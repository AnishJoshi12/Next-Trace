"""SQLite time-series store: tiered (hot 1 Hz / warm 1-min rollups / events) as in design doc section 3.3.

* ``Store``       read side. Opened ``mode=ro`` with ``PRAGMA query_only`` AND a SQLite authorizer that
                  denies everything except SELECT. The assistant process only ever imports this class.
* ``StoreWriter`` write side, used by services/store_writer.py and the seeding script only.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils import get_logger, tz_offset_s

log = get_logger("vgpt.store")

_SCHEMA = Path(__file__).with_name("schema.sql")

# SQLite action codes allowed for the read-only assistant connection.
_SQLITE_RECURSIVE = 33
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, _SQLITE_RECURSIVE}


def _ro_authorizer(action: int, *_: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class StoreUnavailable(RuntimeError):
    pass


class Store:
    """Read-only view of the vehicle database (thread-safe: one connection per thread)."""

    def __init__(self, path: str, readonly: bool = True, hot_days: int = 7, tz: str = "Asia/Kolkata",
                 _shared: bool = False):
        self.path = str(path)
        self.readonly = readonly
        self.hot_days = hot_days
        self.tz = tz
        self._shared = _shared
        self._lock = threading.RLock()
        self._local = threading.local()
        self._shared_conn: sqlite3.Connection | None = None
        self._ids: dict[str, int] = {}
        self._names: dict[int, str] = {}
        if readonly and not Path(self.path).exists():
            raise StoreUnavailable(f"database not found: {self.path}")
        self._load_catalog()

    # ---- connections -----------------------------------------------------
    def _open(self) -> sqlite3.Connection:
        if self.readonly:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=5, check_same_thread=False)
            conn.execute("PRAGMA query_only = ON")
            conn.set_authorizer(_ro_authorizer)
        else:
            conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
            conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _conn(self) -> sqlite3.Connection:
        if self._shared:
            if self._shared_conn is None:
                self._shared_conn = self._open()
            return self._shared_conn
        c = getattr(self._local, "c", None)
        if c is None:
            c = self._local.c = self._open()
        return c

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        if self._shared:
            with self._lock:
                return self._conn().execute(sql, params).fetchall()
        return self._conn().execute(sql, params).fetchall()

    def close(self) -> None:
        for c in (self._shared_conn, getattr(self._local, "c", None)):
            if c is not None:
                try:
                    c.close()
                except sqlite3.Error:
                    pass
        self._shared_conn = None
        self._local = threading.local()

    # ---- catalog ---------------------------------------------------------
    def _load_catalog(self) -> None:
        try:
            rows = self.query("SELECT sig_id, name FROM signal_catalog")
        except sqlite3.Error:
            rows = []
        self._ids = {r["name"]: r["sig_id"] for r in rows}
        self._names = {v: k for k, v in self._ids.items()}

    def sig_id(self, name: str) -> int | None:
        sid = self._ids.get(name)
        if sid is None:
            self._load_catalog()
            sid = self._ids.get(name)
        return sid

    def sig_name(self, sid: int) -> str:
        return self._names.get(sid, f"sig{sid}")

    # ---- point / aggregate / series --------------------------------------
    def latest(self, name: str, before_ts: float | None = None) -> dict | None:
        sid = self.sig_id(name)
        if sid is None:
            return None
        if before_ts is None:
            rows = self.query("SELECT ts, val FROM signals_hot WHERE sig_id=? ORDER BY ts DESC LIMIT 1", (sid,))
            cap = 2 ** 40
        else:
            rows = self.query("SELECT ts, val FROM signals_hot WHERE sig_id=? AND ts<=? ORDER BY ts DESC LIMIT 1",
                              (sid, int(before_ts)))
            cap = int(before_ts)
        if rows:
            return {"ts": float(rows[0]["ts"]), "val": rows[0]["val"], "approx": False}
        rows = self.query("SELECT minute_ts + 30 AS ts, vmean AS val FROM rollup_1m "
                          "WHERE sig_id=? AND minute_ts<=? ORDER BY minute_ts DESC LIMIT 1", (sid, cap))
        if rows:  # older than the hot tier: minute mean, flagged approximate
            return {"ts": float(rows[0]["ts"]), "val": rows[0]["val"], "approx": True}
        return None

    def latest_values(self, names: Iterable[str]) -> dict[str, dict]:
        wanted = set(names)
        rows = self.query(
            "SELECT c.name AS name, h.ts AS ts, h.val AS val FROM signal_catalog c "
            "JOIN signals_hot h ON h.sig_id = c.sig_id "
            "AND h.ts = (SELECT MAX(ts) FROM signals_hot WHERE sig_id = c.sig_id)")
        return {r["name"]: {"ts": r["ts"], "val": r["val"]} for r in rows if r["name"] in wanted}

    def agg(self, name: str, start: float, end: float, now: float | None = None) -> dict | None:
        """min / max / mean / n over [start, end). Short recent windows read the hot tier, others the rollups."""
        sid = self.sig_id(name)
        if sid is None:
            return None
        now = time.time() if now is None else now
        if (end - start) <= 900 and start >= now - self.hot_days * 86400:
            r = self.query("SELECT MIN(val) mn, MAX(val) mx, AVG(val) mean, COUNT(*) n FROM signals_hot "
                           "WHERE sig_id=? AND ts>=? AND ts<?", (sid, int(start), int(end)))[0]
        else:
            r = self.query("SELECT MIN(vmin) mn, MAX(vmax) mx, SUM(vmean*n)/SUM(n) mean, SUM(n) n FROM rollup_1m "
                           "WHERE sig_id=? AND minute_ts>=? AND minute_ts<?",
                           (sid, int(start) // 60 * 60, int(end)))[0]
        if not r["n"]:
            return None
        return {"min": r["mn"], "max": r["mx"], "mean": r["mean"], "n": int(r["n"])}

    def series(self, name: str, start: float, end: float, bucket_s: int) -> list[dict]:
        """Bucketed means from the 1-minute rollups. Buckets are aligned to local midnight."""
        sid = self.sig_id(name)
        if sid is None:
            return []
        off = tz_offset_s(start, self.tz)
        rows = self.query(
            "SELECT ((minute_ts + ?) / ?) * ? - ? AS b, SUM(vmean*n)/SUM(n) AS mean, MIN(vmin) AS mn, "
            "MAX(vmax) AS mx, SUM(n) AS n FROM rollup_1m WHERE sig_id=? AND minute_ts>=? AND minute_ts<? "
            "GROUP BY (minute_ts + ?) / ? ORDER BY b",
            (off, bucket_s, bucket_s, off, sid, int(start) // 60 * 60, int(end), off, bucket_s))
        return [{"t": r["b"], "mean": r["mean"], "min": r["mn"], "max": r["mx"], "n": r["n"]} for r in rows]

    def full_range_series(self, start: float, end: float, bucket_s: int, min_soc: float = 20.0) -> list[dict]:
        """Projected full-battery range: est_range_km / (soc/100), only where SOC >= min_soc (avoids divide-by-small)."""
        rid, sid = self.sig_id("est_range_km"), self.sig_id("soc_pct")
        if rid is None or sid is None:
            return []
        off = tz_offset_s(start, self.tz)
        rows = self.query(
            "SELECT ((r.minute_ts + ?) / ?) * ? - ? AS b, AVG(r.vmean / (s.vmean / 100.0)) AS mean, "
            "MIN(r.vmean / (s.vmean / 100.0)) AS mn, MAX(r.vmean / (s.vmean / 100.0)) AS mx, COUNT(*) AS n "
            "FROM rollup_1m r JOIN rollup_1m s ON s.minute_ts = r.minute_ts AND s.sig_id = ? "
            "WHERE r.sig_id = ? AND r.minute_ts>=? AND r.minute_ts<? AND s.vmean >= ? "
            "GROUP BY (r.minute_ts + ?) / ? ORDER BY b",
            (off, bucket_s, bucket_s, off, sid, rid, int(start) // 60 * 60, int(end), min_soc, off, bucket_s))
        return [{"t": r["b"], "mean": r["mean"], "min": r["mn"], "max": r["mx"], "n": r["n"]} for r in rows]

    # ---- drive-mode comparisons ------------------------------------------
    def mode_means(self, name: str, start: float, end: float) -> list[dict]:
        sid = self.sig_id(name)
        if sid is None:
            return []
        rows = self.query(
            "SELECT m.drive_mode AS mode, SUM(r.vmean*r.n)/SUM(r.n) AS mean, SUM(r.n) AS n "
            "FROM rollup_1m r JOIN minute_mode m ON m.minute_ts = r.minute_ts "
            "WHERE r.sig_id=? AND r.minute_ts>=? AND r.minute_ts<? GROUP BY m.drive_mode",
            (sid, int(start) // 60 * 60, int(end)))
        return [{"mode": r["mode"], "mean": r["mean"], "n": r["n"]} for r in rows]

    def energy_by_mode(self, start: float, end: float) -> list[dict]:
        """Net traction energy per km for each drive mode (idle minutes excluded: no distance)."""
        pid, vid = self.sig_id("motor_power_kw"), self.sig_id("vehicle_speed_kph")
        if pid is None or vid is None:
            return []
        rows = self.query(
            "SELECT m.drive_mode AS mode, SUM(p.vmean*1000.0/60.0) AS wh, SUM(s.vmean/60.0) AS km, COUNT(*) AS minutes "
            "FROM rollup_1m p JOIN rollup_1m s ON s.minute_ts = p.minute_ts AND s.sig_id = ? "
            "JOIN minute_mode m ON m.minute_ts = p.minute_ts "
            "WHERE p.sig_id = ? AND p.minute_ts>=? AND p.minute_ts<? AND m.drive_mode != 'idle' "
            "GROUP BY m.drive_mode",
            (vid, pid, int(start) // 60 * 60, int(end)))
        return [{"mode": r["mode"], "wh": r["wh"], "km": r["km"], "minutes": r["minutes"],
                 "wh_per_km": (r["wh"] / r["km"]) if r["km"] and r["km"] > 0.01 else None} for r in rows]

    # ---- events ----------------------------------------------------------
    def events(self, types: Sequence[str] | None, start: float, end: float, limit: int = 50,
               newest_first: bool = False) -> list[dict]:
        where, params = "ts>=? AND ts<?", [start, end]
        if types:
            where += f" AND type IN ({','.join('?' * len(types))})"
            params += list(types)
        order = "DESC" if newest_first else "ASC"
        rows = self.query(f"SELECT id, ts, type, severity, source, detail FROM events WHERE {where} "
                          f"ORDER BY ts {order} LIMIT ?", params + [limit])
        return [self._event(r) for r in rows]

    def count_events(self, types: Sequence[str] | None, start: float, end: float) -> dict[str, int]:
        where, params = "ts>=? AND ts<?", [start, end]
        if types:
            where += f" AND type IN ({','.join('?' * len(types))})"
            params += list(types)
        rows = self.query(f"SELECT type, COUNT(*) AS n FROM events WHERE {where} GROUP BY type", params)
        return {r["type"]: r["n"] for r in rows}

    def last_event(self, types: Sequence[str] | None, start: float, end: float, with_context: bool = True) -> dict | None:
        where, params = "e.ts>=? AND e.ts<?", [start, end]
        if types:
            where += f" AND e.type IN ({','.join('?' * len(types))})"
            params += list(types)
        if with_context:
            where += " AND EXISTS (SELECT 1 FROM event_context c WHERE c.event_id = e.id)"
        rows = self.query(f"SELECT e.id, e.ts, e.type, e.severity, e.source, e.detail FROM events e "
                          f"WHERE {where} ORDER BY e.ts DESC LIMIT 1", params)
        return self._event(rows[0]) if rows else None

    def context(self, event_id: int) -> list[tuple[float, str, float]]:
        rows = self.query("SELECT ts, sig_id, val FROM event_context WHERE event_id=? ORDER BY ts", (event_id,))
        return [(r["ts"], self.sig_name(r["sig_id"]), r["val"]) for r in rows]

    def dtc_occurrences(self, code: str, start: float, end: float) -> list[float]:
        rows = self.query("SELECT ts FROM events WHERE type='DTC' AND ts>=? AND ts<? AND detail LIKE ? ORDER BY ts",
                          (start, end, f'%"{code}"%'))
        return [r["ts"] for r in rows]

    @staticmethod
    def _event(r: sqlite3.Row) -> dict:
        try:
            detail = json.loads(r["detail"]) if r["detail"] else {}
        except (TypeError, ValueError):
            detail = {"raw": str(r["detail"])[:200]}
        return {"id": r["id"], "ts": r["ts"], "type": r["type"], "severity": r["severity"],
                "source": r["source"], "detail": detail}

    # ---- health ----------------------------------------------------------
    def last_data_ts(self) -> float | None:
        sid = self.sig_id("vehicle_speed_kph")
        if sid is None:
            return None
        rows = self.query("SELECT MAX(ts) AS t FROM signals_hot WHERE sig_id=?", (sid,))
        return float(rows[0]["t"]) if rows and rows[0]["t"] is not None else None

    def health(self) -> dict:
        try:
            last = self.last_data_ts()
            n_ev = self.query("SELECT COUNT(*) AS n FROM events")[0]["n"]
            return {"ok": True, "last_sample_ts": last, "events": n_ev, "signals": len(self._ids)}
        except sqlite3.Error as exc:
            return {"ok": False, "error": str(exc)}


class StoreWriter(Store):
    """Read-write handle. One shared connection guarded by a lock (MQTT thread + maintenance thread)."""

    def __init__(self, path: str, catalog: dict[str, dict] | None = None, hot_days: int = 7,
                 warm_days: int = 70, tz: str = "Asia/Kolkata", idle_kph: float = 1.0, highway_kph: float = 60.0,
                 disk_guard_pct: float = 85.0):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.warm_days, self.idle_kph, self.highway_kph = warm_days, idle_kph, highway_kph
        self.disk_guard_pct = disk_guard_pct
        self._bulk = False
        self.unknown_signals: dict[str, int] = {}
        super().__init__(path, readonly=False, hot_days=hot_days, tz=tz, _shared=True)
        self.init_schema()
        if catalog:
            self.sync_catalog(catalog)

    # ---- setup -----------------------------------------------------------
    def init_schema(self) -> None:
        with self._lock:
            self._conn().executescript(_SCHEMA.read_text(encoding="utf-8"))
        self._load_catalog()

    def sync_catalog(self, catalog: dict[str, dict]) -> None:
        with self._lock:
            c = self._conn()
            for name, meta in catalog.items():
                c.execute("INSERT OR IGNORE INTO signal_catalog(name, unit, description) VALUES (?,?,?)",
                          (name, meta.get("unit", ""), meta.get("label", "")))
            c.commit()
        self._load_catalog()

    # ---- transactions ----------------------------------------------------
    def _commit(self) -> None:
        if not self._bulk:
            self._conn().commit()

    @contextmanager
    def bulk(self):
        """Hold one transaction across many writes (seeding / replay)."""
        with self._lock:
            self._bulk = True
            try:
                yield self
            finally:
                self._bulk = False
                self._conn().commit()

    # ---- writes ----------------------------------------------------------
    def write_signal_rows(self, rows: Sequence[tuple[int, int, float]]) -> None:
        with self._lock:
            self._conn().executemany("INSERT OR REPLACE INTO signals_hot(sig_id, ts, val) VALUES (?,?,?)", rows)
            self._commit()

    def write_signals(self, t: float, values: dict[str, float]) -> int:
        rows = []
        ts = int(t)
        for name, v in values.items():
            sid = self.sig_id(name)
            if sid is None:
                self.unknown_signals[name] = self.unknown_signals.get(name, 0) + 1
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv or fv in (float("inf"), float("-inf")):
                continue
            rows.append((sid, ts, fv))
        if rows:
            self.write_signal_rows(rows)
        return len(rows)

    def write_event(self, ev: dict) -> int:
        """ev = {t, type, severity?, source?, detail?, context?: [[t, name, value], ...]}"""
        with self._lock:
            c = self._conn()
            cur = c.execute("INSERT INTO events(ts, type, severity, source, detail) VALUES (?,?,?,?,?)",
                            (float(ev["t"]), str(ev["type"]), ev.get("severity"), ev.get("source"),
                             json.dumps(ev.get("detail") or {}, separators=(",", ":"))))
            eid = int(cur.lastrowid)
            ctx = []
            for t, name, v in ev.get("context") or []:
                sid = self.sig_id(name)
                if sid is not None:
                    ctx.append((eid, float(t), sid, float(v)))
            if ctx:
                c.executemany("INSERT INTO event_context(event_id, ts, sig_id, val) VALUES (?,?,?,?)", ctx)
            self._commit()
            return eid

    def write_log(self, t: float, source: str, level: str, message: str) -> None:
        with self._lock:
            self._conn().execute("INSERT INTO logs(ts, source, level, message) VALUES (?,?,?,?)",
                                 (float(t), source, level, str(message)[:500]))
            self._commit()

    # ---- meta ------------------------------------------------------------
    def meta_get(self, key: str) -> str | None:
        rows = self.query("SELECT v FROM meta WHERE k=?", (key,))
        return rows[0]["v"] if rows else None

    def meta_set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn().execute("INSERT OR REPLACE INTO meta(k, v) VALUES (?,?)", (key, value))
            self._commit()

    # ---- maintenance -----------------------------------------------------
    def rollup(self, now: float | None = None) -> int:
        """Incrementally build 1-minute rollups for completed minutes and classify each minute's drive mode."""
        now = time.time() if now is None else now
        end_min = int(now) // 60 * 60
        last = self.meta_get("last_rollup_minute")
        if last is None:
            speed = self.sig_id("vehicle_speed_kph")
            r = self.query("SELECT MIN(ts) AS t FROM signals_hot WHERE sig_id=?", (speed,)) if speed else []
            first = r[0]["t"] if r and r[0]["t"] is not None else None
            if first is None:
                return 0
            start_min = int(first) // 60 * 60
        else:
            start_min = int(last) - 60  # re-roll the last minute: late samples may have arrived
        if end_min <= start_min:
            return 0
        with self._lock:
            c = self._conn()
            for sid in list(self._ids.values()):
                c.execute(
                    "INSERT OR REPLACE INTO rollup_1m(sig_id, minute_ts, vmin, vmax, vmean, n) "
                    "SELECT sig_id, (ts/60)*60, MIN(val), MAX(val), AVG(val), COUNT(*) FROM signals_hot "
                    "WHERE sig_id=? AND ts>=? AND ts<? GROUP BY ts/60", (sid, start_min, end_min))
            speed = self._ids.get("vehicle_speed_kph")
            if speed is not None:
                c.execute(
                    "INSERT OR REPLACE INTO minute_mode(minute_ts, drive_mode) "
                    "SELECT minute_ts, CASE WHEN vmean < ? THEN 'idle' WHEN vmean >= ? THEN 'highway' ELSE 'city' END "
                    "FROM rollup_1m WHERE sig_id=? AND minute_ts>=? AND minute_ts<?",
                    (self.idle_kph, self.highway_kph, speed, start_min, end_min))
            c.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('last_rollup_minute', ?)", (str(end_min),))
            self._commit()
        return (end_min - start_min) // 60

    def retention(self, now: float | None = None, hot_days: int | None = None) -> dict:
        now = time.time() if now is None else now
        hot_cut = int(now - (hot_days or self.hot_days) * 86400)
        warm_cut = int(now - self.warm_days * 86400)
        with self._lock:
            c = self._conn()
            n_hot = 0
            for sid in list(self._ids.values()):  # per-signal deletes use the primary-key range (no ts index)
                n_hot += c.execute("DELETE FROM signals_hot WHERE sig_id=? AND ts<?", (sid, hot_cut)).rowcount
            n_roll = c.execute("DELETE FROM rollup_1m WHERE minute_ts<?", (warm_cut,)).rowcount
            c.execute("DELETE FROM minute_mode WHERE minute_ts<?", (warm_cut,))
            n_ev = c.execute("DELETE FROM events WHERE ts<?", (warm_cut,)).rowcount  # context rows cascade
            c.execute("DELETE FROM logs WHERE ts<?", (warm_cut,))
            c.commit()
            try:
                c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                c.execute("PRAGMA incremental_vacuum(2000)")
                c.commit()
            except sqlite3.Error as exc:
                log.warning("maintenance pragma failed: %s", exc)
        return {"hot_deleted": n_hot, "rollups_deleted": n_roll, "events_deleted": n_ev}

    def disk_guard(self, now: float | None = None) -> float:
        """Used-percent of the volume holding the DB. At/over the guard level the hot tier is halved."""
        usage = shutil.disk_usage(Path(self.path).parent)
        pct = 100.0 * usage.used / usage.total
        if pct >= self.disk_guard_pct:
            log.warning("disk %.1f%% used: trimming hot tier to %d days", pct, max(1, self.hot_days // 2))
            self.retention(now, hot_days=max(1, self.hot_days // 2))
        return pct

    def checkpoint(self) -> None:
        with self._lock:
            try:
                self._conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
