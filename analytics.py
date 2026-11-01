"""Deterministic batch analytics (design doc section 4, module [4]).

* ``range_attribution``  regression of daily Wh/km on ambient temperature, highway share and tyre pressure.
                         Produces statements like "range fell 12%: about 7 points cold weather, 4 highway, 1 tyres".
                         All arithmetic is NumPy; the LLM only narrates the result.
* ``anomaly_scan``       Isolation Forest (scikit-learn, optional) on 5-minute battery-thermal windows built from the
                         1-minute rollups; falls back to a robust z-score when scikit-learn is not installed.

The design document also names ONNX Runtime for inference. Exporting these small models to ONNX is a packaging step
(skl2onnx) that does not change behaviour, so it is left as future work and the fitted models stay in-process.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from store import Store, StoreWriter
from utils import get_logger, tz_offset_s

log = get_logger("vgpt.analytics")

_DAILY_SQL = """
SELECT ((p.minute_ts + :off) / 86400) AS day,
       SUM(p.vmean * 1000.0 / 60.0)                                   AS wh,
       SUM(s.vmean / 60.0)                                            AS km,
       SUM(CASE WHEN m.drive_mode = 'highway' THEN s.vmean / 60.0 ELSE 0 END) AS hwy_km,
       SUM(a.vmean * s.vmean / 60.0)                                  AS amb_km,
       SUM(t.vmean * s.vmean / 60.0)                                  AS tire_km
FROM rollup_1m p
JOIN rollup_1m s ON s.minute_ts = p.minute_ts AND s.sig_id = :sid_speed
JOIN rollup_1m a ON a.minute_ts = p.minute_ts AND a.sig_id = :sid_amb
JOIN rollup_1m t ON t.minute_ts = p.minute_ts AND t.sig_id = :sid_tire
JOIN minute_mode m ON m.minute_ts = p.minute_ts
WHERE p.sig_id = :sid_power AND p.minute_ts >= :start AND p.minute_ts < :end AND m.drive_mode != 'idle'
GROUP BY day
HAVING km > :min_km
ORDER BY day
"""


def daily_energy(store: Store, start: float, end: float, min_km: float = 5.0) -> list[dict]:
    ids = {k: store.sig_id(n) for k, n in [("sid_power", "motor_power_kw"), ("sid_speed", "vehicle_speed_kph"),
                                           ("sid_amb", "ambient_temp_c"), ("sid_tire", "tire_pressure_kpa")]}
    if any(v is None for v in ids.values()):
        return []
    off = tz_offset_s(start, store.tz)
    rows = store.query(_DAILY_SQL, {**ids, "off": off, "start": int(start) // 60 * 60, "end": int(end), "min_km": min_km})
    out = []
    for r in rows:
        km = r["km"]
        out.append({"day_ts": r["day"] * 86400 - off, "km": km, "wh_per_km": r["wh"] / km,
                    "hwy_share": 100.0 * r["hwy_km"] / km, "ambient_c": r["amb_km"] / km, "tire_kpa": r["tire_km"] / km})
    return out


def range_attribution(store: Store, start: float, end: float, min_days: int = 6) -> dict[str, Any]:
    """Compare the first and last half of the window; attribute the Wh/km change with a 3-factor OLS fit."""
    days = daily_energy(store, start, end)
    n = len(days)
    if n < min_days:
        return {"ok": False, "reason": f"only {n} driving day(s) with data in the window; at least {min_days} are needed",
                "days": n}
    y = np.array([d["wh_per_km"] for d in days])
    feats = np.array([[d["ambient_c"], d["hwy_share"], d["tire_kpa"]] for d in days])
    X = np.column_stack([np.ones(n), feats])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 1e-12 else 0.0

    half = n // 2
    base_idx, rec_idx = slice(0, half), slice(n - half, n)
    y0, y1 = float(y[base_idx].mean()), float(y[rec_idx].mean())
    f0, f1 = feats[base_idx].mean(axis=0), feats[rec_idx].mean(axis=0)
    contrib_wh = coef[1:] * (f1 - f0)                    # Wh/km attributed to each factor
    total_wh = y1 - y0
    resid_wh = total_wh - float(contrib_wh.sum())
    pct = lambda wh: 100.0 * wh / y0                     # noqa: E731  (points of the baseline Wh/km)
    names = ["ambient temperature", "highway share", "tyre pressure"]
    return {
        "ok": True, "days": n, "r2": r2, "reliable": r2 >= 0.6,
        "base_wh_per_km": y0, "recent_wh_per_km": y1,
        "energy_change_pct": pct(total_wh),
        "range_change_pct": 100.0 * (y0 / y1 - 1.0),    # range ~ 1 / (Wh per km) for a fixed battery
        "factors": [{"name": nm, "from": float(a), "to": float(b), "points": pct(float(c))}
                    for nm, a, b, c in zip(names, f0, f1, contrib_wh)],
        "unexplained_points": pct(resid_wh),
        "window": {"start": days[0]["day_ts"], "end": days[-1]["day_ts"] + 86400},
    }


# --------------------------------------------------------------------------
# anomaly scan
# --------------------------------------------------------------------------

def _thermal_windows(store: Store, since: float, win_min: int = 5) -> tuple[np.ndarray, np.ndarray]:
    t_id, i_id = store.sig_id("battery_temp_max_c"), store.sig_id("pack_current_a")
    if t_id is None or i_id is None:
        return np.empty((0, 4)), np.empty(0)
    rows = store.query(
        "SELECT t.minute_ts AS m, t.vmean AS tm, t.vmax AS tx, i.vmean AS im FROM rollup_1m t "
        "JOIN rollup_1m i ON i.minute_ts = t.minute_ts AND i.sig_id = ? WHERE t.sig_id = ? AND t.minute_ts >= ? "
        "ORDER BY t.minute_ts", (i_id, t_id, int(since)))
    buckets: dict[int, list] = {}
    for r in rows:
        buckets.setdefault(r["m"] // (win_min * 60), []).append(r)
    feats, starts = [], []
    for b, rs in sorted(buckets.items()):
        if len(rs) < max(3, win_min - 1):
            continue
        tm = np.array([r["tm"] for r in rs])
        slope = (tm[-1] - tm[0]) / max(1, len(rs) - 1)          # deg C per minute
        feats.append([tm.mean(), max(r["tx"] for r in rs), slope, float(np.mean([abs(r["im"]) for r in rs]))])
        starts.append(b * win_min * 60)
    return np.array(feats), np.array(starts)


def anomaly_scan(store: Store, writer: StoreWriter | None = None, now: float | None = None,
                 train_days: int = 7, recent_min: int = 30, min_temp_c: float = 45.0) -> list[dict]:
    """Score the most recent windows against the vehicle's own recent history. Returns detections
    (and records THERMAL_ANOMALY events if a writer is given and none was logged in the last 30 minutes)."""
    now = time.time() if now is None else now
    X, starts = _thermal_windows(store, now - train_days * 86400)
    if len(X) < 40:
        return []
    recent = starts >= now - recent_min * 60
    train = ~recent
    if recent.sum() == 0 or train.sum() < 30:
        return []
    try:
        from sklearn.ensemble import IsolationForest  # optional dependency

        model = IsolationForest(n_estimators=100, random_state=0, contamination="auto").fit(X[train])
        score = -model.decision_function(X[recent])          # higher = more anomalous
        flagged = score > 0.10
        method = "isolation_forest"
    except ImportError:
        med = np.median(X[train], axis=0)
        mad = np.median(np.abs(X[train] - med), axis=0) * 1.4826 + 1e-6
        score = np.abs((X[recent] - med) / mad).max(axis=1) / 10.0
        flagged = score > 1.0
        method = "robust_z"
    detections = []
    for row, st, sc, fl in zip(X[recent], starts[recent], score, flagged):
        if fl and row[1] >= min_temp_c:                       # ignore "unusual" but harmless cold-weather windows
            detections.append({"t": float(st), "score": float(sc), "method": method,
                               "temp_mean_c": float(row[0]), "temp_max_c": float(row[1]), "slope_c_per_min": float(row[2])})
    if detections and writer is not None:
        if not writer.events(["THERMAL_ANOMALY"], now - 1800, now + 1, limit=1):
            d = max(detections, key=lambda x: x["score"])
            writer.write_event({"t": d["t"], "type": "THERMAL_ANOMALY", "severity": "medium", "source": "analytics",
                                "detail": {k: round(v, 2) if isinstance(v, float) else v for k, v in d.items()}})
    return detections
