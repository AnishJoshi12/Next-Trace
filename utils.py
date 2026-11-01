"""Shared helpers: logging, time parsing, slot catalogs, grounding check, ring buffer, auth tokens.

Everything here is pure Python (stdlib only) so it can be unit-tested anywhere.
"""
from __future__ import annotations

import base64
import collections
import difflib
import hashlib
import hmac
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------------
# logging / timing
# --------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    return log


class Timer:
    """``with Timer() as t: ...; t.ms``"""

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0


class TTLCache:
    """Tiny thread-safe TTL cache (tool results are cached for ~30 s)."""

    def __init__(self, ttl_s: float = 30.0, max_items: int = 256):
        self.ttl_s, self.max_items = ttl_s, max_items
        self._d: "collections.OrderedDict[Any, tuple[float, Any]]" = collections.OrderedDict()
        self._lock = threading.Lock()

    def get_or_compute(self, key: Any, fn: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            hit = self._d.get(key)
            if hit and hit[0] > now:
                return hit[1]
        value = fn()
        with self._lock:
            self._d[key] = (now + self.ttl_s, value)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)
        return value


# --------------------------------------------------------------------------
# time parsing (deterministic, no ML)
# --------------------------------------------------------------------------

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60,
}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + r")"
_UNIT = r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)"


def _to_num(tok: str) -> float:
    return float(tok) if tok[0].isdigit() else float(_NUM_WORDS[tok])


def _unit_s(unit: str) -> int:
    u = unit.lower()
    if u.startswith("sec"):
        return 1
    if u.startswith("min"):
        return 60
    if u.startswith("h"):
        return 3600
    if u.startswith("d"):
        return 86400
    return 604800  # week


@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float
    label: str
    explicit: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start


def _midnight(dt: datetime) -> datetime:
    return datetime.combine(dt.date(), dtime.min, tzinfo=dt.tzinfo)


def _fmt_span(n: float, unit_name: str) -> str:
    n_txt = str(int(n)) if float(n).is_integer() else f"{n:g}"
    return f"the last {n_txt} {unit_name}{'' if n == 1 else 's'}"


def parse_time_range(text: str, now: float | None = None, tz: str = "Asia/Kolkata",
                     default_hours: float = 24.0) -> TimeRange:
    """Resolve phrases such as 'last two weeks', 'yesterday', 'today', '3 days ago', 'on 2026-10-12'."""
    now = time.time() if now is None else now
    z = ZoneInfo(tz)
    now_dt = datetime.fromtimestamp(now, z)
    t = text.lower()

    m = re.search(rf"\b(?:last|past|previous|over the last|in the last)\s+{_NUM}\s+{_UNIT}\b", t)
    if m:
        n, unit = _to_num(m.group(1)), m.group(2)
        name = "second" if unit.startswith("sec") else "minute" if unit.startswith("min") else \
            "hour" if unit.startswith("h") else "day" if unit.startswith("d") else "week"
        return TimeRange(now - n * _unit_s(unit), now, _fmt_span(n, name))
    m = re.search(r"\b(?:last|past)\s+(minute|hour|day|week|month)\b", t)
    if m:
        unit = m.group(1)
        if unit == "month":
            return TimeRange(now - 30 * 86400, now, "the last 30 days")
        return TimeRange(now - _unit_s(unit), now, _fmt_span(1, unit))
    if re.search(r"\byesterday\b", t) and not re.search(r"\byesterday\s+(?:at|around|@)", t):
        start = _midnight(now_dt) - timedelta(days=1)
        return TimeRange(start.timestamp(), _midnight(now_dt).timestamp(), "yesterday")
    if re.search(r"\bthis morning\b", t):
        start = _midnight(now_dt)
        return TimeRange(start.timestamp(), min(now, (start + timedelta(hours=12)).timestamp()), "this morning")
    if re.search(r"\b(today|so far today)\b", t):
        return TimeRange(_midnight(now_dt).timestamp(), now, "today")
    if re.search(r"\bthis week\b", t):
        start = _midnight(now_dt) - timedelta(days=now_dt.weekday())
        return TimeRange(start.timestamp(), now, "this week")
    if re.search(r"\bthis month\b", t):
        start = _midnight(now_dt).replace(day=1)
        return TimeRange(start.timestamp(), now, "this month")
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        try:
            d0 = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=z)
            return TimeRange(d0.timestamp(), (d0 + timedelta(days=1)).timestamp(), d0.strftime("%d %b %Y"))
        except ValueError:
            pass
    m = re.search(rf"\b{_NUM}\s+days?\s+ago\b", t)
    if m:
        n = int(_to_num(m.group(1)))
        d0 = _midnight(now_dt) - timedelta(days=n)
        return TimeRange(d0.timestamp(), (d0 + timedelta(days=1)).timestamp(), f"{n} days ago")
    return TimeRange(now - default_hours * 3600, now, _fmt_span(default_hours, "hour"), explicit=False)


def parse_time_point(text: str, now: float | None = None, tz: str = "Asia/Kolkata") -> float | None:
    """Return a timestamp for 'at 14:03', 'yesterday at 2 pm', '10 minutes ago'; None if no point in time is given."""
    now = time.time() if now is None else now
    z = ZoneInfo(tz)
    now_dt = datetime.fromtimestamp(now, z)
    t = text.lower()

    hh = mm = None
    m = re.search(r"(?<![\d:])(\d{1,2}):(\d{2})\s*(am|pm)?\b", t)
    if m:
        hh, mm, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    else:
        m = re.search(r"\b(?:at|around|@)\s*(\d{1,2})\s*(am|pm)\b", t)
        if m:
            hh, mm, ap = int(m.group(1)), 0, m.group(2)
    if hh is not None:
        if ap == "pm" and hh < 12:
            hh += 12
        elif ap == "am" and hh == 12:
            hh = 0
        if hh > 23 or mm > 59:
            return None
        day = _midnight(now_dt)
        explicit_day = False
        iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
        if iso:
            try:
                day = datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)), tzinfo=z)
                explicit_day = True
            except ValueError:
                pass
        elif re.search(r"\byesterday\b", t):
            day, explicit_day = day - timedelta(days=1), True
        else:
            ago = re.search(rf"\b{_NUM}\s+days?\s+ago\b", t)
            if ago:
                day, explicit_day = day - timedelta(days=int(_to_num(ago.group(1)))), True
            elif re.search(r"\btoday\b", t):
                explicit_day = True
        ts = (day + timedelta(hours=hh, minutes=mm)).timestamp()
        if ts > now and not explicit_day:
            ts -= 86400
        return ts

    m = re.search(rf"\b{_NUM}\s+(seconds?|minutes?|mins?|hours?|hrs?|days?)\s+ago\b", t)
    if m:
        return now - _to_num(m.group(1)) * _unit_s(m.group(2))
    return None


def parse_window_seconds(text: str, default: int = 60, maximum: int = 300) -> int:
    """'the 30 seconds before ...' -> 30; '2 minutes before' -> 120."""
    m = re.search(r"\b(\d+)\s*(s|sec|secs|seconds?|m|min|mins|minutes?)\b", text.lower())
    if not m:
        return default
    n = int(m.group(1))
    secs = n * (60 if m.group(2).startswith("m") else 1)
    return max(5, min(secs, maximum))


def tz_offset_s(ts: float, tz: str) -> int:
    off = datetime.fromtimestamp(ts, ZoneInfo(tz)).utcoffset()
    return int(off.total_seconds()) if off else 0


def fmt_time(ts: float, tz: str = "Asia/Kolkata", seconds: bool = False) -> str:
    fmt = "%d %b %H:%M:%S" if seconds else "%d %b %H:%M"
    return datetime.fromtimestamp(ts, ZoneInfo(tz)).strftime(fmt)


def fmt_day(ts: float, tz: str = "Asia/Kolkata") -> str:
    return datetime.fromtimestamp(ts, ZoneInfo(tz)).strftime("%d %b")


# --------------------------------------------------------------------------
# slot catalogs (signal / event name resolution)
# --------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


@dataclass
class Resolution:
    name: str | None
    score: float = 0.0
    ambiguous: list[str] = field(default_factory=list)


class AliasCatalog:
    """Maps free text ('battery temp', 'hard braking') to canonical names."""

    def __init__(self, items: dict[str, dict]):
        self.items = items
        pairs: list[tuple[str, str]] = []
        for name, meta in items.items():
            for a in list(meta.get("aliases", [])) + [meta.get("label", "")]:
                na = _norm(a)
                if na:
                    pairs.append((na, name))
        self._pairs = sorted(set(pairs), key=lambda p: -len(p[0]))

    @classmethod
    def from_json(cls, path: str | Path) -> "AliasCatalog":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolve(self, text: str, fuzzy_cutoff: float = 0.86) -> Resolution:
        t = f" {_norm(text)} "
        best_len, names = 0, set()
        for alias, name in self._pairs:
            if f" {alias} " in t:
                if len(alias) > best_len:
                    best_len, names = len(alias), {name}
                elif len(alias) == best_len:
                    names.add(name)
        if names:
            if len(names) == 1:
                return Resolution(next(iter(names)), 1.0)
            return Resolution(None, 1.0, sorted(names))
        # fuzzy fallback for typos ('batery temp')
        toks = t.split()
        cands = {" ".join(toks[i:i + n]) for n in (1, 2, 3) for i in range(len(toks) - n + 1)}
        best, best_names = 0.0, set()
        for alias, name in self._pairs:
            if len(alias) < 4:
                continue
            for c in cands:
                if abs(len(c) - len(alias)) > 3:
                    continue
                r = difflib.SequenceMatcher(None, c, alias).ratio()
                if r >= fuzzy_cutoff:
                    if r > best + 1e-9:
                        best, best_names = r, {name}
                    elif abs(r - best) <= 1e-9:
                        best_names.add(name)
        if len(best_names) == 1:
            return Resolution(next(iter(best_names)), best)
        if best_names:
            return Resolution(None, best, sorted(best_names))
        return Resolution(None, 0.0)


# --------------------------------------------------------------------------
# grounding check: every number in an answer must exist in the evidence
# --------------------------------------------------------------------------

_NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")
_EVID_MARK = re.compile(r"\[E\d+(?:\s*,\s*E\d+)*\]")


def extract_numbers(text: str) -> list[tuple[float, int]]:
    """Return (|value|, decimals) for each number in ``text`` (evidence markers like [E2] are ignored)."""
    out: list[tuple[float, int]] = []
    for tok in _NUM_RE.findall(_EVID_MARK.sub(" ", text)):
        clean = tok.replace(",", "").lstrip("-")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        try:
            out.append((float(clean), decimals))
        except ValueError:
            continue
    return out


def grounding_check(answer: str, evidence: str, allow_text: str = "") -> tuple[bool, list[str]]:
    """True when every number in ``answer`` is the evidence value (or the evidence rounded to the
    precision shown in the answer). Numbers repeated from the user's question are allowed."""
    ev = [v for v, _ in extract_numbers(evidence)] + [v for v, _ in extract_numbers(allow_text)]
    bad: list[str] = []
    for val, dec in extract_numbers(answer):
        tol = 0.5 * 10 ** (-dec) + 1e-9
        if not any(abs(val - e) <= tol for e in ev):
            bad.append(f"{val:g}")
    return (not bad), bad


_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize_untrusted(text: Any, max_len: int = 200) -> str:
    """Neutralise text that came from vehicle logs / DTC strings before it enters a prompt."""
    s = _CTRL.sub(" ", str(text))
    s = s.replace("<<<", "«").replace(">>>", "»").replace("LOG>>>", "LOG").replace("[E", "(E")
    s = " ".join(s.split())
    return s[:max_len]


# --------------------------------------------------------------------------
# ring buffer (full-rate pre-event context, 60 s)
# --------------------------------------------------------------------------


class RingBuffer:
    def __init__(self, seconds: float = 60.0):
        self.seconds = seconds
        self._d: dict[str, collections.deque] = {}

    def push(self, name: str, t: float, v: float) -> None:
        dq = self._d.setdefault(name, collections.deque())
        dq.append((t, v))
        cutoff = t - self.seconds
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def snapshot(self, names: Iterable[str] | None = None, upto: float | None = None) -> list[list]:
        """[[t, name, value], ...] sorted by time."""
        rows: list[list] = []
        for name, dq in self._d.items():
            if names is not None and name not in names:
                continue
            for t, v in dq:
                if upto is None or t <= upto:
                    rows.append([t, name, v])
        rows.sort(key=lambda r: r[0])
        return rows


# --------------------------------------------------------------------------
# auth tokens (HMAC signed, no external deps)
# --------------------------------------------------------------------------


def make_token(role: str, secret: str, ttl_s: int, now: float | None = None) -> str:
    exp = int((time.time() if now is None else now) + ttl_s)
    payload = base64.urlsafe_b64encode(f"{role}|{exp}".encode()).decode().rstrip("=")
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}.{sig}"


def verify_token(token: str, secret: str, now: float | None = None) -> str | None:
    try:
        payload, sig = token.split(".", 1)
        good = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig, good):
            return None
        role, exp = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode().split("|")
        if (time.time() if now is None else now) > int(exp):
            return None
        return role
    except Exception:
        return None


# --------------------------------------------------------------------------
# resource headroom + audit log
# --------------------------------------------------------------------------


def memory_headroom_mb() -> float | None:
    """MB still available to this process: cgroup v2 limit if set, else /proc/meminfo MemAvailable."""
    try:
        mx = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if mx != "max":
            cur = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            return (int(mx) - cur) / 1e6
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return None


class AuditLog:
    """Append-only JSONL. Stores a hash and length of each query, never the text."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, **fields: Any) -> None:
        q = fields.pop("query", None)
        if q is not None:
            fields["q_sha"] = hashlib.sha256(q.encode()).hexdigest()[:12]
            fields["q_len"] = len(q)
        fields["ts"] = round(time.time(), 3)
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(fields, separators=(",", ":")) + "\n")
        except OSError:
            pass  # audit failure must never break an answer


def fmt_num(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{decimals}f}"
  
