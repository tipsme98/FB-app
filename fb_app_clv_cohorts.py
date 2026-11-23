"""CLV evidence and cohort rules. No network, database, Streamlit, or model calls."""

import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import NamedTuple

HKT = timezone(timedelta(hours=8))
MARKETS = {"讓球": ("Home", "Away"), "入球大小": ("Over", "Under"), "角球大小": ("Over", "Under")}
QUOTE = "_clv_tipsme_hkjc_v1"
ENTRY = "CLV_ENTRY_V1:"
CUTOFF = datetime(2026, 11, 22, tzinfo=HKT)


class Cohort(NamedTuple):
    group: str
    clv: float | None = None
    closing: float | None = None
    close_time: str = ""


def fixture_id(value):
    m = re.fullmatch(r"([1-9][0-9]{0,19})(?:\.0+)?", str(value).strip())
    return m[1] if m else ""


def local(value, *, zone=False):
    if not isinstance(value, (datetime, str)):
        return None
    try:
        ts = datetime.fromisoformat(value) if isinstance(value, str) else value
        if ts != ts:  # pandas NaT (without importing pandas)
            return None
        if ts.tzinfo is not None:
            return ts.astimezone(HKT)
        return None if zone else ts.replace(tzinfo=HKT)
    except (ValueError, OverflowError):
        return None


def report_period(value):
    """Use the recorded entry clock in HKT, never the match/settlement date."""
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return None  # a date without a recorded time is not an entry timestamp
    dt = local(value.strip() if isinstance(value, str) else value)
    return None if dt is None else int(dt >= CUTOFF)


def number(value, minimum=None):
    if isinstance(value, bool):
        return None
    try:
        used = float(value)
        return used if math.isfinite(used) and (minimum is None or used > minimum) else None
    except (TypeError, ValueError, OverflowError):
        return None


def history(value):
    try:
        if isinstance(value, str):
            value = json.loads(value)
    except (ValueError, RecursionError):
        pass
    return value if isinstance(value, list) else []


def quote_tag(mid, timestamp, kickoff=None):
    """Timestamp is the original source time normalized to HKT, not HTTP observation time."""
    m, t = fixture_id(mid), local(timestamp, zone=True)
    k = local(kickoff)
    return [m, t.isoformat(), k.isoformat() if k else ""] if m and t else None


def row_key(row):
    market, side = str(row.get("Bet_Type", "")).strip(), str(row.get("Selection", "")).strip()
    line, price = number(row.get("Initial_Line")), number(row.get("Initial_Odds"), 1)
    mid, rid = fixture_id(row.get("Tipsme_Match_ID")), str(row.get("ID", "")).strip()
    if side not in MARKETS.get(market, ()) or line is None or price is None:
        return None
    return [rid, mid, market, side, round(line, 6), round(price, 6)]


def tagged_quote(q, key, *, same_line=True):
    if not isinstance(q, dict) or q.get("type") != key[2]:
        return None
    tag = q.get(QUOTE)
    if not isinstance(tag, list) or len(tag) != 3 or tag[0] != key[1] or not key[1]:
        return None
    t, k = local(tag[1], zone=True), local(tag[2], zone=True)
    line, up, lo = number(q.get("line")), number(q.get("upper"), 1), number(q.get("lower"), 1)
    if t is None or line is None or (same_line and round(line, 6) != key[4]) or None in (up, lo):
        return None
    return t, (up, lo), k


def entry_note(row, candidate, prepared, scheduled, recorded):
    """Build evidence only when the actual selected quote is in the locked provider snapshot."""
    key = row_key(row)
    if not key or not all(key[:2]) or not isinstance(candidate, dict):
        return None
    fill = prepared.get("fill") or {}
    info = fill.get("info") or {}
    token = prepared.get("token") or ()
    if len(token) != 3 or fixture_id(token[2]) != key[1] or fixture_id(info.get("match_id")) != key[1]:
        return None
    kickoff, taken, entered = local(info.get("kickoff")), local(fill.get("clv_observed"), zone=True), local(recorded)
    try:
        queue_ko = datetime.fromtimestamp(float(scheduled), HKT)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    if not (kickoff and taken and entered and fill.get("odds_ok")
            and info.get("clv_id_confirmed") is True and info.get("finished") is False):
        return None
    if (queue_ko.replace(second=0, microsecond=0) != kickoff.replace(second=0, microsecond=0)
            or not taken <= entered <= min(queue_ko, kickoff) - timedelta(minutes=1)
            or entered - taken > timedelta(minutes=5)
            or str(row.get("Match_Date")) != kickoff.date().isoformat()):
        return None
    snapshots = prepared.get("odds") or []
    idx = candidate.get("history_idx")
    if type(idx) is not int or not 0 <= idx < len(snapshots) or candidate.get("is_latest") is not True:
        return None
    if any(isinstance(s, dict) and s.get("type") == key[2] for s in snapshots[idx + 1:]):
        return None
    quote = tagged_quote(snapshots[idx], key)
    if quote is None:
        return None
    qtime, pair, archive_ko = quote
    side = MARKETS[key[2]].index(key[3])
    if (archive_ko is not None or not taken - timedelta(minutes=30) <= qtime <= taken
            or not math.isclose(pair[side], key[5], abs_tol=1e-6, rel_tol=0)):
        return None
    return ENTRY + json.dumps(
        [key, qtime.isoformat(), taken.isoformat(), entered.isoformat(), kickoff.isoformat(), pair],
        ensure_ascii=False, separators=(",", ":"),
    )


def verified_entry(row):
    """Return the locked entry before looking at settlement or the closing history."""
    key = row_key(row)
    note = str(row.get("Note", "")).strip()
    if not key or not note.startswith(ENTRY):
        return None
    try:
        parts = json.loads(note[len(ENTRY):])
        if not isinstance(parts, list) or len(parts) not in (6, 7):
            return None
        bound, qt, ct, rt, kt, pair = parts[:6]
        q, capture, entered, kickoff = [local(t, zone=True) for t in (qt, ct, rt, kt)]
        row_date = local(row.get("Date"))
        side = MARKETS[key[2]].index(key[3])
        if (bound != key or not all(key[:2]) or not all((q, capture, entered, kickoff, row_date))
                or not capture - timedelta(minutes=30) <= q <= capture <= entered <= kickoff - timedelta(minutes=1)
                or entered - capture > timedelta(minutes=5)
                or report_period(row.get("Date")) is None
                or row_date.replace(second=0, microsecond=0) != entered.replace(second=0, microsecond=0)
                or str(row.get("Match_Date"))[:10] != kickoff.date().isoformat()
                or len(pair) != 2 or any(number(n, 1) is None for n in pair)
                or not math.isclose(float(pair[side]), key[5], abs_tol=1e-6, rel_tol=0)):
            return None
    except (TypeError, ValueError, RecursionError):
        return None
    return key, q, entered, kickoff, tuple(map(float, pair)), parts[6] if len(parts) == 7 else None


def record_handicap_forecast(row, forecast):
    """Freeze the submit-time prediction separately from mutable display columns."""
    proof = verified_entry(row)
    if not proof or proof[0][2] != "讓球" or proof[5] is not None:
        return
    p, pm, edge = [number(forecast.get(k)) for k in ("p_model", "p_mkt", "edge")]
    if p is None or pm is None or edge is None or not 0 < p < 1 or not 0 < pm < 1:
        return
    parts = json.loads(row["Note"][len(ENTRY):])
    parts.append(["V15.8_LR", p, pm])
    row["Note"] = ENTRY + json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def classify(row):
    """Strict for new observations; legacy rows never inherit proof from a matching price."""
    key, note = row_key(row), str(row.get("Note", "")).strip()
    if note == "用家跟投":
        return Cohort("follow")
    if not key:
        return Cohort("out_of_scope")
    if not note.startswith(ENTRY):
        # Diagnostic only: a date on its own cannot prove the entry source or odds.
        try:
            day = date.fromisoformat(str(row.get("Match_Date", ""))[:10])
        except ValueError:
            day = None
        logged = local(row.get("Date"))
        if day and logged and logged.date() > day:
            return Cohort("historical_late")
        return Cohort("historical_unknown")
    proof = verified_entry(row)
    if proof is None:
        return Cohort("entry_invalid")
    key, q, entered, kickoff, pair, _ = proof
    side = MARKETS[key[2]].index(key[3])
    closes = []
    for raw in history(row.get("Odds_History")):
        quote = tagged_quote(raw, key, same_line=False)
        if quote is not None:
            stamp, price_pair, actual_kickoff = quote
            if actual_kickoff is not None and actual_kickoff != kickoff:
                return Cohort("kickoff_conflict")
            if actual_kickoff == kickoff and stamp < kickoff:
                closes.append((stamp, round(float(raw["line"]), 6), price_pair))
    if not closes:
        return Cohort("pending")
    last = max(t for t, _, _ in closes)
    finals = {(line, p) for t, line, p in closes if t == last}
    if len(finals) != 1:
        return Cohort("close_conflict")
    line, final_pair = next(iter(finals))
    if line != key[4]:
        return Cohort("pending")
    # If the pre-kickoff entry actually used the final quote, the independent
    # completed-fixture archive must confirm the very same two-sided quote.
    # A different or older pre-entry quote, or a duplicate live snapshot, is not proof.
    same_final_quote = last == q and all(
        math.isclose(x, float(y), abs_tol=1e-6, rel_tol=0) for x, y in zip(final_pair, pair)
    )
    if not same_final_quote and not (entered < last and kickoff - last <= timedelta(minutes=15)):
        return Cohort("pending")
    closing = final_pair[side]
    if same_final_quote:
        return Cohort("snapshot", 0.0, closing, last.isoformat())
    return Cohort("snapshot", key[5] / closing - 1, closing, last.isoformat())
