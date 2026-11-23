"""Read-only historical CLV overlay; never modifies the ledger or live evidence."""
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import re
from zoneinfo import ZoneInfo

HKT = ZoneInfo("Asia/Hong_Kong")
BATCH = "hkjc-historical-clv-2026-11-23"
SOURCE_SHA = "6800e099feccc33f6603db71964c6033f49b74d1f03dcf19da6052fb2e9ee98a"
TABLE = "fb_historical_clv_1823"
TOMBS = "football_bets_deleted"
FIELDS = ("ID", "Date", "Match_Date", "Home_Team", "Away_Team", "Tipsme_Match_ID",
          "Bet_Type", "Selection", "Initial_Line", "Initial_Odds")
DISPLAY = ("Status", "Unit_Profit", "Result_Label", "System_Stake")
MARKETS = {"讓球": ("Home", "Away"), "入球大小": ("Over", "Under"), "角球大小": ("Over", "Under")}


def compact(values):
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


def number(value):
    n = Decimal(str(value))
    if not n.is_finite():
        raise ValueError("non-finite value")
    return n


def stamp(value):
    if isinstance(value, str):
        if not re.fullmatch(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d(?:\.\d{1,6})?)?(?:Z|[+-]\d\d:\d\d)?", value):
            raise ValueError("unsupported entry time")
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise ValueError("missing entry time")
    return value.astimezone(HKT) if value.tzinfo else value.replace(tzinfo=HKT)


def match_day(value):
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d\d-\d\d", value):
        return date.fromisoformat(value)
    if isinstance(value, (str, datetime)):
        dt = stamp(value)
        if (dt.hour, dt.minute, dt.second, dt.microsecond) == (0, 0, 0, 0):
            return dt.date()
    elif isinstance(value, date):
        return value
    raise ValueError("unsupported match date")


def original_mid(value):
    if value is None or (isinstance(value, str) and value == "") or str(value) in ("<NA>", "nan"):
        return None
    s = str(value)
    if not re.fullmatch(r"[1-9]\d{0,11}(?:\.0+)?", s):
        raise ValueError("unsupported original match ID")
    return s.split(".")[0]


def identity(row):
    v = [row.get(k) for k in FIELDS]
    for i in (0, 3, 4, 6, 7):
        if not isinstance(v[i], str) or not v[i]:
            raise ValueError("missing identity fields")
    if v[7] not in MARKETS.get(v[6], ()):
        raise ValueError("unsupported market or side")
    return (*v[:1], stamp(v[1]), match_day(v[2]), *v[3:5], original_mid(v[5]),
            *v[6:8], number(v[8]), number(v[9]))


def evidence_identity(evidence):
    raw = evidence["original_json"]
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) != len(FIELDS) or digest(raw) != evidence["fingerprint"]:
        raise ValueError("invalid original fingerprint")
    key = identity(dict(zip(FIELDS, values, strict=True)))
    if key[0] != evidence["bet_id"] or key[-1] <= 1:
        raise ValueError("invalid original identity")
    return key


def matches(row, expected):
    try:
        return identity(row) == expected
    except (ValueError, TypeError, InvalidOperation, OverflowError):
        return False


def sql_rows(conn):
    """SQLAlchemy connection; no full-ledger, evidence URLs or histories are read."""
    if conn.exec_driver_sql(f"SELECT to_regclass('public.{TABLE}')").scalar() is None:
        return None
    cols = ", ".join(f'b."{k}"' for k in (*FIELDS, *DISPLAY))
    result = conn.exec_driver_sql(f"""
        SELECT e.bet_id, e.original_json, e.fingerprint, e.closing_odds, {cols}
        FROM public.{TABLE} AS e
        LEFT JOIN public.football_bets AS b ON b."ID" = e.bet_id
          AND NOT EXISTS (SELECT 1 FROM public.{TOMBS} t WHERE t.id = b."ID")
        WHERE e.batch_id = '{BATCH}' AND e.source_sha256 = '{SOURCE_SHA}'
    """)
    return [dict(r) for r in result.mappings()]


def overlay(ledger, classified, evidence):
    """Return a display-only ledger and official groups; keep existing snapshot diagnostics independent."""
    if evidence is None:
        return ledger, classified, []
    out, groups = ledger.copy(), classified.copy()
    if not ledger.index.is_unique or "ID" not in ledger.columns:
        return out, groups, []
    evidence_counts = Counter(r["bet_id"] for r in evidence)
    records = ledger.reindex(columns=FIELDS).to_dict("records")
    pos = {r.get("ID"): (i, r) for i, r in zip(ledger.index, records, strict=True)
           if isinstance(r.get("ID"), str)}
    count = ledger["ID"].value_counts()
    visible = []
    # Pandas infers datetime64 for an empty native quote-time column in some sessions.
    groups["close_time"] = groups["close_time"].astype(object)
    for r in evidence:
        bid = r["bet_id"]
        if evidence_counts[bid] != 1 or count.get(bid, 0) != 1 or bid not in pos:
            continue
        i, previous = pos[bid]
        native = groups.at[i, "group"]
        if native in ("follow", "out_of_scope"):
            continue
        try:
            key = evidence_identity(r)
            close = number(r["closing_odds"])
            if close <= 1 or not matches(r, key) or not matches(previous, key):
                continue
        except (ValueError, KeyError, TypeError, InvalidOperation, OverflowError):
            continue
        status = r["Status"]
        if not isinstance(status, str) or status not in ("Open", "Settled"):
            continue
        clv = key[-1] / close - 1
        for k in DISPLAY:
            if k in out:
                out.at[i, k] = r[k]
        visible.append({"登記時間": key[1].strftime("%Y-%m-%d %H:%M"),
                        "賽事": f"{key[3]} 對 {key[4]}", "盤口": key[6], "方向": key[7],
                        "原線": float(key[-2]), "入注價": float(key[-1]), "馬會收盤價": float(close),
                        "CLV%": round(float(clv * 100), 6),
                        "狀態": "已結算" if status == "Settled" else "待結算",
                        "主統計核證來源": "賽前快照" if native == "snapshot" else "歷史馬會核證"})
        if native != "snapshot":
            groups.loc[i, ["group", "clv", "closing", "close_time", "reason"]] = (
                "historical_verified", float(clv), float(close), None, "historical_verified")
    return out, groups, visible
