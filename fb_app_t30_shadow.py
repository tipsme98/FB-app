"""T−30 research only. No network, bet ledger, credentials, or Streamlit dependency."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3

MARKETS = ("讓球", "入球大小", "角球大小")
VERSION = "t30-shadow-rc1"
SOURCE = "Tipsme/HKJC"
HKT = timezone(timedelta(hours=8))
CUTOFF = 30 * 60
WINDOW = 15 * 60
MAX_LAG = 150
SCHEDULE_TTL = 5 * 60
FETCH_GLOBAL_INTERVAL = 25
APP_ID = 0x46425348  # FBSH; refuse to write over an existing, unrelated SQLite file.
NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")
STAMP = re.compile(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d(?:\.\d{1,6})?)?(?:Z|[+-]\d\d:\d\d)?")
SCHEMA = """
CREATE TABLE meta (name TEXT PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE fixtures (
    mid TEXT NOT NULL, ko REAL NOT NULL, home TEXT NOT NULL, away TEXT NOT NULL,
    league TEXT NOT NULL, seen REAL NOT NULL, state TEXT NOT NULL, status TEXT NOT NULL,
    PRIMARY KEY(mid,ko));
CREATE TABLE captures (
    mid TEXT NOT NULL, ko REAL NOT NULL, observed REAL NOT NULL, profile TEXT NOT NULL,
    data TEXT NOT NULL, PRIMARY KEY(mid,ko,observed,profile));
CREATE TABLE decisions (
    mid TEXT NOT NULL, ko REAL NOT NULL, market TEXT NOT NULL, version TEXT NOT NULL,
    decided REAL NOT NULL, reason TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY(mid,ko,market,version));
CREATE TABLE results (
    mid TEXT NOT NULL, ko REAL NOT NULL, market TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY(mid,ko,market));
"""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _iso(ts):
    return datetime.fromtimestamp(ts, HKT).isoformat()


def _identity(v):
    s = str(v) if type(v) in (int, str) else ""
    return s if re.fullmatch(r"[1-9]\d{0,19}", s) else ""


def _text(v):
    return " ".join(v.split()) if isinstance(v, str) and len(v) <= 250 else ""


def _at(data, path):
    for key in path.split("."):
        if type(data) is not dict:
            return None
        data = data.get(key)
    return data


def _path(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", value))


def _same(v, values):
    return any(type(v) is type(candidate) and v == candidate for candidate in values)


def _values(values):
    return isinstance(values, list) and bool(values) and all(type(v) in (str, int, bool) for v in values)


class Profile:
    """Operator-verified authoritative *current* market snapshots, not legacy odds history."""

    def __init__(self, data):
        if (not isinstance(data, dict) or not _text(data.get("evidence"))
                or not _values(data.get("schedule_prematch"))
                or data.get("naive_zone") not in (None, "UTC", "HKT")
                or type(data.get("handicap_sign")) is not int or data["handicap_sign"] not in (-1, 1)
                or not isinstance(data.get("markets"), dict) or set(data["markets"]) != set(MARKETS)
                or not _path(data.get("fixture_id"))):
            raise ValueError("Invalid/unverified T−30 source profile")
        for spec in data["markets"].values():
            if (not isinstance(spec, dict)
                    or not all(_path(spec.get(k)) for k in ("path", "status", "timestamp", "line", "upper", "lower"))
                    or not _values(spec.get("open")) or not _values(spec.get("closed"))
                    or any(_same(v, spec["closed"]) for v in spec["open"])):
                raise ValueError("Invalid T−30 market profile")
        self.data = json.loads(_json(data))
        self.digest = hashlib.sha256(_json(self.data).encode()).hexdigest()

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def quote(self, raw, mid, market, observed, cut):
        if raw is None:
            return {"reason": "odds_unavailable"}
        binding = self.data.get("fixture_id")
        if _identity(_at(raw, binding)) != mid:
            return {"reason": "quote_fixture_mismatch"}
        spec = self.data["markets"][market]
        item = _at(raw, spec["path"])
        if type(item) is not dict:
            return {"reason": "market_missing"}
        status = _at(item, spec["status"])
        if _same(status, spec["closed"]):
            return {"reason": "market_closed", "source_status": status}
        if not _same(status, spec["open"]):
            return {"reason": "market_status_unknown"}
        stamp = _at(item, spec["timestamp"])
        try:
            if not isinstance(stamp, str) or not STAMP.fullmatch(stamp):
                raise ValueError
            t = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if t.tzinfo is None:
                zone = {"UTC": timezone.utc, "HKT": HKT}.get(self.data.get("naive_zone"))
                if zone is None:
                    raise ValueError
                t = t.replace(tzinfo=zone)
            source = t.timestamp()
        except (ValueError, OverflowError, OSError):
            return {"reason": "quote_time_unknown"}
        if source > observed:
            return {"reason": "quote_after_collection", "source_status": status, "source_record_time": stamp}
        if not cut - WINDOW <= source <= cut:
            return {"reason": "quote_stale_or_after_cutoff", "source_status": status, "source_record_time": stamp}
        values = [_at(item, spec[k]) for k in ("line", "upper", "lower")]
        try:
            if not all(type(v) in (int, float, str) and NUMBER.fullmatch(str(v)) for v in values):
                raise ValueError
            line, upper, lower = map(float, values)
            if market == MARKETS[0]:
                line *= self.data["handicap_sign"]
            if (not math.isfinite(line) or abs(line) > 30 or (market != MARKETS[0] and line <= 0)
                    or abs(line * 4 - round(line * 4)) > 1e-7
                    or not (1.01 <= upper <= 20 and 1.01 <= lower <= 20 and .999 <= 1/upper + 1/lower <= 1.45)):
                raise ValueError
        except (ValueError, OverflowError):
            return {"reason": "invalid_two_sided_quote"}
        return dict(reason="eligible", source_status=status, source_record_time=stamp, quote=dict(
            type=market, line=line, upper=upper, lower=lower, record_time=_iso(source), unlock=True))


class HistoryUnavailable(ValueError):
    """The app did not load any eligible history; do not emit a forecast."""


class Store:
    def __init__(self, directory, *, read_only=False):
        root = Path(directory)
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("FB_APP_SHADOW_DIR must be an existing absolute private directory")
        self.path = root / "fb-app-t30-shadow.sqlite3"
        self.read_only = read_only
        if self.path.is_symlink():
            raise ValueError("Shadow database must not be a symbolic link")
        with self._db() as conn:
            app = conn.execute("PRAGMA application_id").fetchone()[0]
            if not read_only and app == 0 and not conn.execute("SELECT 1 FROM sqlite_master").fetchone():
                for statement in SCHEMA.strip().split(";"):
                    if statement.strip():
                        conn.execute(statement)
                conn.execute(f"PRAGMA application_id={APP_ID}")
                conn.execute("PRAGMA user_version=1")
            else:
                self._verify(conn)
        self.ready = True

    @staticmethod
    def _verify(conn):
        if (conn.execute("PRAGMA application_id").fetchone()[0] != APP_ID
                or conn.execute("PRAGMA user_version").fetchone()[0] != 1):
            raise ValueError("Not an FB-app T−30 research database")

    @contextmanager
    def _db(self, *, read_only=False):
        # SQLite operations are short and do not touch the official database.
        read_only = read_only or self.read_only
        target = self.path.as_uri() + "?mode=ro" if read_only else self.path
        conn = sqlite3.connect(target, uri=read_only, isolation_level=None, timeout=.3)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN" if read_only else "BEGIN IMMEDIATE")
            if getattr(self, "ready", False):
                self._verify(conn)
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _meta(conn, key):
        row = conn.execute("SELECT value FROM meta WHERE name=?", (key,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def _set(conn, key, value):
        conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def reserve(self, key, now, seconds):
        with self._db() as conn:
            prior = self._meta(conn, key)
            if prior is not None and now - prior < seconds:
                return False
            self._set(conn, key, now)
            return True

    def schedule_health(self, now):
        with self._db(read_only=True) as conn:
            observed = self._meta(conn, "feed_observed")
            if observed is None:
                return "waiting", None
            if not self._meta(conn, "feed_ok") or not 0 <= now - observed <= SCHEDULE_TTL:
                return "unavailable", _iso(observed)
            return "loaded", _iso(observed)

    @staticmethod
    def _fixture(row):
        if not isinstance(row, dict):
            return None
        mid = _identity(row.get("hkjc_match_id"))
        home, away, league = (_text(row.get(k)) for k in ("home", "away", "league"))
        ko = row.get("match_time")
        if (not mid or not home or not away or home.casefold() == away.casefold()
                or type(ko) not in (int, float) or not math.isfinite(ko) or not 946684800 <= ko <= 4102444799):
            return None
        status = row.get("shadow_status")
        return mid, ko, home, away, league, status if type(status) in (str, int, bool) else None

    def ingest_schedule(self, rows, observed):
        # A failed refresh revokes freshness; never bless the legacy stale fallback.
        with self._db() as conn:
            prior = self._meta(conn, "feed_observed")
            if prior is not None and prior > observed:
                return
            self._set(conn, "feed_observed", observed)
            self._set(conn, "feed_ok", int(bool(rows)))
            if not rows:
                return
            by_id = {}
            for row in rows:
                f = self._fixture(row)
                if f is not None and row.get("shadow_is_hkjc") is True:
                    by_id.setdefault(f[0], set()).add((*f[:-1], _json(f[-1]),
                                                     bool(row.get("shadow_kickoff_conflict"))))
            for mid, candidates in by_id.items():
                if len(candidates) != 1 or any(entry[-1] for entry in candidates):
                    conn.execute("UPDATE fixtures SET state='schedule_conflict', seen=? "
                                 "WHERE mid=? AND state NOT IN ('rescheduled','identity_conflict')",
                                 (observed, mid))
                    for _, ko, h, a, league, status, _ in candidates:
                        if observed - 5*60 <= ko - CUTOFF <= observed + WINDOW + 5*60:
                            conn.execute("INSERT OR IGNORE INTO fixtures VALUES (?,?,?,?,?,?,?,?)",
                                         (mid, ko, h, a, league, observed, "schedule_conflict", status))
                    continue
                _, ko, h, a, league, status, _ = next(iter(candidates))
                old = conn.execute("SELECT * FROM fixtures WHERE mid=? AND ko=?", (mid, ko)).fetchone()
                # Remember a known fixture's reschedule even if its new kickoff is hours away.
                conn.execute("UPDATE fixtures SET state='rescheduled' WHERE mid=? AND ko<>?", (mid, ko))
                if not old and not observed - 5*60 <= ko - CUTOFF <= observed + WINDOW + 5*60:
                    continue
                if old and (old["home"], old["away"], old["league"]) != (h, a, league):
                    conn.execute("UPDATE fixtures SET state='identity_conflict' WHERE mid=? AND ko=?", (mid, ko))
                elif old and old["state"] in ("rescheduled", "identity_conflict"):
                    continue
                else:
                    conn.execute("INSERT OR REPLACE INTO fixtures VALUES (?,?,?,?,?,?,?,?)",
                                 (mid, ko, h, a, league, observed, "scheduled", status))

    def _schedule_reason(self, conn, fixture, now, profile):
        if fixture["state"] != "scheduled":
            return fixture["state"]
        observed = self._meta(conn, "feed_observed")
        if (not self._meta(conn, "feed_ok") or observed is None or fixture["seen"] != observed
                or not 0 <= now - observed <= SCHEDULE_TTL):
            return "schedule_not_fresh"
        if not fixture["league"]:
            return "fixture_details_missing"
        if profile is None:
            return "source_unverified"
        if not _same(json.loads(fixture["status"]), profile.data["schedule_prematch"]):
            return "fixture_not_verified_prematch"
        return None

    def next_fetch(self, now, profile):
        if profile is None:
            return None
        with self._db() as conn:
            global_last = self._meta(conn, "fetch-global")
            if global_last is not None and now - global_last < FETCH_GLOBAL_INTERVAL:
                return None
            candidates = conn.execute("SELECT * FROM fixtures WHERE ko>? AND ko<=? ORDER BY ko,mid",
                                      (now+CUTOFF, now+CUTOFF+WINDOW))
            for f in candidates:
                if self._schedule_reason(conn, f, now, profile):
                    continue
                key = f"fetch:{f['mid']}:{f['ko']}"
                last = self._meta(conn, key)
                last_observed = conn.execute(
                    "SELECT MAX(observed) FROM captures WHERE mid=? AND ko=? AND profile=?",
                    (f["mid"], f["ko"], profile.digest)).fetchone()[0]
                seen = [v for v in (last, last_observed) if v is not None]
                # Reuse a production capture; refresh more closely in the last 2 min.
                period = 45 if f["ko"] - CUTOFF - now <= 120 else 240
                if seen and now - max(seen) < period:
                    continue
                self._set(conn, key, now)
                self._set(conn, "fetch-global", now)
                return f["mid"]
        return None

    def capture(self, mid, raw, observed, profile):
        if profile is None or not (mid := _identity(mid)):
            return False
        with self._db() as conn:
            rows = conn.execute("SELECT * FROM fixtures WHERE mid=? AND ko>=? AND ko<=?",
                                (mid, observed+CUTOFF, observed+CUTOFF+WINDOW)).fetchall()
            rows = [f for f in rows if not self._schedule_reason(conn, f, observed, profile)]
            if len(rows) != 1:
                return False
            ko = rows[0]["ko"]
            captured = {m: profile.quote(raw, mid, m, observed, ko-CUTOFF) for m in MARKETS}
            # Ambiguous same-clock inputs are not resolved by arbitrary insertion order.
            key = (mid, ko, observed, profile.digest)
            encoded = _json(captured)
            old = conn.execute("SELECT data FROM captures WHERE mid=? AND ko=? AND observed=? AND profile=?",
                               key).fetchone()
            if old and old[0] != encoded:
                encoded = _json({m: {"reason": "quote_conflict"} for m in MARKETS})
            conn.execute("INSERT OR REPLACE INTO captures VALUES (?,?,?,?,?)", (*key, encoded))
            return True

    def run_due(self, clock, profile, score, limit=2):
        """score(fixture, aware_now) returns (engine rows, provenance). No network under this lock."""
        with self._db() as conn:
            now = clock()
            due = conn.execute(
                "SELECT * FROM fixtures f WHERE f.ko<=? AND NOT EXISTS "
                "(SELECT 1 FROM decisions d WHERE d.mid=f.mid AND d.ko=f.ko AND d.version=?) "
                "ORDER BY (f.ko<?),f.ko,f.mid LIMIT ?", (now+CUTOFF, VERSION, now+CUTOFF-MAX_LAG, limit)
            ).fetchall()
            for f in due:
                self._decide(conn, f, clock, profile, score)
            return len(due)

    def _decide(self, conn, f, clock, profile, score):
        now = clock()
        cut = f["ko"] - CUTOFF
        global_reason = self._schedule_reason(conn, f, now, profile)
        if f["state"] == "scheduled" and now - cut > MAX_LAG:
            global_reason = "decision_window_missed"
        latest, offers = None, {}
        historical = []
        if not global_reason:
            history = conn.execute(
                "SELECT * FROM captures WHERE mid=? AND ko=? AND profile=? AND observed BETWEEN ? AND ? "
                "ORDER BY observed DESC", (f["mid"], f["ko"], profile.digest, cut-WINDOW, cut)
            ).fetchall()
            if history:
                latest = history[0]
                offers = json.loads(latest["data"])
                # Distinct source times only; conflicting values at the same source time
                # must not cause the engine to fall back or silently use an older quote.
                for m, offer in offers.items():
                    if "quote" not in offer:
                        continue
                    trail = {}
                    for entry in reversed(history):
                        item = json.loads(entry["data"]).get(m, {})
                        if "quote" not in item:
                            trail.clear()
                            continue
                        q = item["quote"]
                        stamp = q["record_time"]
                        token = _json(q)
                        trail.setdefault(stamp, set()).add(token)
                    newest = offer["quote"]["record_time"]
                    if len(trail.get(newest, ())) != 1 or any(t > newest for t in trail):
                        offers[m] = {"reason": "quote_conflict_or_time_regression"}
                    else:
                        historical.extend(json.loads(next(iter(s))) for t, s in sorted(trail.items())
                                          if len(s) == 1 and t <= newest)
        predictions, provenance, engine_reason = {}, {}, "training_or_model_unavailable"
        if historical:
            fixture = dict(fixture_id=f["mid"], home=f["home"], away=f["away"], league=f["league"],
                           kickoff=_iso(f["ko"]), odds=historical)
            started = clock()
            if cut <= started <= cut+MAX_LAG:
                try:
                    output, provenance = score(fixture, datetime.fromtimestamp(started, HKT))
                    eligible = {m for m, o in offers.items() if "quote" in o}
                    predictions = {r["盤口"]: r for r in output if r["盤口"] in eligible}
                    if len(predictions) != len(output):
                        raise ValueError("Unrecognized or duplicate model result")
                    for m, r in predictions.copy().items():
                        q = offers[m]["quote"]
                        if ((r["主盤"], r["主或大賠率"], r["客或小賠率"], r["報價時間"])
                                != (q["line"], q["upper"], q["lower"], q["record_time"])):
                            raise ValueError("Engine did not use the latest locked quote")
                        if (r["預測方向"] not in (("主", "客") if m == MARKETS[0] else ("大", "小"))
                                or r["建議"] not in ("略過", "影子模擬", "對照訊號")):
                            raise ValueError("Unexpected model direction or decision")
                    _json([predictions, provenance])
                except HistoryUnavailable:
                    predictions, provenance, engine_reason = {}, {}, "training_or_model_unavailable"
                except (ValueError, RuntimeError, KeyError, TypeError, ArithmeticError):
                    predictions, provenance, engine_reason = {}, {}, "engine_rejected_inputs"
            else:
                engine_reason = "decision_window_missed"
        finished = clock()
        if finished - cut > MAX_LAG and historical:
            predictions, engine_reason = {}, "decision_window_missed"
        for market in MARKETS:
            offer = offers.get(market, {})
            pred = predictions.get(market)
            reason = global_reason or ({"略過": "forecast_no_pick", "對照訊號": "forecast_reference",
                                        "影子模擬": "forecast_pick"}[pred["建議"]]
                                       if pred else (engine_reason if "quote" in offer
                                                     else offer.get("reason", "no_cutoff_capture")))
            payload = dict(quote=offer.get("quote"), forecast=pred, provenance=provenance,
                           source=SOURCE, source_profile=profile.digest if profile else None,
                           source_status=offer.get("source_status"), source_record_time=offer.get("source_record_time"),
                           collected_at=_iso(latest["observed"]) if latest else None, cutoff=_iso(cut),
                           decided_at=_iso(finished), decision_lag_s=round(finished-cut, 3),
                           timing="at_cutoff" if finished == cut else "delayed")
            conn.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?)",
                         (f["mid"], f["ko"], market, VERSION, finished, reason, _json(payload)))

    def ingest_results(self, rows, observed, source="Tipsme finished results (complete date read)"):
        """Exact ID/kickoff/teams, with market-specific missing results and conflict quarantine."""
        with self._db() as conn:
            count = 0
            for row in rows:
                f = self._fixture(row)
                if f is None or row.get("finished") is not True:
                    continue
                mid, ko, h, a, _, _ = f
                original = conn.execute("SELECT * FROM fixtures WHERE mid=? AND ko=?", (mid, ko)).fetchone()
                if (not original or original["state"] != "scheduled" or observed <= ko
                        or (h, a) != (original["home"], original["away"])):
                    continue
                for m in MARKETS:
                    pair = ("home_corner", "away_corner") if m == MARKETS[2] else ("home_goal", "away_goal")
                    hv, av = (row.get(k) for k in pair)
                    if not all(type(n) is int and 0 <= n <= 80 for n in (hv, av)):
                        continue
                    data = dict(home=hv, away=av, source=source, verified_at=_iso(observed), state="verified")
                    old = conn.execute("SELECT data FROM results WHERE mid=? AND ko=? AND market=?",
                                       (mid, ko, m)).fetchone()
                    if old:
                        saved = json.loads(old[0])
                        if saved["state"] == "verified" and (saved["home"], saved["away"]) != (hv, av):
                            saved.update(state="conflict", conflicting_home=hv, conflicting_away=av,
                                         conflict_at=_iso(observed))
                            conn.execute("UPDATE results SET data=? WHERE mid=? AND ko=? AND market=?",
                                         (_json(saved), mid, ko, m))
                            count += 1
                    else:
                        conn.execute("INSERT INTO results VALUES (?,?,?,?)", (mid, ko, m, _json(data)))
                        count += 1
            return count

    def rows(self, *, recent=None):
        """Read a consistent snapshot; a UI read never acquires a writer lock."""
        with self._db(read_only=True) as conn:
            sql = """SELECT d.*, f.home,f.away,f.league,f.state AS fixture_state,r.data AS result
                     FROM decisions d JOIN fixtures f ON f.mid=d.mid AND f.ko=d.ko
                     LEFT JOIN results r ON r.mid=d.mid AND r.ko=d.ko AND r.market=d.market"""
            if recent is None:
                sql += " ORDER BY d.ko,d.mid,d.market"
                args = ()
            else:
                # Limit by fixtures, not by rows, so the oldest displayed fixture still has all three markets.
                sql += """ WHERE (d.mid,d.ko) IN
                           (SELECT mid,ko FROM decisions GROUP BY mid,ko ORDER BY ko DESC,mid LIMIT ?)
                           ORDER BY d.ko DESC,d.mid,CASE d.market WHEN '讓球' THEN 1
                           WHEN '入球大小' THEN 2 ELSE 3 END"""
                args = (recent,)
            for row in conn.execute(sql, args):
                d = json.loads(row["data"])
                result = json.loads(row["result"]) if row["result"] else None
                pred, q = d.pop("forecast"), d.pop("quote")
                provenance = d.pop("provenance")
                settled = {}
                if result and pred and q and result["state"] == "verified" and row["fixture_state"] == "scheduled":
                    settled = settlement(row["market"], q, pred["預測方向"], result["home"], result["away"])
                yield dict(fixture_id=row["mid"], kickoff=_iso(row["ko"]), market=row["market"],
                           home=row["home"], away=row["away"], league=row["league"],
                           fixture_state=row["fixture_state"], version=row["version"], reason=row["reason"],
                           **d, **{f"quote_{k}": v for k, v in (q or {}).items()},
                           **{f"forecast_{k}": v for k, v in (pred or {}).items()},
                           **{f"provenance_{k}": v for k, v in provenance.items()},
                           **{f"result_{k}": v for k, v in (result or {"state": "pending"}).items()},
                           hypothetical_units=settled.get("hypothetical_units"),
                           outcome=settled.get("outcome"),
                           strategy_units=(settled.get("hypothetical_units") if pred and pred["建議"] == "影子模擬" else None))


def decisions_csv(rows):
    """BOM for Excel and neutralise spreadsheet formulas in provider text."""
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["fixture_id", "kickoff", "market", "reason"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows({k: ("'" + v if isinstance(v, str) and (v.startswith(("\t", "\r", "\n"))
                           or v.lstrip().startswith(("=", "+", "-", "@")))
                          else v) for k, v in r.items()} for r in rows)
    return stream.getvalue().encode("utf-8-sig")


def settlement(market, q, direction, h, a):
    """Unit return for either direction using the exact recorded Asian line."""
    diff = h-a+q["line"] if market == MARKETS[0] else h+a-q["line"]
    upper = direction == ("主" if market == MARKETS[0] else "大")
    if not upper and direction != ("客" if market == MARKETS[0] else "小"):
        raise ValueError("Unknown direction")
    effective_diff = diff if upper else -diff
    w, l = min(1, max(0, 2*effective_diff)), min(1, max(0, -2*effective_diff))
    state = {(1, 0): "全贏", (.5, 0): "半贏", (0, 0): "走盤", (0, .5): "半輸", (0, 1): "全輸"}[(w, l)]
    return dict(outcome=state, hypothetical_units=round(w*((q["upper"] if upper else q["lower"])-1)-l, 8))


def main():
    parser = argparse.ArgumentParser(description="Export local FB-app T−30 research decisions and results")
    parser.add_argument("directory", help="existing absolute shadow directory")
    parser.add_argument("output", help="new CSV destination (will not overwrite)")
    args = parser.parse_args()
    if not (Path(args.directory) / "fb-app-t30-shadow.sqlite3").is_file():
        parser.error("No existing shadow database in this directory")
    rows = list(Store(args.directory, read_only=True).rows())
    with open(args.output, "xb") as f:
        f.write(decisions_csv(rows))


if __name__ == "__main__":
    main()
