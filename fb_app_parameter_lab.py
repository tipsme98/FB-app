"""FB-app parameter lab. Pure research; never reads networks or writes bets.

The workbook adapter deliberately does not use stake, selection, model or P/L
columns. Only fixture identity, two-sided snapshots, prior results and scores.
"""
import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import poisson, skellam
from sklearn.linear_model import PoissonRegressor
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits


MARKETS = ("讓球", "入球大小", "角球大小")
GROUPS = ("賠率走勢", "逐場近況攻守", "長期球隊實力", "聯賽及對賽")
SIZES = (4, 6, 5, 5)
MODELS = {"LR": (10, 60), "BOOST": (1, 2), "POISSON": (0.35, 0.65)}
THRESHOLDS = (0., .02, .04, .06)
MIN_BETS, MIN_FOLD = 30, 5
LAB_VERSION = "parameter-lab-1"
HKT = "Asia/Hong_Kong"
DAY = pd.Timedelta(days=1)


def _text(value):
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _object(value, typ):
    if isinstance(value, typ):
        return value
    if isinstance(value, str):
        try:
            v = json.loads(value)
            return v if isinstance(v, typ) else typ()
        except (ValueError, RecursionError):
            pass
    return typ()


def _day(value):
    if not isinstance(value, (str, datetime, pd.Timestamp)) or pd.isna(value):
        return None
    try:
        d = pd.Timestamp(value)
        if pd.isna(d):
            return None
        if d.tzinfo is not None:
            d = d.tz_convert(HKT).tz_localize(None)
        return d.normalize()
    except (ValueError, TypeError, OverflowError):
        return None


def _number(v):
    try:
        n = float(v)
        return n if np.isfinite(n) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _hkt_timestamp(value, require_zone=False):
    if not isinstance(value, str) or not re.match(r"^\d{4}-\d\d-\d\d[T ]\d\d:\d\d", value):
        return None
    if require_zone and not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d(?::\d\d(?:\.\d{1,6})?)?(?:Z|[+-]\d\d:\d\d)", value):
        return None
    try:
        t = pd.Timestamp(value)
        if not pd.notna(t) or (require_zone and t.tzinfo is None):
            return None
        return t.tz_convert(HKT).tz_localize(None) if t.tzinfo else t
    except (TypeError, ValueError, OverflowError):
        return None


def _historical_times(history, day, strict=False):
    """Infer missing year only near the fixture; flag unknown day/month order."""
    pattern = re.compile(r"(\d{1,2})[-/.](\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?")
    short = [pattern.fullmatch(str(h.get("record_time", "")).strip()) for h in history]
    dm = any(m and int(m[1]) > 12 for m in short)
    md = any(m and int(m[2]) > 12 for m in short)
    for h, m in zip(history, short):
        t = _hkt_timestamp(h.get("record_time"), require_zone=strict)
        if t is not None or not m or strict:
            yield t
            continue
        x, y, hr, mn = map(int, m.groups()[:4])
        f = [(x, y)] if md and not dm else [(y, x)] if dm and not md else [(x, y), (y, x)]
        dates = set()
        for month, d in f:
            for year in (day.year - 1, day.year, day.year + 1):
                try:
                    t = pd.Timestamp(datetime(year, month, d, hr, mn, int(m[5] or 0)))
                    if day - 14 * DAY <= t < day + 2 * DAY:
                        dates.add(t)
                except (ValueError, OverflowError):
                    pass
        yield next(iter(dates)) if len(dates) == 1 else None


def _raw_fixtures(raw):
    required = {"Status", "Match_Date", "Home_Team", "Away_Team", "Home_Goal",
                "Away_Goal", "Odds_History"}
    if not required.issubset(raw.columns):
        raise ValueError("缺少歷史欄位：" + "、".join(sorted(required.difference(raw.columns))))
    groups = {}
    conflicts = 0
    cols = sorted(required | {"Home_Corner", "Away_Corner", "Home_Recent_Matches",
                             "Away_Recent_Matches", "Tournament_Name", "Bet_Type"})
    for q in raw.reindex(columns=cols).to_dict("records"):
        if not isinstance(q["Status"], str) or q["Status"] != "Settled":
            continue
        d, h, a = _day(q["Match_Date"]), _text(q["Home_Team"]), _text(q["Away_Team"])
        if d is None or not h or not a or h == a:
            continue
        key = (d, h, a)
        r = groups.setdefault(key, {"date": d, "home": h, "away": a, "scores": [set() for _ in range(4)],
                                   "key": "|".join((d.strftime("%Y-%m-%d"), h, a)),
                                   "tourn": set(), "odds": {}, "recent": [{}, {}]})
        league = _text(q["Tournament_Name"])
        if league:
            r["tourn"].add(league)
        for k, name in enumerate(("Home_Goal", "Away_Goal", "Home_Corner", "Away_Corner")):
            v = _number(q[name])
            if np.isfinite(v) and v == round(v) and 0 <= v <= 80:
                r["scores"][k].add(v)
        # Exclude rows explicitly designated in-play from the quote/feature pool.
        if "即場" in _text(q["Bet_Type"]):
            continue
        for store, name in zip(r["recent"], ("Home_Recent_Matches", "Away_Recent_Matches")):
            for g in _object(q[name], list):
                if isinstance(g, dict):
                    store[json.dumps(g, sort_keys=True, default=str, ensure_ascii=False)] = g
        for g in _object(q["Odds_History"], list):
            if isinstance(g, dict):
                fields = {k: g.get(k) for k in ("record_time", "type", "line", "upper", "lower", "unlock", "main")}
                r["odds"][json.dumps(fields, sort_keys=True, default=str, ensure_ascii=False)] = fields
    out = []
    for k, r in sorted(groups.items()):
        if any(len(v) > 1 for v in r["scores"][:2]):
            conflicts += 1
            continue
        if not all(r["scores"][:2]):
            continue
        cs = sum(len(v) > 1 for v in r["scores"][2:])
        scores = tuple(next(iter(v)) if len(v) == 1 else np.nan for v in r["scores"])
        r.update(scores=scores, tourn=next(iter(r["tourn"])) if len(r["tourn"]) == 1 else "",
                 odds=list(r["odds"].values()), recent=[list(v.values()) for v in r["recent"]],
                 corner_conflict=bool(cs))
        out.append(r)
    return out, {"identity_matches": len(groups), "result_conflicts_skipped": conflicts,
                 "scored_matches": len(out), "corner_conflicts": sum(r["corner_conflict"] for r in out)}


def _market_quotes(history, day, cutoff, strict=False):
    by = defaultdict(set)
    latest = {}
    data = [h for h in history if isinstance(h, dict)]
    for g, t in zip(data, _historical_times(data, day, strict)):
        bt = g.get("type")
        if not isinstance(bt, str) or bt not in MARKETS or t is None or t > cutoff:
            continue
        latest[bt] = max(latest.get(bt, t), t)
        if strict and g.get("unlock") is not True:
            continue
        if g.get("unlock") in (False, "false", "False", "0"):
            continue
        line, u, l = map(_number, (g.get("line"), g.get("upper"), g.get("lower")))
        if (not np.isfinite(line) or abs(line) > 30 or abs(line * 4 - round(line * 4)) > 1e-7
                or not (1.01 <= u <= 20 and 1.01 <= l <= 20) or not .999 <= 1/u + 1/l <= 1.45):
            continue
        by[(bt, t, line)].add((u, l, g.get("main") is True))
    schedules = defaultdict(lambda: defaultdict(list))
    for (bt, t, ln), choices in by.items():
        odds = {(u, l) for u, l, _ in choices}
        if len(odds) != 1:
            continue
        up, lo = next(iter(odds))
        p = lo / (up + lo)
        schedules[bt][t].append((ln, up, lo, p, any(c[2] for c in choices)))
    out = {}
    for bt, times in schedules.items():
        timeline = [(t, min(times[t], key=lambda v: (not v[4], abs(v[3] - .5), v[0])))
                    for t in sorted(times)]
        first_t, (first_ln, _, _, first_p, _) = timeline[0]
        last_t, (line, up, lo, p, _) = timeline[-1]
        # Historical time is a date proxy. Live offers must be fresh at T−30.
        if cutoff - last_t > (pd.Timedelta(minutes=15) if strict else 2 * DAY) or (strict and last_t != latest[bt]):
            continue
        span = max(0., (last_t - first_t).total_seconds() / 3600)
        out[bt] = dict(line=line, upper=up, lower=lo, market=p, t=last_t,
                       moves=[p - first_p, line - first_ln, np.std([v[3] for _, v in timeline]), math.log1p(span)])
    return out


def _fractions(diff):
    d = np.asarray(diff, dtype=float)
    return np.clip(2*d, 0, 1), np.clip(-2*d, 0, 1)


def _past_context(fixture, known, cutoff):
    """Only results dated strictly before the one-calendar-day embargo."""
    bt = defaultdict(list)
    league = []
    h2h = []
    h, a = fixture["home"], fixture["away"]
    for q in known:
        if q["date"] >= cutoff:
            break
        hg, ag, hc, ac = q["scores"]
        if q["tourn"] and q["tourn"] == fixture["tourn"]:
            league.append([hg, ag, hc, ac])
        for team in (h, a):
            if team == q["home"]:
                bt[team].append((q["date"], q["away"], hg, ag, hc, ac))
            elif team == q["away"]:
                bt[team].append((q["date"], q["home"], ag, hg, ac, hc))
        if q["home"] == h and q["away"] == a:
            h2h.append([hg, ag, hc, ac])
        elif q["home"] == a and q["away"] == h:
            h2h.append([ag, hg, ac, hc])
    return bt, league, h2h


def _external_recent(items, team, cutoff):
    matches = {}
    for g in items:
        if not isinstance(g, dict):
            continue
        d, h, a = _day(g.get("date")), _text(g.get("home")), _text(g.get("away"))
        if d is None or d >= cutoff or not h or not a or (team == h) == (team == a):
            continue
        vals = []
        for field in ("score", "corner"):
            m = re.fullmatch(r"\s*(\d{1,2})\s*[-:]\s*(\d{1,2})\s*", str(g.get(field)))
            g1, g2 = map(float, m.groups()) if m else (np.nan, np.nan)
            vals.extend([g1, g2] if team == h else [g2, g1])
        key = (d, a if team == h else h)
        # Different copies may disagree. Drop the fixture if both give a value
        # and they conflict; otherwise retain the more complete observations.
        if key in matches and matches[key] is None:
            continue
        if key in matches:
            prev = np.array(matches[key])
            v = np.array(vals)
            if (np.isfinite(prev) & np.isfinite(v) & (prev != v)).any():
                matches[key] = None
                continue
            vals = np.where(np.isfinite(prev), prev, v).tolist()
        matches[key] = vals
    return [(d, opp, *v) for (d, opp), v in matches.items()
            if v is not None and (np.isfinite(v[:2]).all() or np.isfinite(v[2:]).all())]


def _team_form(records, day, offset, prior, n, halflife):
    selected = sorted((r for r in records if np.isfinite(r[2+offset:4+offset]).all()),
                      key=lambda r: (r[0], r[1]), reverse=True)[:n]
    if not selected:
        return float(prior), float(prior), 0
    w = np.exp2(-np.array([(day-r[0]).days for r in selected], float) / halflife)
    fg, fa = (np.array([r[2+offset:4+offset] for r in selected]).T @ w + 2 * prior) / (w.sum() + 2)
    return fg, fa, len(selected)


def _groups_for(bt, f, q, context):
    teams, league, h2h = context
    h, a, d = f["home"], f["away"], f["date"]
    offset = 2 if bt == "角球大小" else 0
    prior = 4.8 if offset else 1.35

    def explore(team, raw_recent):
        base = teams[team]
        # Same match from two sources must not be counted twice. Internal
        # concluded results take precedence over the imported recent array.
        old = {(r[0], r[1]): r for r in raw_recent}
        for r in base:
            old[(r[0], r[1])] = r
        return (_team_form(old.values(), d, offset, prior, 10, 45),
                _team_form(base, d, offset, prior, 35, 110))

    H, HL = explore(h, _external_recent(f["recent"][0], h, d - DAY))
    A, AL = explore(a, _external_recent(f["recent"][1], a, d - DAY))
    recent = [H[0], H[1], A[0], A[1], math.log1p(H[2]), math.log1p(A[2])]
    long = [HL[0], HL[1], AL[0], AL[1], math.log1p(HL[2] + AL[2])]

    def shr(rows, strength):
        y = np.asarray([g[offset:offset+2] for g in rows if np.isfinite(g[offset:offset+2]).all()]).reshape(-1, 2)
        return (y.sum(axis=0) + strength * prior) / (len(y) + strength), len(y)
    lg, nl = shr(league, 10)
    hh, nh = shr(h2h, 5)
    ctx = [lg[0], lg[1], math.log1p(nl), hh[0] - hh[1], (hh[0] + hh[1] - 2*prior) * bool(nh)]
    return (q["moves"], recent, long, ctx)


def _fixture_rows(f, known, cutoff, strict=False, add_scores=True):
    markets = _market_quotes(f["odds"], f["date"], cutoff, strict)
    context = _past_context(f, known, f["date"] - DAY)
    rows = []
    for bt, q in markets.items():
        groups = _groups_for(bt, f, q, context)
        upper, lower, line = q["upper"], q["lower"], q["line"]
        row = dict(market=bt, date=f["date"], key=f["key"], line=line, p=q["market"],
                   upper=upper, lower=lower, time=q["t"], shape=int(round(line * 4)) % 4,
                   base=[logit(q["market"]), 1/upper+1/lower-1, line],
                   features=[x for vals in groups for x in vals])
        if add_scores:
            h, a = f["scores"][2:] if bt == "角球大小" else f["scores"][:2]
            if not np.isfinite([h, a]).all():
                continue
            diff = h - a + line if bt == "讓球" else h + a - line
            w, l = map(float, _fractions(diff))
            row.update(h=h, a=a, w=w, l=l, rhi=w*(upper-1)-l, rlo=l*(lower-1)-w)
        rows.append(row)
    return rows


def get_dataset(raw):
    fixtures, coverage = _raw_fixtures(raw)
    rows = []
    for f in fixtures:
        rows.extend(_fixture_rows(f, fixtures, f["date"] - pd.Timedelta(seconds=1)))
    coverage.update(usable_matches=len({r["key"] for r in rows}), market_rows=len(rows),
                    quote_cutoff="賽日 00:00 前，最近報價不超過 48 小時")
    return pd.DataFrame(rows), fixtures, coverage


def _specs():
    return [(family, v, mask, ev) for family, ls in MODELS.items() for v in ls
            for mask in range(16) for ev in THRESHOLDS]


def _feature_matrix(B, mask):
    cols = []
    i = 0
    for n, size in enumerate(SIZES):
        if mask & (1 << n):
            cols.extend(range(i, i+size))
        i += size
    b = np.array(B.base.to_list(), dtype=float)
    return np.column_stack((b, np.array(B.features.to_list(), dtype=float)[:, cols]))


def _predict(tr, te, market, family, value, mask):
    ax, az = _feature_matrix(tr, mask), _feature_matrix(te, mask)
    center, scale = ax.mean(axis=0), ax.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1)
    x, z = (ax-center)/scale, np.clip((az-center)/scale, -8., 8.)
    anchor, target_anchor = logit(tr.p.to_numpy()), logit(te.p.to_numpy())
    w, l = tr.w.to_numpy(), tr.l.to_numpy()
    used = w+l
    sm = pd.Series(used).groupby(tr["shape"].to_numpy()).sum()
    nn = tr.groupby("shape").size()
    typical = float(used.mean())
    action = te["shape"].map((sm+10*typical)/(nn+10)).fillna(typical).to_numpy()
    if family == "LR":
        x, z = np.c_[np.ones(len(x)), x], np.c_[np.ones(len(z)), z]
        penalty = np.array([max(4., value/2)] + [value] * (x.shape[1]-1))
        def objective(beta):
            eta = anchor + x@beta
            return (np.dot(used, np.logaddexp(0., eta)) - np.dot(w, eta) + .5*np.dot(penalty, beta**2),
                    x.T@(used*expit(eta)-w)+penalty*beta)
        fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
        if not fit.success:
            raise RuntimeError(f"LR: {fit.message}")
        p = expit(target_anchor + z@fit.x)
    elif family == "BOOST":
        eta, pred = anchor.copy(), target_anchor.copy()
        for i in range(30):
            q = expit(eta)
            h = np.maximum(q*(1-q), .04)*used
            res = np.divide(w-used*q, h, out=np.zeros_like(h), where=h>0)
            tree = DecisionTreeRegressor(max_depth=int(value), min_samples_leaf=24, random_state=i)
            tree.fit(x, res, sample_weight=h)
            eta += .05*np.clip(tree.predict(x), -1.2, 1.2)
            pred += .05*np.clip(tree.predict(z), -1.2, 1.2)
        p = expit(pred)
    elif family == "POISSON":
        lh = PoissonRegressor(alpha=1.0, max_iter=250).fit(x, tr.h).predict(z).clip(.03, 30.)
        la = PoissonRegressor(alpha=1.0, max_iter=250).fit(x, tr.a).predict(z).clip(.03, 30.)
        count = np.arange(-60 if market == "讓球" else 0, 121)
        diff = count[None, :] + te.line.to_numpy()[:, None] if market == "讓球" else count[None, :] - te.line.to_numpy()[:, None]
        mass = (skellam.pmf(count[None, :], lh[:, None], la[:, None]) if market == "讓球"
                else poisson.pmf(count[None, :], (lh+la)[:, None]))
        mass /= mass.sum(axis=1)[:, None]
        fw, fl = _fractions(diff)
        won, lost = (mass*fw).sum(axis=1), (mass*fl).sum(axis=1)
        # Blend conditional win-balance on the market logit scale.
        p = expit((1-value)*target_anchor + value*logit((won/(won+lost)).clip(.001, .999)))
        action = (1-value)*action + value*(won+lost)
    else:
        raise ValueError("未知模型")
    return p.clip(.001, .999), action


def _sample_check(tr):
    return len(tr) >= 65 and float(tr.w.sum()) > 5 and float(tr.l.sum()) > 5


def _sides(te, p, action):
    hi, lo = action*(p*te.upper.to_numpy()-1), action*((1-p)*te.lower.to_numpy()-1)
    return hi >= lo, np.maximum(hi, lo)


def _metrics(d, p, action, ev, fold=None, intervals=False):
    high, best = _sides(d, p, action)
    chosen = best >= ev
    a, b = d.rhi.to_numpy(), d.rlo.to_numpy()
    ret, random = np.where(high, a, b)[chosen], ((a+b)/2)[chosen]
    wins, losses = d.w.to_numpy(), d.l.to_numpy()
    denom = (wins+losses).sum()
    raw = d.p.to_numpy()
    def loss(v):
        return float(-(wins @ np.log(v)+losses @ np.log1p(-v))/denom) if denom else None
    run = np.r_[0, ret.cumsum()]
    n = int(chosen.sum())
    out = dict(n=n, net=float(ret.sum()), roi=float(ret.mean()) if n else None,
               pair_roi=float(random.mean()) if n else None, gap=float((ret-random).mean()) if n else None,
               logloss=loss(p), raw_logloss=loss(raw), max_drawdown=float((np.maximum.accumulate(run)-run).max()))
    if fold is not None:
        out["fold_n"] = [int((chosen & (fold==i)).sum()) for i in range(4)]
        inc = np.where(high, a, b)
        out["fold_net"] = [float(inc[chosen & (fold==i)].sum()) for i in range(4)]
        out["eligible"] = n >= MIN_BETS and min(out["fold_n"]) >= MIN_FOLD
    if intervals and n:
        # Moving-block inference is not warranted on these sparse dated bets.
        # Resample whole Hong Kong match dates, keeping within-day dependence.
        g = pd.DataFrame(dict(d=d.date.to_numpy()[chosen], r=ret, gap=ret-random, n=1)).groupby("d").sum()
        vec = g[["r", "gap", "n"]].to_numpy()
        rng = np.random.default_rng(1649)
        runs = vec[rng.integers(0,len(vec),(2500,len(vec)))].sum(axis=1)
        out["roi_ci"] = np.quantile(runs[:,0]/runs[:,2],[.025,.975]).tolist()
        out["gap_ci"] = np.quantile(runs[:,1]/runs[:,2],[.025,.975]).tolist()
        out["n_days"] = len(g)
    return out


def describe(family, value, mask, ev):
    return dict(model=family, parameter=value, mask=int(mask), threshold=ev,
                groups=[name for n, name in enumerate(GROUPS) if mask & (1 << n)])


def _rank_total(row):
    return (-row["net"], -(row["roi"] if row["roi"] is not None else -1000), row["mask"].bit_count(),
            row["model"], row["parameter"], row["threshold"], row["mask"])


def _period(frame):
    return {"from": frame.date.min().date().isoformat(), "to": frame.date.max().date().isoformat(),
            "fixtures": int(frame.key.nunique()), "market_rows": len(frame)}


def run_search(raw):
    """Search development blocks, freeze one config/market, then score holdout."""
    data, fixtures, coverage = get_dataset(raw)
    if data.empty or data.date.nunique() < 40:
        raise ValueError("有效賽日少於 40，暫不進行參數擂台。")
    data = data.sort_values(["date", "key", "market"]).reset_index(drop=True)
    days = np.array(sorted(data.date.unique()))
    first, split = round(len(days)*.4), round(len(days)*.8)
    initial = pd.Timestamp(days[first])
    hold = pd.Timestamp(days[split])
    windows = [list(chunk) for chunk in np.array_split(days[first:split], 4)]
    y = dict(version=LAB_VERSION, coverage=coverage,
             period={"warmup": _period(data[data.date < initial]),
                     "development": _period(data[data.date.between(initial, hold, inclusive="left")]),
                     "holdout": _period(data[data.date >= hold])},
             folds=[dict(start=pd.Timestamp(w[0]).date().isoformat(), end=pd.Timestamp(w[-1]).date().isoformat())
                    for w in windows], n_configs=len(_specs()), markets={}, leaderboard=[])
    # Complete all selection before scoring any holdout returns.
    pending = {}
    with threadpool_limits(limits=1):
        for market in MARKETS:
            g = data[data.market == market]
            f = [(g[g.date < pd.Timestamp(w[0])-DAY], g[g.date.isin(w)]) for w in windows]
            if any(not _sample_check(tr) or te.empty for tr, te in f):
                y["markets"][market] = {"status": "資料不足"}
                continue
            dev = pd.concat([te for _,te in f], ignore_index=True)
            fold = np.concatenate([np.full(len(te),i) for i,(_,te) in enumerate(f)])
            scored, memo = [], {}
            for family, v, mask, ev in _specs():
                k = (family,v,mask)
                if k not in memo:
                    pe, ac = zip(*[_predict(tr,te,market,*k) for tr,te in f])
                    memo[k] = np.concatenate(pe), np.concatenate(ac)
                row = {**describe(family,v,mask,ev), **_metrics(dev,*memo[k],ev,fold=fold)}
                scored.append(row)
            scored.sort(key=_rank_total)
            eligible = [r for r in scored if r["eligible"]]
            results = {"status": "候選已鎖定" if eligible else "未足最低注數",
                       "searched": len(scored), "eligible": len(eligible),
                       "raw_max_net": scored[0], "winner": eligible[0] if eligible else None,
                       "best_roi": min(eligible,key=lambda r:(-r["roi"],*_rank_total(r))) if eligible else None,
                       "family_max": {m:next((r for r in eligible if r["model"]==m),None) for m in MODELS},
                       "feature_toggles": []}
            if eligible:
                c = eligible[0]
                for i, name in enumerate(GROUPS):
                    flip = next(r for r in scored if r["mask"] == c["mask"] ^ (1<<i) and
                                (r["model"],r["parameter"],r["threshold"]) ==
                                (c["model"],c["parameter"],c["threshold"]))
                    results["feature_toggles"].append(dict(group=name, enabled=bool(c["mask"] & (1<<i)),
                                                           flipped=flip, delta_net=c["net"]-flip["net"]))
            y["markets"][market] = results
            y["leaderboard"].extend(dict(market=market, **r) for r in scored)
            if eligible:
                pending[market] = (eligible[0], g[g.date < hold-DAY], g[g.date >= hold])
        for market,(c,tr,te) in pending.items():
            if te.empty or not _sample_check(tr):
                continue
            p, a = _predict(tr,te,market,c["model"],c["parameter"],c["mask"])
            test = _metrics(te,p,a,c["threshold"],intervals=True)
            y["markets"][market].update(holdout=test, fit_rows=len(tr), test_rows=len(te))
    return y


# Historical winner parameters may be embedded by the release workflow.
# These are shadow defaults, never betting settings or an auto-BET gate.
FROZEN_AS_OF = "2026-11-19"
FROZEN = {
    "讓球": {**describe("POISSON", .35, 12, .04), "holdout_n": 35, "holdout_roi": -1.495/35,
             "use": "研究對照"},
    "入球大小": {**describe("LR", 60, 9, .04), "holdout_n": 29, "holdout_roi": .71/29,
               "use": "優先影子候選"},
    "角球大小": {**describe("LR", 10, 14, 0.), "holdout_n": 204, "holdout_roi": -6.18/204,
               "use": "研究對照"},
}


def score_fixture(raw, fixture, now=None):
    """Refit frozen configs and make a reproducible T−30 prediction; no writes."""
    if not isinstance(fixture, dict):
        raise ValueError("賽事 JSON 必須是物件。")
    kickoff = _hkt_timestamp(fixture.get("kickoff"), require_zone=True)
    d = _day(kickoff)
    if kickoff is None or d <= pd.Timestamp(FROZEN_AS_OF):
        raise ValueError("請提供含時區的實際賽程時間，且須在歷史測試終點之後。")
    cut = kickoff-pd.Timedelta(minutes=30)
    now = pd.Timestamp.now(tz=HKT).tz_localize(None) if now is None else now
    if now < cut or now >= kickoff:
        raise ValueError("請在 T−30 之後、正式開賽之前分析。")
    h, a, fixture_id = _text(fixture.get("home")), _text(fixture.get("away")), fixture.get("fixture_id")
    if (not h or not a or h == a or not isinstance(fixture_id,(str,int)) or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", str(fixture_id))):
        raise ValueError("請提供 Tipsme fixture_id 和主客隊名稱。")
    if "Tipsme_Match_ID" in raw and "Status" in raw:
        stored = raw.loc[raw.Status.eq("Settled"), "Tipsme_Match_ID"].astype(str).str.strip()
        stored = stored.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        if stored.eq(str(fixture_id)).any():
            raise ValueError("這個 Tipsme ID 已有賽果，不可當成新賽事。")
    odds = _object(fixture.get("odds"), list)
    try:
        archive_date = pd.Timestamp(FROZEN_AS_OF)
        if cut <= archive_date:
            raise ValueError("賽事早於研究配置鎖定日期。")
        D, K, _ = get_dataset(raw)
        if D.empty:
            raise ValueError("未載入可用的帶時點歷史盤口。")
        if any(k["date"] == d and k["home"] == h and k["away"] == a for k in K):
            raise ValueError("同日同一主客組合已有賽果，不可當成新賽事。")
    except KeyError as e:
        raise ValueError("歷史資料欄位不完整。") from e
    info = dict(date=d, home=h, away=a, tourn=_text(fixture.get("league")),
                key="fixture:"+str(fixture_id), odds=odds,
                recent=[_object(fixture.get("home_recent"),list), _object(fixture.get("away_recent"),list)])
    table = pd.DataFrame(_fixture_rows(info,K,cut,strict=True,add_scores=False))
    if table.empty:
        raise ValueError("T−30 前 15 分鐘內沒有帶時區、有效且同時的雙邊報價。")
    result = []
    with threadpool_limits(limits=1):
        for bt, te in table.groupby("market"):
            c = FROZEN.get(bt)
            if c is None:
                continue
            tr = D[(D.market==bt) & (D.date<d-DAY)]
            if not _sample_check(tr):
                continue
            p, ac = _predict(tr,te,bt,c["model"],c["parameter"],c["mask"])
            hi, ev = _sides(te,p,ac)
            q = te.iloc[0]
            direction = ("主" if hi[0] else "客") if bt=="讓球" else ("大" if hi[0] else "小")
            signal = ("影子模擬" if c["use"] == "優先影子候選" else "對照訊號") if ev[0] >= c["threshold"] else "略過"
            result.append(dict(賽事ID=str(fixture_id), 盤口=bt, 主盤=float(q.line),
                               主或大賠率=float(q.upper), 客或小賠率=float(q.lower),
                               開賽時間=kickoff.isoformat()+"+08:00", 報價時間=q.time.isoformat()+"+08:00",
                               預測截點=cut.isoformat()+"+08:00",
                               產生時間=now.isoformat()+"+08:00", 預測方向=direction,
                               可用賠率=float(q.upper if hi[0] else q.lower), 估計EV=float(ev[0]),
                               市場有效比例=float(q.p), 模型有效比例=float(p[0]),
                               研究用途=c["use"], 建議=signal, 模型=c["model"], 模型參數=c["parameter"],
                               開關=c["mask"], 最低EV=c["threshold"], 訓練盤口數=len(tr), 研究版本=LAB_VERSION))
    if not result:
        raise ValueError("該盤口沒有凍結配置，或歷史訓練樣本不足。")
    return result


def _cli():
    from argparse import ArgumentParser
    from pathlib import Path
    p = ArgumentParser(description="FB-app 唯讀歷史參數研究")
    p.add_argument("history", type=Path, help="Sheet1 Excel 報表、CSV，或列式報表 JSON")
    p.add_argument("--summary", required=True, type=Path, help="輸出的研究摘要 JSON（含排名）")
    p.add_argument("--csv", required=True, type=Path, help="輸出的全組合排名 CSV")
    a = p.parse_args()
    if len({x.resolve() for x in (a.history, a.summary, a.csv)}) != 3 or a.summary.exists() or a.csv.exists():
        p.error("請使用兩個各自獨立、尚不存在的輸出檔；不可覆蓋輸入。")
    if a.history.suffix.lower() == ".xlsx":
        raw = pd.read_excel(a.history, sheet_name="Sheet1")
    elif a.history.suffix.lower() == ".json":
        raw = pd.DataFrame(json.loads(a.history.read_text(encoding="utf-8")))
    else:
        raw = pd.read_csv(a.history)
    r = run_search(raw)
    with a.summary.open("x", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2, allow_nan=False)
    out = pd.DataFrame(r["leaderboard"])
    with a.csv.open("x", encoding="utf-8-sig", newline="") as f:
        out.to_csv(f, index=False)
    print(f"有效賽事 {r['coverage']['usable_matches']} 場；比較 {len(out)} 組。")


if __name__ == "__main__":
    _cli()
