"""Independent, simulation-only V1 policy. No Streamlit or database I/O."""

import json
import math

import numpy as np

from fb_app_clv_cohorts import fixture_id, history, number

VERSION = 2
TAG = '_fb_kanpan_v1'
TARGET = 0.02
HIT = 0.55
MIN_N = 40
EPS = 1e-10
MARKETS = ('讓球', '入球大小', '角球大小')
STATUS = ('模擬候選', '移出／待重入', '未入選')


def rule_status(record):
    return STATUS[0] if record.get('active') else STATUS[1] if record.get('ever') else STATUS[2]


def frozen_display(legacy, eligible):
    """Do not infer a historical status for records saved before the status field."""
    r = eligible or legacy
    return r, (STATUS[0] if eligible else (r or {}).get('candidate_status', ''))


def unique_market_signals(res):
    """No fallback: reject many rule/side matches before any price/source filtering."""
    markets = {}
    for key, item in res.items():
        for rule in item.get('profitable_triggers', ()):
            markets.setdefault(key[0], []).append((key, item, rule))
    return [matches[0] for matches in markets.values() if len(matches) == 1]


def _earned(n, total, wins):
    return (n > MIN_N) & (total > (TARGET + EPS) * n) & (wins > (HIT + EPS) * n)


def historical(D, matrix, names, *, price_guard=True):
    """Replay admissions by settled match day; positive but weaker rules stay in."""
    if D.empty:
        return {}
    out = {}
    for market in D['bet_type'].unique():
        sub = D['bet_type'].eq(market).to_numpy()
        dates = D.loc[sub, 'date'].to_numpy()
        odds = D.loc[sub, 'odds'].to_numpy(float)
        ret = D.loc[sub, 'ret'].to_numpy(float)
        valid = np.isfinite(ret) & np.isfinite(odds) & (odds > 1)
        order = np.argsort(dates[valid], kind='stable')
        dates, odds, ret = (x[valid][order] for x in (dates, odds, ret))
        M = matrix[sub][valid][order]
        if not len(dates):
            continue
        n = np.zeros(len(names), dtype=int)
        win = n.copy()
        pnl = n.astype(float)
        active = n.astype(bool)
        ever = active.copy()
        endpoints = np.r_[np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(dates)]
        lo = 0
        for hi in endpoints:
            chunk = M[lo:hi]
            r = ret[lo:hi]
            n += chunk.sum(axis=0)
            win += ((r[:, None] > 0) & chunk).sum(axis=0)
            pnl += (r @ chunk)
            earned = _earned(n, pnl, win)
            active = (active & (pnl >= -EPS)) | earned
            ever |= earned
            lo = hi
        exposure = (np.maximum(ret, 0) / (odds - 1)) @ M
        loss = np.maximum(-ret, 0) @ M
        avg = odds @ M
        for i in np.flatnonzero(n >= MIN_N):
            floor = 1 + (loss[i] + TARGET * n[i]) / exposure[i] if exposure[i] > 0 else None
            out[(market, names[i])] = dict(
                rule=names[i], n=int(n[i]), roi=float(pnl[i]/n[i]), hit=float(win[i]/n[i]),
                avg=float(avg[i]/n[i]), floor=float(floor) if price_guard and floor is not None else None,
                active=bool(active[i]), ever=bool(ever[i]))
    return out


def choose(records, price):
    """Stable priority independent of old V10 EV/z: eligible, higher ROI, n, name."""
    price = number(price, 1)
    allowed = [r for r in records if r['active'] and price is not None
               and (r.get('floor') is None or price + EPS >= r['floor'])]
    return min(allowed, key=lambda r: (-r['roi'], -r['n'], r['rule']), default=None)


def terms(row):
    vals = [number(row.get(k)) for k in ('Initial_Line', 'Initial_Odds')]
    if None in vals:
        return None
    return [str(row.get(k, '')) for k in (
        'ID', 'Match_Date', 'Home_Team', 'Away_Team', 'Bet_Type', 'Selection')] + [round(v, 6) for v in vals]


def freeze(row, legacy, eligible=None):
    """Store decision-time context without changing Note or existing quote terms."""
    h = history(row.get('Odds_History'))
    key = terms(row)
    if not (h and isinstance(h[0], dict) and key and (legacy or eligible)):
        return
    h = list(h)
    h[0] = {**h[0], TAG: [key, legacy, eligible]}
    row['Odds_History'] = json.dumps(h, ensure_ascii=False)


def frozen(row):
    h = history(row.get('Odds_History'))
    if not h:
        return None, None
    key = terms(row)
    found = [item for q in h if isinstance(q, dict)
             if isinstance(item := q.get(TAG), list) and len(item) == 3 and item[0] == key]
    if len(found) != 1:
        return None, None
    item = found[0]
    def checked(r):
        if (not isinstance(r, dict) or not isinstance(r.get('rule'), str)
                or type(r.get('n')) is not int or r['n'] < 1
                or number(r.get('roi')) is None or number(r.get('hit')) is None):
            return None
        if 'candidate_status' in r and r['candidate_status'] not in STATUS:
            return {k: v for k, v in r.items() if k != 'candidate_status'}
        return r
    return checked(item[1]), checked(item[2])


def settled_returns(papers, rows, compute):
    """Join by fixture + HKT match date + both teams. Conflicts/missing stay unknown."""
    index = {}
    for r in rows:
        mid = fixture_id(r.get('Tipsme_Match_ID', ''))
        key = (mid, str(r.get('Match_Date', ''))[:10], r.get('Home_Team'), r.get('Away_Team'))
        index.setdefault(key, []).append(r)
    result = []
    for p in papers:
        key = (p['mid'], p['day'], p['home'], p['away'])
        observed = set()
        fields = ('Home_Corner', 'Away_Corner') if p['market'] == '角球大小' else ('Home_Goal', 'Away_Goal')
        for r in index.get(key, ()):
            score = tuple(number(r.get(f)) for f in fields)
            if all(x is not None and x >= 0 and x.is_integer() for x in score):
                observed.add(score)
        outcome = None
        if len(observed) == 1:
            outcome = compute(p, *next(iter(observed)))
            if not math.isfinite(outcome):
                outcome = None
        result.append({**p, 'return': outcome})
    return result


def _daily_summary(day_map):
    """Report pooled unit returns; bootstrap weeks to keep same-date overlaps together."""
    result = dict(n=0, roi=None, ci=None, net=0., hit=None, first=None, second=None)
    if not day_map:
        return result
    days = sorted(day_map)
    values = np.array([day_map[d] for d in days], dtype=float)
    n, net, wins = values.sum(axis=0)
    result.update(n=int(n), roi=float(net/n), net=float(net), hit=float(wins/n))
    if len(days) >= 2:
        mid = len(days)//2
        halves = [part.sum(axis=0) for part in (values[:mid], values[mid:])]
        result.update(first=float(halves[0][1]/halves[0][0]),
                      second=float(halves[1][1]/halves[1][0]))
    by_week = {}
    for d, part in zip(days, values):
        date = d.date()
        week = date.toordinal() - date.weekday()
        by_week.setdefault(week, np.zeros(3))[:] += part
    if len(by_week) >= 2:
        weeks = np.array(list(by_week.values()))
        rng = np.random.default_rng(1831)
        draws = weeks[rng.integers(len(weeks), size=(2000, len(weeks)))].sum(axis=1)
        result['ci'] = np.quantile(draws[:,1]/draws[:,0], [.025,.975]).tolist()
    return result


def chronological_research(design, folds, fit, apply, matrix):
    """All-rule and reverse comparison only. This code cannot create an order."""
    totals = {m:{} for m in ('總體', *MARKETS)}
    reverse_totals = {m:{} for m in totals}
    reverse_by_rule = {}
    reverse_detail = []
    covered = set()
    def record(target, day, values, weights=None):
        if len(values):
            v = np.ones(len(values), dtype=int) if weights is None else weights
            target.setdefault(day, np.zeros(3))[:] += (v.sum(), v@values, v@(values>0))
    for start, end in folds:
        train = design[design['date'] < start]
        test = design[design['date'].between(start, end)]
        if train.empty or test.empty:
            continue
        coef = fit(train)
        train, test = (apply(d, coef).reset_index(drop=True) for d in (train,test))
        names, trained_masks = matrix(train)
        _, M = matrix(test, names)
        stats = historical(train, trained_masks, names)
        for m in MARKETS:
            name_index = [j for j,name in enumerate(names)
                          if (r := stats.get((m,name))) and r['n']>40 and r['roi']<-.10-EPS and r['hit']<.40-EPS]
            md = test[test['bet_type'].eq(m)]
            for d, group in md.groupby('date', sort=True):
                datum = d.to_pydatetime()
                if (d,m) in covered:
                    raise ValueError('Overlapping out-of-sample folds')
                covered.add((d,m))
                indices = group.index.to_numpy()
                vals = group['ret'].to_numpy(float)
                valid = np.isfinite(vals) & group['odds'].gt(1).to_numpy()
                active = M[indices,:]
                weights = active.sum(axis=1)
                keep = valid & (weights>0)
                for k in (m,'總體'):
                    record(totals[k], datum, vals[keep], weights[keep])
                if not name_index:
                    continue
                # Look up the opposite side from the same reconstructed main quote,
                # never use 1-hit or -ROI. Require a unique opposite for each row.
                paired = {}
                for i,(key,line,nsnap,side,ok) in enumerate(zip(group['key'],group['line'],
                            group['nsnap'],group['side'],valid)):
                    if ok:
                        paired.setdefault((key,round(line,6),nsnap),[]).append((i,side))
                for (_,line,_), members in paired.items():
                    if len(members)!=2 or {s for _,s in members} not in ({'Home','Away'},{'Over','Under'}):
                        continue
                    (a,_),(b,_) = members
                    for j in name_index:
                        if active[a,j] == active[b,j]:
                            continue  # neither triggers, or a conflicting two-sided rule
                        original, opposite = (a,b) if active[a,j] else (b,a)
                        own = reverse_by_rule.setdefault((m,names[j]), ({},{}))
                        record(own[0],datum,vals[[original]])
                        record(own[1],datum,vals[[opposite]])
                        orig, inv = group.iloc[original], group.iloc[opposite]
                        reverse_detail.append(dict(day=datum.date().isoformat(), market=m, rule=names[j],
                                                   line=float(line), side=orig['side'], odds=float(orig['odds']),
                                                   opposite=inv['side'], opposite_odds=float(inv['odds']),
                                                   original_return=float(vals[original]),
                                                   opposite_return=float(vals[opposite])))
                        for k in (m,'總體'):
                            record(reverse_totals[k],datum,vals[[opposite]])
    return dict(all={m:_daily_summary(day) for m,day in totals.items()},
                reverse={m:_daily_summary(day) for m,day in reverse_totals.items()},
                reverse_detail=reverse_detail,
                reverse_rules=[dict(market=m, rule=rule, original=_daily_summary(before),
                                    opposite=_daily_summary(after))
                               for (m,rule),(before,after) in sorted(reverse_by_rule.items())])
