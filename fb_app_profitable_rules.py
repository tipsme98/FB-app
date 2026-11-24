"""Independent, simulation-only V1 policy. No Streamlit or database I/O."""

import json
import math

import numpy as np

from fb_app_clv_cohorts import fixture_id, history, number

VERSION = 1
TAG = '_fb_kanpan_v1'
TARGET = 0.02
HIT = 0.55
MIN_N = 40
EPS = 1e-10


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
    if not h or not isinstance(h[0], dict):
        return None, None
    item = h[0].get(TAG)
    if not isinstance(item, list) or len(item) != 3 or item[0] != terms(row):
        return None, None
    def checked(r):
        if (not isinstance(r, dict) or not isinstance(r.get('rule'), str)
                or type(r.get('n')) is not int or r['n'] < 1
                or number(r.get('roi')) is None or number(r.get('hit')) is None):
            return None
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
