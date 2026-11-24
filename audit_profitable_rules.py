"""Private, offline chronological replay; run on a CSV export, never commits data."""
import argparse
import importlib.util
import json

import numpy as np
import pandas as pd

import fb_app_profitable_rules as pr


def replay(app, D):
    folds = app.v9_folds(D['date'], n_folds=6, init_frac=0.4)
    outputs = {k: [] for k in ('legacy_ev0_refit', 'legacy_ev3_refit',
                                'historical_no_price', 'historical_price')}
    for start, end in folds:
        tr = D[D['date'] < start]
        te = D[(D['date'] >= start) & (D['date'] <= end)]
        if tr.empty or te.empty:
            continue
        coef = app.v10_fit_resid(tr)
        tr, te = (app.v10_apply_resid(d, coef).reset_index(drop=True) for d in (tr, te))
        names, train = app.v10_rule_matrix(tr)
        _, test = app.v10_rule_matrix(te, names)
        stat = app.v10_rule_stats(tr, train, names)
        stats = pr.historical(tr, train, names)
        for key, threshold in [('legacy_ev0_refit', 0), ('legacy_ev3_refit', .03)]:
            scores, chosen = app.v10_pick(te, test, stat, np.ones(len(te), dtype=bool), threshold)
            p = te.loc[chosen >= 0].assign(score=scores[chosen >= 0]).sort_values(
                ['score', 'side'], ascending=[False, True]).drop_duplicates(['key', 'bet_type'])
            outputs[key].extend(p[['date', 'key', 'bet_type', 'ret', 'odds']].to_dict('records'))
        for key, use_price in [('historical_no_price', False), ('historical_price', True)]:
            taken = {}
            for i, r in enumerate(te.itertuples(index=False)):
                matched = [stats[(r.bet_type, names[j])] for j in np.flatnonzero(test[i])
                           if (r.bet_type, names[j]) in stats]
                if not use_price:
                    matched = [{**m, 'floor': None} for m in matched]
                choice = pr.choose(matched, r.odds)
                if choice:
                    rank = (-choice['roi'], -choice['n'], choice['rule'], r.side)
                    mk = (r.key, r.bet_type)
                    if mk not in taken or rank < taken[mk][0]:
                        taken[mk] = (rank, r._asdict())
            outputs[key].extend(x[1] for x in taken.values())
    return folds, outputs


def summary(records):
    frame = pd.DataFrame(records)
    if frame.empty:
        return {}
    frame['week'] = frame['date'].dt.to_period('W')
    agg = frame.groupby('week')['ret'].agg(['sum', 'size']).to_numpy()
    rng = np.random.default_rng(1823)
    draws = agg[rng.integers(0, len(agg), size=(2000, len(agg)))].sum(axis=1)
    ci = np.quantile(draws[:, 0] / draws[:, 1], [.025, .975]).tolist()
    return dict(n=len(frame), fixtures=frame['key'].nunique(), roi=frame['ret'].mean(),
                hit=frame['ret'].gt(0).mean(), avg=frame['odds'].mean(), week_bootstrap_ci=ci,
                markets={bt: dict(n=len(g), roi=g['ret'].mean()) for bt, g in frame.groupby('bet_type')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('private_csv')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('fb_offline', __file__.replace('audit_profitable_rules.py', 'FB-app.py'))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    data = pd.read_csv(args.private_csv, low_memory=False)
    mt = app.v9_build_match_table(app._profitable_history_input(data))
    design = app.v10_build_dataset(mt)
    folds, outputs = replay(app, design)
    whole = app.v10_apply_resid(design, app.v10_fit_resid(design))
    names, matrix = app.v10_rule_matrix(whole)
    active = pr.historical(whole, matrix, names)
    strict = [k for k,v in active.items() if v['n']>40 and v['roi']>.0200000001 and v['hit']>.5500000001]
    print(json.dumps(dict(matches=len(mt), rows=len(design), combinations=len(names), strict=len(strict),
                          active=sum(x['active'] for x in active.values()),
                          periods=[(str(a.date()), str(b.date())) for a,b in folds],
                          portfolios={k: summary(v) for k,v in outputs.items()},
                          last_fold={k: summary([r for r in v if r['date'] >= folds[-1][0]]) for k,v in outputs.items()} if folds else {}),
                     default=float, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
