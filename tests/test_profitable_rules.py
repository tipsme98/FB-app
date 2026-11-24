import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import fb_app_profitable_rules as pr
from fb_app_clv_cohorts import HKT, quote_tag


spec = importlib.util.spec_from_file_location('fb_test_profit', Path(__file__).parents[1] / 'FB-app.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def hist(groups):
    pairs = [(i, r) for i, day in enumerate(groups) for r in day]
    frame = pd.DataFrame(pairs, columns=['date', 'ret']).assign(bet_type='讓球', odds=2.0)
    return pr.historical(frame, np.ones((len(frame), 1), dtype=bool), ['r'])[('讓球', 'r')]


class ProfitableRules(unittest.TestCase):
    def test_strict_thresholds(self):
        self.assertFalse(hist([[1]*25+[-1]*15])['active'])  # exactly 40
        self.assertFalse(hist([[1]*33+[0]*27])['active'])  # exactly 55%
        self.assertFalse(hist([[.30]*30+[-.48]*10+[-.32]*10])['active'])  # exactly +2%
        self.assertTrue(hist([[1]*25+[-1]*16])['active'])

    def test_exit_hysteresis_and_reentry(self):
        a = [1]*30 + [-1]*20
        b = [-1]*10
        c = [-1]
        d = [1]*3
        self.assertTrue(hist([a, b])['active'])  # 0 does not deactivate
        self.assertFalse(hist([a, b, c])['active'])
        self.assertFalse(hist([a, b, c, d])['active'])  # positive but below hit hurdle
        self.assertTrue(hist([a, b, c, d, [1]*8])['active'])

    def test_price_weights_and_best(self):
        r = hist([[.5]*36 + [-.5]*12+[0]*12])
        self.assertAlmostEqual(r['floor'], 1+(6+1.2)/18)
        self.assertEqual(pr.choose([r], 1.399), None)
        self.assertEqual(pr.choose([r], 1.40), r)

    def test_bound_freeze_preserves_odds_and_note(self):
        row = dict(ID='B1', Bet_Type='讓球', Selection='Home', Initial_Line=-.25,
                   Initial_Odds=1.95, Odds_History='[{"type":"讓球","upper":1.95}]', Note='CLV_ENTRY_V1:test')
        a = hist([[1]*25+[-1]*16])
        pr.freeze(row, None, a)
        self.assertEqual(pr.frozen(row), (None, a))
        self.assertEqual(json.loads(row['Odds_History'])[0]['upper'], 1.95)
        self.assertEqual(row['Note'], 'CLV_ENTRY_V1:test')
        self.assertEqual(pr.frozen({**row, 'Initial_Odds': 2.0}), (None, None))
        self.assertEqual(pr.frozen({**row, 'ID': 'user-copy'}), (None, None))

    def test_forward_settlement_fails_closed(self):
        p = dict(mid='9', day='2026-11-25', home='H', away='A', market='讓球')
        good = dict(Tipsme_Match_ID='9.0', Match_Date=p['day'], Home_Team='H',
                    Away_Team='A', Home_Goal=1, Away_Goal=0)
        wrong = {**good, 'Home_Team': 'else'}
        self.assertEqual(pr.settled_returns([p], [wrong], lambda *a: 7)[0]['return'], None)
        self.assertEqual(pr.settled_returns([p], [good, good], lambda *a: 0)[0]['return'], 0)
        self.assertEqual(pr.settled_returns([p], [good, {**good,'Away_Goal': 3}], lambda *a: 7)[0]['return'], None)

    def test_source_and_kickoff_and_one_side(self):
        ko = datetime(2026, 11, 25, 20, tzinfo=HKT)
        t = ko - timedelta(minutes=30)
        p = {'date': ko.date(), 'token': ('run',0,'11'), 'names': ('L','H','A'),
             'fill': {'info': {'match_id': '11', 'kickoff': ko, 'clv_id_confirmed': True, 'finished': False},
                      'clv_observed': t.isoformat(), 'odds_ok': True},
             'odds': [{'type':'讓球','line':0,'upper':2.,'lower':1.9,'unlock':True,
                       app.CLV_QUOTE: quote_tag('11', t-timedelta(minutes=1))}]}
        a = hist([[1]*25+[-1]*16])
        one = {'profitable_snapshot': a, 'close_pos': 0, 'snapshots':2}
        res = {('讓球',0,'Home'): {**one,'lookup_odds':2.0},
               ('讓球',0,'Away'): {**one,'lookup_odds':1.9}}
        self.assertEqual(len(app._profitable_lock(res,p,ko.timestamp(),t,True)), 1)
        self.assertEqual(app._profitable_lock(res,p,ko.timestamp(),ko,True), [])
        self.assertEqual(app._profitable_lock(res,p,0,t,True), [])
        p['odds'][0].pop(app.CLV_QUOTE)
        self.assertEqual(app._profitable_lock(res,p,ko.timestamp(),t,True), [])

    def test_history_input_excludes_after_cutoff_and_follow(self):
        rows = [
            ['old','Settled','2026-11-24','讓球',''],
            ['today','Settled','2026-11-25','讓球',''],
            ['bad','Settled','2026-10-33','讓球',''],
            ['copy','Settled','2026-11-24','讓球','用家跟投'],
            ['live','Settled','2026-11-24','讓球 (即場)',''],
        ]
        d = pd.DataFrame(rows, columns=['ID','Status','Match_Date','Bet_Type','Note'])
        with patch.object(app,'v9_engine_input',side_effect=lambda x:x):
            self.assertEqual(app._profitable_history_input(d, pd.Timestamp('2026-11-25')).ID.tolist(), ['old'])

    def test_ui_no_store_not_zero(self):
        src = """
import sys
sys.path.insert(0, {root!r})
import importlib.util
import pandas as pd
s = importlib.util.spec_from_file_location('fb_ui_test', {src!r})
a=importlib.util.module_from_spec(s)
s.loader.exec_module(a)
a._profitable_paper_enabled = lambda: False
def down(days):
    raise TimeoutError
a._profitable_read = down
a._profitable_show(pd.DataFrame())
""".format(root=str(Path(__file__).parents[1]), src=str(Path(__file__).parents[1]/'FB-app.py'))
        at = AppTest.from_string(src).run()
        self.assertFalse(at.exception)
        self.assertIn('未啟用', at.warning[0].value)
        at.button(key='profitable_read_v1').click().run()
        self.assertFalse(at.exception)
        self.assertIn('不可把查詢失敗當成 0 注', at.error[0].value)


if __name__ == '__main__':
    unittest.main()
