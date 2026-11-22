"""Offline regression. Run from the repo: python -m unittest discover -s tests -v."""
import csv
import os
from pathlib import Path
import tempfile
import unittest

from streamlit.testing.v1 import AppTest

CANDIDATE = Path(os.getenv('FB_APP_CANDIDATE', Path(__file__).resolve().parents[1] / 'FB-app.py')).resolve()
BASELINE = os.getenv('FB_APP_DAILY_OLD')

WIRING = r'''
KO = datetime(2026, 11, 20, 23, 59, 37, tzinfo=HKT)
T0 = datetime(2026, 11, 21, 5, 0, tzinfo=HKT)
def clock():
    return T0 + timedelta(seconds=st.session_state.get('test_delta', 0))
get_hkt_now = clock
_has_secret = lambda key: False
_v10_settings_allow_automation = lambda: True
_v162_can_automate = lambda: MODE != 'viewer'
merge_pending_on_load = lambda *args, **kwargs: None
_v1645_sql_read_bet_duplicates = lambda *args: pd.DataFrame()
_v16_clv_load = lambda: {'enabled': '0', 'bet_lead_min': '30', 'snap_hours': '2', 'runner_mode': '0'}
v16_snapshot_open_bets = lambda *args: (0, [])
def fake_get(url, **kwargs):
    raise AssertionError(f'unmocked Tipsme call {url}')
_tipsme_get = fake_get
_tipsme_fetch_page_html = fake_get
FIXTURE = {'round': 0, 'hkjc_match_id': 999001, 'home': '測試主隊', 'away': '測試客隊',
           'league': '日本職業聯賽', 'day': '2026-11-20', 'finished': True,
           'is_hkjc': True, 'match_time': KO.timestamp(), 'home_goal': 2, 'away_goal': 1,
           'home_half': 1, 'away_half': 0, 'home_corner': 5, 'away_corner': 2,
           'home_red': 0, 'away_red': 0, 'home_yellow': 1, 'away_yellow': 2}
MATCH = dict(match_id='999001', home=FIXTURE['home'], away=FIXTURE['away'], league=FIXTURE['league'],
             kickoff=KO.replace(tzinfo=None), is_cup=False, home_rank=3, away_rank=10, finished=True,
             score='2-1', half_score='1-0', corner='5-2', home_red=0, away_red=0,
             home_yellow=1, away_yellow=2)
tipsme_fetch_match_info = lambda mid: dict(MATCH) if str(mid) == '999001' else None
tipsme_fetch_h2h = lambda mid: None
tipsme_fetch_bookmaker_odds = lambda mid: None
tipsme_fetch_hkjc_odds = lambda mid: {'stub': True}
tipsme_build_odds_history = lambda raw, **kwargs: (
    [] if MODE == 'no_picks' else
    [{'id': 0, 'type': '入球大小', 'record_time': '11-20 23:30', 'line': 2.5,
      'upper': 1.88, 'lower': 1.88, 'unlock': True, 'margin': 1.06}], [])
v14_warm_engines = lambda *args: None
def fixture_row(mid='999001', status='Settled', away=FIXTURE['away'], date='2026-11-20'):
    return {'ID': f'old-{mid}-{date}', 'Match_Date': date, 'Date': date + ' 19:00',
            'Status': status, 'Tournament_Name': FIXTURE['league'], 'Tournament_Category': CATEGORY_OPTIONS[0],
            'Match': f"{FIXTURE['home']} vs {away}", 'Home_Team': FIXTURE['home'], 'Away_Team': away,
            'System_Stake': 0, 'User_Stake': 0, 'Bet_Type': '入球大小', 'Selection': 'Over',
            'Initial_Line': 2.5, 'Initial_Odds': 1.88, 'Tipsme_Match_ID': mid}
if 'test_db' not in st.session_state:
    rows = ([fixture_row()] if MODE == 'settled' else
            [fixture_row(status='Open')] if MODE in ('open', 'settle_failure', 'settle_timeout', 'no_commit') else
            [fixture_row('998012', away='不同客隊'), fixture_row(date='2026-11-19')] if MODE == 'nearby' else
            [])
    st.session_state.test_db = pd.DataFrame(rows).reindex(columns=DB_COLUMNS)
def official():
    if MODE == 'db_down':
        return None
    return st.session_state.test_db.copy()
_v16412_authoritative_bets = official
def cfg_frame():
    if MODE == 'cfg_down':
        return None
    raw = {'enabled': '1', 'time': '05:00', 't0': '00:00',
           't1': '21:00' if MODE == 'outside' else '23:59',
           'last_run': MARKER, 'last_run_win': WIN}
    raw.update(st.session_state.get('test_marker', {}))
    return pd.DataFrame(raw.items(), columns=SCHED_COLUMNS)
_v16412_schedule_frame = cfg_frame
def save_cfg(cfg):
    st.session_state['test_marker_attempts'] = st.session_state.get('test_marker_attempts', 0) + 1
    if MODE == 'marker_failure':
        return False
    st.session_state.test_marker = dict(cfg)
    st.session_state['_v1511_sched_cache'] = None
    return True
if FB_APP_RELEASE != 'V16.4.11':
    _v1511_sched_save = save_cfg
def load_db(filename, columns, table_name, **kwargs):
    if table_name == 'football_bets':
        return st.session_state.test_db.copy()
    if table_name == SCHED_TABLE:
        return cfg_frame()
    return pd.DataFrame(columns=columns)
def save_db(df, filename, table_name, **kwargs):
    if table_name != 'football_bets':
        return True
    step = (st.session_state.get('v15_auto') or {}).get('step')
    st.session_state.setdefault('test_writes', []).append(step)
    if step == 'SETTLE_CONFIRM':
        if MODE == 'settle_failure':
            return False
        if MODE == 'settle_timeout':
            raise TimeoutError('mock ambiguous reply')
        if MODE == 'no_commit':
            return True
    st.session_state.test_db = df.copy()
    return True
def read_feed(day, **kwargs):
    st.session_state.setdefault('test_fetched', []).append((day, kwargs))
    if MODE == 'feed_down' or (MODE == 'delayed' and st.session_state.get('test_delta', 0) < 300):
        return None, '賽果未讀齊'
    if MODE == 'partial':
        return [FIXTURE, {**FIXTURE, 'hkjc_match_id': 999002, 'home': '另一主隊'}], 'ok'
    if MODE == 'empty':
        return [], 'ok'
    return [FIXTURE], 'ok'
tipsme_fetch_day_results = read_feed
'''

FEED_WIRING = r'''
def row(mid, ts, utc='0001-01-01T00:00:00'):
    return dict(matchId=mid, matchStatus='8', home=f'h{mid}', away=f'a{mid}',
                fullMatchTime=ts, fullMatchTimeUtc=utc, score='2-1', corner='5-2')
def page(n, more, rows, date='2026-11-20'):
    payload = json.dumps({'props': {'pageProps': {'page': n, 'hasNextPage': more,
                                                  'filters': {'date': date}, 'matchResults': rows}}})
    return '<script id="__NEXT_DATA__" type="application/json">' + payload + '</script>'
late = row(22, '2026-11-20 23:59:59')
previous = row(11, '2026-11-20 00:00:00', '2026-11-19T16:00:00')
today = row(33, '2026-11-21 00:00:00')
pages = {
    ('2026-11-20', 1): page(1, True, [previous]),
    ('2026-11-20', 2): page(2, False, [late]),
    ('2026-11-21', 1): page(1, False, [late, today], '2026-11-21'),
}
if MODE == 'feed_gap':
    pages.pop(('2026-11-20', 2))
elif MODE == 'wrong_page':
    pages['2026-11-20', 2] = pages['2026-11-20', 1]
elif MODE == 'missing_metadata':
    pages['2026-11-20', 2] = pages['2026-11-20', 2].replace('"hasNextPage": false, ', '')
elif MODE == 'bad_kickoff':
    pages['2026-11-20', 2] = page(2, False, [row(22, 'not a date')])
elif MODE == 'wrong_date':
    pages['2026-11-20', 2] = page(2, False, [late], '2026-11-21')
def fixture_get(url, **kwargs):
    day, pg = re.search(r'date=(\d{4}-\d{2}-\d{2})&page=(\d+)', url).groups()
    return pages.get((day, int(pg)))
_tipsme_fetch_page_html = fixture_get
st.session_state.result = tipsme_fetch_day_results('2026-11-20', require_complete=True)
'''


class DailyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        original = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, original)

    def play(self, mode='fresh', marker='nan', win='', *, delta=0, source=CANDIDATE, code=WIRING, body='main()'):
        src = Path(source).read_text(encoding='utf-8').rsplit('if __name__ == "__main__":', 1)[0]
        filename = Path('fixture_app.py').resolve()
        filename.write_text(src + f'\nMODE={mode!r}\nMARKER={marker!r}\nWIN={win!r}\n' + code + '\n' + body,
                            encoding='utf-8')
        at = AppTest.from_file(str(filename), default_timeout=45)
        at.query_params['runner'] = '1'
        at.session_state['test_delta'] = delta
        return self.rerun(at)

    def rerun(self, at):
        at.run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def assert_not_complete(self, at):
        self.assertNotIn('test_marker', at.session_state)
        self.assertFalse(any('賽果已完成自動輸入' in c.value for c in at.caption))

    @unittest.skipUnless(BASELINE, 'Set FB_APP_DAILY_OLD for exact V16.4.11 reproduction')
    def test_deployed_nan_reports_complete_without_fetch_or_write(self):
        at = self.play(source=BASELINE)
        self.assertTrue(any('2026-11-20 賽果已完成自動輸入' in c.value for c in at.caption))
        self.assertNotIn('test_fetched', at.session_state)
        self.assertNotIn('test_writes', at.session_state)

    def test_045959_no_write_then_050000_full_flow(self):
        at = self.play(delta=-1)
        self.assert_not_complete(at)
        self.assertNotIn('test_fetched', at.session_state)
        at.session_state['test_delta'] = 0
        self.rerun(at)
        self.assertEqual(at.session_state.test_fetched, [('2026-11-20', {'require_complete': True})])
        self.assertEqual(at.session_state.test_writes, ['BET', 'SETTLE_CONFIRM'])
        db = at.session_state.test_db
        self.assertEqual(list(db['Status']), ['Settled'])
        self.assertEqual(tuple(db.iloc[0][['Match_Date', 'Tipsme_Match_ID', 'Home_Goal', 'Away_Goal']]),
                         ('2026-11-20', '999001', 2, 1))
        self.assertEqual(dict(at.session_state.test_marker),
                         dict(enabled='1', time='05:00', t0='00:00', t1='23:59',
                              last_run='2026-11-20', last_run_win='00:00-23:59'))
        self.assertTrue(any('賽果已完成自動輸入' in c.value for c in at.caption))
        self.rerun(at)
        self.assertEqual(len(at.session_state.test_writes), 2)

    def test_marker_normalization_and_exact_window(self):
        body = '''
win = '00:00-23:59'
day = datetime(2026, 11, 20).date()
samples = [None, '', float('nan'), pd.NA, 'nan', 'None', '<NA>', '2026-02-30', '2026-11-2',
           '2026-11-21', '2026-11-22', '2026-11-19', '2026-11-20']
settings = [_v16412_schedule_config(dict(enabled='1', time='05:00', last_run=m, last_run_win=win)) for m in samples]
st.session_state.result = ([x['last_run'] for x in settings],
                           [_v16412_schedule_done(x, day) for x in settings],
                           _v16412_schedule_done(dict(settings[-1], t0='01:00'), day))
'''
        at = self.play(code=WIRING, body=body)
        dates, flags, narrowed = at.session_state.result
        self.assertEqual(dates, [''] * 11 + ['2026-11-19', '2026-11-20'])
        self.assertEqual(flags, [False] * 12 + [True])
        self.assertFalse(narrowed)

    def test_verified_marker_blocks_repeat_execution(self):
        at = self.play(marker='2026-11-20', win='00:00-23:59')
        self.assertNotIn('test_fetched', at.session_state)
        self.assertTrue(any('賽果已完成自動輸入' in c.value for c in at.caption))

    def test_next_calendar_day_does_not_replay_backlog(self):
        at = self.play(delta=86400)
        self.assertEqual(at.session_state.test_fetched, [('2026-11-21', {'require_complete': True})])
        self.assertNotIn('test_writes', at.session_state)
        self.assert_not_complete(at)

    def test_setting_saved_and_read_back_from_sql_only(self):
        body = '''
from sqlalchemy import create_engine
engine = create_engine('sqlite://')
_has_secret = lambda k: k == 'DB_URL'
get_hkt_now = lambda: datetime(2026, 11, 21, 5, 0, tzinfo=HKT)
_v163_sql_engine = lambda: engine
cfg = dict(enabled='1', time='05:00', t0='00:00', t1='23:59',
           last_run='2026-11-20', last_run_win='00:00-23:59')
success = _v1511_sched_save(cfg)
saved = _v1511_sched_load(fresh=True)
_v163_sql_engine = lambda: None
failure = _v1511_sched_save(cfg)
st.session_state.result = (success, saved, failure, _v1511_sched_load(fresh=True))
'''
        at = self.play(code='', body=body)
        success, saved, failure, unreadable = at.session_state.result
        self.assertTrue(success)
        self.assertEqual(saved['time'], '05:00')
        self.assertEqual(saved['last_run'], '2026-11-20')
        self.assertFalse(failure)
        self.assertIsNone(unreadable)
        self.assertFalse(Path('football_schedule_db.csv').is_file())

    def test_unscoped_legacy_marker_rechecks_authoritative_rows_without_writing_bets(self):
        at = self.play('settled', marker='2026-11-20')
        self.assertEqual(at.session_state.test_marker['last_run'], '2026-11-20')
        self.assertNotIn('test_writes', at.session_state)

    def test_open_rows_are_only_settled(self):
        at = self.play('open')
        self.assertEqual(at.session_state.test_writes, ['SETTLE_CONFIRM'])
        self.assertEqual(at.session_state.test_db.iloc[0]['Status'], 'Settled')

    def test_one_team_or_other_date_does_not_match(self):
        at = self.play('nearby')
        self.assertEqual(at.session_state.test_writes, ['BET', 'SETTLE_CONFIRM'])
        self.assertEqual(len(at.session_state.test_db), 3)
        self.assertEqual(list(at.session_state.test_db['Match_Date']).count('2026-11-19'), 1)

    def test_read_only_device_never_starts(self):
        at = self.play('viewer')
        self.assertNotIn('test_fetched', at.session_state)
        self.assert_not_complete(at)

    def test_incomplete_feed_throttled_then_recovers(self):
        at = self.play('delayed')
        self.assert_not_complete(at)
        self.assertNotIn('test_writes', at.session_state)
        self.assertTrue(any('賽果未讀齊' in c.value for c in at.caption))
        at.session_state['test_delta'] = 299
        self.rerun(at)
        self.assertEqual(len(at.session_state.test_fetched), 1)
        at.session_state['test_delta'] = 300
        self.rerun(at)
        self.assertEqual(len(at.session_state.test_fetched), 2)
        self.assertEqual(at.session_state.test_marker['last_run'], '2026-11-20')

    def test_empty_or_outside_window_no_false_completion(self):
        for mode in ('empty', 'outside', 'cfg_down', 'db_down'):
            with self.subTest(mode=mode):
                at = self.play(mode)
                self.assert_not_complete(at)
                self.assertNotIn('test_writes', at.session_state)

    def test_partial_batch_is_not_replayed_after_throttle(self):
        at = self.play('partial')
        self.assert_not_complete(at)
        self.assertEqual(at.session_state.test_writes, ['BET', 'SETTLE_CONFIRM'])
        self.assertTrue(any('部分未完成' in c.value for c in at.caption))
        at.session_state['test_delta'] = 301
        self.rerun(at)
        self.assertEqual(at.session_state.test_writes, ['BET', 'SETTLE_CONFIRM'])
        self.assert_not_complete(at)

    def test_scan_without_eligible_bet_is_not_done(self):
        at = self.play('no_picks')
        self.assert_not_complete(at)
        self.assertNotIn('test_writes', at.session_state)

    def test_settlement_failure_or_timeout_preserves_open_and_does_not_resend(self):
        for mode in ('settle_failure', 'settle_timeout'):
            with self.subTest(mode=mode):
                for name in ('auto_status.csv', '.fbapp_daily_status.csv'):
                    Path(name).unlink(missing_ok=True)
                at = self.play(mode)
                self.assert_not_complete(at)
                self.assertEqual(at.session_state.test_db.iloc[0]['Status'], 'Open')
                self.assertEqual(at.session_state.df_db.iloc[0]['Status'], 'Open')
                self.assertEqual(at.session_state.test_writes, ['SETTLE_CONFIRM'])
                self.assertTrue(any('勿重送' in w.value for w in at.warning))
                with open('.fbapp_daily_status.csv', encoding='utf-8') as f:
                    status = next(csv.DictReader(f))
                self.assertEqual((status['active'], status['done']), ('0', '0'))
                at.session_state['test_delta'] = 301
                self.rerun(at)
                self.assertEqual(at.session_state.test_writes, ['SETTLE_CONFIRM'])

    def test_acknowledged_but_not_visible_in_authoritative_db_is_not_done(self):
        at = self.play('no_commit')
        self.assert_not_complete(at)
        self.assertEqual(at.session_state.test_db.iloc[0]['Status'], 'Open')
        self.assertEqual(at.session_state.df_db.iloc[0]['Status'], 'Open')
        self.assertEqual(at.session_state.v15_auto['done'], 0)

    def test_marker_failure_not_complete_or_reinserts(self):
        at = self.play('marker_failure')
        self.assert_not_complete(at)
        self.assertEqual(at.session_state.test_writes, ['BET', 'SETTLE_CONFIRM'])
        self.assertTrue(any('完成日期未獲確認' in c.value for c in at.caption))
        at.session_state['test_delta'] = 301
        self.rerun(at)
        self.assertEqual(at.session_state.test_marker_attempts, 2)
        self.assertEqual(at.session_state.test_writes, ['BET', 'SETTLE_CONFIRM'])
        self.assert_not_complete(at)

    def test_pagination_complete_dedup_and_hong_kong_seconds(self):
        at = self.play('ok', code=FEED_WIRING, body='')
        matches, message = at.session_state.result
        self.assertIsNotNone(matches, message)
        self.assertEqual([m['hkjc_match_id'] for m in matches], [11, 22])
        self.assertEqual(int(matches[-1]['match_time']) - int(matches[0]['match_time']), 86399)

    def test_pagination_partial_or_wrong_not_eligible_for_writes(self):
        for mode in ('feed_gap', 'wrong_page', 'missing_metadata', 'bad_kickoff', 'wrong_date'):
            with self.subTest(mode=mode):
                at = self.play(mode, code=FEED_WIRING, body='')
                rows, message = at.session_state.result
                self.assertIsNone(rows)
                self.assertIn('賽果未讀齊', message)
