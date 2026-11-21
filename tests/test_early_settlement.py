"""Deterministic provider, payout, runner and schedule tests; no real endpoints."""
import ast
import json
import os
import re
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import fb_app_early_settlement as es


APP = Path(__file__).resolve().parents[1] / 'FB-app.py'


def app_defs(*names, **context):
    tree = ast.parse(APP.read_text())
    def selected(n):
        return ((isinstance(n, ast.FunctionDef) and n.name in names) or
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)))
    ns = dict(pd=pd, re=re, os=os, datetime=datetime, timedelta=timedelta, HKT=es.HKT,
              _early16413=es, **context)
    exec(compile(ast.Module(body=[n for n in tree.body if selected(n)], type_ignores=[]), str(APP), 'exec'), ns)
    return ns


def norm(x):
    return str(x or '').replace(' ', '')


KO = datetime(2026, 11, 20, 23, 59, 37, tzinfo=es.HKT)
EV = es.Evidence('999001', '主隊', '客隊', KO, (2, 1), (1, 0), (5, 2), (0, 1, 1, 3))
PAYOUT_NAMES = ('calculate_settlement', '_tipsme_cap_value', '_tipsme_oh_entry',
                '_oh_sort_ts', 'merge_odds_history_snapshots', 'parse_dt_for_comparison', 'v10_parse_time')
PAYOUT = app_defs(*PAYOUT_NAMES, json=json, get_hkt_now=lambda: KO + timedelta(hours=3))


def payload(ev=EV):
    rec = dict(home=ev.home, away=ev.away, matchId=int(ev.mid),
               fullMatchTimeUtc=ev.kickoff.astimezone(es.UTC).replace(tzinfo=None).isoformat(),
               fullMatchTime=ev.kickoff.strftime('%m/%d/%Y %H:%M:%S'), extraScore='', matchStatus='8',
               liveTime='完', additionalTime='', score='2-1', halfScore='1-0', corner='5-2',
               homeRedCard='0', awayRedCard='1', homeYellowCard='1', awayYellowCard='3')
    page = dict(matchId=ev.mid, liveScore=rec.copy(), initialFixtures={'fixtures': {
        'mainDetails': [{'time': '中', 'awayEventDetails': '1-0'}, {'time': '90+8', 'homeEventType': '換人'},
                        {'time': '完', 'awayEventDetails': '2-1'}],
        'techDetails': [{'time': '角球', 'home': '5', 'away': '2'}],
    }})
    return rec, page


def parse(rec, page):
    html = '<html><script type="application/json" id="__NEXT_DATA__">' + json.dumps(
        {'props': {'pageProps': page}}, ensure_ascii=False) + '</script></html>'
    return es.parse_final(EV.mid, rec, html, norm)


def slip(rid='A', ev=EV, pre=False, **kw):
    return {**dict.fromkeys(es.COLUMNS), 'ID': rid, 'Status': 'Open', 'Tipsme_Match_ID': ev.mid,
            'Match_Date': ev.kickoff.date().isoformat(), 'Home_Team': ev.home, 'Away_Team': ev.away,
            'Match': f'{ev.home} vs {ev.away}', 'Bet_Type': '入球大小' if pre else '入球大小 (即場)',
            'Selection': 'Over', 'Initial_Line': '2.75', 'Initial_Odds': '1.9',
            'System_Stake': '100', 'User_Stake': '20', 'Odds_History': '[]', **kw}


def odds(ev=EV):
    cutoff = ev.kickoff.replace(second=0, microsecond=0)
    quotes = []
    for delta, up in ((timedelta(minutes=-10), '1.80'), (timedelta(minutes=-1), '1.85'),
                      (timedelta(seconds=10), '2.00'), (timedelta(minutes=1), '2.50')):
        t = (cutoff + delta).astimezone(es.UTC).replace(tzinfo=None)
        quotes.append(dict(number_of_goals='[2.5/3]', big_odds=up, small_odds='1.95', create_time=t.isoformat()))
    return dict(hkjcHomeTeamName=ev.home, hkjcAwayTeamName=ev.away, hiloOdds=quotes)


def plan(rows, ev, raw=None):
    return es.build_plan(rows, ev, raw, norm=norm, calculate=PAYOUT['calculate_settlement'],
                         parse_line=PAYOUT['_tipsme_cap_value'], odds_entry=PAYOUT['_tipsme_oh_entry'],
                         merge_history=PAYOUT['merge_odds_history_snapshots'])


def fixture(ev=EV, **kw):
    return dict(hkjc_match_id=int(ev.mid), home=ev.home, away=ev.away, match_time=ev.kickoff.timestamp(),
                finished=True, home_goal=ev.goals[0], away_goal=ev.goals[1],
                home_corner=ev.corners[0], away_corner=ev.corners[1]) | kw


class EvidenceTests(unittest.TestCase):
    def test_normal_regulation_and_hong_kong_midnight(self):
        self.assertEqual(EV, parse(*payload()))
        self.assertEqual(EV.kickoff.astimezone(es.UTC).hour, 15)

    def test_zero_result_is_not_missing(self):
        rec, page = payload()
        rec.update(score='0-0', halfScore='0-0', corner='0-0')
        page['liveScore'] = rec.copy()
        for e in page['initialFixtures']['fixtures']['mainDetails']:
            if 'awayEventDetails' in e:
                e['awayEventDetails'] = '0-0'
        page['initialFixtures']['fixtures']['techDetails'][0].update(home='0', away='0')
        self.assertEqual(parse(rec, page).goals, (0, 0))

    def test_one_source_unavailable_or_conflicting(self):
        for key, value in dict(score='3-1', halfScore='2-1', corner='3-2', home='另隊',
                               matchId=999002, fullMatchTime='11/21/2026 23:59:37',
                               homeYellowCard='2').items():
            with self.subTest(key=key):
                rec, page = payload()
                page['liveScore'][key] = value
                with self.assertRaises(es.Review):
                    parse(rec, page)
        with self.assertRaises(es.Review):
            es.parse_final(EV.mid, payload()[0], 'temporarily unavailable', norm)

    def test_delayed_interrupted_cancelled_abandoned_and_extra_time(self):
        for status, label, extra in [('1', '推遲', ''), ('4', '暫停', ''), ('9', '取消', ''),
                                     ('8', '腰斬', ''), ('8', '完', '3-2'), ('8', '中', '')]:
            with self.subTest(label=label, extra=extra):
                rec, page = payload()
                rec.update(matchStatus=status, liveTime=label, extraScore=extra)
                with self.assertRaises(es.Review):
                    parse(rec, page)

    def test_incomplete_and_non_regulation_event_page(self):
        for adjustment in ('missing', 'minutes', 'penalties', 'early', 'no_corners', 'mismatched_score'):
            with self.subTest(adjustment=adjustment):
                rec, page = payload()
                f = page['initialFixtures']['fixtures']
                if adjustment == 'missing':
                    page['initialFixtures'] = None
                if adjustment in ('minutes', 'penalties', 'early'):
                    f['mainDetails'][1]['time'] = {'minutes': '105', 'penalties': '十二碼', 'early': '80'}[adjustment]
                if adjustment == 'no_corners':
                    f['techDetails'].clear()
                if adjustment == 'mismatched_score':
                    f['mainDetails'][2]['awayEventDetails'] = '1-1'
                with self.assertRaises(es.Review):
                    parse(rec, page)

    def test_network_accepts_no_redirect_or_old_cache(self):
        for code, headers in [(302, {}), (500, {}), (200, {'Age': '61'}), (200, {'Age': 'unknown'})]:
            with self.subTest(code=code, headers=headers), patch.object(es.requests, 'get') as get:
                get.return_value = SimpleNamespace(status_code=code, headers=headers)
                with self.assertRaises(es.Review):
                    es.Tipsme(norm).final(EV.mid)
                self.assertFalse(get.call_args.kwargs['allow_redirects'])
                self.assertEqual(get.call_args.kwargs['timeout'], (3, 5))


class GroupAndPlanTests(unittest.TestCase):
    def test_requires_same_id_date_and_both_teams_for_whole_group(self):
        self.assertEqual(len(es.exact_open_group([slip(Tipsme_Match_ID=' +0999001.0 '), slip('B')], EV, norm)), 2)
        for row in (slip('B', Match_Date=None), slip('B', Match_Date='2026-11-21'),
                    slip('B', Home_Team='另隊'), slip('B', Tipsme_Match_ID=None),
                    slip('B', Status='Settled'), slip('B', Tipsme_Match_ID='222222'),
                    slip('B', Home_Team='客隊', Away_Team='主隊'), slip()):
            with self.subTest(row=row), self.assertRaises(es.Review):
                es.exact_open_group([slip(), row], EV, norm)

    def test_reuses_production_payout_for_quarter_lines_and_zero_stakes(self):
        changes = plan([slip(), slip('Z', System_Stake='0', User_Stake='0')], EV)
        self.assertEqual((changes['A']['System_Profit'], changes['A']['User_Profit'],
                          changes['A']['System_Payout'], changes['A']['Half_Score']), (45.0, 9.0, 145.0, '1-0'))
        self.assertEqual(changes['Z']['User_Payout'], 0.0)
        self.assertEqual(changes['A']['Away_Yellow'], 3)
        self.assertFalse(set(es.INPUTS) & changes['A'].keys())
        self.assertNotIn('Odds_History', changes['A'])

    def test_clv_same_market_line_excludes_kickoff_minute_and_later(self):
        changes = plan([slip(pre=True)], EV, odds())
        saved = json.loads(changes['A']['Odds_History'])
        self.assertEqual([x['upper'] for x in saved], [1.80, 1.85])
        self.assertEqual(saved[-1]['record_time'], '2026-11-20 23:58')
        originals = json.dumps(saved, ensure_ascii=False)
        self.assertNotIn('Odds_History', plan([slip(pre=True, Odds_History=originals)], EV, odds())['A'])

    def test_no_odds_wrong_line_swapped_or_contaminated_history_blocks_the_group(self):
        for adjustments in ({'hiloOdds': None}, {'hkjcHomeTeamName': '客隊'},
                            {'hiloOdds': odds()['hiloOdds'][2:]},
                            {'hiloOdds': [{**o, 'number_of_goals': '2.5'} for o in odds()['hiloOdds']]}):
            with self.subTest(adjustments=adjustments), self.assertRaises(es.Review):
                plan([slip(), slip('B', pre=True)], EV, odds() | adjustments)
        for stamp in ('11-20 23:59', '11-21 00:00', '23:00', 'bad'):
            history = json.dumps([{'type': '入球大小', 'line': 2.75, 'record_time': stamp,
                                   'upper': 1.85, 'lower': 1.95}])
            with self.subTest(stamp=stamp), self.assertRaises(es.Review):
                plan([slip(pre=True, Odds_History=history)], EV, odds())

    def test_unsupported_or_invalid_financial_arguments_do_not_default_to_push(self):
        for bad in (dict(Bet_Type='半場大小'), dict(Initial_Odds='nan'), dict(Initial_Line='2.12'),
                    dict(User_Stake='-1'), dict(Selection='Away')):
            with self.subTest(bad=bad), self.assertRaises(es.Review):
                plan([slip(**bad)], EV)

    def test_read_only_reconciliation_flags_missing_wrong_day_open_and_wrong_result(self):
        ready = slip() | plan([slip()], EV)['A']
        for rows, expected in (([ready], set()), ([], {EV.mid}), ([slip()], {EV.mid}),
                               ([ready | {'Home_Goal': '7'}], {EV.mid}),
                               ([ready | {'Match_Date': '2026-11-19'}], {EV.mid}),
                               ([ready, slip('P', Tipsme_Match_ID=None, Home_Team='他隊', Match='他隊 VS 客隊')],
                                {'legacy-open'})):
            self.assertEqual(es.reconciliation_pending(rows, [fixture()], KO.date(), norm), expected)
        self.assertEqual(es.reconciliation_pending([ready], [fixture(home_corner=None)], KO.date(), norm), {EV.mid})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.store.claim.return_value = (KO, [EV.mid])
        self.store.observe.return_value = True
        self.store.group.return_value = [slip()]
        self.store.observed_until.return_value = KO
        self.store.write.return_value = {'A'}
        self.provider = Mock()
        self.provider.final.return_value = EV

    def test_off_and_invalid_never_touch_store_or_provider(self):
        self.assertEqual(es.mode(None), 'off')
        self.assertEqual(es.mode('on'), 'invalid')
        for m in ('off', 'invalid'):
            self.assertEqual(es.early_scan(self.store, self.provider, m, plan), {})
        self.assertEqual(self.store.mock_calls, [])
        self.assertEqual(self.provider.mock_calls, [])

    def test_observe_records_would_settle_without_ledger_writes(self):
        self.assertEqual(es.early_scan(self.store, self.provider, 'observe', plan)[EV.mid][0], 'would_settle')
        self.store.write.assert_not_called()
        self.provider.odds.assert_not_called()

    def test_hold_defers_plan_and_provider_recheck(self):
        self.store.observe.return_value = False
        self.assertEqual(es.early_scan(self.store, self.provider, 'write', plan)[EV.mid][0], 'observing')
        self.store.group.assert_not_called()
        self.provider.final.assert_called_once_with(EV.mid)

    def test_fresh_write_rechecks_then_calls_one_transaction(self):
        self.assertEqual(es.early_scan(self.store, self.provider, 'write', plan), {EV.mid: ('committed', {'A'})})
        self.assertEqual(self.provider.final.call_count, 2)
        self.store.write.assert_called_once()

    def test_correction_and_second_source_failure_reset_instead_of_writing(self):
        for outcome, reason in ((replace(EV, goals=(3, 1)), 'final_corrected'),
                                (es.Review('provider_unavailable'), 'provider_unavailable')):
            self.provider.final.side_effect = [EV, outcome]
            self.assertEqual(es.early_scan(self.store, self.provider, 'write', plan)[EV.mid][0], reason)
            self.store.write.assert_not_called()
            self.assertEqual(self.store.observe.call_args.args[4], reason)

    def test_commit_exception_never_falls_back_to_a_second_write(self):
        self.store.write.side_effect = TimeoutError
        self.assertEqual(es.early_scan(self.store, self.provider, 'write', plan)[EV.mid][0],
                         'sql_or_runtime_unconfirmed')
        self.store.write.assert_called_once()


class ScheduleIntegration(unittest.TestCase):
    def test_write_mode_never_enters_legacy_scheduler_even_if_open_or_no_db(self):
        cfg = dict(enabled='1', time='05:00', t0='00:00', t1='23:59')
        old_path = Mock(side_effect=AssertionError('legacy ledger path'))
        review = Mock()
        st = SimpleNamespace(session_state={})
        ns = app_defs('_v1511_sched_try_start', _v16413_mode=lambda: 'write',
                      _v162_can_automate=lambda: True, _v10_settings_allow_automation=lambda: True,
                      get_hkt_now=lambda: (KO + timedelta(days=1)).replace(hour=5),
                      _v1511_sched_load=lambda **_: cfg, _v16412_schedule_done=lambda *_: False,
                      _v15_load_remote=lambda: None, _v15_status_phase=lambda _: 'missing',
                      _v16412_window=lambda _: '00:00-23:59', st=st,
                      tipsme_fetch_day_results=lambda *a, **kw: ([fixture()], 'ok'),
                      _v15_build_queue_date=lambda q, *a: q,
                      _v16412_authoritative_bets=old_path, _v16413_review_schedule=review)
        self.assertFalse(ns['_v1511_sched_try_start'](pd.DataFrame([slip()])))
        review.assert_called_once()
        old_path.assert_not_called()
        self.assertNotIn('v15_auto', st.session_state)

    def test_a_previously_queued_daily_batch_is_stopped_before_any_step(self):
        st = SimpleNamespace(session_state={'v15_auto': {'active': True, 'sched': True, 'step': 'BET'}},
                             rerun=Mock())
        taken = Mock(side_effect=AssertionError('old batch'))
        ns = app_defs('v15_auto_executor', _v16413_mode=lambda: 'write', _v15_log=Mock(),
                      _v15_taken_over=taken, st=st)
        ns['v15_auto_executor'](None, None, None)
        taken.assert_not_called()
        self.assertFalse(st.session_state['v15_auto']['active'])

    def test_marker_only_on_readback_success(self):
        for response in ((set(), True), ({EV.mid}, True), (set(), False), RuntimeError('db offline')):
            with self.subTest(response=response), patch.object(es, 'Postgres') as postgres:
                check = postgres.return_value.review_day
                if isinstance(response, Exception):
                    check.side_effect = response
                else:
                    check.return_value = response
                mark, notice = Mock(return_value=True), Mock()
                ns = app_defs('_v16413_review_schedule', _v163_sql_engine=lambda: object(), _v14_norm_team=norm,
                              _v1511_mark_done=mark, _v16412_schedule_notice=notice)
                ns['_v16413_review_schedule'](KO.date(), 'all', [fixture()])
                self.assertEqual(mark.call_count, int(response == (set(), True)))
                notice.assert_called_once()

    def test_early_fragment_off_or_viewer_does_not_connect(self):
        for m, can in (('off', True), ('observe', False), ('write', False), ('invalid', True)):
            engine = Mock()
            ns = app_defs('_v16413_early_body', _v16413_mode=lambda: m, _v162_can_automate=lambda: can,
                          _v163_sql_engine=engine)
            ns['_v16413_early_body']()
            engine.assert_not_called()
