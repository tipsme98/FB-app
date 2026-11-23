"""Isolated native PostgreSQL tests; no configured production DSN is accepted."""
import ast
from datetime import date, datetime
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import URL

SOCKET = os.getenv('FBAPP_EGRESS_TEST_SOCKET')
APP = Path(__file__).resolve().parents[1] / 'FB-app.py'
TREE = ast.parse(APP.read_text())
WORD = {'DB_COLUMNS', 'process_legacy_columns', 'enforce_columns', '_v161_sql_read_bets',
        '_v1824_read_clv_ledger_snapshot', '_v161_refresh_db_from_sql', '_v18_daily_bet_precheck',
        'v10_md_series', '_v16412_match_rows', '_v16412_settled', '_v16412_finish_schedule',
        '_v16412_window', '_v16412_authoritative_bets'}


def defs(**extras):
    nodes = [n for n in TREE.body if getattr(n, 'name', None) in WORD or
             (isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) in WORD)]
    ns = dict(pd=pd, re=re, HAS_SQLALCHEMY=True, **extras)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP), 'exec'), ns)
    return ns


@unittest.skipUnless(SOCKET, 'run tests/run_egress_postgres.sh for native database coverage')
class EgressReads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(SOCKET)
        if (not re.fullmatch(r'/tmp/fbapp-egress-test\.[A-Za-z0-9]+/socket', SOCKET)
                or path.resolve() != path or path.stat().st_uid != os.getuid()
                or path.stat().st_mode & 0o077):
            raise RuntimeError('native tests require their own temporary Unix socket')
        url = URL.create('postgresql+psycopg2', username='fbapp_egress_test', database='postgres',
                         query={'host': SOCKET, 'port': '55424'})
        cls.engine = create_engine(url)
        cls.addClassCleanup(cls.engine.dispose)
        with cls.engine.begin() as c:
            who = c.execute(text('SELECT current_user, inet_server_addr(), current_database()')).one()
            if who != ('fbapp_egress_test', None, 'postgres') or c.scalar(text("SELECT to_regclass('public.football_bets')")):
                raise RuntimeError('native test server is not empty and local')
            c.execute(text('CREATE ROLE egress_read_limited'))
            c.execute(text('CREATE TABLE public.football_bets_deleted (id text PRIMARY KEY)'))

    def setUp(self):
        self.state, self.queries = {}, []
        self.ns = defs(st=SimpleNamespace(session_state=self.state, secrets={'DB_URL': 'synthetic'}),
                       _has_secret=lambda key: key == 'DB_URL', _v163_sql_engine=lambda: self.engine,
                       _v164_drop_tombs=self.drop_tombs, _v159_merge_db=lambda cloud, local:
                       pd.concat([cloud, local]).drop_duplicates('ID', keep='first').reset_index(drop=True),
                       _v1512_fp=lambda frame: dict(zip(frame.ID, frame.Status)),
                       _v15_sched_date=lambda match: match['day'],
                       v11_clean_name=lambda s: s, _v14_norm_team=lambda s: s.strip().lower(),
                       _v16412_date=date.fromisoformat, get_hkt_now=lambda: datetime(2026, 11, 23, 5),
                       _v1511_mark_done=Mock(return_value=True), _v16412_schedule_notice=Mock(),
                       _v15_log=Mock())
        cols = ', '.join(f'"{name}" text' for name in self.ns['DB_COLUMNS'])
        with self.engine.begin() as c:
            c.execute(text(f'DROP TABLE IF EXISTS public.football_bets; CREATE TABLE public.football_bets ({cols})'))
            c.execute(text('TRUNCATE public.football_bets_deleted'))
            c.execute(text('''
                INSERT INTO public.football_bets ("ID","Status","Match","Date","Match_Date","Tipsme_Match_ID","Odds_History")
                VALUES ('A','Settled','Home vs Away','2026-11-22 01:00','2026-11-22','100.0','[opening]'),
                       ('B','Open','Other vs Team','2026-11-21 01:00','2026-11-21','200','[retain]'),
                       ('C','Settled','Legacy vs Pair','2026-11-22 02:00',NULL,NULL,'[legacy]')
            '''))
        def capture(conn, cursor, sql, params, ctx, multi):
            self.queries.append(sql)
        event.listen(self.engine, 'before_cursor_execute', capture)
        self.addCleanup(event.remove, self.engine, 'before_cursor_execute', capture)

    def drop_tombs(self, df):
        with self.engine.connect() as c:
            ids = {r[0] for r in c.execute(text('SELECT id FROM public.football_bets_deleted'))}
        return df.loc[~df.ID.astype(str).isin(ids)]

    def sql(self, stmt):
        with self.engine.begin() as conn:
            conn.execute(text(stmt))

    def wide_count(self):
        return sum('SELECT football_bets."ID"' in s for s in self.queries)

    def read(self):
        return self.ns['_v1824_read_clv_ledger_snapshot']()

    def test_same_snapshot_suppresses_only_duplicate_transfer_and_returns_full_frame(self):
        before = self.ns['_v161_sql_read_bets']()
        self.queries.clear()
        first = self.read()
        first.loc[first.ID.eq('A'), 'Odds_History'] = '(local-only)'
        del first['Match']
        second = self.read()
        pd.testing.assert_frame_equal(before, second)
        self.assertEqual(self.wide_count(), 1)
        self.assertTrue(any('pg_current_snapshot' in q for q in self.queries))

    def test_external_update_insert_delete_are_seen_before_reuse(self):
        self.read()
        self.sql('''UPDATE public.football_bets SET "Odds_History"='[new]' WHERE "ID"='A';
                    DELETE FROM public.football_bets WHERE "ID"='B';
                    INSERT INTO public.football_bets ("ID","Status","Odds_History") VALUES ('D','Open','[snap]');''')
        got = self.read()
        self.assertEqual(set(got.ID), {'A', 'C', 'D'})
        self.assertEqual(got.set_index('ID').loc['A', 'Odds_History'], '[new]')
        self.assertEqual(self.wide_count(), 2)
        pd.testing.assert_frame_equal(got, self.read())
        self.assertEqual(self.wide_count(), 2)

    def test_concurrent_uncommitted_update_hidden_until_commit(self):
        initial = self.read()
        with self.engine.begin() as writer:
            writer.execute(text('UPDATE public.football_bets SET "Odds_History"=\'[new]\' WHERE "ID"=\'A\''))
            pd.testing.assert_frame_equal(self.read(), initial)
        self.assertNotEqual(self.read().set_index('ID').loc['A', 'Odds_History'],
                            initial.set_index('ID').loc['A', 'Odds_History'])

    def test_tombstones_and_local_work_survive_refresh(self):
        key = 'df_db'
        self.state[key] = self.read().query("ID == 'A'").assign(ID='local')
        self.assertTrue(self.ns['_v161_refresh_db_from_sql'](reuse_clv_snapshot=True))
        self.sql("INSERT INTO public.football_bets_deleted VALUES ('A')")
        self.assertTrue(self.ns['_v161_refresh_db_from_sql'](reuse_clv_snapshot=True))
        self.assertEqual(set(self.state[key].ID), {'local', 'B', 'C'})
        self.assertTrue(self.state['_v1641_sql_base'])

    def test_truncate_reinsert_and_schema_change_break_cached_snapshot(self):
        self.read()
        self.sql('''TRUNCATE public.football_bets;
                    ALTER TABLE public.football_bets ADD COLUMN aux text;
                    INSERT INTO public.football_bets ("ID",aux) VALUES ('A','new-schema');''')
        got = self.read()
        self.assertEqual(got[['ID', 'aux']].to_dict('records'), [{'ID': 'A', 'aux': 'new-schema'}])
        self.assertEqual(self.wide_count(), 2)

    def test_probe_error_never_falls_back_to_stored_data(self):
        self.read()
        with patch.object(self.engine, 'connect', side_effect=sqlalchemy.exc.SQLAlchemyError('offline')):
            self.assertIsNone(self.read())
        self.assertNotIn('_v1824_clv_ledger_snapshot', self.state)

    def test_active_row_level_policy_uses_original_path(self):
        self.sql('''ALTER TABLE public.football_bets ENABLE ROW LEVEL SECURITY;
                    CREATE POLICY temporary_policy ON public.football_bets USING ("ID" = 'A');
                    GRANT SELECT ON public.football_bets TO egress_read_limited;''')
        limited = create_engine(self.engine.url)
        self.addCleanup(limited.dispose)
        @event.listens_for(limited, 'connect')
        def assume_role(dbapi, _):
            with dbapi.cursor() as cur:
                cur.execute('SET ROLE egress_read_limited')
            dbapi.commit()
        self.ns['_v163_sql_engine'] = lambda: limited
        self.assertEqual(self.read().ID.tolist(), ['A'])
        self.assertNotIn('_v1824_clv_ledger_snapshot', self.state)

    def finish(self, *, should_mark):
        a = dict(rdate='2026-11-22', t0='00:00', t1='23:59',
                 queue=[dict(day=date(2026, 11, 22), home='Home', away='Away', hkjc_match_id=100),
                        dict(day=date(2026, 11, 22), home='Legacy', away='Pair', hkjc_match_id=300)])
        self.ns['_v16412_finish_schedule'](a)
        self.assertEqual(self.ns['_v1511_mark_done'].called, should_mark)
        self.assertEqual(bool(a.get('daily_verified')), should_mark)

    def test_daily_completion_prunes_columns_but_keeps_legacy_dates_and_id_and_no_write_to_session(self):
        self.state['df_db'] = sentinel = object()
        self.finish(should_mark=True)
        self.assertIs(self.state['df_db'], sentinel)
        self.assertNotIn('_v1641_sql_base', self.state)
        self.assertEqual(self.wide_count(), 0)
        selects = [q for q in self.queries if 'FROM public."football_bets" b' in q]
        self.assertEqual(len(selects), 1)
        for field in ('ID', 'Status', 'Match', 'Tipsme_Match_ID', 'Match_Date', 'Date'):
            self.assertIn(f'b."{field}"', selects[0])
        self.assertNotIn('Odds_History', selects[0])

    def test_daily_tombstone_prevents_false_completion(self):
        self.sql("INSERT INTO public.football_bets_deleted VALUES ('A')")
        self.finish(should_mark=False)
        self.assertEqual(self.wide_count(), 0)

    def test_daily_narrow_failure_falls_back_to_authoritative(self):
        self.ns['_v18_daily_bet_precheck'] = Mock(return_value=None)
        self.finish(should_mark=True)
        self.assertEqual(self.wide_count(), 1)
