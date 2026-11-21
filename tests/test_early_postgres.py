"""Native PostgreSQL tests; use run_early_postgres.sh, never an existing server."""
import os
import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError

import fb_app_early_settlement as es
from test_early_settlement import APP, EV, app_defs, fixture, norm, odds, plan, slip


SOCKET = os.getenv('FBAPP_TEST_SOCKET', '')
APP_DATA = app_defs('DB_COLUMNS')
DB_COLUMNS = APP_DATA['DB_COLUMNS']


@unittest.skipUnless(SOCKET, 'run tests/run_early_postgres.sh to start an isolated native PostgreSQL')
class NativePostgres(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Hardcoded database/user and private temporary Unix socket; no configurable DSN or TCP.
        path = Path(SOCKET)
        if (not re.fullmatch(r'/tmp/fbapp-16413-test\.[A-Za-z0-9]+/socket', SOCKET)
                or path.resolve() != path or path.stat().st_uid != os.getuid()
                or path.stat().st_mode & 0o077):
            raise RuntimeError('refusing to test outside the dedicated temporary PostgreSQL socket')
        cls.engine = create_engine(URL.create('postgresql+psycopg2', username='fbapp_early_test',
                                             database='postgres', query={'host': SOCKET, 'port': '55413'}))
        cls.addClassCleanup(cls.engine.dispose)
        with cls.engine.begin() as c:
            who = c.execute(text('SELECT current_user, inet_server_addr(), current_database()')).one()
            if who != ('fbapp_early_test', None, 'postgres') or c.scalar(text("SELECT to_regclass(:name)"), {'name': es.BETS}):
                raise RuntimeError('test server is not an empty, dedicated local cluster')
            cols = ', '.join(f'"{k}" ' + ('DATE' if k == 'Match_Date' else
                             'NUMERIC' if k in es.INPUTS[2:] else 'TEXT') for k in DB_COLUMNS)
            c.execute(text(f'CREATE TABLE {es.BETS} ({cols})'))
            c.execute(text(f'CREATE TABLE {es.TOMBS} (id TEXT PRIMARY KEY, deleted_at TEXT)'))
            c.execute(text('CREATE ROLE anon; CREATE ROLE authenticated'))
            c.execute(text('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO anon, authenticated'))
        migration = (APP.parent / 'migrations/16413_early_settlement.sql').read_text()
        with cls.engine.connect().execution_options(isolation_level='AUTOCOMMIT') as c:
            with c.connection.driver_connection.cursor() as cursor:
                cursor.execute(migration)
                cursor.execute(migration)

    def setUp(self):
        self.store = es.Postgres(self.engine, norm)
        self.now = es.datetime.now(es.UTC).replace(microsecond=0)
        self.ev = replace(EV, kickoff=(self.now - timedelta(hours=6)).astimezone(es.HKT))
        p = patch.object(es, '_now', lambda _: self.now)
        p.start()
        self.addCleanup(p.stop)
        with self.engine.begin() as c:
            c.execute(text(f'TRUNCATE {es.BETS}, {es.TOMBS}, {es.STATE}, {es.RECEIPT}'))
            c.execute(text(f'UPDATE {es.LEASE} SET last_scan=NULL'))

    def insert(self, *rows):
        for row in rows:
            columns = ','.join(f'"{k}"' for k in row)
            args = ','.join(':' + k for k in row)
            with self.engine.begin() as c:
                c.execute(text(f'INSERT INTO {es.BETS} ({columns}) VALUES ({args})'), row)

    def sql(self, query, **params):
        with self.engine.begin() as c:
            result = c.execute(text(query), params)
            return result.fetchall() if result.returns_rows else None

    def tick(self, *, minutes=0, ev=None, missing=False, code='normal', manual=False):
        self.now += timedelta(minutes=minutes)
        lease, queue = self.store.claim('write')
        self.assertIn(self.ev.mid, queue)
        return lease, self.store.observe(lease, self.now, self.ev.mid, None if missing else ev or self.ev,
                                         code, manual)

    def prime(self, *, rows=None):
        self.insert(*(rows if rows is not None else [slip(ev=self.ev), slip('B', ev=self.ev, pre=True)]))
        self.assertFalse(self.tick()[1])
        for t in range(1, 12):
            self.assertFalse(self.tick(minutes=10)[1], f'early at minute {t*10}')
        lease, mature = self.tick(minutes=10)
        self.assertTrue(mature)
        group = self.store.group(self.ev)
        return lease, group, plan(group, self.ev, odds(self.ev))

    def current_status(self):
        return [r[0] for r in self.sql(f'SELECT "Status" FROM {es.BETS} ORDER BY "ID"')]

    def test_migration_is_idempotent_and_anon_and_authenticated_are_revoked(self):
        rows = self.sql("""SELECT c.relrowsecurity, has_table_privilege('anon',c.oid,'SELECT'),
            has_table_privilege('authenticated',c.oid,'UPDATE') FROM pg_class c
            WHERE c.relname IN ('fb_early_state_16413','fb_early_lease_16413','fb_early_receipt_16413')""")
        self.assertEqual(rows, [(True, False, False)] * 3)
        self.assertEqual(self.sql(f'SELECT count(*) FROM {es.LEASE}'), [(1,)])

    def test_database_clock_starts_at_first_observed_ft_and_writes_at_two_hours_only(self):
        lease, group, changes = self.prime()
        self.assertEqual(self.current_status(), ['Open', 'Open'])
        self.assertEqual(self.store.write(lease, self.ev, group, changes, self.now), {'A', 'B'})
        self.assertEqual(self.current_status(), ['Settled', 'Settled'])
        self.assertEqual(self.sql(f'''SELECT "Initial_Odds", "User_Stake", "Half_Score", "Away_Red"
                                     FROM {es.BETS} ORDER BY "ID"'''),
                         [(group[0]['Initial_Odds'], 20, '1-0', '1')] * 2)
        self.assertEqual(self.store.review_day(self.ev.kickoff.date(), [fixture(self.ev)]), (set(), True))

    def test_82_candidates_preserve_first_stability_window_and_room_for_new_probes(self):
        """The observed production queue size, with synthetic finals and no ledger writes."""
        mids = [str(int(self.ev.mid) + i) for i in range(82)]
        self.sql(f'''INSERT INTO {es.BETS} ("ID", "Tipsme_Match_ID", "Status", "Match_Date")
            SELECT 'queue-' || n, n, 'Open', :day FROM unnest(CAST(:mids AS text[])) AS n''',
                 mids=mids, day=self.ev.kickoff.date())
        seen, ready, new_per_scan = set(), {}, []
        for scan in range(13):
            if scan:
                self.now += timedelta(minutes=11)  # scheduler jitter; do not assume exact cadence
            lease, queue = self.store.claim('observe')
            self.assertEqual(len(set(queue)), es.MAX_MATCHES)
            new_per_scan.append(len(set(queue) - seen))
            seen.update(queue)
            for mid in queue:
                if self.store.observe(lease, self.now, mid, replace(self.ev, mid=mid), 'normal_final'):
                    ready.setdefault(mid, scan * 11)
                    self.store.record_reason(lease, mid, 'would_settle')
        self.assertEqual(ready.get(mids[0]), 121)
        self.assertGreaterEqual(min(new_per_scan), 2)
        self.assertEqual(self.sql(f"SELECT count(*) FROM {es.BETS} WHERE \"Status\"='Open'"), [(82,)])
        self.assertEqual(self.sql(f'SELECT count(*) FROM {es.RECEIPT}'), [(0,)])

    def test_follow_up_slots_exclude_expired_and_already_mature_observations(self):
        # The two old observations should wait behind unseen probe candidates.
        # Only the fresh, unfinished timer should be promoted.
        special = [str(int(self.ev.mid) + i) for i in range(12)]
        self.sql(f'''INSERT INTO {es.BETS} ("ID", "Tipsme_Match_ID", "Status", "Match_Date")
            SELECT 'case-' || n, n, 'Open', :day FROM unnest(CAST(:mids AS text[])) AS n''',
                 mids=special, day=self.ev.kickoff.date())
        for mid, first_age, last_age in [(special[-3], 41, 31), (special[-2], 130, 10), (special[-1], 15, 5)]:
            self.sql(f'''INSERT INTO {es.STATE}
              (match_id, fingerprint, first_good, last_good, last_attempt, reason)
              VALUES (:mid, repeat('a',64), :first, :last, :last, 'normal_final')''',
              mid=mid, first=self.now - timedelta(minutes=first_age),
              last=self.now - timedelta(minutes=last_age))
        _, queue = self.store.claim('observe')
        self.assertIn(special[-1], queue)
        self.assertNotIn(special[-2], queue)
        self.assertNotIn(special[-3], queue)
        self.assertEqual(len(queue), es.MAX_MATCHES)

    def test_write_gate_itself_rejects_119_minutes_and_stale_recheck(self):
        self.insert(slip(ev=self.ev))
        self.tick()
        self.tick(minutes=30)
        self.tick(minutes=30)
        self.tick(minutes=30)
        lease, mature = self.tick(minutes=29)
        self.assertFalse(mature)
        group = self.store.group(self.ev)
        with self.assertRaisesRegex(es.Review, 'expired_write_evidence'):
            self.store.write(lease, self.ev, group, plan(group, self.ev), self.now)
        self.now += timedelta(minutes=1)
        with self.assertRaisesRegex(es.Review, 'expired_write_evidence'):
            self.store.write(lease, self.ev, group, plan(group, self.ev), self.now - timedelta(seconds=61))
        self.assertEqual(self.current_status(), ['Open'])

    def test_global_scan_limit_and_old_open_are_not_selected(self):
        for i in range(9):
            self.insert(slip(str(i), ev=replace(self.ev, mid=str(999010+i))))
        self.insert(slip('old', ev=self.ev, Match_Date=(self.ev.kickoff.date()-timedelta(days=30)).isoformat()))
        _, matches = self.store.claim('write')
        self.assertEqual(len(matches), es.MAX_MATCHES)
        self.assertNotIn(self.ev.mid, matches)

    def test_outage_and_greater_than_30_minute_gap_reset_stability(self):
        self.insert(slip(ev=self.ev))
        self.tick()
        self.tick(minutes=30)
        first, = self.sql(f'SELECT first_good FROM {es.STATE}')[0]
        self.assertEqual(first, self.now - timedelta(minutes=30))
        self.tick(minutes=31)
        self.assertEqual(self.sql(f'SELECT first_good FROM {es.STATE}'), [(self.now,)])
        self.tick(minutes=10, missing=True, code='provider_unavailable')
        self.assertEqual(self.sql(f'SELECT first_good,last_good FROM {es.STATE}'), [(None, None)])
        self.tick(minutes=10)
        self.assertEqual(self.sql(f'SELECT first_good FROM {es.STATE}'), [(self.now,)])

    def test_delay_resume_correction_and_manual_latch(self):
        self.insert(slip(ev=self.ev))
        self.tick(missing=True, code='postponed')
        self.tick(minutes=10)
        self.tick(minutes=10, ev=replace(self.ev, goals=(3, 1)))
        self.assertEqual(self.sql(f'SELECT first_good FROM {es.STATE}'), [(self.now,)])
        self.tick(minutes=10, missing=True, code='abandoned', manual=True)
        self.now += timedelta(minutes=10)
        _, queue = self.store.claim('write')
        self.assertNotIn(self.ev.mid, queue)
        self.assertEqual(self.current_status(), ['Open'])

    def test_single_global_lease_and_superseded_worker_fencing(self):
        self.insert(slip(ev=self.ev))
        gate = Barrier(2)
        def claim():
            gate.wait()
            return self.store.claim('write')
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: claim(), range(2)))
        (old,) = [lease for lease, hits in outcomes if hits]
        self.now += es.POLL
        new, hits = self.store.claim('write')
        self.assertIn(self.ev.mid, hits)
        self.assertNotEqual(old, new)
        with self.assertRaisesRegex(es.Review, 'scan_superseded'):
            self.store.observe(old, self.now, self.ev.mid, self.ev, 'late')

    def test_two_writers_same_match_only_one_commit(self):
        lease, group, changes = self.prime()
        barrier = Barrier(2)
        def writer():
            barrier.wait()
            try:
                return self.store.write(lease, self.ev, group, changes, self.now)
            except es.Review as exc:
                return exc.code
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(lambda _: writer(), range(2)))
        self.assertCountEqual(result, [{'A', 'B'}, 'non_open_or_ambiguous_group'])
        self.assertEqual(self.sql(f'SELECT count(*) FROM {es.RECEIPT}'), [(2,)])

    def test_manual_settlement_wins_table_lock_runner_does_not_overwrite(self):
        lease, group, changes = self.prime()
        started = Event()
        def writer():
            started.set()
            return self.store.write(lease, self.ev, group, changes, self.now)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.engine.begin() as c:
                c.execute(text(f'LOCK TABLE {es.BETS} IN SHARE ROW EXCLUSIVE MODE'))
                job = pool.submit(writer)
                self.assertTrue(started.wait(2))
                c.execute(text(f'''UPDATE {es.BETS} SET "Status"='Settled', "Home_Goal"='22' '''))
            with self.assertRaisesRegex(es.Review, 'non_open_or_ambiguous_group'):
                job.result(5)
        self.assertEqual(self.sql(f'SELECT DISTINCT "Home_Goal" FROM {es.BETS}'), [('22',)])
        self.assertEqual(self.sql(f'SELECT count(*) FROM {es.RECEIPT}'), [(0,)])

    def test_database_lock_timeout_leaves_no_partial_change_or_receipt(self):
        lease, group, changes = self.prime()
        with self.engine.begin() as c:
            c.execute(text(f'LOCK TABLE {es.BETS} IN SHARE ROW EXCLUSIVE MODE'))
            with self.assertRaises(DBAPIError) as caught:
                self.store.write(lease, self.ev, group, changes, self.now)
            self.assertEqual(caught.exception.orig.pgcode, '55P03')
        self.assertEqual(self.current_status(), ['Open', 'Open'])
        self.assertEqual(self.sql(f'SELECT count(*) FROM {es.RECEIPT}'), [(0,)])

    def test_tombstones_duplicate_elsewhere_and_new_slip_invalidate_write(self):
        lease, group, changes = self.prime()
        self.sql(f"INSERT INTO {es.TOMBS} (id) VALUES ('B')")
        with self.assertRaisesRegex(es.Review, 'tombstoned_slip'):
            self.store.write(lease, self.ev, group, changes, self.now)
        self.sql(f'DELETE FROM {es.TOMBS}')
        self.insert(slip('B', ev=replace(self.ev, mid='777000'), Home_Team='OTHER', Match='OTHER vs XX',
                         Match_Date='2026-01-01'))
        with self.assertRaisesRegex(es.Review, 'duplicate_or_invalid_slip_id'):
            self.store.write(lease, self.ev, group, changes, self.now)
        self.sql(f"""DELETE FROM {es.BETS} WHERE "Tipsme_Match_ID"='777000'""")
        self.insert(slip('C', ev=self.ev))
        with self.assertRaisesRegex(es.Review, 'group_changed_since_read'):
            self.store.write(lease, self.ev, group, changes, self.now)
        self.assertEqual(self.current_status(), ['Open', 'Open', 'Open'])

    def test_transaction_readback_mismatch_rolls_back_entire_group(self):
        lease, group, changes = self.prime()
        self.sql(f'''CREATE FUNCTION public.fb_16413_test_trigger() RETURNS trigger LANGUAGE plpgsql AS $$
                    BEGIN IF NEW."ID" = 'B' THEN NEW."System_Stake" := 177; END IF; RETURN NEW; END $$''')
        self.sql(f'''CREATE TRIGGER alter_after BEFORE UPDATE ON {es.BETS}
                    FOR EACH ROW EXECUTE FUNCTION public.fb_16413_test_trigger()''')
        try:
            with self.assertRaisesRegex(es.Review, 'transaction_readback'):
                self.store.write(lease, self.ev, group, changes, self.now)
            self.assertEqual(self.current_status(), ['Open', 'Open'])
            self.assertEqual(self.sql(f'SELECT count(*) FROM {es.RECEIPT}'), [(0,)])
        finally:
            self.sql(f'DROP TRIGGER alter_after ON {es.BETS}')
            self.sql('DROP FUNCTION public.fb_16413_test_trigger()')

    def test_post_commit_disconnect_resolves_via_receipts_and_never_replays(self):
        lease, group, changes = self.prime()
        original = es._in
        def disconnect(ids, sql, conn):
            if sql.startswith('SELECT slip_id, fingerprint, expected'):
                raise TimeoutError('simulated disconnect after COMMIT')
            return original(ids, sql, conn)
        with patch.object(es, '_in', disconnect), self.assertRaises(TimeoutError):
            self.store.write(lease, self.ev, group, changes, self.now)
        self.assertEqual(self.store.review_day(self.ev.kickoff.date(), [fixture(self.ev)]), (set(), True))
        self.sql(f"""UPDATE {es.BETS} SET "Status"='Open'""")
        group = self.store.group(self.ev)
        changes = plan(group, self.ev, odds(self.ev))
        with self.assertRaisesRegex(es.Review, 'match_already_receipted'):
            self.store.write(lease, self.ev, group, changes, self.now)
        self.assertEqual(self.sql(f'SELECT count(*) FROM {es.RECEIPT}'), [(2,)])

    def test_audit_reports_cash_or_odds_history_drift_even_when_scores_match(self):
        lease, group, changes = self.prime()
        self.store.write(lease, self.ev, group, changes, self.now)
        self.sql(f"""UPDATE {es.BETS} SET "User_Profit"='17' WHERE "ID"='A' """)
        self.assertEqual(self.store.review_day(self.ev.kickoff.date(), [fixture(self.ev)]), (set(), False))

    def test_end_to_end_store_observe_then_write_with_fake_provider(self):
        self.insert(slip(ev=self.ev, pre=True))
        upstream = Mock()
        upstream.final.return_value = self.ev
        upstream.odds.return_value = odds(self.ev)
        for index in range(13):
            result = es.early_scan(self.store, upstream, 'observe', plan)
            self.assertEqual(result[self.ev.mid][0], 'observing' if index < 12 else 'would_settle')
            self.now += es.POLL
        self.assertEqual(self.current_status(), ['Open'])
        self.assertEqual(self.sql(f'SELECT count(*) FROM {es.RECEIPT}'), [(0,)])
        self.assertEqual(es.early_scan(self.store, upstream, 'write', plan)[self.ev.mid], ('committed', {'A'}))
        self.assertEqual(self.store.review_day(self.ev.kickoff.date(), [fixture(self.ev)]), (set(), True))

    def test_stale_generic_settled_overwrite_is_blocked_but_fresh_manual_edit_works(self):
        lease, group, changes = self.prime()
        with self.engine.connect() as c:
            import pandas as pd
            stale = pd.read_sql_table('football_bets', c, schema='public')
        state = {'_v1641_fallback_base': False}
        st = SimpleNamespace(session_state=state, error=Mock())
        ns = app_defs('_v1512_fp', 'V1512_FP_COLS', '_V16413_RESULT_FP', 'DB_COLUMNS',
                      '_v1512_save_sql_main', 'process_legacy_columns', 'enforce_columns',
                      V164_TOMB_TABLE='football_bets_deleted', _V10_SQL_TABLE='public.football_settings',
                      get_hkt_now=lambda: self.now, _v163_sql_engine=lambda: self.engine,
                      is_missing_match_date=lambda x: x is None, st=st)
        fp, save = ns['_v1512_fp'], ns['_v1512_save_sql_main']
        state['_v1641_sql_base'] = fp(stale)
        stale['Status'], stale['Home_Goal'] = 'Settled', '99'
        self.store.write(lease, self.ev, group, changes, self.now)
        self.assertFalse(save(stale, 'football_bets'))
        self.assertEqual(self.sql(f'SELECT DISTINCT "Home_Goal" FROM {es.BETS}'), [('2',)])
        with self.engine.connect() as c:
            fresh = pd.read_sql_table('football_bets', c, schema='public')
        state['_v1641_sql_base'] = fp(fresh)
        fresh.loc[fresh['ID'].eq('A'), 'Note'] = 'deliberate correction'
        self.assertTrue(save(fresh, 'football_bets'), st.error.call_args_list)
        self.assertEqual(self.sql(f'''SELECT "Note" FROM {es.BETS} WHERE "ID"='A' '''), [('deliberate correction',)])
