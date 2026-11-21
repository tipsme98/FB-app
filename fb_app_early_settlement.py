"""V17: conservative settlement of existing slips only, never creates a bet."""

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone

import requests

try:
    from sqlalchemy import bindparam, text
except ImportError:
    # Legacy CSV-only use can still start; observe/write require the existing SQL dependencies.
    bindparam = text = None


HKT = timezone(timedelta(hours=8))
UTC = timezone.utc
POLL = timedelta(minutes=10)
GAP = timedelta(minutes=30)
HOLD = timedelta(hours=2)
FRESH = timedelta(seconds=60)
LOOKBACK = timedelta(days=14)
MAX_MATCHES = 8
API = "https://tipsme-web.azurewebsites.net/api/Score"
PAGE = "https://tipsme.hk/zh-HK/match/{}/live"
STATE = 'public.fb_early_state_16413'
RECEIPT = 'public.fb_early_receipt_16413'
LEASE = 'public.fb_early_lease_16413'
BETS = 'public."football_bets"'
TOMBS = 'public."football_bets_deleted"'
STRICT_ID = r'^[+]?0*[0-9]{1,12}([.]0+)?$'
IDENTITY = ('ID', 'Status', 'Tipsme_Match_ID', 'Match_Date', 'Home_Team', 'Away_Team', 'Match')
INPUTS = ('Bet_Type', 'Selection', 'Initial_Line', 'Initial_Odds', 'System_Stake', 'User_Stake')
RESULT = ('Home_Goal', 'Away_Goal', 'Home_Corner', 'Away_Corner', 'Home_Red', 'Away_Red',
          'Home_Yellow', 'Away_Yellow', 'Half_Score', 'System_Profit', 'User_Profit',
          'System_Payout', 'User_Payout', 'Unit_Profit', 'Result_Label')
COLUMNS = IDENTITY + INPUTS + RESULT + ('Odds_History',)
MARKETS = {'讓球': ('hdpOdds', 'home_cap', 'home_win_odds', 'away_win_odds', ('Home', 'Away')),
           '入球大小': ('hiloOdds', 'number_of_goals', 'big_odds', 'small_odds', ('Over', 'Under')),
           '角球大小': ('chiloOdds', 'number_of_corners', 'big_odds', 'small_odds', ('Over', 'Under'))}
ABNORMAL = re.compile(r'腰斬|取消|改期|延期|押後|暫停|暂停|中斷|中断|中止|加時|加时|十二碼|点球大战|'
                      r'abandon|cancel|postpon|suspend|interrupt|extra[ -]time|shoot[ -]?out', re.I)
NEXT = re.compile(r'<script\b[^>]*\bid=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', re.S)


class Review(ValueError):
    """A controlled fail-closed outcome; code is safe for logs and UI."""

    def __init__(self, code, *, manual=False):
        super().__init__(code)
        self.code = code
        self.manual = manual


def mode(value):
    value = str(value or 'off').strip().lower()
    return value if value in ('off', 'observe', 'write') else 'invalid'


def identifier(value):
    s = str(value or '').strip()
    if not re.fullmatch(STRICT_ID, s):
        return None
    return s.split('.')[0].lstrip('+').lstrip('0') or None


def explicit_date(value):
    try:
        return date.fromisoformat(str(value)) if re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(value)) else None
    except ValueError:
        return None


def digest(value):
    return hashlib.sha256(json.dumps(value, default=str, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def pair(value):
    m = re.fullmatch(r'(\d{1,2})\s*[-:]\s*(\d{1,2})', str(value or '').strip())
    if not m:
        raise Review('missing_or_invalid_score')
    return tuple(map(int, m.groups()))


def provider_time(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if 'T' not in str(value):
            raise ValueError
        return (stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)).astimezone(HKT)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Review('provider_time') from exc


@dataclass(frozen=True)
class Evidence:
    mid: str
    home: str
    away: str
    kickoff: datetime
    goals: tuple
    half: tuple
    corners: tuple
    cards: tuple

    @property
    def key(self):
        return digest(asdict(self))


def _score_record(record, mid, norm):
    if any(ABNORMAL.search(str(record.get(k) or '')) for k in
           ('liveTime', 'matchStatus', 'additionalTime')) or record.get('extraScore') not in ('',):
        # Missing extraScore is unknown, not positive evidence of extra time.
        raise Review('extra_time_or_abnormal_status',
                     manual=bool(record.get('extraScore')) or any(
                         ABNORMAL.search(str(record.get(k) or '')) and
                         not re.search(r'延期|押後|暫停|暂停|中斷|中断|改期|postpon|suspend|interrupt',
                                       str(record.get(k) or ''), re.I)
                         for k in ('liveTime', 'matchStatus', 'additionalTime')))
    if str(record.get('matchStatus')) != '8' or record.get('liveTime') not in ('完', 'FT'):
        raise Review('not_normal_final')
    home, away = norm(record.get('home')), norm(record.get('away'))
    if identifier(record.get('matchId')) != mid or not home or not away or home == away:
        raise Review('provider_identity')
    kickoff = provider_time(record.get('fullMatchTimeUtc'))
    try:
        local = datetime.strptime(record['fullMatchTime'], '%m/%d/%Y %H:%M:%S').replace(tzinfo=HKT)
    except (KeyError, ValueError, TypeError) as exc:
        raise Review('local_kickoff') from exc
    if kickoff != local:
        raise Review('kickoff_disagreement')
    goals, half, corners = (pair(record.get(x)) for x in ('score', 'halfScore', 'corner'))
    if any(h > f for h, f in zip(half, goals)):
        raise Review('impossible_half_score')
    cards = tuple(str(record.get(k)) for k in ('homeRedCard', 'awayRedCard', 'homeYellowCard', 'awayYellowCard'))
    if any(not re.fullmatch(r'\d{1,2}', v) for v in cards):
        raise Review('missing_cards')
    return Evidence(mid, home, away, kickoff, goals, half, corners, tuple(map(int, cards)))


def parse_final(mid, api, html, norm):
    """Two Tipsme surfaces, plus regulation event and statistics evidence on /live."""
    try:
        pg = json.loads(NEXT.search(html)[1])['props']['pageProps']
        a = _score_record(api, mid, norm)
        b = _score_record(pg['liveScore'], mid, norm)
        if a != b or identifier(pg['matchId']) != mid:
            raise Review('source_disagreement')
        fixtures = pg['initialFixtures']['fixtures']
        events = fixtures['mainDetails']
        if not isinstance(events, list) or not events:
            raise Review('missing_event_timeline')
        halves, finals, late = [], [], False
        for event in events:
            minute = str(event.get('time') or '').strip()
            kinds = ' '.join(str(event.get(k) or '') for k in ('homeEventType', 'awayEventType'))
            if ABNORMAL.search(minute + ' ' + kinds):
                raise Review('abnormal_event_timeline')
            if minute in ('中', '完'):
                shown = [pair(event[k]) for k in ('awayEventDetails', 'homeEventDetails') if event.get(k)]
                if len(shown) != 1:
                    raise Review('ambiguous_event_marker')
                (halves if minute == '中' else finals).append(shown[0])
            else:
                m = re.fullmatch(r'(\d{1,3})(?:\+\d{1,2})?', minute)
                if not m or int(m[1]) > 90:
                    raise Review('non_regulation_event_minute')
                late |= int(m[1]) == 90
        tech = [pair(f"{r.get('home')}-{r.get('away')}") for r in fixtures['techDetails']
                if r.get('time') in ('角球', 'Corners')]
        if (halves, finals, tech) != ([a.half], [a.goals], [a.corners]) or not late or events[-1].get('time') != '完':
            raise Review('incomplete_or_conflicting_timeline')
        return a
    except (KeyError, AttributeError, TypeError, IndexError, json.JSONDecodeError) as exc:
        raise Review('provider_payload') from exc


class Tipsme:
    def __init__(self, norm):
        self.norm = norm

    @staticmethod
    def _get(url):
        headers = {'Accept-Language': 'zh-HK', 'Cache-Control': 'no-cache', 'Pragma': 'no-cache'}
        if url.startswith('https://tipsme.hk/'):
            try:
                from curl_cffi import requests as page_client
                from curl_cffi.requests.exceptions import RequestException as PageError
            except ImportError as exc:
                raise Review('event_page_client_unavailable') from exc
            try:
                response = page_client.get(url, headers=headers, timeout=(3, 5), allow_redirects=False,
                                           impersonate='chrome')
            except PageError as exc:
                raise Review('event_page_unavailable') from exc
        else:
            response = requests.get(url, headers=headers, timeout=(3, 5), allow_redirects=False)
        if response.status_code != 200:
            raise Review('provider_unavailable')
        try:
            age = float(response.headers.get('Age', 0))
        except ValueError as exc:
            raise Review('provider_cache_age') from exc
        if not 0 <= age <= 60:
            raise Review('provider_cached')
        return response

    def final(self, mid):
        try:
            api = self._get(f'{API}/match/{mid}').json()
            # Do not spend a page call on clearly non-final fixtures.
            _score_record(api, mid, self.norm)
            return parse_final(mid, api, self._get(PAGE.format(mid)).text, self.norm)
        except (requests.RequestException, AttributeError) as exc:
            raise Review('provider_unavailable') from exc

    def odds(self, mid):
        try:
            return self._get(f'{API}/odds/hkjc/{mid}').json()
        except requests.RequestException as exc:
            raise Review('odds_unavailable') from exc


def relevant(rows, mid, day, home, away, norm):
    for row in rows:
        rid, dt = identifier(row['Tipsme_Match_ID']), explicit_date(row['Match_Date'])
        teams = (norm(row['Home_Team']), norm(row['Away_Team']))
        label = re.split(r'\s+vs\s+', str(row['Match'] or '').strip(), flags=re.I)
        named = tuple(map(norm, label)) == (home, away)
        same_pair = teams == (home, away) or named
        swapped = teams == (away, home) or tuple(map(norm, label)) == (away, home)
        if rid == mid or (dt == day and (same_pair or swapped)) or (dt is None and (same_pair or swapped) and rid is None):
            yield row, (rid == mid and dt == day and teams == (home, away)
                        and (len(label) == 1 or named))


def exact_open_group(rows, ev, norm):
    matches = list(relevant(rows, ev.mid, ev.kickoff.date(), ev.home, ev.away, norm))
    if not matches or any(not exact or r['Status'] != 'Open' for r, exact in matches):
        raise Review('non_open_or_ambiguous_group')
    group = sorted((r for r, _ in matches), key=lambda r: str(r['ID']))
    ids = [str(r['ID'] or '') for r in group]
    if any(not rid or rid.startswith('TOTAL') for rid in ids) or len(set(ids)) != len(ids):
        raise Review('duplicate_or_invalid_slip_id')
    return group


def number(value):
    try:
        n = float(value)
        if not math.isfinite(n):
            raise ValueError
        return n
    except (TypeError, ValueError, OverflowError) as exc:
        raise Review('invalid_slip_numbers') from exc


def _history_stamp(value, cutoff):
    v = str(value or '')
    if re.fullmatch(r'\d{4}-\d\d-\d\d \d\d:\d\d', v):
        try:
            t = datetime.strptime(v, '%Y-%m-%d %H:%M').replace(tzinfo=HKT)
        except ValueError as exc:
            raise Review('unverified_existing_clv_timestamp') from exc
        if cutoff - LOOKBACK <= t < cutoff:
            return t
    if re.fullmatch(r'\d\d-\d\d \d\d:\d\d', v):
        for year in (cutoff.year, cutoff.year - 1):
            try:
                t = datetime.strptime(f'{year}-{v}', '%Y-%m-%d %H:%M').replace(tzinfo=HKT)
            except ValueError:
                continue
            if cutoff - LOOKBACK <= t < cutoff:
                return t
    raise Review('unverified_existing_clv_timestamp')


def _closing(raw, ev, market, line, norm, parse_line, odds_entry):
    if not isinstance(raw, dict) or (norm(raw.get('hkjcHomeTeamName')), norm(raw.get('hkjcAwayTeamName'))) != (ev.home, ev.away):
        raise Review('odds_identity')
    key, line_key, upp, low, _ = MARKETS[market]
    recs = raw.get(key)
    if not isinstance(recs, list):
        raise Review('missing_closing_market')
    cutoff = ev.kickoff.replace(second=0, microsecond=0)
    quotes = {}
    for r in recs:
        if not isinstance(r, dict) or r.get(line_key) is None:
            raise Review('invalid_closing_market')
        if parse_line(r[line_key]) != line:
            continue
        t = provider_time(r.get('create_time'))
        if not cutoff - LOOKBACK <= t < cutoff:
            continue
        up, lo = number(r.get(upp)), number(r.get(low))
        if min(up, lo) <= 1:
            raise Review('invalid_closing_quote')
        minute = t.replace(second=0, microsecond=0)
        if minute in quotes and quotes[minute] != (up, lo):
            raise Review('ambiguous_closing_minute')
        quotes[minute] = (up, lo)
    if not quotes:
        raise Review('missing_pre_kickoff_price')
    return cutoff, quotes, [odds_entry(i, market, t.strftime('%Y-%m-%d %H:%M'), line, *v)
                            for i, (t, v) in enumerate(sorted(quotes.items()))]


def build_plan(group, ev, raw_odds, *, norm, calculate, parse_line, odds_entry, merge_history):
    """Calculate with existing payout math; refuse ambiguous CLV rather than rewrite history."""
    updates = {}
    for r in group:
        bt = str(r['Bet_Type'] or '').strip()
        live = bt.endswith(' (即場)')
        market = bt.removesuffix(' (即場)')
        if market not in MARKETS or r['Selection'] not in MARKETS[market][-1]:
            raise Review('unsupported_slip')
        line, price, system, user = (number(r[k]) for k in INPUTS[2:])
        if not -50 <= line <= 50 or line * 4 != round(line * 4) or price <= 1 or min(system, user) < 0:
            raise Review('invalid_slip_numbers')
        sp, up, sa, ua, unit, label, _ = calculate(bt, r['Selection'], line, price, system, user,
                                                   *ev.goals, *ev.corners)
        if not all(math.isfinite(v) for v in (sp, up, sa, ua, unit)):
            raise Review('invalid_calculated_amount')
        values = ('Settled', *ev.goals, *ev.corners, *ev.cards, f'{ev.half[0]}-{ev.half[1]}',
                  sp, up, sa, ua, unit, label)
        changes = dict(zip(('Status',) + RESULT, values))
        if not live:
            cutoff, quotes, snapshots = _closing(raw_odds, ev, market, line, norm, parse_line, odds_entry)
            try:
                history = json.loads(r['Odds_History'] or '[]')
            except (ValueError, TypeError) as exc:
                raise Review('invalid_odds_history') from exc
            if not isinstance(history, list) or any(not isinstance(s, dict) for s in history):
                raise Review('invalid_odds_history')
            known = {}
            for snap in history:
                if str(snap.get('type') or '').strip() != market:
                    continue
                if number(snap.get('line')) != line:
                    continue
                t = _history_stamp(snap.get('record_time'), cutoff)
                existing = number(snap.get('upper')), number(snap.get('lower'))
                if t > max(quotes) or min(existing) <= 1 or (t in quotes and existing != quotes[t]):
                    raise Review('conflicting_existing_clv_price')
                if t in known and existing != known[t]:
                    raise Review('conflicting_existing_clv_price')
                known[t] = existing
            snapshots = [s for s in snapshots if _history_stamp(s['record_time'], cutoff) not in known]
            merged, added = merge_history(r['Odds_History'], snapshots)
            if added:
                changes['Odds_History'] = merged
        updates[str(r['ID'])] = changes
    return updates


def _in(values, sql, conn):
    return conn.execute(text(sql).bindparams(bindparam('ids', expanding=True)), {'ids': sorted(values)})


def _fields(cols):
    return ', '.join(f'"{c}"' for c in cols)


def _short(conn):
    conn.execute(text("SET LOCAL lock_timeout = '3s'"))
    conn.execute(text("SET LOCAL statement_timeout = '8s'"))


def _now(conn):
    return conn.scalar(text('SELECT clock_timestamp()'))


def _rows(conn, ev):
    pattern = rf'^[+]?0*{ev.mid}([.]0+)?$'
    # Alias checks need only identity columns; don't load large odds histories for unrelated legacy rows.
    core = ', '.join(f'CASE WHEN match_id_ok THEN "{c}" END AS "{c}"' for c in COLUMNS[len(IDENTITY):])
    return list(conn.execute(text(f'''
        SELECT {_fields(IDENTITY)}, {core} FROM (
            SELECT *, btrim(CAST("Tipsme_Match_ID" AS text)) ~ :pattern AS match_id_ok FROM {BETS}
        ) AS slips
        WHERE match_id_ok
           OR CAST("Match_Date" AS text) = :day
           OR ("Match_Date" IS NULL OR CAST("Match_Date" AS text) !~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$')
        '''), {'pattern': pattern, 'day': ev.kickoff.date().isoformat()}).mappings())


def _locked_state(conn, mid):
    return conn.execute(text(f'SELECT * FROM {STATE} WHERE match_id = :mid FOR UPDATE'),
                        {'mid': mid}).mappings().one_or_none()


def _fence(conn, lease):
    latest = conn.scalar(text(f'SELECT last_scan FROM {LEASE} WHERE singleton FOR SHARE'))
    if latest != lease:
        raise Review('scan_superseded')


class Postgres:
    def __init__(self, engine, norm):
        self.engine = engine
        self.norm = norm

    def claim(self, run_mode):
        """One shared start per ten minutes across all Streamlit processes."""
        with self.engine.begin() as conn:
            _short(conn)
            # Even empty scans must not look healthy with an unapplied/partial migration.
            conn.execute(text(f'SELECT slip_id FROM {RECEIPT} LIMIT 0'))
            if run_mode == 'write':
                conn.execute(text(f'SELECT id FROM {TOMBS} LIMIT 0'))
            now = _now(conn)
            last = conn.execute(text(f'''SELECT last_scan FROM {LEASE} WHERE singleton FOR UPDATE'''
                                     )).scalar_one()
            if last is not None and now - last < POLL:
                return None, []
            today = now.astimezone(HKT).date()
            hits = conn.execute(text(f'''
                WITH candidates AS (
                    SELECT DISTINCT ltrim(regexp_replace(split_part(btrim(CAST("Tipsme_Match_ID" AS text)),
                        '.', 1), '^[+]', ''), '0') AS mid
                    FROM {BETS} WHERE "Status" = 'Open'
                    AND CAST("Match_Date" AS text) BETWEEN :lo AND :hi
                    AND btrim(CAST("Tipsme_Match_ID" AS text)) ~ :pattern
                )
                SELECT c.mid FROM candidates c LEFT JOIN {STATE} s ON s.match_id = c.mid
                WHERE c.mid <> '' AND NOT coalesce(s.manual, false)
                ORDER BY s.last_attempt NULLS FIRST, c.mid LIMIT :limit
                '''), {'lo': (today - LOOKBACK).isoformat(), 'hi': today.isoformat(),
                       'pattern': STRICT_ID, 'limit': MAX_MATCHES}).scalars().all()
            conn.execute(text(f'UPDATE {LEASE} SET last_scan = :now WHERE singleton'), {'now': now})
            return now, hits

    def observe(self, lease, began, mid, ev, code, manual=False):
        with self.engine.begin() as conn:
            _short(conn)
            _fence(conn, lease)
            conn.execute(text(f'''INSERT INTO {STATE} (match_id, reason) VALUES (:mid, 'new')
                                  ON CONFLICT (match_id) DO NOTHING'''), {'mid': mid})
            old = _locked_state(conn, mid)
            now = _now(conn)
            if old['manual']:
                return False
            if not timedelta(0) <= now - began <= FRESH:
                ev, code, manual = None, 'stale_observation', False
            if ev is not None and not timedelta(minutes=90) <= now - ev.kickoff <= LOOKBACK + timedelta(days=1):
                ev, code = None, 'kickoff_out_of_range'
            first = None
            if ev is not None:
                continuous = old['fingerprint'] == ev.key and old['last_good'] is not None and (
                    timedelta(0) <= now - old['last_good'] <= GAP)
                first = old['first_good'] if continuous else now
            if manual:
                ev, first = None, None
            conn.execute(text(f'''UPDATE {STATE} SET first_good=:first, last_good=:last,
                          last_attempt=:now, fingerprint=:fp, reason=:code, manual=:manual WHERE match_id=:mid
                          '''), dict(mid=mid, first=first, last=now if ev else None, now=now,
                                     fp=ev.key if ev else None, code=code, manual=manual))
            return first is not None and now - first >= HOLD

    def group(self, ev):
        with self.engine.begin() as conn:
            _short(conn)
            return exact_open_group(_rows(conn, ev), ev, self.norm)

    def record_reason(self, lease, mid, code):
        with self.engine.begin() as conn:
            _short(conn)
            _fence(conn, lease)
            conn.execute(text(f'UPDATE {STATE} SET reason=:code WHERE match_id=:mid AND NOT manual'),
                         {'mid': mid, 'code': code})

    def write(self, lease, ev, group, plan, began):
        with self.engine.begin() as conn:
            _short(conn)
            _fence(conn, lease)
            conn.execute(text(f'LOCK TABLE {BETS} IN SHARE ROW EXCLUSIVE MODE'))
            state = _locked_state(conn, ev.mid)
            now = _now(conn)
            if (not state or state['manual'] or state['fingerprint'] != ev.key or
                    not timedelta(0) <= now - state['last_good'] <= GAP or
                    now - state['first_good'] < HOLD or not timedelta(0) <= now - began <= FRESH):
                raise Review('expired_write_evidence')
            current = exact_open_group(_rows(conn, ev), ev, self.norm)
            if digest([dict(r) for r in current]) != digest([dict(r) for r in group]):
                raise Review('group_changed_since_read')
            ids = set(plan)
            if ids != {str(r['ID']) for r in current}:
                raise Review('invalid_write_plan')
            if _in(ids, f'SELECT id FROM {TOMBS} WHERE id IN :ids', conn).first():
                raise Review('tombstoned_slip')
            if _in(ids, f'SELECT 1 FROM {BETS} WHERE "ID" IN :ids GROUP BY "ID" HAVING count(*) <> 1', conn).first():
                raise Review('duplicate_or_invalid_slip_id')
            if conn.scalar(text(f'SELECT EXISTS (SELECT 1 FROM {RECEIPT} WHERE match_id=:mid)'),
                           {'mid': ev.mid}):
                raise Review('match_already_receipted')
            expected = {}
            original = {str(r['ID']): r for r in current}
            for rid, changes in plan.items():
                if (not {'Status', *RESULT} <= changes.keys() <= {'Status', *RESULT, 'Odds_History'}
                        or changes['Status'] != 'Settled'):
                    raise Review('invalid_write_columns')
                before = original[rid]
                expected[rid] = {
                    'Match_Date': ev.kickoff.date().isoformat(),
                    'Tipsme_Match_ID': ev.mid,
                    **{k: before[k] for k in ('Home_Team', 'Away_Team', 'Bet_Type', 'Selection', 'Odds_History')},
                    **{k: number(before[k]) for k in INPUTS[2:]},
                    **changes,
                }
                assignments = ', '.join(f'"{k}"=:{k}' for k in changes)
                res = conn.execute(text(f'''UPDATE {BETS} SET {assignments}
                                             WHERE "ID"=:rid AND "Status"='Open' '''),
                                   {'rid': rid, **{k: str(v) for k, v in changes.items()}})
                if res.rowcount != 1:
                    raise Review('update_count')
            saved = list(_in(ids, f'SELECT {_fields(COLUMNS)} FROM {BETS} WHERE "ID" IN :ids', conn).mappings())
            if (len(saved) != len(ids) or {str(r['ID']) for r in saved} != ids or
                    any(not equal_changes(r, expected[str(r['ID'])]) for r in saved)):
                raise Review('transaction_readback')
            for rid, fields in expected.items():
                conn.execute(text(f'''INSERT INTO {RECEIPT} (slip_id, match_id, match_day, fingerprint,
                                    committed_at, expected) VALUES (:rid,:mid,:day,:fp,:now,CAST(:expected AS jsonb))'''),
                             dict(rid=rid, mid=ev.mid, day=ev.kickoff.date(), fp=ev.key, now=now,
                                  expected=json.dumps(fields, ensure_ascii=False)))
            conn.execute(text(f"UPDATE {STATE} SET reason='committed' WHERE match_id=:mid"), {'mid': ev.mid})
        # A different transaction confirms the commit. If uncertain, never replay a receipted match.
        with self.engine.begin() as conn:
            _short(conn)
            ledger = list(_in(ids, f'SELECT slip_id, fingerprint, expected FROM {RECEIPT} WHERE slip_id IN :ids',
                              conn).mappings())
            got = list(_in(ids, f'SELECT {_fields(COLUMNS)} FROM {BETS} WHERE "ID" IN :ids', conn).mappings())
            if (len(ledger) != len(ids) or len(got) != len(ids) or {str(r['ID']) for r in got} != ids
                    or any(r['fingerprint'] != ev.key or r['expected'] != expected[r['slip_id']] for r in ledger)
                    or any(not equal_changes(r, expected[str(r['ID'])]) for r in got)):
                raise Review('commit_readback_unconfirmed')
            return ids

    def observed_until(self):
        with self.engine.begin() as conn:
            _short(conn)
            return _now(conn)

    def review_day(self, day, fixtures):
        with self.engine.connect().execution_options(isolation_level='REPEATABLE READ') as conn, conn.begin():
            _short(conn)
            conn.execute(text('SET TRANSACTION READ ONLY'))
            rows = conn.execute(text(f'''SELECT {_fields(IDENTITY + RESULT[:4])}
                     FROM {BETS} b WHERE NOT EXISTS (SELECT 1 FROM {TOMBS} t WHERE t.id=b."ID")'''
                                    )).mappings().all()
            pending = reconciliation_pending(rows, fixtures, day, self.norm)
            audits = conn.execute(text(f'''SELECT a.slip_id, a.expected, to_jsonb(b) AS actual,
                       EXISTS (SELECT 1 FROM {TOMBS} t WHERE t.id=a.slip_id) AS deleted
                       FROM {RECEIPT} a LEFT JOIN {BETS} b ON b."ID"=a.slip_id
                       WHERE a.match_day=:day'''), {'day': day}).mappings().all()
        return pending, (len({r['slip_id'] for r in audits}) == len(audits) and
                         all(r['actual'] and not r['deleted'] and equal_changes(r['actual'], r['expected'])
                             for r in audits))


def equal_changes(row, changes):
    for k, exp in changes.items():
        val = row.get(k)
        if k == 'Match_Date':
            if str(val) != exp:
                return False
        elif k == 'Tipsme_Match_ID':
            if identifier(val) != exp:
                return False
        elif isinstance(exp, (int, float)):
            try:
                if number(val) != exp:
                    return False
            except Review:
                return False
        elif val != exp:
            return False
    return True


def protected_current(conn, ids):
    """Generic writer's freshness check; stays on after mode is switched back to off."""
    if not ids or not conn.scalar(text('SELECT to_regclass(:name)'), {'name': RECEIPT}):
        return []
    return list(_in(ids, f'''SELECT b.* FROM {BETS} b JOIN {RECEIPT} r ON r.slip_id=b."ID"
                             WHERE b."ID" IN :ids''', conn).mappings())


def early_scan(store, provider, run_mode, planner):
    if run_mode not in ('observe', 'write'):
        return {}

    def one(mid):
        began = store.observed_until()
        try:
            ev = provider.final(mid)
        except Review as exc:
            store.observe(lease, began, mid, None, exc.code, exc.manual)
            return (exc.code, set())
        if not store.observe(lease, began, mid, ev, 'normal_final'):
            return ('observing', set())
        try:
            group = store.group(ev)
            pre = any(not str(r['Bet_Type']).strip().endswith(' (即場)') for r in group)
            plan = planner(group, ev, provider.odds(mid) if pre else None)
            if run_mode == 'observe':
                store.record_reason(lease, mid, 'would_settle')
                return ('would_settle', set())
            began = store.observed_until()
            try:
                fresh = provider.final(mid)
            except Review as exc:
                store.observe(lease, began, mid, None, exc.code, exc.manual)
                return (exc.code, set())
            if fresh != ev:
                store.observe(lease, began, mid, fresh, 'final_corrected')
                return ('final_corrected', set())
            return ('committed', store.write(lease, ev, group, plan, began))
        except Review as exc:
            store.record_reason(lease, mid, exc.code)
            return (exc.code, set())

    def guarded(mid):
        try:
            return one(mid)
        except Exception:
            # No fallback write. A failed/unknown commit is resolved by official rows + receipts.
            return ('sql_or_runtime_unconfirmed', set())

    lease, matches = store.claim(run_mode)
    if not matches:
        return {}
    with ThreadPoolExecutor(max_workers=4) as workers:
        return dict(zip(matches, workers.map(guarded, matches)))


def reconciliation_pending(rows, fixtures, day, norm):
    """05:00 write mode: compare official rows to feed; never insert or settle here."""
    pending = set()
    day_rows = [r for r in rows if explicit_date(r['Match_Date']) == day]
    for r in day_rows:
        if r['Status'] != 'Settled':
            pending.add(identifier(r['Tipsme_Match_ID']) or 'legacy-open')
    for f in fixtures:
        mid = identifier(f.get('hkjc_match_id'))
        home, away = norm(f.get('home')), norm(f.get('away'))
        group = list(relevant(rows, mid, day, home, away, norm))
        values = tuple(f.get(k) for k in ('home_goal', 'away_goal', 'home_corner', 'away_corner'))
        if not mid or not home or not away or not f.get('finished') or any(
                type(v) is not int or not 0 <= v <= 99 for v in values):
            pending.add(mid or 'unknown-fixture')
            continue
        counts = dict(zip(RESULT[:4], values))
        if (not group or any(not exact or r['Status'] != 'Settled' or not equal_changes(r, counts)
                             for r, exact in group) or
                len({r['ID'] for r, _ in group}) != len(group)):
            pending.add(mid or 'unknown-fixture')
    return pending
