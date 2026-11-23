"""Emit approved-batch SQL to stdout. This tool never connects to a database."""
from collections import Counter
from decimal import Decimal
from hashlib import sha256
from itertools import product
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import argparse
import csv
import io
import json

from fb_app_clv_cohorts import ENTRY
from fb_app_historical_clv import (BATCH, FIELDS, SOURCE_SHA, TABLE, TOMBS, compact, digest,
                                  evidence_identity, number, stamp)

FULL_SHA = "4eea377925d6619b26cf1e142f1152c33cc669e884f7832d3e3f37f884b80423"
KEYS = ("注單 ID", "帳本時間（香港）", "原帳本賽日", "主隊", "客隊", "Tipsme ID",
        "市場", "方向", "原線（讓球為主隊視角）", "原 Initial_Odds")
LINKS = {"entry_url": "寫入或帳本證據", "fill_url": "填充日誌",
         "fixture_url": "賽程", "closing_url": "馬會最後賠率證據",
         "hkjc_fixture_url": "馬會賽事佐證", "tipsme_odds_url": "Tipsme 馬會賠率來源",
         "other_fixture_url": "額外賽事配對佐證"}


def read_csv(path, expected):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("unexpected CSV SHA-256")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))


def original(row):
    """Recover the exact exported JSON via the prior SHA-256, not from parsed IDs."""
    values = [[row[k]] for k in KEYS]
    values[5] = list(dict.fromkeys([None, "", row[KEYS[5]]]))
    for i in (8, 9):
        n = number(row[KEYS[i]])
        values[i].append(float(n))
        if n == n.to_integral_value():
            values[i].append(int(n))
    results = {typed for v in product(*values) if digest(typed := compact(v)) == row["原列核對指紋 SHA-256"]}
    if len(results) != 1:
        raise ValueError(f"unresolved original fingerprint on worksheet row {row['Sheet1 列']}")
    return results.pop()


def batch(formal, full):
    a, b = read_csv(formal, SOURCE_SHA), read_csv(full, FULL_SHA)
    if len(a) != 708 or len(b) != 924 or a != [r for r in b if r["正式 CLV%"]]:
        raise ValueError("not the verified subset of the 924-row export")
    if len({r["注單 ID"] for r in a}) != len(a) or len({r["Sheet1 列"] for r in a}) != len(a):
        raise ValueError("duplicate source records")
    if Counter(r["注單狀態（匯出版）"] for r in a) != {"Settled": 681, "Open": 27}:
        raise ValueError("unexpected export status distribution")
    for r in a:
        bid, fp = r["注單 ID"], r["原列核對指紋 SHA-256"]
        raw = original(r)
        expected = evidence_identity({"bet_id": bid, "original_json": raw, "fingerprint": fp})
        closing = number(r["正式核證收盤價"])
        kickoff = stamp(r["排定開賽（香港）"])
        close_err = abs((expected[-1] / closing - 1) * 100 - number(r["正式 CLV%"]))
        method, _, _ = r["入注覆核"].partition("：")
        urls = {k: r[v] for k, v in LINKS.items() if r[v]}
        close_url = urlsplit(urls["closing_url"])
        last_odds = parse_qs(close_url.query)
        if (closing <= 1 or close_err > Decimal("0.0000005") or kickoff <= expected[1]
                or method not in ("A1", "A2") or r["獨立賽前入注價已核"] != "True"
                or r["狀態及原因"] != "入注與馬會賽前最後賠率已核"
                or not r["馬會賽前最後賠率狀態"].startswith("HKJC closedLines（賽前）")
                or close_url.scheme != "https" or close_url.hostname != "info.cld.hkjc.com"
                or last_odds.get("operationName") != ["lastOdds"]
                or not urls.get("entry_url") or not urls.get("fixture_url")):
            raise ValueError(f"invalid evidence on worksheet row {r['Sheet1 列']}")
        yield {"bid": bid, "fp": fp, "raw": raw, "close": str(closing), "provenance": {
            "worksheet_row": int(r["Sheet1 列"]), "kickoff_hkt": kickoff.isoformat(),
            "entry_method": method, "hkjc_number": r["馬會賽事編號"],
            "resolved_tipsme_id": r["Tipsme ID"], "closing_method": r["馬會賽前最後賠率狀態"],
            "tipsme_same_quote_hkt_not_closing_time": r["同價同線的 Tipsme 報價時間（非關盤時間）"] or None,
            "status_at_export": r["注單狀態（匯出版）"], **urls}}


def source_sql(rows, with_proof=False):
    # Factor repeated URLs across the batch; the database stores the exact original URLs.
    urls, payload = {}, []
    for r in rows:
        v = {k: r[k] for k in ("bid", "fp", "raw", "close")}
        v["sheet"] = r["provenance"]["worksheet_row"]
        if with_proof:
            proof = r["provenance"].copy()
            refs = {k: urls.setdefault(proof.pop(k), len(urls)) for k in LINKS if k in proof}
            v.update(proof=proof, refs=refs)
        payload.append(v)
    data = compact(payload)
    url_data = compact(list(urls))
    if "$fbclv$" in data or "$fblinks$" in url_data:
        raise ValueError("SQL literal delimiter in source")
    return f"""
src AS (
    SELECT p.bid, p.fp, p.raw, p.close, p.sheet, p.raw::jsonb orig,
        p.proof || COALESCE((SELECT jsonb_object_agg(k, ($fblinks${url_data}$fblinks$::jsonb)->(v::int))
                            FROM jsonb_each_text(p.refs) u(k,v)), '{{}}'::jsonb) AS proof
    FROM jsonb_to_recordset($fbclv${data}$fbclv$::jsonb)
         AS p(bid text, fp text, raw text, close numeric, sheet int, proof jsonb, refs jsonb)
)"""


def script(rows, *, with_table, approved=None):
    if approved and (not with_table or len(rows) != 708
                     or len(approved) != 32 or any(c not in "0123456789abcdef" for c in approved)):
        raise ValueError("apply requires the full batch, existing migration and valid preflight digest")
    prior = (f"SELECT bet_id, batch_id, source_sha256, fingerprint, original_json, closing_odds FROM public.{TABLE}"
             if with_table else "SELECT NULL::text, NULL::text, NULL::text, NULL::text, NULL::text, NULL::numeric WHERE FALSE")
    guarded = f"""
stable AS MATERIALIZED (
    SELECT b."ID", b."Date", b."Match_Date", b."Home_Team", b."Away_Team", b."Tipsme_Match_ID",
           b."Bet_Type", b."Selection", b."Initial_Line", b."Initial_Odds", b."Status", b."Note"
    FROM public.football_bets b JOIN src s ON s.bid = b."ID"{' FOR SHARE OF b' if approved else ''}
)"""
    return f"""WITH {source_sql(rows, bool(approved))},
{guarded},
prior(bet_id, batch_id, source_sha256, fingerprint, original_json, closing_odds) AS ({prior}),
checked AS (
    SELECT s.*, b."Status" status, b."Date" entry_date,
    CASE
      WHEN EXISTS (SELECT 1 FROM public.{TOMBS} t WHERE t.id = s.bid) THEN 'tombstoned'
      WHEN b.n IS NULL THEN 'missing'
      WHEN b.n <> 1 THEN 'duplicate'
      WHEN NOT COALESCE(b."Date" = s.orig->>1 AND b."Match_Date" = s.orig->>2
        AND b."Home_Team" = s.orig->>3 AND b."Away_Team" = s.orig->>4
        AND (NULLIF(regexp_replace(b."Tipsme_Match_ID", '\\.0+$', ''), '') IS NOT DISTINCT FROM (s.orig->>5))
        AND b."Bet_Type" = s.orig->>6 AND b."Selection" = s.orig->>7
        AND b."Initial_Line" = (s.orig->>8)::double precision
        AND b."Initial_Odds" = (s.orig->>9)::double precision, FALSE) THEN 'identity_mismatch'
      WHEN b."Status" IS NULL OR b."Status" NOT IN ('Open', 'Settled') THEN 'unsupported_status'
      WHEN left(btrim(COALESCE(b."Note", '')), {len(ENTRY)}) = '{ENTRY}' THEN 'native_entry_evidence'
      WHEN p.bet_id IS NULL THEN 'new'
      WHEN (p.batch_id, p.source_sha256, p.fingerprint, p.original_json, p.closing_odds)
           = ('{BATCH}', '{SOURCE_SHA}', s.fp, s.raw, s.close) THEN 'already_imported'
      ELSE 'existing_evidence_conflict' END AS decision
    FROM src s
    LEFT JOIN (SELECT *, count(*) OVER (PARTITION BY "ID") n FROM stable) b ON b."ID" = s.bid
    LEFT JOIN prior p ON p.bet_id = s.bid
),
cases AS (
    SELECT DISTINCT bid, fp, decision, status, entry_date, close, (orig->>9)::numeric entry FROM checked
),
plan AS (
    SELECT md5(string_agg(concat_ws('|', bid, fp, decision, COALESCE(status,''), COALESCE(entry_date,'')),
              E'\\n' ORDER BY bid, decision, status, entry_date)) AS plan_hash FROM cases
),
stats AS (
    SELECT decision, status, count(*) n, round(avg((entry / close - 1) * 100), 6) mean_clv_pct
    FROM cases GROUP BY decision, status
)
{f""", inserted AS (
    INSERT INTO public.{TABLE} (bet_id,batch_id,source_sha256,fingerprint,original_json,closing_odds,evidence)
    SELECT c.bid, '{BATCH}', '{SOURCE_SHA}', c.fp, c.raw, c.close, c.proof
    FROM checked c CROSS JOIN plan p WHERE c.decision = 'new' AND p.plan_hash = '{approved}'
    ON CONFLICT (bet_id) DO NOTHING RETURNING bet_id
)""" if approved else ""}
SELECT plan_hash,
  (SELECT jsonb_agg(to_jsonb(stats) ORDER BY decision,status) FROM stats) breakdown,
  (SELECT jsonb_agg(jsonb_build_object('id',bid,'reason',decision) ORDER BY bid)
     FROM cases WHERE decision NOT IN ('new','already_imported')) isolated,
  (SELECT jsonb_agg(jsonb_build_object('status',status,'decision',decision,'clv_pct',round((entry/close-1)*100,6)))
     FROM cases WHERE bid IN (SELECT bid FROM src WHERE sheet = 528)) sample_528
  {", (SELECT count(*) FROM inserted) inserted" if approved else ""}
FROM plan
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("formal")
    p.add_argument("full")
    p.add_argument("--side-table-exists", action="store_true")
    p.add_argument("--approve-plan")
    p.add_argument("--section", type=int, help=argparse.SUPPRESS)
    p.add_argument("--section-size", type=int, default=24000, help=argparse.SUPPRESS)
    p.add_argument("--length", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--verify-files", action="store_true")
    args = p.parse_args()
    rows = list(batch(args.formal, args.full))
    if args.verify_files:
        print(f"{len(rows)} fingerprints and same-line CLV values verified against immutable source CSVs")
        return
    sql = script(rows, with_table=args.side_table_exists, approved=args.approve_plan)
    if args.length:
        print(len(sql))
    elif args.section is not None:
        start = args.section * args.section_size
        print(sql[start:start + args.section_size], end="")
    else:
        print(sql)


if __name__ == "__main__":
    main()
