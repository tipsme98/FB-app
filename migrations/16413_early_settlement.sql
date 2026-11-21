-- V17 (schema identifier 16413). Run only on the intended PostgreSQL project after approval.
-- Apply as the existing app DB owner, or explicitly provision a narrowly scoped service role.
-- No ledger changes, extensions, grants, backfill, or automated DROP/rollback.
BEGIN;
SET LOCAL lock_timeout = '5s';
CREATE TABLE IF NOT EXISTS public.fb_early_state_16413 (
    match_id TEXT PRIMARY KEY CHECK (match_id ~ '^[1-9][0-9]{0,11}$'),
    fingerprint TEXT CHECK (fingerprint ~ '^[0-9a-f]{64}$'),
    first_good TIMESTAMPTZ,
    last_good TIMESTAMPTZ,
    last_attempt TIMESTAMPTZ,
    reason TEXT NOT NULL,
    manual BOOLEAN NOT NULL DEFAULT false,
    CHECK ((fingerprint IS NULL AND first_good IS NULL AND last_good IS NULL)
        OR (fingerprint IS NOT NULL AND first_good IS NOT NULL AND last_good IS NOT NULL AND last_attempt IS NOT NULL
            AND first_good <= last_good AND last_good <= last_attempt))
);
CREATE TABLE IF NOT EXISTS public.fb_early_lease_16413 (
    singleton BOOLEAN PRIMARY KEY CHECK (singleton),
    last_scan TIMESTAMPTZ
);
INSERT INTO public.fb_early_lease_16413 (singleton) VALUES (true) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS public.fb_early_receipt_16413 (
    slip_id TEXT PRIMARY KEY,
    match_id TEXT NOT NULL CHECK (match_id ~ '^[1-9][0-9]{0,11}$'),
    match_day DATE NOT NULL,
    fingerprint TEXT NOT NULL CHECK (fingerprint ~ '^[0-9a-f]{64}$'),
    committed_at TIMESTAMPTZ NOT NULL,
    expected JSONB NOT NULL CHECK (jsonb_typeof(expected) = 'object')
);
CREATE INDEX IF NOT EXISTS fb_early_receipt_16413_match ON public.fb_early_receipt_16413 (match_id);
CREATE INDEX IF NOT EXISTS fb_early_receipt_16413_day ON public.fb_early_receipt_16413 (match_day);
ALTER TABLE public.fb_early_state_16413 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fb_early_lease_16413 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fb_early_receipt_16413 ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.fb_early_state_16413, public.fb_early_lease_16413, public.fb_early_receipt_16413 FROM PUBLIC;
-- Supabase default grants can be explicit even when PUBLIC has been revoked.
DO $$
DECLARE role_name TEXT;
BEGIN
    FOR role_name IN SELECT rolname FROM pg_roles WHERE rolname IN ('anon', 'authenticated')
    LOOP
        EXECUTE format('REVOKE ALL ON public.fb_early_state_16413, public.fb_early_lease_16413, '
                       'public.fb_early_receipt_16413 FROM %I', role_name);
    END LOOP;
END $$;
COMMIT;
