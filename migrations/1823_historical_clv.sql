-- Additive only. Run via import_historical_clv.py after reading the target schema.
CREATE TABLE IF NOT EXISTS public.fb_historical_clv_1823 (
    bet_id text PRIMARY KEY,
    batch_id text NOT NULL,
    source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    fingerprint text NOT NULL CHECK (fingerprint ~ '^[0-9a-f]{64}$'),
    original_json text NOT NULL CHECK (jsonb_typeof(original_json::jsonb) = 'array'),
    closing_odds numeric NOT NULL CHECK (closing_odds > 1 AND closing_odds < 'Infinity'::numeric),
    evidence jsonb NOT NULL CHECK (jsonb_typeof(evidence) = 'object'),
    imported_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.fb_historical_clv_1823 ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.fb_historical_clv_1823 FROM PUBLIC;
DO $$
DECLARE r text;
BEGIN
    FOR r IN SELECT rolname FROM pg_roles WHERE rolname IN ('anon', 'authenticated', 'service_role')
    LOOP
        EXECUTE format('REVOKE ALL ON public.fb_historical_clv_1823 FROM %I', r);
    END LOOP;
END $$;
