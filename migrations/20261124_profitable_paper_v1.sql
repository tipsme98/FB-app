-- Deploy separately after production schema/role review. Never modifies football_bets.
CREATE TABLE IF NOT EXISTS public.fb_profitable_paper_v1 (
    mid text NOT NULL CHECK (mid ~ '^[1-9][0-9]{0,19}$'),
    kickoff timestamptz NOT NULL,
    observed timestamptz NOT NULL,
    market text NOT NULL CHECK (market IN ('讓球', '入球大小', '角球大小')),
    payload jsonb NOT NULL,
    PRIMARY KEY (mid, kickoff, market),
    CHECK (observed <= kickoff - interval '1 minute')
);
ALTER TABLE public.fb_profitable_paper_v1 ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.fb_profitable_paper_v1 FROM PUBLIC, anon, authenticated;
CREATE INDEX IF NOT EXISTS fb_profitable_paper_v1_kickoff_idx ON public.fb_profitable_paper_v1(kickoff);
