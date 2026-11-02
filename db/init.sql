-- Fraud Detection Platform - PostgreSQL schema (free & open-source)

CREATE TABLE IF NOT EXISTS scored_transactions (
    txn_id        TEXT PRIMARY KEY,
    event_ts      TIMESTAMPTZ NOT NULL,
    user_id       TEXT,
    card_id       TEXT,
    merchant_id   TEXT,
    amount        NUMERIC(12,2),
    p_fraud       NUMERIC(6,4),
    decision      TEXT,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS cases (
    case_id         SERIAL PRIMARY KEY,
    txn_id          TEXT UNIQUE NOT NULL,
    status          TEXT NOT NULL DEFAULT 'OPEN',
    decision        TEXT,
    p_fraud         NUMERIC(6,4),
    amount          NUMERIC(12,2),
    user_id         TEXT,
    created_ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_ts     TIMESTAMPTZ,
    analyst_verdict TEXT
);

CREATE TABLE IF NOT EXISTS analyst_labels (
    label_id   SERIAL PRIMARY KEY,
    txn_id     TEXT NOT NULL,
    true_label INT  NOT NULL,
    analyst    TEXT NOT NULL DEFAULT 'analyst',
    created_ts TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cases_status ON cases (status);
CREATE INDEX IF NOT EXISTS idx_scored_ts ON scored_transactions (event_ts DESC);
