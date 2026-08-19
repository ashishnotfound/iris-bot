-- Iris Phase 2 Schema Extensions
-- Run these in Supabase SQL Editor AFTER the original schema.sql
-- These are idempotent ALTER TABLE statements — safe to run multiple times.

-- ============================================================
-- Extend cron_jobs with Phase 2 fields
-- ============================================================

ALTER TABLE cron_jobs
    ADD COLUMN IF NOT EXISTS timezone        TEXT         NOT NULL DEFAULT 'UTC',
    ADD COLUMN IF NOT EXISTS retry_count     INT          NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS max_retries     INT          NOT NULL DEFAULT 3,
    ADD COLUMN IF NOT EXISTS last_result     TEXT,
    ADD COLUMN IF NOT EXISTS last_error      TEXT,
    ADD COLUMN IF NOT EXISTS running_since   TIMESTAMPTZ,  -- set when job starts, cleared when done
    ADD COLUMN IF NOT EXISTS action_type     TEXT,         -- 'social_post', 'business_sync', 'report', 'custom'
    ADD COLUMN IF NOT EXISTS action_params   JSONB;        -- structured params for the action

-- ============================================================
-- Business Snapshots: cached Amazon/Flipkart data per chat
-- ============================================================

CREATE TABLE IF NOT EXISTS business_snapshots (
    id               BIGSERIAL PRIMARY KEY,
    chat_id          BIGINT NOT NULL,
    platform         TEXT NOT NULL,           -- 'amazon', 'flipkart'
    snapshot_date    DATE NOT NULL,           -- the date this snapshot covers
    data             JSONB NOT NULL,          -- the actual business data blob
    sync_cursor      TEXT,                    -- incremental sync checkpoint (e.g. last order_id, ISO date)
    synced_at        TIMESTAMPTZ DEFAULT NOW(),
    sync_error       TEXT,
    UNIQUE (chat_id, platform, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_business_snapshots_lookup
    ON business_snapshots (chat_id, platform, snapshot_date DESC);

ALTER TABLE business_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all" ON business_snapshots FOR ALL USING (true);

-- ============================================================
-- Action Log: Composio idempotency record
-- Each external action (post, API call) gets a dedup key.
-- ============================================================

CREATE TABLE IF NOT EXISTS action_log (
    dedup_key     TEXT PRIMARY KEY,           -- unique key per action (job_id + date + action)
    chat_id       BIGINT NOT NULL,
    job_id        TEXT,                       -- cron job that triggered this
    action_type   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'success', 'failed')),
    request_data  JSONB,
    response_data JSONB,
    error         TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_action_log_chat
    ON action_log (chat_id, created_at DESC);

ALTER TABLE action_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all" ON action_log FOR ALL USING (true);

CREATE OR REPLACE TRIGGER trg_action_log_updated_at
    BEFORE UPDATE ON action_log
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- Extend tasks table (already exists) with timeout tracking
-- ============================================================

ALTER TABLE tasks
    ADD COLUMN IF NOT EXISTS task_type   TEXT DEFAULT 'conversation',
    ADD COLUMN IF NOT EXISTS started_at  TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS timeout_at  TIMESTAMPTZ;
