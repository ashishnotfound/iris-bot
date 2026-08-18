-- Hermes Cloud Telegram Agent — Supabase Schema
-- Run this in Supabase SQL Editor to initialize the database

-- ============================================================
-- 1. Sessions: active session metadata per Telegram chat
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    chat_id     BIGINT PRIMARY KEY,
    session_id  TEXT NOT NULL DEFAULT gen_random_uuid()::TEXT,
    model       TEXT,
    platform    TEXT DEFAULT 'telegram',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- 2. Messages: full conversation history per chat
--    Reconstructed and passed to run_conversation() each turn
-- ============================================================
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    chat_id     BIGINT NOT NULL,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool', 'system')),
    content     JSONB NOT NULL,  -- string or array of content parts
    metadata    JSONB,           -- optional: tool_name, tool_call_id, etc.
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (chat_id) REFERENCES sessions(chat_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_session
    ON messages (chat_id, session_id, created_at);

-- ============================================================
-- 3. Memory: MEMORY.md and USER.md text blobs per chat
--    Written back to disk before AIAgent init each turn
-- ============================================================
CREATE TABLE IF NOT EXISTS memory (
    chat_id     BIGINT PRIMARY KEY,
    memory_md   TEXT NOT NULL DEFAULT '',
    user_md     TEXT NOT NULL DEFAULT '',
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (chat_id) REFERENCES sessions(chat_id) ON DELETE CASCADE
);

-- ============================================================
-- 4. Processed Updates: idempotency guard for Telegram retries
--    Telegram may deliver the same update multiple times.
-- ============================================================
CREATE TABLE IF NOT EXISTS processed_updates (
    update_id   BIGINT PRIMARY KEY,
    chat_id     BIGINT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Auto-expire old idempotency records after 24 hours
-- (Requires pg_cron extension in Supabase, or handle in app code)
CREATE INDEX IF NOT EXISTS idx_processed_updates_created
    ON processed_updates (created_at);

-- ============================================================
-- 5. Tasks: track async/long-running agent tasks
-- ============================================================
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    chat_id     BIGINT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'running', 'done', 'error')),
    user_message TEXT,
    result      TEXT,
    error       TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tasks_chat_status
    ON tasks (chat_id, status, created_at);

-- ============================================================
-- Helper: update updated_at automatically
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE TRIGGER trg_memory_updated_at
    BEFORE UPDATE ON memory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE TRIGGER trg_tasks_updated_at
    BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- Row-Level Security: restrict access to service_role key only
-- ============================================================
ALTER TABLE sessions        ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages        ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory          ENABLE ROW LEVEL SECURITY;
ALTER TABLE processed_updates ENABLE ROW LEVEL SECURITY;
ALTER TABLE tasks           ENABLE ROW LEVEL SECURITY;

-- Allow full access via service_role (used by Vercel Functions)
CREATE POLICY "service_role_all" ON sessions        FOR ALL USING (true);
CREATE POLICY "service_role_all" ON messages        FOR ALL USING (true);
CREATE POLICY "service_role_all" ON memory          FOR ALL USING (true);
CREATE POLICY "service_role_all" ON processed_updates FOR ALL USING (true);
CREATE POLICY "service_role_all" ON tasks           FOR ALL USING (true);

-- ============================================================
-- 6. Cron Jobs: autonomous scheduled task runner
--    Each job runs on a cron schedule and executes as a
--    full agent turn, delivering results to the chat.
-- ============================================================
CREATE TABLE IF NOT EXISTS cron_jobs (
    job_id           TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    chat_id          BIGINT NOT NULL,
    cron_expression  TEXT NOT NULL,          -- 5-field cron: "0 9 * * *"
    task_description TEXT NOT NULL,          -- Natural language task to execute
    enabled          BOOLEAN NOT NULL DEFAULT true,
    last_run_at      TIMESTAMPTZ,
    next_run_at      TIMESTAMPTZ,            -- Pre-computed; updated after each run
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cron_jobs_due
    ON cron_jobs (enabled, next_run_at)
    WHERE enabled = true;

CREATE INDEX IF NOT EXISTS idx_cron_jobs_chat
    ON cron_jobs (chat_id, created_at);

ALTER TABLE cron_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "service_role_all" ON cron_jobs FOR ALL USING (true);

CREATE OR REPLACE TRIGGER trg_cron_jobs_updated_at
    BEFORE UPDATE ON cron_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

