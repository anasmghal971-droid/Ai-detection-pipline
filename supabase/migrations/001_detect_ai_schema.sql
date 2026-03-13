-- ============================================================
-- DETECT-AI: Complete Database Schema Migration
-- Apply via: Supabase MCP → apply_migration
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- ── 1. Sources Queue ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sources_queue (
  source_id           TEXT PRIMARY KEY,
  source_url          TEXT NOT NULL,
  content_type        TEXT NOT NULL CHECK (content_type IN ('text','image','video','audio')),
  language_hint       TEXT NOT NULL DEFAULT 'en',
  crawl_frequency     TEXT NOT NULL CHECK (crawl_frequency IN ('hourly','daily','weekly','once')),
  last_crawled_at     TIMESTAMPTZ,
  status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','active','completed','failed','rate_limited')),
  priority_score      INTEGER NOT NULL DEFAULT 50 CHECK (priority_score BETWEEN 0 AND 100),
  access_method       TEXT NOT NULL CHECK (access_method IN ('http','rss','api','dataset_download')),
  worker_id           TEXT,
  heartbeat_at        TIMESTAMPTZ,
  retry_count         INTEGER NOT NULL DEFAULT 0,
  error_message       TEXT,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sources_status_priority 
  ON sources_queue(status, priority_score DESC);
CREATE INDEX IF NOT EXISTS idx_sources_worker 
  ON sources_queue(worker_id);

-- ── 2. Staged Samples (raw, pre-labeling) ─────────────────────
CREATE TABLE IF NOT EXISTS samples_staging (
  sample_id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source_id           TEXT NOT NULL REFERENCES sources_queue(source_id),
  source_url          TEXT NOT NULL,
  content_type        TEXT NOT NULL CHECK (content_type IN ('text','image','video','audio')),
  language            TEXT NOT NULL DEFAULT 'unknown',
  raw_content         TEXT,
  storage_path        TEXT,           -- For binary (image/video/audio)
  metadata            JSONB NOT NULL DEFAULT '{}',
  scraped_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  worker_id           TEXT NOT NULL,
  status              TEXT NOT NULL DEFAULT 'staged'
                        CHECK (status IN ('staged','labeling','labeled','failed'))
);

CREATE INDEX IF NOT EXISTS idx_staging_status 
  ON samples_staging(status);
CREATE INDEX IF NOT EXISTS idx_staging_content_type 
  ON samples_staging(content_type);
CREATE INDEX IF NOT EXISTS idx_staging_language 
  ON samples_staging(language);
CREATE INDEX IF NOT EXISTS idx_staging_source 
  ON samples_staging(source_id);

-- ── 3. Processed Samples (labeled, shard-ready) ───────────────
CREATE TABLE IF NOT EXISTS samples_processed (
  sample_id           UUID PRIMARY KEY,
  source_id           TEXT NOT NULL,
  source_url          TEXT NOT NULL,
  content_type        TEXT NOT NULL CHECK (content_type IN ('text','image','video','audio')),
  language            TEXT NOT NULL,
  raw_content         TEXT,
  storage_path        TEXT,
  metadata            JSONB NOT NULL DEFAULT '{}',
  scraped_at          TIMESTAMPTZ NOT NULL,
  worker_id           TEXT NOT NULL,
  -- Labeling fields
  label               TEXT NOT NULL CHECK (label IN ('AI_GENERATED','HUMAN','UNCERTAIN')),
  final_confidence    NUMERIC(4,3) NOT NULL CHECK (final_confidence BETWEEN 0 AND 1),
  model_scores        JSONB NOT NULL DEFAULT '{}',
  verified            BOOLEAN NOT NULL DEFAULT FALSE,
  labeled_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  -- Shard tracking
  shard_id            TEXT,
  exported_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_processed_content_lang 
  ON samples_processed(content_type, language);
CREATE INDEX IF NOT EXISTS idx_processed_label 
  ON samples_processed(label);
CREATE INDEX IF NOT EXISTS idx_processed_shard 
  ON samples_processed(shard_id);
CREATE INDEX IF NOT EXISTS idx_processed_verified 
  ON samples_processed(verified);
CREATE INDEX IF NOT EXISTS idx_processed_confidence 
  ON samples_processed(final_confidence);

-- ── 4. Human Verification Queue ───────────────────────────────
CREATE TABLE IF NOT EXISTS verification_queue (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  sample_id           UUID NOT NULL REFERENCES samples_processed(sample_id),
  assigned_to         TEXT,
  status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','in_review','completed','skipped')),
  human_label         TEXT CHECK (human_label IN ('AI_GENERATED','HUMAN')),
  reviewer_notes      TEXT,
  queued_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_verification_status 
  ON verification_queue(status);

-- ── 5. Frame / Face Metadata ──────────────────────────────────
CREATE TABLE IF NOT EXISTS frame_metadata (
  frame_id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  sample_id           UUID NOT NULL REFERENCES samples_staging(sample_id),
  video_id            TEXT NOT NULL,
  frame_index         INTEGER NOT NULL,
  timestamp_ms        INTEGER NOT NULL,
  motion_score        NUMERIC(4,3),
  faces_detected      INTEGER NOT NULL DEFAULT 0,
  bounding_boxes      JSONB NOT NULL DEFAULT '[]',
  full_frame_path     TEXT NOT NULL,
  face_crop_paths     JSONB NOT NULL DEFAULT '[]',
  face_texture_mask_paths JSONB NOT NULL DEFAULT '[]',
  extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_frames_sample 
  ON frame_metadata(sample_id);
CREATE INDEX IF NOT EXISTS idx_frames_video 
  ON frame_metadata(video_id);
CREATE INDEX IF NOT EXISTS idx_frames_faces 
  ON frame_metadata(faces_detected);

-- ── 6. Shard Registry ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS shard_registry (
  shard_id            TEXT PRIMARY KEY,
  content_type        TEXT NOT NULL,
  language            TEXT NOT NULL,
  sample_count        INTEGER NOT NULL,
  size_bytes          BIGINT NOT NULL,
  sha256_hash         TEXT NOT NULL,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  schema_version      TEXT NOT NULL DEFAULT '1.0',
  hf_url              TEXT,
  push_status         TEXT NOT NULL DEFAULT 'pending'
                        CHECK (push_status IN ('pending','pushed','failed')),
  source_distribution JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_shards_content_lang 
  ON shard_registry(content_type, language);
CREATE INDEX IF NOT EXISTS idx_shards_push_status 
  ON shard_registry(push_status);

-- ── 7. Worker Logs ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS worker_logs (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event               TEXT NOT NULL,
  worker_id           TEXT NOT NULL,
  source_id           TEXT,
  sample_count        INTEGER,
  duration_ms         INTEGER,
  error_message       TEXT,
  timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logs_worker 
  ON worker_logs(worker_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_logs_event 
  ON worker_logs(event, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_logs_source 
  ON worker_logs(source_id);

-- ── 8. Pipeline Metrics (updated every 5 min) ────────────────
CREATE TABLE IF NOT EXISTS pipeline_metrics (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  measured_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  total_staged        BIGINT NOT NULL DEFAULT 0,
  total_labeled       BIGINT NOT NULL DEFAULT 0,
  total_exported      BIGINT NOT NULL DEFAULT 0,
  total_shards_pushed INTEGER NOT NULL DEFAULT 0,
  active_workers      INTEGER NOT NULL DEFAULT 0,
  error_rate_pct      NUMERIC(5,2),
  samples_per_min     NUMERIC(10,2),
  source_breakdown    JSONB NOT NULL DEFAULT '{}',
  label_distribution  JSONB NOT NULL DEFAULT '{}'
);

-- ── 9. Seed: All 18 Data Sources ──────────────────────────────
INSERT INTO sources_queue 
  (source_id, source_url, content_type, language_hint, crawl_frequency, 
   priority_score, access_method) 
VALUES
  ('bbc-news',       'https://feeds.bbci.co.uk/news/rss.xml',           'text',  'en',    'daily',  90, 'rss'),
  ('reuters',        'https://feeds.reuters.com/reuters/topNews',        'text',  'en',    'daily',  90, 'rss'),
  ('aljazeera',      'https://www.aljazeera.com/xml/rss/all.xml',        'text',  'en',    'daily',  85, 'rss'),
  ('arxiv',          'https://export.arxiv.org/api/query',               'text',  'multi', 'daily',  85, 'api'),
  ('wikipedia',      'https://en.wikipedia.org/w/api.php',               'text',  'multi', 'weekly', 80, 'api'),
  ('paperswithcode', 'https://paperswithcode.com/api/v1/papers/',        'text',  'en',    'daily',  75, 'api'),
  ('newsapi',        'https://newsapi.org/v2/top-headlines',             'text',  'multi', 'hourly', 95, 'api'),
  ('unsplash',       'https://api.unsplash.com/photos',                  'image', 'en',    'daily',  80, 'api'),
  ('pexels',         'https://api.pexels.com/v1/curated',                'image', 'en',    'daily',  80, 'api'),
  ('pexels-video',   'https://api.pexels.com/videos/popular',            'video', 'en',    'daily',  80, 'api'),
  ('pixabay',        'https://pixabay.com/api/',                         'image', 'multi', 'daily',  75, 'api'),
  ('flickr-cc',      'https://www.flickr.com/services/rest/',            'image', 'multi', 'daily',  70, 'api'),
  ('wikimedia',      'https://commons.wikimedia.org/w/api.php',          'image', 'multi', 'weekly', 75, 'api'),
  ('youtube',        'https://www.googleapis.com/youtube/v3/search',     'video', 'multi', 'hourly', 95, 'api'),
  ('ted-talks',      'https://www.ted.com/talks',                        'video', 'multi', 'weekly', 70, 'http'),
  ('voxceleb',       'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/',  'video', 'en',    'once',   60, 'dataset_download'),
  ('stackexchange',  'https://api.stackexchange.com/2.3/questions',      'text',  'en',    'daily',  75, 'api'),
  ('reddit',         'https://oauth.reddit.com/r/all/hot',               'text',  'multi', 'daily',  70, 'api'),
  ('worldbank',      'https://api.worldbank.org/v2/en/indicator',        'text',  'multi', 'weekly', 65, 'api')
ON CONFLICT (source_id) DO NOTHING;

-- ── 10. Auto-update updated_at trigger ───────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_sources_updated_at
  BEFORE UPDATE ON sources_queue
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ── 11. View: Pipeline Dashboard ──────────────────────────────
CREATE OR REPLACE VIEW pipeline_dashboard AS
SELECT
  (SELECT COUNT(*) FROM samples_staging  WHERE status = 'staged')    AS staged_count,
  (SELECT COUNT(*) FROM samples_processed)                             AS processed_count,
  (SELECT COUNT(*) FROM samples_processed WHERE verified = TRUE)       AS verified_count,
  (SELECT COUNT(*) FROM samples_processed WHERE label = 'AI_GENERATED') AS ai_generated,
  (SELECT COUNT(*) FROM samples_processed WHERE label = 'HUMAN')       AS human_count,
  (SELECT COUNT(*) FROM samples_processed WHERE label = 'UNCERTAIN')   AS uncertain_count,
  (SELECT COUNT(*) FROM shard_registry   WHERE push_status = 'pushed') AS shards_pushed,
  (SELECT COUNT(*) FROM verification_queue WHERE status = 'pending')   AS pending_verification,
  (SELECT COUNT(*) FROM sources_queue    WHERE status = 'active')      AS active_sources;
