// ============================================================
// DETECT-AI: Unified Type Definitions
// Shared across all Cloudflare Workers, Vercel API, and Python
// ============================================================

export type ContentType = "text" | "image" | "video" | "audio";
export type SampleLabel = "AI_GENERATED" | "HUMAN" | "UNCERTAIN";
export type SampleStatus = "staged" | "labeled" | "verified" | "exported" | "failed";
export type CrawlFrequency = "hourly" | "daily" | "weekly" | "once";

// ── Core Sample Schema ────────────────────────────────────────
export interface DetectAISample {
  sample_id: string;               // UUID v4
  source_id: string;               // References sources_queue.source_id
  source_url: string;
  content_type: ContentType;
  language: string;                // ISO-639-1 code
  raw_content: string;             // Text content or storage path ref
  metadata: SampleMetadata;
  scraped_at: string;              // ISO-8601
  worker_id: string;
  status: SampleStatus;
  // Populated after labeling:
  label?: SampleLabel;
  final_confidence?: number;       // 0.0 – 1.0
  model_scores?: Record<string, number>;
  verified?: boolean;
  labeled_at?: string;
}

export interface SampleMetadata {
  title?: string;
  author?: string;
  publish_date?: string;
  license?: string;
  description?: string;
  tags?: string[];
  dimensions?: { width: number; height: number };
  duration_seconds?: number;
  file_size_bytes?: number;
}

// ── Source Queue Schema ───────────────────────────────────────
export interface SourceQueueItem {
  source_id: string;
  source_url: string;
  content_type: ContentType;
  language_hint: string;
  crawl_frequency: CrawlFrequency;
  last_crawled_at: string | null;
  status: "pending" | "active" | "completed" | "failed" | "rate_limited";
  priority_score: number;          // 0–100; higher = scraped first
  access_method: "http" | "rss" | "api" | "dataset_download";
  worker_id: string | null;
  heartbeat_at: string | null;
  retry_count: number;
  error_message: string | null;
}

// ── Frame / Face Extraction Schema ───────────────────────────
export interface FrameMetadata {
  frame_id: string;                // UUID v4
  sample_id: string;               // Parent video sample
  video_id: string;
  frame_index: number;
  timestamp_ms: number;
  motion_score: number;            // 0.0 – 1.0
  faces_detected: number;
  bounding_boxes: BoundingBox[];
  full_frame_path: string;
  face_crop_paths: string[];
  face_texture_mask_paths: string[];
  extracted_at: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
}

// ── Worker Log Schema ─────────────────────────────────────────
export type WorkerEvent =
  | "SCRAPE_START" | "SCRAPE_COMPLETE" | "SCRAPE_ERROR"
  | "LABEL_START"  | "LABEL_COMPLETE"  | "LABEL_ERROR"
  | "SHARD_PUSH"   | "SHARD_ERROR"
  | "RATE_LIMIT_HIT" | "WORKER_TIMEOUT" | "WORKER_START"
  | "KILL_SWITCH_ACTIVE"
  | "CRON_TICK" | "SLOT_ERROR" | "RATE_LIMIT_HIT" | "INSERT_ERROR";

export interface WorkerLog {
  event: WorkerEvent;
  worker_id: string;
  source_id?: string;
  sample_count?: number;
  duration_ms?: number;
  error_message?: string;
  timestamp: string;
}

// ── Shard Metadata ─────────────────────────────────────────────
export interface ShardMetadata {
  shard_id: string;
  content_type: ContentType;
  language: string;
  sample_count: number;
  size_bytes: number;
  sha256_hash: string;
  created_at: string;
  schema_version: string;
  hf_url?: string;
  push_status: "pending" | "pushed" | "failed";
  source_distribution: Record<string, number>;
}

// ── Pipeline Env Vars (Cloudflare Workers) ────────────────────
export interface Env {
  PIPELINE_ENABLED: string;           // "true" | "false" — global kill switch
  SUPABASE_URL: string;
  SUPABASE_SERVICE_KEY: string;
  HF_TOKEN: string;
  HF_DATASET_REPO: string;            // e.g. "detect-ai/DETECT-AI-Dataset"
  NEWSAPI_KEY: string;
  YOUTUBE_API_KEY: string;
  FLICKR_API_KEY: string;
  UNSPLASH_ACCESS_KEY: string;
  PEXELS_API_KEY: string;
  PIXABAY_API_KEY: string;
  REDDIT_CLIENT_ID: string;
  REDDIT_CLIENT_SECRET: string;
  WORKER_ID: string;                  // Injected per worker instance
  UPSTASH_REDIS_URL: string;
  UPSTASH_REDIS_TOKEN: string;
}
