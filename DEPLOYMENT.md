# DETECT-AI — Complete Deployment Guide
# ============================================================
# Stack: Cloudflare Workers + Supabase + HF + Vercel
# ============================================================

## 0. Prerequisites

```bash
npm install -g wrangler           # Cloudflare Workers CLI
pip install -r python/requirements.txt
pip install yt-dlp                # Video downloader
wrangler login                    # Authenticate Cloudflare
```

---

## 1. Environment Variables

Create `python/.env`:

```env
# Supabase (get from: supabase.com → Settings → API)
SUPABASE_URL=https://igwimowqtbgatqvdrqjf.supabase.co
SUPABASE_SERVICE_KEY=<your-service-role-key>

# Hugging Face
HF_TOKEN=<your-hf-token>  # Set as GitHub Secret: HF_TOKEN
HF_DATASET_REPO=anas775/DETECT-AI-Dataset

# External APIs
NEWSAPI_KEY=<your-newsapi-key>
YOUTUBE_API_KEY=<your-youtube-data-api-key>
UNSPLASH_ACCESS_KEY=<your-unsplash-key>
PEXELS_API_KEY=<your-pexels-key>
PIXABAY_API_KEY=<your-pixabay-key>
FLICKR_API_KEY=<your-flickr-key>
REDDIT_CLIENT_ID=<your-reddit-app-id>
REDDIT_CLIENT_SECRET=<your-reddit-secret>

# Upstash Redis (get from: upstash.com)
UPSTASH_REDIS_REST_URL=<your-upstash-url>
UPSTASH_REDIS_REST_TOKEN=<your-upstash-token>

# Worker settings
LABELER_BATCH_SIZE=50
SHARD_THRESHOLD=200000
FRAME_JOBS_PER_CYCLE=3
```

---

## 2. Initialize HF Dataset Repo (Run ONCE)

```bash
cd python/hf_push
pip install huggingface_hub
python3 init_hf_repo.py
# ✅ Creates: https://huggingface.co/datasets/anas775/DETECT-AI-Dataset
```

---

## 3. Create Supabase Storage Bucket

Run in Supabase SQL Editor:
```sql
INSERT INTO storage.buckets (id, name, public)
VALUES ('detect-ai-frames', 'detect-ai-frames', true)
ON CONFLICT DO NOTHING;
```

---

## 4. Deploy Cloudflare Workers

### Step 4a — Set secrets for all workers
```bash
# Run for EACH worker (dispatcher, scraper-text, scraper-image, scraper-video)
for WORKER in detect-ai-dispatcher detect-ai-scraper-text detect-ai-scraper-image detect-ai-scraper-video; do
  wrangler secret put SUPABASE_URL          --name $WORKER
  wrangler secret put SUPABASE_SERVICE_KEY  --name $WORKER
  wrangler secret put HF_TOKEN              --name $WORKER
  wrangler secret put HF_DATASET_REPO       --name $WORKER
  wrangler secret put NEWSAPI_KEY           --name $WORKER
  wrangler secret put YOUTUBE_API_KEY       --name $WORKER
  wrangler secret put UNSPLASH_ACCESS_KEY   --name $WORKER
  wrangler secret put PEXELS_API_KEY        --name $WORKER
  wrangler secret put PIXABAY_API_KEY       --name $WORKER
  wrangler secret put FLICKR_API_KEY        --name $WORKER
  wrangler secret put REDDIT_CLIENT_ID      --name $WORKER
  wrangler secret put REDDIT_CLIENT_SECRET  --name $WORKER
  wrangler secret put PIPELINE_ENABLED      --name $WORKER  # value: "true"
done
```

### Step 4b — Deploy workers
```bash
# Dispatcher (has cron trigger)
cd cloudflare-workers/dispatcher
wrangler deploy

# Text scraper
cd ../scrapers
wrangler deploy --config wrangler-text.toml

# Image scraper
wrangler deploy --config wrangler-image.toml

# Video scraper
wrangler deploy --config wrangler-video.toml
```

### Step 4c — Verify deployment
```bash
wrangler tail detect-ai-dispatcher   # Watch live logs
```

---

## 5. Deploy Vercel API

### Step 5a — Set Vercel environment variables
In Vercel Dashboard → Project → Settings → Environment Variables:

```
SUPABASE_URL              = https://igwimowqtbgatqvdrqjf.supabase.co
SUPABASE_SERVICE_KEY      = <service-role-key>
UPSTASH_REDIS_REST_URL    = <upstash-url>
UPSTASH_REDIS_REST_TOKEN  = <upstash-token>
```

### Step 5b — Deploy
```bash
cd vercel-api
vercel deploy --prod
```

### Step 5c — Test endpoints
```bash
# Query samples
curl "https://your-app.vercel.app/api/dataset/query?content_type=text&language=en&page=1"

# List shards
curl "https://your-app.vercel.app/api/dataset/shards?content_type=text"

# Pipeline stats
curl "https://your-app.vercel.app/api/dataset/stats"
```

---

## 6. Start Python Workers (on a VPS / cloud instance)

```bash
cd python
pip install -r requirements.txt
cp .env.example .env  # fill in values

# Start labeler (runs continuously)
nohup python3 labeling/ensemble_labeler.py > logs/labeler.log 2>&1 &

# Start frame extractor via cron (every 5 min)
echo "*/5 * * * * cd /path/to/detect-ai/python && python3 frame_extraction/frame_extractor.py >> logs/frames.log 2>&1" | crontab -

# Monitor
tail -f logs/labeler.log
tail -f logs/frames.log
```

---

## 7. Verify Full Pipeline

```bash
# 1. Check Cloudflare Worker is running
open https://dash.cloudflare.com → Workers → detect-ai-dispatcher → Logs

# 2. Check Supabase tables have data
# supabase.com → Table Editor → samples_staging

# 3. Check labeler is processing
# Look for rows in samples_processed

# 4. Check HF dataset
open https://huggingface.co/datasets/anas775/DETECT-AI-Dataset

# 5. Check API
curl https://your-app.vercel.app/api/dataset/stats
```

---

## 8. Kill Switch

To immediately stop ALL workers:
```bash
wrangler secret put PIPELINE_ENABLED --name detect-ai-dispatcher
# Enter value: false
```

All workers check this flag before each cycle and halt within 30s.

---

## 9. Architecture Summary

```
Every 5 minutes (CF Cron):
  detect-ai-dispatcher
    ↓ claims source from sources_queue
    ↓ calls scraper worker (text/image/video)
      ↓ fetches content from BBC/YouTube/Unsplash/etc
      ↓ writes to samples_staging (Supabase)
      ↓ queues frame_jobs (for video)

Continuously (Python on VPS):
  ensemble_labeler.py
    ↓ reads staged samples
    ↓ calls HF Inference API ensemble
    ↓ writes to samples_processed
    ↓ when 200k threshold → triggers HF shard push

Every 5 minutes (Python cron):
  frame_extractor.py
    ↓ reads pending frame_jobs
    ↓ downloads video (yt-dlp)
    ↓ extracts frames (OpenCV)
    ↓ detects faces (MediaPipe)
    ↓ uploads to Supabase Storage + HF

Vercel API (always on):
  /api/dataset/query  → filtered paginated samples
  /api/dataset/shards → HF shard listing
  /api/dataset/stats  → pipeline health dashboard
```

---

## 10. Expected Throughput

| Component | Rate |
|---|---|
| 20 CF Workers × 5min cycle | ~1,000–2,000 samples/cycle |
| Labeler (50 batch, 2s gap) | ~1,500 samples/min |
| Frame extractor | 3 videos × 500 frames = 1,500 frames/cycle |
| HF shard push | 1 shard per 200k samples |
| **Target** | **1B samples/month** |
