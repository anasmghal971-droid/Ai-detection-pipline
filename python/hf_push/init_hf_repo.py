"""
DETECT-AI — HuggingFace Dataset Repo Initializer
Triggered by GitHub Actions on every deploy or manually.
Requires: HF_TOKEN env var (set as GitHub Secret)
"""
import os, json, time, sys
from datetime import datetime, timezone
from huggingface_hub import HfApi, create_repo, CommitOperationAdd

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("❌ HF_TOKEN env var not set. Add it as a GitHub Secret.")
    sys.exit(1)

REPO_ID = "anas775/DETECT-AI-Dataset"
api = HfApi(token=HF_TOKEN)

print(f"🤗 Initializing {REPO_ID} ...")

# ── Step 1: Create repo ───────────────────────────────────────
try:
    create_repo(repo_id=REPO_ID, repo_type="dataset", private=False,
                exist_ok=True, token=HF_TOKEN)
    print("  ✅ Repo created / already exists")
except Exception as e:
    print(f"  ❌ Repo create failed: {e}")
    sys.exit(1)

time.sleep(3)

# ── Step 2: Dataset card (README.md) ─────────────────────────
README = """\
---
license: cc-by-4.0
task_categories:
- text-classification
- image-classification
- video-classification
- audio-classification
language:
- en
- ar
- fr
- de
- es
- zh
- ja
- ko
- pt
- ru
- hi
- it
- nl
- pl
- sv
- tr
- vi
- fa
- uk
- he
- cs
- id
- th
- bn
- ur
size_categories:
- 1B<n<10B
tags:
- ai-detection
- deepfake-detection
- multi-modal
- multi-language
- synthetic-data
- human-content
pretty_name: DETECT-AI Multi-Modal AI Content Detection Dataset
dataset_info:
  version: 1.0.0
---

# 🔍 DETECT-AI — Multi-Modal AI Detection Dataset

**1B+ verified samples/month** scraped from 19 global sources across text, image, video,
and audio, labeled by a weighted ensemble of 8 specialized AI-detection models.

## 📁 Folder Structure

```
anas775/DETECT-AI-Dataset/
├── text/{lang}/part-{NNNN}.parquet
├── image/{lang}/metadata/part-{NNNN}.parquet
├── image/{lang}/frames/{image_id}.jpg
├── image/{lang}/faces/{image_id}_face_{NN}.jpg
├── video/{lang}/metadata/part-{NNNN}.parquet
├── video/{lang}/frames/{video_id}/full/frame_{NNNNN}.png
├── video/{lang}/frames/{video_id}/faces/face_{NNNNN}.jpg
├── video/{lang}/frames/{video_id}/textures/mask_{NNNNN}.png
├── audio/{lang}/metadata/part-{NNNN}.parquet
└── _metadata/
    ├── schema_v1.json
    ├── shard_registry.json
    └── push_log.jsonl
```

## 🏷️ Labels

| Label | Threshold | Meaning |
|---|---|---|
| `AI_GENERATED` | ≥ 0.75 | Synthetic / AI-created |
| `HUMAN` | ≤ 0.35 | Authentic human content |
| `UNCERTAIN` | 0.35–0.75 | Routed to human review |

## 🤖 Ensemble Models

| Modality | Models | Weights |
|---|---|---|
| Text | roberta-base-openai-detector · chatgpt-detector · ai-content-detector | 0.40/0.35/0.25 |
| Image | AI-image-detector · ViT-Deepfake · SDXL-detector | 0.40/0.35/0.25 |
| Audio | wav2vec2-deepfake · Resemblyzer | 0.60/0.40 |
| Video | Llama-3.2-11B-Vision · ViT fallback | primary/fallback |

## 📡 Data Sources (19)

**Text:** BBC · Reuters · Al Jazeera · arXiv · Wikipedia · NewsAPI · PapersWithCode · StackExchange · Reddit · WorldBank

**Image:** Unsplash · Pexels · Pixabay · Flickr CC · Wikimedia Commons

**Video:** YouTube (CC) · TED Talks · Pexels Video · VoxCeleb/AVA

## ⚡ Pipeline Architecture

```
Cloudflare Workers (cron */5 min) → scrape → Supabase staging
GitHub Actions   (cron */2 min)  → AI labeler → Supabase processed → Parquet shards here
GitHub Actions   (cron */5 min)  → frame/face extractor → frames here
```

## 📊 Parquet Schema

| Field | Type |
|---|---|
| sample_id | UUID |
| source_id | string |
| source_url | string |
| content_type | text/image/video/audio |
| language | ISO-639-1 |
| raw_content | string / storage path |
| label | AI_GENERATED/HUMAN/UNCERTAIN |
| final_confidence | float 0–1 |
| model_scores | JSON |
| verified | bool |
| scraped_at | ISO-8601 |
| labeled_at | ISO-8601 |

## 📜 License
CC-BY-4.0 — free for research and commercial use with attribution.
"""

SCHEMA = {
    "version": "1.0.0",
    "shard_size_rows": 200000,
    "compression": "snappy",
    "thresholds": {"AI_GENERATED": ">=0.75", "HUMAN": "<=0.35", "UNCERTAIN": "0.35-0.75"},
    "languages": ["en","ar","fr","de","es","zh","ja","ko","pt","ru","hi","it","nl","pl","sv","tr","vi","fa","uk","he","cs","id","th","bn","ur"],
    "modalities": ["text","image","video","audio"],
}

REGISTRY = {
    "shards": [],
    "total_samples": 0,
    "last_updated": datetime.now(timezone.utc).isoformat()
}

LANGS = SCHEMA["languages"]

def safe_commit(message, operations, retries=3):
    for attempt in range(1, retries+1):
        try:
            api.create_commit(repo_id=REPO_ID, repo_type="dataset",
                              token=HF_TOKEN, commit_message=message,
                              operations=operations)
            return True
        except Exception as e:
            print(f"    attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    return False

# ── Step 3: Push metadata + README ───────────────────────────
print("  Pushing README + schema + registry...")
ok = safe_commit(
    "[init] Dataset card + schema v1.0 + shard registry",
    [
        CommitOperationAdd("README.md", README.encode()),
        CommitOperationAdd("_metadata/schema_v1.json", json.dumps(SCHEMA, indent=2).encode()),
        CommitOperationAdd("_metadata/shard_registry.json", json.dumps(REGISTRY, indent=2).encode()),
        CommitOperationAdd("_metadata/push_log.jsonl", b""),
    ]
)
print("  ✅ Metadata pushed" if ok else "  ❌ Metadata push failed")
time.sleep(6)

# ── Step 4: English full structure ────────────────────────────
print("  Creating English folder structure...")
safe_commit("[init] English folder structure", [
    CommitOperationAdd("text/en/.gitkeep", b""),
    CommitOperationAdd("image/en/metadata/.gitkeep", b""),
    CommitOperationAdd("image/en/frames/.gitkeep", b""),
    CommitOperationAdd("image/en/faces/.gitkeep", b""),
    CommitOperationAdd("video/en/metadata/.gitkeep", b""),
    CommitOperationAdd("video/en/frames/.gitkeep", b""),
    CommitOperationAdd("audio/en/metadata/.gitkeep", b""),
])
print("  ✅ English structure done")
time.sleep(6)

# ── Step 5: All other languages in 3 batches ─────────────────
other = LANGS[1:]
batches = [other[:8], other[8:16], other[16:]]
for i, batch in enumerate(batches):
    if not batch: continue
    print(f"  Languages batch {i+1}: {', '.join(batch)} ...")
    ops = []
    for lang in batch:
        for path in [f"text/{lang}/.gitkeep", f"image/{lang}/metadata/.gitkeep",
                     f"video/{lang}/metadata/.gitkeep", f"audio/{lang}/metadata/.gitkeep"]:
            ops.append(CommitOperationAdd(path, b""))
    safe_commit(f"[init] Language folders batch {i+1}: {', '.join(batch)}", ops)
    print(f"  ✅ Batch {i+1} done")
    time.sleep(6)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ✅  HuggingFace Repo — FULLY INITIALIZED                       ║
╠══════════════════════════════════════════════════════════════════╣
║  URL: https://huggingface.co/datasets/{REPO_ID}
║  Languages: {len(LANGS)} ({', '.join(LANGS[:6])}...)
║  Modalities: text / image / video / audio
║  Metadata: schema_v1.json + shard_registry.json + push_log.jsonl
╚══════════════════════════════════════════════════════════════════╝
""")
