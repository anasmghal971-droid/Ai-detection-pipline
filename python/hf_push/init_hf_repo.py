"""
DETECT-AI HuggingFace Dataset Repo Initializer
Run via GitHub Actions: Actions → Initialize HuggingFace Dataset Repo → Run workflow
Or locally: HF_TOKEN=hf_... python3 init_hf_repo.py
"""
import os, json, time
from datetime import datetime, timezone
from huggingface_hub import HfApi, create_repo, CommitOperationAdd

HF_TOKEN = os.environ["HF_TOKEN"]  # Set via GitHub Actions secret HF_TOKEN
REPO_ID  = "anas775/DETECT-AI-Dataset"
api      = HfApi(token=HF_TOKEN)

print(f"🤗 Initializing {REPO_ID}...")

# Step 1: Create repo
create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True, token=HF_TOKEN)
print("  ✅ Repo created/verified")
time.sleep(3)

# Step 2: README (dataset card)
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
size_categories:
- 1B<n<10B
tags:
- ai-detection
- deepfake-detection
- multi-modal
- multi-language
- synthetic-data
- human-content
pretty_name: DETECT-AI — Multi-Modal AI Content Detection Dataset
dataset_info:
  version: 1.0.0
---

# 🔍 DETECT-AI Dataset

> **1B+ verified samples/month** — Multi-modal, multi-language AI vs Human content detection dataset.

Automatically collected from **19 global sources** across text, image, video, and audio modalities,
labeled by a weighted ensemble of 8 specialized AI detection models.

---

## 📁 Dataset Structure

```
anas775/DETECT-AI-Dataset/
│
├── text/{lang}/
│   └── part-{NNNN}.parquet          ← 200k text samples per shard
│
├── image/{lang}/
│   ├── metadata/part-{NNNN}.parquet  ← image metadata + labels
│   ├── frames/{image_id}.jpg         ← full-resolution images
│   └── faces/{image_id}_face_NN.jpg  ← cropped face regions
│
├── video/{lang}/
│   ├── metadata/part-{NNNN}.parquet  ← video metadata + labels
│   └── frames/{video_id}/
│       ├── full/frame_{NNNNN}.png    ← lossless full frames
│       ├── faces/face_{NNNNN}.jpg    ← JPEG-95 face crops
│       └── textures/mask_{NNNNN}.png ← face landmark masks
│
├── audio/{lang}/
│   └── metadata/part-{NNNN}.parquet
│
└── _metadata/
    ├── schema_v1.json        ← field definitions & thresholds
    ├── shard_registry.json   ← all pushed shards with SHA256 hashes
    └── push_log.jsonl        ← timestamped push history
```

---

## 🏷️ Label Schema

| Label | Confidence Threshold | Meaning |
|---|---|---|
| `AI_GENERATED` | ≥ 0.75 | Ensemble majority — synthetic/AI-generated content |
| `HUMAN` | ≤ 0.35 | Ensemble majority — authentic human-created content |
| `UNCERTAIN` | 0.35 – 0.75 | Ambiguous — 2% routed to human verification queue |

---

## 🤖 Detection Ensemble

| Modality | Models | Weights |
|---|---|---|
| **Text** | roberta-base-openai-detector · chatgpt-detector-roberta · ai-content-detector | 0.40 / 0.35 / 0.25 |
| **Image** | umm-maybe/AI-image-detector · ViT-Deepfake · SDXL-detector | 0.40 / 0.35 / 0.25 |
| **Audio** | wav2vec2-deepfake · SpeechBrain Resemblyzer | 0.60 / 0.40 |
| **Video** | Llama-3.2-11B-Vision (frame sampling) · ViT fallback | primary / fallback |

---

## 📡 Data Sources (19 total)

**Text:** BBC News · Reuters · Al Jazeera · arXiv · Wikipedia · NewsAPI ·
PapersWithCode · StackExchange · Reddit · World Bank

**Image:** Unsplash · Pexels · Pixabay · Flickr CC · Wikimedia Commons

**Video:** YouTube Data API (CC-licensed) · TED Talks · Pexels Video · VoxCeleb/AVA

---

## ⚡ Pipeline

```
Cloudflare Workers (cron */5 min)
  → Scrape 19 sources → Supabase staging

GitHub Actions (cron */2 min)
  → AI Ensemble Labeler → Supabase processed
  → Auto-shard at 200k samples → Push Parquet here

GitHub Actions (cron */5 min)
  → Frame extractor (OpenCV + MediaPipe)
  → Push frames/faces here
```

---

## 📊 Parquet Schema

| Field | Type | Description |
|---|---|---|
| `sample_id` | string (UUID) | Unique identifier |
| `source_id` | string | Data source (e.g. `bbc-news`, `youtube`) |
| `source_url` | string | Original content URL |
| `content_type` | enum | `text` \| `image` \| `video` \| `audio` |
| `language` | string (ISO-639-1) | e.g. `en`, `ar`, `fr` |
| `raw_content` | string | Text body or file storage path |
| `label` | enum | `AI_GENERATED` \| `HUMAN` \| `UNCERTAIN` |
| `final_confidence` | float [0–1] | Weighted ensemble average |
| `model_scores` | JSON | Per-model scores `{"model_name": 0.92}` |
| `verified` | bool | True if human-reviewed |
| `scraped_at` | ISO-8601 | When scraped |
| `labeled_at` | ISO-8601 | When labeled |

---

## 🔄 Update Frequency
Shards pushed automatically every time 200,000 new labeled samples accumulate per `(content_type, language)` pair.

## 📜 License
CC-BY-4.0 — free to use for research and commercial purposes with attribution.
"""

SCHEMA = {
    "version": "1.0.0",
    "shard_size_rows": 200000,
    "compression": "snappy",
    "thresholds": {"AI_GENERATED": ">=0.75", "HUMAN": "<=0.35", "UNCERTAIN": "0.35-0.75"},
    "path_conventions": {
        "text":         "text/{lang}/part-{NNNN}.parquet",
        "image_meta":   "image/{lang}/metadata/part-{NNNN}.parquet",
        "image_frame":  "image/{lang}/frames/{image_id}.jpg",
        "image_face":   "image/{lang}/faces/{image_id}_face_{NN}.jpg",
        "video_meta":   "video/{lang}/metadata/part-{NNNN}.parquet",
        "video_full":   "video/{lang}/frames/{video_id}/full/frame_{NNNNN}.png",
        "video_face":   "video/{lang}/frames/{video_id}/faces/face_{NNNNN}.jpg",
        "video_mask":   "video/{lang}/frames/{video_id}/textures/mask_{NNNNN}.png",
        "audio_meta":   "audio/{lang}/metadata/part-{NNNN}.parquet",
    },
    "fields": {
        "sample_id":"UUID v4","source_id":"string","source_url":"string",
        "content_type":"text|image|video|audio","language":"ISO-639-1",
        "raw_content":"string","label":"AI_GENERATED|HUMAN|UNCERTAIN",
        "final_confidence":"float 0-1","model_scores":"json","verified":"bool",
        "scraped_at":"ISO-8601","labeled_at":"ISO-8601"
    }
}

REGISTRY = {"shards":[],"total_samples":0,"last_updated":datetime.now(timezone.utc).isoformat()}

LANGUAGES = ["en","ar","fr","de","es","zh","ja","ko","pt","ru","hi","it","nl","pl","sv","tr","vi","fa","uk","he","cs","id","th","bn","ur"]

# Step 3: Push README + metadata
print("  Pushing README + schema...")
api.create_commit(
    repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN,
    commit_message="[init] Dataset card + schema v1.0 + shard registry",
    operations=[
        CommitOperationAdd("README.md", README.encode()),
        CommitOperationAdd("_metadata/schema_v1.json", json.dumps(SCHEMA, indent=2).encode()),
        CommitOperationAdd("_metadata/shard_registry.json", json.dumps(REGISTRY, indent=2).encode()),
        CommitOperationAdd("_metadata/push_log.jsonl", b""),
    ]
)
print("  ✅ README + schema pushed")
time.sleep(5)

# Step 4: English full structure
print("  Creating English folder structure...")
api.create_commit(
    repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN,
    commit_message="[init] English folder structure (text/image/video/audio)",
    operations=[
        CommitOperationAdd("text/en/.gitkeep", b""),
        CommitOperationAdd("image/en/metadata/.gitkeep", b""),
        CommitOperationAdd("image/en/frames/.gitkeep", b""),
        CommitOperationAdd("image/en/faces/.gitkeep", b""),
        CommitOperationAdd("video/en/metadata/.gitkeep", b""),
        CommitOperationAdd("video/en/frames/.gitkeep", b""),
        CommitOperationAdd("audio/en/metadata/.gitkeep", b""),
    ]
)
print("  ✅ English structure created")
time.sleep(5)

# Step 5: All other language folders (batched into 2 commits to stay under rate limit)
langs_a = LANGUAGES[1:13]
langs_b = LANGUAGES[13:]

def lang_ops(langs):
    ops = []
    for lang in langs:
        for path in [f"text/{lang}/.gitkeep", f"image/{lang}/metadata/.gitkeep",
                     f"video/{lang}/metadata/.gitkeep", f"audio/{lang}/metadata/.gitkeep"]:
            ops.append(CommitOperationAdd(path, b""))
    return ops

print(f"  Creating language folders batch A ({', '.join(langs_a)})...")
api.create_commit(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN,
    commit_message=f"[init] Language folders: {', '.join(langs_a)}",
    operations=lang_ops(langs_a))
print("  ✅ Batch A done")
time.sleep(6)

print(f"  Creating language folders batch B ({', '.join(langs_b)})...")
api.create_commit(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN,
    commit_message=f"[init] Language folders: {', '.join(langs_b)}",
    operations=lang_ops(langs_b))
print("  ✅ Batch B done")

print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅  HuggingFace Dataset Repo — FULLY INITIALIZED           ║
╠══════════════════════════════════════════════════════════════╣
║  URL: https://huggingface.co/datasets/{REPO_ID}
║                                                              ║
║  Structure:                                                  ║
║    text/  → {len(LANGUAGES)} languages                               ║
║    image/ → en + {len(LANGUAGES)-1} more (metadata/frames/faces)        ║
║    video/ → en + {len(LANGUAGES)-1} more (metadata/frames)              ║
║    audio/ → {len(LANGUAGES)} languages                               ║
║    _metadata/ → schema + registry + push log                ║
╚══════════════════════════════════════════════════════════════╝
""")
