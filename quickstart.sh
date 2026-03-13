#!/bin/bash
# ============================================================
# DETECT-AI — ONE COMMAND FULL CLOUD LAUNCH
# Usage:  bash quickstart.sh
# All tokens/keys are loaded from environment variables.
# Set them before running:
#
#   export GITHUB_TOKEN="ghp_..."
#   export HF_TOKEN="hf_..."
#   export CF_API_TOKEN="pZ5V..."
#   export SUPABASE_SERVICE_KEY="eyJ..."
#   bash quickstart.sh
# ============================================================
set -euo pipefail

# ── Required tokens (set as env vars) ────────────────────────
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN env var before running}"
: "${HF_TOKEN:?Set HF_TOKEN env var before running}"
: "${CF_API_TOKEN:?Set CF_API_TOKEN env var before running}"
: "${SUPABASE_SERVICE_KEY:?Set SUPABASE_SERVICE_KEY env var before running}"

CF_ACCOUNT_ID="${CF_ACCOUNT_ID:-82108deb1676ae1d0b50dea4276b6cb2}"
SUPABASE_URL="${SUPABASE_URL:-https://igwimowqtbgatqvdrqjf.supabase.co}"

# ── Optional API keys (add via GitHub Secrets UI if not set) ──
NEWSAPI_KEY="${NEWSAPI_KEY:-}"
YOUTUBE_API_KEY="${YOUTUBE_API_KEY:-}"
UNSPLASH_ACCESS_KEY="${UNSPLASH_ACCESS_KEY:-}"
PEXELS_API_KEY="${PEXELS_API_KEY:-}"
PIXABAY_API_KEY="${PIXABAY_API_KEY:-}"
FLICKR_API_KEY="${FLICKR_API_KEY:-}"
REDDIT_CLIENT_ID="${REDDIT_CLIENT_ID:-}"
REDDIT_CLIENT_SECRET="${REDDIT_CLIENT_SECRET:-}"
UPSTASH_REDIS_URL="${UPSTASH_REDIS_URL:-}"
UPSTASH_REDIS_TOKEN="${UPSTASH_REDIS_TOKEN:-}"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     DETECT-AI Pipeline — Cloud Launch           ║"
echo "╚══════════════════════════════════════════════════╝"

if ! command -v gh &>/dev/null; then
  echo "❌ gh CLI not found. Install: brew install gh  OR  apt install gh"
  exit 1
fi

# 1. Auth
echo "🔐 [1/4] Authenticating GitHub..."
echo "$GITHUB_TOKEN" | gh auth login --with-token
GITHUB_USER=$(gh api user --jq .login)
echo "   ✅ Logged in as: $GITHUB_USER"

# 2. Push code
echo "🚀 [2/4] Pushing to GitHub..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
git remote remove origin 2>/dev/null || true
git remote add origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/Ai-detection-pipline.git"
git branch -M main
git push -u origin main --force
echo "   ✅ All files pushed"

# 3. Set secrets
echo "🔑 [3/4] Setting GitHub Actions secrets..."
R="${GITHUB_USER}/Ai-detection-pipline"
_s() { [[ -n "$2" ]] && gh secret set "$1" --body "$2" --repo "$R" && echo "   ✅ $1" || echo "   ⚠️  $1 skipped (empty)"; }

_s CF_API_TOKEN         "$CF_API_TOKEN"
_s CF_ACCOUNT_ID        "$CF_ACCOUNT_ID"
_s HF_TOKEN             "$HF_TOKEN"
_s HF_DATASET_REPO      "anas775/DETECT-AI-Dataset"
_s SUPABASE_URL         "$SUPABASE_URL"
_s SUPABASE_SERVICE_KEY "$SUPABASE_SERVICE_KEY"
_s NEWSAPI_KEY          "$NEWSAPI_KEY"
_s YOUTUBE_API_KEY      "$YOUTUBE_API_KEY"
_s UNSPLASH_ACCESS_KEY  "$UNSPLASH_ACCESS_KEY"
_s PEXELS_API_KEY       "$PEXELS_API_KEY"
_s PIXABAY_API_KEY      "$PIXABAY_API_KEY"
_s FLICKR_API_KEY       "$FLICKR_API_KEY"
_s REDDIT_CLIENT_ID     "$REDDIT_CLIENT_ID"
_s REDDIT_CLIENT_SECRET "$REDDIT_CLIENT_SECRET"
_s UPSTASH_REDIS_URL    "$UPSTASH_REDIS_URL"
_s UPSTASH_REDIS_TOKEN  "$UPSTASH_REDIS_TOKEN"

# 4. Trigger HF init
echo "🤗 [4/4] Triggering HuggingFace repo init..."
sleep 3
gh workflow run init-hf-repo.yml --repo "$R" 2>/dev/null \
  && echo "   ✅ HF init triggered" \
  || echo "   ℹ️  Trigger manually: Actions → Initialize HuggingFace Dataset Repo → Run"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ✅  DETECT-AI — LIVE IN CLOUD                                  ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  GitHub:  https://github.com/${GITHUB_USER}/Ai-detection-pipline"
echo "║  Actions: https://github.com/${GITHUB_USER}/Ai-detection-pipline/actions"
echo "║  Dataset: https://huggingface.co/datasets/anas775/DETECT-AI-Dataset"
echo "║  DB:      https://supabase.com/dashboard/project/igwimowqtbgatqvdrqjf"
echo "╚══════════════════════════════════════════════════════════════════╝"
