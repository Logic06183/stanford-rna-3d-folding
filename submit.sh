#!/usr/bin/env zsh
# ─────────────────────────────────────────────────────────────────────────────
# submit.sh — Push notebook to Kaggle, wait for it to run, submit the output
#
# Usage:
#   ./submit.sh "My submission message"
#   ./submit.sh                           # uses default message
#
# What it does:
#   1. Copies the latest baseline_kaggle.ipynb into kernel/
#   2. Pushes the kernel to Kaggle (creates or updates it)
#   3. Polls every 30s until the kernel finishes (complete/error)
#   4. Submits the kernel output to the competition
# ─────────────────────────────────────────────────────────────────────────────

set -e

KAGGLE=/opt/anaconda3/bin/kaggle
COMPETITION=stanford-rna-3d-folding-2
KERNEL_SLUG=craigparker/rna-3d-baseline-tbm
KERNEL_DIR="$(dirname "$0")/kernel"
NOTEBOOK_SRC="$(dirname "$0")/notebooks/baseline_kaggle.ipynb"
MESSAGE="${1:-Baseline TBM auto-submit}"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()   { echo "${GREEN}[submit]${NC} $*"; }
warn()  { echo "${YELLOW}[submit]${NC} $*"; }
error() { echo "${RED}[submit]${NC} $*"; exit 1; }

# ── Step 1: Sync latest notebook ─────────────────────────────────────────────
log "Syncing notebook to kernel directory..."
cp "$NOTEBOOK_SRC" "$KERNEL_DIR/baseline_kaggle.ipynb"
log "  kernel/ now contains:"
ls -lh "$KERNEL_DIR"

# ── Step 2: Push kernel to Kaggle ─────────────────────────────────────────────
log "Pushing kernel to Kaggle..."
KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN $KAGGLE kernels push -p "$KERNEL_DIR"
log "  Kernel pushed: $KERNEL_SLUG"
log "  Kaggle will now run it on their servers (~5–15 min for CPU baseline)"

# ── Step 3: Poll until complete ───────────────────────────────────────────────
log "Waiting for kernel to finish (polling every 30s)..."
DOTS=0
while true; do
    STATUS=$(KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN $KAGGLE kernels status "$KERNEL_SLUG" 2>&1)
    # Status is one of: running, complete, error, cancelAcknowledged, queued
    if echo "$STATUS" | grep -q "complete"; then
        echo ""
        log "Kernel finished successfully!"
        break
    elif echo "$STATUS" | grep -q "error"; then
        echo ""
        error "Kernel failed with error. Check: https://www.kaggle.com/code/$KERNEL_SLUG"
    elif echo "$STATUS" | grep -q "running\|queued"; then
        printf "."
        DOTS=$((DOTS+1))
        if [ $((DOTS % 20)) -eq 0 ]; then
            echo ""
            warn "  Still running... (${DOTS} checks)"
        fi
        sleep 30
    else
        echo ""
        warn "Unexpected status: $STATUS"
        sleep 30
    fi
done

# ── Step 4: Submit kernel output ─────────────────────────────────────────────
log "Submitting kernel output to competition..."
KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN $KAGGLE competitions submit \
    "$COMPETITION" \
    --kernel "$KERNEL_SLUG" \
    --file  "submission.csv" \
    --message "$MESSAGE"

log "Submission complete!"
log "View your score at: https://www.kaggle.com/competitions/$COMPETITION/submissions"
