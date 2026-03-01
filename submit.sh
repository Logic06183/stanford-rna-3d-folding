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
MESSAGE="${1:-Baseline TBM auto-submit}"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()   { echo "${GREEN}[submit]${NC} $*"; }
warn()  { echo "${YELLOW}[submit]${NC} $*"; }
error() { echo "${RED}[submit]${NC} $*"; exit 1; }

# ── Step 1: Push kernel to Kaggle, capture version number ─────────────────────
log "Pushing kernel to Kaggle..."
PUSH_OUTPUT=$(KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN $KAGGLE kernels push -p "$KERNEL_DIR" 2>&1)
echo "$PUSH_OUTPUT"
VERSION=$(echo "$PUSH_OUTPUT" | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' | tail -1)
if [ -z "$VERSION" ]; then
    error "Could not parse version number from push output: $PUSH_OUTPUT"
fi
log "  Kernel v${VERSION} pushed: $KERNEL_SLUG"
log "  Kaggle will now run it on their servers (~15–25 min for CPU baseline)"

# ── Step 2: Poll until complete ───────────────────────────────────────────────
log "Waiting for kernel to finish (polling every 30s)..."
DOTS=0
while true; do
    STATUS=$(KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN $KAGGLE kernels status "$KERNEL_SLUG" 2>&1)
    # Status is one of: running, complete, error, cancelAcknowledged, queued
    if echo "$STATUS" | grep -qi "complete"; then
        echo ""
        log "Kernel finished successfully!"
        break
    elif echo "$STATUS" | grep -qi "error"; then
        echo ""
        error "Kernel failed with error. Check: https://www.kaggle.com/code/$KERNEL_SLUG"
    elif echo "$STATUS" | grep -qi "running\|queued"; then
        printf "."
        DOTS=$((DOTS+1))
        if [ $((DOTS % 20)) -eq 0 ]; then
            echo ""
            warn "  Still running... (${DOTS} checks, ${DOTS} min)"
        fi
        sleep 30
    else
        echo ""
        warn "Unexpected status: $STATUS"
        sleep 30
    fi
done

# ── Step 3: Submit kernel output ─────────────────────────────────────────────
log "Submitting kernel v${VERSION} output to competition..."
KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN $KAGGLE competitions submit \
    "$COMPETITION" \
    --kernel "$KERNEL_SLUG" \
    -v "$VERSION" \
    --file  "submission.csv" \
    --message "$MESSAGE"

log "Submission complete!"
log "View your score at: https://www.kaggle.com/competitions/$COMPETITION/submissions"
