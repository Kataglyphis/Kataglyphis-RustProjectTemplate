#!/usr/bin/env bash
# Ensure a per-version ref exists in an ostree repo for a flatpak export.
# Usage: ensure-ref.sh <repo-path> <app-id> <arch> <version>
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <repo-path> <app-id> <arch> <version>" >&2
  exit 2
fi
REPO=$1
APP=$2
ARCH=$3
VER=$4

REF_DIR="$REPO/refs/heads/app/$APP/$ARCH"
MASTER_REF_FILE="$REF_DIR/master"
TARGET_REF_FILE="$REF_DIR/$VER"

if command -v ostree >/dev/null 2>&1; then
  # If ostree is present, read the commit for master and create the new ref
  if ! ostree refs --repo="$REPO" | rg -q "refs/heads/app/$APP/$ARCH/master" 2>/dev/null; then
    # Fallback: if ostree can't see the ref, try to read file
    if [ -f "$MASTER_REF_FILE" ]; then
      COMMIT=$(cat "$MASTER_REF_FILE")
    else
      echo "Master ref not found in repo: $MASTER_REF_FILE" >&2
      exit 1
    fi
  else
    COMMIT=$(ostree rev-parse --repo="$REPO" refs/heads/app/$APP/$ARCH/master)
  fi
else
  if [ -f "$MASTER_REF_FILE" ]; then
    COMMIT=$(cat "$MASTER_REF_FILE")
  else
    echo "ostree not available and master ref file not found: $MASTER_REF_FILE" >&2
    exit 1
  fi
fi

# Create target ref file with the same commit
mkdir -p "$REF_DIR"
echo "$COMMIT" > "$TARGET_REF_FILE"
echo "Created ref: refs/heads/app/$APP/$ARCH/$VER -> $COMMIT"
