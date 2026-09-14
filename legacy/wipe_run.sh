#!/usr/bin/env bash
set -e

RUNS_DIR="runs"

# run from repo root
cd "$(dirname "$0")"

# pick newest directory (not a symlink)
LAST_RUN=$(ls -dt "$RUNS_DIR"/* 2>/dev/null | head -n 1)

if [[ -z "$LAST_RUN" ]]; then
  echo "No runs found."
  exit 1
fi

# safety: refuse if it's a symlink
if [[ -L "$LAST_RUN" ]]; then
  echo "Refusing to delete symlink: $LAST_RUN"
  exit 1
fi

echo "Deleting: $LAST_RUN"
rm -rf "$LAST_RUN"
