#!/usr/bin/env bash
# Publish GGUF files (and their model card) to the Hugging Face Hub (step 6). PRIVATE by default.
#
#   hf auth login                                   # once: paste a token with WRITE access
#                                                   #   (https://huggingface.co/settings/tokens)
#   .venv/bin/python -I scripts/model_card.py --dir gguf/q35-reap-k128   # writes DIR/README.md, read it first
#   scripts/upload_gguf.sh YOUR_NAME/Qwen3.5-19B-A3B-REAP-GGUF gguf/q35-reap-k128
#   PUBLIC=1 scripts/upload_gguf.sh ...             # only when you really want it public
#
# Uploads README.md, the quantized *.gguf files and imatrix.gguf from DIR; never the BF16 file (too big, and
# anyone can rebuild it). The repo is created if it does not exist. Large files go through Hugging Face's
# Xet storage, so a re-upload of a changed file only sends the changed chunks.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$1; DIR=$2
hf auth whoami > /dev/null 2>&1 || { echo "Not logged in: run  hf auth login  (token with write access)"; exit 1; }
[ -f "$DIR/README.md" ] || { echo "No $DIR/README.md: write a model card first (scripts/model_card.py)"; exit 1; }
VIS=--private; [ "${PUBLIC:-0}" = 1 ] && VIS=--no-private
ls -la "$DIR"
read -r -p "Upload these files to $REPO (${VIS#--})? [y/N] " OK
[ "$OK" = y ] || exit 0
hf upload "$REPO" "$DIR" . $VIS --include "README.md" --include "*.gguf" --exclude "*-BF16.gguf" \
    --commit-message "Upload GGUF files and model card"
echo "https://huggingface.co/$REPO"
