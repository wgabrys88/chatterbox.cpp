#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 TEXT OUT.wav [--seed N] [--threads N] [...extra chatterbox args]" >&2
    exit 1
fi

TEXT="$1"
OUT="$2"
shift 2
EXTRA_ARGS="$*"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/tts-cli"
T3_GGUF="$ROOT/models/chatterbox-t3-turbo.gguf"
S3G_GGUF="$ROOT/models/chatterbox-s3gen.gguf"

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not built; run 'cmake --build build --target tts-cli' first" >&2
    exit 1
fi
for f in "$T3_GGUF" "$S3G_GGUF"; do
    [[ -f "$f" ]] || { echo "error: missing $f" >&2; exit 1; }
done

"$BIN" \
    --model       "$T3_GGUF" \
    --s3gen-gguf  "$S3G_GGUF" \
    --text        "$TEXT" \
    --out         "$OUT" \
    ${EXTRA_ARGS}

echo "done: $OUT"
