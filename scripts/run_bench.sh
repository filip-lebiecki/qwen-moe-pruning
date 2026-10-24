#!/usr/bin/env bash
# Benchmark one GGUF: start llama-server, run 08_bench.py against it, stop the server.
#   scripts/run_bench.sh gguf/q35-code-k48/q35-code-k48-Q4_K_M.gguf code-k48
# --fit on puts everything on the GPU when it fits, and keeps the rest in system RAM when it
# doesn't (the 20.5 GiB unpruned baseline). -np 4: four requests are answered at the same time.
# It also measures generation speed (speed_test.py); SKIP_BENCH=1 measures only the speed.
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:?gguf file}; TAG=${2:?tag}; shift 2
pkill -x llama-server || true
sleep 2
llama.cpp/build/bin/llama-server -m "$M" --fit on -np 4 -c 16384 --port 8089 > "logs/server-bench-$TAG.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -s localhost:8089/health | grep -q ok; do
    sleep 2
    kill -0 $SERVER 2>/dev/null || { tail -5 "logs/server-bench-$TAG.log"; exit 1; }
done
.venv/bin/python -I scripts/speed_test.py --tag "$TAG"
[ "${SKIP_BENCH:-0}" = 1 ] || .venv/bin/python -I scripts/08_bench.py --tag "$TAG" "$@"
