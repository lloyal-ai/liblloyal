#!/usr/bin/env bash
# Regenerates fixtures/embedding/embeddinggemma2-reference.json: the vectors
# llama.cpp's own server gives EmbeddingGemma 2 for the embedding fixtures, the
# reference the native embedding input must reproduce. Run offline, by hand;
# nothing in a test or in CI builds or calls the server.
#
#   LLAMA_TEST_MODEL=... LLAMA_MMPROJ_MODEL=... tests/scripts/embedding_reference.sh [build-dir]
#
# The model is embeddinggemma-2-Q8_0.gguf and the projector its mmproj-BF16.gguf.
# The build dir is a llama.cpp build with the server and MTMD_VIDEO=ON (created
# if absent); liblloyal's own builds keep both off.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
llama="${LLAMA_CPP_DIR:-$root/../llama.cpp}"
build="${1:-$root/tests/build_embedding_reference}"
fixtures="$root/tests/fixtures"
port="${LLAMA_REFERENCE_PORT:-8791}"
: "${LLAMA_TEST_MODEL:?set LLAMA_TEST_MODEL to embeddinggemma-2-Q8_0.gguf}"
: "${LLAMA_MMPROJ_MODEL:?set LLAMA_MMPROJ_MODEL to its mmproj-BF16.gguf}"

# Configured every run, so a reused build dir cannot keep the server or video off; CMake keeps what is cached.
cmake -S "$llama" -B "$build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DMTMD_VIDEO=ON
cmake --build "$build" --target llama-server -j

# One slot, the whole input in one ubatch, on the CPU as CI runs the embedding cases.
"$build/bin/llama-server" -m "$LLAMA_TEST_MODEL" --mmproj "$LLAMA_MMPROJ_MODEL" --embeddings \
  --pooling mean -c 8192 -b 8192 -ub 8192 -ngl 0 --no-mmproj-offload -np 1 --port "$port" &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT

until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
  kill -0 "$server" 2>/dev/null || { echo "llama-server exited before it was ready" >&2; exit 1; }
  sleep 1
done

# Written aside and moved into place only on success, so a failed run keeps the last good reference.
out="$fixtures/embedding/embeddinggemma2-reference.json"
python3 -I "$here/embedding_reference.py" "http://127.0.0.1:$port" "$fixtures" "$out.tmp"
mv "$out.tmp" "$out"
echo "wrote $out"
