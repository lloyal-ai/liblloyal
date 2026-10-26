#!/usr/bin/env bash
# Regenerates fixtures/embedding/embeddinggemma2-reference.json: the vectors
# llama.cpp's own server gives EmbeddingGemma 2 for the embedding fixtures (images,
# the clip, a recording, mixed inputs, every task framing, other languages and the
# image budgets), the reference the native embedding input must reproduce. Run offline, by hand;
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

# One server for the projector's default image budget, and one for each budget the reference pins (the
# projector's smallest and largest), since the budget is fixed when a server loads its projector.
budgets=(70 1120)
servers=()
trap 'kill "${servers[@]}" 2>/dev/null || true' EXIT

# One slot, the whole input in one ubatch, on the GPU. The reference records these flags as written here.
flags=(--embeddings --pooling mean -c 8192 -b 8192 -ub 8192 -ngl 99 -np 1)
serve() {
  local at="$1"; shift
  "$build/bin/llama-server" -m "$LLAMA_TEST_MODEL" --mmproj "$LLAMA_MMPROJ_MODEL" "${flags[@]}" --port "$at" "$@" &
  servers+=($!)
}

serve "$port"
budgeted=()
for i in "${!budgets[@]}"; do
  at=$((port + 1 + i))
  serve "$at" --image-min-tokens "${budgets[$i]}" --image-max-tokens "${budgets[$i]}"
  budgeted+=("${budgets[$i]}=http://127.0.0.1:$at")
done

for at in "$port" $(seq $((port + 1)) $((port + ${#budgets[@]}))); do
  until curl -sf "http://127.0.0.1:$at/health" > /dev/null; do
    for server in "${servers[@]}"; do
      kill -0 "$server" 2>/dev/null || { echo "a llama-server exited before it was ready" >&2; exit 1; }
    done
    sleep 1
  done
done

# Written aside and moved into place only on success, so a failed run keeps the last good reference.
out="$fixtures/embedding/embeddinggemma2-reference.json"
python3 -I "$here/embedding_reference.py" "$fixtures" "$out.tmp" "llama-server ${flags[*]}" \
  "http://127.0.0.1:$port" "${budgeted[@]}"
mv "$out.tmp" "$out"
echo "wrote $out"
