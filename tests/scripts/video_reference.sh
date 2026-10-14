#!/usr/bin/env bash
# Regenerates fixtures/video/video-reference.json: llama.cpp's own video helper
# tokenizing fixtures/video/red-square-moves.mp4, the reference the native video
# input must reproduce. Needs ffmpeg and ffprobe on PATH and a matched
# Qwen3.5-4B decoder/projector pair.
#
#   LLAMA_TEST_MODEL=... LLAMA_MMPROJ_MODEL=... tests/scripts/video_reference.sh [build-dir]
#
# The build dir is a llama.cpp build configured with MTMD_VIDEO=ON (created if
# absent); liblloyal's own builds keep video off.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
llama="${LLAMA_CPP_DIR:-$root/../llama.cpp}"
build="${1:-$root/tests/build_video_reference}"
fixtures="$root/tests/fixtures/video"
: "${LLAMA_TEST_MODEL:?set LLAMA_TEST_MODEL to Qwen3.5-4B-Q4_K_M.gguf}"
: "${LLAMA_MMPROJ_MODEL:?set LLAMA_MMPROJ_MODEL to its mmproj-F16.gguf}"

# Configured every run, so a reused build dir cannot keep MTMD_VIDEO off; CMake keeps what is cached.
cmake -S "$llama" -B "$build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DMTMD_VIDEO=ON
cmake --build "$build" --target mtmd -j

libs="$(dirname "$(find "$build" -name 'libmtmd.*' -print -quit)")"
c++ -std=c++17 -O2 "$here/video_reference.cpp" -o "$build/video_reference" \
  -I"$llama/include" -I"$llama/ggml/include" -I"$llama/tools/mtmd" \
  -L"$libs" -L"$build/bin" -L"$build/src" -L"$build/ggml/src" \
  -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,"$libs" -Wl,-rpath,"$build/bin"

prompt="$(printf '<__media__>\nWhich direction does the red square move?')"
# Written aside and moved into place only on success, so a failed run keeps the last good reference.
"$build/video_reference" "$LLAMA_TEST_MODEL" "$LLAMA_MMPROJ_MODEL" \
  "$fixtures/red-square-moves.mp4" "$prompt" > "$fixtures/video-reference.json.tmp"
mv "$fixtures/video-reference.json.tmp" "$fixtures/video-reference.json"
echo "wrote $fixtures/video-reference.json"
