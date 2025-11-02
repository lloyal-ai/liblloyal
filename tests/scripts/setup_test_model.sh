#!/bin/bash
# Download tiny-random-llama test model
# This script downloads the model for local testing and CI

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIXTURES_DIR="$SCRIPT_DIR/../fixtures"
MODEL_PATH="$FIXTURES_DIR/tiny-random-llama.gguf"

echo "=== Setting up test model for behavioral contract tests ==="
echo ""

# Create fixtures directory
mkdir -p "$FIXTURES_DIR"

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Test model already exists at: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
    exit 0
fi

echo "📥 Downloading tiny-random-llama.gguf (~12MB)..."
echo "   Source: tensorblock/tiny-random-LlamaForCausalLM-ONNX-GGUF"
echo "   (Official HuggingFace test model - 4.11M params, Q4_K_M quantization)"
echo ""

# Download from HuggingFace
# Using tensorblock's tiny-random model (smallest viable option, 4.11M params)
# This is a standard test model with random weights - produces gibberish but tests API
# Using Q4_K_M quantization (good balance of size and compatibility)
curl -L "https://huggingface.co/tensorblock/tiny-random-LlamaForCausalLM-ONNX-GGUF/resolve/main/tiny-random-LlamaForCausalLM-ONNX-Q4_K_M.gguf" \
    --progress-bar \
    -o "$MODEL_PATH"

echo ""
echo "✅ Model downloaded successfully!"
echo "   Path: $MODEL_PATH"
ls -lh "$MODEL_PATH"
echo ""
echo "Next steps:"
echo "  1. Build integration tests: cmake -S . -B build_integration -DLLOYAL_BUILD_INTEGRATION_TESTS=ON -DLLAMA_CPP_FRAMEWORK_PATH=<path>"
echo "  2. Run tests: LLAMA_TEST_MODEL=\"$MODEL_PATH\" ./build_integration/IntegrationRunner"
