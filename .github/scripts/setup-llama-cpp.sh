#!/usr/bin/env bash
#
# Setup llama.cpp for local integration testing
# Reads version from .llama-cpp-version and clones to llama.cpp/
#
# Usage:
#   .github/scripts/setup-llama-cpp.sh
#
# Then build and run integration tests:
#   LLAMA_DIR=llama.cpp .github/scripts/build-llama.sh
#   cmake -B tests/build -S tests -DLLOYAL_BUILD_INTEGRATION_TESTS=ON -DLLAMA_CPP_DIR=llama.cpp
#   cmake --build tests/build
#   LLAMA_TEST_MODEL=path/to/model.gguf tests/build/IntegrationRunner

set -e

# Read version from .llama-cpp-version
if [ ! -f ".llama-cpp-version" ]; then
    echo "Error: .llama-cpp-version file not found"
    exit 1
fi

LLAMA_VERSION=$(grep -v '^#' .llama-cpp-version | head -n1 | tr -d '[:space:]')

if [ -z "$LLAMA_VERSION" ]; then
    echo "Error: Could not read version from .llama-cpp-version"
    exit 1
fi

echo "Setting up llama.cpp version: $LLAMA_VERSION"

# Check if llama.cpp already exists
if [ -d "llama.cpp" ]; then
    echo "llama.cpp directory already exists"
    cd llama.cpp

    # Check if it's the correct version
    CURRENT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")

    if [ "$CURRENT_COMMIT" = "$LLAMA_VERSION" ]; then
        echo "✓ llama.cpp is already at version $LLAMA_VERSION"
        exit 0
    else
        echo "Updating llama.cpp to $LLAMA_VERSION..."
        git fetch --depth 1 origin $LLAMA_VERSION
        git checkout $LLAMA_VERSION
    fi
else
    echo "Cloning llama.cpp..."
    git clone --depth 1 --branch $LLAMA_VERSION https://github.com/ggerganov/llama.cpp
    cd llama.cpp
fi

echo "✓ llama.cpp setup complete at:"
git log -1 --oneline
