#!/bin/bash

# Generate API documentation using Doxygen + doxygen-awesome-css
# Usage: ./scripts/generate-docs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check dependencies
if ! command -v doxygen &> /dev/null; then
    echo "Error: doxygen not found"
    echo "Install with: brew install doxygen"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found"
    echo "Install with: brew install cmake"
    exit 1
fi

echo "Generating API documentation..."

# Clean previous output
rm -rf docs/api/html

# CMake fetches doxygen-awesome-css, configures Doxyfile, runs doxygen
cmake -S docs -B docs/build
cmake --build docs/build --target docs

echo ""
echo "Documentation generated successfully!"
echo "  open docs/api/html/index.html"
echo ""
