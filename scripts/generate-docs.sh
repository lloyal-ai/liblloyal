#!/bin/bash

# Generate API documentation using Doxygen
# Usage: ./scripts/generate-docs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check if doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "Error: doxygen not found"
    echo "Install with: brew install doxygen"
    exit 1
fi

echo "Generating API documentation..."

# Clean previous output
rm -rf docs/api/html

# Generate documentation
doxygen Doxyfile

echo ""
echo "✅ Documentation generated successfully!"
echo ""
echo "View documentation:"
echo "  - HTML: open docs/api/html/index.html"
echo "  - Or run: open docs/api/html/index.html"
echo ""
