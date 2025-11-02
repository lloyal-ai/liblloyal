# liblloyal Test Dependencies

This directory contains pre-built binaries required for integration testing, organized by llama.cpp version.

## Directory Structure

```
lib/
├── b6870/                    # llama.cpp tag
│   └── llama.xcframework/   # Pre-built XCFramework
└── README.md
```

## Current Version: b6870

**llama.cpp tag:** `b6870`
**Commit:** `338074c3` (October 29, 2025)
**Size:** ~331MB (stored in Git LFS)
**Platforms:** iOS, macOS, tvOS, visionOS (simulator + device)

Built using llama.cpp's `build-xcframework.sh` with the following configuration:
- Metal acceleration enabled
- BLAS (Accelerate framework) enabled
- BF16 support enabled
- Examples/tools/server disabled

### Updating the Framework

To update llama.cpp to a newer version:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Checkout desired tag/commit
git checkout <tag-or-commit>

# Get the tag name
TAG=$(git describe --tags --exact-match 2>/dev/null || git describe --tags)
echo "Building for tag: $TAG"

# Build xcframework (macOS only, requires Xcode)
bash build-xcframework.sh

# Copy to liblloyal with version directory
mkdir -p ../liblloyal/tests/lib/$TAG
cp -r build-apple/llama.xcframework ../liblloyal/tests/lib/$TAG/

# Update CMakeLists.txt to use new version
cd ../liblloyal/tests
# Edit CMakeLists.txt: set(LLAMA_CPP_VERSION "$TAG")

# Commit with Git LFS
git add tests/lib/$TAG
git commit -m "Add llama.cpp xcframework $TAG"
```

**Note:** Keep old versions in the repo until all tests are updated to the new version.

### Why Pre-built?

Building llama.cpp xcframework in CI takes 6+ minutes and requires macOS runners. By committing the pre-built framework:
- ✅ CI runs are much faster (~30 seconds vs 6+ minutes)
- ✅ Consistent build across all contributors
- ✅ Integration tests work immediately after checkout
- ✅ No dependency on external build systems in CI

### Git LFS

The xcframework binaries are stored with Git LFS. Ensure Git LFS is installed:

```bash
git lfs install
git lfs pull
```

Without LFS, you'll see pointer files instead of actual binaries, and integration tests will fail to link.
