# API Documentation

This directory contains auto-generated API documentation using [Doxygen](https://www.doxygen.nl/).

## Generating Documentation

**Prerequisites:**

```bash
brew install doxygen
```

**Generate:**

```bash
# From liblloyal root
./scripts/generate-docs.sh

# Or manually
doxygen Doxyfile
```

**View:**

```bash
open docs/api/html/index.html
```

## What Gets Generated

- **HTML documentation** - Complete API reference with:
  - Function signatures with parameter descriptions
  - Return values and exceptions
  - Usage examples from header comments
  - Source code browser
  - Namespace/class hierarchy
  - Search functionality

## Source of Truth

The API documentation is generated directly from inline comments in:

```
include/lloyal/*.hpp        # API documentation
docs/mainpage.dox           # Homepage content
```

- **API docs**: Maintained in header files themselves
- **Homepage**: Maintained in `docs/mainpage.dox` (overview, quick start, architecture)

If you find documentation issues, fix them in the source files and regenerate.

## Configuration

The Doxygen configuration is in `Doxyfile` at the repository root. Key settings:

- **Input**: `include/lloyal/` (all headers)
- **Output**: `docs/api/html/`
- **Excluded**: `nlohmann/` (third-party library)
- **Format**: HTML with tree navigation

## CI/CD

The generated HTML is **not** committed to git (see `.gitignore`).

**Automatic Publishing:**

- GitHub Actions workflow (`.github/workflows/docs.yml`) auto-publishes on every commit to main
- Docs are deployed to GitHub Pages: https://lloyal-ai.github.io/liblloyal/
- Triggered by changes to `include/lloyal/**`, `Doxyfile`, or `docs/mainpage.dox`
- See `docs/PUBLISHING.md` for GitHub Pages setup details

## For Contributors

When adding new functions or classes:

1. Document them inline in the header using Doxygen-style comments
2. Follow existing patterns (see `include/lloyal/metrics.hpp` for examples)
3. Regenerate docs to verify: `./scripts/generate-docs.sh`
4. Check `docs/api/html/index.html` for rendering issues

### Documentation Style

```cpp
/**
 * @brief One-line brief description
 *
 * Detailed description here. Can be multiple paragraphs.
 *
 * @param ctx Llama context (must not be null)
 * @param tokens Token array to process
 * @return Number of tokens processed
 * @throws std::runtime_error if ctx is null
 *
 * @example
 *   auto tokens = tokenizer::tokenize(vocab, "Hello");
 *   int count = process(ctx, tokens);
 */
inline int process(llama_context* ctx, const std::vector<llama_token>& tokens);
```

## Troubleshooting

**"doxygen not found"**

```bash
brew install doxygen
```

**"Lots of warnings from nlohmann/json.hpp"**

- This is expected and filtered out
- nlohmann/json is excluded via `EXCLUDE` in Doxyfile

**"Missing documentation for my new function"**

- Ensure you added `/** ... */` comment block above the function
- Check that the header is in `include/lloyal/` (not excluded)
- Regenerate: `./scripts/generate-docs.sh`
