# Contributing to liblloyal

Thank you for your interest in contributing to liblloyal!

## Getting Started

```bash
# Clone with submodules
git clone --recursive https://github.com/lloyal-ai/liblloyal.git
cd liblloyal

# Build and run tests
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## Development Guidelines

### Code Style
- C++20 standard
- Header-only implementations with `inline` specifiers
- Use `lloyal::` namespace for all public APIs
- Document functions with doxygen-style comments

### Testing
- All new features must include tests
- Unit tests go in `tests/` directory
- Use doctest framework
- Integration tests for complex workflows

### Commit Messages
- Use conventional commit format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
- Examples:
  - `feat(sampler): add DRY repetition penalty support`
  - `fix(kv): correct position calculation in clear_and_reseed`
  - `docs(readme): update integration examples`

## Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes with tests
4. Run full test suite
5. Submit PR with clear description

### PR Requirements
- All tests must pass
- Code must compile without warnings
- New features need documentation
- Breaking changes must be clearly marked

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- API design questions
- Integration help
