# Security Policy

## Reporting Security Issues

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in liblloyal, please report it privately:

1. **Email**: Send details to security@lloyal.ai
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge your email within 48 hours and provide a detailed response within 7 days.

## Scope

Security issues in liblloyal include:

- **Memory safety**: Buffer overflows, use-after-free, null pointer dereferences
- **API misuse**: Operations that could cause crashes or undefined behavior
- **Resource exhaustion**: Unbounded allocations, infinite loops
- **Integer overflows**: In position calculations, buffer sizing, etc.

## Out of Scope

The following are not considered security issues:

- Bugs in llama.cpp itself (report to llama.cpp project)
- Model-level vulnerabilities (prompt injection, jailbreaks)
- Performance issues without security impact
- Compilation warnings without runtime impact

## Supported Versions

We provide security updates for:

- Current release (latest stable version)
- Previous minor version (for 90 days after new release)

## Disclosure Policy

- We aim to patch critical vulnerabilities within 30 days
- We will coordinate disclosure timing with the reporter
- Credit will be given to reporters (unless anonymity requested)

## Security Best Practices

When using liblloyal:

1. **Validate inputs**: Always check model paths, token arrays, and parameters
2. **Resource limits**: Set appropriate context sizes and batch limits
3. **Error handling**: Handle all exceptions and check return values
4. **Memory monitoring**: Track KV cache size and state memory usage

## Past Security Issues

None reported to date.
