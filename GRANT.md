# Lloyal Harness Builder Grant

**Version 1.0 — Effective 2026-06-12.** Lloyal may publish revised versions for
future releases of the Software; under Section 5, no revision affects any
version already published.

This Grant accompanies the Functional Source License, Version 1.1, with Apache 2.0
Future License (**FSL-1.1-Apache-2.0**), under which Lloyal Labs Pty Ltd
(**"Lloyal," "we," "us"**) makes the Covered Packages available. It applies to each
version of the Covered Packages published while this Grant is in effect.

This Grant only **adds** permissions and assurances. It does not remove,
narrow, or condition any right granted by the License. If this Grant and the
License can be read differently, the reading more permissive to you controls.

## 1. Covered Packages

This Grant covers every version of the following software published by Lloyal
while this Grant is in effect, including all transitive inclusion of one Covered
Package within another:

- `liblloyal` (C++ inference kernel)
- `@lloyal-labs/lloyal.node` (native bindings)
- `@lloyal-labs/sdk`
- `@lloyal-labs/lloyal-agents`
- `@lloyal-labs/rig`
- `@lloyal-labs/web-app`, `@lloyal-labs/corpus-app`, `@lloyal-labs/wikipedia-app`
- any other package Lloyal publishes under FSL-1.1-Apache-2.0 that identifies
  this Grant in its repository

(together, the **"Software"**, matching the License's defined term for each
package). Packages Lloyal publishes under Apache 2.0 (for example, the
`harness.dev` CLI) carry no use restrictions and need no grant.

## 2. Definitions

A **"Harness"** is an application that embeds or uses the Software in order to
provide the application's own functionality to its users — for example a
desktop application, mobile application, CLI tool, server application, embedded
or on-device system, or a hosted product whose backend uses the Software to
serve that product's own end users.

An **"App"** is a capability bundle conforming to the HDK App protocol (a
manifest, Source, Tools, and/or skill template validated by the App protocol),
whether distributed through the canonical channel, distributed privately, or
used internally.

## 3. The Grant

For each version of the Software this Grant covers, we irrevocably grant you
the following, on no conditions beyond those already in the License:

**3.1 Harnesses and Apps are always a Permitted Purpose.** Building,
distributing, selling, licensing, hosting, and otherwise commercializing a
Harness or an App is a Permitted Purpose under the License and is **never a
Competing Use** — regardless of any overlap in category, functionality,
market, audience, or business model with any product or service offered by
Lloyal, now or in the future, including without limitation **reasoning.run**,
Lloyal's first-party Apps, and any vertical or industry deployment by Lloyal.

For clarity: a commercial deep-research product, a medical-practice product, a
product in any category Lloyal occupies or later enters — all are Permitted
Purposes when built as a Harness or App.

**3.2 Internal and private distribution is always a Permitted Purpose.**
Operating a private registry, mirror, or distribution mechanism to deploy
Harnesses and Apps **within your own organization or to your own customers as
part of your Harness** is a Permitted Purpose and is not an "alternative App
distribution channel" or other Competing Use.

**3.3 Plugin systems of a Harness are out of scope.** A Harness's own
extension or plugin mechanism, in the Harness's own format, is part of that
Harness. Clause 4(c) below applies only to the distribution of bundles
conforming to the HDK App protocol to third-party Harness developers in
general.

## 4. Covenant on the Meaning of Competing Use

We irrevocably covenant not to assert that any use of the Software is a
"Competing Use" under the License unless it is one of the following:

**(a)** making the Software available to others as a software development
framework, runtime, SDK, or library **whose primary purpose is enabling
third-party developers to build agentic AI applications** — that is, a
substitute for the Software itself as a developer product, rather than an
application built with it;

**(b)** a managed or hosted service that provides the Software's functionality
**as a service to third-party developers** (for example, hosted HDK runtimes
or HDK-as-a-service). A hosted product that uses the Software to serve that
product's own end users is a Harness under Section 2 and is permitted;

**(c)** a general distribution channel, registry, or marketplace for HDK
Apps, other than the canonical channel operated by Lloyal, offered to
third-party Harness developers. Private and internal distribution under
Section 3.2 and Harness plugin systems under Section 3.3 are not within this
clause.

**Boundary note (4.1):** An application whose end users visually or
programmatically compose agents *within that application* is a Harness. It
becomes a framework under 4(a) only if its primary purpose is to expose the
Software's developer APIs so that third parties can build and ship **their
own separate applications** with it.

## 5. Irrevocability and Survival

This Grant is irrevocable for every version of the Software published while it
is in effect. We may amend this Grant for future versions only; no amendment
affects any version already published. This Grant survives, and is in any case
subsumed by, each version's conversion to Apache 2.0 under the License's
Future License terms.

## 6. No Other Changes

The License text is unmodified and governs. This Grant does not modify the
License's trademark, patent, redistribution, or disclaimer provisions, except
to the extent it adds the permissions and covenants above.

---

*Questions about whether your use case is covered: see
[LICENSE-FAQ.md](./LICENSE-FAQ.md) or contact legal@lloyal.ai.*
