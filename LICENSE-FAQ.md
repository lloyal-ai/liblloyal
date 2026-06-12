# Licensing FAQ

> Canonical version at https://docs.lloyal.ai/licensing/faq.
> This file is a synced copy. Edit the canonical source and re-run
> `scripts/sync-license-faq.sh` in lloyal-sdk to update all copies.


**You can build and sell commercial products using HDK.**

> HDK is free to build products with; it is not free to become the
> replacement HDK platform.

That single sentence is the entire restriction reduced to one line. The rest
of this page is illustration.

## The short version

HDK 3.0 runtime packages — `liblloyal`, `lloyal-node`, and the lloyal-sdk
packages (`agents`, `sdk`, `rig`, `apps/corpus`, `apps/web`) — are
source-available under **FSL-1.1-Apache-2.0** (the Functional Source License,
Apache 2.0 future grant). FSL is a [Fair Source](https://fair.io) license.

Each version converts to Apache 2.0 two years after its release. The
restriction during those two years is narrow: **you cannot offer a competing
HDK runtime, a managed HDK service, or an alternative HDK App distribution
channel.** Everything else — commercial use, redistribution, modification,
sale, embedding in shipped products — is freely permitted.

The `harness.dev` CLI and `hdk-create-app` scaffolder are licensed under
**Apache 2.0** (unrestricted). They're not part of the runtime stack.

## Can I ship a commercial product built on HDK?

**Yes.** This is the question everyone has and the answer is straightforward.
Concretely:

- **Shipping a paid intelligent inbox app to consumers** — permitted ✅
- **Selling an Excel-with-AI desktop app to enterprises** — permitted ✅
- **Embedding HDK in a medical device sold commercially** — permitted ✅
- **A consulting firm building a custom intelligent harness for a Fortune
  500 client, charging $500K for the engagement, deploying on client
  infrastructure** — permitted ✅
- **An indie dev shipping a paid productivity app on the Mac App Store** —
  permitted ✅
- **An OEM shipping HDK inside an infotainment system or industrial device**
  — permitted ✅
- **A startup building an end-user product on top of HDK** — including
  vertical research apps, workflow tools, and agent applications — permitted
  ✅, *as long as the product is not offering HDK itself as a substitute
  runtime, managed HDK service, or competing App distribution channel*.
- **A research lab using HDK in published academic work** — permitted ✅
- **Forking HDK on GitHub to learn, modify, demo, or contribute** — permitted ✅
- **Running HDK internally inside your company for any business use** —
  permitted ✅

If you are building something *with* HDK, you are almost certainly fine.

## What is actually restricted?

The restriction is narrow and specific: **don't become the replacement
platform vendor**. Concretely:

- **AWS / Google / Microsoft launching "Bedrock Managed HDK" or "Vertex HDK"
  as a hosted runtime service** — restricted ❌
- **A competitor publishing "OpenHDK" as a forked harness runtime under a
  different name** — restricted ❌
- **A clean-room reimplementation of the HDK runtime in Python / Rust / Go
  intended as a drop-in replacement** — restricted ❌ (Competing Use doesn't
  require forking source code — a reimplementation that competes is the
  same problem)
- **Launching `apps.competitor.com` as an alternative HDK App distribution
  channel** — restricted ❌
- **Offering "managed HDK hosting" or "HDK-as-a-Service" as a competing
  SaaS** — restricted ❌
- **A hosted orchestration service exposing HDK-compatible APIs as a
  substitute for the HDK runtime** — restricted ❌

Notice the pattern: every restricted scenario is "become the platform
vendor," not "build products with HDK." If your project doesn't compete
directly with the HDK runtime or its distribution channel, FSL doesn't
affect you.

## Why does the LICENSE list four narrow Permitted Purposes then?

You may read the FSL LICENSE and see this section:

> Permitted Purposes specifically include using the Software:
> 1. for your internal use and access;
> 2. for non-commercial education;
> 3. for non-commercial research; and
> 4. in connection with professional services that you provide to a
>    Licensee using the Software in accordance with these Terms and
>    Conditions.

A careful first reading can mistake this list for the *exclusive* set of
permitted uses — leading to the (incorrect) conclusion that commercial
product distribution is prohibited.

It isn't. **The operative definition is broader.** Earlier in the same
section, the license states:

> A "Permitted Purpose" is any purpose other than a Competing Use.

The four enumerated items are **illustrative examples** added because those
specific cases are ones a careful reader might otherwise hesitate about
("is research permitted? is consulting permitted?"). The four items are
additive clarifications, not a closing of the open-ended definition.

Sentry, who authored FSL and uses it on their own software, [confirms this
explicitly in their FAQ](https://fsl.software):

> "You can do anything with FSL software except undermine its producer. You
> can run it for almost all purposes, study it, modify it, and distribute
> your changes…"

If the license felt restrictive on first read, that's a documented
[FSL adoption hazard](https://fair.io) — many developers hit the same wall.
The answer is to read the "any purpose other than a Competing Use" line as
the operative definition and treat the four enumerated items as examples,
not as a closed list.

## Will it become Apache 2.0?

**Yes — automatically, on a per-version schedule.** Each released version of
the runtime stack converts to Apache 2.0 exactly two years after its release
date. The conversion is irrevocable and written into the license text — it's
not a promise from Lloyal Labs, it's a contractual clause.

For example: if `lloyal-sdk @lloyal-labs/lloyal-agents` v3.0.0 is released
on 2026-06-01, that exact version becomes available under Apache 2.0 on
2028-06-01. Any consumer can elect to use that version under Apache 2.0
from that date forward — Lloyal Labs takes no action; the grant is
automatic.

New versions released after v3.0.0 start their own two-year clock from
their own release dates. There is no single global Change Date.

## Is this OSI-approved open source?

**No, and we want to be honest about that.** FSL is not OSI-approved
because the OSI definition of open source (clause 6, "No Discrimination
Against Fields of Endeavor") does not permit restrictions on specific
use cases. FSL restricts Competing Use. That restriction takes it out of
strict OSI compliance.

FSL falls under the [Fair Source](https://fair.io) classification —
source-available licenses that are explicitly developer-friendly:
commercial use permitted, free redistribution, eventual open-source
conversion. Fair Source is a more developer-friendly framing than the
generic "source-available" label, which has been tainted by the
SSPL / Elastic / MongoDB relicensing trauma cycles.

What this means practically:

- **You can read the source.** ✅
- **You can modify it.** ✅
- **You can sell products built with it.** ✅
- **You can redistribute it (with the same FSL terms).** ✅
- **It will be Apache 2.0 in two years.** ✅
- Some enterprise procurement policies that strictly require OSI-approved
  licenses will require an exception for this. We're working on making
  that exception easy to grant.

## Why FSL specifically — why not stay Apache?

HDK 3.0 introduces installable HDK Apps. Every App declares against a
specific App protocol — the bytes-locked intro, catalog format,
tool-selection rule, and boundary marker that the runtime renders into
the spine. Your App's reliability depends on every HDK runtime your users
install agreeing on the same protocol.

Under a permissive license alone, the protocol is forkable. A
well-resourced redistributor could fork the runtime, modify the protocol
surface, and distribute a variant under a different name with captive
distribution. App developers then face a fragmented ecosystem: target one
protocol, target both, or pick the bigger distribution and abandon the
others. The cost of that split is paid by App developers in testing
burden, divergent behavior, and reliability degradation across runtimes.

FSL's two-year Competing Use restriction is shaped to block that
fragmentation specifically. After the conversion, anyone can build
whatever they want — by which time the protocol has had enough time to
stabilize through ecosystem use and the protection is no longer the load-
bearing thing keeping it coherent.

For the longer treatment of this argument, see
[Why FSL](./why-fsl).

We could have used Apache 2.0 and tried to protect only the channel via
terms-of-service. We could have written a custom license. We chose
standard FSL because it's:

- **Off-the-shelf** — no bespoke license review at every adopter
- **Recognizable** — Sentry, PowerSync, and others use it
- **Documented** — the FAQ, definitions, and edge cases have been
  litigated publicly
- **Time-bounded** — the protocolual Apache 2.0 conversion is the answer
  to the "is this just source-available forever?" critique
- **Pre-launch** — relicensing at HDK 3.0 launch is structurally
  different from MongoDB / Elastic / HashiCorp relicensing under an
  existing installed base, which is what causes the backlash cycle

## What about the lloyal stack — what's under FSL and what's not?

| Component | License | Why |
|---|---|---|
| `liblloyal` (C++ engine) | FSL-1.1-Apache-2.0 | Native primitives the runtime is built on |
| `lloyal-node` (N-API binding) | FSL-1.1-Apache-2.0 | The binding that lets Effection drive llama.cpp |
| `@lloyal-labs/lloyal-agents` | FSL-1.1-Apache-2.0 | Runtime framework |
| `@lloyal-labs/lloyal-sdk` | FSL-1.1-Apache-2.0 | Runtime framework |
| `@lloyal-labs/rig` | FSL-1.1-Apache-2.0 | Runtime framework — holds the App protocol |
| `@lloyal-labs/corpus`, `@lloyal-labs/web` | FSL-1.1-Apache-2.0 | Reference Apps shipped in-tree |
| **`@lloyal-labs/harness-cli` (the `harness.dev` CLI)** | **Apache 2.0** | Scaffolder — unrestricted for scaffolding new harnesses and Apps |
| **`hdk-create-app`** (when shipped) | **Apache 2.0** | Scaffolder — same as above |
| `llama.cpp` (vendored dependency) | MIT (unchanged) | External upstream library; we don't relicense their code |

## Can I contribute to HDK?

**Yes.** Contributions are welcome under the same FSL license terms. If you
submit a PR, you're granting Lloyal Labs the right to distribute your
contribution under FSL-1.1-Apache-2.0 (and automatically under Apache 2.0
two years after each release that includes it). The CONTRIBUTING file in
each repo has the details.

## I have a use case that's borderline — who do I ask?

Email <legal@lloyal.ai> (or open a discussion in the repo). The runtime
team will help you confirm whether your use case falls under Permitted
Purpose or Competing Use. We'd rather give you a quick yes than have you
worry about it.

## Further reading

- [FSL official site](https://fsl.software) — Sentry's canonical FSL
  resources and FAQ
- [Fair Source](https://fair.io) — the category FSL belongs to
- [The FSL template, instantiated for each repo](./fsl-template)
- [Why we chose FSL over BSL, Apache, or a custom license](./why-fsl)
