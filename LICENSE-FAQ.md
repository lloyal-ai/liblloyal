# Licensing FAQ

> Canonical version at https://docs.lloyal.ai/licensing/faq.
> This file is a synced copy. Edit the canonical source and re-run
> `scripts/sync-license-faq.sh --docs-dir ../hdk-docs --native-dir ../lloyal.node --kernel-dir ../liblloyal`
> in hdk to update all copies.

**You can use Lloyal to build and sell your own products.**

You may use, copy, modify and distribute the covered software for any purpose
other than the competing developer-platform uses described below, subject to
the license's ordinary conditions.

Permission is not limited to particular industries, application types,
deployment environments or business models. You do not need to find your
project on a list of approved use cases.

## Can I ship a commercial product built with Lloyal?

**Yes.** You can build, distribute, sell, license and host applications using
Lloyal.

The [Developer Grant](https://github.com/lloyal-ai/hdk/blob/8678ebcd02c425b9e569d97e58063e2af143ffb7/GRANT.md)
expressly protects this permission, even when your application competes directly
with an application or service offered by Lloyal Labs.

Your application may let its users create workflows, compose agents and install
extensions. Those features do not, by themselves, make it a competing developer
platform.

## What is restricted?

During each version's FSL period, the competing-use restriction applies to
commercially making the covered software available as:

- A substitute framework, runtime, SDK or library whose primary purpose is
  enabling third-party developers to build and ship their own agentic AI
  applications.
- A managed or hosted service providing the covered runtime's functionality to
  third-party developers.
- A general distribution channel, registry or marketplace for Lloyal abilities,
  other than Lloyal's canonical channel, offered to third-party Harness developers.

Hosting your own application for its users is permitted. So are private
distribution within your organization, distribution to your customers as part
of your application, and your application's own plugin system.

These boundaries are defined in Section 4 of the Developer Grant. The
grant only adds permissions; it does not take away rights provided by the
license.

## Which packages does the grant cover?

The Developer Grant covers the FSL runtime, application packages and abilities:

| Layer | Covered packages |
|---|---|
| Native inference | `liblloyal`, `@lloyal-labs/lloyal.node` |
| Runtime | `@lloyal-labs/sdk`, `@lloyal-labs/lloyal-agents`, `@lloyal-labs/rig` |
| Application integration | `@lloyal-labs/binding`, `@lloyal-labs/host`, `@lloyal-labs/relay`, `@lloyal-labs/desktop`, `@lloyal-labs/ui`, `@lloyal-labs/dev-tools` |
| Media and abilities | `@lloyal-labs/media`, `@lloyal-labs/web-ability`, `@lloyal-labs/corpus-ability`, `@lloyal-labs/documents-ability`, `@lloyal-labs/wikipedia-ability` |

Previously named `@lloyal-labs/web-app`, `@lloyal-labs/corpus-app` and
`@lloyal-labs/wikipedia-app` packages remain covered. The grant also covers other
Lloyal FSL packages that identify it in their repository. See Section 1 of the
[Developer Grant](https://github.com/lloyal-ai/hdk/blob/8678ebcd02c425b9e569d97e58063e2af143ffb7/GRANT.md) for the governing coverage.

The `lloyal-ai` CLI is MIT and `@lloyal-labs/channel-verify` is Apache 2.0.
Neither needs this grant. Model weights retain their respective licenses.

## Can the grant change for a version I already use?

**No.** The grant is irrevocable for every version published while it is in
effect. Revisions apply only to future versions; they do not remove permissions
from versions already published.

## What obligations remain?

Keep the required license and copyright notices when redistributing the covered
software. The license's patent, trademark and other conditions continue to
apply.

The runtime is provided under **FSL-1.1-Apache-2.0**, supplemented by the
Developer Grant. Each version becomes available under Apache 2.0 two years
after that version is first made available.

This FAQ explains the terms. The
[LICENSE](https://github.com/lloyal-ai/liblloyal/blob/4858217cd33a1f55266315160cc490fd6008d725/LICENSE) and
[Developer Grant](https://github.com/lloyal-ai/hdk/blob/8678ebcd02c425b9e569d97e58063e2af143ffb7/GRANT.md) contain the
governing text.
