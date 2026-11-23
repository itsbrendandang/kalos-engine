# Agent skills

Canonical skills live in `.agents/skills/<name>/SKILL.md`.
Byte-identical copies live in `.claude/skills/<name>/SKILL.md`.
Edit the canonical file, update its mirror, and refresh `skills-lock.json`.
The lock uses agent-test's version 2 fields. Each computedHash is the SHA-256
of the canonical SKILL.md bytes, including its final newline.
For skills authored here, source points to this repo rather than an external
skill package. Attribution and licensing are recorded per entry.

Layout reference: https://github.com/itsbrendandang/agent-test
(paths `.agents/skills/`, `.claude/skills/` and `skills-lock.json`).
The review skill is original Kalos documentation, licensed under the root LICENSE.
No upstream skill text was copied.

## Biomni license check

The agent-test biomni-lab folder contains copied Biomni Hub overviews but has
no LICENSE file. Its root MIT license excludes copied third-party files.
Its skills-lock.json has no biomni-lab entry.
That does not establish redistribution rights for the overviews.

The Hub at https://biomni.phylo.bio/skills did not state a redistribution license
in the fetched page. Its /terms URL returned 404.
The operator's terms at https://phylo.bio/terms grant access for personal or
internal business use, not a clear grant to republish the skill text.
The skill guide at https://docs.biomni.phylo.bio/features/skills-and-capabilities
describes copying inside Biomni and sharing skills you own.
It does not establish a license to publish these copied overviews in this repo.

Biomni content is not installed. Do not add it without a clear license or
permission covering public redistribution. The open-source Biomni project's
license is not evidence that these hosted skill overviews have the same terms.
