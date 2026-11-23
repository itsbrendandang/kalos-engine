### Fixed - assistant review quota and agent rules

- Reviews now cost at most the per-minute quota, so limits of 1 or 2 allow one review per minute. A quota of 0 still disables rate limiting.
- Agent rules describe critic severities and code-computed verdicts, and direct change notes to changes/.

### Added - assistant review skill

- Add original Kalos review guidance in .agents/skills/ with a .claude/skills/ mirror and a source, hash and attribution lock.
- Do not include Biomni skill text: public redistribution rights were not established. Record the license check in .agents/skills/README.md.
