# Changelog entries

Each PR adds one file here instead of editing `CHANGELOG.md`, so PRs never conflict on
the changelog.

- Name: `<short-slug>.md`, for example `bm25-small-corpus.md`.
- Content: the entry exactly as it should appear in `CHANGELOG.md`, starting with a
  `### Added - ...`, `### Changed - ...`, `### Fixed - ...` or `### Removed - ...`
  heading. Say when numbers change.

At release, `python scripts/changelog.py` moves every entry into the `## Unreleased`
section of `CHANGELOG.md` (newest first) and deletes the files.
