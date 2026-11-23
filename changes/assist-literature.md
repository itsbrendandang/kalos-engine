### Added - Paper excerpts in the assistant

- `KALOS_ASSIST_LITERATURE_DIR`: Ask can quote verbatim paper excerpts from `kalos-literature/1` packs (docs/ASSISTANT.md, "Paper excerpts"). A sentence citing a paper may hold no number except the paper's year (`literature_number`); paper text supports no claim word; a pack that fails its checks is left out whole and listed in the index status (`literature.refused`).
- `/api/assist/ask` sources of kind `literature` carry a `citation`; `/api/assist/index/status` gains `literature{items, refused}`.
- No change when the setting is unset: same index content, same prompts, same answers. No ML output changes.
