### Changed - The assistant can say what a paper found, as the paper's finding

- A sentence that cites only paper excerpts and attributes the finding ("Example A reports ...", "according to the paper ...") may now use a claim word such as *increased* when the cited excerpt uses that same word. It still may not state a number from the paper, strengthen the paper's wording, or apply the finding to the campaign: a paper-only sentence naming *your*, *this campaign* or a column is removed (`literature_applied`). Sentences about the lab's data, and sentences mixing data and papers, keep the strict rules.
- Before this, almost every paper finding was removed because papers describe effects. No change without `KALOS_ASSIST_LITERATURE_DIR`; no ML output changes.
