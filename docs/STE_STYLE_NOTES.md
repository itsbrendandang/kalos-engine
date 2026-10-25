# README style notes

The top-level `README.md` is written in simplified technical English, inspired by ASD-STE100.
This document lists the rules that were applied. It does not claim compliance with ASD-STE100. The official STE dictionary is copyrighted, and this text was not checked against it.

## Rules applied

1. **Sentence length.** Procedure steps: 20 words or fewer. Descriptions: 25 words or fewer.
2. **One idea per sentence.** Long sentences are split. Lists replace long lists of clauses.
3. **Active voice and present tense.** Use "Kalos returns" and "The engine reads", not "is returned" or "will be read".
4. **One term for one thing.** Use "surrogate model", "acquisition function", "optimization", "repository", "tenant", and "license". Do not mix in variants such as "surrogate" alone, "repo", "organisation", or "licence".
5. **Define each technical term once, at first use.** Examples: Bayesian optimization (BO), surrogate model, acquisition function, uncertainty, Gaussian process (GP), tenant, backtest, in-flight, Latin hypercube sampling (LHS), split-conformal band, fail closed, pool replay, Holm correction.
6. **No idioms.** Phrases such as "the gap closes" or "the best of both" were rewritten in plain words.
7. **Warnings and cautions come before the step they apply to.** Examples: the API caution before the start-up step, and the private-network warning before the deploy steps.
8. **Procedures use numbered steps.** Each step has one action. Commands stay in code blocks.
9. **Tables are kept.** Table cells use short phrases, and the column layout is unchanged.
10. **Approved-type vocabulary where possible.** Common words are used (for example "use", "show", "start"). Product, library, and code names are not changed.

## Exceptions (kept as they were)

- Code blocks, commands, file paths, URLs, numbers, PR numbers, identifiers, the status diagram, and the benchmark tables are kept exactly. The diagram keeps its original labels.
- Terms that are not in the rules above, such as product, library, and code names, are not changed.

## How the rewrite was checked

Facts were compared with the previous README (`origin/main` before the rewrite):

- Links, bare URLs, code spans, code-block lines, PR references, and numbers were compared. Every item from the old README is present in the new README.
- Sentence length (25 words, 20 for steps) and the idiom list were checked by hand.
