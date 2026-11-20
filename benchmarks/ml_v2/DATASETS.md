# Datasets for ML benchmark v2: provenance, licences, decisions

No data file, and no row-level file derived from internal data, is committed.
`python -m kalos.bench.ml_v2 fetch` downloads the open ones into `KALOS_BENCH_DATA`
(default `~/kalos-bench-data`) and checks their SHA-256 (`datasets.DOWNLOADS`).
`python -m kalos.bench.ml_v2 datasets` lists what is available.

## Used

| Dataset | Rows (recipes) | Inputs | Licence / status | Source | Use |
|---|---|---|---|---|---|
| Amii DBO_rat_myocyte | 209 | 14 + IS indicator | open-access article data, attribution; datasets keep their papers' terms (Data/SOURCES.md); not redistributed | Cosenza et al. 2023, doi:10.1002/elsc.202300005, via github.com/Amii-Applied-AI/amii-cell-ag-tools | pool, emulator, snapshot, backtest (seeded orderings) |
| Amii MOBO_rat_myocyte | 442 (highest fidelity kept) | 26 | same | same | same (held-out only) |
| Amii HeLa_regular / HeLa_timesaving | 308 / 403 | 29 | same | Hashizume et al. 2023, doi:10.1038/s41540-023-00284-7 | pool, emulator, snapshot, backtest (real `Round` order) |
| Amii Tcell_expansion / TF1_expansion | 2124 / 2340 | 14 / 15 | same | Kim & Audet 2019, doi:10.1038/s42003-019-0296-7 | pool, emulator, snapshot, backtest (real `gen #` order) |
| Zenodo 18270843, inclusion-body refolding | 22 (DoE1, included rows) + 25 (BO) | 5 (pH, urea, dilution, GSSG, DTT) | CC-BY-4.0 | doi:10.5281/zenodo.18270843 (`Results_Tabular.zip`) | pool/snapshot/emulator, held-out only |
| Olympus `hplc` | 1386 | 6 continuous | MIT (Olympus repository, commit 440b6b5) | Roch et al., ChemOS (via aspuru-guzik-group/olympus) | out-of-domain process DoE |
| Olympus `lnp3` | 768 | 4 discrete numeric + 1 categorical | MIT (same) | Olympus dataset_lnp3 | out-of-domain formulation, mixed |
| repo fixtures synthetic_bioprocess / synthetic_scaleup | 160 / 90 | 10 + medium / 5 incl. scale | repo (Apache-2.0), GENERATED | examples/ | fixture pool, scale LOSO |
| Kaggle fermentation dataset_0 | 715 | 6 | unclear (uploader says Apache-2.0; rights to the runs unverified); 696 of 715 rows flagged Synthetic | examples/datasets/README.md | **information only** |

Amii: `~/kalos-data/amii` and `~/AI/workspace/data/amii/active-learning-for-cell-media/Data`
hold byte-identical copies of all six files (checked with `cmp`). Set `KALOS_AMII_DATA`.

## Internal (rights unconfirmed): local only, report-only, labelled

| Dataset | What it is | Use |
|---|---|---|
| `leadgene_cho_dynamics.csv` (Voyager repo, LeadGene/Cytena) | real multi-scale CHO clone data, 504 clones, transposed | internal clone-ranking and scale-transfer tasks |
| `anagram_compiled.csv` S26JAN06 bench rows | real 3 mL Pichia lipase DoE, 32 recipes x 3 (customer project) | backtest (pseudo-time), per REAL_BACKTEST_PLAN.md |
| `anagram_compiled.csv` large-scale rows | 5 single fed-batch runs, 3 L to 35 m3 | Pichia scale task (refusal expected) |
| Combined S26APR07 + S26APR29 TSV | real lipase media DoE, two experiments; zeros are censored non-detects; APR07 plate 2 harvested at 8 h | backtest (the one real chronological cut) |

## Considered and not added (reason)

| Candidate | Licence | Reason |
|---|---|---|
| Bioqore `mab_scaleup_dataset.csv` (+ sensor/HPLC) | n/a | GENERATED (its README: "synthetic longitudinal sensor data, kinetics modeled"); usable only as a labelled synthetic task, never as evidence; not added in v2.0 |
| Zenodo 7674351, lactic/succinic acid fermentation (Olszewska-Widdrat 2023) | CC-BY-4.0 | time courses, not a recipe -> outcome table |
| Zenodo 7246462, Cosenza 2022 multi-information-source BO | CC-BY-4.0 | record holds only the PDF; the data are the Amii MOBO/DBO sets already used |
| Zenodo 23343771, cultivated-meat operational-mode screening | MIT | computational (model-generated) screening, 400 MB; not measured |
| Figshare 30165037, tobacco BY-2 medium Bayesian design (Frontiers 2025) | CC-BY-4.0 | data only as a PDF supplement; eligible, needs manual extraction (candidate for an amendment) |
| Figshare 32297136 / 29919245, folate and laccase production optimization tables | CC-BY-4.0 | DOCX tables of small RSM designs; eligible, needs extraction |
| Zenodo 17177597, augmented microalgae datasets | CC-BY-4.0 | mostly augmented (generated) rows |
| Olympus chemistry sets (suzuki, snar, buchwald, ...), Summit emulators | MIT | chemistry, not process/bioprocess; `hplc` and `lnp3` kept as out-of-domain checks |
| Bioqore BayOp augmentation (copula) results | n/a | an RF-oracle judgement with no lab confirmation; not a dataset or an arm |
