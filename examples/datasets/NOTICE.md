# Dataset notice

`dataset_0.csv` is derived from the Kaggle dataset **"Fermentation optimization"** by
Aditya Narayan Konwar (https://www.kaggle.com/datasets/adityanarayankonwar/fermentation-optimization),
whose dataset page lists the **Apache License 2.0**. The licence name, creator and the "hybrid" description
were read from Kaggle's own dataset metadata (API `licenseName: "Apache 2.0"`) and match the page. This is the
uploader's self-declared licence: the provenance of the underlying laboratory runs and the uploader's right to
license them were NOT independently verified. Apache-2.0 permits redistribution and commercial use with
attribution and a notice of changes (both given here); keep this file with the CSV.

- It is a *hybrid* dataset: per the listing, the first 20 source runs are laboratory measurements; the rest are
  synthesized with Pandas/NumPy (696 of the 715 rows here are flagged `Synthetic`, 19 `Real`). The `data_type` column marks each row `Real` or `Synthetic`; the `Real` flag itself is unverified (the listing says 20 real runs but 19 rows are flagged; run id TRICH_15 appears twice, both flagged `Real`, with different values; the other 714 ids are unique). Do not describe the
  full table as real data.
- This copy is a cleaned subset (715 rows), modified from the original; see the source for the
  original files.
- Apache-2.0 notice: this file is provided "AS IS", without warranties of any kind.
- Not scientifically validated: the pool-replay result on this table was weak (Spearman 0.048). Use it only as a
  semi-synthetic smoke test; real-data claims need the independently sourced datasets.
