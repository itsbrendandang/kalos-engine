# Dataset notice

`dataset_0.csv` is derived from the Kaggle dataset **"Fermentation optimization"** by
Aditya Narayan Konwar (https://www.kaggle.com/datasets/adityanarayankonwar/fermentation-optimization),
which lists an **Apache License 2.0** licence on its dataset page. That licence line was read from a
search-result excerpt of the Kaggle page and has NOT been independently verified here; confirm it (and the
attribution wording) on Kaggle before redistributing this file.

- It is a *hybrid* dataset: the first ~20 source runs are laboratory measurements; the rest are
  ML-synthesized. The `data_type` column marks each row `Real` or `Synthetic`; the `Real` flag itself is unverified (some flagged rows look synthetic, and run id TRICH_15 appears twice with different values). Do not describe the
  full table as real data.
- This copy is a cleaned subset (715 rows), modified from the original; see the source for the
  original files.
- Apache-2.0 notice: this file is provided "AS IS", without warranties of any kind.
