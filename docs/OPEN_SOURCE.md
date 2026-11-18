# Open-source models, libraries and data for Kalos

What Kalos uses today, and open-source work a scientist could bring into a campaign. Licenses were read from each project's repository or model card on 2026-11-18; check them again before bundling anything.

## ESM protein models

There is no model called "ESM 5". The ESM line is ESM-1b, ESM-2 and ESMFold (Meta, 2019-2022), then ESM3, ESMC and ESMFold2 (EvolutionaryScale, now Biohub). Code: [Biohub/esm](https://github.com/Biohub/esm) (MIT); the archived predecessor is [facebookresearch/esm](https://github.com/facebookresearch/esm).

| Model | What it does | License | In Kalos |
| --- | --- | --- | --- |
| [ESMC 300M](https://huggingface.co/biohub/ESMC-300M), [600M](https://huggingface.co/biohub/ESMC-600M), [6B](https://huggingface.co/biohub/ESMC-6B) | Protein language model: a sequence in, one embedding per residue out | MIT, with third-party notices | `ESMCEmbedder` (300M by default) |
| [ESM-2](https://huggingface.co/facebook/esm2_t6_8M_UR50D) | Earlier protein language model, 8M to 15B parameters | MIT | `ESM2Embedder` (8M by default) |
| [ESMFold2](https://huggingface.co/biohub/ESMFold2) | Structure prediction from a single sequence, built on ESMC 6B | MIT | Not used |
| [ESM3 open small](https://huggingface.co/biohub/esm3-sm-open-v1) | Generative model over sequence, structure and function | MIT | Not used |
| [ESM Atlas](https://esmatlas.com/) | About 600 million predicted metagenomic protein structures | Check the site's terms | Not used |

`ESMCEmbedder` and `ESM2Embedder` turn the expressed product's sequence into features, so one surrogate can learn across products (`kalos/features/protein.py`, `examples/run_protein_embed.py`).

## Libraries Kalos is built on

| Project | What it does | License |
| --- | --- | --- |
| [BoTorch](https://github.com/meta-pytorch/botorch) | Bayesian optimization; Kalos's acquisition functions and batch proposals | MIT |
| [GPyTorch](https://github.com/cornellius-gp/gpytorch) | Gaussian-process models; Kalos's surrogate | MIT |
| [PyTorch](https://github.com/pytorch/pytorch) | Tensors and autodiff under both | BSD-3-Clause |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Loads ESMC and ESM-2 | Apache-2.0 |
| [Ax](https://github.com/meta/Ax), [Optuna](https://github.com/optuna/optuna) | Optimizers Kalos is benchmarked against (`docs/BENCHMARK.md`) | MIT |

## Other open-source tools worth considering

None of these are integrated. Each would need its own decision.

| Project | What it would add | License |
| --- | --- | --- |
| [BayBE](https://github.com/emdgroup/baybe) | Bayesian design of experiments for chemistry and bioprocess, with chemical encodings | Apache-2.0 |
| [pyFOOMB](https://github.com/MicroPhen/pyFOOMB) | Mechanistic bioprocess models (ODEs) to fit alongside the GP | MIT |
| [COBRApy](https://github.com/opencobra/cobrapy) | Genome-scale metabolic models for strain and media design | GPL-2.0 (copyleft; keep it a separate process) |
| [AlphaFold](https://github.com/google-deepmind/alphafold) | Protein structure prediction | Apache-2.0 code; weights under separate terms |
| [AlphaFold 3](https://github.com/google-deepmind/alphafold3) | Complexes of proteins, nucleic acids and ligands | Apache-2.0 code; weights on request under non-commercial terms |
| [ColabFold](https://github.com/sokrypton/ColabFold) | Faster AlphaFold with MMseqs2 search | MIT |
| [Boltz](https://github.com/jwohlwend/boltz) | Open structure and binding-affinity prediction | MIT |
| [Chai-1](https://github.com/chaidiscovery/chai-lab) | Structure prediction for biomolecular complexes | Apache-2.0 |
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Designs sequences for a given backbone | MIT |
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | Designs new protein backbones | BSD |

## Public data sources

The [agent-test registry](https://github.com/itsbrendandang/agent-test/blob/main/datasets/scripts/seeds/registry.json) lists 98 public sources with access notes. The ones closest to Kalos's work are [UniProt](https://www.uniprot.org/) (sequences), [PDB](https://www.rcsb.org/), [AlphaFold DB](https://alphafold.ebi.ac.uk/) and ESM Atlas (structures), and [ChEMBL](https://www.ebi.ac.uk/chembl/) and [PubChem](https://pubchem.ncbi.nlm.nih.gov/) (compounds). Two bioprocess sources it does not list are [BRENDA](https://www.brenda-enzymes.org/) (enzyme kinetics; CC BY 4.0) and [BiGG Models](http://bigg.ucsd.edu/) (curated metabolic models for COBRApy).
