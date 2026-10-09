# HyenaDNA inference on PACE ICE

This repo runs the pretrained HyenaDNA small-32k checkpoint on a GPU on Georgia Tech's PACE ICE cluster (Slurm, account `coc`, partition `coc-gpu`). A smoke test checks that inference runs, and a validation step compares human DNA to shuffled copies and a random-guess baseline.

## Quick start

```bash
git clone https://github.com/WilliamChen768/hyenadna-inference.git
cd hyenadna-inference
bash setup_env.sh            # once on a login node, builds the conda env
mkdir -p logs                # Slurm needs this to exist before the job starts
sbatch run_inference.sbatch  # runs infer.py, then validate.py on one GPU
```

After the job completes, `logs/hyenadna-infer-<jobid>.out` should end with three `PASS` lines, and `outputs/validation_results.json` should contain the mean loss on real and shuffled DNA, the ln(4) baseline, and the per-region values (see Validation).

If your allocation differs, change `#SBATCH -A coc` and `#SBATCH -p coc-gpu` in `run_inference.sbatch`, or override them at submit time:

`sbatch -A <account> -p <partition> run_inference.sbatch`

If you built the environment at a non-default location, run `export ENV_PREFIX=<path>` in your shell before `sbatch`. The job reads it from the submitting shell, and without it the job falls back to the default path.

## Files

- `setup_env.sh`: builds the conda env and installs `requirements.txt`
- `requirements.txt`: every package pinned
- `run_inference.sbatch`: Slurm job (1 GPU, 4 CPUs, 32 GB, 15 min); logs the node, `nvidia-smi`, and `pip freeze`, then runs `infer.py` and `validate.py`
- `infer.py`: smoke test on `ACTGACTGACTGACTG`; saves logits to `outputs/`
- `validate.py`: real vs. shuffled hg38 regions; writes `outputs/validation_results.json`, exits 1 if a check fails
- `fetch_hg38_regions.py`: downloads the validation regions (run once; output is committed)
- `data/hg38_regions.fasta`, `logs/`, `outputs/`

## Environment

Python 3.10.21 (conda-forge) comes from the `miniforge/24.3.0-0` module. The environment also contains torch 2.5.1+cu121, transformers 5.17.0, numpy 2.2.6, tokenizers 0.23.2, safetensors 0.8.0, and huggingface_hub 1.33.0. The cu121 wheel worked on nodes that reported CUDA drivers from 12.9 to 13.4.

`setup_env.sh` builds the environment at `$SCRATCH_DIR/.conda/envs/hyenadna_env`. `$SCRATCH_DIR` defaults to `$HOME/scratch`, which is a symlink to your storage directory on PACE ICE. The script also moves the conda and pip caches to scratch, because the home quota is only 30 GB. The sbatch file looks for the environment at the same location. To change it, run:

`SCRATCH_DIR=... ENV_PREFIX=... bash setup_env.sh`

`requirements.txt` pins every package, not only the top-level ones. Two builds from a file that listed only top-level packages differed in 7 transitive packages (see `logs/pip_freeze_diff.txt`), so I replaced that file with the freeze from the first successful run. `packaging` was pinned by hand to 26.3 because `pip freeze` listed it as a local conda path.

## Model and data

- **Model:** `LongSafari/hyenadna-small-32k-seqlen-hf` at commit `8fe770c78eb13fe33bf81501612faeddf4d6f331`, which every logged run used. The weights are licensed under BSD-3-Clause, per the model card. The model is loaded with `trust_remote_code=True`. `validate.py` hardcodes the commit, while `infer.py` resolves the latest commit and prints it, so it is not pinned there.
- **Source:** Nguyen et al., "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution" (2023), arXiv:2306.15794. The code is at https://github.com/HazyResearch/hyena-dna (Apache-2.0) and is not used directly. The model was pretrained on hg38.
- **Validation input:** 8 regions of 1,000 bp from hg38, fetched on 2026-10-08 from the UCSC API (`https://api.genome.ucsc.edu/getData/sequence`) using `fetch_hg38_regions.py`, with requests spaced 2 seconds apart. The start positions (0-based, half-open, end = start + 1000) are chr1:100000000, chr2:50000000, chr3:120000000, chr5:80000000, chr7:100000000, chr9:100000000, chr12:80000000, and chr16:60000000. They are arbitrary picks. All sequences were uppercased. A region would be skipped if it contained characters other than A, C, G, or T, or if its length was not 1,000 bases. None were skipped. The FASTA file is committed, so validation does not need access to UCSC.
- The model downloads to `$HF_HOME` on the first run (default `$SCRATCH_DIR/.hf_cache`). `infer.py` also queries the Hugging Face API, so compute nodes need internet access, which they had in every run.

## Runs

| Job | GPU | Env | Outcome |
|---|---|---|---|
| 6109681, 6109780 | V100 | `hyenadna_env` | Failed in 2-4 s (see Troubleshooting) |
| 6109847 | V100 | `hyenadna_env` | First successful smoke test; walltime 2:56 including model download |
| 6110540 | A100 | `hyenadna_env_test` | `setup_env.sh` rebuild from the top-level pins; smoke test OK |
| 6111003 | V100 | `hyenadna_env_test` | Rebuild from the pinned `requirements.txt`; `pip freeze` identical to 6109847 |
| 6114425 | L40S | `hyenadna_env` | Smoke test and validation pass; walltime 3:01, about 1.25 GB RAM |
| 6115131 | V100 | `hyenadna_env_fresh` | Fresh clone of `main`, env rebuilt with `setup_env.sh` from `requirements.txt`; smoke test and validation pass; `pip freeze` identical to 6114425; walltime 0:29 |

Logs for each job are in `logs/hyenadna-infer-<jobid>.out` and `.err`, and the pip freeze records are in `logs/pip_freeze_<jobid>.txt`. The environment setup logs are `logs/setup_env_test.log` and `logs/setup_env_lock_test.log`. The `outputs/smoke_test_*` files and `outputs/validation_results.json` hold the most recent run, and the original V100 output is in `outputs/run_6109847/`.

Smoke-test logits match across GPU types. The V100 and L40S runs had identical token IDs and a maximum logit difference of 1.05e-5 (see `logs/v100_vs_l40s_comparison.txt`). The V100 and A100 runs agreed within `allclose` at 1e-4. I checked that interactively and did not save the result. The times printed by `infer.py` range from 0.3 s to 6.7 s. They are single first-call measurements that I can't explain, so don't treat them as benchmarks.

The `.err` files usually contain a Hugging Face warning about unauthenticated requests, a deprecation notice for `use_return_dict`, and weight-loading progress bars. None of these are an error.

## Validation

**Smoke test:** the logits shape is `(1, 17, 16)`: one sequence, 17 tokens (16 DNA bases plus one special token), and a vocabulary of 16 symbols. A, C, G, and T map to token IDs 7, 8, 9, and 10.

**Substantive check:** for each region, `validate.py` calculates the mean next-base loss (nats per base) over A, C, G, and T only, so the random-guess baseline is ln(4) = 1.3863. It does this for the real sequence and for a shuffled copy (same composition, order destroyed, `random.Random(0)`). The full-vocabulary loss is also saved and differs by roughly 3e-5. If a checkpoint loads incorrectly, the shapes can look reasonable while the predictions are near random. The smoke test can't detect that, but this check can. The pass criteria were set before the first run: mean real loss below ln(4), mean real loss below mean shuffled loss, and real loss below shuffled loss in most regions.

| ln(4) | Shuffled (mean) | Real (mean) | Real below shuffled |
|---|---|---|---|
| 1.3863 | 1.3908 | 1.0944 | 8 of 8 regions |

All three checks passed for job 6114425 and again for the fresh-clone job 6115131, which gave the same means (real 1.0944, shuffled 1.3908). Per-region values are in `outputs/validation_results.json`. Real-region loss ranges from 0.83 to 1.31, and I haven't investigated why. The shuffled copies sit at the baseline, so the gap on real DNA comes from base order. This is a sanity check on eight regions with one shuffle each, and the regions were not held out from pretraining.

## Troubleshooting

- **Job ends in 2-4 s with an empty `.out`, and `.err` shows `xml_catalog_files_libxml2: unbound variable`** (jobs 6109681 and 6109780). The cause is `set -u`, because conda's activation scripts reference unset variables. The fix is to use `set -eo pipefail` without `-u`.
- **`sbatch` produces no log.** Slurm opens the `-o` and `-e` files before the script starts, so `logs/` must already exist. Create it with `mkdir -p logs`.
- **`Disk quota exceeded`.** `~/.conda/pkgs` held 21 GB of the 30 GB home quota. `conda clean --all` freed only 58 MB because the files are hardlinked into existing environments. Removing `~/.conda/pkgs` and keeping the caches on scratch (as `setup_env.sh` does) fixed it. Check usage with `quota -s`.
- **Job runs in the wrong environment.** If you built the environment with a custom `ENV_PREFIX`, export it before `sbatch`. My first attempt at the fresh-clone run (6115127, cancelled) would have used the default environment because I had only set it on the `setup_env.sh` command. Check the `Env:` line near the top of the `.out` file.
- **Install from `requirements.txt`, not from the `logs/pip_freeze_*.txt` files.** The freeze files contain entries like `packaging @ file:///...`, a path that exists only on the conda build machine.

## Cleanup

```bash
rm -rf "$HOME/scratch/.conda/envs/hyenadna_env" "$HOME/scratch/.hf_cache" \
       "$HOME/scratch/.conda_pkgs" "$HOME/scratch/.pip_cache"
```

Adjust the paths if you set `SCRATCH_DIR`, `ENV_PREFIX`, or `HF_HOME`.

## Limitations

- Everything ran on PACE ICE (V100, A100, L40S). Other clusters need different `-A`/`-p` values and scratch paths.