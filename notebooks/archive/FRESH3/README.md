# Detecting Genuine Price Anomalies for Battery Arbitrage

Full deliverable set for the IEEE-style paper and its supporting notebook. Nothing in the paper
is hand-typed — every number traces back to `Paper_Code_v2.ipynb`'s actual printed output.

## Setup (fresh venv)

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r pipeline_source/requirements.txt

# put your CAISO cache pickles somewhere, then point the pipeline at them:
export CAISO_DATA_DIR=/path/to/your/pickles    # or just drop them in pipeline_source/data/
```

**If you tried this before and it didn't run:** earlier versions of these scripts had paths
hardcoded to the sandbox they were built in (`/mnt/user-data/uploads`, `/home/claude/work/...`),
which don't exist anywhere else — that's almost certainly why. Fixed as of this version: every
script now uses `CAISO_DATA_DIR` (env var, defaults to `./data`) for input and plain relative
paths (`artifacts/`, `figs/`) for output, verified by actually running the full pipeline from a
different, unrelated directory before shipping this update.

## Files

| File | What it is |
|---|---|
| `main.pdf` | The compiled paper (9 pages, IEEE journal format) |
| `main.tex` | LaTeX source |
| `references.bib` | Bibliography |
| `Paper_Code_v2.ipynb` | The full, executed notebook — data loading through detection, evaluation, all economic backtests (DQN check + multivariate diffusion ablation included), on **both** test windows, plus statistical-significance analysis. Runs end-to-end with zero errors; all figures are embedded. |
| `Paper_Code_v2_partA.ipynb` / `partA_figstats.ipynb` / `partA_mv.ipynb` / `partA_econ.ipynb` / `partA2.ipynb` / `partB.ipynb` | The six pieces the full notebook is assembled from — split because the whole thing doesn't fit in one execution session. Part A = 5-seed detector training (fast, reliable); Part A-figstats = ROC/timeline figures + statistical testing; Part A-mv = multivariate diffusion ablation; Part A-econ = Window-1 economics; Part A2 = the DQN check; Part B = the Window-2 robustness check. All parts after Part A reload its results from `artifacts/` rather than depending on kept-alive kernel state. |
| `figs/` | The 4 figures used in the paper (architecture diagram, ROC curves, test-window timeline, profit comparison) |
| `artifacts/` | Raw JSON results the paper's tables are built from (detector AUROC/AP/F1 per seed and ensemble, economics profit numbers), for both test windows |
| `pipeline_source/` | The standalone `.py` modules the notebook is assembled from (see below) |

## Pipeline source files

The notebook is generated from these modules rather than written directly, so the same code can
run either as a notebook or as plain scripts:

- `data_prep.py` — loads your cached CAISO pickles, merges LMP + ancillary services, defines both
  chronological splits (`split` = primary window, test Jul–Dec 2025; `split_w2` = rolling-origin
  second window, test Jan–Jun 2025)
- `features.py` — rolling-stat feature engineering + the statistics-only reference anomaly flag
  (evaluation-only, never used for training)
- `detectors.py` — all four detectors: Isolation Forest prefilter, VAE, diffusion (DDPM, residual
  blocks + cosine schedule) in both univariate and multivariate (`MVDiffusion`) form, LSTM autoencoder
- `strategies.py` — the four profit/control strategies: myopic battery dispatch, signal-to-PnL
  long/short backtest, tabular Q-learning agent, and a DQN (experience replay + target network)
- `params_local.py` — battery/market constants (mirrors your original `params.py`)
- `repeat_eval.py` — trains all 4 detectors across 5 independently-seeded runs, computes both the
  validation-selected checkpoint and the 5-seed rank-ensemble; also defines `eval_split`/`eval_ensemble`
  reused by every other script below
- `run_economics.py` — runs the 3 core profit strategies on the primary window using the ensemble scores
- `rolling_window_eval.py` — repeats detector training + economics on the second (rolling-origin) window
- `mv_diffusion_comparison.py` — the paired univariate-vs-multivariate diffusion ablation (5 seeds each)
- `run_dqn_comparison.py` — the compute-matched DQN vs. tabular Q-learning comparison (5 seeds each)
- `raw_seed_values.py` — standalone script that saves per-seed AUROC arrays (not just mean/std) for
  the paired statistical significance tests; the notebook now does this automatically in Part A
  (saved as `artifacts/all_runs.json`), so this script is kept for anyone who wants just that piece
  without running the rest of the pipeline
- `diff_seeds_detectors.py` / `diff_seeds_rl.py` — reruns detector training and the RL comparison on
  an entirely disjoint seed set ({100..104} / {200..204}) to check the headline findings aren't
  specific to one arbitrary seed choice
- `make_architecture_fig.py` — generates `figs/architecture.png`
- `build_notebook.py` / `split_notebook.py` — assemble `Paper_Code_v2.ipynb` from the modules above,
  split into six parts so each finishes within a single execution session, then merged (see
  `README`'s file table above for what each part covers)

## Reproducing from scratch

1. Put your CAISO cache pickles in `/mnt/user-data/uploads/` (same filenames `data_prep.py` expects).
2. `python3 build_notebook.py && python3 split_notebook.py` — assembles and splits the notebook
   into six parts (`Paper_Code_v2_partA.ipynb`, `partA_figstats.ipynb`, `partA_mv.ipynb`,
   `partA_econ.ipynb`, `partA2.ipynb`, `partB.ipynb`).
3. Execute the six parts **in that order** via `jupyter nbconvert --to notebook --execute --inplace
   <part>.ipynb` — each part after Part A reloads Part A's results from `artifacts/`, so the order
   matters but each part is independently a fast, single execution session (~1–5 min each).
4. Merge the six executed parts' cells (in the same order) into one notebook to get the final
   `Paper_Code_v2.ipynb` — a few lines of `nbformat` (read each part, concatenate `cells`, write).
5. `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` to rebuild the PDF.

The standalone scripts (`repeat_eval.py`, `run_economics.py`, `rolling_window_eval.py`,
`mv_diffusion_comparison.py`, `run_dqn_comparison.py`, `diff_seeds_detectors.py`, `diff_seeds_rl.py`)
can also be run directly with `python3 <script>.py` if you just want the numbers without rebuilding
the notebook — this is how each finding was originally produced and cross-checked before being
folded into the notebook.

## Headline results

- **Detection**: LSTM autoencoder wins on both test windows (ensemble AUROC 0.859 / 0.921), ahead
  of Isolation Forest, VAE, and a diffusion detector. iForest's "most stable" claim from Window 1
  does **not** hold on Window 2 (AUROC drops to 0.453) — reported explicitly, not hidden.
- **Economics**: value of detection is strategy-dependent. Battery dispatch (already
  physically constrained) barely moves. A signal-to-PnL backtest beats random timing on every
  detector, both windows. A **tabular** Q-learning agent conditioned on the anomaly signal both
  profits more and is far more stable across training seeds than the same agent without it — on
  both windows and under a disjoint seed set.
- **Statistical honesty on small samples**: with only 5 seeds, no standard significance test can
  reach p<0.05 even under a perfect win record (Wilcoxon's floor at n=5 is p=0.0625). Rather than
  compute a misleading test, the paper reports per-seed win counts and bootstrap confidence
  intervals instead — e.g. LSTM-AE beats iForest in 4/5 seeds, 95% CI on the mean AUROC gap
  [0.015, 0.231]; the LSTM-AE vs. Diffusion comparison is the least clean at only 3/5 wins.
- **Multivariate input does not help the diffusion detector either.** The paper's own Discussion
  speculated that feeding all 3 nodes + 4 ancillary-service prices jointly (instead of the single
  aggregated series) might close diffusion's gap to the LSTM-AE. Tested directly: it doesn't
  (0.528±0.066 AUROC vs. 0.564±0.071 univariate; univariate wins 4/5 seeds; bootstrap CI on the
  difference includes zero, so the honest conclusion is "no evidence it helps," not "it hurts").
- **A DQN does not confirm the tabular RL finding.** We tested whether a function-approximation
  upgrade would strengthen the tabular Q-learning result rather than just assuming it would — under
  a compute-matched budget, the DQN's with-signal profit is not reliably above its own no-signal
  condition. Reported as-is; this corrects a claim the paper's Limitations section used to make
  on pure speculation.
- **Two bugs caught and fixed during this work, disclosed in the paper rather than silently fixed**:
  (1) a shared RNG stream across models let one model's architecture change silently perturb
  another model's initialization; (2) CPU thread-parallel matmul isn't bit-reproducible across
  process launches, which is why every headline economic number is averaged over multiple seeds
  rather than read from one run.
