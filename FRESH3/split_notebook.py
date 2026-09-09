import importlib.util, nbformat as nbf, os
OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # write alongside this script

spec = importlib.util.spec_from_file_location("build_notebook", "build_notebook.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cells = m.cells

def find_idx(marker):
    return next(i for i, c in enumerate(cells)
                if c["cell_type"] == "markdown" and marker in c["source"])

def find_code_idx(marker):
    """Find a code cell whose source contains the given marker substring."""
    return next(i for i, c in enumerate(cells)
                if c["cell_type"] == "code" and marker in c["source"])

SPLIT_TRAIN_END = find_code_idx('NAMES = {"iso_score": "Isolation Forest"')  # end of training+save cell
SPLIT_MV = find_idx("## 5c. Does multivariate context")
SPLIT_ECON = find_idx("## 6. Turning detection into profit")
SPLIT_A2 = find_idx("## 6b. Does function approximation")
SPLIT_B = find_idx("## 7. Robustness check")
print(f"Part A-train: 0..{SPLIT_TRAIN_END}   Part A-mv: {SPLIT_TRAIN_END+1}..{SPLIT_ECON-1}   "
      f"Part A-econ: {SPLIT_ECON}..{SPLIT_A2-1}   Part A2 (DQN): {SPLIT_A2}..{SPLIT_B-1}   Part B: {SPLIT_B}..end")

part_a_train_cells = cells[:SPLIT_TRAIN_END+1]
part_a_figstats_src_cells = cells[SPLIT_TRAIN_END+1:SPLIT_MV]  # ROC/timeline figs + stats (5, 5b)
part_a_mv_src_cells = cells[SPLIT_MV:SPLIT_ECON]                # multivariate diffusion (5c)
part_a_econ_src_cells = cells[SPLIT_ECON:SPLIT_A2]               # economics (6)
part_a2_src_cells = cells[SPLIT_A2:SPLIT_B]
part_b_src_cells = cells[SPLIT_B:]

# Definition cells identified by content signature, not position -- robust to
# cells being added/removed elsewhere in build_notebook.py.
DEFINITION_CELL_IDX = [
    find_code_idx("%matplotlib inline"),                                    # imports
    find_code_idx('"""Real-data loading, aggregation'),                     # data_prep.py
    find_code_idx('"""Feature engineering + a weak'),                       # features.py
    find_code_idx('"""Four anomaly-scoring mechanisms'),                    # detectors.py
    find_code_idx('"""Repeat detector training/evaluation'),                # repeat_eval funcs
    find_code_idx('"""Battery/market constants'),                           # params_local.py
    find_code_idx('"""Three ways to turn a detector'),                      # strategies.py
]
print("Definition cell indices:", DEFINITION_CELL_IDX)

RELOAD_W1_CELL = '''# --- Reload Part A's results from disk instead of re-training ---
agg = build_aggregated()
agg = add_time_features(agg)
agg = reference_anomaly_flag(agg)
train, val, test = split(agg)

with open("artifacts/detector_eval_ensemble.json") as f:
    ens_report = json.load(f)
ens_df = pd.read_pickle("artifacts/agg_scored_ensemble.pkl")
with open("artifacts/all_runs.json") as f:
    all_runs = json.load(f)  # per-seed results, for the statistical significance subsection
try:
    with open("artifacts/economics_results.json") as f:
        econ = json.load(f)
except FileNotFoundError:
    econ = {}  # not yet produced (e.g. this IS the part that produces it)

winner_col = max(SCORE_COLS, key=lambda c: ens_report[c]["AUROC"])
RL_SEEDS = [42, 43, 44, 45, 46]
NAMES = {"iso_score": "Isolation Forest", "vae_score": "VAE",
         "diffusion_score": "Diffusion (DDPM)", "lstm_score": "LSTM-AE"}

def flags_from_budget(scores, budget):
    cut = int(np.ceil(len(scores) * budget))
    thr = np.sort(scores)[-cut]
    return (scores >= thr).astype(int)

ens_full = ens_df.merge(agg[["datetime","SP15"]], on="datetime")
_, val_f, test_f = split(ens_full)
prices = test_f["SP15"].values
price_mean, price_std = train["SP15"].mean(), train["SP15"].std()
train_scores = ens_df.loc[train.index, winner_col].values
test_scores = test_f[winner_col].values

print("Reloaded Part A artifacts. Winning detector (Window 1):", NAMES[winner_col])
print("Window 1 ensemble AUROC:", {c: round(ens_report[c]["AUROC"], 3) for c in SCORE_COLS})
'''

def build_part(name, header_md, extra_setup_src, body_cells, out_path):
    nb = nbf.v4.new_notebook()
    out_cells = [nbf.v4.new_markdown_cell(header_md)]
    for i in DEFINITION_CELL_IDX:
        out_cells.append(cells[i])
    if extra_setup_src:
        out_cells.append(nbf.v4.new_code_cell(extra_setup_src))
    out_cells.extend(body_cells)
    nb["cells"] = out_cells
    nbf.write(nb, out_path)
    print(f"{name}: {len(out_cells)} cells -> {out_path}")

# Part A (training only): the reliably-fast part -- data through 5-seed
# detector training and the ensemble/checkpoint save. Everything downstream
# reloads from artifacts/ rather than depending on kept-alive kernel state.
nbA = nbf.v4.new_notebook()
nbA["cells"] = part_a_train_cells
nbf.write(nbA, os.path.join(OUT_DIR, "Paper_Code_v2_partA.ipynb"))
print(f"Part A (train): {len(part_a_train_cells)} cells -> Paper_Code_v2_partA.ipynb")

# Part A-figstats: ROC/timeline figures + statistical significance testing
build_part(
    "Part A-figstats",
    """# Part A-figstats: figures and statistical testing (continues from Part A)

Continues `Paper_Code_v2_partA.ipynb` in a fresh kernel -- redefines functions/classes (fast) and
reloads Part A's trained detector outputs from disk, then produces the ROC/timeline figures and
the per-seed statistical significance comparison.""",
    RELOAD_W1_CELL,
    part_a_figstats_src_cells,
    os.path.join(OUT_DIR, "Paper_Code_v2_partA_figstats.ipynb"),
)

# Part A-mv: multivariate diffusion ablation, self-contained, doesn't feed downstream
build_part(
    "Part A-mv",
    """# Part A-mv: Multivariate diffusion ablation (continues from Part A)

Continues `Paper_Code_v2_partA.ipynb` in a fresh kernel -- redefines functions/classes (fast) and
reloads Part A's results from disk, then trains a paired univariate-vs-multivariate diffusion
comparison (the slow part, ~5 seeds x 2 variants).""",
    RELOAD_W1_CELL,
    part_a_mv_src_cells,
    os.path.join(OUT_DIR, "Paper_Code_v2_partA_mv.ipynb"),
)

# Part A-econ: Window-1 economics only (figures/stats moved to Part A-figstats)
build_part(
    "Part A-econ",
    """# Part A-econ: Window-1 economics (continues from Part A)

Continues `Paper_Code_v2_partA.ipynb` in a fresh kernel -- redefines functions/classes (fast) and
reloads Part A's trained detector outputs from disk, then runs the three Window-1 profit
strategies (dispatch, signal-to-PnL, tabular Q-learning).""",
    RELOAD_W1_CELL,
    part_a_econ_src_cells,
    os.path.join(OUT_DIR, "Paper_Code_v2_partA_econ.ipynb"),
)

# Part A2: DQN section, fresh kernel, reloads Part A's checkpoints
build_part(
    "Part A2",
    """# Part A2: DQN check (continues from Part A)

Continues `Paper_Code_v2_partA.ipynb` in a fresh kernel -- redefines functions/classes (fast) and
reloads Part A's results from disk, then runs the DQN comparison (the slow part, ~5 seeds x 2
conditions x 10 episodes).""",
    RELOAD_W1_CELL,
    part_a2_src_cells,
    os.path.join(OUT_DIR, "Paper_Code_v2_partA2.ipynb"),
)

# Part B: window 2 robustness check, fresh kernel, reloads Part A's checkpoints
build_part(
    "Part B",
    """# Part B: Rolling-origin robustness check (continues from Part A)

Continues `Paper_Code_v2_partA.ipynb` in a fresh kernel -- redefines functions/classes (fast) and
reloads Part A's results from disk, then runs the Window-2 robustness check.""",
    RELOAD_W1_CELL,
    part_b_src_cells,
    os.path.join(OUT_DIR, "Paper_Code_v2_partB.ipynb"),
)
