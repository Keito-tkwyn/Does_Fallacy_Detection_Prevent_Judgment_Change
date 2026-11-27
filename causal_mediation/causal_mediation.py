
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from safetensors.torch import load_file

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

from tqdm import tqdm


# ============================================================
# 1. Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden-root", type=Path, required=True, help="Directory containing sample_* hidden-state directories.")

    parser.add_argument("--label-csv", "--csv_path", dest="csv_path", type=Path, required=True, help="Headerless CSV: first_belief, second_belief, detection.")


    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True, help="Directory for mediation results and intermediate memmap.")

    parser.add_argument("--fallacy-last-id", "--fallacy_last_id", type=int, default=5199, help="Inclusive final sample ID assigned treatment T=1; subsequent IDs have T=0.")

    parser.add_argument("--n-splits", "--n_splits", type=int, default=5, help="Number of stratified cross-validation folds.")

    parser.add_argument("--skip-first-layer", "--skip_first_layer", action="store_true", help="Skip hidden-state index 0 (embedding output).")

    parser.add_argument("--use-first-belief-control", "--use_first_belief_control", action="store_true", help="Use first judgment as a control variable.")

    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap replicates; 0 disables confidence intervals.")

    parser.add_argument("--random-state", "--random_state", type=int, default=42)

    return parser.parse_args()


# ============================================================
# 2. Load labels from CSV
# ============================================================

def normalize_detection_label(x):
    """
    Convert the detection column to binary labels: Detect = 1, Not Detect = 0.
    """
    v = str(x).strip().lower()

    if v in {"detect", "detected", "yes", "true", "1"}:
        return 1

    if v in {"not detect", "not_detect", "not detected", "no", "false", "0"}:
        return 0

    raise ValueError(f"Unknown detection label: {x}")


def load_labels(csv_path):
    """
    Construct labels from the CSV. detection_label: Detect = 1, Not Detect = 0.
    belief_changed: 1 if first_belief differs from second_belief, otherwise 0.
    """
    df = pd.read_csv(csv_path,names=['first_belief','second_belief','detection'],encoding='utf-8')


    # Use the CSV row number as sample_id.
    # Assume that rows correspond to sample_0, sample_1, and so on.
    df["sample_id"] = df.index.astype(int)

    # detection label
    df["detection_label"] = df["detection"].apply(normalize_detection_label)

    # Judgment-change label
    first = df["first_belief"].astype(str).str.strip().str.lower()
    second = df["second_belief"].astype(str).str.strip().str.lower()

    df["belief_changed"] = (first != second).astype(int)

    return df


# ============================================================
# 3. Locate hidden-state files
# ============================================================

def extract_sample_id(path):
    """
    Extract sample_id from a path containing sample_<number>, e.g.,
    sample_123/persuation/hidden_states.safetensors -> 123.
    """
    m = re.search(r"sample_(\d+)", str(path))
    if m is None:
        raise ValueError(f"Could not extract sample_id: {path}")
    return int(m.group(1))


def find_hidden_state_files(hidden_root):
    """
    Locate hidden-state files. Expected layout:
    hidden_root/sample_*/persuasion/hidden_states.safetensors. Prefer the legacy
    spelling 'persuation', falling back to 'persuasion'.
    """
    hidden_root = Path(hidden_root)

    paths = list(hidden_root.glob("sample_*/persuation/hidden_states.safetensors"))

    if len(paths) == 0:
        paths = list(hidden_root.glob("sample_*/persuasion/hidden_states.safetensors"))

    if len(paths) == 0:
        raise FileNotFoundError(f"Hidden states not found: {hidden_root}")

    path_by_id = {}
    for p in paths:
        sid = extract_sample_id(p)
        path_by_id[sid] = p

    return path_by_id


# ============================================================
# 4. Extract final-token vectors for all layers from safetensors
# ============================================================

def _sort_key_by_layer_number(key):
    """
    Sort tensor keys by their numeric layer index when layers are stored separately.
    """
    nums = re.findall(r"\d+", key)
    if len(nums) == 0:
        return 10**9
    return int(nums[-1])


def tensor_to_final_token_vector(x):
    """
    Extract the final-token hidden vector from a tensor. Supported shapes: [hidden_dim],
    [seq_len, hidden_dim], [1, seq_len, hidden_dim], or [batch, seq_len, hidden_dim]
    with batch=1. Convert bfloat16 to float32 with .float() before converting to NumPy.
    """
    x = x.detach().cpu()

    # Remove the batch dimension if its size is 1.
    while x.ndim >= 3 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim == 1:
        # Already a vector of size hidden_dim.
        return x.float().numpy().astype(np.float32)

    if x.ndim == 2:
        # Treat as [seq_len, hidden_dim] and use the final token.
        return x[-1].float().numpy().astype(np.float32)

    raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")


def load_final_token_all_layers(safetensors_path):
    """
    Extract the final-token residual stream or hidden state for each layer from a
    hidden_states.safetensors file. Returns a NumPy array of shape [num_layers,
    hidden_dim].
    """
    tensors = load_file(str(safetensors_path), device="cpu")

    # --------------------------------------------------------
    # Case A:
    # All layers are stored under a single key.
    # Example:
    #   hidden_states: [num_layers, seq_len, hidden_dim]
    #   hidden_states: [1, num_layers, seq_len, hidden_dim]
    # --------------------------------------------------------
    if len(tensors) == 1:
        x = next(iter(tensors.values())).detach().cpu()

        # [1, num_layers, seq_len, hidden_dim] -> [num_layers, seq_len, hidden_dim]
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)

        if x.ndim == 3:
            # Typically [num_layers, seq_len, hidden_dim].
            # Also allow [seq_len, num_layers, hidden_dim] using a simple heuristic.
            if x.shape[0] <= 200:
                # [num_layers, seq_len, hidden_dim]
                vectors = x[:, -1, :]
            elif x.shape[1] <= 200:
                # [seq_len, num_layers, hidden_dim]
                vectors = x[-1, :, :]
            else:
                raise ValueError(f"Could not identify the layer dimension: {tuple(x.shape)}")

            # Handle bfloat16 conversion.
            return vectors.float().numpy().astype(np.float32)

        if x.ndim == 2:
            # Treat as [num_layers, hidden_dim].
            return x.float().numpy().astype(np.float32)

        raise ValueError(f"Unexpected hidden-state shape: {tuple(x.shape)}")

    # --------------------------------------------------------
    # Case B:
    # Each layer is stored under a separate key.
    # Example:
    #   layer_0: [seq_len, hidden_dim]
    #   layer_1: [seq_len, hidden_dim]
    # --------------------------------------------------------
    layer_keys = sorted(tensors.keys(), key=_sort_key_by_layer_number)

    vectors = []
    for k in layer_keys:
        v = tensor_to_final_token_vector(tensors[k])
        vectors.append(v)

    return np.stack(vectors, axis=0).astype(np.float32)


# ============================================================
# 5. Combine hidden states in a memory-mapped array
# ============================================================

def build_hidden_state_memmap(
    sample_ids,
    path_by_id,
    output_dir,
    skip_first_layer=False,
):
    """
    Load each sample's safetensors into a memory-mapped array of shape [num_samples,
    num_layers, hidden_dim]. A memmap avoids the high memory cost of holding all samples
    and layers in a regular NumPy array.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine the number of layers and hidden size from the first sample.
    first_vecs = load_final_token_all_layers(path_by_id[sample_ids[0]])

    if skip_first_layer:
        first_vecs = first_vecs[1:]

    num_layers, hidden_dim = first_vecs.shape
    num_samples = len(sample_ids)

    memmap_path = output_dir / "hidden_final_token_all_layers.dat"

    H = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(num_samples, num_layers, hidden_dim),
    )

    # Store the first sample.
    H[0] = first_vecs

    # Store the remaining samples.
    for row_idx, sid in enumerate(tqdm(sample_ids[1:], desc="Loading hidden states"), start=1):
        vecs = load_final_token_all_layers(path_by_id[sid])

        if skip_first_layer:
            vecs = vecs[1:]

        if vecs.shape != (num_layers, hidden_dim):
            raise ValueError(
                f"Shape mismatch for sample_{sid}: "
                f"{vecs.shape} != {(num_layers, hidden_dim)}"
            )

        H[row_idx] = vecs

    H.flush()

    metadata = {
        "memmap_path": str(memmap_path),
        "num_samples": num_samples,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
    }

    return H, metadata


# ============================================================
# 6. Compute detection probe scores
# ============================================================

def compute_out_of_fold_probe_scores(X, y, n_splits=5, random_state=42):
    """
    Train a linear probe to predict detection_label y from residual-stream features X at
    a layer. Use out-of-fold scores, not in-sample scores, as the mediation variable.
    Inputs: X of shape [num_samples, hidden_dim] and y of shape [num_samples] (Detect=1,
    Not Detect=0). Returns decision_function scores for mediator M_l, AUROC, and
    accuracy.
    """
    y = np.asarray(y).astype(int)

    class_counts = np.bincount(y)
    min_class_count = class_counts.min()

    if min_class_count < 2:
        raise ValueError("One detection class has too few samples to train the probe.")

    n_splits = min(n_splits, min_class_count)

    scores = np.zeros(len(y), dtype=np.float32)
    preds = np.zeros(len(y), dtype=np.int64)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for train_idx, test_idx in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
            )
        )

        clf.fit(X[train_idx], y[train_idx])

        # decision_function = w^T h + b
        # Use as the mediator M in the mediation analysis.
        fold_scores = clf.decision_function(X[test_idx])
        scores[test_idx] = fold_scores.astype(np.float32)

        preds[test_idx] = (fold_scores >= 0).astype(int)

    auc = roc_auc_score(y, scores)
    acc = accuracy_score(y, preds)

    return scores, auc, acc


# ============================================================
# 7. Construct control variables
# ============================================================

def build_controls(df, use_first_belief_control=False):
    """
    Construct control variables for mediation. The CSV is assumed not to contain topic
    or fallacy_type, so no controls are used by default. Optionally include first_belief
    as one-hot features.
    """
    if not use_first_belief_control:
        return np.zeros((len(df), 0), dtype=np.float32)

    controls = pd.get_dummies(
        df["first_belief"].astype(str).str.strip().str.lower(),
        prefix="first_belief",
        drop_first=True,
    )

    return controls.values.astype(np.float32)


# ============================================================
# 8. Estimate statistical mediation effects
# ============================================================

def fit_mediation(T, M, Y, C=None):
    """
    Estimate statistical mediation with a binary outcome. T: fallacy=1, no fallacy=0. M:
    standardized continuous detection probe score. Y: belief_changed (changed=1,
    maintained=0). C: controls of shape [num_samples, num_controls]. Mediator model: M ~
    T + C. Outcome model: Y ~ T + M + T*M + C. Effects are differences on the predicted
    probability scale for Y=1.
    """
    T = np.asarray(T).astype(int)
    M = np.asarray(M).astype(float)
    Y = np.asarray(Y).astype(int)

    n = len(T)

    if C is None:
        C = np.zeros((n, 0), dtype=np.float32)

    C = np.asarray(C).astype(np.float32)

    # --------------------------------------------------------
    # 8.1 Mediator model: M ~ T + C
    # --------------------------------------------------------
    X_mediator = np.column_stack([T, C])

    mediator_model = LinearRegression()
    mediator_model.fit(X_mediator, M)

    # Counterfactual mediator
    # M(0): probe score under the no-fallacy condition.
    # M(1): probe score under the fallacy condition.
    T0 = np.zeros(n, dtype=int)
    T1 = np.ones(n, dtype=int)

    X_m0 = np.column_stack([T0, C])
    X_m1 = np.column_stack([T1, C])

    M0 = mediator_model.predict(X_m0)
    M1 = mediator_model.predict(X_m1)

    # --------------------------------------------------------
    # 8.2 Outcome model: Y ~ T + M + T*M + C
    # --------------------------------------------------------
    def make_outcome_features(t, m):
        interaction = t * m
        return np.column_stack([t, m, interaction, C])

    X_outcome = make_outcome_features(T, M)

    outcome_model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )
    outcome_model.fit(X_outcome, Y)

    # --------------------------------------------------------
    # 8.3 Predict counterfactual outcomes
    # --------------------------------------------------------
    def predict_y_prob(t, m):
        X = make_outcome_features(t, m)
        return outcome_model.predict_proba(X)[:, 1]

    # Y(1, M(1))
    Y_1_M1 = predict_y_prob(T1, M1)

    # Y(1, M(0))
    Y_1_M0 = predict_y_prob(T1, M0)

    # Y(0, M(1))
    Y_0_M1 = predict_y_prob(T0, M1)

    # Y(0, M(0))
    Y_0_M0 = predict_y_prob(T0, M0)

    # --------------------------------------------------------
    # 8.4 Compute indirect and direct effects
    # --------------------------------------------------------

    # ACME(1): hold T=1 fixed and change only M from M(0) to M(1).
    ACME_1 = np.mean(Y_1_M1 - Y_1_M0)

    # ACME(0): hold T=0 fixed and change only M from M(0) to M(1).
    ACME_0 = np.mean(Y_0_M1 - Y_0_M0)

    # ADE(1): hold M=M(1) fixed and change only T from 0 to 1.
    ADE_1 = np.mean(Y_1_M1 - Y_0_M1)

    # ADE(0): hold M=M(0) fixed and change only T from 0 to 1.
    ADE_0 = np.mean(Y_1_M0 - Y_0_M0)

    # Total Effect
    TE = np.mean(Y_1_M1 - Y_0_M0)

    ACME_avg = 0.5 * (ACME_0 + ACME_1)
    ADE_avg = 0.5 * (ADE_0 + ADE_1)

    if np.abs(TE) > 1e-12:
        mediated_prop = ACME_avg / TE
    else:
        mediated_prop = np.nan

    return {
        "ACME_0": ACME_0,
        "ACME_1": ACME_1,
        "ACME_avg": ACME_avg,
        "ADE_0": ADE_0,
        "ADE_1": ADE_1,
        "ADE_avg": ADE_avg,
        "TE": TE,
        "mediated_prop": mediated_prop,
        "mediator_coef_T": mediator_model.coef_[0],
    }


def bootstrap_mediation(T, M, Y, C=None, n_boot=200, random_state=42):
    """
    Compute bootstrap confidence intervals for mediation effects. The probe is not
    retrained: the out-of-fold mediator M is held fixed, so intervals reflect
    uncertainty in the mediation models only.
    """
    rng = np.random.default_rng(random_state)
    n = len(T)

    keys = [
        "ACME_0",
        "ACME_1",
        "ACME_avg",
        "ADE_0",
        "ADE_1",
        "ADE_avg",
        "TE",
        "mediated_prop",
    ]

    values = {k: [] for k in keys}

    if C is None:
        C = np.zeros((n, 0), dtype=np.float32)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        # Skip bootstrap samples containing only one class of Y.
        if len(np.unique(Y[idx])) < 2:
            continue

        try:
            res = fit_mediation(
                T=T[idx],
                M=M[idx],
                Y=Y[idx],
                C=C[idx],
            )

            for k in keys:
                values[k].append(res[k])

        except Exception:
            continue

    ci = {}
    for k in keys:
        arr = np.asarray(values[k], dtype=float)
        if len(arr) == 0:
            ci[f"{k}_ci_low"] = np.nan
            ci[f"{k}_ci_high"] = np.nan
        else:
            ci[f"{k}_ci_low"] = np.nanpercentile(arr, 2.5)
            ci[f"{k}_ci_high"] = np.nanpercentile(arr, 97.5)

    return ci


# ============================================================
# 9. Main procedure
# ============================================================

def main():
    args = parse_args()

    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 9.1 Load labels
    # --------------------------------------------------------
    print("Loading labels...")
    labels_df = load_labels(args.csv_path)

    # --------------------------------------------------------
    # 9.2 Locate hidden-state files
    # --------------------------------------------------------
    print("Finding hidden state files...")
    path_by_id = find_hidden_state_files(args.hidden_root)

    # --------------------------------------------------------
    # 9.3 Keep only samples with both CSV labels and hidden states
    # --------------------------------------------------------
    available_sample_ids = sorted(set(labels_df["sample_id"]) & set(path_by_id.keys()))


    if len(available_sample_ids) == 0:
        raise ValueError(
            "No matching sample IDs remain between the CSV and hidden states after filtering."
        )

    labels_df = labels_df.set_index("sample_id").loc[available_sample_ids].reset_index()

    sample_ids = labels_df["sample_id"].tolist()

    print(f"Number of usable samples: {len(sample_ids)}")

    # --------------------------------------------------------
    # 9.4 Construct T, Y, and D
    # --------------------------------------------------------
    # T:
    # Treatment is defined by --fallacy-last-id.
    T = (labels_df["sample_id"].values <= args.fallacy_last_id).astype(int)

    # Y:
    # If first_belief != second_belief, judgment change = 1.
    # If they match, judgment change = 0.
    Y = labels_df["belief_changed"].values.astype(int)

    # D:
    #   Detect = 1
    #   Not Detect = 0
    D = labels_df["detection_label"].values.astype(int)

    print("Treatment distribution:")
    print(pd.Series(T).value_counts().rename(index={0: "no_fallacy", 1: "fallacy"}))

    print("Belief change distribution:")
    print(pd.Series(Y).value_counts().rename(index={0: "unchanged", 1: "changed"}))

    print("Detection distribution:")
    print(pd.Series(D).value_counts().rename(index={0: "not_detect", 1: "detect"}))

    # --------------------------------------------------------
    # 9.5 Control variables
    # --------------------------------------------------------
    C = build_controls(
        labels_df,
        use_first_belief_control=args.use_first_belief_control,
    )

    # --------------------------------------------------------
    # 9.6 Store hidden states in a memory-mapped array
    # --------------------------------------------------------
    print("Building hidden-state memmap...")
    H, meta = build_hidden_state_memmap(
        sample_ids=sample_ids,
        path_by_id=path_by_id,
        output_dir=output_dir,
        skip_first_layer=args.skip_first_layer,
    )

    num_layers = meta["num_layers"]
    hidden_dim = meta["hidden_dim"]

    print(f"num_layers = {num_layers}")
    print(f"hidden_dim  = {hidden_dim}")

    # --------------------------------------------------------
    # 9.7 Compute probe scores and run mediation for each layer
    # --------------------------------------------------------
    all_results = []
    probe_score_table = pd.DataFrame({
        "sample_id": sample_ids,
        "T_fallacy": T,
        "Y_belief_changed": Y,
        "D_detection": D,
    })

    for layer in tqdm(range(num_layers), desc="Layer-wise mediation"):
        # ----------------------------------------------------
        # 9.7.1 Residual stream at layer l
        # ----------------------------------------------------
        X_l = np.asarray(H[:, layer, :], dtype=np.float32)

        # ----------------------------------------------------
        # 9.7.2 Compute out-of-fold detection probe scores
        # ----------------------------------------------------
        M_l, probe_auc, probe_acc = compute_out_of_fold_probe_scores(
            X=X_l,
            y=D,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )

        # ----------------------------------------------------
        # 9.7.3 Standardize mediator M_l
        # ----------------------------------------------------
        # Probe score scales differ across layers, so
        # standardize to zero mean and unit variance before mediation.
        M_l_std = (M_l - M_l.mean()) / (M_l.std() + 1e-8)

        # Add probe scores to the output table.
        probe_score_table[f"M_layer_{layer}"] = M_l_std

        # ----------------------------------------------------
        # 9.7.4 statistical mediation
        # ----------------------------------------------------
        res = fit_mediation(
            T=T,
            M=M_l_std,
            Y=Y,
            C=C,
        )

        # ----------------------------------------------------
        # 9.7.5 bootstrap CI
        # ----------------------------------------------------
        if args.bootstrap > 0:
            ci = bootstrap_mediation(
                T=T,
                M=M_l_std,
                Y=Y,
                C=C,
                n_boot=args.bootstrap,
                random_state=args.random_state + layer,
            )
            res.update(ci)

        # ----------------------------------------------------
        # 9.7.6 Save results
        # ----------------------------------------------------
        res.update({
            "layer": layer,
            "probe_auc": probe_auc,
            "probe_acc": probe_acc,
            "n_samples": len(T),
            "n_fallacy": int(T.sum()),
            "n_no_fallacy": int((1 - T).sum()),
            "n_detect": int(D.sum()),
            "n_not_detect": int((1 - D).sum()),
            "n_changed": int(Y.sum()),
            "n_unchanged": int((1 - Y).sum()),
        })

        all_results.append(res)

    # --------------------------------------------------------
    # 9.8 Save results
    # --------------------------------------------------------
    results_df = pd.DataFrame(all_results)

    results_path = output_dir / "layerwise_statistical_mediation_results.csv"
    scores_path = output_dir / "out_of_fold_probe_scores_by_sample.csv"

    results_df.to_csv(results_path, index=False)
    probe_score_table.to_csv(scores_path, index=False)

    print("\nSaved:")
    print(f"  {results_path}")
    print(f"  {scores_path}")

    print("\nTop layers by |ACME_avg|:")
    print(
        results_df
        .assign(abs_ACME=lambda x: x["ACME_avg"].abs())
        .sort_values("abs_ACME", ascending=False)
        .head(10)[
            [
                "layer",
                "probe_auc",
                "probe_acc",
                "ACME_avg",
                "ADE_avg",
                "TE",
                "mediated_prop",
            ]
        ]
    )


if __name__ == "__main__":
    main()
