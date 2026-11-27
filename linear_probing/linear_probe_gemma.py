import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


# ============================================================
# 1. Load hidden states from safetensors
# ============================================================

def take_last_token_vector(x: torch.Tensor) -> np.ndarray:
    """
    Extract a vector for linear probing from one layer's hidden states. For [seq_len,
    hidden_dim], use the final token's hidden state.
    """

    x = x.detach().cpu()
    x = x.squeeze()

    if x.ndim == 1:
        # [hidden_dim]
        return x.double().numpy()

    elif x.ndim == 2:
        # [seq_len, hidden_dim]
        return x[-1].double().numpy()

    else:
        raise ValueError(f"Unsupported tensor shape after squeeze: {tuple(x.shape)}")


def tensor_to_layer_vectors(t: torch.Tensor) -> dict[int, np.ndarray]:
    """
    Split an all-layer tensor into individual layer vectors. Supported shapes:
    [num_layers, hidden_dim], [num_layers, seq_len, hidden_dim], and [1, num_layers,
    seq_len, hidden_dim].
    """

    t = t.detach().cpu()

    # [1, num_layers, seq_len, hidden_dim] -> [num_layers, seq_len, hidden_dim]
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]

    # [num_layers, 1, seq_len, hidden_dim] -> [num_layers, seq_len, hidden_dim]
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]

    layer_vectors = {}

    if t.ndim == 2:
        # [num_layers, hidden_dim]
        for layer_idx in range(t.shape[0]):
            layer_vectors[layer_idx] = t[layer_idx].double().numpy()

    elif t.ndim == 3:
        # [num_layers, seq_len, hidden_dim]
        for layer_idx in range(t.shape[0]):
            layer_vectors[layer_idx] = take_last_token_vector(t[layer_idx])

    else:
        raise ValueError(f"Unsupported all-layer tensor shape: {tuple(t.shape)}")

    return layer_vectors


def load_hidden_states_by_layer(path: Path) -> dict[int, np.ndarray]:
    """
    Load hidden_states.safetensors and return a dictionary mapping layer indices to
    hidden-state vectors.
    """

    tensors = load_file(str(path))

    # ------------------------------------------------------------
    # Case 1: all layers are stored under a single key.
    # Example: hidden_states: [num_layers, seq_len, hidden_dim]
    # ------------------------------------------------------------
    if len(tensors) == 1:
        key = list(tensors.keys())[0]
        return tensor_to_layer_vectors(tensors[key])

    # ------------------------------------------------------------
    # Case 2: layers are stored under separate keys.
    # Example: layer_0, layer_1, ...
    # ------------------------------------------------------------
    layer_vectors = {}
    layer_pattern = re.compile(r"(\d+)$")

    for key, tensor in tensors.items():
        match = layer_pattern.search(key)
        if match is None:
            continue

        layer_idx = int(match.group(1))
        layer_vectors[layer_idx] = take_last_token_vector(tensor)

    if len(layer_vectors) == 0:
        raise ValueError(
            f"Could not infer layer keys from {path}. "
            f"Available keys: {list(tensors.keys())}"
        )

    return dict(sorted(layer_vectors.items()))


# ============================================================
# 2. Extract hidden states for each sample
# ============================================================

def get_hidden_path(sample_dir: Path, state_name: str) -> Path:
    """
    Construct the hidden_states.safetensors path within sample_x. state_name may be
    first_belief or persuasion. Prefer 'persuasion'; also accept the legacy spelling
    'persuation'.
    """

    path = sample_dir / state_name / "hidden_states.safetensors"

    if path.exists():
        return path

    if state_name == "persuasion":
        alt_path = sample_dir / "persuation" / "hidden_states.safetensors"
        if alt_path.exists():
            return alt_path

    raise FileNotFoundError(f"hidden_states.safetensors not found: {path}")


def load_sample_hidden_by_layer(sample_dir: Path, state_name: str) -> dict[int, np.ndarray]:
    """
    Load a sample's hidden states for the specified state without computing differences.
    For example, state_name='persuasion' uses
    sample_x/persuasion/hidden_states.safetensors.
    """

    hidden_path = get_hidden_path(sample_dir, state_name)
    hidden_by_layer = load_hidden_states_by_layer(hidden_path)

    return hidden_by_layer


# ============================================================
# 3. Construct labels from the CSV
# ============================================================

def normalize_text(x) -> str:
    """
    Normalize label formatting.
    """

    if pd.isna(x):
        return ""

    return str(x).strip().lower()


def make_labels(df: pd.DataFrame, target: str) -> np.ndarray:
    """
    Construct target labels y from the CSV. For target='belief_change', return 1 if
    first_belief differs from second_belief, otherwise 0. For target='detection',
    Detect=1 and Not Detect=0.
    """

    if target == "belief_change":
        first = df["first_belief"].map(normalize_text)
        second = df["second_belief"].map(normalize_text)

        y = (first != second).astype(int).values
        return y

    elif target == "detection":
        det = df["detection"].map(normalize_text)

        label_map = {
            "detect": 1,
            "detected": 1,
            "yes": 1,
            "true": 1,
            "1": 1,
            "not detect": 0,
            "not_detect": 0,
            "not-detect": 0,
            "not detected": 0,
            "no": 0,
            "false": 0,
            "0": 0,
        }

        y = det.map(label_map)

        if y.isna().any():
            bad_values = sorted(det[y.isna()].unique())
            raise ValueError(
                f"Unknown detection labels: {bad_values}. "
                "Expected Detect / Not Detect."
            )

        return y.astype(int).values

    else:
        raise ValueError(f"Unknown target: {target}")


# ============================================================
# 4. Load all samples and construct a feature matrix X for each layer
# ============================================================

def natural_sample_id(path: Path) -> int:
    """
    Sort sample_0, sample_1, ... by numeric index.
    """

    match = re.search(r"sample_(\d+)", path.name)

    if match is None:
        return 10**18

    return int(match.group(1))


def build_features_by_layer(
    hidden_root: Path,
    sample_ids: np.ndarray,
    state_name: str,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Construct a feature matrix X for each layer from hidden-state directories. Use
    hidden states from the specified state directly rather than differences. For
    example, with state_name='persuasion', each row contains a sample's persuasion
    hidden state. Return a dictionary mapping layer indices to matrices of shape
    [num_samples, hidden_dim].
    """

    sample_dirs = sorted(
        [p for p in hidden_root.glob("sample_*") if p.is_dir()],
        key=natural_sample_id,
    )

    sample_dir_map = {natural_sample_id(p): p for p in sample_dirs}
    requested_sample_ids = np.asarray(sample_ids, dtype=np.int64)

    missing_sample_ids = [
        int(sample_id)
        for sample_id in requested_sample_ids
        if int(sample_id) not in sample_dir_map
    ]
    if missing_sample_ids:
        preview = missing_sample_ids[:20]
        raise ValueError(
            "Some sample IDs have no hidden states: "
            f"{preview}"
            + (" ..." if len(missing_sample_ids) > len(preview) else "")
        )

    # Match exactly the sample IDs used in the CSV.
    # Selecting the first N samples would misalign labels after excluding intermediate samples.
    sample_dirs = [sample_dir_map[int(sample_id)] for sample_id in requested_sample_ids]

    features = defaultdict(list)

    for i, sample_dir in enumerate(sample_dirs):
        hidden_by_layer = load_sample_hidden_by_layer(
            sample_dir=sample_dir,
            state_name=state_name,
        )

        for layer_idx, hidden_vec in hidden_by_layer.items():
            features[layer_idx].append(hidden_vec)

        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1}/{len(sample_dirs)} samples")

    features_by_layer = {}

    for layer_idx, vecs in features.items():
        features_by_layer[layer_idx] = np.stack(vecs, axis=0)

    return dict(sorted(features_by_layer.items())), requested_sample_ids


# ============================================================
# 5. Detect nonfinite values and apply exclusions across layers
# ============================================================

def clean_features_globally(
    features_by_layer: dict[int, np.ndarray],
    sample_ids: np.ndarray,
    nonfinite_policy: str = "drop_layers",
) -> tuple[
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    list[dict],
    list[int],
]:
    """
    Convert all hidden states to float64 and detect NaN/Inf. With
    nonfinite_policy='drop_layers' (the Gemma default), exclude any layer containing
    nonfinite values and retain all samples in valid layers. With
    nonfinite_policy='drop_samples', exclude samples containing nonfinite values in any
    layer consistently across all layers.
    """

    if nonfinite_policy not in {"drop_layers", "drop_samples"}:
        raise ValueError(
            "nonfinite_policy must be 'drop_layers' or 'drop_samples', "
            f"but got {nonfinite_policy!r}"
        )

    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    n_samples = len(sample_ids)
    global_valid_mask = np.ones(n_samples, dtype=bool)
    diagnostics = []
    converted_features = {}
    invalid_layers = []

    for layer_idx, X in features_by_layer.items():
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                f"Layer {layer_idx}: X must be 2-dimensional, "
                f"but got shape={X.shape}"
            )
        if X.shape[0] != n_samples:
            raise ValueError(
                f"Layer {layer_idx}: sample count mismatch: "
                f"X={X.shape[0]}, sample_ids={n_samples}"
            )

        converted_features[layer_idx] = X
        finite_element_mask = np.isfinite(X)
        layer_valid_mask = finite_element_mask.all(axis=1)

        invalid_row_indices = np.flatnonzero(~layer_valid_mask)
        for row_idx in invalid_row_indices:
            row = X[row_idx]
            finite_values = row[np.isfinite(row)]
            finite_max_abs = (
                float(np.max(np.abs(finite_values)))
                if finite_values.size > 0
                else np.nan
            )
            diagnostics.append(
                {
                    "layer": int(layer_idx),
                    "sample_id": int(sample_ids[row_idx]),
                    "nan_count": int(np.isnan(row).sum()),
                    "posinf_count": int(np.isposinf(row).sum()),
                    "neginf_count": int(np.isneginf(row).sum()),
                    "finite_max_abs": finite_max_abs,
                }
            )

        if invalid_row_indices.size > 0:
            invalid_layers.append(int(layer_idx))
            invalid_ids = sample_ids[invalid_row_indices]
            preview = invalid_ids[:20].tolist()
            print(
                f"Layer {layer_idx}: found {invalid_row_indices.size} sample(s) "
                f"containing NaN/Inf. sample_id={preview}"
                + (" ..." if invalid_row_indices.size > len(preview) else "")
            )

            if nonfinite_policy == "drop_layers":
                print(
                    f"Skipping layer {layer_idx} because it contains "
                    "non-finite hidden-state values."
                )

        if nonfinite_policy == "drop_samples":
            global_valid_mask &= layer_valid_mask

    if nonfinite_policy == "drop_layers":
        cleaned_features = {
            layer_idx: X
            for layer_idx, X in converted_features.items()
            if layer_idx not in invalid_layers
        }

        if not cleaned_features:
            raise ValueError(
                "Every layer contains NaN or Inf. No layer can be probed; "
                "regenerate the hidden states in float32."
            )

        if invalid_layers:
            print(
                "Skipped layers containing NaN/Inf: "
                f"{invalid_layers}. Retained {len(cleaned_features)} clean layer(s)."
            )

        return (
            cleaned_features,
            sample_ids,
            global_valid_mask,
            diagnostics,
            invalid_layers,
        )

    removed_indices = np.flatnonzero(~global_valid_mask)
    removed_sample_ids = sample_ids[removed_indices]

    if removed_indices.size > 0:
        preview = removed_sample_ids[:20].tolist()
        print(
            f"Globally removing {removed_indices.size} sample(s) from every layer: "
            f"sample_id={preview}"
            + (" ..." if removed_indices.size > len(preview) else "")
        )

    if not global_valid_mask.any():
        raise ValueError("All samples contain NaN or Inf in at least one layer.")

    cleaned_features = {
        layer_idx: X[global_valid_mask]
        for layer_idx, X in converted_features.items()
    }
    cleaned_sample_ids = sample_ids[global_valid_mask]

    for layer_idx, X in cleaned_features.items():
        if not np.isfinite(X).all():
            raise ValueError(
                f"Layer {layer_idx}: non-finite values remain after cleaning."
            )

    return (
        cleaned_features,
        cleaned_sample_ids,
        global_valid_mask,
        diagnostics,
        invalid_layers,
    )


def save_invalid_diagnostics(
    diagnostics: list[dict],
    output_dir: Path,
    state_name: str,
):
    """Save the layers, sample IDs, and counts of nonfinite values to CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = output_dir / f"{state_name}_invalid_hidden_states.csv"
    diagnostic_columns = [
        "layer",
        "sample_id",
        "nan_count",
        "posinf_count",
        "neginf_count",
        "finite_max_abs",
    ]
    diagnostic_df = pd.DataFrame(diagnostics, columns=diagnostic_columns)
    diagnostic_df.to_csv(diagnostic_path, index=False)
    print(f"Saved invalid-value diagnostics to: {diagnostic_path}")


# ============================================================
# 6. Train and evaluate a linear probe for each layer
# ============================================================

def evaluate_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    max_iter: int = 2000,
) -> dict:
    """
    Train a linear probe for the feature matrix X of a single layer and evaluate it by
    cross-validation. The model is StandardScaler followed by LogisticRegression. Test
    whether judgment-change or fallacy-detection labels are linearly decodable from that
    layer's hidden states.
    """

    unique, counts = np.unique(y, return_counts=True)

    if len(unique) < 2:
        raise ValueError("Only one class exists in labels.")

    min_class_count = counts.min()

    if min_class_count < n_splits:
        n_splits = int(min_class_count)

    if n_splits < 2:
        raise ValueError("Not enough samples for cross validation.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
    }

    for train_idx, test_idx in cv.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=max_iter,
                random_state=random_state,
            ),
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="binary",
            zero_division=0,
        )

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)

        try:
            y_score = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_score)
            metrics["auc"].append(auc)
        except ValueError:
            pass

    result = {}

    for name, values in metrics.items():
        if len(values) == 0:
            result[f"{name}_mean"] = np.nan
            result[f"{name}_std"] = np.nan
        else:
            result[f"{name}_mean"] = float(np.mean(values))
            result[f"{name}_std"] = float(np.std(values))

    result["n_splits"] = n_splits
    result["n_samples"] = int(X.shape[0])
    result["hidden_dim"] = int(X.shape[1])

    return result


def run_probe(
    features_by_layer: dict[int, np.ndarray],
    y: np.ndarray,
    target: str,
    state_name: str,
    output_dir: Path,
    n_splits: int,
    random_state: int,
    max_iter: int,
    n_samples_before_cleaning: int,
):
    """
    Run linear probing for all layers and save results to CSV.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for layer_idx, X in features_by_layer.items():
        print(
            f"Running probe: target={target}, "
            f"state={state_name}, layer={layer_idx}, X={X.shape}"
        )

        try:
            X = np.asarray(X, dtype=np.float64)
            if not np.isfinite(X).all():
                raise ValueError(
                    "NaN/Inf remains. clean_features_globally must be called first."
                )

            result = evaluate_linear_probe(
                X=X,
                y=y,
                n_splits=n_splits,
                random_state=random_state,
                max_iter=max_iter,
            )

            result["layer"] = layer_idx
            result["target"] = target
            result["state"] = state_name
            result["n_samples_before_cleaning"] = int(n_samples_before_cleaning)
            result["n_removed_nonfinite"] = int(
                n_samples_before_cleaning - X.shape[0]
            )
            result["n_samples_after_cleaning"] = int(X.shape[0])
            rows.append(result)

        except ValueError as e:
            print(f"Skipped layer {layer_idx}: {e}")

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("layer")

    output_path = output_dir / f"{target}_{state_name}_linear_probe_results.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Saved results to: {output_path}")


# ============================================================
# 7. Main procedure
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden-root", type=Path, required=True, help="Directory containing sample_* hidden-state directories.")

    parser.add_argument("--label-csv", "--label_csv", type=Path, required=True, help="Headerless CSV: first_belief, second_belief, detection.")

    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for probe results.")

    parser.add_argument(
        "--state",
        type=str,
        default="persuasion",
        choices=["first_belief", "persuasion"],
        help="Which hidden states to use for probing.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="both",
        choices=["belief_change", "detection", "both"],
        help="Which label to probe.",
    )

    parser.add_argument("--n-splits", "--n_splits", type=int, default=5, help="Number of stratified cross-validation folds.")

    parser.add_argument("--random-state", "--random_state", type=int, default=42)

    parser.add_argument("--max-iter", "--max_iter", type=int, default=2000)

    parser.add_argument(
        "--nonfinite-policy", "--nonfinite_policy",
        type=str,
        default="drop_layers",
        choices=["drop_layers", "drop_samples"],
        help=(
            "How to handle NaN/Inf hidden states. 'drop_layers' skips affected "
            "layers and is recommended for Gemma; 'drop_samples' removes affected "
            "samples from every layer."
        ),
    )

    args = parser.parse_args()

    hidden_root = args.hidden_root
    output_dir = args.output_dir

    # ------------------------------------------------------------
    # Load the CSV.
    # ------------------------------------------------------------
    df = pd.read_csv(
        args.label_csv,
        names=["first_belief", "second_belief", "detection"],
    )

    # Retain the original CSV row number as sample_id.
    # Resetting the index after filtering would break the sample mapping,
    # so create the sample_id column before filtering.
    df["sample_id"] = df.index.astype(int)

    original_count = len(df)


    print(f"Loaded label CSV: {args.label_csv}")
    print(f"Original number of rows: {original_count}")
    print(f"Number of usable rows: {len(df)}")

    if len(df) == 0:
        raise ValueError("No usable samples remain after filtering.")

    # ------------------------------------------------------------
    # Load the hidden states directly as features.
    # ------------------------------------------------------------
    print("Building features from hidden states directly")
    print(f"State: {args.state}")

    features_by_layer, loaded_sample_ids = build_features_by_layer(
        hidden_root=hidden_root,
        sample_ids=df["sample_id"].to_numpy(),
        state_name=args.state,
    )

    print(f"Loaded layers: {list(features_by_layer.keys())}")

    # ------------------------------------------------------------
    # Detect NaN/Inf values and exclude affected samples consistently across all layers.
    # ------------------------------------------------------------
    n_samples_before_cleaning = len(df)
    (
        features_by_layer,
        loaded_sample_ids,
        global_valid_mask,
        invalid_diagnostics,
        skipped_nonfinite_layers,
    ) = clean_features_globally(
        features_by_layer=features_by_layer,
        sample_ids=loaded_sample_ids,
        nonfinite_policy=args.nonfinite_policy,
    )

    # Align the retained hidden-state samples with the label rows.
    df = df.iloc[np.flatnonzero(global_valid_mask)].copy()

    if not np.array_equal(
        df["sample_id"].to_numpy(dtype=np.int64),
        loaded_sample_ids,
    ):
        raise ValueError("Sample IDs in the label CSV and hidden states do not match after cleaning.")

    save_invalid_diagnostics(
        diagnostics=invalid_diagnostics,
        output_dir=output_dir,
        state_name=args.state,
    )

    print(
        "Samples after global cleaning: "
        f"{len(df)}/{n_samples_before_cleaning}"
    )
    if skipped_nonfinite_layers:
        print(f"Layers excluded from probing: {skipped_nonfinite_layers}")

    # ------------------------------------------------------------
    # Select the target labels to evaluate.
    # ------------------------------------------------------------
    if args.target == "both":
        targets = ["belief_change", "detection"]
    else:
        targets = [args.target]

    # ------------------------------------------------------------
    # Run linear probing for each target.
    # ------------------------------------------------------------
    for target in targets:
        y = make_labels(df, target)

        print("=" * 80)
        print(f"Target: {target}")
        print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        run_probe(
            features_by_layer=features_by_layer,
            y=y,
            target=target,
            state_name=args.state,
            output_dir=output_dir,
            n_splits=args.n_splits,
            random_state=args.random_state,
            max_iter=args.max_iter,
            n_samples_before_cleaning=n_samples_before_cleaning,
        )


if __name__ == "__main__":
    main()
