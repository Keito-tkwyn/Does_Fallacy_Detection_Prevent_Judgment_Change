import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


# ============================================================
# 1. Label processing
# ============================================================


def normalize_detection_label(x: object) -> int:
    """Convert Detect / Not Detect to 1 / 0."""
    v = str(x).strip().lower()

    if v in {"detect", "detected", "yes", "true", "1"}:
        return 1
    if v in {"not detect", "not_detect", "not detected", "no", "false", "0"}:
        return 0

    raise ValueError(f"Unknown detection label: {x!r}")


def normalize_boolq_answer(x: object) -> str:
    """Normalize BoolQ gold answers and model outputs to true / false."""
    v = str(x).strip().lower()
    v = v.strip("`'\"{}[]() ")

    if v in {"true", "1", "yes"}:
        return "true"
    if v in {"false", "0", "no"}:
        return "false"

    # Extract a standalone True/False token even if additional formatting is present.
    match = re.search(r"\b(true|false)\b", v)
    if match is not None:
        return match.group(1)

    raise ValueError(f"Could not parse the BoolQ answer: {x!r}")


def normalize_mmlu_answer(x: object) -> str:
    """
    Normalize MMLU gold answers and model outputs to a / b / c / d. Map zero-based
    integer labels 0, 1, 2, 3 to a, b, c, d, respectively.
    """
    if pd.isna(x):
        raise ValueError("The MMLU answer is missing.")

    if isinstance(x, (int, np.integer)):
        value = int(x)
        if 0 <= value <= 3:
            return "abcd"[value]

    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        value = int(x)
        if 0 <= value <= 3:
            return "abcd"[value]

    v = str(x).strip().lower()
    v = v.strip("`'\"{}[]() .,:;\n\t")

    if v in {"a", "b", "c", "d"}:
        return v

    # Handle numbers stored as strings.
    if v in {"0", "1", "2", "3"}:
        return "abcd"[int(v)]

    # Accept outputs containing extra text, such as "answer: a".
    match = re.search(r"(?:answer\s*[:=]?\s*)?\b([abcd])\b", v)
    if match is not None:
        return match.group(1)

    raise ValueError(f"Could not parse the MMLU answer: {x!r}")


def normalize_task_answer(x: object, dataset: str) -> str:
    if dataset == "boolq":
        return normalize_boolq_answer(x)
    if dataset == "mmlu":
        return normalize_mmlu_answer(x)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _drop_possible_header(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Remove a header row only if one exists in a CSV read with header=None."""
    if df.empty:
        return df

    first = str(df.iloc[0][label_column]).strip().lower()
    if first in {"label", "answer", "target", "gold_label", "correct_answer"}:
        return df.iloc[1:].reset_index(drop=True)
    return df


def load_gold_labels(
    dataset: str,
    gold_path: str,
    question_start: int,
    num_questions: int,
    gold_label_column: Optional[str],
) -> list[str]:
    """
    Read gold labels for the question subset used in the experiment from a BoolQ CSV or
    MMLU JSONL file. Assume sample_id % num_questions maps to the question's position in
    the selected list.
    """
    if question_start < 0:
        raise ValueError("--question-offset must be nonnegative.")
    if num_questions <= 0:
        raise ValueError("--question-limit must be at least 1.")

    if dataset == "boolq":
        gold_df = pd.read_csv(
            gold_path,
            header=None,
            names=["question", "gold_label"],
        )
        gold_df = _drop_possible_header(gold_df, "gold_label")
        label_column = "gold_label"

    elif dataset == "mmlu":
        gold_df = pd.read_json(gold_path, lines=True, orient="records")

        if gold_label_column is not None:
            if gold_label_column not in gold_df.columns:
                raise ValueError(
                    f"Column --gold-label-column={gold_label_column!r} was not found in the JSONL file."
                    f" Available columns: {list(gold_df.columns)}"
                )
            label_column = gold_label_column
        else:
            candidates = [
                "answer",
                "label",
                "target",
                "gold_label",
                "correct_answer",
            ]
            label_column = next(
                (column for column in candidates if column in gold_df.columns),
                None,
            )
            if label_column is None:
                raise ValueError(
                    "Could not automatically identify the gold-label column in the MMLU JSONL file."
                    f" Available columns: {list(gold_df.columns)}\n"
                    "Specify the gold-label column with --gold-label-column."
                )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    end = question_start + num_questions
    selected = gold_df.iloc[question_start:end].copy()

    if len(selected) != num_questions:
        raise ValueError(
            f"Could not retrieve the requested number of gold labels: "
            f"start={question_start}, requested={num_questions}, got={len(selected)}, "
            f"total={len(gold_df)}"
        )

    labels = [
        normalize_task_answer(value, dataset)
        for value in selected[label_column].tolist()
    ]

    return labels


def load_result_labels(
    csv_path: str,
    dataset: str,
    gold_path: str,
    question_start: int,
    num_questions: int,
    gold_label_column: Optional[str],
) -> pd.DataFrame:
    """
    Load result labels and align them with gold answers. Add sample_id, detection_label,
    first_answer, second_answer, gold_answer, first_correct, second_correct, and
    judgment_changed.
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["first_belief", "second_belief", "detection"],
    )

    if df.empty:
        raise ValueError(f"The result CSV is empty: {csv_path}")

    # Remove the header only if the result CSV contains one.
    first_cell = str(df.iloc[0]["first_belief"]).strip().lower()
    if first_cell in {"first_belief", "first_judgment", "first_answer"}:
        df = df.iloc[1:].reset_index(drop=True)

    df["sample_id"] = df.index.astype(int)
    df["detection_label"] = df["detection"].apply(normalize_detection_label)

    df["first_answer"] = df["first_belief"].apply(
        lambda value: normalize_task_answer(value, dataset)
    )
    df["second_answer"] = df["second_belief"].apply(
        lambda value: normalize_task_answer(value, dataset)
    )

    gold_labels = load_gold_labels(
        dataset=dataset,
        gold_path=gold_path,
        question_start=question_start,
        num_questions=num_questions,
        gold_label_column=gold_label_column,
    )

    df["question_index"] = df["sample_id"] % num_questions
    df["gold_answer"] = df["question_index"].map(dict(enumerate(gold_labels)))

    if df["gold_answer"].isna().any():
        bad_ids = df.loc[df["gold_answer"].isna(), "sample_id"].head(10).tolist()
        raise ValueError(f"Could not match gold labels to some sample IDs: {bad_ids}")

    df["first_correct"] = (df["first_answer"] == df["gold_answer"]).astype(int)
    df["second_correct"] = (df["second_answer"] == df["gold_answer"]).astype(int)
    df["judgment_changed"] = (df["first_answer"] != df["second_answer"]).astype(int)

    return df


# ============================================================
# 2. Load hidden states
# ============================================================


def extract_sample_id(path: Path) -> int:
    match = re.search(r"sample_(\d+)", str(path))
    if match is None:
        raise ValueError(f"Could not extract sample_id: {path}")
    return int(match.group(1))


def find_hidden_state_files(
    hidden_root: Path,
) -> Dict[int, Path]:
    """
    Locate persuasion-stage hidden states. Supported layouts include
    hidden_states_boolQ/sample_*/persuation/hidden_states.safetensors,
    hidden_states_boolQ/sample_*/persuasion/hidden_states_boolQ.safetensors, and
    hidden_states_MMLU/sample_*/persuasion/hidden_states.safetensors.
    """

    candidates = [
        "sample_*/persuation/hidden_states.safetensors",
        "sample_*/persuasion/hidden_states.safetensors",
        "sample_*/persuation/hidden_states_boolQ.safetensors",
        "sample_*/persuasion/hidden_states_boolQ.safetensors",
        "sample_*/persuation/hidden_states_MMLU.safetensors",
        "sample_*/persuasion/hidden_states_MMLU.safetensors",
    ]

    paths: list[Path] = []
    for pattern in candidates:
        paths.extend(hidden_root.glob(pattern))

    # Explicitly check for duplicate candidates for a sample instead of silently overwriting.
    path_by_id: Dict[int, Path] = {}
    for path in sorted(set(paths)):
        sample_id = extract_sample_id(path)
        if sample_id in path_by_id and path_by_id[sample_id] != path:
            raise ValueError(
                f"Multiple hidden-state files found for sample_{sample_id}:\n"
                f"  {path_by_id[sample_id]}\n  {path}"
            )
        path_by_id[sample_id] = path

    if not path_by_id:
        raise FileNotFoundError(f"Hidden states not found: {hidden_root}")

    return path_by_id


def _sort_key_by_layer_number(key: str) -> int:
    numbers = re.findall(r"\d+", key)
    return int(numbers[-1]) if numbers else 10**9


def tensor_to_final_token_vector(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()

    while x.ndim >= 3 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim == 1:
        return x.float().numpy().astype(np.float32)
    if x.ndim == 2:
        return x[-1].float().numpy().astype(np.float32)

    raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")


def load_final_token_all_layers(safetensors_path: Path) -> np.ndarray:
    tensors = load_file(str(safetensors_path), device="cpu")

    # All layers are stored in one tensor.
    if len(tensors) == 1:
        x = next(iter(tensors.values())).detach().cpu()

        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)

        if x.ndim == 3:
            if x.shape[0] <= 200:
                vectors = x[:, -1, :]
            elif x.shape[1] <= 200:
                vectors = x[-1, :, :]
            else:
                raise ValueError(f"Could not identify the layer dimension: {tuple(x.shape)}")
            return vectors.float().numpy().astype(np.float32)

        if x.ndim == 2:
            return x.float().numpy().astype(np.float32)

        raise ValueError(f"Unexpected hidden-state shape: {tuple(x.shape)}")

    # Layers are stored under separate keys such as layer_0, layer_1, ...
    layer_keys = sorted(tensors.keys(), key=_sort_key_by_layer_number)
    vectors = [tensor_to_final_token_vector(tensors[key]) for key in layer_keys]
    return np.stack(vectors, axis=0).astype(np.float32)


# ============================================================
# 3. Construct the robust vector
# ============================================================


def find_nonfinite_samples(sample_ids, path_by_id):
    """Inspect all hidden states and report samples containing NaN/Inf."""
    excluded = []
    for sample_id in tqdm(sample_ids, desc="Checking finite hidden states"):
        tensors = load_file(str(path_by_id[sample_id]), device="cpu")
        bad_keys = [key for key, tensor in tensors.items()
                    if not torch.isfinite(tensor).all()]
        if bad_keys:
            excluded.append({
                "sample_id": sample_id,
                "path": str(path_by_id[sample_id]),
                "nonfinite_keys": ";".join(bad_keys),
            })
    return pd.DataFrame(excluded, columns=["sample_id", "path", "nonfinite_keys"])


def compute_group_mean(
    sample_ids: Iterable[int],
    group_name: str,
    path_by_id: Dict[int, Path],
    num_layers: int,
    hidden_dim: int,
    skip_first_layer: bool,
) -> np.ndarray:
    sample_id_list = list(sample_ids)
    if not sample_id_list:
        raise ValueError(f"No samples found in group {group_name}.")

    group_sum = np.zeros((num_layers, hidden_dim), dtype=np.float64)

    for sample_id in tqdm(sample_id_list, desc=f"Accumulating {group_name}"):
        vectors = load_final_token_all_layers(path_by_id[sample_id])
        if skip_first_layer:
            vectors = vectors[1:]

        if vectors.shape != (num_layers, hidden_dim):
            raise ValueError(
                f"Shape mismatch for sample_{sample_id}: "
                f"{vectors.shape} != {(num_layers, hidden_dim)}"
            )

        if not np.isfinite(vectors).all():
            bad_layers = np.flatnonzero(~np.isfinite(vectors).all(axis=1)).tolist()
            raise ValueError(
                f"{group_name}: Hidden states for sample_{sample_id} contain NaN or Inf."
                f" layers={bad_layers}, path={path_by_id[sample_id]}。"
                "Recollect finite hidden states before constructing the vector."
            )

        group_sum += vectors.astype(np.float64)

    return group_sum / len(sample_id_list)


def build_robust_vectors(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading result and gold labels...")
    df = load_result_labels(
        csv_path=args.csv_path,
        dataset=args.dataset,
        gold_path=args.gold_path,
        question_start=args.question_start,
        num_questions=args.num_questions,
        gold_label_column=args.gold_label_column,
    )

    print("Finding hidden state files...")
    path_by_id = find_hidden_state_files(
        hidden_root=Path(args.hidden_root),
    )

    # Keep only samples with available hidden states.
    before_hidden_filter = len(df)
    df = df[df["sample_id"].isin(path_by_id.keys())].copy()
    print(f"Hidden-state filter: {before_hidden_filter} -> {len(df)}")

    # Optionally restrict to the fallacy condition.
    if args.fallacy_only:
        before_fallacy_filter = len(df)
        df = df[df["sample_id"] <= args.fallacy_last_id].copy()
        print(
            f"Fallacy-only filter (sample_id <= {args.fallacy_last_id}): "
            f"{before_fallacy_filter} -> {len(df)}"
        )

    # By default, use only samples with a correct First Judgment.
    if not args.include_first_incorrect:
        before_correct_filter = len(df)
        df = df[df["first_correct"] == 1].copy()
        print(f"First-correct filter: {before_correct_filter} -> {len(df)}")

    if df.empty:
        raise ValueError("No usable samples remain after filtering.")

    # Exclude nonfinite samples before grouping to keep means, counts, and saved lists consistent.
    excluded = find_nonfinite_samples(df["sample_id"].astype(int).tolist(), path_by_id)
    dataset_tag = "boolQ" if args.dataset == "boolq" else "MMLU"
    excluded_path = output_dir / f"robust_vector_excluded_{dataset_tag}.csv"
    excluded.to_csv(excluded_path, index=False)
    before_finite_filter = len(df)
    df = df[~df["sample_id"].isin(excluded["sample_id"])].copy()
    print(f"Finite hidden-state filter: {before_finite_filter} -> {len(df)} "
          f"(excluded: {len(excluded)})")
    print(f"Excluded sample report: {excluded_path}")
    if df.empty:
        raise ValueError("No usable samples remain after excluding samples with NaN or Inf.")

    # After restricting to initially correct samples, Second Judgment correctness determines maintenance or change.
    df_detect = df[df["detection_label"] == 1].copy()
    df_robust = df_detect[df_detect["second_correct"] == 1].copy()
    df_changed = df_detect[df_detect["second_correct"] == 0].copy()

    df_detect_all = df[df["detection_label"] == 1].copy()
    df_not_detect = df[df["detection_label"] == 0].copy()

    print("\nGroup sizes:")
    print(f"  Detect                         : {len(df_detect_all)}")
    print(f"  Not Detect                     : {len(df_not_detect)}")
    print(f"  Detect & First Correct & Second Correct   : {len(df_robust)}")
    print(f"  Detect & First Correct & Second Incorrect : {len(df_changed)}")

    if df_detect_all.empty:
        raise ValueError("No Detect samples found.")
    if df_not_detect.empty:
        raise ValueError("No Not Detect samples found. Cannot construct the detection direction.")
    if df_robust.empty:
        raise ValueError("No Detect & First Correct & Second Correct samples found.")
    if df_changed.empty:
        raise ValueError("No Detect & First Correct & Second Incorrect samples found.")

    first_sample_id = int(df_detect.iloc[0]["sample_id"])
    first_vectors = load_final_token_all_layers(path_by_id[first_sample_id])
    if args.skip_first_layer:
        first_vectors = first_vectors[1:]

    num_layers, hidden_dim = first_vectors.shape
    print("\nHidden state shape:")
    print(f"  num_layers = {num_layers}")
    print(f"  hidden_dim = {hidden_dim}")

    common_kwargs = {
        "path_by_id": path_by_id,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "skip_first_layer": args.skip_first_layer,
    }

    mean_robust = compute_group_mean(
        df_robust["sample_id"].astype(int).tolist(),
        "Detect & First Correct & Second Correct",
        **common_kwargs,
    )
    mean_changed = compute_group_mean(
        df_changed["sample_id"].astype(int).tolist(),
        "Detect & First Correct & Second Incorrect",
        **common_kwargs,
    )
    mean_detect = compute_group_mean(
        df_detect_all["sample_id"].astype(int).tolist(),
        "Detect",
        **common_kwargs,
    )
    mean_not_detect = compute_group_mean(
        df_not_detect["sample_id"].astype(int).tolist(),
        "Not Detect",
        **common_kwargs,
    )

    # Robust direction: maintained correct judgments minus judgments changed from correct to incorrect.
    robust_raw = mean_robust - mean_changed

    # Detection direction: Detect group minus Not Detect group.
    detect_raw = mean_detect - mean_not_detect

    # Remove the detection direction from the robust vector.
    dot = np.sum(robust_raw * detect_raw, axis=1, keepdims=True)
    detect_norm_sq = np.sum(detect_raw * detect_raw, axis=1, keepdims=True)
    projection = dot / (detect_norm_sq + 1e-8) * detect_raw
    robust_orth_raw = robust_raw - projection

    robust_norms = np.linalg.norm(robust_raw, axis=1, keepdims=True)
    detect_norms = np.linalg.norm(detect_raw, axis=1, keepdims=True)
    robust_orth_norms = np.linalg.norm(robust_orth_raw, axis=1, keepdims=True)

    robust_unit = robust_raw / (robust_norms + 1e-8)
    detect_unit = detect_raw / (detect_norms + 1e-8)
    robust_orth_unit = robust_orth_raw / (robust_orth_norms + 1e-8)

    cos_before = np.sum(robust_unit * detect_unit, axis=1)
    cos_after = np.sum(robust_orth_unit * detect_unit, axis=1)

    print("\nOrthogonalization check:")
    print(f"  mean cos(robust, detect) before = {cos_before.mean():.6f}")
    print(f"  mean cos(robust, detect) after  = {cos_after.mean():.6f}")

    dataset_tag = "boolQ" if args.dataset == "boolq" else "MMLU"
    save_path = output_dir / f"robust_vectors_{dataset_tag}.safetensors"

    tensors_to_save = {
        "robust_raw": torch.tensor(robust_raw, dtype=torch.float32),
        "robust_unit": torch.tensor(robust_unit, dtype=torch.float32),
        "detect_raw": torch.tensor(detect_raw, dtype=torch.float32),
        "detect_unit": torch.tensor(detect_unit, dtype=torch.float32),
        "robust_orth_raw": torch.tensor(robust_orth_raw, dtype=torch.float32),
        "robust_orth_unit": torch.tensor(robust_orth_unit, dtype=torch.float32),
        # Retain compatibility with legacy key names.
        "mean_detect_unchanged": torch.tensor(mean_robust, dtype=torch.float32),
        "mean_detect_changed": torch.tensor(mean_changed, dtype=torch.float32),
        # Use new key names that describe the definitions.
        "mean_detect_first_correct_second_correct": torch.tensor(
            mean_robust, dtype=torch.float32
        ),
        "mean_detect_first_correct_second_incorrect": torch.tensor(
            mean_changed, dtype=torch.float32
        ),
        "mean_detect": torch.tensor(mean_detect, dtype=torch.float32),
        "mean_not_detect": torch.tensor(mean_not_detect, dtype=torch.float32),
        "robust_norms": torch.tensor(robust_norms.squeeze(-1), dtype=torch.float32),
        "detect_norms": torch.tensor(detect_norms.squeeze(-1), dtype=torch.float32),
        "robust_orth_norms": torch.tensor(
            robust_orth_norms.squeeze(-1), dtype=torch.float32
        ),
        "cos_before": torch.tensor(cos_before, dtype=torch.float32),
        "cos_after": torch.tensor(cos_after, dtype=torch.float32),
    }

    metadata = {
        "dataset": args.dataset,
        "n_excluded_nonfinite": str(len(excluded)),
        "definition": (
            "v_robust = mean(H | Detect, FirstCorrect, SecondCorrect) - "
            "mean(H | Detect, FirstCorrect, SecondIncorrect)"
        ),
        "n_detect_first_correct_second_correct": str(len(df_robust)),
        "n_detect_first_correct_second_incorrect": str(len(df_changed)),
        "n_detect": str(len(df_detect_all)),
        "n_not_detect": str(len(df_not_detect)),
        "fallacy_only": str(args.fallacy_only),
        "fallacy_last_id": str(args.fallacy_last_id),
        "include_first_incorrect": str(args.include_first_incorrect),
        "question_start": str(args.question_start),
        "num_questions": str(args.num_questions),
        "skip_first_layer": str(args.skip_first_layer),
        "num_layers": str(num_layers),
        "hidden_dim": str(hidden_dim),
    }

    for key, tensor in tensors_to_save.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Output vector {key} contains NaN or Inf. Saving aborted.")

    save_file(tensors_to_save, str(save_path), metadata=metadata)

    norm_df = pd.DataFrame(
        {
            "layer": np.arange(num_layers),
            "robust_raw_norm": robust_norms.squeeze(-1),
            "detect_raw_norm": detect_norms.squeeze(-1),
            "robust_orth_raw_norm": robust_orth_norms.squeeze(-1),
            "cos_robust_detect_before": cos_before,
            "cos_robust_detect_after": cos_after,
        }
    )
    norm_csv_path = output_dir / f"robust_vector_norms_{dataset_tag}.csv"
    norm_df.to_csv(norm_csv_path, index=False)

    # Save the sample lists so the samples used can be inspected.
    group_csv_path = output_dir / f"robust_vector_groups_{dataset_tag}.csv"
    group_df = df[
        [
            "sample_id",
            "question_index",
            "first_answer",
            "second_answer",
            "gold_answer",
            "first_correct",
            "second_correct",
            "detection_label",
            "judgment_changed",
        ]
    ].copy()
    group_df["robust_group"] = "other"
    group_df.loc[group_df["sample_id"].isin(df_robust["sample_id"]), "robust_group"] = (
        "detect_first_correct_second_correct"
    )
    group_df.loc[group_df["sample_id"].isin(df_changed["sample_id"]), "robust_group"] = (
        "detect_first_correct_second_incorrect"
    )
    group_df.to_csv(group_csv_path, index=False)

    print("\nSaved:")
    print(f"  vectors : {save_path}")
    print(f"  norms   : {norm_csv_path}")
    print(f"  groups  : {group_csv_path}")


# ============================================================
# 4. CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construct steering vectors from saved hidden states and judgment/detection labels.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build",
        help="Build steering vectors from existing hidden states.",
    )

    build.add_argument(
        "--dataset",
        choices=["boolq", "mmlu"],
        required=True,
        help="Dataset type.",
    )
    build.add_argument("--hidden-root", type=Path, required=True, help="Directory containing sample_* folders.")
    build.add_argument("--label-csv", dest="csv_path", type=Path, required=True, help="Result CSV containing first_belief, second_belief, detection.")
    build.add_argument("--gold-path", type=Path, required=True, help="Question file with gold answers (BoolQ CSV or MMLU JSONL).")
    build.add_argument("--gold-label-column", default=None, help="Optional MMLU gold-answer column; otherwise inferred.")
    build.add_argument("--question-offset", dest="question_start", type=int, default=0, help="Zero-based offset in the gold question file.")
    build.add_argument("--question-limit", dest="num_questions", type=int, default=400, help="Number of questions per prompt condition.")
    build.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for steering vectors.",
    )
    build.add_argument(
        "--fallacy-last-id",
        type=int,
        default=5199,
        help="Last sample ID in the fallacy condition (inclusive).",
    )
    build.add_argument(
        "--use-all-samples",
        action="store_true",
        help="Include the no-fallacy control condition.",
    )
    build.add_argument(
        "--include-first-incorrect",
        action="store_true",
        help="Also include initially incorrect judgments; groups are still defined by second-answer correctness.",
    )
    build.add_argument(
        "--skip-first-layer",
        action="store_true",
        help="Exclude the embedding output from saved vector rows.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        args.fallacy_only = not args.use_all_samples
        build_robust_vectors(args)


if __name__ == "__main__":
    main()