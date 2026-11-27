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
# 1. safetensors から hidden states を読み込む部分
# ============================================================

def take_last_token_vector(x: torch.Tensor) -> np.ndarray:
    """
    1つの層の hidden states から、線形プローブに使うベクトルを取り出す。

    hidden states が [seq_len, hidden_dim] の場合は、
    最後の token の hidden state を使う。
    """

    x = x.detach().cpu()
    x = x.squeeze()

    if x.ndim == 1:
        # [hidden_dim]
        return x.float().numpy()

    elif x.ndim == 2:
        # [seq_len, hidden_dim]
        return x[-1].float().numpy()

    else:
        raise ValueError(f"Unsupported tensor shape after squeeze: {tuple(x.shape)}")


def tensor_to_layer_vectors(t: torch.Tensor) -> dict[int, np.ndarray]:
    """
    1つの tensor に全層が保存されている場合、
    層ごとのベクトルに分解する。

    想定する形:
    - [num_layers, hidden_dim]
    - [num_layers, seq_len, hidden_dim]
    - [1, num_layers, seq_len, hidden_dim]
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
            layer_vectors[layer_idx] = t[layer_idx].float().numpy()

    elif t.ndim == 3:
        # [num_layers, seq_len, hidden_dim]
        for layer_idx in range(t.shape[0]):
            layer_vectors[layer_idx] = take_last_token_vector(t[layer_idx])

    else:
        raise ValueError(f"Unsupported all-layer tensor shape: {tuple(t.shape)}")

    return layer_vectors


def load_hidden_states_by_layer(path: Path) -> dict[int, np.ndarray]:
    """
    hidden_states.safetensors を読み込み、
    層ごとの hidden state ベクトルを返す。

    戻り値:
    {
        0: layer0_vector,
        1: layer1_vector,
        ...
    }
    """

    tensors = load_file(str(path))

    # ------------------------------------------------------------
    # ケース1: 1つの key に全層が入っている場合
    # 例: hidden_states: [num_layers, seq_len, hidden_dim]
    # ------------------------------------------------------------
    if len(tensors) == 1:
        key = list(tensors.keys())[0]
        return tensor_to_layer_vectors(tensors[key])

    # ------------------------------------------------------------
    # ケース2: keyごとに層が分かれている場合
    # 例: layer_0, layer_1, ...
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
# 2. sampleごとに単純な hidden states を取り出す部分
# ============================================================

def get_hidden_path(sample_dir: Path, state_name: str) -> Path:
    """
    sample_x の中から hidden_states.safetensors のパスを作る。

    state_name:
    - first_belief
    - persuasion

    注意:
    Prefer 'persuasion'; also accept the legacy spelling 'persuation'.
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
    1つの sample について、指定した状態の hidden states を読み込む。

    ここでは差分は取らない。

    例:
        state_name='persuasion'
        -> sample_x/persuasion/hidden_states.safetensors を使う
    """

    hidden_path = get_hidden_path(sample_dir, state_name)
    hidden_by_layer = load_hidden_states_by_layer(hidden_path)

    return hidden_by_layer


# ============================================================
# 3. CSVからラベルを作る部分
# ============================================================

def normalize_text(x) -> str:
    """
    ラベルの表記ゆれを減らす。
    """

    if pd.isna(x):
        return ""

    return str(x).strip().lower()


def make_labels(df: pd.DataFrame, target: str) -> np.ndarray:
    """
    CSVから教師ラベル y を作る。

    target='belief_change':
        first_belief と second_belief が違うなら 1
        同じなら 0

    target='detection':
        Detect なら 1
        Not Detect なら 0
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
# 4. 全sampleを読み込み、層ごとの特徴量行列 X を作る部分
# ============================================================

def natural_sample_id(path: Path) -> int:
    """
    sample_0, sample_1, ... を数値順に並べる。
    """

    match = re.search(r"sample_(\d+)", path.name)

    if match is None:
        return 10**18

    return int(match.group(1))


def build_features_by_layer(
    hidden_root: Path,
    num_samples: int,
    state_name: str,
) -> dict[int, np.ndarray]:
    """
    hidden states のディレクトリから、層ごとの特徴量 X を作る。

    今回は差分ではなく、指定した state の hidden states そのものを使う。

    例:
        state_name='persuasion'

        X_layer = [
            sample_0 の persuasion hidden state,
            sample_1 の persuasion hidden state,
            ...
        ]

    戻り値:
    {
        0: X_layer0,  # shape = [num_samples, hidden_dim]
        1: X_layer1,
        ...
    }
    """

    sample_dirs = sorted(
        [p for p in hidden_root.glob("sample_*") if p.is_dir()],
        key=natural_sample_id,
    )

    if len(sample_dirs) < num_samples:
        raise ValueError(
            f"Number of sample dirs is smaller than CSV rows: "
            f"{len(sample_dirs)} < {num_samples}"
        )

    # CSVの行数に合わせる
    sample_dirs = sample_dirs[:num_samples]
    if [natural_sample_id(p) for p in sample_dirs] != list(range(num_samples)):
        raise ValueError("Sample IDs must match the zero-based CSV row indices.")

    features = defaultdict(list)

    for i, sample_dir in enumerate(sample_dirs):
        hidden_by_layer = load_sample_hidden_by_layer(
            sample_dir=sample_dir,
            state_name=state_name,
        )

        for layer_idx, hidden_vec in hidden_by_layer.items():
            features[layer_idx].append(hidden_vec)

        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1}/{num_samples} samples")

    features_by_layer = {}

    for layer_idx, vecs in features.items():
        features_by_layer[layer_idx] = np.stack(vecs, axis=0)

    return dict(sorted(features_by_layer.items()))


# ============================================================
# 5. 各層で線形プローブを学習・評価する部分
# ============================================================

def evaluate_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    max_iter: int = 2000,
) -> dict:
    """
    1つの層の特徴量 X に対して線形プローブを学習し、
    交差検証で評価する。

    モデル:
        StandardScaler + LogisticRegression

    ここで見ていること:
        その層の hidden state から、
        信念変化や誤謬検知のラベルが線形に読み出せるか。
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
):
    """
    全層について線形プローブを実行し、結果をCSVに保存する。
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for layer_idx, X in features_by_layer.items():
        print(
            f"Running probe: target={target}, "
            f"state={state_name}, layer={layer_idx}, X={X.shape}"
        )

        try:
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
            rows.append(result)

        except ValueError as e:
            print(f"Skipped layer {layer_idx}: {e}")

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("layer")

    output_path = output_dir / f"{target}_{state_name}_linear_probe_results.csv"
    result_df.to_csv(output_path, index=False)

    print(f"Saved results to: {output_path}")


# ============================================================
# 6. 実行部分
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
        default="detection",
        choices=["belief_change", "detection", "both"],
        help="Which label to probe.",
    )

    parser.add_argument("--n-splits", "--n_splits", type=int, default=5, help="Number of stratified cross-validation folds.")

    parser.add_argument("--random-state", "--random_state", type=int, default=42)

    parser.add_argument("--max-iter", "--max_iter", type=int, default=2000)

    args = parser.parse_args()

    hidden_root = args.hidden_root
    output_dir = args.output_dir

    # ------------------------------------------------------------
    # CSV読み込み
    # ------------------------------------------------------------
    df = pd.read_csv(
        args.label_csv,
        names=["first_belief", "second_belief", "detection"],
    )

    # CSVの元の行番号をsample_idとして保持する。
    # フィルタ後にreset_indexしてしまうとsample番号との対応が崩れるため、
    # sample_id列を作ってから除外する。
    df["sample_id"] = df.index.astype(int)

    original_count = len(df)


    print(f"Loaded label CSV: {args.label_csv}")
    print(f"Original number of rows: {original_count}")
    print(f"Number of usable rows: {len(df)}")

    if len(df) == 0:
        raise ValueError("除外処理後、使用可能なサンプルがありません。")

    # ------------------------------------------------------------
    # hidden states そのものを特徴量として読み込む
    # ------------------------------------------------------------
    print("Building features from hidden states directly")
    print(f"State: {args.state}")

    features_by_layer = build_features_by_layer(
        hidden_root=hidden_root,
        num_samples=len(df),
        state_name=args.state,
    )

    print(f"Loaded layers: {list(features_by_layer.keys())}")

    # ------------------------------------------------------------
    # 実行する target を決める
    # ------------------------------------------------------------
    if args.target == "both":
        targets = ["belief_change", "detection"]
    else:
        targets = [args.target]

    # ------------------------------------------------------------
    # targetごとに線形プローブ
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
        )


if __name__ == "__main__":
    main()
