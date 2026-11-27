
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
# 1. 引数設定
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
# 2. ラベルCSVの読み込み
# ============================================================

def normalize_detection_label(x):
    """
    detection 列を 0/1 に変換する。
    Detect     -> 1
    Not Detect -> 0
    """
    v = str(x).strip().lower()

    if v in {"detect", "detected", "yes", "true", "1"}:
        return 1

    if v in {"not detect", "not_detect", "not detected", "no", "false", "0"}:
        return 0

    raise ValueError(f"未知の detection label です: {x}")


def load_labels(csv_path):
    """
    CSVから以下を作る。

    detection_label:
        Detect = 1, Not Detect = 0

    belief_changed:
        first_belief と second_belief が異なるなら 1
        一致するなら 0
    """
    df = pd.read_csv(csv_path,names=['first_belief','second_belief','detection'],encoding='utf-8')


    # CSVの行番号を sample_id として使う
    # sample_0, sample_1, ... と対応している前提
    df["sample_id"] = df.index.astype(int)

    # detection label
    df["detection_label"] = df["detection"].apply(normalize_detection_label)

    # 信念変化ラベル
    first = df["first_belief"].astype(str).str.strip().str.lower()
    second = df["second_belief"].astype(str).str.strip().str.lower()

    df["belief_changed"] = (first != second).astype(int)

    return df


# ============================================================
# 3. hidden states ファイルの探索
# ============================================================

def extract_sample_id(path):
    """
    パス中の sample_数字 から sample_id を取り出す。
    例:
        sample_123/persuation/hidden_states.safetensors
        -> 123
    """
    m = re.search(r"sample_(\d+)", str(path))
    if m is None:
        raise ValueError(f"sample_id を取り出せません: {path}")
    return int(m.group(1))


def find_hidden_state_files(hidden_root):
    """
    hidden states のファイルを探す。

    Expected layout: hidden_root/sample_*/persuasion/hidden_states.safetensors

    persuation というスペルを優先し、
    念のため persuasion も fallback として見る。
    """
    hidden_root = Path(hidden_root)

    paths = list(hidden_root.glob("sample_*/persuation/hidden_states.safetensors"))

    if len(paths) == 0:
        paths = list(hidden_root.glob("sample_*/persuasion/hidden_states.safetensors"))

    if len(paths) == 0:
        raise FileNotFoundError(f"hidden states が見つかりません: {hidden_root}")

    path_by_id = {}
    for p in paths:
        sid = extract_sample_id(p)
        path_by_id[sid] = p

    return path_by_id


# ============================================================
# 4. safetensors から final-token の全層ベクトルを取り出す
# ============================================================

def _sort_key_by_layer_number(key):
    """
    複数keyに層ごとのtensorが保存されている場合に、
    key名の数字でソートするための関数。
    """
    nums = re.findall(r"\d+", key)
    if len(nums) == 0:
        return 10**9
    return int(nums[-1])


def tensor_to_final_token_vector(x):
    """
    1つのtensorから final-token の hidden vector を取り出す。

    想定する形:
        [hidden_dim]
        [seq_len, hidden_dim]
        [1, seq_len, hidden_dim]
        [batch, seq_len, hidden_dim] ただし batch=1 を想定

    注意:
        hidden states が bfloat16 で保存されている場合、
        そのまま numpy に変換できない。
        そのため .float() で float32 に変換してから numpy にする。
    """
    x = x.detach().cpu()

    # batch dim が 1 の場合は落とす
    while x.ndim >= 3 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim == 1:
        # すでに hidden_dim のベクトル
        return x.float().numpy().astype(np.float32)

    if x.ndim == 2:
        # [seq_len, hidden_dim] とみなし、最後のtokenを使う
        return x[-1].float().numpy().astype(np.float32)

    raise ValueError(f"想定外のtensor形状です: {tuple(x.shape)}")


def load_final_token_all_layers(safetensors_path):
    """
    1つの hidden_states.safetensors から、
    各層の final-token residual stream / hidden state を取り出す。

    戻り値:
        vectors: np.ndarray
            shape = [num_layers, hidden_dim]
    """
    tensors = load_file(str(safetensors_path), device="cpu")

    # --------------------------------------------------------
    # ケースA:
    # 1つのkeyに全層がまとめて保存されている場合
    # 例:
    #   hidden_states: [num_layers, seq_len, hidden_dim]
    #   hidden_states: [1, num_layers, seq_len, hidden_dim]
    # --------------------------------------------------------
    if len(tensors) == 1:
        x = next(iter(tensors.values())).detach().cpu()

        # [1, num_layers, seq_len, hidden_dim] -> [num_layers, seq_len, hidden_dim]
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)

        if x.ndim == 3:
            # 多くの場合 [num_layers, seq_len, hidden_dim]
            # ただし [seq_len, num_layers, hidden_dim] の可能性もあるので簡単に判定
            if x.shape[0] <= 200:
                # [num_layers, seq_len, hidden_dim]
                vectors = x[:, -1, :]
            elif x.shape[1] <= 200:
                # [seq_len, num_layers, hidden_dim]
                vectors = x[-1, :, :]
            else:
                raise ValueError(f"層方向を判定できません: {tuple(x.shape)}")

            # bfloat16 対策
            return vectors.float().numpy().astype(np.float32)

        if x.ndim == 2:
            # [num_layers, hidden_dim] とみなす
            return x.float().numpy().astype(np.float32)

        raise ValueError(f"想定外のhidden_states形状です: {tuple(x.shape)}")

    # --------------------------------------------------------
    # ケースB:
    # keyごとに各層が保存されている場合
    # 例:
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
# 5. hidden states を memmap にまとめる
# ============================================================

def build_hidden_state_memmap(
    sample_ids,
    path_by_id,
    output_dir,
    skip_first_layer=False,
):
    """
    各sampleのsafetensorsを読み込み、
    [num_samples, num_layers, hidden_dim] のmemmapにまとめる。

    memmapを使う理由:
        全サンプル・全層を普通のnumpy配列で持つとメモリ使用量が大きくなるため。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 最初のサンプルで層数と次元数を確認
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

    # 1つ目を保存
    H[0] = first_vecs

    # 残りを保存
    for row_idx, sid in enumerate(tqdm(sample_ids[1:], desc="Loading hidden states"), start=1):
        vecs = load_final_token_all_layers(path_by_id[sid])

        if skip_first_layer:
            vecs = vecs[1:]

        if vecs.shape != (num_layers, hidden_dim):
            raise ValueError(
                f"sample_{sid} の形状が他と違います: "
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
# 6. detection probe score の作成
# ============================================================

def compute_out_of_fold_probe_scores(X, y, n_splits=5, random_state=42):
    """
    各層の残差ストリーム X から detection_label y を予測する線形probeを学習する。

    重要:
        mediationで使うMは、学習データ上のscoreではなく、
        out-of-fold scoreにする。

    入力:
        X: shape = [num_samples, hidden_dim]
        y: shape = [num_samples]
            Detect=1, Not Detect=0

    出力:
        scores:
            各サンプルの decision_function score
            これを mediator M_l として使う

        auc:
            probe の AUROC

        acc:
            probe の accuracy
    """
    y = np.asarray(y).astype(int)

    class_counts = np.bincount(y)
    min_class_count = class_counts.min()

    if min_class_count < 2:
        raise ValueError("detection_label の片方のクラスが少なすぎます。probe を学習できません。")

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
        # mediationのMとして使う
        fold_scores = clf.decision_function(X[test_idx])
        scores[test_idx] = fold_scores.astype(np.float32)

        preds[test_idx] = (fold_scores >= 0).astype(int)

    auc = roc_auc_score(y, scores)
    acc = accuracy_score(y, preds)

    return scores, auc, acc


# ============================================================
# 7. control 変数の作成
# ============================================================

def build_controls(df, use_first_belief_control=False):
    """
    mediation に入れる control 変数を作る。

    今回のCSVには topic や fallacy_type がない前提なので、
    デフォルトでは control なし。

    必要に応じて first_belief を one-hot で入れる。
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
# 8. 統計的 mediation の推定
# ============================================================

def fit_mediation(T, M, Y, C=None):
    """
    binary outcome 用の statistical mediation。

    T:
        誤謬あり = 1, 誤謬なし = 0

    M:
        detection probe score
        ここでは標準化済みの連続値を想定

    Y:
        belief_changed
        変化あり = 1, 変化なし = 0

    C:
        controls
        shape = [num_samples, num_controls]

    推定するモデル:

        Mediator model:
            M ~ T + C

        Outcome model:
            Y ~ T + M + T*M + C

    出力する効果は、Y=1 の予測確率スケールでの差。
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

    # 反実仮想 mediator
    # M(0): 誤謬なし条件だった場合の probe score
    # M(1): 誤謬あり条件だった場合の probe score
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
    # 8.3 反実仮想 outcome の予測
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
    # 8.4 indirect effect / direct effect の計算
    # --------------------------------------------------------

    # ACME(1): T=1に固定し、Mだけ M(0) -> M(1) に変える
    ACME_1 = np.mean(Y_1_M1 - Y_1_M0)

    # ACME(0): T=0に固定し、Mだけ M(0) -> M(1) に変える
    ACME_0 = np.mean(Y_0_M1 - Y_0_M0)

    # ADE(1): M=M(1)に固定し、Tだけ 0 -> 1 に変える
    ADE_1 = np.mean(Y_1_M1 - Y_0_M1)

    # ADE(0): M=M(0)に固定し、Tだけ 0 -> 1 に変える
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
    mediation effect の bootstrap 信頼区間を計算する。

    注意:
        ここでは probe 自体は再学習しない。
        すでに得られた out-of-fold M を固定して、
        mediation model の不確実性だけを見る。
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

        # bootstrap sample 内で Y が1クラスしかない場合はスキップ
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
# 9. メイン処理
# ============================================================

def main():
    args = parse_args()

    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 9.1 ラベル読み込み
    # --------------------------------------------------------
    print("Loading labels...")
    labels_df = load_labels(args.csv_path)

    # --------------------------------------------------------
    # 9.2 hidden states ファイルを探す
    # --------------------------------------------------------
    print("Finding hidden state files...")
    path_by_id = find_hidden_state_files(args.hidden_root)

    # --------------------------------------------------------
    # 9.3 CSVとhidden statesの両方が存在するサンプルだけ使う
    # --------------------------------------------------------
    available_sample_ids = sorted(set(labels_df["sample_id"]) & set(path_by_id.keys()))


    if len(available_sample_ids) == 0:
        raise ValueError(
            "除外処理後、CSVとhidden statesで対応する sample_id がありません。"
        )

    labels_df = labels_df.set_index("sample_id").loc[available_sample_ids].reset_index()

    sample_ids = labels_df["sample_id"].tolist()

    print(f"Number of usable samples: {len(sample_ids)}")

    # --------------------------------------------------------
    # 9.4 T, Y, D を作る
    # --------------------------------------------------------
    # T:
    # Treatment is defined by --fallacy-last-id.
    T = (labels_df["sample_id"].values <= args.fallacy_last_id).astype(int)

    # Y:
    #   first_belief != second_belief なら信念変化 = 1
    #   一致なら信念変化 = 0
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
    # 9.5 control 変数
    # --------------------------------------------------------
    C = build_controls(
        labels_df,
        use_first_belief_control=args.use_first_belief_control,
    )

    # --------------------------------------------------------
    # 9.6 hidden states を memmap 化
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
    # 9.7 各層で probe score を作り、mediation を実行
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
        # 9.7.1 layer l の残差ストリーム
        # ----------------------------------------------------
        X_l = np.asarray(H[:, layer, :], dtype=np.float32)

        # ----------------------------------------------------
        # 9.7.2 detection probe score を out-of-fold で作成
        # ----------------------------------------------------
        M_l, probe_auc, probe_acc = compute_out_of_fold_probe_scores(
            X=X_l,
            y=D,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )

        # ----------------------------------------------------
        # 9.7.3 mediator M_l を標準化
        # ----------------------------------------------------
        # 層ごとに probe score のスケールが違うため、
        # mediation に入れる前に平均0・標準偏差1にする。
        M_l_std = (M_l - M_l.mean()) / (M_l.std() + 1e-8)

        # probe score を保存用テーブルに追加
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
        # 9.7.6 結果を保存
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
    # 9.8 結果保存
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
