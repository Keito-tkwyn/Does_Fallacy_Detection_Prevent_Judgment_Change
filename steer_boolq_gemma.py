import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

def parse_args():
    parser = argparse.ArgumentParser(description="Run the steering experiment.")
    parser.add_argument("--model-name", required=True, help="Hugging Face model ID or local model directory.")
    parser.add_argument("--questions", type=Path, required=True, help="Input question file (CSV for BoolQ; JSONL for MMLU).")
    parser.add_argument("--persuasion-prompts", type=Path, required=True, help="Text file with one persuasion prompt per line, in question order.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for this model and run.")
    parser.add_argument("--question-offset", type=int, default=0, help="Zero-based offset into the question file.")
    parser.add_argument("--question-limit", type=int, default=100, help="Number of questions in each block of persuasion prompts.")
    parser.add_argument("--classification-max-new-tokens", type=int, default=16)
    parser.add_argument("--persuasion-max-new-tokens", type=int, default=64)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--vector-path", type=Path, help="Safetensors file containing the steering vectors.")
    parser.add_argument("--vector-key", default="robust_orth_unit")
    parser.add_argument("--layers", nargs="+", help="Zero-based transformer block indices, or 'all'.")
    parser.add_argument("--alpha", type=float, help="Steering strength (including zero for ablation).")
    parser.add_argument("--vector-layer-offset", type=int, default=1, help="Offset from transformer block to saved hidden-state index.")
    parser.add_argument("--target-positions", choices=["last", "all"], default="last")
    parser.add_argument("--no-steering", action="store_true", help="Disable additive steering.")
    parser.add_argument("--projection-removal", action="store_true")
    parser.add_argument("--detect-vector-key", default="detect_unit")
    parser.add_argument("--projection-beta", type=float, default=1.0)
    args = parser.parse_args()
    if args.question_offset < 0 or min(args.question_limit, args.classification_max_new_tokens, args.persuasion_max_new_tokens, args.flush_every) <= 0:
        parser.error("Question offset must be nonnegative; limits and flush interval must be positive.")
    for path in (args.questions, args.persuasion_prompts):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    active = not args.no_steering or args.projection_removal
    if active and (args.vector_path is None or not args.vector_path.is_file() or args.layers is None):
        parser.error("Active intervention requires an existing --vector-path and --layers.")
    if not args.no_steering and args.alpha is None:
        parser.error("Additive steering requires --alpha.")
    if args.layers == ["all"]:
        args.layers = "all"
    elif args.layers is not None:
        try:
            args.layers = [int(layer) for layer in args.layers]
        except ValueError:
            parser.error("--layers must contain integers or be 'all'.")
        if any(layer < 0 for layer in args.layers) or len(set(args.layers)) != len(args.layers):
            parser.error("Layer indices must be nonnegative and unique.")
    else:
        args.layers = []
    if args.alpha is None:
        args.alpha = 0.0
    return args


args = parse_args()


import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoProcessor, Gemma3ForConditionalGeneration,
    LogitsProcessor, LogitsProcessorList,
)


# ============================================================
# モデル設定
# ============================================================

MODEL_NAME = args.model_name

# Gemma 3ではBF16を優先する。
# BF16非対応環境では、数値安定性を優先してFP32を使用する。
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    MODEL_DTYPE = torch.bfloat16
else:
    MODEL_DTYPE = torch.float32

CLASSIFICATION_MAX_NEW_TOKENS = args.classification_max_new_tokens
PERSUASION_MAX_NEW_TOKENS = args.persuasion_max_new_tokens


# ============================================================
# Robust-detection steering 設定
# ============================================================

USE_ROBUST_STEERING = not args.no_steering
USE_PROJECTION_REMOVAL = args.projection_removal

ROBUST_VECTOR_PATH = str(args.vector_path) if args.vector_path is not None else None

ROBUST_VECTOR_KEY = args.vector_key
DETECT_VECTOR_KEY = args.detect_vector_key

ROBUST_LAYERS = args.layers
ROBUST_ALPHA = args.alpha
PROJECTION_BETA = args.projection_beta

# hidden_states[0]はembedding出力であり、block 0の出力は
# hidden_states[1]に対応するため、offset=1とする。
ROBUST_VECTOR_LAYER_OFFSET = args.vector_layer_offset
ROBUST_TARGET_POSITIONS = args.target_positions


# ============================================================
# Gemma 3の内部モジュール取得
# ============================================================


def get_input_device(current_model: torch.nn.Module) -> torch.device:
    """入力embeddingが置かれているdeviceを取得する。"""
    if (
        hasattr(current_model, "model")
        and hasattr(current_model.model, "language_model")
        and hasattr(current_model.model.language_model, "embed_tokens")
    ):
        return current_model.model.language_model.embed_tokens.weight.device

    return next(current_model.parameters()).device


def get_transformer_layers(current_model: torch.nn.Module):
    """
    Gemma 3を優先し、代表的なdecoder-onlyモデルの
    transformer blockもフォールバックとして取得する。
    """
    # Gemma3ForConditionalGeneration
    if (
        hasattr(current_model, "model")
        and hasattr(current_model.model, "language_model")
        and hasattr(current_model.model.language_model, "layers")
    ):
        return current_model.model.language_model.layers

    # Gemma3TextForCausalLM等
    if (
        hasattr(current_model, "language_model")
        and hasattr(current_model.language_model, "layers")
    ):
        return current_model.language_model.layers

    # Llama / Qwen / Mistral系
    if hasattr(current_model, "model") and hasattr(current_model.model, "layers"):
        return current_model.model.layers

    if hasattr(current_model, "transformer") and hasattr(current_model.transformer, "h"):
        return current_model.transformer.h

    if hasattr(current_model, "gpt_neox") and hasattr(current_model.gpt_neox, "layers"):
        return current_model.gpt_neox.layers

    raise ValueError(
        "transformer layersを自動検出できませんでした。\n"
        f"model class={type(current_model).__name__}"
    )


# ============================================================
# Steering / projection removal用vectorの読み込み
# ============================================================

if USE_ROBUST_STEERING or USE_PROJECTION_REMOVAL:
    vector_tensor_dict = load_file(ROBUST_VECTOR_PATH, device="cpu")

    print("Loaded vector file")
    print("  path:", Path(ROBUST_VECTOR_PATH).resolve())
    print("  keys:", list(vector_tensor_dict.keys()))
else:
    vector_tensor_dict = None


if USE_ROBUST_STEERING:
    assert vector_tensor_dict is not None

    if ROBUST_VECTOR_KEY not in vector_tensor_dict:
        raise ValueError(
            f"{ROBUST_VECTOR_KEY}が{ROBUST_VECTOR_PATH}にありません。"
            f"利用可能なkey: {list(vector_tensor_dict.keys())}"
        )

    robust_vectors = vector_tensor_dict[ROBUST_VECTOR_KEY].float()

    print("Loaded robust vectors")
    print("  key  :", ROBUST_VECTOR_KEY)
    print("  shape:", tuple(robust_vectors.shape))
else:
    robust_vectors = None


if USE_PROJECTION_REMOVAL:
    assert vector_tensor_dict is not None

    if DETECT_VECTOR_KEY not in vector_tensor_dict:
        raise ValueError(
            f"{DETECT_VECTOR_KEY}が{ROBUST_VECTOR_PATH}にありません。"
            f"利用可能なkey: {list(vector_tensor_dict.keys())}"
        )

    detect_vectors = vector_tensor_dict[DETECT_VECTOR_KEY].float()

    print("Loaded detection vectors")
    print("  key  :", DETECT_VECTOR_KEY)
    print("  shape:", tuple(detect_vectors.shape))
else:
    detect_vectors = None


def validate_intervention_vectors(vectors, name):
    if vectors is None:
        return
    if vectors.ndim != 2:
        raise ValueError(f"{name}は[層数, hidden size]である必要があります。")
    indices = (list(range(ROBUST_VECTOR_LAYER_OFFSET, vectors.shape[0]))
               if ROBUST_LAYERS == "all"
               else [i + ROBUST_VECTOR_LAYER_OFFSET for i in ROBUST_LAYERS])
    for idx in indices:
        if idx < 0 or idx >= vectors.shape[0]:
            raise ValueError(f"{name}: vector_idx={idx}が範囲外です。")
        if not torch.isfinite(vectors[idx]).all():
            raise ValueError(
                f"{ROBUST_VECTOR_PATH} の {name}[{idx}] にNaN/Infがあります。"
                "元のhidden statesを確認し、有限値のデータからvectorを再生成してください。"
                "介入なしで実行する場合はUSE_ROBUST_STEERINGと"
                "USE_PROJECTION_REMOVALをFalseにしてください。"
            )


validate_intervention_vectors(robust_vectors, ROBUST_VECTOR_KEY)
validate_intervention_vectors(detect_vectors, DETECT_VECTOR_KEY)


processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=MODEL_DTYPE,
    device_map="auto",
    attn_implementation="sdpa",
).eval()

# max_new_tokensのみをgenerate()で指定し、max_lengthとの重複警告を防ぐ。
model.generation_config.max_length = None
model.generation_config.max_new_tokens = None
model.generation_config.do_sample = False
model.generation_config.num_beams = 1

PAD_TOKEN_ID = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
)
EOS_TOKEN_ID = model.generation_config.eos_token_id


MODEL_INPUT_DEVICE = get_input_device(model)

print("model class:", type(model).__name__)
print("model dtype:", next(model.parameters()).dtype)
print("model input device:", MODEL_INPUT_DEVICE)
print("pad_token_id:", PAD_TOKEN_ID)
print("eos_token_id:", EOS_TOKEN_ID)


# ============================================================
# Representation intervention hook
# ============================================================


class RepresentationInterventionHook:
    """
    指定層のhidden stateに対して、必要に応じて以下を行う。

    1. detection方向の射影除去
    2. robust vectorの加算

    h'_l = h_l - beta * proj_detect(h_l) + alpha * v_robust
    """

    def __init__(
        self,
        current_model: torch.nn.Module,
        robust_vectors: Optional[torch.Tensor] = None,
        detect_vectors: Optional[torch.Tensor] = None,
        layers: Union[str, Sequence[int]] = "all",
        alpha: float = 1.0,
        beta: float = 1.0,
        vector_layer_offset: int = 1,
        target_positions: str = "last",
        use_robust_steering: bool = True,
        use_projection_removal: bool = True,
    ) -> None:
        self.model = current_model
        self.robust_vectors = robust_vectors
        self.detect_vectors = detect_vectors
        self.layers = layers
        self.alpha = alpha
        self.beta = beta
        self.vector_layer_offset = vector_layer_offset
        self.target_positions = target_positions
        self.use_robust_steering = use_robust_steering
        self.use_projection_removal = use_projection_removal
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        if target_positions not in {"last", "all"}:
            raise ValueError(
                "target_positionsは'last'または'all'を指定してください。"
            )

        if self.use_robust_steering and self.robust_vectors is None:
            raise ValueError(
                "use_robust_steering=Trueですがrobust_vectorsがありません。"
            )

        if self.use_projection_removal and self.detect_vectors is None:
            raise ValueError(
                "use_projection_removal=Trueですがdetect_vectorsがありません。"
            )

    def _remove_projection(
        self,
        hidden_part: torch.Tensor,
        detect_vec: torch.Tensor,
    ) -> torch.Tensor:
        """hidden_partからdetect_vec方向の射影成分を除去する。"""
        detect_vec = detect_vec / (detect_vec.norm() + 1e-8)
        direction = detect_vec.view(1, 1, -1)

        dot = torch.sum(
            hidden_part * direction,
            dim=-1,
            keepdim=True,
        )
        projection = dot * direction

        return hidden_part - self.beta * projection

    def _add_robust_vector(
        self,
        hidden_part: torch.Tensor,
        robust_vec: torch.Tensor,
    ) -> torch.Tensor:
        """hidden_partへalpha倍した単位robust vectorを加える。"""
        robust_vec = robust_vec / (robust_vec.norm() + 1e-8)
        direction = robust_vec.view(1, 1, -1)

        return hidden_part + self.alpha * direction

    def _make_hook(self, block_idx: int):
        def hook(module, inputs, output):
            del module, inputs

            # Gemmaのdecoder layerは通常tupleを返すが、
            # Transformersの版によってTensorの場合にも対応する。
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            if not isinstance(hidden, torch.Tensor):
                raise TypeError(
                    "Transformer blockの先頭出力がTensorではありません。"
                    f"type={type(hidden)}"
                )

            hidden = hidden.clone()
            vector_idx = block_idx + self.vector_layer_offset

            if self.target_positions == "last":
                hidden_part = hidden[:, -1:, :]
            else:
                hidden_part = hidden

            if self.use_projection_removal:
                assert self.detect_vectors is not None

                if vector_idx < 0 or vector_idx >= self.detect_vectors.shape[0]:
                    raise ValueError(
                        f"vector_idx={vector_idx}がdetect_vectorsの範囲外です。"
                        f"detect_vectors.shape={tuple(self.detect_vectors.shape)}"
                    )

                detect_vec = self.detect_vectors[vector_idx].to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )

                if detect_vec.numel() != hidden.shape[-1]:
                    raise ValueError(
                        "detect vectorとモデルのhidden sizeが一致しません。"
                        f"vector={detect_vec.numel()}, "
                        f"hidden={hidden.shape[-1]}"
                    )

                hidden_part = self._remove_projection(
                    hidden_part=hidden_part,
                    detect_vec=detect_vec,
                )

            if self.use_robust_steering:
                assert self.robust_vectors is not None

                if vector_idx < 0 or vector_idx >= self.robust_vectors.shape[0]:
                    raise ValueError(
                        f"vector_idx={vector_idx}がrobust_vectorsの範囲外です。"
                        f"robust_vectors.shape={tuple(self.robust_vectors.shape)}"
                    )

                robust_vec = self.robust_vectors[vector_idx].to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )

                if robust_vec.numel() != hidden.shape[-1]:
                    raise ValueError(
                        "robust vectorとモデルのhidden sizeが一致しません。"
                        f"vector={robust_vec.numel()}, "
                        f"hidden={hidden.shape[-1]}"
                    )

                hidden_part = self._add_robust_vector(
                    hidden_part=hidden_part,
                    robust_vec=robust_vec,
                )

            if self.target_positions == "last":
                hidden[:, -1:, :] = hidden_part
            else:
                hidden = hidden_part

            if rest is not None:
                return (hidden,) + rest

            return hidden

        return hook

    def __enter__(self):
        transformer_layers = get_transformer_layers(self.model)

        if self.layers == "all":
            target_layers = list(range(len(transformer_layers)))
        else:
            target_layers = list(self.layers)

        for block_idx in target_layers:
            if block_idx < 0 or block_idx >= len(transformer_layers):
                raise ValueError(
                    f"block_idx={block_idx}は範囲外です。"
                    f"このモデルのblock数は{len(transformer_layers)}です。"
                )

            handle = transformer_layers[block_idx].register_forward_hook(
                self._make_hook(block_idx)
            )
            self.handles.append(handle)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback

        for handle in self.handles:
            handle.remove()

        self.handles = []


# ============================================================
# 入力作成・生成・hidden states取得
# ============================================================


class ValidateGenerationScores(LogitsProcessor):
    """数値異常を検出し、padding tokenの生成を防ぐ。"""

    def __call__(self, input_ids, scores):
        if torch.isnan(scores).any() or torch.isposinf(scores).any():
            raise RuntimeError(
                "生成logitsにNaNまたは+Infがあります。"
                "モデル計算や介入vectorを確認してください。"
                f"input_length={input_ids.shape[1]}, dtype={scores.dtype}"
            )
        eos_ids = ([EOS_TOKEN_ID] if isinstance(EOS_TOKEN_ID, int)
                   else (EOS_TOKEN_ID or []))
        if PAD_TOKEN_ID is not None and PAD_TOKEN_ID not in eos_ids:
            scores[:, PAD_TOKEN_ID] = -float("inf")
        if not torch.isfinite(scores).any(dim=-1).all():
            raise RuntimeError("生成可能な有限logitsがありません。")
        return scores


def prepare_inputs(messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    """Gemmaのchat templateを適用してモデル入力を作る。"""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(MODEL_INPUT_DEVICE)


def generate_and_collect(
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    use_robust_steering: bool = False,
    collect_hidden_states: bool = True,
) -> Dict[str, object]:
    """
    貪欲法で応答を生成する。

    collect_hidden_states=Trueの場合は、最初の生成トークンを予測する際に
    プロンプト全体を処理して得られた全層hidden statesを返す。
    """
    inputs = prepare_inputs(messages)
    input_length = inputs["input_ids"].shape[1]

    def run_generation():
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=PAD_TOKEN_ID,
                eos_token_id=EOS_TOKEN_ID,
                logits_processor=LogitsProcessorList([ValidateGenerationScores()]),
                return_dict_in_generate=True,
                output_hidden_states=collect_hidden_states,
            )

        hidden_states = None

        if collect_hidden_states:
            if generated.hidden_states is None:
                raise RuntimeError(
                    "generate()からhidden statesを取得できませんでした。"
                )

            # generated.hidden_states[0]は、最初の生成トークンを
            # 予測するためにプロンプトを処理した際の全層出力。
            hidden_states = generated.hidden_states[0]

        new_tokens = generated.sequences[0, input_length:]

        response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        if not response:
            raw_decoded = tokenizer.decode(
                new_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            raise RuntimeError(
                "モデルの生成結果が空文字になりました。\n"
                f"generated_token_ids={new_tokens.tolist()}\n"
                f"decoded_with_special_tokens={raw_decoded!r}"
            )

        del generated

        return {
            "response": response,
            "hidden_states": hidden_states,
        }

    try:
        if use_robust_steering:
            if USE_ROBUST_STEERING and robust_vectors is None:
                raise ValueError(
                    "USE_ROBUST_STEERING=Trueですが、"
                    "robust_vectorsが読み込まれていません。"
                )

            if USE_PROJECTION_REMOVAL and detect_vectors is None:
                raise ValueError(
                    "USE_PROJECTION_REMOVAL=Trueですが、"
                    "detect_vectorsが読み込まれていません。"
                )

            with RepresentationInterventionHook(
                current_model=model,
                robust_vectors=robust_vectors,
                detect_vectors=detect_vectors,
                layers=ROBUST_LAYERS,
                alpha=ROBUST_ALPHA,
                beta=PROJECTION_BETA,
                vector_layer_offset=ROBUST_VECTOR_LAYER_OFFSET,
                target_positions=ROBUST_TARGET_POSITIONS,
                use_robust_steering=USE_ROBUST_STEERING,
                use_projection_removal=USE_PROJECTION_REMOVAL,
            ):
                return run_generation()

        return run_generation()
    finally:
        del inputs


def save_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    save_dir: Path,
) -> None:
    """
    元コードと同じく、プロンプト全トークンのhidden statesを
    bfloat16で保存する。
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        f"layer_{layer_idx}": layer_hidden[0]
        .to(device="cpu", dtype=torch.bfloat16)
        .contiguous()
        for layer_idx, layer_hidden in enumerate(hidden_states)
    }

    output_path = save_dir / "hidden_states_boolQ.safetensors"
    temporary_path = save_dir / "hidden_states_boolQ.safetensors.tmp"

    save_file(save_dict, str(temporary_path))
    os.replace(temporary_path, output_path)


# ============================================================
# プロンプト
# ============================================================

text_belief = '''
I will now ask questions. Please answer with ``just one word'' : "True" or "False".

Answer Format: { answer(True/False) }
Attention: Please output only the results; do not include any extra text beyond the required format.
'''

text_detect = '''
A logical fallacy is a concept that refers to an invalid argument resulting from flaws in the relationship between premises and conclusions, or in the structure of the argument itself.
Did the argument presented earlier contain a logical fallacy?
If it did, please select “Detect”; if not, please select “Not Detect.”

Answer Format: { answer(Detect/Not Detect) }
Attention: Please output only the results; do not include any extra text beyond the required format.
'''


# ============================================================
# 実験本体
# ============================================================


def main() -> None:
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    questions = pd.read_csv(
        args.questions,
        names=["question", "label"],
    )

    questions = questions.iloc[args.question_offset:args.question_offset + args.question_limit].reset_index(drop=True)
    if len(questions) != args.question_limit:
        raise ValueError(f"Expected {args.question_limit} questions, found {len(questions)}. Check --question-offset and --question-limit.")

    if questions.empty:
        raise ValueError(
            "No questions found in the input CSV."
        )

    result_csv_path = output_root / "result_boolQ.csv"
    persuasion_response_path = output_root / "persuasion_response_boolQ.txt"
    hidden_states_root = output_root / "hidden_states_boolQ"

    print("start")

    with (
        open(
            args.persuasion_prompts,
            "r",
            encoding="utf-8",
        ) as persuasion_file,
        open(
            result_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file,
        open(
            persuasion_response_path,
            "w",
            encoding="utf-8",
        ) as response_file,
    ):
        writer = csv.writer(csv_file, lineterminator="\n")

        for i, text in enumerate(persuasion_file):
            question_idx = i % len(questions)
            question = str(questions.iloc[question_idx]["question"])

            sample_dir = hidden_states_root / f"sample_{i:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # ------------------------------------------------------
            # 1. first belief：介入なし
            # ------------------------------------------------------
            messages: List[Dict[str, str]] = [
                {
                    "role": "user",
                    "content": text_belief + "question:" + question,
                }
            ]

            result = generate_and_collect(
                messages=messages,
                max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS,
                use_robust_steering=False,
                collect_hidden_states=True,
            )

            first_belief = str(result["response"])
            first_hidden_states = result["hidden_states"]

            if first_hidden_states is None:
                raise RuntimeError(
                    "first beliefのhidden statesが取得できませんでした。"
                )

            save_hidden_states(
                first_hidden_states,
                sample_dir / "first_belief",
            )
            del first_hidden_states

            # ------------------------------------------------------
            # 2. persuasion：介入あり
            # ------------------------------------------------------
            messages.append(
                {
                    "role": "assistant",
                    "content": first_belief,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": text.rstrip("\n"),
                }
            )

            result = generate_and_collect(
                messages=messages,
                max_new_tokens=PERSUASION_MAX_NEW_TOKENS,
                use_robust_steering=True,
                collect_hidden_states=True,
            )

            persuasion_response = str(result["response"])
            persuasion_hidden_states = result["hidden_states"]

            if persuasion_hidden_states is None:
                raise RuntimeError(
                    "persuasionのhidden statesが取得できませんでした。"
                )

            save_hidden_states(
                persuasion_hidden_states,
                sample_dir / "persuasion",
            )
            del persuasion_hidden_states

            messages.append(
                {
                    "role": "assistant",
                    "content": persuasion_response,
                }
            )

            # ------------------------------------------------------
            # 3. second belief：介入あり、hidden states保存なし
            # ------------------------------------------------------
            messages_belief = [
                {
                    "role": "user",
                    "content": (
                        text_belief
                        + "question:"
                        + question
                    ),
                }
            ]

            result = generate_and_collect(
                messages=messages + messages_belief,
                max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS,
                use_robust_steering=True,
                collect_hidden_states=False,
            )
            second_belief = str(result["response"])

            # ------------------------------------------------------
            # 4. fallacy detection：介入あり、hidden states保存なし
            # ------------------------------------------------------
            messages_detection = [
                {
                    "role": "user",
                    "content": text_detect,
                }
            ]

            result = generate_and_collect(
                messages=messages + messages_detection,
                max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS,
                use_robust_steering=True,
                collect_hidden_states=False,
            )
            detection = str(result["response"])

            writer.writerow(
                [
                    first_belief,
                    second_belief,
                    detection,
                ]
            )
            response_file.write(
                persuasion_response.replace("\r", " ").replace("\n", " ")
                + "\n"
            )

            # 各サンプル終了時に書き込み、途中停止時の損失を抑える。
            csv_file.flush()
            response_file.flush()

            if (i+1) % args.flush_every == 0:
                print(
                    f"completed={i + 1}, sample={i:04d}",
                    flush=True,
                )

    print("complete")


if __name__ == "__main__":
    main()
