import argparse
import csv
import os
import shutil
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="Run the steering experiment.")
    parser.add_argument("--model-name", required=True, help="Hugging Face model ID or local model directory.")
    parser.add_argument("--questions", type=Path, required=True, help="Input question file (CSV for BoolQ; JSONL for MMLU).")
    parser.add_argument("--persuasion-prompts", type=Path, required=True, help="Text file with one persuasion prompt per line, in question order.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for this model and run.")
    parser.add_argument("--question-offset", type=int, default=0, help="Zero-based offset into the question file.")
    parser.add_argument("--question-limit", type=int, default=100, help="Number of questions in each block of persuasion prompts.")
    parser.add_argument("--classification-max-new-tokens", type=int, default=16)
    parser.add_argument("--persuasion-max-new-tokens", type=int, default=32)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--vector-path", type=Path, help="Safetensors file containing the steering vectors.")
    parser.add_argument("--vector-key", default="robust_unit")
    parser.add_argument("--layers", nargs="+", help="Zero-based transformer block indices, or 'all'.")
    parser.add_argument("--alpha", type=float, help="Steering strength (including zero for ablation).")
    parser.add_argument("--vector-layer-offset", type=int, default=1, help="Offset from transformer block to saved hidden-state index.")
    parser.add_argument("--target-positions", choices=["last", "all"], default="last")
    parser.add_argument("--no-steering", action="store_true", help="Disable additive steering.")
    parser.add_argument("--projection-removal", action="store_true")
    parser.add_argument("--detect-vector-key", default="detect_unit")
    parser.add_argument("--projection-beta", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None, help="Override the chat template thinking setting when supported.")
    parser.add_argument("--start-index", type=int, default=None, help="Resume from this zero-based sample index; defaults to the common completed prefix.")
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
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = args.model_name
OUTPUT_ROOT = args.output_dir
CLASSIFICATION_MAX_NEW_TOKENS = args.classification_max_new_tokens
PERSUASION_MAX_NEW_TOKENS = args.persuasion_max_new_tokens
FLUSH_EVERY = args.flush_every
USE_ROBUST_STEERING = not args.no_steering
ROBUST_VECTOR_PATH = str(args.vector_path) if args.vector_path is not None else None
ROBUST_VECTOR_KEY = args.vector_key
ROBUST_LAYERS = args.layers
ROBUST_ALPHA = args.alpha
ROBUST_VECTOR_LAYER_OFFSET = args.vector_layer_offset
ROBUST_TARGET_POSITIONS = args.target_positions
USE_PROJECTION_REMOVAL = args.projection_removal
DETECT_VECTOR_KEY = args.detect_vector_key
PROJECTION_BETA = args.projection_beta


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

# 呼び出し時の max_new_tokens だけを使用する。
model.generation_config.max_length = None
model.generation_config.max_new_tokens = None
model.generation_config.do_sample = False
model.generation_config.num_beams = 1

MODEL_INPUT_DEVICE = next(model.parameters()).device
PAD_TOKEN_ID = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
)

eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
EOS_TOKEN_IDS = list(
    dict.fromkeys(
        token_id
        for token_id in [tokenizer.eos_token_id, eot_token_id]
        if token_id is not None
    )
)


# ============================================================
# Steering / projection removal 用 vector の読み込み
# ============================================================
if USE_ROBUST_STEERING or USE_PROJECTION_REMOVAL:
    vector_tensor_dict = load_file(ROBUST_VECTOR_PATH, device="cpu")
    print("Loaded vector file")
    print("  path:", ROBUST_VECTOR_PATH)
    print("  keys:", list(vector_tensor_dict.keys()))
else:
    vector_tensor_dict = None

if USE_ROBUST_STEERING:
    if ROBUST_VECTOR_KEY not in vector_tensor_dict:
        raise ValueError(
            f"{ROBUST_VECTOR_KEY} が {ROBUST_VECTOR_PATH} にありません。"
            f"利用可能なkey: {list(vector_tensor_dict.keys())}"
        )
    robust_vectors = vector_tensor_dict[ROBUST_VECTOR_KEY].float()
    print("Loaded robust vectors")
    print("  key  :", ROBUST_VECTOR_KEY)
    print("  shape:", tuple(robust_vectors.shape))
else:
    robust_vectors = None

if USE_PROJECTION_REMOVAL:
    if DETECT_VECTOR_KEY not in vector_tensor_dict:
        raise ValueError(
            f"{DETECT_VECTOR_KEY} が {ROBUST_VECTOR_PATH} にありません。"
            f"利用可能なkey: {list(vector_tensor_dict.keys())}"
        )
    detect_vectors = vector_tensor_dict[DETECT_VECTOR_KEY].float()
    print("Loaded detection vectors")
    print("  key  :", DETECT_VECTOR_KEY)
    print("  shape:", tuple(detect_vectors.shape))
else:
    detect_vectors = None


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("transformer layers を自動検出できませんでした。")


class RepresentationInterventionHook:
    def __init__(
        self,
        model,
        robust_vectors=None,
        detect_vectors=None,
        layers="all",
        alpha=1.0,
        beta=1.0,
        vector_layer_offset=1,
        target_positions="last",
        use_robust_steering=True,
        use_projection_removal=True,
    ):
        self.model = model
        self.robust_vectors = robust_vectors
        self.detect_vectors = detect_vectors
        self.layers = layers
        self.alpha = alpha
        self.beta = beta
        self.vector_layer_offset = vector_layer_offset
        self.target_positions = target_positions
        self.use_robust_steering = use_robust_steering
        self.use_projection_removal = use_projection_removal
        self.handles = []
        self.cached_robust_vectors = {}
        self.cached_detect_vectors = {}

        if target_positions not in {"last", "all"}:
            raise ValueError(
                "target_positions は 'last' または 'all' を指定してください。"
            )
        if self.use_robust_steering and self.robust_vectors is None:
            raise ValueError(
                "use_robust_steering=True ですが robust_vectors がありません。"
            )
        if self.use_projection_removal and self.detect_vectors is None:
            raise ValueError(
                "use_projection_removal=True ですが detect_vectors がありません。"
            )

    def _remove_projection(self, hidden_part, detect_vec):
        d = detect_vec.view(1, 1, -1)
        dot = torch.sum(hidden_part * d, dim=-1, keepdim=True)
        return hidden_part - self.beta * dot * d

    def _add_robust_vector(self, hidden_part, robust_vec):
        return hidden_part + self.alpha * robust_vec.view(1, 1, -1)

    def _make_hook(self, block_idx):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            # 安全性を優先して出力を複製する。
            hidden = hidden.clone()
            hidden_part = (
                hidden[:, -1:, :]
                if self.target_positions == "last"
                else hidden
            )

            if self.use_projection_removal:
                hidden_part = self._remove_projection(
                    hidden_part,
                    self.cached_detect_vectors[block_idx],
                )

            if self.use_robust_steering:
                hidden_part = self._add_robust_vector(
                    hidden_part,
                    self.cached_robust_vectors[block_idx],
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
        target_layers = (
            list(range(len(transformer_layers)))
            if self.layers == "all"
            else list(self.layers)
        )

        for block_idx in target_layers:
            if block_idx < 0 or block_idx >= len(transformer_layers):
                raise ValueError(
                    f"block_idx={block_idx} は範囲外です。"
                    f"このモデルのblock数は {len(transformer_layers)} です。"
                )

            vector_idx = block_idx + self.vector_layer_offset
            block_parameter = next(transformer_layers[block_idx].parameters())
            device = block_parameter.device
            dtype = block_parameter.dtype

            if self.use_robust_steering:
                if vector_idx < 0 or vector_idx >= self.robust_vectors.shape[0]:
                    raise ValueError(
                        f"vector_idx={vector_idx} が robust_vectors の範囲外です。"
                    )
                robust_vec = self.robust_vectors[vector_idx]
                robust_vec = robust_vec / (robust_vec.norm() + 1e-8)
                self.cached_robust_vectors[block_idx] = robust_vec.to(
                    device=device,
                    dtype=dtype,
                )

            if self.use_projection_removal:
                if vector_idx < 0 or vector_idx >= self.detect_vectors.shape[0]:
                    raise ValueError(
                        f"vector_idx={vector_idx} が detect_vectors の範囲外です。"
                    )
                detect_vec = self.detect_vectors[vector_idx]
                detect_vec = detect_vec / (detect_vec.norm() + 1e-8)
                self.cached_detect_vectors[block_idx] = detect_vec.to(
                    device=device,
                    dtype=dtype,
                )

            self.handles.append(
                transformer_layers[block_idx].register_forward_hook(
                    self._make_hook(block_idx)
                )
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.cached_robust_vectors = {}
        self.cached_detect_vectors = {}


def prepare_inputs(messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **({"enable_thinking": args.enable_thinking} if args.enable_thinking is not None else {}),
    )
    return tokenizer(prompt, return_tensors="pt").to(MODEL_INPUT_DEVICE)


def generate_and_collect(
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    use_robust_steering: bool = False,
    collect_hidden_states: bool = False,
) -> Dict[str, object]:
    inputs = prepare_inputs(messages)
    input_length = inputs["input_ids"].shape[1]

    def run_generation() -> Dict[str, object]:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=EOS_TOKEN_IDS,
                pad_token_id=PAD_TOKEN_ID,
                return_dict_in_generate=True,
                output_hidden_states=collect_hidden_states,
            )

        hidden_states_cpu = None
        if collect_hidden_states:
            if generated.hidden_states is None:
                raise RuntimeError(
                    "generate()からhidden statesを取得できませんでした。"
                )

            # 最初の生成ステップは、入力プロンプト全体のforwardに対応する。
            prompt_hidden_states = generated.hidden_states[0]
            hidden_states_cpu = tuple(
                layer_hidden[0, -1:, :]
                .to(device="cpu", dtype=torch.float16)
                .contiguous()
                for layer_hidden in prompt_hidden_states
            )

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
            "hidden_states": hidden_states_cpu,
        }

    if use_robust_steering and (USE_ROBUST_STEERING or USE_PROJECTION_REMOVAL):
        with RepresentationInterventionHook(
            model=model,
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
            result = run_generation()
    else:
        result = run_generation()

    del inputs
    return result


def save_hidden_states(
    hidden_states: Optional[Tuple[torch.Tensor, ...]],
    save_dir: Path,
) -> None:
    if hidden_states is None:
        raise ValueError(
            "保存対象のhidden statesがありません。"
            "generate_and_collect(..., collect_hidden_states=True) "
            "を指定してください。"
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {
        f"layer_{layer_idx}": layer_hidden.contiguous()
        for layer_idx, layer_hidden in enumerate(hidden_states)
    }

    output_path = save_dir / "hidden_states_boolQ.safetensors"
    temporary_path = save_dir / "hidden_states_boolQ.safetensors.tmp"
    save_file(save_dict, str(temporary_path))
    os.replace(temporary_path, output_path)


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


def read_valid_csv_rows(path: Path) -> List[List[str]]:
    if not path.exists():
        return []

    rows: List[List[str]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) != 3:
                break
            rows.append(row)
    return rows


def read_response_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file]


def rewrite_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())


def rewrite_responses(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")
        file.flush()
        os.fsync(file.fileno())


def resolve_start_index(
    result_csv_path: Path,
    response_path: Path,
    requested_start_index: Optional[int],
) -> int:
    csv_rows = read_valid_csv_rows(result_csv_path)
    response_lines = read_response_lines(response_path)
    common_completed = min(len(csv_rows), len(response_lines))

    if requested_start_index is None:
        start_index = common_completed
    else:
        if requested_start_index < 0:
            raise ValueError("--start-indexは0以上にしてください。")
        if requested_start_index > common_completed:
            raise ValueError(
                f"--start-index={requested_start_index}ですが、"
                f"CSVとTXTの共通完了件数は{common_completed}です。"
            )
        start_index = requested_start_index

    if len(csv_rows) != len(response_lines) or requested_start_index is not None:
        print(
            f"output alignment: csv={len(csv_rows)}, "
            f"responses={len(response_lines)} -> {start_index}",
            flush=True,
        )

    rewrite_csv(result_csv_path, csv_rows[:start_index])
    rewrite_responses(response_path, response_lines[:start_index])
    return start_index


def main() -> None:
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    result_csv_path = output_root / "result_boolQ.csv"
    response_path = output_root / "persuasion_response_boolQ.txt"
    hidden_states_root = output_root / "hidden_states_boolQ"

    questions = pd.read_csv(
        args.questions,
        names=["question", "label"],
    )
    questions = questions.iloc[args.question_offset:args.question_offset + args.question_limit].reset_index(drop=True)
    if len(questions) != args.question_limit:
        raise ValueError(f"Expected {args.question_limit} questions, found {len(questions)}. Check --question-offset and --question-limit.")
    question_list = questions["question"].astype(str).tolist()

    start_index = resolve_start_index(
        result_csv_path,
        response_path,
        args.start_index,
    )

    session_processed = 0
    print(f"start: resume_index={start_index}", flush=True)

    with (
        open(args.persuasion_prompts, "r", encoding="utf-8") as persuasion_file,
        result_csv_path.open("a", newline="", encoding="utf-8") as csv_file,
        response_path.open("a", encoding="utf-8") as response_file,
    ):
        writer = csv.writer(csv_file, lineterminator="\n")
        remaining_lines = islice(persuasion_file, start_index, None)

        for i, text in enumerate(remaining_lines, start=start_index):
            question = question_list[i % len(question_list)]

            sample_dir = hidden_states_root / f"sample_{i:04d}"
            # 中断したサンプルの不完全なファイルを除去して作り直す。
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
            sample_dir.mkdir(parents=True, exist_ok=True)

            messages = [
                {
                    "role": "user",
                    "content": text_belief + "question:" + question,
                }
            ]

            # 1. first belief: 保存するので True
            result = generate_and_collect(
                messages=messages,
                max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS,
                use_robust_steering=False,
                collect_hidden_states=True,
            )
            first_belief = str(result["response"])
            save_hidden_states(
                result["hidden_states"],
                sample_dir / "first_belief",
            )

            messages.append({"role": "assistant", "content": first_belief})
            messages.append({"role": "user", "content": text.rstrip("\n")})

            # 2. persuasion: 保存するので True
            result = generate_and_collect(
                messages=messages,
                max_new_tokens=PERSUASION_MAX_NEW_TOKENS,
                use_robust_steering=True,
                collect_hidden_states=True,
            )
            persuasion_response = str(result["response"])
            save_hidden_states(
                result["hidden_states"],
                sample_dir / "persuasion",
            )

            messages.append(
                {"role": "assistant", "content": persuasion_response}
            )

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

            # 3. second belief: 保存しないので False
            result = generate_and_collect(
                messages=messages + messages_belief,
                max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS,
                use_robust_steering=True,
                collect_hidden_states=False,
            )
            second_belief = str(result["response"])

            # 4. Fallacy detection: intervention enabled, no hidden-state collection.
            messages_detection = [{"role": "user", "content": text_detect}]
            result = generate_and_collect(
                messages=messages + messages_detection,
                max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS,
                use_robust_steering=True,
                collect_hidden_states=False,
            )
            detection = str(result["response"])

            writer.writerow([first_belief, second_belief, detection])
            response_file.write(
                persuasion_response.replace("\r", " ").replace("\n", " ")
                + "\n"
            )

            session_processed += 1
            if session_processed % FLUSH_EVERY == 0:
                csv_file.flush()
                response_file.flush()
                print(
                    f"completed={i + 1}, sample={i:04d}",
                    flush=True,
                )

    print(
        f"complete: total={start_index + session_processed}, "
        f"this_run={session_processed}",
        flush=True,
    )


if __name__ == "__main__":
    main()