import argparse
import csv
import os
import shutil
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
    parser.add_argument("--persuasion-max-new-tokens", type=int, default=64)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--vector-path", type=Path, help="Safetensors file containing the steering vectors.")
    parser.add_argument("--vector-key", default="robust_unit")
    parser.add_argument("--layers", nargs="+", help="Zero-based transformer block indices, or 'all'.")
    parser.add_argument("--alpha", type=float, help="Steering strength (including zero for ablation).")
    parser.add_argument("--vector-layer-offset", type=int, default=1, help="Offset from transformer block to saved hidden-state index.")
    parser.add_argument("--target-positions", choices=["last", "all"], default="last")
    parser.add_argument("--no-steering", action="store_true", help="Disable additive steering.")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None, help="Override the chat template thinking setting when supported.")
    args = parser.parse_args()
    if args.question_offset < 0 or min(args.question_limit, args.classification_max_new_tokens, args.persuasion_max_new_tokens, args.flush_every) <= 0:
        parser.error("Question offset must be nonnegative; limits and flush interval must be positive.")
    for path in (args.questions, args.persuasion_prompts):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    active = not args.no_steering
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


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

# Use only max_new_tokens specified at each generate() call.
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

# Llama 3 Instruct may end a chat turn with <|eot_id|>.
eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
EOS_TOKEN_IDS = list(
    dict.fromkeys(
        token_id
        for token_id in [tokenizer.eos_token_id, eot_token_id]
        if token_id is not None
    )
)


# ============================================================
# Robust vector loading
# ============================================================

if USE_ROBUST_STEERING:
    robust_tensor_dict = load_file(ROBUST_VECTOR_PATH, device="cpu")

    if ROBUST_VECTOR_KEY not in robust_tensor_dict:
        raise ValueError(
            f"Key {ROBUST_VECTOR_KEY} not found in {ROBUST_VECTOR_PATH}. "
            f"Available keys: {list(robust_tensor_dict.keys())}"
        )

    robust_vectors = robust_tensor_dict[ROBUST_VECTOR_KEY].float()

    print("Loaded robust vectors", flush=True)
    print(f"  path : {ROBUST_VECTOR_PATH}", flush=True)
    print(f"  key  : {ROBUST_VECTOR_KEY}", flush=True)
    print(f"  shape: {tuple(robust_vectors.shape)}", flush=True)
else:
    robust_vectors = None


# ============================================================
# Prompt definitions: aligned with the baseline MMLU evaluation
# ============================================================

text_belief = '''
For multiple-choice questions, please select the answer(s) you believe to be correct.
The questions follow this format:

"question"
"choices:[a:text A, b:text B, c:text C, d:text D]"

For example, in a question with four choices (“Text A”, “Text B”, “Text C”, “Text D”), answer a if Text A is correct, b if Text B is correct, c if Text C is correct, and d if Text D is correct.
Please answer using alphabet only.

Answer format: 'a' or 'b' or 'c' or 'd'
Attention: No other response formats are permitted (e.g., directly stating the content of the options, including messages other than the answer, etc.).
'''

text_detect = '''
A logical fallacy is a concept that refers to an invalid argument resulting from flaws in the relationship between premises and conclusions, or in the structure of the argument itself.
Did the argument presented earlier contain a logical fallacy?
If it did, please select “Detect”; if not, please select “Not Detect.”

Answer Format: {answer(Detect/Not Detect)}
Attention: Please output only the results; do not include any extra text beyond the required format.
'''


def build_question_prompt(question: str, choices: object) -> str:
    """Construct the MMLU question and choices in a common format."""
    return (
        text_belief
        + "\nquestion:"
        + question
        + "\na:"
        + choices[0]
        + " b:"
        + choices[1]
        + " c:"
        + choices[2]
        + " d:"
        + choices[3]
    )

# ============================================================
# Steering hook
# ============================================================


def get_transformer_layers(target_model):
    """Return transformer blocks for Llama/Qwen/Mistral-like models."""
    if hasattr(target_model, "model") and hasattr(target_model.model, "layers"):
        return target_model.model.layers

    if hasattr(target_model, "transformer") and hasattr(
        target_model.transformer, "h"
    ):
        return target_model.transformer.h

    if hasattr(target_model, "gpt_neox") and hasattr(
        target_model.gpt_neox, "layers"
    ):
        return target_model.gpt_neox.layers

    raise ValueError("Could not automatically locate the transformer layers.")


class RobustSteeringHook:
    """
    Register hooks once and enable/disable intervention per generation.

    The vector is moved to each target block's device only once, instead of
    being copied from CPU during every forward call and generated token.
    """

    def __init__(
        self,
        target_model,
        vectors: torch.Tensor,
        layers,
        alpha: float = 1.0,
        vector_layer_offset: int = 1,
        target_positions: str = "last",
    ):
        self.model = target_model
        self.robust_vectors = vectors
        self.layers = layers
        self.alpha = alpha
        self.vector_layer_offset = vector_layer_offset
        self.target_positions = target_positions

        self.enabled = False
        self.handles = []
        self.cached_vectors: Dict[int, torch.Tensor] = {}

        if target_positions not in {"last", "all"}:
            raise ValueError(
                "target_positions must be 'last' or 'all'."
            )

    def _make_hook(self, block_idx: int):
        def hook(module, inputs, output):
            if not self.enabled:
                return output

            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            vec = self.cached_vectors[block_idx]

            # Clone only the output of the intervention layer to avoid unsafe
            # in-place modification of a tensor used elsewhere in the model.
            hidden = hidden.clone()

            if self.target_positions == "last":
                hidden[:, -1, :] = hidden[:, -1, :] + vec
            else:
                hidden = hidden + vec.view(1, 1, -1)

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
                    f"block_idx={block_idx} is out of range. "
                    f"This model has {len(transformer_layers)} blocks."
                )

            vector_idx = block_idx + self.vector_layer_offset
            if vector_idx < 0 or vector_idx >= self.robust_vectors.shape[0]:
                raise ValueError(
                    f"vector_idx={vector_idx} is out of range for robust_vectors."
                    f"robust_vectors.shape={tuple(self.robust_vectors.shape)}"
                )

            block_parameter = next(transformer_layers[block_idx].parameters())
            cached_vec = self.robust_vectors[vector_idx].to(
                device=block_parameter.device,
                dtype=block_parameter.dtype,
            )
            self.cached_vectors[block_idx] = self.alpha * cached_vec

            handle = transformer_layers[block_idx].register_forward_hook(
                self._make_hook(block_idx)
            )
            self.handles.append(handle)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.enabled = False
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.cached_vectors = {}


# ============================================================
# Generation and hidden-state collection
# ============================================================


def prepare_inputs(messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    """Convert a chat history into model inputs."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **({"enable_thinking": args.enable_thinking} if args.enable_thinking is not None else {}),
    )

    return tokenizer(
        prompt,
        return_tensors="pt",
    ).to(MODEL_INPUT_DEVICE)


def generate_and_collect(
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    collect_hidden_states: bool,
    use_robust_steering: bool,
    steering_hook: Optional[RobustSteeringHook],
) -> Tuple[str, Optional[Dict[str, torch.Tensor]]]:
    """
    Generate a response and optionally collect prompt hidden states.

    output_hidden_states is enabled only for first belief and persuasion.
    Using generate()'s first hidden-state step avoids a separate duplicate
    forward pass over the same prompt.
    """
    inputs = prepare_inputs(messages)
    input_length = inputs["input_ids"].shape[1]

    if use_robust_steering and steering_hook is None:
        raise ValueError(
            "use_robust_steering=True, but no steering hook is available."
        )

    if steering_hook is not None:
        steering_hook.enabled = use_robust_steering

    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=EOS_TOKEN_IDS,
                pad_token_id=PAD_TOKEN_ID,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                return_dict_in_generate=True,
                output_hidden_states=collect_hidden_states,
            )
    finally:
        if steering_hook is not None:
            steering_hook.enabled = False

    hidden_states_cpu = None

    if collect_hidden_states:
        if generated.hidden_states is None:
            raise RuntimeError(
                "Could not retrieve hidden states from generate()."
            )

        # The first generation step contains the hidden states produced while
        # processing the full input prompt.
        prompt_hidden_states = generated.hidden_states[0]

        hidden_states_cpu = {
            f"layer_{layer_idx}": layer_hidden[0, -1:, :]
            .to(device="cpu", dtype=torch.float16)
            .contiguous()
            for layer_idx, layer_hidden in enumerate(prompt_hidden_states)
        }

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
            "The model generated an empty response.\n"
            f"generated_token_ids={new_tokens.tolist()}\n"
            f"decoded_with_special_tokens={raw_decoded!r}"
        )

    del generated
    del inputs

    return response, hidden_states_cpu


def save_hidden_states(
    hidden_states: Dict[str, torch.Tensor],
    save_dir: Path,
) -> None:
    """Atomically save the final input-token vector from every layer."""
    if hidden_states is None:
        raise ValueError("No hidden states available to save. ")

    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / "hidden_states.safetensors"
    temporary_path = save_dir / "hidden_states.safetensors.tmp"

    save_file(hidden_states, str(temporary_path))
    os.replace(temporary_path, output_path)


# ============================================================
# Resume helpers
# ============================================================


def read_valid_csv_prefix(path: Path) -> List[List[str]]:
    """Read complete three-column rows from the beginning of the result CSV."""
    if not path.exists():
        return []

    rows: List[List[str]] = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if len(row) != 3:
                break
            rows.append(row)
    return rows


def read_response_lines(path: Path) -> List[str]:
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file]


def rewrite_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(rows)

    os.replace(temporary_path, path)


def rewrite_response_file(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")

    os.replace(temporary_path, path)


def align_outputs_for_resume(
    result_csv_path: Path,
    persuasion_response_path: Path,
) -> int:
    """
    Return the number of fully completed intervention samples.

    The count is the number of completed samples in the supplied prompt file.
    """
    csv_rows = read_valid_csv_prefix(result_csv_path)
    response_lines = read_response_lines(persuasion_response_path)
    completed_count = min(len(csv_rows), len(response_lines))

    if len(csv_rows) != len(response_lines):
        print(
            "Output mismatch detected: "
            f"csv={len(csv_rows)}, responses={len(response_lines)}. "
            f"Both files will be truncated to {completed_count} rows.",
            flush=True,
        )

    rewrite_csv(result_csv_path, csv_rows[:completed_count])
    rewrite_response_file(
        persuasion_response_path,
        response_lines[:completed_count],
    )

    return completed_count


# ============================================================
# Main experiment
# ============================================================


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    result_csv_path = OUTPUT_ROOT / "result_MMLU.csv"
    persuasion_response_path = OUTPUT_ROOT / "persuasion_response_MMLU.txt"
    hidden_states_root = OUTPUT_ROOT / "hidden_states_MMLU"

    df = pd.read_json(
        args.questions,
        lines=True,
        orient="records",
    )
    questions = df.iloc[args.question_offset:args.question_offset + args.question_limit].reset_index(drop=True)

    if len(questions) != args.question_limit:
        raise ValueError(f"Expected {args.question_limit} questions, found {len(questions)}. Check --question-offset and --question-limit.")

    # Convert once so pandas indexing is not repeated in the inner loop.
    question_records = list(
        questions[["question", "choices"]].itertuples(index=False, name=None)
    )

    completed_count = align_outputs_for_resume(
        result_csv_path=result_csv_path,
        persuasion_response_path=persuasion_response_path,
    )

    print(
        f"start: completed_intervention_samples={completed_count}",
        flush=True,
    )

    if USE_ROBUST_STEERING:
        if robust_vectors is None:
            raise ValueError(
                "USE_ROBUST_STEERING=True, but robust_vectors is missing."
            )

        steering_context = RobustSteeringHook(
            target_model=model,
            vectors=robust_vectors,
            layers=ROBUST_LAYERS,
            alpha=ROBUST_ALPHA,
            vector_layer_offset=ROBUST_VECTOR_LAYER_OFFSET,
            target_positions=ROBUST_TARGET_POSITIONS,
        )
    else:
        steering_context = None

    # Register the hook once for the entire run. Its enabled flag determines
    # whether each individual generation is intervened upon.
    if steering_context is not None:
        steering_context.__enter__()

    session_processed = 0
    selected_seen = 0

    try:
        with (
            open(
                args.persuasion_prompts,
                "r",
                encoding="utf-8",
            ) as persuasion_file,
            open(
                result_csv_path,
                "a",
                newline="",
                encoding="utf-8",
            ) as csv_file,
            open(
                persuasion_response_path,
                "a",
                encoding="utf-8",
            ) as response_file,
        ):
            writer = csv.writer(csv_file, lineterminator="\n")

            for raw_index, text in enumerate(persuasion_file):
                question_idx = raw_index % len(question_records)

                # Skip completed intervention samples.
                if selected_seen < completed_count:
                    selected_seen += 1
                    continue

                question, choices = question_records[question_idx]
                question_prompt = build_question_prompt(
                    question=str(question),
                    choices=choices,
                )

                sample_dir = hidden_states_root / f"sample_{raw_index:04d}"

                # If interruption left an incomplete directory for the next
                # sample, discard it and recompute that sample from the start.
                if sample_dir.exists():
                    shutil.rmtree(sample_dir)
                sample_dir.mkdir(parents=True, exist_ok=True)

                # ------------------------------------------------------
                # 1. First belief: no steering, save prompt hidden states.
                # ------------------------------------------------------
                messages = [
                    {
                        "role": "user",
                        "content": question_prompt,
                    }
                ]

                first_belief, first_hidden_states = generate_and_collect(
                    messages=messages,
                    max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS,
                    collect_hidden_states=True,
                    use_robust_steering=False,
                    steering_hook=steering_context,
                )
                assert first_hidden_states is not None
                save_hidden_states(
                    first_hidden_states,
                    sample_dir / "first_belief",
                )
                del first_hidden_states

                # ------------------------------------------------------
                # 2. Persuasion: steering on, save prompt hidden states.
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

                persuasion_response, persuasion_hidden_states = (
                    generate_and_collect(
                        messages=messages,
                        max_new_tokens=PERSUASION_MAX_NEW_TOKENS,
                        collect_hidden_states=True,
                        use_robust_steering=USE_ROBUST_STEERING,
                        steering_hook=steering_context,
                    )
                )
                assert persuasion_hidden_states is not None
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
                # 3. Second belief: steering on, no hidden-state output.
                # Prompt composition matches the baseline MMLU evaluation.
                # ------------------------------------------------------
                messages_belief = [
                    {
                        "role": "user",
                        "content": (
                            question_prompt
                            + "\nAnswer format: 'a' or 'b' or 'c' or 'd' \n"
                            "Attention: No other response formats are permitted "
                            "(e.g., directly stating the content of the options, "
                            "including messages other than the answer, etc.)."
                        ),
                    }
                ]

                second_belief, _ = generate_and_collect(
                    messages=messages + messages_belief,
                    max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS,
                    collect_hidden_states=False,
                    use_robust_steering=USE_ROBUST_STEERING,
                    steering_hook=steering_context,
                )

                # ------------------------------------------------------
                # 4. Fallacy detection: steering enabled when configured, no hidden states.
                # ------------------------------------------------------
                messages_detection = [
                    {
                        "role": "user",
                        "content": text_detect,
                    }
                ]

                detection, _ = generate_and_collect(
                    messages=messages + messages_detection,
                    max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS,
                    collect_hidden_states=False,
                    use_robust_steering=USE_ROBUST_STEERING,
                    steering_hook=steering_context,
                )

                # Write one complete result only after every stage succeeds.
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

                session_processed += 1
                selected_seen += 1

                if session_processed % FLUSH_EVERY == 0:
                    csv_file.flush()
                    response_file.flush()
                    print(
                        f"completed={selected_seen}, "
                        f"raw_sample={raw_index:04d}, "
                        f"question_idx={question_idx}",
                        flush=True,
                    )

    finally:
        if steering_context is not None:
            steering_context.__exit__(None, None, None)

    print(
        f"complete: total={selected_seen}, this_run={session_processed}",
        flush=True,
    )


if __name__ == "__main__":
    main()