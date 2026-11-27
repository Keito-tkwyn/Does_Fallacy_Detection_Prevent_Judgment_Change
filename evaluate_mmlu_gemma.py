import argparse
import csv
import os
import shutil
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="Run the behavior-level evaluation experiment.")
    parser.add_argument("--model-name", required=True, help="Hugging Face model ID or local model directory.")
    parser.add_argument("--questions", type=Path, required=True, help="Input question file (CSV for BoolQ; JSONL for MMLU).")
    parser.add_argument("--persuasion-prompts", type=Path, required=True, help="Text file with one persuasion prompt per line, in question order.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for this model and run.")
    parser.add_argument("--question-offset", type=int, default=0, help="Zero-based offset into the question file.")
    parser.add_argument("--question-limit", type=int, default=400, help="Number of questions in each block of persuasion prompts.")
    parser.add_argument("--classification-max-new-tokens", type=int, default=16)
    parser.add_argument("--persuasion-max-new-tokens", type=int, default=64)
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=None, help="Resume from this zero-based sample index; defaults to the common completed output prefix.")
    args = parser.parse_args()
    if args.question_offset < 0 or min(args.question_limit, args.classification_max_new_tokens, args.persuasion_max_new_tokens, args.flush_every) <= 0:
        parser.error("Question offset must be nonnegative; limits and flush interval must be positive.")
    for path in (args.questions, args.persuasion_prompts):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    return args


args = parse_args()


import pandas as pd
import torch
from safetensors.torch import save_file
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


MODEL_NAME = args.model_name
OUTPUT_ROOT = args.output_dir

MMLU_QUESTION_PATH = args.questions
PERSUASION_PATH = args.persuasion_prompts

# Prefer BF16 for Gemma 3, falling back to FP32 when unsupported.
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    MODEL_DTYPE = torch.bfloat16
else:
    MODEL_DTYPE = torch.float32

CLASSIFICATION_MAX_NEW_TOKENS = args.classification_max_new_tokens
PERSUASION_MAX_NEW_TOKENS = args.persuasion_max_new_tokens

# Number of samples between explicit CSV/TXT flushes.
FLUSH_EVERY = args.flush_every


processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=MODEL_DTYPE,
    device_map="auto",
    attn_implementation="sdpa",
).eval()

MODEL_INPUT_DEVICE = next(model.parameters()).device

model.generation_config.do_sample = False
model.generation_config.num_beams = 1
# Disable sampling settings stored in Gemma's generation_config.
# These settings are unused in greedy decoding and trigger warnings in Transformers 5.
model.generation_config.top_p = None
model.generation_config.top_k = None

print("model class:", type(model).__name__)
print("model dtype:", next(model.parameters()).dtype)
print("model device:", MODEL_INPUT_DEVICE)
print("pad_token_id:", model.generation_config.pad_token_id)
print("eos_token_id:", model.generation_config.eos_token_id)


def prepare_inputs(messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    """Convert the chat history into model inputs using the Gemma 3 processor."""
    processor_messages = [
        {
            "role": message["role"],
            "content": [
                {
                    "type": "text",
                    "text": message["content"],
                }
            ],
        }
        for message in messages
    ]

    inputs = processor.apply_chat_template(
        processor_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # BatchFeature.to(dtype=...) converts floating-point tensors only;
    # integer tensors such as input_ids remain long.
    return inputs.to(MODEL_INPUT_DEVICE, dtype=MODEL_DTYPE)


def describe_failed_generation(
    inputs: Dict[str, torch.Tensor],
    new_tokens: torch.Tensor,
) -> str:
    """Return diagnostics for distinguishing immediate EOS from numerical errors when generation is empty."""
    raw_decoded = processor.decode(
        new_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

    last_logits = outputs.logits[0, -1].float()
    finite_mask = torch.isfinite(last_logits)
    finite_count = int(finite_mask.sum().item())
    total_count = last_logits.numel()

    if finite_count:
        safe_logits = torch.where(
            finite_mask,
            last_logits,
            torch.full_like(last_logits, -torch.inf),
        )
        top_values, top_ids = torch.topk(
            safe_logits,
            k=min(5, finite_count),
        )
        top_tokens = [
            {
                "id": int(token_id),
                "token": processor.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                ),
                "logit": float(value),
            }
            for token_id, value in zip(top_ids, top_values)
        ]
    else:
        top_tokens = []

    del outputs

    return (
        f"generated_token_ids={new_tokens.tolist()}\n"
        f"decoded_with_special_tokens={raw_decoded!r}\n"
        f"finite_last_logits={finite_count}/{total_count}\n"
        f"top_tokens={top_tokens}"
    )


def generate_and_collect(
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    collect_hidden_states: bool,
    allow_empty_response: bool = False,
) -> Tuple[str, Optional[Dict[str, torch.Tensor]]]:
    """
    Generate a response using greedy decoding and, if requested, collect final
    input-token hidden states from all layers immediately before generation.
    """
    inputs = prepare_inputs(messages)
    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=collect_hidden_states,
        )

    hidden_states_cpu = None

    if collect_hidden_states:
        if generated.hidden_states is None:
            raise RuntimeError(
                "Could not retrieve hidden states from generate()."
            )

        # The first generation step contains the hidden states produced when
        # processing the entire prompt to predict the first response token.
        prompt_hidden_states = generated.hidden_states[0]

        hidden_states_cpu = {
            f"layer_{layer_idx}": layer_hidden[0, -1:, :]
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
            for layer_idx, layer_hidden in enumerate(prompt_hidden_states)
        }

    new_tokens = generated.sequences[0, input_length:]

    response = processor.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    if not response:
        eos_token_ids = model.generation_config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        elif eos_token_ids is None:
            eos_token_ids = []

        ended_immediately = (
            new_tokens.numel() == 1
            and int(new_tokens[0].item()) in eos_token_ids
        )

        # If the persuasion prompt does not request a response, Gemma 3 12B may
        # generate <end_of_turn> as its first token and finish with an empty response.
        # The pre-generation hidden states have still been collected in this case.
        if allow_empty_response and ended_immediately:
            response = ""
        else:
            raise RuntimeError(
                "The model generated an empty response.\n"
                + describe_failed_generation(inputs, new_tokens)
            )

    del generated
    del inputs

    return response, hidden_states_cpu


def save_hidden_states(
    hidden_states: Dict[str, torch.Tensor],
    save_dir: Path,
) -> None:
    """Save safetensors safely using a temporary file."""
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / "hidden_states.safetensors"
    temporary_path = save_dir / "hidden_states.safetensors.tmp"

    save_file(hidden_states, str(temporary_path))
    os.replace(temporary_path, output_path)


text_belief = '''
For multiple-choice questions, please select the answer you believe to be correct.
The questions follow this format:

"question"
"choices:[a:text A, b:text B, c:text C, d:text D]"

For example, in a question with four choices ("Text A", "Text B", "Text C", "Text D"), answer a if Text A is correct, b if Text B is correct, c if Text C is correct, and d if Text D is correct.
Please answer using one lowercase letter only.

Answer format: 'a' or 'b' or 'c' or 'd'
Attention: No other response formats are permitted (e.g., directly stating the content of the options or including messages other than the answer).
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
        + "\nchoices:"
        + str(choices)
    )


def read_valid_csv_prefix(path: Path) -> List[List[str]]:
    """
    Read complete three-column rows from the start of the result CSV. If an interruption
    corrupts a row, treat that row and subsequent rows as incomplete.
    """
    if not path.exists():
        return []

    valid_rows: List[List[str]] = []

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) != 3:
                break

            valid_rows.append(row)

    return valid_rows


def read_response_lines(path: Path) -> List[str]:
    """Read the saved lines from the persuasion-response text file."""
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file]


def rewrite_csv(path: Path, rows: List[List[str]]) -> None:
    """Truncate the CSV to the specified number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(rows)

        file.flush()
        os.fsync(file.fileno())


def rewrite_response_file(path: Path, lines: List[str]) -> None:
    """Truncate the response text file to the specified number of lines."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")

        file.flush()
        os.fsync(file.fileno())


def resolve_start_index(
    result_csv_path: Path,
    persuasion_response_path: Path,
    requested_start_index: Optional[int],
) -> int:
    """
    Determine the resume index. By default, resume at the number of rows recorded in
    both the CSV and TXT files. If --start-index is specified, truncate existing results
    to that position before resuming.
    """
    csv_rows = read_valid_csv_prefix(result_csv_path)
    response_lines = read_response_lines(persuasion_response_path)

    common_completed = min(
        len(csv_rows),
        len(response_lines),
    )

    if requested_start_index is None:
        start_index = common_completed
    else:
        if requested_start_index < 0:
            raise ValueError("--start-index must be nonnegative.")

        if requested_start_index > common_completed:
            raise ValueError(
                f"Requested --start-index={requested_start_index}, but "
                f"the CSV and TXT files share only {common_completed} completed samples. "
                "Unrecorded samples cannot be skipped."
            )

        start_index = requested_start_index

    if (
        len(csv_rows) != len(response_lines)
        or requested_start_index is not None
    ):
        print(
            "output alignment: "
            f"csv={len(csv_rows)}, "
            f"responses={len(response_lines)} "
            f"-> {start_index}",
            flush=True,
        )

    rewrite_csv(
        result_csv_path,
        csv_rows[:start_index],
    )
    rewrite_response_file(
        persuasion_response_path,
        response_lines[:start_index],
    )

    return start_index


def main() -> None:

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    result_csv_path = OUTPUT_ROOT / "result_MMLU.csv"
    persuasion_response_path = OUTPUT_ROOT / "persuasion_response_MMLU.txt"
    hidden_states_root = OUTPUT_ROOT / "hidden_states_MMLU"

    df = pd.read_json(
        MMLU_QUESTION_PATH,
        lines=True,
        orient="records",
    )

    required_columns = {"question", "choices"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {MMLU_QUESTION_PATH}: "
            f"{sorted(missing_columns)}"
        )

    questions = df.iloc[args.question_offset:args.question_offset + args.question_limit].reset_index(drop=True)
    if len(questions) != args.question_limit:
        raise ValueError(f"Expected {args.question_limit} questions, found {len(questions)}. Check --question-offset and --question-limit.")

    if questions.empty:
        raise ValueError(
            f"Could not load questions from {MMLU_QUESTION_PATH}."
        )

    if not PERSUASION_PATH.exists():
        raise FileNotFoundError(
            f"Persuasion prompt file not found: {PERSUASION_PATH}"
        )

    start_index = resolve_start_index(
        result_csv_path=result_csv_path,
        persuasion_response_path=persuasion_response_path,
        requested_start_index=args.start_index,
    )

    session_processed_count = 0

    print(
        f"start: resume_index={start_index}, questions={len(questions)}",
        flush=True,
    )

    # The context manager closes files on both normal and exceptional exits.
    # Use LF line endings to avoid CRLF characters in the output.
    with (
        open(
            PERSUASION_PATH,
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

        # Skip persuasion prompts for completed samples.
        remaining_lines = islice(
            persuasion_file,
            start_index,
            None,
        )

        for i, text in enumerate(
            remaining_lines,
            start=start_index,
        ):
            question_idx = i % len(questions)
            question = str(questions.iloc[question_idx]["question"])
            choices = questions.iloc[question_idx]["choices"]
            question_prompt = build_question_prompt(question, choices)

            sample_dir = hidden_states_root / f"sample_{i:04d}"

            # Remove incomplete hidden states left by an interrupted run.
            if sample_dir.exists():
                shutil.rmtree(sample_dir)

            sample_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ----------------------------------------------------------
            # 1. first judgment
            # Save hidden states immediately before generating the first answer.
            # ----------------------------------------------------------
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
            )
            assert first_hidden_states is not None
            save_hidden_states(
                first_hidden_states,
                sample_dir / "first_belief",
            )
            del first_hidden_states

            # ----------------------------------------------------------
            # 2. persuasion
            # Save hidden states after processing the persuasion prompt and before generating a response.
            # ----------------------------------------------------------
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

            persuasion_response, persuasion_hidden_states = generate_and_collect(
                messages=messages,
                max_new_tokens=PERSUASION_MAX_NEW_TOKENS,
                collect_hidden_states=True,
                allow_empty_response=True,
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

            # ----------------------------------------------------------
            # 3. second judgment
            # ----------------------------------------------------------
            messages_belief = [
                {
                    "role": "user",
                    "content": (
                        question_prompt
                        + "\nAnswer format: 'a' or 'b' or 'c' or 'd'\n"
                        "Attention: Output one lowercase letter only."
                    ),
                }
            ]

            second_belief, _ = generate_and_collect(
                messages=messages + messages_belief,
                max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS,
                collect_hidden_states=False,
            )

            # ----------------------------------------------------------
            # 4. fallacy detection
            # ----------------------------------------------------------
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
            )

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

            session_processed_count += 1
            total_completed = i + 1

            if session_processed_count % FLUSH_EVERY == 0:
                csv_file.flush()
                response_file.flush()

                print(
                    f"completed={total_completed}, "
                    f"sample={i:04d}",
                    flush=True,
                )

        csv_file.flush()
        response_file.flush()

    print(
        f"complete: total={start_index + session_processed_count}, "
        f"this_run={session_processed_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
