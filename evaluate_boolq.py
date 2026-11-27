import argparse
import csv
import os
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
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None, help="Override the chat template thinking setting when supported.")
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
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = args.model_name
OUTPUT_ROOT = args.output_dir


CLASSIFICATION_MAX_NEW_TOKENS = args.classification_max_new_tokens
PERSUASION_MAX_NEW_TOKENS = args.persuasion_max_new_tokens

# 何サンプルごとにCSV/TXTを明示的にflushするか。
FLUSH_EVERY = args.flush_every


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

# generate()では呼び出し時のmax_new_tokensだけを使用する
model.generation_config.max_length = None
model.generation_config.max_new_tokens = None

eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

EOS_TOKEN_IDS = list(
    dict.fromkeys(
        token_id
        for token_id in [
            tokenizer.eos_token_id,
            eot_token_id,
        ]
        if token_id is not None
    )
)


MODEL_INPUT_DEVICE = next(model.parameters()).device
PAD_TOKEN_ID = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
)


def prepare_inputs(messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    """チャット履歴をモデル入力へ変換する。"""
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
) -> Tuple[str, Optional[Dict[str, torch.Tensor]]]:
    inputs = prepare_inputs(messages)
    input_length = inputs["input_ids"].shape[1]

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
            output_hidden_states=collect_hidden_states
        )

    hidden_states_cpu = None

    if collect_hidden_states:
        if generated.hidden_states is None:
            raise RuntimeError("generate()からhidden statesを取得できませんでした。")

        # 最初の生成ステップは、最初の応答トークンを予測するために
        # プロンプトを処理した際のhidden states。
        prompt_hidden_states = generated.hidden_states[0]

        hidden_states_cpu = {
            f"layer_{layer_idx}": layer_hidden[0, -1:, :]
            .to(
                device="cpu",
                dtype=torch.float16,
            )
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
            "モデルの生成結果が空文字になりました。\n"
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
    """safetensorsを一時ファイル経由で安全に保存する。"""
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / "hidden_states.safetensors"
    temporary_path = save_dir / "hidden_states.safetensors.tmp"

    save_file(hidden_states, str(temporary_path))
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


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    result_csv_path = OUTPUT_ROOT / "result_boolQ.csv"
    persuasion_response_path = OUTPUT_ROOT / "persuasion_response_boolQ.txt"
    hidden_states_root = OUTPUT_ROOT / "hidden_states_boolQ"

    df = pd.read_csv(
        args.questions,
        names=["question", "label"],
    )
    questions = df.iloc[args.question_offset:args.question_offset + args.question_limit].reset_index(drop=True)
    if len(questions) != args.question_limit:
        raise ValueError(f"Expected {args.question_limit} questions, found {len(questions)}. Check --question-offset and --question-limit.")

    if questions.empty:
        raise ValueError("No questions found in the input CSV.")

    processed_count = 0
    print("start", flush=True)

    # context managerにより、正常終了・例外終了のどちらでもファイルを閉じる。
    # lineterminator='\\n'により、cat -Aで^Mが表示されるCRLFを避ける。
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

        # readlines()を使わず1行ずつ処理し、入力ファイル全体をメモリに載せない。
        for i, text in enumerate(persuasion_file):

            
            question_idx = i % len(questions)
            question = str(questions.iloc[question_idx]["question"])

            sample_dir = hidden_states_root / f"sample_{i:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # ----------------------------------------------------------
            # 1. first belief
            # 生成前hidden statesを保存する。
            # ----------------------------------------------------------
            messages = [
                {
                    "role": "user",
                    "content": text_belief + "question:" + question,
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
            # 説得文を読み終え、応答を生成する前のhidden statesを保存する。
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
            # 3. second belief
            # hidden statesを保存しないので、生成前forwardを省略する。
            # ----------------------------------------------------------
            messages_belief = [
                {
                    "role": "user",
                    "content": (
                        text_belief
                        + "\n"
                        + question
                        + "\nAnswer Format: { answer(True/False) } "
                        "Attention: Please output only the results; "
                        "do not include any extra text beyond the required format."
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
            # hidden statesを保存しないので、生成前forwardを省略する。
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

            processed_count += 1

            # 途中で停止しても直近の結果がファイルへ反映されるよう定期flushする。
            if processed_count % FLUSH_EVERY == 0:
                csv_file.flush()
                response_file.flush()
                print(
                    f"processed={processed_count}, sample={i:04d}",
                    flush=True,
                )

    print(f"complete: {processed_count} samples", flush=True)


if __name__ == "__main__":
    main()
