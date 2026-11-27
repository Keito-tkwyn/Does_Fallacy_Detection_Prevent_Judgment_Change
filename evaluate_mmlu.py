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
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=None, help="Resume from this zero-based sample index; defaults to the common completed output prefix.")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False, help="Override the chat template thinking setting when supported.")
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

# generate()ではmax_new_tokensだけを指定する。
# モデル側のmax_lengthが残っていることによる警告を避ける。
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


def collect_prompt_hidden_states(
    inputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    応答生成前のプロンプトに対するhidden statesを取得する。

    保存形式は元コードと同じで、各層について
    [1, hidden_size] の最終入力トークンのテンソルを保存する。
    GPUメモリを早く解放するため、ここでfloat16・CPUへ移す。
    """
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states_cpu = {
        f"layer_{layer_idx}": layer_hidden[0, -1:, :]
        .to(
            device="cpu",
            dtype=torch.float16,
        )
        .contiguous()
        for layer_idx, layer_hidden in enumerate(outputs.hidden_states)
    }

    del outputs
    return hidden_states_cpu


def generate_response(
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
) -> str:
    """貪欲法で応答を生成する。"""
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=EOS_TOKEN_IDS,
            pad_token_id=PAD_TOKEN_ID,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated[0, input_length:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    # 空文字をそのままCSVへ書き込まず、原因確認ができる状態で停止する。
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
    return response


def generate_and_collect(
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    collect_hidden_states: bool,
) -> Tuple[str, Optional[Dict[str, torch.Tensor]]]:
    """
    応答を生成し、必要な場合だけ生成前hidden statesを取得する。

    second_beliefとdetectionではcollect_hidden_states=Falseとし、
    不要な全層forwardを実行しない。
    """
    inputs = prepare_inputs(messages)

    hidden_states = None
    if collect_hidden_states:
        hidden_states = collect_prompt_hidden_states(inputs)

    response = generate_response(
        inputs=inputs,
        max_new_tokens=max_new_tokens,
    )

    del inputs
    return response, hidden_states


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
For multiple-choice questions, please select the answer(s) you believe to be correct.
The questions follow this format:

"question"
"a:text A b:text B c:text C d:text D"

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
    """MMLUの質問と選択肢を共通形式で組み立てる。"""
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

def read_valid_csv_prefix(path: Path) -> List[List[str]]:
    """
    結果CSVの先頭から、3列そろった正常な行だけを読み込む。

    中断によって最終行が壊れている場合は、
    その行以降を未完了とみなす。
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
    """説得応答TXTに保存済みの行を読み込む。"""
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file]


def rewrite_csv(path: Path, rows: List[List[str]]) -> None:
    """CSVを指定した行数まで切り戻す。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(rows)

        file.flush()
        os.fsync(file.fileno())


def rewrite_response_file(path: Path, lines: List[str]) -> None:
    """応答TXTを指定した行数まで切り戻す。"""
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
    再開インデックスを決定する。

    自動再開時：
        CSVとTXTの両方に記録されている共通行数から再開する。

    --start-index指定時：
        指定位置まで既存結果を切り戻して再開する。
    """
    csv_rows = read_valid_csv_prefix(result_csv_path)
    response_lines = read_response_lines(persuasion_response_path)

    # 両方に保存されている位置だけを完了済みとする。
    common_completed = min(
        len(csv_rows),
        len(response_lines),
    )

    if requested_start_index is None:
        start_index = common_completed

    else:
        if requested_start_index < 0:
            raise ValueError(
                "--start-indexは0以上にしてください。"
            )

        if requested_start_index > common_completed:
            raise ValueError(
                f"--start-index={requested_start_index}を指定しましたが、"
                f"CSVとTXTの共通完了件数は{common_completed}です。"
                "未記録の区間を飛ばすことはできません。"
            )

        start_index = requested_start_index

    # CSVとTXTの行数が異なる場合や手動指定した場合は、
    # 両方を同じ開始位置まで切り戻す。
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
        args.questions,
        lines=True,
        orient="records",
    )
    questions = df.iloc[args.question_offset:args.question_offset + args.question_limit].reset_index(drop=True)
    if len(questions) != args.question_limit:
        raise ValueError(f"Expected {args.question_limit} questions, found {len(questions)}. Check --question-offset and --question-limit.")

    if questions.empty:
        raise ValueError(
            "No questions found in the input JSONL."
        )

    start_index = resolve_start_index(
        result_csv_path=result_csv_path,
        persuasion_response_path=persuasion_response_path,
        requested_start_index=args.start_index,
    )

    session_processed_count = 0

    print(
        f"start: resume_index={start_index}",
        flush=True,
    )

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

        # 完了済みの説得文を読み飛ばす。
        remaining_lines = islice(persuasion_file,start_index,None,)

        # iには元ファイル上の絶対インデックスを入れる。
        for i, text in enumerate(remaining_lines,start=start_index,):
            question_idx = i % len(questions)
            question = str(questions.iloc[question_idx]["question"])
            choices = questions.iloc[question_idx]["choices"]
            question_prompt = build_question_prompt(question, choices)

            sample_dir = hidden_states_root / f"sample_{i:04d}"

            # 中断時に残った不完全なhidden statesを削除する。
            if sample_dir.exists():
                shutil.rmtree(sample_dir)

            sample_dir.mkdir(parents=True,exist_ok=True,)

            # ----------------------------------------------------------
            # 1. first belief
            # 生成前hidden statesを保存する。
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

            # 途中で停止しても直近の結果がファイルへ反映されるよう定期flushする。
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

                print(
                    f"complete: "
                    f"total={start_index + session_processed_count}, "
                    f"this_run={session_processed_count}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
