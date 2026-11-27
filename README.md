# QA Datasets, Persuasion Prompts, and Experiment Code

This repository contains the QA datasets, persuasion prompts, and code used in our experiments on judgment change in large language models (LLMs) under persuasion containing logical fallacies.

The repository is organized as follows:

```text
.
├── QAdataset/
│   ├── boolQ.csv
│   ├── boolQ_val.csv
│   ├── MMLU.jsonl
│   └── MMLU_val.jsonl
├── persuasion_prompts/
│   ├── boolQ/
│   │   ├── [fallacy_type].txt
│   │   ├── ...
│   │   ├── dummy.txt
│   │   ├── all_fallacies.txt
│   │   └── val.txt
│   └── MMLU/
│       ├── [fallacy_type].txt
│       ├── ...
│       ├── dummy.txt
│       ├── all_fallacies.txt
│       └── val.txt
├── behavior_evaluation/
│   ├── evaluate_boolq.py
│   ├── evaluate_mmlu.py
│   ├── evaluate_boolq_gemma.py
│   └── evaluate_mmlu_gemma.py
├── linear_probing/
│   ├── linear_probe.py
│   └── linear_probe_gemma.py
├── causal_mediation/
│   ├── causal_mediation.py
│   └── causal_mediation_gemma.py
├── vector_intervention/
│   ├── construct_steering_vector.py
│   ├── steer_boolq.py
│   ├── steer_mmlu.py
│   ├── steer_boolq_gemma.py
│   └── steer_mmlu_gemma.py
├── requirements.txt
├── .gitignore
└── README.md
```

## QA Datasets

The `QAdataset/` directory contains the QA examples used in our experiments.

We use questions from two datasets:

* **BoolQ**, a binary QA task
* **MMLU**, a four-choice multiple-choice QA task

For each dataset, we use a total of **500 questions**. These are divided into 400 questions for the main analyses and 100 held-out questions for evaluating the vector intervention.

| File             | Number of questions | Usage                                           |
| ---------------- | ------------------: | ----------------------------------------------- |
| `boolQ.csv`      |                 400 | Behavior-level evaluation and internal analyses |
| `boolQ_val.csv`  |                 100 | Vector intervention evaluation                  |
| `MMLU.jsonl`     |                 400 | Behavior-level evaluation and internal analyses |
| `MMLU_val.jsonl` |                 100 | Vector intervention evaluation                  |

The `_val` files contain the held-out subset used for the vector intervention experiments.

### BoolQ data format

The BoolQ files are stored in CSV format.

Each example consists of:

* a question
* the correct answer

The answer is either `True` or `False`.

Conceptually, each row has the following structure:

```text
(question, answer)
```

### MMLU data format

The MMLU files are stored in JSONL format.

Each example contains the following fields:

| Field       | Description                                                 |
| ----------- | ----------------------------------------------------------- |
| `question`  | The question text                                           |
| `choices`   | The four answer choices                                     |
| `answer`    | The correct answer                                          |
| `incorrect` | A predefined incorrect answer used as the persuasion target |

The `incorrect` field specifies the incorrect answer toward which the model is intended to be persuaded.

The fallacy-based persuasion prompts for each MMLU question are generated to support this predefined incorrect answer.

For example, if the correct answer is `b` and `incorrect` is `d`, the persuasion prompts corresponding to that question are constructed to support answer `d`.

## Persuasion Prompts

The `persuasion_prompts/` directory contains the persuasion prompts corresponding to the QA examples.

Separate directories are provided for BoolQ and MMLU:

```text
persuasion_prompts/
├── boolQ/
└── MMLU/
```

Both directories follow the same structure.

### Fallacy-specific prompt files

For each of the **13 logical fallacy types**, a separate text file is provided:

```text
[fallacy_type].txt
```

Each file contains **500 persuasion prompts**, with one prompt corresponding to each of the 500 QA examples.

The prompts are constructed to support an incorrect answer while containing the designated logical fallacy.

For MMLU, the supported incorrect answer corresponds to the answer specified by the `incorrect` field in the QA data.

Across the 13 fallacy types, each dataset therefore contains:

```text
13 fallacy types × 500 questions = 6,500 fallacy-based prompts
```

### `dummy.txt`

`dummy.txt` contains **500 prompts without logical fallacies**.

Unlike the fallacy-based prompts, which support an incorrect answer, the prompts in `dummy.txt` support the **correct answer** to the corresponding question.

These prompts are used as the no-fallacy control condition.

### `all_fallacies.txt`

`all_fallacies.txt` contains the prompts corresponding to the 400-question subset used for the behavior-level evaluation and internal analyses.

It combines all 13 fallacy conditions and the no-fallacy control condition:

```text
14 conditions × 400 questions = 5,600 prompts
```

The 14 conditions consist of:

* 13 logical fallacy conditions, each supporting an incorrect answer
* 1 no-fallacy (`dummy`) condition supporting the correct answer

### `val.txt`

`val.txt` contains the prompts corresponding to the held-out 100-question subset used for evaluating the vector intervention.

It contains the same 14 conditions:

```text
14 conditions × 100 questions = 1,400 prompts
```

The 14 conditions again consist of the 13 fallacy conditions and the no-fallacy control condition.

## Data Summary

For each of BoolQ and MMLU, the persuasion prompt directory contains:

| Data                      | Number of prompts |
| ------------------------- | ----------------: |
| 13 fallacy-specific files |             6,500 |
| `dummy.txt`               |               500 |
| `all_fallacies.txt`       |             5,600 |
| `val.txt`                 |             1,400 |

`all_fallacies.txt` and `val.txt` are reorganized subsets of the prompts contained in the 13 fallacy-specific files and `dummy.txt`. They are provided for convenience and do not contain independently generated additional prompts.

The 500 QA examples and their corresponding prompts are divided as follows:

```text
500 questions
├── 400 questions → main analyses
│   └── 14 × 400 = 5,600 prompts in all_fallacies.txt
│
└── 100 questions → vector intervention evaluation
    └── 14 × 100 = 1,400 prompts in val.txt
```

## Intended Use

These data were prepared for experiments examining how LLM judgments change after receiving persuasive prompts containing logical fallacies.

The 400-question subset is used for behavior-level evaluation and analyses of internal model representations. The held-out 100-question subset is used to evaluate vector-based interventions.

The released data can also be used for research on topics such as:

* judgment change in LLMs
* persuasion under flawed reasoning
* logical fallacy detection
* robustness of model judgments
* analysis of internal model representations

## Experiment Code

### Requirements

Use Python 3.10 or later. Install a PyTorch build appropriate for your hardware and CUDA version, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The inference scripts require a compatible Transformers release (the supplied implementations use the `dtype` argument to `from_pretrained` and Gemma 3 APIs), sufficient device memory, and access to the selected model. Some model repositories require Hugging Face authentication. Authenticate outside the source code. The exact versions from the original experimental environments were not supplied with these scripts; record and verify them before claiming exact reproduction. `requirements.txt` lists dependencies rather than a verified environment lockfile.

All scripts are command-line entry points. Run any script with `--help` to see its parameters. Use distinct output directories for each model, dataset, and intervention condition. Some scripts append/resume, while others overwrite output CSV/TXT files; do not reuse a directory for a different experimental configuration.

The following examples assume the existing `QAdataset/` and `persuasion_prompts/` directories are at the repository root.

### 1. Behavior-level evaluation

For BoolQ, the question file is a headerless CSV with columns `question,label`. For MMLU, the question file is JSONL containing at least `question` and `choices`. The prompt file contains one persuasion prompt per line; each block must follow the same question order as the selected input questions. A sample is matched to a question by its index modulo the number of selected questions. Do not shuffle either file independently.

Example for a Llama model on BoolQ:

```bash
python behavior_evaluation/evaluate_boolq.py \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --questions QAdataset/boolQ.csv \
  --persuasion-prompts persuasion_prompts/boolQ/all_fallacies.txt \
  --output-dir experiment/llama_boolq \
  --question-limit 400
```

Example for Gemma 3 on MMLU:

```bash
python behavior_evaluation/evaluate_mmlu_gemma.py \
  --model-name google/gemma-3-4b-it \
  --questions QAdataset/MMLU.jsonl \
  --persuasion-prompts persuasion_prompts/MMLU/all_fallacies.txt \
  --output-dir experiment/gemma_mmlu \
  --question-limit 400
```

Choose `evaluate_mmlu.py` for the non-Gemma MMLU implementation and `evaluate_boolq_gemma.py` for Gemma 3 on BoolQ. The non-Gemma scripts expose `--enable-thinking` / `--no-enable-thinking` for models that support the corresponding chat-template parameter. By default, the BoolQ implementation leaves the template setting unspecified; the MMLU implementation passes `False`, as in the original scripts. Classification and persuasion response limits default to 16 and 64 new tokens, respectively. The original generation settings, precision, conversation prompts, hidden-state collection methods, and resume behavior are retained.

The behavior scripts produce a headerless `result_boolQ.csv` or `result_MMLU.csv` containing `first_belief,second_belief,detection`, a persuasion-response text file, and `hidden_states_boolQ/` or `hidden_states_MMLU/`. Hidden-state files use the following layout:

```text
hidden_states_boolQ/
    sample_0000/
        first_belief/hidden_states.safetensors
        persuasion/hidden_states.safetensors
    sample_0001/
        ...
```

The CSV's zero-based row index corresponds to the sample ID. Hidden-state index 0 is the embedding output in the original representation layout. For a question file containing both the training and validation partitions, select a contiguous subset using `--question-offset` and `--question-limit`.

`evaluate_mmlu.py`, `evaluate_mmlu_gemma.py`, and `evaluate_boolq_gemma.py` support resuming from existing CSV/TXT outputs and accept `--start-index`. `evaluate_boolq.py` starts a new CSV/TXT run.

### 2. Linear probing

```bash
python linear_probing/linear_probe.py \
  --hidden-root experiment/llama_boolq/hidden_states_boolQ \
  --label-csv experiment/llama_boolq/result_boolQ.csv \
  --output-dir experiment/llama_boolq/probe_results \
  --state persuasion --target detection --n-splits 5 --random-state 42
```

Use `linear_probe_gemma.py` for the Gemma-specific implementation. It retains its original float64 conversion and non-finite filtering; `--nonfinite-policy drop_layers` (default) or `drop_samples` selects the filtering policy. Both implementations accept `--target detection`, `belief_change`, or `both`. Set this explicitly when comparing results, since the original defaults differ (`detection` for the standard script, `both` for Gemma). The original cross-validation and metric calculation logic is unchanged. Use separate output folders for different configurations.

### 3. Statistical causal mediation analysis

```bash
python causal_mediation/causal_mediation.py \
  --hidden-root experiment/llama_boolq/hidden_states_boolQ \
  --label-csv experiment/llama_boolq/result_boolQ.csv \
  --output-dir experiment/llama_boolq/mediation_results \
  --fallacy-last-id 5199 --n-splits 5 --random-state 42
```

Use `causal_mediation_gemma.py` for the Gemma-specific float64/non-finite handling. Both scripts accept `--skip-first-layer`, `--use-first-belief-control`, and `--bootstrap N`. The default bootstrap count is 0; set it to the count used in the paper if confidence intervals are required. The input sample IDs through `--fallacy-last-id` (inclusive) receive treatment 1; the remaining samples receive treatment 0. The default 5199 assumes 13 blocks of 400 fallacious prompts followed by a dummy block. Change it if the input ordering or block size differs.

**Layer numbering:** the original standard mediation implementation reports array indices after removing the embedding when `--skip-first-layer` is used (output layer 0 then corresponds to saved hidden-state index 1). The Gemma mediation implementation retains the original hidden-state indices in its reported layers. This difference is intentionally retained to avoid changing historical result conventions; account for it when selecting intervention blocks or comparing results.

The analysis writes layer-wise mediation results, out-of-fold probe scores, and a potentially large intermediate memory-mapped array. Keep output directories separate by model/dataset.

### 4. Steering vector construction

Use the behavior-level outputs from the 400-question subset to construct vectors. The script matches each CSV row to its question and saved persuasion hidden states. By default, it uses fallacy samples with a correct first judgment and excludes samples containing non-finite hidden states. The robust direction is the mean of hidden states for Detect & Maintained minus the mean for Detect & Changed (within the initially correct subset). The script also constructs a detection direction and an orthogonalized robust direction.

```bash
python vector_intervention/construct_steering_vector.py build \
  --dataset boolq \
  --hidden-root experiment/llama_boolq/hidden_states_boolQ \
  --label-csv experiment/llama_boolq/result_boolQ.csv \
  --gold-path QAdataset/boolQ.csv \
  --output-dir experiment/llama_boolq/vectors \
  --question-limit 400 --fallacy-last-id 5199
```

For MMLU, use `--dataset mmlu`, the corresponding MMLU hidden-state directory and result CSV, and `QAdataset/MMLU.jsonl`. For Gemma, use the outputs from the corresponding Gemma evaluation script. The script produces `robust_vectors_boolQ.safetensors` or `robust_vectors_MMLU.safetensors`, along with group and vector diagnostics. Available vector keys include `robust_raw`, `robust_unit`, `detect_raw`, `detect_unit`, `robust_orth_raw`, and `robust_orth_unit`.

Use `--use-all-samples` only if the control condition is to be included, and `--include-first-incorrect` only if initially incorrect samples are to be included (the groups still use second-answer correctness). These options change the construction sample set. With the default vector construction setting, row 0 represents the embedding output; if `--skip-first-layer` is used, pass `--vector-layer-offset 0` to the intervention script instead of 1.

### 5. Vector intervention

Use the held-out 100-question subset and vectors constructed from the same model and dataset. The following command illustrates BoolQ intervention for a non-Gemma model:

```bash
python vector_intervention/steer_boolq.py \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --questions QAdataset/boolQ_val.csv \
  --persuasion-prompts persuasion_prompts/boolQ/val.txt \
  --output-dir experiment/llama_boolq_steer \
  --vector-path experiment/llama_boolq/vectors/robust_vectors_boolQ.safetensors \
  --vector-key robust_unit --layers 15 --alpha 1.0 \
  --question-limit 100
```

For Gemma 3 on MMLU, for example:

```bash
python vector_intervention/steer_mmlu_gemma.py \
  --model-name google/gemma-3-4b-it \
  --questions QAdataset/MMLU_val.jsonl \
  --persuasion-prompts persuasion_prompts/MMLU/val.txt \
  --output-dir experiment/gemma_mmlu_steer \
  --vector-path experiment/gemma_mmlu/vectors/robust_vectors_MMLU.safetensors \
  --vector-key robust_unit --layers 18 --alpha 1.0 \
  --question-limit 100
```

Use `steer_mmlu.py` for non-Gemma MMLU models and `steer_boolq_gemma.py` for Gemma on BoolQ. The examples are illustrative; use the model, vector key, layer, strength, and token limits corresponding to the reported experiment. The default question offset is 0 for the separate validation files. For a combined 500-question input file, use `--question-offset 400 --question-limit 100` with the matching validation prompts.

When intervention is enabled, all four steering scripts use the following schedule:

| Stage | Intervention |
| --- | --- |
| First judgment | Off |
| Persuasion response | On |
| Second judgment | On |
| Fallacy detection | On |

In this release, the fallacy-detection stage is also intervened upon in the standard BoolQ, standard MMLU, and Gemma MMLU scripts; their supplied original scripts had it disabled. This changes the experimental protocol relative to those versions. Re-run the affected intervention evaluations before claiming that the results match previously reported values.

`--layers` accepts zero-based transformer block indices (e.g., `--layers 13 18 23`) or `all`. The default `--vector-layer-offset 1` maps block i to saved hidden-state index i+1. `--alpha` specifies additive strength; 0.0 is valid for a strength ablation. `--target-positions last` is the default. `--no-steering` disables additive steering for the entire run. Both BoolQ scripts additionally support `--projection-removal`, `--projection-beta`, and `--detect-vector-key`. Their default vector keys and token limits are retained from the supplied implementations (check `--help`). The standard BoolQ steering script defaults to 32 persuasion response tokens, while the other steering scripts default to 64.

The standard and Gemma BoolQ steering scripts save hidden states as `hidden_states_boolQ.safetensors`; the Gemma BoolQ implementation retains its original full-input-token BF16 storage, whereas the standard BoolQ implementation saves the final input token. The MMLU steering scripts save `hidden_states.safetensors`. The vector construction script recognizes the legacy filenames. For subsequent probing of BoolQ steering outputs, the analysis scripts expect `hidden_states.safetensors`; use copies with that filename if needed.

### Verification and limitations

- Python syntax and command-line help checks were performed for all 13 scripts. No real-model inference or steering run was performed in the preparation environment. Validate the revised intervention schedule with the actual models before publication.
- The exact package versions and model revisions used for the paper were not supplied as a lockfile. Record them for exact reproducibility.
- The code archive does not include the QA datasets, persuasion prompts, model weights, hidden states, or a separate behavior-metric aggregation script. Place the scripts alongside the existing dataset and prompt directories.
- Some original Japanese comments and error messages remain; experimental prompt strings were preserved.

## Citation

If you use the data, persuasion prompts, or experiment code in your research, please cite our paper:

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  year    = {2026}
}
```

## License

Please refer to the license information of this repository and the original BoolQ and MMLU datasets when using the released data.
