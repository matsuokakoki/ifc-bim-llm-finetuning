# -*- coding: utf-8 -*-
"""
IFC/BIM向け Gemma 3 1B LoRA 再実験コード

目的:
- QLoRAではなく通常LoRAで学習する
- system/user prompt部分にはlossをかけず、assistant output部分のみ学習する
- 前回、ハイパーパラメータで実行し、最良だった LoRA rank=16 / batch size=4 / 2 epoch の1条件だけを今回実行する
- Gemma評価データと重複する質問をAlpaca学習データ側から除去して再学習する
- Gemma評価データは1,000件のまま変更せずに学習前後を評価する
- ROUGE-1 / ROUGE-L / BLEU、生成回答、学習曲線、LoRA adapterを保存する
- 質問単位で学習データ汚染を検査・除去し、レポートを保存する
- セッション終了前に成果物をZIPへまとめ、画面にも生成例を出力する

実行環境:
- /kaggle/input にGemma 3 1BモデルとIFC/BIMデータセットを追加しておく
- AcceleratorはGPUにする
"""

# ============================================================
# 0. Setup
# ============================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLE_CODE"] = "true"

import re
import gc
import json
import math
import time
import shutil
import random
import inspect
import traceback
import hashlib
import unicodedata
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

try:
    from IPython.display import display, FileLink
except Exception:
    display = print
    FileLink = None

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

# ============================================================
# 1. Basic config
# ============================================================

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_NAME = "ifc-bim-gemma-lora-output-only-r16-bs4-ep2-train_clean_eval_full"
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

MODEL_NAME_HINT = "gemma-3-1b-it"
TRAIN_DATASET_HINT = "ifc-bim-high-quality-alpaca"
GEMMA_SUBSET_HINT = "ifc-bim-gemma3-subset-1k"

WORK_ROOT = "/kaggle/working"
RESULTS_ROOT = f"{WORK_ROOT}/results_output_only_r16_bs4_ep2_train_clean_eval_full"
EXPERIMENT_ROOT = f"{WORK_ROOT}/experiments_output_only_r16_bs4_ep2_train_clean_eval_full"
PACKAGE_ROOT = f"{WORK_ROOT}/final_output_only_r16_bs4_ep2_train_clean_eval_full_package"

os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

USE_FULL_DATA = True
SMOKE_TRAIN_SAMPLES = 1000
SMOKE_VAL_SAMPLES = 200
SMOKE_GEMMA_EVAL_SAMPLES = 200

NUM_TRAIN_EPOCHS = 2
MANUAL_MAX_LENGTH = None
GENERATION_SAMPLES = 50

# データ汚染対策
# Gemma評価データは変更しない。
# Gemma評価質問と一致する行を、Alpaca全体から除去した後でtrain/validationへ分割する。
REMOVE_ALPACA_OVERLAP_WITH_GEMMA = True
GROUP_SPLIT_BY_QUESTION = True
VALIDATION_RATIO = 0.2

# dtype: RTX PRO 6000ではbf16を使う
TORCH_DTYPE = torch.bfloat16
USE_BF16 = True
USE_FP16 = False

# 評価は全件で実行する。時間を短くしたい場合はTrueにしてsample数を絞る。
EVAL_FULL_DATASETS = True

# 単一実験設定: 前回最良だった条件だけを実行する
RUN_HPO = False
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 1e-4
LORA_DROPOUT = 0.05
BEST_LORA_R = 16
BEST_BATCH_SIZE = 4

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

print("WANDB_AVAILABLE:", WANDB_AVAILABLE)
print("PROJECT_NAME:", PROJECT_NAME)
print("TIMESTAMP:", TIMESTAMP)

# ============================================================
# 2. Run config: best setting only
# ============================================================

def make_run_label(lora_r, batch_size, grad_accum, epochs):
    return f"lora_r{lora_r}_bs{batch_size}_acc{grad_accum}_ep{epochs}"


def make_cfg(lora_r, batch_size):
    return {
        "run_label": make_run_label(
            lora_r,
            batch_size,
            GRAD_ACCUM_STEPS,
            NUM_TRAIN_EPOCHS,
        ),
        "lora_r": lora_r,
        "lora_alpha": lora_r * 2,
        "lora_dropout": LORA_DROPOUT,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "target_modules": TARGET_MODULES,
    }


RUN_CONFIGS = [make_cfg(BEST_LORA_R, BEST_BATCH_SIZE)]

print("RUN_HPO:", RUN_HPO)
print("RUN_CONFIGS:")
for cfg in RUN_CONFIGS:
    print(cfg)

# ============================================================
# 3. GPU check
# ============================================================

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("GPUが有効ではありません。Kaggle NotebookのAcceleratorを確認してください。")

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"GPU {i}: {name}")
    print(f"Total memory: {total:.2f} GB")

print("\nNVIDIA-SMI:")
os.system("nvidia-smi")

# ============================================================
# 4. Utilities: paths
# ============================================================

def find_model_dir(root="/kaggle/input", name_hint="gemma-3-1b-it"):
    root = Path(root)
    candidates = []

    for p in root.rglob("config.json"):
        d = p.parent
        try:
            files = [x.name for x in d.iterdir() if x.is_file()]
        except Exception:
            continue

        has_model = any(name.endswith(".safetensors") or name.endswith(".bin") for name in files)
        has_tokenizer = any(
            name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
            for name in files
        )

        if has_model and has_tokenizer:
            score = 0
            path_lower = str(d).lower()
            if name_hint.lower() in path_lower:
                score += 20
            if "gemma" in path_lower:
                score += 10
            candidates.append((score, str(d)))

    if not candidates:
        raise FileNotFoundError("/kaggle/input にGemmaモデルフォルダが見つかりません。")

    candidates = sorted(candidates, reverse=True)
    print("Model candidates:")
    for score, path in candidates[:5]:
        print(score, path)
    return candidates[0][1]


def find_dataset_dir(root="/kaggle/input", name_hint="ifc-bim-high-quality-alpaca"):
    root = Path(root)
    candidates = []

    for p in list(root.rglob("dataset_dict.json")) + list(root.rglob("state.json")):
        d = p.parent
        path_lower = str(d).lower()
        score = 0
        if name_hint.lower() in path_lower:
            score += 20
        if "alpaca" in path_lower:
            score += 5
        if "gemma3" in path_lower:
            score += 5
        if "subset" in path_lower:
            score += 5
        if "ifc" in path_lower or "bim" in path_lower:
            score += 3
        candidates.append((score, str(d)))

    if not candidates:
        raise FileNotFoundError(f"/kaggle/input に {name_hint} のdatasetが見つかりません。")

    candidates = sorted(candidates, reverse=True)
    print(f"Dataset candidates for {name_hint}:")
    for score, path in candidates[:10]:
        print(score, path)
    return candidates[0][1]


def copy_dataset_to_working(input_path, work_path):
    if os.path.exists(work_path):
        shutil.rmtree(work_path)
    raw_readonly = load_from_disk(input_path)
    raw_readonly.save_to_disk(work_path)
    return load_from_disk(work_path)

# ============================================================
# 5. Load local assets
# ============================================================

BASE_MODEL_PATH = find_model_dir(name_hint=MODEL_NAME_HINT)
TRAIN_DATASET_PATH = find_dataset_dir(name_hint=TRAIN_DATASET_HINT)
GEMMA_SUBSET_PATH = find_dataset_dir(name_hint=GEMMA_SUBSET_HINT)

print("\nSelected model path:", BASE_MODEL_PATH)
print("Selected train dataset path:", TRAIN_DATASET_PATH)
print("Selected gemma subset path:", GEMMA_SUBSET_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("Tokenizer loaded.")
print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
print("chat_template exists:", tokenizer.chat_template is not None)

# ============================================================
# 6. Load train dataset
# ============================================================

TRAIN_WORK_PATH = f"{WORK_ROOT}/ifc-bim-high-quality-alpaca-working"
raw_train_ds = copy_dataset_to_working(TRAIN_DATASET_PATH, TRAIN_WORK_PATH)

print("\nRaw train dataset copied to writable path:")
print(TRAIN_WORK_PATH)
print(raw_train_ds)

if "train" in raw_train_ds:
    dataset = raw_train_ds["train"]
else:
    dataset = raw_train_ds

print("\nTrain source rows:", len(dataset))
print("Columns:", dataset.column_names)
print("Sample:")
print(dataset[0])

# この時点ではsplitしない。
# 先にAlpaca全体とGemma評価データを共通形式へ変換し、質問単位で重複除去する。

# ============================================================
# 7. Load gemma3-subset eval dataset
# ============================================================

GEMMA_WORK_PATH = f"{WORK_ROOT}/ifc-bim-gemma3-subset-1k-working"
raw_gemma_ds = copy_dataset_to_working(GEMMA_SUBSET_PATH, GEMMA_WORK_PATH)

print("\nRaw gemma-subset copied to writable path:")
print(GEMMA_WORK_PATH)
print(raw_gemma_ds)

if "train" in raw_gemma_ds and "test" in raw_gemma_ds:
    gemma_eval_raw = concatenate_datasets([raw_gemma_ds["train"], raw_gemma_ds["test"]])
elif "test" in raw_gemma_ds:
    gemma_eval_raw = raw_gemma_ds["test"]
elif "train" in raw_gemma_ds:
    gemma_eval_raw = raw_gemma_ds["train"]
else:
    gemma_eval_raw = raw_gemma_ds

print("\nGemma-subset source rows:", len(gemma_eval_raw))
print("Columns:", gemma_eval_raw.column_names)
print("Sample:")
print(gemma_eval_raw[0])

# ============================================================
# 8. Formatting and training-data decontamination
# ============================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are an IFC (Industry Foundation Classes) and BIM "
    "(Building Information Modeling) expert. Provide accurate, detailed, "
    "and practical answers about IFC schemas, BIM implementation, and AEC workflows."
)

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "system": "system",
}


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()


def alpaca_to_messages(sample):
    instruction = clean_text(sample.get("instruction", ""))
    user_input = clean_text(sample.get("input", ""))
    output = clean_text(sample.get("output", ""))

    if not output:
        return []

    messages = []

    # このデータではinstructionは共通の指示、inputが実際の質問として扱われる。
    # inputが空の場合のみinstructionを質問として使用する。
    if instruction and user_input:
        messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": user_input})
    elif user_input:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": user_input})
    elif instruction:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": instruction})
    else:
        return []

    messages.append({"role": "assistant", "content": output})
    return messages


def conversations_to_messages(sample):
    conversations = sample.get("conversations", None)
    if conversations is None:
        conversations = sample.get("messages", None)
    if conversations is None:
        return []

    messages = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", turn.get("from", ""))
        content = turn.get("content", turn.get("value", ""))
        role = ROLE_MAP.get(clean_text(role).lower(), clean_text(role).lower())
        content = clean_text(content)
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def safe_apply_chat_template(messages, add_generation_prompt):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        # Gemma tokenizerがsystem roleを直接受けられない場合はuser側へ畳み込む
        folded = []
        pending_system = ""
        for m in messages:
            if m["role"] == "system":
                pending_system += m["content"].strip() + "\n\n"
            elif m["role"] == "user":
                content = pending_system + m["content"]
                pending_system = ""
                folded.append({"role": "user", "content": content.strip()})
            else:
                folded.append(m)
        return tokenizer.apply_chat_template(
            folded,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def find_last_assistant_index(messages):
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            return i
    return -1


def messages_to_formatted_record(messages):
    last_assistant_idx = find_last_assistant_index(messages)
    if last_assistant_idx <= 0:
        return {
            "prompt_text": "",
            "text": "",
            "question": "",
            "reference": "",
            "question_len": 0,
            "reference_len": 0,
        }

    prompt_messages = messages[:last_assistant_idx]
    full_messages = messages[:last_assistant_idx + 1]

    prompt_text = safe_apply_chat_template(prompt_messages, add_generation_prompt=True)
    full_text = safe_apply_chat_template(full_messages, add_generation_prompt=False)

    question_parts = [
        m["content"] for m in prompt_messages if m["role"] == "user"
    ]
    question = "\n\n".join(question_parts)
    reference = messages[last_assistant_idx]["content"]

    return {
        "prompt_text": prompt_text,
        "text": full_text,
        "question": question,
        "reference": reference,
        "question_len": len(question),
        "reference_len": len(reference),
    }


def format_alpaca_for_gemma(example):
    return messages_to_formatted_record(alpaca_to_messages(example))


def format_conversations_for_gemma(example):
    return messages_to_formatted_record(conversations_to_messages(example))


def has_text_question_and_reference(x):
    return (
        len(x["text"]) > 0
        and len(x["question"]) > 0
        and len(x["reference"]) > 0
    )


# Unicode・大文字小文字・空白・改行の差を吸収して質問を比較する。
def normalize_for_overlap(text):
    text = unicodedata.normalize("NFKC", clean_text(text)).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stable_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def question_key(question):
    return stable_hash(normalize_for_overlap(question))


def qa_key(question, reference):
    normalized = normalize_for_overlap(question) + "\n<ANSWER>\n" + normalize_for_overlap(reference)
    return stable_hash(normalized)


def dataset_keys(formatted_ds):
    question_keys = [question_key(q) for q in formatted_ds["question"]]
    qa_keys = [
        qa_key(q, a)
        for q, a in zip(formatted_ds["question"], formatted_ds["reference"])
    ]
    return question_keys, qa_keys


def records_from_indices(ds, indices, reason):
    return [
        {
            "source_index": int(i),
            "reason": reason,
            "question": ds[int(i)]["question"],
            "reference": ds[int(i)]["reference"],
        }
        for i in indices
    ]


def split_dataset_by_question_group(formatted_ds, validation_ratio=0.2, seed=42):
    """同一の正規化質問がtrainとvalidationにまたがらないように分割する。"""
    q_keys, _ = dataset_keys(formatted_ds)
    unique_q_keys = list(dict.fromkeys(q_keys))

    rng = random.Random(seed)
    rng.shuffle(unique_q_keys)

    if len(unique_q_keys) < 2:
        raise RuntimeError("Need at least two unique questions for train/validation split.")

    n_val_groups = max(1, int(round(len(unique_q_keys) * validation_ratio)))
    n_val_groups = min(n_val_groups, len(unique_q_keys) - 1)
    val_q_set = set(unique_q_keys[:n_val_groups])

    train_indices = [i for i, key in enumerate(q_keys) if key not in val_q_set]
    val_indices = [i for i, key in enumerate(q_keys) if key in val_q_set]

    return formatted_ds.select(train_indices), formatted_ds.select(val_indices)


print("\nFormatting complete Alpaca source before filtering...")
formatted_alpaca_all = dataset.map(
    format_alpaca_for_gemma,
    remove_columns=dataset.column_names,
    desc="Formatting complete Alpaca data",
    load_from_cache_file=False,
)
formatted_alpaca_all = formatted_alpaca_all.filter(
    has_text_question_and_reference,
    desc="Filtering complete Alpaca data",
)

print("\nFormatting full Gemma evaluation data...")
formatted_gemma_all = gemma_eval_raw.map(
    format_conversations_for_gemma,
    remove_columns=gemma_eval_raw.column_names,
    desc="Formatting full Gemma eval data",
    load_from_cache_file=False,
)
formatted_gemma_all = formatted_gemma_all.filter(
    has_text_question_and_reference,
    desc="Filtering full Gemma eval data",
)

print("Formatted Alpaca rows before removal:", len(formatted_alpaca_all))
print("Formatted Gemma evaluation rows:", len(formatted_gemma_all))

alpaca_q_keys_before, alpaca_qa_keys_before = dataset_keys(formatted_alpaca_all)
gemma_q_keys, gemma_qa_keys = dataset_keys(formatted_gemma_all)

gemma_q_set = set(gemma_q_keys)
gemma_qa_set = set(gemma_qa_keys)

alpaca_question_overlap_before = sum(
    key in gemma_q_set for key in alpaca_q_keys_before
)
alpaca_exact_qa_overlap_before = sum(
    key in gemma_qa_set for key in alpaca_qa_keys_before
)

if REMOVE_ALPACA_OVERLAP_WITH_GEMMA:
    removed_alpaca_indices = [
        i for i, key in enumerate(alpaca_q_keys_before) if key in gemma_q_set
    ]
    kept_alpaca_indices = [
        i for i, key in enumerate(alpaca_q_keys_before) if key not in gemma_q_set
    ]
else:
    removed_alpaca_indices = []
    kept_alpaca_indices = list(range(len(formatted_alpaca_all)))

formatted_alpaca_clean = formatted_alpaca_all.select(kept_alpaca_indices)
removed_training_records = records_from_indices(
    formatted_alpaca_all,
    removed_alpaca_indices,
    "question_overlap_with_full_gemma_eval",
)

# 評価データは一切削除・重複排除しない。
formatted_gemma_eval = formatted_gemma_all
assert len(formatted_gemma_eval) == len(formatted_gemma_all)

# 汚染除去後のAlpacaをtrain/validationへ分割する。
# 同一質問がtrainとvalidationへまたがらないよう、質問グループ単位で分割する。
if GROUP_SPLIT_BY_QUESTION:
    formatted_train, formatted_val = split_dataset_by_question_group(
        formatted_alpaca_clean,
        validation_ratio=VALIDATION_RATIO,
        seed=SEED,
    )
else:
    split_formatted = formatted_alpaca_clean.train_test_split(
        test_size=VALIDATION_RATIO,
        seed=SEED,
        shuffle=True,
    )
    formatted_train = split_formatted["train"]
    formatted_val = split_formatted["test"]

# スモークテストの場合のみ各データを縮小する。
if not USE_FULL_DATA:
    formatted_train = formatted_train.select(
        range(min(SMOKE_TRAIN_SAMPLES, len(formatted_train)))
    )
    formatted_val = formatted_val.select(
        range(min(SMOKE_VAL_SAMPLES, len(formatted_val)))
    )
    formatted_gemma_eval = formatted_gemma_eval.select(
        range(min(SMOKE_GEMMA_EVAL_SAMPLES, len(formatted_gemma_eval)))
    )

# 最終状態を機械的に検証する。
final_train_q_keys, final_train_qa_keys = dataset_keys(formatted_train)
final_val_q_keys, final_val_qa_keys = dataset_keys(formatted_val)
final_gemma_q_keys, final_gemma_qa_keys = dataset_keys(formatted_gemma_eval)

final_train_q_set = set(final_train_q_keys)
final_train_qa_set = set(final_train_qa_keys)
final_val_q_set = set(final_val_q_keys)
final_val_qa_set = set(final_val_qa_keys)
final_gemma_q_set = set(final_gemma_q_keys)
final_gemma_qa_set = set(final_gemma_qa_keys)

train_question_overlap_with_gemma_after = sum(
    key in final_gemma_q_set for key in final_train_q_keys
)
train_exact_qa_overlap_with_gemma_after = sum(
    key in final_gemma_qa_set for key in final_train_qa_keys
)
val_question_overlap_with_gemma_after = sum(
    key in final_gemma_q_set for key in final_val_q_keys
)
val_exact_qa_overlap_with_gemma_after = sum(
    key in final_gemma_qa_set for key in final_val_qa_keys
)
train_val_question_overlap_after = len(final_train_q_set & final_val_q_set)
train_val_exact_qa_overlap_after = len(final_train_qa_set & final_val_qa_set)

gemma_internal_duplicate_question_rows = len(final_gemma_q_keys) - len(final_gemma_q_set)
gemma_internal_duplicate_qa_rows = len(final_gemma_qa_keys) - len(final_gemma_qa_set)

assert train_question_overlap_with_gemma_after == 0, (
    "Training questions still overlap with Gemma evaluation data."
)
assert train_exact_qa_overlap_with_gemma_after == 0, (
    "Training QA pairs still overlap with Gemma evaluation data."
)
assert val_question_overlap_with_gemma_after == 0, (
    "Validation questions still overlap with Gemma evaluation data."
)
assert val_exact_qa_overlap_with_gemma_after == 0, (
    "Validation QA pairs still overlap with Gemma evaluation data."
)
if GROUP_SPLIT_BY_QUESTION:
    assert train_val_question_overlap_after == 0, (
        "A normalized question appears in both train and validation."
    )
    assert train_val_exact_qa_overlap_after == 0, (
        "An exact QA pair appears in both train and validation."
    )

contamination_policy = "remove_overlapping_questions_from_alpaca_before_split_keep_gemma_eval_full"

contamination_report = {
    "timestamp": TIMESTAMP,
    "normalization": "Unicode NFKC + casefold + whitespace normalization",
    "comparison_unit_for_removal": "question-only",
    "policy": contamination_policy,
    "alpaca_rows_before_removal": len(formatted_alpaca_all),
    "alpaca_question_overlap_rows_before": alpaca_question_overlap_before,
    "alpaca_exact_qa_overlap_rows_before": alpaca_exact_qa_overlap_before,
    "alpaca_rows_removed": len(removed_alpaca_indices),
    "alpaca_rows_after_removal": len(formatted_alpaca_clean),
    "train_rows_after_split": len(formatted_train),
    "validation_rows_after_split": len(formatted_val),
    "gemma_eval_rows_before": len(formatted_gemma_all),
    "gemma_eval_rows_removed": 0,
    "gemma_eval_rows_after": len(formatted_gemma_eval),
    "gemma_internal_duplicate_question_rows_kept": gemma_internal_duplicate_question_rows,
    "gemma_internal_duplicate_qa_rows_kept": gemma_internal_duplicate_qa_rows,
    "train_question_overlap_with_gemma_after": train_question_overlap_with_gemma_after,
    "train_exact_qa_overlap_with_gemma_after": train_exact_qa_overlap_with_gemma_after,
    "validation_question_overlap_with_gemma_after": val_question_overlap_with_gemma_after,
    "validation_exact_qa_overlap_with_gemma_after": val_exact_qa_overlap_with_gemma_after,
    "train_validation_question_overlap_after": train_val_question_overlap_after,
    "train_validation_exact_qa_overlap_after": train_val_exact_qa_overlap_after,
}

contamination_report_df = pd.DataFrame([contamination_report])
contamination_report_df.to_csv(
    f"{RESULTS_ROOT}/data_contamination_report.csv",
    index=False,
)
with open(f"{RESULTS_ROOT}/data_contamination_report.json", "w", encoding="utf-8") as f:
    json.dump(contamination_report, f, ensure_ascii=False, indent=2)

pd.DataFrame(removed_training_records).to_csv(
    f"{RESULTS_ROOT}/removed_training_overlap_rows.csv",
    index=False,
)

print("\n================ DATA CONTAMINATION REPORT ================")
display(contamination_report_df.T.rename(columns={0: "value"}))
print("\nAlpaca rows removed:", len(removed_alpaca_indices))
print("Final train rows:", len(formatted_train))
print("Final validation rows:", len(formatted_val))
print("Final Gemma eval rows (unchanged):", len(formatted_gemma_eval))
print("Train question overlap with Gemma after filtering:", train_question_overlap_with_gemma_after)
print("Validation question overlap with Gemma after filtering:", val_question_overlap_with_gemma_after)
print("Train/validation question overlap after grouped split:", train_val_question_overlap_after)

print("\nFormatted train:", formatted_train)
print("Formatted validation:", formatted_val)
print("Formatted gemma eval:", formatted_gemma_eval)
print("\nFormatted sample:")
print(formatted_train[0]["text"][:1500])

# ============================================================
# 9. Token length distribution
# ============================================================

def count_tokens(texts, name="dataset"):
    lengths = []
    for i, text in enumerate(texts):
        ids = tokenizer(text, truncation=False)["input_ids"]
        lengths.append(len(ids))
        if (i + 1) % 5000 == 0:
            print(f"{name}: processed {i+1}/{len(texts)}")
    return lengths


print("\nCounting token lengths...")
all_texts_for_length = list(formatted_train["text"]) + list(formatted_val["text"])
token_lengths = count_tokens(all_texts_for_length, name="train+val")

length_df = pd.DataFrame({"token_length": token_lengths})
length_df.to_csv(f"{RESULTS_ROOT}/token_length_distribution.csv", index=False)

print("\nToken length statistics:")
display(length_df.describe())

coverage_rows = []
for threshold in [256, 512, 768, 1024, 1536, 2048, 4096]:
    ratio = (length_df["token_length"] <= threshold).mean()
    coverage_rows.append({
        "max_length": threshold,
        "coverage_ratio": ratio,
        "coverage_percent": ratio * 100,
        "truncation_percent": (1 - ratio) * 100,
    })
    print(f"<= {threshold}: {ratio*100:.2f}%")

coverage_df = pd.DataFrame(coverage_rows)
coverage_df.to_csv(f"{RESULTS_ROOT}/token_length_coverage.csv", index=False)
display(coverage_df)

if MANUAL_MAX_LENGTH is not None:
    MAX_LENGTH = MANUAL_MAX_LENGTH
else:
    MAX_LENGTH = 1024
    for threshold in [512, 768, 1024, 1536, 2048]:
        ratio = (length_df["token_length"] <= threshold).mean()
        if ratio >= 0.95:
            MAX_LENGTH = threshold
            break

print("\nSelected MAX_LENGTH:", MAX_LENGTH)

# ============================================================
# 10. Tokenize output-only
# ============================================================

def tokenize_output_only(example):
    prompt_ids = tokenizer(
        example["prompt_text"],
        add_special_tokens=False,
    )["input_ids"]

    full_enc = tokenizer(
        example["text"],
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    if len(input_ids) == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    start = min(len(prompt_ids), len(input_ids))
    labels = [-100] * len(input_ids)

    # assistant output部分だけloss対象にする
    for i in range(start, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def has_trainable_label(example):
    return (
        len(example["input_ids"]) > 0
        and len(example["labels"]) == len(example["input_ids"])
        and any(x != -100 for x in example["labels"])
    )


tokenized_train = formatted_train.map(
    tokenize_output_only,
    remove_columns=formatted_train.column_names,
    desc="Tokenizing train output-only",
    load_from_cache_file=False,
)

tokenized_val = formatted_val.map(
    tokenize_output_only,
    remove_columns=formatted_val.column_names,
    desc="Tokenizing validation output-only",
    load_from_cache_file=False,
)

tokenized_gemma_eval = formatted_gemma_eval.map(
    tokenize_output_only,
    remove_columns=formatted_gemma_eval.column_names,
    desc="Tokenizing gemma-subset output-only",
    load_from_cache_file=False,
)

tokenized_train = tokenized_train.filter(has_trainable_label, desc="Filtering tokenized train")
tokenized_val = tokenized_val.filter(has_trainable_label, desc="Filtering tokenized validation")
tokenized_gemma_eval = tokenized_gemma_eval.filter(has_trainable_label, desc="Filtering tokenized gemma eval")

print("\nTokenized train:", tokenized_train)
print("Tokenized validation:", tokenized_val)
print("Tokenized gemma eval:", tokenized_gemma_eval)

# ============================================================
# 11. Dataset summary and generation samples
# ============================================================

dataset_summary = {
    "timestamp": TIMESTAMP,
    "model_path": BASE_MODEL_PATH,
    "train_dataset_path": TRAIN_DATASET_PATH,
    "gemma_subset_path": GEMMA_SUBSET_PATH,
    "use_full_data": USE_FULL_DATA,
    "train_rows": len(formatted_train),
    "validation_rows": len(formatted_val),
    "gemma_eval_rows": len(formatted_gemma_eval),
    "decontamination_policy": contamination_policy,
    "alpaca_overlap_rows_removed": len(removed_alpaca_indices),
    "gemma_eval_rows_removed": 0,
    "train_question_overlap_with_gemma_after": train_question_overlap_with_gemma_after,
    "validation_question_overlap_with_gemma_after": val_question_overlap_with_gemma_after,
    "max_length": MAX_LENGTH,
    "train_question_len_mean": pd.Series(formatted_train["question_len"]).mean(),
    "train_reference_len_mean": pd.Series(formatted_train["reference_len"]).mean(),
    "val_question_len_mean": pd.Series(formatted_val["question_len"]).mean(),
    "val_reference_len_mean": pd.Series(formatted_val["reference_len"]).mean(),
}

dataset_summary_df = pd.DataFrame([dataset_summary])
dataset_summary_df.to_csv(f"{RESULTS_ROOT}/dataset_summary.csv", index=False)
display(dataset_summary_df)


def build_generation_df(formatted_ds, n_samples=30, seed=42, source_name="validation"):
    df = pd.DataFrame({
        "question": formatted_ds["question"],
        "reference": formatted_ds["reference"],
        "question_len": formatted_ds["question_len"],
        "reference_len": formatted_ds["reference_len"],
    })

    candidate_df = df[
        (df["question_len"] <= 2500)
        & (df["reference_len"] <= 2500)
        & (df["reference_len"] >= 30)
    ].copy()

    n_gen = min(n_samples, len(candidate_df))
    sampled = candidate_df.sample(n=n_gen, random_state=seed).reset_index(drop=True)
    sampled.insert(0, "id", range(len(sampled)))
    sampled.insert(1, "source", source_name)
    return sampled


generation_gemma_df = build_generation_df(
    formatted_gemma_eval,
    n_samples=GENERATION_SAMPLES,
    seed=SEED,
    source_name="gemma_subset",
)
generation_gemma_df.to_csv(f"{RESULTS_ROOT}/generation_questions_gemma_subset.csv", index=False)

print("\nGemma-subset generation samples:", len(generation_gemma_df))
display(generation_gemma_df.head())

# ============================================================
# 12. Data collator
# ============================================================

def output_only_collator(features):
    input_features = [
        {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
        for f in features
    ]

    batch = tokenizer.pad(
        input_features,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    max_len = batch["input_ids"].shape[1]
    labels = []
    for f in features:
        label = list(f["labels"])
        pad_len = max_len - len(label)
        labels.append(label + [-100] * pad_len)

    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return batch

# ============================================================
# 13. Evaluation metrics and generation helpers
# ============================================================

def get_model_device(model):
    return next(model.parameters()).device


def print_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"{prefix}GPU allocated: {allocated:.2f} GB")
        print(f"{prefix}GPU reserved:  {reserved:.2f} GB")
        print(f"{prefix}GPU max allocated: {max_allocated:.2f} GB")
        print(f"{prefix}GPU total:     {total:.2f} GB")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def generate_answer(model, question, max_new_tokens=300):
    model.eval()
    device = get_model_device(model)

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt = safe_apply_chat_template(messages, add_generation_prompt=True)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def normalize_for_metric(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    return text.strip().split()


def rouge1_f1(pred, ref):
    pred_tokens = normalize_for_metric(pred)
    ref_tokens = normalize_for_metric(ref)
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_length(a, b):
    max_len = 700
    a = a[:max_len]
    b = b[:max_len]
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            temp = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l_f1(pred, ref):
    pred_tokens = normalize_for_metric(pred)
    ref_tokens = normalize_for_metric(ref)
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def sentence_bleu_simple(pred, ref, max_n=4):
    pred_tokens = normalize_for_metric(pred)
    ref_tokens = normalize_for_metric(ref)
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        if sum(pred_ngrams.values()) == 0:
            precisions.append(1e-9)
            continue
        overlap = sum((pred_ngrams & ref_ngrams).values())
        precision = overlap / sum(pred_ngrams.values())
        precisions.append(max(precision, 1e-9))

    log_precision = sum(math.log(p) for p in precisions) / max_n
    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / max(pred_len, 1))
    return bp * math.exp(log_precision)


def compute_metrics_for_rows(df, answer_col):
    rouge1_scores = []
    rougeL_scores = []
    bleu_scores = []
    for _, row in df.iterrows():
        pred = row[answer_col]
        ref = row["reference"]
        rouge1_scores.append(rouge1_f1(pred, ref))
        rougeL_scores.append(rouge_l_f1(pred, ref))
        bleu_scores.append(sentence_bleu_simple(pred, ref))
    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
        "bleu": sum(bleu_scores) / len(bleu_scores),
    }

# ============================================================
# 14. Resource logging callback
# ============================================================

class ResourceLoggingCallback(TrainerCallback):
    def __init__(self, record_every_n_steps=20):
        self.record_every_n_steps = record_every_n_steps
        self.step_start_time = None
        self.last_log_time = time.time()
        self.step_times_since_log = []
        self.records = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start_time is None:
            return

        now = time.time()
        step_time = now - self.step_start_time
        self.step_times_since_log.append(step_time)

        step = int(state.global_step)
        if step <= 0:
            return

        if step % self.record_every_n_steps == 0:
            row = {
                "step": step,
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "step_time_sec": step_time,
            }
            if torch.cuda.is_available():
                row.update({
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_memory_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                })
            self.records.append(row)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        now = time.time()
        logs["time_since_last_log_sec"] = now - self.last_log_time
        self.last_log_time = now

        if self.step_times_since_log:
            logs["avg_step_time_sec_since_last_log"] = sum(self.step_times_since_log) / len(self.step_times_since_log)
            logs["max_step_time_sec_since_last_log"] = max(self.step_times_since_log)
            self.step_times_since_log = []

        if torch.cuda.is_available():
            logs["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            logs["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            logs["gpu_memory_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3

    def save_csv(self, path):
        pd.DataFrame(self.records).to_csv(path, index=False)

# ============================================================
# 15. Model loading / trainer helpers
# ============================================================

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=TORCH_DTYPE,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False
    return model


def get_found_target_modules(model, target_modules):
    found = sorted({
        name.split(".")[-1]
        for name, module in model.named_modules()
        if name.split(".")[-1] in target_modules
    })
    if len(found) == 0:
        raise RuntimeError("LoRA target modules were not found. Check model architecture.")
    return found


def create_eval_trainer(model, output_dir, batch_size=8):
    args_kwargs = dict(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        bf16=USE_BF16,
        fp16=USE_FP16,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
    )

    params = inspect.signature(TrainingArguments.__init__).parameters
    filtered = {k: v for k, v in args_kwargs.items() if k in params}
    args = TrainingArguments(**filtered)

    return Trainer(
        model=model,
        args=args,
        data_collator=output_only_collator,
        processing_class=tokenizer,
    )


def evaluate_loss_full(model, dataset, output_dir, batch_size=8, metric_key_prefix="eval"):
    eval_trainer = create_eval_trainer(model, output_dir, batch_size=batch_size)
    metrics = eval_trainer.evaluate(
        eval_dataset=dataset,
        metric_key_prefix=metric_key_prefix,
    )
    del eval_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def make_training_args(cfg, run_dir, run_name, report_to):
    effective_batch_size = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(tokenized_train) / effective_batch_size)
    estimated_total_steps = steps_per_epoch * cfg["num_train_epochs"]
    warmup_steps = max(1, int(estimated_total_steps * 0.03))

    args_kwargs = dict(
        output_dir=run_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=2,
        bf16=USE_BF16,
        fp16=USE_FP16,
        optim="adamw_torch",
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=run_name if report_to == "wandb" else None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        args_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in params:
        args_kwargs["evaluation_strategy"] = "steps"

    if "group_by_length" in params:
        args_kwargs["group_by_length"] = True

    filtered = {k: v for k, v in args_kwargs.items() if k in params}
    removed = sorted(set(args_kwargs) - set(filtered))
    print("Removed unsupported TrainingArguments:", removed)
    return TrainingArguments(**filtered)

# ============================================================
# 16. Base evaluation once
# ============================================================

print("\n================ BASE EVALUATION START ================")
clear_memory()
base_model = load_base_model()
print("Base model loaded.")
print_gpu_memory("[base] ")

base_val_metrics = evaluate_loss_full(
    base_model,
    tokenized_val,
    output_dir=f"{RESULTS_ROOT}/base_eval_val",
    batch_size=8,
    metric_key_prefix="base_val",
)

base_gemma_metrics = evaluate_loss_full(
    base_model,
    tokenized_gemma_eval,
    output_dir=f"{RESULTS_ROOT}/base_eval_gemma",
    batch_size=8,
    metric_key_prefix="base_gemma",
)

base_answers = []
for _, row in generation_gemma_df.iterrows():
    ans = generate_answer(base_model, row["question"], max_new_tokens=300)
    base_answers.append(ans)

generation_gemma_base_df = generation_gemma_df.copy()
generation_gemma_base_df["base_answer"] = base_answers
base_metrics = compute_metrics_for_rows(generation_gemma_base_df, "base_answer")
generation_gemma_base_df.to_csv(f"{RESULTS_ROOT}/base_generation_gemma_subset.csv", index=False)

base_summary = {
    "base_val_output_only_loss": base_val_metrics.get("base_val_loss"),
    "base_gemma_output_only_loss": base_gemma_metrics.get("base_gemma_loss"),
    "base_rouge1": base_metrics["rouge1"],
    "base_rougeL": base_metrics["rougeL"],
    "base_bleu": base_metrics["bleu"],
}

print("Base summary:")
print(base_summary)

pd.DataFrame([base_summary]).to_csv(f"{RESULTS_ROOT}/base_summary.csv", index=False)

del base_model
gc.collect()
torch.cuda.empty_cache()
print("Base model cleared.")
print_gpu_memory("[after base clear] ")

# ============================================================
# 17. Training / evaluation function
# ============================================================

def make_failure_summary(cfg, status, error_message, traceback_text):
    return {
        "run_label": cfg["run_label"],
        "status": status,
        "error_message": error_message,
        "traceback": traceback_text,
        "run_name": f"{cfg['run_label']}-{TIMESTAMP}",
        "model_path": BASE_MODEL_PATH,
        "train_dataset_path": TRAIN_DATASET_PATH,
        "gemma_subset_path": GEMMA_SUBSET_PATH,
        "use_full_data": USE_FULL_DATA,
        "train_rows": len(tokenized_train),
        "validation_rows": len(tokenized_val),
        "gemma_eval_rows": len(tokenized_gemma_eval),
        "decontamination_policy": contamination_policy,
        "alpaca_overlap_rows_removed": len(removed_alpaca_indices),
        "gemma_eval_rows_removed": 0,
        "train_question_overlap_with_gemma_after": train_question_overlap_with_gemma_after,
        "num_train_epochs": cfg["num_train_epochs"],
        "max_length": MAX_LENGTH,
        "learning_rate": cfg["learning_rate"],
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "effective_batch_size": cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
    }


def run_one_experiment(cfg):
    run_label = cfg["run_label"]
    run_name = f"{run_label}-{TIMESTAMP}"
    run_dir = f"{EXPERIMENT_ROOT}/{run_label}"
    adapter_dir = f"{run_dir}/adapter"
    result_dir = f"{run_dir}/results"

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    wandb_run = None
    trainer = None
    model = None

    print("\n" + "=" * 90)
    print("RUN START:", run_label)
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    print("=" * 90)

    try:
        clear_memory()
        torch.cuda.reset_peak_memory_stats()

        if WANDB_AVAILABLE:
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_PROJECT"] = PROJECT_NAME
            wandb_run = wandb.init(
                project=PROJECT_NAME,
                name=run_name,
                config={
                    **cfg,
                    "model_path": BASE_MODEL_PATH,
                    "train_dataset_path": TRAIN_DATASET_PATH,
                    "gemma_subset_path": GEMMA_SUBSET_PATH,
                    "use_full_data": USE_FULL_DATA,
                    "max_length": MAX_LENGTH,
                    "train_rows": len(tokenized_train),
                    "validation_rows": len(tokenized_val),
                    "gemma_eval_rows": len(tokenized_gemma_eval),
                    "output_only_training": True,
                    "qlora": False,
                    "precision": str(TORCH_DTYPE),
                },
                settings=wandb.Settings(save_code=False),
            )
            report_to = "wandb"
        else:
            report_to = "none"

        model = load_base_model()
        print("Training base model loaded.")
        print_gpu_memory(f"[{run_label} load] ")

        found_modules = get_found_target_modules(model, cfg["target_modules"])
        print("Found LoRA target modules:", found_modules)

        peft_config = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=found_modules,
        )

        model = get_peft_model(model, peft_config)
        print("LoRA applied.")
        model.print_trainable_parameters()

        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        training_args = make_training_args(cfg, run_dir, run_name, report_to)
        resource_callback = ResourceLoggingCallback(record_every_n_steps=20)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=output_only_collator,
            processing_class=tokenizer,
            callbacks=[resource_callback],
        )

        print("Trainer ready.")
        print_gpu_memory(f"[{run_label} trainer ready] ")

        train_result = trainer.train(resume_from_checkpoint=False)
        print("Train result:")
        print(train_result)

        log_history_df = pd.DataFrame(trainer.state.log_history)
        log_history_path = f"{result_dir}/trainer_log_history.csv"
        log_history_df.to_csv(log_history_path, index=False)
        resource_callback.save_csv(f"{result_dir}/resource_step_log.csv")

        after_val_metrics = trainer.evaluate(eval_dataset=tokenized_val, metric_key_prefix="after_val")
        print("After validation eval:")
        print(after_val_metrics)

        after_gemma_metrics = trainer.evaluate(eval_dataset=tokenized_gemma_eval, metric_key_prefix="after_gemma")
        print("After gemma-subset eval:")
        print(after_gemma_metrics)

        generation_df = generation_gemma_base_df.copy()
        after_answers = []
        for _, row in generation_df.iterrows():
            ans = generate_answer(trainer.model, row["question"], max_new_tokens=300)
            after_answers.append(ans)

        generation_df["after_answer"] = after_answers
        after_metrics = compute_metrics_for_rows(generation_df, "after_answer")

        comparison_csv_path = f"{result_dir}/before_after_generation_gemma_subset.csv"
        generation_df.to_csv(comparison_csv_path, index=False)

        # セッション終了時にも内容を確認できるよう、先頭10件を画面へ全文表示する
        pd.set_option("display.max_colwidth", None)
        print("\n================ GENERATION EXAMPLES ================")
        for _, sample_row in generation_df.head(10).iterrows():
            print("\n" + "=" * 100)
            print(f"Sample {sample_row['id']}")
            print("\n[Question]")
            print(sample_row["question"])
            print("\n[Reference]")
            print(sample_row["reference"])
            print("\n[Base model]")
            print(sample_row["base_answer"])
            print("\n[Fine-tuned model]")
            print(sample_row["after_answer"])

        # NotionやブログへコピーしやすいMarkdownも保存する
        examples_md_path = f"{result_dir}/generation_examples_top10.md"
        with open(examples_md_path, "w", encoding="utf-8") as examples_file:
            examples_file.write("# Generation Examples: Base vs Fine-tuned\n\n")
            for _, sample_row in generation_df.head(10).iterrows():
                examples_file.write(f"## Sample {sample_row['id']}\n\n")
                examples_file.write("### Question\n\n")
                examples_file.write(str(sample_row["question"]) + "\n\n")
                examples_file.write("### Reference\n\n")
                examples_file.write(str(sample_row["reference"]) + "\n\n")
                examples_file.write("### Base model\n\n")
                examples_file.write(str(sample_row["base_answer"]) + "\n\n")
                examples_file.write("### Fine-tuned model\n\n")
                examples_file.write(str(sample_row["after_answer"]) + "\n\n")

        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        max_gpu_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        max_gpu_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None

        train_metrics = train_result.metrics if hasattr(train_result, "metrics") else {}

        summary = {
            "status": "ok",
            "run_label": run_label,
            "run_name": run_name,
            "model_path": BASE_MODEL_PATH,
            "train_dataset_path": TRAIN_DATASET_PATH,
            "gemma_subset_path": GEMMA_SUBSET_PATH,
            "use_full_data": USE_FULL_DATA,
            "train_rows": len(tokenized_train),
            "validation_rows": len(tokenized_val),
            "gemma_eval_rows": len(tokenized_gemma_eval),
            "num_train_epochs": cfg["num_train_epochs"],
            "max_length": MAX_LENGTH,
            "learning_rate": cfg["learning_rate"],
            "per_device_train_batch_size": cfg["per_device_train_batch_size"],
            "per_device_eval_batch_size": cfg["per_device_eval_batch_size"],
            "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
            "effective_batch_size": cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
            "lora_r": cfg["lora_r"],
            "lora_alpha": cfg["lora_alpha"],
            "lora_dropout": cfg["lora_dropout"],
            "lora_target_modules": ",".join(found_modules),
            "trainable_params_note": "see run log and adapter config",
            "base_val_output_only_loss": base_summary["base_val_output_only_loss"],
            "after_val_output_only_loss": after_val_metrics.get("after_val_loss"),
            "base_gemma_output_only_loss": base_summary["base_gemma_output_only_loss"],
            "after_gemma_output_only_loss": after_gemma_metrics.get("after_gemma_loss"),
            "base_rouge1": base_summary["base_rouge1"],
            "after_rouge1": after_metrics["rouge1"],
            "base_rougeL": base_summary["base_rougeL"],
            "after_rougeL": after_metrics["rougeL"],
            "base_bleu": base_summary["base_bleu"],
            "after_bleu": after_metrics["bleu"],
            "train_runtime": train_metrics.get("train_runtime"),
            "train_samples_per_second": train_metrics.get("train_samples_per_second"),
            "train_steps_per_second": train_metrics.get("train_steps_per_second"),
            "train_loss": train_metrics.get("train_loss"),
            "max_gpu_allocated_gb": max_gpu_allocated_gb,
            "max_gpu_reserved_gb": max_gpu_reserved_gb,
            "adapter_dir": adapter_dir,
            "result_dir": result_dir,
            "trainer_log_history_path": log_history_path,
            "resource_step_log_path": f"{result_dir}/resource_step_log.csv",
        }

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f"{result_dir}/run_summary.csv", index=False)

        JUDGE_TEMPLATE = """You are an expert evaluator for IFC/BIM question answering.

Evaluate the model answer based on the question and the reference answer.

Score from 1 to 5:
1: The answer is irrelevant or clearly wrong.
2: The answer is partially related but mostly incomplete or inaccurate.
3: The answer is generally relevant but shallow or vague.
4: The answer is accurate, specific, and uses appropriate IFC/BIM terminology.
5: The answer is highly accurate, detailed, and comprehensive.

Focus on whether the model answer properly answers the question.

Question:
{question}

Reference Answer:
{reference}

Model Answer:
{model_answer}

Return JSON only:
{{
  "score": 1,
  "reason": "brief reason"
}}
"""

        judge_rows = []
        for _, row in generation_df.iterrows():
            for target, answer_col in [("base", "base_answer"), ("after", "after_answer")]:
                judge_rows.append({
                    "id": row["id"],
                    "target": target,
                    "question": row["question"],
                    "reference": row["reference"],
                    "model_answer": row[answer_col],
                    "judge_prompt": JUDGE_TEMPLATE.format(
                        question=row["question"],
                        reference=row["reference"],
                        model_answer=row[answer_col],
                    ),
                })

        judge_df = pd.DataFrame(judge_rows)
        judge_df.to_csv(f"{result_dir}/llm_judge_prompts.csv", index=False)

        report_path = f"{result_dir}/run_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# IFC/BIM Gemma LoRA Output-only Report: {run_label}\n\n")
            f.write("## Setup\n\n")
            f.write(f"- Base model path: `{BASE_MODEL_PATH}`\n")
            f.write(f"- Train dataset: `{TRAIN_DATASET_PATH}`\n")
            f.write(f"- Eval dataset: `{GEMMA_SUBSET_PATH}`\n")
            f.write(f"- Decontamination policy: `{contamination_policy}`\n")
            f.write(f"- Removed Alpaca rows due to Gemma-question overlap: `{len(removed_alpaca_indices)}`\n")
            f.write("- Removed Gemma evaluation rows: `0`\n")
            f.write(f"- Gemma evaluation rows used: `{len(formatted_gemma_eval)}`\n")
            f.write(f"- Remaining train-question overlap with Gemma: `{train_question_overlap_with_gemma_after}`\n")
            f.write("- Output-only training: `True`\n")
            f.write("- QLoRA: `False`\n")
            f.write(f"- Precision: `{str(TORCH_DTYPE)}`\n")
            f.write(f"- Epochs: `{cfg['num_train_epochs']}`\n")
            f.write(f"- Max length: `{MAX_LENGTH}`\n")
            f.write(f"- LoRA rank: `{cfg['lora_r']}`\n")
            f.write(f"- LoRA alpha: `{cfg['lora_alpha']}`\n")
            f.write(f"- Batch size: `{cfg['per_device_train_batch_size']}`\n")
            f.write(f"- Gradient accumulation: `{cfg['gradient_accumulation_steps']}`\n")
            f.write(f"- Effective batch size: `{cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']}`\n\n")

            f.write("## Results\n\n")
            f.write("| Metric | Base | After |\n")
            f.write("|---|---:|---:|\n")
            f.write(f"| Validation output-only loss | {base_summary['base_val_output_only_loss']} | {summary['after_val_output_only_loss']} |\n")
            f.write(f"| Gemma-subset output-only loss | {base_summary['base_gemma_output_only_loss']} | {summary['after_gemma_output_only_loss']} |\n")
            f.write(f"| ROUGE-1 | {base_summary['base_rouge1']} | {after_metrics['rouge1']} |\n")
            f.write(f"| ROUGE-L | {base_summary['base_rougeL']} | {after_metrics['rougeL']} |\n")
            f.write(f"| BLEU | {base_summary['base_bleu']} | {after_metrics['bleu']} |\n\n")

            f.write("## Resource\n\n")
            f.write(f"- Max GPU allocated GB: `{max_gpu_allocated_gb}`\n")
            f.write(f"- Max GPU reserved GB: `{max_gpu_reserved_gb}`\n")
            f.write(f"- Train runtime sec: `{summary['train_runtime']}`\n\n")

            f.write("## Notes\n\n")
            f.write("ROUGE/BLEU are surface-form similarity metrics and are reported together with the generated examples.\n")

        if WANDB_AVAILABLE and wandb_run is not None:
            try:
                wandb.log(summary)
            except Exception as e:
                print("wandb.log failed:", e)

        print("RUN SUMMARY:")
        display(summary_df)
        return summary

    except torch.cuda.OutOfMemoryError as e:
        tb = traceback.format_exc()
        print("CUDA OOM in run:", run_label)
        print(tb)
        return make_failure_summary(cfg, "oom", str(e), tb)

    except RuntimeError as e:
        tb = traceback.format_exc()
        if "out of memory" in str(e).lower():
            print("Runtime OOM in run:", run_label)
            print(tb)
            return make_failure_summary(cfg, "oom", str(e), tb)
        print("Runtime error in run:", run_label)
        print(tb)
        return make_failure_summary(cfg, "failed", str(e), tb)

    except Exception as e:
        tb = traceback.format_exc()
        print("Error in run:", run_label)
        print(tb)
        return make_failure_summary(cfg, "failed", str(e), tb)

    finally:
        if WANDB_AVAILABLE:
            try:
                wandb.finish()
            except Exception:
                pass
        try:
            del trainer
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_gpu_memory(f"[{run_label} after cleanup] ")

# ============================================================
# 18. Run experiments
# ============================================================

all_summaries = []
for cfg in RUN_CONFIGS:
    summary = run_one_experiment(cfg)
    all_summaries.append(summary)

all_summary_df = pd.DataFrame(all_summaries)
all_summary_path = f"{RESULTS_ROOT}/all_runs_summary.csv"
all_summary_df.to_csv(all_summary_path, index=False)

print("\nAll runs summary:")
display(all_summary_df)

ok_df = all_summary_df[all_summary_df.get("status", "") == "ok"].copy()
if len(ok_df) > 0:
    sort_cols = ["after_val_output_only_loss"]
    if "after_gemma_output_only_loss" in ok_df.columns:
        sort_cols.append("after_gemma_output_only_loss")
    best_df = ok_df.sort_values(sort_cols, ascending=True).head(1).copy()
else:
    best_df = all_summary_df.head(1).copy()

best_path = f"{RESULTS_ROOT}/best_run_summary.csv"
best_df.to_csv(best_path, index=False)
print("\nBest run:")
display(best_df)

# ============================================================
# 19. Plot curves for each run
# ============================================================

def save_plot_from_log(log_df, x_col, y_col, title, ylabel, out_path):
    if x_col not in log_df.columns or y_col not in log_df.columns:
        return
    plot_df = log_df.dropna(subset=[x_col, y_col])
    if len(plot_df) == 0:
        return
    plt.figure()
    plt.plot(plot_df[x_col], plot_df[y_col], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


for _, row in all_summary_df.iterrows():
    if row.get("status") != "ok":
        continue
    run_label = row["run_label"]
    result_dir = row["result_dir"]
    log_path = f"{result_dir}/trainer_log_history.csv"
    resource_log_path = f"{result_dir}/resource_step_log.csv"

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        save_plot_from_log(
            log_df,
            "step",
            "loss",
            f"Training Loss over Steps ({run_label})",
            "Training Loss",
            f"{result_dir}/training_loss_curve.png",
        )
        save_plot_from_log(
            log_df,
            "step",
            "eval_loss",
            f"Validation Loss over Steps ({run_label})",
            "Validation Loss",
            f"{result_dir}/validation_loss_curve.png",
        )
        save_plot_from_log(
            log_df,
            "step",
            "gpu_memory_max_allocated_gb",
            f"GPU Max Allocated over Log Steps ({run_label})",
            "Max GPU Memory Allocated (GB)",
            f"{result_dir}/gpu_memory_log_curve.png",
        )
        save_plot_from_log(
            log_df,
            "step",
            "avg_step_time_sec_since_last_log",
            f"Average Step Time over Log Steps ({run_label})",
            "Average Step Time (sec)",
            f"{result_dir}/step_time_log_curve.png",
        )

    if os.path.exists(resource_log_path):
        resource_df = pd.read_csv(resource_log_path)
        save_plot_from_log(
            resource_df,
            "step",
            "step_time_sec",
            f"Step Time over Steps ({run_label})",
            "Step Time (sec)",
            f"{result_dir}/step_time_curve.png",
        )
        save_plot_from_log(
            resource_df,
            "step",
            "gpu_memory_max_allocated_gb",
            f"GPU Memory over Steps ({run_label})",
            "Max GPU Memory Allocated (GB)",
            f"{result_dir}/gpu_memory_curve.png",
        )

# ============================================================
# 20. Final aggregated report
# ============================================================

final_report_path = f"{RESULTS_ROOT}/final_output_only_r16_bs4_ep2_train_clean_eval_full_report.md"
with open(final_report_path, "w", encoding="utf-8") as f:
    f.write("# IFC/BIM Gemma 3 1B LoRA Output-only Fine-tuning Report\n\n")

    f.write("## Overview\n\n")
    f.write("This experiment fine-tunes Gemma 3 1B on IFC/BIM data using normal LoRA, not QLoRA. The training loss is applied only to the assistant output part. Questions appearing in the full Gemma evaluation set are removed from the Alpaca source before training, while the Gemma evaluation set itself is kept unchanged.\n\n")

    f.write("## Key Setup\n\n")
    f.write(f"- Base model: `{BASE_MODEL_PATH}`\n")
    f.write(f"- Train dataset: `{TRAIN_DATASET_PATH}`\n")
    f.write(f"- Eval dataset: `{GEMMA_SUBSET_PATH}`\n")
    f.write(f"- Decontamination policy: `{contamination_policy}`\n")
    f.write(f"- Alpaca rows before removal: `{len(formatted_alpaca_all)}`\n")
    f.write(f"- Alpaca question-overlap rows removed: `{len(removed_alpaca_indices)}`\n")
    f.write(f"- Alpaca rows after removal: `{len(formatted_alpaca_clean)}`\n")
    f.write(f"- Gemma evaluation rows removed: `0`\n")
    f.write(f"- Gemma evaluation rows used: `{len(formatted_gemma_eval)}`\n")
    f.write(f"- Remaining train-question overlap with Gemma: `{train_question_overlap_with_gemma_after}`\n")
    f.write(f"- Max length: `{MAX_LENGTH}`\n")
    f.write("- Output-only training: `True`\n")
    f.write("- QLoRA: `False`\n")
    f.write(f"- Epochs: `{NUM_TRAIN_EPOCHS}`\n")
    f.write(f"- Gradient accumulation: `{GRAD_ACCUM_STEPS}`\n")
    f.write(f"- Number of runs: `{len(all_summary_df)}`\n")
    f.write("- W&B mode: `offline`\n\n")

    f.write("## All Runs\n\n")
    f.write(all_summary_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Best Run\n\n")
    f.write(best_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Notes\n\n")
    f.write("Validation loss, separate-dataset loss, ROUGE/BLEU, and generated examples are recorded for the selected configuration.\n")

print("Final report saved:", final_report_path)

# Copy-ready compact markdown table
compact_cols = [
    "status",
    "run_label",
    "lora_r",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "num_train_epochs",
    "after_val_output_only_loss",
    "after_gemma_output_only_loss",
    "after_rouge1",
    "after_rougeL",
    "after_bleu",
    "train_runtime",
    "train_loss",
    "max_gpu_allocated_gb",
    "max_gpu_reserved_gb",
]
compact_cols = [c for c in compact_cols if c in all_summary_df.columns]
compact_df = all_summary_df[compact_cols].copy()
for c in compact_df.select_dtypes(include="number").columns:
    compact_df[c] = compact_df[c].round(4)

copy_ready_path = f"{RESULTS_ROOT}/copy_ready_summary.md"
with open(copy_ready_path, "w", encoding="utf-8") as f:
    f.write("# Copy-ready Summary\n\n")
    f.write(compact_df.to_markdown(index=False))
    f.write("\n\n")
    f.write("## Best Run\n\n")
    f.write(best_df.to_markdown(index=False))
    f.write("\n")

print("Copy-ready summary saved:", copy_ready_path)
print("\nCopy-ready compact table:")
print(compact_df.to_markdown(index=False))

# ============================================================
# 21. Zip outputs
# ============================================================

if os.path.exists(PACKAGE_ROOT):
    shutil.rmtree(PACKAGE_ROOT)
os.makedirs(PACKAGE_ROOT, exist_ok=True)

# Copy root results
for src in [
    f"{RESULTS_ROOT}/dataset_summary.csv",
    f"{RESULTS_ROOT}/data_contamination_report.csv",
    f"{RESULTS_ROOT}/data_contamination_report.json",
    f"{RESULTS_ROOT}/removed_training_overlap_rows.csv",
    f"{RESULTS_ROOT}/token_length_distribution.csv",
    f"{RESULTS_ROOT}/token_length_coverage.csv",
    f"{RESULTS_ROOT}/generation_questions_gemma_subset.csv",
    f"{RESULTS_ROOT}/base_generation_gemma_subset.csv",
    f"{RESULTS_ROOT}/base_summary.csv",
    f"{RESULTS_ROOT}/all_runs_summary.csv",
    f"{RESULTS_ROOT}/best_run_summary.csv",
    f"{RESULTS_ROOT}/final_output_only_r16_bs4_ep2_train_clean_eval_full_report.md",
    f"{RESULTS_ROOT}/copy_ready_summary.md",
]:
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(PACKAGE_ROOT, os.path.basename(src)))
        print("copied:", src)
    else:
        print("missing:", src)

# Copy experiment directories
exp_dst = os.path.join(PACKAGE_ROOT, "experiments_output_only_r16_bs4_ep2_train_clean_eval_full")
if os.path.exists(exp_dst):
    shutil.rmtree(exp_dst)
shutil.copytree(EXPERIMENT_ROOT, exp_dst)
print("copied experiments:", EXPERIMENT_ROOT)

# Copy wandb logs if exists
wandb_src = f"{WORK_ROOT}/wandb"
if os.path.exists(wandb_src):
    wandb_dst = os.path.join(PACKAGE_ROOT, "wandb")
    if os.path.exists(wandb_dst):
        shutil.rmtree(wandb_dst)
    shutil.copytree(wandb_src, wandb_dst)
    print("copied wandb logs:", wandb_src)

zip_base = f"{WORK_ROOT}/ifc_bim_gemma_lora_output_only_r16_bs4_ep2_train_clean_eval_full_complete_outputs"
zip_file = shutil.make_archive(zip_base, "zip", PACKAGE_ROOT)
print("created:", zip_file)
print("size MB:", os.path.getsize(zip_file) / 1024 / 1024)

if FileLink is not None:
    os.chdir(WORK_ROOT)
    display(FileLink(os.path.basename(zip_file)))

print("\nAll done.")
print("Package:", zip_file)
print("ZIP exists:", os.path.exists(zip_file))
print("ZIP size MB:", os.path.getsize(zip_file) / 1024**2 if os.path.exists(zip_file) else None)
print("\nImportant outputs:")
print("- LoRA adapter:", f"{EXPERIMENT_ROOT}/{RUN_CONFIGS[0]['run_label']}/adapter")
print("- Before/after CSV:", f"{EXPERIMENT_ROOT}/{RUN_CONFIGS[0]['run_label']}/results/before_after_generation_gemma_subset.csv")
print("- Generation examples:", f"{EXPERIMENT_ROOT}/{RUN_CONFIGS[0]['run_label']}/results/generation_examples_top10.md")
print("- Validation curve:", f"{EXPERIMENT_ROOT}/{RUN_CONFIGS[0]['run_label']}/results/validation_loss_curve.png")
print("- Complete ZIP:", zip_file)
