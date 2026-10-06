#!/usr/bin/env python3
import glob
import json
import math
import os
import random
import re
import shutil
import site
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path


os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="strict")


# Mode selection. For the first real Kaggle run keep TRAIN_ON_KAGGLE=1.
TRAIN_ON_KAGGLE = 1
USE_PRETRAINED = 0
assert (TRAIN_ON_KAGGLE + USE_PRETRAINED) == 1, "Set exactly one mode to 1."

# Set to 512 for a short smoke run. Keep 0 for the full first run.
MAX_RECORDS = 0
FILTER_BOXED_COT = False

REQUIRE_RTX_PRO_6000 = True
SEED = 42
MAX_SEQ_LEN = 8192
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LEARNING_RATE = 2e-4
GRAD_ACCUM_STEPS = 32
PER_DEVICE_BATCH_SIZE = 1

BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
KAGGLE_MODEL_REF = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

PRETRAINED_ADAPTER_DATASET_PATH = "/kaggle/input/datasets/dgxchen/trained-adapter"
OUTPUT_DIR = Path("/kaggle/working")
ADAPTER_DIR = OUTPUT_DIR / "sft_adapter"
SUBMISSION_ADAPTER_DIR = OUTPUT_DIR / "submission_adapter"


def run(cmd, *, shell=False):
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"+ {printable}", flush=True)
    subprocess.run(cmd, shell=shell, check=True)


def add_utility_paths():
    candidates = [
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script",
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script",
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages",
    ]
    candidates.extend(glob.glob("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia*"))
    candidates.extend(glob.glob("/kaggle/input/**/pydeps", recursive=True))
    candidates.extend(glob.glob("/kaggle/input/**/unsloth_compiled_cache", recursive=True))
    for path in candidates:
        if os.path.isdir(path):
            site.addsitedir(path)
            sys.path.insert(0, path)
            sys.path.insert(0, os.path.dirname(path))
            print(f"Added utility path: {path}")


def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def install_offline_training_deps():
    already_available = []
    for module in ["unsloth", "trl", "peft", "transformers", "datasets", "accelerate"]:
        try:
            __import__(module)
            already_available.append(module)
        except Exception:
            pass
    if {"unsloth", "trl", "peft", "transformers", "datasets"}.issubset(set(already_available)):
        print("Training dependencies already importable:", already_available)
        return

    packages_dir = first_existing(
        [
            "/kaggle/input/datasets/mayukh18/nemotron-packages/packages",
            "/kaggle/input/nemotron-packages/packages",
        ]
    )
    if not packages_dir:
        package_dirs = []
        for d in glob.glob("/kaggle/input/**/packages", recursive=True):
            if glob.glob(os.path.join(d, "unsloth*.whl")) or glob.glob(os.path.join(d, "trl*.whl")):
                package_dirs.append(d)
        if package_dirs:
            packages_dir = package_dirs[0]
    if not packages_dir:
        print("Top Kaggle inputs:")
        for path in sorted(glob.glob("/kaggle/input/*")):
            print(" ", path)
        raise FileNotFoundError(
            "Could not find offline training wheels. Add `mayukh18/nemotron-packages` "
            "or attach a notebook/dataset that provides unsloth, trl, peft, transformers, and datasets."
        )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-index",
            "--find-links",
            packages_dir,
            "unsloth",
            "trl",
            "peft",
            "transformers",
            "datasets",
            "accelerate",
            "bitsandbytes",
        ]
    )

    def module_available(name):
        try:
            __import__(name)
            return True
        except Exception as exc:
            print(f"{name} import not ready yet: {exc}")
            return False

    if not module_available("causal_conv1d"):
        wheels = sorted(glob.glob("/kaggle/input/**/*causal*conv1d*.whl", recursive=True))
        if wheels:
            run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", wheels[-1]])
    if not module_available("mamba_ssm"):
        wheels = sorted(glob.glob("/kaggle/input/**/mamba_ssm-*.whl", recursive=True))
        if wheels:
            run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", wheels[-1]])


def print_gpu_sanity():
    import torch

    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("This kernel requires a Kaggle GPU runtime.")

    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name}")
    print(f"Compute capability: sm_{cc[0] * 10 + cc[1]}")
    print(f"VRAM: {vram:.1f} GB")

    looks_like_rtx6000 = ("RTX" in name.upper() and "6000" in name) or cc == (12, 0)
    if REQUIRE_RTX_PRO_6000 and not looks_like_rtx6000:
        raise RuntimeError(
            "Expected RTX Pro 6000 / Blackwell. Push with: "
            "kaggle kernels push -p kaggle_rtx6000_sft --accelerator NvidiaRtxPro6000"
        )

    import causal_conv1d
    import mamba_ssm

    print("mamba_ssm:", getattr(mamba_ssm, "__version__", "unknown"))
    print("causal_conv1d:", getattr(causal_conv1d, "__version__", "unknown"))


def find_training_csv():
    candidates = [
        "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv",
        "/kaggle/input/nemotron-cot-tong/problem_ids_matched.csv",
    ]
    found = first_existing(candidates)
    if found:
        return found
    matches = sorted(glob.glob("/kaggle/input/**/problem_ids_matched.csv", recursive=True))
    if not matches:
        raise FileNotFoundError("Could not find problem_ids_matched.csv in Kaggle inputs.")
    return matches[0]


def build_sft_records(tokenizer):
    import pandas as pd
    from datasets import Dataset as HFDataset

    dataset_path = find_training_csv()
    print(f"Training data: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Source rows: {len(df)}")
    if FILTER_BOXED_COT:
        before = len(df)

        def extract_boxed(text):
            matches = re.findall(r"\\boxed\{([^}]*)\}", str(text))
            return matches[-1].strip() if matches else ""

        df["boxed_from_cot"] = df["generated_cot"].map(extract_boxed)
        df["answer_str"] = df["answer"].astype(str).str.strip()
        has_boxed = df["boxed_from_cot"].astype(bool)
        clean = (~has_boxed) | (df["boxed_from_cot"] == df["answer_str"])
        rejected = df[~clean].copy()
        df = df[clean].copy()
        rejected.to_csv(OUTPUT_DIR / "rejected_boxed_cot_mismatch.csv", index=False)
        print(f"FILTER_BOXED_COT kept {len(df)}/{before}; rejected {len(rejected)}")
        if len(df) < 1000:
            raise RuntimeError("Too few rows after FILTER_BOXED_COT. Set FILTER_BOXED_COT=False or inspect data.")
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    if MAX_RECORDS:
        df = df.head(MAX_RECORDS)
        print(f"Smoke-limited rows: {len(df)}")

    records = []
    record_types = []
    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        cot = str(row["generated_cot"])
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            continue
        cot_cleaned = re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()
        user_content = prompt + PROMPT_SUFFIX
        assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
        )
        record_types.append(str(row.get("type", "unknown")))

    print(f"SFT records: {len(records)}")
    print("Records by type:", dict(sorted(pd.Series(record_types).value_counts().to_dict().items())))

    def formatting_prompts_func(example):
        messages = example["messages"]
        conversations = [messages] if messages and isinstance(messages[0], dict) else messages
        texts = []
        for conversation in conversations:
            try:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append(text)
        return texts

    return HFDataset.from_list(records), record_types, formatting_prompts_func


def build_stratified_index_order(labels, batch_size, seed):
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    rng = random.Random(seed)
    for idx_list in by_label.values():
        rng.shuffle(idx_list)

    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)

    assigned = 0
    for label in sorted(by_label.keys()):
        for idx in by_label[label]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1

    order = [idx for batch in batches for idx in batch]
    if len(order) != len(labels):
        raise ValueError("Stratified order size mismatch")
    return order


def train_adapter():
    import kagglehub
    import torch
    from torch.utils.data import DataLoader, Sampler
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model_path = kagglehub.model_download(KAGGLE_MODEL_REF)
    print(f"Model path: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj",
        "out_proj",
        "up_proj",
        "down_proj",
    ]
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    model.print_trainable_parameters()

    dataset, record_types, formatting_prompts_func = build_sft_records(tokenizer)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "sft_output"),
        num_train_epochs=1,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_length=MAX_SEQ_LEN,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1e9,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=SEED,
        report_to="none",
        packing=False,
    )

    class PrecomputedOrderSampler(Sampler):
        def __init__(self, order):
            self.order = list(order)

        def __iter__(self):
            return iter(self.order)

        def __len__(self):
            return len(self.order)

    class StratifiedSFTTrainer(SFTTrainer):
        def __init__(self, *args, stratified_order=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.stratified_order = stratified_order

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer requires a train_dataset.")
            if self.stratified_order is None:
                return super().get_train_dataloader()
            dataloader_kwargs = {
                "batch_size": self.args.per_device_train_batch_size,
                "sampler": PrecomputedOrderSampler(self.stratified_order),
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "drop_last": self.args.dataloader_drop_last,
            }
            if self.args.dataloader_num_workers > 0:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
            return DataLoader(self.train_dataset, **dataloader_kwargs)

    effective_batch_size = max(1, PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS)
    stratified_order = build_stratified_index_order(record_types, effective_batch_size, SEED)
    print(f"Effective batch size: {effective_batch_size}")

    trainer = StratifiedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        stratified_order=stratified_order,
    )

    print("Starting SFT training...")
    t0 = time.time()
    trainer.train()
    print(f"Training done in {(time.time() - t0) / 60:.1f} min")

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}")


def package_adapter():
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    src_adapter_dir = ADAPTER_DIR if TRAIN_ON_KAGGLE else Path(PRETRAINED_ADAPTER_DATASET_PATH)

    SUBMISSION_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Packaging adapter from: {src_adapter_dir}")
    for fname in required_files:
        src = Path(src_adapter_dir) / fname
        dst = SUBMISSION_ADAPTER_DIR / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing required adapter file: {src}")
        shutil.copy2(src, dst)
        print(f"Copied {fname}: {dst.stat().st_size / 1024 / 1024:.1f} MB")

    config_path = SUBMISSION_ADAPTER_DIR / "adapter_config.json"
    with config_path.open("r") as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = BASE_MODEL_NAME
    cfg["inference_mode"] = True
    cfg["lora_dropout"] = 0.0
    with config_path.open("w") as f:
        json.dump(cfg, f, indent=2)

    zip_path = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in required_files:
            zf.write(SUBMISSION_ADAPTER_DIR / fname, fname)
            print(f"Added {fname}")
    print(f"submission.zip: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    print(
        {
            "TRAIN_ON_KAGGLE": TRAIN_ON_KAGGLE,
            "USE_PRETRAINED": USE_PRETRAINED,
            "MAX_RECORDS": MAX_RECORDS,
            "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "LORA_RANK": LORA_RANK,
            "LORA_ALPHA": LORA_ALPHA,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
        }
    )
    add_utility_paths()
    if TRAIN_ON_KAGGLE:
        install_offline_training_deps()
        print_gpu_sanity()
        train_adapter()
    package_adapter()
    print("Done. Kaggle output is /kaggle/working/submission.zip")


if __name__ == "__main__":
    main()
