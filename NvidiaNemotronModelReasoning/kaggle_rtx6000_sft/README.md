# Kaggle RTX Pro 6000 SFT Kernel

This folder is a Kaggle kernel package for the NVIDIA Nemotron Model Reasoning Challenge.

It starts from the public DGXChen/Tong SFT lane:

- rank-32 LoRA
- max sequence length 8192
- SFTTrainer
- boxed final answers
- no internet in the main kernel
- intended accelerator: `NvidiaRtxPro6000`

## Files

```text
kernel-metadata.json     Kaggle kernel metadata
notebooke4b8ccf598.ipynb Kaggle notebook code generated from the script
train_rtx6000_sft.py     Main training and packaging script
```

## Push And Run

Use the GitHub/2.x Kaggle CLI because the old PyPI 1.7.x CLI does not support the accelerator flag.

```bash
kaggle kernels push -p kaggle_rtx6000_sft --accelerator NvidiaRtxPro6000
```

From the repository root, the helper script does the same:

```bash
scripts/push_kaggle_rtx6000.sh
```

## Output

The kernel writes:

```text
/kaggle/working/sft_output/
/kaggle/working/sft_adapter/
/kaggle/working/submission_adapter/
/kaggle/working/submission.zip
```

Download after the Kaggle run finishes:

```bash
kaggle kernels output chinnaphatkhun/notebooke4b8ccf598 -p artifacts/kaggle_rtx6000_sft
```

## Quick Smoke Option

For a short Kaggle smoke run, set this near the top of `train_rtx6000_sft.py`:

```python
MAX_RECORDS = 512
```

For the real first run, keep `MAX_RECORDS = 0`.

## Fine-Tune Mode

Use these settings for real training:

```python
TRAIN_ON_KAGGLE = 1
USE_PRETRAINED = 0
MAX_RECORDS = 0
FILTER_BOXED_COT = False
```

Set `FILTER_BOXED_COT = True` only after confirming the attached CoT CSV has boxed answers worth filtering.
