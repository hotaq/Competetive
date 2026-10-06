"""Leakage-safe denoising-autoencoder + residual MLP for Playground S6E8.

The autoencoder is trained transductively on train and test covariates without labels.
Every target-derived feature is rebuilt inside each outer fold; the outer validation
labels never enter its training features.  The script exports aligned OOF/test streams
for downstream fixed blending or a truly nested level-2 experiment.
"""

from __future__ import annotations

import gc
import glob
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TARGET = "addicted_label"
ID = "id"
NUMERIC = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
LEVEL_COLUMNS = NUMERIC + CATEGORICAL

N_SPLITS = 5
OUTER_SEED = 42
INNER_SEED = 1701
BASE_SEEDS = (42,)
AE_SEEDS = (42, 1042)
SMOOTHING = 10.0
LATENT_DIM = 24
AE_EPOCHS = 8
NN_EPOCHS = 18
AE_BATCH_SIZE = 8192
NN_BATCH_SIZE = 4096


def log(message: str) -> None:
    print(message, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_competition_dir() -> str:
    preferred = "/kaggle/input/competitions/playground-series-s6e8"
    if os.path.exists(os.path.join(preferred, "train.csv")):
        return preferred
    for path in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        directory = os.path.dirname(path)
        if os.path.exists(os.path.join(directory, "test.csv")):
            return directory
    raise FileNotFoundError("competition train.csv/test.csv not found")


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=out,
        where=np.abs(denominator) > 1e-6,
    )
    return out


def make_label_free_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build label-free transductive features for the autoencoder and classifier."""
    combined = pd.concat(
        [train[LEVEL_COLUMNS], test[LEVEL_COLUMNS]], ignore_index=True
    )
    values: dict[str, np.ndarray] = {}

    filled: dict[str, np.ndarray] = {}
    for column in NUMERIC:
        raw = pd.to_numeric(combined[column], errors="coerce").to_numpy(np.float64)
        missing = np.isnan(raw)
        median = float(np.nanmedian(raw))
        x = np.where(missing, median, raw).astype(np.float32)
        filled[column] = x
        values[f"raw_{column}"] = x
        values[f"na_{column}"] = missing.astype(np.float32)
        values[f"log1p_{column}"] = np.log1p(np.clip(x, 0.0, None)).astype(np.float32)
        values[f"sqrt_{column}"] = np.sqrt(np.clip(x, 0.0, None)).astype(np.float32)
        values[f"square_{column}"] = np.square(x).astype(np.float32)
        values[f"frac_{column}"] = (x - np.floor(x)).astype(np.float32)
        values[f"digit1_{column}"] = ((np.floor(x * 10.0) % 10.0) / 9.0).astype(np.float32)
        values[f"digit2_{column}"] = ((np.floor(x * 100.0) % 10.0) / 9.0).astype(np.float32)

    daily = filled["daily_screen_time_hours"]
    social = filled["social_media_hours"]
    gaming = filled["gaming_hours"]
    work = filled["work_study_hours"]
    sleep = filled["sleep_hours"]
    notifications = filled["notifications_per_day"]
    opens = filled["app_opens_per_day"]
    weekend = filled["weekend_screen_time"]

    parts = social + gaming + work
    values.update(
        {
            "other_screen": daily - parts,
            "entertainment": social + gaming,
            "leisure": daily - work,
            "weekly_screen": 5.0 * daily + 2.0 * weekend,
            "weekend_delta": weekend - daily,
            "weekend_daily_ratio": safe_divide(weekend, daily),
            "social_share": safe_divide(social, daily),
            "gaming_share": safe_divide(gaming, daily),
            "work_share": safe_divide(work, daily),
            "other_share": safe_divide(daily - parts, daily),
            "screen_sleep_ratio": safe_divide(daily, sleep),
            "awake_screen_share": safe_divide(daily, 24.0 - sleep),
            "notifications_per_open": safe_divide(notifications, opens),
            "notifications_per_screen": safe_divide(notifications, daily),
            "opens_per_screen": safe_divide(opens, daily),
        }
    )

    numeric_frame = pd.DataFrame(values, copy=False)
    category_frame = pd.DataFrame(index=combined.index)
    for column in CATEGORICAL:
        levels = combined[column].astype("string").fillna("__MISSING__")
        dummies = pd.get_dummies(levels, prefix=column, dtype=np.float32)
        category_frame = pd.concat([category_frame, dummies], axis=1)

    frame = pd.concat([numeric_frame, category_frame], axis=1)
    matrix = frame.to_numpy(dtype=np.float32, copy=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler(copy=False)
    matrix = scaler.fit_transform(matrix).astype(np.float32, copy=False)
    n_train = len(train)
    return matrix[:n_train], matrix[n_train:], frame.columns.tolist()


def make_level_codes(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Factorize exact values jointly without using labels."""
    combined = pd.concat(
        [train[LEVEL_COLUMNS], test[LEVEL_COLUMNS]], ignore_index=True
    )
    codes = np.empty((len(combined), len(LEVEL_COLUMNS)), dtype=np.int32)
    level_counts: list[int] = []
    for j, column in enumerate(LEVEL_COLUMNS):
        keys = combined[column].astype("string").fillna("__MISSING__")
        column_codes, uniques = pd.factorize(keys, sort=True)
        if (column_codes < 0).any():
            raise ValueError(f"factorization produced missing codes for {column}")
        codes[:, j] = column_codes.astype(np.int32)
        level_counts.append(int(len(uniques)))
    return codes[: len(train)], codes[len(train) :], level_counts


def apply_target_maps(
    fit_codes: np.ndarray,
    fit_y: np.ndarray,
    query_codes: np.ndarray,
    level_counts: list[int],
) -> np.ndarray:
    """Apply smoothed target means and log frequencies fitted on one index set."""
    output = np.empty((len(query_codes), 2 * fit_codes.shape[1]), dtype=np.float32)
    global_mean = float(np.mean(fit_y))
    frequency_scale = np.log1p(max(len(fit_y), 1))
    for j, n_levels in enumerate(level_counts):
        counts = np.bincount(fit_codes[:, j], minlength=n_levels).astype(np.float64)
        sums = np.bincount(
            fit_codes[:, j], weights=fit_y, minlength=n_levels
        ).astype(np.float64)
        means = (sums + SMOOTHING * global_mean) / (counts + SMOOTHING)
        output[:, 2 * j] = means[query_codes[:, j]].astype(np.float32)
        output[:, 2 * j + 1] = (
            np.log1p(counts[query_codes[:, j]]) / frequency_scale
        ).astype(np.float32)
    return output


def build_nested_target_features(
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    valid_idx: np.ndarray,
    level_counts: list[int],
    outer_fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create inner-OOF train encodings and outer-safe validation/test encodings."""
    fit_codes = train_codes[fit_idx]
    fit_y = y[fit_idx]
    fit_features = np.empty((len(fit_idx), 2 * train_codes.shape[1]), dtype=np.float32)
    inner = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=INNER_SEED + outer_fold,
    )
    for inner_fit, inner_valid in inner.split(np.zeros(len(fit_y)), fit_y):
        fit_features[inner_valid] = apply_target_maps(
            fit_codes[inner_fit],
            fit_y[inner_fit],
            fit_codes[inner_valid],
            level_counts,
        )
    valid_features = apply_target_maps(
        fit_codes, fit_y, train_codes[valid_idx], level_counts
    )
    test_features = apply_target_maps(fit_codes, fit_y, test_codes, level_counts)
    return fit_features, valid_features, test_features


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Linear(256, 96),
            nn.BatchNorm1d(96),
            nn.Mish(),
            nn.Linear(96, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.BatchNorm1d(96),
            nn.Mish(),
            nn.Linear(96, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Linear(256, input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.activation = nn.Mish()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.block(values))


class ResidualClassifier(nn.Module):
    def __init__(self, input_dim: int, width: int = 256, depth: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.Mish(),
            nn.Dropout(0.08),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, dropout=0.10) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.Linear(width, 96),
            nn.Mish(),
            nn.Dropout(0.05),
            nn.Linear(96, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(values))).squeeze(1)


def make_loader(
    features: np.ndarray,
    labels: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors: tuple[torch.Tensor, ...]
    x_tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    if labels is None:
        tensors = (x_tensor,)
    else:
        y_tensor = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))
        tensors = (x_tensor, y_tensor)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=shuffle,
    )


def train_autoencoder(
    combined: np.ndarray, device: torch.device
) -> tuple[DenoisingAutoencoder, list[float]]:
    seed_everything(OUTER_SEED)
    model = DenoisingAutoencoder(combined.shape[1]).to(device)
    loader = make_loader(combined, None, AE_BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=AE_EPOCHS)
    amp = device.type == "cuda"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    losses: list[float] = []
    for epoch in range(AE_EPOCHS):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            mask = torch.rand_like(batch) < 0.15
            noisy = batch.masked_fill(mask, 0.0) + 0.02 * torch.randn_like(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                reconstruction = model(noisy)
                loss = nn.functional.mse_loss(reconstruction, batch)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            total_loss += float(loss.detach().cpu()) * len(batch)
            total_rows += len(batch)
        scheduler.step()
        epoch_loss = total_loss / total_rows
        losses.append(epoch_loss)
        log(f"autoencoder epoch={epoch + 1} mse={epoch_loss:.7f}")
    return model, losses


@torch.no_grad()
def encode_matrix(
    model: DenoisingAutoencoder, matrix: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    loader = make_loader(matrix, None, AE_BATCH_SIZE, shuffle=False)
    encoded: list[np.ndarray] = []
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        encoded.append(model.encoder(batch).float().cpu().numpy())
    return np.concatenate(encoded).astype(np.float32, copy=False)


@torch.no_grad()
def predict_classifier(
    model: ResidualClassifier,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = make_loader(features, None, NN_BATCH_SIZE, shuffle=False)
    predictions: list[np.ndarray] = []
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        logits = model(batch)
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(predictions).astype(np.float64, copy=False)


def train_classifier(
    fit_features: np.ndarray,
    fit_y: np.ndarray,
    valid_features: np.ndarray,
    valid_y: np.ndarray,
    test_features: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    seed_everything(seed)
    model = ResidualClassifier(fit_features.shape[1]).to(device)
    train_loader = make_loader(
        fit_features, fit_y.astype(np.float32), NN_BATCH_SIZE, shuffle=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NN_EPOCHS, eta_min=5e-5
    )
    loss_fn = nn.BCEWithLogitsLoss()
    amp = device.type == "cuda"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    history: list[dict[str, float]] = []
    for epoch in range(NN_EPOCHS):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch, labels in train_loader:
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(batch)
                loss = loss_fn(logits, labels)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            total_loss += float(loss.detach().cpu()) * len(batch)
            total_rows += len(batch)

        train_loss = total_loss / total_rows
        scheduler.step()
        history.append({"epoch": float(epoch + 1), "loss": train_loss})
        log(f"  seed={seed} epoch={epoch + 1} loss={train_loss:.7f}")

    valid_prediction = predict_classifier(model, valid_features, device)
    test_prediction = predict_classifier(model, test_features, device)
    return valid_prediction, test_prediction, {
        "seed": seed,
        "valid_auc": float(roc_auc_score(valid_y, valid_prediction)),
        "fixed_epochs": NN_EPOCHS,
        "history": history,
    }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("This training kernel requires a Kaggle GPU")
    device = torch.device("cuda")
    log(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    competition_dir = find_competition_dir()
    train = pd.read_csv(os.path.join(competition_dir, "train.csv"))
    test = pd.read_csv(os.path.join(competition_dir, "test.csv"))
    sample = pd.read_csv(os.path.join(competition_dir, "sample_submission.csv"))
    y = train[TARGET].to_numpy(dtype=np.int8)
    log(f"train={train.shape} test={test.shape} positive_rate={y.mean():.6f}")

    base_train, base_test, base_feature_names = make_label_free_features(train, test)
    train_codes, test_codes, level_counts = make_level_codes(train, test)
    log(
        f"label_free_features={base_train.shape[1]} exact_level_features={2 * len(LEVEL_COLUMNS)}"
    )

    combined_base = np.vstack([base_train, base_test])
    autoencoder, ae_losses = train_autoencoder(combined_base, device)
    combined_latent = encode_matrix(autoencoder, combined_base, device)
    latent_train = combined_latent[: len(train)]
    latent_test = combined_latent[len(train) :]
    del combined_base, combined_latent, autoencoder
    gc.collect()
    torch.cuda.empty_cache()

    folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=OUTER_SEED
        ).split(np.zeros(len(y)), y)
    )
    oof_base = np.zeros(len(train), dtype=np.float64)
    test_base = np.zeros(len(test), dtype=np.float64)
    oof_ae_by_seed = {
        seed: np.zeros(len(train), dtype=np.float64) for seed in AE_SEEDS
    }
    test_ae_by_seed = {
        seed: np.zeros(len(test), dtype=np.float64) for seed in AE_SEEDS
    }
    base_fold_metrics: list[dict[str, object]] = []
    ae_fold_metrics: list[dict[str, object]] = []

    for outer_fold, (fit_idx, valid_idx) in enumerate(folds):
        log(f"\nouter_fold={outer_fold + 1}/{N_SPLITS} building nested encodings")
        te_fit, te_valid, te_test = build_nested_target_features(
            train_codes,
            test_codes,
            y,
            fit_idx,
            valid_idx,
            level_counts,
            outer_fold,
        )

        raw_fit = np.hstack([base_train[fit_idx], te_fit]).astype(np.float32)
        raw_valid = np.hstack([base_train[valid_idx], te_valid]).astype(np.float32)
        raw_test = np.hstack([base_test, te_test]).astype(np.float32)
        scaler = StandardScaler(copy=False)
        raw_fit = scaler.fit_transform(raw_fit).astype(np.float32, copy=False)
        raw_valid = scaler.transform(raw_valid).astype(np.float32, copy=False)
        raw_test = scaler.transform(raw_test).astype(np.float32, copy=False)

        log(f"outer_fold={outer_fold + 1} training ResNet without AE")
        base_valid_predictions: list[np.ndarray] = []
        base_test_predictions: list[np.ndarray] = []
        for seed in BASE_SEEDS:
            valid_prediction, test_prediction, details = train_classifier(
                raw_fit,
                y[fit_idx],
                raw_valid,
                y[valid_idx],
                raw_test,
                seed + 10_000 * outer_fold,
                device,
            )
            details["outer_fold"] = outer_fold + 1
            base_fold_metrics.append(details)
            base_valid_predictions.append(valid_prediction)
            base_test_predictions.append(test_prediction)
        oof_base[valid_idx] = np.mean(base_valid_predictions, axis=0)
        test_base += np.mean(base_test_predictions, axis=0) / N_SPLITS

        ae_fit = np.hstack([base_train[fit_idx], latent_train[fit_idx], te_fit]).astype(
            np.float32
        )
        ae_valid = np.hstack(
            [base_train[valid_idx], latent_train[valid_idx], te_valid]
        ).astype(np.float32)
        ae_test = np.hstack([base_test, latent_test, te_test]).astype(np.float32)
        ae_scaler = StandardScaler(copy=False)
        ae_fit = ae_scaler.fit_transform(ae_fit).astype(np.float32, copy=False)
        ae_valid = ae_scaler.transform(ae_valid).astype(np.float32, copy=False)
        ae_test = ae_scaler.transform(ae_test).astype(np.float32, copy=False)

        log(f"outer_fold={outer_fold + 1} training AE ResNet seeds={AE_SEEDS}")
        for seed in AE_SEEDS:
            valid_prediction, test_prediction, details = train_classifier(
                ae_fit,
                y[fit_idx],
                ae_valid,
                y[valid_idx],
                ae_test,
                seed + 10_000 * outer_fold,
                device,
            )
            details["outer_fold"] = outer_fold + 1
            details["base_seed"] = seed
            ae_fold_metrics.append(details)
            oof_ae_by_seed[seed][valid_idx] = valid_prediction
            test_ae_by_seed[seed] += test_prediction / N_SPLITS

        fold_base_auc = roc_auc_score(y[valid_idx], oof_base[valid_idx])
        fold_ae_auc = roc_auc_score(
            y[valid_idx],
            np.mean([pred[valid_idx] for pred in oof_ae_by_seed.values()], axis=0),
        )
        log(
            f"outer_fold={outer_fold + 1} base_auc={fold_base_auc:.8f} "
            f"ae_seedavg_auc={fold_ae_auc:.8f}"
        )

        del (
            te_fit,
            te_valid,
            te_test,
            raw_fit,
            raw_valid,
            raw_test,
            ae_fit,
            ae_valid,
            ae_test,
        )
        gc.collect()
        torch.cuda.empty_cache()

    oof_ae = np.mean(list(oof_ae_by_seed.values()), axis=0)
    test_ae = np.mean(list(test_ae_by_seed.values()), axis=0)
    base_auc = float(roc_auc_score(y, oof_base))
    ae_seed_auc = {
        str(seed): float(roc_auc_score(y, prediction))
        for seed, prediction in oof_ae_by_seed.items()
    }
    ae_auc = float(roc_auc_score(y, oof_ae))
    fold_auc_base = [
        float(roc_auc_score(y[valid_idx], oof_base[valid_idx]))
        for _, valid_idx in folds
    ]
    fold_auc_ae = [
        float(roc_auc_score(y[valid_idx], oof_ae[valid_idx]))
        for _, valid_idx in folds
    ]

    np.save("oof_resnet_base.npy", oof_base)
    np.save("test_resnet_base.npy", test_base)
    np.save("oof_resnet_ae.npy", oof_ae)
    np.save("test_resnet_ae.npy", test_ae)
    for seed in AE_SEEDS:
        np.save(f"oof_resnet_ae_seed{seed}.npy", oof_ae_by_seed[seed])
        np.save(f"test_resnet_ae_seed{seed}.npy", test_ae_by_seed[seed])

    oof_frame = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "resnet_base": oof_base,
            "resnet_ae": oof_ae,
            **{
                f"resnet_ae_seed{seed}": oof_ae_by_seed[seed]
                for seed in AE_SEEDS
            },
        }
    )
    oof_frame.to_csv("oof.csv", index=False)
    pd.DataFrame(
        {
            ID: test[ID],
            "resnet_base": test_base,
            "resnet_ae": test_ae,
            **{
                f"resnet_ae_seed{seed}": test_ae_by_seed[seed]
                for seed in AE_SEEDS
            },
        }
    ).to_csv("test_predictions.csv", index=False)
    submission = sample.copy()
    submission[TARGET] = test_ae
    submission.to_csv("submission_resnet_ae.csv", index=False)

    metrics: dict[str, object] = {
        "protocol": {
            "outer_cv": f"StratifiedKFold({N_SPLITS}, shuffle=True, random_state={OUTER_SEED})",
            "inner_target_encoding_cv": (
                f"StratifiedKFold({N_SPLITS}, shuffle=True, random_state={INNER_SEED}+outer_fold)"
            ),
            "transductive_label_free_autoencoder": True,
            "target_encoding_smoothing": SMOOTHING,
        },
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "rows": {"train": len(train), "test": len(test)},
        "feature_counts": {
            "label_free": len(base_feature_names),
            "latent": LATENT_DIM,
            "target_and_frequency": 2 * len(LEVEL_COLUMNS),
        },
        "autoencoder_mse": ae_losses,
        "resnet_base_oof_auc": base_auc,
        "resnet_base_fold_auc": fold_auc_base,
        "resnet_ae_seed_oof_auc": ae_seed_auc,
        "resnet_ae_oof_auc": ae_auc,
        "resnet_ae_fold_auc": fold_auc_ae,
        "ae_delta_vs_base": ae_auc - base_auc,
        "spearman_base_ae": float(spearmanr(oof_base, oof_ae).statistic),
        "base_training_details": base_fold_metrics,
        "ae_training_details": ae_fold_metrics,
        "runtime_seconds": time.time() - started,
    }
    with open("metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    hashes = {}
    for output in [
        "oof_resnet_base.npy",
        "test_resnet_base.npy",
        "oof_resnet_ae.npy",
        "test_resnet_ae.npy",
        "oof.csv",
        "test_predictions.csv",
        "submission_resnet_ae.csv",
        "metrics.json",
    ]:
        hashes[output] = sha256(output)
    with open("sha256.json", "w", encoding="utf-8") as handle:
        json.dump(hashes, handle, indent=2)

    log(
        json.dumps(
            {
                "resnet_base_oof_auc": base_auc,
                "resnet_ae_seed_oof_auc": ae_seed_auc,
                "resnet_ae_oof_auc": ae_auc,
                "ae_delta_vs_base": ae_auc - base_auc,
                "runtime_seconds": metrics["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
