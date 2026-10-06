"""FT-Transformer farm kernel: per-feature tokenizer + Income/Commute ID embeddings on naji matrix.
GPU (T4). 5-fold seed42 (aligned). Saves oof/test npy + submission."""
import os, glob, gc, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SMOKE = os.environ.get("SMOKE") == "1"
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.1
BATCH = 4096
LR = 1e-3
MAX_EPOCHS = 2 if SMOKE else 40
PATIENCE = 6
SEEDS = [42] if not SMOKE else [42]

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0), flush=True)
USE_AMP = torch.cuda.is_available()

KAGGLE = os.path.exists("/kaggle/input")
if KAGGLE:
    cands = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    assert cands, "no train.csv"
    TRAIN_P = cands[0]
    ROOTD = os.path.dirname(TRAIN_P)
    TEST_P = os.path.join(ROOTD, "test.csv")
    SUB_P = os.path.join(ROOTD, "sample_submission.csv")
    OUT = "/kaggle/working"
    _og = glob.glob("/kaggle/input/**/EV_Adoption*.csv", recursive=True)
else:
    TRAIN_P, TEST_P, SUB_P, OUT = "train.csv", "test.csv", "sample_submission.csv", "farm_out_tt_smoke"
    os.makedirs(OUT, exist_ok=True)
    _og = glob.glob("orig_data/EV_Adoption*.csv")
assert _og, "original missing"
ORIG_P = _og[0]

NROWS = int(os.environ.get("SMOKE_ROWS", 30000)) if SMOKE else None
tr = pd.read_csv(TRAIN_P, nrows=NROWS); te = pd.read_csv(TEST_P, nrows=(5000 if SMOKE else None))
orig = pd.read_csv(ORIG_P)
y = (tr[TARGET] == "Yes").astype(int).values
print(f"train {tr.shape} test {te.shape}", flush=True)

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
M = build_matrices(tr, te, "naji", y=y, folds=folds, orig=orig)
Xo = M["oh_train"].astype(np.float32); Xt = M["oh_test"].astype(np.float32)
Xo = np.nan_to_num(Xo); Xt = np.nan_to_num(Xt)
print("matrix:", Xo.shape, flush=True)

# ---- ID embeddings (label-free, train+test vocabulary; 0 = unknown) ----
def id_index(col):
    vals = pd.concat([tr[col], te[col]]).astype(str)
    vocab = {v: i + 1 for i, v in enumerate(sorted(vals.unique()))}
    a = tr[col].astype(str).map(vocab).fillna(0).astype(np.int64).values
    b = te[col].astype(str).map(vocab).fillna(0).astype(np.int64).values
    return a, b, len(vocab) + 1
inc_tr, inc_te, N_INC = id_index("Annual_Income_USD")
com_tr, com_te, N_COM = id_index("Daily_Commute_km")
print(f"id vocab: income {N_INC} commute {N_COM}", flush=True)


class FTTransformer(nn.Module):
    def __init__(self, n_feat, n_inc, n_com, d=D_MODEL, layers=N_LAYERS, heads=N_HEADS, drop=DROPOUT):
        super().__init__()
        # per-feature tokenizer: token_j = x_j * W_j + b_j  (b_j doubles as field embedding)
        self.W = nn.Parameter(torch.randn(n_feat, d) * 0.02)
        self.B = nn.Parameter(torch.randn(n_feat, d) * 0.02)
        self.emb_inc = nn.Embedding(n_inc, d)
        self.emb_com = nn.Embedding(n_com, d)
        nn.init.normal_(self.emb_inc.weight, std=0.02); nn.init.normal_(self.emb_com.weight, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=d * 4,
                                               dropout=drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, x, inc, com):
        t = x.unsqueeze(-1) * self.W + self.B                       # [B, F, d]
        ids = torch.stack([self.emb_inc(inc), self.emb_com(com)], 1)  # [B, 2, d]
        cls = self.cls.expand(x.size(0), -1, -1)
        h = self.enc(torch.cat([cls, t, ids], dim=1))
        return self.head(self.norm(h[:, 0])).squeeze(-1)


def predict(model, X, I, C):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(X), BATCH * 2):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                o = model(X[s:s + BATCH * 2], I[s:s + BATCH * 2], C[s:s + BATCH * 2])
            out.append(torch.sigmoid(o.float()).cpu().numpy())
    return np.concatenate(out)


def run_fold(ti, vi, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler()
    Xtr = torch.from_numpy(sc.fit_transform(Xo[ti]).astype(np.float32)).to(device)
    Xva = torch.from_numpy(sc.transform(Xo[vi]).astype(np.float32)).to(device)
    Xs = torch.from_numpy(sc.transform(Xt).astype(np.float32)).to(device)
    Itr, Iva, Is = [torch.from_numpy(a).to(device) for a in (inc_tr[ti], inc_tr[vi], inc_te)]
    Ctr, Cva, Cs = [torch.from_numpy(a).to(device) for a in (com_tr[ti], com_tr[vi], com_te)]
    ytr = torch.from_numpy(y[ti].astype(np.float32)).to(device); yva = y[vi]
    model = FTTransformer(Xtr.shape[1], N_INC, N_COM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    crit = nn.BCEWithLogitsLoss()
    n = len(ti); best_auc, best_state, bad = 0.0, None, 0
    for ep in range(MAX_EPOCHS):
        model.train(); t0 = time.monotonic()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                loss = crit(model(Xtr[idx], Itr[idx], Ctr[idx]).float(), ytr[idx])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        auc = roc_auc_score(yva, predict(model, Xva, Iva, Cva))
        print(f"    ep{ep:02d} val_auc={auc:.5f} loss={loss.item():.4f} ({time.monotonic() - t0:.0f}s)", flush=True)
        if auc > best_auc + 1e-5:
            best_auc = auc; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    pv = predict(model, Xva, Iva, Cva); pt = predict(model, Xs, Is, Cs)
    del model, Xtr, Xva, Xs; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pv, pt, best_auc


sub = pd.read_csv(SUB_P, nrows=(5000 if SMOKE else None))
weights = {}
for seed in SEEDS:
    oof = np.zeros(len(y)); test_pred = np.zeros(len(Xt))
    for fold, (ti, vi) in enumerate(folds):
        t0 = time.monotonic()
        pv, pt, auc = run_fold(ti, vi, seed + fold)
        oof[vi] = pv; test_pred += pt / len(folds)
        print(f"<<< ft_s{seed} fold {fold} AUC={auc:.5f} ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    auc = roc_auc_score(y, oof)
    print(f"<<< ft_s{seed} OOF={auc:.5f}", flush=True)
    np.save(os.path.join(OUT, f"oof_ft_s{seed}.npy"), oof)
    np.save(os.path.join(OUT, f"test_ft_s{seed}.npy"), test_pred)
    weights[f"ft_s{seed}"] = float(auc)
    sub[TARGET] = test_pred
    sub.to_csv(os.path.join(OUT, f"submission_ft_s{seed}.csv"), index=False)
json.dump({"weights": {k: 1.0 / len(weights) for k in weights}, "solo_oof": weights,
           "blend_oof": max(weights.values())}, open(os.path.join(OUT, "weights.json"), "w"))
print("saved", flush=True)
