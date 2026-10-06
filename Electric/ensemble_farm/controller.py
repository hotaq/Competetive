"""Farm controller (runs LOCAL, trains on KAGGLE)."""
import os, sys, json, subprocess, datetime, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMP = "playground-series-s6e9"
USER = "chinnaphatkhun"
NL = chr(10)

BATCHES = {
    1: ["lgb_fe_s42_A", "lgb_fe_s7_A", "lgb_base_s42_B", "cat_fe_s42_d7",
        "cat_fe_s7_d6", "xgb_fe_s42_d7", "xgb_fe_s7_d6", "hgb_fe_s42_df",
        "logreg_fe_s42_C0.1", "lgb_fe_s11_A"],
    2: ["lgb_fe_s23_C", "lgb_base_s99_A", "cat_fe_s11_d8", "cat_base_s23_d6",
        "xgb_fe_s11_d8", "xgb_base_s23_d7", "hgb_base_s7_df", "logreg_base_s42_C0.05"],
    # shallow trees + Income target-encoding + original dataset (EDA 2026-09-05: OOF ~0.9455)
    3: ["lgb_te_s42_d3", "lgb_te_s42_d2", "lgb_te_s7_d3", "lgb_te_s42_d4", "lgb_tefe_s42_d3",
        "cat_te_s42_d4", "cat_te_s7_d3", "cat_te_s11_d5", "xgb_te_s42_d3", "xgb_te_s7_d2",
        "hgb_te_s42_d3", "logreg_te_s42_C0.1"],
    4: ["lgb_te_s11_d3", "lgb_te_s7_d2", "lgb_te_s11_d4", "lgb_tefe_s7_d4", "cat_te_s23_d6",
        "cat_tefe_s42_d4", "xgb_te_s11_d4", "xgb_tefe_s42_d3", "hgb_te_s7_d4", "logreg_tefe_s42_C0.1"],
    # it5: same best families as it3 but 10-fold (see N_SPLITS_BY_BATCH); hgb/logreg dropped (0 blend weight in it3)
    5: ["lgb_te_s42_d3", "lgb_te_s7_d3", "lgb_te_s11_d3", "lgb_te_s42_d2", "lgb_te_s7_d2",
        "lgb_te_s42_d4", "lgb_te_s11_d4", "cat_te_s42_d4", "cat_te_s7_d3", "cat_te_s11_d5",
        "xgb_te_s42_d3", "xgb_te_s7_d2"],
    # it6: digit + frequency + Income TE (LB-proven artifacts); 5-fold for speed
    6: ["lgb_te2_s42_d3", "lgb_te2_s7_d3", "lgb_te2_s42_d2", "lgb_te2_s42_d4",
        "cat_te2_s42_d4", "xgb_te2_s42_d3"],
    # it7: fe interactions + digit + frequency + Income TE
    7: ["lgb_tefe2_s42_d3", "lgb_tefe2_s7_d3", "lgb_tefe2_s42_d4",
        "cat_tefe2_s42_d4", "xgb_tefe2_s42_d3", "logreg_tefe2_s42_C0.1"],
    # it8: Naji-style artifact set (dual sklearn TE, orig means, lr 0.02 / col 0.3)
    8: ["lgb_naji_s42_n5", "lgb_naji_s7_n5", "lgb_naji_s42_n4", "lgb_naji_s11_n6",
        "cat_naji_s42_d4", "xgb_naji_s42_d3"],
    # it9: exotics (dart / extra-trees / mlp / hgb-alt / logreg-C1)
    9: ["lgb_te2_s42_dart", "lgb_te2_s7_dart", "lgb_naji_s42_xtra",
        "mlp_naji_s42_h32", "hgb_te2_s42_d3b", "logreg_naji_s42_C1"],
    # it10: MLP v2 (128,64 / alpha 1e-4 / 300 iters / patience 15)
    10: ["mlp_naji_s42_h128", "mlp_naji_s7_h128", "mlp_te2_s42_h64"],
    # it12: coarse param sweep on naji (lr 0.01/0.02/0.04 x leaves 16/32)
    12: ["lgb_naji_s42_g01_16", "lgb_naji_s42_g01_32", "lgb_naji_s42_g02_16",
        "lgb_naji_s42_g04_16", "lgb_naji_s42_g04_32", "lgb_naji_s7_g01_16"],
    # it15: focal-loss LGB (gamma 2.0/2.0/1.0)
    15: ["lgb_naji_s42_focal", "lgb_naji_s7_focal", "lgb_te2_s42_focal"],
    # it16: Deotte recipe (buy_score feats + base-margin XGB)
    16: ["lgb_te2_s42_r1", "lgb_te2_s7_r1", "cat_te2_s42_r1",
        "xgb_te2_s42_r1", "xgbm_te2_s42_m1", "xgbm_te2_s7_m1"],
    # it17: residual tuning (xgbm lr/depth/margin-scale grid + xgb lr0.01)
    17: ["xgbm_te2_s42_r2", "xgbm_te2_s42_r3", "xgbm_te2_s42_r4",
        "xgbm_te2_s42_r5", "xgbm_te2_s7_r2", "xgb_te2_s42_r2"],
    # it18: lamhuy economic/conditional set
    18: ["lgb_lam_s42_d3", "lgb_lam_s7_d3", "lgb_lam_s42_d4",
        "lgb_lam_s42_d2", "cat_lam_s42_d4", "xgb_lam_s42_d3"],
    # it19: agent-A conditional set (gates, city×home, peak, tier, seg prior)
    19: ["lgb_eda_s42_d3", "lgb_eda_s7_d3", "lgb_eda_s42_d4",
        "lgb_eda_s42_d2", "cat_eda_s42_d4", "xgb_eda_s42_d3"],
    # it20: feature views (megayak C/D/F mechanisms) + linear-leaf LGBM
    20: ["lgb_win_s42_d5", "lgb_nokey_s42_d5", "lgb_init_s42_d5", "lgb_te2_s42_lin"],
}
ORIG_DS = "itzzomkar/ev-adoption-behavior-and-range-anxiety"
# per-batch fold count override (default 5). 10-fold: TE on 90% of rows -> local +0.0004 OOF (2026-09-05)
N_SPLITS_BY_BATCH = {5: 10}


def run(*a):
    p = subprocess.run(a, capture_output=True, text=True, cwd=ROOT)
    return (p.stdout + p.stderr).strip()


def slug(n):
    return f"{USER}/ev-farm-it{n}"


def _build_single_file(d, batch, n_splits=None):
    lib = open(os.path.join(HERE, "farm_lib.py")).read()
    body = open(os.path.join(HERE, "kernel_train.py")).read()
    imp = "from farm_lib import zoo, get_folds, build_matrices, train_config, greedy_blend, TARGET"
    assert imp in body, "import line not found"
    body = body.replace(imp, "# farm_lib inlined above")
    old_batch = 'batch = json.load(open(os.path.join(HERE, "batch.json")))'
    assert old_batch in body, "batch line not found"
    body = body.replace(old_batch, "batch = BATCH")
    head = "# ================= BATCH (injected) ================="
    inj = "BATCH = " + repr(batch) + NL
    if n_splits:
        inj += f"N_SPLITS_OVERRIDE = {int(n_splits)}" + NL
    out = lib + NL + NL + head + NL + inj + body
    open(os.path.join(d, "train.py"), "w").write(out)
    print(f"built single-file train.py ({len(out) // 1024} KB) batch={len(batch)} n_splits={n_splits or 'default'}")


def cmd_new(n, prev):
    n = int(n)
    d = os.path.join(ROOT, f"farm_it{n}")
    os.makedirs(d, exist_ok=True)
    batch = BATCHES.get(n)
    if not batch:
        print(f"no BATCHES[{n}] defined"); sys.exit(1)
    _build_single_file(d, batch, N_SPLITS_BY_BATCH.get(n))
    meta = {"id": slug(n), "title": f"ev-farm-it{n}", "code_file": "train.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": [ORIG_DS], "kernel_sources": [], "model_sources": [],
            "competition_sources": [COMP]}
    if prev:
        meta["kernel_sources"] = [prev]
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    print(f"made {d} prev={prev}")


def cmd_push(n):
    d = os.path.join(ROOT, f"farm_it{n}")
    print(run("kaggle", "kernels", "push", "-p", d))


def cmd_status(s):
    print(run("kaggle", "kernels", "status", s))


def cmd_pull(n, s):
    d = os.path.join(ROOT, f"farm_out_it{n}")
    os.makedirs(d, exist_ok=True)
    print(run("kaggle", "kernels", "output", s, "-p", d))
    w = os.path.join(d, "weights.json")
    if os.path.exists(w):
        print(open(w).read()[:600])
    else:
        print("no weights.json yet; files:", os.listdir(d))


def today_count():
    out = run("kaggle", "competitions", "submissions", "-c", COMP, "-q")
    today = datetime.date.today().isoformat()
    n = sum(1 for ln in out.splitlines() if today in ln)
    return n, out


def cmd_quota():
    n, _ = today_count()
    print(f"submissions today: {n} (playground daily limit ~20)")


def cmd_submit(n):
    n_, out = today_count()
    print(f"submissions today: {n_}")
    if n_ >= 15:
        print("quota guard: >=15 today, refusing auto-submit.")
        return
    d = os.path.join(ROOT, f"farm_out_it{n}")
    f = os.path.join(d, "submission_farm.csv")
    w = json.load(open(os.path.join(d, "weights.json")))
    msg = f"farm-it{n} blend-oof-{w['blend_oof']:.5f} n={len(w['weights'])}"
    print(run("kaggle", "competitions", "submit", "-c", COMP, "-f", f, "-m", msg))
    with open(os.path.join(ROOT, "farm_log.csv"), "a") as fh:
        fh.write(f"{datetime.datetime.now().isoformat()},it{n},{w['blend_oof']:.5f},{len(w['weights'])},{msg}" + NL)
    print("logged to farm_log.csv")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "new":
        cmd_new(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif c == "push":
        cmd_push(sys.argv[2])
    elif c == "status":
        cmd_status(sys.argv[2])
    elif c == "status-it":
        cmd_status(slug(sys.argv[2]))
    elif c == "pull":
        cmd_pull(sys.argv[2], sys.argv[3])
    elif c == "submit":
        cmd_submit(sys.argv[2])
    elif c == "quota":
        cmd_quota()
