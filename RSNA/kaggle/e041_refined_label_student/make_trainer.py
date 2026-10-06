"""Generate assets/train_e041.py from the E021 trainer with exact, counted replacements."""
import ast
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = (HERE.parent / "e021_train_single_coatnet/e021-train-single-coatnet.py").read_text()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


sub('"""Knee MRI: training the twelve-finding model\n',
    '"""E041 trainer = E021 (dreaddevelopment twelve-finding recipe) with five changes:\n'
    '  input search skips the competition DICOM trees;\n'
    '  --labels is required (refined or control parquet from the e041 assets dataset);\n'
    '  224 px coatnet_rmlp_2_rw_224 by default; fp16 + GradScaler on pre-Ampere GPUs (T4 has no\n'
    '  bf16 tensor cores); --max_hours wall-clock guard so a Kaggle timeout cannot eat the output.\n\n'
    'Knee MRI: training the twelve-finding model\n')
# Labels now arrive as a file, so the in-kernel E021 consensus build goes.
s, n = re.subn(r'E021_LABEL_SOURCES = \{.*?\n\}\n\n', '', s, flags=re.S)
assert n == 1
s, n = re.subn(r'def _e021_source_file\(name\):.*?\n\n\ndef main\(\):\n    label_path = _build_e021_labels\(\)\n',
               'def main():\n', s, flags=re.S)
assert n == 1
# Never walk the competition's DICOM trees (train_series/, test_series/) while searching inputs.
sub('    for d, _, fs in os.walk(root):\n',
    '    for d, dirs, fs in os.walk(root):\n'
    '        dirs[:] = [x for x in dirs if not x.endswith(("_series", "_images"))]\n')
sub('default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")', 'default="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k")')
sub('ap.add_argument("--res", type=int, default=384)', 'ap.add_argument("--res", type=int, default=224)')
sub('ap.add_argument("--tag", default="e021_cc0")',
    'ap.add_argument("--tag", default="e041")\n    ap.add_argument("--max_hours", type=float, default=10.0)')
sub('ap.add_argument("--labels", default=label_path,', 'ap.add_argument("--labels", required=True,')
sub('ap.add_argument("--smoke", action="store_true", default=True)', 'ap.add_argument("--smoke", action="store_true")')
sub('    dev = "cuda" if torch.cuda.is_available() else "cpu"\n',
    '    dev = "cuda" if torch.cuda.is_available() else "cpu"\n'
    '    AMP = torch.bfloat16 if dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16\n'
    '    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda" and AMP == torch.float16)\n'
    '    print(f"[amp] {AMP} | grad scaler {scaler.is_enabled()}", flush=True)\n')
sub('dtype=torch.bfloat16', 'dtype=AMP', count=5)
sub('            loss.backward()\n'
    '            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)\n'
    '            opt.step(); sched.step(); tot += loss.item()\n',
    '            scaler.scale(loss).backward()\n'
    '            scaler.unscale_(opt)\n'
    '            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)\n'
    '            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()\n')
sub('''              flush=True)
    _, aucs, _, Y = evaluate()
''', '''              flush=True)
        if time.time() - t0 > a.max_hours * 3600:
            print(f"[guard] {a.max_hours}h budget reached after ep{ep}; stopping early", flush=True)
            break
    _, aucs, _, Y = evaluate()
''')
assert "_build_e021_labels" not in s and "E021_LABEL_SOURCES" not in s
ast.parse(s)
(HERE / "assets/train_e041.py").write_text(s)
print("wrote assets/train_e041.py", len(s.splitlines()), "lines")
