import os
HERE = 'ensemble_farm'
cp = os.path.join(HERE, 'controller.py')
src = open(cp).read()

old_block = '''    shutil.copy(os.path.join(HERE, "kernel_train.py"), d)
    shutil.copy(os.path.join(HERE, "farm_lib.py"), d)
    batch = BATCHES.get(n)
    if not batch:
        print(f"no BATCHES[{n}] defined - pass ids explicitly"); sys.exit(1)
    json.dump(batch, open(os.path.join(d, "batch.json"), "w"), indent=1)'''
new_block = '''    batch = BATCHES.get(n)
    if not batch:
        print(f"no BATCHES[{n}] defined - pass ids explicitly"); sys.exit(1)
    _build_single_file(d, batch)'''
assert old_block in src, 'block1 not found'
src = src.replace(old_block, new_block)

old_meta = '"code_file": "kernel_train.py"'
assert old_meta in src
src = src.replace(old_meta, '"code_file": "train.py"')

builder = '''

def _build_single_file(d, batch):
    lib = open(os.path.join(HERE, "farm_lib.py")).read()
    body = open(os.path.join(HERE, "kernel_train.py")).read()
    imp = "from farm_lib import zoo, get_folds, build_matrices, train_config, greedy_blend, TARGET"
    assert imp in body, "import line not found in kernel_train.py"
    body = body.replace(imp, "# farm_lib inlined below")
    old_batch = 'batch = json.load(open(os.path.join(HERE, "batch.json")))'
    assert old_batch in body, "batch load line not found"
    body = body.replace(old_batch, "batch = BATCH")
    out = lib + "\n\n# ================= BATCH (injected) =================\nBATCH = " + repr(batch) + "\n" + body
    open(os.path.join(d, "train.py"), "w").write(out)
    print(f"built single-file train.py ({len(out) // 1024} KB) batch={len(batch)}")
'''
marker = '\n\ndef cmd_new('
assert marker in src, 'cmd_new marker not found'
src = src.replace(marker, builder + '\n\ndef cmd_new(')
open(cp, 'w').write(src)
print('controller patched')
