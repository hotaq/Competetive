import json, os
nb_path = 'kaggle_push_kernel/kaggle_push.ipynb'
with open(nb_path) as f:
    nb = json.load(f)
cell = nb['cells'][2]
src = cell['source']
if isinstance(src, list):
    src = ''.join(src)
if '_CANDS' not in src:
    old = 'tr=pd.read_csv(TRAIN_P); te=pd.read_csv(TEST_P)'
    new = """import glob
_CANDS = glob.glob('/kaggle/input/**/train.csv', recursive=True)
print('INPUT ROOT:', os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else 'NO INPUT')
print('CANDS:', _CANDS)
if _CANDS:
    TRAIN_P = _CANDS[0]
    _root = os.path.dirname(TRAIN_P)
    TEST_P = os.path.join(_root, 'test.csv'); SUB_P = os.path.join(_root, 'sample_submission.csv')
print('USING:', TRAIN_P, TEST_P, SUB_P)
tr=pd.read_csv(TRAIN_P); te=pd.read_csv(TEST_P)"""
    assert old in src, 'anchor not found'
    src = src.replace(old, new)
    lines = src.split('\n')
    cell['source'] = [l + '\n' if i < len(lines)-1 else l for i, l in enumerate(lines)]
    with open(nb_path, 'w') as f:
        json.dump(nb, f)
    print('patched OK')
else:
    print('already patched')
