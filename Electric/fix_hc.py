p = 'hill_climb.py'
s = open(p).read()
old = '    feats = BASE_NUM + CATL + FE + feats_extra\n    Xtr, Xva = F.iloc[idx_tr][feats].copy(), F.iloc[idx_va][feats].copy()'
new = '    feats = BASE_NUM + CATL + FE\n    Xtr, Xva = F.iloc[idx_tr][feats].copy(), F.iloc[idx_va][feats].copy()'
assert old in s, 'anchor missing'
s = s.replace(old, new)
open(p, 'w').write(s)
print('fixed')
