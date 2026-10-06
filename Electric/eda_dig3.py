import pandas as pd
ROOT="/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
tr=pd.read_csv(f"{ROOT}/train.csv", usecols=["Annual_Income_USD","Environmental_Concern_Level","Will_Buy_EV"])
te=pd.read_csv(f"{ROOT}/test.csv", usecols=["Annual_Income_USD","Environmental_Concern_Level"])
for th in [150000,155000,160000,162000,165000,170000,170537]:
    s=tr[tr.Annual_Income_USD>=th]
    print(f"train inc>={th}: n={len(s)} rate={(s.Will_Buy_EV==chr(89)+chr(101)+chr(115)).mean():.4f}")
print("--- test counts above thresholds ---")
for th in [160000,165000,170537]:
    print(f"test inc>={th}: n={(te.Annual_Income_USD>=th).sum()}")
print("--- rate by Env for inc>=160k ---")
print(tr[tr.Annual_Income_USD>=160000].groupby("Environmental_Concern_Level")["Will_Buy_EV"].apply(lambda s:(s=="Yes").mean()).round(4).to_string())
print("--- top incomes: distinct values >=160k in train ---")
print(tr[tr.Annual_Income_USD>=160000].Annual_Income_USD.value_counts().head(10).to_string())
