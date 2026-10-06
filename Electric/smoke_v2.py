import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
tr=pd.read_csv('train.csv', nrows=20000)
te=pd.read_csv('test.csv', nrows=5000)
y=(tr['Will_Buy_EV']=='Yes').astype(int).values
def add_features(df):
    d=df.copy()
    d['Income_log']=np.log1p(d['Annual_Income_USD'])
    d['TotalStations']=d['Charging_Stations_Near_Home']+d['Charging_Stations_Near_Work']
    d['StationDiff']=d['Charging_Stations_Near_Home']-d['Charging_Stations_Near_Work']
    d['Commute_per_station']=d['Daily_Commute_km']/(d['TotalStations']+1)
    d['SubsidyYes']=(d['Subsidy_Available']=='Yes').astype(int)
    d['HomeYes']=(d['Home_Charging_Possible']=='Yes').astype(int)
    d['AnxietyOrd']=d['Range_Anxiety_Level'].map({'Low':0,'Medium':1,'High':2}).astype(int)
    d['Env_x_Subsidy']=d['Environmental_Concern_Level']*d['SubsidyYes']
    d['Income_x_Env']=d['Income_log']*d['Environmental_Concern_Level']
    d['Env_x_Home']=d['Environmental_Concern_Level']*d['HomeYes']
    d['HomeSubsidy']=d['HomeYes']*2+d['SubsidyYes']
    d['City_Car']=d['City_Type'].astype(str)+'_'+d['Current_Car_Type'].astype(str)
    d['Env5']=(d['Environmental_Concern_Level']==5).astype(int)
    d['AnxietyHigh']=(d['Range_Anxiety_Level']=='High').astype(int)
    return d
CAT=['Gender','City_Type','Current_Car_Type','Home_Charging_Possible','Subsidy_Available','Range_Anxiety_Level','City_Car']
trf=add_features(tr); tef=add_features(te)
FE=[c for c in trf.columns if c not in ['id','Will_Buy_EV']]
for c in CAT:
    trf[c]=trf[c].astype('category'); tef[c]=tef[c].astype('category')
X=trf[FE]; Xt=tef[FE]
skf=StratifiedKFold(n_splits=2,shuffle=True,random_state=42)
for ti,vi in skf.split(X,y):
    dtr=lgb.Dataset(X.iloc[ti],label=y[ti],categorical_feature=CAT)
    dva=lgb.Dataset(X.iloc[vi],label=y[vi],categorical_feature=CAT)
    m=lgb.train({'objective':'binary','metric':'auc','verbosity':-1,'seed':42,'learning_rate':0.1,'num_leaves':31,'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':1,'device_type':'cpu'},dtr,num_boost_round=50,valid_sets=[dva],callbacks=[lgb.log_evaluation(0)])
    print('smoke fold AUC=',round(roc_auc_score(y[vi],m.predict(X.iloc[vi])),5))
    break
print('FE OK:',len(FE),'features')
print('smoke passed')
