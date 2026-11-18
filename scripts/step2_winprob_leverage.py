
"""
STEP 2 - Win-probability (WP) model + 'Leverage' (how much the NEXT ball can swing the match).
 * Separate logistic-regression WP model for innings 1 (setting a total) and innings 2 (chasing).
 * Out-of-fold predictions grouped by match (5 folds) -> no match is ever scored by a model trained on it.
 * Logistic beat gradient boosting on out-of-fold log-loss (0.636 vs 0.641 inn1, 0.398 vs 0.411 inn2) and is better calibrated.
Output: balls_lev.pkl  (adds wp_before, wp_after, dwp, leverage columns)
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss

b = pd.read_pickle(OUTPUT_DIR / "balls_state.pkl")

def make_X(inn, runs, wkts, legal, inn1_total, is_ipl):
    runs=np.asarray(runs,float); wkts=np.asarray(wkts,float); legal=np.asarray(legal,float)
    inn1_total=np.asarray(inn1_total,float); is_ipl=np.asarray(is_ipl,float)
    bl = 120-legal; crr = runs*6/np.maximum(legal,1)
    X = {'runs':runs,'wkts':wkts,'balls_left':bl,'is_ipl':is_ipl,'crr':crr,'wb':(10-wkts)*bl}
    if inn==2:
        need = inn1_total+1-runs; rrr = need*6/np.maximum(bl,1); r = np.clip(rrr,0,40)
        X.update({'need':need,'rrr':rrr,'inn1_total':inn1_total,'rrr_minus_crr':rrr-crr,'rrr_c2':r**2,
                  'log_need':np.log1p(np.maximum(need,0)),
                  'need_per_wkt':np.maximum(need,0)/np.maximum(10-wkts,1),'sqrt_bl':np.sqrt(bl)})
        X['need_per_wkt_sq']=X['need_per_wkt']**2
    return pd.DataFrame(X)

def apply_rules(p, inn, runs, wkts, legal, inn1_total):
    p = np.asarray(p,float).copy()
    if inn==2:
        need = np.asarray(inn1_total)+1-np.asarray(runs)
        p[need<=0] = 1.0
        p[(need>0) & ((np.asarray(wkts)>=10)|(np.asarray(legal)>=120))] = 0.0
    return p

# ---------- weights of next-ball outcomes (league-wide, by innings x phase) ----------
b['phase'] = np.where(b.overid<=6,'PP',np.where(b.overid<=15,'MID','DEATH'))
L = b[b.is_legal].copy()
L['out'] = np.where(L.wkt==1,'W', np.where(L.bat_runs.isin([0,1,2,3,4,6]), L.bat_runs.astype(str), 'x'))
L = L[L.out!='x']
W = L.groupby(['innings','phase']).out.value_counts(normalize=True).unstack().fillna(0)
OUTS = [('0',0,0),('1',1,0),('2',2,0),('3',3,0),('4',4,0),('6',6,0),('W',0,1)]

def train_states(df, inn):
    d = df[df.innings==inn]
    st = pd.DataFrame({'match_id':d.match_id,'runs':d.runs_after,'wkts':d.wkts_after,'legal':d.legal_after,
                       'inn1_total':d.inn1_total,'is_ipl':d.is_ipl,'y':d.bat_team_won})
    first = d.groupby('match_id').first()
    z = pd.DataFrame({'match_id':first.index,'runs':0,'wkts':0,'legal':0,'inn1_total':first.inn1_total.values,
                      'is_ipl':first.is_ipl.values,'y':first.bat_team_won.values})
    return pd.concat([st,z], ignore_index=True)

mids = b.match_id.unique()
fold_of = dict(zip(mids, np.random.RandomState(7).randint(0,5,len(mids))))
b['fold'] = b.match_id.map(fold_of)
b['wp_before']=np.nan; b['wp_after']=np.nan; b['leverage']=np.nan

for inn in [1,2]:
    T = train_states(b, inn); T['fold']=T.match_id.map(fold_of)
    X = make_X(inn, T.runs, T.wkts, T.legal, T.inn1_total, T.is_ipl)
    oof = np.zeros(len(T))
    for f in range(5):
        tr=(T.fold!=f).values; te=~tr
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X[tr], T.y[tr])
        oof[te] = apply_rules(mdl.predict_proba(X[te])[:,1], inn, T.runs[te], T.wkts[te], T.legal[te], T.inn1_total[te])
        idx = (b.innings==inn)&(b.fold==f); d = b[idx]
        def wp(runs,wkts,legal):
            p = mdl.predict_proba(make_X(inn,runs,wkts,legal,d.inn1_total,d.is_ipl))[:,1]
            return apply_rules(p, inn, runs, wkts, legal, d.inn1_total)
        base = wp(d.runs_before, d.wkts_before, d.legal_before)
        b.loc[idx,'wp_before'] = base
        b.loc[idx,'wp_after']  = wp(d.runs_after, d.wkts_after, d.legal_after)
        lev = np.zeros(len(d))
        phase_probs = W.loc[inn].reindex(d.phase.values).fillna(0).to_numpy()
        for name,r,w in OUTS:
            p_o = phase_probs[:, W.columns.get_loc(name)]
            nxt = wp(d.runs_before.values+r, d.wkts_before.values+w, d.legal_before.values+1)
            lev += p_o*np.abs(nxt-base)
        b.loc[idx,'leverage'] = lev
    T['oof']=oof
    print('innings %d  OOF AUC=%.3f  logloss=%.3f  (n states=%d)'%(inn, roc_auc_score(T.y,oof), log_loss(T.y,np.clip(oof,1e-4,1-1e-4)), len(T)))
    T['bin']=pd.cut(T.oof,[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0],include_lowest=True)
    print(T.groupby('bin',observed=True).agg(pred=('oof','mean'),actual=('y','mean'),n=('y','size')).round(3).T.to_string())

b['dwp'] = b.wp_after - b.wp_before
b.to_pickle(OUTPUT_DIR / "balls_lev.pkl")
print(b.groupby('innings').leverage.describe(percentiles=[.5,.75,.9,.95]).round(4))
print(b.groupby(['innings','phase']).leverage.mean().unstack().round(4))
