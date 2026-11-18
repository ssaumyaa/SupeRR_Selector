"""
STEP 8 - Stability of the ENSEMBLE ranking.
 (a) bootstrap over matches  (b) design-choice sensitivity (whole ensemble rebuilt)  (c) split-half reliability
"""
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT = PROJECT_DIR / "outputs"
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd, numpy as np, sys, itertools, pickle, warnings, time; warnings.filterwarnings('ignore')
sys.path.insert(0, str(SCRIPT_DIR))
from step4_pressure_index import add_pi
from step5_ratings import player_ps, rate
import step7_ensemble as E
from scipy.stats import spearmanr
pd.set_option('display.width',250)
nm = E.nm
PS = pickle.load(open(str(OUT / 'ens_ps.pkl'),'rb'))

def ens(PS_role, w=None, min_share=0.7, **kw):
    tab = pd.DataFrame({cfg: rate(ps, w=w, **kw)['PRS'] for cfg,ps in PS_role.items()})
    return tab[tab.notna().mean(axis=1)>=min_share].mean(axis=1).sort_values(ascending=False)
BASE = {r: ens(PS[r]) for r in ['bat','bowl']}

mode = sys.argv[1]
if mode=='variants':
    d0 = E.d0
    VAR = {'ordinary = bottom 40%':dict(lo=0.40),'ordinary = bottom 60%':dict(lo=0.60),'decided cut-off 10-90%':dict(live=(0.10,0.90)),
           'no decided-ball exclusion':dict(live=(0.0,1.0)),'percentiles over whole innings':dict(by_phase=False)}
    rows=[]
    for name,kw in VAR.items():
        P2 = {'bat':{}, 'bowl':{}}
        for (w,hi) in E.CONFIGS:
            dd = add_pi(d0, w=w, hi=hi, **kw)
            for role in P2: P2[role][(w,hi)] = player_ps(dd, role)
        for role in ['bat','bowl']:
            t = list(ens(P2[role]).index[:10]); rows.append((name,role,len(set(t)&set(BASE[role].index[:10])), len(set(t[:5])&set(BASE[role].index[:5]))))
    RK = {'min 90 high-pressure balls':dict(min_H=90),'seasonal blocks need 30 balls':dict(kb=30),'min 4 blocks':dict(min_blocks=4)}
    for name,kw in RK.items():
        for role in ['bat','bowl']:
            t = list(ens(PS[role], **kw).index[:10]); rows.append((name,role,len(set(t)&set(BASE[role].index[:10])), len(set(t[:5])&set(BASE[role].index[:5]))))
    for lg in ['IPL','T20I']:
        for role in ['bat','bowl']:
            P3 = {cfg: ps[ps.block.str.endswith(lg)] for cfg,ps in PS[role].items()}
            t = list(ens(P3, min_H=40, min_O=60, kb=15, min_blocks=2).index[:10]); rows.append((f'{lg} data only',role,len(set(t)&set(BASE[role].index[:10])), len(set(t[:5])&set(BASE[role].index[:5]))))
    R = pd.DataFrame(rows, columns=['variant','role','overlap_top10','overlap_top5']).pivot(index='variant',columns='role',values=['overlap_top10','overlap_top5'])
    R.to_csv(str(OUT / 'ensemble_sensitivity.csv')); print('Overlap of the ENSEMBLE top 10 / top 5 with the default ensemble'); print(R.to_string())
    # split-half reliability
    rng = np.random.RandomState(5); kw = dict(min_H=30,min_O=50,kb=10,min_blocks=2)
    for role in ['bat','bowl']:
        mids = PS[role][((1,1,1),0.20)].match_id.unique(); rho=[]; ov=[]
        for i in range(30):
            perm = rng.permutation(mids); h1 = pd.Series(1.0, index=perm[:len(perm)//2]); h2 = pd.Series(1.0, index=perm[len(perm)//2:])
            a = ens(PS[role], w=h1, min_share=0.5, **kw); b2 = ens(PS[role], w=h2, min_share=0.5, **kw); c = a.index.intersection(b2.index)
            rho.append(spearmanr(a[c], b2[c])[0]); ov.append(len(set(a.index[:10])&set(b2.index[:10])))
        print(f'\nsplit-half ({role}): mean Spearman between halves = {np.mean(rho):.2f} (range {np.min(rho):.2f}..{np.max(rho):.2f}), n players per half ~{len(c)}; top-10 overlap between halves = {np.mean(ov):.1f}/10; Spearman-Brown full-sample reliability = {2*np.mean(rho)/(1+np.mean(rho)):.2f}')
else:
    reps = int(sys.argv[2]); rng = np.random.RandomState(2024); t0=time.time()
    for role in ['bat','bowl']:
        mids = PS[role][((1,1,1),0.20)].match_id.unique(); top = BASE[role].index[:15]
        inc = {}; rk = {p:[] for p in top}
        for i in range(reps):
            draw = pd.Series(rng.choice(mids, len(mids))).value_counts(); s = ens(PS[role], w=draw); order = list(s.index)
            for p in order[:10]: inc[p]=inc.get(p,0)+1
            pos = {p:r+1 for r,p in enumerate(order)}
            for p in top: rk[p].append(pos.get(p,np.nan))
        out = pd.DataFrame({'name':[nm[p] for p in top]}, index=top); out.insert(0,'rank',range(1,16)); out['score']=BASE[role].loc[top].round(3).values
        out['P_top10_%'] = [round(100*inc.get(p,0)/reps) for p in top]; out['median_rank']=[np.nanmedian(rk[p]) for p in top]
        out['rank_p10']=[np.nanpercentile(rk[p],10) for p in top]; out['rank_p90']=[np.nanpercentile(rk[p],90) for p in top]
        out.to_csv(str(OUT / f'ensemble_bootstrap_{role}.csv'))
        print(f'\n=== ENSEMBLE BOOTSTRAP ({role}, {reps} resamples of matches; {time.time()-t0:.0f}s) ==='); print(out.to_string())
        print('most frequent outsiders:', [(nm[p], round(100*v/reps)) for p,v in sorted(inc.items(), key=lambda x:-x[1]) if p not in top[:10]][:5])
