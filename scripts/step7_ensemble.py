"""
STEP 7 - Definition-robust ENSEMBLE ranking.
Instead of betting on one pressure definition, score every player under a grid of 15 defensible definitions
(5 Pressure-Index weightings x 3 'high pressure' cut-offs) and rank by the AVERAGE Pressure Reliability Score.
A player only ranks high if he delivers whichever reasonable way pressure is defined.
Outputs: ensemble_<role>.csv, ens_ps_<role>.pkl (per-config sufficient stats for bootstrap)
"""
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT = PROJECT_DIR / "outputs"
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd, numpy as np, sys, time, itertools, pickle, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, str(SCRIPT_DIR))
from step4_pressure_index import add_pi
from step5_ratings import player_ps, rate

d0 = pd.read_pickle(str(OUT / 'balls_values.pkl'))
b = pd.read_pickle(str(OUT / 'balls_state.pkl'))
nm = pd.concat([b[['striker_id','striker_name']].rename(columns={'striker_id':'pid','striker_name':'name'}),
                b[['bowler_id','bowler_name']].rename(columns={'bowler_id':'pid','bowler_name':'name'})]).drop_duplicates('pid').set_index('pid').name
dup = nm[nm.duplicated(keep=False)].index; nm.loc[dup] = nm.loc[dup]+' (id '+dup.astype(str)+')'
WEIGHTS = [(1,1,1),(2,1,1),(1,2,1),(1,1,2),(1,0,1)]; HIS = [0.15,0.20,0.30]
CONFIGS = list(itertools.product(WEIGHTS, HIS))

def build():
    PS = {'bat':{}, 'bowl':{}}
    for (w,hi) in CONFIGS:
        dd = add_pi(d0, w=w, hi=hi)
        for role in PS: PS[role][(w,hi)] = player_ps(dd, role)
    pickle.dump(PS, open(str(OUT / 'ens_ps.pkl'),'wb')); return PS

def ensemble(PS_role, w=None, min_share=0.7):
    tab = pd.DataFrame({cfg: rate(ps, w=w)['PRS'] for cfg,ps in PS_role.items()})     # players x configs (NaN if ineligible)
    keep = tab.notna().mean(axis=1) >= min_share
    return tab[keep].mean(axis=1).sort_values(ascending=False), tab

if __name__=='__main__':
    t0=time.time(); PS = build(); print('built 15 configs in %.0fs'%(time.time()-t0))
    for role in ['bat','bowl']:
        t1=time.time(); score, tab = ensemble(PS[role]); print(role,'ensemble in %.1fs'%(time.time()-t1))
        base = rate(PS[role][((1,1,1),0.20)])
        out = pd.DataFrame({'score':score}); out['name']=out.index.map(nm)
        out['configs_in_top10_%'] = [round(100*(tab.loc[p].rank(ascending=False) <= 0).sum(),1) for p in out.index]  # placeholder replaced below
        ranks = tab.rank(ascending=False, axis=0)
        out['share_of_defs_in_top10_%'] = [round(100*(ranks.loc[p]<=10).mean(),0) for p in out.index]
        out['median_rank_across_defs'] = [ranks.loc[p].median() for p in out.index]
        out = out.drop(columns='configs_in_top10_%')
        for c in ['nH','nO','mH','mO','H_s','U_s','C_s']: out[c] = base[c].reindex(out.index)
        out.insert(0,'rank',range(1,len(out)+1)); out.to_csv(str(OUT / f'ensemble_{role}.csv'))
        print(f'\n===== ENSEMBLE {role.upper()} : players eligible in >=70% of definitions = {len(out)} =====')
        pd.set_option('display.width',250)
        print(out.head(15)[['rank','name','score','share_of_defs_in_top10_%','median_rank_across_defs','nH','mH','mO','H_s','U_s','C_s']].round(3).to_string())
