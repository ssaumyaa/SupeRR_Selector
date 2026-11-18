"""
STEP 6 - Robustness: (A) sensitivity to every design choice, (B) bootstrap over matches (300 resamples).
Outputs: final_batters.csv, final_bowlers.csv (point-estimate rankings), sensitivity_overlap.csv, bootstrap_top10.csv
"""
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT = PROJECT_DIR / "outputs"
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd, numpy as np, sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, str(SCRIPT_DIR))
from step4_pressure_index import add_pi
from step5_ratings import player_ps, rate

d0 = pd.read_pickle(str(OUT / 'balls_values.pkl'))      # per-ball values (BV, BoV) are independent of the pressure grouping
b = pd.read_pickle(str(OUT / 'balls_state.pkl'))
nm = pd.concat([b[['striker_id','striker_name']].rename(columns={'striker_id':'pid','striker_name':'name'}),
                b[['bowler_id','bowler_name']].rename(columns={'bowler_id':'pid','bowler_name':'name'})]).drop_duplicates('pid').set_index('pid').name
dup = nm[nm.duplicated(keep=False)].index; nm.loc[dup] = nm.loc[dup] + ' (id ' + dup.astype(str) + ')'

def top10(el): return list(el.head(10).index)
BASE = {}
for role in ['bat','bowl']:
    ps = pd.read_pickle(str(OUT / f'ps_{role}.pkl')); BASE[role] = (ps, rate(ps))
    t = BASE[role][1].head(10).copy(); t.insert(0,'rank',range(1,11)); t.insert(1,'name',t.index.map(nm))
    t[['rank','name','nH','nO','mH','mO','H_s','U_s','hits','blocks','C_s','PRS']].round(4).to_csv(str(OUT / f'final_{"batters" if role=="bat" else "bowlers"}.csv'))

# ---------------- (A) sensitivity ----------------
variants = {
 'default (PI parts equal, top20%/bottom50%, by-phase)': dict(),
 'PI weights: leverage-heavy (2,1,1)': dict(pi=dict(w=(2,1,1))),
 'PI weights: momentum-heavy (1,2,1)': dict(pi=dict(w=(1,2,1))),
 'PI weights: strain-heavy (1,1,2)': dict(pi=dict(w=(1,1,2))),
 'PI without momentum (1,0,1)  [circularity check]': dict(pi=dict(w=(1,0,1))),
 'PI leverage only (1,0,0)': dict(pi=dict(w=(1,0,0))),
 'high = top 15%': dict(pi=dict(hi=0.15)),
 'high = top 30%': dict(pi=dict(hi=0.30)),
 'ordinary = bottom 40%': dict(pi=dict(lo=0.40)),
 'ordinary = bottom 60%': dict(pi=dict(lo=0.60)),
 'decided cut-off 10%-90% (stricter live)': dict(pi=dict(live=(0.10,0.90))),
 'no decided-ball exclusion': dict(pi=dict(live=(0.0,1.0))),
 'percentiles over whole innings (not by phase)': dict(pi=dict(by_phase=False)),
 'min sample 90 high-pressure balls': dict(rate=dict(min_H=90)),
 'blocks need 30 high-pressure balls': dict(rate=dict(kb=30)),
 'composite = agreed (uplift + hit-rate only)': dict(rate=dict(base_w=(0,1,1))),
 'composite = level only': dict(rate=dict(base_w=(1,0,0))),
 'IPL data only': dict(league='IPL'),
 'T20I data only': dict(league='T20I'),
}
rows=[]; tops={}
for name,cfg in variants.items():
    dd = add_pi(d0, **cfg['pi']) if 'pi' in cfg else d0
    for role in ['bat','bowl']:
        ps = player_ps(dd, role) if 'pi' in cfg else BASE[role][0]
        if 'league' in cfg: ps = ps[ps.block.str.endswith(cfg['league'])]
        el = rate(ps, **cfg.get('rate',{}))
        t = top10(el); tops[(name,role)] = t
        base_t = top10(BASE[role][1])
        rows.append(dict(variant=name, role=role, overlap_top10=len(set(t)&set(base_t)), overlap_top5=len(set(t[:5])&set(base_t[:5])), n_eligible=len(el)))
S = pd.DataFrame(rows).pivot(index='variant', columns='role', values=['overlap_top10','overlap_top5','n_eligible'])
S = S.loc[list(variants)]; S.to_csv(str(OUT / 'sensitivity_overlap.csv'))
print('Overlap of each variant\'s top 10 with the default top 10 (max 10)'); print(S.to_string())

# ---------------- (B) bootstrap over matches ----------------
rng = np.random.RandomState(11); B_REPS = 300
for role in ['bat','bowl']:
    ps, base = BASE[role]; mids = ps.match_id.unique()
    cnt = {p:0 for p in base.index}; ranks = {p:[] for p in base.index}; allcnt = {}
    for i in range(B_REPS):
        draw = pd.Series(rng.choice(mids, len(mids))).value_counts()
        el = rate(ps, w=draw)
        order = list(el.index)
        for r,p in enumerate(order[:10]): allcnt[p]=allcnt.get(p,0)+1
        pos = {p:r+1 for r,p in enumerate(order)}
        for p in base.index[:30]: ranks[p].append(pos.get(p, np.nan))
    out = base.head(15)[['nH','H_s','U_s','C_s','PRS']].copy(); out.insert(0,'name',out.index.map(nm)); out.insert(0,'point_rank',range(1,16))
    out['P_top10_%'] = [round(100*allcnt.get(p,0)/B_REPS,1) for p in out.index]
    out['median_rank'] = [np.nanmedian(ranks[p]) for p in out.index]
    out['rank_p10'] = [np.nanpercentile(ranks[p],10) for p in out.index]; out['rank_p90']=[np.nanpercentile(ranks[p],90) for p in out.index]
    out.round(3).to_csv(str(OUT / f'bootstrap_top10_{role}.csv'))
    hi_not_top = sorted([(v,p) for p,v in allcnt.items() if p not in base.index[:10]], reverse=True)[:5]
    print(f'\n=== BOOTSTRAP ({role}) : how often each point-estimate top-15 player lands in the top 10 across {B_REPS} resampled seasons ===')
    print(out[['point_rank','name','P_top10_%','median_rank','rank_p10','rank_p90','PRS']].to_string())
    print('players outside point top 10 that most often enter top 10:', [(nm[p], round(100*v/B_REPS)) for v,p in hi_not_top])
