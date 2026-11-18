"""
STEP 5 - Per-ball performance value, player baselines, and the Pressure Reliability Score (PRS).

Per-ball VALUE (context-adjusted, in 'runs'):
  Batting  BV  = (runs off bat - expected runs) - (dismissed - expected dismissal rate) * wicket_cost
  Bowling  BoV = -(runs conceded - expected)   + (wicket taken - expected wicket rate)  * wicket_cost
  'expected' = what a typical player gets on that ball given league (IPL/T20I), innings, over number, how set the batter is,
               plus a ridge-estimated opponent-strength adjustment. It deliberately does NOT use score/pressure info,
  so pressure effects stay visible. wicket_cost = runs a side loses by losing a wicket from that state (step 4).

Per player (batters use grp_bat, bowlers grp_bowl):
  HIGH-pressure mean (H), ORDINARY mean (O, the player's own usual level),
  Uplift = H - O, shrunk toward the pool (empirical Bayes) because pressure samples are small.
  Hit-rate = share of match-innings (>= 4 high-pressure balls) where the player's HIGH-pressure average >= own ordinary level (Beta-shrunk).
  PRS = 0.5*z(shrunk uplift) + 0.5*z(shrunk hit-rate).  Gate: shrunk HIGH-pressure level must be >= pool average.
"""
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT = PROJECT_DIR / "outputs"
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

def add_values(d):
    d = d.copy()
    d['sb'] = np.minimum(d.striker_balls, 30)
    d['dis'] = (d.is_wicket & (d.dismissed_player_id==d.striker_id)).astype(float)
    F = ['is_ipl','innings','overid','sb']
    def hgb(): return HistGradientBoostingRegressor(max_depth=4, learning_rate=.08, max_iter=120, min_samples_leaf=400, random_state=0)
    faced = ~d.extras_type.eq('wide')
    for tgt,name,mask in [('bat_runs','E_br',faced),('dis','E_dis',faced),('bowler_runs','E_bor',d.index==d.index),('bowler_wkt','E_bwk',d.index==d.index)]:
        m = hgb().fit(d.loc[mask,F], d.loc[mask,tgt]); d[name] = m.predict(d[F])
    # opponent-strength adjustment (ridge; teams as dummies, batting+bowling team jointly)
    enc = OneHotEncoder(handle_unknown='ignore'); Xt = enc.fit_transform(d[['batting_team_id','bowling_team_id']].astype(str))
    nb = len(enc.categories_[0])
    def team_eff(tgt, E, mask):
        r = (d.loc[mask,tgt]-d.loc[mask,E]).values
        rg = Ridge(alpha=1500, fit_intercept=True).fit(Xt[np.where(mask)[0]], r)
        coef = rg.coef_
        return dict(zip(enc.categories_[0], coef[:nb])), dict(zip(enc.categories_[1], coef[nb:]))
    bat_eff_bat, bowl_eff_bat = team_eff('bat_runs','E_br',faced)             # effect on batter runs/ball
    bat_eff_bow, bowl_eff_bow = team_eff('bowler_runs','E_bor',d.index==d.index)
    d['opp_for_batter'] = d.bowling_team_id.astype(str).map(bowl_eff_bat).fillna(0)   # leaky/tight bowling side faced
    d['opp_for_bowler'] = d.batting_team_id.astype(str).map(bat_eff_bow).fillna(0)    # strong/weak batting side faced
    d['BV']  = (d.bat_runs-(d.E_br+d.opp_for_batter)) - (d.dis-d.E_dis)*d.wkt_cost
    d['BoV'] = -(d.bowler_runs-(d.E_bor+d.opp_for_bowler)) + (d.bowler_wkt-d.E_bwk)*d.wkt_cost
    d.loc[~faced,'BV'] = np.nan                                                        # batters do not face wides
    return d

def player_ps(d, role):
    """sufficient statistics per player x match x innings, for HIGH and ORD balls"""
    if role=='bat': pid,val,grp,mask = 'striker_id','BV','grp_bat', ~d.BV.isna()
    else:           pid,val,grp,mask = 'bowler_id','BoV','grp_bowl', d.BoV.notna()
    d = d.assign(block=d.match_date.astype(str).str[:4] + np.where(d.is_ipl==1,'-IPL','-T20I'))
    x = d.loc[mask & d[grp].isin(['HIGH','ORD']), ['match_id','innings','block',pid,val,grp]].rename(columns={pid:'pid',val:'v',grp:'g'})
    x['v2'] = x.v**2
    a = x.groupby(['pid','match_id','innings','block','g']).agg(n=('v','size'), s=('v','sum'), ss=('v2','sum')).unstack('g', fill_value=0)
    a.columns = [f'{c}_{g}' for c,g in a.columns]
    a = a.rename(columns={'n_HIGH':'nH','s_HIGH':'sH','ss_HIGH':'ssH','n_ORD':'nO','s_ORD':'sO','ss_ORD':'ssO'}).reset_index()
    return a

def _eb2(O, H, sO2, sH2):
    """bivariate empirical Bayes: borrow strength between a player's ordinary (O) and pressure (H) performance"""
    mu = np.array([np.average(O,weights=1/sO2), np.average(H,weights=1/sH2)])
    SOO = max(np.var(O,ddof=1)-sO2.mean(), 1e-6); SHH = max(np.var(H,ddof=1)-sH2.mean(), 1e-6)
    lim = 0.98*np.sqrt(SOO*SHH); SOH = float(np.clip(np.cov(O,H,ddof=1)[0,1], -lim, lim))
    Sig = np.array([[SOO,SOH],[SOH,SHH]])
    n = len(O); Sn = np.zeros((n,2,2)); Sn[:,0,0]=sO2; Sn[:,1,1]=sH2
    K = np.einsum('ij,njk->nik', Sig, np.linalg.inv(Sig[None]+Sn))
    x = np.stack([O,H],1)-mu
    m = mu + np.einsum('nij,nj->ni', K, x)
    V = Sig[None] - np.einsum('nij,jk->nik', K, Sig)
    return m, V, Sig, mu

def rate(ps, w=None, min_H=60, min_O=100, kb=20, min_blocks=3, base_w=(1,1,1), n0=None, return_R=False):
    """ps = player-match-innings sufficient stats.  w = optional match multiplicity weights (bootstrap)."""
    p = ps.copy(); p['w'] = 1.0 if w is None else p.match_id.map(w).fillna(0).values
    for c in ['nH','sH','ssH','nO','sO','ssO']: p[c+'_w'] = p[c]*p.w
    agg = p.groupby('pid')[[c+'_w' for c in ['nH','sH','ssH','nO','sO','ssO']]].sum(); agg.columns=['nH','sH','ssH','nO','sO','ssO']
    nz = lambda s: s.replace(0,np.nan)
    varH = (agg.ssH-agg.sH**2/nz(agg.nH)).sum()/(agg.nH.sum()-len(agg)); varO = (agg.ssO-agg.sO**2/nz(agg.nO)).sum()/(agg.nO.sum()-len(agg))
    agg['mH']=agg.sH/nz(agg.nH); agg['mO']=agg.sO/nz(agg.nO)
    pool_H = agg.sH.sum()/agg.nH.sum()                                   # what a typical player does under pressure
    # blocks (season x league): consistency units
    bl = p.groupby(['pid','block'])[['nH_w','sH_w']].sum(); bl = bl[bl.nH_w>=kb]
    bl['hit'] = ((bl.sH_w/bl.nH_w) >= pool_H).astype(float)
    hh = bl.groupby('pid').agg(hits=('hit','sum'), blocks=('hit','size'))
    el = agg[(agg.nH>=min_H)&(agg.nO>=min_O)].join(hh, how='inner'); el = el[el.blocks>=min_blocks].copy()
    # --- ingredient 1+2: Level (true pressure performance) and Uplift (true pressure minus usual) via bivariate EB
    m, V, Sig, mu = _eb2(el.mO.values, el.mH.values, varO/el.nO.values, varH/el.nH.values)
    el['O_s'] = m[:,0]; el['H_s'] = m[:,1]; el['U_raw'] = m[:,1]-m[:,0]          # raw uplift H-O (descriptive only: mostly rewards weak ordinary form)
    beta = Sig[0,1]/Sig[0,0]                                                      # how much pressure performance rises with usual performance
    el['U_s'] = m[:,1] - (mu[1] + beta*(m[:,0]-mu[0]))                            # RESILIENCE: pressure performance beyond what the player's usual level predicts
    vL = V[:,1,1]; vU = V[:,1,1] + beta**2*V[:,0,0] - 2*beta*V[:,0,1]
    # --- ingredient 3: hit-rate across blocks, Beta-shrunk
    pbar = el.hits.sum()/el.blocks.sum()
    if n0 is None:
        ph = el.hits/el.blocks; v_obs = np.average((ph-pbar)**2, weights=el.blocks); v_bin = np.average(pbar*(1-pbar)/el.blocks, weights=el.blocks)
        n0 = float(np.clip(pbar*(1-pbar)/max(v_obs-v_bin,1e-4)-1, 2, 30))
    a = el.hits+n0*pbar; bq = el.blocks-el.hits+n0*(1-pbar)
    el['C_s'] = a/(a+bq); vC = a*bq/((a+bq)**2*(a+bq+1))
    # --- reliability of each ingredient = real-signal share of its variation across players
    R = {}
    for k,col,v in [('L','H_s',vL),('U','U_s',vU),('C','C_s',vC)]:
        vm = el[col].var(); R[k] = float(vm/(vm+np.mean(v)))
    wt = np.array(base_w,float)*np.array([R['L'],R['U'],R['C']]); wt = wt/wt.sum()
    z = lambda s: (s-s.mean())/s.std()
    el['PRS'] = wt[0]*z(el.H_s) + wt[1]*z(el.U_s) + wt[2]*z(el.C_s)
    el = el.sort_values('PRS', ascending=False)
    el.attrs.update(dict(beta=beta, R=R, weights=dict(zip(['Level','Uplift','HitRate'],wt.round(3))), n0=n0, pbar=pbar, pool_H=pool_H, n_eligible=len(el), Sigma=Sig))
    return el

if __name__=='__main__':
    import sys; sys.path.insert(0, str(SCRIPT_DIR))
    from step4_pressure_index import build_base, add_pi
    d = add_pi(build_base())
    d = add_values(d)
    d.to_pickle(str(OUT / 'balls_values.pkl'))
    # --- validation: does HIGH pressure really differ from ORDINARY? (phase mix is identical by construction) ---
    d['is_dot'] = ((d.bat_runs==0)&(d.team_runs==0)&d.is_legal); d['bnd']=d.is_four|d.is_six
    for k,val,lab in [('bat','BV','batting'),('bowl','BoV','bowling')]:
        g = d[d[f'grp_{k}'].isin(['HIGH','ORD']) & d.is_legal]
        t = g.groupby(['innings',f'grp_{k}']).agg(n=('is_legal','size'), runs_per_ball=('bat_runs','mean'), dot_pct=('is_dot','mean'), boundary_pct=('bnd','mean'),
                  dismissal_pct=('wkt','mean'), mean_value=(val,'mean'))
        t[['dot_pct','boundary_pct','dismissal_pct']] *= 100
        print(f'\n=== VALIDATION ({lab} perspective, PI_{k}): HIGH vs ORD, identical phase mix ===\n', t.round(3).to_string())
    print('\nteam adjustments sd (runs/ball): batter-side', d.opp_for_batter.std().round(3), ' bowler-side', d.opp_for_bowler.std().round(3))
    for role in ['bat','bowl']:
        ps = player_ps(d, role); ps.to_pickle(str(OUT / f'ps_{role}.pkl'))
        a = ps.groupby('pid')[['nH','nO']].sum()
        print(f'\n{role}: players={len(a)}  nH quantiles', a.nH.quantile([.5,.75,.9,.95,.99]).round(0).to_dict(), ' players with nH>=60 & nO>=100:', ((a.nH>=60)&(a.nO>=100)).sum())
