"""
STEP 1 - Clean ball-by-ball data and build the match-state table.
Output: /home/claude/superr/balls_state.pkl  (one row per delivery, with state BEFORE and AFTER the ball)
"""
import pandas as pd, numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
D = PROJECT_DIR/ "data"
b = pd.read_csv(D / 'ball_by_ball.csv')
m = pd.read_csv(D / 'matches.csv')
c = pd.read_csv(D / 'competitions.csv')

# ---------- 1. eligibility filters ----------
m = m.merge(c[['comp_id','comp_name','match_format','season']], on='comp_id')
m['is_ipl'] = m.comp_name.str.startswith('IPL').astype(int)
# only IPL + full T20Is (drop 'Emerging Teams Asia Cup' and 'Asia Cup Rising Stars' = A-team events)
keep_m = m[(m.is_ipl==1) | (m.match_format=='T20I')]
# drop rain-affected (DLS) games: target / overs changed, so required-rate logic breaks
dls = keep_m.revised_target.notna() | keep_m.first_inn_revised_overs.notna() | keep_m.second_inn_revised_overs.notna()
keep_m = keep_m[~dls]
# need a winner (drops No Result / Tie / Abandoned)
keep_m = keep_m[keep_m.win_team_id.notna() & (keep_m.win_team_id>0)]
print('matches kept:', len(keep_m), ' IPL:', keep_m.is_ipl.sum(), ' T20I:', (1-keep_m.is_ipl).sum())

b = b[b.innings.isin([1,2]) & b.match_id.isin(keep_m.match_id)].copy()   # drops super overs
b = b.merge(keep_m[['match_id','comp_id','comp_name','match_date','stage','knock_out','win_team_id','is_ipl','season']], on='match_id')
b = b.sort_values(['match_id','innings','innings_ball_num']).reset_index(drop=True)

# ---------- 2. ball-level quantities ----------
nb = b.extras_type.isin(['Overstep','other'])          # no-balls (extra delivery, bat can score)
wd = b.extras_type.eq('wide')
b['is_legal'] = ~(nb | wd | b.extras_type.eq('Penalty'))
b['team_runs'] = b.runs + b.extras1_runs + b.extras_2_runs          # validated vs match results (99.5% exact)
b['bat_runs']  = b.runs                                              # runs off the bat
b['faced']     = ~wd                                                 # batter faces everything except wides
b['bowler_runs'] = b.runs + np.where(wd|nb, b.extras1_runs, 0)       # byes/leg-byes are not charged to bowler
b['wkt'] = b.is_wicket.astype(int)
# wicket credited to bowler (not run-outs / retired / obstructing)
b['bowler_wkt'] = (b.is_wicket & b.dismissal_type.isin(['Caught','Bowled','LBW','Stump','Hit Wicket'])).astype(int)

g = b.groupby(['match_id','innings'])
b['legal_after'] = g.is_legal.cumsum()
b['runs_after']  = g.team_runs.cumsum()
b['wkts_after']  = g.wkt.cumsum()
b['legal_before'] = b.legal_after - b.is_legal.astype(int)
b['runs_before']  = b.runs_after - b.team_runs
b['wkts_before']  = b.wkts_after - b.wkt
b['balls_left_before'] = 120 - b.legal_before
b['balls_left_after']  = 120 - b.legal_after

inn1_tot = b[b.innings==1].groupby('match_id').team_runs.sum().rename('inn1_total')
b = b.merge(inn1_tot, on='match_id')
b['target'] = np.where(b.innings==2, b.inn1_total+1, np.nan)
b['bat_team_won'] = (b.batting_team_id==b.win_team_id).astype(int)

# keep only matches that have both innings
ok = b.groupby('match_id').innings.nunique()
b = b[b.match_id.isin(ok[ok==2].index)].reset_index(drop=True)

# drop ties (decided by super over) and chases whose ball-by-ball ends before a real finish (incomplete data)
last = b[b.innings==2].groupby('match_id').tail(1).set_index('match_id')
finished = (last.runs_after>=last.target) | (last.wkts_after>=10) | (last.legal_after>=120)
tied = (last.runs_after==last.target-1)
bad = last.index[(~finished) | tied]
print('dropped ties / incomplete chases:', len(bad))
b = b[~b.match_id.isin(bad)].reset_index(drop=True)


OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
b.to_pickle(OUTPUT_DIR / "balls_state.pkl")
print(b.shape, b.match_id.nunique(), 'matches')
print(b.groupby('is_ipl').match_id.nunique())
print('batting-first win %:', b[(b.innings==1)].groupby('match_id').bat_team_won.first().mean().round(3))
