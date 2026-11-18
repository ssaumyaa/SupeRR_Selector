# SupeRR Selector Season 2: Most Reliable Players Under Pressure

A data-driven method for ranking the IPL and T20I batters and bowlers who perform best in high-pressure moments, measured against how they normally play. Built for Round 1 of Rajasthan Royals' SupeRR Selector Season 2 challenge.

## Results

| Rank | Batter | Bowler |
| --- | --- | --- |
| 1 | Tim David | Wanindu Hasaranga |
| 2 | Suryakumar Yadav | Josh Hazlewood |
| 3 | Rajat Patidar | Sunil Narine |
| 4 | Sai Sudharsan | Harshit Rana |
| 5 | Shreyas Iyer | Rashid Khan |
| 6 | Heinrich Klaasen | Varun Chakravarthy |
| 7 | Mitchell Marsh | Jasprit Bumrah |
| 8 | Glenn Phillips | Kagiso Rabada |
| 9 | Daryl Mitchell | Kuldeep Yadav |
| 10 | Dinesh Karthik | Noor Ahmad |

Based on 769 matches (352 IPL, 417 T20I) and 183,736 deliveries from 2022 onwards.

## Approach

1. **Label every ball as high pressure, ordinary, or neither.** Before each delivery, three questions are asked from the point of view of the player under pressure:
   - How much can this ball change who wins? This is measured by *leverage*, the expected swing in win probability, using a logistic-regression win-probability model.
   - Has the batting side just been squeezed (wickets and dots in the last 12 balls), or, for bowlers, hit around (runs conceded in the last 12 balls)?
   - Is the batting side behind or ahead of where a typical team would be from this position?

   The answers are ranked against balls at the same stage of the same innings and averaged into a Pressure Index. The top 20% of balls are *high pressure* and the bottom 50% are *ordinary*. Balls where the result is effectively decided (win probability above 95% for either side) are excluded.

2. **Score each ball in runs.** A ball's value is what the player achieved minus what an average player achieves on a comparable ball (same league, innings, over and batter's balls faced), adjusted for opposition strength. Wickets are converted into the runs they cost from that exact match position.

3. **Rate each player on three ingredients:**
   - **Pressure level:** average ball value on high-pressure balls.
   - **Resilience:** pressure level beyond what the player's ordinary form predicts.
   - **Consistency:** share of seasons in which the player beat the all-player average under pressure.

   Small samples are shrunk toward the average with empirical Bayes, and each ingredient is weighted by its measured reliability.

4. **Average over 15 definitions of pressure.** These are five weightings of the three questions crossed with three high-pressure cut-offs (top 15%, 20% and 30%). The final ranking is the average score across all 15, so it does not hinge on one arbitrary choice.

Robustness is checked by changing one design rule at a time, by a 200-sample bootstrap over matches, and by split-half reliability.

## Repository structure

```
SupeRR_Selector/
├── data/                         # competition CSVs (not included, see below)
├── outputs/                      # created by the scripts
├── scripts/
│   ├── build_state.py            # cleaning + match state before/after every ball
│   ├── step2_winprob_leverage.py # win-probability models + leverage
│   ├── step2b_try_the_model.py   # optional: query the chase win-probability model
│   ├── step3_pressure_signals.py # tests candidate pressure signals
│   ├── step4_pressure_index.py   # expected remaining runs, strain, Pressure Index
│   ├── step5_ratings.py          # per-ball values + player rating functions
│   ├── step6_robustness.py       # optional: single-definition sensitivity check
│   ├── step7_ensemble.py         # scores players under 15 pressure definitions
│   ├── step8_stability.py        # design-choice checks, split-half, bootstrap
│   ├── step9_final_tables.py     # merges rankings and stability
│   └── run_all.py                # runs the full pipeline in order
├── requirements.txt
└── README.md
```

## Data

The data was supplied by the competition and is not included in this repository. To run the pipeline, place these six files in `data/`:

`ball_by_ball.csv`, `matches.csv`, `competitions.csv`, `teams.csv`, `playing_eleven.csv`, `competition_team_player_mapping.csv`

## How to run

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py          # full run, about 10–15 minutes
python scripts/run_all.py 30       # quicker run with 30 bootstrap samples
```

The final rankings and evidence are written to `outputs/final_rankings.csv`. All random steps use fixed seeds, so reruns give the same results.

## Data cleaning

| Step | Matches left | Removed |
| --- | --- | --- |
| All matches supplied | 883 | |
| IPL and full T20Is only (A-team events dropped) | 853 | 30 |
| Rain-affected matches removed (revised target or overs) | 819 | 34 |
| Matches without a winner removed | 788 | 31 |
| Super-over ties and incomplete chases removed | 769 | 19 |

## Limitations

- With a few hundred high-pressure balls per player, real pressure skill is small next to luck. Split-half agreement is weak (0.10 for batters, 0.22 for bowlers), so places 7–10 on each list are close calls.
- Rain-affected matches, venue and pitch conditions, batter-against-bowler match-ups and batting role are not modelled.
