# Boat-race outcome feature research

Updated: 2026-07-20

## Evidence incorporated

- BOAT RACE's official guide defines national win rate as performance over all
  venues and local win rate as performance at the current venue. The model uses
  both levels and their difference instead of treating branch membership as a
  substitute for venue performance.
  <https://www.boatrace.jp/owpc/pc/extra/enjoy/guide/level2/l2_01_01_05.html>
- A 2025 SHAP study covering 484,006 races found racer ability to be the dominant
  feature family. It also found that outer-lane racer ability contributes to the
  decision whether lane 1 wins, while motor performance is less influential in
  ordinary races. This motivates separate racer/equipment scores, lane-1 defense,
  outer-threat, and field-balance features.
  <https://www.jstage.jst.go.jp/article/sci/SCI25/0/SCI25_508/_pdf/-char/en>
- A nationwide 24-venue study found only weak overall rank correlation between
  exhibition time and finish, but stronger Top-k usefulness under some venue,
  rain, water, and distance strata. Exhibition time is therefore race-normalized
  and added through rank/Top-2 and context interactions, not as a stand-alone
  universal rule.
  <https://www.jstage.jst.go.jp/article/jsik/35/4/35_2026_009/_article/-char/ja>
- A venue-clustering study reports that venue, weather/water conditions, lane,
  racer-course history, and motor differences should be modeled jointly. The
  implementation uses venue/lane/weather interactions and prior-only rolling
  racer-venue, motor-venue, and boat-venue histories.
  <https://www.msi.co.jp/solution/stuaward/2021/VMS_5.pdf>

## Added candidate features

- Local racer: official branch-to-venue match, local-minus-national performance,
  venue/branch and home/lane interactions, and prior racer-at-venue performance.
- Racer matchup: normalized racer strength, rank in field, gap to lane 1 and the
  best racer, lane-1 strength, and maximum lane 5/6 threat.
- Equipment: motor/boat strength separated from racer skill and discounted when
  the racer field is not balanced.
- Live context: actual exhibition course versus lane, waku-nari flag, normalized
  exhibition Top-1/Top-2, and exhibition interactions with venue, weather,
  distance, wind, wave, and racer strength.

All feature names use the `research_` prefix. `feature_tuning` can remove the
whole group with `--drop-feature-groups research_correlates`, enabling an exact
same-race-set ablation. Race results and same-day later outcomes are never used
to construct a pre-race feature.

## Promotion rule

The candidate is not promoted merely because training finishes. Compare it with
the prior model on the same chronological 365-day test set and the same JPY
10,000/day bankroll policy. Promote only when calibration/ranking does not
materially regress and out-of-sample ROI or loss improves across multiple time
blocks rather than one venue or one month.


## 2026-07-22 market-residual structure probes

- Winner-only residual: 260-race daily walk-forward LogLoss 3.84071 versus the retained global Newton residual 3.83357. The winner residual coefficient converged near zero, so the candidate was rejected.
- Market-entropy-conditioned residual: 260-race daily walk-forward LogLoss 3.83974 versus 3.83357, with Top-5 equal to the market at 33.08%. The entropy interaction did not add stable signal and was rejected.
- T-10 to T-5 outcome momentum and signed disagreement curvature were also rejected on their untouched comparison folds. T-10 to T-5 movement is retained only for closing-price forecasting, where it reduced 2026-07-22 log-odds MAE from 0.17318 to 0.16537 on the same 126 races.
- Rejected probe implementations are not kept on the production import path. Their exact code and tests remain recoverable from Git history.

## 2026-07-23 preregistered market candidate

- Position-specific stagewise probabilities plus the retained two-coefficient Newton market residual scored LogLoss 3.84268 on the 133-race 2026-07-22 fold, versus 3.84354 for listwise plus Newton residual and 3.85637 for stagewise plus grid calibration.
- The 0.00086 difference from the retained residual model is development evidence only. The existing stagewise shadow track is fixed to Newton residual before 2026-07-23 outcomes and that day is the next untouched architecture comparison.
- No wagering or production promotion is allowed from this one-day result; the 30-day, 1,000-race, paired market-confidence, positive-profit, ROI, and fold-stability gates remain unchanged.
