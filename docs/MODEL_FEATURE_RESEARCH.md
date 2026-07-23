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

## 2026-07-22 finish-position/lane residual probe

- A 17-parameter Newton residual added strongly regularized lane 2-6 offsets for each of first, second, and third place. Regularization was selected on the 2026-07-20 to 2026-07-21 forward folds before testing 2026-07-22.
- With listwise probabilities, LogLoss changed from 3.84354 to 3.84256, but 3T5 fell from 30.83% to 29.32%. With stagewise probabilities, LogLoss changed from 3.84268 to 3.84171 and 3T5 again fell to 29.32%.
- The incremental LogLoss differences were about -0.001 for both source models and both 95% intervals crossed zero. Because ranking utility regressed and confidence was absent, the structured residual is rejected and is not added to a production shadow or promotion candidate.
- The generic implementation and exact probe remain in Git so the hypothesis can be retested only after substantially more strict T-5 days accumulate; no coefficients from this development fold are deployed.

## 2026-07-22 full-day source ensemble check

- The earlier intraday ensemble result covered only 113 races, so it was rerun on the same final 133-race fold used by every v14 candidate. Source subset and regularization selection still used only 2026-07-20 and 2026-07-21.
- The selected market, fixed-cutoff listwise, and stagewise ensemble produced LogLoss 3.90027 and 3T5 29.32%. This was worse than both the T-5 market at 3.86070/30.08% and stagewise plus global Newton residual at 3.84268/30.83%.
- The ensemble is rejected as an unstable two-day source-weight fit. It is not registered for 2026-07-23 and will not be reconsidered until enough full-day strict T-5 folds support source-weight stability.

## 2026-07-23 formal T-5 timestamp tolerance

- Formal market evaluation still rejects every snapshot captured after the T-5 decision cutoff, so the change cannot introduce lookahead.
- The freshness ceiling is 65 seconds before the cutoff instead of 60. This is a five-second scheduler-jitter tolerance: the three previously excluded 2026-07-22 races had complete 120-combination snapshots 62, 63, and 63 seconds before the cutoff.
- The collector continues to target a snapshot within 60 seconds and now reserves the process 90 seconds before an imminent T-5 window. The 65-second evaluation ceiling does not relax collection frequency.
- Evaluation version 16 and the scored-cache contract prevent mixing the new tolerance with earlier reports. Daily coverage must still be 100%.


## 2026-07-23 calibration/evaluation population separation

- Evaluation version 17 keeps the formal evaluation population unchanged: a
  scored day is eligible only when every completed race has a pre-cutoff T-5
  snapshot within 65 seconds and every payout is complete.
- Earlier races with valid pre-cutoff T-5 snapshots are now retained for
  calibration and policy selection even when another race on the same earlier
  day is missing a snapshot. Discarding those valid rows reduced calibration
  data without making the later complete-day holdout safer.
- Every fold still fits the calibrator, closing-odds model, and bankroll policy
  exclusively on dates strictly earlier than its complete evaluation day. A
  regression test fixes this temporal boundary and verifies that partial prior
  days can never enter the evaluation metrics.
- Promotion still requires 30 complete evaluation days, at least 1,000
  evaluation races, paired market-confidence gates, positive profit, ROI above
  one, and fold stability. Calibration-only races do not count toward those
  promotion sample gates.


## 2026-07-23 conservative expected closing-odds correction

- The previous price model minimized absolute log error and used
  `exp(E[log closing odds])` directly for bankroll expected value. That quantity
  estimates a conditional median, not `E[closing odds]`, and therefore
  systematically omits the Jensen correction needed for expected payout.
- Evaluation version 18 keeps median log-odds forecasts for price-accuracy model
  selection, but bankroll decisions use a separate expected-odds multiplier.
  The multiplier is the 95% lower confidence bound of per-race mean
  `closing_odds / predicted_median_odds`, estimated only on prior dates and
  bounded to prevent unstable tail extrapolation.
- The correction can create a wager only when the prior-day policy search also
  passes its ticket-count, profitable-day, ROI, and drawdown gates. The 7/22
  result is development evidence because this correction was specified after
  that day; 7/23 or later is required for untouched confirmation.
