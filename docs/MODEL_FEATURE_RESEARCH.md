# Boat-race outcome feature research

Updated: 2026-08-03

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

## 2026-08-02 market-difference and two-stage genetic search

The 2025 LightGBM study is a useful prediction baseline, not evidence of a
profitable policy. Its 484,006-race experiment reports win-bet returns of
93.4%/95.1% and trifecta returns of 83.5%-85.8%; its non-binary models use a
7:3 data split rather than a documented rolling walk-forward protocol. The
study also predicts finish positions separately. Production evaluation must
therefore retain chronological folds and a jointly normalized 120-order
distribution.

Official guidance states that at least 75% of non-refunded sales are distributed
to winning tickets. Define terminology precisely:

- Raw break-even probability is 1 / observed_decimal_odds.
- Market pool share is the normalized reciprocal-odds distribution.
- Under the idealized payout relation odds = payout_rate / pool_share,
  break-even probability is pool_share / payout_rate.

The approximately 33% relative hurdle at a 75% payout rate applies when the
20% number means pool share. It must not be applied a second time when 20%
already means raw 1 / odds. Rounding, refunds, dead heats, late money, and
the system's own order impact require separate execution adjustments.

This ratio is a theoretical special case, not the production betting rule.
For ticket i, use its own decision-time information set F_t and model the final
decimal payout D_final as uncertain:

    conditional_EV_i = E[1{outcome = i} * D_i_final | F_t] - 1

Probability and final payout are not assumed independent. New information that
raises the estimated outcome probability can attract late money and lower the
payout at the same time. Closing-price scenarios must therefore preserve their
joint dependence by decision offset, venue, popularity band, and ticket.
Factoring this expectation into p_i times expected payout is valid only under
the corresponding conditional-independence assumption.

When the joint distribution is not estimable with adequate support, use this
conservative sufficient gate rather than treating it as an EV identity:

    p_i_LCB * D_i_final_low > 1 + safety_margin

The safety margin covers payout rounding, late-price error, model drift, and
order impact. The closing-price lower bound is estimated from snapshots that
were available at the same decision offset; official final odds remain a label.

The retained architecture is role-separated:

1. A no-odds fundamental rank model estimates a joint 120-order distribution.
2. A decision-time market model normalizes reciprocal T-5 odds.
3. A residual model learns when the fundamental distribution adds information
   to the market. Selection fitness is market-relative LogLoss, Brier score,
   calibration slope/intercept, and 3T5 stability on strict prior-day folds.
4. A closing-price model predicts the distribution of executable odds from
   predecision snapshots. Final odds are a label, never a decision feature.
5. A separate policy consumes frozen, calibrated probabilities and conservative
   closing-price estimates. It may bet only when a lower-confidence probability
   yields positive expected value after execution stress.

Do not evolve prediction and bankroll genes under one realized-return fitness.
The 2026-08-02 V34 GA evaluated 48 unique allocation candidates and converged
to near-zero exposure; aggressive candidates suffered severe daily losses.
This is evidence that a joint return teacher has a degenerate no-bet optimum,
not evidence that the fundamental or market-residual probability models lack
signal.

Use two genetic stages:

- Prediction GA: fundamental blend, residual regularization, calibration family,
  and feature-group switches; fitness uses paired market-relative probability
  metrics and worst-fold penalties.
- Policy GA: EV lower-confidence quantile, closing-odds quantile, fractional
  Kelly coefficient, race exposure, and daily exposure; fitness uses log
  bankroll growth, worst-day loss, drawdown, activity, and odds stress.

Hyperparameter discovery uses only selection folds. The final candidate is
frozen before one untouched period and is applied to that period once. GA
fitness, candidate rules, feature switches, thresholds, or bet types cannot be
reselected from the sealed result. Confidence intervals use complete race-day
blocks, with nested day-by-venue attribution where sample size permits; tickets
from the same race are never resampled independently. Promotion still requires
at least 30 complete days, 1,000 races, 300 tickets, 20 hits, paired probability
confidence, positive profit, ROI above one under 5% odds stress, and robustness
after removing the largest hit. Profit concentration by period and venue,
probability calibration, and maximum drawdown must also pass. A positive ROI
based on hundreds of high-variance tickets is research evidence only.

Primary references:

- <https://doi.org/10.11509/sci.SCI25.0_508>
- <https://www.boatrace.jp/owsp/sp/extra/enjoy/guide/jiten/26/y_213.html>
- <https://eprints.soton.ac.uk/51684/>


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
  that day. Because v18 was fixed during 7/23 racing, 7/24 is the first full day eligible for untouched confirmation.


## 2026-07-23 formal release boundary

- Evaluation version 19 sets `2026-07-24` as the first promotion-eligible
  complete day for the expected-closing-odds policy. The boundary is part of
  the evaluation output and cannot be inferred from file timestamps later.
- Clean 7/22 and 7/23 data remain usable for calibration and development
  diagnostics, but are listed as pre-registration dates and never count toward
  formal evaluation races, days, confidence intervals, ROI, or profit gates.
- This intentionally resets formal policy evidence to zero. It prevents a
  mathematically plausible correction designed after a result from appearing
  as untouched production evidence.

## Joint decision-time value contract

The production purchase target is not a fixed probability ratio and is not the
product of two independently forecast marginal means.  At decision time `t`,
the value of ticket `i` is

```text
V_i(t) = E[Y_i * D_i,T(b) | F_t] - 1
       = E[pi_i,T * D_i,T(b) | F_t] - 1.
```

`pi_T` and the final market are generated from the same future path.  A path
contains post-decision information updates, market inflow and reaction, and a
draw from model or latent-state uncertainty.  For trifecta, each path emits one
120-element outcome simplex and one 120-element market-share simplex.  Final
popularity is an output of the path, never a decision-time feature.  The system
must not replace `E[pi * D]` with `E[pi] * E[D]`; the omitted covariance can be
material and its sign is estimated rather than assumed.

Settlement is a function of the complete proposed bet vector. The payoff
engine returns integer-yen gross receipts for each ticket and terminal state,
including ordinary trifecta outcomes, cancellations and refunds. It, rather
than the probability model, owns self-impact, rounding and special payouts.
The evaluator deliberately has no approximate payout default. The purchase
gate applies only to the complete fixed bet vector; per-ticket values are
diagnostics. Marginal ticket contribution reruns settlement after removing the
ticket because self-impact changes with the vector.

`boatrace_ai.parimutuel_settlement` implements that boundary with integer yen
and 10-yen face units. It follows the statutory pool formula and minimum
face-value return, supports full and ticket-specific refund terminal states,
and adds the complete proposed bet vector to both total sales and winning-ticket
sales before computing receipts. The legal basis is Article 15 of the Motorboat
Racing Act and Article 28 / Appendix 2 of its enforcement regulation:

- <https://laws.e-gov.go.jp/law/326AC1000000242>
- <https://laws.e-gov.go.jp/document?lawid=326M50000800059>

The adapter deliberately requires absolute external ticket stakes in every
scenario. Closing odds or normalized market shares identify relative prices but
not absolute pool size, so they cannot identify the price impact of a JPY 100
or JPY 10,000 order. No nominal pool size is invented. Until an audited pool
size source or conservative pool-size model is attached, the production
purchase gate is unavailable even when probability and closing-share scenarios
exist.

`boatrace_ai.joint_policy_ga` connects pre-generated parameter/path draws to
the settlement callback and searches complete 100-yen stake vectors. Fitness is
the portfolio lower-quantile value above the fixed safety margin multiplied by
stake, not a sum of independently gated tickets. Unevaluable vectors receive a
finite search-disqualification score, while the no-bet vector retains value
zero. Every GA candidate is repriced as a whole; the selected vector alone is
rerun with ticket-removal marginal diagnostics. Version v0 is diagnostic-only
and has no path to the wagering API.

Parameter uncertainty and future-path uncertainty are kept separate.  For each
outer parameter/refit draw `r`, future paths `s` are integrated to produce an
expected edge `mu_i(r)`.  The default purchase gate is a preregistered lower
quantile of `{mu_i(r)}` above the operating margin.  A stricter mode replaces
the inner path mean with the lower-tail expected edge before applying the outer
quantile.  It never applies a low quantile directly to single-ticket realized
Bernoulli returns, which would usually reduce to `-1` and is not a useful value
test.

Venue, decision horizon, decision-time popularity, wager type and race state
condition the joint model.  Sparse interactions use partial pooling toward the
global effect.  The first implementation may fit calibrated marginal update
models and join their residuals with a conditional copula.  A shared-latent
state-space or conditional generative model is a later challenger, not a
prerequisite for preserving the joint scenario identity.

Selection resampling and sealed evaluation are distinct:

- Training uncertainty resamples training days and reruns calibration, joint
  distribution fitting and GA selection in the outer loop.
- Sealed operating uncertainty freezes every model, threshold and genome, then
  resamples complete operating days.
- The primary sealed interval keeps every venue within a sampled day together.
  Day-by-venue resampling is sensitivity analysis because it adds conditional
  independence assumptions.
- Meeting blocks and consecutive-day moving blocks test persistent racer,
  motor and venue effects.

`boatrace_ai.joint_market_value.evaluate_joint_market_value` is explicitly the
`joint_market_value_evaluator_v0`. It evaluates pre-generated joint scenarios,
computes portfolio-path lower-tail expectation before outer aggregation, uses
an `inverted_cdf` empirical outer quantile, reports scenario ESS, and disables
the purchase gate when outer or tail evidence is insufficient. It does not
generate joint scenarios. Shared-state generation, partial pooling, and
closing-market distribution fitting now live in the separate diagnostic
`joint_scenario_model`; connection of generated paths to settlement and policy
GA remains unfinished. The current market-residual GA uses market-relative
probability metrics and is not described as a joint-value GA.

`boatrace_ai.terminal_probability_oof` now supplies the previously missing
diagnostic terminal-probability teacher. For each evaluation day it fits a
regularized log pool of the base probability and official closing market using
strictly earlier dates, then emits a soft 120-outcome `P(Y | F_T)` prediction.
The realized one-hot result is used only as a prior-fold likelihood target and
is never exported as a terminal probability. Fold manifests, model hashes,
prediction hashes, outcome schema and the artifact contract are verified before
the teacher can enter the scenario fitter. Aggregate OOF LogLoss, Brier and 3T5
remain reported against the closing-market identity baseline.

`boatrace_ai.joint_scenario_model` is the first diagnostic shared-path
generator. It fits paired CLR residuals from decision probability to terminal
probability and from decision market share to closing market share. A low-rank
shared latent draw produces both terminal simplexes, while venue, decision
horizon and decision-time popularity effects use hierarchical shrinkage. This
preserves empirical probability-price dependence without claiming that a
single fit represents parameter uncertainty.

Neither artifact is deployment eligible yet. The terminal teacher is partly
conditioned on the final market by definition. The generator has passed an
initial real-data chronological diagnostic, but only across five outer days,
and its scenarios are not connected to the purchase GA or settlement evaluator.
Outer parameter uncertainty requires day-block refits. Promotion remains
prohibited until at least 30 complete days and the sealed bankroll gates pass.

### 2026-08-03 strict joint walk-forward diagnostic

DB evaluation job 11785 used the 2,034-race scored cache through 2026-08-02.
It retained 1,896 races with official closing odds; 137 excluded rows were
unfinished 2026-08-02 closes and one 2026-07-28 row was a true missing close.
No closing value was imputed.

The terminal teacher trained on at least five strictly earlier days and scored
1,181 races from 2026-07-25 through 2026-08-01. Relative to closing-market
identity, LogLoss improved by 0.00769 and Brier by 0.000269; 3T5 was unchanged
at 37.93%. Seven of eight daily LogLoss deltas were negative, but this remains
short of the 30-day promotion evidence requirement.

The joint generator then used an additional prior-day refit boundary and
evaluated 713 races from 2026-07-28 through 2026-08-01. Closing-market forecasts
improved cross entropy by 0.01388 and total-variation error by 0.01657, with an
improvement on every evaluated day. The generated outcome mean improved
LogLoss by only 0.00144 against the decision model, while Brier regressed by
0.000023 and 3T5 regressed by 0.42 percentage points. Its generated paired
residual inner product, 0.03802, was close to the observed 0.03823.

The result supports retaining the shared market-path component but rejects a
full-strength probability residual for deployment. Probability and market
residual strengths must be selected separately on inner prior-day folds before
each outer day. Job 11785 is diagnostic only and remains disconnected from
settlement, policy GA and automated purchasing.

### 2026-08-03 residual and dependence calibration comparison

Jobs 11799, 11805, 11806 and 11808 reused the exact job-11785 population: 713
races on five outer days. This is a controlled diagnostic comparison, not a
sealed promotion test.

| Job | Residual treatment | LogLoss delta | Brier delta | 3T5 delta | Generated dependency |
| --- | --- | ---: | ---: | ---: | ---: |
| 11799 | Scale the mean and shared shock together | -0.002944 | -0.000092 | -0.004208 | 0.01074 |
| 11805 | Scale means; keep shared shock at 1.0 | -0.002360 | +0.000017 | +0.001403 | 0.04747 |
| 11806 | Select shock on only the last prior day | -0.002796 | -0.000026 | 0.000000 | 0.04644 |
| 11808 | Match the full-training dependency moment | -0.002878 | -0.000012 | +0.001403 | 0.04353 |

The observed dependency inner product was 0.03823. Scaling the whole residual
destroyed the shared-path dependence, while last-day shock selection was
unstable and overreacted to one day. Job 11808 therefore becomes the retained
diagnostic design: mean probability and market residuals remain role-separated,
and the shared shock is selected by deterministic moment matching over at most
256 date-spread training races. The outer evaluation day is never used for
that selection.

Job 11808 improved LogLoss by 0.002878, Brier by 0.000012 and 3T5 by 0.140
percentage points versus the decision model. It improved closing-market cross
entropy by 0.01595 and total-variation error by 0.01895. Dependency error fell
from 0.00821 in job 11806 to 0.00531, a 35% reduction. The five-day sample is
too short, daily outcome performance is mixed, and no bankroll result exists;
the artifact remains diagnostic-only and cannot be promoted or purchased from.
