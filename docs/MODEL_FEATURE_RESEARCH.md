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

The adapter deliberately requires either absolute external ticket stakes or an
absolute total wager-pool amount plus the 120-way market shares in every
scenario. In the latter case, total face units are allocated by deterministic
largest remainder before adding the system's order. Closing odds or normalized
market shares alone identify relative prices but not absolute pool size, so they
cannot uniquely identify the price impact of a JPY 100 or JPY 10,000 order.
`boatrace_ai.pool_scale_lower_bound` supplies a fail-closed fallback rather than
inventing a nominal pool: it finds the smallest integer 10-yen-face-unit pool
whose statutory settlement reproduces every available decision-time decimal
odd. Missing outcomes are rejected unless the caller explicitly marks them as
unpriced. The resulting amount is a lower bound, not an estimate of the actual
closing pool. Holding it fixed through close deliberately overstates self
impact and can reject otherwise viable bets. Audited wager-type sales remain
the preferred input and replace this fallback when available.

An initial source probe found race-level wager-type sales in venue-published
official record sheets. For example, the 2026-06-25 Tokoname record sheet lists
trifecta sales and winning ticket counts for each race in addition to total
race sales. This establishes that absolute trifecta pool labels exist, but it
does not establish nationwide historical coverage or a realtime endpoint:

- <https://www.boatrace-tokoname.jp/uploads/cdn/pdf/syussou/08_20260629_2.pdf>

`boatrace_ai.joint_policy_ga` connects pre-generated parameter/path draws to
the settlement callback and searches complete 100-yen stake vectors. Version v2
maximizes the lower parameter quantile of scenario-tail expected
`log(terminal wealth / available bankroll)`, subject to the existing
portfolio expected-edge gate. The terminal wealth calculation integrates every
ordinary and refund outcome through the integer gross-payoff callback. This
penalizes ruin and determines both ticket composition and stake fraction;
positive expected edge alone cannot force a full-bankroll allocation.
Unevaluable vectors receive a finite search-disqualification score, while the
no-bet vector retains value zero. Every GA candidate is repriced as a whole;
the selected vector alone is rerun with ticket-removal marginal diagnostics.
The module is diagnostic-only and has no path to the wagering API.

Rejected GA vectors retain a continuous feasibility score based on their
shortfall from the portfolio-edge and expected-log-growth gates. This score is
used only to guide evolution; it cannot authorize a purchase. Feasible vectors
always outrank rejected vectors, and when no feasible vector exists the final
selection is the explicit zero-stake vector. This avoids the flat-penalty
failure mode where every rejected initial candidate had identical fitness and
subsequent evolution was effectively random.

The pool lower bound can be attached to every generated path while retaining
that path's final 120-way market shares. The settlement adapter then allocates
the absolute lower-bound scale across those generated shares and reprices each
complete GA vector. The attachment records its method, decision-time as-of
label and deterministic allocation audit hash. It refuses to overwrite an
exact pool amount or exact ticket stakes already present in a scenario.

`boatrace_ai.joint_bankroll_evaluation` is the chronological operating
evaluation that connects these components. For each evaluation day it refits
outer parameter draws from strictly earlier complete days, generates paired
terminal-probability and closing-share paths, attaches the decision-time pool
lower bound, and asks the complete-vector GA for zero or more 100-yen units.
The realized ledger uses only the official final payout. Each day opens with
JPY 10,000; stake is removed at purchase, and a return is unavailable to later
purchases until ten minutes after that race's recorded odds deadline. This
prevents same-day bankroll leakage from a result that was not yet knowable.

Promotion requires at least 30 complete operating days, 1,000 ticket orders,
positive aggregate profit, a complete-day bootstrap ROI lower bound above
1.0, maximum within-day drawdown no greater than half the opening bankroll,
and generated LogLoss no worse than the decision model. Current evidence has
fewer than 30 collected market days, so every result is provisional even if
its point ROI is above one. The evaluator never submits a wager.

`boatrace_ai.joint_parameter_uncertainty` supplies genuine outer parameter
draws rather than duplicating paths from one fit. It resamples complete strictly
prior race days with replacement, refits the terminal/market residual model for
each draw, and then generates an independent inner path set from every refit.
Repeated blocks carry an internal resample key while retaining the original OOF
teacher race ID and prediction hash. Each draw manifest hashes the sampled day
sequence, teacher prediction/schema hashes, fit options and calibration seed;
the manifest and draw index are copied into every generated path. Any current
or future-day training observation is rejected before fitting.

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
`joint_scenario_model`; generated paths are now connected to integer settlement
and complete-vector policy GA. A chronological provisional bankroll evaluator
is connected, but exact nationwide pool labels, a 30-day sealed bankroll
evaluation and production promotion remain unfinished. The current
market-residual GA uses market-relative probability metrics and is not
described as a joint-value GA.

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
conditioned on the final market by definition. The generator is connected to
complete-vector policy GA and integer settlement, but its real-data evaluation
still covers only five outer days. Outer parameter uncertainty requires
day-block refits. Promotion remains prohibited until at least 30 complete days,
the sealed bankroll gates pass, and timestamped shadow-operation evidence is
recorded.

The formal purchase value is the empirical `inverted_cdf` outer quantile of the
portfolio-path lower-tail expected edge, `V_buy(b)`. A portfolio is authorized
only when `V_buy(b) > m_buy`; ticket-level values remain diagnostics. The formal
sealed return gate is exclusively `Q0.05(ROI) > 1` under complete operating-day
resampling. `P(ROI > 1)` is displayed only as a diagnostic. Every result stores
a resampling condition ID covering only the quantile method, sample count, block
definition and seed, with independent day-by-venue and consecutive
venue-meeting blocks shown as sensitivity analyses. This ID is not the complete
evaluation identity.

Every new artifact additionally stores an evaluation protocol ID: the SHA-256
of canonical model and terminal-teacher identities, exact scored-cache content,
every per-race evaluation time `t`, venue, trifecta wager type, decision-time
popularity band, outcome schema, joint-distribution controls, GA controls,
purchase and bankroll rules, integer settlement rules, and the resampling
condition ID. Here `t` is the `captured_at` timestamp of the odds snapshot and
is the upper boundary of information available to the purchase decision
`F_t`; deadline or closing information after `t` is not part of that decision.
Changing `t` or any purchase rule changes the evaluation protocol ID even when
the resampling condition ID remains unchanged. Legacy artifacts explicitly
report both values as unrecorded instead of inferring an identity.

Evaluation artifact v4 also persists the observed probability-multiplier
covariance, the edge overstatement from the independent approximation, inner
scenario ESS, outer draw count, portfolio-path aggregation, complete-vector
repricing and integer settlement capabilities. The public model report embeds a
compact server-rendered table and JSON payload, so these values and explicit
legacy `not recorded` states are auditable without executing browser JavaScript.

Current maturity is stage 3 of 6: late research. Stage 4 requires a fully frozen
model, calibration, genome and purchase threshold over at least 30 previously
unseen consecutive days, positive formal purchase value, sealed ROI lower bound
above one, profit after removing the largest hit, and a complete timestamped
shadow journal. Stage 5 additionally requires low fixed-stake live operation
without changing the model or purchase rules during the evidence period.

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
each outer day. Job 11785 was diagnostic only; later jobs connected its retained
successor to settlement and policy evaluation, never to automated purchasing.

### 2026-08-03 provisional joint bankroll correction

Job 11845 applied the first connected chronological policy to the same five
days and 713 races. Generated trifecta LogLoss improved by 0.003128, but the
policy staked JPY 63,200 for JPY 13,250 return: ROI 0.20965 and profit
JPY -49,950. Only one race returned money, all return depended on that hit,
the complete-day bootstrap ROI lower bound was zero, and five of six promotion
gates failed.

The probability result does not validate the purchase policy. The v0 GA
maximized conservative expected profit in yen, which is conservative about
unit edge but still scales a positive edge toward the stake ceiling. It spent
the opening bankroll in the first one or two opportunities on most days. This
objective is rejected.

Policy v1 instead integrates each terminal settlement state and maximizes the
lower parameter quantile of scenario-tail expected log bankroll growth. The
expected-edge gate remains mandatory, while the growth objective selects the
stake fraction and penalizes terminal ruin. Matched-window job 11852 compares
this correction with job 11845. It staked JPY 100 on one ticket, returned zero,
and limited maximum drawdown to JPY 100. The generated probabilities retained
the trifecta LogLoss improvement of 0.003128 and Brier improvement of 0.000076,
while 3T5 regressed by 0.14 percentage points. The growth objective therefore
fixed the catastrophic stake sizing but did not establish a profitable policy.

Inspection found that every rejected vector had the same search penalty, so an
initial population without a feasible vector had no selection gradient. Policy
GA v2 fixes that search-only defect while preserving the exact same hard
authorization gates and zero-stake fallback. Matched-window job 11895 evaluates
the change as joint bankroll evaluation v3. Both results remain provisional
regardless of point ROI because only five complete operating days are available.

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
