# Binned Poisson deviance: a loss for counting-process predictions

Status: research stage (2026-09-26). Answers the user's request to "nail down" the loss from `tvc-research.md` §3, and supersedes that section where they differ. Numbers come from `bench/tvc_deviance_check.py`; the output is in `docs/scratch/tvc_deviance_check.txt`. No design decisions yet.
Tags as in `tvc-research.md`: [S] source fetched/searched, [S: title] title/abstract only, [K] background or own derivation (proof sketch given), [R] repository, [N] numerical check in this note.

---

## 0. Summary
1. **Use the piecewise-exponential log score:** `S = Σ_{r,m} [N_rm log λ̃_m(x_r) − λ̃_m(x_r) e_rm]`, with `λ̃_m(x) = ΔΛ̂_m(x) / |W_m|` the forest's window-average hazard. The deviance is `−2S` plus a constant that depends only on the data.
2. **It is proper in a precise, qualified sense.** It is the exact continuous-time log-likelihood of the piecewise-constant predictor `λ̃`, and the log-likelihood is a strictly proper score for intensities. Among piecewise-constant predictors, its unique maximiser is the **exposure-weighted window hazard `c*_m(x)`**. The truth's uniform window average can miss `c*` by at most the hazard's oscillation within the window (§2).
3. **The original proposal is improper.** `N log Ê − Ê`, with `Ê` integrated over the row's own exposure in the window, is proved and checked to be improper: a constant-hazard prediction beats the truth [K, N] (§2.3). The fix is to take the log of a *rate* that does not depend on when the row's exposure ends.
4. **Zero-rate event cells** (`λ̃ = 0`, `N = 1` → `−∞`) are real at fine windows: 2.5–23 % of events at M = 32–64. Fix: score the mixture `(1 − α) λ̃ + α λ₀` with the training-set null rate `λ₀`. That is a legitimate predictor, so the score stays proper for it, unlike a floor (§3).
5. **The window count M acts as a bandwidth.** The absolute score degrades quickly as M grows, even with a truly time-constant hazard, but **importance differences are stable for M = 2–16** [N] (§4).
6. It decomposes exactly by window, cause and subject, which gives time-resolved importance, cause-specific importance and subject-level inference for free (§5). It needs no IPCW (§6).

## 1. Setup
- **Row data.** Row `r` has covariates `x_r` on `(s_r, u_r]` (start, stop) and event indicator `δ_r`, with at most one event, at `u_r`. `Y_r(t) = 1{s_r < t ≤ u_r}`.
- **Windows.** `W_m = (w_{m−1}, w_m]`, `m = 1..M`, partition `(0, τ]`, right-closed like the rows [R: rows are `(start, stop]`].
- **Cells.** Per (row, window): the exposure `e_rm = |W_m ∩ (s_r, u_r]|` and the count `N_rm = δ_r 1{u_r ∈ W_m} ∈ {0, 1}`.
- **Prediction.** The fixed-profile cumulative hazard `Λ̂(t | x)`, which is `predict_cumulative_hazard(X, times=edges)` [R]. The window rate is `λ̃_m(x) = [Λ̂(w_m | x) − Λ̂(w_{m−1} | x)] / |W_m|`. Any `aggregate` mode works, because only `Λ̂` is used.
- **Truth.** The intensity of `N_r` is `Y_r(t) λ(t, x_r)`, with `x_r` predictable (§1.2 of `tvc-research.md`).

**Definition (piecewise-exponential log score):**
```
S(λ̃) = Σ_r Σ_m [ N_rm log λ̃_m(x_r) − λ̃_m(x_r) e_rm ]      (higher is better; 0·log 0 = 0)
```
- **Deviance form.** With `μ_rm = λ̃_m(x_r) e_rm` and `N ∈ {0, 1}`, `D = 2 Σ [N log(N/μ) − (N − μ)] = −2 S − 2 Σ N_rm log e_rm − 2 Σ N_rm`. The last two terms depend only on the data, so **`ΔD = −2 ΔS`**. D and S rank predictors identically, and importance can be reported on either scale [K].
- **Link to Poisson regression.** This is the likelihood of the piecewise-exponential model, which equals a Poisson GLM on (row × window) cells with offset `log e_rm` (Holford 1980; Laird & Olivier 1981; asymptotics in Friedman 1982) [S]. PAMMs (Bender, Groll & Scheipl 2018, `pammtools`) use the same data expansion with TVCs [S]. `randomForestRHF`'s empirical risk is the continuous-time version [S: arXiv 2608.21597].

## 2. Propriety

### 2.1 The continuous log score is strictly proper for intensities
For any deterministic predictor `λ̂(t, x)` (fitted on other data), the continuous log score is `ℓ(λ̂) = Σ_r [∫ log λ̂(t, x_r) dN_r(t) − ∫ Y_r(t) λ̂(t, x_r) dt]`. `log λ̂(t, x_r)` is predictable (deterministic given `x_r`), so the compensator gives
```
E ℓ(λ̂) = E Σ_r ∫ Y_r(t) [ λ(t, x_r) log λ̂(t, x_r) − λ̂(t, x_r) ] dt .
```
Pointwise, `c ↦ λ log c − c` is strictly concave with maximum at `c = λ`. So `E ℓ` is uniquely maximised by `λ̂ = λ` almost everywhere on the exposure support [K]. The expected shortfall is an exposure-weighted Poisson Kullback–Leibler divergence, `E ℓ(λ) − E ℓ(λ̂) = E Σ_r ∫ Y_r [λ log(λ/λ̂) − (λ − λ̂)] dt ≥ 0` [K].
- This mirrors the right-censored log-likelihood (RCLL), shown to be proper for survival distributions by Rindt, Hu, Steinsaltz & Sejdinovic (AISTATS 2022). The same paper proves that the integrated Brier score, survival-CRPS and time-dependent C are **not** proper [S]. Yanagisawa (2023) studies proper scores for survival further [S: title].

### 2.2 The binned score is the log score of a piecewise-constant predictor
`S(λ̃)` equals `ℓ` evaluated at the predictor `λ̂_W(t, x) = λ̃_m(x)` for `t ∈ W_m`, since `∫_{W_m} Y_r λ̃_m dt = λ̃_m e_rm`. Hence:
- **Proper over all intensities.** No predictor beats the truth: `E S(λ̂_W) ≤ E ℓ(λ)`.
- **Strictly proper within the piecewise-constant class.** For each covariate state `x` and window `m`, the expected score is `a log c − c b`, with `a = E Σ_{r: x_r = x} N_rm` and `b = E Σ_{r: x_r = x} e_rm`. It is uniquely maximised at
  ```
  c*_m(x) = a / b = ∫_{W_m} λ(t, x) ρ_x(t) dt / ∫_{W_m} ρ_x(t) dt ,
  ```
  where `ρ_x(t)` is the at-risk density of state `x` at time `t`: an **exposure-weighted** window hazard [K]. [N]: the Monte Carlo maximum is at `c* = 0.693`, as predicted.
- **The discretisation gap.** A forest that estimated `λ` perfectly would report the *uniform* average `λ̄_m(x) = |W_m|⁻¹ ∫_{W_m} λ(t, x) dt`, not `c*`. Both are weighted averages of `λ(·, x)` over `W_m`, so `|c* − λ̄| ≤ osc_{W_m} λ(·, x)`, and the score loss is `≈ b (c* − λ̄)² / (2 c*)`: second order [K]. The gap is 0 when `λ(·, x)` or `ρ_x` is constant on the window.
  - [N]: for a Weibull hazard of shape 2.5 on a single window (hazard from 0 to 2.5, strong exits), `c* = 0.69` against `λ̄ = 1.00`, and the uniform average loses 0.032 per subject. This is a deliberately extreme case.
  - **Practical rule:** windows must be narrow wherever the hazard moves fast. Quantile windows on the training event times do this automatically, because they are dense where events are dense.
- **For importance.** The same gap affects the intact and the permuted predictions alike, so the importance difference is affected only at second order [K].

### 2.3 The original proposal is improper
- **The v0 score.** `tvc-research.md` §3 v0 proposed `Σ [N log Ê − Ê]` with `Ê_rm = Λ̂(min(u_r, w_m) | x) − Λ̂(max(s_r, w_{m−1}) | x)`. Here `Ê` depends on the row's own exit time, which is the event time when `N = 1`. So `log Ê` is not predictable, and the argument of §2.1 fails.
- **Analytic counterexample** [K]. Rescale time so the true hazard is 1 on the window. Let `T ~ Exp(1)` and the window end `B`, and perturb the prediction to `G(s) = s + ε s²`. Then `d/dε E[1{T ≤ B} log G(T) − G(T ∧ B)] |_{ε=0} = −1 + e^{−B}(1 + B) < 0` for all `B > 0`. So the truth is not even a stationary point: a hazard that falls within the window (`ε < 0`) scores better.
- **Monte Carlo** [N]. For a Weibull truth with shape 2.5, v0 ranks predicted shapes 1.0 > 1.5 > 2.0 > 2.5 (the truth) > 3.0. A constant hazard beats the truth, −0.858 vs −1.036.
- Equivalently, v0 = `Σ N log(Ê/e) − Ê` + a data-only term. It scores each row's rate averaged over its *own* exposure, which the event time truncates.

## 3. Zero-rate event cells
- **Why they occur.** Leaves jump only at their in-bag event times [R: `tree.rs`]. A window with no in-bag event in any of the row's leaves gives `λ̃ = 0`. OOB evaluation uses only the ~37 % of trees that are out of bag, which makes zero rates more likely.
- **How often** [N, S3 simulation, held-out rows; 200 trees; the 70-tree row mimics OOB]:

  | M | zero-rate share of events, 200 trees | zero-rate share of events, 70 trees |
  |---|---|---|
  | 4 | 0 | 0 |
  | 8 | 0 | 0 |
  | 16 | 0.07 % | 0.45 % |
  | 32 | 2.5 % | 6.0 % |
  | 64 | 16 % | 23 % |

- **Options:**
  1. *Keep `−∞`* and report the cells. This is honest, but one cell makes every comparison uninformative.
  2. *Floor*: `max(λ̃, ε)`. It is a legitimate predictor, but it makes every rate below `ε` indistinguishable, and `ε` has no natural scale (Codex review) [K].
  3. **Mixture with the null rate:** `λ̃_α = (1 − α) λ̃ + α λ₀,m`, where `λ₀,m` is the **training-set** occurrence/exposure rate in window `m` (the covariate-free piecewise-exponential fit).
     - The mixture is itself a predictor, so §2 applies unchanged: the score is proper *for the mixed predictor*.
     - Mixing preserves the ordering of rates, and it bounds each event cell's log term below by `log(α λ₀,m)`.
     - [N]: at M ≤ 8, α ∈ [0, 0.1] moves the score by < 0.02 per event.
- **Candidate default:** α = 0.01, and always report the zero-rate share of events. A share well above 0 means the windows are too fine for the forest (§4).

## 4. Choosing the windows (M is a bandwidth)
- **The absolute score depends strongly on M.** [N] With a truly time-constant hazard (S3), the forest's score per event is −1.93 at M = 4, −1.96 at M = 8, −2.11 at M = 16 and −3.1 at M = 64 (α = 0.01). The truth stays at −1.77 throughout.
  - Each leaf's window rate rests on few in-bag events, so its variance grows as windows shrink, and the log score penalises low rates heavily.
  - So M sets how much time-smoothing the evaluated predictor gets. **Scores are comparable only at the same M and on the same windows.**
- **Importance differences are stable.** [N] Permutation importance (score drop per event, α = 0.01, 5 replications) at M = 2, 4, 8 and 16:
  - `z`: 0.78, 0.80, 0.85, 0.86;
  - `x0`: 0.08, 0.08, 0.10, 0.10;
  - pure noise: ≈ 0 (|mean| ≤ 0.011);
  - the replicate sd grows with M (noise sd 0.016 → 0.036).
- **Windows:** quantiles of the training event times, with edges `0` and `τ`. Under coarse mode (`ntime`), reuse `coarse_grid_` [R].
- **M:** small. The evidence supports 4–8 for forests with ~400 ids and the default `min_events_leaf=3`. An `"auto"` rule tied to events per leaf (e.g. median in-bag leaf events / M ≥ ~2) is a design question; §8 would check it.
- **Timing resolution conflicts with M.** Time-resolved importance (§5) wants more windows; the score wants fewer. The two resolutions can be separated: score each cell at a coarse M, then *report* over coarser groups of windows. Or smooth `Λ̂` with a kernel before scoring. A kernel-smoothed `λ̂_h` is still a deterministic predictor, so §2.1 applies; it is a design option [K].

## 5. Decompositions and inference
- **By window.** `S = Σ_m S_m` exactly. Time-resolved importance is `ΔS_m` for the same permuted predictions, with no extra prediction pass. Codex noted that per-window deviances sum to the *binned* total, not to a deviance recomputed on merged windows; that is intended.
- **By cause.** For competing risks, `S = Σ_k S_k`, with `S_k` built from cause-`k` counts and `Λ̂_k`. The cause-specific likelihood factorises, so this is the log score of the vector of cause-specific hazards, and `ΔS_k` is the importance for `λ_k` [K; R: `predict_cumulative_hazard` of `CompetingRisksForestTV`]. Importance for `F_k` still needs the CR Brier (`tvc-research.md` §6).
- **By subject.** `S = Σ_i S_i`, summing each id's cells.
  - **On held-out data**, conditional on the fitted forest, ids are independent. So `ΔS = Σ_i ΔS_i` has the paired standard error `sd(ΔS_i) · √n_ids`, including under M2/M3 permutation (averaged over repeats).
  - **Out of bag**, trees are shared across ids, so use Ishwaran & Lu (2019) subsampling / delete-d jackknife [S].
  - Block resampling (S10) uses id-blocks as units [R].
- **Normalisation.**
  - *Per event:* `S / Σ N`.
  - *Share of the model's gain over the null:* `ΔS_j / (S_model − S_null)`, i.e. the fraction of the forest's improvement over the covariate-free piecewise-exponential fit that permuting `j` destroys. It is unit-free and comparable across datasets [K].
- **Interpretation.** By §2.1, the expected importance is the increase in exposure-weighted Poisson KL divergence from the (binned) truth. Time counts in proportion to at-risk exposure: long follow-up and many rows weigh more, and in landmark stacks every landmark copy counts [K].

## 6. Assumptions and scope
- **No IPCW.** Validity needs the compensator identity `E[dN_r | past] = Y_r λ(t, x_r) dt` on the observed data. This is independent censoring and entry given the covariate state, the standard Andersen–Gill condition [K].
  - If censoring depends on history that is not in `x`, the score targets the hazard among the uncensored at risk with state `x`. That is also what the forest estimates, so the two stay aligned [K].
  - Contrast: the IPCW Brier needs a censoring model and is not proper [S: Rindt et al.].
- **Estimators.**
  - `SurvivalForestTV` and `CompetingRisksForestTV` on counting-process rows [R].
  - The landmark models: stacked rows are `(0, stop]` on the horizon clock [R: `make_landmark_data`], so windows live on `w ∈ (0, horizon]`. Outcome overlap between landmarks affects only the variance.
- **What it does not measure.** The score targets the hazard, not `P(T ≤ s + w)` at a horizon. Users who act on horizon risks still want the landmark IBS alongside, and CR `F_k` importance needs the CR Brier.

## 7. Recommendation for the design stage (not a decision)
Adopt the α-mixed piecewise-exponential log score as the **default importance loss for the counting-process estimators**, with:
- quantile windows, small M;
- α = 0.01, and the zero-rate share always reported;
- per-window and per-cause decompositions;
- subject-level paired SEs on held-out data;
- per-event and share-of-gain normalisations.

Keep the landmark IBS as the horizon-risk alternative. Also consider exposing the score as a metric (`rftvc.metrics`), because it doubles as a proper, IPCW-free replacement for C in `oob_score_`/`score` (`tvc-research.md` §1.3, release pass).

## 8. Checks still to run (feed the plan)
1. **Discretisation gap with a time-varying baseline** (e.g. Weibull `λ0(t)` in the S3 generator): score and importance stability vs M.
2. **OOB version:** zero-rate share and importance agreement vs held-out.
3. **`"auto"` M rule:** does a rule tied to leaf events keep the zero-rate share ≈ 0 across n and `min_events_leaf`?
4. **Propriety regression test:** the §2.3 Monte Carlo (v0 improper; piecewise-constant maximised at `c*`) becomes a unit test of the metric.
5. **CR:** the cause-specific decomposition on the S14 generator.
