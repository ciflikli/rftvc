# Research: n-sweep beyond Slice 1's two points (findings.md row 5)

Answers to `docs/plans/n-sweep-gate-questions.md`, from local pilot runs (`.venv` + cargo `PATH`).

## Q1: what design.md's Approach 3 actually proposed

`docs/plans/design.md`'s Approach 3 representative shape (`run(n_values=(200, 1000, 5000),
n_estimators=500, seed=0)`) is a genuine 3-point sweep — not vague. But its own trade-off analysis
explicitly assumed the sweep "must live entirely in `bench/` (manual-run), not `slow`-marked pytest,
per research's finding that even single-n R=10-20 sims are already `slow`-tier" — a conclusion from
before Slices 5-9 established that reduced-scale, verified versions of "already slow" rules *can*
run in the default tier. That assumption is worth re-testing, not assuming still holds.

## Q2/Q3: does a 3+ point monotonic-decrease claim hold at reduced scale?

**It does not, cleanly.** Tested `(100,500,2000)` and `(200,1000,3000)` at `n_estimators=50-150`,
R=8-15, across 3 seed batches (0, 10000, 20010): the **first** adjacent pair (small→mid n) shows a
robust, significant decrease every time (one-sided 95% lower bound > 0 in all batches, e.g.
0.007-0.052). The **second** adjacent pair (mid→large n) is **never significant** at this scale —
lower bound is negative in every batch tested (-0.010 to -0.026), and the point estimate itself
sometimes increases slightly. This isn't noise from too few reps at the *first* jump; it's a real
plateau: most of the oracle-to-model gap closes early, and the marginal benefit of more data
shrinks fast, so a 3-point sweep at reduced scale cannot honestly claim "decreases at every step."

## Q4: does the existing endpoint claim (200 vs 5000) even hold robustly?

This is the more important finding. The existing `slow`-tier gate (`n_values=(200,5000),
n_estimators=200, R=10, seed0=0` — its only ever-run configuration) was **never checked against any
other seed range**. Testing it at an out-of-band seed batch:

| seed0 | R | lower bound (200 vs 5000) | runtime |
|---|---|---|---|
| 0 (the gate's own) | 10 | 0.027 to 0.070 (across two runs) | ~16-17s |
| 10000 | 10 | 0.024 | ~17s |
| **20010** | **10** | **-0.015 (fails)** | ~17s |
| 20010 | **20** | **0.014 (passes, thin margin)** | ~30s |
| 0 | 20 | 0.070 (comfortable) | ~30s |

**The existing R=10 rule is not robust across seed batches** — it fails outright at seed0=20010.
R=20 restores a pass at that batch, but the margin is thin (0.014) compared to R=20 at the gate's
own seeds (0.070). This mirrors the S19 history scenario's lesson: the originally-declared rep
count is sometimes not enough for genuine robustness, and this is only found by testing seeds the
rule wasn't tuned on.

## Q5: cost

This DGP/model combination is the most expensive of any gate closed this session: R=10 at
`n_estimators=200, n=5000` already takes ~16-17s (vs. 0.4-9s for every Slice 5-9 gate). Getting a
robust version (R=20) costs ~30s for the 2-point endpoint claim alone — before adding a third point
(which would roughly scale runtime by another ~50%, since `replicate` fits one model per n-value in
`n_values`). This is a real, load-bearing cost/rigor trade-off, not a "just verify and it'll be
fine" situation like the other rows closed this session.

## Bottom line

1. A genuine 3+ point "decreases at every step" sweep is **not achievable honestly** at any scale
   tested — the second half of the range plateaus. A 3-point sweep could report the middle point
   as a diagnostic (not gated) while only formally asserting the outer-pair claim, matching the
   S19 history-scenario precedent of "assert only what's robust."
2. Making even the *existing* 2-point endpoint claim reliable enough for the default tier requires
   bumping R from 10 to ~20, which roughly doubles its already-largest-of-any-gate runtime (~17s to
   ~30s). This is a genuine cost question for the user, not a research question: is a single ~30s
   default-tier test (vs. every other gate's 0.4-9s) an acceptable trade to close this blind spot,
   or should the heavier, more-rigorous version stay `slow`-tier while something lighter (e.g., a
   directional-only, cheap smoke check) moves to the default tier instead?
