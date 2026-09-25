# Research Questions: Random Forests for Survival with Time-Varying Covariates

Goal (context only, not a solution): an efficient, scikit-learn-compatible Python
random forest for right-censored survival data with time-varying covariates (TVCs).

## Questions

### i) State of the art — random forests for survival
1. Which RF-based survival models exist today (RSF, conditional inference forests,
   ranger survival, oblique RSF, extremely randomized survival trees, etc.)?
   What split rules, terminal-node estimators, and outputs does each use?
2. Which Python / R implementations are maintained (scikit-survival, aorsf, ranger,
   randomForestSRC, pycox, others)? What are their performance characteristics and
   limitations (e.g. memory, O(n²) splitting)?
3. What newer tree-ensemble families compete with RSF (gradient-boosted survival,
   survival BART, deep-learning hybrids), and how do they compare in benchmarks?

### ii) Time-varying covariates in tree ensembles
4. What published methods extend survival trees/forests to TVCs (e.g. Bou-Hamad
   et al. 2011, Bertolet LTRC trees/forests `LTRCtrees`/`LTRCforests`, Fu & Simonoff
   2017, Wongvibulsin et al. 2020 "RF-SLAM", Yao et al. 2022 LTRC forests,
   landmarking + RSF, discrete-time/piecewise-exponential Poisson forests)?
5. How does each represent data: counting-process (start, stop, event] rows,
   person-period expansion, landmark snapshots, summary features?
6. What practical challenges arise: row dependence within subject, bootstrapping
   by subject vs by row, left truncation in nodes, prediction requiring a future
   covariate path, computational cost of data expansion?

### iii) Statistical implications
7. What are the known biases/validity issues of each approach: internal vs external
   covariates, conditional-hazard vs marginal-survival interpretation, pseudo-subject
   treatment, informative observation times, look-ahead bias, calibration?
8. How are these models evaluated (time-dependent C-index, Brier score / IBS,
   dynamic prediction AUC) when covariates change over time?
