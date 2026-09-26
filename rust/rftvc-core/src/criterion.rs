/// Summaries of one node (or candidate child) on the parent node's event times.
///
/// A child profile shares the parent's `times`. Log-rank reads only `at_risk`
/// and `events`; the other fields serve likelihood and impurity criteria.
#[derive(Clone, Copy)]
pub struct Profile<'a> {
    pub at_risk: &'a [f64],
    /// Events of any cause.
    pub events: &'a [f64],
    /// Events per cause, cause-major: `cause_events[j * K + k]` for cause `j`
    /// (0-based) at time `k` of `K = at_risk.len()`. For `n_causes == 1` it is `events`.
    pub cause_events: &'a [f64],
    pub n_causes: usize,
    /// Event times (the parent node's), aligned with `at_risk` / `events`.
    pub times: &'a [f64],
    /// Person-time: the sum of `stop - start` over the node's rows.
    pub exposure: f64,
    /// Distinct resampling units. A unit with rows on both sides of a split
    /// counts in both children (the `min_ids_leaf` convention).
    pub n_units: f64,
}

pub trait SplitCriterion: Sync {
    /// Score a candidate split from the left child's and the parent's profiles.
    /// The right child is the parent minus the left, except for its unit
    /// count (units may straddle), which is passed as `n_units_right`.
    fn score(&self, left: &Profile, parent: &Profile, n_units_right: f64) -> f64;

    /// A scorer for many candidates of one parent. Criteria may precompute
    /// parent-only terms; the default defers to `score`.
    fn node_scorer<'a>(&'a self, parent: Profile<'a>) -> Box<dyn NodeScorer + 'a> {
        Box::new(Deferred {
            criterion: self,
            parent,
        })
    }
}

/// Scores left-child profiles against a fixed parent.
pub trait NodeScorer {
    fn score(&self, left: &Profile, n_units_right: f64) -> f64;
}

struct Deferred<'a, C: SplitCriterion + ?Sized> {
    criterion: &'a C,
    parent: Profile<'a>,
}

impl<C: SplitCriterion + ?Sized> NodeScorer for Deferred<'_, C> {
    fn score(&self, left: &Profile, n_units_right: f64) -> f64 {
        self.criterion.score(left, &self.parent, n_units_right)
    }
}

/// Left-truncated right-censored log-rank chi-square statistic.
///
/// Uses the hypergeometric variance, which is valid for the integer
/// unit-weight counts used in v1 (design.md, D6).
pub struct LtrcLogRank;

impl SplitCriterion for LtrcLogRank {
    fn score(&self, l: &Profile, p: &Profile, _n_units_right: f64) -> f64 {
        let (mut num, mut var) = (0.0, 0.0);
        for k in 0..p.at_risk.len() {
            let (y, d, yl) = (p.at_risk[k], p.events[k], l.at_risk[k]);
            if y < 2.0 || d == 0.0 {
                continue;
            }
            let frac = yl / y;
            num += l.events[k] - d * frac;
            var += d * frac * (1.0 - frac) * (y - d) / (y - 1.0);
        }
        if var > 0.0 { num * num / var } else { 0.0 }
    }

    /// Precomputes, per event time with `y >= 2`: `mask`, `e = d / y` and
    /// `c = d (y - d) / (y^2 (y - 1))`, so that
    /// `num = sum mask·d_l - y_l·e` and `var = sum y_l (y - y_l) c`,
    /// the same statistic as `score` with only multiply-adds per candidate.
    fn node_scorer<'a>(&'a self, p: Profile<'a>) -> Box<dyn NodeScorer + 'a> {
        let k = p.at_risk.len();
        let (mut mask, mut e, mut c) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
        for j in 0..k {
            let (y, d) = (p.at_risk[j], p.events[j]);
            if y < 2.0 || d == 0.0 {
                continue;
            }
            mask[j] = 1.0;
            e[j] = d / y;
            c[j] = d * (y - d) / (y * y * (y - 1.0));
        }
        Box::new(LogRankNode {
            y: p.at_risk,
            mask,
            e,
            c,
        })
    }
}

struct LogRankNode<'a> {
    y: &'a [f64],
    mask: Vec<f64>,
    e: Vec<f64>,
    c: Vec<f64>,
}

impl NodeScorer for LogRankNode<'_> {
    fn score(&self, l: &Profile, _n_units_right: f64) -> f64 {
        let (mut num, mut var) = (0.0, 0.0);
        for j in 0..self.y.len() {
            let yl = l.at_risk[j];
            num += self.mask[j] * l.events[j] - yl * self.e[j];
            var += yl * (self.y[j] - yl) * self.c[j];
        }
        if var > 0.0 { num * num / var } else { 0.0 }
    }
}

/// Composite cause-specific LTRC log-rank: `sum_j U_j^2 / V_j` over causes with
/// `V_j > 0`, where `U_j`, `V_j` are the log-rank numerator and hypergeometric
/// variance with "event" = cause `j` (cr-design.md C3). Unlike the all-cause
/// statistic, opposing effects on two causes do not cancel.
pub struct CompositeCauseLogRank;

impl SplitCriterion for CompositeCauseLogRank {
    fn score(&self, l: &Profile, p: &Profile, _n_units_right: f64) -> f64 {
        let k = p.at_risk.len();
        let mut total = 0.0;
        for j in 0..p.n_causes {
            let (pd, ld) = (cause_col(p, j), cause_col(l, j));
            let (mut num, mut var) = (0.0, 0.0);
            for t in 0..k {
                let (y, d, yl) = (p.at_risk[t], pd[t], l.at_risk[t]);
                if y < 2.0 || d == 0.0 {
                    continue;
                }
                let frac = yl / y;
                num += ld[t] - d * frac;
                var += d * frac * (1.0 - frac) * (y - d) / (y - 1.0);
            }
            if var > 0.0 {
                total += num * num / var;
            }
        }
        total
    }

    /// Per cause, the `LogRankNode` terms `mask`, `e`, `c` (cause-major); causes
    /// without an informative event time in the node are skipped.
    fn node_scorer<'a>(&'a self, p: Profile<'a>) -> Box<dyn NodeScorer + 'a> {
        let k = p.at_risk.len();
        let mut causes = Vec::new();
        for j in 0..p.n_causes {
            let terms = log_rank_terms(p.at_risk, cause_col(&p, j));
            if terms.0.iter().any(|&m| m > 0.0) {
                causes.push((j, terms));
            }
        }
        Box::new(CompositeNode {
            y: p.at_risk,
            k,
            causes,
        })
    }
}

/// Cause `j`'s event counts of a profile (a contiguous column).
#[inline]
fn cause_col<'a>(p: &Profile<'a>, j: usize) -> &'a [f64] {
    let k = p.at_risk.len();
    &p.cause_events[j * k..(j + 1) * k]
}

/// `(mask, e, c)` of `LtrcLogRank::node_scorer` for events `d` at risk `y`.
fn log_rank_terms(y: &[f64], d: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let k = y.len();
    let (mut mask, mut e, mut c) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
    for j in 0..k {
        let (y, d) = (y[j], d[j]);
        if y < 2.0 || d == 0.0 {
            continue;
        }
        mask[j] = 1.0;
        e[j] = d / y;
        c[j] = d * (y - d) / (y * y * (y - 1.0));
    }
    (mask, e, c)
}

/// The `LogRankNode` candidate loop on one column of left-child events.
#[inline]
fn log_rank_candidate(
    y: &[f64],
    yl: &[f64],
    dl: &[f64],
    mask: &[f64],
    e: &[f64],
    c: &[f64],
) -> f64 {
    let (mut num, mut var) = (0.0, 0.0);
    for j in 0..y.len() {
        let yl = yl[j];
        num += mask[j] * dl[j] - yl * e[j];
        var += yl * (y[j] - yl) * c[j];
    }
    if var > 0.0 { num * num / var } else { 0.0 }
}

type Terms = (Vec<f64>, Vec<f64>, Vec<f64>);

struct CompositeNode<'a> {
    y: &'a [f64],
    k: usize,
    /// Informative causes and their log-rank terms.
    causes: Vec<(usize, Terms)>,
}

impl NodeScorer for CompositeNode<'_> {
    fn score(&self, l: &Profile, _n_units_right: f64) -> f64 {
        let k = self.k;
        self.causes
            .iter()
            .map(|(j, (mask, e, c))| {
                let dl = &l.cause_events[j * k..(j + 1) * k];
                log_rank_candidate(self.y, l.at_risk, dl, mask, e, c)
            })
            .sum()
    }
}

/// LTRC log-rank on one cause (`cause`, 0-based) with the other causes treated
/// as censoring. It runs the operations of `LtrcLogRank`'s node scorer in the
/// same order on the cause's column; times without a cause-`cause` event add
/// exact zeros, so the score equals `LtrcLogRank` on "cause vs rest" data bit-for-bit.
pub struct SingleCause {
    pub cause: usize,
}

impl SplitCriterion for SingleCause {
    fn score(&self, l: &Profile, p: &Profile, n_units_right: f64) -> f64 {
        self.node_scorer(*p).score(l, n_units_right)
    }

    fn node_scorer<'a>(&'a self, p: Profile<'a>) -> Box<dyn NodeScorer + 'a> {
        assert!(self.cause < p.n_causes, "split cause out of range");
        Box::new(SingleCauseNode {
            y: p.at_risk,
            cause: self.cause,
            terms: log_rank_terms(p.at_risk, cause_col(&p, self.cause)),
        })
    }
}

struct SingleCauseNode<'a> {
    y: &'a [f64],
    cause: usize,
    terms: Terms,
}

impl NodeScorer for SingleCauseNode<'_> {
    fn score(&self, l: &Profile, _n_units_right: f64) -> f64 {
        let (mask, e, c) = &self.terms;
        log_rank_candidate(self.y, l.at_risk, cause_col(l, self.cause), mask, e, c)
    }
}

// --- S14 bake-off challengers (P5: removed unless adopted) ---------------------

/// Per-candidate cause-specific numerators `U_j` and the multivariate
/// hypergeometric covariance `V` (`J x J`, row-major) over times with `y >= 2`.
///
/// At one time, with `f = y_L / y`: `Var(d_Lj) = d_j f(1-f)(y-d_j)/(y-1)` and
/// `Cov(d_Lj, d_Lk) = -d_j d_k f(1-f)/(y-1)`.
fn cause_moments(l: &Profile, p: &Profile) -> (Vec<f64>, Vec<f64>) {
    let (k, nc) = (p.at_risk.len(), p.n_causes);
    let (mut u, mut v) = (vec![0.0; nc], vec![0.0; nc * nc]);
    for t in 0..k {
        let (y, yl) = (p.at_risk[t], l.at_risk[t]);
        if y < 2.0 {
            continue;
        }
        let f = yl / y;
        let w = f * (1.0 - f) / (y - 1.0);
        for a in 0..nc {
            let da = p.cause_events[a * k + t];
            if da == 0.0 {
                continue;
            }
            u[a] += l.cause_events[a * k + t] - da * f;
            v[a * nc + a] += da * (y - da) * w;
            for b in 0..nc {
                let db = p.cause_events[b * k + t];
                if b != a && db != 0.0 {
                    v[a * nc + b] -= da * db * w;
                }
            }
        }
    }
    (u, v)
}

/// `u' V^+ u` with the pseudo-inverse of the symmetric `n x n` matrix `v`
/// (cyclic Jacobi eigendecomposition; eigenvalues below `1e-10 * max` dropped).
pub fn pinv_quadratic_form(u: &[f64], v: &[f64]) -> f64 {
    let n = u.len();
    let mut a = v.to_vec();
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum();
        if off <= 1e-30 {
            break;
        }
        for p in 0..n {
            for r in p + 1..n {
                let apr = a[p * n + r];
                if apr.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[r * n + r] - a[p * n + p]) / (2.0 * apr);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akr) = (a[k * n + p], a[k * n + r]);
                    a[k * n + p] = c * akp - s * akr;
                    a[k * n + r] = s * akp + c * akr;
                }
                for k in 0..n {
                    let (apk, ark) = (a[p * n + k], a[r * n + k]);
                    a[p * n + k] = c * apk - s * ark;
                    a[r * n + k] = s * apk + c * ark;
                }
                for k in 0..n {
                    let (qkp, qkr) = (q[k * n + p], q[k * n + r]);
                    q[k * n + p] = c * qkp - s * qkr;
                    q[k * n + r] = s * qkp + c * qkr;
                }
            }
        }
    }
    let lam_max = (0..n).map(|i| a[i * n + i]).fold(0.0, f64::max);
    if lam_max <= 0.0 {
        return 0.0;
    }
    (0..n)
        .filter(|&i| a[i * n + i] > 1e-10 * lam_max)
        .map(|i| {
            let proj: f64 = (0..n).map(|k| q[k * n + i] * u[k]).sum();
            proj * proj / a[i * n + i]
        })
        .sum()
}

/// Full quadratic form `U' V^+ U` with the multivariate hypergeometric
/// cross-cause covariance (bake-off challenger).
pub struct QuadraticCauseLogRank;

impl SplitCriterion for QuadraticCauseLogRank {
    fn score(&self, l: &Profile, p: &Profile, _n_units_right: f64) -> f64 {
        let (u, v) = cause_moments(l, p);
        pinv_quadratic_form(&u, &v)
    }
}

/// Ishwaran et al. (2014) eq. 3.2 as implemented by randomForestSRC's
/// cause-specific composite log-rank with equal weights:
/// `(sum_j U_j)^2 / sum_j V_j` (bake-off challenger).
pub struct IshwaranComposite;

impl SplitCriterion for IshwaranComposite {
    fn score(&self, l: &Profile, p: &Profile, _n_units_right: f64) -> f64 {
        let (u, v) = cause_moments(l, p);
        let nc = p.n_causes;
        let num: f64 = u.iter().sum();
        let var: f64 = (0..nc).map(|j| v[j * nc + j]).sum();
        if var > 0.0 { num * num / var } else { 0.0 }
    }
}
