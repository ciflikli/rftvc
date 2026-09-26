/// Summaries of one node (or candidate child) on the parent node's event times.
///
/// A child profile shares the parent's `times`. Log-rank reads only `at_risk`
/// and `events`; the other fields serve likelihood and impurity criteria.
pub struct Profile<'a> {
    pub at_risk: &'a [f64],
    pub events: &'a [f64],
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
