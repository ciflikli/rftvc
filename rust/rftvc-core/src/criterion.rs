/// At-risk and event counts on a node's event-time grid.
pub struct Profile<'a> {
    pub at_risk: &'a [f64],
    pub events: &'a [f64],
}

pub trait SplitCriterion: Sync {
    /// Score a candidate split from the left child's and the parent's profiles.
    fn score(&self, left: &Profile, parent: &Profile) -> f64;
}

/// Left-truncated right-censored log-rank chi-square statistic.
///
/// Uses the hypergeometric variance, which is valid for the integer
/// unit-weight counts used in v1 (design.md, D6).
pub struct LtrcLogRank;

impl SplitCriterion for LtrcLogRank {
    fn score(&self, l: &Profile, p: &Profile) -> f64 {
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
}
