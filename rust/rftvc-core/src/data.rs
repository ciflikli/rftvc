use crate::grid::Grid;

/// Features binned to `u8`, stored column-major.
///
/// For feature `f` with sorted edges `e`, `bin(x) = #{j : e_j < x}`, so
/// `bin(x) <= b  <=>  x <= e_b`. A split at bin `b` therefore sends raw
/// values `x <= edges[f][b]` left, which is what prediction uses.
///
/// `NaN` is a legitimate ("missing") value: `edges` are computed from the
/// non-missing values of a column only, and `missing[i]` is set wherever
/// `x` was `NaN` there. `bins[i]` is never meaningful for a missing entry
/// (routing/splitting on it must check `missing` first) -- `bin_of` still
/// returns a well-defined value for `NaN` (bin `0`, since no edge is `< NaN`)
/// so nothing panics if it's read anyway, but it carries no information.
#[derive(Clone, Debug)]
pub struct Binned {
    pub n_rows: usize,
    pub n_features: usize,
    pub bins: Vec<u8>,
    pub missing: Vec<bool>,
    /// Fast path for features with no missing values anywhere in training X.
    pub feature_has_missing: Vec<bool>,
    pub edges: Vec<Vec<f64>>,
}

impl Binned {
    /// Bin a row-major `n_rows x n_features` matrix into at most `max_bins` bins per feature.
    pub fn fit(x: &[f64], n_rows: usize, n_features: usize, max_bins: usize) -> Binned {
        assert!((2..=256).contains(&max_bins), "max_bins must be in 2..=256");
        assert_eq!(x.len(), n_rows * n_features);
        let mut bins = vec![0u8; n_rows * n_features];
        let mut missing = vec![false; n_rows * n_features];
        let mut feature_has_missing = vec![false; n_features];
        let mut edges = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let mut col: Vec<f64> = (0..n_rows)
                .map(|i| x[i * n_features + f])
                .filter(|v| !v.is_nan())
                .collect();
            // NaN was filtered out above; every remaining f64 (including +/-inf) is
            // totally ordered by partial_cmp, so this can never actually be None.
            col.sort_by(|a, b| a.partial_cmp(b).expect("unreachable: NaN already filtered"));
            let e = feature_edges(&col, max_bins);
            let out = &mut bins[f * n_rows..(f + 1) * n_rows];
            let miss = &mut missing[f * n_rows..(f + 1) * n_rows];
            for i in 0..n_rows {
                let v = x[i * n_features + f];
                if v.is_nan() {
                    miss[i] = true;
                    feature_has_missing[f] = true;
                } else {
                    out[i] = bin_of(&e, v);
                }
            }
            edges.push(e);
        }
        Binned {
            n_rows,
            n_features,
            bins,
            missing,
            feature_has_missing,
            edges,
        }
    }

    #[inline]
    pub fn column(&self, f: usize) -> &[u8] {
        &self.bins[f * self.n_rows..(f + 1) * self.n_rows]
    }

    #[inline]
    pub fn missing_column(&self, f: usize) -> &[bool] {
        &self.missing[f * self.n_rows..(f + 1) * self.n_rows]
    }
}

#[inline]
pub fn bin_of(edges: &[f64], x: f64) -> u8 {
    edges.partition_point(|&e| e < x) as u8
}

/// An edge `e` with `a <= e < b`, so `a` bins left and `b` right. Computed as
/// `a/2 + b/2`, which cannot overflow even for `[-f64::MAX, f64::MAX]`; falls
/// back to `a` when rounding lands on `b` (adjacent floats).
fn midpoint(a: f64, b: f64) -> f64 {
    let m = (a / 2.0 + b / 2.0).max(a);
    if m < b { m } else { a }
}

/// Midpoints between consecutive unique values when they fit in `max_bins`,
/// otherwise distinct quantile values of the sorted column.
fn feature_edges(sorted: &[f64], max_bins: usize) -> Vec<f64> {
    let mut uniq = sorted.to_vec();
    uniq.dedup();
    if uniq.len() <= max_bins {
        return uniq.windows(2).map(|w| midpoint(w[0], w[1])).collect();
    }
    let n = sorted.len();
    let mut e: Vec<f64> = (1..max_bins).map(|q| sorted[q * n / max_bins]).collect();
    e.dedup();
    // The maximum as an edge would create an empty right bin.
    if e.last() == sorted.last() {
        e.pop();
    }
    e
}

/// Survival response mapped onto the event-time grid.
///
/// Row `r` is at risk at grid index `k` iff `a[r] <= k < b[r]`,
/// i.e. `start_r < t_k <= stop_r`. An event row fails at `k = b[r] - 1`.
#[derive(Clone, Debug)]
pub struct SurvData {
    pub grid: Grid,
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    /// Cause code of each row: 0 = censored, `1..=n_causes` = event of that cause.
    pub event: Vec<u8>,
    /// Number of causes `J` (1 for single-event survival).
    pub n_causes: usize,
    /// Person-time `stop - start` of each row (on the snapped times in coarse mode,
    /// which are the times passed in).
    pub duration: Vec<f64>,
}

impl SurvData {
    /// Single-event data (`J = 1`).
    pub fn new(start: &[f64], stop: &[f64], event: &[bool]) -> SurvData {
        let codes: Vec<u8> = event.iter().map(|&e| e as u8).collect();
        SurvData::with_causes(start, stop, &codes, 1)
    }

    /// Competing-risks data: `codes[r]` in `0..=n_causes`.
    pub fn with_causes(start: &[f64], stop: &[f64], codes: &[u8], n_causes: usize) -> SurvData {
        assert!(start.len() == stop.len() && stop.len() == codes.len());
        assert!((1..=255).contains(&n_causes), "n_causes must be in 1..=255");
        assert!(
            codes.iter().all(|&c| (c as usize) <= n_causes),
            "cause code above n_causes"
        );
        let grid = Grid::exact(stop, codes);
        let a = start
            .iter()
            .map(|&s| grid.first_greater(s) as u32)
            .collect();
        let b = stop.iter().map(|&s| grid.first_greater(s) as u32).collect();
        let duration = start.iter().zip(stop).map(|(s, t)| t - s).collect();
        SurvData {
            grid,
            a,
            b,
            event: codes.to_vec(),
            n_causes,
            duration,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.event.len()
    }
}
