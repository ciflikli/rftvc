use crate::grid::Grid;

/// Features binned to `u8`, stored column-major.
///
/// For feature `f` with sorted edges `e`, `bin(x) = #{j : e_j < x}`, so
/// `bin(x) <= b  <=>  x <= e_b`. A split at bin `b` therefore sends raw
/// values `x <= edges[f][b]` left, which is what prediction uses.
#[derive(Clone, Debug)]
pub struct Binned {
    pub n_rows: usize,
    pub n_features: usize,
    pub bins: Vec<u8>,
    pub edges: Vec<Vec<f64>>,
}

impl Binned {
    /// Bin a row-major `n_rows x n_features` matrix into at most `max_bins` bins per feature.
    pub fn fit(x: &[f64], n_rows: usize, n_features: usize, max_bins: usize) -> Binned {
        assert!((2..=256).contains(&max_bins), "max_bins must be in 2..=256");
        assert_eq!(x.len(), n_rows * n_features);
        let mut bins = vec![0u8; n_rows * n_features];
        let mut edges = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let mut col: Vec<f64> = (0..n_rows).map(|i| x[i * n_features + f]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).expect("NaN in X"));
            let e = feature_edges(&col, max_bins);
            let out = &mut bins[f * n_rows..(f + 1) * n_rows];
            for i in 0..n_rows {
                out[i] = bin_of(&e, x[i * n_features + f]);
            }
            edges.push(e);
        }
        Binned {
            n_rows,
            n_features,
            bins,
            edges,
        }
    }

    #[inline]
    pub fn column(&self, f: usize) -> &[u8] {
        &self.bins[f * self.n_rows..(f + 1) * self.n_rows]
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
    pub event: Vec<bool>,
    /// Person-time `stop - start` of each row (on the snapped times in coarse mode,
    /// which are the times passed in).
    pub duration: Vec<f64>,
}

impl SurvData {
    pub fn new(start: &[f64], stop: &[f64], event: &[bool]) -> SurvData {
        assert!(start.len() == stop.len() && stop.len() == event.len());
        let grid = Grid::exact(stop, event);
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
            event: event.to_vec(),
            duration,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.event.len()
    }
}
