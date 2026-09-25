use std::sync::Arc;

/// Sorted, unique event times `t_0 < … < t_{K-1}`, shared (cheaply cloned) by all trees.
#[derive(Clone, Debug)]
pub struct Grid {
    pub times: Arc<Vec<f64>>,
}

impl Grid {
    /// Exact grid: all unique times at which an event occurs.
    pub fn exact(stop: &[f64], event: &[bool]) -> Grid {
        let mut times: Vec<f64> = stop
            .iter()
            .zip(event)
            .filter(|(_, e)| **e)
            .map(|(t, _)| *t)
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).expect("NaN time"));
        times.dedup();
        Grid {
            times: Arc::new(times),
        }
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// First index `k` with `t_k > x` (`len()` if none).
    pub fn first_greater(&self, x: f64) -> usize {
        self.times.partition_point(|&t| t <= x)
    }
}

impl Grid {
    /// Coarse event grid (design.md D8): empirical quantiles (inverse CDF) of
    /// the event times, with multiplicity, at levels `j / k` for `j = 1..=k`,
    /// deduplicated. At most `k` points; each is an observed event time and the
    /// last is the largest.
    pub fn quantile(stop: &[f64], event: &[bool], k: usize) -> Grid {
        assert!(k >= 1, "k must be >= 1");
        let mut t: Vec<f64> = stop
            .iter()
            .zip(event)
            .filter(|(_, e)| **e)
            .map(|(s, _)| *s)
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).expect("NaN time"));
        let n = t.len();
        let mut times: Vec<f64> = if n == 0 {
            Vec::new()
        } else {
            (1..=k).map(|j| t[(j * n).div_ceil(k) - 1]).collect()
        };
        times.dedup();
        Grid {
            times: Arc::new(times),
        }
    }
}

/// Counting-process rows after snapping times to a coarse grid (D8).
#[derive(Clone, Debug)]
pub struct Coarsened {
    /// Original index of each kept row.
    pub kept: Vec<u32>,
    pub start: Vec<f64>,
    pub stop: Vec<f64>,
    pub event: Vec<bool>,
    /// Events whose chain had no kept row to carry them.
    pub lost_events: usize,
}

/// Snap rows to `points` (sorted snapping grid: origin + event grid) and
/// rebuild them (design.md D8.2).
///
/// `g(t)` is the smallest point `>= t`, clamped to the last point: times past
/// the last event carry no further at-risk contribution, so the clamp leaves
/// every risk set unchanged. Chain `c` owns rows `order[offsets[c]..offsets[c+1]]`,
/// sorted by start and contiguous. A row with `g(start) == g(stop)` is dropped;
/// its event moves to the chain's previous kept row (which ends at the same
/// point by contiguity) or, if there is none, is lost and counted.
pub fn coarsen(
    start: &[f64],
    stop: &[f64],
    event: &[bool],
    order: &[u32],
    offsets: &[usize],
    points: &[f64],
) -> Coarsened {
    assert!(!points.is_empty(), "empty snapping grid");
    let last = *points.last().unwrap();
    let g = |t: f64| -> f64 {
        let i = points.partition_point(|&p| p < t);
        if i < points.len() { points[i] } else { last }
    };
    let mut out = Coarsened {
        kept: Vec::new(),
        start: Vec::new(),
        stop: Vec::new(),
        event: Vec::new(),
        lost_events: 0,
    };
    for w in offsets.windows(2) {
        let chain_first = out.kept.len();
        for &r in &order[w[0]..w[1]] {
            let r = r as usize;
            let (s, e) = (g(start[r]), g(stop[r]));
            if s < e {
                out.kept.push(r as u32);
                out.start.push(s);
                out.stop.push(e);
                out.event.push(event[r]);
            } else if event[r] {
                if out.kept.len() > chain_first {
                    *out.event.last_mut().unwrap() = true;
                } else {
                    out.lost_events += 1;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_grid_uses_inverse_cdf_and_keeps_the_maximum() {
        let stop = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 9.0];
        let event = [true, true, true, true, false, true, true];
        // Event times (n = 6): 1 2 2 3 5 9; k = 3 -> ranks ceil(2), ceil(4), ceil(6).
        let g = Grid::quantile(&stop, &event, 3);
        assert_eq!(*g.times, vec![2.0, 3.0, 9.0]);
        assert_eq!(
            *Grid::quantile(&stop, &event, 100).times,
            vec![1.0, 2.0, 3.0, 5.0, 9.0]
        );
    }
}
