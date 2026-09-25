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
