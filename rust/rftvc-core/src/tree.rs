use std::sync::Arc;

use crate::criterion::SplitCriterion;
use crate::data::{Binned, SurvData};
use crate::rng::Rng;
use crate::splitter::{SplitParams, best_split_in, count_units, node_profile};

#[derive(Clone, Debug)]
pub enum Node {
    Split {
        feature: u32,
        threshold: f64,
        left: u32,
        right: u32,
    },
    Leaf {
        leaf: u32,
    },
}

/// Leaf risk-set counts at the leaf's own event times, plus the
/// Nelson–Aalen cumulative hazard (right-continuous) at those times.
#[derive(Clone, Debug)]
pub struct Leaf {
    pub event_idx: Vec<u32>,
    pub d: Vec<f64>,
    pub y: Vec<f64>,
    pub cumhaz: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct TreeParams {
    /// `None` = unlimited; `Some(0)` = a single root leaf.
    pub max_depth: Option<usize>,
    pub min_ids_leaf: usize,
    pub min_events_leaf: usize,
    /// Features tried per node (without replacement).
    pub max_features: usize,
}

#[derive(Clone, Debug)]
pub struct Tree {
    pub nodes: Vec<Node>,
    pub leaves: Vec<Leaf>,
    pub grid_times: Arc<Vec<f64>>,
}

/// Grow one tree on `rows`, where `units[i]` is the resampling unit (an id, or
/// one bootstrap copy of an id) of `rows[i]`. Each unit's rows must be
/// contiguous; partitioning keeps relative order, so this holds in every node.
pub fn build_tree(
    binned: &Binned,
    surv: &SurvData,
    rows: Vec<u32>,
    units: Vec<u32>,
    params: &TreeParams,
    criterion: &dyn SplitCriterion,
    rng: &mut Rng,
) -> Tree {
    let split_params = SplitParams {
        min_leaf: params.min_ids_leaf,
        min_events_leaf: params.min_events_leaf,
    };
    let mut nodes = vec![Node::Leaf { leaf: 0 }];
    let mut leaves = Vec::new();
    let mut stack = vec![(0usize, rows, units, 0usize)];

    while let Some((node_id, rows, units, depth)) = stack.pop() {
        let profile = node_profile(surv, &rows);
        let can_split = params.max_depth.is_none_or(|m| depth < m)
            && count_units(&units) >= 2 * params.min_ids_leaf
            && profile.n_events >= 2 * params.min_events_leaf;
        let split = if can_split {
            let features = rng.sample_without_replacement(binned.n_features, params.max_features);
            best_split_in(
                binned,
                &profile,
                &rows,
                &units,
                &features,
                &split_params,
                criterion,
            )
        } else {
            None
        };
        match split {
            Some(s) => {
                let col = binned.column(s.feature);
                // Stable partition of (row, unit) pairs keeps each unit contiguous.
                let (mut l_rows, mut l_units, mut r_rows, mut r_units) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for (&row, &unit) in rows.iter().zip(&units) {
                    if col[row as usize] <= s.bin {
                        l_rows.push(row);
                        l_units.push(unit);
                    } else {
                        r_rows.push(row);
                        r_units.push(unit);
                    }
                }
                let (li, ri) = (nodes.len(), nodes.len() + 1);
                nodes.push(Node::Leaf { leaf: 0 });
                nodes.push(Node::Leaf { leaf: 0 });
                nodes[node_id] = Node::Split {
                    feature: s.feature as u32,
                    threshold: s.threshold,
                    left: li as u32,
                    right: ri as u32,
                };
                stack.push((ri, r_rows, r_units, depth + 1));
                stack.push((li, l_rows, l_units, depth + 1));
            }
            None => {
                let mut cum = 0.0;
                let cumhaz = profile
                    .events
                    .iter()
                    .zip(&profile.at_risk)
                    .map(|(d, y)| {
                        cum += d / y;
                        cum
                    })
                    .collect();
                nodes[node_id] = Node::Leaf {
                    leaf: leaves.len() as u32,
                };
                leaves.push(Leaf {
                    event_idx: profile.event_idx,
                    d: profile.events,
                    y: profile.at_risk,
                    cumhaz,
                });
            }
        }
    }
    Tree {
        nodes,
        leaves,
        grid_times: Arc::clone(&surv.grid.times),
    }
}

impl Tree {
    /// Leaf index reached by one raw feature row.
    pub fn apply(&self, x: &[f64]) -> usize {
        let mut i = 0usize;
        loop {
            match &self.nodes[i] {
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    i = if x[*feature as usize] <= *threshold {
                        *left
                    } else {
                        *right
                    } as usize;
                }
                Node::Leaf { leaf } => return *leaf as usize,
            }
        }
    }

    /// Nelson–Aalen hazard increments `d_k / Y_k` at the leaf's event times.
    pub fn leaf_hazard(&self, leaf: usize) -> Vec<f64> {
        let l = &self.leaves[leaf];
        l.d.iter().zip(&l.y).map(|(d, y)| d / y).collect()
    }

    /// Leaf cumulative hazard at time `t` (right-continuous step function).
    pub fn cumhaz_at(&self, leaf: usize, t: f64) -> f64 {
        let l = &self.leaves[leaf];
        let pos = l
            .event_idx
            .partition_point(|&k| self.grid_times[k as usize] <= t);
        if pos == 0 { 0.0 } else { l.cumhaz[pos - 1] }
    }
}
