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

#[derive(Clone, Debug, Default)]
pub struct TreeParams {
    /// `None` = unlimited; `Some(0)` = a single root leaf.
    pub max_depth: Option<usize>,
    pub min_ids_leaf: usize,
    /// Minimum events of any cause per child.
    pub min_events_leaf: usize,
    /// Features tried per node (without replacement).
    pub max_features: usize,
    /// `(cause, m)` (0-based cause): each child needs `m` events of that cause,
    /// and a node needs `2 m` before any feature is drawn (cr-design.md C5).
    pub cause_floor: Option<(usize, usize)>,
    /// Store per-leaf per-cause in-bag event counts (`Tree::leaf_cause_events`).
    pub leaf_events: bool,
}

#[derive(Clone, Debug)]
/// Leaves are stored flat: leaf `l` owns entries `leaf_offsets[l]..leaf_offsets[l + 1]`
/// of `event_idx` (its event times of any cause, as sorted grid indices) and
/// `cumhaz` (the per-cause Nelson–Aalen cumulative hazards, right-continuous,
/// at those times). `cumhaz` is entry-major: entry `e`, cause `j` is at
/// `e * n_causes + j`, so `cumhaz.len() == event_idx.len() * n_causes`.
pub struct Tree {
    pub nodes: Vec<Node>,
    pub leaf_offsets: Vec<u32>,
    pub event_idx: Vec<u32>,
    pub cumhaz: Vec<f64>,
    pub n_causes: usize,
    /// In-bag events of each cause per leaf, `[leaf * n_causes + j]` (bootstrap
    /// copies count); empty unless `TreeParams::leaf_events`.
    pub leaf_cause_events: Vec<u32>,
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
        cause_floor: params.cause_floor,
    };
    let mut nodes = vec![Node::Leaf { leaf: 0 }];
    let (mut leaf_offsets, mut event_idx, mut cumhaz) = (vec![0u32], Vec::new(), Vec::new());
    let mut leaf_cause_events = Vec::new();
    let mut stack = vec![(0usize, rows, units, 0usize)];

    while let Some((node_id, rows, units, depth)) = stack.pop() {
        let profile = node_profile(surv, &rows);
        let can_split = params.max_depth.is_none_or(|m| depth < m)
            && count_units(&units) >= 2 * params.min_ids_leaf
            && profile.n_events >= 2 * params.min_events_leaf
            && params
                .cause_floor
                .is_none_or(|(c, m)| profile.n_cause_events[c] >= 2 * m);
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
            None if surv.n_causes == 1 => {
                let mut cum = 0.0;
                cumhaz.extend(profile.events.iter().zip(&profile.at_risk).map(|(d, y)| {
                    cum += d / y;
                    cum
                }));
                if params.leaf_events {
                    leaf_cause_events.extend(profile.n_cause_events.iter().map(|&c| c as u32));
                }
                event_idx.extend_from_slice(&profile.event_idx);
                nodes[node_id] = Node::Leaf {
                    leaf: leaf_offsets.len() as u32 - 1,
                };
                leaf_offsets.push(u32::try_from(event_idx.len()).expect("leaf entries exceed u32"));
            }
            None => {
                let nc = surv.n_causes;
                let mut cum = vec![0.0; nc];
                let k = profile.at_risk.len();
                for (t, y) in profile.at_risk.iter().enumerate() {
                    for (j, c) in cum.iter_mut().enumerate() {
                        *c += profile.cause_events[j * k + t] / y;
                    }
                    cumhaz.extend_from_slice(&cum);
                }
                if params.leaf_events {
                    leaf_cause_events.extend(profile.n_cause_events.iter().map(|&c| c as u32));
                }
                event_idx.extend_from_slice(&profile.event_idx);
                nodes[node_id] = Node::Leaf {
                    leaf: leaf_offsets.len() as u32 - 1,
                };
                leaf_offsets.push(u32::try_from(event_idx.len()).expect("leaf entries exceed u32"));
            }
        }
    }
    // Growth by doubling leaves up to half of each array unused.
    nodes.shrink_to_fit();
    leaf_offsets.shrink_to_fit();
    event_idx.shrink_to_fit();
    cumhaz.shrink_to_fit();
    leaf_cause_events.shrink_to_fit();
    Tree {
        nodes,
        leaf_offsets,
        event_idx,
        cumhaz,
        n_causes: surv.n_causes,
        leaf_cause_events,
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

    pub fn n_leaves(&self) -> usize {
        self.leaf_offsets.len() - 1
    }

    pub(crate) fn entries(&self, leaf: usize) -> std::ops::Range<usize> {
        self.leaf_offsets[leaf] as usize..self.leaf_offsets[leaf + 1] as usize
    }

    /// Grid indices of the leaf's event times (strictly increasing).
    pub fn leaf_event_idx(&self, leaf: usize) -> &[u32] {
        &self.event_idx[self.entries(leaf)]
    }

    /// Leaf cumulative hazards at its event times, entry-major (`n_causes` per entry).
    pub fn leaf_cumhaz(&self, leaf: usize) -> &[f64] {
        let r = self.entries(leaf);
        &self.cumhaz[r.start * self.n_causes..r.end * self.n_causes]
    }

    /// Cause `j`'s leaf cumulative hazard at time `t` (right-continuous step function).
    pub fn cause_cumhaz_at(&self, leaf: usize, t: f64, j: usize) -> f64 {
        let r = self.entries(leaf);
        let pos = self.event_idx[r.clone()].partition_point(|&k| self.grid_times[k as usize] <= t);
        if pos == 0 {
            0.0
        } else {
            self.cumhaz[(r.start + pos - 1) * self.n_causes + j]
        }
    }

    /// Leaf cumulative hazard at time `t` (right-continuous step function), single-event trees.
    pub fn cumhaz_at(&self, leaf: usize, t: f64) -> f64 {
        debug_assert_eq!(self.n_causes, 1);
        let r = self.entries(leaf);
        let pos = self.event_idx[r.clone()].partition_point(|&k| self.grid_times[k as usize] <= t);
        if pos == 0 {
            0.0
        } else {
            self.cumhaz[r.start + pos - 1]
        }
    }

    /// Allocated bytes of the nodes and leaf arrays (capacity, not length;
    /// the shared grid is not counted).
    pub fn nbytes(&self) -> usize {
        use std::mem::size_of;
        self.nodes.capacity() * size_of::<Node>()
            + self.leaf_offsets.capacity() * size_of::<u32>()
            + self.event_idx.capacity() * size_of::<u32>()
            + self.cumhaz.capacity() * size_of::<f64>()
            + self.leaf_cause_events.capacity() * size_of::<u32>()
    }
}
