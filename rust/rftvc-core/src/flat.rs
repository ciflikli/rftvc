//! Flat (struct-of-arrays) representation of a fitted forest, used for
//! pickling from Python.

use std::sync::Arc;

use crate::forest::Forest;
use crate::tree::{Node, Tree};

/// Version of the flat state. v1 (before S9) stored leaf counts `d`, `y`;
/// v2 stores the leaf cumulative hazards; v3 (S11) adds `n_causes`, with
/// `n_causes` hazards per leaf entry. v4 adds `node_missing_right` (missing-value
/// routing) and allows a Split node's `node_threshold` to be `NaN` (a
/// `MissingVsObserved` split). A v2 state loads as `n_causes = 1`; a v3 (or v2)
/// state loads with every node's `missing_goes_right = true` (irrelevant: a
/// pre-v4 forest was never fit with missing values, so this is never exercised).
pub const FORMAT_VERSION: u64 = 4;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FlatForest {
    pub format_version: u64,
    pub grid: Vec<f64>,
    pub tree_seeds: Vec<u64>,
    pub n_features: u64,
    pub n_groups: u64,
    pub n_draw: u64,
    pub bootstrap: bool,
    /// Per tree, start of its nodes; length `n_trees + 1`.
    pub node_offsets: Vec<u64>,
    /// `-1` marks a leaf, whose leaf index is stored in `node_left`.
    pub node_feature: Vec<i64>,
    pub node_threshold: Vec<f64>,
    /// Where a missing `node_feature` value routes; meaningless (but present)
    /// for a `Leaf` entry. Absent (empty) when loading a pre-v4 state.
    pub node_missing_right: Vec<bool>,
    pub node_left: Vec<u32>,
    pub node_right: Vec<u32>,
    /// Per tree, start of its leaves; length `n_trees + 1`.
    pub leaf_offsets: Vec<u64>,
    /// Per leaf (all trees), start of its event entries; length `n_leaves + 1`.
    pub event_offsets: Vec<u64>,
    pub event_idx: Vec<u32>,
    /// Entry-major: `n_causes` values per entry of `event_idx`.
    pub cumhaz: Vec<f64>,
    pub n_causes: u64,
    /// Per leaf (all trees) `n_causes` in-bag event counts; empty when the
    /// forest did not store them (optional in the pickled state).
    pub leaf_cause_events: Vec<u32>,
}

impl FlatForest {
    pub fn from_forest(forest: &Forest) -> FlatForest {
        let mut f = FlatForest {
            format_version: FORMAT_VERSION,
            grid: forest
                .trees
                .first()
                .map(|t| t.grid_times.to_vec())
                .unwrap_or_default(),
            tree_seeds: forest.tree_seeds.clone(),
            n_features: forest.n_features as u64,
            n_groups: forest.n_groups as u64,
            n_draw: forest.n_draw as u64,
            bootstrap: forest.bootstrap,
            n_causes: forest.n_causes as u64,
            node_offsets: vec![0],
            leaf_offsets: vec![0],
            event_offsets: vec![0],
            ..Default::default()
        };
        for tree in &forest.trees {
            for node in &tree.nodes {
                match node {
                    Node::Split {
                        feature,
                        threshold,
                        missing_goes_right,
                        left,
                        right,
                    } => {
                        f.node_feature.push(*feature as i64);
                        f.node_threshold.push(*threshold);
                        f.node_missing_right.push(*missing_goes_right);
                        f.node_left.push(*left);
                        f.node_right.push(*right);
                    }
                    Node::Leaf { leaf } => {
                        f.node_feature.push(-1);
                        f.node_threshold.push(0.0);
                        f.node_missing_right.push(false);
                        f.node_left.push(*leaf);
                        f.node_right.push(0);
                    }
                }
            }
            f.node_offsets.push(f.node_feature.len() as u64);
            let base = f.event_idx.len() as u64;
            f.event_offsets
                .extend(tree.leaf_offsets[1..].iter().map(|&o| base + o as u64));
            f.event_idx.extend_from_slice(&tree.event_idx);
            f.cumhaz.extend_from_slice(&tree.cumhaz);
            f.leaf_cause_events
                .extend_from_slice(&tree.leaf_cause_events);
            f.leaf_offsets.push(f.event_offsets.len() as u64 - 1);
        }
        f
    }

    /// Rebuild a forest, validating offsets and indices so a corrupt state
    /// errors instead of panicking.
    pub fn to_forest(&self) -> Result<Forest, String> {
        let n_trees = self.node_offsets.len().saturating_sub(1);
        let n_nodes = self.node_feature.len();
        let n_leaves = self.event_offsets.len().saturating_sub(1);
        let n_events = self.event_idx.len();
        if !(2..=FORMAT_VERSION).contains(&self.format_version) {
            return Err(format!(
                "forest state has format version {}, supported 2..={FORMAT_VERSION}; refit the model",
                self.format_version
            ));
        }
        // Scalars must be ones a fit can produce; `cumhaz_at` binary-searches
        // the grid, so it must be finite and strictly increasing.
        // v2 has no `n_causes` (single event); the loader passes 1. `n_causes`
        // has been real (not a loader default) since v3, so a v3 *or* v4 state
        // may legitimately have n_causes > 1 -- only a pre-v3 state is
        // restricted to the single-cause default. (Caught by
        // flat_v3_state_still_loads_and_predicts_identically: this used to read
        // `format_version == FORMAT_VERSION`, which wrongly started rejecting
        // real multi-cause v3 states the moment v4 became the latest version.)
        let nc = self.n_causes as usize;
        let scalars_ok = (1..=255).contains(&self.n_causes)
            && (self.format_version >= 3 || self.n_causes == 1)
            && self.n_features >= 1
            && self.n_groups >= 1
            && self.n_draw >= 1
            && (self.bootstrap || self.n_draw <= self.n_groups);
        let grid_ok =
            self.grid.iter().all(|t| t.is_finite()) && self.grid.windows(2).all(|w| w[0] < w[1]);
        let consistent = scalars_ok
            && grid_ok
            && n_trees >= 1
            && self.leaf_offsets.len() == n_trees + 1
            && self.tree_seeds.len() == n_trees
            && [
                self.node_threshold.len(),
                self.node_left.len(),
                self.node_right.len(),
            ]
            .iter()
            .all(|&l| l == n_nodes)
            && (self.node_missing_right.len() == n_nodes
                || (self.format_version < 4 && self.node_missing_right.is_empty()))
            && Some(self.cumhaz.len()) == n_events.checked_mul(nc)
            && (self.leaf_cause_events.is_empty()
                || Some(self.leaf_cause_events.len()) == n_leaves.checked_mul(nc))
            && monotone(&self.node_offsets, n_nodes)
            && monotone(&self.leaf_offsets, n_leaves)
            && monotone(&self.event_offsets, n_events)
            && self
                .event_idx
                .iter()
                .all(|&k| (k as usize) < self.grid.len());
        if !consistent {
            return Err("inconsistent forest state".into());
        }
        let grid = Arc::new(self.grid.clone());
        let mut trees = Vec::with_capacity(n_trees);
        for t in 0..n_trees {
            let (n0, n1) = (
                self.node_offsets[t] as usize,
                self.node_offsets[t + 1] as usize,
            );
            let (l0, l1) = (
                self.leaf_offsets[t] as usize,
                self.leaf_offsets[t + 1] as usize,
            );
            let (tree_nodes, tree_leaves) = (n1 - n0, l1 - l0);
            if tree_nodes == 0 || tree_leaves == 0 {
                return Err(format!("tree {t} has no nodes or no leaves"));
            }
            let mut nodes = Vec::with_capacity(tree_nodes);
            // Children are always created after their parent, so requiring
            // `parent < child < tree_nodes` also rules out cycles in `apply`.
            let child_ok =
                |c: u32, parent: usize| (parent < c as usize) && (c as usize) < tree_nodes;
            for i in n0..n1 {
                let node = match self.node_feature[i] {
                    -1 if (self.node_left[i] as usize) < tree_leaves => Node::Leaf {
                        leaf: self.node_left[i],
                    },
                    feat if feat >= 0
                        && (feat as u64) < self.n_features
                        // A NaN threshold is a valid `MissingVsObserved` split (v4+
                        // only; a pre-v4 state's writer never produced one).
                        && (self.node_threshold[i].is_finite()
                            || (self.node_threshold[i].is_nan()
                                && self.format_version >= 4
                                && !self.node_missing_right[i]))
                        && child_ok(self.node_left[i], i - n0)
                        && child_ok(self.node_right[i], i - n0) =>
                    {
                        Node::Split {
                            feature: feat as u32,
                            threshold: self.node_threshold[i],
                            missing_goes_right: self
                                .node_missing_right
                                .get(i)
                                .copied()
                                .unwrap_or(true),
                            left: self.node_left[i],
                            right: self.node_right[i],
                        }
                    }
                    _ => return Err(format!("invalid node {i}")),
                };
                nodes.push(node);
            }
            let (e0, e1) = (
                self.event_offsets[l0] as usize,
                self.event_offsets[l1] as usize,
            );
            if u32::try_from(e1 - e0).is_err() {
                return Err(format!("tree {t} has too many leaf entries"));
            }
            // Offsets relative to the tree's first entry (monotone, so in range).
            let leaf_offsets: Vec<u32> = self.event_offsets[l0..=l1]
                .iter()
                .map(|&o| (o as usize - e0) as u32)
                .collect();
            for (l, w) in leaf_offsets.windows(2).enumerate() {
                // `cumhaz_at` binary-searches event times, so they must be strictly
                // increasing; the cumulative hazard must be finite, non-negative
                // and non-decreasing.
                let (a, b) = (e0 + w[0] as usize, e0 + w[1] as usize);
                let (idx, ch) = (&self.event_idx[a..b], &self.cumhaz[a * nc..b * nc]);
                // Per cause: entries `j, j + nc, ...` of the leaf.
                let ok = idx.windows(2).all(|p| p[0] < p[1])
                    && ch.iter().all(|c| c.is_finite())
                    && (0..nc).all(|j| {
                        let col = ch.iter().skip(j).step_by(nc);
                        col.clone().next().is_none_or(|&c| c >= 0.0)
                            && col.clone().zip(col.skip(1)).all(|(p, q)| p <= q)
                    });
                if !ok {
                    return Err(format!("invalid leaf {}", l0 + l));
                }
            }
            trees.push(Tree {
                nodes,
                leaf_offsets,
                event_idx: self.event_idx[e0..e1].to_vec(),
                cumhaz: self.cumhaz[e0 * nc..e1 * nc].to_vec(),
                n_causes: nc,
                leaf_cause_events: if self.leaf_cause_events.is_empty() {
                    Vec::new()
                } else {
                    self.leaf_cause_events[l0 * nc..l1 * nc].to_vec()
                },
                grid_times: Arc::clone(&grid),
            });
        }
        Ok(Forest {
            trees,
            tree_seeds: self.tree_seeds.clone(),
            n_features: self.n_features as usize,
            n_groups: self.n_groups as usize,
            n_draw: self.n_draw as usize,
            bootstrap: self.bootstrap,
            n_causes: nc,
        })
    }
}

/// Offsets start at 0, never decrease, and end at `total`.
fn monotone(offsets: &[u64], total: usize) -> bool {
    offsets.first() == Some(&0)
        && offsets.windows(2).all(|w| w[0] <= w[1])
        && offsets.last().map(|&l| l as usize) == Some(total)
}
