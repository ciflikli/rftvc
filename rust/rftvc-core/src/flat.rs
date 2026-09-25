//! Flat (struct-of-arrays) representation of a fitted forest, used for
//! pickling from Python. Leaf cumulative hazards are rebuilt on load.

use std::sync::Arc;

use crate::forest::Forest;
use crate::tree::{Leaf, Node, Tree};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FlatForest {
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
    pub node_left: Vec<u32>,
    pub node_right: Vec<u32>,
    /// Per tree, start of its leaves; length `n_trees + 1`.
    pub leaf_offsets: Vec<u64>,
    /// Per leaf (all trees), start of its event entries; length `n_leaves + 1`.
    pub event_offsets: Vec<u64>,
    pub event_idx: Vec<u32>,
    pub d: Vec<f64>,
    pub y: Vec<f64>,
}

impl FlatForest {
    pub fn from_forest(forest: &Forest) -> FlatForest {
        let mut f = FlatForest {
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
                        left,
                        right,
                    } => {
                        f.node_feature.push(*feature as i64);
                        f.node_threshold.push(*threshold);
                        f.node_left.push(*left);
                        f.node_right.push(*right);
                    }
                    Node::Leaf { leaf } => {
                        f.node_feature.push(-1);
                        f.node_threshold.push(0.0);
                        f.node_left.push(*leaf);
                        f.node_right.push(0);
                    }
                }
            }
            f.node_offsets.push(f.node_feature.len() as u64);
            for leaf in &tree.leaves {
                f.event_idx.extend_from_slice(&leaf.event_idx);
                f.d.extend_from_slice(&leaf.d);
                f.y.extend_from_slice(&leaf.y);
                f.event_offsets.push(f.event_idx.len() as u64);
            }
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
        let consistent = n_trees >= 1
            && self.leaf_offsets.len() == n_trees + 1
            && self.tree_seeds.len() == n_trees
            && [
                self.node_threshold.len(),
                self.node_left.len(),
                self.node_right.len(),
            ]
            .iter()
            .all(|&l| l == n_nodes)
            && self.d.len() == n_events
            && self.y.len() == n_events
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
                        && self.node_threshold[i].is_finite()
                        && child_ok(self.node_left[i], i - n0)
                        && child_ok(self.node_right[i], i - n0) =>
                    {
                        Node::Split {
                            feature: feat as u32,
                            threshold: self.node_threshold[i],
                            left: self.node_left[i],
                            right: self.node_right[i],
                        }
                    }
                    _ => return Err(format!("invalid node {i}")),
                };
                nodes.push(node);
            }
            let mut leaves = Vec::with_capacity(tree_leaves);
            for l in l0..l1 {
                let (e0, e1) = (
                    self.event_offsets[l] as usize,
                    self.event_offsets[l + 1] as usize,
                );
                // `cumhaz_at` binary-searches event times, so they must be strictly
                // increasing; counts must give finite, non-negative increments.
                let event_idx = self.event_idx[e0..e1].to_vec();
                let (d, y) = (self.d[e0..e1].to_vec(), self.y[e0..e1].to_vec());
                let counts_ok = d.iter().zip(&y).all(|(&d, &y)| {
                    d.is_finite() && y.is_finite() && 0.0 <= d && d <= y && y > 0.0
                });
                if !(event_idx.windows(2).all(|w| w[0] < w[1]) && counts_ok) {
                    return Err(format!("invalid leaf {l}"));
                }
                let mut cum = 0.0;
                let cumhaz = d
                    .iter()
                    .zip(&y)
                    .map(|(d, y)| {
                        cum += d / y;
                        cum
                    })
                    .collect();
                leaves.push(Leaf {
                    event_idx,
                    d,
                    y,
                    cumhaz,
                });
            }
            trees.push(Tree {
                nodes,
                leaves,
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
        })
    }
}

/// Offsets start at 0, never decrease, and end at `total`.
fn monotone(offsets: &[u64], total: usize) -> bool {
    offsets.first() == Some(&0)
        && offsets.windows(2).all(|w| w[0] <= w[1])
        && offsets.last().map(|&l| l as usize) == Some(total)
}
