//! Survival forest engine for counting-process data `(start, stop, event, X)`.
//!
//! Rows are left-truncated at `start` and right-censored or failing at `stop`.
//! Everything is exact on the event-time grid (design.md D8, exact mode).

pub mod criterion;
pub mod data;
pub mod grid;
pub mod rng;
pub mod splitter;
pub mod tree;

pub use criterion::{LtrcLogRank, Profile, SplitCriterion};
pub use data::{Binned, SurvData};
pub use grid::Grid;
pub use splitter::{
    NodeProfile, SplitCandidate, SplitParams, best_split, node_profile, profile_on,
};
pub use tree::{Tree, TreeParams, build_tree};
