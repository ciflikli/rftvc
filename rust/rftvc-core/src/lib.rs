//! Survival forest engine for counting-process data `(start, stop, event, X)`.
//!
//! Rows are left-truncated at `start` and right-censored or failing at `stop`.
//! Everything is exact on the event-time grid (design.md D8, exact mode).

pub mod criterion;
pub mod data;
pub mod flat;
pub mod forest;
pub mod grid;
pub mod rng;
pub mod splitter;
pub mod tree;

pub use criterion::{LtrcLogRank, NodeScorer, Profile, SplitCriterion};
pub use data::{Binned, SurvData};
pub use flat::{FORMAT_VERSION, FlatForest};
pub use forest::{Aggregate, Extrapolate, Forest, ForestParams, Groups, draw_ids, fit_forest};
pub use grid::{Coarsened, Grid, coarsen};
pub use rng::Rng;
pub use splitter::{
    NodeProfile, SplitCandidate, SplitParams, best_split, count_units, exposure_of, node_profile,
    profile_on,
};
pub use tree::{Tree, TreeParams, build_tree};
