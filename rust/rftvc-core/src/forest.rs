//! Forest: whole-id resampling, parallel tree building, aggregated prediction.

use rayon::prelude::*;

use crate::criterion::LtrcLogRank;
use crate::data::{Binned, SurvData};
use crate::rng::Rng;
use crate::tree::{Tree, TreeParams, build_tree};

/// Salt separating a tree's resampling stream from its feature-sampling stream.
const SPLIT_STREAM: u64 = 0xA5A5_5A5A_C3C3_3C3C;

/// Covariates beyond a path's last `stop`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extrapolate {
    /// Undefined (NaN): the future covariate path is unknown.
    None,
    /// Named scenario: the last row's covariates stay in force.
    Locf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aggregate {
    /// `S = exp(-mean_b Λ_b)`.
    Hazard,
    /// `S = mean_b exp(-Λ_b)`.
    Survival,
}

#[derive(Clone, Debug)]
pub struct ForestParams {
    pub tree: TreeParams,
    pub n_trees: usize,
    /// Ids drawn per tree.
    pub n_draw: usize,
    /// With replacement (classic bootstrap) or without (subsampling, D10).
    pub bootstrap: bool,
    pub seed: u64,
}

/// Rows grouped by id: `rows_of[g]` are the rows of id `g`.
#[derive(Clone, Debug)]
pub struct Groups {
    pub rows_of: Vec<Vec<u32>>,
}

impl Groups {
    /// `groups[r]` is the id index (`0..n_groups`) of row `r`.
    pub fn new(groups: &[u32], n_groups: usize) -> Groups {
        let mut rows_of = vec![Vec::new(); n_groups];
        for (r, &g) in groups.iter().enumerate() {
            rows_of[g as usize].push(r as u32);
        }
        Groups { rows_of }
    }

    pub fn len(&self) -> usize {
        self.rows_of.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows_of.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct Forest {
    pub trees: Vec<Tree>,
    pub tree_seeds: Vec<u64>,
    pub n_features: usize,
    pub n_groups: usize,
    pub n_draw: usize,
    pub bootstrap: bool,
}

/// Id indices drawn for one tree. Deterministic in `seed`, so bags are
/// recomputed on demand (OOB, diagnostics) rather than stored.
pub fn draw_ids(seed: u64, n_groups: usize, n_draw: usize, bootstrap: bool) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut ids: Vec<u32> = if bootstrap {
        (0..n_draw).map(|_| rng.below(n_groups) as u32).collect()
    } else {
        rng.sample_without_replacement(n_groups, n_draw)
            .into_iter()
            .map(|g| g as u32)
            .collect()
    };
    ids.sort_unstable();
    ids
}

pub fn fit_forest(
    binned: &Binned,
    surv: &SurvData,
    groups: &Groups,
    params: &ForestParams,
) -> Forest {
    let mut master = Rng::new(params.seed);
    let tree_seeds: Vec<u64> = (0..params.n_trees).map(|_| master.next_u64()).collect();
    let trees = tree_seeds
        .par_iter()
        .map(|&seed| {
            let ids = draw_ids(seed, groups.len(), params.n_draw, params.bootstrap);
            // Unit = position in the bag, so bootstrap copies of an id are distinct units.
            let (mut rows, mut units) = (Vec::new(), Vec::new());
            for (unit, &g) in ids.iter().enumerate() {
                for &r in &groups.rows_of[g as usize] {
                    rows.push(r);
                    units.push(unit as u32);
                }
            }
            let mut rng = Rng::new(seed ^ SPLIT_STREAM);
            build_tree(
                binned,
                surv,
                rows,
                units,
                &params.tree,
                &LtrcLogRank,
                &mut rng,
            )
        })
        .collect();
    Forest {
        trees,
        tree_seeds,
        n_features: binned.n_features,
        n_groups: groups.len(),
        n_draw: params.n_draw,
        bootstrap: params.bootstrap,
    }
}

impl Forest {
    pub fn in_bag_ids(&self, tree: usize) -> Vec<u32> {
        draw_ids(
            self.tree_seeds[tree],
            self.n_groups,
            self.n_draw,
            self.bootstrap,
        )
    }

    /// Ensemble cumulative hazard, row-major `(n_rows, times.len())`.
    ///
    /// Under `Survival` aggregation this is `-log(mean_b exp(-Λ_b))`.
    pub fn predict_cumhaz(
        &self,
        x: &[f64],
        n_features: usize,
        times: &[f64],
        agg: Aggregate,
    ) -> Vec<f64> {
        let m = times.len();
        let n_trees = self.trees.len() as f64;
        let mut out = vec![0.0; x.len() / n_features * m];
        out.par_chunks_mut(m.max(1))
            .zip(x.par_chunks(n_features))
            .for_each(|(row_out, xr)| match agg {
                Aggregate::Hazard => {
                    for tree in &self.trees {
                        let leaf = tree.apply(xr);
                        for (o, &t) in row_out.iter_mut().zip(times) {
                            *o += tree.cumhaz_at(leaf, t);
                        }
                    }
                    row_out.iter_mut().for_each(|v| *v /= n_trees);
                }
                Aggregate::Survival => {
                    // -log(mean_b exp(-Λ_b)) via an online log-sum-exp, so large
                    // hazards stay finite instead of underflowing exp() to 0.
                    let mut max = vec![f64::NEG_INFINITY; m];
                    let mut sum = vec![0.0; m];
                    for tree in &self.trees {
                        let leaf = tree.apply(xr);
                        for (j, &t) in times.iter().enumerate() {
                            let a = -tree.cumhaz_at(leaf, t);
                            if a > max[j] {
                                sum[j] = sum[j] * (max[j] - a).exp() + 1.0;
                                max[j] = a;
                            } else {
                                sum[j] += (a - max[j]).exp();
                            }
                        }
                    }
                    for (j, o) in row_out.iter_mut().enumerate() {
                        *o = -(max[j] + (sum[j] / n_trees).ln());
                    }
                }
            });
        out
    }

    /// Conditional cumulative hazard along covariate paths, `(n_paths, times.len())`.
    ///
    /// Path `s` owns rows `offsets[s]..offsets[s+1]`, which must be contiguous
    /// in time and sorted by `start`. Each row routes its own covariates, and the
    /// row's leaf hazard accrues on `(start, stop]`. The value at `t` is
    /// `Λ(t) - Λ(origin[s])`; it is NaN for `t < origin[s]`, and for `t` beyond
    /// the last `stop` unless `extrapolate` is `Locf` (last row's covariates continue).
    ///
    /// `Survival` aggregation averages each tree's conditional survival,
    /// `mean_b exp(-(Λ_b(t) - Λ_b(origin)))`, in log space.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_paths(
        &self,
        x: &[f64],
        n_features: usize,
        start: &[f64],
        stop: &[f64],
        offsets: &[usize],
        origin: &[f64],
        times: &[f64],
        agg: Aggregate,
        extrapolate: Extrapolate,
    ) -> Vec<f64> {
        let m = times.len();
        let n_trees = self.trees.len() as f64;
        let n_paths = offsets.len().saturating_sub(1);
        let mut out = vec![0.0; n_paths * m];
        out.par_chunks_mut(m.max(1))
            .enumerate()
            .for_each(|(p, row_out)| {
                let (r0, r1) = (offsets[p], offsets[p + 1]);
                let last_stop = stop[r1 - 1];
                let u = origin[p];
                let mut leaves = vec![0usize; r1 - r0];
                let (mut max, mut sum) = (vec![f64::NEG_INFINITY; m], vec![0.0; m]);
                for tree in &self.trees {
                    for (l, r) in leaves.iter_mut().zip(r0..r1) {
                        *l = tree.apply(&x[r * n_features..(r + 1) * n_features]);
                    }
                    // Cumulative hazard of this tree along the path up to time t.
                    let path_cumhaz = |t: f64| -> f64 {
                        let mut h = 0.0;
                        for (i, r) in (r0..r1).enumerate() {
                            if t <= start[r] {
                                break;
                            }
                            let end = if r == r1 - 1 && extrapolate == Extrapolate::Locf {
                                t
                            } else {
                                t.min(stop[r])
                            };
                            h += tree.cumhaz_at(leaves[i], end)
                                - tree.cumhaz_at(leaves[i], start[r]);
                        }
                        h
                    };
                    let h_origin = path_cumhaz(u);
                    for (j, &t) in times.iter().enumerate() {
                        let h = path_cumhaz(t) - h_origin;
                        match agg {
                            Aggregate::Hazard => row_out[j] += h,
                            Aggregate::Survival => {
                                let a = -h;
                                if a > max[j] {
                                    sum[j] = sum[j] * (max[j] - a).exp() + 1.0;
                                    max[j] = a;
                                } else {
                                    sum[j] += (a - max[j]).exp();
                                }
                            }
                        }
                    }
                }
                for (j, (o, &t)) in row_out.iter_mut().zip(times).enumerate() {
                    let undefined = t < u || (t > last_stop && extrapolate == Extrapolate::None);
                    *o = if undefined {
                        f64::NAN
                    } else {
                        match agg {
                            Aggregate::Hazard => *o / n_trees,
                            Aggregate::Survival => -(max[j] + (sum[j] / n_trees).ln()),
                        }
                    };
                }
            });
        out
    }

    /// Leaf index per (row, tree), row-major.
    pub fn apply(&self, x: &[f64], n_features: usize) -> Vec<u32> {
        let k = self.trees.len();
        let mut out = vec![0u32; x.len() / n_features * k];
        out.par_chunks_mut(k.max(1))
            .zip(x.par_chunks(n_features))
            .for_each(|(row_out, xr)| {
                for (o, tree) in row_out.iter_mut().zip(&self.trees) {
                    *o = tree.apply(xr) as u32;
                }
            });
        out
    }
}
