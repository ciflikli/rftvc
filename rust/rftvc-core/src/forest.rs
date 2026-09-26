//! Forest: whole-id resampling, parallel tree building, aggregated prediction.

use std::sync::Arc;

use rayon::prelude::*;

use crate::criterion::SplitCriterion;
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
    /// Causes `J` of the leaf hazards (1 for single-event survival).
    pub n_causes: usize,
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
    criterion: &dyn SplitCriterion,
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
            build_tree(binned, surv, rows, units, &params.tree, criterion, &mut rng)
        })
        .collect();
    Forest {
        trees,
        tree_seeds,
        n_features: binned.n_features,
        n_groups: groups.len(),
        n_draw: params.n_draw,
        bootstrap: params.bootstrap,
        n_causes: surv.n_causes,
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
        let mut out = vec![0.0; x.len() / n_features * m];
        out.par_chunks_mut(m.max(1))
            .zip(x.par_chunks(n_features))
            .for_each(|(row_out, xr)| {
                ensemble_cumhaz(self.trees.iter(), xr, times, agg, row_out);
            });
        out
    }

    /// Ensemble per-cause cumulative hazards (mean over trees), row-major
    /// `(n_rows, n_causes, times.len())`. Accumulated in the order of
    /// `predict_cumhaz`, so for one cause the two are bit-identical under `Hazard`.
    pub fn predict_cause_cumhaz(&self, x: &[f64], n_features: usize, times: &[f64]) -> Vec<f64> {
        let (m, nc) = (times.len(), self.n_causes);
        let n = self.trees.len() as f64;
        let mut out = vec![0.0; x.len() / n_features * nc * m];
        out.par_chunks_mut((nc * m).max(1))
            .zip(x.par_chunks(n_features))
            .for_each(|(row_out, xr)| {
                for tree in &self.trees {
                    let leaf = tree.apply(xr);
                    for (j, out_j) in row_out.chunks_mut(m.max(1)).enumerate() {
                        for (o, &t) in out_j.iter_mut().zip(times) {
                            *o += tree.cause_cumhaz_at(leaf, t, j);
                        }
                    }
                }
                row_out.iter_mut().for_each(|v| *v /= n);
            });
        out
    }

    /// Cumulative incidence per cause and event-free survival for rows whose
    /// covariates hold from time 0, by the discrete Aalen–Johansen estimator on
    /// the ensemble (tree-averaged) cause-specific hazard increments:
    /// `F_j(t) = sum_{v <= t} S(v-) dL_j(v)`, `S(t) = prod_{v <= t} (1 - sum_j dL_j(v))`.
    ///
    /// Returns `(cif (n_rows, n_causes, T), surv (n_rows, T), n_clamped)`, where
    /// `n_clamped` counts the grid points at which `1 - sum_j dL_j < 0` was set
    /// to 0 (only rounding can cause it: every leaf has `sum_j dL_j = d / y <= 1`).
    ///
    /// Per row, each tree's leaf increments are added into a dense `K x J`
    /// buffer; only the touched grid points are swept and reset. `times` must
    /// not contain NaN.
    pub fn predict_cif(
        &self,
        x: &[f64],
        n_features: usize,
        times: &[f64],
    ) -> (Vec<f64>, Vec<f64>, usize) {
        let (m, nc) = (times.len(), self.n_causes);
        let n_rows = x.len() / n_features;
        let grid = match self.trees.first() {
            Some(t) => Arc::clone(&t.grid_times),
            None => {
                return (
                    vec![f64::NAN; n_rows * nc * m],
                    vec![f64::NAN; n_rows * m],
                    0,
                );
            }
        };
        let kg = grid.len();
        let n_trees = self.trees.len() as f64;
        // Requested times in increasing order (ties kept), to merge with the sweep.
        let mut t_order: Vec<usize> = (0..m).collect();
        t_order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
        let (mut cif, mut surv) = (vec![0.0; n_rows * nc * m], vec![0.0; n_rows * m]);
        let n_clamped = cif
            .par_chunks_mut((nc * m).max(1))
            .zip(surv.par_chunks_mut(m.max(1)))
            .zip(x.par_chunks(n_features))
            .map_init(
                || (vec![0.0; kg * nc], vec![false; kg], Vec::<u32>::new()),
                |(inc, seen, touched), ((cif_row, surv_row), xr)| {
                    for tree in &self.trees {
                        let leaf = tree.apply(xr);
                        let r = tree.entries(leaf);
                        for e in r.clone() {
                            let k = tree.event_idx[e] as usize;
                            if !seen[k] {
                                seen[k] = true;
                                touched.push(k as u32);
                            }
                            for j in 0..nc {
                                let prev = if e == r.start {
                                    0.0
                                } else {
                                    tree.cumhaz[(e - 1) * nc + j]
                                };
                                inc[k * nc + j] += tree.cumhaz[e * nc + j] - prev;
                            }
                        }
                    }
                    touched.sort_unstable();
                    let (mut s, mut f) = (1.0, vec![0.0; nc]);
                    let (mut clamped, mut next) = (0usize, 0usize);
                    let mut write = |upto: f64, s: f64, f: &[f64], next: &mut usize| {
                        while *next < m && times[t_order[*next]] < upto {
                            let ti = t_order[*next];
                            surv_row[ti] = s;
                            for j in 0..nc {
                                cif_row[j * m + ti] = f[j];
                            }
                            *next += 1;
                        }
                    };
                    for &k in touched.iter() {
                        let k = k as usize;
                        // Requested times before grid point k see the state before it.
                        write(grid[k], s, &f, &mut next);
                        let d = &mut inc[k * nc..(k + 1) * nc];
                        let mut total = 0.0;
                        for (fj, dj) in f.iter_mut().zip(d.iter_mut()) {
                            *dj /= n_trees;
                            *fj += s * *dj;
                            total += *dj;
                        }
                        let step = 1.0 - total;
                        if step < 0.0 {
                            clamped += 1;
                        }
                        s *= step.max(0.0);
                        d.iter_mut().for_each(|v| *v = 0.0);
                        seen[k] = false;
                    }
                    write(f64::INFINITY, s, &f, &mut next);
                    debug_assert_eq!(next, m, "times must not be NaN");
                    touched.clear();
                    clamped
                },
            )
            .sum();
        (cif, surv, n_clamped)
    }

    /// Out-of-bag ensemble mortality per row: `sum_k Λ_oob(t_k | x_row)`, and
    /// the number of trees in each row's out-of-bag ensemble.
    ///
    /// Row `r` must be out of bag in units `units[offsets[r]..offsets[r + 1]]`
    /// (its own unit, plus any buffer units); only trees whose bag contains none
    /// of them enter its ensemble. NaN when there is no such tree.
    pub fn oob_mortality(
        &self,
        x: &[f64],
        n_features: usize,
        offsets: &[usize],
        units: &[u32],
        times: &[f64],
        agg: Aggregate,
    ) -> (Vec<f64>, Vec<u32>) {
        let words = self.n_groups.div_ceil(64);
        let in_bag: Vec<Vec<u64>> = (0..self.trees.len())
            .into_par_iter()
            .map(|b| {
                let mut bits = vec![0u64; words];
                for g in self.in_bag_ids(b) {
                    bits[g as usize / 64] |= 1 << (g % 64);
                }
                bits
            })
            .collect();
        let n_rows = offsets.len() - 1;
        let (mut out, mut n_oob) = (vec![0.0; n_rows], vec![0u32; n_rows]);
        out.par_iter_mut()
            .zip(n_oob.par_iter_mut())
            .zip(x.par_chunks(n_features))
            .zip(offsets.par_windows(2))
            .for_each_init(
                || vec![0.0; times.len()],
                |buf, (((o, k), xr), w)| {
                    let row_units = &units[w[0]..w[1]];
                    let oob = self
                        .trees
                        .iter()
                        .zip(&in_bag)
                        .filter(|(_, bits)| {
                            row_units
                                .iter()
                                .all(|&g| bits[g as usize / 64] & (1 << (g % 64)) == 0)
                        })
                        .map(|(t, _)| t);
                    *k = ensemble_cumhaz(oob, xr, times, agg, buf) as u32;
                    *o = if *k == 0 { f64::NAN } else { buf.iter().sum() };
                },
            );
        (out, n_oob)
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

    /// Allocated bytes of all trees' node and leaf arrays, plus one copy of the grid.
    pub fn nbytes(&self) -> usize {
        self.trees.iter().map(Tree::nbytes).sum::<usize>()
            + self
                .trees
                .first()
                .map_or(0, |t| t.grid_times.capacity() * std::mem::size_of::<f64>())
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

/// Aggregate the cumulative hazard of `x` over `trees` into `out` (one value
/// per time); returns the number of trees. Leaves `out` zeroed for no trees.
fn ensemble_cumhaz<'a>(
    trees: impl Iterator<Item = &'a Tree>,
    x: &[f64],
    times: &[f64],
    agg: Aggregate,
    out: &mut [f64],
) -> usize {
    let m = times.len();
    out.iter_mut().for_each(|v| *v = 0.0);
    let mut n = 0usize;
    match agg {
        Aggregate::Hazard => {
            for tree in trees {
                let leaf = tree.apply(x);
                for (o, &t) in out.iter_mut().zip(times) {
                    *o += tree.cumhaz_at(leaf, t);
                }
                n += 1;
            }
            if n > 0 {
                out.iter_mut().for_each(|v| *v /= n as f64);
            }
        }
        Aggregate::Survival => {
            // -log(mean_b exp(-Λ_b)) via an online log-sum-exp, so large
            // hazards stay finite instead of underflowing exp() to 0.
            let mut max = vec![f64::NEG_INFINITY; m];
            let mut sum = vec![0.0; m];
            for tree in trees {
                let leaf = tree.apply(x);
                for (j, &t) in times.iter().enumerate() {
                    let a = -tree.cumhaz_at(leaf, t);
                    if a > max[j] {
                        sum[j] = sum[j] * (max[j] - a).exp() + 1.0;
                        max[j] = a;
                    } else {
                        sum[j] += (a - max[j]).exp();
                    }
                }
                n += 1;
            }
            if n > 0 {
                for (j, o) in out.iter_mut().enumerate() {
                    *o = -(max[j] + (sum[j] / n as f64).ln());
                }
            }
        }
    }
    n
}
