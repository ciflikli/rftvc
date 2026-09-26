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
    /// covariates hold from time 0, by the discrete Aalen–Johansen estimator
    /// (see `predict_cif_paths`; each row is a path of one row covering all times).
    ///
    /// Returns `(cif (n_rows, n_causes, T), surv (n_rows, T), n_clamped)`.
    /// `times` must not contain NaN.
    pub fn predict_cif(
        &self,
        x: &[f64],
        n_features: usize,
        times: &[f64],
        agg: CifAggregate,
    ) -> (Vec<f64>, Vec<f64>, usize) {
        let out = self.aj_rows(x, n_features, times, agg, &|_, _| true);
        (out.cif, out.surv, out.n_clamped)
    }

    /// Cumulative incidence, event-free survival and cause-specific cumulative
    /// hazard along covariate paths, conditional on being event-free at `origin`.
    ///
    /// Path `p` owns rows `offsets[p]..offsets[p+1]` (contiguous in time, sorted
    /// by `start`). Row `r` routes its own covariates; its leaf's per-cause
    /// increments `dL_j(v)` at grid times `v` in `(max(start_r, u), stop_r]`
    /// enter (the last row's interval is unbounded above under `Locf`). With
    /// `u = origin[p]`, the discrete Aalen–Johansen estimator gives
    /// `S(t | u) = prod_{u < v <= t} (1 - sum_j dL_j(v))` and
    /// `F_j(t | u) = sum_{u < v <= t} S(v- | u) dL_j(v)`; the hazard output is
    /// `L_j(t) - L_j(u)`. `Hazard` aggregation averages the increments over trees,
    /// then applies the estimator; `Cif` applies it per tree and averages `F`,
    /// `S` and the hazard. Values are NaN for `t < u`, and for `t` beyond the
    /// last `stop` unless `extrapolate` is `Locf`.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_cif_paths(
        &self,
        x: &[f64],
        n_features: usize,
        start: &[f64],
        stop: &[f64],
        offsets: &[usize],
        origin: &[f64],
        times: &[f64],
        agg: CifAggregate,
        extrapolate: Extrapolate,
    ) -> AjOutput {
        let paths = Paths {
            x,
            n_features,
            start,
            stop,
            offsets,
            origin,
            extrapolate,
        };
        self.aj_paths(&paths, times, agg, &|_, _| true)
    }

    /// Out-of-bag Aalen–Johansen per row (covariates fixed from time 0): the
    /// ensemble of trees whose bag contains none of the row's units
    /// `units[offsets[r]..offsets[r + 1]]` (as in `oob_mortality`). Rows with no
    /// such tree are NaN; `n_trees` gives each row's ensemble size.
    #[allow(clippy::too_many_arguments)]
    pub fn oob_cif(
        &self,
        x: &[f64],
        n_features: usize,
        offsets: &[usize],
        units: &[u32],
        times: &[f64],
        agg: CifAggregate,
    ) -> AjOutput {
        let in_bag = self.in_bag_bits();
        let keep = |row: usize, b: usize| {
            units[offsets[row]..offsets[row + 1]]
                .iter()
                .all(|&g| in_bag[b][g as usize / 64] & (1 << (g % 64)) == 0)
        };
        self.aj_rows(x, n_features, times, agg, &keep)
    }

    /// In-bag id bitsets per tree.
    fn in_bag_bits(&self) -> Vec<Vec<u64>> {
        let words = self.n_groups.div_ceil(64);
        (0..self.trees.len())
            .into_par_iter()
            .map(|b| {
                let mut bits = vec![0u64; words];
                for g in self.in_bag_ids(b) {
                    bits[g as usize / 64] |= 1 << (g % 64);
                }
                bits
            })
            .collect()
    }

    /// Each row as a path of one row with covariates in force over all times.
    fn aj_rows(
        &self,
        x: &[f64],
        n_features: usize,
        times: &[f64],
        agg: CifAggregate,
        keep: &(dyn Fn(usize, usize) -> bool + Sync),
    ) -> AjOutput {
        let n = x.len() / n_features;
        let (lo, hi) = (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]);
        let offsets: Vec<usize> = (0..=n).collect();
        let paths = Paths {
            x,
            n_features,
            start: &lo,
            stop: &hi,
            offsets: &offsets,
            origin: &lo,
            extrapolate: Extrapolate::None,
        };
        self.aj_paths(&paths, times, agg, keep)
    }

    /// The Aalen–Johansen kernel over paths; `keep(path, tree)` selects each
    /// path's ensemble.
    fn aj_paths(
        &self,
        paths: &Paths,
        times: &[f64],
        agg: CifAggregate,
        keep: &(dyn Fn(usize, usize) -> bool + Sync),
    ) -> AjOutput {
        let (m, nc) = (times.len(), self.n_causes);
        let n_paths = paths.offsets.len().saturating_sub(1);
        let mut out = AjOutput {
            cif: vec![0.0; n_paths * nc * m],
            surv: vec![0.0; n_paths * m],
            cumhaz: vec![0.0; n_paths * nc * m],
            n_trees: vec![0; n_paths],
            n_clamped: 0,
        };
        let Some(first) = self.trees.first() else {
            out.cif.fill(f64::NAN);
            out.surv.fill(f64::NAN);
            out.cumhaz.fill(f64::NAN);
            return out;
        };
        let grid = Arc::clone(&first.grid_times);
        // Requested times in increasing order (ties kept), to merge with the sweep.
        let mut t_order: Vec<usize> = (0..m).collect();
        t_order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
        let sweep = Sweep {
            grid: &grid,
            times,
            t_order: &t_order,
            n_causes: nc,
        };
        let (cw, sw) = ((nc * m).max(1), m.max(1));
        out.n_clamped = out
            .cif
            .par_chunks_mut(cw)
            .zip(out.surv.par_chunks_mut(sw))
            .zip(out.cumhaz.par_chunks_mut(cw))
            .zip(out.n_trees.par_iter_mut())
            .enumerate()
            .map_init(
                || AjBuffer::new(grid.len(), nc),
                |buf, (p, (((cif, surv), cumhaz), n_used))| {
                    let (r0, r1) = (paths.offsets[p], paths.offsets[p + 1]);
                    let u = paths.origin[p];
                    let locf = paths.extrapolate == Extrapolate::Locf;
                    let mut clamped = 0;
                    let mut n = 0usize;
                    for (b, tree) in self.trees.iter().enumerate() {
                        if !keep(p, b) {
                            continue;
                        }
                        n += 1;
                        for r in r0..r1 {
                            let xr = &paths.x[r * paths.n_features..(r + 1) * paths.n_features];
                            let lo = paths.start[r].max(u);
                            let hi = if r == r1 - 1 && locf {
                                f64::INFINITY
                            } else {
                                paths.stop[r]
                            };
                            if lo < hi {
                                buf.add_leaf(tree, tree.apply(xr), lo, hi);
                            }
                        }
                        if agg == CifAggregate::Cif {
                            clamped += sweep.run(buf, 1.0, cif, surv, cumhaz);
                        }
                    }
                    *n_used = n as u32;
                    if n == 0 {
                        cif.fill(f64::NAN);
                        surv.fill(f64::NAN);
                        cumhaz.fill(f64::NAN);
                        return 0;
                    }
                    match agg {
                        CifAggregate::Hazard => {
                            clamped += sweep.run(buf, n as f64, cif, surv, cumhaz)
                        }
                        CifAggregate::Cif => {
                            let nf = n as f64;
                            for v in cif
                                .iter_mut()
                                .chain(surv.iter_mut())
                                .chain(cumhaz.iter_mut())
                            {
                                *v /= nf;
                            }
                        }
                    }
                    let last_stop = paths.stop[r1 - 1];
                    for (ti, &t) in times.iter().enumerate() {
                        if t < u || (t > last_stop && !locf) {
                            surv[ti] = f64::NAN;
                            for j in 0..nc {
                                cif[j * m + ti] = f64::NAN;
                                cumhaz[j * m + ti] = f64::NAN;
                            }
                        }
                    }
                    clamped
                },
            )
            .sum();
        out
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
        let in_bag = self.in_bag_bits();
        let n_rows = offsets.len() - 1;
        let (mut out, mut n_oob) = (vec![0.0; n_rows], vec![0u32; n_rows]);
        out.par_iter_mut()
            .zip(n_oob.par_iter_mut())
            .zip(x.par_chunks(n_features))
            .zip(offsets.par_windows(2))
            .for_each_init(
                || vec![0.0; times.len()],
                |buf, (((o, k), xr), w)| {
                    let oob = self.oob_trees(&in_bag, &units[w[0]..w[1]]);
                    *k = ensemble_cumhaz(oob, xr, times, agg, buf) as u32;
                    *o = if *k == 0 { f64::NAN } else { buf.iter().sum() };
                },
            );
        (out, n_oob)
    }

    /// Out-of-bag ensemble cumulative hazard per row, `(n_rows, times.len())`
    /// row-major, and each row's out-of-bag tree count; the same ensemble and
    /// aggregation as `oob_mortality` (whose value is the row's sum). Rows with
    /// no out-of-bag tree are NaN.
    pub fn oob_cumhaz(
        &self,
        x: &[f64],
        n_features: usize,
        offsets: &[usize],
        units: &[u32],
        times: &[f64],
        agg: Aggregate,
    ) -> (Vec<f64>, Vec<u32>) {
        let in_bag = self.in_bag_bits();
        let (n_rows, m) = (offsets.len() - 1, times.len());
        let (mut out, mut n_oob) = (vec![0.0; n_rows * m], vec![0u32; n_rows]);
        if m == 0 {
            return (out, n_oob);
        }
        out.par_chunks_mut(m)
            .zip(n_oob.par_iter_mut())
            .zip(x.par_chunks(n_features))
            .zip(offsets.par_windows(2))
            .for_each(|(((o, k), xr), w)| {
                let oob = self.oob_trees(&in_bag, &units[w[0]..w[1]]);
                *k = ensemble_cumhaz(oob, xr, times, agg, o) as u32;
                if *k == 0 {
                    o.fill(f64::NAN);
                }
            });
        (out, n_oob)
    }

    /// Out-of-bag per-cause cumulative hazards per row, `(n_rows, n_causes,
    /// times.len())` row-major: the out-of-bag twin of `predict_cause_cumhaz`
    /// (tree-averaged). Rows with no out-of-bag tree are NaN.
    pub fn oob_cause_cumhaz(
        &self,
        x: &[f64],
        n_features: usize,
        offsets: &[usize],
        units: &[u32],
        times: &[f64],
    ) -> (Vec<f64>, Vec<u32>) {
        let in_bag = self.in_bag_bits();
        let (n_rows, m, nc) = (offsets.len() - 1, times.len(), self.n_causes);
        let (mut out, mut n_oob) = (vec![0.0; n_rows * nc * m], vec![0u32; n_rows]);
        if m == 0 {
            return (out, n_oob);
        }
        out.par_chunks_mut(nc * m)
            .zip(n_oob.par_iter_mut())
            .zip(x.par_chunks(n_features))
            .zip(offsets.par_windows(2))
            .for_each(|(((row_out, k), xr), w)| {
                let mut n = 0u32;
                for tree in self.oob_trees(&in_bag, &units[w[0]..w[1]]) {
                    let leaf = tree.apply(xr);
                    for (j, out_j) in row_out.chunks_mut(m).enumerate() {
                        for (o, &t) in out_j.iter_mut().zip(times) {
                            *o += tree.cause_cumhaz_at(leaf, t, j);
                        }
                    }
                    n += 1;
                }
                *k = n;
                if n == 0 {
                    row_out.fill(f64::NAN);
                } else {
                    row_out.iter_mut().for_each(|v| *v /= n as f64);
                }
            });
        (out, n_oob)
    }

    /// Trees whose bag contains none of `row_units`.
    fn oob_trees<'a>(
        &'a self,
        in_bag: &'a [Vec<u64>],
        row_units: &'a [u32],
    ) -> impl Iterator<Item = &'a Tree> + 'a {
        self.trees
            .iter()
            .zip(in_bag)
            .filter(move |(_, bits)| {
                row_units
                    .iter()
                    .all(|&g| bits[g as usize / 64] & (1 << (g % 64)) == 0)
            })
            .map(|(t, _)| t)
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

/// Ensemble rule for competing-risks predictions (cr-design.md C4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CifAggregate {
    /// Average the cause-specific hazard increments over trees, then Aalen–Johansen.
    Hazard,
    /// Aalen–Johansen per tree, then average `F`, `S` (and the hazard).
    Cif,
}

/// Row-major outputs of the Aalen–Johansen kernel: `cif` and `cumhaz` are
/// `(n_paths, n_causes, T)`, `surv` is `(n_paths, T)`.
#[derive(Clone, Debug, PartialEq)]
pub struct AjOutput {
    pub cif: Vec<f64>,
    pub surv: Vec<f64>,
    pub cumhaz: Vec<f64>,
    /// Trees in each path's ensemble.
    pub n_trees: Vec<u32>,
    /// Grid points at which `1 - sum_j dL_j < 0` was set to 0 (only rounding
    /// can cause it: every leaf has `sum_j dL_j = d / y <= 1`).
    pub n_clamped: usize,
}

struct Paths<'a> {
    x: &'a [f64],
    n_features: usize,
    start: &'a [f64],
    stop: &'a [f64],
    offsets: &'a [usize],
    origin: &'a [f64],
    extrapolate: Extrapolate,
}

/// Per-thread increments on the grid: a dense `K x J` buffer plus the list of
/// touched grid points, so a path costs its leaf entries, not `K J`.
struct AjBuffer {
    inc: Vec<f64>,
    seen: Vec<bool>,
    touched: Vec<u32>,
    n_causes: usize,
}

impl AjBuffer {
    fn new(k: usize, n_causes: usize) -> AjBuffer {
        AjBuffer {
            inc: vec![0.0; k * n_causes],
            seen: vec![false; k],
            touched: Vec::new(),
            n_causes,
        }
    }

    /// Add the leaf's per-cause increments at its event times in `(lo, hi]`.
    fn add_leaf(&mut self, tree: &Tree, leaf: usize, lo: f64, hi: f64) {
        let nc = self.n_causes;
        let r = tree.entries(leaf);
        let idx = &tree.event_idx[r.clone()];
        let grid = &tree.grid_times;
        let first = r.start + idx.partition_point(|&k| grid[k as usize] <= lo);
        for e in first..r.end {
            let k = tree.event_idx[e] as usize;
            if grid[k] > hi {
                break;
            }
            if !self.seen[k] {
                self.seen[k] = true;
                self.touched.push(k as u32);
            }
            for j in 0..nc {
                let prev = if e == r.start {
                    0.0
                } else {
                    tree.cumhaz[(e - 1) * nc + j]
                };
                self.inc[k * nc + j] += tree.cumhaz[e * nc + j] - prev;
            }
        }
    }
}

/// The Aalen–Johansen sweep over touched grid points, merged with the requested times.
struct Sweep<'a> {
    grid: &'a [f64],
    times: &'a [f64],
    t_order: &'a [usize],
    n_causes: usize,
}

impl Sweep<'_> {
    /// Divide the buffered increments by `div`, run the estimator, add the
    /// values at the requested times into `cif`, `surv` and `cumhaz`, and
    /// reset the buffer. Returns the number of clamped grid points.
    fn run(
        &self,
        buf: &mut AjBuffer,
        div: f64,
        cif: &mut [f64],
        surv: &mut [f64],
        cumhaz: &mut [f64],
    ) -> usize {
        let (m, nc) = (self.times.len(), self.n_causes);
        buf.touched.sort_unstable();
        let (mut s, mut f, mut h) = (1.0, vec![0.0; nc], vec![0.0; nc]);
        let (mut clamped, mut next) = (0usize, 0usize);
        // Writes the current state at every remaining requested time below
        // `upto` (all remaining ones for `None`, including `+inf`).
        let mut write = |upto: Option<f64>, s: f64, f: &[f64], h: &[f64], next: &mut usize| {
            while *next < m && upto.is_none_or(|u| self.times[self.t_order[*next]] < u) {
                let ti = self.t_order[*next];
                surv[ti] += s;
                for j in 0..nc {
                    cif[j * m + ti] += f[j];
                    cumhaz[j * m + ti] += h[j];
                }
                *next += 1;
            }
        };
        for &k in buf.touched.iter() {
            let k = k as usize;
            // Requested times before grid point k see the state before it.
            write(Some(self.grid[k]), s, &f, &h, &mut next);
            let d = &mut buf.inc[k * nc..(k + 1) * nc];
            let mut total = 0.0;
            for ((fj, hj), dj) in f.iter_mut().zip(h.iter_mut()).zip(d.iter_mut()) {
                *dj /= div;
                *fj += s * *dj;
                *hj += *dj;
                total += *dj;
            }
            let step = 1.0 - total;
            if step < 0.0 {
                clamped += 1;
            }
            s *= step.max(0.0);
            d.iter_mut().for_each(|v| *v = 0.0);
            buf.seen[k] = false;
        }
        write(None, s, &f, &h, &mut next);
        buf.touched.clear();
        clamped
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
