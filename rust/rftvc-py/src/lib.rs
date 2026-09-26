//! Python bindings: `rftvc._core`.

use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rftvc_core::{
    Aggregate, AjOutput, Binned, CifAggregate, CompositeCauseLogRank, Extrapolate, FlatForest,
    Forest, ForestParams, Grid, Groups, LtrcLogRank, Profile, SingleCause, SplitCriterion,
    SplitParams, SurvData, TreeParams, best_split as core_best_split, cause_profile_on, coarsen,
    exposure_of, fit_forest, node_profile, profile_on,
};

/// Contiguous 1-d input as a Vec. Strided views (e.g. a field of a structured
/// array, whose stride need not be a multiple of the item size) are rejected:
/// reading them through ndarray is not reliable.
fn vec1<T: numpy::Element + Copy>(a: &PyReadonlyArray1<T>, name: &str) -> PyResult<Vec<T>> {
    a.as_slice().map(|s| s.to_vec()).map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be a contiguous array (use numpy.ascontiguousarray)"
        ))
    })
}

/// C-contiguous 2-d input as a row-major Vec plus its shape.
fn matrix(x: &PyReadonlyArray2<f64>) -> PyResult<(Vec<f64>, usize, usize)> {
    let (n, p) = x.as_array().dim();
    // `as_slice` also accepts Fortran order, which would be misread as row-major.
    let v = x
        .as_slice()
        .ok()
        .filter(|_| x.is_c_contiguous())
        .ok_or_else(|| PyValueError::new_err("X must be a C-contiguous float64 array"))?;
    Ok((v.to_vec(), n, p))
}

/// Response arrays must all have `n` entries (`n = X.shape[0]` where X is given).
fn check_lengths(n: usize, arrays: &[(&str, usize)]) -> PyResult<()> {
    for (name, len) in arrays {
        if *len != n {
            return Err(PyValueError::new_err(format!(
                "{name} has {len} entries, expected {n}"
            )));
        }
    }
    Ok(())
}

fn surv_data(
    start: &PyReadonlyArray1<f64>,
    stop: &PyReadonlyArray1<f64>,
    event: &PyReadonlyArray1<bool>,
) -> PyResult<SurvData> {
    let s = vec1(start, "start")?;
    let t = vec1(stop, "stop")?;
    let e = vec1(event, "event")?;
    check_times(&s, &t)?;
    Ok(SurvData::new(&s, &t, &e))
}

/// Finite times with `start < stop` on every row (the grid sorts them, and a
/// row with no at-risk time cannot carry an event).
fn check_times(start: &[f64], stop: &[f64]) -> PyResult<()> {
    let ok = start
        .iter()
        .zip(stop)
        .all(|(a, b)| a.is_finite() && b.is_finite() && a < b);
    if start.len() != stop.len() || !ok {
        return Err(PyValueError::new_err(
            "start and stop must be finite with start < stop on every row",
        ));
    }
    Ok(())
}

fn pool(n_jobs: usize) -> PyResult<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_jobs.max(1))
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

fn aggregate(name: &str) -> PyResult<Aggregate> {
    match name {
        "hazard" => Ok(Aggregate::Hazard),
        "survival" => Ok(Aggregate::Survival),
        _ => Err(PyValueError::new_err(
            "aggregate must be 'hazard' or 'survival'",
        )),
    }
}

/// `(event_times, cumhaz (n_entries, n_causes))` of one leaf.
type LeafProfile<'py> = (Vec<f64>, Bound<'py, PyArray2<f64>>);

/// Cause-coded response: codes in `0..=n_causes`, `n_causes` in `1..=255`.
fn cause_data(
    start: &PyReadonlyArray1<f64>,
    stop: &PyReadonlyArray1<f64>,
    event: &PyReadonlyArray1<u8>,
    n_causes: usize,
) -> PyResult<SurvData> {
    let (s, t, e) = (
        vec1(start, "start")?,
        vec1(stop, "stop")?,
        vec1(event, "event")?,
    );
    if !(1..=255).contains(&n_causes) {
        return Err(PyValueError::new_err("n_causes must be in [1, 255]"));
    }
    check_times(&s, &t)?;
    if e.iter().any(|&c| c as usize > n_causes) {
        return Err(PyValueError::new_err(
            "event codes must be in [0, n_causes]",
        ));
    }
    Ok(SurvData::with_causes(&s, &t, &e, n_causes))
}

/// Split rule for `n_causes` causes (cr-plan.md P4a): one cause always uses
/// `LtrcLogRank`; otherwise `SingleCause` for `split_cause` (1-based) or the composite.
fn criterion_for(
    n_causes: usize,
    split_cause: Option<usize>,
    criterion: &str,
) -> PyResult<Box<dyn SplitCriterion>> {
    match (split_cause, criterion) {
        (Some(k), _) if !(1..=n_causes).contains(&k) => Err(PyValueError::new_err(
            "split_cause must be in [1, n_causes]",
        )),
        // S14 bake-off: "composite" won; the challengers were removed (P5).
        (_, c) if c != "composite" => Err(PyValueError::new_err("criterion must be 'composite'")),
        _ if n_causes == 1 => Ok(Box::new(LtrcLogRank)),
        (Some(k), _) => Ok(Box::new(SingleCause { cause: k - 1 })),
        (None, _) => Ok(Box::new(CompositeCauseLogRank)),
    }
}

fn cif_aggregate(name: &str) -> PyResult<CifAggregate> {
    match name {
        "hazard" => Ok(CifAggregate::Hazard),
        "cif" => Ok(CifAggregate::Cif),
        _ => Err(PyValueError::new_err("aggregate must be 'hazard' or 'cif'")),
    }
}

fn extrapolate_of(name: &str) -> PyResult<Extrapolate> {
    match name {
        "none" => Ok(Extrapolate::None),
        "locf" => Ok(Extrapolate::Locf),
        _ => Err(PyValueError::new_err(
            "extrapolate must be 'none' or 'locf'",
        )),
    }
}

/// Validated path inputs `(start, stop, offsets, origin)` for `n` rows.
type PathInputs = (Vec<f64>, Vec<f64>, Vec<usize>, Vec<f64>);

fn path_inputs(
    n: usize,
    start: &PyReadonlyArray1<f64>,
    stop: &PyReadonlyArray1<f64>,
    offsets: &PyReadonlyArray1<u64>,
    origin: &PyReadonlyArray1<f64>,
) -> PyResult<PathInputs> {
    check_lengths(
        n,
        &[
            ("start", start.as_array().len()),
            ("stop", stop.as_array().len()),
        ],
    )?;
    let offsets: Vec<usize> = vec1(offsets, "offsets")?
        .into_iter()
        .map(|o| o as usize)
        .collect();
    let n_paths = offsets.len().saturating_sub(1);
    let valid = offsets.first() == Some(&0)
        && offsets.last() == Some(&n)
        && offsets.windows(2).all(|w| w[0] < w[1]);
    if !valid || origin.as_array().len() != n_paths {
        return Err(PyValueError::new_err(
            "offsets must start at 0, strictly increase and end at n_rows; one origin per path",
        ));
    }
    Ok((
        vec1(start, "start")?,
        vec1(stop, "stop")?,
        offsets,
        vec1(origin, "origin")?,
    ))
}

/// CSR per-row OOB unit sets, validated against `n` rows and `n_groups` units.
fn oob_sets(
    n: usize,
    n_groups: usize,
    offsets: &PyReadonlyArray1<u64>,
    units: &PyReadonlyArray1<u32>,
) -> PyResult<(Vec<usize>, Vec<u32>)> {
    check_lengths(n + 1, &[("offsets", offsets.as_array().len())])?;
    let offsets: Vec<usize> = vec1(offsets, "offsets")?
        .into_iter()
        .map(|o: u64| o as usize)
        .collect();
    let units: Vec<u32> = vec1(units, "units")?;
    if offsets[0] != 0 || offsets.windows(2).any(|w| w[0] >= w[1]) || offsets[n] != units.len() {
        return Err(PyValueError::new_err(
            "offsets must start at 0, give each row at least one unit and end at len(units)",
        ));
    }
    if units.iter().any(|&g| g as usize >= n_groups) {
        return Err(PyValueError::new_err("units must be in [0, n_groups)"));
    }
    Ok((offsets, units))
}

type Array3Out<'py> = Bound<'py, PyArray3<f64>>;

/// `(cif, surv, cumhaz)` of a kernel output as `(n, J, T)`, `(n, T)`, `(n, J, T)` arrays.
fn aj_arrays<'py>(
    py: Python<'py>,
    out: AjOutput,
    n: usize,
    nc: usize,
    m: usize,
) -> (Array3Out<'py>, Bound<'py, PyArray2<f64>>, Array3Out<'py>) {
    (
        Array3::from_shape_vec((n, nc, m), out.cif)
            .expect("shape")
            .into_pyarray(py),
        Array2::from_shape_vec((n, m), out.surv)
            .expect("shape")
            .into_pyarray(py),
        Array3::from_shape_vec((n, nc, m), out.cumhaz)
            .expect("shape")
            .into_pyarray(py),
    )
}

fn times_vec(times: &PyReadonlyArray1<f64>) -> PyResult<Vec<f64>> {
    let t = vec1(times, "times")?;
    if t.iter().any(|v| v.is_nan()) {
        return Err(PyValueError::new_err("times must not contain NaN"));
    }
    Ok(t)
}

#[pyclass(module = "rftvc._core", name = "Forest", frozen)]
struct PyForest {
    inner: Forest,
}

impl PyForest {
    fn check_x(&self, x: &PyReadonlyArray2<f64>) -> PyResult<(Vec<f64>, usize, usize)> {
        let (v, n, p) = matrix(x)?;
        if p != self.inner.n_features {
            return Err(PyValueError::new_err(format!(
                "X has {p} features, expected {}",
                self.inner.n_features
            )));
        }
        Ok((v, n, p))
    }

    fn tree(&self, tree: usize) -> PyResult<&rftvc_core::Tree> {
        self.inner
            .trees
            .get(tree)
            .ok_or_else(|| PyValueError::new_err("tree index out of range"))
    }
}

#[pymethods]
impl PyForest {
    #[getter]
    fn n_trees(&self) -> usize {
        self.inner.trees.len()
    }

    #[getter]
    fn n_causes(&self) -> usize {
        self.inner.n_causes
    }

    fn n_leaves(&self, tree: usize) -> PyResult<usize> {
        Ok(self.tree(tree)?.n_leaves())
    }

    /// Allocated bytes of the fitted trees' node and leaf arrays, plus one copy of the grid.
    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    /// `(event_times, cumhaz)` of one leaf of one tree: the per-cause
    /// Nelson–Aalen cumulative hazards, shape `(n_event_times, n_causes)`, at the
    /// leaf's event times (of any cause).
    fn leaf_profile<'py>(
        &self,
        py: Python<'py>,
        tree: usize,
        leaf: usize,
    ) -> PyResult<LeafProfile<'py>> {
        let t = self.tree(tree)?;
        if leaf >= t.n_leaves() {
            return Err(PyValueError::new_err("leaf index out of range"));
        }
        let times: Vec<f64> = t
            .leaf_event_idx(leaf)
            .iter()
            .map(|&k| t.grid_times[k as usize])
            .collect();
        let cumhaz =
            Array2::from_shape_vec((times.len(), t.n_causes), t.leaf_cumhaz(leaf).to_vec())
                .expect("shape")
                .into_pyarray(py);
        Ok((times, cumhaz))
    }

    /// Id indices each tree was grown on (sorted; repeats under bootstrap).
    fn in_bag_ids<'py>(&self, py: Python<'py>, tree: usize) -> PyResult<Bound<'py, PyArray1<u32>>> {
        self.tree(tree)?;
        Ok(self.inner.in_bag_ids(tree).into_pyarray(py))
    }

    /// Leaf index per (row, tree).
    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        n_jobs: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let (v, n, p) = self.check_x(&x)?;
        let k = self.inner.trees.len();
        let forest = &self.inner;
        let pool = pool(n_jobs)?;
        let out = py.detach(|| pool.install(|| forest.apply(&v, p)));
        Ok(Array2::from_shape_vec((n, k), out)
            .expect("shape")
            .into_pyarray(py))
    }

    /// Ensemble cumulative hazard `(n_rows, n_times)` for rows starting at time 0.
    fn predict_cumhaz<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
        aggregate_by: &str,
        n_jobs: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (v, n, p) = self.check_x(&x)?;
        let agg = aggregate(aggregate_by)?;
        let times: Vec<f64> = vec1(&times, "times")?;
        let m = times.len();
        let forest = &self.inner;
        let pool = pool(n_jobs)?;
        let out = py.detach(|| pool.install(|| forest.predict_cumhaz(&v, p, &times, agg)));
        Ok(Array2::from_shape_vec((n, m), out)
            .expect("shape")
            .into_pyarray(py))
    }

    /// Ensemble per-cause cumulative hazards `(n_rows, n_causes, n_times)` for
    /// rows starting at time 0 (mean over trees).
    fn predict_cause_cumhaz<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
        n_jobs: usize,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let (v, n, p) = self.check_x(&x)?;
        let times = times_vec(&times)?;
        let (m, nc) = (times.len(), self.inner.n_causes);
        let forest = &self.inner;
        let pool = pool(n_jobs)?;
        let out = py.detach(|| pool.install(|| forest.predict_cause_cumhaz(&v, p, &times)));
        Ok(Array3::from_shape_vec((n, nc, m), out)
            .expect("shape")
            .into_pyarray(py))
    }

    /// Aalen–Johansen `(cif (n_rows, n_causes, n_times), surv (n_rows, n_times),
    /// n_clamped)` from the tree-averaged cause-specific hazards, for rows
    /// starting at time 0.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (x, times, n_jobs, aggregate_by="hazard"))]
    fn predict_cif<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
        n_jobs: usize,
        aggregate_by: &str,
    ) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray2<f64>>, usize)> {
        let (v, n, p) = self.check_x(&x)?;
        let agg = cif_aggregate(aggregate_by)?;
        let times = times_vec(&times)?;
        let (m, nc) = (times.len(), self.inner.n_causes);
        let forest = &self.inner;
        let pool = pool(n_jobs)?;
        let (cif, surv, clamped) =
            py.detach(|| pool.install(|| forest.predict_cif(&v, p, &times, agg)));
        Ok((
            Array3::from_shape_vec((n, nc, m), cif)
                .expect("shape")
                .into_pyarray(py),
            Array2::from_shape_vec((n, m), surv)
                .expect("shape")
                .into_pyarray(py),
            clamped,
        ))
    }

    /// Aalen–Johansen along covariate paths: `(cif (n_paths, J, T), surv
    /// (n_paths, T), cumhaz (n_paths, J, T), n_clamped)`, conditional on the
    /// origins. Rows of path `p` are `offsets[p]..offsets[p+1]`, contiguous and
    /// sorted by start.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn predict_cif_paths<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        start: PyReadonlyArray1<'py, f64>,
        stop: PyReadonlyArray1<'py, f64>,
        offsets: PyReadonlyArray1<'py, u64>,
        origin: PyReadonlyArray1<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
        aggregate_by: &str,
        extrapolate: &str,
        n_jobs: usize,
    ) -> PyResult<(
        Array3Out<'py>,
        Bound<'py, PyArray2<f64>>,
        Array3Out<'py>,
        usize,
    )> {
        let (v, n, p) = self.check_x(&x)?;
        let (start, stop, offsets, origin) = path_inputs(n, &start, &stop, &offsets, &origin)?;
        let agg = cif_aggregate(aggregate_by)?;
        let extrapolate = extrapolate_of(extrapolate)?;
        let times = times_vec(&times)?;
        let (m, nc, n_paths) = (times.len(), self.inner.n_causes, offsets.len() - 1);
        let forest = &self.inner;
        let pool = pool(n_jobs)?;
        let out = py.detach(|| {
            pool.install(|| {
                forest.predict_cif_paths(
                    &v,
                    p,
                    &start,
                    &stop,
                    &offsets,
                    &origin,
                    &times,
                    agg,
                    extrapolate,
                )
            })
        });
        let clamped = out.n_clamped;
        let (cif, surv, cumhaz) = aj_arrays(py, out, n_paths, nc, m);
        Ok((cif, surv, cumhaz, clamped))
    }

    /// Out-of-bag cumulative incidence `(n_rows, J, T)` per row (covariates
    /// fixed from time 0) and each row's ensemble size; rows must be out of bag
    /// in every unit of `units[offsets[r]..offsets[r+1]]`. NaN without a tree.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn oob_cif<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        offsets: PyReadonlyArray1<'py, u64>,
        units: PyReadonlyArray1<'py, u32>,
        times: PyReadonlyArray1<'py, f64>,
        aggregate_by: &str,
        n_jobs: usize,
    ) -> PyResult<(Array3Out<'py>, Bound<'py, PyArray1<u32>>)> {
        let (v, n, p) = self.check_x(&x)?;
        let (offsets, units) = oob_sets(n, self.inner.n_groups, &offsets, &units)?;
        let agg = cif_aggregate(aggregate_by)?;
        let times = times_vec(&times)?;
        let (m, nc) = (times.len(), self.inner.n_causes);
        let forest = &self.inner;
        let pool = pool(n_jobs)?;
        let out =
            py.detach(|| pool.install(|| forest.oob_cif(&v, p, &offsets, &units, &times, agg)));
        let n_trees = out.n_trees.clone();
        let (cif, _, _) = aj_arrays(py, out, n, nc, m);
        Ok((cif, n_trees.into_pyarray(py)))
    }

    /// Whether per-leaf per-cause in-bag event counts are stored.
    #[getter]
    fn has_leaf_cause_events(&self) -> bool {
        self.inner
            .trees
            .first()
            .is_some_and(|t| !t.leaf_cause_events.is_empty())
    }

    /// In-bag events of each cause per leaf of one tree, `(n_leaves, J)`.
    fn leaf_cause_events<'py>(
        &self,
        py: Python<'py>,
        tree: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let t = self.tree(tree)?;
        if t.leaf_cause_events.is_empty() {
            return Err(PyValueError::new_err(
                "this forest stores no per-leaf cause counts (fitted before S12 or by SurvivalForestTV)",
            ));
        }
        Ok(
            Array2::from_shape_vec((t.n_leaves(), t.n_causes), t.leaf_cause_events.clone())
                .expect("shape")
                .into_pyarray(py),
        )
    }

    /// Out-of-bag ensemble mortality per row and the size of its ensemble.
    ///
    /// Row `r` must be out of bag in every unit of
    /// `units[offsets[r]..offsets[r+1]]` (at least one). Mortality is NaN when
    /// no tree qualifies.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn oob_mortality<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        offsets: PyReadonlyArray1<'py, u64>,
        units: PyReadonlyArray1<'py, u32>,
        times: PyReadonlyArray1<'py, f64>,
        aggregate_by: &str,
        n_jobs: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u32>>)> {
        let (v, n, p) = self.check_x(&x)?;
        let (offsets, units) = oob_sets(n, self.inner.n_groups, &offsets, &units)?;
        let agg = aggregate(aggregate_by)?;
        let times: Vec<f64> = vec1(&times, "times")?;
        let forest = &self.inner;
        let pool = pool(n_jobs)?;
        let (out, n_oob) = py
            .detach(|| pool.install(|| forest.oob_mortality(&v, p, &offsets, &units, &times, agg)));
        Ok((out.into_pyarray(py), n_oob.into_pyarray(py)))
    }

    /// Conditional cumulative hazard along covariate paths `(n_paths, n_times)`.
    ///
    /// Rows of path `p` are `offsets[p]..offsets[p+1]`, contiguous and sorted by start.
    #[allow(clippy::too_many_arguments)]
    fn predict_paths<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        start: PyReadonlyArray1<'py, f64>,
        stop: PyReadonlyArray1<'py, f64>,
        offsets: PyReadonlyArray1<'py, u64>,
        origin: PyReadonlyArray1<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
        aggregate_by: &str,
        extrapolate: &str,
        n_jobs: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (v, n, p) = self.check_x(&x)?;
        let (start, stop, offsets, origin) = path_inputs(n, &start, &stop, &offsets, &origin)?;
        let n_paths = offsets.len() - 1;
        let agg = aggregate(aggregate_by)?;
        let extrapolate = extrapolate_of(extrapolate)?;
        let times: Vec<f64> = vec1(&times, "times")?;
        let m = times.len();
        let forest = &self.inner;
        let pool = pool(n_jobs)?;
        let out = py.detach(|| {
            pool.install(|| {
                forest.predict_paths(
                    &v,
                    p,
                    &start,
                    &stop,
                    &offsets,
                    &origin,
                    &times,
                    agg,
                    extrapolate,
                )
            })
        });
        Ok(Array2::from_shape_vec((n_paths, m), out)
            .expect("shape")
            .into_pyarray(py))
    }

    /// Pickle support: rebuild via `Forest._from_state(state)`.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let py = slf.py();
        let f = FlatForest::from_forest(&slf.get().inner);
        let d = PyDict::new(py);
        d.set_item("format_version", f.format_version)?;
        d.set_item("grid", f.grid.into_pyarray(py))?;
        d.set_item("tree_seeds", f.tree_seeds.into_pyarray(py))?;
        d.set_item("n_features", f.n_features)?;
        d.set_item("n_groups", f.n_groups)?;
        d.set_item("n_draw", f.n_draw)?;
        d.set_item("bootstrap", f.bootstrap)?;
        d.set_item("node_offsets", f.node_offsets.into_pyarray(py))?;
        d.set_item("node_feature", f.node_feature.into_pyarray(py))?;
        d.set_item("node_threshold", f.node_threshold.into_pyarray(py))?;
        d.set_item("node_left", f.node_left.into_pyarray(py))?;
        d.set_item("node_right", f.node_right.into_pyarray(py))?;
        d.set_item("leaf_offsets", f.leaf_offsets.into_pyarray(py))?;
        d.set_item("event_offsets", f.event_offsets.into_pyarray(py))?;
        d.set_item("event_idx", f.event_idx.into_pyarray(py))?;
        d.set_item("cumhaz", f.cumhaz.into_pyarray(py))?;
        d.set_item("n_causes", f.n_causes)?;
        // Only forests that stored counts carry the key, so other pickles are unchanged.
        if !f.leaf_cause_events.is_empty() {
            d.set_item("leaf_cause_events", f.leaf_cause_events.into_pyarray(py))?;
        }
        Ok((slf.getattr("_from_state")?, (d,)))
    }

    #[staticmethod]
    fn _from_state(state: &Bound<'_, PyDict>) -> PyResult<PyForest> {
        fn item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
            d.get_item(key)?
                .ok_or_else(|| PyValueError::new_err(format!("forest state missing '{key}'")))
        }
        macro_rules! vec_of {
            ($key:expr, $t:ty) => {
                vec1(&item(state, $key)?.extract::<PyReadonlyArray1<$t>>()?, $key)?
            };
        }
        if !state.contains("format_version")? {
            return Err(PyValueError::new_err(
                "forest state from an older rftvc build (pre-S9 leaf format); refit the model",
            ));
        }
        let format_version: u64 = item(state, "format_version")?.extract()?;
        // v2 states (before S11) are single-event and carry no `n_causes`.
        let n_causes: u64 = match state.get_item("n_causes")? {
            Some(v) => v.extract()?,
            None if format_version == 2 => 1,
            None => return Err(PyValueError::new_err("forest state missing 'n_causes'")),
        };
        let flat = FlatForest {
            format_version,
            grid: vec_of!("grid", f64),
            tree_seeds: vec_of!("tree_seeds", u64),
            n_features: item(state, "n_features")?.extract()?,
            n_groups: item(state, "n_groups")?.extract()?,
            n_draw: item(state, "n_draw")?.extract()?,
            bootstrap: item(state, "bootstrap")?.extract()?,
            node_offsets: vec_of!("node_offsets", u64),
            node_feature: vec_of!("node_feature", i64),
            node_threshold: vec_of!("node_threshold", f64),
            node_left: vec_of!("node_left", u32),
            node_right: vec_of!("node_right", u32),
            leaf_offsets: vec_of!("leaf_offsets", u64),
            event_offsets: vec_of!("event_offsets", u64),
            event_idx: vec_of!("event_idx", u32),
            cumhaz: vec_of!("cumhaz", f64),
            n_causes,
            leaf_cause_events: if state.contains("leaf_cause_events")? {
                // Present means stored: an empty array is corrupt, not "absent".
                let v = vec_of!("leaf_cause_events", u32);
                if v.is_empty() {
                    return Err(PyValueError::new_err(
                        "forest state has an empty 'leaf_cause_events'",
                    ));
                }
                v
            } else {
                Vec::new()
            },
        };
        let inner = flat.to_forest().map_err(PyValueError::new_err)?;
        Ok(PyForest { inner })
    }
}

#[pyfunction]
#[pyo3(name = "fit_forest", signature = (
    x, start, stop, event, groups, n_groups, *, n_trees, n_draw, bootstrap,
    max_depth, min_ids_leaf, min_events_leaf, max_features, max_bins, seed, n_jobs,
    n_causes=1, split_cause=None, min_events_leaf_cause=None, leaf_events=false, criterion="composite"
))]
#[allow(clippy::too_many_arguments)]
fn fit_forest_py(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    start: PyReadonlyArray1<'_, f64>,
    stop: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, u8>,
    groups: PyReadonlyArray1<'_, u32>,
    n_groups: usize,
    n_trees: usize,
    n_draw: usize,
    bootstrap: bool,
    max_depth: Option<usize>,
    min_ids_leaf: usize,
    min_events_leaf: usize,
    max_features: usize,
    max_bins: usize,
    seed: u64,
    n_jobs: usize,
    n_causes: usize,
    split_cause: Option<usize>,
    min_events_leaf_cause: Option<usize>,
    leaf_events: bool,
    criterion: &str,
) -> PyResult<PyForest> {
    let (v, n, p) = matrix(&x)?;
    check_lengths(
        n,
        &[
            ("start", start.as_array().len()),
            ("stop", stop.as_array().len()),
            ("event", event.as_array().len()),
            ("groups", groups.as_array().len()),
        ],
    )?;
    if !(2..=256).contains(&max_bins) {
        return Err(PyValueError::new_err("max_bins must be in [2, 256]"));
    }
    if p == 0 {
        return Err(PyValueError::new_err("X must have at least one feature"));
    }
    let groups: Vec<u32> = vec1(&groups, "groups")?;
    if n_groups == 0 || groups.iter().any(|&g| g as usize >= n_groups) {
        return Err(PyValueError::new_err("groups must be in [0, n_groups)"));
    }
    if n_trees == 0 || n_draw == 0 || (!bootstrap && n_draw > n_groups) {
        return Err(PyValueError::new_err(
            "need n_trees >= 1 and 1 <= n_draw (<= n_groups without bootstrap)",
        ));
    }
    let surv = cause_data(&start, &stop, &event, n_causes)?;
    let criterion = criterion_for(n_causes, split_cause, criterion)?;
    let cause_floor = match (split_cause, min_events_leaf_cause) {
        (_, None) => None,
        (Some(k), Some(m)) => Some((k - 1, m)),
        (None, Some(_)) => {
            return Err(PyValueError::new_err(
                "min_events_leaf_cause requires split_cause",
            ));
        }
    };
    let params = ForestParams {
        tree: TreeParams {
            max_depth,
            min_ids_leaf,
            min_events_leaf,
            max_features: max_features.clamp(1, p),
            cause_floor,
            leaf_events,
        },
        n_trees,
        n_draw,
        bootstrap,
        seed,
    };
    let pool = pool(n_jobs)?;
    let inner = py.detach(|| {
        pool.install(|| {
            let binned = Binned::fit(&v, n, p, max_bins);
            fit_forest(
                &binned,
                &surv,
                &Groups::new(&groups, n_groups),
                &params,
                criterion.as_ref(),
            )
        })
    });
    Ok(PyForest { inner })
}

type CoarsenOut<'py> = (
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<f64>>,
    usize,
);

/// Coarse mode (D8): quantile event grid of `ntime` points, snapping grid =
/// earliest start + event grid, then the row transform of `coarsen`.
///
/// Returns `(kept, start, stop, event, event_grid, lost_events)`.
#[pyfunction]
#[pyo3(name = "coarsen")]
fn coarsen_py<'py>(
    py: Python<'py>,
    start: PyReadonlyArray1<'py, f64>,
    stop: PyReadonlyArray1<'py, f64>,
    event: PyReadonlyArray1<'py, u8>,
    order: PyReadonlyArray1<'py, u32>,
    offsets: PyReadonlyArray1<'py, u64>,
    ntime: usize,
) -> PyResult<CoarsenOut<'py>> {
    let (s, t, e) = (
        vec1(&start, "start")?,
        vec1(&stop, "stop")?,
        vec1(&event, "event")?,
    );
    let n = s.len();
    check_lengths(
        n,
        &[
            ("stop", t.len()),
            ("event", e.len()),
            ("order", order.as_array().len()),
        ],
    )?;
    let order = vec1(&order, "order")?;
    let offsets: Vec<usize> = vec1(&offsets, "offsets")?
        .into_iter()
        .map(|o| o as usize)
        .collect();
    let mut seen = vec![false; n];
    let perm = order
        .iter()
        .all(|&r| (r as usize) < n && !std::mem::replace(&mut seen[r as usize], true));
    let valid = perm
        && offsets.first() == Some(&0)
        && offsets.last() == Some(&n)
        && offsets.windows(2).all(|w| w[0] < w[1]);
    if !valid || ntime == 0 || !e.iter().any(|&x| x != 0) {
        return Err(PyValueError::new_err(
            "need a row permutation `order`, chain offsets from 0 to n_rows, ntime >= 1 and an event",
        ));
    }
    // Event relocation needs each chain in time order with contiguous rows.
    let chained = offsets.windows(2).all(|w| {
        order[w[0]..w[1]]
            .windows(2)
            .all(|p| t[p[0] as usize] == s[p[1] as usize])
    });
    if !chained {
        return Err(PyValueError::new_err(
            "each chain's rows must be in time order and contiguous (stop == next start)",
        ));
    }
    let grid = Grid::quantile(&t, &e, ntime);
    let origin = s.iter().copied().fold(f64::INFINITY, f64::min);
    let mut points = vec![origin];
    points.extend(grid.times.iter().copied().filter(|&p| p > origin));
    let c = coarsen(&s, &t, &e, &order, &offsets, &points);
    Ok((
        c.kept.into_pyarray(py),
        c.start.into_pyarray(py),
        c.stop.into_pyarray(py),
        c.event.into_pyarray(py),
        grid.times.to_vec().into_pyarray(py),
        c.lost_events,
    ))
}

/// LTRC log-rank chi-square comparing `left` rows against the rest.
#[pyfunction]
fn logrank_score(
    start: PyReadonlyArray1<'_, f64>,
    stop: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, bool>,
    left: PyReadonlyArray1<'_, bool>,
) -> PyResult<f64> {
    check_lengths(
        start.as_array().len(),
        &[
            ("stop", stop.as_array().len()),
            ("event", event.as_array().len()),
            ("left", left.as_array().len()),
        ],
    )?;
    let surv = surv_data(&start, &stop, &event)?;
    let all: Vec<u32> = (0..surv.n_rows() as u32).collect();
    let parent = node_profile(&surv, &all);
    let left_rows: Vec<u32> = vec1(&left, "left")?
        .iter()
        .enumerate()
        .filter(|(_, l)| **l)
        .map(|(i, _)| i as u32)
        .collect();
    let (l_at, l_ev) = profile_on(&surv, &left_rows, &parent.event_idx);
    // One unit per row.
    let (n, n_left) = (surv.n_rows() as f64, left_rows.len() as f64);
    Ok(LtrcLogRank.score(
        &Profile {
            at_risk: &l_at,
            events: &l_ev,
            cause_events: &l_ev,
            n_causes: 1,
            times: &parent.times,
            exposure: exposure_of(&surv, &left_rows),
            n_units: n_left,
        },
        &parent.view(n),
        n - n_left,
    ))
}

/// Split score of `left` rows against the rest for cause-coded data, with the
/// rule `fit_forest` would use (`LtrcLogRank` for one cause, else `SingleCause`
/// for `split_cause` or the composite), both through the node scorer.
#[pyfunction]
#[pyo3(signature = (start, stop, event, left, n_causes, split_cause=None, criterion="composite"))]
fn cause_score(
    start: PyReadonlyArray1<'_, f64>,
    stop: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, u8>,
    left: PyReadonlyArray1<'_, bool>,
    n_causes: usize,
    split_cause: Option<usize>,
    criterion: &str,
) -> PyResult<f64> {
    check_lengths(
        start.as_array().len(),
        &[
            ("stop", stop.as_array().len()),
            ("event", event.as_array().len()),
            ("left", left.as_array().len()),
        ],
    )?;
    let surv = cause_data(&start, &stop, &event, n_causes)?;
    let criterion = criterion_for(n_causes, split_cause, criterion)?;
    let all: Vec<u32> = (0..surv.n_rows() as u32).collect();
    let parent = node_profile(&surv, &all);
    let left_rows: Vec<u32> = vec1(&left, "left")?
        .iter()
        .enumerate()
        .filter(|(_, l)| **l)
        .map(|(i, _)| i as u32)
        .collect();
    let (l_at, l_ev, l_cev) = cause_profile_on(&surv, &left_rows, &parent.event_idx);
    let (n, n_left) = (surv.n_rows() as f64, left_rows.len() as f64);
    let left = Profile {
        at_risk: &l_at,
        events: &l_ev,
        cause_events: if n_causes == 1 { &l_ev } else { &l_cev },
        n_causes,
        times: &parent.times,
        exposure: exposure_of(&surv, &left_rows),
        n_units: n_left,
    };
    Ok(criterion
        .node_scorer(parent.view(n))
        .score(&left, n - n_left))
}

/// Best root split: `(feature, threshold, score, left_mask, ids_left, ids_right)` or `None`.
///
/// `units` (default: one per row) must be contiguous per unit.
#[pyfunction]
#[pyo3(signature = (x, start, stop, event, *, min_ids_leaf, min_events_leaf, max_bins=256, units=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn best_split(
    x: PyReadonlyArray2<'_, f64>,
    start: PyReadonlyArray1<'_, f64>,
    stop: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, bool>,
    min_ids_leaf: usize,
    min_events_leaf: usize,
    max_bins: usize,
    units: Option<PyReadonlyArray1<'_, u32>>,
) -> PyResult<Option<(usize, f64, f64, Vec<bool>, usize, usize)>> {
    let (v, n, p) = matrix(&x)?;
    check_lengths(
        n,
        &[
            ("start", start.as_array().len()),
            ("stop", stop.as_array().len()),
            ("event", event.as_array().len()),
        ],
    )?;
    if !(2..=256).contains(&max_bins) {
        return Err(PyValueError::new_err("max_bins must be in [2, 256]"));
    }
    let surv = surv_data(&start, &stop, &event)?;
    let binned = Binned::fit(&v, n, p, max_bins);
    let rows: Vec<u32> = (0..n as u32).collect();
    let units: Vec<u32> = match units {
        Some(u) => vec1(&u, "u")?,
        None => rows.clone(),
    };
    check_lengths(n, &[("units", units.len())])?;
    // The splitter counts runs of equal units, so each unit must form one run.
    let mut seen = std::collections::HashSet::new();
    for (i, &u) in units.iter().enumerate() {
        if (i == 0 || units[i - 1] != u) && !seen.insert(u) {
            return Err(PyValueError::new_err(
                "units must be contiguous: each unit's rows in one run",
            ));
        }
    }
    let features: Vec<usize> = (0..p).collect();
    let params = SplitParams {
        min_leaf: min_ids_leaf,
        min_events_leaf,
        cause_floor: None,
    };
    Ok(core_best_split(
        &binned,
        &surv,
        &rows,
        &units,
        &features,
        &params,
        &LtrcLogRank,
    )
    .map(|s| {
        let col = binned.column(s.feature);
        let mask = (0..n).map(|i| col[i] <= s.bin).collect();
        (
            s.feature,
            s.threshold,
            s.score,
            mask,
            s.ids_left,
            s.ids_right,
        )
    }))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForest>()?;
    m.add_function(wrap_pyfunction!(fit_forest_py, m)?)?;
    m.add_function(wrap_pyfunction!(logrank_score, m)?)?;
    m.add_function(wrap_pyfunction!(cause_score, m)?)?;
    m.add_function(wrap_pyfunction!(coarsen_py, m)?)?;
    m.add_function(wrap_pyfunction!(best_split, m)?)?;
    Ok(())
}
