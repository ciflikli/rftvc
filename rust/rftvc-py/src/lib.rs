//! Python bindings: `rftvc._core`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rftvc_core::{
    Aggregate, Binned, FlatForest, Forest, ForestParams, Groups, LtrcLogRank, Profile,
    SplitCriterion, SplitParams, SurvData, TreeParams, best_split as core_best_split, fit_forest,
    node_profile, profile_on,
};

fn matrix(x: &PyReadonlyArray2<f64>) -> (Vec<f64>, usize, usize) {
    let a = x.as_array();
    let (n, p) = a.dim();
    (a.iter().copied().collect(), n, p)
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
) -> SurvData {
    let s: Vec<f64> = start.as_array().to_vec();
    let t: Vec<f64> = stop.as_array().to_vec();
    let e: Vec<bool> = event.as_array().to_vec();
    SurvData::new(&s, &t, &e)
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

/// `(event_times, d, y, cumhaz)` of one leaf.
type LeafProfile = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

#[pyclass(module = "rftvc._core", name = "Forest", frozen)]
struct PyForest {
    inner: Forest,
}

impl PyForest {
    fn check_x(&self, x: &PyReadonlyArray2<f64>) -> PyResult<(Vec<f64>, usize, usize)> {
        let (v, n, p) = matrix(x);
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

    fn n_leaves(&self, tree: usize) -> PyResult<usize> {
        Ok(self.tree(tree)?.leaves.len())
    }

    /// `(event_times, d, y, cumhaz)` of one leaf of one tree.
    fn leaf_profile(&self, tree: usize, leaf: usize) -> PyResult<LeafProfile> {
        let t = self.tree(tree)?;
        let l = t
            .leaves
            .get(leaf)
            .ok_or_else(|| PyValueError::new_err("leaf index out of range"))?;
        let times = l
            .event_idx
            .iter()
            .map(|&k| t.grid_times[k as usize])
            .collect();
        Ok((times, l.d.clone(), l.y.clone(), l.cumhaz.clone()))
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
        let times: Vec<f64> = times.as_array().to_vec();
        let m = times.len();
        let forest = &self.inner;
        let pool = pool(n_jobs)?;
        let out = py.detach(|| pool.install(|| forest.predict_cumhaz(&v, p, &times, agg)));
        Ok(Array2::from_shape_vec((n, m), out)
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
        d.set_item("d", f.d.into_pyarray(py))?;
        d.set_item("y", f.y.into_pyarray(py))?;
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
                item(state, $key)?
                    .extract::<PyReadonlyArray1<$t>>()?
                    .as_array()
                    .to_vec()
            };
        }
        let flat = FlatForest {
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
            d: vec_of!("d", f64),
            y: vec_of!("y", f64),
        };
        let inner = flat.to_forest().map_err(PyValueError::new_err)?;
        Ok(PyForest { inner })
    }
}

#[pyfunction]
#[pyo3(name = "fit_forest", signature = (
    x, start, stop, event, groups, n_groups, *, n_trees, n_draw, bootstrap,
    max_depth, min_ids_leaf, min_events_leaf, max_features, max_bins, seed, n_jobs
))]
#[allow(clippy::too_many_arguments)]
fn fit_forest_py(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    start: PyReadonlyArray1<'_, f64>,
    stop: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, bool>,
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
) -> PyResult<PyForest> {
    let (v, n, p) = matrix(&x);
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
    let groups: Vec<u32> = groups.as_array().to_vec();
    if n_groups == 0 || groups.iter().any(|&g| g as usize >= n_groups) {
        return Err(PyValueError::new_err("groups must be in [0, n_groups)"));
    }
    if n_trees == 0 || n_draw == 0 || (!bootstrap && n_draw > n_groups) {
        return Err(PyValueError::new_err(
            "need n_trees >= 1 and 1 <= n_draw (<= n_groups without bootstrap)",
        ));
    }
    let surv = surv_data(&start, &stop, &event);
    let params = ForestParams {
        tree: TreeParams {
            max_depth,
            min_ids_leaf,
            min_events_leaf,
            max_features: max_features.clamp(1, p),
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
            fit_forest(&binned, &surv, &Groups::new(&groups, n_groups), &params)
        })
    });
    Ok(PyForest { inner })
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
    let surv = surv_data(&start, &stop, &event);
    let all: Vec<u32> = (0..surv.n_rows() as u32).collect();
    let parent = node_profile(&surv, &all);
    let left_rows: Vec<u32> = left
        .as_array()
        .iter()
        .enumerate()
        .filter(|(_, l)| **l)
        .map(|(i, _)| i as u32)
        .collect();
    let (l_at, l_ev) = profile_on(&surv, &left_rows, &parent.event_idx);
    Ok(LtrcLogRank.score(
        &Profile {
            at_risk: &l_at,
            events: &l_ev,
        },
        &Profile {
            at_risk: &parent.at_risk,
            events: &parent.events,
        },
    ))
}

/// Best root split over all features: `(feature, threshold, score, left_mask)` or `None`.
#[pyfunction]
#[pyo3(signature = (x, start, stop, event, *, min_ids_leaf, min_events_leaf, max_bins=256))]
#[allow(clippy::type_complexity)]
fn best_split(
    x: PyReadonlyArray2<'_, f64>,
    start: PyReadonlyArray1<'_, f64>,
    stop: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, bool>,
    min_ids_leaf: usize,
    min_events_leaf: usize,
    max_bins: usize,
) -> PyResult<Option<(usize, f64, f64, Vec<bool>)>> {
    let (v, n, p) = matrix(&x);
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
    let surv = surv_data(&start, &stop, &event);
    let binned = Binned::fit(&v, n, p, max_bins);
    let rows: Vec<u32> = (0..n as u32).collect();
    let features: Vec<usize> = (0..p).collect();
    let params = SplitParams {
        min_leaf: min_ids_leaf,
        min_events_leaf,
    };
    Ok(
        core_best_split(&binned, &surv, &rows, &features, &params, &LtrcLogRank).map(|s| {
            let col = binned.column(s.feature);
            let mask = (0..n).map(|i| col[i] <= s.bin).collect();
            (s.feature, s.threshold, s.score, mask)
        }),
    )
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForest>()?;
    m.add_function(wrap_pyfunction!(fit_forest_py, m)?)?;
    m.add_function(wrap_pyfunction!(logrank_score, m)?)?;
    m.add_function(wrap_pyfunction!(best_split, m)?)?;
    Ok(())
}
