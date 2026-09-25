//! Python bindings: `rftvc._core`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rftvc_core::rng::Rng;
use rftvc_core::{
    Binned, LtrcLogRank, Profile, SplitCriterion, SplitParams, SurvData, Tree, TreeParams,
    best_split as core_best_split, build_tree, node_profile, profile_on,
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

/// `(event_times, d, y, cumhaz)` of one leaf.
type LeafProfile = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

#[pyclass(module = "rftvc._core", name = "Tree", frozen)]
struct PyTree {
    inner: Tree,
}

#[pymethods]
impl PyTree {
    #[getter]
    fn n_leaves(&self) -> usize {
        self.inner.leaves.len()
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.nodes.len()
    }

    /// Leaf index for each row of `x`.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<i64>> {
        let (v, n, p) = matrix(&x);
        let out: Vec<i64> = (0..n)
            .map(|i| self.inner.apply(&v[i * p..(i + 1) * p]) as i64)
            .collect();
        out.into_pyarray(py)
    }

    /// Cumulative hazard `(n_rows, n_times)` for rows starting at time 0.
    fn predict_cumhaz<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let (v, n, p) = matrix(&x);
        let times: Vec<f64> = times.as_array().to_vec();
        let m = times.len();
        let tree = &self.inner;
        let out = py.detach(|| {
            let mut out = vec![0.0; n * m];
            for i in 0..n {
                let leaf = tree.apply(&v[i * p..(i + 1) * p]);
                for (j, &t) in times.iter().enumerate() {
                    out[i * m + j] = tree.cumhaz_at(leaf, t);
                }
            }
            out
        });
        Array2::from_shape_vec((n, m), out)
            .expect("shape")
            .into_pyarray(py)
    }

    /// `(event_times, d, y, cumhaz)` of one leaf.
    fn leaf_profile(&self, leaf: usize) -> PyResult<LeafProfile> {
        let l = self
            .inner
            .leaves
            .get(leaf)
            .ok_or_else(|| PyValueError::new_err("leaf index out of range"))?;
        let times = l
            .event_idx
            .iter()
            .map(|&k| self.inner.grid_times[k as usize])
            .collect();
        Ok((times, l.d.clone(), l.y.clone(), l.cumhaz.clone()))
    }
}

#[pyfunction]
#[pyo3(signature = (x, start, stop, event, *, max_depth, min_ids_leaf, min_events_leaf, max_features, max_bins, seed))]
#[allow(clippy::too_many_arguments)]
fn fit_tree(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    start: PyReadonlyArray1<'_, f64>,
    stop: PyReadonlyArray1<'_, f64>,
    event: PyReadonlyArray1<'_, bool>,
    max_depth: Option<usize>,
    min_ids_leaf: usize,
    min_events_leaf: usize,
    max_features: usize,
    max_bins: usize,
    seed: u64,
) -> PyResult<PyTree> {
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
    let params = TreeParams {
        max_depth,
        min_ids_leaf,
        min_events_leaf,
        max_features: max_features.clamp(1, p),
    };
    let inner = py.detach(|| {
        let binned = Binned::fit(&v, n, p, max_bins);
        let rows: Vec<u32> = (0..n as u32).collect();
        build_tree(
            &binned,
            &surv,
            rows,
            &params,
            &LtrcLogRank,
            &mut Rng::new(seed),
        )
    });
    Ok(PyTree { inner })
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
    m.add_class::<PyTree>()?;
    m.add_function(wrap_pyfunction!(fit_tree, m)?)?;
    m.add_function(wrap_pyfunction!(logrank_score, m)?)?;
    m.add_function(wrap_pyfunction!(best_split, m)?)?;
    Ok(())
}
