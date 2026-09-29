"""Optional plotting helpers for rftvc results (``pip install rftvc[viz]``).

Every function renders an existing rftvc return value; none compute new
statistics. The five tabular charts (``plot_survival_curve``,
``plot_cumulative_incidence``, ``plot_importance``, ``plot_hazard_effect``,
``plot_calibration``) use `Altair <https://altair-viz.github.io/>`_ and need
the ``viz`` extra. ``plot_tree`` needs no extra dependency at all: it returns
a plain SVG string built directly from :meth:`~rftvc.SurvivalForestTV.export_tree`.
"""

import numbers

import numpy as np
import polars as pl

__all__ = [
    "plot_calibration",
    "plot_cumulative_incidence",
    "plot_hazard_effect",
    "plot_importance",
    "plot_survival_curve",
    "plot_tree",
]


def _require_altair():
    try:
        import altair as alt
    except ImportError as exc:
        raise ImportError("this function needs altair; install it with `pip install rftvc[viz]`") from exc
    return alt


def plot_survival_curve(times, curve, *, labels=None, kind="survival"):
    """Step-line chart of a survival or cumulative-hazard curve.

    Parameters
    ----------
    times : array-like of shape (n_times,)
        As passed to ``predict_survival_function`` / ``predict_cumulative_hazard``.
    curve : array-like of shape (n_subjects, n_times)
        Their return value.
    labels : array-like of shape (n_subjects,), default=None
        Name per row (subject); defaults to its row index.
    kind : {"survival", "hazard"}, default="survival"
        Only controls the y-axis title.

    Returns
    -------
    altair.Chart
    """
    alt = _require_altair()
    if kind not in ("survival", "hazard"):
        raise ValueError(f"kind must be 'survival' or 'hazard', got {kind!r}")
    times = np.asarray(times, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != times.shape[0]:
        raise ValueError(f"curve must have shape (n_subjects, {times.shape[0]}), got {curve.shape}")
    labels = np.arange(curve.shape[0]) if labels is None else np.asarray(labels)
    if labels.shape[0] != curve.shape[0]:
        raise ValueError("labels must have one entry per row of curve")
    df = pl.DataFrame(
        {
            "time": np.tile(times, curve.shape[0]),
            "value": curve.ravel(),
            "subject": np.repeat(labels.astype(str), times.shape[0]),
        }
    )
    y_title = "S(t)" if kind == "survival" else "cumulative hazard"
    return (
        alt.Chart(df)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("time:Q", title="time"),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("subject:N", title="subject"),
        )
        .properties(width=500, height=300)
    )


def plot_cumulative_incidence(times, cif, *, cause_labels=None, subject_labels=None):
    """Step-line chart of cumulative incidence per cause (and per subject, if more than one).

    Parameters
    ----------
    times : array-like of shape (n_times,)
        As passed to ``predict_cumulative_incidence``.
    cif : array-like of shape (n_subjects, n_causes, n_times) or (n_subjects, n_times)
        Its return value (the 2-D form is a single-``cause`` call).
    cause_labels : array-like of shape (n_causes,), default=None
        The model's own cause labels, in ``causes_`` order (or the single label for a
        2-D, one-``cause`` ``cif``) -- pass ``forest.causes_`` (or ``[cause]``) to show
        them. Cause labels need not be contiguous or start at 1 (``make_competing_risks_y``
        allows any positive integers), so the default is an uninterpreted axis position
        (``"cause[0]"``, ``"cause[1]"``, ...), not a guess at the real label.
    subject_labels : array-like of shape (n_subjects,), default=None
        Defaults to the row index.

    Returns
    -------
    altair.Chart
        Faceted by subject when there is more than one.
    """
    alt = _require_altair()
    times = np.asarray(times, dtype=float)
    cif = np.asarray(cif, dtype=float)
    if cif.ndim == 2:
        cif = cif[:, None, :]
    if cif.ndim != 3 or cif.shape[2] != times.shape[0]:
        raise ValueError(
            f"cif must have shape (n_subjects, n_causes, {times.shape[0]}) or (n_subjects, {times.shape[0]}), "
            f"got {cif.shape}"
        )
    n_subjects, n_causes, n_times = cif.shape
    cause_labels = (
        np.array([f"cause[{j}]" for j in range(n_causes)]) if cause_labels is None else np.asarray(cause_labels)
    )
    subject_labels = np.arange(n_subjects) if subject_labels is None else np.asarray(subject_labels)
    if cause_labels.shape[0] != n_causes:
        raise ValueError("cause_labels must have one entry per cause")
    if subject_labels.shape[0] != n_subjects:
        raise ValueError("subject_labels must have one entry per subject")
    df = pl.DataFrame(
        {
            "time": np.tile(times, n_subjects * n_causes),
            "value": cif.ravel(),
            "cause": np.tile(np.repeat(cause_labels.astype(str), n_times), n_subjects),
            "subject": np.repeat(subject_labels.astype(str), n_causes * n_times),
        }
    )
    chart = (
        alt.Chart(df)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("time:Q", title="time"),
            y=alt.Y("value:Q", title="cumulative incidence"),
            color=alt.Color("cause:N", title="cause"),
        )
        .properties(width=350 if n_subjects > 1 else 500, height=300)
    )
    return chart.facet(column=alt.Column("subject:N", title="subject")) if n_subjects > 1 else chart


def plot_importance(result, *, top_n=None):
    """Horizontal bar chart of a feature-importance result, with its standard error.

    Parameters
    ----------
    result : Bunch
        The return value of ``inspection.permutation_importance`` or
        ``inspection.drop_column_importance`` (anything with ``importances_mean``,
        ``importances_se`` and ``feature_names`` works).
    top_n : int, default=None
        Keep only the ``top_n`` units by ``|importances_mean|``; default all.

    Returns
    -------
    altair.LayerChart
    """
    alt = _require_altair()
    for attr in ("importances_mean", "importances_se", "feature_names"):
        if not hasattr(result, attr):
            raise TypeError(f"result must be a permutation_importance/drop_column_importance Bunch (missing {attr!r})")
    mean = np.asarray(result.importances_mean, dtype=float)
    se = np.asarray(result.importances_se, dtype=float)
    names = np.asarray(result.feature_names, dtype=str)
    if not mean.shape == se.shape == names.shape:
        raise ValueError("result.importances_mean, importances_se and feature_names must have the same shape")
    order = np.argsort(-np.abs(mean))
    if top_n is not None:
        order = order[:top_n]
    df = pl.DataFrame(
        {"feature": names[order], "importance": mean[order], "lo": (mean - se)[order], "hi": (mean + se)[order]}
    )
    y = alt.Y("feature:N", sort=None, title=None)
    bars = alt.Chart(df).mark_bar().encode(x=alt.X("importance:Q"), y=y)
    err = alt.Chart(df).mark_errorbar().encode(x=alt.X("lo:Q", title="importance"), x2="hi:Q", y=y)
    return (bars + err).properties(width=500, height=max(200, 22 * len(df)))


def plot_hazard_effect(result, *, cause=None):
    """Line chart of ``inspection.hazard_effect``'s window hazard, one line per grid value.

    Parameters
    ----------
    result : Bunch
        The return value of ``inspection.hazard_effect``.
    cause : int, default=None
        0-based index into the ``causes_`` axis (``result.hazard.ndim == 3``);
        required for a competing-risks result, disallowed otherwise.

    Returns
    -------
    altair.Chart
    """
    alt = _require_altair()
    for attr in ("hazard", "window_edges", "feature_name"):
        if not hasattr(result, attr):
            raise TypeError(f"result must be a hazard_effect Bunch (missing {attr!r})")
    values = np.asarray(result["values"])
    hazard = np.asarray(result.hazard, dtype=float)
    edges = np.asarray(result.window_edges, dtype=float)
    if hazard.ndim == 3:
        if cause is None:
            raise ValueError("cause (a 0-based index into the causes_ axis) is required: result.hazard is 3-D")
        if not (isinstance(cause, numbers.Integral) and not isinstance(cause, (bool, np.bool_))):
            raise TypeError(f"cause must be an int, got {type(cause).__name__}")
        if not 0 <= cause < hazard.shape[1]:
            raise ValueError(f"cause must be in [0, {hazard.shape[1]}), got {cause}")
        hazard = hazard[:, cause, :]
    elif cause is not None:
        raise ValueError("cause is only for a competing-risks result (result.hazard is already 2-D here)")
    mid = (edges[:-1] + edges[1:]) / 2.0
    if hazard.shape != (values.shape[0], mid.shape[0]):
        raise ValueError(f"hazard must have shape ({values.shape[0]}, {mid.shape[0]}), got {hazard.shape}")
    df = pl.DataFrame(
        {
            "window_mid": np.tile(mid, values.shape[0]),
            "hazard": hazard.ravel(),
            "value": np.repeat(np.asarray(values, dtype=str), mid.shape[0]),
        }
    )
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("window_mid:Q", title="window midpoint"),
            y=alt.Y("hazard:Q", title="window hazard"),
            color=alt.Color("value:N", title=str(result.feature_name)),
        )
        .properties(width=500, height=300)
    )


def plot_calibration(table):
    """Scatter of observed vs. predicted risk, from ``metrics.calibration_table``.

    Parameters
    ----------
    table : polars.DataFrame
        The return value of ``metrics.calibration_table``.

    Returns
    -------
    altair.LayerChart
        Points (sized by bin count) plus a ``y = x`` reference line.
    """
    alt = _require_altair()
    required = {"mean_risk", "observed_risk", "n"}
    cols = set(table.columns) if hasattr(table, "columns") else set()
    if not required <= cols:
        raise TypeError(f"table must be a calibration_table result (missing columns {sorted(required - cols)})")
    df = table if isinstance(table, pl.DataFrame) else pl.from_pandas(table)
    hi = float(max(df["mean_risk"].max(), df["observed_risk"].max())) * 1.05
    lims = [0.0, hi if hi > 0 else 1.0]
    ref = pl.DataFrame({"x": lims, "y": lims})
    tooltip = [c for c in ("bin", "n", "n_events", "mean_risk", "observed_risk") if c in df.columns]
    points = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("mean_risk:Q", title="mean predicted risk", scale=alt.Scale(domain=lims)),
            y=alt.Y("observed_risk:Q", title="observed risk (1 - KM)", scale=alt.Scale(domain=lims)),
            size=alt.Size("n:Q", title="n"),
            tooltip=tooltip,
        )
    )
    line = alt.Chart(ref).mark_line(strokeDash=[4, 4], color="gray").encode(x="x:Q", y="y:Q")
    return (points + line).properties(width=400, height=400)


def _tree_layout(children_left, children_right, leaf, max_depth):
    """``(depth, x, cut, visited)`` per node: iterative (no recursion-depth limit).

    ``x`` sequences leaves left to right (0, 1, 2, ...); an internal node's ``x``
    is the mean of its children's. ``cut[i]`` marks a node drawn as a collapsed
    "..." marker because it is past ``max_depth`` (only when ``max_depth`` is given).
    ``visited[i]`` is ``False`` for a node past a cut marker: it keeps a stale
    default ``depth``/``x`` and must not be drawn.
    """
    n = children_left.shape[0]
    depth = np.zeros(n, dtype=int)
    cut = np.zeros(n, dtype=bool)
    visited = np.zeros(n, dtype=bool)
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depth[node] = d
        visited[node] = True
        is_leaf = leaf[node] >= 0
        cut[node] = (not is_leaf) and max_depth is not None and d >= max_depth
        if not is_leaf and not cut[node]:
            stack.append((int(children_left[node]), d + 1))
            stack.append((int(children_right[node]), d + 1))
    x = np.zeros(n, dtype=float)
    next_x = [0.0]
    stack = [(0, False)]
    while stack:
        node, processed = stack.pop()
        terminal = leaf[node] >= 0 or cut[node]
        if terminal:
            x[node] = next_x[0]
            next_x[0] += 1.0
        elif processed:
            x[node] = (x[int(children_left[node])] + x[int(children_right[node])]) / 2.0
        else:
            stack.append((node, True))
            stack.append((int(children_right[node]), False))
            stack.append((int(children_left[node]), False))
    return depth, x, cut, visited


def _svg_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def plot_tree(forest_or_tree, tree=0, *, max_depth=None, feature_names=None):
    """SVG diagram of one tree's split structure (no extra dependency).

    Parameters
    ----------
    forest_or_tree : fitted SurvivalForestTV or CompetingRisksForestTV, or a Bunch
        A fitted estimator (its own ``export_tree(tree)`` is called), or the
        ``Bunch`` that ``export_tree`` itself returns.
    tree : int, default=0
        Only used when ``forest_or_tree`` is a fitted estimator.
    max_depth : int, default=None
        Nodes deeper than this are drawn as a collapsed ``"..."`` marker;
        ``None`` draws the whole tree.
    feature_names : sequence, default=None
        Overrides the exported ``feature_names`` (e.g. for a forest fit on a bare array).

    Returns
    -------
    str
        A standalone SVG document.
    """
    et = forest_or_tree.export_tree(tree) if hasattr(forest_or_tree, "export_tree") else forest_or_tree
    for attr in ("children_left", "children_right", "feature", "threshold", "leaf"):
        if not hasattr(et, attr):
            raise TypeError(f"forest_or_tree must be a fitted estimator or an export_tree() Bunch (missing {attr!r})")
    if max_depth is not None and not (isinstance(max_depth, numbers.Integral) and max_depth >= 0):
        raise ValueError(f"max_depth must be a non-negative int or None, got {max_depth!r}")
    names = feature_names if feature_names is not None else getattr(et, "feature_names", None)
    cl = np.asarray(et.children_left)
    cr = np.asarray(et.children_right)
    feat = np.asarray(et.feature)
    thr = np.asarray(et.threshold)
    leaf = np.asarray(et.leaf)
    missing_right = np.asarray(getattr(et, "missing_goes_right", np.ones_like(leaf, dtype=bool)))
    depth, x, cut, visited = _tree_layout(cl, cr, leaf, max_depth)

    unit_x, unit_y, margin, box_w, box_h = 155, 80, 30, 140, 40
    width = margin * 2 + (x[visited].max() + 1) * unit_x
    height = margin * 2 + (depth[visited].max() + 1) * unit_y
    cx = margin + x * unit_x + box_w / 2
    cy = margin + depth * unit_y + box_h / 2

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" font-family="sans-serif" font-size="12">']
    for node in range(cl.shape[0]):
        if not visited[node] or leaf[node] >= 0 or cut[node]:
            continue
        for child in (int(cl[node]), int(cr[node])):
            parts.append(f'<line x1="{cx[node]:.1f}" y1="{cy[node]:.1f}" x2="{cx[child]:.1f}" y2="{cy[child]:.1f}" stroke="#999"/>')
    for node in range(cl.shape[0]):
        if not visited[node]:
            continue
        left = cx[node] - box_w / 2
        top = cy[node] - box_h / 2
        is_leaf = leaf[node] >= 0
        if is_leaf:
            fill, text = "#dbe9f6", f"leaf {int(leaf[node])}"
        elif cut[node]:
            fill, text = "#eeeeee", "..."
        else:
            fname = f"x[{feat[node]}]" if names is None else _svg_escape(names[feat[node]])
            if np.isnan(thr[node]):
                text = f"{fname} missing?"
            else:
                direction = "R" if missing_right[node] else "L"
                text = f"{fname} ≤ {thr[node]:.3g}; NaN → {direction}"
            fill = "#f6e9db"
        parts.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{box_w}" height="{box_h}" rx="6" fill="{fill}" stroke="#333"/>')
        parts.append(
            f'<text x="{cx[node]:.1f}" y="{cy[node]:.1f}" text-anchor="middle" dominant-baseline="middle">{text}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)
