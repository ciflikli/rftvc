"""Sphinx configuration for rftvc."""

import ast
import csv
from pathlib import Path

import rftvc

project = "rftvc"
author = "Gokhan Ciflikli"
release = rftvc.__version__
extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.mathjax", "numpydoc"]
autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_validation_checks = set()
html_theme = "pydata_sphinx_theme"
html_title = "rftvc"
exclude_patterns = ["_build"]
nitpicky = False

HERE = Path(__file__).parent
ROOT = HERE.parents[1]


def _compat_matrix():
    """docs/source/generated/compat.csv from the EXPECTED_FAILED table in the tests (single source)."""
    tree = ast.parse((ROOT / "tests" / "test_sklearn_compat.py").read_text())
    table = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "EXPECTED_FAILED"
    )
    out = HERE / "generated"
    out.mkdir(exist_ok=True)
    with open(out / "compat.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sklearn check", "covered by (tests/test_sklearn_compat.py)"])
        for check, test in sorted(table.items()):
            w.writerow([f"``{check}``", f"``{test}``"])


_compat_matrix()
