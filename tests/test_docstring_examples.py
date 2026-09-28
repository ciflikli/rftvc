"""Release-pass T3-slice-1: run every module's docstring ``Examples`` as doctests."""

import doctest

import pytest

from rftvc import _competing, _estimator, inspection, landmark, metrics, model_selection

MODULES = [_estimator, _competing, landmark, inspection, metrics, model_selection]


@pytest.mark.parametrize("module", MODULES, ids=[m.__name__ for m in MODULES])
def test_docstring_examples(module):
    results = doctest.testmod(module, verbose=False, optionflags=doctest.ELLIPSIS)
    assert results.failed == 0, f"{results.failed} doctest failure(s) in {module.__name__}"
