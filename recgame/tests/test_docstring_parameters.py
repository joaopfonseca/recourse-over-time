# Adapted from:
# https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/tests/test_docstring_parameters.py

import inspect
import warnings
import importlib
from pkgutil import walk_packages
from inspect import signature

import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils import IS_PYPY
from sklearn.utils._testing import check_docstring_parameters
from sklearn.utils._testing import _get_func_name
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.deprecation import _is_deprecated

import recgame
from recgame.recourse import NFeatureRecourse
from recgame.populations import Population
from recgame.utils import numpy_to_pandas
from recgame.utils._testing import all_environments, all_recourse, all_population


def is_environment(estimator):
    """Return True if the given estimator is an environment, False otherwise.
    Parameters
    ----------
    estimator : object
        Estimator to test.
    Returns
    -------
    is_sampler : bool
        True if estimator is an environment, otherwise False.
    """
    if estimator._estimator_type == "environment":
        return True
    return False


def is_recourse(estimator):
    """Return True if the given estimator is a recourse method, False otherwise.
    Parameters
    ----------
    estimator : object
        Estimator to test.
    Returns
    -------
    is_sampler : bool
        True if estimator is a recourse method, otherwise False.
    """
    if estimator._estimator_type == "recourse":
        return True
    return False


def is_population(estimator):
    """Return True if the given estimator is a population method, False otherwise.
    Parameters
    ----------
    estimator : object
        Estimator to test.
    Returns
    -------
    is_sampler : bool
        True if estimator is a population method, otherwise False.
    """
    if estimator._estimator_type == "population":
        return True
    return False


def _generate_data(self, n_agents):
    X, y = make_classification(
        n_samples=n_agents,
        n_features=3,
        n_redundant=0,
        n_classes=2,
        random_state=self._rng.integers(10000),
    )
    return numpy_to_pandas(X)


# walk_packages() ignores DeprecationWarnings, now we need to ignore
# FutureWarnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    recgame_path = recgame.__path__
    PUBLIC_MODULES = set(
        [
            pckg[1]
            for pckg in walk_packages(prefix="recgame.", path=recgame_path)
            if not ("._" in pckg[1] or ".tests." in pckg[1])
        ]
    )

# functions to ignore args / docstring of
_DOCSTRING_IGNORES = []

# Methods where y param should be ignored if y=None by default
_METHODS_IGNORE_NONE_Y = [
    "fit",
    "score",
    "fit_predict",
    "fit_transform",
    "partial_fit",
    "predict",
]


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skipif(IS_PYPY, reason="test segfaults on PyPy")
def test_docstring_parameters():
    # Test module docstring formatting

    # Skip test if numpydoc is not found
    pytest.importorskip(
        "numpydoc", reason="numpydoc is required to test the docstrings"
    )

    from numpydoc import docscrape

    incorrect = []
    for name in PUBLIC_MODULES:
        if name.endswith(".conftest"):
            # pytest tooling, not part of the scikit-learn API
            continue
        with warnings.catch_warnings(record=True):
            module = importlib.import_module(name)
        classes = inspect.getmembers(module, inspect.isclass)
        # Exclude non-recourse-game classes
        classes = [cls for cls in classes if cls[1].__module__.startswith("recgame")]
        for cname, cls in classes:
            this_incorrect = []
            if cname in _DOCSTRING_IGNORES or cname.startswith("_"):
                continue
            if inspect.isabstract(cls):
                continue
            with warnings.catch_warnings(record=True) as w:
                cdoc = docscrape.ClassDoc(cls)
            if len(w):
                raise RuntimeError(
                    "Error for __init__ of %s in %s:\n%s" % (cls, name, w[0])
                )

            cls_init = getattr(cls, "__init__", None)

            if _is_deprecated(cls_init):
                continue
            elif cls_init is not None:
                this_incorrect += check_docstring_parameters(cls.__init__, cdoc)

            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                if _is_deprecated(method):
                    continue
                param_ignore = None
                # Now skip docstring test for y when y is None
                # by default for API reason
                if method_name in _METHODS_IGNORE_NONE_Y:
                    sig = signature(method)
                    if "y" in sig.parameters and sig.parameters["y"].default is None:
                        param_ignore = ["y"]  # ignore y for fit and score
                result = check_docstring_parameters(method, ignore=param_ignore)
                this_incorrect += result

            incorrect += this_incorrect

        functions = inspect.getmembers(module, inspect.isfunction)
        # Exclude imported functions
        functions = [fn for fn in functions if fn[1].__module__ == name]
        for fname, func in functions:
            # Don't test private methods / functions
            if fname.startswith("_"):
                continue
            if fname == "configuration" and name.endswith("setup"):
                continue
            name_ = _get_func_name(func)
            if not any(d in name_ for d in _DOCSTRING_IGNORES) and not _is_deprecated(
                func
            ):
                incorrect += check_docstring_parameters(func)

    msg = "\n".join(incorrect)
    if len(incorrect) > 0:
        raise AssertionError("Docstring Error:\n" + msg)


@ignore_warnings(category=FutureWarning)
def test_tabs():
    # Test that there are no tabs in our source files
    for importer, modname, ispkg in walk_packages(recgame.__path__, prefix="recgame."):
        if IS_PYPY:
            continue

        # because we don't import
        mod = importlib.import_module(modname)

        try:
            source = inspect.getsource(mod)
        except IOError:  # user probably should have run "make clean"
            continue
        assert "\t" not in source, (
            f'"{modname}" has tabs, please remove them ' "or add it to the ignore list"
        )


@pytest.mark.parametrize(
    "name, Estimator", [*all_environments(), *all_recourse(), *all_population()]
)
def test_fit_docstring_attributes(name, Estimator):
    pytest.importorskip("numpydoc")
    from numpydoc import docscrape

    if Estimator.__name__ in _DOCSTRING_IGNORES:
        return

    doc = docscrape.ClassDoc(Estimator)
    attributes = doc["Attributes"]

    X, y = make_classification(
        n_samples=20,
        n_features=3,
        n_redundant=0,
        n_classes=2,
        random_state=2,
    )
    X = numpy_to_pandas(X)

    if is_environment(Estimator):
        clf = LogisticRegression(random_state=2).fit(X, y)

        # monkey patch add_agents since the generative process is different from the
        # default function
        Estimator.add_agents = _generate_data

        est = Estimator(
            population=Population(X),
            recourse=NFeatureRecourse(model=clf),
            random_state=2,
        )
    elif is_recourse(Estimator):
        clf = LogisticRegression(random_state=2).fit(X, y)
        est = Estimator(model=clf)
    elif is_population(Estimator):
        est = Estimator(X=X)
    else:
        raise TypeError(f"Could not recognize the object type of {Estimator}")

    if "oob_score" in est.get_params():
        est.set_params(oob_score=True)

    if is_environment(est):
        est.update()
    elif is_recourse(est):
        est.counterfactual(Population(X))

    skipped_attributes = set([])

    for attr in attributes:
        if attr.name in skipped_attributes:
            continue
        desc = " ".join(attr.desc).lower()
        # As certain attributes are present "only" if a certain parameter is
        # provided, this checks if the word "only" is present in the attribute
        # description, and if not the attribute is required to be present.
        if "only " in desc:
            continue
        # ignore deprecation warnings
        with ignore_warnings(category=FutureWarning):
            assert hasattr(est, attr.name)

    fit_attr = _get_all_fitted_attributes(est)
    fit_attr_names = [attr.name for attr in attributes]
    undocumented_attrs = set(fit_attr).difference(fit_attr_names)
    undocumented_attrs = set(undocumented_attrs).difference(skipped_attributes)
    if undocumented_attrs:
        raise AssertionError(
            f"Undocumented attributes for {Estimator.__name__}: {undocumented_attrs}"
        )


def _get_all_fitted_attributes(estimator):
    "Get all the fitted attributes of an estimator including properties"
    # attributes
    fit_attr = list(estimator.__dict__.keys())

    # properties
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)

        for name in dir(estimator.__class__):
            obj = getattr(estimator.__class__, name)
            if not isinstance(obj, property):
                continue

            # ignore properties that raises an AttributeError and deprecated
            # properties
            try:
                getattr(estimator, name)
            except (AttributeError, FutureWarning):
                continue
            fit_attr.append(name)

    return [k for k in fit_attr if k.endswith("_") and not k.startswith("_")]
