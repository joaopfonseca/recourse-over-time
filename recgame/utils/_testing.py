# Adapted from scikit-learn
# Authors: Joao Fonseca <jpfonseca@novaims.unl.pt>
# License: MIT

import inspect
import pkgutil
from importlib import import_module
from pathlib import Path

from operator import itemgetter

from sklearn.utils._testing import ignore_warnings
from imblearn.base import SamplerMixin


def all_environments(
    type_filter=None,
):
    """Get a list of all environments from recgame.

    This function crawls the module and gets all classes that inherit
    from BaseEnvironment. Classes that are defined in test-modules are not
    included.
    By default meta_estimators are also not included.
    This function is adapted from imblearn.

    Parameters
    ----------
    type_filter : str, list of str, or None, default=None
        Which kind of estimators should be returned. If None, no
        filter is applied and all estimators are returned.  Possible
        values are 'sampler' to get estimators only of these specific
        types, or a list of these to get the estimators that fit at
        least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    from recgame.environments.base import BaseEnvironment

    return [("BaseEnvironment", BaseEnvironment), *_all_objs(BaseEnvironment)]


def all_recourse(
    type_filter=None,
):
    """Get a list of all recourse methods from recgame.

    This function crawls the module and gets all classes that inherit
    from BaseRecourse. Classes that are defined in test-modules are not
    included.
    By default meta_estimators are also not included.
    This function is adapted from imblearn.

    Parameters
    ----------
    type_filter : str, list of str, or None, default=None
        Which kind of estimators should be returned. If None, no
        filter is applied and all estimators are returned.  Possible
        values are 'sampler' to get estimators only of these specific
        types, or a list of these to get the estimators that fit at
        least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    from recgame.recourse.base import BaseRecourse

    return _all_objs(BaseRecourse)


def _all_objs(
    base_class,
    type_filter=None,
):
    """Get a list of all classes from a certain base class from recgame.

    This function crawls the module and gets all classes that inherit
    from a base class. Classes that are defined in test-modules are not
    included.
    By default meta_estimators are also not included.
    This function is adapted from imblearn.

    Parameters
    ----------
    base_class : class
        The base class to find the classes that inherit it.
    type_filter : str, list of str, or None, default=None
        Which kind of estimators should be returned. If None, no
        filter is applied and all estimators are returned.  Possible
        values are 'sampler' to get estimators only of these specific
        types, or a list of these to get the estimators that fit at
        least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    modules_to_ignore = {"tests"}
    root = str(Path(__file__).parent.parent)
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=[root], prefix="recgame."
        ):
            mod_parts = modname.split(".")
            if any(part in modules_to_ignore for part in mod_parts) or "._" in modname:
                continue
            module = import_module(modname)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls) for name, est_cls in classes if not name.startswith("_")
            ]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], base_class) and c[0] != base_class.__name__)
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    # get rid of sklearn estimators which have been imported in some classes
    # estimators = [c for c in estimators if "sklearn" not in c[1].__module__]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {"sampler": SamplerMixin}
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)]
                )
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'sampler' or "
                "None, got"
                " %s." % repr(type_filter)
            )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))
