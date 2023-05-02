"""
Implementation of the framework proposed in the paper ``Multi Agent Dynamic
Counterfactual Recourse''.
"""
import sys

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    # mypy error: Cannot determine type of '__SKLEARN_SETUP__'
    __RECGAME_SETUP__  # type: ignore
except NameError:
    __RECGAME_SETUP__ = False

if __RECGAME_SETUP__:
    sys.stderr.write("Partial import of recgame during the build process.\n")
    # We are not importing the rest of recourse-game during the build
    # process, as it may not be compiled yet
else:
    from . import recourse, populations, environments
    from ._version import __version__

    __all__ = ["recourse", "populations", "environments", "utils", "__version__"]
