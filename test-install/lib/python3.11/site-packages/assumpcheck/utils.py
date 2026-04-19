from __future__ import annotations

import importlib
from typing import Any


def require_dependency(module_name: str, package_hint: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = package_hint or module_name
        raise ImportError(
            f"{hint} is required for this feature. Install assumpcheck with its core "
            f"dependencies, for example: pip install assumpcheck or pip install {hint}."
        ) from exc


def require_core_dependencies() -> dict[str, Any]:
    numpy = require_dependency("numpy", "numpy")
    pandas = require_dependency("pandas", "pandas")
    scipy_stats = require_dependency("scipy.stats", "scipy")
    matplotlib_pyplot = require_dependency("matplotlib.pyplot", "matplotlib")
    statsmodels_api = require_dependency("statsmodels.api", "statsmodels")
    return {
        "np": numpy,
        "pd": pandas,
        "stats": scipy_stats,
        "plt": matplotlib_pyplot,
        "sm": statsmodels_api,
    }


def to_1d_array(values: Any, np: Any) -> Any:
    array = np.asarray(values)
    if array.ndim != 1:
        array = np.ravel(array)
    return array


def to_dataframe(X: Any, pd: Any) -> Any:
    if hasattr(X, "columns"):
        return X.copy()
    return pd.DataFrame(X)


def add_constant(X: Any, sm: Any) -> Any:
    return sm.add_constant(X, has_constant="add")


def bool_from_user_input(value: bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise TypeError("Expected a boolean or None.")


def p_value_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
