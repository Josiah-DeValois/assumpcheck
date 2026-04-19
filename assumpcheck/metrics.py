from __future__ import annotations

from typing import Any

from .utils import p_value_text, require_dependency, require_core_dependencies, safe_float, to_1d_array


def shapiro_wilk(residuals: Any) -> dict[str, float | str | None]:
    deps = require_core_dependencies()
    np = deps["np"]
    stats = deps["stats"]
    residuals = to_1d_array(residuals, np)
    sample = residuals[:5000] if residuals.size > 5000 else residuals
    if sample.size < 3:
        return {"statistic": None, "p_value": None, "note": "At least 3 residuals are required."}
    statistic, p_value = stats.shapiro(sample)
    return {"statistic": float(statistic), "p_value": float(p_value), "note": None}


def levene_test(values: Any, groups: Any) -> dict[str, float | str | None]:
    deps = require_core_dependencies()
    np = deps["np"]
    stats = deps["stats"]
    values = to_1d_array(values, np)
    groups = to_1d_array(groups, np)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return {"statistic": None, "p_value": None, "note": "At least two groups are required."}
    grouped = [values[groups == group] for group in unique_groups]
    statistic, p_value = stats.levene(*grouped, center="median")
    return {"statistic": float(statistic), "p_value": float(p_value), "note": None}


def durbin_watson(residuals: Any) -> float:
    stattools = require_dependency("statsmodels.stats.stattools", "statsmodels")
    return float(stattools.durbin_watson(residuals))


def breusch_pagan(residuals: Any, exog: Any) -> dict[str, float]:
    diagnostic = require_dependency("statsmodels.stats.diagnostic", "statsmodels")
    lm_stat, lm_pvalue, f_stat, f_pvalue = diagnostic.het_breuschpagan(residuals, exog)
    return {
        "lm_stat": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }


def linear_reset(model: Any) -> dict[str, float | None]:
    diagnostic = require_dependency("statsmodels.stats.diagnostic", "statsmodels")
    try:
        result = diagnostic.linear_reset(model, power=2, use_f=True)
    except Exception:
        return {"statistic": None, "p_value": None}
    return {"statistic": safe_float(result.statistic), "p_value": safe_float(result.pvalue)}


def vif_table(exog: Any, columns: list[str]) -> list[dict[str, float | str]]:
    influence = require_dependency("statsmodels.stats.outliers_influence", "statsmodels")
    table: list[dict[str, float | str]] = []
    for idx, column in enumerate(columns):
        if str(column).lower() == "const":
            continue
        vif_value = influence.variance_inflation_factor(exog, idx)
        table.append({"predictor": str(column), "vif": float(vif_value)})
    return table


def cooks_distance(model: Any) -> dict[str, Any]:
    influence = model.get_influence()
    distances = influence.cooks_distance[0]
    leverage = getattr(influence, "hat_matrix_diag", None)
    studentized = getattr(influence, "resid_studentized_internal", None)
    return {
        "distances": distances,
        "max_distance": float(distances.max()),
        "leverage": leverage,
        "studentized_residuals": studentized,
    }


def standardized_residuals(model: Any) -> Any:
    influence = model.get_influence()
    studentized = getattr(influence, "resid_studentized_internal", None)
    if studentized is not None:
        return studentized
    return model.resid / model.resid.std(ddof=1)


def box_tidwell(
    X: Any,
    y: Any,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    deps = require_core_dependencies()
    np = deps["np"]
    pd = deps["pd"]
    sm = deps["sm"]
    X = X.copy()
    if not hasattr(X, "columns"):
        X = pd.DataFrame(X)
    y = to_1d_array(y, np)

    continuous_columns = [
        column
        for column in X.columns
        if np.issubdtype(X[column].dtype, np.number) and X[column].nunique(dropna=True) > 5
    ]
    if not continuous_columns:
        return {"results": [], "worst_predictor": None}

    transformed = X.copy()
    tested_columns: list[str] = []
    for column in continuous_columns:
        values = transformed[column].astype(float)
        offset = 0.0
        if values.min() <= 0:
            offset = abs(float(values.min())) + 1e-6
        positive_values = values + offset
        transformed[f"{column}__box_tidwell"] = positive_values * np.log(positive_values)
        tested_columns.append(column)

    design = sm.add_constant(transformed, has_constant="add")
    try:
        result = sm.Logit(y, design).fit(disp=0)
    except Exception as exc:
        return {"results": [], "worst_predictor": None, "error": str(exc)}

    rows = []
    for column in tested_columns:
        term = f"{column}__box_tidwell"
        p_value = safe_float(result.pvalues.get(term))
        rows.append(
            {
                "predictor": column,
                "p_value": p_value,
                "status": "FAIL" if p_value is not None and p_value < alpha else "PASS",
            }
        )

    worst = None
    if rows:
        worst = min(rows, key=lambda row: row["p_value"] if row["p_value"] is not None else 1.0)
    return {"results": rows, "worst_predictor": worst}


def roc_curve_points(y_true: Any, y_score: Any) -> dict[str, Any]:
    deps = require_core_dependencies()
    np = deps["np"]
    y_true = to_1d_array(y_true, np).astype(int)
    y_score = to_1d_array(y_score, np).astype(float)

    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    positives = (y_true == 1).sum()
    negatives = (y_true == 0).sum()
    if positives == 0 or negatives == 0:
        return {"fpr": np.array([0.0, 1.0]), "tpr": np.array([0.0, 1.0]), "auc": None}

    tps = 0
    fps = 0
    tpr = [0.0]
    fpr = [0.0]
    previous_score = None
    for truth, score in zip(y_true, y_score):
        if previous_score is not None and score != previous_score:
            tpr.append(tps / positives)
            fpr.append(fps / negatives)
        if truth == 1:
            tps += 1
        else:
            fps += 1
        previous_score = score
    tpr.append(tps / positives)
    fpr.append(fps / negatives)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(integrate(tpr, fpr))
    return {"fpr": np.asarray(fpr), "tpr": np.asarray(tpr), "auc": auc}


def format_metric_result(label: str, value: float | None) -> str:
    if value is None:
        return f"{label} = n/a"
    if "p" in label.lower():
        return f"{label} = {p_value_text(value)}"
    return f"{label} = {value:.3f}"
