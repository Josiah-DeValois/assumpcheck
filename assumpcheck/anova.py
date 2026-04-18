from __future__ import annotations

from typing import Any

from .interpret import HEURISTIC_NOTE, combine_interpretation, mitigation
from .metrics import levene_test, shapiro_wilk, standardized_residuals
from .plots import boxplot_by_group, qq_plot
from .reporting import finalize_report
from .types import AssumptionCheck, AssumptionReport
from .utils import bool_from_user_input, p_value_text, require_core_dependencies, require_dependency, to_1d_array


def check_anova(
    model: Any = None,
    y: Any = None,
    groups: Any = None,
    *,
    alpha: float = 0.05,
    show_all: bool = False,
    plots_on_fail: bool = True,
    verbose: bool = False,
    return_dict: bool = False,
    design_independent: bool | None = None,
):
    design_independent = bool_from_user_input(design_independent)
    fitted_model, values, group_values = _resolve_anova_inputs(model=model, y=y, groups=groups)
    residuals = fitted_model.resid
    studentized = standardized_residuals(fitted_model)

    checks = [
        _independence_check(design_independent),
        _normality_check(residuals, alpha),
        _variance_check(values, group_values, alpha),
        _outlier_check(values, group_values, studentized),
    ]
    report = AssumptionReport(
        model_type="anova",
        checks=checks,
        title="ANOVA ASSUMPTION CHECKS",
    )
    return finalize_report(
        report,
        verbose=verbose,
        show_all=show_all,
        plots_on_fail=plots_on_fail,
        return_dict=return_dict,
    )


def _resolve_anova_inputs(model: Any, y: Any, groups: Any) -> tuple[Any, Any, Any]:
    deps = require_core_dependencies()
    np = deps["np"]
    pd = deps["pd"]

    values = to_1d_array(y, np) if y is not None else None
    group_values = to_1d_array(groups, np) if groups is not None else None

    if model is None:
        if values is None or group_values is None:
            raise ValueError("Provide either a fitted statsmodels ANOVA model or both y and groups.")
        if values.shape[0] != group_values.shape[0]:
            raise ValueError("y and groups must have the same length.")
        smf = require_dependency("statsmodels.formula.api", "statsmodels")
        frame = pd.DataFrame({"response": values, "group": group_values})
        fitted_model = smf.ols("response ~ C(group)", data=frame).fit()
        return fitted_model, values, group_values

    fitted_model = model
    if values is None:
        values = np.asarray(fitted_model.model.endog)
    if group_values is None:
        frame = getattr(getattr(fitted_model.model, "data", None), "frame", None)
        response_name = getattr(fitted_model.model, "endog_names", None)
        if frame is not None:
            if response_name in frame:
                values = frame[response_name].to_numpy()
            candidate_columns = [column for column in frame.columns if column != response_name]
            if len(candidate_columns) == 1:
                group_values = frame[candidate_columns[0]].to_numpy()
    return fitted_model, values, group_values


def _independence_check(design_independent: bool | None) -> AssumptionCheck:
    if design_independent is None:
        return AssumptionCheck(
            assumption="Independence",
            status="INFO",
            threshold="Verify independence from the study design rather than the data alone.",
            interpretation="Independence is usually a design question, so this package does not infer it automatically.",
            mitigation=mitigation("anova_independence"),
        )
    if design_independent:
        return AssumptionCheck(
            assumption="Independence",
            status="PASS",
            interpretation="You indicated that the study design supports independent observations.",
        )
    return AssumptionCheck(
        assumption="Independence",
        status="FAIL",
        interpretation="You indicated that observations are not independent, so standard ANOVA inference may be invalid.",
        mitigation=mitigation("anova_independence"),
    )


def _normality_check(residuals: Any, alpha: float) -> AssumptionCheck:
    result = shapiro_wilk(residuals)
    p_value = result["p_value"]
    note = result.get("note")
    if p_value is None:
        return AssumptionCheck(
            assumption="Normality of residuals",
            status="INFO",
            metric=note,
            threshold=f"Shapiro-Wilk requires at least 3 residuals. {HEURISTIC_NOTE}",
            interpretation="There is not enough data to evaluate residual normality.",
            visual="Q-Q plot of residuals",
            plotter=lambda residuals=residuals: qq_plot(residuals, title="ANOVA: Q-Q Plot of Residuals"),
        )

    status = "PASS" if p_value >= alpha else "FAIL"
    interpretation = (
        "Residuals do not show strong evidence against normality."
        if status == "PASS"
        else combine_interpretation(
            "Residuals may depart from normality.",
            "Use the Q-Q plot alongside the p-value, especially in large samples.",
        )
    )
    return AssumptionCheck(
        assumption="Normality of residuals",
        status=status,
        metric=f"Shapiro-Wilk p = {p_value_text(p_value)}",
        threshold=f"p < {alpha:.2f} suggests possible non-normality. {HEURISTIC_NOTE}",
        interpretation=interpretation,
        mitigation=[] if status == "PASS" else mitigation("anova_normality"),
        visual="Q-Q plot of residuals",
        details=result,
        plotter=lambda residuals=residuals: qq_plot(residuals, title="ANOVA: Q-Q Plot of Residuals"),
    )


def _variance_check(values: Any, groups: Any, alpha: float) -> AssumptionCheck:
    if groups is None:
        return AssumptionCheck(
            assumption="Equal variance across groups",
            status="INFO",
            threshold="Original group assignments are required to run Levene's test.",
            interpretation="Homogeneity of variance could not be checked because group labels were not available.",
            mitigation=mitigation("anova_variance"),
        )

    result = levene_test(values, groups)
    p_value = result["p_value"]
    if p_value is None:
        return AssumptionCheck(
            assumption="Equal variance across groups",
            status="INFO",
            metric=result.get("note"),
            threshold=f"Levene's test needs at least two groups. {HEURISTIC_NOTE}",
            interpretation="Homogeneity of variance could not be evaluated from the supplied data.",
            visual="Boxplot by group",
            plotter=lambda values=values, groups=groups: boxplot_by_group(
                values,
                groups,
                title="ANOVA: Boxplot by Group",
                y_label="Response",
            ),
        )

    status = "PASS" if p_value >= alpha else "FAIL"
    interpretation = (
        "Group spreads look reasonable for a standard ANOVA."
        if status == "PASS"
        else "Group variances may differ enough to affect the standard ANOVA."
    )
    return AssumptionCheck(
        assumption="Equal variance across groups",
        status=status,
        metric=f"Levene's test p = {p_value_text(p_value)}",
        threshold=f"p < {alpha:.2f} suggests unequal variance. {HEURISTIC_NOTE}",
        interpretation=interpretation,
        mitigation=[] if status == "PASS" else mitigation("anova_variance"),
        visual="Boxplot by group",
        details=result,
        plotter=lambda values=values, groups=groups: boxplot_by_group(
            values,
            groups,
            title="ANOVA: Boxplot by Group",
            y_label="Response",
        ),
    )


def _outlier_check(values: Any, groups: Any, studentized: Any) -> AssumptionCheck:
    deps = require_core_dependencies()
    np = deps["np"]
    max_abs = float(np.max(np.abs(studentized)))
    flagged = int(np.sum(np.abs(studentized) > 3))
    if max_abs > 3:
        status = "FAIL"
        interpretation = "At least one observation has a standardized residual above the common concern threshold."
    elif max_abs > 2:
        status = "WARN"
        interpretation = "A residual is moderately large and worth checking, even though it is not extreme."
    else:
        status = "PASS"
        interpretation = "No extreme residual outliers were detected by the common standardized residual heuristic."

    plotter = None
    if groups is not None:
        plotter = lambda values=values, groups=groups: boxplot_by_group(
            values,
            groups,
            title="ANOVA: Boxplot by Group",
            y_label="Response",
        )

    return AssumptionCheck(
        assumption="Extreme outliers",
        status=status,
        metric=f"Max |standardized residual| = {max_abs:.3f}; flagged points > 3: {flagged}",
        threshold="Values above 2 deserve review and values above 3 are concerning.",
        interpretation=interpretation,
        mitigation=[] if status == "PASS" else mitigation("anova_outliers"),
        visual="Boxplot by group" if groups is not None else None,
        details={"max_abs_standardized_residual": max_abs, "flagged_gt_3": flagged},
        plotter=plotter,
    )
