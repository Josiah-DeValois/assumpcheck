from __future__ import annotations

from typing import Any

from .interpret import HEURISTIC_NOTE, combine_interpretation, mitigation
from .metrics import breusch_pagan, cooks_distance, durbin_watson, linear_reset, shapiro_wilk, vif_table
from .plots import qq_plot, residuals_vs_fitted_plot, residuals_vs_leverage_plot
from .reporting import finalize_report
from .types import AssumptionCheck, AssumptionReport
from .utils import (
    add_constant,
    bool_from_user_input,
    p_value_text,
    require_core_dependencies,
    to_1d_array,
    to_dataframe,
)


def check_linear_regression(
    model: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    alpha: float = 0.05,
    show_all: bool = False,
    plots_on_fail: bool = True,
    verbose: bool = False,
    return_dict: bool = False,
    design_independent: bool | None = None,
    ordered: bool = False,
):
    """Check ordinary linear regression assumptions with statsmodels-first support.

    Supply either a fitted statsmodels OLS model or raw ``X`` and ``y``.
    The function prints a pytest-style summary by default, optionally shows
    diagnostic plots, and returns either an ``AssumptionReport`` or a
    serializable dictionary when ``return_dict=True``.
    """

    design_independent = bool_from_user_input(design_independent)
    fitted_model, design_frame, outcome = _resolve_linear_inputs(model=model, X=X, y=y)
    residuals = fitted_model.resid
    fitted = fitted_model.fittedvalues
    exog = fitted_model.model.exog
    exog_names = list(fitted_model.model.exog_names)

    influence_result = cooks_distance(fitted_model)
    checks = [
        _linearity_check(fitted_model, fitted, residuals, alpha),
        _independence_check(residuals, design_independent, ordered),
        _normality_check(residuals, alpha),
        _homoscedasticity_check(residuals, exog, fitted, alpha),
        _multicollinearity_check(exog, exog_names),
        _influence_check(influence_result, len(outcome)),
    ]
    report = AssumptionReport(
        model_type="linear_regression",
        checks=checks,
        title="LINEAR REGRESSION ASSUMPTION CHECKS",
    )
    return finalize_report(
        report,
        verbose=verbose,
        show_all=show_all,
        plots_on_fail=plots_on_fail,
        return_dict=return_dict,
    )


def _resolve_linear_inputs(model: Any, X: Any, y: Any) -> tuple[Any, Any, Any]:
    deps = require_core_dependencies()
    np = deps["np"]
    pd = deps["pd"]
    sm = deps["sm"]

    if model is None:
        if X is None or y is None:
            raise ValueError("Provide either a fitted statsmodels OLS model or both X and y.")
        design_frame = _named_feature_frame(X, pd)
        outcome = to_1d_array(y, np)
        if design_frame.shape[0] != outcome.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        design = add_constant(design_frame, sm)
        fitted_model = sm.OLS(outcome, design).fit()
        return fitted_model, design_frame, outcome

    fitted_model = model
    exog_names = list(fitted_model.model.exog_names)
    design_frame = pd.DataFrame(fitted_model.model.exog, columns=exog_names)
    outcome = np.asarray(fitted_model.model.endog)
    if "const" in design_frame.columns:
        design_frame = design_frame.drop(columns=["const"])
    return fitted_model, design_frame, outcome


def _named_feature_frame(X: Any, pd: Any) -> Any:
    frame = to_dataframe(X, pd)
    if all(isinstance(column, int) for column in frame.columns):
        frame.columns = [f"x{i + 1}" for i in range(frame.shape[1])]
    return frame


def _linearity_check(model: Any, fitted: Any, residuals: Any, alpha: float) -> AssumptionCheck:
    result = linear_reset(model)
    p_value = result["p_value"]
    if p_value is None:
        status = "INFO"
        metric = "Visual check only"
        interpretation = "Inspect the residuals vs fitted plot for curvature or other systematic structure."
        mitigation_items = mitigation("linear_linearity")
    elif p_value < alpha:
        status = "WARN"
        metric = f"Ramsey RESET p = {p_value_text(p_value)}"
        interpretation = combine_interpretation(
            "There may be curvature or missing structure in the model.",
            "Use the residuals vs fitted plot as the primary diagnostic.",
        )
        mitigation_items = mitigation("linear_linearity")
    else:
        status = "PASS"
        metric = f"Ramsey RESET p = {p_value_text(p_value)}"
        interpretation = "Residual diagnostics do not show strong evidence of nonlinearity."
        mitigation_items = []

    return AssumptionCheck(
        assumption="Linearity",
        status=status,
        metric=metric,
        threshold=f"Residuals should look like a random cloud; RESET p < {alpha:.2f} is a warning sign, not proof. {HEURISTIC_NOTE}",
        interpretation=interpretation,
        mitigation=mitigation_items,
        visual="Residuals vs fitted",
        details=result,
        plotter=lambda fitted=fitted, residuals=residuals: residuals_vs_fitted_plot(
            fitted,
            residuals,
            title="Linear Regression: Residuals vs Fitted",
        ),
    )


def _independence_check(residuals: Any, design_independent: bool | None, ordered: bool) -> AssumptionCheck:
    dw_value = durbin_watson(residuals) if ordered else None
    threshold = "Around 2 is ideal; values below 1.5 or above 2.5 suggest autocorrelation." if ordered else (
        "Verify independence from the study design rather than the residuals alone."
    )
    metric = f"Durbin-Watson = {dw_value:.3f}" if dw_value is not None else None

    if design_independent is None:
        interpretation = "Independence is primarily a design question."
        if ordered and dw_value is not None:
            interpretation = combine_interpretation(
                interpretation,
                "Durbin-Watson is included as a supplemental ordered-data check.",
            )
        return AssumptionCheck(
            assumption="Independence",
            status="INFO",
            metric=metric,
            threshold=threshold,
            interpretation=interpretation,
            mitigation=mitigation("linear_independence"),
        )

    if not design_independent:
        return AssumptionCheck(
            assumption="Independence",
            status="FAIL",
            metric=metric,
            threshold=threshold,
            interpretation="You indicated that the errors are not independent, so standard regression inference may be unreliable.",
            mitigation=mitigation("linear_independence"),
        )

    if ordered and dw_value is not None:
        if dw_value < 1.5 or dw_value > 2.5:
            status = "FAIL"
            interpretation = "Durbin-Watson suggests meaningful autocorrelation in the residuals."
        elif dw_value < 1.75 or dw_value > 2.25:
            status = "WARN"
            interpretation = "Durbin-Watson is somewhat away from 2, so residual dependence may be worth checking."
        else:
            status = "PASS"
            interpretation = "The design was marked independent and Durbin-Watson does not suggest strong autocorrelation."
        return AssumptionCheck(
            assumption="Independence",
            status=status,
            metric=metric,
            threshold=threshold,
            interpretation=interpretation,
            mitigation=[] if status == "PASS" else mitigation("linear_independence"),
        )

    return AssumptionCheck(
        assumption="Independence",
        status="PASS",
        interpretation="You indicated that the study design supports independent errors.",
    )


def _normality_check(residuals: Any, alpha: float) -> AssumptionCheck:
    result = shapiro_wilk(residuals)
    p_value = result["p_value"]
    if p_value is None:
        return AssumptionCheck(
            assumption="Normality of residuals",
            status="INFO",
            metric=result.get("note"),
            threshold=f"Shapiro-Wilk requires at least 3 residuals. {HEURISTIC_NOTE}",
            interpretation="Residual normality could not be evaluated from the supplied model.",
            mitigation=mitigation("linear_normality"),
            visual="Q-Q plot of residuals",
            plotter=lambda residuals=residuals: qq_plot(residuals, title="Linear Regression: Q-Q Plot of Residuals"),
        )

    status = "PASS" if p_value >= alpha else "FAIL"
    interpretation = (
        "Residuals do not show strong evidence against normality."
        if status == "PASS"
        else "Residuals may depart from normality, especially in the tails."
    )
    return AssumptionCheck(
        assumption="Normality of residuals",
        status=status,
        metric=f"Shapiro-Wilk p = {p_value_text(p_value)}",
        threshold=f"p < {alpha:.2f} suggests possible non-normality. {HEURISTIC_NOTE}",
        interpretation=interpretation,
        mitigation=[] if status == "PASS" else mitigation("linear_normality"),
        visual="Q-Q plot of residuals",
        details=result,
        plotter=lambda residuals=residuals: qq_plot(residuals, title="Linear Regression: Q-Q Plot of Residuals"),
    )


def _homoscedasticity_check(residuals: Any, exog: Any, fitted: Any, alpha: float) -> AssumptionCheck:
    result = breusch_pagan(residuals, exog)
    p_value = result["lm_pvalue"]
    status = "PASS" if p_value >= alpha else "FAIL"
    interpretation = (
        "Residual spread looks broadly compatible with constant variance."
        if status == "PASS"
        else "Residual variance may change across the fitted range."
    )
    return AssumptionCheck(
        assumption="Homoscedasticity",
        status=status,
        metric=f"Breusch-Pagan p = {p_value_text(p_value)}",
        threshold=f"p < {alpha:.2f} suggests heteroscedasticity. {HEURISTIC_NOTE}",
        interpretation=interpretation,
        mitigation=[] if status == "PASS" else mitigation("linear_homoscedasticity"),
        visual="Residuals vs fitted",
        details=result,
        plotter=lambda fitted=fitted, residuals=residuals: residuals_vs_fitted_plot(
            fitted,
            residuals,
            title="Linear Regression: Residuals vs Fitted",
        ),
    )


def _multicollinearity_check(exog: Any, exog_names: list[str]) -> AssumptionCheck:
    rows = vif_table(exog, exog_names)
    if not rows:
        return AssumptionCheck(
            assumption="Multicollinearity",
            status="INFO",
            interpretation="VIF values were not available for this model specification.",
            mitigation=mitigation("linear_multicollinearity"),
        )
    worst = max(rows, key=lambda row: row["vif"])
    max_vif = float(worst["vif"])
    if max_vif > 10:
        status = "FAIL"
        interpretation = f"Predictor '{worst['predictor']}' has a VIF high enough to make coefficients unstable."
    elif max_vif > 5:
        status = "WARN"
        interpretation = f"Predictor '{worst['predictor']}' shows noticeable collinearity."
    else:
        status = "PASS"
        interpretation = "Predictor VIF values are in a commonly acceptable range."
    return AssumptionCheck(
        assumption="Multicollinearity",
        status=status,
        metric=f"Max VIF = {max_vif:.3f} ({worst['predictor']})",
        threshold="VIF > 5 deserves caution and VIF > 10 is a serious concern.",
        interpretation=interpretation,
        mitigation=[] if status == "PASS" else mitigation("linear_multicollinearity"),
        details={"vif": rows},
    )


def _influence_check(influence_result: dict[str, Any], sample_size: int) -> AssumptionCheck:
    max_distance = float(influence_result["max_distance"])
    threshold = 4 / sample_size
    warned_threshold = 2 / sample_size
    flagged = int((influence_result["distances"] > threshold).sum())
    if max_distance > threshold:
        status = "FAIL"
        interpretation = "At least one observation exceeds the common Cook's distance concern threshold."
    elif max_distance > warned_threshold:
        status = "WARN"
        interpretation = "There may be a moderately influential point worth reviewing."
    else:
        status = "PASS"
        interpretation = "No highly influential observations were flagged by Cook's distance."

    leverage = influence_result["leverage"]
    studentized = influence_result["studentized_residuals"]
    plotter = None
    if leverage is not None and studentized is not None:
        plotter = lambda leverage=leverage, studentized=studentized, distances=influence_result["distances"]: residuals_vs_leverage_plot(
            leverage,
            studentized,
            distances,
            title="Linear Regression: Residuals vs Leverage",
        )

    return AssumptionCheck(
        assumption="Extreme influential points",
        status=status,
        metric=f"Max Cook's D = {max_distance:.3f}; flagged points > 4/n: {flagged}",
        threshold=f"Cook's D > 4/n is concerning; here 4/n = {threshold:.3f}.",
        interpretation=interpretation,
        mitigation=[] if status == "PASS" else mitigation("linear_influence"),
        visual="Residuals vs leverage" if plotter else None,
        details={
            "max_cooks_distance": max_distance,
            "flagged_gt_4_over_n": flagged,
            "threshold_4_over_n": threshold,
        },
        plotter=plotter,
    )
