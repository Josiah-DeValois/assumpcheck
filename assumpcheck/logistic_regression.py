from __future__ import annotations

from typing import Any

from .interpret import HEURISTIC_NOTE, combine_interpretation, mitigation
from .metrics import box_tidwell, cooks_distance, roc_curve_points, vif_table
from .plots import binned_logit_plot, residuals_vs_leverage_plot, roc_curve_plot
from .reporting import finalize_report
from .types import AssumptionCheck, AssumptionReport
from .utils import (
    add_constant,
    bool_from_user_input,
    p_value_text,
    require_core_dependencies,
    require_dependency,
    safe_float,
    to_1d_array,
    to_dataframe,
)


def check_logistic_regression(
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
):
    design_independent = bool_from_user_input(design_independent)
    fitted_model, feature_frame, outcome, fit_error = _resolve_logistic_inputs(model=model, X=X, y=y)

    checks = [
        _linearity_check(feature_frame, outcome, alpha),
        _independence_check(design_independent),
        _multicollinearity_check(feature_frame),
        _influence_check(fitted_model, len(outcome)),
        _separation_check(fitted_model, fit_error),
    ]
    diagnostics = []
    if fitted_model is not None:
        diagnostics.append(_fit_diagnostic(fitted_model, outcome))

    report = AssumptionReport(
        model_type="logistic_regression",
        checks=checks,
        diagnostics=diagnostics,
        title="LOGISTIC REGRESSION ASSUMPTION CHECKS",
    )
    return finalize_report(
        report,
        verbose=verbose,
        show_all=show_all,
        plots_on_fail=plots_on_fail,
        return_dict=return_dict,
    )


def _resolve_logistic_inputs(model: Any, X: Any, y: Any) -> tuple[Any, Any, Any, str | None]:
    deps = require_core_dependencies()
    np = deps["np"]
    pd = deps["pd"]
    sm = deps["sm"]

    if model is None:
        if X is None or y is None:
            raise ValueError("Provide either a fitted statsmodels logistic model or both X and y.")
        feature_frame = _named_feature_frame(X, pd)
        outcome = to_1d_array(y, np).astype(int)
        if feature_frame.shape[0] != outcome.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        design = add_constant(feature_frame, sm)
        sm_exceptions = require_dependency("statsmodels.tools.sm_exceptions", "statsmodels")
        try:
            fitted_model = sm.Logit(outcome, design).fit(disp=0)
            return fitted_model, feature_frame, outcome, None
        except sm_exceptions.PerfectSeparationError as exc:
            return None, feature_frame, outcome, str(exc)
        except Exception as exc:
            return None, feature_frame, outcome, str(exc)

    fitted_model = model
    outcome = np.asarray(fitted_model.model.endog).astype(int)
    exog_names = list(fitted_model.model.exog_names)
    feature_frame = pd.DataFrame(fitted_model.model.exog, columns=exog_names)
    if "const" in feature_frame.columns:
        feature_frame = feature_frame.drop(columns=["const"])
    return fitted_model, feature_frame, outcome, None


def _named_feature_frame(X: Any, pd: Any) -> Any:
    frame = to_dataframe(X, pd)
    if all(isinstance(column, int) for column in frame.columns):
        frame.columns = [f"x{i + 1}" for i in range(frame.shape[1])]
    return frame


def _linearity_check(feature_frame: Any, outcome: Any, alpha: float) -> AssumptionCheck:
    result = box_tidwell(feature_frame, outcome, alpha=alpha)
    if result.get("error"):
        return AssumptionCheck(
            assumption="Linearity in the log-odds",
            status="INFO",
            metric=result["error"],
            threshold=f"Box-Tidwell p < {alpha:.2f} suggests possible nonlinearity. {HEURISTIC_NOTE}",
            interpretation="Linearity in the log-odds could not be assessed cleanly for the supplied predictors.",
            mitigation=mitigation("logistic_linearity"),
        )

    rows = result["results"]
    if not rows:
        return AssumptionCheck(
            assumption="Linearity in the log-odds",
            status="INFO",
            threshold="Only continuous predictors are checked for log-odds linearity.",
            interpretation="No continuous predictors were available for a Box-Tidwell check.",
            mitigation=mitigation("logistic_linearity"),
        )

    worst = result["worst_predictor"]
    failing = [row for row in rows if row["status"] == "FAIL"]
    status = "FAIL" if failing else "PASS"
    interpretation = (
        "Continuous predictors do not show strong evidence against linearity in the log-odds."
        if status == "PASS"
        else f"Predictor '{worst['predictor']}' shows evidence of curvature on the log-odds scale."
    )
    plotter = lambda values=feature_frame[worst["predictor"]].to_numpy(), outcome=outcome, predictor=worst["predictor"]: binned_logit_plot(
        values,
        outcome,
        title=f"Logistic Regression: Binned Logit Plot for {predictor}",
    )
    return AssumptionCheck(
        assumption="Linearity in the log-odds",
        status=status,
        metric=f"Worst Box-Tidwell p = {p_value_text(safe_float(worst['p_value']))} ({worst['predictor']})",
        threshold=f"p < {alpha:.2f} suggests nonlinearity in the log-odds. {HEURISTIC_NOTE}",
        interpretation=interpretation,
        mitigation=[] if status == "PASS" else mitigation("logistic_linearity"),
        visual=f"Binned logit plot for {worst['predictor']}",
        details={"box_tidwell": rows},
        plotter=plotter,
    )


def _independence_check(design_independent: bool | None) -> AssumptionCheck:
    if design_independent is None:
        return AssumptionCheck(
            assumption="Independence",
            status="INFO",
            threshold="Verify independence from the study design rather than the data alone.",
            interpretation="Independence cannot usually be confirmed from logistic model output by itself.",
            mitigation=mitigation("logistic_independence"),
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
        interpretation="You indicated that observations are not independent, so standard logistic regression inference may be unreliable.",
        mitigation=mitigation("logistic_independence"),
    )


def _multicollinearity_check(feature_frame: Any) -> AssumptionCheck:
    deps = require_core_dependencies()
    sm = deps["sm"]
    design = add_constant(feature_frame, sm)
    rows = vif_table(design.to_numpy(), list(design.columns))
    if not rows:
        return AssumptionCheck(
            assumption="Multicollinearity",
            status="INFO",
            interpretation="VIF values were not available for this model specification.",
            mitigation=mitigation("logistic_multicollinearity"),
        )
    worst = max(rows, key=lambda row: row["vif"])
    max_vif = float(worst["vif"])
    if max_vif > 10:
        status = "FAIL"
        interpretation = f"Predictor '{worst['predictor']}' has a VIF high enough to destabilize coefficient estimates."
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
        mitigation=[] if status == "PASS" else mitigation("logistic_multicollinearity"),
        details={"vif": rows},
    )


def _influence_check(model: Any, sample_size: int) -> AssumptionCheck:
    if model is None:
        return AssumptionCheck(
            assumption="Extreme influential points",
            status="INFO",
            interpretation="Influence diagnostics require a fitted logistic model.",
            mitigation=mitigation("logistic_influence"),
        )
    try:
        influence_result = cooks_distance(model)
    except Exception as exc:
        return AssumptionCheck(
            assumption="Extreme influential points",
            status="INFO",
            metric=str(exc),
            interpretation="This fitted model did not expose a usable influence calculation.",
            mitigation=mitigation("logistic_influence"),
        )

    max_distance = float(influence_result["max_distance"])
    threshold = 4 / sample_size
    flagged = int((influence_result["distances"] > threshold).sum())
    if max_distance > threshold:
        status = "FAIL"
        interpretation = "At least one case exceeds the common Cook's distance concern threshold."
    elif max_distance > (2 / sample_size):
        status = "WARN"
        interpretation = "There may be a moderately influential case worth checking."
    else:
        status = "PASS"
        interpretation = "No highly influential cases were flagged by Cook's distance."

    leverage = influence_result["leverage"]
    studentized = influence_result["studentized_residuals"]
    plotter = None
    if leverage is not None and studentized is not None:
        plotter = lambda leverage=leverage, studentized=studentized, distances=influence_result["distances"]: residuals_vs_leverage_plot(
            leverage,
            studentized,
            distances,
            title="Logistic Regression: Residuals vs Leverage",
        )

    return AssumptionCheck(
        assumption="Extreme influential points",
        status=status,
        metric=f"Max Cook's D = {max_distance:.3f}; flagged points > 4/n: {flagged}",
        threshold=f"Cook's D > 4/n is concerning; here 4/n = {threshold:.3f}.",
        interpretation=interpretation,
        mitigation=[] if status == "PASS" else mitigation("logistic_influence"),
        visual="Residuals vs leverage" if plotter else None,
        details={
            "max_cooks_distance": max_distance,
            "flagged_gt_4_over_n": flagged,
            "threshold_4_over_n": threshold,
        },
        plotter=plotter,
    )


def _separation_check(model: Any, fit_error: str | None) -> AssumptionCheck:
    if fit_error:
        return AssumptionCheck(
            assumption="Adequate sample / no separation",
            status="FAIL",
            metric=fit_error,
            threshold="The model should converge normally without signs of complete or quasi-complete separation.",
            interpretation="The logistic fit failed, which often indicates separation, sparse cells, or an unstable specification.",
            mitigation=mitigation("logistic_separation"),
        )
    if model is None:
        return AssumptionCheck(
            assumption="Adequate sample / no separation",
            status="INFO",
            interpretation="Separation could not be assessed because the model was not available.",
            mitigation=mitigation("logistic_separation"),
        )

    deps = require_core_dependencies()
    np = deps["np"]
    converged = getattr(model, "converged", None)
    params = np.asarray(model.params, dtype=float)
    bse = np.asarray(model.bse, dtype=float)
    max_coef = float(np.max(np.abs(params)))
    max_se = float(np.max(np.abs(bse)))

    if converged is False or max_coef > 20 or max_se > 20:
        status = "FAIL"
        interpretation = "The fit shows strong signs of separation or sparse-data instability."
    elif max_coef > 10 or max_se > 10:
        status = "WARN"
        interpretation = "The fit converged, but very large coefficients or standard errors suggest possible separation."
    else:
        status = "PASS"
        interpretation = "The fit converged without obvious signs of severe separation."

    converged_text = "yes" if converged is not False else "no"
    return AssumptionCheck(
        assumption="Adequate sample / no separation",
        status=status,
        metric=f"Converged = {converged_text}; max |coef| = {max_coef:.3f}; max SE = {max_se:.3f}",
        threshold="Non-convergence or very large coefficients / SEs can indicate separation or sparse data.",
        interpretation=interpretation,
        mitigation=[] if status == "PASS" else mitigation("logistic_separation"),
        details={"converged": converged, "max_abs_coef": max_coef, "max_se": max_se},
    )


def _fit_diagnostic(model: Any, outcome: Any) -> AssumptionCheck:
    predicted = model.predict()
    roc = roc_curve_points(outcome, predicted)
    auc = roc["auc"]
    if auc is None:
        interpretation = "AUC is unavailable because the observed outcome contains only one class."
    elif auc < 0.60:
        interpretation = "Discrimination is weak."
    elif auc < 0.70:
        interpretation = "Discrimination is modest."
    elif auc < 0.80:
        interpretation = "Discrimination is acceptable."
    elif auc < 0.90:
        interpretation = "Discrimination is good."
    else:
        interpretation = "Discrimination is excellent."

    return AssumptionCheck(
        assumption="Model fit summary",
        status="INFO",
        metric="AUC = n/a" if auc is None else f"AUC = {auc:.3f}",
        threshold="AUC around 0.50 indicates no discrimination; higher values indicate better separation.",
        interpretation=interpretation,
        mitigation=mitigation("logistic_fit") if auc is not None and auc < 0.70 else [],
        visual="ROC curve",
        category="diagnostic",
        details={"auc": auc},
        plotter=lambda fpr=roc["fpr"], tpr=roc["tpr"], auc=auc: roc_curve_plot(
            fpr,
            tpr,
            auc,
            title="Logistic Regression: ROC Curve",
        ),
    )
