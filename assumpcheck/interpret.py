from __future__ import annotations

from typing import Iterable

HEURISTIC_NOTE = "Thresholds here are heuristics, not hard laws."

MITIGATIONS = {
    "anova_independence": [
        "Use repeated-measures ANOVA if observations are repeated.",
        "Use a mixed-effects model if data are clustered.",
        "Rework the analysis if dependence was introduced by the study design.",
    ],
    "anova_normality": [
        "Check for outliers before changing the model.",
        "Transform the response with a log, square-root, or Box-Cox transform when appropriate.",
        "Consider a robust or nonparametric alternative such as Kruskal-Wallis if the problem is substantial.",
    ],
    "anova_variance": [
        "Try transforming the response.",
        "Use Welch's ANOVA when group variances differ.",
        "Check whether unequal spread is being driven by skew or outliers.",
    ],
    "anova_outliers": [
        "Verify data entry for flagged cases.",
        "Check whether the observation is legitimate but unusual.",
        "Consider a transformation, robust method, or nonparametric alternative if outliers remain influential.",
    ],
    "linear_linearity": [
        "Add polynomial terms if curvature is plausible.",
        "Add interactions if the effect changes across predictors.",
        "Transform predictors or the response if the scale is causing the pattern.",
        "Use a more flexible model form if the relationship is clearly nonlinear.",
    ],
    "linear_independence": [
        "Use time-series methods for ordered data with autocorrelation.",
        "Add lag terms if the process is genuinely sequential.",
        "Use clustered or correlated error structures when dependence is expected.",
    ],
    "linear_normality": [
        "Check for outliers before transforming.",
        "Transform the response with a log, square-root, or Box-Cox transform when appropriate.",
        "Use robust methods if inference is sensitive to non-normal residuals.",
    ],
    "linear_homoscedasticity": [
        "Transform the response.",
        "Use heteroscedasticity-robust standard errors.",
        "Consider weighted least squares if the variance pattern is known.",
        "Inspect the model for missing structure or nonlinearity.",
    ],
    "linear_multicollinearity": [
        "Remove or combine redundant predictors.",
        "Keep only predictors that matter substantively.",
        "Consider regularization if prediction matters more than coefficient interpretation.",
    ],
    "linear_influence": [
        "Verify that influential observations are valid.",
        "Compare the model with and without flagged cases.",
        "Use robust regression or re-specify the model if single points are driving the fit.",
    ],
    "logistic_linearity": [
        "Transform the predictor if the effect is curved on the log-odds scale.",
        "Add polynomial terms or splines for continuous predictors.",
        "Bin predictors only when there is a substantive reason to do so.",
    ],
    "logistic_independence": [
        "Use GEE, mixed-effects logistic regression, or clustered standard errors when data are dependent.",
        "Rework the analysis if repeated measures or clustering were ignored.",
    ],
    "logistic_multicollinearity": [
        "Remove or combine redundant predictors.",
        "Use regularization if stable prediction matters more than coefficient interpretation.",
    ],
    "logistic_influence": [
        "Verify unusual cases before making changes.",
        "Compare the fit with and without flagged cases.",
        "Re-specify the model if influential cases reveal a structural issue.",
    ],
    "logistic_separation": [
        "Collect more data if the current sample is sparse.",
        "Combine rare levels or simplify sparse predictors when justified.",
        "Use penalized logistic regression if separation is present.",
    ],
    "logistic_fit": [
        "Add more predictive structure with better features, interactions, or nonlinear terms.",
        "Re-specify the model if discrimination is weak.",
    ],
}


def mitigation(key: str) -> list[str]:
    return list(MITIGATIONS.get(key, []))


def combine_interpretation(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def bullets(items: Iterable[str]) -> list[str]:
    return [item for item in items if item]

