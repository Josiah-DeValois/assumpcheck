# assumpcheck workflow

This guide shows a realistic first workflow for each public checker.

The snippets below are deterministic and the console output was captured from the current package implementation.

## Setup

```python
import numpy as np
import pandas as pd

from assumpcheck import (
    check_anova,
    check_linear_regression,
    check_logistic_regression,
)
```

> Note
>
> The current MVP uses conservative influence heuristics. In the clean linear and logistic examples, the package still emits a mild `WARN` for influential points even though the rest of the diagnostics look reasonable.

## ANOVA

### Clean Example

A small balanced one-way ANOVA example where all four current checks pass.

```python
groups = np.repeat(["A", "B", "C"], 12)
base = np.array([9.80, 9.90, 10.00, 10.10, 10.20, 9.95, 10.05, 10.00, 9.85, 10.15, 9.92, 10.08])
y = np.concatenate([base, base + 0.40, base + 0.80]) + np.tile(np.linspace(-0.05, 0.05, 12), 3)

anova_clean = check_anova(
    y=y,
    groups=groups,
    design_independent=True,
    plots_on_fail=False,
    return_dict=True,
)

print()
print("Returned summary:", anova_clean["summary"])
```

Expected console output:

```text
ANOVA ASSUMPTION CHECKS
[PASS] Independence
[PASS] Normality of residuals
[PASS] Equal variance across groups
[PASS] Extreme outliers

Summary: 4 pass

Returned summary: {'pass': 4, 'fail': 0, 'warn': 0, 'info': 0}
```

### Problematic Example

A similar ANOVA setup with an injected outlier and extra spread in one group.

```python
groups = np.repeat(["A", "B", "C"], 12)
base = np.array([9.80, 9.90, 10.00, 10.10, 10.20, 9.95, 10.05, 10.00, 9.85, 10.15, 9.92, 10.08])
y = np.concatenate([base, base + 0.40, base + 0.80]) + np.tile(np.linspace(-0.05, 0.05, 12), 3)
y[4] = 14.50
y[24:36] = y[24:36] + np.linspace(-0.45, 0.45, 12)

anova_fail = check_anova(
    y=y,
    groups=groups,
    design_independent=True,
    return_dict=True,
)

print()
print("Returned summary:", anova_fail["summary"])
```

Expected console output:

```text
ANOVA ASSUMPTION CHECKS
[PASS] Independence
[FAIL] Normality of residuals
[PASS] Equal variance across groups
[FAIL] Extreme outliers

Summary: 2 pass, 2 fail

Details:
- Normality of residuals [FAIL]
  Metric: Shapiro-Wilk p = < 0.001
  Threshold: p < 0.05 suggests possible non-normality. Thresholds here are heuristics, not hard laws.
  Interpretation: Residuals may depart from normality. Use the Q-Q plot alongside the p-value, especially in large samples.
  Visual: Q-Q plot of residuals
  Possible mitigation:
    - Check for outliers before changing the model.
    - Transform the response with a log, square-root, or Box-Cox transform when appropriate.
    - Consider a robust or nonparametric alternative such as Kruskal-Wallis if the problem is substantial.
- Extreme outliers [FAIL]
  Metric: Max |standardized residual| = 5.469; flagged points > 3: 1
  Threshold: Values above 2 deserve review and values above 3 are concerning.
  Interpretation: At least one observation has a standardized residual above the common concern threshold.
  Visual: Boxplot by group
  Possible mitigation:
    - Verify data entry for flagged cases.
    - Check whether the observation is legitimate but unusual.
    - Consider a transformation, robust method, or nonparametric alternative if outliers remain influential.

Returned summary: {'pass': 2, 'fail': 2, 'warn': 0, 'info': 0}
```

![ANOVA Problematic Example plot](assets/anova_fail_1.png)

![ANOVA Problematic Example plot](assets/anova_fail_2.png)

### Full Diagnostics

Even when the model looks fine, you can request every available metric and plot.

```python
groups = np.repeat(["A", "B", "C"], 12)
base = np.array([9.80, 9.90, 10.00, 10.10, 10.20, 9.95, 10.05, 10.00, 9.85, 10.15, 9.92, 10.08])
y = np.concatenate([base, base + 0.40, base + 0.80]) + np.tile(np.linspace(-0.05, 0.05, 12), 3)

anova_full = check_anova(
    y=y,
    groups=groups,
    design_independent=True,
    show_all=True,
    return_dict=True,
)

print()
print("Returned summary:", anova_full["summary"])
```

Expected console output:

```text
ANOVA ASSUMPTION CHECKS
[PASS] Independence
[PASS] Normality of residuals
[PASS] Equal variance across groups
[PASS] Extreme outliers

Summary: 4 pass

Details:
- Independence [PASS]
  Interpretation: You indicated that the study design supports independent observations.
- Normality of residuals [PASS]
  Metric: Shapiro-Wilk p = 0.062
  Threshold: p < 0.05 suggests possible non-normality. Thresholds here are heuristics, not hard laws.
  Interpretation: Residuals do not show strong evidence against normality.
  Visual: Q-Q plot of residuals
- Equal variance across groups [PASS]
  Metric: Levene's test p = 1.000
  Threshold: p < 0.05 suggests unequal variance. Thresholds here are heuristics, not hard laws.
  Interpretation: Group spreads look reasonable for a standard ANOVA.
  Visual: Boxplot by group
- Extreme outliers [PASS]
  Metric: Max |standardized residual| = 1.947; flagged points > 3: 0
  Threshold: Values above 2 deserve review and values above 3 are concerning.
  Interpretation: No extreme residual outliers were detected by the common standardized residual heuristic.
  Visual: Boxplot by group

Returned summary: {'pass': 4, 'fail': 0, 'warn': 0, 'info': 0}
```

![ANOVA Full Diagnostics plot](assets/anova_show_all_1.png)

![ANOVA Full Diagnostics plot](assets/anova_show_all_2.png)

![ANOVA Full Diagnostics plot](assets/anova_show_all_3.png)

## Linear Regression

### Clean Example

A well-behaved synthetic linear model. Under the current MVP, the influence check still gives a mild warning.

```python
rng = np.random.default_rng(27)
X = pd.DataFrame({
    "x1": rng.normal(size=35),
    "x2": rng.normal(size=35),
})
y = 1.0 + 1.8 * X["x1"] - 0.6 * X["x2"] + rng.normal(scale=0.35, size=35)

linear_clean = check_linear_regression(
    X=X,
    y=y,
    design_independent=True,
    plots_on_fail=False,
    return_dict=True,
)

print()
print("Returned summary:", linear_clean["summary"])
```

Expected console output:

```text
LINEAR REGRESSION ASSUMPTION CHECKS
[PASS] Linearity
[PASS] Independence
[PASS] Normality of residuals
[PASS] Homoscedasticity
[PASS] Multicollinearity
[WARN] Extreme influential points

Summary: 5 pass, 1 warn

Details:
- Extreme influential points [WARN]
  Metric: Max Cook's D = 0.097; flagged points > 4/n: 0
  Threshold: Cook's D > 4/n is concerning; here 4/n = 0.114.
  Interpretation: There may be a moderately influential point worth reviewing.
  Visual: Residuals vs leverage
  Possible mitigation:
    - Verify that influential observations are valid.
    - Compare the model with and without flagged cases.
    - Use robust regression or re-specify the model if single points are driving the fit.

Returned summary: {'pass': 5, 'fail': 0, 'warn': 1, 'info': 0}
```

### Problematic Example

A nonlinear signal with non-constant variance to trigger multiple diagnostics.

```python
rng = np.random.default_rng(7)
x = np.linspace(-2.5, 2.5, 90)
X = pd.DataFrame({"x1": x})
noise = rng.normal(scale=0.20 + 0.20 * np.abs(x), size=x.size)
y = 1.0 + 1.8 * x + 0.9 * x**2 + noise

linear_fail = check_linear_regression(
    X=X,
    y=y,
    design_independent=True,
    return_dict=True,
)

print()
print("Returned summary:", linear_fail["summary"])
```

Expected console output:

```text
LINEAR REGRESSION ASSUMPTION CHECKS
[WARN] Linearity
[PASS] Independence
[FAIL] Normality of residuals
[PASS] Homoscedasticity
[PASS] Multicollinearity
[FAIL] Extreme influential points

Summary: 3 pass, 2 fail, 1 warn

Details:
- Linearity [WARN]
  Metric: Ramsey RESET p = < 0.001
  Threshold: Residuals should look like a random cloud; RESET p < 0.05 is a warning sign, not proof. Thresholds here are heuristics, not hard laws.
  Interpretation: There may be curvature or missing structure in the model. Use the residuals vs fitted plot as the primary diagnostic.
  Visual: Residuals vs fitted
  Possible mitigation:
    - Add polynomial terms if curvature is plausible.
    - Add interactions if the effect changes across predictors.
    - Transform predictors or the response if the scale is causing the pattern.
    - Use a more flexible model form if the relationship is clearly nonlinear.
- Normality of residuals [FAIL]
  Metric: Shapiro-Wilk p = < 0.001
  Threshold: p < 0.05 suggests possible non-normality. Thresholds here are heuristics, not hard laws.
  Interpretation: Residuals may depart from normality, especially in the tails.
  Visual: Q-Q plot of residuals
  Possible mitigation:
    - Check for outliers before transforming.
    - Transform the response with a log, square-root, or Box-Cox transform when appropriate.
    - Use robust methods if inference is sensitive to non-normal residuals.
- Extreme influential points [FAIL]
  Metric: Max Cook's D = 0.129; flagged points > 4/n: 10
  Threshold: Cook's D > 4/n is concerning; here 4/n = 0.044.
  Interpretation: At least one observation exceeds the common Cook's distance concern threshold.
  Visual: Residuals vs leverage
  Possible mitigation:
    - Verify that influential observations are valid.
    - Compare the model with and without flagged cases.
    - Use robust regression or re-specify the model if single points are driving the fit.

Returned summary: {'pass': 3, 'fail': 2, 'warn': 1, 'info': 0}
```

![Linear Regression Problematic Example plot](assets/linear_fail_1.png)

![Linear Regression Problematic Example plot](assets/linear_fail_2.png)

![Linear Regression Problematic Example plot](assets/linear_fail_3.png)

### Full Diagnostics

Request all detail and all plots for the cleaner linear example.

```python
rng = np.random.default_rng(27)
X = pd.DataFrame({
    "x1": rng.normal(size=35),
    "x2": rng.normal(size=35),
})
y = 1.0 + 1.8 * X["x1"] - 0.6 * X["x2"] + rng.normal(scale=0.35, size=35)

linear_full = check_linear_regression(
    X=X,
    y=y,
    design_independent=True,
    show_all=True,
    return_dict=True,
)

print()
print("Returned summary:", linear_full["summary"])
```

Expected console output:

```text
LINEAR REGRESSION ASSUMPTION CHECKS
[PASS] Linearity
[PASS] Independence
[PASS] Normality of residuals
[PASS] Homoscedasticity
[PASS] Multicollinearity
[WARN] Extreme influential points

Summary: 5 pass, 1 warn

Details:
- Linearity [PASS]
  Metric: Ramsey RESET p = 0.230
  Threshold: Residuals should look like a random cloud; RESET p < 0.05 is a warning sign, not proof. Thresholds here are heuristics, not hard laws.
  Interpretation: Residual diagnostics do not show strong evidence of nonlinearity.
  Visual: Residuals vs fitted
- Independence [PASS]
  Interpretation: You indicated that the study design supports independent errors.
- Normality of residuals [PASS]
  Metric: Shapiro-Wilk p = 0.431
  Threshold: p < 0.05 suggests possible non-normality. Thresholds here are heuristics, not hard laws.
  Interpretation: Residuals do not show strong evidence against normality.
  Visual: Q-Q plot of residuals
- Homoscedasticity [PASS]
  Metric: Breusch-Pagan p = 0.472
  Threshold: p < 0.05 suggests heteroscedasticity. Thresholds here are heuristics, not hard laws.
  Interpretation: Residual spread looks broadly compatible with constant variance.
  Visual: Residuals vs fitted
- Multicollinearity [PASS]
  Metric: Max VIF = 1.000 (x1)
  Threshold: VIF > 5 deserves caution and VIF > 10 is a serious concern.
  Interpretation: Predictor VIF values are in a commonly acceptable range.
- Extreme influential points [WARN]
  Metric: Max Cook's D = 0.097; flagged points > 4/n: 0
  Threshold: Cook's D > 4/n is concerning; here 4/n = 0.114.
  Interpretation: There may be a moderately influential point worth reviewing.
  Visual: Residuals vs leverage
  Possible mitigation:
    - Verify that influential observations are valid.
    - Compare the model with and without flagged cases.
    - Use robust regression or re-specify the model if single points are driving the fit.

Returned summary: {'pass': 5, 'fail': 0, 'warn': 1, 'info': 0}
```

![Linear Regression Full Diagnostics plot](assets/linear_show_all_1.png)

![Linear Regression Full Diagnostics plot](assets/linear_show_all_2.png)

![Linear Regression Full Diagnostics plot](assets/linear_show_all_3.png)

![Linear Regression Full Diagnostics plot](assets/linear_show_all_4.png)

## Logistic Regression

### Clean Example

A stable logistic model. The current influence heuristic still tends to emit a mild warning.

```python
rng = np.random.default_rng(44)
X = pd.DataFrame({
    "x1": rng.normal(size=45),
    "x2": rng.normal(size=45),
})
logits = -0.2 + 0.9 * X["x1"] - 0.5 * X["x2"]
p = 1 / (1 + np.exp(-logits))
y = rng.binomial(1, p)

logistic_clean = check_logistic_regression(
    X=X,
    y=y,
    design_independent=True,
    plots_on_fail=False,
    return_dict=True,
)

print()
print("Returned summary:", logistic_clean["summary"])
print("Model fit metric:", logistic_clean["diagnostics"][0]["metric"])
```

Expected console output:

```text
LOGISTIC REGRESSION ASSUMPTION CHECKS
[PASS] Linearity in the log-odds
[PASS] Independence
[PASS] Multicollinearity
[WARN] Extreme influential points
[PASS] Adequate sample / no separation

Summary: 4 pass, 1 warn

Details:
- Extreme influential points [WARN]
  Metric: Max Cook's D = 0.080; flagged points > 4/n: 0
  Threshold: Cook's D > 4/n is concerning; here 4/n = 0.089.
  Interpretation: There may be a moderately influential case worth checking.
  Possible mitigation:
    - Verify unusual cases before making changes.
    - Compare the fit with and without flagged cases.
    - Re-specify the model if influential cases reveal a structural issue.

Returned summary: {'pass': 4, 'fail': 0, 'warn': 1, 'info': 0}
Model fit metric: AUC = 0.620
```

### Problematic Example

A near-perfect separation setup that should clearly fail the stability check.

```python
x = np.linspace(-2.5, 2.5, 80)
X = pd.DataFrame({
    "x1": x,
    "x2": np.sin(x),
})
y = (x > 0.0).astype(int)

logistic_fail = check_logistic_regression(
    X=X,
    y=y,
    design_independent=True,
    return_dict=True,
)

print()
print("Returned summary:", logistic_fail["summary"])
```

Expected console output:

```text
LOGISTIC REGRESSION ASSUMPTION CHECKS
[INFO] Linearity in the log-odds
[PASS] Independence
[WARN] Multicollinearity
[FAIL] Extreme influential points
[FAIL] Adequate sample / no separation

Summary: 1 pass, 2 fail, 1 warn, 1 info

Details:
- Multicollinearity [WARN]
  Metric: Max VIF = 7.360 (x2)
  Threshold: VIF > 5 deserves caution and VIF > 10 is a serious concern.
  Interpretation: Predictor 'x2' shows noticeable collinearity.
  Possible mitigation:
    - Remove or combine redundant predictors.
    - Use regularization if stable prediction matters more than coefficient interpretation.
- Extreme influential points [FAIL]
  Metric: Max Cook's D = 1479330280139.682; flagged points > 4/n: 80
  Threshold: Cook's D > 4/n is concerning; here 4/n = 0.050.
  Interpretation: At least one case exceeds the common Cook's distance concern threshold.
  Possible mitigation:
    - Verify unusual cases before making changes.
    - Compare the fit with and without flagged cases.
    - Re-specify the model if influential cases reveal a structural issue.
- Adequate sample / no separation [FAIL]
  Metric: Converged = no; max |coef| = 647.674; max SE = 69789.733
  Threshold: Non-convergence or very large coefficients / SEs can indicate separation or sparse data.
  Interpretation: The fit shows strong signs of separation or sparse-data instability.
  Possible mitigation:
    - Collect more data if the current sample is sparse.
    - Combine rare levels or simplify sparse predictors when justified.
    - Use penalized logistic regression if separation is present.

Returned summary: {'pass': 1, 'fail': 2, 'warn': 1, 'info': 1}
```

### Full Diagnostics

Request the full logistic diagnostic view, including the ROC curve.

```python
rng = np.random.default_rng(44)
X = pd.DataFrame({
    "x1": rng.normal(size=45),
    "x2": rng.normal(size=45),
})
logits = -0.2 + 0.9 * X["x1"] - 0.5 * X["x2"]
p = 1 / (1 + np.exp(-logits))
y = rng.binomial(1, p)

logistic_full = check_logistic_regression(
    X=X,
    y=y,
    design_independent=True,
    show_all=True,
    return_dict=True,
)

print()
print("Returned summary:", logistic_full["summary"])
print("Model fit metric:", logistic_full["diagnostics"][0]["metric"])
```

Expected console output:

```text
LOGISTIC REGRESSION ASSUMPTION CHECKS
[PASS] Linearity in the log-odds
[PASS] Independence
[PASS] Multicollinearity
[WARN] Extreme influential points
[PASS] Adequate sample / no separation

Summary: 4 pass, 1 warn

Details:
- Linearity in the log-odds [PASS]
  Metric: Worst Box-Tidwell p = 0.564 (x2)
  Threshold: p < 0.05 suggests nonlinearity in the log-odds. Thresholds here are heuristics, not hard laws.
  Interpretation: Continuous predictors do not show strong evidence against linearity in the log-odds.
  Visual: Binned logit plot for x2
- Independence [PASS]
  Interpretation: You indicated that the study design supports independent observations.
- Multicollinearity [PASS]
  Metric: Max VIF = 1.051 (x1)
  Threshold: VIF > 5 deserves caution and VIF > 10 is a serious concern.
  Interpretation: Predictor VIF values are in a commonly acceptable range.
- Extreme influential points [WARN]
  Metric: Max Cook's D = 0.080; flagged points > 4/n: 0
  Threshold: Cook's D > 4/n is concerning; here 4/n = 0.089.
  Interpretation: There may be a moderately influential case worth checking.
  Possible mitigation:
    - Verify unusual cases before making changes.
    - Compare the fit with and without flagged cases.
    - Re-specify the model if influential cases reveal a structural issue.
- Adequate sample / no separation [PASS]
  Metric: Converged = yes; max |coef| = 0.631; max SE = 0.416
  Threshold: Non-convergence or very large coefficients / SEs can indicate separation or sparse data.
  Interpretation: The fit converged without obvious signs of severe separation.

Diagnostics:
- Model fit summary [INFO]
  Metric: AUC = 0.620
  Threshold: AUC around 0.50 indicates no discrimination; higher values indicate better separation.
  Interpretation: Discrimination is modest.
  Visual: ROC curve
  Possible mitigation:
    - Add more predictive structure with better features, interactions, or nonlinear terms.
    - Re-specify the model if discrimination is weak.

Returned summary: {'pass': 4, 'fail': 0, 'warn': 1, 'info': 0}
Model fit metric: AUC = 0.620
```

![Logistic Regression Full Diagnostics plot](assets/logistic_show_all_1.png)

![Logistic Regression Full Diagnostics plot](assets/logistic_show_all_2.png)
