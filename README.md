# assumpcheck

`assumpcheck` is a small Python package for checking core assumptions for:

- ANOVA
- Linear regression
- Logistic regression

The package is designed to stay simple:

- concise terminal output by default
- plots only when they matter
- mitigation suggestions when something fails
- optional structured output for programmatic use

## Status

This repository currently contains an MVP that is:

- statsmodels-first
- focused on clear terminal output
- designed for local analysis workflows

## Installation

```bash
pip install -e .
```

Core dependencies:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `matplotlib`

## Public API

```python
from assumpcheck import (
    check_anova,
    check_linear_regression,
    check_logistic_regression,
)
```

### ANOVA

```python
report = check_anova(y=y, groups=groups)
```

### Linear regression

```python
report = check_linear_regression(model=fitted_ols_model)
```

### Logistic regression

```python
report = check_logistic_regression(model=fitted_logit_model)
```

## Example output

```text
ANOVA ASSUMPTION CHECKS
[INFO] Independence
[PASS] Normality of residuals
[PASS] Equal variance across groups
[FAIL] Extreme outliers

Summary: 2 pass, 1 fail, 1 info

Details:
- Extreme outliers [FAIL]
  Metric: Max |standardized residual| = 3.420; flagged points > 3: 1
  Threshold: Values above 2 deserve review and values above 3 are concerning.
  Interpretation: At least one observation has a standardized residual above the common concern threshold.
  Possible mitigation:
    - Verify data entry for flagged cases.
    - Check whether the observation is legitimate but unusual.
    - Consider a transformation, robust method, or nonparametric alternative if outliers remain influential.
```

## Options

All three public functions support these core options:

- `alpha=0.05`
- `show_all=False`
- `plots_on_fail=True`
- `verbose=False`
- `return_dict=False`
- `design_independent=None`

Additional model-specific option:

- `check_linear_regression(..., ordered=False)`

### Output behavior

Default behavior:

- prints a concise summary
- shows plots for failed or warning-level checks if `plots_on_fail=True`
- returns an `AssumptionReport` object

Optional behavior:

- `verbose=True` prints detail for every check
- `show_all=True` prints all details and shows all available plots
- `return_dict=True` returns a serializable dictionary

### Independence handling

By default, independence is treated as a design question:

- `design_independent=None` gives an `INFO` result
- `design_independent=True` gives a `PASS` unless an ordered linear model also shows autocorrelation warnings
- `design_independent=False` gives a `FAIL`

## Current checks

### ANOVA

- Independence
- Normality of residuals
- Equal variance across groups
- Extreme outliers

### Linear regression

- Linearity
- Independence
- Normality of residuals
- Homoscedasticity
- Multicollinearity
- Extreme influential points

### Logistic regression

- Linearity in the log-odds
- Independence
- Multicollinearity
- Extreme influential points
- Adequate sample / no separation
- Model fit summary via ROC / AUC

## Notes on the MVP

- The package prioritizes `statsmodels` models first.
- Thresholds are intentionally presented as heuristics.
- Logistic ROC / AUC is treated as a fit diagnostic, not a strict assumption.
- Some diagnostics need access to original data or design metadata to be fully informative.

## Examples

- Walkthrough: `examples/workflow.md`
- Script: `examples/basic_usage.py`
- Notebook: `examples/assumpcheck_examples.ipynb`

To rebuild the notebook, Markdown walkthrough, and example plot assets:

```bash
python examples/build_workflow_artifacts.py
```

## Tests

Run the lightweight test suite with:

```bash
python3 -m unittest discover -s tests
```

If the scientific Python dependencies are installed, the end-to-end diagnostic tests will run as well.
