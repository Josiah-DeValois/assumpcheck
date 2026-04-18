from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from assumpcheck import check_anova, check_linear_regression, check_logistic_regression


def anova_example() -> None:
    rng = np.random.default_rng(42)
    groups = np.repeat(["A", "B", "C"], 40)
    y = np.concatenate(
        [
            rng.normal(10, 1.0, size=40),
            rng.normal(11, 1.1, size=40),
            rng.normal(12, 1.0, size=40),
        ]
    )
    check_anova(y=y, groups=groups)


def linear_regression_example() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=200),
            "x2": rng.normal(size=200),
        }
    )
    y = 2.0 + 1.5 * X["x1"] - 0.8 * X["x2"] + rng.normal(scale=0.8, size=200)
    model = sm.OLS(y, sm.add_constant(X)).fit()
    check_linear_regression(model=model, design_independent=True)


def logistic_regression_example() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=250),
            "x2": rng.normal(size=250),
        }
    )
    linear_term = -0.4 + 1.2 * X["x1"] - 0.7 * X["x2"]
    probabilities = 1 / (1 + np.exp(-linear_term))
    y = rng.binomial(1, probabilities)
    model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    check_logistic_regression(model=model, design_independent=True, verbose=True)


if __name__ == "__main__":
    anova_example()
    linear_regression_example()
    logistic_regression_example()
