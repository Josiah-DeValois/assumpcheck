from __future__ import annotations

import importlib.util
import unittest

from assumpcheck import check_anova, check_linear_regression, check_logistic_regression


def _has_core_stack() -> bool:
    required = ["numpy", "pandas", "scipy", "statsmodels"]
    return all(importlib.util.find_spec(name) is not None for name in required)


@unittest.skipUnless(_has_core_stack(), "Scientific Python dependencies are not installed.")
class EndToEndTests(unittest.TestCase):
    def test_anova_returns_structured_summary(self) -> None:
        import numpy as np

        rng = np.random.default_rng(7)
        groups = np.repeat(["A", "B", "C"], 30)
        y = np.concatenate(
            [
                rng.normal(0, 1.0, 30),
                rng.normal(0.1, 1.0, 30),
                rng.normal(-0.1, 1.0, 30),
            ]
        )

        result = check_anova(y=y, groups=groups, return_dict=True, plots_on_fail=False)
        self.assertEqual(result["model_type"], "anova")
        self.assertEqual(len(result["checks"]), 4)

    def test_linear_regression_includes_expected_checks(self) -> None:
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(11)
        X = pd.DataFrame({"x1": rng.normal(size=120), "x2": rng.normal(size=120)})
        y = 1.0 + 2.0 * X["x1"] - 0.5 * X["x2"] + rng.normal(scale=0.6, size=120)

        result = check_linear_regression(
            X=X,
            y=y,
            design_independent=True,
            return_dict=True,
            plots_on_fail=False,
        )
        assumptions = {row["assumption"] for row in result["checks"]}
        self.assertIn("Linearity", assumptions)
        self.assertIn("Homoscedasticity", assumptions)
        self.assertIn("Extreme influential points", assumptions)

    def test_logistic_regression_returns_fit_diagnostic(self) -> None:
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(23)
        X = pd.DataFrame({"x1": rng.normal(size=180), "x2": rng.normal(size=180)})
        logits = -0.2 + 1.1 * X["x1"] - 0.7 * X["x2"]
        probabilities = 1 / (1 + np.exp(-logits))
        y = rng.binomial(1, probabilities)

        result = check_logistic_regression(
            X=X,
            y=y,
            design_independent=True,
            return_dict=True,
            plots_on_fail=False,
        )
        self.assertEqual(result["model_type"], "logistic_regression")
        self.assertTrue(result["diagnostics"])
        self.assertEqual(result["diagnostics"][0]["assumption"], "Model fit summary")


if __name__ == "__main__":
    unittest.main()
