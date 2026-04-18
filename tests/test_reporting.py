from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from assumpcheck.reporting import counts_as_text, print_report
from assumpcheck.types import AssumptionCheck, AssumptionReport


class ReportingTests(unittest.TestCase):
    def test_summary_counts_and_dict_shape(self) -> None:
        report = AssumptionReport(
            model_type="anova",
            checks=[
                AssumptionCheck("Normality", "PASS"),
                AssumptionCheck("Variance", "FAIL"),
                AssumptionCheck("Independence", "INFO"),
            ],
        )

        self.assertEqual(report.summary(), {"pass": 1, "fail": 1, "warn": 0, "info": 1})
        as_dict = report.to_dict()
        self.assertEqual(as_dict["model_type"], "anova")
        self.assertEqual(len(as_dict["checks"]), 3)

    def test_counts_as_text_omits_zeroes(self) -> None:
        self.assertEqual(counts_as_text({"pass": 2, "fail": 1, "warn": 0, "info": 1}), "2 pass, 1 fail, 1 info")

    def test_print_report_includes_details_for_failures(self) -> None:
        report = AssumptionReport(
            model_type="linear_regression",
            checks=[
                AssumptionCheck("Linearity", "PASS"),
                AssumptionCheck(
                    "Homoscedasticity",
                    "FAIL",
                    metric="Breusch-Pagan p = 0.010",
                    threshold="p < 0.05 suggests heteroscedasticity.",
                    interpretation="Residual spread may vary.",
                    mitigation=["Use robust standard errors."],
                ),
            ],
            title="LINEAR REGRESSION ASSUMPTION CHECKS",
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_report(report)

        output = buffer.getvalue()
        self.assertIn("[FAIL] Homoscedasticity", output)
        self.assertIn("Details:", output)
        self.assertIn("Use robust standard errors.", output)


if __name__ == "__main__":
    unittest.main()
