import unittest

import numpy as np
import pandas as pd

from methods.interpolation import calculate_metrics, run_patient_loocv


class InterpolationValidationTest(unittest.TestCase):
    def test_linear_series_is_recovered_and_endpoints_are_excluded(self):
        frame = pd.DataFrame(
            {
                "id": [1] * 5,
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "test": [1.0, 3.0, 5.0, 7.0, 9.0],
            }
        )
        predictions, eligibility = run_patient_loocv(frame, ["test"])
        self.assertEqual(eligibility.loc[0, "eligible_exams_per_method"], 3)
        self.assertEqual(set(predictions.time), {1.0, 2.0, 3.0})
        linear = predictions[predictions.method.eq("linear")]
        np.testing.assert_allclose(linear.predicted, linear.observed)
        np.testing.assert_allclose(linear.predicted_original, linear.observed_original)

    def test_minmax_does_not_fit_the_held_out_exam(self):
        frame = pd.DataFrame(
            {
                "id": [1] * 5 + [2] * 5,
                "time": [0.0, 1.0, 2.0, 3.0, 4.0] * 2,
                "test": [0.0, 1.0, 100.0, 3.0, 4.0]
                + [1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
            }
        )
        predictions, _ = run_patient_loocv(frame, ["test"])
        held_maximum = predictions.loc[
            predictions.id.eq(1) & predictions.time.eq(2.0)
        ]
        self.assertTrue(held_maximum.scale_max.eq(4.0).all())
        self.assertTrue(
            held_maximum.normalization.eq("patient-train-minmax").all()
        )

    def test_macro_metrics_give_each_patient_equal_weight(self):
        rows = []
        for patient_id, errors in ((1, [0.0] * 10), (2, [10.0])):
            for error in errors:
                rows.append(
                    {
                        "series": "x",
                        "method": "m",
                        "id": patient_id,
                        "error": error,
                        "absolute_error": abs(error),
                        "squared_error": error**2,
                    }
                )
        _, macro = calculate_metrics(pd.DataFrame(rows))
        result = macro.iloc[0]
        self.assertAlmostEqual(result.mae, 5.0)
        self.assertAlmostEqual(result.bias, 5.0)
        self.assertAlmostEqual(result.rmse, np.sqrt(50.0))
        self.assertNotAlmostEqual(result.mae, result.mae_micro)

    def test_requires_five_observations_for_common_method_set(self):
        frame = pd.DataFrame({"id": [1] * 4, "time": range(4), "x": range(4)})
        predictions, eligibility = run_patient_loocv(frame, ["x"])
        self.assertTrue(predictions.empty)
        self.assertEqual(eligibility.loc[0, "eligible_patients"], 0)

    def test_m155_structural_zeros_are_not_observed_exams(self):
        frame = pd.DataFrame(
            {
                "id": [1] * 7,
                "time": range(7),
                "m155": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
            }
        )
        predictions, eligibility = run_patient_loocv(frame, ["m155"])
        self.assertEqual(eligibility.loc[0, "observed_exams"], 5)
        self.assertEqual(eligibility.loc[0, "eligible_exams_per_method"], 3)
        self.assertNotIn(0.0, predictions.observed_original.to_list())


if __name__ == "__main__":
    unittest.main()

