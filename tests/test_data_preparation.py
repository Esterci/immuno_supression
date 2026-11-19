import unittest
import numpy as np
import pandas as pd
from data_preparation import (
    INTERPOLATION_METHODS,
    data_cleaning,
    merge_db,
    select_occ5,
    ts_interpolation,
)
from methods.interpolation import METHODS


class PreparationTest(unittest.TestCase):
    def test_cleaning_propagates_cisq_and_removes_patients_without_it(self):
        frame = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "time": [0, 1, 2, 0, 1],
                "cisq": [np.nan, 7.0, np.nan, np.nan, np.nan],
                "measurement": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = data_cleaning(frame)

        self.assertEqual(result["id"].unique().tolist(), [1])
        self.assertEqual(result["cisq"].tolist(), [7.0, 7.0, 7.0])
        self.assertFalse(result["cisq"].isna().any())

    def test_occ5_removed_before_matching_and_interpolation(self):
        lr = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "time": [0.0, 10.0, 100.0],
                "occ": [1, 1, 5],
                "imc": [0] * 3,
                "ar": [0] * 3,
                "nta": [0] * 3,
            }
        )
        drug = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "time": [0.0, 11.0, 100.0],
                "occ": [1, 5, 5],
                "amt": [0] * 3,
                "dv": [2.0, 999.0, 999.0],
                "mdv": [0] * 3,
            }
        )
        merged = merge_db(
            select_occ5(lr), select_occ5(drug), select_occ5(drug), occ=1
        )
        self.assertEqual(merged.time.tolist(), [0.0, 10.0])
        self.assertTrue(np.isnan(merged.dv.iloc[1]))
        prepared = merged.rename(columns={"dv": "dv_tac"})
        for column in ("gf", "lym", "m155", "wgt"):
            prepared[column] = 1.0
        interpolated = ts_interpolation(prepared)
        self.assertEqual(interpolated.dv_tac.iloc[0], 2.0)
        self.assertEqual(interpolated.dv_map.iloc[0], 2.0)
        self.assertEqual(interpolated.dv_tac.iloc[1], 2.0)
        self.assertEqual(interpolated.dv_map.iloc[1], 2.0)
        self.assertEqual(len(drug), 3)

    def test_selected_methods_fill_internal_gaps_using_time(self):
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        frame = pd.DataFrame({"id": 1, "time": times})
        for column in INTERPOLATION_METHODS:
            frame[column] = [1.0, 2.0, np.nan, 10.0, 17.0]

        result = ts_interpolation(frame)

        keep = np.array([True, True, False, True, True])
        for column, method_name in INTERPOLATION_METHODS.items():
            expected = METHODS[method_name](
                times[keep], frame.loc[keep, column].to_numpy(), 2.0
            )
            self.assertAlmostEqual(result.loc[2, column], expected)

    def test_edges_use_nearest_observation_and_empty_series_stays_missing(self):
        frame = pd.DataFrame({"id": [1] * 5, "time": range(5)})
        for column in INTERPOLATION_METHODS:
            frame[column] = [np.nan, 2.0, 3.0, 4.0, np.nan]
        frame["gf"] = np.nan

        result = ts_interpolation(frame)

        for column in set(INTERPOLATION_METHODS) - {"gf"}:
            self.assertEqual(result[column].tolist(), [2.0, 2.0, 3.0, 4.0, 4.0])
        self.assertTrue(result["gf"].isna().all())

    def test_string_occ_and_missing_column(self):
        self.assertEqual(
            select_occ5(pd.DataFrame({"occ": ["1", "5"]})).occ.tolist(), ["1"]
        )
        with self.assertRaisesRegex(ValueError, "occ"):
            select_occ5(pd.DataFrame({"time": [0]}))

    def test_multiple_occasions_are_selected(self):
        frame = pd.DataFrame({"occ": [1, 2, 3, 5], "value": range(4)})

        selected = select_occ5(frame, [1, 3])

        self.assertEqual(selected["occ"].tolist(), [1, 3])
