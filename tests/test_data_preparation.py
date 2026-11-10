import unittest
import numpy as np
import pandas as pd
from data_preparation import (
    INTERPOLATION_METHODS,
    exclude_occ5,
    merge_db,
    ts_interpolation,
)
from methods.interpolation import METHODS


class PreparationTest(unittest.TestCase):
    def test_occ5_removed_before_matching_and_interpolation(self):
        lr = pd.DataFrame({
            "id": [1, 1, 1], "time": [0., 10., 100.], "occ": [1, 2, 5],
            "imc": [0]*3, "ar": [0]*3, "nta": [0]*3,
        })
        drug = pd.DataFrame({
            "id": [1, 1, 1], "time": [0., 11., 100.], "occ": [1, 5, 5],
            "amt": [0]*3, "dv": [2., 999., 999.], "mdv": [0]*3,
        })
        merged = merge_db(lr, drug, drug)
        self.assertEqual(merged.time.tolist(), [0., 10.])
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
        self.assertEqual(exclude_occ5(pd.DataFrame({"occ": ["1", "5"]})).occ.tolist(), ["1"])
        with self.assertRaisesRegex(ValueError, "occ"):
            exclude_occ5(pd.DataFrame({"time": [0]}))
