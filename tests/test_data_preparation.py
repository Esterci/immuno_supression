import unittest
import numpy as np
import pandas as pd
from data_preparation import (
    INTERPOLATION_METHODS,
    data_cleaning,
    merge_db,
    parse_args,
    select_occ5,
    translating_flags,
    ts_interpolation,
)
from methods.interpolation import METHODS


class PreparationTest(unittest.TestCase):
    def test_interpolation_is_optional_and_disabled_by_default(self):
        self.assertEqual(parse_args([]).interpolation, 0)
        self.assertEqual(parse_args(["--interpolation", "1"]).interpolation, 1)

    def test_flags_remove_globally_excluded_variables_from_processed_data(self):
        frame = pd.DataFrame(
            {
                "id": [1],
                "ii": [12],
                "cmt": [2],
                "tad": [4],
                "gfc": [3],
                "amt": [1.0],
                "dv": [2.0],
                "mdv": [0],
                "dv_map": [3.0],
                "mdv_map": [0],
                "lym": [4.0],
                "dylc": [0],
                "dyl": [5.0],
                "cisq": [6.0],
                "gf": [7.0],
                "ifc": [8.0],
                "m155": [9.0],
            }
        )

        result = translating_flags(frame)

        self.assertFalse({"ii", "cmt", "tad", "gfc"} & set(result.columns))

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

    def test_time_series_cleaning_ignores_missing_non_temporal_data(self):
        frame = pd.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "cisq": [7.0, 7.0, np.nan, np.nan],
                "lym": [1.0, 2.0, 3.0, 4.0],
                "dv_tac": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = data_cleaning(frame, var_mode="time-series")

        self.assertEqual(result["id"].unique().tolist(), [1, 2])

    def test_cleaning_does_not_treat_interpolation_flags_as_measurements(self):
        frame = pd.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "cisq": [1.0, 1.0, 1.0, 1.0],
                "lym": [1.0, 2.0, np.nan, np.nan],
                "int_fg_lym": [0, 1, 0, 0],
            }
        )

        result = data_cleaning(frame, var_mode="time-series")

        self.assertEqual(result["id"].unique().tolist(), [1])

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
        self.assertTrue(np.isnan(interpolated.dv_tac.iloc[1]))
        self.assertTrue(np.isnan(interpolated.dv_map.iloc[1]))
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
            self.assertEqual(result.loc[2, f"int_fg_{column}"], 1)
            self.assertEqual(result[f"int_fg_{column}"].sum(), 1)

    def test_edges_and_empty_series_stay_missing(self):
        frame = pd.DataFrame({"id": [1] * 5, "time": range(5)})
        for column in INTERPOLATION_METHODS:
            frame[column] = [np.nan, 2.0, 3.0, 4.0, np.nan]
        frame["gf"] = np.nan

        result = ts_interpolation(frame)

        for column in set(INTERPOLATION_METHODS) - {"gf"}:
            self.assertTrue(np.isnan(result.loc[0, column]))
            self.assertEqual(result.loc[1:3, column].tolist(), [2.0, 3.0, 4.0])
            self.assertTrue(np.isnan(result.loc[4, column]))
            self.assertEqual(result[f"int_fg_{column}"].sum(), 0)
        self.assertTrue(result["gf"].isna().all())
        self.assertEqual(result["int_fg_gf"].sum(), 0)

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
