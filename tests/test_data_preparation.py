import unittest
import numpy as np
import pandas as pd
from data_preparation import exclude_occ5, merge_db, ts_interpolation


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
        interpolated = ts_interpolation(merged.rename(columns={"dv": "dv_tac"}))
        self.assertEqual(interpolated.dv_tac.tolist(), [2., 2.])
        self.assertEqual(interpolated.dv_map.tolist(), [2., 2.])
        self.assertEqual(len(drug), 3)

    def test_string_occ_and_missing_column(self):
        self.assertEqual(exclude_occ5(pd.DataFrame({"occ": ["1", "5"]})).occ.tolist(), ["1"])
        with self.assertRaisesRegex(ValueError, "occ"):
            exclude_occ5(pd.DataFrame({"time": [0]}))
