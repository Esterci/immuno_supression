import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone
from methods.ml_models import prepare_features, model_specs, make_search
from runners.run_ml_experiment import parse_args, prepare_measurement_features
from utils.ml_reports import regression_metrics


class MLMethodsTest(unittest.TestCase):
    def test_default_dataset_is_clean_processed(self):
        default = parse_args([])
        temporal = parse_args(["--var-mode", "time-series"])
        custom = parse_args(["--data", "custom.parquet"])

        self.assertEqual(default.data.name, "clean_processed.parquet")
        self.assertEqual(temporal.data.name, "clean_processed.parquet")
        self.assertEqual(custom.data, Path("custom.parquet"))
        self.assertEqual(default.models, list(model_specs()))

    def test_measurement_time_series_mode_excludes_non_temporal_variables(self):
        raw = self.data()
        raw["cisq"] = np.nan

        X, y, rows = prepare_measurement_features(raw, var_mode="time-series")

        self.assertNotIn("age", X.columns)
        self.assertNotIn("cisq", X.columns)
        self.assertIn("time", X.columns)
        self.assertEqual(rows["id"].nunique(), raw["id"].nunique())
        self.assertFalse(X.isna().any().any())
        self.assertEqual(len(X), len(y))

    def test_global_exclusions_apply_in_both_variable_modes(self):
        raw = self.data()
        raw["ii"] = 12
        raw["cmt"] = 2
        raw["tad"] = 4
        raw["gfc"] = 3

        for var_mode in ("all", "time-series"):
            X, _, _ = prepare_measurement_features(raw, var_mode=var_mode)
            for excluded in ("ii", "cmt", "tad", "gfc"):
                self.assertFalse(any(excluded in column for column in X.columns))

            legacy_X, _, _ = prepare_features(
                raw, 1, "none", var_mode=var_mode
            )
            for excluded in ("ii", "cmt", "tad", "gfc"):
                self.assertFalse(
                    any(excluded in column for column in legacy_X.columns)
                )

    def test_interpolation_flags_are_metadata_not_model_features(self):
        raw = self.data()
        raw["int_fg_dv_tac"] = 0
        raw["int_fg_lym"] = raw["time"].eq(3).astype("int8")

        X, _, rows = prepare_measurement_features(raw)
        legacy_X, _, legacy_rows = prepare_features(raw, 1, "none")

        self.assertFalse(any("int_fg_" in column for column in X.columns))
        self.assertFalse(any("int_fg_" in column for column in legacy_X.columns))
        self.assertFalse(legacy_rows["time"].eq(3).any())
        self.assertIn("interpolated", rows.columns)
        self.assertFalse(rows["interpolated"].any())
        self.assertFalse(rows["time"].eq(3).any())
        self.assertEqual(len(rows), 2 * raw["id"].nunique())

        # O LYM interpolado em t=3 não vira alvo, mas é o valor anterior usado
        # para construir os atributos da medida real seguinte, em t=4.
        time_four = rows["time"].eq(4)
        expected_previous_lym = (
            raw.loc[raw["time"].eq(3)].sort_values("id")["lym"].to_numpy()
        )
        np.testing.assert_array_equal(
            X.loc[time_four, "val_lym"], expected_previous_lym
        )

    def test_measurement_format_uses_history_without_target_leakage(self):
        raw = self.data().sort_values(["id", "time"])
        X, y, rows = prepare_measurement_features(raw, use_lym=False)

        self.assertEqual(len(X), 3 * raw["id"].nunique())
        self.assertIn("time", X.columns)
        self.assertIn("age", X.columns)
        self.assertIn("initial_dv_tac", X.columns)
        self.assertIn("val_dv_tac", X.columns)
        self.assertIn("d_time_dv_tac", X.columns)
        self.assertIn("tend_dv_tac", X.columns)
        self.assertFalse(any("lym" in column for column in X.columns))
        self.assertNotIn("id", X.columns)
        predictable = raw["time"] >= 2
        pd.testing.assert_series_equal(
            y, raw.loc[predictable, "lym"].reset_index(drop=True), check_names=True
        )
        pd.testing.assert_frame_equal(
            rows,
            raw.loc[predictable, ["id", "time"]].reset_index(drop=True),
        )

        patient = rows.id == 0
        np.testing.assert_array_equal(X.loc[patient, "initial_dv_tac"], 0)
        np.testing.assert_array_equal(
            X.loc[patient, "val_dv_tac"], [1, 2, 3]
        )
        np.testing.assert_array_equal(rows.loc[patient, "time"], [2, 3, 4])
        self.assertFalse(X.isna().any().any())

    def test_measurement_format_optionally_uses_only_previous_lym(self):
        raw = self.data().sort_values(["id", "time"])

        with_lym, target, rows = prepare_measurement_features(raw, use_lym=True)
        without_lym, _, _ = prepare_measurement_features(raw, use_lym=False)

        self.assertTrue(
            {"initial_lym", "d_time_lym", "val_lym", "tend_lym"}.issubset(
                with_lym.columns
            )
        )
        self.assertFalse(any(column.endswith("_lym") for column in without_lym))
        patient = rows["id"].eq(0)
        np.testing.assert_array_equal(with_lym.loc[patient, "val_lym"], [1, 2, 3])
        np.testing.assert_array_equal(target.loc[patient], [2, 3, 4])

        changed = raw.copy()
        changed.loc[(changed["id"] == 0) & (changed["time"] == 4), "lym"] = 999
        changed_X, _, _ = prepare_measurement_features(changed, use_lym=True)
        pd.testing.assert_frame_equal(with_lym, changed_X)

    def test_all_pipeline_normalizes_measurement_features(self):
        X, y, rows = prepare_measurement_features(self.data())
        train = rows.id < 6
        model, params = model_specs(quick=True)["random_forest"]
        search = make_search(model, params, "grid", 2, 1, 42, 1, "all")
        search.fit(X[train], y[train], groups=rows.loc[train, "id"])

        transformed = search.best_estimator_[:-1].transform(X[train])
        self.assertFalse(transformed.isna().any().any())
        self.assertTrue((transformed.min() >= 0).all())
        self.assertTrue((transformed.max() <= 1).all())

    def test_patients_with_only_one_predictable_measurement_are_kept(self):
        raw = self.data()
        raw = raw.loc[~((raw["id"] == 0) & (raw["time"] > 2))]

        X, _, rows = prepare_measurement_features(raw)

        self.assertIn(0, rows["id"].unique())
        self.assertEqual((rows["id"] == 0).sum(), 1)
        self.assertEqual(len(X), 1 + 3 * 7)

    def test_normalization_modes_preserve_metadata_and_target(self):
        raw = self.data()
        original = raw.copy(deep=True)
        baseline, target, rows = prepare_features(raw, 1, "none")
        for mode in ("all", "time-series", "none"):
            X, y, meta = prepare_features(raw, 1, mode)
            pd.testing.assert_frame_equal(meta, rows)
            pd.testing.assert_series_equal(y, target)
            self.assertIsInstance(X, pd.DataFrame)
            if mode == "time-series":
                np.testing.assert_allclose(X.amt_tac_t1, (meta.time - 1) / 4)
            else:
                pd.testing.assert_frame_equal(X, baseline)
        pd.testing.assert_frame_equal(raw, original)
        with self.assertRaisesRegex(ValueError, "all, time-series ou none"):
            prepare_features(raw, 1, "patient")

    def test_variable_modes(self):
        raw = self.data()
        all_features, _, _ = prepare_features(raw, 2, "none", var_mode="all")
        temporal, _, _ = prepare_features(raw, 2, "none", var_mode="time-series")
        expected = {
            f"{column}_t{lag}"
            for column in ("amt_tac", "dv_tac", "amt_map", "dv_map", "time", "lym")
            for lag in range(3)
        }
        self.assertEqual(set(temporal.columns), expected | {"time"})
        self.assertIn("age", all_features.columns)
        self.assertNotIn("age", temporal.columns)
        with self.assertRaisesRegex(ValueError, "all ou time-series"):
            prepare_features(raw, 2, "none", var_mode="invalid")

    def test_time_series_keeps_missing_clinical_data_but_drops_missing_target(self):
        raw = self.data()
        raw.loc[(raw.id == 0) & (raw.time == 2), "age"] = np.nan
        raw.loc[(raw.id == 0) & (raw.time == 4), "lym"] = np.nan

        X, y, rows = prepare_features(raw, 1, "none", var_mode="time-series")

        self.assertTrue(((rows.id == 0) & (rows.time == 2)).any())
        self.assertFalse(((rows.id == 0) & (rows.time == 4)).any())
        self.assertFalse(y.isna().any())
        self.assertTrue(np.isfinite(X.to_numpy(dtype=float)).all())

    def test_current_temporal_inputs_excluded_in_every_mode(self):
        current = {"amt_tac", "dv_tac", "amt_map", "dv_map"}
        for normalization in ("none", "all", "time-series"):
            for var_mode in ("all", "time-series"):
                for lags in (0, 1, 3):
                    with self.subTest(
                        normalization=normalization, var_mode=var_mode, lags=lags
                    ):
                        X, _, rows = prepare_features(
                            self.data(), lags, normalization, var_mode
                        )
                        self.assertFalse(current & set(X.columns))
                        self.assertTrue((rows.time > 0).all())
                        self.assertIn("dv_tac_t0", X.columns)

    def test_all_scaler_fits_only_training_rows(self):
        X, y, rows = prepare_features(self.data(), 1, "all")
        train = rows.id < 6
        for kind in ("grid", "random"):
            model, params = model_specs(quick=True)["random_forest"]
            search = make_search(model, params, kind, 2, 1, 42, 1, normalization="all")
            for a, b in search.cv.split(X[train], y[train], rows[train].id):
                fitted = clone(search.estimator).fit(X[train].iloc[a], y[train].iloc[a])
                np.testing.assert_allclose(
                    fitted.named_steps["scaler"].data_max_, X[train].iloc[a].max()
                )
                target_scaler = fitted.named_steps["model"].transformer_
                np.testing.assert_allclose(
                    target_scaler.data_min_, [y[train].iloc[a].min()]
                )
                np.testing.assert_allclose(
                    target_scaler.data_max_, [y[train].iloc[a].max()]
                )
            search.fit(X[train], y[train], groups=rows[train].id)
            scaler = search.best_estimator_.named_steps["scaler"]
            np.testing.assert_allclose(scaler.data_max_, X[train].max())
            target_scaler = search.best_estimator_.named_steps["model"].transformer_
            np.testing.assert_allclose(target_scaler.data_min_, [y[train].min()])
            np.testing.assert_allclose(target_scaler.data_max_, [y[train].max()])
            self.assertGreater(scaler.transform(X[~train]).amt_tac_t1.max(), 1)
            self.assertTrue(np.isfinite(search.predict(X[~train])).all())

    def data(self):
        rows = []
        for patient in range(8):
            for t in range(5):
                rows.append(
                    dict(
                        id=patient,
                        time=t,
                        lym=t + patient,
                        age=20 + patient,
                        amt_tac=100 * patient + t,
                        dv_tac=t,
                        amt_map=t,
                        dv_map=t,
                    )
                )
        return pd.DataFrame(rows).sample(frac=1, random_state=42)

    def test_lags_do_not_cross_patients(self):
        X, y, meta = prepare_features(self.data(), 2, "none")
        self.assertEqual(len(X), 24)
        np.testing.assert_array_equal(X.amt_tac_t1, 100 * meta.id + meta.time - 1)
        np.testing.assert_array_equal(X.amt_tac_t0, 100 * meta.id)
        self.assertFalse({"id", "lym"} & set(X.columns))

    def test_zero_lags_and_empty_result(self):
        X, _, _ = prepare_features(self.data(), 0, "none")
        self.assertEqual(len(X), 32)
        self.assertIn("dv_tac_t0", X.columns)
        with self.assertRaisesRegex(ValueError, "Nenhuma linha"):
            prepare_features(self.data(), 5)

    def test_searches_and_group_isolation(self):
        X, y, rows = prepare_features(self.data(), 1, "none")
        train, test = next(
            GroupShuffleSplit(test_size=0.25, random_state=42).split(X, y, rows.id)
        )
        self.assertFalse(set(rows.iloc[train].id) & set(rows.iloc[test].id))
        for kind in ("grid", "random"):
            model, params = model_specs(quick=True)["random_forest"]
            search = make_search(model, params, kind, 2, 1, 42, 1)
            self.assertEqual(search.scoring, "r2")
            groups = rows.iloc[train].id
            for a, b in search.cv.split(X.iloc[train], y.iloc[train], groups):
                self.assertFalse(set(groups.iloc[a]) & set(groups.iloc[b]))
            search.fit(X.iloc[train], y.iloc[train], groups=groups)
            self.assertEqual(
                len(search.cv_results_["params"]), 2 if kind == "grid" else 1
            )

    def test_irregular_time_and_lym_history(self):
        raw = self.data()
        raw["time"] = raw.time.map({0: 0, 1: 2, 2: 7, 3: 20, 4: 100})
        X, y, rows = prepare_features(raw, 2, "none", "time-series")
        patient = rows.id == 0
        np.testing.assert_array_equal(X.loc[patient, "time"], [7, 20, 100])
        np.testing.assert_array_equal(X.loc[patient, "time_t1"], [2, 7, 20])
        np.testing.assert_array_equal(X.loc[patient, "lym_t1"], [1, 2, 3])
        np.testing.assert_array_equal(X.loc[patient, "lym_t2"], [0, 1, 2])
        np.testing.assert_array_equal(y[patient], [2, 3, 4])
        changed = raw.copy()
        changed.loc[(changed.id == 0) & (changed.time == 100), "lym"] = 999
        changed_X, _, _ = prepare_features(changed, 2, "none", "time-series")
        pd.testing.assert_frame_equal(X, changed_X)

    def test_required_history_scaling_in_every_normalization_mode(self):
        for mode in ("all", "none", "time-series"):
            X, y, rows = prepare_features(self.data(), 1, mode)
            train = rows.id < 6
            model, params = model_specs(quick=True)["random_forest"]
            search = make_search(model, params, "grid", 2, 1, 42, 1, mode)
            for a, _ in search.cv.split(X[train], y[train], rows[train].id):
                fold_X = X[train].iloc[a]
                fitted = clone(search.estimator).fit(fold_X, y[train].iloc[a])
                transformed = fitted.named_steps["scaler"].transform(fold_X)
                for col in ("time", "time_t1", "lym_t0", "lym_t1"):
                    self.assertAlmostEqual(transformed[col].min(), 0)
                    self.assertAlmostEqual(transformed[col].max(), 1)
                scaled = fitted.named_steps["scaler"].transform(X[~train])
                self.assertGreater(scaled.lym_t1.max(), 1)

    def test_undefined_metrics_are_json_safe(self):
        result = regression_metrics([0, 0], np.array([1, 1]))
        self.assertIsNone(result["r2"])
        self.assertIsNone(result["mape_percent_nonzero"])
        self.assertEqual(result["mape_excluded_zeros"], 2)

    def test_metric_standard_deviation_is_calculated_between_patients(self):
        result = regression_metrics(
            [1, 3, 2, 6],
            [0, 2, 0, 2],
            groups=[10, 10, 20, 20],
        )

        # Patient RMSE values are 1 and sqrt(10); MAE values are 1 and 3.
        self.assertAlmostEqual(
            result["rmse_patient_sd"], np.std([1, np.sqrt(10)], ddof=1)
        )
        self.assertAlmostEqual(result["mae_patient_sd"], np.std([1, 3], ddof=1))
        self.assertAlmostEqual(
            result["mape_percent_nonzero_patient_sd"],
            np.std([100 * (1 + 1 / 3) / 2, 100 * (1 + 4 / 6) / 2], ddof=1),
        )
        self.assertEqual(result["metric_patient_count"], 2)


if __name__ == "__main__":
    unittest.main()
