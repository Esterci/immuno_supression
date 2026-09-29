"""Verifica SHAP contra a previsão do Pipeline, inclusive fallback de TreeSHAP."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor

from methods.ml_models import model_specs
from utils.ml_reports import shap_report


class ShapPipelineTest(unittest.TestCase):
    def test_pipeline_explanations_and_tree_fallback(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(24, 3)), columns=['a', 'b', 'c'])
        y = 2 * X.a + X.b
        train, test = X.iloc[:18], X.iloc[18:]
        rows = pd.DataFrame({'id': range(6), 'time': range(6)}, index=test.index)
        for name, (model, _) in model_specs(quick=True).items():
            model.set_params(n_estimators=5)
            pipeline = Pipeline([
                ('scaler', MinMaxScaler().set_output(transform='pandas')),
                ('model', TransformedTargetRegressor(
                    regressor=model, transformer=MinMaxScaler())),
            ]).fit(train, y.iloc[:18])
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                meta = shap_report(pipeline, name, train, test, rows, output, 3, 4, 42, 1)
                self.assertEqual(meta['algorithm'], 'permutation_pipeline')
                samples = pd.read_parquet(output / 'shap_samples.parquet')
                values = pd.read_parquet(output / 'shap_values.parquet')
                np.testing.assert_allclose(values.sum(axis=1) + samples.base_value,
                                           pipeline.predict(test.loc[samples.index]), rtol=1e-4, atol=1e-5)
                self.assertEqual(meta['feature_scale'], 'prepared_features')
                self.assertEqual(meta['target_scale'], 'original_lym')
                json.dumps(meta, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
