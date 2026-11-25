import pandas as pd

from runners.run_interpolation_study import classify_series, parse_args


def test_default_series_are_discovered_and_semantic_exclusions_are_audited():
    data = pd.DataFrame(
        {
            "id": [1, 1],
            "time": [0, 1],
            "age": [20, 20],
            "ii": [12, 0],
            "m155": [1.0, 2.0],
            "gfc": [3, 4],
            "tad": [0, 12],
            "amt_tac": [1.0, 0.0],
        }
    )

    catalog = classify_series(data).set_index("series")

    assert catalog.loc["m155", "interpolation_eligible"]
    assert not catalog.loc["amt_tac", "interpolation_eligible"]
    assert "age" not in catalog.index
    assert "ii" not in catalog.index
    assert "gfc" not in catalog.index
    assert "tad" not in catalog.index
    assert "observed_exams_per_patient_sd" in catalog.columns


def test_interpolation_study_defaults_to_occ_one_and_automatic_series():
    args = parse_args([])

    assert args.occ == [1]
    assert args.series is None
