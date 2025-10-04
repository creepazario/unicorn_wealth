from __future__ import annotations

from pathlib import Path
from typing import Dict

import json
import numpy as np
import pandas as pd
import pytest

from ml_lifecycle.data_preparer import DataPreparer


@pytest.fixture()
def synthetic_parquet_setup(tmp_path: Path) -> Dict[str, Path | pd.DataFrame]:
    # Create directories
    output_dir = tmp_path / "output" / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_dir = tmp_path / "specifications"
    spec_dir.mkdir(parents=True, exist_ok=True)

    # Build synthetic dataset with 100 rows
    n = 100
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            # numeric feature
            "price_feature": np.linspace(0.0, 1.0, n),
            # categorical feature we will mark as categorical in the spec
            "token_id": ["BTC", "ETH", "UNI", "MATIC"] * (n // 4) + ["BTC"] * (n % 4),
            # target for 1h horizon
            "target_1h": np.random.randint(-2, 3, size=n),
        }
    )

    # Save parquet for horizon 1h
    parquet_path = output_dir / "ml_training_dataset_1h.parquet"
    df.to_parquet(parquet_path)

    # Minimal feature spec JSON indicating token_id is categorical
    spec = [
        {"operation": "token_id", "is_categorical_feature": True},
        {"operation": "price_feature", "is_categorical_feature": False},
    ]
    spec_path = spec_dir / "Unicorn_Wealth_Feature_Set.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "spec_path": spec_path,
        "df": df,
    }


def test_data_preparation_logic(
    synthetic_parquet_setup: Dict[str, Path | pd.DataFrame],
) -> None:
    output_dir = synthetic_parquet_setup["output_dir"]  # type: ignore[assignment]
    spec_path = synthetic_parquet_setup["spec_path"]  # type: ignore[assignment]
    original_df: pd.DataFrame = synthetic_parquet_setup["df"]  # type: ignore[assignment]

    preparer = DataPreparer(
        feature_spec_path=spec_path,
        output_dir=output_dir,
        horizons=["1h"],
    )

    results = preparer.prepare_data_for_training()
    assert "1h" in results

    prepared = results["1h"]

    # Assert split sizes (n=100, train=70%, val=10%, test=20%)
    assert len(prepared.X_train) == 70
    assert len(prepared.X_val) == 10
    assert len(prepared.X_test) == 20

    # Assert chronology: map indices back to original timestamps
    # Note: DataPreparer sorts and resets index, so indices align with 0..n-1
    train_ts = original_df.loc[prepared.X_train.index, "timestamp"]
    val_ts = original_df.loc[prepared.X_val.index, "timestamp"]
    test_ts = original_df.loc[prepared.X_test.index, "timestamp"]

    assert train_ts.max() < val_ts.min()
    assert val_ts.max() < test_ts.min()

    # Assert dtypes: token_id should be categorical
    assert str(prepared.X_train["token_id"].dtype) == "category"

    # Assert sample weights shape and monotonicity (more recent -> larger)
    sw = prepared.sample_weights
    assert len(sw) == len(prepared.X_train)
    # training data is chronological; last index corresponds to most recent
    assert sw.iloc[-1] > sw.iloc[0]
