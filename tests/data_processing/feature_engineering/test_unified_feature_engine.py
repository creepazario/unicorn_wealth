from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from features.feature_engine import UnifiedFeatureEngine


def test_run_15m_pipeline_returns_final_dataframe(mocker):
    # Arrange: settings dict with needed paths
    settings: Dict[str, Any] = {
        "15m": {
            "rsi_15m": {"window": 14},
            "atr_15m": {"window": 14},
        }
    }

    # Prepare identifiable dummy data
    index = pd.RangeIndex(start=0, stop=5)
    ohlcv = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0, 1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 20, 30, 40, 50],
        },
        index=index,
    )

    # Mock the three feature functions to return unique outputs
    rsi_series = pd.Series([10, 20, 30, 40, 50], index=index, name="rsi_15m")
    atr_df = pd.DataFrame(
        {"average_true_range": [0.1, 0.2, 0.3, 0.4, 0.5]}, index=index
    )
    atr_norm_series = pd.Series([1, 2, 3, 4, 5], index=index, name="atr_normalized_15m")

    mocker.patch("features.feature_engine.rsi_15m", return_value=rsi_series)
    mocker.patch("features.feature_engine.atr_15m", return_value=atr_df)
    mocker.patch(
        "features.feature_engine.atr_normalized_15m",
        return_value=atr_norm_series,
    )

    engine = UnifiedFeatureEngine(registry=None, sql_engine=None)  # registry unused now

    # Act
    df = engine.run_15m_pipeline(token="BTC", ohlcv_15m_df=ohlcv, settings=settings)

    # Assert: final df is single-row and contains the mocked feature columns
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    expected_cols = set(["rsi_15m", "average_true_range", "atr_normalized_15m"])
    assert expected_cols.issubset(df.columns)

    # Verify the mocked functions were called
    import features.feature_engine as fe_mod

    fe_mod.rsi_15m.assert_called_once()
    fe_mod.atr_15m.assert_called_once()
    fe_mod.atr_normalized_15m.assert_called_once()
