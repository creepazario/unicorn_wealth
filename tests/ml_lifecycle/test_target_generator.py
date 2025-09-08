import pandas as pd
import numpy as np

from ml_lifecycle.target_generator import TargetGenerator


def _mk_ts(n: int, start: str = "2025-01-01 00:00:00+00:00", freq: str = "15min"):
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC")


def test_barrier_calculation():
    # Build OHLCV with constant True Range = 10 each bar so ATR (Wilder EWM) == 10
    n = 10
    ts = _mk_ts(n)
    # Start values
    lows = np.arange(100, 100 + n, dtype=float)
    highs = lows + 10.0  # high - low = 10
    closes = lows + 5.0  # mid price, prev_close changes by 1 => max(high-prev_close)=6

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": closes,  # not used but realistic
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 0.0,
        }
    )

    barriers = TargetGenerator._calculate_barriers(df)

    # Expected ATR
    atr = 10.0
    expected = pd.DataFrame(
        {
            "barrier_strong_bull": closes + atr * 2.5,
            "barrier_bull": closes + atr * 1.25,
            "barrier_bull_invalid": closes - atr * 0.75,
            "barrier_strong_bear": closes - atr * 2.5,
            "barrier_bear": closes - atr * 1.25,
            "barrier_bear_invalid": closes + atr * 0.75,
        }
    )

    # Compare numeric arrays
    for col in expected.columns:
        assert np.allclose(
            barriers[col].to_numpy(), expected[col].to_numpy(), atol=1e-8
        )


def _make_ohlcv_for_scenario(
    base_close: float,
    highs: list[float],
    lows: list[float],
):
    n = len(highs)
    ts = _mk_ts(n)
    # Close as midpoint for simplicity
    closes = (np.array(highs) + np.array(lows)) / 2.0
    opens = closes.copy()
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 0.0,
        }
    )


def _make_constant_barriers(
    df: pd.DataFrame, close0: float, atr: float
) -> pd.DataFrame:
    # Create a barriers_df aligned by timestamp with constant levels derived from close0
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "barrier_strong_bull": close0 + 2.5 * atr,
            "barrier_bull": close0 + 1.25 * atr,
            "barrier_bull_invalid": close0 - 0.75 * atr,
            "barrier_strong_bear": close0 - 2.5 * atr,
            "barrier_bear": close0 - 1.25 * atr,
            "barrier_bear_invalid": close0 + 0.75 * atr,
        }
    )


def test_target_labeling_scenarios():
    L = 32
    close0 = 100.0
    atr = 2.0

    # Scenario 1: Strong Bull (label 2)
    # Make price hit strong bull quickly without invalidation first.
    highs = [101.0] + [105.1] + [106.0] * (L - 2)  # reach >= 105 at j=2
    lows = [99.5] + [99.0] + [99.0] * (L - 2)  # stay above bull_invalid (98.5)
    ohlcv1 = _make_ohlcv_for_scenario(close0, highs, lows)
    barriers1 = _make_constant_barriers(ohlcv1, close0, atr)
    labels1 = TargetGenerator._generate_labels(ohlcv1, barriers1, look_forward_period=L)
    # Expect label 2 at the first row
    assert int(labels1.loc[0, "target_prediction"]) == 2

    # Scenario 2: Bear (label -1)
    # Price hits bear before strong_bear and invalidation.
    highs = [100.5] + [100.8] + [100.7] * (L - 2)  # below bear_invalid (101.5)
    lows = [99.0] + [97.4] + [97.0] * (L - 2)  # touch bear (97.5) at j=2
    ohlcv2 = _make_ohlcv_for_scenario(close0, highs, lows)
    barriers2 = _make_constant_barriers(ohlcv2, close0, atr)
    labels2 = TargetGenerator._generate_labels(ohlcv2, barriers2, look_forward_period=L)
    assert int(labels2.loc[0, "target_prediction"]) == -1

    # Scenario 3: Invalidation first (label 0) - hit bull_invalid before strong_bull
    highs = [100.2] + [100.3] + [100.4] * (L - 2)  # never reach strong_bull (105)
    lows = [99.0] + [98.4] + [98.7] * (L - 2)  # hit bull_invalid (98.5) at j=1
    ohlcv3 = _make_ohlcv_for_scenario(close0, highs, lows)
    barriers3 = _make_constant_barriers(ohlcv3, close0, atr)
    labels3 = TargetGenerator._generate_labels(ohlcv3, barriers3, look_forward_period=L)
    assert int(labels3.loc[0, "target_prediction"]) == 0

    # Scenario 4: No Hit (label 0) - price trades inside barriers
    highs = [101.0] * L  # below bull (102.5)
    lows = [99.5] * L  # above bear (97.5)
    ohlcv4 = _make_ohlcv_for_scenario(close0, highs, lows)
    barriers4 = _make_constant_barriers(ohlcv4, close0, atr)
    labels4 = TargetGenerator._generate_labels(ohlcv4, barriers4, look_forward_period=L)
    assert int(labels4.loc[0, "target_prediction"]) == 0
