from types import SimpleNamespace

import pandas as pd
import pytest

from execution.consensus_engine import ConsensusEngine


class MockRegistry:
    def __init__(self, atr_value: float | None = None):
        self._dfs = {}
        if atr_value is not None:
            # Create simple ATR df with a single 'atr' column
            self._dfs["BTC_atr_1h_df"] = pd.DataFrame({"atr": [atr_value]})

    async def get_df(self, name: str) -> pd.DataFrame:
        if name not in self._dfs:
            raise KeyError(name)
        return self._dfs[name]


def make_settings(
    *,
    use_fixed_tp: bool,
    fixed_tp_pct: float = 3.0,
    atr_mult: float = 2.5,
    min_move_pct: float = 0.5,
    min_consensus: float = 0.55,
    model_weights: dict | None = None,
    backtesting_settings: dict | None = None,
):
    if model_weights is None:
        model_weights = {"model_a": 1.0}
    if backtesting_settings is None:
        backtesting_settings = {"SLIPPAGE_PCT": 0.05, "TAKER_FEE": 0.04}
    settings = SimpleNamespace(
        ATR_STOP_LOSS_MULTIPLIER=atr_mult,
        USE_FIXED_TAKE_PROFIT=use_fixed_tp,
        FIXED_TAKE_PROFIT_PCT=fixed_tp_pct,
        MINIMUM_MOVE_PCT=min_move_pct,
        MIN_CONSENSUS_PROBABILITY_THRESHOLD=min_consensus,
        MODEL_WEIGHTS=model_weights,
        BACKTESTING_SETTINGS=backtesting_settings,
        CLASS_TO_EXPECTED_MOVE_PCT={
            "OPEN_LONG": 1.0,
            "OPEN_SHORT": 1.0,
            "EXIT_LONG": 0.0,
            "EXIT_SHORT": 0.0,
            "HOLD": 0.0,
        },
    )
    return settings


@pytest.mark.asyncio
async def test_atr_stop_loss_calculation():
    # Given
    entry_price = 50000.0
    atr_value = 1000.0
    settings = make_settings(use_fixed_tp=False, atr_mult=2.5)
    registry = MockRegistry(atr_value=atr_value)
    engine = ConsensusEngine(settings, registry)

    # Prediction with clear OPEN_LONG above threshold
    predictions = {
        "token": "BTC",
        "entry_price": entry_price,
        "models": {
            "model_a": {
                "probs": {
                    "OPEN_LONG": 0.9,
                    "OPEN_SHORT": 0.02,
                    "EXIT_LONG": 0.02,
                    "EXIT_SHORT": 0.02,
                    "HOLD": 0.04,
                }
            }
        },
    }

    # When
    directive = await engine.generate_directive(predictions)

    # Then
    assert directive["directive"] == "OPEN_LONG"
    expected_sl = entry_price - (atr_value * settings.ATR_STOP_LOSS_MULTIPLIER)
    assert pytest.approx(directive["stop_loss"], rel=1e-9) == expected_sl


@pytest.mark.asyncio
async def test_signal_based_take_profit():
    # Given USE_FIXED_TAKE_PROFIT = False
    entry_price = 50000.0
    atr_value = 1000.0
    settings = make_settings(use_fixed_tp=False)
    registry = MockRegistry(atr_value=atr_value)
    engine = ConsensusEngine(settings, registry)

    predictions = {
        "token": "BTC",
        "entry_price": entry_price,
        "models": {
            "model_a": {
                "probs": {
                    "OPEN_LONG": 0.8,
                    "OPEN_SHORT": 0.05,
                    "EXIT_LONG": 0.05,
                    "EXIT_SHORT": 0.05,
                    "HOLD": 0.05,
                }
            }
        },
    }

    # When
    directive = await engine.generate_directive(predictions)

    # Then
    assert directive["directive"] == "OPEN_LONG"
    assert directive.get("take_profit") is None


@pytest.mark.asyncio
async def test_fixed_take_profit_calculation():
    # Given USE_FIXED_TAKE_PROFIT = True with costs
    entry_price = 50000.0
    atr_value = 1000.0
    slippage = 0.10  # percent per fill
    fee = 0.05  # percent per fill
    settings = make_settings(
        use_fixed_tp=True,
        fixed_tp_pct=3.0,
        backtesting_settings={"SLIPPAGE_PCT": slippage, "TAKER_FEE": fee},
    )
    registry = MockRegistry(atr_value=atr_value)
    engine = ConsensusEngine(settings, registry)

    predictions = {
        "token": "BTC",
        "entry_price": entry_price,
        "models": {
            "model_a": {
                "probs": {
                    "OPEN_LONG": 0.7,
                    "OPEN_SHORT": 0.1,
                    "EXIT_LONG": 0.05,
                    "EXIT_SHORT": 0.05,
                    "HOLD": 0.1,
                }
            }
        },
    }

    # When
    directive = await engine.generate_directive(predictions)

    # Then
    assert directive["directive"] == "OPEN_LONG"
    # Effective pct = base + round-trip costs
    raw = settings.FIXED_TAKE_PROFIT_PCT / 100.0
    round_trip = 2.0 * (
        (slippage / 100.0) + (fee / 100.0)
    )  # round-trip costs added to base target
    effective = raw + round_trip
    expected_tp = entry_price * (1.0 + effective)
    assert pytest.approx(directive["take_profit"], rel=1e-12) == expected_tp
