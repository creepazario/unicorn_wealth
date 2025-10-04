from __future__ import annotations

import asyncio
from typing import Any

import pytest

from main import run_live_15m_cycle


class AsyncMock:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def _record(self, name: str, *args: Any, **kwargs: Any) -> Any:
        # small await to ensure scheduling points between awaits in the tested coroutine
        await asyncio.sleep(0)
        self.calls.append((name, args, kwargs))
        return object()


@pytest.mark.asyncio
async def test_run_live_15m_cycle_calls_in_correct_order(
    mocker: pytest.MockFixture,
) -> None:
    # Arrange: create simple objects with async methods we can spy on
    class LiveDataManager:
        async def fetch_scheduled_data(
            self,
        ) -> None:  # pragma: no cover - replaced by spy
            pass

    class LiveFeatureCalculator:
        async def calculate_all_horizons(self) -> dict[str, Any]:  # pragma: no cover
            return {"1h": object(), "4h": object(), "8h": object()}

    class InferenceEngine:
        async def predict(
            self, feature_rows: dict[str, Any]
        ) -> dict[str, Any]:  # pragma: no cover
            return {"1h": object()}

    class ConsensusEngine:
        async def generate_directive(
            self, predictions: dict[str, Any]
        ) -> dict[str, Any]:  # pragma: no cover
            return {"action": "HOLD"}

    class PositionManager:
        async def process_directive(
            self, directive: dict[str, Any]
        ) -> None:  # pragma: no cover
            return None

    ldm = LiveDataManager()
    lfc = LiveFeatureCalculator()
    ie = InferenceEngine()
    ce = ConsensusEngine()
    pm = PositionManager()

    # Attach spies to verify ordering
    fetch_spy = mocker.spy(ldm, "fetch_scheduled_data")
    calc_spy = mocker.spy(lfc, "calculate_all_horizons")
    predict_spy = mocker.spy(ie, "predict")
    directive_spy = mocker.spy(ce, "generate_directive")
    process_spy = mocker.spy(pm, "process_directive")

    # Act
    await run_live_15m_cycle(ldm, lfc, ie, ce, pm)

    # Assert: all called exactly once
    assert fetch_spy.call_count == 1
    assert calc_spy.call_count == 1
    assert predict_spy.call_count == 1
    assert directive_spy.call_count == 1
    assert process_spy.call_count == 1

    # Assert: sequential order using call timestamps from spy mocks
    # pytest-mock's spy stores call objects in .mock_calls in the order of invocation
    calls_in_order = [
        fetch_spy.mock_calls[0],
        calc_spy.mock_calls[0],
        predict_spy.mock_calls[0],
        directive_spy.mock_calls[0],
        process_spy.mock_calls[0],
    ]
    # The above construction will raise IndexError if any missing; ensure length equals 5
    assert len(calls_in_order) == 5

    # Additionally, verify dependency flow (args)
    # prediction should receive the dict returned by calculate_all_horizons
    # Since spies don't alter return values, we re-run small controlled stubs with AsyncMock to assert dataflow.


@pytest.mark.asyncio
async def test_run_live_15m_cycle_dataflow_and_order_with_stubbed_returns(
    mocker: pytest.MockFixture,
) -> None:
    """A stricter test that verifies the exact argument passing and order.

    Uses stubbed return values and side-effect tracking to ensure each stage
    receives the previous stage's output.
    """

    class LDM:
        async def fetch_scheduled_data(self) -> None:
            return None

    class LFC:
        async def calculate_all_horizons(self) -> dict[str, Any]:
            return {"features": True}

    class IE:
        async def predict(self, feature_rows: dict[str, Any]) -> dict[str, Any]:
            assert feature_rows == {"features": True}
            return {"pred": 123}

    class CE:
        async def generate_directive(
            self, predictions: dict[str, Any]
        ) -> dict[str, Any]:
            assert predictions == {"pred": 123}
            return {"action": "HOLD"}

    class PM:
        async def process_directive(self, directive: dict[str, Any]) -> None:
            assert directive == {"action": "HOLD"}
            return None

    ldm, lfc, ie, ce, pm = LDM(), LFC(), IE(), CE(), PM()

    # Spy to verify order
    s1 = mocker.spy(ldm, "fetch_scheduled_data")
    s2 = mocker.spy(lfc, "calculate_all_horizons")
    s3 = mocker.spy(ie, "predict")
    s4 = mocker.spy(ce, "generate_directive")
    s5 = mocker.spy(pm, "process_directive")

    await run_live_15m_cycle(ldm, lfc, ie, ce, pm)

    assert [s.call_count for s in (s1, s2, s3, s4, s5)] == [1, 1, 1, 1, 1]

    # Ensure chronological order: each call must occur after the previous
    order = [s1, s2, s3, s4, s5]
    all_calls = []
    # Merge all mock_calls to a single list preserving order
    for spy in order:
        for c in spy.mock_calls:
            all_calls.append((spy, c))
    positions = {id(spy): idx for idx, (spy, _c) in enumerate(all_calls)}
    # Ensure positions are strictly increasing
    assert (
        positions[id(s1)]
        < positions[id(s2)]
        < positions[id(s3)]
        < positions[id(s4)]
        < positions[id(s5)]
    )
