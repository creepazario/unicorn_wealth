from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.dataframe_registry import DataFrameRegistry


class ConsensusEngine:
    """Compute weighted consensus signals and trade parameters.

    This engine consumes model predictions, applies a viability filter,
    aggregates via model weights, and produces actionable directives
    for a PositionManager.
    """

    OUTCOME_LABELS: List[str] = [
        "OPEN_LONG",
        "OPEN_SHORT",
        "EXIT_LONG",
        "EXIT_SHORT",
        "HOLD",
    ]

    def __init__(self, settings_module, dataframe_registry: DataFrameRegistry) -> None:
        self.settings = settings_module
        self.dataframe_registry = dataframe_registry
        # Validate required settings exist to fail fast
        required_attrs = [
            "MODEL_WEIGHTS",
            "MINIMUM_MOVE_PCT",
            "MIN_CONSENSUS_PROBABILITY_THRESHOLD",
            "CLASS_TO_EXPECTED_MOVE_PCT",
            "ATR_STOP_LOSS_MULTIPLIER",
            "USE_FIXED_TAKE_PROFIT",
            "FIXED_TAKE_PROFIT_PCT",
        ]
        for a in required_attrs:
            if not hasattr(self.settings, a):
                raise AttributeError(f"Missing required setting: {a}")

    async def generate_directive(self, predictions: Dict) -> Dict:
        """Generate a strategic directive from raw model predictions.

        Args:
            predictions: A dict with structure:
                {
                  'token': 'BTC',
                  'entry_price': 50000.0,
                  'models': {
                      'model_a': {'probs': {'OPEN_LONG': 0.4, 'OPEN_SHORT': 0.1, 'EXIT_LONG': 0.1, 'EXIT_SHORT': 0.1, 'HOLD': 0.3}, 'top_class': 'OPEN_LONG'},
                      'model_b': {...}
                  }
                }

                If provided in class-index form, ensure labels match OUTCOME_LABELS.
        Returns:
            Directive dict, e.g., {'directive': 'OPEN_LONG', 'token': 'BTC', 'stop_loss': sl, 'take_profit': tp}
        """
        token: str = predictions.get("token", "BTC")
        entry_price: Optional[float] = predictions.get("entry_price")
        model_preds: Dict = predictions.get("models", {})

        # 1) Viability filter per model
        viable_model_probs: Dict[str, Dict[str, float]] = {}
        for model_name, payload in model_preds.items():
            probs: Dict[str, float] = payload.get("probs", {})
            # Determine top class and expected move
            top_class = max(probs.items(), key=lambda kv: kv[1])[0] if probs else "HOLD"
            expected_move = self._expected_move_pct(top_class)
            if expected_move < getattr(self.settings, "MINIMUM_MOVE_PCT"):
                # Treat as HOLD
                viable_model_probs[model_name] = {
                    label: 0.0 for label in self.OUTCOME_LABELS
                }
                viable_model_probs[model_name]["HOLD"] = 1.0
            else:
                # Normalize to labels
                viable_model_probs[model_name] = {
                    label: float(probs.get(label, 0.0)) for label in self.OUTCOME_LABELS
                }

        if not viable_model_probs:
            return {"directive": "HOLD", "token": token}

        # 2) Weighted consensus across models
        weighted_scores = {label: 0.0 for label in self.OUTCOME_LABELS}
        total_weight = 0.0
        for model_name, probs in viable_model_probs.items():
            weight = float(self.settings.MODEL_WEIGHTS.get(model_name, 1.0))
            total_weight += weight
            for label in self.OUTCOME_LABELS:
                weighted_scores[label] += weight * probs.get(label, 0.0)

        if total_weight > 0:
            consensus = {k: v / total_weight for k, v in weighted_scores.items()}
        else:
            consensus = weighted_scores

        # 3) Determine directive if above threshold
        top_label, top_prob = max(consensus.items(), key=lambda kv: kv[1])
        threshold = float(self.settings.MIN_CONSENSUS_PROBABILITY_THRESHOLD)
        if top_prob < threshold:
            return {"directive": "HOLD", "token": token}

        if top_label in ("OPEN_LONG", "OPEN_SHORT"):
            if entry_price is None:
                # Cannot open without entry price context
                return {"directive": "HOLD", "token": token}
            sl, tp = await self._calculate_trade_parameters(
                token=token, entry_price=float(entry_price), direction=top_label
            )
            return {
                "directive": top_label,
                "token": token,
                "stop_loss": sl,
                "take_profit": tp,
            }

        # For EXIT or HOLD decisions, return minimal info
        return {"directive": top_label, "token": token}

    async def _calculate_trade_parameters(
        self, token: str, entry_price: float, direction: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute stop-loss and take-profit per settings and ATR.

        Stop-loss uses latest 1h ATR multiplied by ATR_STOP_LOSS_MULTIPLIER.
        Take-profit is None unless USE_FIXED_TAKE_PROFIT is True, in which case
        we set a cost-aware target that covers slippage and taker fees.
        """
        atr_name = f"{token}_atr_1h_df"
        try:
            atr_df: pd.DataFrame = await self.dataframe_registry.get_df(atr_name)
        except KeyError:
            atr_df = pd.DataFrame()

        latest_atr_value = None
        if not atr_df.empty:
            # Accept common ATR column names
            for col in ("atr", "ATR", "value"):
                if col in atr_df.columns and not atr_df[col].dropna().empty:
                    latest_atr_value = float(atr_df[col].dropna().iloc[-1])
                    break
            # If value may be in a single-column DF
            if latest_atr_value is None and atr_df.shape[1] == 1:
                latest_atr_value = float(atr_df.iloc[-1, 0])

        stop_loss: Optional[float]
        if latest_atr_value is not None and latest_atr_value > 0:
            delta = latest_atr_value * float(self.settings.ATR_STOP_LOSS_MULTIPLIER)
            if direction == "OPEN_LONG":
                stop_loss = max(0.0, entry_price - delta)
            else:  # OPEN_SHORT
                stop_loss = entry_price + delta
        else:
            # Fallback: 2% away if ATR unavailable
            pct = 0.02
            if direction == "OPEN_LONG":
                stop_loss = max(0.0, entry_price * (1 - pct))
            else:
                stop_loss = entry_price * (1 + pct)

        # TODO: Implement future SL logic based on SMC swing points.
        # latest_swing_low = await self.dataframe_registry.get_df(f"{token}_smc_swing_1h_df")["low"].iloc[-1]
        # stop_loss = latest_swing_low * 0.99

        take_profit: Optional[float]
        if bool(getattr(self.settings, "USE_FIXED_TAKE_PROFIT")):
            raw_tp_pct = float(getattr(self.settings, "FIXED_TAKE_PROFIT_PCT")) / 100.0
            # Cost aware: account for round-trip slippage and fees
            slippage_pct = (
                float(
                    self._get_setting_nested(
                        "BACKTESTING_SETTINGS", "SLIPPAGE_PCT", default=0.0
                    )
                )
                / 100.0
            )
            taker_fee_pct = (
                float(
                    self._get_setting_nested(
                        "BACKTESTING_SETTINGS", "TAKER_FEE", default=0.0
                    )
                )
                / 100.0
            )
            round_trip_cost = 2.0 * (slippage_pct + taker_fee_pct)
            effective_pct = max(0.0, raw_tp_pct + round_trip_cost)
            if direction == "OPEN_LONG":
                take_profit = entry_price * (1.0 + effective_pct)
            else:
                take_profit = entry_price * (1.0 - effective_pct)
        else:
            take_profit = None

        return stop_loss, take_profit

    def _expected_move_pct(self, outcome_label: str) -> float:
        mapping = getattr(self.settings, "CLASS_TO_EXPECTED_MOVE_PCT", {})
        return float(mapping.get(outcome_label, 0.0))

    def _get_setting_nested(self, container_attr: str, key: str, default=None):
        container = getattr(self.settings, container_attr, {})
        if isinstance(container, dict):
            return container.get(key, default)
        return getattr(self.settings, key, default)
