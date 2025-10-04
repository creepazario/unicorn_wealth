from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import BACKTESTING_SETTINGS


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""

    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_pnl": float(self.total_pnl),
            "sharpe_ratio": float(self.sharpe_ratio),
            "max_drawdown": float(self.max_drawdown),
            "num_trades": int(self.num_trades),
        }


class ModelValidator:
    """Cadence-based backtester and champion–challenger validator.

    This class simulates trades on 15-minute bars based on three model predictions
    (1h, 4h, 8h) and a placeholder directive function mimicking a consensus engine.

    Parameters
    ----------
    test_features : pd.DataFrame
        Feature matrix aligned to 15m bars. Index or a column should align to
        test_prices. The backtester will pass the current-row features to
        each model to obtain predictions.
    test_prices : pd.DataFrame
        OHLCV dataframe with at least columns: ["open", "high", "low", "close"]
        on a 15-minute cadence. Index must be aligned with test_features.
    challenger_run_id : str
        MLflow run ID containing challenger models as artifacts.
    model_artifact_map : Optional[Dict[str, str]]
        Optional mapping for models' artifact locations. Keys should be
        {"1h", "4h", "8h"}. Defaults to
        {"1h": "models/1h", "4h": "models/4h", "8h": "models/8h"}.
    registry_model_names : Optional[Dict[str, str]]
        Mapping for champion registered model names in MLflow Model Registry.
        Defaults to {"1h": "unicorn_wealth_1h", "4h": "unicorn_wealth_4h", "8h": "unicorn_wealth_8h"}.
    """

    def __init__(
        self,
        test_features: pd.DataFrame,
        test_prices: pd.DataFrame,
        challenger_run_id: str,
        model_artifact_map: Optional[Dict[str, str]] = None,
        registry_model_names: Optional[Dict[str, str]] = None,
    ) -> None:
        self.test_features = test_features.copy()
        self.test_prices = test_prices.copy()
        self.challenger_run_id = challenger_run_id
        self.model_artifact_map = model_artifact_map or {
            "1h": "models/1h",
            "4h": "models/4h",
            "8h": "models/8h",
        }
        self.registry_model_names = registry_model_names or {
            "1h": "unicorn_wealth_1h",
            "4h": "unicorn_wealth_4h",
            "8h": "unicorn_wealth_8h",
        }

    # ------------------------------- Public API -------------------------------
    def validate_models(self) -> bool:
        """Run backtests for challenger and champion, compare, and decide promotion.

        Returns
        -------
        bool
            True if challenger Sharpe >= (1 + threshold) * champion Sharpe, else False.
        """

        challenger_models = self._load_challenger_models(self.challenger_run_id)
        champion_models = self._load_champion_models()

        challenger_metrics = self._run_backtest(
            challenger_models, self.test_features, self.test_prices
        )
        champion_metrics = (
            self._run_backtest(champion_models, self.test_features, self.test_prices)
            if champion_models
            else BacktestMetrics(
                total_pnl=0.0, sharpe_ratio=0.0, max_drawdown=0.0, num_trades=0
            )
        )

        # Log metrics to the challenger run (parent)
        self._log_metrics_to_mlflow(
            run_id=self.challenger_run_id,
            challenger=challenger_metrics.to_dict(),
            champion=champion_metrics.to_dict(),
        )

        # Promotion decision
        threshold_pct = float(
            BACKTESTING_SETTINGS.get("CHAMPION_PROMOTION_THRESHOLD_PCT", 10.0)
        )
        factor = 1.0 + threshold_pct / 100.0
        promote = (
            challenger_metrics.sharpe_ratio >= factor * champion_metrics.sharpe_ratio
        )
        return bool(promote)

    # ---------------------------- Backtest Mechanics ---------------------------
    def _run_backtest(
        self,
        models: Dict[str, object],
        test_features: pd.DataFrame,
        test_prices: pd.DataFrame,
    ) -> BacktestMetrics:
        """Run a simple 15m-cadence backtest using three horizon models.

        Notes
        -----
        - Avoids look-ahead: decisions at T, entry at T+1 open.
        - Exits when SL/TP hit using future high/low from (T+1 ..).
        - Costs applied: slippage and taker fees from config BACKTESTING_SETTINGS.
        """

        # Basic validations
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(set(test_prices.columns)):
            raise ValueError(f"test_prices must contain columns {required_cols}")
        if len(test_features) != len(test_prices):
            # Align on index if sizes differ
            test_features, test_prices = test_features.align(
                test_prices, join="inner", axis=0
            )

        # Generate predictions per horizon
        preds = self._predict_all(models, test_features)
        # Derive directives row-wise without look-ahead
        directives = preds.apply(self._get_strategic_directive, axis=1)

        # Build a DataFrame for entries: entry occurs on next bar open
        prices = test_prices.copy()
        prices = prices.astype(
            {"open": float, "high": float, "low": float, "close": float}
        )
        next_open = prices["open"].shift(-1)

        slippage_pct = float(BACKTESTING_SETTINGS.get("SLIPPAGE_PCT", 0.05)) / 100.0
        taker_fee_pct = float(BACKTESTING_SETTINGS.get("TAKER_FEE", 0.04)) / 100.0

        trade_entries: List[Tuple[pd.Timestamp, float, float, float]] = []
        trade_exits: List[Tuple[pd.Timestamp, float]] = []

        # We'll loop trades (sparser than bars). This remains efficient because directives
        # often produce fewer trades than bars; bar-wise computations (preds, directives,
        # entry price shift) are vectorized above.
        in_position = False
        entry_price = 0.0
        entry_time: Optional[pd.Timestamp] = None
        sl_level = 0.0
        tp_level = 0.0

        index = prices.index
        for i in range(len(prices) - 1):  # up to second last due to shift(-1) entry
            if not in_position:
                directive, sl_mult, tp_mult = directives.iloc[i]
                if directive == "OPEN_LONG":
                    # Enter next bar
                    in_position = True
                    entry_time = index[i + 1]
                    # Apply entry costs via adverse slippage; modeled as price disadvantage
                    raw_entry = next_open.iloc[i]
                    if pd.isna(raw_entry):
                        # Can't enter on the very last bar
                        in_position = False
                        entry_time = None
                        continue
                    entry_price = float(
                        raw_entry * (1.0 + slippage_pct + taker_fee_pct)
                    )
                    # Set absolute SL/TP based on multiples of entry
                    sl_level = float(entry_price * (1.0 - abs(sl_mult)))
                    tp_level = float(entry_price * (1.0 + abs(tp_mult)))
                    trade_entries.append((entry_time, entry_price, sl_level, tp_level))
            else:
                # Manage the open position: check if SL/TP hit within this bar using H/L
                high = float(prices["high"].iloc[i])
                low = float(prices["low"].iloc[i])
                exit_now = False
                exit_price = 0.0

                # For a long: SL hit if low <= sl_level; TP hit if high >= tp_level
                sl_hit = low <= sl_level
                tp_hit = high >= tp_level

                if sl_hit and tp_hit:
                    # If both hit in same bar, assume worst-case for long: SL first
                    exit_now = True
                    exit_price = sl_level
                elif sl_hit:
                    exit_now = True
                    exit_price = sl_level
                elif tp_hit:
                    exit_now = True
                    exit_price = tp_level

                if exit_now:
                    # Apply exit costs against trader
                    exit_time = index[i]
                    filled = float(exit_price * (1.0 - slippage_pct - taker_fee_pct))
                    trade_exits.append((exit_time, filled))
                    in_position = False
                    entry_price = 0.0
                    sl_level = 0.0
                    tp_level = 0.0
                    entry_time = None

        # Close any open trade at the final close price
        if in_position and entry_time is not None:
            close_fill = float(
                prices["close"].iloc[-1] * (1.0 - slippage_pct - taker_fee_pct)
            )
            trade_exits.append((prices.index[-1], close_fill))

        # Compute PnL per trade
        pnls: List[float] = []
        for (et, ep, _sl, _tp), (xt, xp) in zip(trade_entries, trade_exits):
            pnl = xp - ep
            # Costs already integrated into effective prices via adverse fills
            pnls.append(float(pnl))

        total_pnl = float(np.nansum(pnls))

        # Equity curve: assume unit notional per trade (1 contract/coin)
        equity = np.cumsum(pnls) if pnls else np.array([0.0])
        # Create pseudo 15m returns series from trade-level PnL distributed at exits
        # For Sharpe, allocate returns at exit times; zeros elsewhere
        returns = pd.Series(0.0, index=prices.index, dtype=float)
        for (xt, xp), (et, ep, _a, _b) in zip(trade_exits, trade_entries):
            trade_ret = (xp - ep) / ep if ep else 0.0
            returns.loc[xt] = trade_ret

        sharpe = self._sharpe_ratio(returns)
        mdd = self._max_drawdown(pd.Series(equity))

        return BacktestMetrics(
            total_pnl=total_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            num_trades=len(pnls),
        )

    # -------------------------- Strategy Placeholder --------------------------
    @staticmethod
    def _get_strategic_directive(row: pd.Series) -> Tuple[str, float, float]:
        """Placeholder for ConsensusEngine.

        Returns a tuple: (directive, sl_multiple, tp_multiple)
        - directive: "OPEN_LONG" to open a long next bar, or "FLAT" to do nothing.
        - sl_multiple: stop-loss as fraction of entry price (e.g., 0.005 => 0.5%).
        - tp_multiple: take-profit as fraction of entry price (e.g., 0.01 => 1%).

        Simple rule:
        - If the mean of the three horizon predictions > 0, open long with SL=0.5% and TP=1%.
        - Otherwise, FLAT.
        """

        # Collect any prediction columns starting with 'pred_'
        pred_vals: List[float] = []
        for col, val in row.items():
            if isinstance(col, str) and col.startswith("pred_") and pd.notna(val):
                try:
                    pred_vals.append(float(val))
                except Exception:
                    continue
        if not pred_vals:
            return ("FLAT", 0.0, 0.0)
        if float(np.nanmean(pred_vals)) > 0.0:
            return ("OPEN_LONG", 0.005, 0.01)
        return ("FLAT", 0.0, 0.0)

    # ------------------------------ Model Loading ------------------------------
    def _load_challenger_models(self, run_id: str) -> Dict[str, object]:
        """Load challenger models from an MLflow run's artifacts.

        This attempts to load models logged under child runs of the given parent
        training run (named "{horizon}_model"), default artifact_path="model".
        Falls back to attempting to load artifacts directly under the parent run
        using the provided relative paths.
        """
        models: Dict[str, object] = {}
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            parent = client.get_run(run_id)
            exp_id = parent.info.experiment_id
            # Find child runs
            runs = client.search_runs(
                [exp_id], filter_string=f"tags.mlflow.parentRunId = '{run_id}'"
            )
            horizon_to_child: Dict[str, str] = {}
            for r in runs:
                name = r.data.tags.get("mlflow.runName") or ""
                if name.endswith("_model"):
                    h = name[:-6]
                    horizon_to_child[h] = r.info.run_id

            for horizon in self.model_artifact_map.keys():
                child_id = horizon_to_child.get(horizon)
                if child_id:
                    uri = f"runs:/{child_id}/model"
                else:
                    # Fallback to parent-relative path
                    rel_path = self.model_artifact_map[horizon]
                    uri = f"runs:/{run_id}/{rel_path}"
                try:
                    models[horizon] = mlflow.pyfunc.load_model(uri)
                except Exception:
                    continue
        except Exception:
            # MLflow not available or search failed; best-effort fallback to parent paths
            try:
                import mlflow  # type: ignore

                for horizon, rel_path in self.model_artifact_map.items():
                    uri = f"runs:/{run_id}/{rel_path}"
                    try:
                        models[horizon] = mlflow.pyfunc.load_model(uri)
                    except Exception:
                        continue
            except Exception:
                pass
        return models

    def _load_champion_models(self) -> Dict[str, object]:
        """Load current champion models from MLflow Model Registry using alias 'production',
        with a fallback to deprecated stage lookup if alias is unavailable.
        """
        models: Dict[str, object] = {}
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            for horizon, name in self.registry_model_names.items():
                mv = None
                # Prefer alias-based resolution
                try:
                    mv = client.get_model_version_by_alias(
                        name=name, alias="production"
                    )
                except Exception:
                    mv = None
                # Fallback to stage-based resolution if alias not set
                if mv is None:
                    try:
                        versions = client.get_latest_versions(name, stages=["Production"])  # type: ignore[arg-type]
                        if versions:
                            mv = versions[0]
                    except Exception:
                        mv = None
                if mv is None:
                    continue
                try:
                    models[horizon] = mlflow.pyfunc.load_model(mv.source)
                except Exception:
                    continue
        except Exception:
            pass
        return models

    def _predict_all(
        self, models: Dict[str, object], features: pd.DataFrame
    ) -> pd.DataFrame:
        """Run predictions for available horizons and return a DataFrame of preds.

        The expected columns are pred_1h, pred_4h, pred_8h when available.
        Missing horizons will yield NaNs.
        """
        out = pd.DataFrame(index=features.index)
        # Create prediction columns dynamically based on available models
        for horizon, model in models.items():
            col = f"pred_{horizon}"
            try:
                preds = model.predict(features)
                if (
                    isinstance(preds, (pd.DataFrame, np.ndarray))
                    and getattr(preds, "ndim", 1) > 1
                    and getattr(preds, "shape", (0, 0))[1] == 1
                ):
                    preds = np.ravel(preds)
                out[col] = pd.Series(preds, index=features.index)
            except Exception:
                out[col] = np.nan
        return out

    # ------------------------------ Metrics Utils ------------------------------
    @staticmethod
    def _sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Compute annualized Sharpe ratio for 15-minute returns series.

        Annualization factor for 15m bars: 365 days * 24 hours * 4 bars/hour.
        """
        returns = returns.fillna(0.0).astype(float)
        if returns.empty:
            return 0.0
        excess = returns - risk_free_rate / (365.0 * 24.0 * 4.0)
        std = float(excess.std(ddof=1))
        if std == 0.0 or np.isnan(std):
            return 0.0
        mean = float(excess.mean())
        ann_factor = np.sqrt(365.0 * 24.0 * 4.0)
        return float((mean / std) * ann_factor)

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        """Compute maximum drawdown from an equity curve series."""
        if equity_curve.empty:
            return 0.0
        eq = equity_curve.astype(float).ffill().fillna(0.0)
        cummax = eq.cummax()
        drawdown = eq - cummax
        return float(drawdown.min())

    # ------------------------------ MLflow Logger ------------------------------
    @staticmethod
    def _log_metrics_to_mlflow(
        run_id: str, challenger: Dict[str, float], champion: Dict[str, float]
    ) -> None:
        """Log both challenger and champion metrics to the given MLflow run, if possible."""
        try:
            import mlflow

            # Log to existing run context
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(
                    {f"challenger_{k}": v for k, v in challenger.items()}
                )
                mlflow.log_metrics({f"champion_{k}": v for k, v in champion.items()})
        except Exception:
            # Silently ignore if MLflow not available or logging fails
            return
