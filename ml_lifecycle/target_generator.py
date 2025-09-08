"""Target variable generator (multi-horizon).

- Computes six ATR-based price barriers per spec.
- Labels using first-touch logic for three horizons.
- Joins with feature store and saves Parquet outputs.

Barriers (Wilder ATR, period 14):
- strong_bull: close + 2.5*ATR
- bull: close + 1.25*ATR
- bull_invalid: close - 0.75*ATR
- strong_bear: close - 2.5*ATR
- bear: close - 1.25*ATR
- bear_invalid: close + 0.75*ATR

Label rule within look-forward L bars (use T+1..T+L only):
- 2 if time_to_strong_bull < time_to_bull_invalid
- 1 if time_to_bull < time_to_strong_bull and time_to_bull < time_to_bull_invalid
- -2 if time_to_strong_bear < time_to_bear_invalid
- -1 if time_to_bear < time_to_strong_bear and time_to_bear < time_to_bear_invalid
- 0 otherwise
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # Prefer relative import when available
    from ..config import MASTER_TOKEN_LIST
    from ..core.dataframe_registry import DataFrameRegistry
except Exception:  # pragma: no cover - fallback for direct execution context
    from unicorn_wealth.config import MASTER_TOKEN_LIST  # type: ignore
    from unicorn_wealth.core.dataframe_registry import DataFrameRegistry  # type: ignore


__all__ = ["TargetGenerator"]


def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'timestamp' column exists and is datetime64[ns, UTC].

    Accepts common variations: a numeric epoch in 'timestamp' or an index.
    """
    if df is None or df.empty:
        empty = pd.DataFrame(columns=["timestamp"]).astype(
            {"timestamp": "datetime64[ns, UTC]"}
        )
        return empty

    out = df.copy()
    if "timestamp" not in out.columns:
        # Try from index
        if isinstance(out.index, pd.DatetimeIndex):
            out["timestamp"] = out.index
        else:
            # Best effort: if any column looks like a time, try it
            for cand in ("time", "date", "datetime"):
                if cand in out.columns:
                    out["timestamp"] = out[cand]
                    break
            if "timestamp" not in out.columns:
                # fabricate monotonic timestamps if nothing available
                out["timestamp"] = pd.to_datetime(out.index, utc=True, errors="coerce")

    # Normalize to UTC datetime
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).copy()
    return out


def _epoch_seconds(ts: pd.Series) -> pd.Series:
    """Convert a timestamp-like series to Unix epoch seconds (int)."""
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    return (dt.view("int64") // 10**9).astype("int64")


def _build_future_matrix(arr: np.ndarray, horizon: int) -> np.ndarray:
    """Build a matrix of future values for each row.

    mat[i, j] = arr[i + (j + 1)] for j in [0..horizon-1], NaN where out of bounds.
    """
    n = arr.shape[0]
    mat = np.full((n, horizon), np.nan, dtype=float)
    for j in range(1, horizon + 1):
        mat[:-j, j - 1] = arr[j:]
    return mat


def _first_touch_times(
    barrier: np.ndarray,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    direction: str,
) -> np.ndarray:
    """Compute first-touch times (1..L) for a barrier per row.

    Parameters
    ----------
    barrier: np.ndarray
        1D array of barrier levels per row (length n).
    future_highs: np.ndarray
        2D array (n, L) of future highs for each row and step j.
    future_lows: np.ndarray
        2D array (n, L) of future lows for each row and step j.
    direction: str
        One of {"above", "below"}. If "above" we test highs >= barrier.
        If "below" we test lows <= barrier.

    Returns
    -------
    np.ndarray
        Array of length n with first touch time in bars (1..L) or np.inf if never.
    """
    if direction not in {"above", "below"}:
        raise ValueError("direction must be 'above' or 'below'")

    n, L = future_highs.shape
    b = barrier.reshape(-1, 1).astype(float)

    if direction == "above":
        cond = future_highs >= b
    else:  # below
        cond = future_lows <= b

    any_hit = cond.any(axis=1)
    # argmax returns the first True index due to how booleans work in NumPy
    first_idx = cond.argmax(axis=1)  # 0-based
    times = np.where(any_hit, first_idx + 1, np.inf)
    return times.astype(float)


class TargetGenerator:
    """Generate ATR-barrier based categorical targets for ML training.

    This class relies on OHLCV 15m data from the DataFrameRegistry. It attempts
    keys of the form '{TOKEN}_ohlcv_15m_df'. Labels are produced per token and
    joined against the feature store for each horizon ('1h', '4h', '8h').
    """

    def __init__(self, registry: DataFrameRegistry) -> None:
        self._registry = registry

    @staticmethod
    def _calculate_barriers(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate six ATR-based price barriers for the given OHLCV DataFrame.

        Expects columns: 'timestamp', 'high', 'low', 'close'. Returns a DataFrame
        with 'timestamp' and six barrier columns.
        """
        df = _ensure_timestamp_column(ohlcv_df)
        required = {"high", "low", "close"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"OHLCV DataFrame missing required columns: {missing}")

        # Compute ATR (Wilder RMA, period=14)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        n = 14
        atr = tr.ewm(alpha=1 / n, adjust=False).mean()
        atr = atr.ffill()

        out = pd.DataFrame(
            {
                "timestamp": df["timestamp"].values,
                "barrier_strong_bull": (close + atr * 2.5).values,
                "barrier_bull": (close + atr * 1.25).values,
                "barrier_bull_invalid": (close - atr * 0.75).values,
                "barrier_strong_bear": (close - atr * 2.5).values,
                "barrier_bear": (close - atr * 1.25).values,
                "barrier_bear_invalid": (close + atr * 0.75).values,
            }
        )
        return out

    @staticmethod
    def _generate_labels(
        ohlcv_df: pd.DataFrame, barriers_df: pd.DataFrame, look_forward_period: int
    ) -> pd.DataFrame:
        """Generate categorical labels using first-touch ATR-barrier logic.

        Parameters
        ----------
        ohlcv_df : pd.DataFrame
            OHLCV with columns ['timestamp','high','low','close'].
        barriers_df : pd.DataFrame
            Output of _calculate_barriers aligned by timestamp.
        look_forward_period : int
            Number of future bars to scan (e.g., 4,16,32).

        Returns
        -------
        pd.DataFrame
            DataFrame with ['timestamp', 'target_prediction'] for the given horizon.
        """
        price = _ensure_timestamp_column(ohlcv_df)
        # Merge to ensure alignment on timestamp
        df = price.merge(barriers_df, on="timestamp", how="inner")

        H = _build_future_matrix(
            df["high"].astype(float).to_numpy(),
            look_forward_period,
        )
        L = _build_future_matrix(
            df["low"].astype(float).to_numpy(),
            look_forward_period,
        )

        # Extract barriers as numpy arrays aligned with df's order
        # Ensure same order as price by re-merging indices
        b = df
        sbull = b["barrier_strong_bull"].to_numpy()
        bull = b["barrier_bull"].to_numpy()
        bull_inv = b["barrier_bull_invalid"].to_numpy()
        sbear = b["barrier_strong_bear"].to_numpy()
        bear = b["barrier_bear"].to_numpy()
        bear_inv = b["barrier_bear_invalid"].to_numpy()

        # Compute first-touch times
        t_sbull = _first_touch_times(sbull, H, L, direction="above")
        t_bull = _first_touch_times(bull, H, L, direction="above")
        t_bull_inv = _first_touch_times(bull_inv, H, L, direction="below")

        t_sbear = _first_touch_times(sbear, H, L, direction="below")
        t_bear = _first_touch_times(bear, H, L, direction="below")
        t_bear_inv = _first_touch_times(bear_inv, H, L, direction="above")

        # Apply nested conditional logic using vectorized booleans
        # Initialize labels to 0
        label = np.zeros(len(df), dtype=np.int8)

        cond_2 = t_sbull < t_bull_inv
        label = np.where(cond_2, 2, label)

        cond_1 = (~cond_2) & (t_bull < t_sbull) & (t_bull < t_bull_inv)
        label = np.where(cond_1, 1, label)

        cond_m2 = (~cond_2) & (~cond_1) & (t_sbear < t_bear_inv)
        label = np.where(cond_m2, -2, label)

        cond_m1 = (
            (~cond_2)
            & (~cond_1)
            & (~cond_m2)
            & (t_bear < t_sbear)
            & (t_bear < t_bear_inv)
        )
        label = np.where(cond_m1, -1, label)

        out = df.loc[:, ["timestamp"]].copy()
        out["target_prediction"] = label.astype("int8")
        return out

    async def generate_targets(self) -> None:
        """Orchestrate target label generation for 1h, 4h, 8h horizons.

        Steps per horizon (("1h", 4), ("4h", 16), ("8h", 32)):
        1. For each token in MASTER_TOKEN_LIST, fetch OHLCV 15m from registry
           and compute barriers and labels.
        2. Concatenate labels across tokens, adding the token column.
        3. Fetch the horizon feature store (key 'feature_store_{h}' in registry),
           else fallback to CSV in output/training_data/.
        4. Join labels with the feature store on ['timestamp', 'token'].
        5. Save final labeled DataFrame to Parquet:
           output/training_data/ml_training_dataset_{h}.parquet
        """
        horizons: List[Tuple[str, int]] = [("1h", 4), ("4h", 16), ("8h", 32)]

        output_dir = Path("output") / "training_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Preload all per-token OHLCV DataFrames available in the registry
        tokens = list(MASTER_TOKEN_LIST.keys())

        for horizon, fwd in horizons:
            per_token_labels: List[pd.DataFrame] = []

            for token in tokens:
                # Try common registry keys for OHLCV 15m per token
                oh_keys = [
                    f"{token}_ohlcv_15m_df",
                    f"{token}_ohlcv_df",
                    f"ohlcv_15m_df_{token}",
                ]
                ohlcv: Optional[pd.DataFrame] = None
                for name in oh_keys:
                    try:
                        ohlcv = await self._registry.get_df(name)
                        break
                    except KeyError:
                        continue

                if ohlcv is None or ohlcv.empty:
                    # Skip token if no OHLCV available
                    continue

                # Compute barriers and labels for this token
                barriers = self._calculate_barriers(ohlcv)
                labels = self._generate_labels(ohlcv, barriers, look_forward_period=fwd)

                # Add token and normalized numeric timestamp to align with feature store
                labels["token"] = token
                # Numeric epoch for join alignment with feature store
                labels["timestamp_int"] = _epoch_seconds(labels["timestamp"])

                per_token_labels.append(labels)

            if not per_token_labels:
                # Nothing to do for this horizon
                continue

            labels_all = pd.concat(per_token_labels, axis=0, ignore_index=True)

            # Fetch feature store for this horizon
            fs: Optional[pd.DataFrame] = None
            try:
                fs = await self._registry.get_df(f"feature_store_{horizon}")
            except KeyError:
                fs = None

            if fs is None:
                # Fallback to CSV export if present
                csv_path = output_dir / f"feature_store_{horizon}.csv"
                if csv_path.exists():
                    try:
                        fs = pd.read_csv(csv_path.as_posix())
                    except Exception:
                        fs = None

            if fs is None or fs.empty:
                # If no feature store, still save labels per token for inspection
                out = labels_all.copy()
                out_path = output_dir / f"ml_training_dataset_{horizon}.parquet"
                # Save with pyarrow engine
                out.to_parquet(out_path.as_posix(), index=False)
                continue

            # Normalize feature store timestamp and join
            fs = fs.copy()
            if "timestamp" not in fs.columns or "token" not in fs.columns:
                # Cannot join safely; just save labels
                out = labels_all.copy()
                out_path = output_dir / f"ml_training_dataset_{horizon}.parquet"
                out.to_parquet(out_path.as_posix(), index=False)
                continue

            # Detect timestamp type in feature store (int epoch vs datetime)
            ts_is_int = pd.api.types.is_integer_dtype(
                fs["timestamp"]
            ) or pd.api.types.is_numeric_dtype(fs["timestamp"])
            if ts_is_int:
                fs["timestamp_int"] = fs["timestamp"].astype("int64")
            else:
                fs["timestamp"] = pd.to_datetime(
                    fs["timestamp"], utc=True, errors="coerce"
                )
                fs["timestamp_int"] = _epoch_seconds(fs["timestamp"]).astype("int64")

            # Left join: keep all feature rows and add the label when available
            merged = fs.merge(
                labels_all[["timestamp_int", "token", "target_prediction"]],
                on=["timestamp_int", "token"],
                how="left",
            )

            # If there is already a target column, prefer the freshly computed one
            if "target_prediction" in fs.columns:
                tp_y = merged["target_prediction_y"]
                tp_x = merged.get("target_prediction_x")
                merged["target_prediction"] = tp_y.combine_first(tp_x)
                drop_cols = [
                    c for c in merged.columns if c.endswith("_x") or c.endswith("_y")
                ]
                merged = merged.drop(columns=drop_cols)

            # Save to Parquet
            out_path = output_dir / f"ml_training_dataset_{horizon}.parquet"
            merged.to_parquet(out_path.as_posix(), index=False)

        return None
