from __future__ import annotations

from typing import Optional

from core.dataframe_registry import DataFrameRegistry
from database.sql_engine import RawDataSQLEngine
from features.technical.momentum import rsi_15m
from features.technical.volatility import (
    atr_15m,
    atr_normalized_15m,
)
from features.temporal.cyclical_time import (
    day_of_week_cos,
    day_of_week_sin,
    hour_of_day_cos,
    hour_of_day_sin,
    minute_of_hour_cos,
    minute_of_hour_sin,
)
from features.temporal.sessions import compute_session_features

__all__ = ["UnifiedFeatureEngine"]


class UnifiedFeatureEngine:
    """Hardcoded feature engineering engine (MVFS) for 15m cadence.

    This refactored engine does not read any external JSON specs. It provides
    an explicit pipeline that computes a minimal viable set of features for
    each token and uses the DataFrameRegistry for state management.
    """

    def __init__(
        self,
        *,
        registry: DataFrameRegistry,
        sql_engine: Optional[RawDataSQLEngine] = None,
        feature_spec_path: Optional[str] = None,
    ) -> None:
        # sql_engine is kept for backward compatibility but not used here.
        self._registry = registry
        self._sql_engine = sql_engine

    def run_15m_pipeline(self, token: str, ohlcv_15m_df, settings: dict):
        """Compute 15m features synchronously using provided source DataFrame.

        Returns a single-row DataFrame with the latest calculated features for the token.
        """
        # Compute indicators directly from provided data
        rsi_series = rsi_15m(ohlcv_15m_df, settings)
        atr_df_local = atr_15m(ohlcv_15m_df, settings)
        atr_norm_series = atr_normalized_15m(atr_df_local, ohlcv_15m_df)
        # Assemble final single-row DF
        import pandas as pd

        frames = [
            rsi_series.to_frame(name="rsi_15m"),
            atr_df_local,
            atr_norm_series.to_frame(name="atr_normalized_15m"),
        ]
        merged = pd.concat(frames, axis=1)
        final_row = merged.tail(1).reset_index(drop=True)
        return final_row

    # --------------------- Live multi-timeframe pipelines --------------------- #
    async def run_1h_pipeline(self) -> None:
        """Compute and store 1h features (ATR, ADX, ATR normalized) for all tokens.

        Reads OHLCV 1h frames from the in-memory DataFrameRegistry and stores
        results back to the registry under the following keys per token:
        - {t}_atr_1h_df
        - {t}_adx_1h_df
        - {t}_atr_normalized_1h_df
        """
        import logging
        import pandas as pd
        from config import MASTER_TOKEN_LIST, FEATURE_PARAMS
        from features.technical.volatility import atr_1h as _atr_1h
        from features.technical.volatility import atr_normalized_1h as _atrn_1h
        from features.technical.trend import adx_1h as _adx_1h

        log = logging.getLogger(__name__)
        log.info("UnifiedFeatureEngine: run_1h_pipeline() called")

        # Build settings for 1h
        params = (FEATURE_PARAMS or {}).get("1h", {})
        settings = {
            "1h": {
                "atr_1h": params.get("atr_1h", {"window": 40}),
                "adx_1h": params.get("adx_1h", {"window": 30}),
            }
        }

        for token in (MASTER_TOKEN_LIST or {}).keys():
            key_ohlcv = f"{token}_ohlcv_1h_df"
            try:
                ohlcv_1h = await self._registry.get_df(key_ohlcv)
            except KeyError:
                continue
            if ohlcv_1h is None or len(ohlcv_1h) == 0:
                continue
            # Ensure DataFrame index or columns consistent
            df = ohlcv_1h.copy()
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)

            atr = _atr_1h(df, settings)
            adx = _adx_1h(df, settings)
            atrn = _atrn_1h(atr, df).to_frame(name="atr_normalized_1h")

            await self._registry.store_df(
                name=f"{token}_atr_1h_df",
                df=atr,
                update_mode="rolling_append",
                storage_period=5000,
            )
            await self._registry.store_df(
                name=f"{token}_adx_1h_df",
                df=adx,
                update_mode="rolling_append",
                storage_period=5000,
            )
            await self._registry.store_df(
                name=f"{token}_atr_normalized_1h_df",
                df=atrn,
                update_mode="rolling_append",
                storage_period=5000,
            )

    async def run_4h_pipeline(self) -> None:
        """Compute and store 4h features (ATR, ADX, ATR normalized) for all tokens."""
        import logging
        import pandas as pd
        from config import MASTER_TOKEN_LIST, FEATURE_PARAMS
        from features.technical.volatility import atr_4h as _atr_4h
        from features.technical.volatility import atr_normalized_4h as _atrn_4h
        from features.technical.trend import adx_4h as _adx_4h

        log = logging.getLogger(__name__)
        log.info("UnifiedFeatureEngine: run_4h_pipeline() called")

        params = (FEATURE_PARAMS or {}).get("4h", {})
        settings = {
            "4h": {
                "atr_4h": params.get("atr_4h", {"window": 14}),
                "adx_4h": params.get("adx_4h", {"window": 40}),
            }
        }

        for token in (MASTER_TOKEN_LIST or {}).keys():
            key_ohlcv = f"{token}_ohlcv_4h_df"
            try:
                ohlcv_4h = await self._registry.get_df(key_ohlcv)
            except KeyError:
                continue
            if ohlcv_4h is None or len(ohlcv_4h) == 0:
                continue
            df = ohlcv_4h.copy()
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)

            atr = _atr_4h(df, settings)
            adx = _adx_4h(df, settings)
            atrn = _atrn_4h(atr, df).to_frame(name="atr_normalized_4h")

            await self._registry.store_df(
                name=f"{token}_atr_4h_df",
                df=atr,
                update_mode="rolling_append",
                storage_period=5000,
            )
            await self._registry.store_df(
                name=f"{token}_adx_4h_df",
                df=adx,
                update_mode="rolling_append",
                storage_period=5000,
            )
            await self._registry.store_df(
                name=f"{token}_atr_normalized_4h_df",
                df=atrn,
                update_mode="rolling_append",
                storage_period=5000,
            )

    async def run_1d_pipeline(self) -> None:
        """Compute and store 1d features (ATR, ADX, ATR normalized) for all tokens."""
        import logging
        import pandas as pd
        from config import MASTER_TOKEN_LIST, FEATURE_PARAMS
        from features.technical.volatility import atr_1d as _atr_1d
        from features.technical.volatility import atr_normalized_1d as _atrn_1d
        from features.technical.trend import adx_1d as _adx_1d

        log = logging.getLogger(__name__)
        log.info("UnifiedFeatureEngine: run_1d_pipeline() called")

        params = (FEATURE_PARAMS or {}).get("1d", {})
        settings = {
            "1d": {
                "atr_1d": params.get("atr_1d", {"window": 40}),
                "adx_1d": params.get("adx_1d", {"window": 30}),
            }
        }

        for token in (MASTER_TOKEN_LIST or {}).keys():
            key_ohlcv = f"{token}_ohlcv_1d_df"
            try:
                ohlcv_1d = await self._registry.get_df(key_ohlcv)
            except KeyError:
                continue
            if ohlcv_1d is None or len(ohlcv_1d) == 0:
                continue
            df = ohlcv_1d.copy()
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)

            atr = _atr_1d(df, settings)
            adx = _adx_1d(df, settings)
            atrn = _atrn_1d(atr, df).to_frame(name="atr_normalized_1d")

            await self._registry.store_df(
                name=f"{token}_atr_1d_df",
                df=atr,
                update_mode="rolling_append",
                storage_period=5000,
            )
            await self._registry.store_df(
                name=f"{token}_adx_1d_df",
                df=adx,
                update_mode="rolling_append",
                storage_period=5000,
            )
            await self._registry.store_df(
                name=f"{token}_atr_normalized_1d_df",
                df=atrn,
                update_mode="rolling_append",
                storage_period=5000,
            )

    async def run_historical_pipeline(self, tokens: list[str], db_session):
        """Historical pipeline: three-pass, multi-horizon export.

        For each horizon in ["1h", "4h", "8h"], this method:
          - loads required raw OHLCV data;
          - computes the full hardcoded feature set using horizon-specific FEATURE_PARAMS;
          - assembles a wide DataFrame containing only ML training features;
          - saves to the corresponding feature_store_{horizon} table.
        """
        import pandas as pd
        from sqlalchemy import text

        from config import FEATURE_PARAMS, HISTORICAL_LOOKBACK_DAYS
        from database.sql_engine import FeatureStoreSQLStorageEngine
        from features.technical.trend import (
            adx_15m as _adx_15m,
            adx_1h as _adx_1h,
            adx_4h as _adx_4h,
            adx_1d as _adx_1d,
        )
        from features.technical.volatility import atr_15m as _atr_15m
        from features.technical.volatility import atr_1h as _atr_1h
        from features.technical.volatility import atr_4h as _atr_4h
        from features.technical.volatility import atr_1d as _atr_1d
        from features.technical.momentum import rsi_15m as _rsi_15m
        from features.technical.volatility import atr_normalized_15m as _atr_norm_15m
        from features.technical.volatility import atr_normalized_1h as _atr_norm_1h
        from features.technical.volatility import atr_normalized_4h as _atr_norm_4h
        from features.technical.volatility import atr_normalized_1d as _atr_norm_1d

        # Determine time window bounds (in BigInt epoch ms as stored)
        end_ts = pd.Timestamp.now(tz="UTC")
        start_ts = end_ts - pd.Timedelta(days=int(HISTORICAL_LOOKBACK_DAYS))
        start_epoch_ms = int(start_ts.timestamp() * 1000)
        end_epoch_ms = int(end_ts.timestamp() * 1000)

        storage = FeatureStoreSQLStorageEngine(db_session)

        # Helper to load OHLCV for a token
        async def _load_raw_ohlcv(token: str):
            async with db_session.begin() as conn:
                queries = {
                    "15m": text(
                        """
                        SELECT timestamp, token, open, high, low, close, volume
                        FROM raw_ohlcv_15m
                        WHERE token = :token AND timestamp BETWEEN :start_ts AND :end_ts
                        ORDER BY timestamp
                        """
                    ),
                    "1h": text(
                        """
                        SELECT timestamp, token, open, high, low, close, volume
                        FROM raw_ohlcv_1h
                        WHERE token = :token AND timestamp BETWEEN :start_ts AND :end_ts
                        ORDER BY timestamp
                        """
                    ),
                    "4h": text(
                        """
                        SELECT timestamp, token, open, high, low, close, volume
                        FROM raw_ohlcv_4h
                        WHERE token = :token AND timestamp BETWEEN :start_ts AND :end_ts
                        ORDER BY timestamp
                        """
                    ),
                    "1d": text(
                        """
                        SELECT timestamp, token, open, high, low, close, volume
                        FROM raw_ohlcv_1d
                        WHERE token = :token AND timestamp BETWEEN :start_ts AND :end_ts
                        ORDER BY timestamp
                        """
                    ),
                }

                data = {}
                for tf, q in queries.items():
                    res = await conn.execute(
                        q,
                        {
                            "token": token,
                            "start_ts": start_epoch_ms,
                            "end_ts": end_epoch_ms,
                        },
                    )
                    rows = res.fetchall()
                    cols = res.keys()
                    import pandas as _pdi

                    data[tf] = (
                        _pdi.DataFrame(rows, columns=cols) if rows else _pdi.DataFrame()
                    )
                return data

        # Three-pass system
        for horizon in ["1h", "4h", "8h"]:
            params = FEATURE_PARAMS.get(horizon, {})

            for token in tokens:
                raw = await _load_raw_ohlcv(token)
                if all(df.empty for df in raw.values()):
                    continue

                # Compute all required features using horizon-specific params
                # Build settings dicts matching each function's expected path
                settings_15m = {
                    "15m": {
                        "rsi_15m": params.get("rsi_15m", {"window": 14}),
                        "atr_15m": params.get("atr_15m", {"window": 40}),
                        "adx_15m": params.get("adx_15m", {"window": 30}),
                    }
                }
                settings_1h = {
                    "1h": {
                        "atr_1h": params.get("atr_1h", {"window": 40}),
                        "adx_1h": params.get("adx_1h", {"window": 30}),
                    }
                }
                settings_4h = {
                    "4h": {
                        "atr_4h": params.get("atr_4h", {"window": 14}),
                        "adx_4h": params.get("adx_4h", {"window": 40}),
                    }
                }
                settings_1d = {
                    "1d": {
                        "atr_1d": params.get("atr_1d", {"window": 40}),
                        "adx_1d": params.get("adx_1d", {"window": 30}),
                    }
                }

                import pandas as _pd

                frames = []

                # 15m-sourced features
                if not raw["15m"].empty:
                    o15 = raw["15m"].copy()
                    # indicators
                    rsi15 = _rsi_15m(o15, settings_15m)
                    atr15 = _atr_15m(o15, settings_15m)
                    atrn15 = _atr_norm_15m(atr15, o15)
                    adx15 = _adx_15m(o15, settings_15m)
                    # assemble ML training features only: include RSI, ADX components, ATR normalized as a training feature per config
                    f15 = _pd.DataFrame(
                        {
                            "timestamp": o15["timestamp"].astype("int64"),
                            "token": token,
                            "rsi_15m": _pd.Series(rsi15).astype("float64"),
                            "adx_15m": adx15["adx"].astype("float64"),
                            "adx_pos_15m": adx15["adx_pos"].astype("float64"),
                            "adx_neg_15m": adx15["adx_neg"].astype("float64"),
                            "atr_normalized_15m": _pd.Series(atrn15).astype("float64"),
                        }
                    )
                    frames.append(f15)

                # 1h-sourced features
                if not raw["1h"].empty:
                    o1 = raw["1h"].copy()
                    adx1h = _adx_1h(o1, settings_1h)
                    atr1h = _atr_1h(o1, settings_1h)
                    atrn1h = _atr_norm_1h(atr1h, o1)
                    f1h = _pd.DataFrame(
                        {
                            "timestamp": o1["timestamp"].astype("int64"),
                            "token": token,
                            "adx_1h": adx1h["adx"].astype("float64"),
                            "adx_pos_1h": adx1h["adx_pos"].astype("float64"),
                            "adx_neg_1h": adx1h["adx_neg"].astype("float64"),
                            "atr_normalized_1h": _pd.Series(atrn1h).astype("float64"),
                        }
                    )
                    frames.append(f1h)

                # 4h-sourced features
                if not raw["4h"].empty:
                    o4 = raw["4h"].copy()
                    adx4h = _adx_4h(o4, settings_4h)
                    atr4h = _atr_4h(o4, settings_4h)
                    atrn4h = _atr_norm_4h(atr4h, o4)
                    f4h = _pd.DataFrame(
                        {
                            "timestamp": o4["timestamp"].astype("int64"),
                            "token": token,
                            "adx_4h": adx4h["adx"].astype("float64"),
                            "adx_pos_4h": adx4h["adx_pos"].astype("float64"),
                            "adx_neg_4h": adx4h["adx_neg"].astype("float64"),
                            "atr_normalized_4h": _pd.Series(atrn4h).astype("float64"),
                        }
                    )
                    frames.append(f4h)

                # 1d-sourced features
                if not raw["1d"].empty:
                    o1d = raw["1d"].copy()
                    adx1d = _adx_1d(o1d, settings_1d)
                    atr1d = _atr_1d(o1d, settings_1d)
                    atrn1d = _atr_norm_1d(atr1d, o1d)
                    f1d = _pd.DataFrame(
                        {
                            "timestamp": o1d["timestamp"].astype("int64"),
                            "token": token,
                            "adx_1d": adx1d["adx"].astype("float64"),
                            "adx_pos_1d": adx1d["adx_pos"].astype("float64"),
                            "adx_neg_1d": adx1d["adx_neg"].astype("float64"),
                            "atr_normalized_1d": _pd.Series(atrn1d).astype("float64"),
                        }
                    )
                    frames.append(f1d)

                if not frames:
                    continue

                # Build 15m base timeline and forward-fill higher timeframe features

                if raw.get("15m") is None or raw["15m"].empty:
                    # We require 15m base rows for all feature_store tables
                    continue

                base15 = raw["15m"][["timestamp"]].copy()
                base15["timestamp"] = base15["timestamp"].astype("int64")
                base15["token"] = token
                # Temporal cyclical time features relative to UTC
                base15["day_of_week_cos"] = day_of_week_cos(base15["timestamp"]).astype(
                    "float64"
                )
                base15["day_of_week_sin"] = day_of_week_sin(base15["timestamp"]).astype(
                    "float64"
                )
                base15["hour_of_day_cos"] = hour_of_day_cos(base15["timestamp"]).astype(
                    "float64"
                )
                base15["hour_of_day_sin"] = hour_of_day_sin(base15["timestamp"]).astype(
                    "float64"
                )
                base15["minute_of_hour_cos"] = minute_of_hour_cos(
                    base15["timestamp"]
                ).astype("float64")
                base15["minute_of_hour_sin"] = minute_of_hour_sin(
                    base15["timestamp"]
                ).astype("float64")

                # Sessions & Kill-Zone features (15m base, UTC)
                sess_df = compute_session_features(base15["timestamp"], bar_minutes=15)
                for c in sess_df.columns:
                    if str(sess_df[c].dtype) in ("boolean", "bool"):
                        base15[c] = sess_df[c].astype("float64")
                    else:
                        base15[c] = pd.to_numeric(sess_df[c], errors="coerce").astype(
                            "float64"
                        )

                # External boolean feature: economic_event
                # Prefer in-memory calendar during live; for historical, lazily fetch once
                econ_flag = None

                # Lazy-load and cache a calendar DataFrame for historical pipeline scope
                try:
                    _cal_df = None
                    # If a registry exists (live), try retrieving from it
                    if getattr(self, "_registry", None) is not None:
                        try:
                            _cal_df = await self._registry.get_df(
                                "finnhub_econ_calendar_df"
                            )
                        except Exception:
                            try:
                                _cal_df = await self._registry.get_df(
                                    "economic_event_df"
                                )
                            except Exception:
                                _cal_df = None
                    # If still None (e.g., historical path), attempt best-effort fetch via Finnhub
                    if _cal_df is None:
                        try:
                            import aiohttp
                            from core.config_loader import (
                                load_settings as _load_settings,
                            )
                            from data_ingestion.api.finnhub_client import (
                                FinnhubClient as _Finn,
                            )
                            import pandas as _pd

                            setts = _load_settings()
                            api_key = getattr(setts, "finnhub_api_key", "") or ""
                            if api_key:
                                # Determine a reasonable window from base15 timestamps
                                ts_series = _pd.to_datetime(
                                    base15["timestamp"], unit="ms", utc=True
                                )
                                start_d = ts_series.min().date().isoformat()
                                end_d = ts_series.max().date().isoformat()
                                async with aiohttp.ClientSession() as _sess:
                                    _cli = _Finn(api_key=api_key, session=_sess)
                                    # Fetch in monthly chunks to avoid provider limits and ensure coverage
                                    # Build month boundaries between start_d and end_d
                                    try:
                                        s_dt = _pd.to_datetime(start_d).date()
                                        e_dt = _pd.to_datetime(end_d).date()
                                        months: list[tuple[str, str]] = []
                                        cur = _pd.Timestamp(s_dt)
                                        last = _pd.Timestamp(e_dt)
                                        while cur <= last:
                                            month_start = cur.normalize()
                                            next_month = (
                                                month_start + _pd.offsets.MonthBegin(1)
                                            ).normalize()
                                            # end of chunk is min(last+1day, next_month) - 1 day
                                            chunk_end = min(
                                                last + _pd.Timedelta(days=1), next_month
                                            ) - _pd.Timedelta(days=1)
                                            months.append(
                                                (
                                                    month_start.date().isoformat(),
                                                    chunk_end.date().isoformat(),
                                                )
                                            )
                                            cur = next_month
                                        frames = []
                                        for f, t in months:
                                            try:
                                                df_part = await _cli.fetch_data(
                                                    **{"from": f, "to": t}
                                                )
                                                if (
                                                    isinstance(df_part, _pd.DataFrame)
                                                    and not df_part.empty
                                                ):
                                                    frames.append(df_part)
                                            except Exception:
                                                continue
                                        if frames:
                                            _cal_df = _pd.concat(
                                                frames, ignore_index=True
                                            )
                                            # Deduplicate by timestamp if present to avoid overlap across chunks
                                            try:
                                                if "timestamp" in _cal_df.columns:
                                                    _cal_df = _cal_df.drop_duplicates(subset=["timestamp"])  # type: ignore[arg-type]
                                            except Exception:
                                                pass
                                        else:
                                            _cal_df = _pd.DataFrame()
                                    except Exception:
                                        # Fallback to single-shot if chunking fails
                                        _cal_df = await _cli.fetch_data(
                                            **{"from": start_d, "to": end_d}
                                        )
                        except Exception:
                            _cal_df = None

                    cal_df = _cal_df
                except Exception:
                    cal_df = None

                try:
                    if cal_df is not None and getattr(cal_df, "empty", True) is False:
                        df_cal = cal_df.copy()
                        # Normalize timestamp column name first
                        if "timestamp" not in df_cal.columns:
                            for cand in (
                                "time",
                                "datetime",
                                "date",
                                "time_period_start",
                            ):
                                if cand in df_cal.columns:
                                    df_cal = df_cal.rename(columns={cand: "timestamp"})
                                    break
                        # Ensure timestamps are timezone-aware UTC
                        df_cal["timestamp"] = pd.to_datetime(
                            df_cal["timestamp"], utc=True, errors="coerce"
                        )
                        df_cal = df_cal[df_cal["timestamp"].notna()].copy()
                        # Drop the last row to avoid incomplete trailing data as per spec
                        if len(df_cal) > 0:
                            df_cal = df_cal.iloc[:-1]
                        # Normalize impact column and filter strictly for 'high' impact
                        impact_series = None
                        for col in ("impact", "Impact", "importance", "Importance"):
                            if col in df_cal.columns:
                                impact_series = df_cal[col]
                                break
                        if impact_series is None:
                            impact_series = pd.Series(
                                [None] * len(df_cal), index=df_cal.index
                            )
                        # Strict 'high' match (case-insensitive); remove previous numeric >=7 logic
                        impact_str = impact_series.astype(str).str.lower().str.strip()
                        is_high = impact_str.eq("high")
                        df_high = df_cal[is_high][["timestamp"]].copy()
                        if not df_high.empty:
                            # Prepare base timestamps as datetime UTC
                            ts_base = pd.to_datetime(
                                base15["timestamp"], unit="ms", utc=True
                            )
                            base_dt = pd.DataFrame({"timestamp": ts_base})
                            # Ensure sorted for merge_asof
                            base_dt = base_dt.sort_values("timestamp").reset_index(
                                drop=True
                            )
                            df_high = df_high.sort_values("timestamp").reset_index(
                                drop=True
                            )
                            # Nearest-asof within 12h tolerance (before or after)
                            merged = pd.merge_asof(
                                base_dt,
                                df_high,
                                on="timestamp",
                                direction="nearest",
                                tolerance=pd.Timedelta(hours=12),
                            )
                            econ_flag = merged["timestamp_y"].notna().astype("boolean")
                        else:
                            econ_flag = pd.Series(
                                [False] * len(base15), dtype="boolean"
                            )
                    else:
                        econ_flag = pd.Series([False] * len(base15), dtype="boolean")
                except Exception:
                    # Safety fallback: set False
                    econ_flag = pd.Series([False] * len(base15), dtype="boolean")
                base15["economic_event"] = (
                    econ_flag
                    if econ_flag is not None
                    else pd.Series([False] * len(base15), dtype="boolean")
                )

                # Start with 15m-sourced features (already aligned to 15m)
                # Be tolerant if the first feature frame lacks 'token' due to upstream quirks.
                if frames:
                    right0 = frames[0]
                    if "token" not in right0.columns:
                        try:
                            right0 = right0.copy()
                            right0["token"] = token
                        except Exception:
                            pass
                    # Normalize timestamp dtype on the right to match base (int64 epoch ms)
                    try:
                        import pandas as _pd  # local alias

                        if "timestamp" in right0.columns:
                            if _pd.api.types.is_datetime64_any_dtype(
                                right0["timestamp"]
                            ):
                                right0["timestamp"] = (
                                    _pd.to_datetime(
                                        right0["timestamp"], utc=True, errors="coerce"
                                    ).astype("int64")
                                    // 10**6
                                ).astype("int64")
                            else:
                                right0["timestamp"] = (
                                    _pd.to_numeric(right0["timestamp"], errors="coerce")
                                    .astype("Int64")
                                    .astype("int64")
                                )
                    except Exception:
                        pass
                    on_keys = (
                        ["timestamp", "token"]
                        if "token" in right0.columns
                        else ["timestamp"]
                    )
                    wide = base15.merge(right0, on=on_keys, how="left")
                    if "token" not in wide.columns:
                        wide["token"] = token
                else:
                    wide = base15

                def _asof_merge(
                    base: _pd.DataFrame, feat: _pd.DataFrame, cols: list[str]
                ) -> _pd.DataFrame:
                    if feat is None or feat.empty:
                        for c in cols:
                            if c not in base.columns:
                                base[c] = _pd.NA
                        return base
                    left = base.sort_values("timestamp")
                    right = feat.sort_values("timestamp")
                    merged = _pd.merge_asof(
                        left,
                        right[["timestamp"] + cols],
                        on="timestamp",
                        direction="backward",
                    )
                    return merged

                # Determine which frames correspond to 1h/4h/1d features
                # frames were appended in order: f15 (if exists), f1h, f4h, f1d

                # 1h features
                f1h_cols = ["adx_1h", "adx_pos_1h", "adx_neg_1h", "atr_normalized_1h"]
                f4h_cols = ["adx_4h", "adx_pos_4h", "adx_neg_4h", "atr_normalized_4h"]
                f1d_cols = ["adx_1d", "adx_pos_1d", "adx_neg_1d", "atr_normalized_1d"]

                # Extract frames safely by name (recompute small frames dictionary)
                fr_named = {c: None for c in ("f15", "f1h", "f4h", "f1d")}
                # Rebuild based on availability
                if not raw["15m"].empty:
                    fr_named["f15"] = frames[0]
                # Find others by matching columns present
                for fr in frames[1:]:
                    cols = set(fr.columns)
                    if {
                        "adx_1h",
                        "adx_pos_1h",
                        "adx_neg_1h",
                        "atr_normalized_1h",
                    }.issubset(cols):
                        fr_named["f1h"] = fr
                    elif {
                        "adx_4h",
                        "adx_pos_4h",
                        "adx_neg_4h",
                        "atr_normalized_4h",
                    }.issubset(cols):
                        fr_named["f4h"] = fr
                    elif {
                        "adx_1d",
                        "adx_pos_1d",
                        "adx_neg_1d",
                        "atr_normalized_1d",
                    }.issubset(cols):
                        fr_named["f1d"] = fr

                # Forward-fill higher TF features onto 15m timeline using backward-asof
                wide = _asof_merge(wide, fr_named["f1h"], f1h_cols)
                wide = _asof_merge(wide, fr_named["f4h"], f4h_cols)
                wide = _asof_merge(wide, fr_named["f1d"], f1d_cols)

                # Final sort and drop duplicates safeguard
                wide = wide.sort_values(["timestamp", "token"]).drop_duplicates(
                    subset=["timestamp", "token"], keep="last"
                )

                # Save to corresponding feature store table by horizon
                await storage.save_features(wide, horizon=horizon)
