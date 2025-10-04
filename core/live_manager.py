from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import numpy as np

from .dataframe_registry import DataFrameRegistry
from data_ingestion.api.yfinance_client import YFinanceClient

LOGGER = logging.getLogger(__name__)


class LiveDataManager:
    """Central hub for live data processing and scheduled fetches.

    Responsibilities:
    - Consume raw trade ticks and build 1-minute OHLCV candles per symbol.
    - Store completed 1m candles into the DataFrameRegistry.
    - Run a resampling cascade to produce higher timeframes (15m, 1h, 4h, 1d, 7d).
    - Periodically fetch API-based raw data on a hardcoded schedule (no JSON).

    The queue is expected to deliver trade dictionaries with at least:
      {"symbol": str, "price": float, "quantity": float, "timestamp": int(ms)}
    """

    def __init__(
        self,
        *,
        trade_queue: asyncio.Queue,
        registry: DataFrameRegistry,
        historical_api_clients: Optional[Iterable[Any]] = None,
        santiment_client: Any | None = None,
        coinmarketcap_client: Any | None = None,
        coinapi_client: Any | None = None,
    ) -> None:
        self.trade_queue = trade_queue
        self.registry = registry
        self.historical_api_clients: List[Any] = list(historical_api_clients or [])
        # Optional dedicated clients (preferred over historical_api_clients)
        self.santiment_client = santiment_client
        self.coinmarketcap_client = coinmarketcap_client
        self.coinapi_client = coinapi_client
        # In-progress candle state per symbol
        self._current_candles: Dict[str, Dict[str, Any]] = {}
        # Hardcoded schedule of jobs per cadence
        self._scheduled_jobs = {
            "15m": [
                self._fetch_active_addresses_24h_15m,
                self._fetch_social_dominance_15m,
                self._fetch_sentiment_weighted_total_15m,
                self._fetch_oi_total_15m,
                self._fetch_short_liq_notional_15m,
                self._fetch_long_liq_notional_15m,
                self._fetch_spx_close_15m,
                self._fetch_dxy_close_15m,
                self._fetch_gold_close_15m,
                self._fetch_fees_burnt_usd_15m,
                self._fetch_fees_usd_intraday_15m,
                self._fetch_median_fees_usd_15m,
                self._fetch_total_gas_used_15m,
                self._fetch_whale_transaction_count_100k_usd_to_inf_15m,
                self._fetch_whale_transaction_volume_1m_usd_to_inf_15m,
                self._fetch_futures_mark_price_15m,
                self._fetch_eth_supply_usd_15m,
            ],
            "30m": [
                self._fetch_funding_rate_30m,
            ],
            "1h": [
                self._fetch_perp_close_1h,
            ],
            "1d": [
                self._fetch_active_addresses_1d,
                self._fetch_social_dominance_1d,
                self._fetch_mvrv_long_short_diff_1d,
                self._fetch_network_profit_loss_1d,
                self._fetch_percent_supply_on_exchanges_1d,
                self._fetch_btc_etf_flow_1d,
                self._fetch_spx_close_1d,
                self._fetch_dxy_close_1d,
                self._fetch_gold_close_1d,
                self._fetch_ust10y_yield_1d,
                self._fetch_transaction_volume_usd_1d,
                self._fetch_btc_supply_usd_1d,
                self._fetch_btcd_close_1d,
                self._fetch_btc_mcap_1d,
                self._fetch_eth_mcap_1d,
                self._fetch_crypto_fng_index_1d,
            ],
        }

    # --------------------------- Tick Consumer --------------------------- #
    async def run_tick_consumer(self) -> None:
        """Continuously consume raw trades and build 1-minute candles.

        This loop never ends on its own; cancel the task to stop it.
        """
        while True:
            trade = await self.trade_queue.get()
            # Live tick identification log
            try:
                symbol = trade.get("symbol") if isinstance(trade, dict) else None
                price_val = trade.get("price") if isinstance(trade, dict) else None
                qty_val = (
                    (trade.get("quantity") if isinstance(trade, dict) else None)
                    if trade is not None
                    else None
                )
                if isinstance(trade, dict) and qty_val is None:
                    qty_val = trade.get("qty")
                try:
                    price_fmt = (
                        f"{float(price_val):.2f}" if price_val is not None else "nan"
                    )
                except Exception:
                    price_fmt = str(price_val)
                LOGGER.info(
                    "Tick Received: %s | Price: %s | Size: %s",
                    symbol,
                    price_fmt,
                    qty_val,
                )
            except Exception:
                # Don't let logging issues impact processing
                pass
            try:
                await self._handle_trade(trade)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Failed processing trade: %s", exc)
            finally:
                self.trade_queue.task_done()

    async def _handle_trade(self, trade: Dict[str, Any]) -> None:
        symbol = str(trade.get("symbol"))
        price = float(trade.get("price"))
        qty = float(trade.get("quantity", trade.get("qty", 0.0)))
        ts_ms = int(trade.get("timestamp"))

        # Determine 1-minute interval start (ms)
        minute_start_ms = ts_ms - (ts_ms % 60000)
        minute_start_ts = pd.to_datetime(minute_start_ms, unit="ms", utc=True)

        state = self._current_candles.get(symbol)
        if state is None:
            # Start first candle
            self._current_candles[symbol] = {
                "timestamp": minute_start_ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": qty,
            }
            return

        # If we moved to a new minute -> finalize previous and start a new one
        if pd.Timestamp(state["timestamp"]) != minute_start_ts:
            finalized = state.copy()
            await self._finalize_and_resample(symbol=symbol, candle_data=finalized)
            # Start new candle for current minute
            self._current_candles[symbol] = {
                "timestamp": minute_start_ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": qty,
            }
        else:
            # Update current minute candle
            state["high"] = max(state["high"], price)
            state["low"] = min(state["low"], price)
            state["close"] = price
            state["volume"] = float(state.get("volume", 0.0)) + qty

    # -------------------- Finalize & Resampling Cascade ----------------- #
    async def _finalize_and_resample(
        self, symbol: str, candle_data: Dict[str, Any]
    ) -> None:
        """Store the completed 1m candle and run the resampling cascade.

        Stores the 1m row as rolling_append and, when boundary conditions are met,
        resamples to 15m, 1h, 4h, 1d, and 7d.
        """
        # Build a single-row DataFrame with UTC timestamp index or column
        df_1m = pd.DataFrame([candle_data])
        # Ensure column order
        df_1m = df_1m[["timestamp", "open", "high", "low", "close", "volume"]]

        # Store 1m into registry (keep last 5,000 1m rows by default)
        await self.registry.store_df(
            name=f"{symbol}_ohlcv_1m_df",
            df=df_1m.set_index("timestamp"),
            update_mode="rolling_append",
            storage_period=5000,
        )

        ts: pd.Timestamp = pd.to_datetime(candle_data["timestamp"], utc=True)
        # Resampling boundaries to check
        boundaries = [
            ("15m", "15min", 15),
            ("1h", "1h", 60),
            ("4h", "4h", 60 * 4),
            ("1d", "1D", 60 * 24),
            ("7d", "7D", 60 * 24 * 7),
        ]

        # Pull latest source series as needed and resample
        for label, rule, minutes in boundaries:
            if self._completes_boundary(ts, minutes):
                await self._resample_from_lower(
                    symbol, lower_key="1m", rule=rule, out_label=label, boundary_ts=ts
                )

    def _completes_boundary(self, ts: pd.Timestamp, minutes: int) -> bool:
        # Normalize to minute precision
        ts = ts.floor("min")
        epoch = pd.Timestamp(0, tz="UTC")
        delta = int((ts - epoch).total_seconds() // 60)
        return delta % minutes == 0

    async def _resample_from_lower(
        self,
        symbol: str,
        lower_key: str,
        rule: str,
        out_label: str,
        boundary_ts: pd.Timestamp,
    ) -> None:
        """Generic resampler using pandas.resample on the lower timeframe.

        Aggregation: first, max, min, last, sum per spec.
        """
        src_name = f"{symbol}_ohlcv_{lower_key}_df"
        try:
            src_df = await self.registry.get_df(src_name)
        except KeyError:
            return
        if src_df is None or len(src_df) == 0:
            return

        # Ensure datetime index named 'timestamp'
        df = src_df.copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)

        # Take only the most recent window to compute a single new bar
        # Here we resample the last N periods where N depends on rule; taking
        # a reasonable slice from end to avoid huge memory cost.
        tail = df.tail(10000)
        agg = (
            tail.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(how="any")
        )

        if len(agg) == 0:
            return

        # Only append the last completed resampled bar before the current boundary
        boundary_ts = pd.to_datetime(boundary_ts, utc=True)
        completed = agg[agg.index < boundary_ts]
        if len(completed) == 0:
            return
        last_row = completed.tail(1).reset_index()
        out_name = f"{symbol}_ohlcv_{out_label}_df"
        await self.registry.store_df(
            name=out_name,
            df=last_row.set_index("timestamp"),
            update_mode="rolling_append",
            storage_period=5000,
        )

    # ------------------------ Scheduled Fetcher ------------------------- #
    async def fetch_scheduled_data(self, *, spec_path: Optional[str] = None) -> None:
        """Run hardcoded scheduled API fetches concurrently.

        Note: spec_path is accepted for backward compatibility but ignored.
        The schedule is defined in self._scheduled_jobs.
        """
        # Determine active cadences based on current UTC time
        # Derive from Python datetime.now so freezegun can freeze time
        from datetime import datetime, timezone

        now_dt = datetime.now(timezone.utc)
        now = pd.Timestamp(now_dt)
        tasks_to_run: List[Any] = []

        # Check for 15-minute cadence
        if now.minute % 15 == 0:
            tasks_to_run.extend(self._scheduled_jobs.get("15m", []))

        # Check for 30-minute cadence
        if now.minute % 30 == 0:
            tasks_to_run.extend(self._scheduled_jobs.get("30m", []))

        # Check for 1-hour cadence (top of the hour)
        if now.minute == 0:
            tasks_to_run.extend(self._scheduled_jobs.get("1h", []))

        # Check for 1-day cadence (midnight UTC)
        if now.hour == 0 and now.minute == 0:
            tasks_to_run.extend(self._scheduled_jobs.get("1d", []))

        if tasks_to_run:
            await asyncio.gather(*[job() for job in tasks_to_run])

    def _select_client_for_source(
        self, hint: str, client_map: Dict[str, Any]
    ) -> Optional[Any]:
        """Best-effort mapping from transform_data_source hint to a client.

        This is intentionally heuristic to keep coupling minimal. It looks for
        substrings like 'santiment', 'coinmarketcap', or 'coinapi'.
        """
        if "santiment" in hint:
            for key, cli in client_map.items():
                if "santiment" in key:
                    return cli
        if "coinmarketcap" in hint or "cmc" in hint:
            for key, cli in client_map.items():
                if "coinmarketcap" in key or "cmc" in key:
                    return cli
        if "coinapi" in hint or "ohlcv" in hint:
            for key, cli in client_map.items():
                if "coinapi" in key:
                    return cli
        # fallback: any client
        return next(iter(client_map.values()), None)

    # ------------------------ Hardcoded Job Helpers ---------------------- #
    async def _fetch_active_addresses_24h_15m(self) -> None:
        """Fetch Santiment active_addresses_24h at 15m cadence and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: active_addresses_24h
        - df_variable_id: {t}_active_addresses_24h_df
        - df_keys: timestamp, active_addresses_24h
        - df_update_mode: rolling_append
        - df_storage_period: 31
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="active_addresses_24h", slug=slug, interval="15m"
                )
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                # Normalize columns: value -> active_addresses_24h
                if "value" in df.columns and "active_addresses_24h" not in df.columns:
                    df = df.rename(columns={"value": "active_addresses_24h"})
                await self.registry.store_df(
                    name=f"{t}_active_addresses_24h_df",
                    df=df,
                    update_mode="rolling_append",
                    storage_period=31,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning(
                    "_fetch_active_addresses_24h_15m failed for %s: %s", t, exc
                )

    async def _fetch_active_addresses_1d(self) -> None:
        """Fetch Santiment active_addresses_24h for selected tokens (1d)."""
        client = self.santiment_client or None
        if client is None:
            # attempt to find in provided list for backward compat
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="active_addresses_24h", slug=slug, interval="1d"
                )
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                # Normalize columns
                if "value" in df.columns and "active_addresses_24h" not in df.columns:
                    df = df.rename(columns={"value": "active_addresses_24h"})
                await self.registry.store_df(
                    name=f"{t}_active_addresses_24h_df",
                    df=df,
                    update_mode="overwrite",
                    storage_period=0,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning("_fetch_active_addresses_1d failed for %s: %s", t, exc)

    async def _fetch_transaction_volume_usd_1d(self) -> None:
        """Fetch Santiment transaction_volume_usd at 1d cadence and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: transaction_volume_usd_1d
        - df_variable_id: {t}_transaction_volume_usd_1d_df
        - df_keys: timestamp, transaction_volume_usd_1d
        - df_update_mode: rolling_append
        - df_storage_period: 1
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="transaction_volume_usd", slug=slug, interval="1d"
                )
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                # Normalize columns: value -> transaction_volume_usd_1d
                if (
                    "value" in df.columns
                    and "transaction_volume_usd_1d" not in df.columns
                ):
                    df = df.rename(columns={"value": "transaction_volume_usd_1d"})
                await self.registry.store_df(
                    name=f"{t}_transaction_volume_usd_1d_df",
                    df=df,
                    update_mode="rolling_append",
                    storage_period=1,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning(
                    "_fetch_transaction_volume_usd_1d failed for %s: %s", t, exc
                )

    async def _fetch_whale_transaction_count_100k_usd_to_inf_15m(self) -> None:
        """Fetch Santiment whale_transaction_count_100k_usd_to_inf at 15m cadence and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: whale_transaction_count_100k_usd_to_inf
        - df_variable_id: {t}_whale_transaction_count_100k_usd_to_inf_df
        - df_keys: timestamp, whale_transaction_count_100k_usd_to_inf
        - df_update_mode: rolling_append
        - df_storage_period: 31
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="whale_transaction_count_100k_usd_to_inf",
                    slug=slug,
                    interval="15m",
                )
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                # Normalize columns: value -> whale_transaction_count_100k_usd_to_inf
                if (
                    "value" in df.columns
                    and "whale_transaction_count_100k_usd_to_inf" not in df.columns
                ):
                    df = df.rename(
                        columns={"value": "whale_transaction_count_100k_usd_to_inf"}
                    )
                await self.registry.store_df(
                    name=f"{t}_whale_transaction_count_100k_usd_to_inf_df",
                    df=df,
                    update_mode="rolling_append",
                    storage_period=31,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning(
                    "_fetch_whale_transaction_count_100k_usd_to_inf_15m failed for %s: %s",
                    t,
                    exc,
                )

    async def _fetch_social_dominance_15m(self) -> None:
        """Fetch Santiment social_dominance_total at 15m cadence."""
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="social_dominance_total", slug=slug, interval="15m"
                )
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                if (
                    "value" in df.columns
                    and "social_dominance_total_15m" not in df.columns
                ):
                    df = df.rename(columns={"value": "social_dominance_total_15m"})
                await self.registry.store_df(
                    name=f"{t}_social_dominance_total_15m_df",
                    df=df,
                    update_mode="rolling_append",
                    storage_period=97,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("_fetch_social_dominance_15m failed for %s: %s", t, exc)

    async def _fetch_sentiment_weighted_total_15m(self) -> None:
        """Fetch Santiment sentiment_weighted_total at 15m cadence and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: sentiment_weighted_total_15m
        - df_variable_id: {t}_sentiment_weighted_total_15m_df
        - df_keys: timestamp, sentiment_weighted_total_15m
        - df_update_mode: rolling_append
        - df_storage_period: 97
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="sentiment_weighted_total", slug=slug, interval="15m"
                )
                import pandas as _pd

                if not isinstance(df, _pd.DataFrame):
                    df = _pd.DataFrame(df)
                # Normalize columns: value -> sentiment_weighted_total_15m
                if (
                    "value" in df.columns
                    and "sentiment_weighted_total_15m" not in df.columns
                ):
                    df = df.rename(columns={"value": "sentiment_weighted_total_15m"})
                await self.registry.store_df(
                    name=f"{t}_sentiment_weighted_total_15m_df",
                    df=df,
                    update_mode="rolling_append",
                    storage_period=97,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning(
                    "_fetch_sentiment_weighted_total_15m failed for %s: %s", t, exc
                )

    async def _fetch_oi_total_15m(self) -> None:
        """Fetch Santiment exchange_open_interest at 15m cadence and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: oi_total_15m
        - df_variable_id: {t}_oi_total_15m_df
        - df_keys: timestamp, oi_total_15m
        - df_update_mode: rolling_append
        - df_storage_period: 97
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="exchange_open_interest", slug=slug, interval="15m"
                )
                import pandas as _pd

                if not isinstance(df, _pd.DataFrame):
                    df = _pd.DataFrame(df)
                # Normalize columns: value -> oi_total_15m
                if "value" in df.columns and "oi_total_15m" not in df.columns:
                    df = df.rename(columns={"value": "oi_total_15m"})
                await self.registry.store_df(
                    name=f"{t}_oi_total_15m_df",
                    df=df,
                    update_mode="rolling_append",
                    storage_period=97,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning("_fetch_oi_total_15m failed for %s: %s", t, exc)

    async def _fetch_short_liq_notional_15m(self) -> None:
        """Fetch Santiment liquidations (15m) and compute SHORT notional per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: short_liq_notional_15m
        - df_variable_id: {t}_short_liq_notional_15m_df
        - df_keys: timestamp, shortLiqNotional_15m
        - df_update_mode: rolling_append
        - df_storage_period: 193
        - calculate_per_token: true
        - transform: if positionType == 'SHORT' then price*volume; sum over last 15m bucket
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        import pandas as _pd

        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                # Prefer a dedicated method if available
                if hasattr(client, "fetch_liquidations"):
                    df = await client.fetch_liquidations(slug=slug, interval="15m")
                else:
                    # Fallback to a generic metric name if supported in environment
                    df = await client.fetch_data(
                        metric="liquidations_historical", slug=slug, interval="15m"
                    )
                if not isinstance(df, _pd.DataFrame):
                    df = _pd.DataFrame(df)
                if len(df) == 0:
                    # Store empty frame with correct columns
                    empty = _pd.DataFrame(columns=["timestamp", "shortLiqNotional_15m"])
                    await self.registry.store_df(
                        name=f"{t}_short_liq_notional_15m_df",
                        df=empty,
                        update_mode="rolling_append",
                        storage_period=193,
                    )
                    continue
                # Ensure timestamp as UTC datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
                elif "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "timestamp"})
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
                # Column robustness: recognize variants
                pos_col = None
                for cand in ("positionType", "position_type", "side", "position"):
                    if cand in df.columns:
                        pos_col = cand
                        break
                price_col = None
                for cand in ("price", "price_usd", "p"):
                    if cand in df.columns:
                        price_col = cand
                        break
                vol_col = None
                for cand in ("volume", "qty", "amount", "size", "v"):
                    if cand in df.columns:
                        vol_col = cand
                        break
                if pos_col is None or price_col is None or vol_col is None:
                    # If a pre-aggregated column is present, use it directly
                    for agg_cand in ("shortLiqNotional_15m", "short_liq_notional_15m"):
                        if agg_cand in df.columns:
                            out = df[["timestamp", agg_cand]].copy()
                            if agg_cand != "shortLiqNotional_15m":
                                out = out.rename(
                                    columns={agg_cand: "shortLiqNotional_15m"}
                                )
                            await self.registry.store_df(
                                name=f"{t}_short_liq_notional_15m_df",
                                df=out.reset_index(drop=True),
                                update_mode="rolling_append",
                                storage_period=193,
                            )
                            break
                    else:
                        # Can't compute; skip gracefully
                        continue
                    continue
                # Filter SHORT rows (case-insensitive)
                mask_short = df[pos_col].astype(str).str.upper().eq("SHORT")
                short_df = df[mask_short].copy()
                if len(short_df) == 0:
                    out = _pd.DataFrame(columns=["timestamp", "shortLiqNotional_15m"])
                else:
                    # Compute notional and aggregate per timestamp bucket
                    short_df["_notional"] = _pd.to_numeric(
                        short_df[price_col], errors="coerce"
                    ) * _pd.to_numeric(short_df[vol_col], errors="coerce")
                    # Group by timestamp (assumed aligned to 15m by provider or fetch window)
                    agg = (
                        short_df.dropna(subset=["timestamp", "_notional"])
                        .groupby("timestamp", as_index=False)["_notional"]
                        .sum()
                        .rename(columns={"_notional": "shortLiqNotional_15m"})
                    )
                    out = agg[["timestamp", "shortLiqNotional_15m"]]
                await self.registry.store_df(
                    name=f"{t}_short_liq_notional_15m_df",
                    df=out.reset_index(drop=True),
                    update_mode="rolling_append",
                    storage_period=193,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning(
                    "_fetch_short_liq_notional_15m failed for %s: %s", t, exc
                )

    async def _fetch_long_liq_notional_15m(self) -> None:
        """Fetch Santiment liquidations (15m) and compute LONG notional per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: long_liq_notional_15m
        - df_variable_id: {t}_long_liq_notional_15m_df
        - df_keys: timestamp, longLiqNotional_15m
        - df_update_mode: rolling_append
        - df_storage_period: 193
        - calculate_per_token: true
        - transform: if positionType == 'LONG' then price*volume; sum over last 15m bucket
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        import pandas as _pd

        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                # Prefer a dedicated method if available
                if hasattr(client, "fetch_liquidations"):
                    df = await client.fetch_liquidations(slug=slug, interval="15m")
                else:
                    # Fallback to a generic metric name if supported in environment
                    df = await client.fetch_data(
                        metric="liquidations_historical", slug=slug, interval="15m"
                    )
                if not isinstance(df, _pd.DataFrame):
                    df = _pd.DataFrame(df)
                if len(df) == 0:
                    # Store empty frame with correct columns
                    empty = _pd.DataFrame(columns=["timestamp", "longLiqNotional_15m"])
                    await self.registry.store_df(
                        name=f"{t}_long_liq_notional_15m_df",
                        df=empty,
                        update_mode="rolling_append",
                        storage_period=193,
                    )
                    continue
                # Ensure timestamp as UTC datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
                elif "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "timestamp"})
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
                # Column robustness: recognize variants
                pos_col = None
                for cand in ("positionType", "position_type", "side", "position"):
                    if cand in df.columns:
                        pos_col = cand
                        break
                price_col = None
                for cand in ("price", "price_usd", "p"):
                    if cand in df.columns:
                        price_col = cand
                        break
                vol_col = None
                for cand in ("volume", "qty", "amount", "size", "v"):
                    if cand in df.columns:
                        vol_col = cand
                        break
                if pos_col is None or price_col is None or vol_col is None:
                    # If a pre-aggregated column is present, use it directly
                    for agg_cand in ("longLiqNotional_15m", "long_liq_notional_15m"):
                        if agg_cand in df.columns:
                            out = df[["timestamp", agg_cand]].copy()
                            if agg_cand != "longLiqNotional_15m":
                                out = out.rename(
                                    columns={agg_cand: "longLiqNotional_15m"}
                                )
                            await self.registry.store_df(
                                name=f"{t}_long_liq_notional_15m_df",
                                df=out.reset_index(drop=True),
                                update_mode="rolling_append",
                                storage_period=193,
                            )
                            break
                    else:
                        # Can't compute; skip gracefully
                        continue
                    continue
                # Filter LONG rows (case-insensitive)
                mask_long = df[pos_col].astype(str).str.upper().eq("LONG")
                long_df = df[mask_long].copy()
                if len(long_df) == 0:
                    out = _pd.DataFrame(columns=["timestamp", "longLiqNotional_15m"])
                else:
                    # Compute notional and aggregate per timestamp bucket
                    long_df["_notional"] = _pd.to_numeric(
                        long_df[price_col], errors="coerce"
                    ) * _pd.to_numeric(long_df[vol_col], errors="coerce")
                    # Group by timestamp (assumed aligned to 15m by provider or fetch window)
                    agg = (
                        long_df.dropna(subset=["timestamp", "_notional"])
                        .groupby("timestamp", as_index=False)["_notional"]
                        .sum()
                        .rename(columns={"_notional": "longLiqNotional_15m"})
                    )
                    out = agg[["timestamp", "longLiqNotional_15m"]]
                await self.registry.store_df(
                    name=f"{t}_long_liq_notional_15m_df",
                    df=out.reset_index(drop=True),
                    update_mode="rolling_append",
                    storage_period=193,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning("_fetch_long_liq_notional_15m failed for %s: %s", t, exc)

    async def _fetch_fees_burnt_usd_15m(self) -> None:
        """Fetch Santiment fees_burnt_usd at 15m cadence and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: fees_burnt_usd_15m
        - df_variable_id: fees_burnt_usd_df
        - df_keys: timestamp, fees_burnt_usd_15m
        - df_update_mode: rolling_append
        - df_storage_period: 2
        - calculate_per_token: false (global metric, use ethereum slug)
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            df = await client.fetch_data(
                metric="fees_burnt_usd", slug="ethereum", interval="15m"
            )
            import pandas as _pd

            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            # Normalize columns: value -> fees_burnt_usd_15m
            if "value" in df.columns and "fees_burnt_usd_15m" not in df.columns:
                df = df.rename(columns={"value": "fees_burnt_usd_15m"})
            await self.registry.store_df(
                name="fees_burnt_usd_df",
                df=df,
                update_mode="rolling_append",
                storage_period=2,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_fees_burnt_usd_15m failed: %s", exc)

    async def _fetch_fees_usd_intraday_15m(self) -> None:
        """Fetch Santiment fees_usd_intraday at 15m cadence and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: fees_usd_intraday_15m
        - df_variable_id: fees_usd_intraday_df
        - df_keys: timestamp, fees_usd_intraday_15m
        - df_update_mode: rolling_append
        - df_storage_period: 168
        - calculate_per_token: false (global metric, use ethereum slug)
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            df = await client.fetch_data(
                metric="fees_usd_intraday", slug="ethereum", interval="15m"
            )
            import pandas as _pd

            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            # Normalize columns: value -> fees_usd_intraday_15m
            if "value" in df.columns and "fees_usd_intraday_15m" not in df.columns:
                df = df.rename(columns={"value": "fees_usd_intraday_15m"})
            await self.registry.store_df(
                name="fees_usd_intraday_df",
                df=df,
                update_mode="rolling_append",
                storage_period=168,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_fees_usd_intraday_15m failed: %s", exc)

    async def _fetch_median_fees_usd_15m(self) -> None:
        """Fetch Santiment median_fees_usd_5m at 15m cadence and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: median_fees_usd_15m
        - df_variable_id: median_fees_15m_df
        - df_keys: timestamp, median_fees_usd_15m
        - df_update_mode: rolling_append
        - df_storage_period: 168
        - calculate_per_token: false (global metric, use ethereum slug)
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            df = await client.fetch_data(
                metric="median_fees_usd_5m", slug="ethereum", interval="15m"
            )
            import pandas as _pd

            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            # Normalize columns: value -> median_fees_usd_15m
            if "value" in df.columns and "median_fees_usd_15m" not in df.columns:
                df = df.rename(columns={"value": "median_fees_usd_15m"})
            await self.registry.store_df(
                name="median_fees_15m_df",
                df=df,
                update_mode="rolling_append",
                storage_period=168,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_median_fees_usd_15m failed: %s", exc)

    async def _fetch_total_gas_used_15m(self) -> None:
        """Fetch Santiment total_gas_used at 15m cadence and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: total_gas_used_15m
        - df_variable_id: total_gas_used_15m_df
        - df_keys: timestamp, total_gas_used_15m
        - df_update_mode: rolling_append
        - df_storage_period: 96
        - calculate_per_token: false (global metric, use ethereum slug)
        - transform: rename 'value' -> 'total_gas_used_15m'
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            df = await client.fetch_data(
                metric="total_gas_used", slug="ethereum", interval="15m"
            )
            import pandas as _pd

            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            # Normalize columns: value -> total_gas_used_15m
            if "value" in df.columns and "total_gas_used_15m" not in df.columns:
                df = df.rename(columns={"value": "total_gas_used_15m"})
            await self.registry.store_df(
                name="total_gas_used_15m_df",
                df=df,
                update_mode="rolling_append",
                storage_period=96,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_total_gas_used_15m failed: %s", exc)

    async def _fetch_social_dominance_1d(self) -> None:
        """Fetch Santiment social_dominance_total at 1d cadence."""
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="social_dominance_total", slug=slug, interval="1d"
                )
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                if (
                    "value" in df.columns
                    and "social_dominance_total_1d" not in df.columns
                ):
                    df = df.rename(columns={"value": "social_dominance_total_1d"})
                await self.registry.store_df(
                    name=f"{t}_social_dominance_total_1d_df",
                    df=df,
                    update_mode="rolling_append",
                    storage_period=7,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("_fetch_social_dominance_1d failed for %s: %s", t, exc)

    async def _fetch_spx_close_15m(self) -> None:
        """Fetch S&P 500 (^GSPC) close at 15m cadence via yfinance.

        Stores DataFrame under key 'spx_close_15m_df' with columns:
        - timestamp (UTC)
        - spx_close_15m
        Uses rolling_append with storage_period=70 as per spec.
        """
        try:
            client = YFinanceClient(api_key=None, session=None)
            # Fetch recent window; drop the last (potentially incomplete) row
            import pandas as _pd

            end = _pd.Timestamp.utcnow().floor("min")
            start = end - _pd.Timedelta(days=10)
            df = await client.fetch_data(
                symbol="^GSPC", interval="15m", start=start, end=end
            )
            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            if len(df) == 0:
                return
            # Ensure timestamp exists and is UTC
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            else:
                # yfinance reset_index() created 'Datetime'/'Date' -> handled in client
                pass
            # Drop the last row to avoid incomplete bar
            df = df.iloc[:-1] if len(df) > 0 else df
            # Keep only needed columns and rename
            if "Close" in df.columns:
                out = df[["timestamp", "Close"]].rename(
                    columns={"Close": "spx_close_15m"}
                )
            elif "close" in df.columns:
                out = df[["timestamp", "close"]].rename(
                    columns={"close": "spx_close_15m"}
                )
            else:
                # Fallback: try 'Adj Close'
                if "Adj Close" in df.columns:
                    out = df[["timestamp", "Adj Close"]].rename(
                        columns={"Adj Close": "spx_close_15m"}
                    )
                else:
                    return
            await self.registry.store_df(
                name="spx_close_15m_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=70,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("_fetch_spx_close_15m failed: %s", exc)

    async def _fetch_mvrv_long_short_diff_1d(self) -> None:
        """Fetch Santiment mvrv_long_short_diff_usd (1d)."""
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                # Some setups may use metric id 'mvrv_long_short_diff_usd' or 'mvrv'
                try:
                    df = await client.fetch_data(
                        metric="mvrv_long_short_diff_usd", slug=slug, interval="1d"
                    )
                except Exception:
                    df = await client.fetch_data(
                        metric="mvrv", slug=slug, interval="1d"
                    )
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                if (
                    "value" in df.columns
                    and "mvrv_long_short_diff_usd_1d" not in df.columns
                ):
                    df = df.rename(columns={"value": "mvrv_long_short_diff_usd_1d"})
                await self.registry.store_df(
                    name=f"{t}_mvrv_long_short_diff_usd_df",
                    df=df,
                    update_mode="overwrite",
                    storage_period=0,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning(
                    "_fetch_mvrv_long_short_diff_1d failed for %s: %s", t, exc
                )

    async def _fetch_network_profit_loss_1d(self) -> None:
        """Fetch Santiment network_profit_loss (1d) and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: network_profit_loss_1d
        - df_variable_id: {t}_network_profit_loss_1d_df
        - df_keys: timestamp, network_profit_loss_1d
        - df_update_mode: rolling_append
        - df_storage_period: 1
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="network_profit_loss", slug=slug, interval="1d"
                )
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                # Normalize columns: value -> network_profit_loss_1d
                if "value" in df.columns and "network_profit_loss_1d" not in df.columns:
                    df = df.rename(columns={"value": "network_profit_loss_1d"})
                await self.registry.store_df(
                    name=f"{t}_network_profit_loss_1d_df",
                    df=df,
                    update_mode="rolling_append",
                    storage_period=1,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning(
                    "_fetch_network_profit_loss_1d failed for %s: %s", t, exc
                )

    async def _fetch_percent_supply_on_exchanges_1d(self) -> None:
        """Fetch Santiment percent_of_total_supply_on_exchanges (1d) and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: percent_of_total_supply_on_exchanges_1d
        - df_variable_id: {t}_percent_of_total_supply_on_exchanges_1d_df
        - df_keys: timestamp, percent_of_total_supply_on_exchanges_1d
        - df_update_mode: rolling_append
        - df_storage_period: 91
        - calculate_per_token: true
        - transform: rename 'value' -> 'percent_of_total_supply_on_exchanges_1d'
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="percent_of_total_supply_on_exchanges",
                    slug=slug,
                    interval="1d",
                )
                import pandas as _pd

                if not isinstance(df, _pd.DataFrame):
                    df = _pd.DataFrame(df)
                # Normalize columns: value -> percent_of_total_supply_on_exchanges_1d
                if (
                    "value" in df.columns
                    and "percent_of_total_supply_on_exchanges_1d" not in df.columns
                ):
                    df = df.rename(
                        columns={"value": "percent_of_total_supply_on_exchanges_1d"}
                    )
                await self.registry.store_df(
                    name=f"{t}_percent_of_total_supply_on_exchanges_1d_df",
                    df=df,
                    update_mode="rolling_append",
                    storage_period=91,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning(
                    "_fetch_percent_supply_on_exchanges_1d failed for %s: %s", t, exc
                )

    async def _fetch_btc_etf_flow_1d(self) -> None:
        """Fetch Santiment ETF Flow for BTC (1d) and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: btc_etf_flow_1d
        - calculate_per_token: false (single BTC series)
        - df_variable_id: btc_etf_flow_1d_df
        - df_keys: timestamp, btc_etf_flow_1d
        - df_update_mode: rolling_append
        - df_storage_period: 7
        - transform: rename 'value' -> 'btc_etf_flow_1d'
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            # Santiment metric id for ETF flow, scoped to bitcoin slug
            df = await client.fetch_data(
                metric="etf_flow", slug="bitcoin", interval="1d"
            )
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            # Normalize columns: value -> btc_etf_flow_1d
            if "value" in df.columns and "btc_etf_flow_1d" not in df.columns:
                df = df.rename(columns={"value": "btc_etf_flow_1d"})
            await self.registry.store_df(
                name="btc_etf_flow_1d_df",
                df=df,
                update_mode="rolling_append",
                storage_period=7,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("_fetch_btc_etf_flow_1d failed: %s", exc)

    async def _fetch_spx_close_1d(self) -> None:
        """Fetch S&P 500 (^GSPC) close at 1d cadence via yfinance.

        Stores DataFrame under key 'spx_close_1d_df' with columns:
        - timestamp (UTC)
        - spx_close_1d
        Uses rolling_append with storage_period=70 as per spec.
        """
        try:
            client = YFinanceClient(api_key=None, session=None)
            import pandas as _pd

            end = _pd.Timestamp.utcnow().floor("D")
            start = end - _pd.Timedelta(days=120)
            df = await client.fetch_data(
                symbol="^GSPC", interval="1d", start=start, end=end
            )
            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            if len(df) == 0:
                return
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            # Drop the last row to avoid incomplete bar
            df = df.iloc[:-1] if len(df) > 0 else df
            if "Close" in df.columns:
                out = df[["timestamp", "Close"]].rename(
                    columns={"Close": "spx_close_1d"}
                )
            elif "close" in df.columns:
                out = df[["timestamp", "close"]].rename(
                    columns={"close": "spx_close_1d"}
                )
            else:
                if "Adj Close" in df.columns:
                    out = df[["timestamp", "Adj Close"]].rename(
                        columns={"Adj Close": "spx_close_1d"}
                    )
                else:
                    return
            await self.registry.store_df(
                name="spx_close_1d_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=70,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("_fetch_spx_close_1d failed: %s", exc)

    async def _fetch_dxy_close_15m(self) -> None:
        """Fetch US Dollar Index (DX-Y.NYB) close at 15m via yfinance.

        Stores under 'dxy_close_15m_df' with columns [timestamp, dxy_close_15m].
        Applies df.iloc[:-1] to drop potentially incomplete last bar.
        """
        try:
            client = YFinanceClient(api_key=None, session=None)
            import pandas as _pd

            end = _pd.Timestamp.utcnow().floor("min")
            start = end - _pd.Timedelta(days=10)
            df = await client.fetch_data(
                symbol="DX-Y.NYB", interval="15m", start=start, end=end
            )
            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            if len(df) == 0:
                return
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            df = df.iloc[:-1] if len(df) > 0 else df
            if "Close" in df.columns:
                out = df[["timestamp", "Close"]].rename(
                    columns={"Close": "dxy_close_15m"}
                )
            elif "close" in df.columns:
                out = df[["timestamp", "close"]].rename(
                    columns={"close": "dxy_close_15m"}
                )
            elif "Adj Close" in df.columns:
                out = df[["timestamp", "Adj Close"]].rename(
                    columns={"Adj Close": "dxy_close_15m"}
                )
            else:
                return
            await self.registry.store_df(
                name="dxy_close_15m_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=70,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("_fetch_dxy_close_15m failed: %s", exc)

    async def _fetch_dxy_close_1d(self) -> None:
        """Fetch US Dollar Index (DX-Y.NYB) close at 1d via yfinance.

        Stores under 'dxy_close_1d_df' with columns [timestamp, dxy_close_1d].
        Applies df.iloc[:-1] to drop potentially incomplete last bar.
        """
        try:
            client = YFinanceClient(api_key=None, session=None)
            import pandas as _pd

            end = _pd.Timestamp.utcnow().floor("D")
            start = end - _pd.Timedelta(days=120)
            df = await client.fetch_data(
                symbol="DX-Y.NYB", interval="1d", start=start, end=end
            )
            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            if len(df) == 0:
                return
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            df = df.iloc[:-1] if len(df) > 0 else df
            if "Close" in df.columns:
                out = df[["timestamp", "Close"]].rename(
                    columns={"Close": "dxy_close_1d"}
                )
            elif "close" in df.columns:
                out = df[["timestamp", "close"]].rename(
                    columns={"close": "dxy_close_1d"}
                )
            elif "Adj Close" in df.columns:
                out = df[["timestamp", "Adj Close"]].rename(
                    columns={"Adj Close": "dxy_close_1d"}
                )
            else:
                return
            await self.registry.store_df(
                name="dxy_close_1d_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=70,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("_fetch_dxy_close_1d failed: %s", exc)

    async def _fetch_gold_close_15m(self) -> None:
        """Fetch Gold futures (GC=F) close at 15m via yfinance.

        Stores under 'gold_close_15m_df' with columns [timestamp, gold_close_15m].
        Applies df.iloc[:-1] to drop potentially incomplete last bar.
        """
        try:
            client = YFinanceClient(api_key=None, session=None)
            import pandas as _pd

            end = _pd.Timestamp.utcnow().floor("min")
            start = end - _pd.Timedelta(days=10)
            df = await client.fetch_data(
                symbol="GC=F", interval="15m", start=start, end=end
            )
            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            if len(df) == 0:
                return
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            df = df.iloc[:-1] if len(df) > 0 else df
            if "Close" in df.columns:
                out = df[["timestamp", "Close"]].rename(
                    columns={"Close": "gold_close_15m"}
                )
            elif "close" in df.columns:
                out = df[["timestamp", "close"]].rename(
                    columns={"close": "gold_close_15m"}
                )
            elif "Adj Close" in df.columns:
                out = df[["timestamp", "Adj Close"]].rename(
                    columns={"Adj Close": "gold_close_15m"}
                )
            else:
                return
            await self.registry.store_df(
                name="gold_close_15m_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=70,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("_fetch_gold_close_15m failed: %s", exc)

    async def _fetch_gold_close_1d(self) -> None:
        """Fetch Gold futures (GC=F) close at 1d via yfinance.

        Stores under 'gold_close_1d_df' with columns [timestamp, gold_close_1d].
        Applies df.iloc[:-1] to drop potentially incomplete last bar.
        """
        try:
            client = YFinanceClient(api_key=None, session=None)
            import pandas as _pd

            end = _pd.Timestamp.utcnow().floor("D")
            start = end - _pd.Timedelta(days=120)
            df = await client.fetch_data(
                symbol="GC=F", interval="1d", start=start, end=end
            )
            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            if len(df) == 0:
                return
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            df = df.iloc[:-1] if len(df) > 0 else df
            if "Close" in df.columns:
                out = df[["timestamp", "Close"]].rename(
                    columns={"Close": "gold_close_1d"}
                )
            elif "close" in df.columns:
                out = df[["timestamp", "close"]].rename(
                    columns={"close": "gold_close_1d"}
                )
            elif "Adj Close" in df.columns:
                out = df[["timestamp", "Adj Close"]].rename(
                    columns={"Adj Close": "gold_close_1d"}
                )
            else:
                return
            await self.registry.store_df(
                name="gold_close_1d_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=70,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("_fetch_gold_close_1d failed: %s", exc)

    async def _fetch_ust10y_yield_1d(self) -> None:
        """Fetch U.S. 10Y Treasury yield (^TNX) at 1d via yfinance.

        Stores under 'ust10y_yield_1d_df' with columns [timestamp, ust10y_yield_1d].
        Applies df.iloc[:-1] to drop potentially incomplete last bar.
        """
        try:
            client = YFinanceClient(api_key=None, session=None)
            import pandas as _pd

            end = _pd.Timestamp.utcnow().floor("D")
            start = end - _pd.Timedelta(days=365)
            df = await client.fetch_data(
                symbol="^TNX", interval="1d", start=start, end=end
            )
            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            if len(df) == 0:
                return
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            df = df.iloc[:-1] if len(df) > 0 else df
            if "Close" in df.columns:
                out = df[["timestamp", "Close"]].rename(
                    columns={"Close": "ust10y_yield_1d"}
                )
            elif "close" in df.columns:
                out = df[["timestamp", "close"]].rename(
                    columns={"close": "ust10y_yield_1d"}
                )
            elif "Adj Close" in df.columns:
                out = df[["timestamp", "Adj Close"]].rename(
                    columns={"Adj Close": "ust10y_yield_1d"}
                )
            else:
                return
            await self.registry.store_df(
                name="ust10y_yield_1d_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=31,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("_fetch_ust10y_yield_1d failed: %s", exc)

    async def _fetch_whale_transaction_volume_1m_usd_to_inf_15m(self) -> None:
        """Fetch Santiment whale_transaction_volume_1m_usd_to_inf at 15m cadence and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: whale_transaction_volume_1m_usd_to_inf
        - df_variable_id: {t}_whale_transaction_volume_1m_usd_to_inf_df
        - df_keys: timestamp, whale_transaction_volume_1m_usd_to_inf
        - df_update_mode: rolling_append
        - df_storage_period: 31
        """
        client = self.santiment_client or None
        if client is None:
            client = self._select_client_for_source(
                "santiment",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        token_slugs = {"btc": "bitcoin", "eth": "ethereum"}
        for t, slug in token_slugs.items():
            try:
                df = await client.fetch_data(
                    metric="whale_transaction_volume_1m_usd_to_inf",
                    slug=slug,
                    interval="15m",
                )
                import pandas as _pd

                if not isinstance(df, _pd.DataFrame):
                    df = _pd.DataFrame(df)
                # Normalize columns: value -> whale_transaction_volume_1m_usd_to_inf
                if (
                    "value" in df.columns
                    and "whale_transaction_volume_1m_usd_to_inf" not in df.columns
                ):
                    df = df.rename(
                        columns={"value": "whale_transaction_volume_1m_usd_to_inf"}
                    )
                await self.registry.store_df(
                    name=f"{t}_whale_transaction_volume_1m_usd_to_inf_df",
                    df=df,
                    update_mode="rolling_append",
                    storage_period=31,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning(
                    "_fetch_whale_transaction_volume_1m_usd_to_inf_15m failed for %s: %s",
                    t,
                    exc,
                )

    async def _fetch_funding_rate_30m(self) -> None:
        """Fetch CoinAPI derivatives funding rate (30m) and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: funding_rate_30m
        - df_variable_id: {t}_funding_rate_30m_df
        - df_keys: timestamp, last
        - df_update_mode: rolling_append
        - df_storage_period: 336
        - calculate_per_token: true
        - transform: rename 'time_period_start' -> 'timestamp' (handled by client), drop last (incomplete) row
        """
        client = self.coinapi_client or None
        if client is None:
            client = self._select_client_for_source(
                "coinapi",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return

        import pandas as _pd

        # Fetch a recent window larger than storage_period to ensure correct rolling truncation
        end = _pd.Timestamp.utcnow().floor("min")
        start = end - _pd.Timedelta(days=10)

        token_symbols = {
            "btc": "BINANCEFTS_PERP_BTC_USDT",
            "eth": "BINANCEFTS_PERP_ETH_USDT",
        }

        for t, symbol_id in token_symbols.items():
            try:
                df = await client.fetch_data(
                    endpoint="metrics",
                    metric_id="DERIVATIVES_FUNDING_RATE_CURRENT",
                    symbol_id=symbol_id,
                    period_id="30MIN",
                    time_start=start.isoformat(timespec="seconds") + "Z",
                    time_end=end.isoformat(timespec="seconds") + "Z",
                )
                if not isinstance(df, _pd.DataFrame):
                    df = _pd.DataFrame(df)
                if len(df) == 0:
                    empty = _pd.DataFrame(columns=["timestamp", "last"])
                    await self.registry.store_df(
                        name=f"{t}_funding_rate_30m_df",
                        df=empty,
                        update_mode="rolling_append",
                        storage_period=336,
                    )
                    continue
                if "timestamp" in df.columns:
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
                elif "time_period_start" in df.columns:
                    df = df.rename(columns={"time_period_start": "timestamp"})
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
                # Drop last row to avoid incomplete bar
                if len(df) > 0:
                    df = df.iloc[:-1]
                # Keep only required columns; handle value->last if present
                if "last" not in df.columns and "value" in df.columns:
                    df = df.rename(columns={"value": "last"})
                if "timestamp" in df.columns and "last" in df.columns:
                    out = df[["timestamp", "last"]].reset_index(drop=True)
                else:
                    cols = [c for c in df.columns if c in ("timestamp", "last")]
                    out = df[cols]
                await self.registry.store_df(
                    name=f"{t}_funding_rate_30m_df",
                    df=out,
                    update_mode="rolling_append",
                    storage_period=336,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning("_fetch_funding_rate_30m failed for %s: %s", t, exc)

    async def _fetch_eth_supply_usd_15m(self) -> None:
        """Fetch ETH circulating supply at 15m cadence via CoinMarketCap and store.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: eth_supply_usd_15m
        - df_variable_id: eth_supply_usd_15m_df
        - df_keys: timestamp, eth_supply_15m
        - df_update_mode: rolling_append
        - df_storage_period: 2
        - note: drop the last row from raw ingestion to avoid incomplete data
        """
        client = self.coinmarketcap_client or None
        if client is None:
            client = self._select_client_for_source(
                "coinmarketcap",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            # Request ETH quotes at 15m interval; include aux if supported
            df = await client.fetch_data(
                endpoint="quotes",
                symbol="ETH",
                interval="15m",
                convert="USD",
                aux="circulating_supply",
            )
            import pandas as _pd

            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            # Ensure timestamp is UTC
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            # Drop last row to avoid incomplete bar
            if len(df) > 0:
                df = df.iloc[:-1]
            # Prefer explicit circulating_supply column
            if "circulating_supply" in df.columns:
                out = df[["timestamp", "circulating_supply"]].rename(
                    columns={"circulating_supply": "eth_supply_15m"}
                )
            else:
                # Fallback: if market_cap and price available, estimate supply
                if "market_cap" in df.columns and "price_usd" in df.columns:
                    df["eth_supply_15m"] = _pd.to_numeric(
                        df["market_cap"], errors="coerce"
                    ) / (_pd.to_numeric(df["price_usd"], errors="coerce") + 1e-12)
                    df["eth_supply_15m"] = df["eth_supply_15m"].replace(
                        [np.inf, -np.inf], np.nan
                    )
                    out = df[["timestamp", "eth_supply_15m"]]
                else:
                    # If not available, store empty frame with correct schema
                    out = _pd.DataFrame(columns=["timestamp", "eth_supply_15m"])
            await self.registry.store_df(
                name="eth_supply_usd_15m_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=2,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_eth_supply_usd_15m failed: %s", exc)

    async def _fetch_btc_supply_usd_1d(self) -> None:
        """Fetch BTC circulating supply at 1d cadence via CoinMarketCap and store.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: btc_supply_usd_1d
        - df_variable_id: btc_circ_supply_1d_df
        - df_keys: timestamp, btc_supply_1d
        - df_update_mode: rolling_append
        - df_storage_period: 11
        - note: drop the last row from raw ingestion to avoid incomplete data
        """
        client = self.coinmarketcap_client or None
        if client is None:
            client = self._select_client_for_source(
                "coinmarketcap",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            df = await client.fetch_data(
                endpoint="quotes",
                symbol="BTC",
                interval="1d",
                convert="USD",
                aux="circulating_supply,market_cap",
            )
            import pandas as _pd

            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            # Ensure timestamp is UTC
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            # Drop last row to avoid incomplete daily bar
            if len(df) > 0:
                df = df.iloc[:-1]
            # Prefer explicit circulating_supply
            if "circulating_supply" in df.columns:
                out = df[["timestamp", "circulating_supply"]].rename(
                    columns={"circulating_supply": "btc_supply_1d"}
                )
            else:
                # Fallback: estimate supply from market_cap and price if present
                if "market_cap" in df.columns and "price_usd" in df.columns:
                    df["btc_supply_1d"] = _pd.to_numeric(
                        df["market_cap"], errors="coerce"
                    ) / (_pd.to_numeric(df["price_usd"], errors="coerce") + 1e-12)
                    df["btc_supply_1d"] = df["btc_supply_1d"].replace(
                        [np.inf, -np.inf], np.nan
                    )
                    out = df[["timestamp", "btc_supply_1d"]]
                else:
                    out = _pd.DataFrame(columns=["timestamp", "btc_supply_1d"])
            await self.registry.store_df(
                name="btc_circ_supply_1d_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=11,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_btc_supply_usd_1d failed: %s", exc)

    async def _fetch_btcd_close_1d(self) -> None:
        """Fetch BTC Dominance (1d) via CoinMarketCap Global Metrics and store.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: btcd_close_1d
        - df_variable_id: btcd_close_1d_df
        - df_keys: timestamp, btc_dominance
        - df_update_mode: rolling_append
        - df_storage_period: 20
        - note: drop the last row from raw ingestion to avoid incomplete data
        """
        client = self.coinmarketcap_client or None
        if client is None:
            client = self._select_client_for_source(
                "coinmarketcap",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            df = await client.fetch_data(
                endpoint="global_metrics",
                interval="1d",
            )
            import pandas as _pd

            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            if len(df) == 0:
                out = _pd.DataFrame(columns=["timestamp", "btc_dominance"])
            else:
                # Ensure timestamp is UTC
                if "timestamp" in df.columns:
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
                # Drop last row
                if len(df) > 0:
                    df = df.iloc[:-1]
                # Determine dominance column
                if "btc_dominance" in df.columns:
                    out = df[["timestamp", "btc_dominance"]]
                elif "usd_btc_dominance" in df.columns:
                    out = df[["timestamp", "usd_btc_dominance"]].rename(
                        columns={"usd_btc_dominance": "btc_dominance"}
                    )
                else:
                    # If global metrics places it under a generic 'value', use that
                    if "value" in df.columns:
                        out = df[["timestamp", "value"]].rename(
                            columns={"value": "btc_dominance"}
                        )
                    else:
                        out = _pd.DataFrame(columns=["timestamp", "btc_dominance"])
            await self.registry.store_df(
                name="btcd_close_1d_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=20,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_btcd_close_1d failed: %s", exc)

    async def _fetch_btc_mcap_1d(self) -> None:
        """Fetch BTC Market Cap (1d) via CoinMarketCap Quotes Historical and store.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: btc_mcap_1d
        - df_variable_id: btc_mcap_1d_df
        - df_keys: timestamp, btc_mcap_1d
        - df_update_mode: rolling_append
        - df_storage_period: 31
        - note: drop the last row from raw ingestion to avoid incomplete data
        """
        client = self.coinmarketcap_client or None
        if client is None:
            client = self._select_client_for_source(
                "coinmarketcap",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            df = await client.fetch_data(
                endpoint="quotes",
                symbol="BTC",
                interval="1d",
                convert="USD",
                aux="market_cap",
            )
            import pandas as _pd

            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            # Ensure timestamp is UTC
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            # Drop last row to avoid incomplete daily bar
            if len(df) > 0:
                df = df.iloc[:-1]
            # Extract market_cap -> btc_mcap_1d; handle type coercion
            if "market_cap" in df.columns:
                df["btc_mcap_1d"] = _pd.to_numeric(df["market_cap"], errors="coerce")
                df["btc_mcap_1d"] = df["btc_mcap_1d"].replace([np.inf, -np.inf], np.nan)
                out = df[["timestamp", "btc_mcap_1d"]]
            else:
                # If market_cap not present, but price and supply exist, compute
                if {"price_usd", "circulating_supply"}.issubset(df.columns):
                    df["btc_mcap_1d"] = _pd.to_numeric(
                        df["price_usd"], errors="coerce"
                    ) * _pd.to_numeric(df["circulating_supply"], errors="coerce")
                    df["btc_mcap_1d"] = df["btc_mcap_1d"].replace(
                        [np.inf, -np.inf], np.nan
                    )
                    out = df[["timestamp", "btc_mcap_1d"]]
                else:
                    out = _pd.DataFrame(columns=["timestamp", "btc_mcap_1d"])
            await self.registry.store_df(
                name="btc_mcap_1d_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=31,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_btc_mcap_1d failed: %s", exc)

    async def _fetch_crypto_fng_index_1d(self) -> None:
        """Fetch Crypto Fear & Greed Index (1d) via CoinMarketCap and store.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: crypto_fng_index_1d
        - df_variable_id: crypto_fng_1d_df
        - df_keys: timestamp, crypto_fng_1d
        - df_update_mode: rolling_append
        - df_storage_period: 108
        - note: drop the last row from raw ingestion to avoid incomplete data
        """
        client = self.coinmarketcap_client or None
        if client is None:
            client = self._select_client_for_source(
                "coinmarketcap",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            df = await client.fetch_data(endpoint="fear_greed", start=1, limit=500)
            import pandas as _pd

            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            if len(df) == 0:
                out = _pd.DataFrame(columns=["timestamp", "crypto_fng_1d"])
            else:
                # Ensure timestamp is UTC
                if "timestamp" in df.columns:
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
                # Drop last row to avoid incomplete latest day
                if len(df) > 0:
                    df = df.iloc[:-1]
                # Map 'value' to 'crypto_fng_1d'; coerce to int32 per spec
                if "value" in df.columns:
                    df["crypto_fng_1d"] = _pd.to_numeric(
                        df["value"], errors="coerce"
                    ).round()
                    # Replace inf with NaN, then cast if possible
                    df["crypto_fng_1d"] = df["crypto_fng_1d"].replace(
                        [np.inf, -np.inf], np.nan
                    )
                    try:
                        df["crypto_fng_1d"] = df["crypto_fng_1d"].astype("Int64")
                    except Exception:
                        pass
                    out = df[["timestamp", "crypto_fng_1d"]].copy()
                    # Convert nullable Int64 to plain int32 where possible
                    try:
                        out["crypto_fng_1d"] = out["crypto_fng_1d"].astype("int32")
                    except Exception:
                        # Keep nullable ints if coercion fails due to NaNs
                        pass
                else:
                    out = _pd.DataFrame(columns=["timestamp", "crypto_fng_1d"])
            await self.registry.store_df(
                name="crypto_fng_1d_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=108,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_crypto_fng_index_1d failed: %s", exc)

    async def _fetch_eth_mcap_1d(self) -> None:
        """Fetch ETH Market Cap (1d) via CoinMarketCap Quotes Historical and store.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: eth_mcap_1d
        - df_variable_id: eth_mcap_1d_df
        - df_keys: timestamp, eth_mcap_1d
        - df_update_mode: rolling_append
        - df_storage_period: 31
        - note: drop the last row from raw ingestion to avoid incomplete data
        """
        client = self.coinmarketcap_client or None
        if client is None:
            client = self._select_client_for_source(
                "coinmarketcap",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return
        try:
            df = await client.fetch_data(
                endpoint="quotes",
                symbol="ETH",
                interval="1d",
                convert="USD",
                aux="market_cap",
            )
            import pandas as _pd

            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            # Ensure timestamp is UTC
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            # Drop last row to avoid incomplete daily bar
            if len(df) > 0:
                df = df.iloc[:-1]
            # Extract market_cap -> eth_mcap_1d; handle type coercion
            if "market_cap" in df.columns:
                df["eth_mcap_1d"] = _pd.to_numeric(df["market_cap"], errors="coerce")
                df["eth_mcap_1d"] = df["eth_mcap_1d"].replace([np.inf, -np.inf], np.nan)
                out = df[["timestamp", "eth_mcap_1d"]]
            else:
                # If market_cap not present, but price and supply exist, compute
                if {"price_usd", "circulating_supply"}.issubset(df.columns):
                    df["eth_mcap_1d"] = _pd.to_numeric(
                        df["price_usd"], errors="coerce"
                    ) * _pd.to_numeric(df["circulating_supply"], errors="coerce")
                    df["eth_mcap_1d"] = df["eth_mcap_1d"].replace(
                        [np.inf, -np.inf], np.nan
                    )
                    out = df[["timestamp", "eth_mcap_1d"]]
                else:
                    out = _pd.DataFrame(columns=["timestamp", "eth_mcap_1d"])
            await self.registry.store_df(
                name="eth_mcap_1d_df",
                df=out.reset_index(drop=True),
                update_mode="rolling_append",
                storage_period=31,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_eth_mcap_1d failed: %s", exc)

    async def _fetch_futures_mark_price_15m(self) -> None:
        """Fetch CoinAPI futures mark price (15m) and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: futures_mark_price_15m
        - df_variable_id: {t}_futures_mark_price_15m_df
        - df_keys: timestamp, last
        - df_update_mode: rolling_append
        - df_storage_period: 672
        - calculate_per_token: true
        - transform: rename 'time_period_start' -> 'timestamp' (handled by client), drop last (incomplete) row
        """
        client = self.coinapi_client or None
        if client is None:
            client = self._select_client_for_source(
                "coinapi",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return

        import pandas as _pd

        # Fetch recent window greater than storage_period to ensure rolling correctness
        end = _pd.Timestamp.utcnow().floor("min")
        start = end - _pd.Timedelta(days=10)

        # Map tokens to futures symbol ids (Binance perpetuals)
        token_symbols = {
            "btc": "BINANCEFTS_PERP_BTC_USDT",
            "eth": "BINANCEFTS_PERP_ETH_USDT",
        }

        for t, symbol_id in token_symbols.items():
            try:
                df = await client.fetch_data(
                    endpoint="metrics",
                    metric_id="DERIVATIVES_MARK_PRICE",
                    symbol_id=symbol_id,
                    period_id="15MIN",
                    time_start=start.isoformat(timespec="seconds") + "Z",
                    time_end=end.isoformat(timespec="seconds") + "Z",
                )
                if not isinstance(df, _pd.DataFrame):
                    df = _pd.DataFrame(df)
                if len(df) == 0:
                    # Store empty frame with correct columns
                    empty = _pd.DataFrame(columns=["timestamp", "last"])
                    await self.registry.store_df(
                        name=f"{t}_futures_mark_price_15m_df",
                        df=empty,
                        update_mode="rolling_append",
                        storage_period=672,
                    )
                    continue
                # Ensure timestamp exists and is UTC (Base client already attempts this)
                if "timestamp" in df.columns:
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
                # Drop the last row to avoid incomplete bar
                if len(df) > 0:
                    df = df.iloc[:-1]
                # Keep only required columns; if value column present instead of 'last', rename
                if "last" not in df.columns and "value" in df.columns:
                    df = df.rename(columns={"value": "last"})
                if "timestamp" in df.columns and "last" in df.columns:
                    out = df[["timestamp", "last"]].reset_index(drop=True)
                else:
                    # If schema unexpected, try best-effort
                    cols = [c for c in df.columns if c in ("timestamp", "last")]
                    out = df[cols]
                await self.registry.store_df(
                    name=f"{t}_futures_mark_price_15m_df",
                    df=out,
                    update_mode="rolling_append",
                    storage_period=672,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning(
                    "_fetch_futures_mark_price_15m failed for %s: %s", t, exc
                )

    async def _fetch_perp_close_1h(self) -> None:
        """Fetch per-token perpetual futures OHLCV (1h) close via CoinAPI and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: perp_close_1h
        - df_variable_id: {t}_perp_close_1h_df
        - df_keys: timestamp, perpClose_1h
        - df_update_mode: rolling_append
        - df_storage_period: 1
        - transform: parse 'price_close' -> 'perpClose_1h'; drop last (incomplete) row
        """
        client = self.coinapi_client or None
        if client is None:
            client = self._select_client_for_source(
                "coinapi",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return

        import pandas as _pd

        # Fetch a window that comfortably covers recent hourly bars
        end = _pd.Timestamp.utcnow().floor("h")
        start = end - _pd.Timedelta(days=14)

        # Map tokens to futures symbol ids (Binance perpetuals)
        token_symbols = {
            "btc": "BINANCEFTS_PERP_BTC_USDT",
            "eth": "BINANCEFTS_PERP_ETH_USDT",
        }

        for t, symbol_id in token_symbols.items():
            try:
                df = await client.fetch_data(
                    endpoint="ohlcv",
                    symbol_id=symbol_id,
                    period_id="1HRS",
                    time_start=start.isoformat(timespec="seconds") + "Z",
                    time_end=end.isoformat(timespec="seconds") + "Z",
                )
                if not isinstance(df, _pd.DataFrame):
                    df = _pd.DataFrame(df)
                if len(df) == 0:
                    empty = _pd.DataFrame(columns=["timestamp", "perpClose_1h"])
                    await self.registry.store_df(
                        name=f"{t}_perp_close_1h_df",
                        df=empty,
                        update_mode="rolling_append",
                        storage_period=1,
                    )
                    continue

                # Normalize timestamp column to UTC
                if "timestamp" in df.columns:
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
                elif "time_period_start" in df.columns:
                    df = df.rename(columns={"time_period_start": "timestamp"})
                    df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)

                # Drop last row to avoid incomplete bar
                if len(df) > 0:
                    df = df.iloc[:-1]

                # Map price_close to perpClose_1h
                if "perpClose_1h" not in df.columns and "price_close" in df.columns:
                    df = df.rename(columns={"price_close": "perpClose_1h"})

                # Keep only required columns
                cols = [c for c in ("timestamp", "perpClose_1h") if c in df.columns]
                out = df[cols].reset_index(drop=True)

                await self.registry.store_df(
                    name=f"{t}_perp_close_1h_df",
                    df=out,
                    update_mode="rolling_append",
                    storage_period=1,
                )
            except Exception as exc:  # pragma: no cover - robustness per job
                LOGGER.warning("_fetch_perp_close_1h failed for %s: %s", t, exc)

    async def _fetch_btc_perp_close_7d(self) -> None:
        """Fetch BTC perpetual futures OHLCV (7d) close via CoinAPI and store per spec.

        Spec mapping (unicorn_wealth_feature_set.json):
        - operation: btc_perp_close_7d
        - df_variable_id: btc_perp_close_7d_df
        - df_keys: timestamp, btc_perpClose_7d
        - df_update_mode: rolling_append
        - df_storage_period: 2
        - transform: parse 'price_close' -> 'btc_perpClose_7d'; drop last (incomplete) row
        """
        client = self.coinapi_client or None
        if client is None:
            client = self._select_client_for_source(
                "coinapi",
                {c.__class__.__name__.lower(): c for c in self.historical_api_clients},
            )
        if client is None:
            return

        import pandas as _pd

        # Fetch a recent window that ensures at least a few weekly bars
        end = _pd.Timestamp.utcnow().floor("D")
        start = end - _pd.Timedelta(days=120)

        symbol_id = "BINANCEFTS_PERP_BTC_USDT"
        try:
            df = await client.fetch_data(
                endpoint="ohlcv",
                symbol_id=symbol_id,
                period_id="7DAY",
                time_start=start.isoformat(timespec="seconds") + "Z",
                time_end=end.isoformat(timespec="seconds") + "Z",
            )
            if not isinstance(df, _pd.DataFrame):
                df = _pd.DataFrame(df)
            if len(df) == 0:
                empty = _pd.DataFrame(columns=["timestamp", "btc_perpClose_7d"])
                await self.registry.store_df(
                    name="btc_perp_close_7d_df",
                    df=empty,
                    update_mode="rolling_append",
                    storage_period=2,
                )
                return

            # Normalize timestamp column
            if "timestamp" in df.columns:
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
            elif "time_period_start" in df.columns:
                df = df.rename(columns={"time_period_start": "timestamp"})
                df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)

            # Drop last row to avoid incomplete bar
            if len(df) > 0:
                df = df.iloc[:-1]

            # Map price_close to btc_perpClose_7d
            if "btc_perpClose_7d" not in df.columns and "price_close" in df.columns:
                df = df.rename(columns={"price_close": "btc_perpClose_7d"})

            # Keep only required columns
            cols = [c for c in ("timestamp", "btc_perpClose_7d") if c in df.columns]
            out = df[cols].reset_index(drop=True)

            await self.registry.store_df(
                name="btc_perp_close_7d_df",
                df=out,
                update_mode="rolling_append",
                storage_period=2,
            )
        except Exception as exc:  # pragma: no cover - robustness per job
            LOGGER.warning("_fetch_btc_perp_close_7d failed: %s", exc)
