from __future__ import annotations

import asyncio
import logging
import sys
import csv
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import typer
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import API_FETCH_DELAY_SECONDS
from core.config_loader import load_settings
from core.dataframe_registry import DataFrameRegistry
from data_ingestion.websocket.coinapi_ws_client import CoinApiWebSocketClient
from execution.notifiers.telegram_notifier import TelegramNotifier
from execution.risk_management import RiskManager
from execution.trade_executor import TradeExecutionEngine
from execution.consensus_engine import ConsensusEngine
from execution.position_manager import PositionManager
from ml_lifecycle.inference import InferenceEngine
from ml_lifecycle.live_calculator import LiveFeatureCalculator
from features.feature_engine import UnifiedFeatureEngine
from database.models.base import get_async_engine
from reporting.performance_reporter import PerformanceReporter
from ml_lifecycle.performance_analyzer import PerformanceAnalyzer


async def run_live_15m_cycle(
    live_data_manager: Any,
    a2: Any,
    a3: Any,
    a4: Any,
    a5: Any,
    a6: Any | None = None,
) -> Dict[str, Any]:
    """Run a single 15-minute live trading cycle.

    Orchestration:
    1) Fetch/resample data.
    2) At time boundaries, compute higher-cadence features (1d -> 4h -> 1h) and store to the registry.
    3) Run 15m-feature assembly and model inference.
    4) Generate trading directive and process positions.
    """
    # Support both legacy (5-arg) and new (6-arg) call signatures
    if a6 is None:

        class _NoOpFE:
            async def run_1d_pipeline(self) -> None:
                return None

            async def run_4h_pipeline(self) -> None:
                return None

            async def run_1h_pipeline(self) -> None:
                return None

        feature_engine = _NoOpFE()
        live_feature_calculator = a2
        inference_engine = a3
        consensus_engine = a4
        position_manager = a5
    else:
        feature_engine = a2
        live_feature_calculator = a3
        inference_engine = a4
        consensus_engine = a5
        position_manager = a6

    # 1) Fetch scheduled data (gap fill, resampling, etc.)
    await live_data_manager.fetch_scheduled_data()

    # 2) Time-aware multi-timeframe feature pipelines (longest -> shortest)
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    # 1d boundary: midnight UTC
    if now.hour == 0 and now.minute == 0:
        try:
            await feature_engine.run_1d_pipeline()
        except Exception as e:  # noqa: BLE001
            logging.getLogger("uw.main").warning("run_1d_pipeline failed: %s", e)
    # 4h boundary: top of hour where hour % 4 == 0
    if now.minute == 0 and now.hour % 4 == 0:
        try:
            await feature_engine.run_4h_pipeline()
        except Exception as e:  # noqa: BLE001
            logging.getLogger("uw.main").warning("run_4h_pipeline failed: %s", e)
    # 1h boundary: top of the hour
    if now.minute == 0:
        try:
            await feature_engine.run_1h_pipeline()
        except Exception as e:  # noqa: BLE001
            logging.getLogger("uw.main").warning("run_1h_pipeline failed: %s", e)

    # 3) Compute features for current 15m cycle and assemble horizon subsets
    from config import MASTER_TOKEN_LIST

    try:
        feature_rows: Dict[str, Any] = (
            await live_feature_calculator.calculate_all_horizons(
                tokens=list(MASTER_TOKEN_LIST.keys())
            )
        )
    except TypeError:
        # Backward-compatible fallback for calculators without 'tokens' parameter (e.g., tests)
        feature_rows = await live_feature_calculator.calculate_all_horizons()

    # 4) Run inference for all horizons
    predictions: Dict[str, Any] = await inference_engine.predict(feature_rows)

    # 5) Build a trading directive from predictions (could be HOLD)
    directive: Any = await consensus_engine.generate_directive(predictions)

    # 6) Let the position manager process the directive (place/close/hold)
    await position_manager.process_directive(directive)

    return predictions


app = typer.Typer()


async def inspect_registry(
    token: str = "BTC", days_back: int = 5, output_path: str = "registry_report.csv"
) -> None:
    """Inspect the in-memory DataFrameRegistry by performing a small, targeted
    data load for a single token, then export a human-readable CSV snapshot.

    Notes:
    - This function populates the in-memory registry only. It does not write to DB.
    - It attempts a best-effort minimal ingestion across a few providers to ensure
      the registry is non-empty for inspection. All external calls are wrapped in
      try/except to keep the report generation robust.
    """
    print(f"Starting registry inspection for token={token}, days_back={days_back}…")

    import aiohttp
    import httpx
    import pandas as pd
    from datetime import datetime, timedelta, timezone

    from data_ingestion.api import (
        CoinApiClient,
        CoinMarketCapClient,
        YFinanceClient,
        FinnhubClient,
    )

    # 1) Initialization
    settings = load_settings()
    registry = DataFrameRegistry()

    # 2) Data Loading (limited, targeted; in-memory only)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(days_back))

    # Build ISO strings for providers that require them
    start_iso = pd.Timestamp(start_dt).isoformat().replace("+00:00", "Z")
    end_iso = pd.Timestamp(end_dt).isoformat().replace("+00:00", "Z")

    # Create sessions and clients
    async with aiohttp.ClientSession() as aiohttp_session, httpx.AsyncClient() as httpx_session:
        coinapi_key = getattr(settings, "coinapi_api_key", "") or ""
        cmc_key = getattr(settings, "coinmarketcap_api_key", "") or ""
        finnhub_key = getattr(settings, "finnhub_api_key", "") or ""
        san_key = getattr(settings, "santiment_api_key", None)

        coinapi_client = (
            CoinApiClient(api_key=coinapi_key, session=aiohttp_session)
            if coinapi_key
            else None
        )
        cmc_client = (
            CoinMarketCapClient(api_key=cmc_key, session=aiohttp_session)
            if cmc_key
            else None
        )
        yfin_client = YFinanceClient(api_key=None, session=aiohttp_session)
        finnhub_client = (
            FinnhubClient(api_key=finnhub_key, session=aiohttp_session)
            if finnhub_key
            else None
        )

        # Santiment client uses httpx and its own small Settings dataclass
        try:
            from data_ingestion.api.santiment_client import (
                Settings as SanSettings,
                SantimentClient as _SanClient,
            )

            san_client = (
                _SanClient(
                    session=httpx_session,
                    settings=SanSettings(SANTIMENT_API_KEY=san_key),
                )
                if san_key
                else None
            )
        except Exception:  # fallback (should not happen since we import above)
            san_client = None

        t_sym = str(token).upper()
        t_l = t_sym.lower()

        # 2.a) YFinance daily candles (robust, no API key)
        try:
            yf_symbol = f"{t_sym}-USD"
            df_yf = await yfin_client.fetch_data(
                symbol=yf_symbol, interval="1d", start=start_dt, end=end_dt
            )
            if isinstance(df_yf, pd.DataFrame) and len(df_yf) > 0:
                await registry.store_df(
                    name=f"{t_l}_yfinance_1d_df",
                    df=df_yf,
                    update_mode="overwrite",
                    storage_period=0,
                )
        except Exception as e:
            print(f"[warn] yfinance fetch failed: {e}")

        # 2.b) Santiment active_addresses_24h daily for tokens with configured slug
        try:
            from config import MASTER_TOKEN_LIST

            slug = MASTER_TOKEN_LIST.get(t_sym, {}).get("santiment_slug")
            if san_client is not None and slug:
                df_san = await san_client.fetch_data(
                    metric="active_addresses_24h",
                    slug=slug,
                    interval="1d",
                    start_date=start_dt,
                    end_date=end_dt,
                )
                if isinstance(df_san, pd.DataFrame):
                    if (
                        "value" in df_san.columns
                        and "active_addresses_24h" not in df_san.columns
                    ):
                        df_san = df_san.rename(
                            columns={"value": "active_addresses_24h"}
                        )
                    await registry.store_df(
                        name=f"{t_l}_active_addresses_24h_df",
                        df=df_san,
                        update_mode="overwrite",
                        storage_period=0,
                    )
        except Exception as e:
            print(f"[warn] santiment fetch failed: {str(e).split(':')[0]}")

        # 2.c) CoinAPI OHLCV 15m for BINANCE_SPOT_<TOKEN>_USDT (best-effort)
        try:
            if coinapi_client is not None:
                from config import MASTER_TOKEN_LIST

                symbol_id = (
                    MASTER_TOKEN_LIST.get(t_sym, {}).get("coinapi_symbol_id")
                    or f"BINANCE_SPOT_{t_sym}_USDT"
                )
                df_ohlcv = await coinapi_client.fetch_data(
                    endpoint="ohlcv",
                    symbol_id=symbol_id,
                    period_id="15MIN",
                    time_start=start_iso,
                    time_end=end_iso,
                    limit=100000,
                )
                if isinstance(df_ohlcv, pd.DataFrame) and len(df_ohlcv) > 0:
                    # Normalize common OHLCV columns to standard names if needed
                    rename_map = {
                        "price_open": "open",
                        "price_high": "high",
                        "price_low": "low",
                        "price_close": "close",
                        "volume_traded": "volume",
                    }
                    cols_to_rename = {
                        k: v for k, v in rename_map.items() if k in df_ohlcv.columns
                    }
                    if cols_to_rename:
                        df_ohlcv = df_ohlcv.rename(columns=cols_to_rename)
                    await registry.store_df(
                        name=f"{t_l}_ohlcv_15m_df",
                        df=df_ohlcv,
                        update_mode="overwrite",
                        storage_period=0,
                    )
        except Exception as e:
            print(f"[warn] coinapi OHLCV fetch failed: {e}")

        # 2.d) CoinMarketCap fear & greed (global), constrained count
        try:
            if cmc_client is not None:
                # Request a small recent window; the API uses 'start' and 'limit'
                df_fng = await cmc_client.fetch_data(
                    endpoint="fear_greed", start=0, limit=days_back * 2
                )
                if isinstance(df_fng, pd.DataFrame) and len(df_fng) > 0:
                    await registry.store_df(
                        name="global_fear_greed_df",
                        df=df_fng,
                        update_mode="overwrite",
                        storage_period=0,
                    )
        except Exception as e:
            print(
                f"[warn] coinmarketcap fear_greed fetch failed: {str(e).split(':')[0]}"
            )

        # 2.e) Finnhub economic calendar between dates
        try:
            if finnhub_client is not None:
                df_fin = await finnhub_client.fetch_data(
                    **{
                        "from": start_dt.date().isoformat(),
                        "to": end_dt.date().isoformat(),
                    }
                )
                if isinstance(df_fin, pd.DataFrame) and len(df_fin) > 0:
                    await registry.store_df(
                        name="finnhub_econ_calendar_df",
                        df=df_fin,
                        update_mode="overwrite",
                        storage_period=0,
                    )
        except Exception as e:
            print(f"[warn] finnhub econ calendar fetch failed: {str(e).split(':')[0]}")

    # 3) Report Preparation
    csv_rows: list[list[Any]] = []
    names = await registry.list_dfs()
    names = sorted(names)

    if not names:
        print(
            "Warning: DataFrameRegistry is empty after limited load. No report generated."
        )
        return

    # 4) Data Aggregation Loop
    import pandas as pd  # local alias to ensure typing

    for name in names:
        try:
            df = await registry.get_df(name)
        except KeyError:
            continue
        # Header row
        shape_str = f"{getattr(df, 'shape', ('?', '?'))}"
        csv_rows.append([f"DataFrame: {name}", "", "Shape:", shape_str])
        # Columns row
        try:
            cols = df.columns.tolist()
        except Exception:
            cols = []
        csv_rows.append(["Columns ->"] + cols)
        # Sample rows (head 3)
        try:
            head_df = df.head(3)
            for idx in range(len(head_df)):
                row = head_df.iloc[idx].tolist() if hasattr(head_df, "iloc") else []
                csv_rows.append([f"Head {idx + 1} ->"] + row)
        except Exception:
            pass
        # Separator
        csv_rows.append([])

    # 5) CSV Export
    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Registry report successfully saved to {output_path}")


async def _fetch_and_store_for_token(
    token: str,
    settings: Any,
    clients: Dict[str, Any],
    storage_engine: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """Intelligently sync historical OHLCV for a token within the lookback window.

    New workflow:
    1) Inspect DB for existing min/max timestamp per timeframe in raw_ohlcv_[tf] table.
    2) Determine missing ranges vs requested [start_time, end_time] window.
       - Handle empty table: fetch the full window.
       - Back-fill (window_start -> min_ts-1) and forward-fill (max_ts+1 -> window_end).
    3) Fetch each missing range from CoinAPI and upsert via RawDataSQLEngine in batches.
    4) Prune rows older than the window start for the token.
    """
    import pandas as pd
    from sqlalchemy import text

    log = logging.getLogger("uw.run_historical.token")

    # Map internal timeframe labels to CoinAPI period_id values and step minutes
    timeframe_to_period = {
        "15m": ("15MIN", 15),
        "1h": ("1HRS", 60),
        "4h": ("4HRS", 240),
        "1d": ("1DAY", 1440),
        "7d": ("7DAY", 10080),
    }

    coinapi_client = clients.get("coinapi")
    if coinapi_client is None:
        raise RuntimeError("coinapi client not provided in clients dictionary")

    # Resolve CoinAPI symbol ID from config.MASTER_TOKEN_LIST if available
    from config import MASTER_TOKEN_LIST

    symbol_id = MASTER_TOKEN_LIST.get(token, {}).get("coinapi_symbol_id")
    if not symbol_id:
        raise ValueError(
            f"Missing coinapi_symbol_id for token '{token}' in MASTER_TOKEN_LIST"
        )

    # Ensure window bounds are tz-aware UTC pandas Timestamps
    window_end = pd.to_datetime(end_time, utc=True)
    window_start = pd.to_datetime(start_time, utc=True)

    # Helper to normalize OHLCV DataFrame to raw_ohlcv schema
    def _normalize_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "token",
                    "interval",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )
        rename_map = {
            "price_open": "open",
            "price_high": "high",
            "price_low": "low",
            "price_close": "close",
            "volume_traded": "volume",
            "time_period_start": "timestamp",
            "time": "timestamp",
        }
        df = df.rename(columns=rename_map).copy()
        # Ensure timestamp to epoch milliseconds int
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df[df["timestamp"].notna()]
            df["timestamp"] = (df["timestamp"].astype("int64") // 10**6).astype("int64")
        # Insert token and interval
        df["token"] = token
        df["interval"] = interval
        # Keep only expected columns and order
        cols = [
            "timestamp",
            "token",
            "interval",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        df = df[[c for c in cols if c in df.columns]]
        # Sort by timestamp ascending to be tidy
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp", ascending=True)
        return df

    # Iterate each timeframe independently
    engine = getattr(storage_engine, "_engine", None)
    if engine is None:
        raise RuntimeError("storage_engine does not expose an underlying async engine")

    for interval, (period_id, step_min) in timeframe_to_period.items():
        table_name = f"raw_ohlcv_{interval}"
        step_ms = int(step_min * 60 * 1000)

        # 1) Query DB for existing min/max within the table for this token
        min_ts: int | None = None
        max_ts: int | None = None
        async with engine.begin() as conn:
            res = await conn.execute(
                text(
                    f"""
                    SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts
                    FROM {table_name}
                    WHERE token = :token
                    """
                ),
                {"token": token},
            )
            row = res.first()
            if row is not None:
                # Row can return a RowMapping or tuple depending on dialect
                min_ts = row[0]
                max_ts = row[1]

        # Compute desired window in epoch ms
        win_start_ms = int(window_start.value // 10**6)
        win_end_ms = int(window_end.value // 10**6)

        # 2) Determine gaps vs [win_start_ms, win_end_ms]
        gaps: list[tuple[int, int]] = []
        if min_ts is None or max_ts is None:
            # Empty table for this token -> fetch full window
            gaps.append((win_start_ms, win_end_ms))
        else:
            # Back-fill gap
            if min_ts > win_start_ms:
                # Align end to just before current min
                gap_end = max(win_start_ms, min_ts - step_ms)
                gaps.append((win_start_ms, gap_end))
            # Forward-fill gap
            if max_ts < win_end_ms:
                gap_start = min(win_end_ms, max_ts + step_ms)
                gaps.append((gap_start, win_end_ms))
            # Note: internal gaps not computed to keep logic minimal and efficient.

        if not gaps:
            log.info(
                "[%s][%s] No gaps detected within lookback window.", token, interval
            )
        else:
            log.info(
                "[%s][%s] Identified %d gap(s): %s", token, interval, len(gaps), gaps
            )

        # 3) Fetch and upsert only missing ranges
        for gap_start_ms, gap_end_ms in gaps:
            if gap_start_ms > gap_end_ms:
                continue  # skip invalid
            # Convert to ISO8601 for CoinAPI
            gap_start_iso = pd.to_datetime(gap_start_ms, unit="ms", utc=True).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            gap_end_iso = pd.to_datetime(gap_end_ms, unit="ms", utc=True).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            try:
                df = await coinapi_client.fetch_data(
                    endpoint="ohlcv",
                    symbol_id=symbol_id,
                    period_id=period_id,
                    time_start=gap_start_iso,
                    time_end=gap_end_iso,
                )
                df_norm = _normalize_ohlcv(df, interval)
                if df_norm is not None and not df_norm.empty:
                    # Remove 'interval' before saving; table implied by interval param
                    await storage_engine.save_ohlcv_data(
                        df=df_norm.drop(
                            columns=[c for c in ["interval"] if c in df_norm.columns]
                        ),
                        interval=interval,
                    )
                log.info(
                    "[%s] Saved OHLCV %s rows=%s for gap %s -> %s",
                    token,
                    interval,
                    0 if df_norm is None else len(df_norm),
                    gap_start_iso,
                    gap_end_iso,
                )
            except Exception as e:  # noqa: BLE001
                log.exception(
                    "[%s] Failed timeframe %s for gap %s-%s: %s",
                    token,
                    interval,
                    gap_start_iso,
                    gap_end_iso,
                    e,
                )
                continue

        # 4) Prune rows older than window start
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        f"DELETE FROM {table_name} WHERE token = :token AND timestamp < :ts"
                    ),
                    {"token": token, "ts": win_start_ms},
                )
            log.info(
                "[%s][%s] Pruned rows older than %s",
                token,
                interval,
                pd.to_datetime(win_start_ms, unit="ms", utc=True).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("[%s][%s] Prune step failed: %s", token, interval, e)

    return None


@app.command("run-live")
def run_live() -> None:  # pragma: no cover - orchestrator wiring
    asyncio.run(_run_live_async())


async def _run_live_async() -> None:  # pragma: no cover - orchestrator wiring
    """Start the 24/7 live trading system with scheduler and background tasks."""
    # --- Initialization ---
    settings = load_settings()

    # Logging
    logging.basicConfig(
        level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("uw.main")
    log.info("System starting up…")

    # Shared state and executors
    registry = DataFrameRegistry()
    process_pool = ProcessPoolExecutor()

    trade_tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
    price_tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

    # Networked components
    notifier = TelegramNotifier(settings)
    risk_manager = RiskManager()

    # Exchange handlers (configure minimal viable set; extend as needed)
    from execution.exchange_clients.binance_client import BinanceClient

    exchange_handlers: Dict[str, Any] = {
        "unicorn": BinanceClient(settings.binance.unicorn),
        "personal": BinanceClient(settings.binance.personal),
    }

    # Trade executor depends on a DB engine factory and notifier
    def _engine_factory():
        return get_async_engine(settings)

    trade_executor = TradeExecutionEngine(
        exchange_handlers=exchange_handlers,
        settings=settings,
        engine_factory=_engine_factory,
        notifier=notifier,
    )

    # Data ingestion and live data orchestration
    from config import MASTER_TOKEN_LIST

    ws_tokens = list(MASTER_TOKEN_LIST.keys())
    coinapi_ws_client = CoinApiWebSocketClient(
        symbols=ws_tokens, queue=trade_tick_queue, settings=settings
    )

    from core.live_manager import LiveDataManager

    live_data_manager = LiveDataManager(
        trade_queue=trade_tick_queue,
        registry=registry,
        historical_api_clients=[],
        santiment_client=None,
        coinmarketcap_client=None,
        coinapi_client=None,
    )

    # Features and Inference
    feature_engine = UnifiedFeatureEngine(registry=registry, sql_engine=None)
    live_feature_calculator = LiveFeatureCalculator(
        engine=feature_engine, registry=registry, process_pool=process_pool
    )
    inference_engine = InferenceEngine(process_pool)

    # Strategy consensus & positions
    consensus_engine = ConsensusEngine(
        settings_module=__import__("config"), dataframe_registry=registry
    )
    position_manager = PositionManager(
        settings=settings,
        risk_manager=risk_manager,
        trade_executor=trade_executor,
        exchange_handlers=exchange_handlers,
        price_tick_queue=price_tick_queue,
        primary_exchange_key="unicorn",
    )

    # Reporting and analysis
    from sqlalchemy.ext.asyncio import async_sessionmaker

    session_maker = async_sessionmaker(
        get_async_engine(settings), expire_on_commit=False
    )
    performance_reporter = PerformanceReporter(
        db_session_factory=session_maker,
        exchange_clients=exchange_handlers,
        notifier=notifier,
    )
    performance_analyzer = PerformanceAnalyzer(
        db_session_factory=session_maker,
        settings=settings,
        risk_manager=risk_manager,
        telegram_notifier=notifier,
    )

    # --- Startup sequence ---
    await notifier.start()
    await inference_engine.load_models()

    # --- Bootstrapping sequence: load raw -> pre-calc features -> gap-fill ---
    try:
        import pandas as pd
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import async_sessionmaker
        from config import MASTER_TOKEN_LIST
        from database.models.raw_data import (
            RawOHLCV15m,
            RawOHLCV1h,
            RawOHLCV4h,
            RawOHLCV1d,
        )

        engine = get_async_engine(settings)
        Session = async_sessionmaker(engine, expire_on_commit=False)

        # 1) Load recent raw OHLCV history for all timeframes into the registry
        tf_to_model = {
            "15m": (RawOHLCV15m, 200),
            "1h": (RawOHLCV1h, 200),
            "4h": (RawOHLCV4h, 200),
            "1d": (RawOHLCV1d, 200),
        }
        async with Session() as session:
            for token in MASTER_TOKEN_LIST.keys():
                for tf, (Model, limit_rows) in tf_to_model.items():
                    try:
                        stmt = (
                            select(
                                Model.timestamp,
                                Model.open,
                                Model.high,
                                Model.low,
                                Model.close,
                                Model.volume,
                            )
                            .where(Model.token == token)
                            .order_by(Model.timestamp.desc())
                            .limit(limit_rows)
                        )
                        result = await session.execute(stmt)
                        rows = result.all()
                        if not rows:
                            continue
                        df = pd.DataFrame(
                            rows,
                            columns=[
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ],
                        )
                        # Ensure ascending chronological order and datetime index
                        df = df.iloc[::-1].copy()
                        df["timestamp"] = pd.to_datetime(
                            df["timestamp"], unit="ms", utc=True, errors="coerce"
                        )
                        await registry.store_df(
                            name=f"{token}_ohlcv_{tf}_df",
                            df=df.set_index("timestamp"),
                            update_mode="overwrite",
                            storage_period=0,
                        )
                    except Exception as e:  # pragma: no cover - best effort per tf
                        log.warning("Bootstrap load failed for %s %s: %s", token, tf, e)

        # 2) Pre-calculate features (store intermediates like ATR/ADX back to registry)
        try:
            await feature_engine.run_1d_pipeline()
        except Exception as e:
            log.warning("Initial run_1d_pipeline failed: %s", e)
        try:
            await feature_engine.run_4h_pipeline()
        except Exception as e:
            log.warning("Initial run_4h_pipeline failed: %s", e)
        try:
            await feature_engine.run_1h_pipeline()
        except Exception as e:
            log.warning("Initial run_1h_pipeline failed: %s", e)

        # 3) Final API gap-fill for recency (scheduled fetches)
        await live_data_manager.fetch_scheduled_data()

        log.info("Bootstrapping complete")
    except Exception as exc:
        log.warning("Bootstrapping step failed or skipped: %s", exc)

    # --- Scheduler setup ---
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_live_15m_cycle,
        "cron",
        minute="*/15",
        second=int(API_FETCH_DELAY_SECONDS),
        max_instances=1,
        misfire_grace_time=60,
        args=[
            live_data_manager,
            feature_engine,
            live_feature_calculator,
            inference_engine,
            consensus_engine,
            position_manager,
        ],
    )
    # Daily and weekly jobs
    scheduler.add_job(
        performance_reporter.generate_daily_report,
        "cron",
        hour=0,
        minute=5,
        max_instances=1,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        performance_reporter.generate_weekly_report,
        "cron",
        day_of_week="mon",
        hour=0,
        minute=10,
        max_instances=1,
        misfire_grace_time=600,
    )
    scheduler.add_job(
        performance_analyzer.run_analysis,
        "cron",
        hour=1,
        minute=0,
        max_instances=1,
        misfire_grace_time=300,
    )

    # --- Start continuous processes ---
    scheduler.start()
    tasks = [
        asyncio.create_task(coinapi_ws_client.run()),
        asyncio.create_task(live_data_manager.run_tick_consumer()),
        asyncio.create_task(position_manager.run_price_monitor()),
    ]

    # --- Run forever with graceful shutdown ---
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    finally:
        log.info("Shutdown initiated; stopping services…")
        try:
            scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass
        # Stop websocket
        try:
            await coinapi_ws_client.stop()
        except Exception:  # noqa: BLE001
            pass
        # Cancel background tasks
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Shutdown process pool
        process_pool.shutdown(wait=False, cancel_futures=True)
        # Close exchange clients
        for _name, client in exchange_handlers.items():
            try:
                if hasattr(client, "close"):
                    await client.close()
            except Exception:  # noqa: BLE001
                pass
        log.info("Shutdown complete.")


@app.command("run-historical")
def run_historical() -> None:
    asyncio.run(_run_historical_async())


async def _run_historical_async() -> None:
    """Download and persist multi-year historical datasets for active tokens.

    This orchestrator initializes API clients and the async DB engine,
    computes the lookback window, and concurrently fetches/saves OHLCV
    data for all active tokens and multiple timeframes.
    """
    import aiohttp
    import pandas as pd
    from typing import Dict as _Dict

    from config import MASTER_TOKEN_LIST, HISTORICAL_LOOKBACK_DAYS
    from database.sql_engine import RawDataSQLEngine
    from data_ingestion.api.coinapi_client import CoinApiClient

    log = logging.getLogger("uw.run_historical")

    settings = load_settings()

    # Configure basic logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    log.info("Starting historical data run…")

    # Engine and clients
    engine = get_async_engine(settings)
    storage_engine = RawDataSQLEngine(engine)

    # Create HTTP session and clients (extend 'clients' container as needed)
    async with aiohttp.ClientSession() as session:
        coinapi_client = CoinApiClient(
            api_key=settings.coinapi_api_key, session=session
        )
        clients: _Dict[str, Any] = {"coinapi": coinapi_client}

        # Define time window with a tz-aware UTC Timestamp (avoid tz_localize/convert pitfalls)
        end_time = pd.Timestamp.now(tz="UTC")
        start_time = end_time - pd.Timedelta(days=int(HISTORICAL_LOOKBACK_DAYS))

        # Launch concurrent tasks per token
        tasks = []
        for token in MASTER_TOKEN_LIST.keys():
            tasks.append(
                asyncio.create_task(
                    _fetch_and_store_for_token(
                        token=token,
                        settings=settings,
                        clients=clients,
                        storage_engine=storage_engine,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any task-level exceptions without raising to keep orchestrator robust
        failures = 0
        for token, res in zip(MASTER_TOKEN_LIST.keys(), results):
            if isinstance(res, Exception):
                failures += 1
                log.error("Token '%s' historical task failed: %s", token, res)
        if failures:
            log.warning("Historical run completed with %d failures.", failures)
        else:
            log.info("Historical run completed successfully for all tokens.")

        # After raw ingestion, run historical feature pipeline
        try:
            feature_engine = UnifiedFeatureEngine(registry=None, sql_engine=None)
            await feature_engine.run_historical_pipeline(
                list(MASTER_TOKEN_LIST.keys()), engine
            )
            log.info(
                "Historical feature pipeline completed and saved to feature store."
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Historical feature pipeline failed: %s", exc)


@app.command()
def train() -> None:
    print("Training pipeline command not yet implemented.")


@app.command()
def optimize() -> None:
    print("Optimization command not yet implemented.")


if __name__ == "__main__":
    # Lightweight CLI hook for inspection outside of Typer commands
    if "inspect-registry" in sys.argv:
        asyncio.run(inspect_registry())
    else:
        app()
