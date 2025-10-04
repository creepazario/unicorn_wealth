from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

# Reuse orchestrator pieces directly from main
from main import run_live_15m_cycle

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


async def main() -> None:
    """Manually trigger a single 15-minute live cycle for debugging.

    This script mirrors the initialization and startup sequence from the
    run-live command in main.py, then executes one run_live_15m_cycle and
    gracefully shuts everything down.
    """
    # --- Initialization (copied from main._run_live_async) ---
    settings = load_settings()

    # Logging
    logging.basicConfig(
        level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("uw.temp_trigger_cycle")
    log.info("Manual single-cycle trigger starting up…")

    # Shared state and executors
    registry = DataFrameRegistry()
    process_pool = ProcessPoolExecutor()

    trade_tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
    price_tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

    # Networked components
    notifier = TelegramNotifier(settings)
    risk_manager = RiskManager()

    # Exchange handlers
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
    coinapi_ws_client = CoinApiWebSocketClient(
        symbols=["BTC"], queue=trade_tick_queue, settings=settings
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

    # --- Startup sequence (copied) ---
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
        from database.models.base import get_async_engine as _get_engine

        engine = _get_engine(settings)
        Session = async_sessionmaker(engine, expire_on_commit=False)

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
                    except Exception as e:
                        log.warning("Bootstrap load failed for %s %s: %s", token, tf, e)

        # Pre-calc features
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

        # Final gap-fill (scheduled API calls)
        await live_data_manager.fetch_scheduled_data()

        log.info("Bootstrapping complete")
    except Exception as exc:
        log.warning("Bootstrapping step failed or skipped: %s", exc)

    # --- Single cycle execution ---
    try:
        preds = await run_live_15m_cycle(
            live_data_manager,
            feature_engine,
            live_feature_calculator,
            inference_engine,
            consensus_engine,
            position_manager,
        )
        print(preds)
    finally:
        # --- Graceful shutdown (copied and adapted) ---
        log.info("Shutdown initiated; stopping services…")
        # Stop websocket (if it had been started; safe to call stop anyway)
        try:
            await coinapi_ws_client.stop()
        except Exception:
            pass
        # Shutdown process pool
        try:
            process_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        # Close exchange clients
        for _name, client in exchange_handlers.items():
            try:
                if hasattr(client, "close"):
                    await client.close()
            except Exception:
                pass
        log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
