from __future__ import annotations

import asyncio
import logging
from typing import Optional

import sqlalchemy.ext.asyncio as sa_async

from core.config_loader import load_settings
from core.dataframe_registry import DataFrameRegistry
from database.sql_engine import RawDataSQLEngine, FeatureStoreSQLEngine
from features.feature_engine import UnifiedFeatureEngine
from ml_lifecycle.data_preparer import DataPreparer
from ml_lifecycle.trainer import ModelTrainer
from ml_lifecycle.registry import ModelRegistryManager
from ml_lifecycle.validator import ModelValidator


# MLflow is used by trainer/registry/validator; import tracking client here for lookups
try:  # pragma: no cover - MLflow may be optional in some CI environments
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - allow script import even if mlflow missing
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore


logger = logging.getLogger("unicorn_wealth.historical_pipeline")


def _configure_logging() -> None:
    """Configure root logging based on settings."""
    try:
        settings = load_settings()
        level = getattr(logging, str(settings.log_level).upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(settings.log_file_path),
                logging.StreamHandler(),
            ],
        )
    except Exception:
        # Fallback minimal config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def _create_async_engine():
    """Create the Async SQLAlchemy engine using application settings."""
    settings = load_settings()
    engine = sa_async.create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle_seconds,
        future=True,
    )
    return engine


def _resolve_training_parent_run_id(
    experiment_name: Optional[str] = None,
) -> Optional[str]:
    """Find the most recent MLflow run named 'UnicornWealth_Training_Run'.

    This serves as the parent run that contains child horizon runs ("1h_model", etc.).
    """
    if mlflow is None or MlflowClient is None:  # pragma: no cover - defensive
        logger.warning("MLflow is not available; cannot resolve parent run id.")
        return None

    client = MlflowClient()

    # Determine which experiments to search
    experiments = []
    try:
        if experiment_name:
            exp = client.get_experiment_by_name(experiment_name)
            if exp is not None:
                experiments = [exp]
        if not experiments:  # fallback to searching all
            experiments = client.search_experiments()
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to enumerate MLflow experiments: %s", e)
        return None

    # Search each experiment for the latest parent training run
    for exp in experiments:
        try:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="tags.mlflow.runName = 'UnicornWealth_Training_Run'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if runs:
                return runs[0].info.run_id
        except Exception as e:  # pragma: no cover - defensive
            logger.exception(
                "Failed to search runs in experiment %s: %s", exp.experiment_id, e
            )
            continue

    logger.warning("Could not locate parent training run 'UnicornWealth_Training_Run'.")
    return None


async def main() -> None:
    """Master orchestration for historical data + MLOps pipeline.

    Phase IV: feature engineering and target generation.
    Phase V: data preparation, training, registration, validation, and promotion.
    """
    _configure_logging()
    logger.info("Starting historical pipeline orchestration...")

    # ------------------------ Setup ------------------------
    engine = None
    try:
        settings = load_settings()
        logger.debug(
            "DB pool_size=%s max_overflow=%s recycle=%s",
            getattr(settings, "db_pool_size", None),
            getattr(settings, "db_max_overflow", None),
            getattr(settings, "db_pool_recycle_seconds", None),
        )
        engine = _create_async_engine()

        # Instantiate core components
        registry = DataFrameRegistry()
        raw_sql_engine = RawDataSQLEngine(engine)
        feature_store_engine = FeatureStoreSQLEngine(engine)
        logger.debug("Feature store engine initialized: %s", feature_store_engine)

        feature_engine = UnifiedFeatureEngine(
            registry=registry,
            sql_engine=raw_sql_engine,
        )

        # --------------------- Phase IV ---------------------
        # Run unified feature engine for all configured tokens from MASTER_TOKEN_LIST
        from config import MASTER_TOKEN_LIST

        tokens = list(MASTER_TOKEN_LIST.keys())
        logger.info(
            "Phase IV: Running unified feature engine (historical) for tokens: %s",
            tokens,
        )
        await feature_engine.run_historical_pipeline(tokens, engine)
        # Note: Target generation and merge with features can be orchestrated separately.

        # --------------------- Phase V ----------------------
        horizon = "15m"
        logger.info("Phase V: Preparing data for training (15m)...")
        data_preparer = DataPreparer(horizons=[horizon])
        prepared = data_preparer.prepare_data_for_training()

        logger.info("Phase V: Training challenger model (15m) with MLflow tracking...")
        experiment_name = "UnicornWealth"
        trainer = ModelTrainer(
            prepared_data=prepared,
            experiment_name=experiment_name,
            horizons=[horizon],
        )
        _ = trainer.train_models()

        # Resolve parent MLflow run ID that contains the horizon child runs
        parent_run_id = _resolve_training_parent_run_id(experiment_name)
        if not parent_run_id:
            raise RuntimeError(
                "Unable to resolve training parent MLflow run ID for registration."
            )

        logger.info("Phase V: Registering model under MLflow Model Registry (15m)...")
        model_names = {horizon: "uw-catboost-15m"}
        registry_manager = ModelRegistryManager(
            horizons=[horizon], model_names=model_names
        )
        try:
            new_versions = registry_manager.register_models(parent_run_id=parent_run_id)
        except ValueError as e:
            logger.warning("Model registration skipped: %s", e)
            new_versions = {}
        logger.info("Registered new model versions: %s", new_versions)

        if not new_versions:
            logger.warning(
                "No model versions registered; skipping validation and promotion."
            )
        else:
            # Prepare inputs for validator using 15m test split
            pdata = prepared.get(horizon)
            if pdata is None:
                raise RuntimeError(f"Prepared data for horizon '{horizon}' not found.")

            test_features = pdata.X_test
            # Ensure OHLCV columns exist in features for backtesting
            price_cols = [
                c
                for c in ["open", "high", "low", "close"]
                if c in test_features.columns
            ]
            if len(price_cols) < 4:
                raise RuntimeError(
                    "Test features are missing OHLCV columns required for validation:"
                    f" have {price_cols}"
                )
            test_prices = test_features.loc[:, ["open", "high", "low", "close"]]

            logger.info("Phase V: Validating challenger model via backtest (15m)...")
            validator = ModelValidator(
                test_features=test_features,
                test_prices=test_prices,
                challenger_run_id=parent_run_id,
                model_artifact_map={horizon: "model"},
                registry_model_names={horizon: "uw-catboost-15m"},
            )
            should_promote = validator.validate_models()
            logger.info("Validation decision: promote=%s", should_promote)

            if should_promote:
                logger.info("Promoting challenger model to Production (15m)...")
                registry_manager.promote_challenger_ensemble(new_versions=new_versions)
                logger.info("Promotion complete.")
            else:
                logger.info(
                    "Challenger did not meet promotion criteria. No changes made."
                )

    except Exception as e:
        logger.exception("Historical pipeline failed with error: %s", e)
        raise
    finally:
        # Ensure the async engine is properly disposed
        if engine is not None:
            try:
                await engine.dispose()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to dispose async DB engine cleanly.")

    logger.info("Historical pipeline orchestration completed.")


if __name__ == "__main__":
    asyncio.run(main())
