from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def _predict_proba(model: Any, df: pd.DataFrame) -> Any:
    """Isolated function to call model.predict_proba for executor.

    Keeping the payload minimal by passing only the model object and the
    small, single-row DataFrame to the subprocess.
    """
    # Some pyfunc models expose predict_proba; if unavailable, fall back to predict
    if hasattr(model, "predict_proba"):
        return model.predict_proba(df)
    return model.predict(df)


class InferenceEngine:
    """Real-time inference orchestrator for multiple prediction horizons.

    This class loads three production-stage models (1h, 4h, 8h) from the
    MLflow Model Registry and performs asynchronous predictions using a
    shared ProcessPoolExecutor. The executor should be the same one used by
    the LiveFeatureCalculator to efficiently manage CPU-bound tasks.
    """

    _MODEL_NAMES = {
        "1h": "uw-catboost-1h",
        "4h": "uw-catboost-4h",
        "8h": "uw-catboost-8h",
    }

    def __init__(self, executor: ProcessPoolExecutor) -> None:
        self._executor = executor
        self._client = MlflowClient()
        self._model_1h: Optional[Any] = None
        self._model_4h: Optional[Any] = None
        self._model_8h: Optional[Any] = None

        # Cache categorical features from the spec to speed up preprocessing
        self._categorical_features: List[str] = self._load_categorical_features()

    async def load_models(self) -> None:
        """Load production-stage models from MLflow Model Registry.

        Uses the models:/<name>/Production URI convention to resolve the
        current production version for each horizon and loads it via
        mlflow.pyfunc.load_model.
        """

        # Load models directly; this avoids cross-process mocking issues and is fast enough.
        def _load_uri(horizon: str) -> Any:
            name = self._MODEL_NAMES[horizon]
            uri = f"models:/{name}/Production"
            logger.info("Loading production model: %s (%s)", horizon, uri)
            return mlflow.pyfunc.load_model(uri)

        self._model_1h = _load_uri("1h")
        self._model_4h = _load_uri("4h")
        self._model_8h = _load_uri("8h")
        logger.info(
            "All production models loaded: 1h=%s, 4h=%s, 8h=%s",
            bool(self._model_1h),
            bool(self._model_4h),
            bool(self._model_8h),
        )

    async def predict(self, feature_rows: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run inference for all horizons concurrently.

        Parameters
        ----------
        feature_rows: Dict[str, pd.DataFrame]
            Mapping from horizon ("1h", "4h", "8h") to a single-row feature
            DataFrame produced by the LiveFeatureCalculator.

        Returns
        -------
        Dict[str, Any]
            Structured dictionary with predictions and probabilities per
            horizon. The exact shape depends on the model output, but will
            include 'horizon' keyed entries.
        """
        self._ensure_models_loaded()

        loop = asyncio.get_running_loop()

        async def _predict_one(
            horizon: str, model: Any, df: pd.DataFrame
        ) -> Dict[str, Any]:
            pre_df = self._preprocess(df)
            # Execute only the predict_proba in the executor
            # Offload the model's predict_proba (or predict) directly
            func = getattr(model, "predict_proba", None)
            if func is None:
                func = getattr(model, "predict")
            proba = await loop.run_in_executor(self._executor, func, pre_df)
            # Best-effort formatting for common outputs
            result: Dict[str, Any] = {"horizon": horizon}
            try:
                if hasattr(proba, "tolist"):
                    result["proba"] = proba.tolist()
                else:
                    result["proba"] = proba  # type: ignore[assignment]
            except Exception:  # noqa: BLE001
                result["proba"] = None
            return result

        tasks = []
        horizon_models = {
            "1h": (self._model_1h, feature_rows.get("1h")),
            "4h": (self._model_4h, feature_rows.get("4h")),
            "8h": (self._model_8h, feature_rows.get("8h")),
        }
        for horizon, (model, df) in horizon_models.items():
            if model is None or df is None or df.empty:
                logger.warning(
                    "Skipping prediction for %s: model or features missing", horizon
                )
                continue
            tasks.append(_predict_one(horizon, model, df))

        results_list = await asyncio.gather(*tasks, return_exceptions=False)
        results: Dict[str, Any] = {r["horizon"]: r for r in results_list}
        return results

    # ------------------------- internal helpers ----------------------
    def _ensure_models_loaded(self) -> None:
        if self._model_1h is None or self._model_4h is None or self._model_8h is None:
            raise RuntimeError(
                "Models are not loaded. Call await InferenceEngine.load_models() at startup."
            )

    def _load_categorical_features(self) -> List[str]:
        """Read categorical feature names from specifications JSON.

        Falls back to an empty list if the spec file is missing or malformed.
        """
        spec_path = Path("specifications") / "Unicorn_Wealth_Feature_Set.json"
        if not spec_path.exists():
            logger.warning(
                "Feature spec not found at %s; no categorical casting will be applied.",
                spec_path,
            )
            return []
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # The schema seems to store feature names under the 'operation' field
            return [
                str(item.get("operation"))
                for item in data
                if item.get("is_categorical_feature") is True
                and item.get("operation") is not None
            ]
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse feature spec at %s: %s", spec_path, exc)
            return []

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast categorical features to string dtype as expected by models.

        The training pipeline marks categoricals; here we ensure incoming
        single-row frames use string dtype for those columns.
        """
        out = df.copy()
        for col in self._categorical_features:
            if col in out.columns:
                try:
                    out[col] = out[col].astype("string")
                except Exception:  # noqa: BLE001
                    # Best effort casting
                    out[col] = out[col].astype(str)
        return out
