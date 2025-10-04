from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlflow
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.config_loader import Settings
from database.models.feature_stores import (
    FeatureStore1h,
    FeatureStore4h,
    FeatureStore8h,
)
from database.models.operational import TradeSignals
from execution.risk_management import RiskManager

try:
    # Evidently v0.7 API
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except (
    Exception
):  # pragma: no cover - allow import without evidently installed in some envs
    Report = None  # type: ignore
    DataDriftPreset = None  # type: ignore


@dataclass
class _SignalRecord:
    signal_id: str
    timestamp_ms: int
    token: str
    account_name: str
    strategic_directive: str
    prediction_label: int
    avg_probability: float
    mlflow_run_id: Optional[str]


class PerformanceAnalyzer:
    """Daily analyzer for model performance and drift detection.

    Responsibilities:
      - Evaluate historical signal accuracy and log metrics to MLflow.
      - Detect data/concept drift using Evidently and take automated actions
        via RiskManager and TelegramNotifier.

    Notes:
      - This implementation degrades gracefully when linkages (e.g.,
        TradeSignals.mlflow_run_id) are missing. It logs warnings and continues.
    """

    def __init__(
        self,
        *,
        db_session_factory: async_sessionmaker[AsyncSession],
        settings: Settings | Any,
        risk_manager: RiskManager,
        telegram_notifier: Any,
    ) -> None:
        self._session_factory = db_session_factory
        self.settings = settings
        self.risk_manager = risk_manager
        self.notifier = telegram_notifier
        self._log = logging.getLogger(__name__)

    async def run_analysis(self) -> None:
        """Main entry point for the daily scheduler."""
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=1)
        since_ms = int(since.timestamp() * 1000)

        signals = await self._fetch_recent_signals(since_ms)
        if not signals:
            self._log.info("PerformanceAnalyzer: No signals found in the last 24h.")
            return

        # Group by mlflow run id for logging per model version
        by_run: Dict[Optional[str], List[_SignalRecord]] = {}
        for rec in signals:
            by_run.setdefault(rec.mlflow_run_id, []).append(rec)

        # For drift, we compare current live features vs reference (training) data.
        # We'll construct combined current/reference datasets across all involved runs.
        all_current_frames: List[pd.DataFrame] = []
        all_reference_frames: List[pd.DataFrame] = []

        # Process each run for performance metrics and accumulate data for drift.
        for run_id, recs in by_run.items():
            # Load reference data for this run if possible
            reference_df = await self._load_reference_training_data(run_id)
            if reference_df is not None and not reference_df.empty:
                all_reference_frames.append(reference_df)

            # Load live feature rows for the tokens referenced in these signals
            tokens = sorted({r.token for r in recs})
            try:
                current_df = await self._load_current_features(
                    tokens=tokens, since_ms=since_ms
                )
            except Exception as exc:  # pragma: no cover - db/infra dependent
                current_df = None
                self._log.exception("Failed loading current features: %s", exc)

            if current_df is not None and not current_df.empty:
                all_current_frames.append(current_df)

            # Historical accuracy metrics (best-effort)
            try:
                precision, recall = await self._compute_precision_recall_best_effort(
                    recs
                )
                if run_id:
                    # Log to the specific MLflow run
                    with mlflow.start_run(run_id=run_id):
                        mlflow.log_metric("daily_precision", precision)
                        mlflow.log_metric("daily_recall", recall)
                else:
                    self._log.warning(
                        "TradeSignals missing mlflow_run_id; cannot log metrics to MLflow for %d records.",
                        len(recs),
                    )
            except Exception as exc:  # pragma: no cover - depends on infra linkage
                self._log.exception(
                    "Failed computing/logging performance metrics: %s", exc
                )

        # Drift analysis across aggregated datasets
        try:
            reference_data = (
                pd.concat(all_reference_frames, ignore_index=True)
                if all_reference_frames
                else None
            )
            current_data = (
                pd.concat(all_current_frames, ignore_index=True)
                if all_current_frames
                else None
            )

            if reference_data is None or current_data is None:
                self._log.warning(
                    "Skipping drift analysis due to missing reference (%s) or current (%s) data.",
                    reference_data is not None,
                    current_data is not None,
                )
                return

            drift_score, html_bytes = self._run_evidently_drift(
                reference_data, current_data
            )
            self._log.info("Evidently drift_score=%.4f", drift_score)

            # Actions based on thresholds
            thresholds = self._get_drift_thresholds()
            if drift_score >= thresholds["SEVERE_DRIFT_SCORE"]:
                self.risk_manager.activate_killswitch()
                await self._send_system_alert(
                    level="CRITICAL",
                    message=(
                        f"Severe drift detected (score={drift_score:.3f}). "
                        "Kill-switch activated."
                    ),
                )
            elif drift_score >= thresholds["MODERATE_DRIFT_SCORE"]:
                await self._send_system_alert(
                    level="WARNING",
                    message=(
                        f"Moderate drift detected (score={drift_score:.3f}). "
                        "Retraining recommended."
                    ),
                )
                self._log.warning(
                    "Trigger retraining pipeline due to drift >= MODERATE threshold."
                )

            # Log HTML report to a generic drift run if any run_id exists; otherwise create one
            any_run_id = next((k for k in by_run.keys() if k), None)
            if any_run_id:
                with mlflow.start_run(run_id=any_run_id):
                    mlflow.log_metric("daily_drift_score", drift_score)
                    if html_bytes:
                        mlflow.log_artifact(
                            local_path=self._save_temp_html(
                                "evidently_drift_report.html", html_bytes
                            )
                        )
            else:
                # Create a new run under default experiment to store the artifact
                with mlflow.start_run():
                    mlflow.set_tag("component", "PerformanceAnalyzer")
                    mlflow.log_metric("daily_drift_score", drift_score)
                    if html_bytes:
                        mlflow.log_artifact(
                            local_path=self._save_temp_html(
                                "evidently_drift_report.html", html_bytes
                            )
                        )
        except Exception as exc:  # pragma: no cover - external libs and env dependent
            self._log.exception("Drift analysis failed: %s", exc)

    # -----------------------
    # Data loading helpers
    # -----------------------
    async def _fetch_recent_signals(self, since_ms: int) -> List[_SignalRecord]:
        sm = self._session_factory
        out: List[_SignalRecord] = []
        async with sm() as session:
            stmt = (
                select(TradeSignals)
                .where(TradeSignals.timestamp >= since_ms)
                .order_by(TradeSignals.timestamp.asc())
            )
            res = await session.execute(stmt)
            rows: Sequence[TradeSignals] = [r[0] for r in res.all()]

        for row in rows:
            # Try to get mlflow_run_id if it exists in the table (schema may not yet include it)
            run_id = getattr(row, "mlflow_run_id", None)
            try:
                out.append(
                    _SignalRecord(
                        signal_id=str(row.signal_id),
                        timestamp_ms=int(row.timestamp),
                        token=str(row.token),
                        account_name=str(row.account_name),
                        strategic_directive=str(row.strategic_directive),
                        prediction_label=int(row.prediction_label),
                        avg_probability=float(row.avg_probability),
                        mlflow_run_id=str(run_id) if run_id else None,
                    )
                )
            except Exception:
                continue
        return out

    async def _load_reference_training_data(
        self, run_id: Optional[str]
    ) -> Optional[pd.DataFrame]:
        if not run_id:
            return None
        try:
            client = mlflow.tracking.MlflowClient()
            # Try common artifact names
            for name in [
                "training_data.parquet",
                "training_data.csv",
                "artifacts/training_data.parquet",
                "artifacts/training_data.csv",
            ]:
                try:
                    local_path = client.download_artifacts(run_id, name)
                except Exception:
                    continue
                if local_path:
                    if local_path.endswith(".parquet"):
                        return pd.read_parquet(local_path)
                    if local_path.endswith(".csv"):
                        return pd.read_csv(local_path)
            self._log.warning(
                "No training dataset artifact found for run_id=%s", run_id
            )
        except Exception as exc:  # pragma: no cover
            self._log.exception("Failed to load training data for %s: %s", run_id, exc)
        return None

    async def _load_current_features(
        self, *, tokens: List[str], since_ms: int
    ) -> Optional[pd.DataFrame]:
        # Try each horizon table and union rows, as we may not know which was used
        frames: List[pd.DataFrame] = []
        for model in (FeatureStore1h, FeatureStore4h, FeatureStore8h):
            df = await self._load_features_from_table(
                model, tokens=tokens, since_ms=since_ms
            )
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return None
        # Align columns by intersection to avoid NaNs explosion
        common_cols = set(frames[0].columns)
        for f in frames[1:]:
            common_cols &= set(f.columns)
        if not common_cols:
            return None
        frames = [f[list(sorted(common_cols))] for f in frames]
        return pd.concat(frames, ignore_index=True)

    async def _load_features_from_table(
        self,
        model: Any,
        *,
        tokens: List[str],
        since_ms: int,
    ) -> Optional[pd.DataFrame]:
        sm = self._session_factory
        cols = [c.name for c in model.__table__.columns]
        async with sm() as session:
            stmt = (
                select(model)
                .where(model.timestamp >= since_ms)
                .where(model.token.in_(tokens))
            )
            res = await session.execute(stmt)
            rows = [r[0] for r in res.all()]
        if not rows:
            return None
        data = [{c: getattr(row, c, None) for c in cols} for row in rows]
        return pd.DataFrame.from_records(data)

    # -----------------------
    # Metric computation (best-effort placeholders until full linkage exists)
    # -----------------------
    async def _compute_precision_recall_best_effort(
        self, recs: List[_SignalRecord]
    ) -> Tuple[float, float]:
        """Compute simple precision/recall for open-direction signals.

        Since we do not yet have a reliable linkage from TradeSignals to realized
        outcomes, this function approximates:
          - Treat OPEN_LONG/OPEN_SHORT as positive predictions for movement in that
            direction.
          - Without realized market movement data, we cannot compute true labels.
            We therefore return NaN-safe zeros and log a warning.

        Returns (precision, recall).
        """
        pos = [r for r in recs if r.strategic_directive in ("OPEN_LONG", "OPEN_SHORT")]
        if not pos:
            return 0.0, 0.0
        # Placeholder: unknown ground truth in current repo scope
        self._log.warning(
            "Insufficient linkage to compute true precision/recall (no realized outcomes). Returning 0.0."
        )
        return 0.0, 0.0

    # -----------------------
    # Drift with Evidently
    # -----------------------
    def _run_evidently_drift(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> Tuple[float, Optional[bytes]]:
        if Report is None or DataDriftPreset is None:
            raise RuntimeError("Evidently is not installed; cannot compute drift.")

        # Align columns to intersection and drop obvious identifiers
        common = list(sorted(set(reference_data.columns) & set(current_data.columns)))
        drop_cols = {"timestamp", "token"}
        cols = [c for c in common if c not in drop_cols]
        ref = reference_data[cols].copy()
        cur = current_data[cols].copy()

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

        drift_score = self._extract_drift_score(report)
        html_bytes = None
        try:
            html_str = report.as_html()
            html_bytes = html_str.encode("utf-8")
        except Exception:  # pragma: no cover - optional artifact
            html_bytes = None
        return drift_score, html_bytes

    def _get_drift_thresholds(self) -> Dict[str, float]:
        """Retrieve drift thresholds from settings or config defaults."""
        # Access via settings or fallback to module-level config constants
        try:
            from config import DRIFT_ACTION_THRESHOLDS as DEFAULT_THRESHOLDS  # type: ignore
        except Exception:  # pragma: no cover
            DEFAULT_THRESHOLDS = {  # type: ignore[assignment]
                "MODERATE_DRIFT_SCORE": 0.3,
                "SEVERE_DRIFT_SCORE": 0.6,
            }
        # Try settings first (supports lowercase via getattr)
        val = getattr(self.settings, "DRIFT_ACTION_THRESHOLDS", None)
        if val is None:
            val = getattr(self.settings, "drift_action_thresholds", None)
        if isinstance(val, dict):
            # Ensure keys exist; fill from defaults as needed
            out = dict(DEFAULT_THRESHOLDS)
            out.update({k: float(v) for k, v in val.items() if k in DEFAULT_THRESHOLDS})
            return out
        return dict(DEFAULT_THRESHOLDS)

    @staticmethod
    def _extract_drift_score(report: Any) -> float:
        """Extract an overall drift score from Evidently report.as_dict().

        Evidently structures may change; we attempt several known paths.
        Prefer dataset_drift.drift_share when available.
        """
        try:
            obj = report.as_dict()
        except Exception:  # pragma: no cover
            return 0.0
        # Try common paths
        try:
            metrics = obj.get("metrics", [])
            for m in metrics:
                res = m.get("result", {})
                ds = res.get("dataset_drift") or {}
                share = ds.get("drift_share")
                if isinstance(share, (int, float)):
                    return float(share)
        except Exception:
            pass
        # Fallback: any numeric score field
        try:
            return float(obj.get("score", 0.0))
        except Exception:
            return 0.0

    # -----------------------
    # Notifications & utilities
    # -----------------------
    async def _send_system_alert(self, *, level: str, message: str) -> None:
        data = {
            "level": level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
        }
        try:
            # The project's TelegramNotifier expects type 'SYSTEM_ALERTS' and an ADMIN channel
            await self.notifier.send_notification(
                type="SYSTEM_ALERTS", data=data, channel_type="ADMIN"
            )
        except Exception:  # pragma: no cover - notifier may be disabled in some envs
            self._log.exception("Failed to send Telegram system alert.")

    @staticmethod
    def _save_temp_html(filename: str, html_bytes: bytes) -> str:
        # Persist to a temp file in the project logs/ directory for MLflow to pick up
        import os

        os.makedirs("logs", exist_ok=True)
        path = f"logs/{filename}"
        with open(path, "wb") as f:
            f.write(html_bytes)
        return path
