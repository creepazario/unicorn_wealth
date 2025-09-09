from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from unicorn_wealth.ml_lifecycle.data_preparer import PreparedData
    from unicorn_wealth.ml_lifecycle.trainer import ModelTrainer
except Exception:  # pragma: no cover - fallback for direct execution context
    from ml_lifecycle.data_preparer import PreparedData  # type: ignore
    from ml_lifecycle.trainer import ModelTrainer  # type: ignore


def test_train_models_orchestration(mocker):
    # Mock mlflow used inside the trainer module
    mlflow_mock = mocker.patch("unicorn_wealth.ml_lifecycle.trainer.mlflow")

    # start_run should behave as a context manager
    mlflow_mock.start_run.return_value.__enter__.return_value = SimpleNamespace()
    mlflow_mock.start_run.return_value.__exit__.return_value = False

    # Ensure nested mlflow.catboost.log_model exists and is trackable
    mlflow_mock.catboost = SimpleNamespace()
    mlflow_mock.catboost.log_model = mocker.MagicMock()

    # Mock CatBoostClassifier constructor and instance methods
    cb_cls_mock = mocker.patch("unicorn_wealth.ml_lifecycle.trainer.CatBoostClassifier")
    cb_instance = mocker.MagicMock()

    # Configure instance behavior
    cb_instance.fit.return_value = None
    # For validation predictions
    cb_instance.predict_proba.return_value = np.array([[0.4, 0.6]])
    cb_instance.predict.return_value = np.array([1])
    cb_instance.classes_ = np.array([0, 1])
    cb_instance.get_all_params.return_value = {"task_type": "GPU"}
    cb_instance.get_params.return_value = {"_used_task_type": "GPU"}
    cb_instance.set_params.return_value = cb_instance

    # Each instantiation returns a fresh mock (but we can reuse the same for simplicity)
    cb_cls_mock.return_value = cb_instance

    # Build minimal dummy PreparedData for three horizons
    def make_pd():
        X_train = pd.DataFrame({"f1": [0.1, 0.2]})
        y_train = pd.Series([0, 1])
        X_val = pd.DataFrame({"f1": [0.15]})
        y_val = pd.Series([1])
        X_test = pd.DataFrame({"f1": [0.3]})
        y_test = pd.Series([0])
        sample_weights = pd.Series([1.0, 1.0])
        return PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            sample_weights=sample_weights,
        )

    pdata = {"1h": make_pd(), "4h": make_pd(), "8h": make_pd()}

    trainer = ModelTrainer(prepared_data=pdata)

    # Execute training (orchestration only due to mocks)
    trainer.train_models()

    # Assertions
    # 1 parent run + 3 nested runs
    assert mlflow_mock.start_run.call_count == 4

    # CatBoost instantiated 3 times with GPU task type
    assert cb_cls_mock.call_count == 3
    for call in cb_cls_mock.call_args_list:
        kwargs = call.kwargs
        # Ensure task_type='GPU' was requested
        assert kwargs.get("task_type") == "GPU"

    # fit called once per horizon
    assert cb_instance.fit.call_count == 3

    # Logging calls once per nested run
    assert mlflow_mock.log_params.call_count == 3
    assert mlflow_mock.log_metrics.call_count == 3
    assert mlflow_mock.catboost.log_model.call_count == 3
