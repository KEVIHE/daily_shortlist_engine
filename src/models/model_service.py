"""Model orchestration for training, validation, and inference."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.evaluation import evaluate_classifier, evaluate_regressor
from src.backtest.walkforward import build_walkforward_splits
from src.data.dataset_builder import build_training_dataset
from src.engine.ml_features import ML_FEATURE_COLUMNS, prepare_ml_feature_frame, prepare_ml_prediction_frame
from src.models.lightgbm_classifier import predict_classifier, train_classifier
from src.models.lightgbm_ranker import predict_ranker, train_ranker
from src.models.lightgbm_regressor import predict_regressor, train_regressor
from src.models.model_registry import load_metadata, model_path, save_metadata
from src.storage.db import record_model_run

MODEL_VERSION = "lgbm_v1"
MIN_TRAINING_ROWS = 40
MIN_TRAINING_GROUPS = 8


def run_ml_pipeline(project_root: Path, db_path: Path, candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train models when enough data exists and predict the current candidate frame."""
    fallback_df = candidate_df.copy()
    for column, default in {
        "ml_score": 0.0,
        "breakout_probability": 0.0,
        "failure_probability": 0.0,
        "predicted_upside_pct": 0.0,
        "predicted_downside_pct": 0.0,
        "predicted_upper_band": fallback_df.get("base_high", 0.0),
        "predicted_lower_band": fallback_df.get("base_low", 0.0),
        "ml_rank_raw": 0.0,
    }.items():
        if column not in fallback_df.columns:
            fallback_df[column] = default
    if not _lightgbm_available():
        return fallback_df, {"status": "fallback", "message": "lightgbm not installed", "metadata": load_metadata(project_root)}

    training_df = build_training_dataset(db_path)
    if training_df.empty or len(training_df) < MIN_TRAINING_ROWS:
        return fallback_df, {"status": "fallback", "message": "insufficient replay samples for model training", "metadata": load_metadata(project_root)}
    unique_groups = training_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").nunique()
    if unique_groups < MIN_TRAINING_GROUPS:
        return fallback_df, {"status": "fallback", "message": "insufficient time groups for walk-forward validation", "metadata": load_metadata(project_root)}

    training_df = _normalize_training_frame(training_df)
    splits = build_walkforward_splits(training_df, min_train_groups=5, valid_groups=1, test_groups=1)
    if not splits:
        return fallback_df, {"status": "fallback", "message": "walk-forward split generation failed", "metadata": load_metadata(project_root)}

    latest_split = splits[-1]
    split_meta = {
        "train_start": latest_split.train_start,
        "train_end": latest_split.train_end,
        "valid_start": latest_split.valid_start,
        "valid_end": latest_split.valid_end,
        "test_start": latest_split.test_start,
        "test_end": latest_split.test_end,
    }
    ranker_path = model_path(project_root, "ranker")
    classifier_path = model_path(project_root, "classifier")
    failure_path = project_root / "data" / "models" / "classifier_failure.txt"
    reg_up_path = model_path(project_root, "regressor_up")
    reg_down_path = model_path(project_root, "regressor_down")

    feature_frame = prepare_ml_feature_frame(training_df)
    deduped_features = list(dict.fromkeys(ML_FEATURE_COLUMNS))
    training_df = training_df.reset_index(drop=True)
    for column in deduped_features:
        training_df[column] = feature_frame[column].values

    ranker_info = train_ranker(training_df, latest_split, ranker_path, deduped_features)
    classifier_info = train_classifier(training_df, latest_split, "breakout_first_15m", classifier_path, deduped_features)
    failure_info = train_classifier(training_df, latest_split, "invalidation_first_15m", failure_path, deduped_features)
    reg_up_info = train_regressor(training_df, latest_split, "future_max_upside_15m", reg_up_path, deduped_features)
    reg_down_info = train_regressor(training_df, latest_split, "future_max_drawdown_15m", reg_down_path, deduped_features)

    metadata = {
        "model_version": MODEL_VERSION,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ranker": ranker_info,
        "classifier": classifier_info,
        "failure_classifier": failure_info,
        "regressor_up": reg_up_info,
        "regressor_down": reg_down_info,
    }
    save_metadata(project_root, metadata)

    evaluation_frame = training_df.loc[latest_split.test_index].copy()
    if ranker_info.get("trained"):
        evaluation_frame["rank_prediction"] = predict_ranker(evaluation_frame[deduped_features], ranker_path, deduped_features)
        rank_metrics = {
            "sample_count": int(len(evaluation_frame)),
            "avg_rank_target_top3": round(float(evaluation_frame.sort_values("rank_prediction", ascending=False).head(3)["rank_target"].mean()), 4),
            "avg_rank_target_all": round(float(evaluation_frame["rank_target"].mean()), 4),
        }
        record_model_run(db_path, "ranker", MODEL_VERSION, split_meta, rank_metrics, ranker_info.get("feature_importance", []))
    if classifier_info.get("trained"):
        evaluation_frame["breakout_probability"] = predict_classifier(evaluation_frame[deduped_features], classifier_path, deduped_features)
        record_model_run(db_path, "classifier_breakout", MODEL_VERSION, split_meta, evaluate_classifier(evaluation_frame, "breakout_first_15m", "breakout_probability"), classifier_info.get("feature_importance", []))
    if failure_info.get("trained"):
        evaluation_frame["failure_probability"] = predict_classifier(evaluation_frame[deduped_features], failure_path, deduped_features)
        record_model_run(db_path, "classifier_failure", MODEL_VERSION, split_meta, evaluate_classifier(evaluation_frame, "invalidation_first_15m", "failure_probability"), failure_info.get("feature_importance", []))
    if reg_up_info.get("trained"):
        evaluation_frame["predicted_upside_pct"] = predict_regressor(evaluation_frame[deduped_features], reg_up_path, deduped_features)
        record_model_run(db_path, "regressor_up", MODEL_VERSION, split_meta, evaluate_regressor(evaluation_frame, "future_max_upside_15m", "predicted_upside_pct"), reg_up_info.get("feature_importance", []))
    if reg_down_info.get("trained"):
        evaluation_frame["predicted_downside_pct"] = predict_regressor(evaluation_frame[deduped_features], reg_down_path, deduped_features)
        record_model_run(db_path, "regressor_down", MODEL_VERSION, split_meta, evaluate_regressor(evaluation_frame, "future_max_drawdown_15m", "predicted_downside_pct"), reg_down_info.get("feature_importance", []))

    prediction_df = candidate_df.copy()
    if prediction_df.empty:
        return prediction_df, {"status": "trained", "message": "models updated; no current candidates", "metadata": metadata}
    pred_features = prepare_ml_prediction_frame(prediction_df)
    deduped_pred_features = list(dict.fromkeys(pred_features.columns.tolist()))
    prediction_df["ml_rank_raw"] = predict_ranker(pred_features, ranker_path, deduped_pred_features) if ranker_path.exists() else 0.0
    prediction_df["breakout_probability"] = predict_classifier(pred_features, classifier_path, deduped_pred_features) if classifier_path.exists() else 0.0
    prediction_df["failure_probability"] = predict_classifier(pred_features, failure_path, deduped_pred_features) if failure_path.exists() else 0.0
    prediction_df["predicted_upside_pct"] = predict_regressor(pred_features, reg_up_path, deduped_pred_features) if reg_up_path.exists() else 0.0
    prediction_df["predicted_downside_pct"] = predict_regressor(pred_features, reg_down_path, deduped_pred_features) if reg_down_path.exists() else 0.0
    prediction_df["ml_score"] = _compose_ml_score(prediction_df)
    prediction_df["predicted_upper_band"] = prediction_df["current_price"] * (1.0 + prediction_df["predicted_upside_pct"].clip(lower=0.0) / 100.0)
    prediction_df["predicted_lower_band"] = prediction_df["current_price"] * (1.0 + prediction_df["predicted_downside_pct"].clip(upper=0.0) / 100.0)
    return prediction_df, {"status": "trained", "message": "lightgbm models trained and predictions refreshed", "metadata": metadata}


def _normalize_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["tradeable"] = frame.get("tradeable", 0).fillna(0).astype(int)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    return frame


def _compose_ml_score(df: pd.DataFrame) -> pd.Series:
    breakout = pd.to_numeric(df.get("breakout_probability"), errors="coerce").fillna(0.0)
    failure = pd.to_numeric(df.get("failure_probability"), errors="coerce").fillna(0.0)
    upside = pd.to_numeric(df.get("predicted_upside_pct"), errors="coerce").fillna(0.0)
    downside = pd.to_numeric(df.get("predicted_downside_pct"), errors="coerce").fillna(0.0).abs()
    rank_raw = pd.to_numeric(df.get("ml_rank_raw"), errors="coerce").fillna(0.0)
    if len(rank_raw) > 1:
        rank_component = rank_raw.rank(pct=True)
    else:
        rank_component = pd.Series([0.5] * len(rank_raw), index=df.index)
    score = 4.0 * breakout + 2.5 * rank_component + 0.6 * upside - 0.8 * failure - 0.5 * downside
    return score.clip(lower=0.0, upper=10.0).round(2)


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False
