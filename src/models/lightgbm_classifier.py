"""LightGBM classifier for breakout / failure probabilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.engine.ml_features import CATEGORICAL_FEATURES, ML_FEATURE_COLUMNS


def train_classifier(training_df: pd.DataFrame, split, target_column: str, artifact_path, feature_columns: list[str] | None = None) -> dict[str, Any]:
    """Train an LGBMClassifier on a time-based split."""
    if training_df.empty or target_column not in training_df.columns:
        return {"trained": False, "reason": f"missing target {target_column}"}
    import lightgbm as lgb

    features = feature_columns or ML_FEATURE_COLUMNS
    train_df = training_df.loc[split.train_index].copy()
    valid_df = training_df.loc[split.valid_index].copy()
    if train_df.empty or valid_df.empty or train_df[target_column].nunique() < 2:
        return {"trained": False, "reason": "insufficient class variation"}
    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=250,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(
        train_df[features],
        train_df[target_column].astype(int),
        eval_set=[(valid_df[features], valid_df[target_column].astype(int))],
        eval_metric="binary_logloss",
        categorical_feature=[column for column in CATEGORICAL_FEATURES if column in features],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    model.booster_.save_model(str(artifact_path))
    return {
        "trained": True,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "feature_importance": _feature_importance(model, features),
    }


def predict_classifier(feature_df: pd.DataFrame, artifact_path, feature_columns: list[str] | None = None) -> pd.Series:
    """Predict binary probabilities for a candidate dataframe."""
    if feature_df.empty or not artifact_path.exists():
        return pd.Series(dtype="float64")
    import lightgbm as lgb

    features = feature_columns or ML_FEATURE_COLUMNS
    booster = lgb.Booster(model_file=str(artifact_path))
    return pd.Series(booster.predict(feature_df[features]), index=feature_df.index)


def _feature_importance(model, features: list[str]) -> list[dict[str, Any]]:
    values = model.booster_.feature_importance(importance_type="gain")
    frame = pd.DataFrame({"feature": features, "importance": values}).sort_values("importance", ascending=False)
    return frame.head(20).to_dict("records")
