"""LightGBM ranking model for candidate ordering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.engine.ml_features import CATEGORICAL_FEATURES, ML_FEATURE_COLUMNS


def train_ranker(training_df: pd.DataFrame, split, artifact_path, feature_columns: list[str] | None = None) -> dict[str, Any]:
    """Train an LGBMRanker on time-split candidate groups."""
    if training_df.empty:
        return {"trained": False, "reason": "empty dataset"}
    import lightgbm as lgb

    features = feature_columns or ML_FEATURE_COLUMNS
    train_df = training_df.loc[split.train_index].copy()
    valid_df = training_df.loc[split.valid_index].copy()
    if train_df.empty or valid_df.empty:
        return {"trained": False, "reason": "empty split"}

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.05,
        n_estimators=250,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    group_train = train_df.groupby(train_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")).size().tolist()
    group_valid = valid_df.groupby(valid_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")).size().tolist()
    model.fit(
        train_df[features],
        train_df["rank_target"],
        group=group_train,
        eval_set=[(valid_df[features], valid_df["rank_target"])],
        eval_group=[group_valid],
        eval_at=[1, 3, 5],
        categorical_feature=[column for column in CATEGORICAL_FEATURES if column in features],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    model.booster_.save_model(str(artifact_path))
    return {
        "trained": True,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "feature_importance": _feature_importance(model, features),
    }


def predict_ranker(feature_df: pd.DataFrame, artifact_path, feature_columns: list[str] | None = None) -> pd.Series:
    """Predict ranking scores for a candidate dataframe."""
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
