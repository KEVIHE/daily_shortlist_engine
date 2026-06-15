"""Model and strategy evaluation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def evaluate_classifier(df: pd.DataFrame, label_column: str, prediction_column: str, threshold: float = 0.5) -> dict[str, Any]:
    if df.empty or label_column not in df.columns or prediction_column not in df.columns:
        return {}
    actual = df[label_column].astype(int)
    score = pd.to_numeric(df[prediction_column], errors="coerce").fillna(0.0)
    predicted = (score >= threshold).astype(int)
    accuracy = float((predicted == actual).mean()) if not actual.empty else 0.0
    precision = float(((predicted == 1) & (actual == 1)).sum() / max((predicted == 1).sum(), 1))
    recall = float(((predicted == 1) & (actual == 1)).sum() / max((actual == 1).sum(), 1))
    return {
        "sample_count": int(len(df)),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "positive_rate": round(float(actual.mean()) if not actual.empty else 0.0, 4),
    }


def evaluate_regressor(df: pd.DataFrame, target_column: str, prediction_column: str) -> dict[str, Any]:
    if df.empty or target_column not in df.columns or prediction_column not in df.columns:
        return {}
    target = pd.to_numeric(df[target_column], errors="coerce").fillna(0.0)
    pred = pd.to_numeric(df[prediction_column], errors="coerce").fillna(0.0)
    error = pred - target
    mae = float(error.abs().mean()) if not error.empty else 0.0
    rmse = float((error.pow(2).mean() ** 0.5)) if not error.empty else 0.0
    direction = float(((pred >= 0) == (target >= 0)).mean()) if not target.empty else 0.0
    return {
        "sample_count": int(len(df)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "directional_accuracy": round(direction, 4),
    }


def evaluate_strategy(df: pd.DataFrame, return_column: str, filter_mask: pd.Series) -> dict[str, Any]:
    if df.empty or return_column not in df.columns:
        return {}
    sample = df[filter_mask].copy()
    if sample.empty:
        return {"sample_count": 0}
    returns = pd.to_numeric(sample[return_column], errors="coerce").fillna(0.0)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    equity = returns.cumsum()
    running_peak = equity.cummax()
    drawdown = equity - running_peak
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    payoff = (float(wins.mean()) / abs(avg_loss)) if wins.size and avg_loss < 0 else 0.0
    return {
        "sample_count": int(len(sample)),
        "win_rate": round(float((returns > 0).mean()), 4),
        "average_return": round(float(returns.mean()), 4),
        "average_loss": round(avg_loss, 4),
        "payoff_ratio": round(payoff, 4),
        "max_drawdown": round(float(drawdown.min()) if not drawdown.empty else 0.0, 4),
    }
