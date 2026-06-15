"""Time-series walk-forward helpers for model validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkforwardSplit:
    train_index: list[int]
    valid_index: list[int]
    test_index: list[int]
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str


def build_walkforward_splits(
    df: pd.DataFrame,
    group_column: str = "timestamp",
    min_train_groups: int = 5,
    valid_groups: int = 1,
    test_groups: int = 1,
) -> list[WalkforwardSplit]:
    """Build expanding-window splits using sorted time groups."""
    if df.empty or group_column not in df.columns:
        return []
    groups = pd.Series(pd.to_datetime(df[group_column], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")).dropna().unique().tolist()
    groups = sorted(groups)
    splits: list[WalkforwardSplit] = []
    total_needed = min_train_groups + valid_groups + test_groups
    if len(groups) < total_needed:
        return []
    for end_train in range(min_train_groups, len(groups) - valid_groups - test_groups + 1):
        train_keys = groups[:end_train]
        valid_keys = groups[end_train : end_train + valid_groups]
        test_keys = groups[end_train + valid_groups : end_train + valid_groups + test_groups]
        train_index = df.index[df[group_column].dt.strftime("%Y-%m-%d %H:%M:%S").isin(train_keys)].tolist()
        valid_index = df.index[df[group_column].dt.strftime("%Y-%m-%d %H:%M:%S").isin(valid_keys)].tolist()
        test_index = df.index[df[group_column].dt.strftime("%Y-%m-%d %H:%M:%S").isin(test_keys)].tolist()
        if not train_index or not valid_index or not test_index:
            continue
        splits.append(
            WalkforwardSplit(
                train_index=train_index,
                valid_index=valid_index,
                test_index=test_index,
                train_start=train_keys[0],
                train_end=train_keys[-1],
                valid_start=valid_keys[0],
                valid_end=valid_keys[-1],
                test_start=test_keys[0],
                test_end=test_keys[-1],
            )
        )
    return splits
