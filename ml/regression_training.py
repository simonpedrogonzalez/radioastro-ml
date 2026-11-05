"""Type-specific training for the combined6 regression comparison."""

import numpy as np
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

from ml.regression import GRID, TARGETS


def training_mask(kinds):
    """Each output learns from clean samples and samples of its own fault type."""
    kinds = np.asarray(kinds)
    if not np.isin(kinds, ("clean", *TARGETS)).all():
        raise ValueError("Type-specific training requires clean, amp, phase or noise samples")
    return np.column_stack([(kinds == "clean") | (kinds == target) for target in TARGETS])


def predict_outputs(model, X):
    if isinstance(model, list):
        return np.column_stack([estimator.predict(X) for estimator in model])
    return model.predict(X)


def fit_filtered_candidate(X, y, mask, params, *, jobs):
    models = []
    for j in range(len(TARGETS)):
        model = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                             multi_strategy="one_output_per_tree", random_state=42, n_jobs=jobs, **params)
        models.append(model.fit(X[mask[:, j]], y[mask[:, j], j]))
    return models


def masked_squared_error(mask):
    """Derivatives of sum(weight * (prediction - truth)^2 / 2).

    Boolean masks retain the original behavior; fractional weights support
    inverse observation probabilities for recovery-damage labels.
    """
    weight = np.asarray(mask, dtype=float)
    if not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("Training weights must be finite and nonnegative")
    def objective(truth, prediction):
        truth = truth.reshape(prediction.shape)
        if prediction.shape != weight.shape:
            raise ValueError("Training mask and predictions have different shapes")
        active = weight > 0
        residual = prediction - np.where(active, truth, 0.)
        return np.where(active, weight * residual, 0.), weight.astype(prediction.dtype)
    return objective


def fit_shared_candidate(X, y, mask, params, *, jobs):
    # The intercept uses only each output's eligible training rows, as in the
    # separate filtered models. It is stored in the booster for prediction.
    base_score = [float(y[mask[:, j], j].mean()) for j in range(len(TARGETS))]
    model = XGBRegressor(objective=masked_squared_error(mask), tree_method="hist",
                         multi_strategy="multi_output_tree", base_score=base_score,
                         random_state=42, n_jobs=jobs, **params)
    return model.fit(X, y)


def fit_type_specific(X, y, kinds, folds, *, jobs, multi_strategy):
    mask = training_mask(kinds)
    if multi_strategy == "one_output_per_tree":
        fit_candidate = fit_filtered_candidate
    elif multi_strategy == "multi_output_tree":
        fit_candidate = fit_shared_candidate
    else:
        raise ValueError(f"Unsupported output strategy: {multi_strategy}")
    rows = []
    for params in ParameterGrid(GRID):
        scores = []
        for train, validation in folds:
            # Keep the original CV population and scoring scale for all setups.
            scale = y[train].std(0)
            active = scale > 0
            if not active.any():
                raise ValueError("All training-fold regression targets are constant")
            model = fit_candidate(X[train], y[train], mask[train], params, jobs=jobs)
            prediction = np.maximum(0, predict_outputs(model, X[validation]))
            scores.append(float(np.mean(np.abs(prediction[:, active] - y[validation][:, active]) / scale[active])))
        rows.append({"params": params, "fold_scores": scores, "mean_scaled_mae": float(np.mean(scores))})
    best = min(rows, key=lambda row: row["mean_scaled_mae"])["params"]
    return fit_candidate(X, y, mask, best, jobs=jobs), best, rows
