"""Create a test submission for the predeclared 0.50/0.25/0.25 WavLM blend."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from run_repeated_nested_blend_cv import make_strat_bins


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "gsf"
SEED = 42
WEIGHTS = np.array([0.50, 0.25, 0.25])


def main():
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    train_cache = np.load(CACHE / "gsf_train_wavlm_layer_mean_std.npz", allow_pickle=False)
    test_cache = np.load(CACHE / "gsf_test_wavlm_layer_mean_std.npz", allow_pickle=False)
    if not np.array_equal(train_cache["filenames"].astype(str), train.filename.to_numpy(str)):
        raise ValueError("Training WavLM features are not aligned with train.csv")
    if not np.array_equal(test_cache["filenames"].astype(str), test.filename.to_numpy(str)):
        raise ValueError("Test WavLM features are not aligned with test.csv")

    y = train.label.to_numpy(float)
    train_features = train_cache["features"].astype(np.float32)
    test_features = test_cache["features"].astype(np.float32)
    names = [f"wavlm_layer_{i:02d}" for i in range(train_features.shape[1])]
    models = {
        "RidgeCV": make_pipeline(
            SimpleImputer(), StandardScaler(),
            RidgeCV(alphas=(0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
                    scoring="neg_mean_absolute_error"),
        ),
        "SVR_RBF": make_pipeline(
            SimpleImputer(), StandardScaler(), SVR(C=10.0, epsilon=0.1, gamma="scale"),
        ),
    }
    candidates = [(layer, model) for layer in names for model in models]

    bins = make_strat_bins(y, min_count=6)
    splitter = StratifiedKFold(3, shuffle=True, random_state=SEED + 3000)
    inner_oof = np.zeros((len(y), len(candidates)), dtype=float)
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), bins), start=1):
        for j, (layer, model_name) in enumerate(candidates):
            layer_idx = int(layer.rsplit("_", 1)[1])
            estimator = clone(models[model_name])
            estimator.fit(train_features[fit_idx, layer_idx, :], y[fit_idx])
            inner_oof[valid_idx, j] = np.clip(
                estimator.predict(train_features[valid_idx, layer_idx, :]), 0.0, 5.0
            )
        print(f"Completed full-training inner fold {fold}/3", flush=True)

    scores = np.mean(np.abs(inner_oof - y[:, None]), axis=0)
    selected_indices = np.argsort(scores)[:3]
    selected = [candidates[j] for j in selected_indices]
    if abs(float(WEIGHTS.sum()) - 1.0) > 1e-12:
        raise ValueError("Blend weights must sum to one")

    member_predictions = []
    selected_report = []
    for layer, model_name in selected:
        layer_idx = int(layer.rsplit("_", 1)[1])
        estimator = clone(models[model_name])
        estimator.fit(train_features[:, layer_idx, :], y)
        member_predictions.append(np.clip(
            estimator.predict(test_features[:, layer_idx, :]), 0.0, 5.0
        ))
        selected_report.append({
            "candidate": f"{layer}__{model_name}",
            "inner_cv_mae": float(scores[candidates.index((layer, model_name))]),
        })

    prediction = member_predictions[0] * WEIGHTS[0]
    prediction += member_predictions[1] * WEIGHTS[1]
    prediction += member_predictions[2] * WEIGHTS[2]
    output = pd.DataFrame({"filename": test.filename, "label": prediction})
    target = OUT / "nested_blend_050_025_025_submission.csv"
    output.to_csv(target, index=False)
    report = {
        "method": "fixed_weights_on_inner_cv_selected_top_three",
        "weights_in_selected_candidate_order": WEIGHTS.tolist(),
        "selected_candidates": selected_report,
        "inner_cv_blend_mae": float(mean_absolute_error(y, inner_oof[:, selected_indices] @ WEIGHTS)),
        "rows": len(output),
        "columns": list(output.columns),
        "prediction_min": float(output.label.min()),
        "prediction_max": float(output.label.max()),
        "submission_path": str(target),
    }
    (OUT / "nested_blend_050_025_025_metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
