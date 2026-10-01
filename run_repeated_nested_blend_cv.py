"""Focused repeated nested-CV comparison of median and nearby WavLM blends."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "gsf"
SEED = 42
N_SPLITS = 5
N_REPEATS = 3
CLIP_MIN, CLIP_MAX = 0.0, 5.0


def make_strat_bins(y, min_count):
    values = np.sort(np.unique(y))
    groups = [[value] for value in values]
    counts = [int((y == value).sum()) for value in values]
    while len(groups) > 1 and min(counts) < min_count:
        index = int(np.argmin(counts))
        if index == 0:
            neighbor = 1
        elif index == len(groups) - 1:
            neighbor = index - 1
        else:
            neighbor = index - 1 if counts[index - 1] <= counts[index + 1] else index + 1
        left, right = sorted((index, neighbor))
        groups[left].extend(groups[right])
        counts[left] += counts[right]
        del groups[right]
        del counts[right]
    bins = np.zeros(len(y), dtype=int)
    for bin_id, group in enumerate(groups):
        bins[np.isin(y, group)] = bin_id
    return bins


def simplex_weights(step=0.25):
    units = round(1 / step)
    return [(a / units, b / units, (units - a - b) / units)
            for a in range(units + 1) for b in range(units - a + 1)]


def mae(y, pred):
    return float(mean_absolute_error(y, pred))


def metrics(y, pred):
    return {
        "mae": mae(y, pred),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
    }


def main():
    train = pd.read_csv(RAW / "train.csv")
    y = train.label.to_numpy(float)
    cache = np.load(CACHE / "gsf_train_wavlm_layer_mean_std.npz", allow_pickle=False)
    if not np.array_equal(cache["filenames"].astype(str), train.filename.to_numpy(str)):
        raise ValueError("WavLM feature cache is not aligned with train.csv")
    features = cache["features"].astype(np.float32)
    layer_names = [f"wavlm_layer_{i:02d}" for i in range(features.shape[1])]
    x = {name: pd.DataFrame(features[:, i, :]) for i, name in enumerate(layer_names)}

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
    candidates = [(name, model) for name in layer_names for model in models]
    variants = {
        "median": None,
        "inner_optimal": None,
        "weights_0.50_0.25_0.25": (0.50, 0.25, 0.25),
        "weights_0.50_0.00_0.50": (0.50, 0.00, 0.50),
        "weights_0.25_0.50_0.25": (0.25, 0.50, 0.25),
        "weights_0.50_0.50_0.00": (0.50, 0.50, 0.00),
    }
    repeat_predictions = {name: [] for name in variants}
    fold_rows = []
    selected_rows = []

    for repeat in range(N_REPEATS):
        bins = make_strat_bins(y, min_count=10)
        outer = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED + repeat)
        predictions = {name: np.zeros(len(y), dtype=float) for name in variants}
        for fold, (train_idx, valid_idx) in enumerate(
            outer.split(np.zeros(len(y)), bins), start=1
        ):
            inner_y = y[train_idx]
            inner_bins = make_strat_bins(inner_y, min_count=6)
            inner = StratifiedKFold(3, shuffle=True,
                                    random_state=SEED + 3000 + repeat * 100 + fold)
            inner_oof = np.zeros((len(train_idx), len(candidates)), dtype=float)
            for inner_train_local, inner_valid_local in inner.split(
                np.zeros(len(inner_y)), inner_bins
            ):
                fit_idx = train_idx[inner_train_local]
                hold_idx = train_idx[inner_valid_local]
                for j, (layer, model_name) in enumerate(candidates):
                    estimator = clone(models[model_name])
                    estimator.fit(x[layer].iloc[fit_idx], y[fit_idx])
                    inner_oof[inner_valid_local, j] = np.clip(
                        estimator.predict(x[layer].iloc[hold_idx]), CLIP_MIN, CLIP_MAX
                    )

            scores = np.mean(np.abs(inner_oof - inner_y[:, None]), axis=0)
            chosen = np.argsort(scores)[:3]
            selected = [candidates[j] for j in chosen]
            inner_selected = inner_oof[:, chosen]
            grid = simplex_weights()
            weight_scores = [mae(inner_y, inner_selected @ np.asarray(w)) for w in grid]
            best_weights = grid[int(np.argmin(weight_scores))]

            outer_matrix = np.column_stack([
                np.clip(
                    clone(models[model_name]).fit(x[layer].iloc[train_idx], y[train_idx])
                    .predict(x[layer].iloc[valid_idx]), CLIP_MIN, CLIP_MAX
                )
                for layer, model_name in selected
            ])
            predictions["median"][valid_idx] = np.median(outer_matrix, axis=1)
            predictions["inner_optimal"][valid_idx] = outer_matrix @ np.asarray(best_weights)
            for name, weights in variants.items():
                if weights is not None:
                    predictions[name][valid_idx] = outer_matrix @ np.asarray(weights)

            fold_rows.append({
                "repeat": repeat + 1,
                "fold": fold,
                "n_valid": len(valid_idx),
                "selected": [f"{layer}/{model}" for layer, model in selected],
                "inner_optimal_weights": list(best_weights),
                **{f"{name}_mae": mae(y[valid_idx], predictions[name][valid_idx])
                   for name in variants},
            })
            selected_rows.append({
                "repeat": repeat + 1,
                "fold": fold,
                "selected": [f"{layer}/{model}" for layer, model in selected],
                "inner_optimal_weights": list(best_weights),
            })
            print(f"repeat {repeat + 1}/{N_REPEATS}, fold {fold}/{N_SPLITS}: "
                  f"selected {[f'{a}/{b}' for a, b in selected]}", flush=True)

        for name in variants:
            repeat_predictions[name].append(predictions[name])

    summary = []
    for name, predictions in repeat_predictions.items():
        per_repeat = [metrics(y, pred) for pred in predictions]
        fold_maes = [row[f"{name}_mae"] for row in fold_rows]
        summary.append({
            "method": name,
            "mean_mae": float(np.mean([m["mae"] for m in per_repeat])),
            "std_mae_across_repeats": float(np.std([m["mae"] for m in per_repeat], ddof=0)),
            "mean_fold_mae": float(np.mean(fold_maes)),
            "std_fold_mae": float(np.std(fold_maes, ddof=0)),
            "repeat_maes": [m["mae"] for m in per_repeat],
            "mean_rmse": float(np.mean([m["rmse"] for m in per_repeat])),
            "std_rmse_across_repeats": float(np.std([m["rmse"] for m in per_repeat], ddof=0)),
            "repeat_rmses": [m["rmse"] for m in per_repeat],
        })
    summary.sort(key=lambda row: row["mean_mae"])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(OUT / "repeated_nested_blend_folds.csv", index=False)
    report = {
        "primary_metric": "mae",
        "outer_cv": f"{N_REPEATS} repeats x {N_SPLITS} folds",
        "candidate_pool": "RidgeCV and SVR_RBF on WavLM layers 00-12; top three chosen inside each outer training split using inner 3-fold CV",
        "blend_comparison": "Median and predeclared nearby weight patterns; inner_optimal weights selected in inner CV",
        "summary": summary,
        "folds": fold_rows,
        "selected_candidates_by_outer_fold": selected_rows,
    }
    (OUT / "repeated_nested_blend_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\\nRepeated nested CV summary (lower is better):")
    for row in summary:
        print(f"{row['method']:28s} MAE {row['mean_mae']:.4f} "
              f"repeat SD {row['std_mae_across_repeats']:.4f}; "
              f"fold MAE {row['mean_fold_mae']:.4f} +/- {row['std_fold_mae']:.4f}; "
              f"RMSE {row['mean_rmse']:.4f} +/- {row['std_rmse_across_repeats']:.4f}; "
              f"repeat MAEs {np.round(row['repeat_maes'], 4).tolist()}")
    print(f"Saved report to {OUT / 'repeated_nested_blend_report.json'}")


if __name__ == "__main__":
    main()
