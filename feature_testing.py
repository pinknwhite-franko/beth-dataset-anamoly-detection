"""Validation-only feature testing aligned with model_training.ipynb.

Run: python feature_testing.py

Outputs:
    features_data/permutation_importance_validation.csv
    features_data/best_features_val_k5.csv
    features_data/best_features_val_k10.csv
    features_data/best_features_val_k15.csv
    features_data/validation_selection_results.csv
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from model_class.feature_builder_clean_v2 import FeatureBuilder
from model_class.feature_builder_transformer import FeatureBuilderTransformer


OUT_DIR = os.path.join(os.getcwd(), "features_data")


count_cols = [
    "processId_eventId_past_count",
    "processId_past_count",
    "threadId_past_count",
    "eventId_past_count",
    "userId_past_count",
    "parentProcessId_past_count",
    "mountNamespace_past_count",
]

rarity_cols = [
    "processId_eventId_rarity",
    "processId_rarity",
    "threadId_rarity",
    "eventId_rarity",
    "userId_rarity",
    "parentProcessId_rarity",
    "mountNamespace_rarity",
]

first_seen_cols = [
    "processId_eventId_is_first_seen",
    "processId_is_first_seen",
    "threadId_is_first_seen",
    "eventId_is_first_seen",
    "userId_is_first_seen",
    "parentProcessId_is_first_seen",
    "mountNamespace_is_first_seen",
]

scale_cols = [
    "stackAddresses_len",
    "stackAddresses_unique_ratio",
    "child_process_spawn_rate_so_far",
]

binary_cols = [
    "userId_binary",
    "parentUserId_binary",
]

passthrough_cols = [
    "argsNum",
    "returnValue",
    "returnValue_is_error",
    "args_has_path",
]

ALL_CANDIDATE_COLS = (
    count_cols
    + rarity_cols
    + first_seen_cols
    + scale_cols
    + binary_cols
    + passthrough_cols
)


def read_data(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "datasets", "raw")

    def _load(name):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pd.read_csv(path, header=0)

    def _split(df):
        y = df["evil"]
        X = df.drop(columns=[c for c in ["sus", "evil"] if c in df.columns])
        return X, y

    X_train, y_train = _split(_load("labelled_training_data.csv"))
    X_val, y_val = _split(_load("labelled_validation_data.csv"))
    return X_train, y_train, X_val, y_val


def build_features(X_train, X_val):
    trans = FeatureBuilderTransformer(FeatureBuilder(), return_numpy=False)
    return trans.fit_transform(X_train), trans.transform(X_val)


def keep_candidate_features(X_train, X_val):
    missing = [c for c in ALL_CANDIDATE_COLS if c not in X_train.columns]
    if missing:
        raise ValueError(f"Missing expected features from FeatureBuilder: {missing}")

    def clean(X):
        return X[ALL_CANDIDATE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

    return clean(X_train), clean(X_val)


def build_preprocessor(selected_features):
    selected_count_cols = [c for c in count_cols if c in selected_features]
    selected_rarity_cols = [c for c in rarity_cols if c in selected_features]
    selected_first_seen_cols = [c for c in first_seen_cols if c in selected_features]
    selected_scale_cols = [c for c in scale_cols if c in selected_features]
    selected_binary_cols = [c for c in binary_cols if c in selected_features]
    selected_passthrough_cols = [c for c in passthrough_cols if c in selected_features]

    log_count_pipeline = Pipeline([
        ("log", FunctionTransformer(
            lambda x: np.log1p(np.clip(x, 0, None)),
            feature_names_out="one-to-one",
            validate=False
        )),
        ("scale", RobustScaler()),
    ])

    rarity_pipeline = Pipeline([
        ("scale", RobustScaler()),
    ])

    scale_pipeline = Pipeline([
        ("scale", RobustScaler()),
    ])

    transformers = []

    if selected_count_cols:
        transformers.append(("count", log_count_pipeline, selected_count_cols))
    if selected_rarity_cols:
        transformers.append(("rarity", rarity_pipeline, selected_rarity_cols))
    if selected_first_seen_cols:
        transformers.append(("first_seen", "passthrough", selected_first_seen_cols))
    if selected_scale_cols:
        transformers.append(("scale_other", scale_pipeline, selected_scale_cols))
    if selected_binary_cols:
        transformers.append(("binary", "passthrough", selected_binary_cols))
    if selected_passthrough_cols:
        transformers.append(("pass", "passthrough", selected_passthrough_cols))

    if not transformers:
        raise ValueError("No selected features were mapped into the preprocessor.")

    return ColumnTransformer(transformers=transformers)


def validation_metrics(model, X_train, X_val):
    train_scores = -model.score_samples(X_train)
    val_scores = -model.score_samples(X_val)

    val_pred = model.predict(X_val)
    predicted_outliers = int(np.sum(val_pred == -1))   # false evils on good-only validation
    predicted_inliers = int(np.sum(val_pred == 1))     # true goods on good-only validation
    val_outlier_rate = float(np.mean(val_pred == -1))

    score_shift = float(abs(train_scores.mean() - val_scores.mean()))
    score_std_gap = float(abs(train_scores.std() - val_scores.std()))

    # Lower is better
    objective = val_outlier_rate + 0.5 * score_shift + 0.25 * score_std_gap

    return {
        "objective": objective,
        "predicted_outliers": predicted_outliers,
        "predicted_inliers": predicted_inliers,
        "val_outlier_rate": val_outlier_rate,
        "train_score_mean": float(train_scores.mean()),
        "val_score_mean": float(val_scores.mean()),
        "score_shift": score_shift,
        "train_score_std": float(train_scores.std()),
        "val_score_std": float(val_scores.std()),
        "score_std_gap": score_std_gap,
    }


def fit_iforest_on_features(X_train, X_val, selected_features, random_state=2000):
    preprocessor = build_preprocessor(selected_features)

    model = Pipeline([
        ("scaler", preprocessor),
        ("iforest", IsolationForest(
            n_estimators=100,
            contamination="auto",
            max_features=0.7,
            random_state=random_state
        ))
    ])

    model.fit(X_train[selected_features])
    return model


def permutation_importance_validation(X_train, X_val, selected_features, random_state=42):
    """
    Fit on training good data only.
    Evaluate importance by permuting one feature at a time on validation good data only.
    Higher objective increase = more important feature.
    """
    model = fit_iforest_on_features(X_train, X_val, selected_features, random_state=2000)
    baseline = validation_metrics(model, X_train[selected_features], X_val[selected_features])

    rng = np.random.RandomState(random_state)
    rows = []

    for col in selected_features:
        Xp = X_val[selected_features].copy()
        Xp[col] = rng.permutation(Xp[col].values)

        perm = validation_metrics(model, X_train[selected_features], Xp)

        rows.append({
            "feature": col,
            "baseline_objective": baseline["objective"],
            "permuted_objective": perm["objective"],
            "objective_increase": perm["objective"] - baseline["objective"],
            "baseline_predicted_outliers": baseline["predicted_outliers"],
            "permuted_predicted_outliers": perm["predicted_outliers"],
            "baseline_val_outlier_rate": baseline["val_outlier_rate"],
            "permuted_val_outlier_rate": perm["val_outlier_rate"],
            "baseline_score_shift": baseline["score_shift"],
            "permuted_score_shift": perm["score_shift"],
        })

    return pd.DataFrame(rows).sort_values("objective_increase", ascending=False).reset_index(drop=True)


def evaluate_feature_subsets_on_validation(
    X_train,
    X_val,
    imp_df,
    k_list,
    random_state=2000
):
    """
    Fit on training only.
    Evaluate feature subsets on validation only.
    Lower objective is better.
    """
    all_features = list(X_train.columns)
    imp_sorted = imp_df.sort_values("objective_increase", ascending=False).reset_index(drop=True)

    def fit_eval(features, scenario):
        model = fit_iforest_on_features(X_train, X_val, features, random_state=random_state)
        metrics = validation_metrics(model, X_train[features], X_val[features])

        return {
            "scenario": scenario,
            "n_features": len(features),
            "features": ",".join(features),
            "objective": metrics["objective"],
            "predicted_outliers": metrics["predicted_outliers"],
            "predicted_inliers": metrics["predicted_inliers"],
            "val_outlier_rate": metrics["val_outlier_rate"],
            "score_shift": metrics["score_shift"],
            "score_std_gap": metrics["score_std_gap"],
        }

    results = [fit_eval(all_features, "baseline_all")]

    for k in sorted(set(k_list)):
        k = min(int(k), len(imp_sorted))
        features = imp_sorted["feature"].iloc[:k].tolist()
        results.append(fit_eval(features, f"keep_top_k={k}"))

    return pd.DataFrame(results).sort_values("objective", ascending=True).reset_index(drop=True)


def main():
    X_train, _y_train, X_val, _y_val,= read_data()

    print("Building features on train/validation...")
    X_train_f, X_val_f = build_features(X_train, X_val)

    print("Keeping notebook-aligned candidate features...")
    X_train_f, X_val_f = keep_candidate_features(X_train_f, X_val_f)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Computing validation-only permutation importance...")
    imp_df = permutation_importance_validation(
        X_train_f,
        X_val_f,
        selected_features=ALL_CANDIDATE_COLS,
        random_state=42
    )

    imp_path = os.path.join(OUT_DIR, "permutation_importance_validation.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"Saved: {imp_path}")

    for k in (5, 10, 15):
        best_path = os.path.join(OUT_DIR, f"best_features_val_k{k}.csv")
        imp_df["feature"].iloc[:k].to_frame().to_csv(best_path, index=False)
        print(f"Saved top-{k} features: {best_path}")

    print("Evaluating baseline vs reduced feature sets on validation only...")
    results_df = evaluate_feature_subsets_on_validation(
        X_train_f,
        X_val_f,
        imp_df,
        k_list=[5, 10, 15],
        random_state=2000
    )

    results_path = os.path.join(OUT_DIR, "validation_selection_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    def _fmt_features(s, max_show=6):
        if not isinstance(s, str) or not s:
            return s
        parts = s.split(",")
        return ",".join(parts[:max_show]) + (f",...(+{len(parts) - max_show} more)" if len(parts) > max_show else "")

    disp = results_df.copy()
    disp["features"] = disp["features"].apply(_fmt_features)
    print("\nValidation selection results:")
    print(disp.to_string(index=False))

    summary_path = os.path.join(OUT_DIR, "validation_selection_results.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Validation selection results:\n")
        f.write(disp.to_string(index=False))
    print(f"Saved: {summary_path}")

    top = imp_df.head(15)
    if not top.empty:
        plt.figure(figsize=(10, 5))
        plt.barh(top["feature"].iloc[::-1], top["objective_increase"].iloc[::-1])
        plt.xlabel("Validation objective increase after permutation")
        plt.title("Permutation importance on validation")
        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, "permutation_importance_validation.png")
        plt.savefig(plot_path, dpi=150)
        plt.show()


if __name__ == "__main__":
    main()