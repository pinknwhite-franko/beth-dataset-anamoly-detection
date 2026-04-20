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

FEATURE_GROUPS = {
    "count": [
        "processId_eventId_past_count",
        "processId_past_count",
        "threadId_past_count",
        "eventId_past_count",
        "userId_past_count",
        "parentProcessId_past_count",
        "mountNamespace_past_count",
    ],
    "rarity": [
        "processId_eventId_rarity",
        "processId_rarity",
        "threadId_rarity",
        "eventId_rarity",
        "userId_rarity",
        "parentProcessId_rarity",
        "mountNamespace_rarity",
    ],
    "first_seen": [
        "processId_eventId_is_first_seen",
        "processId_is_first_seen",
        "threadId_is_first_seen",
        "eventId_is_first_seen",
        "userId_is_first_seen",
        "parentProcessId_is_first_seen",
        "mountNamespace_is_first_seen",
    ],
    "scale_other": [
        "stackAddresses_len",
        "stackAddresses_unique_ratio",
        "child_process_spawn_rate_so_far",
    ],
    "binary": [
        "userId_binary",
        "parentUserId_binary",
    ],
    "pass": [
        "argsNum",
        "returnValue",
        "returnValue_is_error",
        "args_has_path",
    ],
}

ALL_CANDIDATE_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]


def read_data(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "datasets", "raw")

    def _load_x(name):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path, header=0)
        return df.drop(columns=[c for c in ["sus", "evil"] if c in df.columns])

    return _load_x("labelled_training_data.csv"), _load_x("labelled_validation_data.csv")


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
    selected = {
        group: [c for c in cols if c in selected_features]
        for group, cols in FEATURE_GROUPS.items()
    }

    log_count_pipeline = Pipeline([
        ("log", FunctionTransformer(
            lambda x: np.log1p(np.clip(x, 0, None)),
            feature_names_out="one-to-one",
            validate=False
        )),
        ("scale", RobustScaler()),
    ])

    scale_pipeline = Pipeline([("scale", RobustScaler())])

    group_transformers = {
        "count": log_count_pipeline,
        "rarity": scale_pipeline,
        "first_seen": "passthrough",
        "scale_other": scale_pipeline,
        "binary": "passthrough",
        "pass": "passthrough",
    }

    transformers = [
        (group, group_transformers[group], cols)
        for group, cols in selected.items()
        if cols
    ]

    if not transformers:
        raise ValueError("No selected features were mapped into the preprocessor.")

    return ColumnTransformer(transformers=transformers)


def validation_metrics(model, X_train, X_val):
    train_scores = -model.score_samples(X_train)
    val_scores = -model.score_samples(X_val)

    val_pred = model.predict(X_val)
    predicted_outliers = int(np.sum(val_pred == -1))
    predicted_inliers = int(np.sum(val_pred == 1))
    val_outlier_rate = float(np.mean(val_pred == -1))

    score_shift = float(abs(train_scores.mean() - val_scores.mean()))
    score_std_gap = float(abs(train_scores.std() - val_scores.std()))
    objective = val_outlier_rate + 0.5 * score_shift + 0.25 * score_std_gap

    return {
        "objective": objective,
        "predicted_outliers": predicted_outliers,
        "predicted_inliers": predicted_inliers,
        "val_outlier_rate": val_outlier_rate,
        "score_shift": score_shift,
        "score_std_gap": score_std_gap,
    }


def fit_iforest_on_features(X_train, selected_features, random_state=2000):
    model = Pipeline([
        ("scaler", build_preprocessor(selected_features)),
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
    model = fit_iforest_on_features(X_train, selected_features, random_state=2000)
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


def evaluate_feature_subsets_on_validation(X_train, X_val, imp_df, k_list, random_state=2000):
    imp_sorted = imp_df.sort_values("objective_increase", ascending=False).reset_index(drop=True)

    def fit_eval(features, scenario):
        model = fit_iforest_on_features(X_train, features, random_state=random_state)
        metrics = validation_metrics(model, X_train[features], X_val[features])
        return {
            "scenario": scenario,
            "n_features": len(features),
            "features": ",".join(features),
            **metrics,
        }

    results = [fit_eval(ALL_CANDIDATE_COLS, "baseline_all")]
    for k in sorted(set(k_list)):
        features = imp_sorted["feature"].iloc[:min(int(k), len(imp_sorted))].tolist()
        results.append(fit_eval(features, f"keep_top_k={k}"))

    return pd.DataFrame(results).sort_values("objective", ascending=True).reset_index(drop=True)


def main():
    X_train, X_val = read_data()

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
    imp_df.to_csv(os.path.join(OUT_DIR, "permutation_importance_validation.csv"), index=False)

    for k in (5, 10, 15):
        imp_df["feature"].iloc[:k].to_frame().to_csv(
            os.path.join(OUT_DIR, f"best_features_val_k{k}.csv"),
            index=False
        )

    print("Evaluating baseline vs reduced feature sets on validation only...")
    results_df = evaluate_feature_subsets_on_validation(
        X_train_f,
        X_val_f,
        imp_df,
        k_list=[5, 10, 15],
        random_state=2000
    )
    results_df.to_csv(os.path.join(OUT_DIR, "validation_selection_results.csv"), index=False)

    print("\nValidation selection results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()