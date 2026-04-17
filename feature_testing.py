"""Feature importance and retrain evaluation.

Run: python feature_testing.py
Outputs: features_data/permutation_importance_full.csv, features_data/retrain_results_full.csv
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from model_class.feature_builder import FeatureBuilder
from model_class.feature_builder_transformer import FeatureBuilderTransformer

OUT_DIR = os.path.join(os.getcwd(), "features_data")


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
    X_test, y_test = _split(_load("labelled_testing_data.csv"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_features(X_train, X_val, X_test_dev, X_test_final):
    trans = FeatureBuilderTransformer(FeatureBuilder(), return_numpy=False)
    return (
        trans.fit_transform(X_train),
        trans.transform(X_val),
        trans.transform(X_test_dev),
        trans.transform(X_test_final),
    )


def safe_roc_auc(y_true, scores):
    y = np.array(y_true).astype(int)
    if len(np.unique(y)) < 2 or np.unique(scores).size < 2:
        return float("nan")
    return roc_auc_score(y, scores)


def permutation_importance(X_train, X_val, X_test, y_test, contamination=0.01, random_state=0):
    """Compute permutation feature importance measured as drop in test ROC AUC."""
    X_fit = pd.concat([X_train, X_val], axis=0)

    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(X_fit)
    baseline_auc = safe_roc_auc(y_test, -clf.decision_function(X_test))

    rng = np.random.RandomState(random_state)
    rows = []
    for col in X_test.columns:
        Xp = X_test.copy()
        Xp[col] = rng.permutation(Xp[col].values)
        auc = safe_roc_auc(y_test, -clf.decision_function(Xp))
        drop = (baseline_auc - auc) if not (np.isnan(baseline_auc) or np.isnan(auc)) else np.nan
        rows.append({"feature": col, "baseline_auc": baseline_auc, "permuted_auc": auc, "auc_drop": drop})

    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)


def retrain_and_evaluate(X_train, X_val, X_test, y_train, y_test, imp_df, k_list, n_estimators=200, random_state=42):
    """Retrain IsolationForest on train+val and evaluate test AUC for each feature subset in k_list."""
    X_fit = pd.concat([X_train, X_val], axis=0)
    all_features = list(X_train.columns)
    contamination = max(0.001, float(np.array(y_train).astype(int).mean()))
    imp_sorted = imp_df.sort_values("auc_drop", ascending=False).reset_index(drop=True)

    def fit_eval(features, scenario):
        scaler = RobustScaler()
        Xf_s = scaler.fit_transform(X_fit[features].values)
        Xt_s = scaler.transform(X_test[features].values)
        clf = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
        clf.fit(Xf_s)
        auc = safe_roc_auc(y_test, -clf.decision_function(Xt_s))
        return {"scenario": scenario, "n_features": len(features), "features": ",".join(features), "auc": auc}

    results = [fit_eval(all_features, "baseline_all")]
    for k in sorted(set(k_list)):
        k = min(int(k), len(imp_sorted))
        features = imp_sorted["feature"].iloc[:k].tolist()
        results.append(fit_eval(features, f"keep_top_k={k}"))

    return pd.DataFrame(results)


def main():
    X_train, y_train, X_val, _y_val, X_test, y_test = read_data()

    # Split labelled test data into:
    # dev = used for feature selection
    # final = used only for final evaluation
    X_test_dev, X_test_final, y_test_dev, y_test_final = train_test_split(
        X_test,
        y_test,
        test_size=0.30,
        stratify=y_test,
        random_state=42
    )

    X_train_f, X_val_f, X_test_dev_f, X_test_final_f = build_features(
        X_train,
        X_val,
        X_test_dev,
        X_test_final
    )

    contamination = max(0.001, float(np.array(y_train).astype(int).mean()))
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Computing permutation importance on dev split...")
    imp_df = permutation_importance(
        X_train_f,
        X_val_f,
        X_test_dev_f,
        y_test_dev,
        contamination=contamination,
        random_state=42
    )

    imp_path = os.path.join(OUT_DIR, "permutation_importance_dev.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"Saved: {imp_path}")

    for k in (5, 10, 15):
        best_path = os.path.join(OUT_DIR, f"best_features_dev_k{k}.csv")
        imp_df["feature"].iloc[:k].to_frame().to_csv(best_path, index=False)
        print(f"Saved top-{k} features: {best_path}")

    print("Retraining and evaluating on final split...")
    n = len(X_train_f.columns)
    k_candidates = sorted({5, 10, 15, max(1, n // 4), max(1, n // 2), max(1, (3 * n) // 4)})

    results_df = retrain_and_evaluate(
        X_train_f,
        X_val_f,
        X_test_final_f,
        y_train,
        y_test_final,
        imp_df,
        k_candidates
    )

    results_path = os.path.join(OUT_DIR, "retrain_results_final.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    def _fmt_features(s, max_show=6):
        if not isinstance(s, str) or not s:
            return s
        parts = s.split(",")
        return ",".join(parts[:max_show]) + (f",...(+{len(parts) - max_show} more)" if len(parts) > max_show else "")

    disp = results_df.copy()
    disp["features"] = disp["features"].apply(_fmt_features)
    print("\nRetrain results:")
    print(disp.to_string(index=False))

    summary_path = os.path.join(OUT_DIR, "retrain_results_final.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Retrain results:\n")
        f.write(disp.to_string(index=False))
    print(f"Saved: {summary_path}")

    print(f"\nAll outputs in: {OUT_DIR}")

    top = imp_df.head(15)
    if not top.empty:
        plt.figure(figsize=(10, 5))
        plt.barh(top["feature"].iloc[::-1], top["auc_drop"].iloc[::-1])
        plt.xlabel("AUC drop (baseline - permuted)")
        plt.title("Permutation importance on dev split")
        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, "permutation_importance_dev.png")
        plt.savefig(plot_path, dpi=150)
        plt.show()


if __name__ == "__main__":
    main()