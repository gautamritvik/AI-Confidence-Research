"""
Step 4: Statistical analysis of confident failures.

Analyses:
  1. Descriptive statistics (accuracy, confidence, ECE per model)
  2. Confident failure rates by question feature
  3. Mixed-effects logistic regression (RQ1 & RQ3)
  4. Per-model interaction analysis (RQ2)
  5. Random forest feature importance
  6. Per-domain ECE analysis
  7. Calibration curves

Input:  data/questions_annotated.parquet + data/all_responses.parquet
Output: results/*.csv, figures/*.png
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR, CONFIDENT_THRESHOLD

warnings.filterwarnings("ignore", category=FutureWarning)

PREDICTOR_COLS = [
    "question_length", "has_negation", "has_numerical",
    "option_similarity", "reasoning_steps", "has_misleading_premise",
]
CATEGORICAL_PREDICTORS = ["domain_category"]


def load_data() -> pd.DataFrame:
    """Load and merge question features with model responses."""
    questions = pd.read_parquet(DATA_DIR / "questions_annotated.parquet")
    responses = pd.read_parquet(DATA_DIR / "all_responses.parquet")

    # Merge
    df = responses.merge(questions, on="question_id", how="inner")

    # Filter out parse errors and invalid confidence
    before = len(df)
    df = df[
        (df["model_answer"] != "PARSE_ERROR") &
        (df["model_answer"] != "ERROR") &
        (df["model_confidence"] >= 0) &
        (df["model_confidence"] <= 100)
    ]
    print(f"Loaded {len(df)} valid responses (dropped {before - len(df)} invalid)")

    # Create key derived variables
    df["confident_failure"] = (
        (df["model_confidence"] >= CONFIDENT_THRESHOLD) &
        (df["is_correct"] == 0)
    ).astype(int)

    df["miscalibration"] = abs(df["model_confidence"] / 100.0 - df["is_correct"])

    df["overconfident"] = (
        df["model_confidence"] / 100.0 > df["is_correct"]
    ).astype(int)

    return df


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray,
                n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if i == n_bins - 1:
            mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return ece


def analysis_1_descriptive(df: pd.DataFrame):
    """Overall accuracy, confidence, ECE per model."""
    print("\n" + "="*70)
    print("ANALYSIS 1: Descriptive Statistics")
    print("="*70)

    results = []
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        acc = mdf["is_correct"].mean()
        avg_conf = mdf["model_confidence"].mean()
        med_conf = mdf["model_confidence"].median()
        conf_norm = mdf["model_confidence"].values / 100.0
        ece = compute_ece(conf_norm, mdf["is_correct"].values)
        cf_rate = mdf["confident_failure"].mean()
        cf_count = mdf["confident_failure"].sum()
        n = len(mdf)

        results.append({
            "model": model, "n": n, "accuracy": acc,
            "avg_confidence": avg_conf, "median_confidence": med_conf,
            "ECE": ece, "confident_failure_rate": cf_rate,
            "confident_failure_count": cf_count,
        })
        print(f"\n  {model}:")
        print(f"    N = {n}")
        print(f"    Accuracy: {acc:.3f}")
        print(f"    Avg Confidence: {avg_conf:.1f}")
        print(f"    Median Confidence: {med_conf:.0f}")
        print(f"    ECE: {ece:.4f}")
        print(f"    Confident failures: {cf_count} ({100*cf_rate:.1f}%)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "descriptive_stats.csv", index=False)
    return results_df


def analysis_2_bivariate(df: pd.DataFrame):
    """Confident failure rates by each predictor variable."""
    print("\n" + "="*70)
    print("ANALYSIS 2: Confident Failure Rates by Question Feature")
    print("="*70)

    all_tables = []

    # Binary predictors
    for col in ["has_negation", "has_numerical", "has_misleading_premise"]:
        if col not in df.columns:
            continue
        table = df.groupby(col).agg(
            n=("confident_failure", "count"),
            cf_rate=("confident_failure", "mean"),
            avg_conf=("model_confidence", "mean"),
            accuracy=("is_correct", "mean"),
        ).round(4)
        print(f"\n  {col}:")
        print(table.to_string())
        table["feature"] = col
        all_tables.append(table.reset_index())

    # Ordinal: reasoning_steps
    if "reasoning_steps" in df.columns:
        table = df.groupby("reasoning_steps").agg(
            n=("confident_failure", "count"),
            cf_rate=("confident_failure", "mean"),
            avg_conf=("model_confidence", "mean"),
            accuracy=("is_correct", "mean"),
        ).round(4)
        print(f"\n  reasoning_steps:")
        print(table.to_string())

    # Categorical: domain
    table = df.groupby("domain_category").agg(
        n=("confident_failure", "count"),
        cf_rate=("confident_failure", "mean"),
        avg_conf=("model_confidence", "mean"),
        accuracy=("is_correct", "mean"),
    ).round(4).sort_values("cf_rate", ascending=False)
    print(f"\n  domain_category:")
    print(table.to_string())

    # Continuous: question_length and option_similarity quartiles
    for col in ["question_length", "option_similarity"]:
        if col not in df.columns:
            continue
        df[f"{col}_quartile"] = pd.qcut(df[col], 4, labels=["Q1", "Q2", "Q3", "Q4"],
                                         duplicates="drop")
        table = df.groupby(f"{col}_quartile").agg(
            n=("confident_failure", "count"),
            cf_rate=("confident_failure", "mean"),
            avg_conf=("model_confidence", "mean"),
            accuracy=("is_correct", "mean"),
        ).round(4)
        print(f"\n  {col} (quartiles):")
        print(table.to_string())


def analysis_3_logistic_regression(df: pd.DataFrame):
    """Mixed-effects-style logistic regression predicting confident failure."""
    print("\n" + "="*70)
    print("ANALYSIS 3: Logistic Regression — Predicting Confident Failure")
    print("="*70)

    # Prepare data
    model_df = df.copy()

    # Create dummy variables for domain and model
    model_df = pd.get_dummies(model_df, columns=["domain_category"], 
                               drop_first=True, dtype=int)
    model_df = pd.get_dummies(model_df, columns=["model"],
                               drop_first=True, dtype=int)

    # Build predictor list
    domain_dummies = [c for c in model_df.columns if c.startswith("domain_category_")]
    model_dummies = [c for c in model_df.columns if c.startswith("model_")]
    
    predictors = PREDICTOR_COLS + domain_dummies + model_dummies
    
    # Filter to valid predictors that exist in the data
    predictors = [p for p in predictors if p in model_df.columns]

    # Drop rows with missing values in predictors
    analysis_df = model_df[predictors + ["confident_failure"]].dropna()
    
    X = analysis_df[predictors]
    y = analysis_df["confident_failure"]

    print(f"  N = {len(analysis_df)}, "
          f"events = {y.sum()} ({100*y.mean():.1f}%)")

    # Fit logistic regression
    X_const = sm.add_constant(X)
    try:
        logit_model = sm.Logit(y, X_const).fit(disp=0, maxiter=100)
        print("\n  Logistic Regression Results:")
        print(logit_model.summary2().tables[1].to_string())

        # Pseudo R-squared
        print(f"\n  Pseudo R² (McFadden): {logit_model.prsquared:.4f}")
        print(f"  AIC: {logit_model.aic:.1f}")
        print(f"  BIC: {logit_model.bic:.1f}")

        # Save coefficients
        coef_df = pd.DataFrame({
            "variable": logit_model.params.index,
            "coefficient": logit_model.params.values,
            "std_error": logit_model.bse.values,
            "z_value": logit_model.tvalues.values,
            "p_value": logit_model.pvalues.values,
            "odds_ratio": np.exp(logit_model.params.values),
        })
        coef_df.to_csv(RESULTS_DIR / "logistic_regression_coefficients.csv",
                        index=False)
        print(f"\n  Coefficients saved to logistic_regression_coefficients.csv")

    except Exception as e:
        print(f"  Logistic regression failed: {e}")


def analysis_4_model_interactions(df: pd.DataFrame):
    """Test whether different models have different failure profiles."""
    print("\n" + "="*70)
    print("ANALYSIS 4: Model-Specific Failure Profiles (RQ2)")
    print("="*70)

    models = df["model"].unique()
    per_model_results = []

    for model in models:
        mdf = df[df["model"] == model].copy()
        mdf = pd.get_dummies(mdf, columns=["domain_category"],
                              drop_first=True, dtype=int)
        
        domain_dummies = [c for c in mdf.columns if c.startswith("domain_category_")]
        predictors = PREDICTOR_COLS + domain_dummies
        predictors = [p for p in predictors if p in mdf.columns]

        analysis_df = mdf[predictors + ["confident_failure"]].dropna()
        X = analysis_df[predictors]
        y = analysis_df["confident_failure"]

        if y.sum() < 10:
            print(f"  {model}: too few confident failures ({y.sum()}), skipping")
            continue

        X_const = sm.add_constant(X)
        try:
            logit = sm.Logit(y, X_const).fit(disp=0, maxiter=100)
            
            print(f"\n  {model} (N={len(analysis_df)}, "
                  f"CF={y.sum()}, R²={logit.prsquared:.4f}):")

            # Show significant predictors
            sig = logit.pvalues[logit.pvalues < 0.05]
            for var in sig.index:
                if var == "const":
                    continue
                coef = logit.params[var]
                p = logit.pvalues[var]
                odds = np.exp(coef)
                direction = "+" if coef > 0 else "-"
                print(f"    {direction} {var}: OR={odds:.3f}, p={p:.4f}")

            per_model_results.append({
                "model": model,
                "n": len(analysis_df),
                "cf_count": int(y.sum()),
                "pseudo_r2": logit.prsquared,
            })

        except Exception as e:
            print(f"  {model}: regression failed: {e}")

    if per_model_results:
        pd.DataFrame(per_model_results).to_csv(
            RESULTS_DIR / "per_model_regression_summary.csv", index=False
        )


def analysis_5_random_forest(df: pd.DataFrame):
    """Random forest for non-linear feature importance."""
    print("\n" + "="*70)
    print("ANALYSIS 5: Random Forest Feature Importance")
    print("="*70)

    # Encode categoricals
    model_df = df.copy()
    le_domain = LabelEncoder()
    model_df["domain_encoded"] = le_domain.fit_transform(model_df["domain_category"])
    le_model = LabelEncoder()
    model_df["model_encoded"] = le_model.fit_transform(model_df["model"])

    feature_cols = PREDICTOR_COLS + ["domain_encoded", "model_encoded"]
    feature_cols = [c for c in feature_cols if c in model_df.columns]

    analysis_df = model_df[feature_cols + ["confident_failure"]].dropna()
    X = analysis_df[feature_cols]
    y = analysis_df["confident_failure"]

    # Fit Random Forest
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  Feature Importance (Gini):")
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"    {row['feature']:30s} {row['importance']:.4f}  {bar}")

    importance.to_csv(RESULTS_DIR / "random_forest_importance.csv", index=False)

    # Cross-validated AUC
    cv_auc = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
    print(f"\n  5-Fold CV AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

    # Gradient Boosting for comparison
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_leaf=20, random_state=42
    )
    cv_auc_gb = cross_val_score(gb, X, y, cv=5, scoring="roc_auc")
    print(f"  5-Fold CV AUC (Gradient Boosting): "
          f"{cv_auc_gb.mean():.4f} (+/- {cv_auc_gb.std():.4f})")


def analysis_6_domain_ece(df: pd.DataFrame):
    """Per-domain and per-subject ECE analysis."""
    print("\n" + "="*70)
    print("ANALYSIS 6: Per-Domain ECE")
    print("="*70)

    domain_results = []
    for domain in df["domain_category"].unique():
        ddf = df[df["domain_category"] == domain]
        conf = ddf["model_confidence"].values / 100.0
        acc = ddf["is_correct"].values
        ece = compute_ece(conf, acc)
        domain_results.append({
            "domain": domain, "n": len(ddf),
            "accuracy": acc.mean(), "avg_confidence": conf.mean(),
            "ECE": ece, "cf_rate": ddf["confident_failure"].mean(),
        })

    domain_df = pd.DataFrame(domain_results).sort_values("ECE", ascending=False)
    print(domain_df.to_string(index=False))
    domain_df.to_csv(RESULTS_DIR / "domain_ece.csv", index=False)

    # Per-subject ECE (MMLU only)
    mmlu = df[df["dataset"] == "mmlu"]
    if len(mmlu) > 0:
        subject_results = []
        for subj in mmlu["subject"].unique():
            sdf = mmlu[mmlu["subject"] == subj]
            if len(sdf) < 20:
                continue
            conf = sdf["model_confidence"].values / 100.0
            acc = sdf["is_correct"].values
            ece = compute_ece(conf, acc)
            subject_results.append({
                "subject": subj, "n": len(sdf),
                "accuracy": acc.mean(), "avg_confidence": conf.mean(),
                "ECE": ece, "cf_rate": sdf["confident_failure"].mean(),
            })

        subj_df = pd.DataFrame(subject_results).sort_values("ECE", ascending=False)
        subj_df.to_csv(RESULTS_DIR / "subject_ece.csv", index=False)
        print(f"\n  Top 10 worst-calibrated MMLU subjects:")
        print(subj_df.head(10).to_string(index=False))


def plot_figures(df: pd.DataFrame):
    """Generate all figures for the paper."""
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Calibration curves per model ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    models = df["model"].unique()
    for ax, model in zip(axes.flat, models):
        mdf = df[df["model"] == model]
        conf = mdf["model_confidence"].values / 100.0
        acc = mdf["is_correct"].values

        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accs = []
        bin_counts = []

        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (conf >= bin_boundaries[i]) & (conf < bin_boundaries[i+1])
            else:
                mask = (conf >= bin_boundaries[i]) & (conf <= bin_boundaries[i+1])
            if mask.sum() > 0:
                bin_centers.append(conf[mask].mean())
                bin_accs.append(acc[mask].mean())
                bin_counts.append(mask.sum())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color="steelblue",
               edgecolor="black", linewidth=0.5)
        ece = compute_ece(conf, acc)
        ax.set_title(f"{model}\nECE = {ece:.4f}", fontsize=12)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_calibration_curves.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print("  Saved fig1_calibration_curves.png")

    # ── Figure 2: Confident failure rate by feature ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 2a: By reasoning steps
    ax = axes[0, 0]
    if "reasoning_steps" in df.columns:
        rates = df.groupby("reasoning_steps")["confident_failure"].mean()
        ax.bar(rates.index.astype(str), rates.values, color="coral",
               edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Reasoning Steps")
        ax.set_ylabel("Confident Failure Rate")
        ax.set_title("By Reasoning Steps")

    # 2b: By has_negation
    ax = axes[0, 1]
    rates = df.groupby("has_negation")["confident_failure"].mean()
    ax.bar(["No Negation", "Has Negation"], rates.values, color="coral",
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Confident Failure Rate")
    ax.set_title("By Negation")

    # 2c: By has_misleading_premise
    ax = axes[0, 2]
    if "has_misleading_premise" in df.columns:
        rates = df.groupby("has_misleading_premise")["confident_failure"].mean()
        ax.bar(["No Misleading", "Misleading"], rates.values, color="coral",
               edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Confident Failure Rate")
        ax.set_title("By Misleading Premise")

    # 2d: By domain
    ax = axes[1, 0]
    rates = df.groupby("domain_category")["confident_failure"].mean().sort_values()
    ax.barh(rates.index, rates.values, color="coral",
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Confident Failure Rate")
    ax.set_title("By Domain")

    # 2e: By option similarity quartile
    ax = axes[1, 1]
    if "option_similarity" in df.columns:
        df["os_q"] = pd.qcut(df["option_similarity"], 4,
                              labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"],
                              duplicates="drop")
        rates = df.groupby("os_q")["confident_failure"].mean()
        ax.bar(rates.index.astype(str), rates.values, color="coral",
               edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Option Similarity")
        ax.set_ylabel("Confident Failure Rate")
        ax.set_title("By Option Similarity")

    # 2f: By has_numerical
    ax = axes[1, 2]
    rates = df.groupby("has_numerical")["confident_failure"].mean()
    ax.bar(["No Numbers", "Has Numbers"], rates.values, color="coral",
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Confident Failure Rate")
    ax.set_title("By Numerical Content")

    plt.suptitle("Confident Failure Rate by Question Feature", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_cf_rate_by_feature.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print("  Saved fig2_cf_rate_by_feature.png")

    # ── Figure 3: Confidence distributions (correct vs. incorrect) ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, model in zip(axes.flat, models):
        mdf = df[df["model"] == model]
        ax.hist(mdf[mdf["is_correct"] == 1]["model_confidence"],
                bins=20, alpha=0.6, label="Correct", color="green", density=True)
        ax.hist(mdf[mdf["is_correct"] == 0]["model_confidence"],
                bins=20, alpha=0.6, label="Incorrect", color="red", density=True)
        ax.set_xlabel("Confidence (0-100)")
        ax.set_ylabel("Density")
        ax.set_title(model)
        ax.legend()

    plt.suptitle("Confidence Distributions: Correct vs. Incorrect", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_confidence_distributions.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print("  Saved fig3_confidence_distributions.png")

    # ── Figure 4: Feature importance (Random Forest) ──
    importance_path = RESULTS_DIR / "random_forest_importance.csv"
    if importance_path.exists():
        imp = pd.read_csv(importance_path).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp["feature"], imp["importance"], color="steelblue",
                edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Feature Importance (Gini)")
        ax.set_title("Random Forest: Predictors of Confident Failure")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig4_feature_importance.png", dpi=300,
                    bbox_inches="tight")
        plt.close()
        print("  Saved fig4_feature_importance.png")

    # ── Figure 5: Heatmap — CF rate by model × domain ──
    pivot = df.pivot_table(
        values="confident_failure", index="domain_category",
        columns="model", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                linewidths=0.5)
    ax.set_title("Confident Failure Rate: Model × Domain")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_model_domain_heatmap.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print("  Saved fig5_model_domain_heatmap.png")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print(f"\n  Models: {df['model'].unique().tolist()}")
    print(f"  Datasets: {df['dataset'].unique().tolist()}")
    print(f"  Total valid responses: {len(df)}")
    print(f"  Confident failures: {df['confident_failure'].sum()} "
          f"({100*df['confident_failure'].mean():.1f}%)")

    analysis_1_descriptive(df)
    analysis_2_bivariate(df)
    analysis_3_logistic_regression(df)
    analysis_4_model_interactions(df)
    analysis_5_random_forest(df)
    analysis_6_domain_ece(df)
    plot_figures(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
