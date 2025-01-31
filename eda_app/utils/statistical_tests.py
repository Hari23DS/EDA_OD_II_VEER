import pandas as pd
import numpy as np
from scipy.stats import shapiro, spearmanr, pearsonr, f_oneway
from statsmodels.stats.multitest import multipletests

def feature_selection_stats(data, target_col):
    """
    Performs statistical tests based on the data type of the target column.
    Identifies suggested features where p_value < 0.05.
    """
    
    numeric_columns = data.select_dtypes(include=np.number).columns
    target_is_numeric = target_col in numeric_columns
    results = {}
    suggested_features = []

    def normality_test(series):
        """Check if the target variable follows a normal distribution."""
        series = series.dropna()
        return shapiro(series)[1] > 0.05  # Returns True if normal

    is_normal = normality_test(data[target_col])

    for feature in data.columns:
        if feature == target_col:
            continue

        if target_is_numeric and feature in numeric_columns:
            if is_normal:
                corr, p_value = pearsonr(data[feature].dropna(), data[target_col].dropna())
                test_name = "Pearson"
            else:
                corr, p_value = spearmanr(data[feature].dropna(), data[target_col].dropna())
                test_name = "Spearman"

            results[feature] = {
                "test": test_name,
                "p_value": p_value,
                "correlation": corr
            }

        elif not target_is_numeric and feature in numeric_columns:
            try:
                stat, p_value = f_oneway(*[data[data[target_col] == cat][feature] for cat in data[target_col].unique()])
                results[feature] = {"test": "ANOVA", "p_value": p_value}
            except:
                continue  # Skip if the test cannot be performed

    # Apply multiple testing correction (FDR)
    p_values = [res["p_value"] for res in results.values()]
    adjusted_p_values = multipletests(p_values, method="fdr_bh")[1]

    for i, feature in enumerate(results):
        results[feature]["adjusted_p_value"] = adjusted_p_values[i]
        if adjusted_p_values[i] < 0.05:
            suggested_features.append(feature)

    return {
        "statistical_results": results,
        "suggested_features": suggested_features
    }

