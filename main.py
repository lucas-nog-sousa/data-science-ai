---

**2. `main.py`**

```python
import numpy as np
import pandas as pd
from scipy import stats

def run_analysis():
    # 1. Simulate Synthetic Zendesk Operational Data (Log-Normal Distribution)
    np.random.seed(42)
    n_active, n_churn = 1500, 300

    # Handling time in minutes (Active vs. Churn)
    active_ht = np.random.lognormal(mean=2.1, sigma=0.75, size=n_active) # Median ~8 min
    churn_ht  = np.random.lognormal(mean=2.4, sigma=0.85, size=n_churn)  # Median ~11 min

    # 2. Exploratory Normality & Skewness Diagnosis
    print("--- Distribution Diagnosis ---")
    print(f"Active Group Skewness: {stats.skew(active_ht):.2f}")
    print(f"Churn Group Skewness:  {stats.skew(churn_ht):.2f}")

    # 3. Parametric vs. Non-Parametric Hypothesis Testing
    t_stat, p_val_t = stats.ttest_ind(churn_ht, active_ht, equal_var=False)
    u_stat, p_val_u = stats.mannwhitneyu(churn_ht, active_ht, alternative='two-sided')

    # Rank-Biserial Correlation (Effect Size for Mann-Whitney U)
    n1, n2 = len(churn_ht), len(active_ht)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    print("\n--- Test Comparison ---")
    print(f"Welch's t-test:     p-value = {p_val_t:.5f}")
    print(f"Mann-Whitney U:     p-value = {p_val_u:.5f}")
    print(f"Rank-Biserial Effect Size: {rank_biserial:.3f}")

    # 4. Non-Parametric Bootstrapping for Median Difference (95% CI)
    def bootstrap_median_diff(data1, data2, n_boot=10000, ci=95):
        boot_diffs = np.empty(n_boot)
        for i in range(n_boot):
            sample1 = np.random.choice(data1, size=len(data1), replace=True)
            sample2 = np.random.choice(data2, size=len(data2), replace=True)
            boot_diffs[i] = np.median(sample1) - np.median(sample2)
        
        alpha = (100 - ci) / 2
        lower = np.percentile(boot_diffs, alpha)
        upper = np.percentile(boot_diffs, 100 - alpha)
        return np.median(data1) - np.median(data2), lower, upper

    diff, ci_lower, ci_upper = bootstrap_median_diff(churn_ht, active_ht)
    print("\n--- Bootstrap Results ---")
    print(f"Observed Median Difference: {diff:.2f} minutes")
    print(f"95% Confidence Interval:    [{ci_lower:.2f}, {ci_upper:.2f}] minutes")

if __name__ == "__main__":
    run_analysis()
