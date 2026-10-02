"""OLS coefficient-table version of driver_features.py's driver regression, for the two headline
Deception+ components (whiff: rate outcome, timing: magnitude outcome), matching the reference
report's methodology (plain multiple OLS, coefficients + SE + p-values + 95% CI per feature), plus a
Benjamini-Hochberg adjusted q across the 14 features of each model.

Uses the exact same feature set and NaN-handling as driver_features.run_driver_analysis (median
fill, same exclusions), just swapping Ridge+RandomForest for statsmodels OLS so real p-values and
confidence intervals exist to report. Same grouped-by-pitcher rule carried over in spirit: this is
explanatory, not predictive, so no CV split is needed for the coefficient table itself, but we keep
n and report it with the same caveat driver_features.py's own docstring already gives (small n,
season aggregates).

Backs report.pdf (built from report.tex). Run after driver_analysis.py.
"""
import json

import pandas as pd
import statsmodels.api as sm

from driver_features import fdr_adjusted_p_values

LABELS = ["whiff", "timing"]

driver_df = pd.read_csv("output/driver_features.csv")
ps = pd.read_csv("output/pitcher_season.csv")
feature_cols = [c for c in driver_df.columns if c not in ("pitcher", "season")]

results = {}
for label in LABELS:
    target_col = f"{label}_diff_adj_shrunk"
    data = driver_df.merge(ps[["pitcher", "season", target_col]], on=["pitcher", "season"], how="inner")
    data = data.dropna(subset=[target_col])
    data = data.dropna(subset=feature_cols, thresh=len(feature_cols) - 2)
    for c in feature_cols:
        data[c] = data[c].fillna(data[c].median())

    X = sm.add_constant(data[feature_cols])
    y = data[target_col]
    model = sm.OLS(y, X).fit()

    ci = model.conf_int(alpha=0.05)
    q = fdr_adjusted_p_values(data[feature_cols], y)
    rows = []
    for feat in feature_cols:
        rows.append({
            "feature": feat,
            "coef": round(float(model.params[feat]), 6),
            "se": round(float(model.bse[feat]), 6),
            "p": round(float(model.pvalues[feat]), 6),
            "q": round(float(q[feat]), 6),
            "ci_lo": round(float(ci.loc[feat, 0]), 6),
            "ci_hi": round(float(ci.loc[feat, 1]), 6),
        })
    rows.sort(key=lambda r: r["p"])

    results[label] = {
        "n": int(model.nobs),
        "n_features": len(feature_cols),
        "r2": round(float(model.rsquared), 4),
        "adj_r2": round(float(model.rsquared_adj), 4),
        "fvalue": round(float(model.fvalue), 3),
        "f_pvalue": float(model.f_pvalue),
        "df_resid": int(model.df_resid),
        "intercept": round(float(model.params["const"]), 6),
        "rows": rows,
    }
    print(f"\n=== {label} ===  n={results[label]['n']}, R2={results[label]['r2']}, "
          f"adj R2={results[label]['adj_r2']}, F={results[label]['fvalue']} (p={results[label]['f_pvalue']:.3g}), "
          f"df_resid={results[label]['df_resid']}", flush=True)
    for r in rows:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        print(f"  {r['feature']:38s} coef={r['coef']:+.5f}  se={r['se']:.5f}  p={r['p']:.4f}{sig:3s} q={r['q']:.4f} "
              f"CI=[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]", flush=True)

with open("output/driver_ols_report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("\nSaved output/driver_ols_report.json")
