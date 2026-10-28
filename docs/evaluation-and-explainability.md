# Evaluation and explainability

Run `python -m src.evaluate` after `python -m src.prepare_data` and `python -m src.train`. This evaluates the saved final XGBoost model on the November holdout and writes calibration, campaign-capacity and SHAP outputs into the Git-ignored `artifacts/` folder.

## November holdout findings

- 2,461 customers were eligible; 42.9% purchased in the next 30 days.
- Average precision was 0.6754 and Brier score was 0.2195.
- Mean predicted probability was 0.3336, below the observed 0.4291 purchase rate. Ten-quantile expected calibration error was 0.0955. The model ranks customers usefully but underestimates November purchase probabilities overall.

The campaign-capacity analysis shows the ranking tradeoff:

| Contact capacity | Customers | Precision | Recall | Lift over all eligible customers |
|---:|---:|---:|---:|---:|
| Top 5% | 124 | 93.5% | 11.0% | 2.18x |
| Top 10% | 247 | 85.8% | 20.1% | 2.00x |
| Top 20% | 493 | 74.8% | 34.9% | 1.74x |
| Top 30% | 739 | 66.6% | 46.6% | 1.55x |
| Top 50% | 1,231 | 56.1% | 65.4% | 1.31x |

These describe November 2011 only. The population purchase rate rose from 35.2% in October to 42.9% in November, so the test month may include seasonal behavior absent from the shorter training history. Calibration should be revisited with more time periods; do not fit a calibrator on the November test labels.

## SHAP summary

Mean absolute SHAP values identify `order_count` as the strongest driver of model output (0.447), followed by total spend (0.136), recency in days (0.131), quantity (0.089), and the missing-value indicator for days between orders (0.074). In the summary plot, customers with more orders generally receive a higher propensity score, while customers with only one recent order tend to receive lower scores. SHAP describes associations encoded by the model; it does not establish that changing a feature would cause a purchase.

The generated `shap_summary.png`, `shap_feature_importance.csv`, customer-level `november_shap_contributions.csv`, calibration chart, and precision-recall chart are local analysis artifacts and are excluded from Git.
