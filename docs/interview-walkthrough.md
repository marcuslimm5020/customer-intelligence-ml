# Interview walkthrough

## A 60–90 second project pitch

> I built an end-to-end customer purchase propensity project using the UCI Online Retail dataset. The business question is: among customers who purchased recently, who is most likely to purchase again in the next 30 days? I created monthly customer snapshots using the prior 90 days of activity, then compared Logistic Regression, Random Forest, and XGBoost with chronological validation. I selected models using average precision and kept November 2011 as a final holdout. XGBoost reached 0.675 average precision on that holdout; in a top-20%-capacity scenario, 74.8% of contacted customers purchased. I also checked calibration and feature contributions. The model underestimates November purchase rates, and the dataset covers only one retailer and one test month, so I present this as a ranking demonstration rather than a production forecast.

## Five-minute walkthrough

### 1. Start with the decision

The model is meant to help prioritize outreach when a business cannot contact every active customer. It ranks eligible customers; a marketing team would still need to choose a contact capacity and weigh campaign costs.

### 2. Explain how the label works

For a snapshot cutoff date, use transactions from the preceding 90 days to calculate features. The target is whether the customer makes a valid purchase in the next 30 days. Only customers with activity in the observation window are included. See `src/feature_engineering.py`.

### 3. Show the data and leakage choices

The raw data has over half a million invoice lines. The cleaning step removes exact duplicates, cancellation invoices, missing customer IDs, and non-positive quantity or price lines. Features stop strictly before the cutoff; future transactions are used only for the label. See `src/data_processing.py` and `src/feature_engineering.py`.

### 4. Describe the validation design

The split follows time: March–August for training, September–October for validation, and November for the final holdout. Validation average precision chooses the model. After selection, the model is refit on training plus validation before the one-time holdout evaluation. See `src/train.py` and `docs/data-and-validation-design.md`.

### 5. Interpret the result without overselling it

XGBoost's validation average precision was 0.6134, only slightly above Logistic Regression at 0.6130. On the November holdout, average precision was 0.6754; the top-ranked 20% had 74.8% precision and 34.9% recall. The predictions were under-calibrated, and November had a higher positive rate than October. This makes ranking and calibration separate discussion points. See `docs/initial-model-results.md` and `docs/evaluation-and-explainability.md`.

### 6. Finish with limitations and next research steps

This is one retailer and about one year of transactions, with one final holdout month. Monthly snapshots reuse customers and have overlapping feature windows. A stronger study would use multiple retailers or years, more untouched time periods, segment-level evaluation, and a calibration set distinct from the final test.

## Suggested code tour

| Topic | File | What to point out |
|---|---|---|
| Cleaning | `src/data_processing.py` | Explicit schema checks and auditable sale-line filters |
| Label/features | `src/feature_engineering.py` | Time boundaries for the 90-day observation and 30-day outcome windows |
| Dataset creation | `src/prepare_data.py` | Rebuildable snapshots and configurable source/date inputs |
| Modeling | `src/train.py` | Preprocessing inside sklearn pipelines, chronological split, validation selection, holdout separation |
| Evaluation | `src/evaluate.py` | Average precision, calibration, contact-capacity scenarios, and SHAP |
| API | `src/api.py` | Typed request validation and model-backed inference |

## Questions an interviewer may ask

**Why average precision?**

The goal is to rank likely purchasers under limited outreach capacity. Average precision summarizes precision and recall across score thresholds and is more informative than accuracy for this ranking problem. The top-k analysis connects model ranking to a concrete contact-capacity scenario.

**Why not use a random split?**

A random split could mix earlier and later behavior and make evaluation less representative of predicting future months. Chronological splits preserve the intended direction of time and make leakage easier to reason about.

**Why is the November result not proof of future performance?**

It is a single month from one historical retailer. The purchase rate also rose sharply in November, so seasonal effects may explain part of the result. More time periods and external validation are needed.

**Would you deploy this score as a probability?**

Not as-is. The November scores were under-calibrated. I would treat them as ranking scores until calibration is assessed on additional time periods and campaign costs/capacity define a useful decision threshold.

**Why XGBoost if Logistic Regression is almost tied?**

XGBoost had the best validation average precision, but the 0.0004 difference is small and may not be meaningful with only two validation months. I would report the near tie and test stability across more periods; if performance stayed comparable, the simpler model could be preferable.

## Live demo sequence

1. Show the README project summary and the 90-day/30-day prediction definition.
2. Show the chronological split and model comparison table.
3. Show the campaign-capacity table and calibration caveat.
4. If the local environment and dataset are ready, run `bash scripts/run_demo.sh` to rebuild and show all results in one sequence.
5. Optionally start `uvicorn src.api:app --reload` and make one request through `/docs`.

Run outputs are ignored by Git, so the data file and local model artifacts must exist for a live code demo. The documented metrics and findings are included in the repo for a source-only walkthrough.
