# Initial model comparison

Run with `python -m src.train` after preparing the customer snapshots. The script fits imputers only on the training rows, selects a model by pooled September–October average precision, refits the selected estimator on March–October snapshots, and evaluates November once as the final holdout.

## Results

| Model | Validation average precision | Validation precision, top 20% | Validation Brier score |
|---|---:|---:|---:|
| Logistic Regression | 0.6130 | 0.669 | 0.203 |
| Random Forest | 0.6020 | 0.675 | 0.206 |
| XGBoost | **0.6134** | 0.665 | **0.202** |

XGBoost was selected on validation average precision. On the November holdout (2,461 eligible customers; 42.9% purchased during the next 30 days), it achieved:

- Average precision: 0.675
- Precision among the top-ranked 20%: 0.748
- Precision at a reference probability threshold of 0.5: 0.793
- Recall at 0.5: 0.294
- Brier score: 0.220

The 0.5 threshold is only a reference. A real campaign threshold should reflect contact capacity and the relative costs of missed purchasers and unnecessary outreach. Brier score is included as an overall probability-quality measure; probability calibration still needs fuller inspection.

## Runtime note and limitations

XGBoost 2.1.4 now runs on this Mac. Homebrew had no prebuilt `libomp` bottle for this system, so a compatible Apple Silicon `libomp.dylib` from the conda-forge `llvm-openmp` package was installed at the Homebrew runtime path. If XGBoost cannot load on another machine, the script falls back to scikit-learn Histogram Gradient Boosting.

On this Python 3.9 / NumPy 2 environment, Logistic Regression also emits NumPy runtime warnings during probability scoring, although its resulting probabilities and reported metrics are finite. Treat those baseline metrics as provisional until the behavior is resolved or reproduced in a clean environment.

November is one test month and has a much higher purchase rate than the validation months. The apparent lift should be treated as a result for this single historical period, not as a stable forecast of future performance. The output metrics and predictions are regenerated under `artifacts/` and are excluded from Git.
