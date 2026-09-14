# Performance modeling

This stage predicts `successful_performance` for a candidate-job pair. The model
uses content-match features plus two noisy pre-work signals available to every
candidate: interview and assessment scores. It never uses relevance scores,
hidden simulator traits, true success probabilities, candidate names, or outcomes
as features.

The embedding representation keeps cosine similarity and adds 64 PCA absolute-
difference features. PCA is fitted only on normalized candidate embeddings, then
the same transformation is applied to candidates and jobs. This gives LightGBM
more semantic detail without passing all 1,536 embedding dimensions.

`successful_performance` is currently a simulated potential outcome available for
the sampled labeled pairs. This is an oracle content-modeling baseline, not yet a
claim that counterfactual outcomes would be observable in a real marketplace.
Later selected/observed data must account for delayed labels and selection bias.

Build the bounded training table:

```bash
PYTHONPATH=matching-pipeline/modeling/src \
  .venv/bin/python matching-pipeline/modeling/scripts/assemble_dataset.py
```

Train and run five-fold job-grouped cross-validation:

```bash
PYTHONPATH=matching-pipeline/modeling/src \
  .venv/bin/python matching-pipeline/modeling/scripts/train_model.py
```

The default trains the stronger base feature set. Reproduce the PCA comparison
with `--feature-set pca --output-dir matching-pipeline/modeling/artifacts/model_pca`.

Compare the out-of-fold model ranking against cosine similarity alone:

```bash
PYTHONPATH=matching-pipeline/modeling/src \
  .venv/bin/python matching-pipeline/modeling/scripts/evaluate_ranking.py
```

Train the directly comparable XGBoost baseline:

```bash
PYTHONPATH=matching-pipeline/modeling/src \
  .venv/bin/python matching-pipeline/modeling/scripts/train_xgboost.py
```

Tune LightGBM on development jobs and evaluate the winner once on an untouched
20% job holdout:

```bash
PYTHONPATH=matching-pipeline/modeling/src \
  .venv/bin/python matching-pipeline/modeling/scripts/tune_lightgbm.py
```

Check whether adding more labeled jobs is likely to help:

```bash
PYTHONPATH=matching-pipeline/modeling/src \
  .venv/bin/python matching-pipeline/modeling/scripts/evaluate_learning_curve.py
```

The test uses one fixed 20% job holdout and compares repeated training runs with
10%, 25%, 50%, and 100% of the remaining jobs.

Generated datasets, predictions, reports, and models are stored under `artifacts/`
and are gitignored. Grouping folds by job tests whether the model generalizes to
jobs it did not encounter during training.
