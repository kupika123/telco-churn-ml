# Telco Customer Churn — From a Linear Baseline to a Class-Weighted Neural Network

An end-to-end study of churn prediction on the IBM Telco Customer Churn dataset, carried out in
three stages. The focus is not on maximising a single score, but on making the **model-selection
reasoning explicit**: how class imbalance shifts which metric matters, what a linear baseline can
and cannot do, and whether a neural network earns its extra complexity on tabular data of this size.

---

## Headline results

All numbers are on a held-out test set that was never used for fitting or model selection.

| Stage | Model | Accuracy | Churn recall | Churn F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | Logistic Regression (scikit-learn Pipeline) | 0.807 | 0.567 | 0.609 | 0.842 |
| 2 | Random Forest (regularised) | 0.721 | 0.737 | 0.606 | 0.838 |
| 2 | HistGradientBoosting (GridSearchCV) | 0.801 | 0.529 | 0.585 | 0.847 |
| 3 | Feed-forward ANN, class-weighted loss (PyTorch) | 0.764 | 0.655 | 0.596 | 0.818 |

The most useful comparison here is **not** the accuracy column. Roughly 26.6% of customers churn,
so a model can score well on accuracy while missing most of the customers the business actually
wants to find. Stages 2 and 3 both trade overall accuracy for a substantially higher churn recall,
which is the trade the task calls for.

---

## Why three stages

**Stage 1 — Establish a baseline and expose the real problem.**
A single `Pipeline` of `StandardScaler` + `LogisticRegression`, evaluated with 5-fold
`StratifiedKFold` (mean CV ROC-AUC ≈ 0.846). The test ROC-AUC of 0.842 says the model ranks
churners well, but recall of 0.567 says it misses over 40% of them at the default 0.5 threshold.
That gap between "ranks well" and "decides well" is what the next two stages attack.

**Stage 2 — Do tree ensembles fix it? (team coursework)**
Two regularised tree pipelines were compared: a Random Forest with `class_weight="balanced"`, and a
`HistGradientBoostingClassifier` tuned over 16 configurations with `GridSearchCV` (scoring on F1,
not accuracy). Boosting won on accuracy (0.801 vs 0.721) and ROC-AUC (0.847 vs 0.838); Random
Forest won on churn recall (0.737 vs 0.529), overall F1 (0.606 vs 0.585), train–validation gap
(0.007 vs 0.027) and training time (101 ms vs 265 ms). Random Forest was selected — on an
imbalanced target with a retention use case, recall and generalisation stability outweigh a 8-point
accuracy lead driven by the majority class.

**Stage 3 — Does a neural network earn its complexity?**
Three feed-forward architectures (7,425 / 31,361 / 53,889 trainable parameters) trained in PyTorch
with `BCEWithLogitsLoss` and a positive-class weight of 1.881 (errors on churners cost ≈2.8× more),
dropout, weight decay, and early stopping on validation F1. All three landed within 0.61–0.64 best
validation F1 despite a 7× difference in parameter count — evidence that the dataset does not
reward extra capacity. The selected model lifts churn recall to 0.655, but its ROC-AUC (0.818) does
not beat the linear baseline (0.842). **The honest conclusion is that on this dataset the ANN buys
recall through its weighted loss, not through superior representation learning.**

---

## Repository layout

```
.
├── notebooks/
│   ├── 01_logistic_regression_pipeline.ipynb   # Stage 1: baseline + 5-fold CV
│   └── 03_pytorch_ann.ipynb                    # Stage 3: architecture search + final model
├── reports/
│   ├── 01_baseline_report.pdf
│   └── 03_ann_report.pdf
├── requirements.txt
└── README.md
```

Stage 2 was a group coursework and its code and report are not redistributed here; the comparison
in the table above is reproduced from the figures I produced for that stage. See *Contributions*.

---

## Data

The dataset is **not** included in this repository. Download
`WA_Fn-UseC_-Telco-Customer-Churn.csv` from
[Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place it in the repository root (or edit the path in the first cells).

7,043 rows, 20 input variables, binary `Churn` target. `TotalCharges` contains 11 blank strings;
these rows are coerced to `NaN` and dropped, leaving 7,032 customers. `customerID` is dropped as a
non-predictive identifier.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Developed on Python 3.13. Stage 3 runs on CPU in a few minutes; it will use CUDA automatically if
available. All random seeds are fixed to 42.

---

## Preventing leakage

Worth stating explicitly, because it is easy to get wrong and invisible when you do:

- Stage 1 puts scaling **inside** the `Pipeline`, so `StandardScaler` is refitted within each CV
  fold rather than on the full training set.
- Stage 3 splits 60/20/20 with stratification, then fits the `ColumnTransformer`
  (`StandardScaler` + `OneHotEncoder(handle_unknown="ignore")`) on the training portion **only**
  before transforming validation and test.
- Architecture selection uses validation F1 only. The test set is touched once, at the end.

---

## Efficiency

Inference latency for the final ANN was measured over the **entire 1,407-sample test set**, repeated
five times: mean total latency 0.020 s (σ = 0.002 s), i.e. roughly 71,500 samples/second on the
hardware used. Stage 1's full 5-fold cross-validation completed in about 0.13 s using ~5.6 MB of
additional memory. Neither model is anywhere near a deployment bottleneck at this data scale, which
is itself part of the argument: when compute is not the constraint, model choice should be driven by
the error profile rather than by speed.

---

## Limitations

- All metrics are reported at a fixed 0.5 decision threshold. Given the strong ROC-AUC, tuning the
  threshold against actual false-positive/false-negative costs would likely help more than any of
  the model changes above.
- No feature engineering: the models consume raw one-hot encoded columns. Interaction terms or
  domain summaries (service count, payment stability) were not explored.
- The ANN architecture search covered three hand-designed configurations, not a systematic sweep.
- Single dataset, single time slice; no test of temporal drift.

---

## Contributions

Stages 1 and 3 are entirely my own work. Stage 2 was a six-person group coursework in which I was
responsible for the **comparative analysis of the two tuned pipelines and the written justification
for the final pipeline selection**; the Random Forest and HistGradientBoosting implementations and
their individual metric write-ups were done by other group members. Their code and the group report
are not included here.

---

## Note

This work originated as coursework for a taught postgraduate course. It is published after
assessment, for reference and discussion. Please do not submit any part of it as your own work.
