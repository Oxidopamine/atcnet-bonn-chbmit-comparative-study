"""Single-class-safe binary metrics and across-fold aggregation for CHB-MIT LOPO.

A LOPO test fold can hold one class only (e.g. a subject without seizure segments). Quantities
that need the missing class are NaN ("undefined"), never an exception and never a misleading
default (MCC = 0 or macro-F1 = 0.5). Aggregates report how many folds define each metric.
Binary decisions use prob_seizure >= 0.5, as in the shared notebooks.
"""

import math
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, cohen_kappa_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)

NAN = float("nan")
THRESHOLD = 0.5
FOLD_METRICS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "npv",
                "f1_seizure", "f1_nonseizure", "macro_f1", "kappa", "mcc", "roc_auc", "average_precision"]
BOTH_CLASS_METRICS = {"balanced_accuracy", "macro_f1", "kappa", "mcc", "roc_auc", "average_precision"}


def json_safe(obj):
    """NaN/inf -> None and numpy scalars -> Python, recursively (for json.dumps(allow_nan=False))."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _div(a, b):
    return float(a / b) if b else NAN


def binary_metrics(truth, prob1, threshold=THRESHOLD):
    """Metrics for one test fold (or pooled predictions). Undefined quantities are NaN.

    sensitivity/f1_seizure need seizure segments, specificity/f1_nonseizure need non-seizure
    segments, precision/npv need at least one predicted positive/negative; balanced accuracy,
    macro-F1, kappa, MCC, ROC-AUC and average precision need both classes in `truth`."""
    truth = np.asarray(truth)
    prob1 = np.asarray(prob1, dtype=float)
    if truth.ndim != 1 or prob1.shape != truth.shape or truth.size == 0:
        raise ValueError(f"truth and prob1 must be matching non-empty 1-D arrays ({truth.shape}, {prob1.shape})")
    if not set(np.unique(truth).tolist()) <= {0, 1}:
        raise ValueError("truth must contain only 0 (non-seizure) and 1 (seizure)")
    if not np.isfinite(prob1).all():
        raise ValueError("non-finite predicted probabilities")
    truth = truth.astype(int)
    pred = (prob1 >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(truth, pred, labels=[0, 1]).ravel())
    n_pos, n_neg = tp + fn, tn + fp
    both = n_pos > 0 and n_neg > 0
    sens, spec = _div(tp, n_pos), _div(tn, n_neg)
    f1_pos = _div(2 * tp, 2 * tp + fp + fn) if n_pos else NAN
    f1_neg = _div(2 * tn, 2 * tn + fn + fp) if n_neg else NAN
    out = dict(n=int(truth.size), n_seizure=n_pos, n_nonseizure=n_neg, single_class=not both,
               tp=tp, tn=tn, fp=fp, fn=fn,
               accuracy=_div(tp + tn, truth.size), balanced_accuracy=(sens + spec) / 2 if both else NAN,
               sensitivity=sens, specificity=spec, precision=_div(tp, tp + fp), npv=_div(tn, tn + fn),
               f1_seizure=f1_pos, f1_nonseizure=f1_neg, macro_f1=(f1_pos + f1_neg) / 2 if both else NAN,
               kappa=NAN, mcc=NAN, roc_auc=NAN, average_precision=NAN)
    if both:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = cohen_kappa_score(truth, pred)
            out["kappa"] = float(kappa) if np.isfinite(kappa) else NAN
            out["mcc"] = float(matthews_corrcoef(truth, pred))
            out["roc_auc"] = float(roc_auc_score(truth, prob1))
            out["average_precision"] = float(average_precision_score(truth, prob1))
    return out


def per_patient_table(oof, unit_col="test_unit", prob_col="prob_1", target_col="target", extra_cols=()):
    """One row of binary_metrics per test fold, from out-of-fold predictions (one row per segment).

    extra_cols: fold-level columns copied from the first row of each group (e.g. val_subject)."""
    rows = []
    for unit, g in oof.groupby(unit_col, sort=False):
        row = {unit_col: unit, **binary_metrics(g[target_col].to_numpy(), g[prob_col].to_numpy())}
        for c in extra_cols:
            row[c] = g[c].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(table, metrics=FOLD_METRICS, weight_col="n"):
    """Across-fold summary per metric over the folds where it is defined: unweighted mean, SD
    (ddof=1), median, quartiles and IQR, n_defined of n_folds, and the segment-weighted mean."""
    rows = []
    for m in metrics:
        v = pd.to_numeric(table[m], errors="coerce") if m in table else pd.Series(dtype=float)
        ok = v.notna()
        d = v[ok].to_numpy(float)
        w = table.loc[ok, weight_col].to_numpy(float) if len(d) else np.array([])
        q1, med, q3 = (np.percentile(d, [25, 50, 75]) if len(d) else (NAN, NAN, NAN))
        rows.append(dict(metric=m, n_folds=int(len(v)), n_defined=int(len(d)),
                         mean=float(d.mean()) if len(d) else NAN,
                         sd=float(d.std(ddof=1)) if len(d) > 1 else NAN,
                         median=float(med), q1=float(q1), q3=float(q3), iqr=float(q3 - q1),
                         weighted_mean=float(np.average(d, weights=w)) if len(d) and w.sum() > 0 else NAN))
    return pd.DataFrame(rows)


def summarize_folds(table, metrics=FOLD_METRICS, weight_col="n"):
    """Aggregates over all folds and over the folds whose test set holds both classes."""
    both = table[~table["single_class"].astype(bool)] if len(table) else table
    return dict(all_folds=aggregate(table, metrics, weight_col), both_class_folds=aggregate(both, metrics, weight_col))


def lopo_report(oof, unit_col="test_unit", prob_col="prob_1", target_col="target", extra_cols=()):
    """Per-fold table, across-fold summaries (all folds; both-class folds) and pooled out-of-fold
    metrics. Pooled scores mix the outputs of K different models; report them next to the per-fold
    mean +- SD, not instead of it."""
    table = per_patient_table(oof, unit_col, prob_col, target_col, extra_cols)
    summaries = summarize_folds(table)
    return dict(per_patient=table, fold_summary_all=summaries["all_folds"],
                fold_summary_both_classes=summaries["both_class_folds"],
                pooled=binary_metrics(oof[target_col].to_numpy(), oof[prob_col].to_numpy()))


def format_mean_sd(row, digits=3):
    """'0.912 ± 0.050 (n=22/23)' from one aggregate() row; 'undefined' when no fold defines it."""
    if not row["n_defined"]:
        return "undefined"
    sd = "n/a" if not np.isfinite(row["sd"]) else f"{row['sd']:.{digits}f}"
    return f"{row['mean']:.{digits}f} ± {sd} (n={int(row['n_defined'])}/{int(row['n_folds'])})"
