"""Tests for the CHB-MIT LOPO components: strict loader, fold plan, fold metrics.

Synthetic cases always run. Real-data cases run when the export is available: set CHBMIT_NPZ and
CHBMIT_META (or place chbmit_8ch.npz and chbmit_8ch_metadata.csv under data/ in the repo).

    python -m unittest tests.test_lopo -v
"""
import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from chbmit import loader as L  # noqa: E402
from chbmit import lopo_folds as F  # noqa: E402
from chbmit import lopo_metrics as M  # noqa: E402
from chbmit.make_plan import CSV_COLUMNS  # noqa: E402

REAL_NPZ = Path(os.environ.get("CHBMIT_NPZ", REPO / "data" / "chbmit_8ch.npz"))
REAL_META = Path(os.environ.get("CHBMIT_META", REPO / "data" / "chbmit_8ch_metadata.csv"))
SAVED_PLAN = REPO / "protocol" / "chbmit_lopo_plan.json"
SAVED_CSV = REPO / "protocol" / "chbmit_lopo_folds.csv"
HAVE_REAL = REAL_NPZ.is_file() and REAL_META.is_file()


def meta_from_counts(counts):
    """Segment table (no signals) with the converter's id layout; counts: {case: (n_nonseizure, n_seizure)}."""
    rows = []
    for case, (n0, n1) in counts.items():
        for label, n in ((0, n0), (1, n1)):
            name = L.LABEL_NAMES[label]
            for i in range(n):
                rows.append(dict(segment_id=f"{name}/{case}/{case}_01_segment_{i + 1:04d}.csv", patient=case,
                                 subject=L.SUBJECT_OF.get(case, case), label=label, label_name=name))
    meta = pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)
    meta.insert(0, "record_id", meta["segment_id"])
    return meta


def write_export(folder, counts, *, seed=0, report_channels=L.CHANNELS, npz_channels=None):
    """Tiny export in the converter's format: <folder>/chbmit_8ch.npz, _metadata.csv, _report.json."""
    folder = Path(folder)
    meta = meta_from_counts(counts).drop(columns="record_id")
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((len(meta), len(L.CHANNELS), L.N_SAMPLES)) * 2e-5).astype(np.float32)  # volts
    y = meta["label"].to_numpy(np.int64)
    meta["signal_sha256"] = [L.signal_sha256(x) for x in X]
    arrays = dict(X=X, y=y)
    if npz_channels is not None:
        arrays["channels"] = np.array(npz_channels)
    np.savez(folder / "chbmit_8ch.npz", **arrays)
    meta.to_csv(folder / "chbmit_8ch_metadata.csv", index=False)
    if report_channels is not None:
        (folder / "chbmit_8ch_report.json").write_text(json.dumps(dict(channels=list(report_channels), fs=L.FS,
                                                                       n_samples=L.N_SAMPLES)))
    return folder / "chbmit_8ch.npz", folder / "chbmit_8ch_metadata.csv", X, meta


def counts_of(meta):
    y = meta["label"].to_numpy()
    return dict(total=len(y), nonseizure=int((y == 0).sum()), seizure=int((y == 1).sum()))


SMALL = {"chb01": (6, 6), "chb02": (6, 5), "chb21": (5, 5)}


class LoaderSynthetic(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def load(self, npz, meta, **kw):
        kw.setdefault("expected", None)
        return L.load_chbmit_npz(npz, meta, kw.pop("unit_scale", 1e6), **kw)

    def test_valid_export_is_scaled_after_hash_check(self):
        npz, meta_csv, raw, meta = write_export(self.tmp, SMALL)
        X, y, out, info = self.load(npz, meta_csv, expected=counts_of(meta), return_info=True)
        self.assertEqual(X.dtype, np.float32)
        np.testing.assert_array_equal(X, raw * np.float32(1e6))
        np.testing.assert_array_equal(y, meta["label"].to_numpy())
        self.assertEqual(out["record_id"].tolist(), meta["segment_id"].tolist())
        self.assertEqual(info["signal_rows_verified"], len(meta))
        self.assertEqual(info["channel_order_checked_against"], ["chbmit_8ch_report.json:channels"])
        self.assertAlmostEqual(info["median_abs_scaled"], info["median_abs_stored"] * 1e6, delta=1e-6)
        record = L.data_handling_record(1e6, info)
        self.assertEqual(record["model_input_units"], "microvolts")
        self.assertEqual(record["normalization"], "none")

    def test_misaligned_rows_are_detected(self):
        npz, meta_csv, _, meta = write_export(self.tmp, SMALL)
        a, b = meta.index[(meta.patient == "chb01") & (meta.label == 0)][:2]       # same label and case
        meta.loc[[a, b], "signal_sha256"] = meta.loc[[b, a], "signal_sha256"].to_numpy()
        meta.to_csv(meta_csv, index=False)
        with self.assertRaisesRegex(L.ChbmitDataError, "not aligned"):
            self.load(npz, meta_csv)

    def test_changed_signal_is_detected(self):
        npz, meta_csv, raw, meta = write_export(self.tmp, SMALL)
        raw[3, 2, 100] += np.float32(1e-6)
        np.savez(npz, X=raw, y=meta["label"].to_numpy(np.int64))
        with self.assertRaisesRegex(L.ChbmitDataError, "signal_sha256 mismatch"):
            self.load(npz, meta_csv)

    def test_nonfinite_values_are_rejected(self):
        npz, meta_csv, raw, meta = write_export(self.tmp, SMALL)
        raw[5, 0, 0] = np.nan
        meta.loc[5, "signal_sha256"] = L.signal_sha256(raw[5])
        meta.to_csv(meta_csv, index=False)
        np.savez(npz, X=raw, y=meta["label"].to_numpy(np.int64))
        with self.assertRaisesRegex(L.ChbmitDataError, "non-finite"):
            self.load(npz, meta_csv)

    def test_shape_dtype_and_counts(self):
        npz, meta_csv, raw, meta = write_export(self.tmp, SMALL)
        y = meta["label"].to_numpy(np.int64)
        with self.assertRaisesRegex(L.ChbmitDataError, "segment counts"):
            self.load(npz, meta_csv, expected=L.EXPECTED_COUNTS)
        np.savez(npz, X=raw.transpose(0, 2, 1), y=y)                               # time-major
        with self.assertRaisesRegex(L.ChbmitDataError, "shape"):
            self.load(npz, meta_csv)
        np.savez(npz, X=raw.astype(np.float64), y=y)
        with self.assertRaisesRegex(L.ChbmitDataError, "dtype"):
            self.load(npz, meta_csv)
        np.savez(npz, X=raw, y=1 - y)
        with self.assertRaisesRegex(L.ChbmitDataError, "label differs"):
            self.load(npz, meta_csv)

    def test_channel_order_is_checked(self):
        swapped = list(L.CHANNELS)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        npz, meta_csv, _, _ = write_export(self.tmp, SMALL, report_channels=swapped)
        with self.assertRaisesRegex(L.ChbmitDataError, "channel order"):
            self.load(npz, meta_csv)
        (self.tmp / "chbmit_8ch_report.json").unlink()
        with self.assertRaisesRegex(L.ChbmitDataError, "cannot verify channel order"):
            self.load(npz, meta_csv)
        X, _, _ = self.load(npz, meta_csv, require_channel_names=False)
        self.assertEqual(X.shape[1:], (8, 2560))
        with tempfile.TemporaryDirectory() as other:
            npz2, meta2, _, _ = write_export(other, SMALL, report_channels=None, npz_channels=L.CHANNELS)
            _, _, _, info = self.load(npz2, meta2, return_info=True)
            self.assertEqual(info["channel_order_checked_against"], ["npz:channels"])

    def test_subject_mapping_and_cross_subject_duplicates(self):
        npz, meta_csv, raw, meta = write_export(self.tmp, SMALL)
        bad = meta.copy()
        bad.loc[bad.patient == "chb21", "subject"] = "chb21"                      # same person split in two
        bad.to_csv(meta_csv, index=False)
        with self.assertRaisesRegex(L.ChbmitDataError, "same-person mapping"):
            self.load(npz, meta_csv)
        i = int(meta.index[meta.patient == "chb02"][0])
        j = int(meta.index[(meta.patient == "chb01") & (meta.label == meta.label[i])][0])
        raw[i] = raw[j]
        meta.loc[i, "signal_sha256"] = meta.loc[j, "signal_sha256"]
        meta.to_csv(meta_csv, index=False)
        np.savez(npz, X=raw, y=meta["label"].to_numpy(np.int64))
        with self.assertRaisesRegex(L.ChbmitDataError, "different subjects"):
            self.load(npz, meta_csv)

    def test_unit_scale_must_be_positive_finite(self):
        npz, meta_csv, _, _ = write_export(self.tmp, SMALL)
        for bad in (0, -1e6, float("nan"), float("inf")):
            with self.assertRaises(L.ChbmitDataError):
                self.load(npz, meta_csv, unit_scale=bad)


# Same shape as the real data: 23 subjects, chb07 single-class, chb16 below the minimum.
REALISTIC = {"chb01": (43, 43), "chb02": (16, 16), "chb03": (37, 37), "chb04": (36, 36), "chb05": (54, 54),
             "chb06": (13, 13), "chb07": (31, 0), "chb08": (91, 91), "chb09": (21, 27), "chb10": (41, 41),
             "chb11": (80, 80), "chb12": (94, 94), "chb13": (41, 41), "chb14": (15, 15), "chb15": (194, 194),
             "chb16": (4, 4), "chb17a": (20, 20), "chb17b": (8, 8), "chb18": (29, 29), "chb19": (22, 22),
             "chb20": (25, 25), "chb21": (19, 19), "chb22": (19, 19), "chb23": (41, 41), "chb24": (45, 45)}


class FoldsSynthetic(unittest.TestCase):
    def test_natural_order(self):
        ids = ["chb10", "chb9", "chb17b", "chb17a", "chb17", "chb1"]
        self.assertEqual(sorted(ids, key=F.natural_key), ["chb1", "chb9", "chb10", "chb17", "chb17a", "chb17b"])

    def test_single_class_subject_is_tested_but_never_validation(self):
        meta = meta_from_counts({"chb01": (6, 6), "chb02": (8, 0), "chb03": (6, 6), "chb04": (6, 6)})
        folds = F.build_lopo_folds(meta, min_val_per_class=5)
        by = {f.test_unit: f for f in folds}
        self.assertEqual(set(by), {"chb01", "chb02", "chb03", "chb04"})
        self.assertNotIn("chb02", {f.val_subject for f in folds})
        self.assertEqual(by["chb01"].val_subject, "chb03")                         # skips chb02
        man = F.manifest(folds, meta).set_index("test_unit")
        self.assertTrue(man.loc["chb02", "test_single_class"])
        self.assertEqual((man.loc["chb02", "n_test_nonseizure"], man.loc["chb02", "n_test_seizure"]), (8, 0))
        m = M.binary_metrics(meta.label[by["chb02"].test_idx], np.full(8, 0.2))
        self.assertTrue(m["single_class"] and np.isnan(m["sensitivity"]) and np.isnan(m["roc_auc"]))
        self.assertEqual(m["specificity"], 1.0)

    def test_subject_below_minimum_is_skipped_for_validation(self):
        meta = meta_from_counts({"chb01": (6, 6), "chb02": (4, 4), "chb03": (5, 9), "chb04": (6, 6)})
        folds = F.build_lopo_folds(meta, min_val_per_class=5)
        v = {f.test_unit: f.val_subject for f in folds}
        self.assertEqual(v, {"chb01": "chb03", "chb02": "chb03", "chb03": "chb04", "chb04": "chb01"})
        self.assertEqual(F.build_lopo_folds(meta, min_val_per_class=4)[0].val_subject, "chb02")
        with self.assertRaises(F.FoldPlanError):                                   # a single eligible subject
            F.build_lopo_folds(meta_from_counts({"chb01": (6, 6), "chb02": (4, 4)}), min_val_per_class=5)

    def test_realistic_subject_plan(self):
        meta = meta_from_counts(REALISTIC)
        folds = F.build_lopo_folds(meta, seed=42)
        self.assertEqual(len(folds), 23)
        self.assertEqual([f.seed for f in folds], list(range(43, 66)))
        self.assertEqual(set(meta.patient[folds[0].test_idx]), {"chb01", "chb21"})
        vals = {f.val_subject for f in folds}
        self.assertFalse(vals & {"chb07", "chb16"})
        v = {f.test_unit: f.val_subject for f in folds}
        self.assertEqual((v["chb06"], v["chb07"], v["chb15"], v["chb16"], v["chb20"], v["chb24"]),
                         ("chb08", "chb08", "chb17", "chb17", "chb22", "chb01"))
        self.assertTrue(F.check_plan(folds, meta, min_val_per_class=5))

    def test_case_mode_excludes_siblings(self):
        meta = meta_from_counts(REALISTIC)
        folds = F.build_lopo_folds(meta, unit="case", seed=42)
        self.assertEqual(len(folds), 25)
        by = {f.test_unit: f for f in folds}
        pats = meta.patient.to_numpy()
        for test, sibling in (("chb01", "chb21"), ("chb21", "chb01"), ("chb17a", "chb17b"), ("chb17b", "chb17a")):
            f = by[test]
            self.assertEqual(f.excluded_units, (sibling,))
            self.assertEqual(set(pats[f.test_idx]), {test})
            used = set(pats[f.train_idx]) | set(pats[f.val_idx])
            self.assertNotIn(sibling, used)
            self.assertNotIn(test, used)
            left_out = np.setdiff1d(np.arange(len(meta)), np.concatenate([f.train_idx, f.val_idx, f.test_idx]))
            self.assertEqual(set(pats[left_out]), {sibling})
        self.assertEqual(by["chb21"].val_subject, "chb02")
        self.assertEqual(by["chb24"].val_subject, "chb01")
        self.assertEqual(set(pats[by["chb24"].val_idx]), {"chb01", "chb21"})       # validation = whole subject
        self.assertEqual(by["chb02"].excluded_units, ())

    def test_invariant_violations_are_caught(self):
        meta = meta_from_counts(REALISTIC)
        folds = F.build_lopo_folds(meta)
        f = folds[2]
        leaked = F.Fold(**{**f.__dict__, "train_idx": np.sort(np.r_[f.train_idx, f.test_idx[:1]])})
        with self.assertRaisesRegex(F.FoldPlanError, "both train and test"):
            F.check_plan([leaked], meta, complete=False)
        sibling = F.build_lopo_folds(meta, unit="case")[0]                          # chb01; chb21 left out
        rows21 = np.flatnonzero(meta.patient == "chb21")
        with self.assertRaisesRegex(F.FoldPlanError, "subjects"):
            F.check_plan([F.Fold(**{**sibling.__dict__, "train_idx": np.sort(np.r_[sibling.train_idx, rows21])})],
                         meta, complete=False)
        half_val = F.Fold(**{**f.__dict__, "val_idx": f.val_idx[:3]})
        with self.assertRaises(F.FoldPlanError):
            F.check_plan([half_val], meta, complete=False)
        with self.assertRaisesRegex(F.FoldPlanError, "tested exactly once|tests every"):
            F.check_plan(folds[:-1], meta)
        with self.assertRaisesRegex(F.FoldPlanError, "segments per class"):
            F.check_plan(folds, meta, min_val_per_class=100)

    def test_select_folds_keeps_plan_numbers_and_seeds(self):
        meta = meta_from_counts(REALISTIC)
        folds = F.build_lopo_folds(meta, seed=42)
        sub = F.select_folds(folds, ["chb09", "chb02"])
        self.assertEqual([(f.test_unit, f.fold, f.seed) for f in sub], [("chb02", 2, 44), ("chb09", 9, 51)])
        self.assertEqual(len(F.select_folds(folds, [])), 23)
        with self.assertRaises(F.FoldPlanError):
            F.select_folds(folds, ["chb21"])                                        # a case, not a subject
        F.check_plan(sub, meta, complete=False)

    def test_plan_json_roundtrip_is_row_order_independent(self):
        meta = meta_from_counts(REALISTIC)
        for unit in ("subject", "case"):
            folds, body = F.plan_or_build(meta, unit=unit, min_val_per_class=5, seed=42)
            shuffled = meta.sample(frac=1, random_state=3).reset_index(drop=True)
            replay = F.plan_from_json(json.loads(json.dumps(body)), shuffled)
            self.assertEqual(F.plan_sha256(replay, shuffled), body["plan_sha256"])
            rebuilt = F.build_lopo_folds(shuffled, unit=unit, min_val_per_class=5, seed=42)
            self.assertEqual(F.plan_sha256(rebuilt, shuffled), body["plan_sha256"])
            rid, rid2 = meta.record_id.to_numpy(), shuffled.record_id.to_numpy()
            for a, b in zip(folds, replay):
                self.assertEqual((a.fold, a.name, a.seed, a.val_subject, a.excluded_units),
                                 (b.fold, b.name, b.seed, b.val_subject, b.excluded_units))
                for part in ("train_idx", "val_idx", "test_idx"):
                    self.assertEqual(set(rid[getattr(a, part)]), set(rid2[getattr(b, part)]))

    def test_plan_json_rejects_edits_and_other_data(self):
        meta = meta_from_counts(REALISTIC)
        folds, body = F.plan_or_build(meta)
        reordered = json.loads(json.dumps(body))                                    # e.g. written in row order
        for entry in reordered["folds"]:
            for part in ("train", "val", "test"):
                entry[part].reverse()
        self.assertEqual(F.plan_sha256(F.plan_from_json(reordered, meta), meta), body["plan_sha256"])
        edited = json.loads(json.dumps(body))
        edited["folds"][0]["test"].append(edited["folds"][0]["train"].pop())
        with self.assertRaisesRegex(F.FoldPlanError, "modified"):
            F.plan_from_json(edited, meta)
        relabeled = meta.copy()
        relabeled.loc[0, "label"] = 1 - relabeled.loc[0, "label"]
        with self.assertRaisesRegex(F.FoldPlanError, "records_sha256"):
            F.plan_from_json(body, relabeled)
        with self.assertRaisesRegex(F.FoldPlanError, "records"):
            F.plan_from_json(body, meta.iloc[1:].reset_index(drop=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            F.save_plan(body, path)
            first = path.read_bytes()
            F.save_plan(F.plan_or_build(meta)[1], path)
            self.assertEqual(first, path.read_bytes())                              # deterministic bytes
            self.assertEqual(F.plan_or_build(meta, plan_path=str(path))[1]["plan_sha256"], body["plan_sha256"])
            with self.assertRaisesRegex(F.FoldPlanError, "different settings"):
                F.plan_or_build(meta, plan_path=str(path), seed=7)

    def test_missing_ids_rejected(self):
        meta = meta_from_counts(SMALL)
        meta.loc[3, "subject"] = None
        with self.assertRaises(F.FoldPlanError):
            F.build_lopo_folds(meta)


class Metrics(unittest.TestCase):
    def test_single_class_fold_never_raises_or_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = M.binary_metrics(np.zeros(40, int), np.linspace(0, 0.6, 40))
            s = M.binary_metrics(np.ones(10, int), np.linspace(0.3, 0.9, 10))
        self.assertTrue(m["single_class"])
        for k in ("sensitivity", "f1_seizure", "balanced_accuracy", "macro_f1", "kappa", "mcc", "roc_auc",
                  "average_precision"):
            self.assertTrue(np.isnan(m[k]), k)
        self.assertEqual(m["specificity"], m["accuracy"])
        self.assertTrue(np.isnan(s["specificity"]) and np.isnan(s["f1_nonseizure"]))
        self.assertEqual(s["npv"], 0.0)                                            # predicted negatives exist
        self.assertEqual(s["sensitivity"], s["accuracy"])
        json.dumps(M.json_safe(m), allow_nan=False)

    def test_both_classes_match_sklearn(self):
        from sklearn.metrics import (average_precision_score, balanced_accuracy_score, cohen_kappa_score,
                                     f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
        rng = np.random.default_rng(0)
        truth = rng.integers(0, 2, 300)
        prob = np.clip(truth * 0.3 + rng.uniform(0, 0.7, 300), 0, 1)
        pred = (prob >= 0.5).astype(int)
        m = M.binary_metrics(truth, prob)
        self.assertAlmostEqual(m["roc_auc"], roc_auc_score(truth, prob))
        self.assertAlmostEqual(m["average_precision"], average_precision_score(truth, prob))
        self.assertAlmostEqual(m["kappa"], cohen_kappa_score(truth, pred))
        self.assertAlmostEqual(m["mcc"], matthews_corrcoef(truth, pred))
        self.assertAlmostEqual(m["macro_f1"], f1_score(truth, pred, average="macro"))
        self.assertAlmostEqual(m["balanced_accuracy"], balanced_accuracy_score(truth, pred))
        self.assertAlmostEqual(m["sensitivity"], recall_score(truth, pred))
        self.assertAlmostEqual(m["precision"], precision_score(truth, pred))

    def test_threshold_matches_notebook_convention(self):
        m = M.binary_metrics(np.array([0, 1]), np.array([0.5, 0.5]))              # probs[:, 1] >= 0.5
        self.assertEqual((m["tp"], m["fp"]), (1, 1))
        with self.assertRaises(ValueError):
            M.binary_metrics(np.array([0, 1]), np.array([np.nan, 0.5]))

    def test_aggregate_statistics(self):
        table = pd.DataFrame(dict(n=[10, 30, 20, 40], single_class=[False, False, False, True],
                                  roc_auc=[0.6, 0.8, 1.0, np.nan], accuracy=[0.5, 0.7, 0.9, 1.0]))
        agg = M.aggregate(table, ["roc_auc", "accuracy"]).set_index("metric")
        r = agg.loc["roc_auc"]
        self.assertEqual((r.n_defined, r.n_folds), (3, 4))
        self.assertAlmostEqual(r["mean"], 0.8)
        self.assertAlmostEqual(r["sd"], np.std([0.6, 0.8, 1.0], ddof=1))
        self.assertAlmostEqual(r["median"], 0.8)
        self.assertAlmostEqual(r["iqr"], 0.2)
        self.assertAlmostEqual(r["weighted_mean"], (0.6 * 10 + 0.8 * 30 + 1.0 * 20) / 60)
        self.assertAlmostEqual(agg.loc["accuracy", "weighted_mean"], (5 + 21 + 18 + 40) / 100)
        both = M.summarize_folds(table)["both_class_folds"].set_index("metric")
        self.assertEqual(both.loc["accuracy", "n_folds"], 3)
        self.assertEqual(M.format_mean_sd(r), "0.800 ± 0.200 (n=3/4)")
        one = M.aggregate(table.iloc[:1], ["roc_auc"]).iloc[0]
        self.assertTrue(np.isnan(one["sd"]))
        self.assertEqual(M.format_mean_sd(M.aggregate(table.iloc[3:], ["roc_auc"]).iloc[0]), "undefined")

    def test_lopo_report(self):
        rng = np.random.default_rng(1)
        parts = []
        for unit, (n0, n1) in {"chb01": (10, 10), "chb07": (12, 0), "chb09": (5, 7)}.items():
            t = np.r_[np.zeros(n0, int), np.ones(n1, int)]
            parts.append(pd.DataFrame(dict(test_unit=unit, val_subject="x", target=t,
                                           prob_1=np.clip(t * 0.4 + rng.uniform(0, 0.6, len(t)), 0, 1))))
        oof = pd.concat(parts, ignore_index=True)
        rep = M.lopo_report(oof, extra_cols=("val_subject",))
        table = rep["per_patient"].set_index("test_unit")
        self.assertEqual(table.loc["chb07", "single_class"], True)
        self.assertTrue(np.isnan(table.loc["chb07", "roc_auc"]))
        summ = rep["fold_summary_all"].set_index("metric")
        self.assertEqual((summ.loc["roc_auc", "n_defined"], summ.loc["accuracy", "n_defined"]), (2, 3))
        self.assertEqual(rep["fold_summary_both_classes"].set_index("metric").loc["accuracy", "n_folds"], 2)
        self.assertEqual(rep["pooled"], M.binary_metrics(oof.target, oof.prob_1))
        json.dumps(M.json_safe({k: v for k, v in rep.items()}), allow_nan=False)


@unittest.skipUnless(HAVE_REAL, "real CHB-MIT export not found (set CHBMIT_NPZ / CHBMIT_META)")
class RealData(unittest.TestCase):
    EXPECTED_VALIDATION = {
        "chb01": "chb02", "chb02": "chb03", "chb03": "chb04", "chb04": "chb05", "chb05": "chb06",
        "chb06": "chb08", "chb07": "chb08", "chb08": "chb09", "chb09": "chb10", "chb10": "chb11",
        "chb11": "chb12", "chb12": "chb13", "chb13": "chb14", "chb14": "chb15", "chb15": "chb17",
        "chb16": "chb17", "chb17": "chb18", "chb18": "chb19", "chb19": "chb20", "chb20": "chb22",
        "chb22": "chb23", "chb23": "chb24", "chb24": "chb01"}

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.meta, cls.info = L.load_chbmit_npz(REAL_NPZ, REAL_META, 1e6, return_info=True)

    def test_loader(self):
        self.assertEqual(self.X.shape, (2053, 8, 2560))
        self.assertEqual(self.X.dtype, np.float32)
        self.assertEqual(self.info["counts"], dict(total=2053, nonseizure=1039, seizure=1014))
        self.assertEqual(self.info["signal_rows_verified"], 2053)
        self.assertTrue(self.info["channel_order_checked_against"])
        self.assertEqual(len(self.info["subjects"]), 23)
        self.assertEqual(len(self.info["patients"]), 25)
        self.assertTrue(5 < self.info["median_abs_scaled"] < 100)                   # microvolts
        with np.load(REAL_NPZ) as d:
            raw = d["X"]
        for i in (0, 1000, 2052):
            np.testing.assert_array_equal(self.X[i], raw[i] * np.float32(1e6))
            self.assertEqual(L.signal_sha256(raw[i]), self.meta.signal_sha256[i])
        self.assertEqual(self.meta.record_id.tolist(), self.meta.segment_id.tolist())

    def test_subject_plan(self):
        folds = F.build_lopo_folds(self.meta, unit="subject", min_val_per_class=5, seed=42)
        self.assertEqual(len(folds), 23)
        self.assertEqual({f.test_unit: f.val_subject for f in folds}, self.EXPECTED_VALIDATION)
        self.assertEqual([f.seed for f in folds], list(range(43, 66)))
        self.assertEqual([f.fold for f in folds], list(range(1, 24)))
        man = F.manifest(folds, self.meta).set_index("test_unit")
        self.assertEqual(man.n_test.sum(), 2053)
        self.assertTrue((man.n_train + man.n_val + man.n_test == 2053).all())
        self.assertEqual(man.index[man.test_single_class].tolist(), ["chb07"])
        self.assertEqual((man.loc["chb07", "n_test_nonseizure"], man.loc["chb07", "n_test_seizure"]), (31, 0))
        self.assertEqual((man.loc["chb16", "n_test_nonseizure"], man.loc["chb16", "n_test_seizure"]), (4, 4))
        self.assertEqual((man.loc["chb01", "n_test"]), 124)                         # chb01 + chb21
        self.assertTrue((man[["n_val_nonseizure", "n_val_seizure"]] >= 5).all().all())
        F.check_plan(folds, self.meta, min_val_per_class=5)

    def test_case_plan(self):
        folds = F.build_lopo_folds(self.meta, unit="case", min_val_per_class=5, seed=42)
        self.assertEqual(len(folds), 25)
        by = {f.test_unit: f for f in folds}
        for test, sibling in (("chb01", "chb21"), ("chb21", "chb01"), ("chb17a", "chb17b"), ("chb17b", "chb17a")):
            self.assertEqual(by[test].excluded_units, (sibling,))
            used = set(self.meta.patient[by[test].train_idx]) | set(self.meta.patient[by[test].val_idx])
            self.assertNotIn(sibling, used)
        self.assertEqual(by["chb21"].val_subject, "chb02")
        self.assertEqual(sum(len(f.test_idx) for f in folds), 2053)
        F.check_plan(folds, self.meta, min_val_per_class=5)

    def test_plan_is_row_order_independent(self):
        folds = F.build_lopo_folds(self.meta)
        shuffled = self.meta.sample(frac=1, random_state=11).reset_index(drop=True)
        self.assertEqual(F.plan_sha256(F.build_lopo_folds(shuffled), shuffled), F.plan_sha256(folds, self.meta))

    @unittest.skipUnless(SAVED_PLAN.is_file() and SAVED_CSV.is_file(), "protocol plan files not generated")
    def test_saved_plan_replays_and_matches_rebuild(self):
        folds, body = F.load_plan(SAVED_PLAN, self.meta)
        self.assertEqual(body["params"]["lopo_unit"], "subject")
        self.assertEqual(body["plan_sha256"], F.plan_sha256(F.build_lopo_folds(self.meta), self.meta))
        self.assertEqual(body["plan_sha256"], F.plan_sha256(folds, self.meta))
        saved = pd.read_csv(SAVED_CSV, keep_default_na=False)
        fresh = F.manifest(folds, self.meta)[CSV_COLUMNS]
        self.assertEqual(saved.columns.tolist(), CSV_COLUMNS)
        self.assertEqual(saved[["fold", "test_unit", "validation_unit", "fit_seed", "n_test"]].values.tolist(),
                         fresh[["fold", "test_unit", "validation_unit", "fit_seed", "n_test"]].values.tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
