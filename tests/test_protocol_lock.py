"""Tests for protocol/protocol_lock.py.

Pure-logic and CHB-MIT tests always run on small synthetic data. The Bonn notebook tests need the
shared protocol notebooks (Bonn_{EEGNet,ATCNet,TCFormer}_5_to_2_Class_Validation.ipynb) in
$BONN_NOTEBOOK_DIR or notebooks/, and are skipped otherwise; they use a synthetic Bonn folder.
The real-data test runs only when $PROTOCOL_LOCK_REAL_DATA points to a folder holding bonn/ and,
optionally, chbmit_8ch.npz + chbmit_8ch_metadata.csv.

    python -m unittest tests.test_protocol_lock -v      (from the repository root)
"""
import copy
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "protocol"))
sys.path.insert(0, str(ROOT))
import protocol_lock as pl  # noqa: E402

try:  # the shared CHB-MIT fold module, if present, must agree with the lock's own rule
    from chbmit import lopo_folds  # noqa: E402
except ImportError:
    lopo_folds = None

MODELS = ("EEGNet", "ATCNet", "TCFormer")
SETS = "ZONFS"


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = pl.main([str(a) for a in argv])
    return code, out.getvalue() + err.getvalue()


def find_notebooks():
    candidates = [os.environ.get("BONN_NOTEBOOK_DIR"), ROOT / "notebooks"]
    for folder in filter(None, candidates):
        found = {m: Path(folder) / f"Bonn_{m}_5_to_2_Class_Validation.ipynb" for m in MODELS}
        if all(p.is_file() for p in found.values()):
            return found
    return None


def write_bonn(root, values, name=lambda s, i: f"{s}/{s}{i:03d}", newline="\n"):
    """values: {set: [int arrays]}; the N set uses upper-case .TXT as in the official files."""
    for s in SETS:
        for i, v in enumerate(values[s], 1):
            path = Path(root) / (name(s, i) + (".TXT" if s == "N" else ".txt"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(newline.join(map(str, v)).encode() + newline.encode())


def float32_sha(v):
    return hashlib.sha256(np.asarray(v, dtype=np.float32).astype("<f4").tobytes()).hexdigest()


def variant(src, dst, set_params=None, tag=False, inject=None, extra_cell=None, edit=None):
    """Copy a notebook with Colab-style parameter edits, a papermill tag/injected cell or code edits."""
    nb = json.loads(Path(src).read_text(encoding="utf-8"))
    cells = nb["cells"]
    roles = pl.classify_cells(cells)["roles"]
    config = cells[roles["config"]]
    lines = pl.cell_source(config).split("\n")
    for name, literal in (set_params or {}).items():
        hits = [i for i, l in enumerate(lines) if l.startswith(f"{name} = ")]
        assert len(hits) == 1, name
        lines[hits[0]] = f"{name} = {literal}"
    config["source"] = "\n".join(lines)
    if tag:
        config.setdefault("metadata", {})["tags"] = ["parameters"]
    if edit:
        role, old, new = edit
        source = pl.cell_source(cells[roles[role]])
        assert source.count(old) == 1, old
        cells[roles[role]]["source"] = source.replace(old, new)
    position = roles["config"] + 1
    if inject is not None:
        body = "# Parameters\n" + "".join(f"{k} = {v!r}\n" for k, v in inject.items())
        cells.insert(position, dict(cell_type="code", metadata={"tags": ["injected-parameters"]},
                                    source=body, outputs=[], execution_count=None))
    if extra_cell is not None:
        cells.insert(position, dict(cell_type="code", metadata={}, source=extra_cell, outputs=[],
                                    execution_count=None))
    Path(dst).write_text(json.dumps(nb), encoding="utf-8")
    return dst


# ----------------------------------------------------------------------------- pure logic
class TestCodeNormalisation(unittest.TestCase):
    def test_formatting_does_not_count_code_does(self):
        a = "x = 1\n# note\n\ny = x + 1   \n"
        self.assertEqual(pl.code_sha(a), pl.code_sha("x = 1\r\ny = x + 1\r\n# other note\r\n"))
        self.assertNotEqual(pl.code_sha(a), pl.code_sha("x = 1\ny = x + 2\n"))

    def test_parameter_lines_are_stripped_and_captured(self):
        src = ('A = "path"  # @param {type:"string"}\nEPOCHS = 100  # @param\nFS, T = 173.61, None\n'
               "OPTS = {'k': (1, 2)}\nassert EPOCHS >= 1\n\ndef f():\n    return EPOCHS\nDEVICE = str(EPOCHS)\n")
        code, params = pl.split_parameter_lines(src)
        self.assertEqual(params, {"A": "path", "EPOCHS": 100, "FS": 173.61, "T": None, "OPTS": {"k": (1, 2)}})
        self.assertIn("assert EPOCHS >= 1", code)
        self.assertIn("DEVICE = str(EPOCHS)", code)  # not a literal: stays in the hashed code
        self.assertNotIn("173.61", code)
        edited, _ = pl.split_parameter_lines(src.replace("EPOCHS = 100", "EPOCHS = 5").replace('"path"', '"/x"'))
        self.assertEqual(pl.code_sha(code), pl.code_sha(edited))

    def test_env_import_lines_ignore_package_list(self):
        base = ('import os\nos.environ.setdefault("A", "1")\npackages = {"torch": "torch"}\n'
                "missing = [p for p in packages]\nfrom x import (a,\n    b)\n")
        other = base.replace('{"torch": "torch"}', '{"torch": "torch", "einops": "einops"}')
        self.assertEqual(pl.env_import_lines(base), pl.env_import_lines(other))
        self.assertIn("from x import (a,\n    b)", pl.env_import_lines(base))
        self.assertNotEqual(pl.code_sha(pl.env_import_lines(base)),
                            pl.code_sha(pl.env_import_lines(base + "import einops\n")))

    def test_load_parameters(self):
        self.assertEqual(pl.load_parameters('{"EPOCHS": 3, "SELECTED_IDS": ["Z_vs_S"]}'),
                         {"EPOCHS": 3, "SELECTED_IDS": ["Z_vs_S"]})
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "p.json").write_text('{"PLOT_EVERY": 100}')
            self.assertEqual(pl.load_parameters(str(tmp / "p.json")), {"PLOT_EVERY": 100})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        for bad in ("{broken", "[1, 2]", str(tmp / "missing.json")):
            with self.assertRaises(pl.LockError):
                pl.load_parameters(bad)

    def test_classify_cells(self):
        code = lambda s, tags=None: dict(cell_type="code", source=s, metadata={"tags": tags or []})  # noqa: E731
        cells = [code("packages = {}"), code("LEARNING_RATE = 1", ["parameters"]), code("LEARNING_RATE = 2", ["injected-parameters"]),
                 code("MODEL_PROVENANCE = {}"), code("PREVIOUS_32 = []"), code("dataset_root = Path(BONN_DATA_DIR)"),
                 code("NOTEBOOK_CODE_SHA256 = ''"), code("x = 1  # stray"), code("def run_fit(): pass"),
                 code("def plan_splits(): pass"), dict(cell_type="markdown", source="LEARNING_RATE = 3", metadata={})]
        layout = pl.classify_cells(cells)
        self.assertEqual(layout["roles"]["config"], 1)
        self.assertEqual((layout["parameters_tag"], layout["injected"], layout["unknown"]), (1, [2], [7]))
        with self.assertRaises(pl.LockError):
            pl.classify_cells(cells + [code("def run_fit(): return 1")])
        with self.assertRaises(pl.LockError):
            pl.classify_cells(cells[:4])


def fake_bonn_lock(tasks, data=None, **protocol):
    base = dict(shared_config={"epochs": 100}, preset="previous_32", tasks=[[t, [["Z"], ["S"]]] for t in tasks],
                raw_data_handling={}, parameters={}, shared_code={"training": "a"}, extra_code_cells=[],
                other_injected_parameters={})
    base.update(protocol)
    data = data or dict(data_order_hash="d", signal_set_hash="s", file_order_hash="f", record_order_hash="r",
                        notebook_data_hash="n")
    per_task = {t: pl.sha(t) for t in tasks}
    return dict(lock_kind="bonn", lock_schema=pl.BONN_SCHEMA, protocol_hash=pl.sha(pl.canon(base)),
                data_order_hash=data["data_order_hash"], split_plan_hash=pl.sha(pl.canon(per_task)),
                split_hash_per_task=per_task, split_hash_per_fit={t: {"cv_fold_01": per_task[t][:16]} for t in tasks},
                protocol=base, data=data, run_scope=dict(n_tasks=len(tasks)), info_only={}, recordings=[])


class TestCompare(unittest.TestCase):
    def test_bytes_differ_signals_identical_is_a_match(self):
        a = fake_bonn_lock(["Z_vs_S"])
        b = copy.deepcopy(a)
        b["data"].update(file_order_hash="other", record_order_hash="other", notebook_data_hash="other")
        status, lines = pl.compare_locks(a, b)
        self.assertEqual(status, "MATCH")
        self.assertTrue(any("bytes differ but signals are identical" in l for l in lines))
        self.assertTrue(any("file names or folder nesting differ" in l for l in lines))

    def test_per_task_comparison_uses_the_task_intersection(self):
        a = fake_bonn_lock(["Z_vs_S", "Z_vs_O", "N_vs_F"])
        b = fake_bonn_lock(["Z_vs_S", "Z_vs_O"], preset="all_26")
        b["split_hash_per_task"]["Z_vs_O"] = "x" * 64
        b["split_hash_per_fit"]["Z_vs_O"]["cv_fold_01"] = "x" * 16
        status, lines = pl.compare_locks(a, b)
        self.assertEqual(status, "MISMATCH")
        text = "\n".join(lines)
        self.assertIn("1/2 common tasks identical; differing: ['Z_vs_O']", text)
        self.assertIn("1 task(s) in only one lock", text)
        self.assertIn("preset: 'previous_32' vs 'all_26'", text)
        self.assertIn("only here: ['N_vs_F']", text)

    def test_protocol_field_differences_are_named(self):
        a = fake_bonn_lock(["Z_vs_S"])
        b = fake_bonn_lock(["Z_vs_S"], shared_config={"epochs": 50}, shared_code={"training": "b"})
        status, lines = pl.compare_locks(a, b)
        self.assertEqual(status, "MISMATCH")
        self.assertIn("  config.epochs: 100 vs 50", lines)
        self.assertIn("  shared code differs: training cell", lines)

    def test_different_kinds_or_schemas_are_not_comparable(self):
        a = fake_bonn_lock(["Z_vs_S"])
        self.assertEqual(pl.compare_locks(a, dict(a, lock_schema=1))[0], "INCOMPARABLE")
        self.assertEqual(pl.compare_locks(a, dict(lock_kind="chbmit-lopo", lock_schema=1))[0], "INCOMPARABLE")

    def test_compare_cli_exit_codes(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            a = fake_bonn_lock(["Z_vs_S"])
            b = fake_bonn_lock(["Z_vs_S"], shared_config={"epochs": 3})
            (tmp / "a.json").write_text(json.dumps(a))
            (tmp / "b.json").write_text(json.dumps(b))
            script = ROOT / "protocol" / "protocol_lock.py"
            same = subprocess.run([sys.executable, str(script), "--compare", tmp / "a.json", tmp / "a.json"],
                                  capture_output=True, text=True)
            diff = subprocess.run([sys.executable, str(script), "--compare", tmp / "a.json", tmp / "b.json"],
                                  capture_output=True, text=True)
            self.assertEqual(same.returncode, 0, same.stdout + same.stderr)
            self.assertEqual(diff.returncode, 3, diff.stdout + diff.stderr)
            self.assertIn("config.epochs: 100 vs 3", diff.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- CHB-MIT
def synthetic_chbmit(root, permute=None, rename=None):
    """Nine recording folders, seven subjects: chb21 -> chb01 and chb17a/b -> chb17 share a subject;
    chb04 has no seizures and chb05 only 2+2 segments, so neither may be validation."""
    rng = np.random.default_rng(7)
    layout = [("chb01", "chb01", 8, 8), ("chb21", "chb01", 4, 4), ("chb02", "chb02", 6, 7), ("chb03", "chb03", 7, 6),
              ("chb04", "chb04", 9, 0), ("chb05", "chb05", 2, 2), ("chb06", "chb06", 6, 6),
              ("chb17a", "chb17", 3, 3), ("chb17b", "chb17", 3, 3)]
    rows, X = [], []
    for patient, subject, n0, n1 in layout:
        for label, count in ((0, n0), (1, n1)):
            for j in range(count):
                x = (rng.standard_normal((4, 16)) * 3e-5).astype(np.float32)
                X.append(x)
                rows.append(dict(segment_id=f"{'seizure' if label else 'nonseizure'}/{patient}/seg_{j:03d}.csv",
                                 patient=patient, subject=subject, label=label, signal_sha256=float32_sha(x)))
    meta, X = pd.DataFrame(rows), np.stack(X)
    if permute is not None:
        order = np.random.default_rng(permute).permutation(len(meta))
        meta, X = meta.iloc[order].reset_index(drop=True), X[order]
    if rename:
        meta["segment_id"] = [rename(s) for s in meta["segment_id"]]
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "x.npz", X=X, y=meta["label"].to_numpy(np.int64))
    meta.to_csv(root / "meta.csv", index=False)
    return root / "x.npz", root / "meta.csv"


class TestChbmit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="plock_chb_"))
        cls.npz, cls.meta = synthetic_chbmit(cls.tmp / "a")
        cls.plan = cls.tmp / "plan.json"
        code, out = run_cli("--chbmit-reference-plan", cls.plan, "--npz", cls.npz, "--meta", cls.meta)
        assert code == 0, out
        code, out = run_cli("--chbmit-plan", cls.plan, "--npz", cls.npz, "--meta", cls.meta, "--out", cls.tmp / "ref.json")
        assert code == 0, out
        cls.ref = json.loads((cls.tmp / "ref.json").read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def edited_plan(self, name, change):
        body = json.loads(self.plan.read_text())
        change(body)
        path = self.tmp / name
        path.write_text(json.dumps(body))
        return path

    def test_decided_rule(self):
        plan = json.loads(self.plan.read_text())
        folds = plan["folds"]
        self.assertEqual([f["test_unit"] for f in folds],
                         ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb17"])
        self.assertEqual([f["val_subject"] for f in folds], ["chb02", "chb03", "chb06", "chb06", "chb06", "chb17", "chb01"])
        self.assertEqual([f["seed"] for f in folds], list(range(43, 50)))
        self.assertEqual((plan["format"], plan["plan_sha256"]), (pl.PLAN_FORMAT, pl.plan_sha256_of(plan)))
        self.assertEqual(len(folds[0]["test"]), 24)  # chb01 + chb21 are one subject
        rc = self.ref["rule_check"]
        self.assertEqual((rc["invariants_ok"], rc["matches_decided_rule"], rc["coverage"]), (True, True, "full"))
        self.assertEqual(self.ref["data"]["n_subjects"], 7)

    @unittest.skipUnless(lopo_folds, "chbmit.lopo_folds not importable")
    def test_reference_plan_equals_the_shared_fold_module(self):
        meta = pd.read_csv(self.meta, dtype={"patient": str, "subject": str})
        for unit in ("subject", "case"):
            ours = self.tmp / f"ref_{unit}.json"
            run_cli("--chbmit-reference-plan", ours, "--npz", self.npz, "--meta", self.meta, "--lopo-unit", unit)
            body = json.loads(ours.read_text())
            folds = lopo_folds.build_lopo_folds(meta, unit=unit, min_val_per_class=5, seed=42)
            theirs = lopo_folds.plan_to_json(folds, meta, dict(lopo_unit=unit, min_val_per_class=5, seed=42))
            self.assertEqual(body["plan_sha256"], theirs["plan_sha256"], unit)
            self.assertEqual(len(lopo_folds.plan_from_json(body, meta)), len(folds))  # replayable as LOPO_PLAN_PATH
            path = self.tmp / f"module_{unit}.json"
            path.write_text(json.dumps(theirs))
            code, out = run_cli("--chbmit-plan", path, "--npz", self.npz, "--meta", self.meta,
                                "--out", self.tmp / f"module_{unit}_lock.json")
            self.assertEqual(code, 0, out)
            self.assertIn("matches_decided_rule=True", out)

    def test_case_units_exclude_the_sibling_case(self):
        case_plan = self.tmp / "case.json"
        run_cli("--chbmit-reference-plan", case_plan, "--npz", self.npz, "--meta", self.meta, "--lopo-unit", "case")
        folds = {f["test_unit"]: f for f in json.loads(case_plan.read_text())["folds"]}
        self.assertEqual(len(folds), 9)
        self.assertEqual(folds["chb21"]["excluded_units"], ["chb01"])
        self.assertFalse(any("/chb01/" in r for r in folds["chb21"]["train"] + folds["chb21"]["val"]))
        code, out = run_cli("--chbmit-plan", case_plan, "--npz", self.npz, "--meta", self.meta, "--out", self.tmp / "c.json")
        self.assertEqual(code, 0, out)

    def test_renamed_ids_and_reordered_rows_still_match_on_signals(self):
        npz, meta = synthetic_chbmit(self.tmp / "b", permute=3, rename=lambda s: "x_" + s.replace("/", "__"))
        plan = self.tmp / "plan_b.json"
        run_cli("--chbmit-reference-plan", plan, "--npz", npz, "--meta", meta)
        code, out = run_cli("--chbmit-plan", plan, "--npz", npz, "--meta", meta, "--out", self.tmp / "b.json",
                            "--expect", self.tmp / "ref.json")
        self.assertEqual(code, 0, out)
        self.assertIn("plan_sha256 differs", out)
        self.assertIn("row order differs", out)

    def test_other_validation_choice_is_a_mismatch(self):
        def swap(body):  # fold 1 validates on chb03 instead of chb02 (still leak-free)
            f = body["folds"][0]
            ids = pd.read_csv(self.meta)
            by_subject = lambda s: ids.loc[ids.subject == s, "segment_id"].tolist()  # noqa: E731
            f["val_subject"], f["val"] = "chb03", by_subject("chb03")
            f["train"] = sorted(set(f["train"]) - set(by_subject("chb03")) | set(by_subject("chb02")))
        plan = self.edited_plan("swap.json", swap)
        code, out = run_cli("--chbmit-plan", plan, "--npz", self.npz, "--meta", self.meta, "--out", self.tmp / "s.json",
                            "--expect", self.tmp / "ref.json")
        self.assertEqual(code, 3, out)
        self.assertIn("fold 1: differs in", out)
        lock = json.loads((self.tmp / "s.json").read_text())
        self.assertTrue(lock["rule_check"]["invariants_ok"])
        self.assertIn("val_subject", lock["rule_check"]["first_difference"])

    def test_leaky_plan_is_invalid_even_without_expect(self):
        def leak(body):
            f = body["folds"][1]
            f["train"].append(f["test"][0])
        code, out = run_cli("--chbmit-plan", self.edited_plan("leak.json", leak), "--npz", self.npz, "--meta", self.meta,
                            "--out", self.tmp / "l.json")
        self.assertEqual(code, 3, out)
        self.assertIn("INVALID", out)

    def test_single_class_validation_is_invalid(self):
        def bad_val(body):
            f = body["folds"][0]
            ids = pd.read_csv(self.meta)
            chb04 = ids.loc[ids.subject == "chb04", "segment_id"].tolist()
            f["train"] = sorted(set(f["train"]) - set(chb04) | set(f["val"]))
            f["val_subject"], f["val"] = "chb04", chb04
        code, out = run_cli("--chbmit-plan", self.edited_plan("v.json", bad_val), "--npz", self.npz, "--meta",
                            self.meta, "--out", self.tmp / "v_lock.json")
        self.assertEqual(code, 3, out)
        self.assertIn("validation has [9, 0] segments per class", out)

    def test_record_level_lock_without_data(self):
        code, out = run_cli("--chbmit-plan", self.plan, "--out", self.tmp / "r.json", "--expect", self.tmp / "ref.json")
        self.assertEqual(code, 0, out)
        self.assertIn("comparing by record id only", out)
        edited = self.edited_plan("seed.json", lambda b: b["folds"][2].update(seed=7))
        code, out = run_cli("--chbmit-plan", edited, "--out", self.tmp / "r2.json", "--expect", self.tmp / "r.json")
        self.assertEqual(code, 3, out)

    def test_npz_and_metadata_must_belong_together(self):
        with np.load(self.npz) as f:
            X, y = f["X"].copy(), f["y"]
        X[5, 0, 0] += 1e-3
        np.savez(self.tmp / "tampered.npz", X=X, y=y)
        code, out = run_cli("--chbmit-plan", self.plan, "--npz", self.tmp / "tampered.npz", "--meta", self.meta,
                            "--out", self.tmp / "t.json")
        self.assertEqual(code, 2, out)
        self.assertIn("does not match NPZ row 5", out)


# ----------------------------------------------------------------------------- Bonn notebooks
NOTEBOOKS = find_notebooks()


@unittest.skipUnless(NOTEBOOKS, "shared protocol notebooks not found (set BONN_NOTEBOOK_DIR)")
class TestBonnNotebooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="plock_bonn_"))
        rng = np.random.default_rng(12345)
        cls.values = {s: [rng.integers(-2000, 2000, 4097) for _ in range(100)] for s in SETS}
        cls.data = cls.tmp / "bonn"
        write_bonn(cls.data, cls.values)
        cls.ref_path = cls.tmp / "ref_eegnet.json"
        code, out = run_cli("--notebook", NOTEBOOKS["EEGNet"], "--data", cls.data, "--out", cls.ref_path)
        assert code == 0, out
        cls.ref = json.loads(cls.ref_path.read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def lock(self, notebook, *extra, data=None):
        out_path = self.tmp / f"lock_{Path(notebook).stem}.json"
        code, out = run_cli("--notebook", notebook, "--data", data or self.data, "--out", out_path,
                            "--expect", self.ref_path, *extra)
        return code, out, json.loads(out_path.read_text()) if out_path.exists() else None

    def test_reference_lock_shape(self):
        self.assertEqual((self.ref["n_tasks"], self.ref["n_fits"], self.ref["n_recordings"]), (32, 352, 500))
        self.assertEqual(self.ref["info_only"]["model"], "EEGNet")
        self.assertEqual(self.ref["protocol"]["extra_code_cells"], [])

    def test_all_three_models_match(self):
        for model in ("ATCNet", "TCFormer"):
            code, out, lock = self.lock(NOTEBOOKS[model])
            self.assertEqual(code, 0, out)
            self.assertIn("32/32 common tasks identical", out)
            self.assertEqual(lock["info_only"]["model"], model)
            self.assertNotEqual(lock["info_only"]["model_options"], self.ref["info_only"]["model_options"])

    def test_splits_match_an_independent_reimplementation(self):
        from sklearn.model_selection import StratifiedKFold, train_test_split
        sets = np.repeat(list(SETS), 100)
        sig = [float32_sha(v) for s in SETS for v in self.values[s]]
        for task, groups in self.ref["protocol"]["tasks"][::7]:
            chosen = [s for g in groups for s in g]
            global_idx = np.flatnonzero(np.isin(sets, chosen))
            target = {s: i for i, g in enumerate(groups) for s in g}
            labels = np.array([target[s] for s in sets[global_idx]])
            strata, local = sets[global_idx], np.arange(len(global_idx))
            plans = []
            for k, (dev, test) in enumerate(StratifiedKFold(10, shuffle=True, random_state=42).split(local, strata), 1):
                tr, va = train_test_split(dev, test_size=0.15, random_state=42 + k, stratify=strata[dev])
                plans.append((f"cv_fold_{k:02d}", tr, va, test, 42 + k))
            dev, test = train_test_split(local, test_size=0.30, random_state=42, stratify=strata)
            tr, va = train_test_split(dev, test_size=0.15, random_state=1042, stratify=strata[dev])
            plans.append(("holdout_70_30", tr, va, test, 1042))
            for name, tr, va, te, seed in plans:
                parts = {p: [f"{sig[global_idx[i]]}:{labels[i]}" for i in idx] for p, idx in (("train", tr), ("val", va), ("test", te))}
                expected = pl.sha(pl.canon(dict(task=task, fit=name, seed=seed, **parts)))[:16]
                self.assertEqual(self.ref["split_hash_per_fit"][task][name], expected, f"{task}/{name}")

    def test_staged_and_sharded_runs_match(self):
        staged = variant(NOTEBOOKS["ATCNet"], self.tmp / "staged.ipynb", {"MAX_CLASSES_TO_RUN": "5"})
        code, out, lock = self.lock(staged)
        self.assertEqual(code, 0, out)
        self.assertEqual(lock["run_scope"]["tasks"], ["Z_vs_O_vs_N_vs_F_vs_S"])
        self.assertEqual(lock["n_fits"], 352)
        sharded = variant(NOTEBOOKS["TCFormer"], self.tmp / "sharded.ipynb",
                          {"SELECTED_IDS": '["Z_vs_S", "N+F_vs_S"]', "PLOT_EVERY": "100", "REUSE_COMPLETED": "False",
                           "STOP_ON_STAGE_FAILURE": "False", "RESULTS_ROOT": '"/elsewhere"'})
        code, out, lock = self.lock(sharded)
        self.assertEqual(code, 0, out)
        self.assertEqual(lock["run_scope"]["tasks"], ["Z_vs_S", "N+F_vs_S"])

    def test_protocol_changes_mismatch(self):
        cases = {
            "epochs": (dict(set_params={"EPOCHS": "50"}), "config.epochs: 50 vs 100"),
            "seed": (dict(set_params={"SEED": "7"}), "config.seed: 7 vs 42"),
            "comment": (dict(edit=("training", "def run_fit(", "# edited in Colab\ndef run_fit(")), None),  # still a match
            "clip": (dict(edit=("training", "parameters(), 1.0)", "parameters(), 2.0)")), "shared code differs: training cell"),
            "holdout": (dict(edit=("splits", "random_state=SEED+1000,", "random_state=SEED+1001,")),
                        ("shared code differs: splits cell", "0/32 common tasks identical", "fits differing ['holdout_70_30']")),
            "extra": (dict(extra_cell="EPOCHS = 5"), "extra code cells: 1 here vs 0 there"),
        }
        for name, (kwargs, message) in cases.items():
            with self.subTest(name):
                nb = variant(NOTEBOOKS["ATCNet"], self.tmp / f"{name}.ipynb", **kwargs)
                code, out, _ = self.lock(nb)
                if message is None:
                    self.assertEqual(code, 0, out)
                else:
                    self.assertEqual(code, 3, out)
                    for text in (message if isinstance(message, tuple) else (message,)):
                        self.assertIn(text, out)

    def test_invalid_selection_is_an_error(self):
        nb = variant(NOTEBOOKS["ATCNet"], self.tmp / "unknown.ipynb", {"SELECTED_IDS": '["Z_vs_Q"]'})
        code, out, _ = self.lock(nb)
        self.assertEqual(code, 2, out)
        self.assertIn("Unknown IDs", out)

    def test_papermill_tag_and_injected_parameters(self):
        injected = dict(SELECTED_IDS=["Z_vs_S"], MAX_CLASSES_TO_RUN=None, PLOT_EVERY=100, BONN_DATA_DIR="/missing",
                        RESULTS_ROOT="/runs/job", CREATE_WORD_SUMMARY_REPORT=False, MODEL_OPTIONS={"f1": 8})
        try:
            import papermill
            logging.getLogger("papermill").setLevel(logging.ERROR)
            tagged = variant(NOTEBOOKS["ATCNet"], self.tmp / "tagged.ipynb", tag=True)
            papermill.execute_notebook(str(tagged), str(self.tmp / "injected.ipynb"), parameters=injected,
                                       prepare_only=True, log_output=False, progress_bar=False)
            nb = self.tmp / "injected.ipynb"
        except ImportError:
            nb = variant(NOTEBOOKS["ATCNet"], self.tmp / "injected.ipynb", tag=True, inject=injected)
        code, out, lock = self.lock(nb)
        self.assertEqual(code, 0, out)
        self.assertEqual(lock["run_scope"]["tasks"], ["Z_vs_S"])
        self.assertEqual(lock["info_only"]["papermill"]["parameters_source"], "notebook")
        self.assertEqual(lock["info_only"]["model_options"]["f1"], 8)
        self.assertEqual(lock["info_only"]["papermill"]["injected"]["BONN_DATA_DIR"], "<path>")
        # an injected protocol change is caught; unknown injected names count as protocol
        tagged = variant(NOTEBOOKS["ATCNet"], self.tmp / "tagged2.ipynb", tag=True)
        code, out, _ = self.lock(tagged, "--parameters", json.dumps({"EPOCHS": 3, "SELECTED_IDS": ["Z_vs_S"]}))
        self.assertEqual(code, 3, out)
        self.assertIn("config.epochs: 3 vs 100", out)
        code, out, _ = self.lock(tagged, "--parameters", json.dumps({"NEW_KNOB": 1}))
        self.assertEqual(code, 3, out)
        self.assertIn("injected NEW_KNOB", out)
        # without the tag papermill injects at the top and the config cell overrides it
        code, out, lock = self.lock(NOTEBOOKS["ATCNet"], "--parameters", json.dumps({"EPOCHS": 3}))
        self.assertEqual(code, 0, out)
        self.assertIn("no cell is tagged 'parameters'", out)

    def test_data_bytes_names_and_order(self):
        crlf = self.tmp / "bonn_crlf"
        write_bonn(crlf, self.values, name=lambda s, i: f"{'ABCDE'[SETS.index(s)]}/{s}/{s}{i:03d}", newline="\r\n")
        code, out, lock = self.lock(NOTEBOOKS["ATCNet"], data=crlf)
        self.assertEqual(code, 0, out)
        self.assertIn("bytes differ but signals are identical", out)
        self.assertIn("file names or folder nesting differ", out)
        renamed = self.tmp / "bonn_nonpadded"
        write_bonn(renamed, self.values, name=lambda s, i: f"{s}/{s}{i}")
        code, out, _ = self.lock(NOTEBOOKS["ATCNet"], data=renamed)
        self.assertEqual(code, 3, out)
        self.assertIn("same recordings at signal level but a different load order", out)
        self.assertIn("0/32 common tasks identical", out)
        changed = {s: list(v) for s, v in self.values.items()}
        changed["S"][0] = changed["S"][0] + 1
        other = self.tmp / "bonn_changed"
        write_bonn(other, changed)
        code, out, _ = self.lock(NOTEBOOKS["ATCNet"], data=other)
        self.assertEqual(code, 3, out)
        self.assertIn("1 recording signals here are not in the expected lock (per set: {'S': 1})", out)


# ----------------------------------------------------------------------------- real data (opt-in)
REAL = os.environ.get("PROTOCOL_LOCK_REAL_DATA")


@unittest.skipUnless(REAL, "set PROTOCOL_LOCK_REAL_DATA to a folder with bonn/ (and chbmit_8ch.npz/_metadata.csv)")
class TestRealData(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="plock_real_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @unittest.skipUnless(NOTEBOOKS, "shared protocol notebooks not found")
    def test_bonn_models_and_run_variants(self):
        data = Path(REAL) / "bonn"
        ref = self.tmp / "eegnet.json"
        code, out = run_cli("--notebook", NOTEBOOKS["EEGNet"], "--data", data, "--out", ref)
        self.assertEqual(code, 0, out)
        runs = {"ATCNet": NOTEBOOKS["ATCNet"], "TCFormer": NOTEBOOKS["TCFormer"],
                "staged": variant(NOTEBOOKS["ATCNet"], self.tmp / "staged.ipynb", {"MAX_CLASSES_TO_RUN": "5"}),
                "selected": variant(NOTEBOOKS["ATCNet"], self.tmp / "sel.ipynb", {"SELECTED_IDS": '["Z_vs_S"]'}),
                "epochs": variant(NOTEBOOKS["ATCNet"], self.tmp / "ep.ipynb", {"EPOCHS": "50"})}
        for name, nb in runs.items():
            with self.subTest(name):
                code, out = run_cli("--notebook", nb, "--data", data, "--out", self.tmp / f"{name}.json", "--expect", ref)
                self.assertEqual(code, 3 if name == "epochs" else 0, out)
        saved = {m: ROOT / "protocol" / f"bonn_lock_{m.lower()}.json" for m in MODELS}
        reference = json.loads(ref.read_text())
        for model, path in saved.items():
            if path.exists():
                lock = json.loads(path.read_text())
                for key in ("protocol_hash", "data_order_hash", "split_plan_hash"):
                    self.assertEqual(lock[key], reference[key], f"{path.name} {key}")

    def test_chbmit_reference_plan(self):
        npz, meta = Path(REAL) / "chbmit_8ch.npz", Path(REAL) / "chbmit_8ch_metadata.csv"
        if not (npz.exists() and meta.exists()):
            self.skipTest("CHB-MIT NPZ/metadata not in PROTOCOL_LOCK_REAL_DATA")
        plan = self.tmp / "lopo_plan.json"
        code, out = run_cli("--chbmit-reference-plan", plan, "--npz", npz, "--meta", meta)
        self.assertEqual(code, 0, out)
        code, out = run_cli("--chbmit-plan", plan, "--npz", npz, "--meta", meta, "--out", self.tmp / "lock.json")
        self.assertEqual(code, 0, out)
        lock = json.loads((self.tmp / "lock.json").read_text())
        folds = json.loads(plan.read_text())["folds"]
        self.assertEqual(len(folds), 23)
        self.assertEqual(lock["data"]["n_segments"], 2053)
        self.assertEqual((lock["data"]["n_nonseizure"], lock["data"]["n_seizure"]), (1039, 1014))
        self.assertTrue(lock["rule_check"]["invariants_ok"] and lock["rule_check"]["matches_decided_rule"])
        self.assertEqual(lock["rule_check"]["coverage"], "full")
        vals = {f["val_subject"] for f in folds}
        self.assertFalse(vals & {"chb07", "chb16"})
        self.assertIn("chb07", [f["test_unit"] for f in folds])
        self.assertEqual([f["seed"] for f in folds], list(range(43, 66)))
        shared = ROOT / "protocol" / "chbmit_lopo_plan.json"  # written by chbmit/make_plan.py, if present
        if shared.exists():
            code, out = run_cli("--chbmit-plan", shared, "--npz", npz, "--meta", meta, "--out", self.tmp / "shared.json",
                                "--expect", self.tmp / "lock.json")
            self.assertEqual(code, 0, out)
            self.assertEqual(json.loads(shared.read_text())["plan_sha256"], json.loads(plan.read_text())["plan_sha256"])


if __name__ == "__main__":
    unittest.main()
