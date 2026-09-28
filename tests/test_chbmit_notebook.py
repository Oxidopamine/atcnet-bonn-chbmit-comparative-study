"""Tests for the CHB-MIT leave-one-patient-out notebook builder (notebooks/build_chbmit_notebook.py).

- The builder is deterministic, the notebook on disk is the current build, and a changed template
  makes the build fail instead of silently patching the wrong code.
- The papermill 'parameters' cell holds only plain literal assignments of the agreed names, parses
  with papermill without warnings, and injected parameters land right after it, before the setup
  cell that validates them.
- NOTEBOOK_CODE_SHA256 equals the shared notebooks' formula over the built notebook, also after
  papermill injects parameters.
- The study fingerprint reads every result-changing setting and none of the operational knobs, so
  jobs that run different folds share one STUDY_ID (checked statically and by executing it).
- The embedded code equals the repository modules (atcnet_torch.py, chbmit/*.py).
- The patched shared cell 16 gives identical metrics when both classes are present, survives
  single-class test folds, and its AGGREGATE_ONLY guard raises before a completion marker is
  deleted or anything is trained.
- The exported inference code reloads a multichannel checkpoint and applies the recorded unit scale.
- On the real export, the notebook's own load and plan cells replaying protocol/chbmit_lopo_plan.json
  (LOPO_PLAN_PATH) give the same 23 folds, plan_sha256 and fold table as building the plan from the
  data; a fold subset keeps the plan's fold numbers and seeds.

notebooks/ and atcnet/ are not public, and rebuilding needs the shared protocol notebook: set
ATCNET_REFERENCE_NB to a local copy of Bonn_ATCNet_5_to_2_Class_Validation.ipynb (otherwise the
builder's default input is used). The plan-replay tests need the real export: set CHBMIT_NPZ and
CHBMIT_META (or place the files under data/ in the repo). Tests skip when their inputs are missing.

Run from the repository root:
    python -m unittest discover -s tests -p "test_chbmit_notebook.py" -v
"""
import ast
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_PATH = NB_DIR / "CHBMIT_ATCNet_LOPO.ipynb"
if (NB_DIR / "build_chbmit_notebook.py").is_file():
    sys.path.insert(0, str(NB_DIR))
    import build_chbmit_notebook as B
    import nbtools as T
else:   # notebooks/ is kept local until the shared notebooks are published
    B = T = None

try:
    import torch
    sys.path.insert(0, str(ROOT))
    from atcnet.atcnet_torch import atcnet_geometry, model_source
except ImportError:
    torch = None

REAL_NPZ = Path(os.environ.get("CHBMIT_NPZ", ROOT / "data" / "chbmit_8ch.npz"))
REAL_META = Path(os.environ.get("CHBMIT_META", ROOT / "data" / "chbmit_8ch_metadata.csv"))
SAVED_PLAN = ROOT / "protocol" / "chbmit_lopo_plan.json"
SAVED_CSV = ROOT / "protocol" / "chbmit_lopo_folds.csv"

HAVE_NB = B is not None and NB_PATH.is_file()
TEMPLATE = Path(os.environ.get("ATCNET_REFERENCE_NB", "").strip() or B.DEFAULT_TEMPLATE) if B else None
HAVE_BUILD_INPUTS = HAVE_NB and TEMPLATE.is_file() and all(p.is_file() for p in B.SOURCES.values())
needs_nb = unittest.skipUnless(HAVE_NB, "notebooks/CHBMIT_ATCNet_LOPO.ipynb or its builder is not available")
needs_build = unittest.skipUnless(HAVE_BUILD_INPUTS, "the shared protocol notebook or the source modules are missing")
needs_torch = unittest.skipUnless(HAVE_NB and torch is not None, "torch or atcnet/ is not available")
needs_real = unittest.skipUnless(HAVE_NB and REAL_NPZ.is_file() and REAL_META.is_file() and SAVED_PLAN.is_file()
                                 and SAVED_CSV.is_file(),
                                 "real CHB-MIT export (set CHBMIT_NPZ / CHBMIT_META) or protocol plan files missing")

CONTRACT = ("CHBMIT_NPZ", "CHBMIT_META", "RESULTS_ROOT", "LOPO_UNIT", "SELECTED_TEST_UNITS", "LOPO_PLAN_PATH",
            "MIN_VAL_PER_CLASS", "UNIT_SCALE", "SEED", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY",
            "PLOT_EVERY", "REUSE_COMPLETED", "AGGREGATE_ONLY", "CREATE_WORD_SUMMARY_REPORT", "MODEL_OPTIONS")

_CACHE = {}


def notebook():
    if "nb" not in _CACHE:
        _CACHE["nb"] = T.load_nb(NB_PATH)
    return copy.deepcopy(_CACHE["nb"])


def code_cell(nb, marker):
    hits = [T.cell_source(c) for c in nb["cells"] if c["cell_type"] == "code" and marker in T.cell_source(c)]
    if len(hits) != 1:
        raise AssertionError(f"{marker!r} in {len(hits)} code cells")
    return hits[0]


def params_cell(nb):
    return T.cell_source(nb["cells"][T.parameters_cell_index(nb)])


def notebook_namespace(nb, **extra):
    """Namespace after the dependency cell (cell 2) plus the fingerprint cell's write_json."""
    ns = {}
    exec(T.cell_source(nb["cells"][2]), ns)
    fp = T.cell_source(nb["cells"][T.code_hash_cell_index(nb)])
    fn = [n for n in ast.parse(fp).body if isinstance(n, ast.FunctionDef) and n.name == "write_json"]
    exec(compile(ast.Module(body=fn, type_ignores=[]), "write_json", "exec"), ns)
    ns.update(extra)
    return ns


# ------------------------------------------------------------------------------ builder

@needs_build
class Builder(unittest.TestCase):
    def test_deterministic_and_current(self):
        first = T.dumps_nb(B.build_from(TEMPLATE)).encode("utf-8")
        second = T.dumps_nb(B.build_from(TEMPLATE)).encode("utf-8")
        self.assertEqual(first, second)
        self.assertEqual(NB_PATH.read_bytes(), first, "notebook on disk is stale: rerun the builder")
        self.assertNotIn(b"\r", first)

    def test_template_is_read_only_input(self):
        before = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
        B.build_from(TEMPLATE)
        self.assertEqual(hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(), before)

    def test_changed_template_fails_loudly(self):
        template = T.load_nb(TEMPLATE)
        broken = copy.deepcopy(template)
        text = T.cell_source(broken["cells"][16]).replace("fpr, tpr, thresholds = roc_curve(", "fpr, tpr, th = roc_curve(")
        T.set_cell_source(broken["cells"][16], text)
        T.set_code_hash_constant(broken)
        with self.assertRaises(T.PatchError):
            B.build(broken, B.sources())
        stale = copy.deepcopy(template)
        T.set_cell_source(stale["cells"][2], T.cell_source(stale["cells"][2]) + "\n# edited")
        with self.assertRaises(B.BuildError):
            B.build(stale, B.sources())


# ------------------------------------------------------------------------------ parameters

@needs_nb
class Parameters(unittest.TestCase):
    def test_single_tagged_cell_of_plain_literal_assignments(self):
        nb = notebook()
        tagged = [i for i, c in enumerate(nb["cells"]) if "parameters" in T.cell_tags(c)]
        self.assertEqual(len(tagged), 1)
        values = B.parameter_assignments(params_cell(nb))
        self.assertEqual(tuple(values), B.PARAMETER_NAMES)
        self.assertTrue(set(CONTRACT) <= set(values))
        self.assertEqual(values["SELECTED_TEST_UNITS"], [])
        self.assertEqual(values["LOPO_PLAN_PATH"], "")
        self.assertEqual((values["LOPO_UNIT"], values["MIN_VAL_PER_CLASS"], values["UNIT_SCALE"]), ("subject", 5, 1e6))
        self.assertEqual((values["SEED"], values["EPOCHS"], values["BATCH_SIZE"]), (42, 100, 8))
        self.assertEqual((values["LEARNING_RATE"], values["WEIGHT_DECAY"]), (3e-4, 1e-3))
        self.assertIs(values["AGGREGATE_ONLY"], False)
        self.assertEqual(values["MODEL_OPTIONS"], dict(B.BONN_MODEL_OPTIONS, n_chans=8, n_times=2560))
        setup = T.cell_source(nb["cells"][tagged[0] + 1])
        self.assertTrue(setup.startswith("assert EPOCHS >= 1"), "setup must directly follow the parameters cell")

    def test_papermill_parses_every_parameter_without_warnings(self):
        import papermill
        with self.assertNoLogs("papermill", level="WARNING"):
            found = papermill.inspect_notebook(str(NB_PATH))
        values = B.parameter_assignments(params_cell(notebook()))
        self.assertEqual(list(found), list(values))
        for name, info in found.items():
            self.assertEqual(ast.literal_eval(info["default"]), values[name], name)

    def test_papermill_injection_order_and_code_hash(self):
        import nbformat
        from papermill.iorw import load_notebook_node
        from papermill.parameterize import parameterize_notebook
        injected = dict(SELECTED_TEST_UNITS=["chb01", "chb07"], AGGREGATE_ONLY=True, EPOCHS=1, PLOT_EVERY=1,
                        MODEL_OPTIONS=dict(B.CHBMIT_MODEL_OPTIONS, tcn_depth=3))
        # load_notebook_node is papermill's own loader (it adds empty tag lists), as in execute_notebook.
        nb = json.loads(nbformat.writes(parameterize_notebook(load_notebook_node(str(NB_PATH)), injected)))
        p = T.parameters_cell_index(nb)
        self.assertIn(T.INJECTED_TAG, T.cell_tags(nb["cells"][p + 1]))
        self.assertTrue(T.cell_source(nb["cells"][p + 2]).startswith("assert EPOCHS >= 1"))
        ns = {}
        exec(T.cell_source(nb["cells"][p]) + "\n" + T.cell_source(nb["cells"][p + 1]), ns)
        for name, value in injected.items():
            self.assertEqual(ns[name], value)
        self.assertEqual(T.code_hash(nb), T.code_hash_constant(nb))

    def test_setup_cell_rejects_invalid_values(self):
        nb = notebook()
        setup = T.cell_source(nb["cells"][T.parameters_cell_index(nb) + 1])
        checks = setup[:setup.index("try:\n")]   # validation part only (no seeding side effects)
        base = {}
        exec("import math\n" + params_cell(nb), base)
        exec(checks, dict(base))
        for change in (dict(LOPO_UNIT="patient"), dict(AGGREGATE_ONLY=True, REUSE_COMPLETED=False),
                       dict(SELECTED_TEST_UNITS="chb01"), dict(UNIT_SCALE=0.0), dict(MIN_VAL_PER_CLASS=0),
                       dict(AGGREGATE_ONLY="yes")):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                exec(checks, dict(base, **change))


# ------------------------------------------------------------------------------ code hash

@needs_nb
class CodeHash(unittest.TestCase):
    def test_constant_equals_the_shared_notebooks_formula(self):
        nb = notebook()
        code = [T.cell_source(c) for c in nb["cells"] if c["cell_type"] == "code"]
        holder = [s for s in code if s.startswith('NOTEBOOK_CODE_SHA256 = "')]
        self.assertEqual(len(holder), 1)
        expected = hashlib.sha256("\n".join(s for s in code if s is not holder[0]).encode("utf-8")).hexdigest()
        self.assertEqual(T.code_hash_constant(nb), expected)
        self.assertEqual(T.code_hash(nb), expected)

    def test_any_code_edit_changes_the_hash(self):
        nb = notebook()
        i = [k for k, c in enumerate(nb["cells"]) if "def run_fit(" in T.cell_source(c)][0]
        T.set_cell_source(nb["cells"][i], T.cell_source(nb["cells"][i]).replace("patience=4", "patience=5"))
        self.assertNotEqual(T.code_hash(nb), T.code_hash_constant(nb))


# ------------------------------------------------------------------------------ fingerprint

@needs_nb
class Fingerprint(unittest.TestCase):
    OPERATIONAL = dict(RESULTS_ROOT="/elsewhere", CHBMIT_NPZ="/x/other.npz", CHBMIT_META="/x/other.csv",
                       SELECTED_TEST_UNITS=["chb07"], LOPO_PLAN_PATH="/x/plan.json", PLOT_EVERY=100,
                       REUSE_COMPLETED=False, AGGREGATE_ONLY=True, CREATE_WORD_SUMMARY_REPORT=False,
                       DOWNLOAD_WORD_SUMMARY_REPORT=False, DOWNLOAD_GRAPHS_AND_TABLES=True, RUN_FOLDS=[])

    def base(self):
        meta = pd.DataFrame(dict(record_id=["a", "b", "c"], signal_sha256=["1" * 64, "2" * 64, "3" * 64]))
        opts = dict(B.CHBMIT_MODEL_OPTIONS)
        return dict(SEED=42, EPOCHS=100, BATCH_SIZE=8, LEARNING_RATE=3e-4, WEIGHT_DECAY=1e-3, LOPO_UNIT="subject",
                    MIN_VAL_PER_CLASS=5, PLAN_SHA256="f" * 64, TASK_ID="nonseizure_vs_seizure", FS=256, N_CHANS=8,
                    TARGET_LENGTH=2560, UNIT_SCALE=1e6, DATA_HANDLING=dict(fs=256, unit_scale=1e6), MODEL_OPTIONS=opts,
                    MODEL_PROVENANCE=dict(name="ATCNet"), metadata=meta, VERSIONS=dict(torch="2", device="cpu"),
                    RESULTS_ROOT="/r", CHBMIT_NPZ="/d.npz", CHBMIT_META="/d.csv", SELECTED_TEST_UNITS=[],
                    LOPO_PLAN_PATH="", PLOT_EVERY=10, REUSE_COMPLETED=True, AGGREGATE_ONLY=False,
                    CREATE_WORD_SUMMARY_REPORT=True, DOWNLOAD_WORD_SUMMARY_REPORT=True,
                    DOWNLOAD_GRAPHS_AND_TABLES=False, RUN_FOLDS=[1, 2])

    def study_id(self, _code=None, **changes):
        nb = notebook()
        text = T.cell_source(nb["cells"][T.code_hash_cell_index(nb)])
        if _code is not None:   # a different code hash (the builder writes it into this cell)
            text = text.replace(T.code_hash_constant(nb), _code)
        body = []
        for node in ast.parse(text).body:
            body.append(node)
            if isinstance(node, ast.Assign) and any(getattr(x, "id", "") == "STUDY_ID" for x in node.targets):
                break
        ns = dict(self.base(), hashlib=hashlib, json=json)
        ns.update(changes)
        exec(compile(ast.Module(body=body, type_ignores=[]), "fingerprint", "exec"), ns)
        return ns["STUDY_ID"]

    def test_static_names(self):
        nb = notebook()
        names = B.fingerprint_names(T.cell_source(nb["cells"][T.code_hash_cell_index(nb)]))
        self.assertFalse(names & set(B.OPERATIONAL_NAMES))
        self.assertTrue(set(B.FINGERPRINT_NAMES) <= names)

    def test_operational_knobs_keep_the_study_id(self):
        reference = self.study_id()
        for name, value in self.OPERATIONAL.items():
            with self.subTest(name=name):
                self.assertEqual(self.study_id(**{name: value}), reference)
        self.assertEqual(self.study_id(**self.OPERATIONAL), reference)

    def test_result_changing_settings_change_the_study_id(self):
        reference = self.study_id()
        base = self.base()
        changed_meta = base["metadata"].assign(signal_sha256=["1" * 64, "2" * 64, "4" * 64])
        changes = dict(SEED=43, EPOCHS=1, BATCH_SIZE=16, LEARNING_RATE=1e-3, WEIGHT_DECAY=0.0, LOPO_UNIT="case",
                       MIN_VAL_PER_CLASS=3, PLAN_SHA256="e" * 64, UNIT_SCALE=1.0,
                       DATA_HANDLING=dict(fs=256, unit_scale=1.0), MODEL_PROVENANCE=dict(name="other"),
                       metadata=changed_meta, VERSIONS=dict(torch="3", device="cuda"), TASK_ID="other")
        seen = {reference}
        for name, value in changes.items():
            with self.subTest(name=name):
                sid = self.study_id(**{name: value})
                self.assertNotIn(sid, seen)
                seen.add(sid)
        for variant in (dict(tcn_depth=3), dict(head_max_norm=0.25)):
            with self.subTest(variant=variant):
                self.assertNotEqual(self.study_id(MODEL_OPTIONS=dict(base["MODEL_OPTIONS"], **variant)), reference)
        nb = notebook()
        self.assertNotEqual(self.study_id(_code="0" * 64), reference, "NOTEBOOK_CODE_SHA256 must enter STUDY_ID")
        self.assertEqual(self.study_id(_code=T.code_hash_constant(nb)), reference)

    def test_write_json_uses_per_writer_temp_names(self):
        nb = notebook()
        ns = notebook_namespace(nb)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.json"
            ns["write_json"](target, dict(x=1))
            ns["write_json"](target, dict(x=2))
            self.assertEqual(json.loads(target.read_text()), dict(x=2))
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["a.json"])
        self.assertIn("os.getpid()", T.cell_source(nb["cells"][T.code_hash_cell_index(nb)]))


# ------------------------------------------------------------------------------ embedded code

@needs_nb
class EmbeddedCode(unittest.TestCase):
    def test_modules_are_embedded_verbatim(self):
        nb = notebook()
        src = B.sources()
        cells = [T.cell_source(c) for c in nb["cells"] if c["cell_type"] == "code"]
        for name, rel in (("loader", "chbmit/loader.py"), ("lopo_folds", "chbmit/lopo_folds.py"),
                          ("lopo_metrics", "chbmit/lopo_metrics.py")):
            self.assertEqual(sum(c == B.PASTED.format(rel=rel) + src[name] for c in cells), 1, name)
        self.assertIn(src["atcnet"], code_cell(nb, "class ATCNet(nn.Module):\n"))
        inference = B.literal_assignment(code_cell(nb, "INFERENCE_SOURCE = "), "INFERENCE_SOURCE")
        self.assertIn(src["atcnet"], inference)
        model_cell = code_cell(nb, "class ATCNet(nn.Module):\n")
        self.assertIn(inference[inference.index('"""Generalized PyTorch ATCNet'):], model_cell)

    @needs_torch
    def test_model_source_helper_matches_the_embedded_text(self):
        self.assertEqual(model_source().replace("\r\n", "\n"), B.sources()["atcnet"])

    def test_hygiene(self):
        nb = notebook()
        for i, c in enumerate(nb["cells"]):
            text = T.cell_source(c)
            self.assertNotIn(B.OFF_TOPIC_MODEL, text, f"cell {i}")
            if c["cell_type"] == "code":
                tree = ast.parse(text)
                names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                self.assertFalse(names & {"BonnClassifier", "BONN_DATA_DIR"}, f"cell {i}")
                self.assertEqual(c["outputs"], [])
                self.assertIsNone(c["execution_count"])
        self.assertEqual(len({c["id"] for c in nb["cells"]}), len(nb["cells"]))

    def test_results_layout_matches_the_job_queue(self):
        nb = notebook()
        fp = T.cell_source(nb["cells"][T.code_hash_cell_index(nb)])
        self.assertIn('STUDY_DIR = Path(RESULTS_ROOT) / f"study_{STUDY_ID}"', fp)
        execute = code_cell(nb, "def fold_positions(fold):")
        self.assertIn("folder = STUDY_DIR / fold.name", execute)            # fold_<unit>/complete.json
        folds = B.sources()["lopo_folds"]
        self.assertIn('name=f"fold_{test_unit}"', folds)
        for name in ("lopo_fold_metrics.csv", "lopo_summary.json", "lopo_fold_summary.csv",
                     "lopo_per_patient_metrics.csv", "lopo_out_of_fold_predictions.csv", '"lopo_pooled"'):
            self.assertIn(name, execute)

    @needs_torch
    def test_markdown_geometry_matches_the_model(self):
        g2 = atcnet_geometry(2560, 256, tcn_depth=2)
        g3 = atcnet_geometry(2560, 256, tcn_depth=3)
        self.assertEqual((g2["seq_len"], g2["window_len"], g2["tcn_rf_steps"], g3["tcn_rf_steps"]), (45, 41, 19, 43))
        self.assertEqual(round(1000 * g2["step_s"]), 219)
        text = B.MD_MODEL
        for phrase in ("45 feature steps of 219 ms", "41 steps", "19 steps", "43 steps"):
            self.assertIn(phrase, text)


# ------------------------------------------------------------------------------ patched cell 16

@needs_nb
class PatchedCell16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def namespace(self, text=None, **extra):
        nb = notebook()
        ns = notebook_namespace(nb, BATCH_SIZE=8, STUDY_ID="s", REUSE_COMPLETED=True, AGGREGATE_ONLY=True,
                                EPOCHS=1, PLOT_EVERY=1, CONFIG={}, **extra)
        ns["DEVICE"] = ns["torch"].device("cpu")
        exec(text if text is not None else code_cell(nb, "def run_fit("), ns)
        return ns

    def evaluate(self, ns, truth, probs, names, tag):
        folder = Path(self.tmp.name) / tag
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ns["evaluation_artifacts"](truth, probs, names, folder, tag)
            ns["write_json"](folder / "metrics.json", result)
        return result, json.loads((folder / "metrics.json").read_text())

    def test_single_class_test_fold_gives_undefined_metrics(self):
        ns = self.namespace()
        p1 = np.random.default_rng(0).uniform(size=31)
        probs = np.c_[1 - p1, p1]
        result, saved = self.evaluate(ns, np.zeros(31, int), probs, ["nonseizure", "seizure"], "only0")
        self.assertTrue(math.isnan(result["roc_auc"]) and math.isnan(result["positive_recall"]))
        self.assertIsNone(saved["roc_auc"])
        self.assertIsNone(saved["average_precision"])
        self.assertEqual(saved["specificity"], result["specificity"])
        result, saved = self.evaluate(ns, np.ones(31, int), probs, ["nonseizure", "seizure"], "only1")
        self.assertIsNone(saved["specificity"])

    @needs_build
    def test_identical_metrics_to_the_unpatched_cell_when_both_classes_are_present(self):
        original = T.cell_source(T.load_nb(TEMPLATE)["cells"][16])
        rng = np.random.default_rng(1)
        truth = np.repeat([0, 1], 40)
        logits = rng.normal(size=(80, 2)) + 1.2 * np.eye(2)[truth]
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        a, _ = self.evaluate(self.namespace(original), truth, probs, ["nonseizure", "seizure"], "orig")
        b, _ = self.evaluate(self.namespace(), truth, probs, ["nonseizure", "seizure"], "patched")
        self.assertEqual(set(a), set(b))
        for key in a:
            self.assertEqual(a[key], b[key], key)

    def test_aggregate_only_guard_never_deletes_or_trains(self):
        n = 12
        meta = pd.DataFrame(dict(record_id=[f"r{i}" for i in range(n)], label=[0, 1] * 6))
        ns = self.namespace(metadata=meta, X_MODEL_INPUT=np.zeros((n, 8, 32), np.float32),
                            DATA_HANDLING=dict(fs=256), MODEL_OPTIONS={})
        labels = meta.label.to_numpy()
        args = (np.arange(n), labels, np.arange(0, 8), np.arange(8, 10), np.arange(10, 12))
        for stale in (True, False):
            with self.subTest(stale_marker=stale):
                folder = Path(self.tmp.name) / f"fold_guard_{stale}"
                folder.mkdir()
                if stale:
                    (folder / "complete.json").write_text(json.dumps(dict(signature="old", files=[])))
                before = sorted(p.name for p in folder.iterdir())
                with self.assertRaises(ns["AggregateOnlyMissing"]):
                    ns["run_fit"]("task", (("nonseizure",), ("seizure",)), *args, folder, 43)
                self.assertEqual(sorted(p.name for p in folder.iterdir()), before)


# ------------------------------------------------------------------------------ plan replay (real data)

@needs_real
class PlanReplay(unittest.TestCase):
    """The notebook's own load and plan cells on the real export: replaying the shared
    protocol/chbmit_lopo_plan.json through LOPO_PLAN_PATH gives the folds built from the data."""

    @classmethod
    def setUpClass(cls):
        nb = notebook()
        cls.ns = notebook_namespace(nb, display=lambda *a, **k: None)
        for marker in ("def load_chbmit_npz(", "def plan_or_build("):
            exec(code_cell(nb, marker), cls.ns)
        cls.ns.update(CHBMIT_NPZ=str(REAL_NPZ), CHBMIT_META=str(REAL_META), UNIT_SCALE=1e6,
                      MODEL_OPTIONS=dict(B.BONN_MODEL_OPTIONS))
        exec(code_cell(nb, "X_raw, y, metadata, DATA_INFO = load_chbmit_npz("), cls.ns)
        cls.plan_cell = code_cell(nb, "FOLDS, PLAN_BODY = plan_or_build(")
        cls.built = cls.run_plan()

    @classmethod
    def run_plan(cls, **params):
        ns = dict(cls.ns, LOPO_PLAN_PATH="", LOPO_UNIT="subject", MIN_VAL_PER_CLASS=5, SEED=42,
                  SELECTED_TEST_UNITS=[], EPOCHS=100)
        ns.update(params)
        exec(cls.plan_cell, ns)
        return ns

    def assert_same_folds(self, a, b):
        self.assertEqual(len(a), len(b))
        for fa, fb in zip(a, b):
            for attr in ("fold", "name", "unit", "test_unit", "test_subject", "val_subject", "excluded_units", "seed"):
                self.assertEqual(getattr(fa, attr), getattr(fb, attr), f"{fa.name}.{attr}")
            for part in ("train_idx", "val_idx", "test_idx"):
                np.testing.assert_array_equal(getattr(fa, part), getattr(fb, part), f"{fa.name}.{part}")

    def test_load_cell_fills_the_input_shape(self):
        self.assertEqual(self.ns["X_raw"].shape, (2053, 8, 2560))
        self.assertEqual((self.ns["MODEL_OPTIONS"]["n_chans"], self.ns["MODEL_OPTIONS"]["n_times"]), (8, 2560))
        self.assertEqual(self.ns["DATA_HANDLING"]["unit_scale"], 1e6)

    def test_replayed_protocol_plan_gives_the_same_folds(self):
        saved = json.loads(SAVED_PLAN.read_text(encoding="utf-8"))
        replay = self.run_plan(LOPO_PLAN_PATH=str(SAVED_PLAN))
        self.assertEqual(len(self.built["FOLDS"]), 23)
        self.assertEqual(self.built["PLAN_SHA256"], saved["plan_sha256"])
        self.assertEqual(replay["PLAN_SHA256"], saved["plan_sha256"])
        self.assert_same_folds(replay["FOLDS"], self.built["FOLDS"])
        table = pd.read_csv(SAVED_CSV, keep_default_na=False)
        fresh = replay["FOLD_MANIFEST"][table.columns.tolist()].astype(str)
        self.assertEqual(fresh.values.tolist(), table.astype(str).values.tolist())

    def test_plan_written_by_the_notebook_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lopo_plan.json"
            self.ns["write_json"](path, self.built["PLAN_BODY"])      # as the fingerprint cell saves it
            replay = self.run_plan(LOPO_PLAN_PATH=str(path))
        self.assertEqual(replay["PLAN_SHA256"], self.built["PLAN_SHA256"])
        self.assert_same_folds(replay["FOLDS"], self.built["FOLDS"])

    def test_selected_units_keep_plan_numbers_and_seeds(self):
        replay = self.run_plan(LOPO_PLAN_PATH=str(SAVED_PLAN), SELECTED_TEST_UNITS=["chb16", "chb01", "chb07"])
        full = {f.test_unit: f for f in self.built["FOLDS"]}
        self.assertEqual([f.test_unit for f in replay["RUN_FOLDS"]], ["chb01", "chb07", "chb16"])
        self.assert_same_folds(replay["RUN_FOLDS"], [full[u] for u in ("chb01", "chb07", "chb16")])
        self.assertEqual([(f.fold, f.seed) for f in replay["RUN_FOLDS"]], [(1, 43), (7, 49), (16, 58)])
        self.assertTrue({"chb07", "chb16"}.isdisjoint(f.val_subject for f in self.built["FOLDS"]))

    def test_replay_with_other_settings_is_refused(self):
        for change in (dict(SEED=43), dict(MIN_VAL_PER_CLASS=3), dict(LOPO_UNIT="case")):
            with self.subTest(change=change), self.assertRaises(self.ns["FoldPlanError"]):
                self.run_plan(LOPO_PLAN_PATH=str(SAVED_PLAN), **change)


# ------------------------------------------------------------------------------ inference export

@needs_torch
class InferenceExport(unittest.TestCase):
    def test_checkpoint_roundtrip_and_unit_scale(self):
        nb = notebook()
        ns = {}
        exec(B.literal_assignment(code_cell(nb, "INFERENCE_SOURCE = "), "INFERENCE_SOURCE"), ns)
        torch.manual_seed(0)
        config = dict(n_classes=2, **B.CHBMIT_MODEL_OPTIONS)
        model = ns["ATCNet"](**config).eval()
        channels = ["FP1-F7", "P3-O1", "P4-O2", "FP2-F8", "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7"]
        saved = dict(model_state=model.state_dict(), model_config=config,
                     data_handling=dict(fs=256, unit_scale=1e6, channels=channels),
                     config=dict(task="binary", class_names=["nonseizure", "seizure"]))
        segment = (np.random.default_rng(0).normal(size=(8, 2560)) * 3e-5).astype(np.float32)   # volts
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.pt"
            torch.save(saved, path)
            loaded, meta = ns["load_classifier"](path)
        name, probs = ns["predict_segment"](segment, loaded, meta)
        with torch.no_grad():
            expected = model(torch.from_numpy(segment * np.float32(1e6))[None]).softmax(1)[0].numpy()
        np.testing.assert_array_equal(probs, expected)
        self.assertEqual(name, ["nonseizure", "seizure"][int(expected.argmax())])
        _, again = ns["predict_segment"](segment * np.float32(1e6), loaded, meta, already_scaled=True)
        np.testing.assert_array_equal(again, expected)
        with self.assertRaises(ValueError):
            ns["predict_segment"](segment.T, loaded, meta)        # time-major layout
        with self.assertRaises(ValueError):
            ns["predict_segment"](segment, loaded, meta, sample_rate=173.61)


if __name__ == "__main__":
    unittest.main(verbosity=2)
