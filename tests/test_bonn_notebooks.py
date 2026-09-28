"""Tests for the Bonn notebook builder (notebooks/build_bonn_notebooks.py, notebooks/nbtools.py).

- Code-hash formula: reproduces the NOTEBOOK_CODE_SHA256 constants of the three shared
  protocol notebooks (ATCNet, EEGNet, TCFormer).
- Primary notebook: every cell source byte-identical to the shared ATCNet notebook; the only
  difference is the papermill 'parameters' tag on cell 4.
- head_max_norm variant: differs only in cells 4, 12, 14 and 20, by exactly the intended lines;
  with head_max_norm=None the model reproduces the original (init, RNG use, outputs, training
  steps); with 0.25 the window classifiers are max-norm constrained; the exported inference
  code loads variant checkpoints.

The shared protocol notebooks and notebooks/ are not public. Set ATCNET_REFERENCE_NB to a local
copy of Bonn_ATCNet_5_to_2_Class_Validation.ipynb (the EEGNet and TCFormer notebooks are looked
up next to it); otherwise the builder's default input is used. Tests skip when either is missing.

Run from the repository root:
    python -m unittest discover -s tests -p "test_bonn_notebooks.py" -v
"""
import ast
import copy
import difflib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
if (NB_DIR / "build_bonn_notebooks.py").is_file():
    sys.path.insert(0, str(NB_DIR))
    import build_bonn_notebooks as B
    import nbtools as T
else:   # notebooks/ is kept local until the shared notebooks are published
    B = T = None

ORIGINAL = None
if B is not None:
    ORIGINAL = Path(os.environ.get("ATCNET_REFERENCE_NB", "").strip() or B.DEFAULT_ORIGINAL)
HAVE_INPUTS = B is not None and ORIGINAL.is_file()
needs_inputs = unittest.skipUnless(HAVE_INPUTS, "notebooks/ or the shared protocol notebook is not available")

EXPECTED_CODE_HASH = {
    "ATCNet": "7fff1ef859330704b6720988fdcce641ff92b1969b2621658ce4925f3ec2a3f3",
    "EEGNet": "59b9f6b210c4279221363072b710d4317bfbb72b7fd923f869f13ea5f81c108b",
    "TCFormer": "2df8d5f6c2625a12308e6b638f6d6c52321b6612025ecf487ebb5127c8dfd433",
}

_CACHE = {}


def built():
    if "built" not in _CACHE:
        out = B.build_all(ORIGINAL)
        _CACHE["built"] = dict(original=out["original"], original_bytes=ORIGINAL.read_bytes(),
                               primary=out["outputs"][B.PRIMARY_NAME], variant=out["outputs"][B.VARIANT_NAME])
    return _CACHE["built"]


def src(nb, i):
    return T.cell_source(nb["cells"][i])


def changed_lines(a, b):
    """(removed, added) lines between two texts."""
    removed, added = [], []
    for line in difflib.ndiff(a.splitlines(), b.splitlines()):
        if line.startswith("- "):
            removed.append(line[2:])
        elif line.startswith("+ "):
            added.append(line[2:])
    return removed, added


# ------------------------------------------------------------------------------ code hash

@needs_inputs
class CodeHash(unittest.TestCase):
    def test_reproduces_the_three_original_constants(self):
        for model, expected in EXPECTED_CODE_HASH.items():
            with self.subTest(model=model):
                path = ORIGINAL.parent / f"Bonn_{model}_5_to_2_Class_Validation.ipynb"
                if not path.is_file():
                    self.skipTest(f"{path.name} not available")
                nb = T.load_nb(path)
                self.assertEqual(T.code_hash_constant(nb), expected)
                self.assertEqual(T.code_hash(nb), expected)
                self.assertEqual(T.code_hash_cell_index(nb), 12)

    def test_ignores_constant_cell_and_injected_parameters(self):
        nb = copy.deepcopy(built()["primary"])
        h = T.code_hash(nb)
        injected = dict(cell_type="code", execution_count=None, id="injected", outputs=[],
                        metadata={"tags": ["injected-parameters"]}, source=['EPOCHS = 1\n'])
        nb["cells"].insert(5, injected)
        self.assertEqual(T.code_hash(nb), h)
        T.set_cell_source(nb["cells"][13], src(nb, 13).replace(h, "0" * 64))
        self.assertEqual(T.code_hash(nb), h)

    def test_tracks_code_edits_and_set_constant(self):
        nb = copy.deepcopy(built()["primary"])
        h = T.code_hash(nb)
        T.set_cell_source(nb["cells"][16], src(nb, 16) + "\n# edit\n")
        self.assertNotEqual(T.code_hash(nb), h)
        new = T.set_code_hash_constant(nb)
        self.assertEqual(T.code_hash_constant(nb), new)
        self.assertEqual(T.code_hash(nb), new)

    def test_source_split_round_trips(self):
        for cell in built()["original"]["cells"]:
            self.assertEqual(T.split_source(T.cell_source(cell)), cell["source"])


# ------------------------------------------------------------------------------ primary

@needs_inputs
class Primary(unittest.TestCase):
    def test_every_cell_source_is_byte_identical(self):
        b = built()
        self.assertEqual(len(b["primary"]["cells"]), len(b["original"]["cells"]))
        for i, (a, p) in enumerate(zip(b["original"]["cells"], b["primary"]["cells"])):
            with self.subTest(cell=i):
                self.assertEqual(p["source"], a["source"])
                self.assertEqual(T.cell_source(p).encode("utf-8"), T.cell_source(a).encode("utf-8"))
                for key in ("cell_type", "id", "outputs", "execution_count"):
                    self.assertEqual(p.get(key), a.get(key))

    def test_only_the_parameters_tag_differs(self):
        b = built()
        self.assertEqual(T.cell_diff(b["original"], b["primary"]),
                         [dict(index=4, cell_type="code", parts=["metadata"])])
        self.assertEqual(b["primary"]["cells"][4]["metadata"], {"tags": ["parameters"]})
        self.assertEqual(b["primary"]["metadata"], b["original"]["metadata"])
        self.assertEqual(T.parameters_cell_index(b["primary"]), 4)

    def test_untagged_serialization_equals_original_file(self):
        b = built()
        nb = copy.deepcopy(b["primary"])
        nb["cells"][4]["metadata"] = {}
        self.assertEqual(T.dumps_nb(nb).encode("utf-8"), b["original_bytes"])

    def test_parameters_cell_holds_the_papermill_contract(self):
        text = src(built()["primary"], 4)
        for name in ("BONN_DATA_DIR", "RESULTS_ROOT", "SELECTED_IDS", "MAX_CLASSES_TO_RUN", "EPOCHS",
                     "PLOT_EVERY", "REUSE_COMPLETED", "MODEL_OPTIONS"):
            self.assertIn(f"\n{name} = ", "\n" + text)

    def test_file_on_disk_is_current(self):
        path = NB_DIR / B.PRIMARY_NAME
        if not path.is_file():
            self.skipTest("primary notebook not built")
        self.assertEqual(path.read_bytes(), T.dumps_nb(built()["primary"]).encode("utf-8"))


# ------------------------------------------------------------------------------ variant

@needs_inputs
class Variant(unittest.TestCase):
    def test_differs_only_in_intended_cells(self):
        b = built()
        diff = T.cell_diff(b["primary"], b["variant"])
        self.assertEqual([d["index"] for d in diff], [4, 12, 14, 20])
        self.assertTrue(all(d["parts"] == ["source"] for d in diff), diff)
        self.assertEqual(b["variant"]["metadata"], b["original"]["metadata"])
        B.check_variant(b["original"], b["variant"])   # raises on any unintended change

    def test_exact_changed_lines(self):
        b = built()
        o, v = b["original"], b["variant"]
        sig_old = "                 kernel_length=64,pool_size=7,n_windows=5,tcn_depth=2,tcn_kernel=4,dropout=.3):"
        model_added = [
            sig_old[:-2] + ",head_max_norm=None):",
            "        self.head_max_norm=head_max_norm  # None: original port (no dense max-norm)",
            "        if self.head_max_norm is not None:",
            "            # Opt-in variant: classic ATCNet dense max-norm on each window classifier.",
            "            for classifier in self.classifiers:",
            "                limit_output_norm(classifier.weight,self.head_max_norm)",
        ]
        self.assertEqual(changed_lines(src(o, 14), src(v, 14)), ([sig_old], model_added))
        inf_o = T.string_assignment(src(o, 20), "INFERENCE_SOURCE")
        inf_v = T.string_assignment(src(v, 20), "INFERENCE_SOURCE")
        self.assertEqual(changed_lines(inf_o, inf_v), ([sig_old], model_added))
        rest_o = [l for l in src(o, 20).splitlines() if not l.startswith("INFERENCE_SOURCE = ")]
        rest_v = [l for l in src(v, 20).splitlines() if not l.startswith("INFERENCE_SOURCE = ")]
        self.assertEqual(rest_o, rest_v)
        removed, added = changed_lines(src(o, 4), src(v, 4))
        self.assertEqual([l.split(" = ")[0] for l in removed], ["RESULTS_ROOT", "MODEL_OPTIONS"])
        self.assertEqual([l.split(" = ")[0] for l in added], ["RESULTS_ROOT", "MODEL_OPTIONS"])
        removed, added = changed_lines(src(o, 12), src(v, 12))
        self.assertEqual(len(removed), 1)
        self.assertEqual(len(added), 1)
        self.assertTrue(removed[0].startswith("NOTEBOOK_CODE_SHA256 = ") and added[0].startswith("NOTEBOOK_CODE_SHA256 = "))

    def test_code_hash_recomputed_with_original_formula(self):
        b = built()
        new = T.code_hash_constant(b["variant"])
        self.assertEqual(new, T.code_hash(b["variant"]))
        self.assertNotEqual(new, T.code_hash_constant(b["original"]))

    def test_model_options_results_root_and_study_id_inputs(self):
        b = built()
        opts_o = B.literal_assignment(src(b["original"], 4), "MODEL_OPTIONS")
        opts_v = B.literal_assignment(src(b["variant"], 4), "MODEL_OPTIONS")
        self.assertEqual(opts_v, dict(opts_o, head_max_norm=0.25))
        root_o = B.literal_assignment(src(b["original"], 4), "RESULTS_ROOT")
        root_v = B.literal_assignment(src(b["variant"], 4), "RESULTS_ROOT")
        self.assertNotEqual(root_o, root_v)
        self.assertIn("HEADMAXNORM", root_v)
        # Both the model options and the code hash enter STUDY_ID, so no primary fit is reused.
        c12 = src(b["variant"], 12)
        self.assertIn("model=MODEL_OPTIONS", c12)
        self.assertIn("code=NOTEBOOK_CODE_SHA256", c12)

    def test_patch_targets_must_match_exactly_once(self):
        original = built()["original"]
        missing = copy.deepcopy(original)
        T.set_cell_source(missing["cells"][14], src(missing, 14).replace(
            "        limit_output_norm(self.spatial.weight,1.0)\n", "        limit_output_norm(self.spatial.weight, 1.0)\n"))
        with self.assertRaises(T.PatchError):
            B.build_variant(missing)
        twice = copy.deepcopy(original)
        T.set_cell_source(twice["cells"][4], src(twice, 4) + B.PARAMS_REPLACEMENTS[1][0])
        with self.assertRaises(T.PatchError):
            B.build_variant(twice)
        with self.assertRaises(T.PatchError):
            T.replace_exact("abc", "x", "y", "demo")
        with self.assertRaises(T.PatchError):
            T.replace_exact("xx", "x", "y", "demo")

    def test_file_on_disk_is_current(self):
        path = NB_DIR / B.VARIANT_NAME
        if not path.is_file():
            self.skipTest("variant notebook not built")
        self.assertEqual(path.read_bytes(), T.dumps_nb(built()["variant"]).encode("utf-8"))


# ------------------------------------------------------------------------------ model code

def exec_model_code(cell14_source):
    """Execute the model definitions of cell 14 (everything before the parameter probe)."""
    ns = {"torch": torch, "nn": nn, "np": np}
    exec(compile(cell14_source[:cell14_source.index(B.PROBE_START)], "<cell 14>", "exec"), ns)
    return ns


def exec_inference(cell20_source):
    module = types.ModuleType("bonn_inference")
    exec(compile(T.string_assignment(cell20_source, "INFERENCE_SOURCE"), "<inference.py>", "exec"), module.__dict__)
    return module


def train_steps(model, x, y, steps=3, seed=7):
    """Cell-16 style updates: AdamW, class-weighted CE, grad clip 1.0, constraints after each step."""
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]))
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.constrain_weights()
    return model


@needs_inputs
class ModelEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b = built()
        cls.orig = exec_model_code(src(b["original"], 14))
        cls.var = exec_model_code(src(b["variant"], 14))
        opts = B.literal_assignment(src(b["original"], 4), "MODEL_OPTIONS")
        cls.options = dict(opts, n_times=4097)   # cell 10 sets n_times from the data
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.threads))
        g = torch.Generator().manual_seed(0)
        cls.x = torch.randn(6, 1, 4097, generator=g) * 100.0   # Bonn-like amplitude (raw values)
        cls.y = torch.tensor([0, 1, 0, 1, 1, 0])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def make(self, ns, n_classes, seed=11, **extra):
        torch.manual_seed(seed)
        model = ns["BonnClassifier"](n_classes=n_classes, **self.options, **extra)
        return model, torch.get_rng_state()

    def assert_same_state(self, a, b):
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(list(sa), list(sb))
        for k in sa:
            self.assertTrue(torch.equal(sa[k], sb[k]), k)

    def test_default_none_reproduces_original(self):
        for n_classes in (2, 5):
            for extra in ({}, {"head_max_norm": None}):
                with self.subTest(n_classes=n_classes, extra=extra):
                    a, rng_a = self.make(self.orig, n_classes)
                    b, rng_b = self.make(self.var, n_classes, **extra)
                    self.assertTrue(torch.equal(rng_a, rng_b))
                    self.assert_same_state(a, b)
                    a.eval(), b.eval()
                    with torch.no_grad():
                        self.assertTrue(torch.equal(a(self.x), b(self.x)))
                    a.train(), b.train()
                    with torch.no_grad():
                        torch.manual_seed(3)
                        out_a = a(self.x)
                        torch.manual_seed(3)
                        out_b = b(self.x)
                    self.assertTrue(torch.equal(out_a, out_b))
                    y = self.y % n_classes
                    if n_classes == 2:
                        train_steps(a, self.x, y)
                        train_steps(b, self.x, y)
                        self.assert_same_state(a, b)
                        a.eval(), b.eval()
                        with torch.no_grad():
                            self.assertTrue(torch.equal(a(self.x), b(self.x)))

    def test_head_max_norm_constrains_window_classifiers(self):
        a, _ = self.make(self.orig, 3)
        b, _ = self.make(self.var, 3, head_max_norm=0.25)
        self.assert_same_state(a, b)   # same initialization; the constraint acts only after updates
        before = {k: v.clone() for k, v in b.state_dict().items()}
        norms_before = torch.stack([c.weight.norm(dim=1) for c in b.classifiers])
        self.assertGreater(norms_before.max().item(), 0.25)
        a.constrain_weights()
        b.constrain_weights()
        for c in b.classifiers:
            self.assertLessEqual(c.weight.norm(dim=1).max().item(), 0.25 + 1e-6)
        for i, (ca, cb) in enumerate(zip(a.classifiers, b.classifiers)):
            self.assertTrue(torch.equal(ca.weight, before[f"classifiers.{i}.weight"]))   # original: unconstrained
            self.assertTrue(torch.equal(cb.bias, ca.bias))
        self.assertTrue(torch.equal(a.spatial.weight, b.spatial.weight))
        after = b.state_dict()
        for k in after:
            if not (k.startswith("classifiers.") and k.endswith(".weight")):
                self.assertTrue(torch.equal(after[k], before[k]), k)

    def test_inference_export_matches_cell14_and_loads_variant_checkpoints(self):
        b = built()
        for key in ("primary", "variant"):
            with self.subTest(notebook=key):
                self.assertEqual(B.model_code_from_cell14(src(b[key], 14)),
                                 B.model_code_from_inference(src(b[key], 20)))
        model, _ = self.make(self.var, 2, head_max_norm=0.25)
        train_steps(model, self.x, self.y, steps=1)
        model.eval()
        config = dict(n_classes=2, **self.options, head_max_norm=0.25)   # as cell 16 stores it
        checkpoint = dict(model_state=model.state_dict(), model_config=config,
                          data_handling=dict(fs=173.61), config=dict(class_names=["Z", "S"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.pt"
            torch.save(checkpoint, path)
            inference = exec_inference(src(b["variant"], 20))
            loaded, saved = inference.load_classifier(path)
            self.assertEqual(loaded.head_max_norm, 0.25)
            with torch.no_grad():
                self.assertTrue(torch.equal(loaded(self.x), model(self.x)))
            label, probabilities = inference.predict_recording(self.x[0, 0].numpy(), loaded, saved)
            self.assertIn(label, ("Z", "S"))
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)
            # The unpatched export could not rebuild the variant model.
            with self.assertRaises(TypeError):
                exec_inference(src(b["original"], 20)).load_classifier(path)


if __name__ == "__main__":
    unittest.main()
