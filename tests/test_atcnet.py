"""Tests for atcnet.atcnet_torch.ATCNet.

ReferenceEquivalence: at n_chans=1 with default options the model is bit-identical to
the single-channel port in the shared protocol notebooks (Bonn ATCNet notebook, class
BonnClassifier): state_dict keys/order/shapes, initialization and RNG use, eval and
train outputs, gradients, constraint and five AdamW training steps.  The notebook is
not public: set ATCNET_REFERENCE_NB to a local copy of
Bonn_ATCNet_5_to_2_Class_Validation.ipynb to run these tests; otherwise they are skipped.

The other groups are self-contained: head_max_norm, the CHB-MIT geometry (8 x 2560),
multi-channel semantics against Keras-layout references, receptive field, model_source.

Run from the repository root:
    python -m unittest discover -s tests -p "test_atcnet.py" -v
"""
import math
import os
import re
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isfile(os.path.join(ROOT, "atcnet", "atcnet_torch.py")):
    sys.path.insert(0, ROOT)
    from atcnet import atcnet_torch as AT
else:   # atcnet/ is kept local until the shared port is published
    AT = None
needs_package = unittest.skipIf(AT is None, "atcnet/ package is not in this checkout")

# Shared protocol MODEL_OPTIONS (Bonn notebook, configuration cell); n_times is set per dataset.
MODEL_OPTIONS = {'f1': 16, 'depth_multiplier': 2, 'kernel_length': 64, 'pool_size': 7,
                 'n_windows': 5, 'tcn_depth': 2, 'tcn_kernel': 4, 'dropout': 0.3}
BONN = dict(MODEL_OPTIONS, n_times=4097)
CHB = dict(MODEL_OPTIONS, n_times=2560, n_chans=8)
CHB_FS = 256.0
_DETERMINISTIC = None


def setUpModule():
    global _DETERMINISTIC
    _DETERMINISTIC = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)


def tearDownModule():
    torch.use_deterministic_algorithms(bool(_DETERMINISTIC))


_REFERENCE = {}


def load_reference():
    """Exec the model part of the notebook's model cell (everything before the probe code)."""
    if "ns" in _REFERENCE:
        return _REFERENCE["ns"]
    path = os.environ.get("ATCNET_REFERENCE_NB", "").strip()
    if not path:
        raise unittest.SkipTest(
            "ATCNET_REFERENCE_NB is not set: the equivalence tests need a local copy of the "
            "shared protocol notebook Bonn_ATCNet_5_to_2_Class_Validation.ipynb (not public)")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ATCNET_REFERENCE_NB does not point to a file: {path}")
    import json
    with open(path, encoding="utf-8") as handle:
        cells = json.load(handle)["cells"]
    sources = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
               for c in cells if c.get("cell_type") == "code"]
    model_cells = [s for s in sources if re.search(r"^class BonnClassifier\(", s, flags=re.M)]
    if len(model_cells) != 1:
        raise ValueError(f"expected one code cell defining BonnClassifier, found {len(model_cells)}")
    src = model_cells[0]
    if "class TCNResidual(" not in src or "class AttentionBlock(" not in src:
        raise ValueError("ATCNET_REFERENCE_NB is not the ATCNet notebook (no TCN/attention blocks)")
    probe = re.search(r"^probe\s*=\s*BonnClassifier", src, flags=re.M)
    if probe is None:
        raise ValueError("model cell has no 'probe=BonnClassifier' line to cut at")
    ns = {"torch": torch, "nn": nn, "math": math}
    try:
        import numpy as np
        ns["np"] = np
    except ImportError:
        pass
    exec(compile(src[:probe.start()], "reference_model_cell", "exec"), ns)
    _REFERENCE["ns"] = ns
    return ns


def make_pair(reference_cls, seed, n_classes, n_times, **extra):
    opts = dict(MODEL_OPTIONS, n_times=n_times)
    torch.manual_seed(seed)
    ref = reference_cls(n_classes=n_classes, **opts)
    rng_ref = torch.get_rng_state()
    torch.manual_seed(seed)
    gen = AT.ATCNet(n_classes=n_classes, **opts, **extra)
    rng_gen = torch.get_rng_state()
    return ref, gen, rng_ref, rng_gen


def assert_state_equal(tc, a, b):
    sa, sb = a.state_dict(), b.state_dict()
    tc.assertEqual(list(sa.keys()), list(sb.keys()))
    for k in sa:
        tc.assertEqual(sa[k].shape, sb[k].shape, k)
        tc.assertTrue(torch.equal(sa[k], sb[k]), f"tensor differs: {k}")


def train_steps(model, x, y, n_steps=5, seed=2024, batch=8, weight=None):
    """Inner loop of the shared training cell: AdamW, grad clip 1.0, constrain after step."""
    crit = nn.CrossEntropyLoss(weight=weight)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    model.train()
    losses = []
    for i in range(n_steps):
        xb, yb = x[i * batch:(i + 1) * batch], y[i * batch:(i + 1) * batch]
        opt.zero_grad(set_to_none=True)
        loss = crit(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.constrain_weights()
        losses.append(float(loss.detach()))
    return losses


def row_norms(weight):
    return weight.detach().square().sum(dim=tuple(range(1, weight.ndim))).sqrt()


# ---------------------------------------------------------------------------
# n_chans == 1: bit-identity with the shared protocol notebook's port
# ---------------------------------------------------------------------------
@needs_package
class ReferenceEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_reference()
        cls.Ref = cls.ns["BonnClassifier"]

    def test_keys_shapes_init_rng(self):
        for n_classes in (2, 3, 4, 5):
            for n_times in (4097, 4096):
                ref, gen, rr, rg = make_pair(self.Ref, 1234 + n_classes, n_classes, n_times)
                assert_state_equal(self, ref, gen)
                self.assertTrue(torch.equal(rr, rg), "RNG consumption during __init__ differs")
                self.assertEqual([type(m).__name__ for m in ref.modules()][1:],
                                 [type(m).__name__ for m in gen.modules()][1:])

    def test_notebook_seed_42(self):
        ref, gen, _, _ = make_pair(self.Ref, 42, 5, 4097)
        assert_state_equal(self, ref, gen)

    def test_strict_checkpoint_roundtrip(self):
        ref, _, _, _ = make_pair(self.Ref, 7, 5, 4097)
        gen = AT.ATCNet(n_classes=5, **BONN)
        gen.load_state_dict(ref.state_dict(), strict=True)
        ref2 = self.Ref(n_classes=5, **BONN)
        ref2.load_state_dict(gen.state_dict(), strict=True)

    def test_outputs_and_grads_bit_identical(self):
        ref, gen, _, _ = make_pair(self.Ref, 3, 5, 4097)
        x = torch.randn(6, 1, 4097) * 50.0
        ref.eval(); gen.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(ref(x), gen(x)), "eval outputs differ")
        ref.train(); gen.train()
        torch.manual_seed(99); out_ref = ref(x)
        torch.manual_seed(99); out_gen = gen(x)
        self.assertTrue(torch.equal(out_ref, out_gen), "train-mode outputs (dropout) differ")
        y = torch.randint(0, 5, (6,))
        F.cross_entropy(out_ref, y).backward(); F.cross_entropy(out_gen, y).backward()
        for (n1, p1), (n2, p2) in zip(ref.named_parameters(), gen.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1.grad, p2.grad), f"grad differs: {n1}")

    def test_constraint_identical(self):
        ref, gen, _, _ = make_pair(self.Ref, 5, 3, 4097)
        with torch.no_grad():
            ref.spatial.weight.mul_(4.0); gen.spatial.weight.mul_(4.0)   # push some |w| above 1
        self.assertGreater(float(ref.spatial.weight.detach().abs().max()), 1.0)
        ref.constrain_weights(); gen.constrain_weights()
        assert_state_equal(self, ref, gen)
        self.assertLessEqual(float(gen.spatial.weight.detach().abs().max()), 1.0 + 1e-6)

    def test_five_training_steps_bit_identical(self):
        ref, gen, _, _ = make_pair(self.Ref, 42, 4, 4097)
        x = torch.randn(40, 1, 4097) * 40.0
        y = torch.randint(0, 4, (40,))
        w = torch.tensor([1.0, 0.8, 1.2, 1.1])
        self.assertEqual(train_steps(ref, x, y, weight=w), train_steps(gen, x, y, weight=w))
        assert_state_equal(self, ref, gen)   # includes BN running stats
        ref.eval(); gen.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(ref(x[:8]), gen(x[:8])))

    def test_head_max_norm_none_is_reference(self):
        # Explicit None is the reference model, including its training trajectory.
        ref, gen, _, _ = make_pair(self.Ref, 11, 2, 4097, head_max_norm=None)
        x = torch.randn(40, 1, 4097) * 40.0
        y = torch.randint(0, 2, (40,))
        train_steps(ref, x, y); train_steps(gen, x, y)
        assert_state_equal(self, ref, gen)

    def test_head_max_norm_is_reference_plus_classifier_limit(self):
        # head_max_norm=0.25 == reference + the one-line classic dense max-norm in constrain_weights.
        limit = self.ns["limit_output_norm"]

        class WithHeadLimit(self.Ref):
            @torch.no_grad()
            def constrain_weights(self):
                super().constrain_weights()
                for c in self.classifiers:
                    limit(c.weight, 0.25)

        ref, gen, rr, rg = make_pair(WithHeadLimit, 42, 5, 4097, head_max_norm=0.25)
        assert_state_equal(self, ref, gen)
        self.assertTrue(torch.equal(rr, rg))
        self.assertGreater(float(row_norms(gen.classifiers[0].weight).max()), 0.25)  # limit binds
        x = torch.randn(40, 1, 4097) * 40.0
        y = torch.randint(0, 5, (40,))
        self.assertEqual(train_steps(ref, x, y), train_steps(gen, x, y))
        assert_state_equal(self, ref, gen)
        plain, _, _, _ = make_pair(self.Ref, 42, 5, 4097)
        train_steps(plain, x, y)
        self.assertFalse(torch.equal(plain.classifiers[0].weight, gen.classifiers[0].weight))

    def test_keras_atcnet_mode_differs_from_reference_at_c1(self):
        ref, _, _, _ = make_pair(self.Ref, 4, 5, 4097)
        gen = AT.ATCNet(5, **BONN, spatial_norm_mode="keras_atcnet")
        gen.load_state_dict(ref.state_dict())
        ref.constrain_weights(); gen.constrain_weights()
        self.assertFalse(torch.allclose(ref.spatial.weight, gen.spatial.weight))


# ---------------------------------------------------------------------------
# head_max_norm (opt-in classic dense max-norm)
# ---------------------------------------------------------------------------
@needs_package
class HeadMaxNorm(unittest.TestCase):
    def test_applies_limit_to_every_classifier(self):
        for opts in (BONN, CHB):
            torch.manual_seed(21)
            m = AT.ATCNet(3, **opts, head_max_norm=0.25)
            with torch.no_grad():
                m.classifiers[2].weight[1].mul_(0.01)                 # one row already inside
            before = [c.weight.detach().clone() for c in m.classifiers]
            m.constrain_weights()
            for b, c in zip(before, m.classifiers):
                n = row_norms(b)
                expected = b * torch.clamp(0.25 / n, max=1.0)[:, None]
                self.assertTrue(torch.allclose(c.weight, expected, rtol=1e-6, atol=1e-8))
                self.assertLessEqual(float(row_norms(c.weight).max()), 0.25 + 1e-6)
                self.assertTrue(torch.equal(c.bias, torch.zeros_like(c.bias)))
            self.assertTrue(torch.equal(m.classifiers[2].weight[1], before[2][1]))
            self.assertGreater(float(row_norms(before[0]).min()), 0.25)   # it did bind

    def test_none_changes_nothing(self):
        for opts in (BONN, CHB):
            torch.manual_seed(22)
            a = AT.ATCNet(2, **opts)
            rng_a = torch.get_rng_state()
            torch.manual_seed(22)
            b = AT.ATCNet(2, **opts, head_max_norm=None)
            self.assertTrue(torch.equal(rng_a, torch.get_rng_state()))
            assert_state_equal(self, a, b)
            with torch.no_grad():
                for c in b.classifiers:
                    c.weight.mul_(10.0)
            heads = [c.weight.detach().clone() for c in b.classifiers]
            b.constrain_weights()
            for h, c in zip(heads, b.classifiers):
                self.assertTrue(torch.equal(h, c.weight))

    def test_does_not_change_init_or_forward(self):
        # The limit acts only in constrain_weights (after optimizer steps), as in the other ports.
        for opts in (BONN, CHB):
            torch.manual_seed(23)
            a = AT.ATCNet(2, **opts).eval()
            torch.manual_seed(23)
            b = AT.ATCNet(2, **opts, head_max_norm=0.25).eval()
            assert_state_equal(self, a, b)
            shape = (3, opts.get("n_chans", 1), opts["n_times"])
            x = torch.randn(*shape) * 30
            with torch.no_grad():
                self.assertTrue(torch.equal(a(x), b(x)))

    def test_checkpoint_compatible(self):
        a = AT.ATCNet(2, **CHB, head_max_norm=0.25)
        b = AT.ATCNet(2, **CHB)
        b.load_state_dict(a.state_dict(), strict=True)       # adds no parameters or buffers

    def test_holds_during_training(self):
        torch.manual_seed(24)
        m = AT.ATCNet(2, **CHB, head_max_norm=0.25)
        x = torch.randn(16, 1, 8, 2560) * 30.0                # [N,1,C,T] as the training cell feeds it
        y = torch.randint(0, 2, (16,))
        losses = train_steps(m, x, y, n_steps=2)
        self.assertTrue(all(math.isfinite(v) for v in losses))
        for c in m.classifiers:
            self.assertLessEqual(float(row_norms(c.weight).max()), 0.25 + 1e-6)
        self.assertLessEqual(float(row_norms(m.spatial.weight).max()), 1.0 + 1e-6)

    def test_rejects_invalid_values(self):
        for bad in (0, 0.0, -0.25):
            with self.assertRaises(ValueError):
                AT.ATCNet(2, **CHB, head_max_norm=bad)


# ---------------------------------------------------------------------------
# CHB-MIT configuration: 8 channels x 2560 samples @ 256 Hz
# ---------------------------------------------------------------------------
@needs_package
class ChbmitGeometry(unittest.TestCase):
    def test_shapes(self):
        torch.manual_seed(31)
        m = AT.ATCNet(2, **CHB).eval()
        x = torch.randn(4, 8, 2560) * 30
        with torch.no_grad():
            out = m(x)
            self.assertEqual(tuple(out.shape), (4, 2))
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(tuple(m.conv_block(x).shape), (4, 32, 45))
            self.assertTrue(torch.equal(m(x[:, None]), out))   # [N,1,8,2560] accepted
        self.assertEqual(tuple(m.spatial.weight.shape), (32, 1, 8))

    def test_rejects_wrong_layouts(self):
        m = AT.ATCNet(2, **CHB).eval()
        for shape in ((2, 2560, 8), (2, 1, 2560), (2, 8, 4097), (2, 1, 2560, 8), (2, 20480)):
            with self.assertRaises(ValueError, msg=str(shape)):
                m(torch.zeros(shape))

    def test_geometry_depth2_and_depth3(self):
        g2 = AT.ATCNet(2, **CHB).geometry(CHB_FS)
        self.assertEqual((g2["after_pool1"], g2["seq_len"], g2["window_len"]), (320, 45, 41))
        self.assertEqual((g2["step_samples"], g2["tcn_rf_steps"]), (56, 19))
        self.assertAlmostEqual(g2["step_s"], 0.21875)
        self.assertAlmostEqual(g2["record_s"], 10.0)
        self.assertAlmostEqual(g2["kernel_s"], 0.25)        # kernel 64 = fs / 4, the paper's design
        self.assertLess(g2["tcn_rf_steps"], g2["window_len"])
        g3 = AT.ATCNet(2, **dict(CHB, tcn_depth=3)).geometry(CHB_FS)
        self.assertEqual(g3["tcn_rf_steps"], 43)
        self.assertGreaterEqual(g3["tcn_rf_steps"], g3["window_len"])
        gb = AT.ATCNet(5, **BONN).geometry(173.61)
        self.assertEqual((gb["seq_len"], gb["window_len"], gb["tcn_rf_steps"]), (73, 69, 19))

    def test_parameter_counts(self):
        self.assertEqual(AT.count_parameters(AT.ATCNet(2, **CHB)), 112954)
        self.assertEqual(AT.count_parameters(AT.ATCNet(2, **dict(CHB, tcn_depth=3))), 154874)
        self.assertEqual(AT.count_parameters(AT.ATCNet(5, **BONN)), 113225)

    def test_every_channel_reaches_output(self):
        torch.manual_seed(32)
        m = AT.ATCNet(2, **CHB).eval()
        x = (torch.randn(2, 8, 2560) * 30).requires_grad_()
        m(x).sum().backward()
        self.assertTrue((x.grad.abs().sum(dim=(0, 2)) > 0).all())

    def test_seeded_construction_is_reproducible(self):
        torch.manual_seed(33); a = AT.ATCNet(2, **CHB)
        torch.manual_seed(33); b = AT.ATCNet(2, **CHB)
        assert_state_equal(self, a, b)


# ---------------------------------------------------------------------------
# Multi-channel semantics against Keras-layout references
# ---------------------------------------------------------------------------
def keras_style_conv_block(model, x):
    """Independent channels-last reference with Conv2d on [B, 1, C, T] (H=C, W=T):
      Conv2D(F1, (K,1), 'same')        -> conv2d with TF SAME padding on W only
      BatchNormalization(axis=-1)      -> batch_norm over F1
      DepthwiseConv2D((1,C), D, valid) -> conv2d(F1 -> F1*D, (C, 1), groups=F1)
    """
    f1, d, c = model.f1, model.depth_multiplier, model.n_chans
    k = model.temporal.kernel_size[0]
    h = x.unsqueeze(1)
    h = F.pad(h, ((k - 1) // 2, (k - 1) - (k - 1) // 2, 0, 0))
    h = F.conv2d(h, model.temporal.weight.unsqueeze(2))              # (F1,1,1,K)
    bn = model.bn1
    h = F.batch_norm(h, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)
    h = F.conv2d(h, model.spatial.weight.reshape(f1 * d, 1, c, 1), groups=f1)
    h = h[:, :, 0, :]
    h = model.pool1(F.elu(model.bn2(h)))
    h = model.pool2(F.elu(model.bn3(model.conv2(h))))
    return h


def keras_maxnorm_reference(w_torch, f1, d, c, max_value, axis):
    """Map torch (F1*D,1,C) -> Keras depthwise kernel (1, C, F1, D), apply the Keras formula."""
    k = w_torch.detach().clone().reshape(f1, d, c).permute(2, 0, 1).unsqueeze(0)  # (1,C,F1,D)
    norms = k.square().sum(dim=axis, keepdim=True).sqrt()
    k = k * (norms.clamp(0, max_value) / (1e-7 + norms))
    return k[0].permute(1, 2, 0).reshape(f1 * d, 1, c)


@needs_package
class MultiChannel(unittest.TestCase):
    def test_shapes_and_reference(self):
        for c, t, ncls in ((8, 2560, 2), (22, 1125, 4), (3, 1000, 3)):
            torch.manual_seed(c)
            m = AT.ATCNet(ncls, n_times=t, n_chans=c).eval()
            with torch.no_grad():
                for bn in (m.bn1, m.bn2, m.bn3):   # non-trivial running stats
                    bn.running_mean.uniform_(-1, 1); bn.running_var.uniform_(0.5, 2)
                x = torch.randn(4, c, t) * 30
                out = m(x)
                self.assertEqual(tuple(out.shape), (4, ncls))
                self.assertTrue(torch.isfinite(out).all())
                ref = keras_style_conv_block(m, x)
                got = m.conv_block(x)
                self.assertEqual(ref.shape, got.shape)
                self.assertTrue(torch.allclose(ref, got, rtol=1e-4, atol=1e-5),
                                float((ref - got).abs().max()))
                self.assertTrue(torch.equal(m(x.unsqueeze(1)), out))

    def test_train_mode_bn1_stats_over_channels(self):
        torch.manual_seed(1)
        m = AT.ATCNet(2, n_times=2560, n_chans=8).train()
        x = torch.randn(4, 8, 2560)
        m(x)
        with torch.no_grad():
            h = m.temporal(x.reshape(32, 1, 2560)).reshape(4, 8, 16, 2560)
            mean = h.mean(dim=(0, 1, 3))
            var = h.var(dim=(0, 1, 3), unbiased=True)
        self.assertTrue(torch.allclose(m.bn1.running_mean, 0.01 * mean, atol=1e-6))
        self.assertTrue(torch.allclose(m.bn1.running_var, 0.99 + 0.01 * var, rtol=1e-5, atol=1e-6))

    def test_depthwise_semantics(self):
        torch.manual_seed(2)
        m = AT.ATCNet(2, n_times=2560, n_chans=8).eval()
        x = torch.randn(2, 16, 8, 50)
        out = F.conv2d(x, m.spatial.weight.unsqueeze(-1), groups=16)[:, :, 0]
        w = m.spatial.weight[:, 0, :]                                  # (32, 8)
        ref = torch.einsum("oc,bfct->boft", w, x)                      # all pairs
        idx = torch.arange(32)
        ref = ref[:, idx, idx // 2]                                    # keep f = o // D
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_maxnorm_modes_vs_keras_formula(self):
        f1, d, c = 16, 2, 8
        for mode, maxv, axis in (("per_filter", 1.0, None), ("keras_classic", 1.0, (0,)),
                                 ("keras_atcnet", 0.6, (0, 1, 2))):
            torch.manual_seed(3)
            m = AT.ATCNet(2, n_times=2560, n_chans=c, spatial_norm_mode=mode)
            with torch.no_grad():
                m.spatial.weight.copy_(torch.randn_like(m.spatial.weight))
            before = m.spatial.weight.detach().clone()
            m.constrain_weights()
            after = m.spatial.weight.detach()
            if mode == "per_filter":
                # EEGNet (arl-eegmodels) semantics: kernel (C,1,F1,D), max_norm axis=0 (over C).
                k = before.reshape(f1, d, c).permute(2, 0, 1).unsqueeze(1)          # (C,1,F1,D)
                n = k.square().sum(0, keepdim=True).sqrt()
                ref = (k * n.clamp(0, maxv) / (1e-7 + n))[:, 0].permute(1, 2, 0).reshape(f1 * d, 1, c)
                self.assertLessEqual(float(after.square().sum((1, 2)).sqrt().max()), maxv + 1e-5)
            else:
                ref = keras_maxnorm_reference(before, f1, d, c, maxv, axis)
            self.assertTrue(torch.allclose(after, ref, rtol=1e-5, atol=1e-6), mode)
        v = after.reshape(f1, d, c)                                    # keras_atcnet: D joint norms
        self.assertEqual(v.square().sum((0, 2)).numel(), d)
        self.assertTrue((v.square().sum((0, 2)).sqrt() <= 0.6 + 1e-5).all())

    def test_keras_classic_equals_per_filter_at_c1(self):
        torch.manual_seed(0)
        a = AT.ATCNet(3, **BONN, spatial_norm_mode="per_filter")
        b = AT.ATCNet(3, **BONN, spatial_norm_mode="keras_classic")
        b.load_state_dict(a.state_dict())
        with torch.no_grad():
            a.spatial.weight.mul_(5); b.spatial.weight.mul_(5)
        a.constrain_weights(); b.constrain_weights()
        self.assertTrue(torch.allclose(a.spatial.weight, b.spatial.weight, rtol=1e-5, atol=1e-7))

    def test_bci2a_param_count_matches_readme(self):
        # Upstream README: ATCNet 113,732 trainable parameters (BCI-IV-2a, 22 channels, 4 classes).
        self.assertEqual(AT.count_parameters(AT.ATCNet(4, n_times=1125, n_chans=22)), 113732)

    def test_keras_extras(self):
        torch.manual_seed(5)
        m = AT.ATCNet(2, n_times=2560, n_chans=8, keras_constraints=True, spatial_norm_mode="keras_atcnet")
        m.constrain_weights()
        convs = [m.temporal.weight, m.conv2.weight]
        convs += [x.weight for x in m.tcns.modules() if isinstance(x, nn.Conv1d)]
        for w in convs:
            self.assertLessEqual(float(w.detach().square().sum((1, 2)).sqrt().max()), 0.6 + 1e-5)
        pen = m.keras_l2_penalty()
        pen.backward()
        self.assertTrue(torch.isfinite(pen) and m.classifiers[0].weight.grad is not None)

    def test_tcn_filters_downsample_path(self):
        torch.manual_seed(6)
        m = AT.ATCNet(2, n_times=2560, n_chans=8, f1=8, tcn_filters=32)   # F2 = 16 != 32
        self.assertTrue(hasattr(m.tcns[0][0], "downsample") and not hasattr(m.tcns[0][1], "downsample"))
        self.assertEqual(tuple(m(torch.randn(2, 8, 2560)).shape), (2, 2))

    def test_spatial_init_keras_bound(self):
        torch.manual_seed(7)
        m = AT.ATCNet(2, n_times=2560, n_chans=8, spatial_init="keras")
        bound = math.sqrt(6 / (16 * 8 + 2 * 8))
        self.assertLessEqual(float(m.spatial.weight.detach().abs().max()), bound + 1e-6)
        self.assertGreater(float(m.spatial.weight.detach().abs().max()), 0.8 * bound)


@needs_package
class ReceptiveField(unittest.TestCase):
    def test_tcn_rf_by_gradient(self):
        for depth, kernel in ((2, 4), (3, 4), (4, 4), (3, 6)):
            torch.manual_seed(8)
            m = AT.ATCNet(2, n_times=2560, n_chans=8, tcn_depth=depth, tcn_kernel=kernel).eval()
            seq = torch.randn(1, 32, 120, requires_grad=True)
            m.tcns[0](seq)[:, :, -1].sum().backward()
            nz = (seq.grad.abs().sum(1)[0] > 0).nonzero().flatten()
            self.assertEqual(int(120 - nz.min()), AT.tcn_receptive_field(kernel, depth), (depth, kernel))


# ---------------------------------------------------------------------------
# model_source(): the text embedded in notebooks and inference.py
# ---------------------------------------------------------------------------
@needs_package
class ModelSource(unittest.TestCase):
    def test_embedded_source_builds_identical_models(self):
        for docstring in (True, False):
            src = AT.model_source(docstring=docstring)
            self.assertIsNone(re.search(r"^\s*from __future__", src, flags=re.M))
            self.assertIn("SPDX-License-Identifier: Apache-2.0", src)
            ns = {}
            exec(compile("PRECEDING_CELL_CODE = 1\n" + src, "embedded_model", "exec"), ns)
            for n_classes, opts, extra in ((5, BONN, {}), (2, CHB, {}), (2, CHB, {"head_max_norm": 0.25})):
                torch.manual_seed(41)
                a = AT.ATCNet(n_classes, **opts, **extra).eval()
                torch.manual_seed(41)
                b = ns["ATCNet"](n_classes, **opts, **extra).eval()
                assert_state_equal(self, a, b)
                shape = (2, opts.get("n_chans", 1), opts["n_times"])
                x = torch.randn(*shape) * 30
                with torch.no_grad():
                    self.assertTrue(torch.equal(a(x), b(x)))

    def test_source_matches_module_file(self):
        with open(AT.__file__, encoding="utf-8") as handle:
            self.assertEqual(AT.model_source(), handle.read())
        short = AT.model_source(docstring=False)
        self.assertTrue(short.startswith("# Generalized PyTorch ATCNet"))
        self.assertLess(len(short), len(AT.model_source()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
