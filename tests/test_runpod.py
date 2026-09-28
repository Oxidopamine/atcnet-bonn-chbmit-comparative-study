"""Offline tests for the RunPod tooling in runpod/ (no network access, no RunPod state changes).

    python -m unittest tests.test_runpod -v        (from the repository root)

Covers: CLI parsing, request bodies against the saved OpenAPI schemas, key redaction, GPU
filtering, duplicate-pod protection, data/result bundles with sha256 manifests, job generation
for Bonn and CHB-MIT, queue claim/requeue/kill safety with fake jobs and processes, merge,
finalize guard, pack, the autostop watchdog, and static checks of remote_setup.sh.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUNPOD = ROOT / "runpod"
sys.path.insert(0, str(RUNPOD))

import rp  # noqa: E402
import rp_queue as rq  # noqa: E402

SNAP = json.loads((RUNPOD / "openapi_snapshot.json").read_text(encoding="utf-8"))
FAKE_KEY = "rpa_FAKEKEY_must_never_be_printed_0123456789"
PREVIOUS_32 = (
    "PREVIOUS_32 = [(('Z',), ('O',)), (('Z',), ('N',)), (('Z',), ('F',)), (('Z',), ('S',)), (('O',), ('N',)), "
    "(('O',), ('F',)), (('O',), ('S',)), (('N',), ('F',)), (('N',), ('S',)), (('F',), ('S',)), (('Z', 'O'), ('S',)), "
    "(('N', 'F'), ('S',)), (('Z', 'O', 'N', 'F'), ('S',)), (('Z', 'O'), ('N', 'F')), (('Z', 'O'), ('N', 'F', 'S')), "
    "(('Z',), ('O',), ('N',)), (('Z',), ('O',), ('F',)), (('Z',), ('O',), ('S',)), (('Z',), ('N',), ('F',)), "
    "(('Z',), ('N',), ('S',)), (('Z',), ('F',), ('S',)), (('O',), ('F',), ('N',)), (('O',), ('N',), ('S',)), "
    "(('O',), ('F',), ('S',)), (('N',), ('F',), ('S',)), (('Z', 'O'), ('N', 'F'), ('S',)), "
    "(('Z',), ('O',), ('N',), ('F',)), (('Z',), ('O',), ('N',), ('S',)), (('Z',), ('O',), ('F',), ('S',)), "
    "(('Z',), ('N',), ('F',), ('S',)), (('O',), ('N',), ('F',), ('S',)), (('Z',), ('O',), ('N',), ('F',), ('S',))]")
BONN_PARAMS = """BONN_DATA_DIR = "/content/drive/MyDrive/Dataset/Bonn"  # @param {type:"string"}
RESULTS_ROOT = "/content/drive/MyDrive/BONN"  # @param {type:"string"}
PRESET = "previous_32"
SELECTED_IDS = []  # [] = all
MAX_CLASSES_TO_RUN = None  # @param {type:"raw"}
STOP_ON_STAGE_FAILURE = True
SEED = 42
EPOCHS = 100
BATCH_SIZE = 8
PLOT_EVERY = 10
REUSE_COMPLETED = True
MODEL_OPTIONS = {'f1': 16, 'depth_multiplier': 2, 'kernel_length': 64, 'pool_size': 7, 'n_windows': 5, 'tcn_depth': 2, 'tcn_kernel': 4, 'dropout': 0.3}
assert EPOCHS >= 1 and PLOT_EVERY >= 1

def seed_everything(seed):
    return seed

seed_everything(SEED)
"""
CHB_PARAMS = """CHBMIT_NPZ = "/content/chbmit_8ch.npz"
CHBMIT_META = "/content/chbmit_8ch_metadata.csv"
RESULTS_ROOT = "/content/results"
LOPO_UNIT = "subject"
SELECTED_TEST_UNITS = []
LOPO_PLAN_PATH = ""
MIN_VAL_PER_CLASS = 5
UNIT_SCALE = 1e6
SEED = 42
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
PLOT_EVERY = 10
REUSE_COMPLETED = True
AGGREGATE_ONLY = False
CREATE_WORD_SUMMARY_REPORT = True
MODEL_OPTIONS = {'n_chans': 8, 'tcn_depth': 2}
"""


def make_notebook(path: Path, params_src: str, extra=(), tagged=True) -> Path:
    def cell(src, tags=None):
        return {"cell_type": "code", "execution_count": None, "outputs": [],
                "metadata": {"tags": tags} if tags else {}, "source": src.splitlines(keepends=True)}
    cells = [{"cell_type": "markdown", "metadata": {}, "source": ["# synthetic notebook\n"]},
             cell(params_src, ["parameters"] if tagged else None)] + [cell(s) for s in extra]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return path


def write_meta(path: Path, units_by_subject: dict):
    """units_by_subject: subject -> list of (patient, n_nonseizure, n_seizure)."""
    rows = ["segment_id,patient,subject,label"]
    for subj, parts in units_by_subject.items():
        for patient, n0, n1 in parts:
            rows += [f"{patient}/n{i},{patient},{subj},0" for i in range(n0)]
            rows += [f"{patient}/s{i},{patient},{subj},1" for i in range(n1)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------ schema validator

def _props(schema, comps, seen=()):
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return _props(comps[name], comps, seen + (name,)) if name not in seen else ({}, set(), None)
    props, required, addl = {}, set(schema.get("required", [])), schema.get("additionalProperties")
    for k in ("allOf", "anyOf", "oneOf"):
        for sub in schema.get(k, []):
            p, r, _ = _props(sub, comps, seen)
            props.update(p)
            if k == "allOf":
                required |= r
    props.update(schema.get("properties") or {})
    return props, required, addl


def _resolve(schema, comps):
    while "$ref" in schema:
        schema = comps[schema["$ref"].split("/")[-1]]
    if "allOf" in schema and len(schema["allOf"]) == 1 and not schema.get("properties"):
        return _resolve(schema["allOf"][0], comps)
    return schema


def validate(body, schema, comps, path="") -> list:
    """Unknown fields, missing required fields, enum and basic type violations."""
    errors = []
    props, required, _ = _props(schema, comps)
    for k in sorted(required - set(body)):
        errors.append(f"{path}{k}: required")
    for k, v in body.items():
        if k not in props:
            errors.append(f"{path}{k}: unknown field")
            continue
        sub = _resolve(props[k], comps)
        t = sub.get("type")
        types = {"string": str, "integer": int, "boolean": bool, "array": list, "object": dict, "number": (int, float)}
        if t in types and not isinstance(v, types[t]) or (t == "integer" and isinstance(v, bool)):
            errors.append(f"{path}{k}: expected {t}, got {type(v).__name__}")
        if "enum" in sub and v not in sub["enum"]:
            errors.append(f"{path}{k}: {v!r} not in enum")
        if isinstance(v, dict) and (sub.get("properties") or "$ref" in props[k] or "allOf" in sub) and k != "env":
            errors += validate(v, props[k], comps, path + k + ".")
        if isinstance(v, list) and "items" in sub:
            item = _resolve(sub["items"], comps)
            for x in v:
                if isinstance(x, dict):
                    errors += validate(x, sub["items"], comps, path + k + "[].")
                elif "enum" in item and x not in item["enum"]:
                    errors.append(f"{path}{k}[]: {x!r} not in enum")
    return errors


class _Env(unittest.TestCase):
    """Temporary workspace ($WS), fake key, fast heartbeat; restores the environment afterwards."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rp_test_"))
        self.env = mock.patch.dict(os.environ, {"WS": str(self.tmp / "ws"), "RUNPOD_API_KEY": FAKE_KEY,
                                                "RP_HEARTBEAT": "0.2", "FOLDS_CSV": str(self.tmp / "no_folds.csv"),
                                                "RP_FAKE_SLEEP": "0"})
        self.env.start()
        rp._KEY_CACHE.clear()
        self.cwd = os.getcwd()
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            with contextlib.suppress(Exception):
                rq.kill_tree(p.pid)
                p.wait(timeout=10)
        os.chdir(self.cwd)
        self.env.stop()
        rp._KEY_CACHE.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def sleeper(self, cwd=None, marker="ipykernel_fake", seconds=120):
        p = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})", marker],
                             cwd=str(cwd) if cwd else None, start_new_session=(os.name != "nt"))
        self.procs.append(p)
        time.sleep(0.3)
        return p

    def dead_pid(self):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        ident = rq.proc_ident(p.pid)
        p.wait()
        return p.pid, ident

    def bonn_nb(self):
        return make_notebook(self.tmp / "bonn.ipynb", BONN_PARAMS, [PREVIOUS_32 + "\nSETS = ['Z']\n"])

    def chb_nb(self):
        return make_notebook(self.tmp / "chb.ipynb", CHB_PARAMS)


# ============================================================================== rp.py: CLI + API

class RecordingHttp:
    """Stands in for rp.http: canned read-only answers; records every call; refuses mutations."""

    def __init__(self, pods=(), fail_post=False):
        self.calls, self.pods, self.fail_post = [], list(pods), fail_post

    def __call__(self, method, url, *, params=None, body=None, ok=(200, 201, 204), retries=2):
        self.calls.append((method, url, body))
        if url == rp.GRAPHQL_URL:
            q = body["query"]
            if "gpuTypes" in q:
                return {"data": {"gpuTypes": [{"id": "NVIDIA RTX A4000", "displayName": "RTX A4000",
                                               "memoryInGb": 16, "sec": {"uninterruptablePrice": 0.25,
                                                                         "stockStatus": "Low", "minVcpu": 16,
                                                                         "minMemory": 62}, "com": {}}]}}
            return {"data": {"myself": {"clientBalance": 10.0, "currentSpendPerHr": 0.0, "spendLimit": 80,
                                        "underBalance": False, "minBalance": 0}}}
        if method == "GET" and url.endswith("/pods"):
            return {"pods": self.pods}
        if method == "GET" and "/catalog/gpus" in url:
            return {"gpus": [{"id": "NVIDIA RTX A4000", "name": "RTX A4000", "memory": 16, "secure": True,
                              "community": True, "price": {"secure": 0.25, "community": 0.17},
                              "availability": "LOW", "cudaVersions": [{"version": "12.8", "available": True}],
                              "dataCenters": [{"id": "EU-X-1", "availability": "LOW"}]}]}
        if method == "GET" and "/pods/" in url:
            raise rp.ApiError(f"GET {url} -> HTTP 404")
        if method == "GET":
            return {}
        if self.fail_post:
            raise rp.ApiError(f"{method} {url} failed: Read timed out")
        raise AssertionError(f"mutating request attempted: {method} {url}")

    @property
    def mutations(self):
        return [(m, u) for m, u, _ in self.calls if u != rp.GRAPHQL_URL and m != "GET"]


def run_rp(argv, http=None):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with mock.patch.object(rp, "http", http or RecordingHttp()), contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(err):
        try:
            rp.main(argv)
        except SystemExit as e:
            code = e.code or 0
    return code, out.getvalue(), err.getvalue()


class TestRpCli(_Env):
    def setUp(self):
        super().setUp()
        self.pub = self.tmp / "id_test.pub"
        self.pub.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAItestkey test\n", encoding="utf-8")

    def test_parser_commands_and_defaults(self):
        ap = rp.build_parser()
        a = ap.parse_args(["create", "--gpu", "NVIDIA RTX A4000"])
        self.assertFalse(a.yes)
        self.assertEqual((a.cloud, a.api, a.min_ram), ("SECURE", "auto", 32))
        for name in ("start", "stop", "terminate"):
            self.assertFalse(ap.parse_args([name, "abc"]).yes)
        self.assertTrue(ap.parse_args(["gpus", "--available", "--min-vcpu", "12"]).available)
        self.assertEqual(ap.parse_args(["verify-results", "a.tgz", "b.tgz"]).tgz, ["a.tgz", "b.tgz"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            ap.parse_args(["create"])                           # --gpu is required

    def test_dry_runs_send_nothing_and_never_print_the_key(self):
        cmds = [["create", "--gpu", "NVIDIA RTX A4000", "--pubkey", str(self.pub)],
                ["create", "--gpu", "NVIDIA RTX A4000", "--cloud", "COMMUNITY", "--pubkey", str(self.pub)],
                ["stop", "podx"], ["terminate", "podx"], ["start", "podx"], ["balance"], ["status"],
                ["gpus", "--available", "--min-vcpu", "12"]]
        for argv in cmds:
            h = RecordingHttp()
            code, out, err = run_rp(argv, h)
            self.assertEqual(code, 0, (argv, err))
            self.assertEqual(h.mutations, [], argv)
            self.assertNotIn(FAKE_KEY, out + err)
            if argv[0] in ("create", "stop", "terminate", "start"):
                self.assertIn("DRY RUN - nothing was sent", out)
                self.assertIn(rp.REDACTED, out)

    def test_scrub_and_headers(self):
        self.assertEqual(rp._key(), FAKE_KEY)
        self.assertNotIn(FAKE_KEY, rp._scrub(f"error body echoing {FAKE_KEY}"))
        self.assertNotIn(FAKE_KEY, json.dumps(rp._redacted_headers()))

    def test_create_refuses_existing_name(self):
        h = RecordingHttp(pods=[{"id": "p1", "name": "atcnet", "status": "RUNNING", "cost": 0.3}])
        code, out, err = run_rp(["create", "--gpu", "NVIDIA RTX A4000", "--name", "atcnet", "--pubkey",
                                 str(self.pub), "--yes"], h)
        self.assertEqual(code, 1)
        self.assertIn("Refusing", err)
        self.assertEqual(h.mutations, [])

    def test_create_error_warns_about_duplicate_pod(self):
        class Flaky(RecordingHttp):             # the pod appears although the POST "timed out"
            def __call__(self, method, url, **kw):
                if method == "POST" and url.endswith("/pods"):
                    self.pods.append({"id": "p9", "name": "fresh", "status": "RUNNING", "cost": 0.25})
                return super().__call__(method, url, **kw)
        h = Flaky(fail_post=True)
        code, out, err = run_rp(["create", "--gpu", "NVIDIA RTX A4000", "--name", "fresh", "--pubkey",
                                 str(self.pub), "--yes"], h)
        self.assertEqual(code, 1)
        self.assertIn("FOUND pod 'fresh'", err)
        self.assertIn("do NOT re-run create", err)
        self.assertNotIn(FAKE_KEY, out + err)

    def test_gpu_rows_honour_vcpu_ram_filter(self):
        cat = {
            "NVIDIA RTX A4000": dict(id="NVIDIA RTX A4000", vram=16, price={"secure": 0.25}, offered={"SECURE": True},
                                     SECURE=dict(availability="LOW", dcs=["X"], cuda=["12.8"]),
                                     gql=dict(SECURE={"stockStatus": None, "uninterruptablePrice": 0.25})),
            "NVIDIA GeForce RTX 4090": dict(id="NVIDIA GeForce RTX 4090", vram=24, price={"secure": 0.74},
                                            offered={"SECURE": True}, SECURE=dict(availability="NONE", dcs=[], cuda=[]),
                                            gql=dict(SECURE={"stockStatus": "Low", "uninterruptablePrice": 0.74,
                                                             "minVcpu": 16, "minMemory": 62})),
            "NVIDIA L4": dict(id="NVIDIA L4", vram=24, price={"secure": 0.49}, offered={"SECURE": True},
                              SECURE=dict(availability="HIGH", dcs=["Y"], cuda=["13.1"]),
                              gql=dict(SECURE={"stockStatus": "High", "uninterruptablePrice": 0.44,
                                               "minVcpu": 12, "minMemory": 50})),
        }
        rows = rp.gpu_rows(cat, clouds=["SECURE"], available=True, min_vcpu=12, min_ram=32)
        self.assertEqual([r["gpu_id"] for r in rows], ["NVIDIA L4", "NVIDIA GeForce RTX 4090"])   # cheapest first
        rows = {r["gpu_id"]: r for r in rp.gpu_rows(cat, clouds=["SECURE"], min_vcpu=12)}
        self.assertEqual(rows["NVIDIA RTX A4000"]["match"], "filter unmet")
        self.assertIsNone(rows["NVIDIA RTX A4000"]["live_usd_h"])
        rows = {r["gpu_id"]: r for r in rp.gpu_rows(cat, clouds=["SECURE"], available=True)}
        self.assertIn("NVIDIA RTX A4000", rows)                                              # no filter

    def test_ssh_cmd_offline(self):
        pod = {"id": "p1", "name": "n", "status": "RUNNING", "cloud": "SECURE",
               "ssh": {"direct": {"host": "1.2.3.4", "port": 2222}}}
        f = self.tmp / "pod.json"
        f.write_text(json.dumps(pod), encoding="utf-8")
        code, out, _ = run_rp(["ssh-cmd", "p1", "--from-json", str(f), "--queue", "q1"])
        self.assertEqual(code, 0)
        self.assertIn('mkdir -p /workspace/code /workspace/data', out)
        self.assertIn("-p 2222 root@1.2.3.4", out)
        self.assertIn("bundle-data", out)
        self.assertIn("verify-results", out)
        self.assertIn("/workspace/packs/results_q1_*", out)
        code, out, _ = run_rp(["pull", "p1", "--from-json", str(f), "--queue", "q1"])
        self.assertIn("remote_setup.sh pack q1", out)


class TestRequestBodies(_Env):
    def setUp(self):
        super().setUp()
        pub = self.tmp / "k.pub"
        pub.write_text("ssh-ed25519 AAAAC3Nza test\n", encoding="utf-8")
        self.ns = rp.build_parser().parse_args(["create", "--gpu", "NVIDIA GeForce RTX 4090", "--pubkey", str(pub),
                                                "--min-vcpu", "12", "--jupyter", "--datacenter", "EU-RO-1"])

    def check(self, api, body_url):
        method, url, body, _ = body_url
        server = SNAP[api]["server"]
        self.assertTrue(url.startswith(server), url)
        return body

    def test_v2_secure(self):
        m, u, b, _ = rp.build_create_request(self.ns)
        self.assertEqual((m, u), ("POST", "https://api.runpod.io/v2/pods"))
        self.assertIn("/v2/pods", SNAP["v2"]["paths"])
        self.assertIn("post", SNAP["v2"]["paths"]["/v2/pods"])
        s = SNAP["v2"]["schemas"]
        self.assertEqual(validate(b, s["CreatePodRequest"], s), [])
        self.assertEqual(b["gpu"]["minCudaVersion"], "12.8")
        self.assertEqual(b["mounts"], {"persistent": {"size": 20, "path": "/workspace"}})

    def test_v2_network_volume(self):
        self.ns.network_volume = "vol123"
        _, _, b, _ = rp.build_create_request(self.ns)
        s = SNAP["v2"]["schemas"]
        self.assertEqual(validate(b, s["CreatePodRequest"], s), [])
        self.assertEqual(b["mounts"], {"network": [{"volumeId": "vol123", "path": "/workspace"}]})

    def test_v1_community_relaxed_cuda(self):
        self.ns.cloud = "COMMUNITY"
        m, u, b, _ = rp.build_create_request(self.ns)
        self.assertEqual(u, "https://rest.runpod.io/v1/pods")
        s = SNAP["v1"]["schemas"]
        self.assertEqual(validate(b, s["PodCreateInput"], s), [])
        self.assertNotIn("allowedCudaVersions", b)
        self.assertTrue(b["supportPublicIp"])

    def test_validator_catches_errors(self):
        s = SNAP["v2"]["schemas"]
        bad = {"name": "x", "image": "i", "cloud": "PUBLIC", "gpu": {"id": "g", "count": "1"}, "bogus": 1}
        errs = validate(bad, s["CreatePodRequest"], s)
        self.assertTrue(any("bogus: unknown field" in e for e in errs), errs)
        self.assertTrue(any("cloud" in e for e in errs), errs)
        self.assertTrue(any("gpu.count" in e for e in errs), errs)

    def test_interruptible_needs_v1(self):
        self.ns.interruptible, self.ns.api = True, "v2"
        with self.assertRaises(rp.ApiError):
            rp.build_create_request(self.ns)

    def test_lifecycle_and_autostop_requests(self):
        s = SNAP["v2"]["schemas"]
        for action in ("start", "stop", "restart", "terminate"):
            self.assertEqual(validate({"action": action}, s["PodActionRequest"], s), [])
        self.assertIn("post", SNAP["v2"]["paths"]["/v2/pods/{id}/action"])
        self.assertIn("delete", SNAP["v2"]["paths"]["/v2/pods/{id}"])
        self.assertIn("post", SNAP["v1"]["paths"]["/pods/{podId}/stop"])


# ============================================================================== bundles + integrity

class TestBundles(_Env):
    def make_data(self):
        root = self.tmp / "eegdata"
        for s in "ZONFS":
            d = root / "bonn" / s
            d.mkdir(parents=True)
            for i in range(3):
                (d / f"{s}{i:03d}.txt").write_text(f"{i}\n{i + 1}\n", encoding="utf-8")
        (root / "bonn" / "Z" / ".DS_Store").write_text("junk")
        (root / "chbmit_8ch.npz").write_bytes(os.urandom(1000))
        (root / "chbmit_8ch_metadata.csv").write_text("segment_id,patient,subject,label\n", encoding="utf-8")
        return root

    def test_bundle_data_stream_and_manifest(self):
        root = self.make_data()
        buf = io.BytesIO()
        rp.write_bundle(rp.data_entries(root / "bonn", root / "chbmit_8ch.npz", root / "chbmit_8ch_metadata.csv"),
                        buf, "DATA_MANIFEST.sha256", log=io.StringIO())
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
            self.assertEqual(names[-1], "DATA_MANIFEST.sha256")
            self.assertIn("bonn/Z/Z000.txt", names)
            self.assertIn("chbmit/chbmit_8ch.npz", names)
            self.assertNotIn("bonn/Z/.DS_Store", names)
            man = dict(line.split("  ")[::-1] for line in
                       tar.extractfile("DATA_MANIFEST.sha256").read().decode().splitlines())
            for n in names[:-1]:
                self.assertEqual(man[n], hashlib.sha256(tar.extractfile(n).read()).hexdigest())
        self.assertEqual(len(man), 5 * 3 + 2)

    def test_bundle_data_refuses_bad_layout(self):
        root = self.make_data()
        with self.assertRaises(rp.ApiError):
            list(rp.data_entries(root / "missing", root / "chbmit_8ch.npz", root / "chbmit_8ch_metadata.csv"))

    def test_bundle_code_excludes_heavy_and_private_files(self):
        buf = io.BytesIO()
        rp.write_bundle(rp.code_entries(ROOT), buf, "seizure_study/CODE_MANIFEST.sha256", log=io.StringIO())
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
        top = ROOT.name
        self.assertIn(f"{top}/runpod/remote_setup.sh", names)
        self.assertFalse(any("/.git/" in n or n.endswith((".pt", ".pyc")) or "__pycache__" in n for n in names))

    def test_pack_and_verify_results(self):
        q = "t"
        ws = rq.ws()
        (ws / "runs" / q / "job1" / "study_x" / "a" / "fit").mkdir(parents=True)
        f = ws / "runs" / q / "job1" / "study_x" / "a" / "fit" / "complete.json"
        f.write_text("{}")
        (ws / "results" / q / "study_x" / "a" / "fit").mkdir(parents=True)
        os.link(f, ws / "results" / q / "study_x" / "a" / "fit" / "complete.json")
        (ws / "jobs" / q / "done").mkdir(parents=True)
        (ws / "jobs" / q / "jobs.jsonl").write_text(json.dumps({"id": "job1", "expect_complete": 1}) + "\n")
        (ws / "jobs" / q / "done" / "job1.json").write_text("{}")
        (ws / "runs" / q / "job1" / "partial.tmp").write_text("in progress")
        with contextlib.redirect_stdout(io.StringIO()):
            tgz = rq.cmd_pack(q)
        with tarfile.open(tgz) as tar:
            members = {m.name: m for m in tar.getmembers()}
        self.assertTrue(members[f"runs/{q}/job1/study_x/a/fit/complete.json"].islnk())
        self.assertNotIn(f"runs/{q}/job1/partial.tmp", members)
        out = io.StringIO()
        self.assertEqual(rp.verify_results(tgz, out=out), 0, out.getvalue())
        self.assertIn("jobs 1/1 done", out.getvalue())
        blob = bytearray(tgz.read_bytes())            # corrupt the transfer
        blob[len(blob) // 2] ^= 0xFF
        bad = tgz.with_name("bad.tgz")
        bad.write_bytes(bytes(blob))
        shutil.copy(tgz.with_name(tgz.name + ".sha256"), bad.with_name("bad.tgz.sha256"))
        self.assertEqual(rp.verify_results(bad, out=io.StringIO()), 1)

    def test_pack_rotation(self):
        (rq.ws() / "results" / "r").mkdir(parents=True)
        (rq.ws() / "results" / "r" / "x.txt").write_text("x")
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3):
                rq.cmd_pack("r", keep=2)
                time.sleep(1.1)                       # distinct timestamps
        self.assertEqual(len(list((rq.ws() / "packs").glob("results_r_*.tgz"))), 2)


# ============================================================================== job generation

class TestJobGeneration(_Env):
    def test_bonn_32_jobs_in_class_order(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_make_bonn("b", str(self.bonn_nb()), str(self.tmp))
        jobs = rq.load_jobs("b")
        self.assertEqual(len(jobs), 32)
        self.assertEqual([j["classes"] for j in jobs], [5] + [4] * 5 + [3] * 11 + [2] * 15)
        self.assertEqual(jobs[0]["id"], "Z_vs_O_vs_N_vs_F_vs_S")
        self.assertEqual(jobs[-1]["id"], "Z+O_vs_N+F+S")
        self.assertTrue(all(j["expect_complete"] == 11 for j in jobs))
        self.assertTrue(all(j["params"]["SELECTED_IDS"] == [j["id"]] for j in jobs))
        self.assertTrue(all(j["params"]["RESULTS_ROOT"] == "{JOB_ROOT}" for j in jobs))
        self.assertEqual(sum(j["expect_complete"] for j in jobs), 352)
        meta = rq.load_meta("b")
        self.assertEqual(Path(meta["notebook"]).read_bytes(), (self.tmp / "bonn.ipynb").read_bytes())

    def test_bonn_subset_idempotent_and_guards(self):
        nb = str(self.bonn_nb())
        with contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_make_bonn("s", nb, str(self.tmp), '{"SELECTED_IDS": ["Z_vs_S", "Z_vs_O_vs_N_vs_F_vs_S"], "EPOCHS": 1}')
            rq.cmd_make_bonn("s", nb, str(self.tmp), '{"SELECTED_IDS": ["Z_vs_S", "Z_vs_O_vs_N_vs_F_vs_S"], "EPOCHS": 1}')
        self.assertEqual([j["id"] for j in rq.load_jobs("s")], ["Z_vs_O_vs_N_vs_F_vs_S", "Z_vs_S"])
        self.assertEqual(rq.load_jobs("s")[0]["params"]["EPOCHS"], 1)
        with self.assertRaisesRegex(rq.QueueError, "different"):
            rq.cmd_make_bonn("s", nb, str(self.tmp), '{"EPOCHS": 2}')
        with self.assertRaisesRegex(rq.QueueError, "not assigned"):
            rq.cmd_make_bonn("u", nb, str(self.tmp), '{"PLOT_EVRY": 100}')
        with self.assertRaisesRegex(rq.QueueError, "already used"):
            rq.cmd_make_bonn("u", nb, str(self.tmp), '{"SEED": 1}')
        with self.assertRaisesRegex(rq.QueueError, "unknown combination"):
            rq.cmd_make_bonn("u", nb, str(self.tmp), '{"SELECTED_IDS": ["Z_vs_Q"]}')
        untagged = make_notebook(self.tmp / "untagged.ipynb", BONN_PARAMS, [PREVIOUS_32], tagged=False)
        with self.assertRaisesRegex(rq.QueueError, "parameters"):
            rq.cmd_make_bonn("u", str(untagged), str(self.tmp))
        self.assertFalse((rq.qdir("u")).exists())

    def test_real_bonn_notebook_if_present(self):
        nb = ROOT / "notebooks" / "Bonn_ATCNet_5_to_2_Class_Validation.ipynb"
        if not nb.is_file():
            self.skipTest("notebooks/ not present (kept local)")
        combos = rq.bonn_combinations(nb)
        self.assertEqual(len(combos), 32)
        self.assertEqual([c[1] for c in combos], [5] + [4] * 5 + [3] * 11 + [2] * 15)
        info = rq.notebook_params(nb)
        self.assertEqual(info["cell"], 4)
        self.assertIn("SEED", info["computed"])
        self.assertLessEqual({"BONN_DATA_DIR", "RESULTS_ROOT", "SELECTED_IDS", "EPOCHS", "PLOT_EVERY"}, info["names"])

    def test_prep_notebook_tags_without_source_change(self):
        src = make_notebook(self.tmp / "u.ipynb", BONN_PARAMS, [PREVIOUS_32], tagged=False)
        with contextlib.redirect_stdout(io.StringIO()):
            out = rq.cmd_prep_notebook(src, self.tmp / "tagged.ipynb")
        a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (src, out))
        self.assertEqual([c["source"] for c in a["cells"]], [c["source"] for c in b["cells"]])
        self.assertEqual(b["cells"][1]["metadata"]["tags"], ["parameters"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rq.cmd_prep_notebook(out), out)          # already tagged: unchanged

    def chb_meta(self):
        subjects = {f"chb{i:02d}": [(f"chb{i:02d}", 6, 6)] for i in (1, 2, 3, 4, 7, 10)}
        subjects["chb01"].append(("chb21", 3, 3))                    # same subject, second case
        subjects["chb07"] = [("chb07", 8, 0)]                       # single-class subject
        return write_meta(self.tmp / "meta.csv", subjects)

    def test_chbmit_rule_order_without_csv(self):
        meta = self.chb_meta()
        with contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_make_chbmit("c", str(self.chb_nb()), str(self.tmp / "x.npz"), str(meta), 1)
        jobs = rq.load_jobs("c")
        self.assertEqual([j["id"] for j in jobs], ["chb01", "chb02", "chb03", "chb04", "chb07", "chb10"])
        self.assertTrue(all(j["expect_complete"] == 1 for j in jobs))
        p = jobs[0]["params"]
        self.assertEqual(p["SELECTED_TEST_UNITS"], ["chb01"])
        self.assertEqual((p["REUSE_COMPLETED"], p["AGGREGATE_ONLY"], p["CREATE_WORD_SUMMARY_REPORT"]),
                         (True, False, False))
        self.assertIn("notebook rule", rq.load_meta("c")["units_source"])

    def test_chbmit_csv_order_chunks_and_subset(self):
        meta = self.chb_meta()
        folds = self.tmp / "folds.csv"
        folds.write_text("fold,test_unit,lopo_unit\n2,chb03,subject\n1,chb10,subject\n3,chb01,subject\n"
                         "4,chb02,subject\n5,chb04,subject\n6,chb07,subject\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_make_chbmit("c", str(self.chb_nb()), str(self.tmp / "x.npz"), str(meta), 4, "{}", str(folds))
        jobs = rq.load_jobs("c")
        self.assertEqual([j["units"] for j in jobs], [["chb10", "chb03", "chb01", "chb02"], ["chb04", "chb07"]])
        self.assertEqual([j["expect_complete"] for j in jobs], [4, 2])
        self.assertEqual(rq.load_meta("c")["units"], ["chb10", "chb03", "chb01", "chb02", "chb04", "chb07"])
        with contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_make_chbmit("d", str(self.chb_nb()), str(self.tmp / "x.npz"), str(meta), 1,
                               '{"SELECTED_TEST_UNITS": ["chb07", "chb03"], "EPOCHS": 2}', str(folds))
        self.assertEqual([j["id"] for j in rq.load_jobs("d")], ["chb03", "chb07"])
        self.assertEqual(rq.load_jobs("d")[0]["params"]["EPOCHS"], 2)
        with self.assertRaisesRegex(rq.QueueError, "not LOPO units"):
            rq.cmd_make_chbmit("e", str(self.chb_nb()), "x", str(meta), 1, '{"SELECTED_TEST_UNITS": ["chb99"]}', str(folds))
        folds.write_text("fold,test_unit,lopo_unit\n1,chb01,case\n", encoding="utf-8")
        with self.assertRaisesRegex(rq.QueueError, "LOPO_UNIT"):
            rq.cmd_make_chbmit("e", str(self.chb_nb()), "x", str(meta), 1, "{}", str(folds))
        folds.write_text("fold,test_unit\n1,chb55\n", encoding="utf-8")
        with self.assertRaisesRegex(rq.QueueError, "do not occur"):
            rq.cmd_make_chbmit("e", str(self.chb_nb()), "x", str(meta), 1, "{}", str(folds))

    def test_chbmit_case_units(self):
        meta = self.chb_meta()
        with contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_make_chbmit("k", str(self.chb_nb()), "x", str(meta), 1, '{"LOPO_UNIT": "case"}')
        self.assertIn("chb21", [j["id"] for j in rq.load_jobs("k")])
        self.assertEqual(len(rq.load_jobs("k")), 7)

    def test_chbmit_protocol_csv_if_present(self):
        csv_path = ROOT / "protocol" / "chbmit_lopo_folds.csv"
        if not csv_path.is_file():
            self.skipTest("protocol/chbmit_lopo_folds.csv not present (kept local)")
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        units = [r["test_unit"] for r in sorted(rows, key=lambda r: int(r["fold"]))]
        meta = write_meta(self.tmp / "m.csv", {u: [(u, 6, 6)] for u in units})
        with contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_make_chbmit("p", str(self.chb_nb()), "x", str(meta), 1, "{}", str(csv_path))
        self.assertEqual([j["id"] for j in rq.load_jobs("p")], units)
        self.assertEqual(len(units), 23)


# ============================================================================== queue safety

class TestQueueRuns(_Env):
    def fake_queue(self, q="f", ids=("Z_vs_S", "N_vs_F")):
        nb = str(self.bonn_nb())
        with contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_make_bonn(q, nb, str(self.tmp), json.dumps({"SELECTED_IDS": list(ids)}))
        return q

    def test_worker_merge_finalize_with_fake_runs(self):
        q = self.fake_queue()
        with mock.patch.dict(os.environ, {"RP_FAKE_RUN": "1"}), contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_work(q, 1)
            self.assertEqual(rq.done_ids(q), {"Z_vs_S", "N_vs_F"})
            dest = rq.cmd_merge(q)
            code = rq.cmd_finalize(q)
        self.assertEqual(code, 0)
        self.assertEqual(rq.count_complete(dest), 22)
        src = rq.job_root(q, "Z_vs_S") / "study_fake0000000000"
        self.assertTrue(os.path.samefile(src / "Z_vs_S" / "fit_00" / "metrics.json",
                                         dest / "Z_vs_S" / "fit_00" / "metrics.json"))     # fit files linked
        self.assertFalse(os.path.samefile(src / "study.json", dest / "study.json"))        # others copied
        fin = rq.read_json(rq.qdir(q) / "finalize.json")
        self.assertEqual(fin["status"], "done")
        self.assertEqual(fin["selected"], ["Z_vs_S", "N_vs_F"])
        self.assertEqual(rq.queue_state(q)["waiting"], [])
        self.assertEqual(rq.live_workers(q), [])                    # worker unregistered on exit

    def test_failed_job_is_recorded_and_requeued(self):
        q = self.fake_queue()
        with mock.patch.dict(os.environ, {"RP_FAKE_RUN": "1"}), contextlib.redirect_stdout(io.StringIO()):
            job = rq.claim_next(q, 1)
            os.environ["RP_FAKE_FAIL"] = job["id"]
            rec = rq.run_job(q, job, 1)
            self.assertEqual(rec["status"], "failed")
            self.assertEqual(rec["complete"], 0)
            self.assertIn("fake failure", rec["error"])
            released = rq.cmd_requeue(q)
        self.assertEqual(released, [job["id"]])
        self.assertEqual(len(list((rq.qdir(q) / "failed" / "history").glob("*.json"))), 1)

    def test_concurrent_claims_are_exclusive(self):
        q = self.fake_queue(ids=())                           # all 32 jobs
        got, lock = [], threading.Lock()

        def grab():
            while True:
                j = rq.claim_next(q, 1)
                if j is None:
                    return
                with lock:
                    got.append(j["id"])
        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(got), 32)
        self.assertEqual(len(set(got)), 32)

    def claim_as(self, q, jid, pid, ident, procs=(), host=None, heartbeat=None):
        (rq.qdir(q) / "claims" / jid).mkdir(parents=True)
        rq.write_json(rq.qdir(q) / "claims" / jid / "owner.json",
                      dict(worker=9, pid=pid, ident=ident, host=host or rq.HOST, job=jid,
                           heartbeat=heartbeat or time.time(), procs=[list(p) for p in procs]))

    def test_requeue_never_releases_live_jobs(self):
        q = self.fake_queue(ids=("Z_vs_S", "N_vs_F", "Z_vs_O", "O_vs_S", "F_vs_S", "N_vs_S"))
        me = os.getpid()
        dead_pid, dead_ident = self.dead_pid()
        kernel = self.sleeper()
        self.claim_as(q, "Z_vs_S", me, rq.proc_ident(me))                                  # worker alive
        self.claim_as(q, "N_vs_F", dead_pid, dead_ident)                                   # everything dead
        self.claim_as(q, "Z_vs_O", dead_pid, dead_ident, [(kernel.pid, rq.proc_ident(kernel.pid))])  # orphan
        self.claim_as(q, "O_vs_S", dead_pid, dead_ident, host="other-host")               # foreign, fresh
        self.claim_as(q, "F_vs_S", dead_pid, dead_ident, host="other-host", heartbeat=time.time() - 3600)
        with contextlib.redirect_stdout(io.StringIO()):
            released = rq.cmd_requeue(q)
        self.assertEqual(sorted(released), ["F_vs_S", "N_vs_F"])
        st = rq.queue_state(q)
        self.assertIn("Z_vs_S", st["running"])
        self.assertIn("Z_vs_O", st["running"])
        rq.kill_tree(kernel.pid)
        kernel.wait(timeout=10)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rq.cmd_requeue(q, ids=["Z_vs_O"]), ["Z_vs_O"])

    def test_requeue_refuses_while_workers_run_unless_forced(self):
        q = self.fake_queue()
        rq.register_worker(q, 1)                                   # this test process acts as a worker
        dead_pid, dead_ident = self.dead_pid()
        self.claim_as(q, "Z_vs_S", dead_pid, dead_ident)
        with self.assertRaisesRegex(rq.QueueError, "--force"):
            rq.cmd_requeue(q)
        self.assertTrue((rq.qdir(q) / "claims" / "Z_vs_S").exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rq.cmd_requeue(q, force=True), ["Z_vs_S"])
        rq.unregister_worker(q, 1)

    def test_pid_reuse_is_not_mistaken_for_a_live_job(self):
        if rq.proc_ident(os.getpid()) == "pid":
            self.skipTest("no start-time identity on this platform without psutil")
        q = self.fake_queue()
        stale_ident = "ps:1.00" if rq.psutil else "proc:1"
        self.claim_as(q, "Z_vs_S", os.getpid(), stale_ident)     # same PID, different process start time
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rq.cmd_requeue(q), ["Z_vs_S"])

    def test_unrecorded_kernel_found_by_working_directory(self):
        if rq.psutil is None and not os.path.isdir("/proc"):
            self.skipTest("needs psutil or /proc")
        q = self.fake_queue()
        root = rq.job_root(q, "Z_vs_S")
        root.mkdir(parents=True)
        dead_pid, dead_ident = self.dead_pid()
        self.claim_as(q, "Z_vs_S", dead_pid, dead_ident)          # the record knows no kernel
        kernel = self.sleeper(cwd=root)
        self.assertIn(kernel.pid, rq.job_live_procs(q, "Z_vs_S"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rq.cmd_requeue(q, force=True), [])
        plain = self.sleeper(cwd=root, marker="unrelated_tool")   # not a python kernel/worker: ignored
        self.assertNotIn(plain.pid, rq.job_live_procs(q, "Z_vs_S"))

    def test_kill_is_scoped_to_one_queue(self):
        qa, qb = self.fake_queue("qa"), self.fake_queue("qb")
        dead_pid, dead_ident = self.dead_pid()
        ka, kb = self.sleeper(), self.sleeper()
        self.claim_as(qa, "Z_vs_S", dead_pid, dead_ident, [(ka.pid, rq.proc_ident(ka.pid))])
        self.claim_as(qb, "Z_vs_S", dead_pid, dead_ident, [(kb.pid, rq.proc_ident(kb.pid))])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rq.cmd_kill("qa"), 0)
        ka.wait(timeout=10)
        self.assertIsNotNone(ka.poll())
        self.assertIsNone(kb.poll())
        self.assertIn("Z_vs_S", rq.queue_state("qb")["running"])

    def test_orphaned_kernel_end_to_end(self):
        """A worker is killed while its (fake) kernel keeps running: requeue keeps the job, kill-workers
        removes the kernel, then requeue releases it and a new worker completes it."""
        q = self.fake_queue(ids=("Z_vs_S",))
        env = dict(os.environ, RP_FAKE_RUN="1", RP_FAKE_KERNEL="1", RP_HEARTBEAT="0.2")
        worker = subprocess.Popen([sys.executable, str(RUNPOD / "rp_queue.py"), "work", q, "1"], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(worker)
        owner = rq.qdir(q) / "claims" / "Z_vs_S" / "owner.json"
        for _ in range(100):
            info = rq.read_json(owner) or {}
            if info.get("procs") and info.get("pid") != worker.pid:
                break
            time.sleep(0.1)
        kernels = [p for p, _ in info["procs"]]
        self.assertTrue(kernels)
        worker_pid = info["pid"]                               # the interpreter (differs from Popen pid on Windows venvs)
        if rq.psutil is not None:
            rq.psutil.Process(worker_pid).kill()               # the worker only, not its tree
        else:
            os.kill(worker_pid, 9)
        time.sleep(0.5)
        st = rq.queue_state(q)
        self.assertIn("Z_vs_S", st["running"])                 # kernel alive -> still live
        self.assertEqual(st["workers"], [])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rq.cmd_requeue(q), [])
            self.assertEqual(rq.cmd_kill(q), 0)
            self.assertEqual(rq.cmd_requeue(q), ["Z_vs_S"])
        with mock.patch.dict(os.environ, {"RP_FAKE_RUN": "1"}), contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_work(q, 2)
        self.assertEqual(rq.done_ids(q), {"Z_vs_S"})

    def test_merge_refuses_mixed_study_ids(self):
        q = self.fake_queue()
        with mock.patch.dict(os.environ, {"RP_FAKE_RUN": "1"}), contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_work(q, 1)
        other = rq.job_root(q, "N_vs_F") / "study_fake0000000000"
        other.rename(other.with_name("study_other"))
        with self.assertRaisesRegex(rq.QueueError, "one study id"):
            rq.cmd_merge(q)

    def test_finalize_refuses_while_live_and_guard_trips(self):
        q = self.fake_queue()
        with mock.patch.dict(os.environ, {"RP_FAKE_RUN": "1"}), contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_work(q, 1)
        rq.register_worker(q, 5)
        with self.assertRaisesRegex(rq.QueueError, "live processes"):
            rq.cmd_finalize(q)
        rq.unregister_worker(q, 5)
        with contextlib.redirect_stdout(io.StringIO()):
            dest = rq.cmd_merge(q)
        guard = rq.train_guard(dest.parent, dest)
        self.assertIsNone(guard())
        (dest.parent / "study_new").mkdir()
        self.assertIn("STUDY_ID differs", guard())
        (dest.parent / "study_new").rmdir()
        next(dest.rglob("complete.json")).unlink()
        self.assertIn("re-training", guard())


class TestAutostop(_Env):
    def done_queue(self):
        nb = str(self.bonn_nb())
        with contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_make_bonn("a", nb, str(self.tmp), '{"SELECTED_IDS": ["Z_vs_S"]}')
        with mock.patch.dict(os.environ, {"RP_FAKE_RUN": "1"}), contextlib.redirect_stdout(io.StringIO()):
            rq.cmd_work("a", 1)
        return "a"

    def test_dry_run_stop_finalizes_packs_and_redacts(self):
        q = self.done_queue()
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"RP_FAKE_RUN": "1", "RUNPOD_POD_ID": "podabc"}), \
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("network in dry run")), \
                contextlib.redirect_stdout(out):
            self.assertEqual(rq.cmd_autostop(q, stop_pod=True, interval=0.05, dry_run=True), 0)
        text = out.getvalue() + (rq.ws() / "logs" / q / "autostop.log").read_text(encoding="utf-8")
        self.assertIn("DRY RUN: would send POST https://api.runpod.io/v2/pods/podabc/action", text)
        self.assertIn(rq.REDACTED, text)
        self.assertNotIn(FAKE_KEY, text)
        self.assertEqual(rq.read_json(rq.qdir(q) / "finalize.json")["status"], "done")
        self.assertEqual(len(list((rq.ws() / "packs").glob("results_a_*.tgz"))), 1)
        self.assertFalse((rq.qdir(q) / "autostop.json").exists())

    def test_without_flag_the_pod_is_never_stopped(self):
        q = self.done_queue()
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"RP_FAKE_RUN": "1", "RUNPOD_POD_ID": "podabc"}), \
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("must not call the API")), \
                contextlib.redirect_stdout(out):
            rq.cmd_autostop(q, stop_pod=False, interval=0.05, no_finalize=True)
        self.assertIn("pod NOT stopped", out.getvalue())

    def test_stop_request_uses_pod_env_and_hides_key(self):
        seen = {}

        class Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            seen["url"], seen["auth"], seen["body"] = req.full_url, req.get_header("Authorization"), req.data
            return Resp()
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"RUNPOD_POD_ID": "podabc"}), \
                mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), contextlib.redirect_stdout(out):
            self.assertTrue(rq.stop_this_pod(dry_run=False))
        self.assertEqual(seen["url"], "https://api.runpod.io/v2/pods/podabc/action")
        self.assertEqual(seen["auth"], "Bearer " + FAKE_KEY)
        self.assertEqual(json.loads(seen["body"]), {"action": "stop"})
        s = SNAP["v2"]["schemas"]
        self.assertEqual(validate(json.loads(seen["body"]), s["PodActionRequest"], s), [])
        self.assertNotIn(FAKE_KEY, out.getvalue())

    def test_no_pod_env_means_no_stop(self):
        with mock.patch.dict(os.environ, {"RUNPOD_POD_ID": ""}), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(rq.stop_this_pod())

    def test_single_watchdog_per_queue(self):
        q = self.done_queue()
        rq.write_json(rq.qdir(q) / "autostop.json", dict(pid=os.getpid(), ident=rq.proc_ident(os.getpid()),
                                                          host=rq.HOST))
        with self.assertRaisesRegex(rq.QueueError, "already running"):
            rq.cmd_autostop(q, interval=0.05)


# ============================================================================== static checks

def find_bash():
    cands = [os.environ.get("RP_TEST_BASH"), shutil.which("bash")]
    if os.name == "nt":
        cands += [r"C:\Program Files\Git\bin\bash.exe",
                  str(Path.home() / "AppData" / "Local" / "Programs" / "Git" / "bin" / "bash.exe")]
    for c in cands:
        if c and Path(c).is_file() and "system32" not in c.lower():      # skip the WSL launcher
            return c
    return None


class TestStatic(unittest.TestCase):
    SCRIPT = RUNPOD / "remote_setup.sh"

    def test_line_endings_lf(self):
        for p in list(RUNPOD.glob("*.sh")) + list(RUNPOD.glob("*.py")):
            self.assertNotIn(b"\r\n", p.read_bytes(), p.name)

    def test_bash_syntax(self):
        bash = find_bash()
        if not bash:
            self.skipTest("bash not found")
        r = subprocess.run([bash, "-n", str(self.SCRIPT)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_shellcheck(self):
        sc = os.environ.get("RP_SHELLCHECK") or shutil.which("shellcheck")
        if not sc:
            self.skipTest("shellcheck not installed (set RP_SHELLCHECK)")
        r = subprocess.run([sc, "-S", "style", str(self.SCRIPT)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_every_command_is_dispatched(self):
        text = self.SCRIPT.read_text(encoding="utf-8")
        for cmd in ("setup", "check-data", "preflight", "prep-notebook", "calibrate", "make-bonn-jobs",
                    "make-chbmit-jobs", "launch", "progress", "kill-workers", "requeue", "merge", "finalize",
                    "autostop", "pack", "mps-on", "mps-off"):
            self.assertRegex(text, rf"\n  {re.escape(cmd)}\) ", cmd)

    def test_public_files_hold_no_secrets_or_personal_paths(self):
        pat = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|org|net|edu)|[A-Z]:[\\/]Users[\\/]|/home/[a-z]+/"
                         r"|rpa_[A-Za-z0-9]{20,}", re.I)
        for p in sorted(RUNPOD.iterdir()):
            if p.is_file() and p.suffix in (".py", ".sh", ".md", ".txt", ".json"):
                hits = [m.group(0) for m in pat.finditer(p.read_text(encoding="utf-8"))
                        if "noreply" not in m.group(0)]
                self.assertEqual(hits, [], p.name)

    def test_requirements_are_pinned_without_torch(self):
        lines = [ln.split("#")[0].strip() for ln in (RUNPOD / "requirements-runpod.txt").read_text().splitlines()]
        lines = [ln for ln in lines if ln]
        self.assertTrue(all("==" in ln for ln in lines), lines)
        self.assertFalse(any(ln.lower().startswith("torch") for ln in lines))


if __name__ == "__main__":
    unittest.main()
