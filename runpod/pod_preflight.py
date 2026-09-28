#!/usr/bin/env python3
"""pod_preflight.py - checks a pod (or a local smoke-test machine) before any training starts.

Read-only except a scratch folder under RESULTS that is removed again.
  environment  python, torch, CUDA device + host driver (>= 12.8 for the cu128 image), a CUDA
               forward/backward, deterministic-algorithms mode (cell 4 of the notebooks), psutil
  packages     every package the notebooks import, versions against requirements-runpod.txt
               (a missing package would be pip-installed by the notebook at run time)
  kernel       the Jupyter kernel papermill uses resolves to this interpreter and imports torch
  Bonn         Z/O/N/F/S (or A-E, Set_X) folders, 100 *.txt/*.TXT each, 4096/4097 finite values,
               equal lengths, 500 distinct signals
  CHB-MIT      npz X [N, 8, 2560] float32 and y, metadata rows == N, labels equal, per-row
               signal_sha256, subjects, units (volts -> the notebook's UNIT_SCALE applies)
  manifest     DATA_MANIFEST.sha256 written by `rp.py bundle-data`, if present
  filesystem   concurrent atomic JSON replace, hard links (used by merge), free disk

usage: python pod_preflight.py [--bonn DIR] [--npz FILE] [--meta FILE] [--results DIR]
                               [--kernel NAME] [--no-gpu] [--skip-kernel] [--json OUT]
Defaults follow $WS (default /workspace): $WS/data/bonn, $WS/data/chbmit/chbmit_8ch.npz, ...
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import io
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = []
MODULES = [("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"), ("scipy", "scipy"),
           ("matplotlib", "matplotlib"), ("requests", "requests"), ("IPython", "ipython"),
           ("docx", "python-docx"), ("papermill", "papermill"), ("ipykernel", "ipykernel"),
           ("nbclient", "nbclient"), ("psutil", "psutil"), ("optree", "optree")]


def check(status, name, msg):
    """status: True/'ok', False/'fail', 'warn', 'skip'."""
    s = {True: "ok", False: "fail"}.get(status, status)
    RESULTS.append(dict(status=s, check=name, detail=msg))
    print(f"{s.upper():5s} {name}: {msg}", flush=True)


def _writer(args):
    target, reps = args
    errors = 0
    for i in range(reps):
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
        tmp.write_text(json.dumps(dict(pid=os.getpid(), i=i)), encoding="utf-8")
        for _ in range(100):
            try:
                tmp.replace(target)
                break
            except PermissionError:
                time.sleep(0.05)
        else:
            errors += 1
    return errors


def pinned_versions(req: Path) -> dict:
    pins = {}
    if req.is_file():
        for line in req.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)", line)
            if m:
                pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def check_environment(a):
    print("python", sys.version.split()[0], sys.executable)
    try:
        import torch
    except ImportError as e:
        check(False, "torch", f"not importable: {e}")
        return
    cuda = torch.cuda.is_available()
    if a.no_gpu:
        check("skip" if not cuda else True, "gpu", f"torch {torch.__version__}; CUDA available: {cuda} (--no-gpu)")
    else:
        check(cuda, "gpu", f"torch {torch.__version__}, CUDA build {torch.version.cuda}, "
              f"device {torch.cuda.get_device_name(0) if cuda else 'none'}")
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi], capture_output=True, text=True).stdout
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            v = tuple(int(x) for x in m.group(1).split("."))
            check(v >= (12, 8), "driver", f"host driver supports CUDA {m.group(1)} (image needs >= 12.8)")
    elif not a.no_gpu:
        check("warn", "driver", "nvidia-smi not found")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        check(True, "determinism", "torch.use_deterministic_algorithms(True, warn_only=True) works")
    except Exception as e:  # noqa: BLE001
        check(False, "determinism", f"{type(e).__name__}: {e} (cell 4 of the notebooks calls this)")
    finally:
        torch.use_deterministic_algorithms(False)
    if cuda:
        try:
            x = torch.randn(8, 1, 4097, device="cuda")
            conv = torch.nn.Conv1d(1, 16, 64).cuda()
            conv(x).sum().backward()
            torch.cuda.synchronize()
            check(True, "cuda-op", "Conv1d forward/backward on the GPU")
        except Exception as e:  # noqa: BLE001
            check(False, "cuda-op", f"{type(e).__name__}: {e}")
    pins = pinned_versions(Path(a.requirements))
    for mod, dist in MODULES:
        if importlib.util.find_spec(mod) is None:
            check(False if mod != "psutil" else "warn", f"pkg {dist}",
                  "MISSING" + (" (the notebook would pip-install it mid-run)" if mod != "psutil" else
                               " (orphan-kernel detection falls back to /proc)"))
            continue
        v = md.version(dist)
        want = pins.get(dist)
        check(True if not want or want == v else "warn", f"pkg {dist}",
              v + (f" (pinned {want})" if want and want != v else ""))


def check_kernel(a):
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        spec = KernelSpecManager().get_kernel_spec(a.kernel)
    except Exception as e:  # noqa: BLE001
        check(False, "kernel", f"kernel '{a.kernel}' not found: {e}")
        return
    exe = spec.argv[0]
    same = exe in ("python", "python3", "{python}") or Path(exe).resolve() == Path(sys.executable).resolve()
    check(True if same else "warn", "kernel", f"'{a.kernel}' -> {exe}" + ("" if same else
                                                                         f" (this interpreter: {sys.executable})"))
    if a.skip_kernel:
        return
    try:
        from jupyter_client.manager import start_new_kernel
        km, kc = start_new_kernel(kernel_name=a.kernel, startup_timeout=120)
        try:
            code = "import sys, torch, numpy; print(sys.executable, torch.__version__, numpy.__version__)"
            box = []
            kc.execute_interactive(code, timeout=120, output_hook=lambda m: _collect(m, box))
            text = "".join(box).strip()
            check(bool(text), "kernel-exec", text or "no output")
        finally:
            kc.stop_channels()
            km.shutdown_kernel(now=True)
    except Exception as e:  # noqa: BLE001
        check(False, "kernel-exec", f"{type(e).__name__}: {e}")


def _collect(msg, box):
    if msg.get("msg_type") == "stream":
        box.append(msg["content"].get("text", ""))
    elif msg.get("msg_type") == "error":
        box.append("ERROR " + msg["content"].get("ename", ""))


def check_bonn(d: Path):
    import numpy as np
    if not d.is_dir():
        check(False, "bonn", f"folder not found: {d}")
        return
    aliases = dict(zip("ZONFS", "ABCDE"))
    lengths, hashes = set(), set()
    for s in "ZONFS":
        cands = [d / n for n in (s, aliases[s], f"Set_{s}") if (d / n).is_dir()]
        if len(cands) != 1:
            check(False, f"bonn set {s}", f"expected exactly one of {s}/{aliases[s]}/Set_{s} under {d}, found {cands}")
            continue
        paths = sorted(p for p in cands[0].rglob("*") if p.is_file() and p.suffix.lower() == ".txt"
                       and "__MACOSX" not in p.parts)
        bad = 0
        for p in paths:
            v = np.loadtxt(io.StringIO(p.read_bytes().decode("utf-8-sig")))
            bad += not (v.ndim == 1 and len(v) in (4096, 4097) and np.isfinite(v).all())
            lengths.add(len(v))
            hashes.add(hashlib.sha256(v.astype("<f4").tobytes()).hexdigest())
        check(len(paths) == 100 and bad == 0, f"bonn set {s}",
              f"{cands[0].name}/ {len(paths)} files {sorted({p.suffix for p in paths})}, invalid={bad}")
    check(len(lengths) == 1, "bonn lengths", f"{sorted(lengths)}")
    check(len(hashes) == 500, "bonn distinct", f"{len(hashes)} / 500 distinct signals")


def check_chbmit(npz: Path, meta: Path):
    import numpy as np
    if not npz.is_file() or not meta.is_file():
        check("skip" if not npz.exists() and not meta.exists() else False, "chbmit",
              f"files not found: {npz} / {meta}")
        return
    import pandas as pd
    report = npz.with_name("chbmit_8ch_report.json")
    check(report.is_file(), "chbmit report", f"{report.name} next to the npz" + ("" if report.is_file() else
          " MISSING (the CHB-MIT notebook checks the channel order against it)"))
    z = np.load(npz)
    X, y = z["X"], z["y"]
    m = pd.read_csv(meta)
    check(X.ndim == 3 and X.shape[1:] == (8, 2560) and X.dtype == np.float32, "chbmit X",
          f"shape {X.shape} {X.dtype} (expected [N, 8, 2560] float32, channel-major)")
    check(len(m) == len(X) == len(y), "chbmit rows", f"X {len(X)}, y {len(y)}, metadata {len(m)}")
    need = {"segment_id", "patient", "subject", "label", "signal_sha256"}
    check(need <= set(m.columns), "chbmit columns", f"missing {sorted(need - set(m.columns))}" if need - set(m.columns)
          else "segment_id, patient, subject, label, signal_sha256 present")
    if len(m) != len(X) or not need <= set(m.columns):
        return
    check(bool((m["label"].to_numpy() == y).all()) and set(np.unique(y)) <= {0, 1}, "chbmit labels",
          f"{int((y == 0).sum())} non-seizure / {int((y == 1).sum())} seizure, metadata labels match y")
    check(bool(np.isfinite(X).all()), "chbmit finite", "all values finite")
    bad = sum(hashlib.sha256(X[i].astype("<f4").tobytes()).hexdigest() != h for i, h in enumerate(m["signal_sha256"]))
    check(bad == 0, "chbmit hashes", f"{len(m) - bad}/{len(m)} rows match signal_sha256")
    check(m["segment_id"].is_unique, "chbmit ids", "segment_id unique")
    check(True, "chbmit units", f"{m['subject'].nunique()} subjects, {m['patient'].nunique()} recording folders; "
          f"median |x| = {float(np.median(np.abs(X))):.3g} "
          + ("(volts: the notebook's UNIT_SCALE=1e6 converts to microvolts)" if np.median(np.abs(X)) < 1e-3
             else "(not volts)"))


def check_manifest(root: Path):
    man = root / "DATA_MANIFEST.sha256"
    if not man.is_file():
        check("skip", "data manifest", f"{man} not present (data not uploaded with rp.py bundle-data)")
        return
    bad, n = [], 0
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, name = line.split(None, 1)
        n += 1
        p = root / name.strip()
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(name.strip())
    check(not bad, "data manifest", f"{n - len(bad)}/{n} files match" + (f"; bad: {bad[:5]}" if bad else ""))


def check_filesystem(results: Path, ws: Path):
    scratch = results / ".preflight"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with mp.get_context("spawn").Pool(4) as pool:
            errors = sum(pool.map(_writer, [(scratch / "x.json", 200)] * 4))
        json.loads((scratch / "x.json").read_text())
        check(errors == 0, "atomic replace", f"concurrent JSON replace on {results}: {errors} failures / 800")
        a, b = scratch / "a.txt", scratch / "b.txt"
        a.write_text("x")
        try:
            os.link(a, b)
            check(True, "hard links", f"supported on {results} (merge stores results once)")
        except OSError as e:
            check("warn", "hard links", f"not supported ({e}); merge falls back to copies")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for p, need in ((ws, 5), (Path("/") if os.name != "nt" else None, 2)):
        if p is None or not p.exists():
            continue
        free = shutil.disk_usage(p).free / 1e9
        check(free > need, f"disk {p}", f"{free:.1f} GB free (need > {need} GB; ~1.4 MB per Bonn fit)")


def main():
    ws = Path(os.environ.get("WS", "/workspace"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bonn", default=str(ws / "data" / "bonn"))
    ap.add_argument("--npz", default=str(ws / "data" / "chbmit" / "chbmit_8ch.npz"))
    ap.add_argument("--meta", default=str(ws / "data" / "chbmit" / "chbmit_8ch_metadata.csv"))
    ap.add_argument("--results", default=str(ws / "results"))
    ap.add_argument("--kernel", default=os.environ.get("KERNEL", "atcnet-venv"))
    ap.add_argument("--requirements", default=str(HERE / "requirements-runpod.txt"))
    ap.add_argument("--no-gpu", action="store_true", help="CPU smoke test: GPU checks become informational")
    ap.add_argument("--skip-kernel", action="store_true", help="do not start the kernel")
    ap.add_argument("--json", help="write the check results here")
    a = ap.parse_args()
    check_environment(a)
    check_kernel(a)
    check_bonn(Path(a.bonn))
    check_chbmit(Path(a.npz), Path(a.meta))
    check_manifest(Path(a.bonn).parent)
    check_filesystem(Path(a.results), ws if ws.exists() else Path(a.results))
    failed = [r for r in RESULTS if r["status"] == "fail"]
    warned = [r for r in RESULTS if r["status"] == "warn"]
    print(f"PREFLIGHT {'PASSED' if not failed else 'FAILED'}: {len(RESULTS)} checks, {len(failed)} failed, "
          f"{len(warned)} warnings")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(dict(time=time.time(), python=sys.executable, results=RESULTS),
                                           indent=1), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
