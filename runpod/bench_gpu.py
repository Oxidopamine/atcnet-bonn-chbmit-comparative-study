#!/usr/bin/env python3
"""bench_gpu.py - how many concurrent notebook workers per GPU, with CUDA MPS off and on?

Each child process trains the notebook's own model class with the notebook's step recipe (batch 8,
class-weighted cross-entropy, AdamW 3e-4 / 1e-3, gradient clip 1.0, constrain_weights, the
per-step isfinite sync, deterministic cuDNN) on random data of the dataset's shape, then runs the
per-epoch evaluation, for --seconds. The sweep prints per-process ms/step under contention and the
projected wall-clock hours and cost of the full Bonn and CHB-MIT workloads.

usage: bench_gpu.py NOTEBOOK [--dataset bonn|chbmit] [--procs 1,2,4,6,8] [--seconds 40]
                    [--price 0.25] [--mps both|off|on] [--model-options JSON] [--device cuda]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MPS_PIPE, MPS_LOG = "/tmp/nvidia-mps", "/tmp/nvidia-mps-log"
MPS_FLAG = Path("/tmp/rp_mps_on")

# Full-study workloads (batch 8, 100 epochs). Bonn: notebook cells 8 + 18 split logic, 352 fits.
# CHB-MIT: protocol/chbmit_lopo_folds.csv, 23 subject folds (recomputed below when the CSV exists).
WORKLOADS = {
    "bonn": dict(fits=352, train_steps=1_049_800, eval_batches=1_255_750, shape=(1, 4097), n_classes=5,
                 n_train=382, n_val=68),
    "chbmit": dict(fits=23, train_steps=537_500, eval_batches=566_866, shape=(8, 2560), n_classes=2,
                   n_train=1880, n_val=80),
}
ARTIFACT_S_PER_FIT = 4.4       # evaluation_artifacts (ROC/PR/confusion figures), measured on CPU


def chbmit_workload():
    p = REPO / "protocol" / "chbmit_lopo_folds.csv"
    if not p.is_file():
        return
    import csv
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    try:
        steps = sum(100 * math.ceil(int(r["n_train"]) / 8) for r in rows)
        evals = sum(100 * (math.ceil(int(r["n_train"]) / 8) + math.ceil(int(r["n_val"]) / 8))
                    + math.ceil(int(r["n_test"]) / 8) for r in rows)
    except (KeyError, ValueError):
        return
    WORKLOADS["chbmit"].update(fits=len(rows), train_steps=steps, eval_batches=evals)


def plot_calls_per_fit(plot_every, epochs=100):
    return len({1, epochs} | {e for e in range(1, epochs + 1) if e % plot_every == 0})


# ------------------------------------------------------------------------------- model loading

def _src(c):
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def notebook_model_options(nb):
    for c in nb["cells"]:
        if "parameters" in c.get("metadata", {}).get("tags", []) or "MODEL_OPTIONS" in _src(c):
            for line in _src(c).splitlines():
                if line.startswith("MODEL_OPTIONS ="):
                    try:
                        return ast.literal_eval(line.split("=", 1)[1].strip())
                    except (ValueError, SyntaxError):
                        pass
    return {}


def load_model_class(nb_path, class_name=None):
    """The notebook's model class: executes its model cell up to the probe/loader code; falls back to
    the repository's generalized ATCNet when the notebook imports the model instead of defining it."""
    import numpy as np
    import torch
    from torch import nn
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    names = [class_name] if class_name else ["BonnClassifier", "ATCNet", "CHBMITClassifier"]
    for name in names:
        for c in nb["cells"]:
            s = _src(c)
            if c.get("cell_type") == "code" and re.search(rf"^class {name}\b", s, flags=re.M):
                cut = re.search(r"^(probe\s*=|def load_classifier)", s, flags=re.M)
                ns = {"np": np, "torch": torch, "nn": nn, "math": math}
                sys.path.insert(0, str(REPO))
                exec(compile(s[:cut.start()] if cut else s, "notebook_model_cell", "exec"), ns)
                return ns[name], name
    sys.path.insert(0, str(REPO))
    from atcnet.atcnet_torch import ATCNet
    return ATCNet, "atcnet.atcnet_torch.ATCNet"


# ------------------------------------------------------------------------------- child

def child(a):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(1)
    dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    Model, _ = load_model_class(a.notebook, a.class_name)
    wl = WORKLOADS[a.dataset]
    c, t = wl["shape"]
    kw = dict(json.loads(a.model_kwargs))
    kw.setdefault("n_times", t)
    if c > 1:
        kw.setdefault("n_chans", c)
    m = Model(n_classes=wl["n_classes"], **kw).to(dev)
    n_tr, n_va = wl["n_train"], wl["n_val"]
    X = (torch.randn(n_tr + n_va, c, t) * 60).float()
    Y = torch.randint(0, wl["n_classes"], (n_tr + n_va,))
    g = torch.Generator().manual_seed(0)

    def loader(idx, shuffle):
        return DataLoader(TensorDataset(X[idx], Y[idx]), batch_size=8, shuffle=shuffle, num_workers=0,
                          generator=g, pin_memory=dev.type == "cuda")
    tr, tre, va = loader(slice(0, n_tr), True), loader(slice(0, n_tr), False), loader(slice(n_tr, None), False)
    crit = nn.CrossEntropyLoss(weight=torch.ones(wl["n_classes"], device=dev))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-3)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)

    def train_epoch(limit=None):
        m.train()
        k = 0
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(m(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if hasattr(m, "constrain_weights"):
                m.constrain_weights()
            k += 1
            if limit and k >= limit:
                break
        return k

    @torch.no_grad()
    def eval_epoch(ld, limit=None):
        m.eval()
        k = 0
        for xb, yb in ld:
            xb, yb = xb.to(dev), yb.to(dev)
            lo = m(xb)
            nn.functional.cross_entropy(lo, yb, reduction="sum").item()
            lo.softmax(1).cpu().numpy()
            k += 1
            if limit and k >= limit:
                break
        return k
    train_epoch(limit=5)
    eval_epoch(va, limit=3)
    sync()
    t_tr = t_ev = 0.0
    n_st = n_ev = 0
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        t0 = time.perf_counter()
        n_st += train_epoch(limit=a.chunk)
        sync()
        t_tr += time.perf_counter() - t0
        t0 = time.perf_counter()
        n_ev += eval_epoch(tre, limit=a.chunk) + eval_epoch(va, limit=a.chunk // 4 or 1)
        sync()
        t_ev += time.perf_counter() - t0
    mem = torch.cuda.max_memory_reserved() / 2 ** 20 if dev.type == "cuda" else 0
    print(json.dumps(dict(train_steps=n_st, eval_batches=n_ev, t_train=t_tr, t_eval=t_ev, mem_mb=mem)), flush=True)


# ------------------------------------------------------------------------------- MPS

def mps_env(on: bool):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    if on:
        env.update(CUDA_MPS_PIPE_DIRECTORY=MPS_PIPE, CUDA_MPS_LOG_DIRECTORY=MPS_LOG)
    else:
        env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
        env.pop("CUDA_MPS_LOG_DIRECTORY", None)
    return env


def mps_start():
    ctl = shutil.which("nvidia-cuda-mps-control")
    if not ctl:
        return "nvidia-cuda-mps-control is not in this image"
    os.makedirs(MPS_PIPE, exist_ok=True)
    os.makedirs(MPS_LOG, exist_ok=True)
    r = subprocess.run([ctl, "-d"], env=mps_env(True), capture_output=True, text=True)
    if r.returncode != 0:
        return f"MPS daemon did not start: {(r.stderr or r.stdout).strip()[:200]}"
    time.sleep(1)
    probe = subprocess.run([sys.executable, "-c", "import torch; torch.zeros(1, device='cuda'); print('ok')"],
                           env=mps_env(True), capture_output=True, text=True, timeout=120)
    if "ok" not in probe.stdout:
        mps_stop()
        return f"CUDA does not work under MPS in this container: {probe.stderr.strip()[-200:]}"
    MPS_FLAG.touch()
    return None


def mps_stop():
    ctl = shutil.which("nvidia-cuda-mps-control")
    if ctl:
        subprocess.run([ctl], input="quit\n", env=mps_env(True), capture_output=True, text=True)
    MPS_FLAG.unlink(missing_ok=True)
    time.sleep(1)


# ------------------------------------------------------------------------------- parent

def plot_cost():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = Path(tempfile.mkdtemp())
    ep = list(range(1, 101))
    t = time.perf_counter()
    for metric in ("accuracy", "loss"):           # the notebook's plot_history: 2 figures, PNG 300 dpi + PDF
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ep, [1 - 1 / e for e in ep], marker=".", markersize=3, label="Training")
        ax.plot(ep, [1 - 1.3 / e for e in ep], marker=".", markersize=3, label="Inner validation")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / f"{metric}.png", dpi=300, bbox_inches="tight")
        fig.savefig(d / f"{metric}.pdf", bbox_inches="tight")
        plt.close(fig)
    shutil.rmtree(d, ignore_errors=True)
    return time.perf_counter() - t


def gpu_sample():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        u, mem = [float(x) for x in out.split(",")]
        return u, mem
    except Exception:  # noqa: BLE001
        return None, None


def sweep(a, procs, mode, t_plot):
    rows = []
    wl = WORKLOADS[a.dataset]
    plots = wl["fits"] * plot_calls_per_fit(a.plot_every)
    for P in procs:
        cmd = [sys.executable, __file__, a.notebook, "--child", "--seconds", str(a.seconds), "--device", a.device,
               "--dataset", a.dataset, "--model-kwargs", a.model_kwargs, "--chunk", str(a.chunk)]
        if a.class_name:
            cmd += ["--class-name", a.class_name]
        ps = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=mps_env(mode == "on")) for _ in range(P)]
        utils, mems = [], []
        while any(p.poll() is None for p in ps):
            u, mm = gpu_sample()
            if u is not None:
                utils.append(u)
                mems.append(mm)
            time.sleep(1)
        res = []
        for p in ps:
            out, err = p.communicate()
            try:
                res.append(json.loads(out.strip().splitlines()[-1]))
            except (ValueError, IndexError):
                print(f"child failed (P={P}, MPS {mode}):", err[-800:], flush=True)
                return rows
        ms_tr = statistics.mean(r["t_train"] / max(1, r["train_steps"]) for r in res) * 1e3
        ms_ev = statistics.mean(r["t_eval"] / max(1, r["eval_batches"]) for r in res) * 1e3
        per_proc_s = (wl["train_steps"] * ms_tr + wl["eval_batches"] * ms_ev) / 1e3 + plots * t_plot \
            + wl["fits"] * ARTIFACT_S_PER_FIT
        hours = per_proc_s / P / 3600
        agg = sum(r["train_steps"] / max(1e-9, r["t_train"] + r["t_eval"]) for r in res)
        row = dict(mps=mode, P=P, ms_train_step=round(ms_tr, 2), ms_eval_batch=round(ms_ev, 2),
                   agg_train_steps_s=round(agg, 1), gpu_util=round(statistics.mean(utils)) if utils else None,
                   gpu_mem_mb=max(mems) if mems else None, proc_mem_mb=round(max(r["mem_mb"] for r in res)),
                   hours=round(hours, 2), usd=round(hours * a.price, 2) if a.price else None)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if len(rows) >= 3 and rows[-1]["hours"] > 0.97 * min(r["hours"] for r in rows[:-1]) \
                and rows[-2]["hours"] > 0.97 * min(r["hours"] for r in rows[:-2]):
            print(f"(MPS {mode}: no further gain after P={P}; stopping this sweep)", flush=True)
            break
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("--dataset", choices=sorted(WORKLOADS), default="bonn")
    ap.add_argument("--procs", default="1,2,4,6,8,10,12,16")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--price", type=float, default=0.0, help="pod USD/h, for the cost column")
    ap.add_argument("--mps", choices=["both", "off", "on"], default="both")
    ap.add_argument("--plot-every", type=int, default=10, help="notebook PLOT_EVERY used for projections")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--class-name")
    ap.add_argument("--model-options", default=None, help="JSON overriding the notebook's MODEL_OPTIONS")
    ap.add_argument("--model-kwargs", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--chunk", type=int, default=40, help=argparse.SUPPRESS)
    ap.add_argument("--out", help="write the result rows as JSON here")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    chbmit_workload()
    if a.child:
        return child(a)
    nb = json.loads(Path(a.notebook).read_text(encoding="utf-8"))
    opts = notebook_model_options(nb)
    opts.update(json.loads(a.model_options or "{}"))
    a.model_kwargs = json.dumps(opts)
    cls = load_model_class(a.notebook, a.class_name)[1]
    wl = WORKLOADS[a.dataset]
    cpus = int(os.environ.get("RUNPOD_CPU_COUNT") or os.cpu_count() or 1)
    procs = [int(x) for x in a.procs.split(",") if int(x) <= max(1, cpus)]
    print(f"model {cls} with {opts}; dataset {a.dataset} {wl['shape']}; workload {wl['fits']} fits, "
          f"{wl['train_steps']:,} train steps, {wl['eval_batches']:,} eval batches; vCPUs {cpus}")
    t_plot = min(plot_cost(), plot_cost())
    print(f"plot_history equivalent: {t_plot:.2f} s/call, {plot_calls_per_fit(a.plot_every)} calls per fit "
          f"(PLOT_EVERY={a.plot_every})", flush=True)
    modes = ["off", "on"] if a.mps == "both" else [a.mps]
    if a.device != "cuda":
        modes = ["off"]
    was_on = MPS_FLAG.exists()
    rows, notes = [], []
    try:
        for mode in modes:
            if mode == "on" and not MPS_FLAG.exists():
                why = mps_start()
                if why:
                    notes.append(f"MPS skipped: {why}")
                    print(notes[-1], flush=True)
                    continue
            if mode == "off" and MPS_FLAG.exists():
                mps_stop()
            rows += sweep(a, procs, mode, t_plot)
    finally:
        if was_on and not MPS_FLAG.exists():
            mps_start()
        elif not was_on and MPS_FLAG.exists():
            mps_stop()
    if not rows:
        print("no successful measurement")
        return 1
    best = min(rows, key=lambda r: r["hours"])
    by_mode = {m: min((r for r in rows if r["mps"] == m), key=lambda r: r["hours"]) for m in {r["mps"] for r in rows}}
    print("\nms_* = per-process time under contention; hours = projected wall clock for the whole "
          f"{a.dataset} workload with P workers (train + eval + plots + artifacts, perfect balance).")
    for m, r in sorted(by_mode.items()):
        print(f"best with MPS {m}: P={r['P']} -> {r['hours']} h" + (f", ${r['usd']}" if r["usd"] is not None else ""))
    print(f"RECOMMENDATION: launch {best['P']} workers with MPS {best['mps']}"
          + (" (run `remote_setup.sh mps-on` first)" if best["mps"] == "on" else "")
          + f"; keep P <= vCPUs - 2 ({max(1, cpus - 2)}).")
    for n in notes:
        print(n)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(rows=rows, best=best, notes=notes, dataset=a.dataset,
                                               model=cls, options=opts), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
