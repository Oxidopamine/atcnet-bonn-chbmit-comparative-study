#!/usr/bin/env python3
"""rp_queue.py - file-based job queue, papermill workers and result handling (pod side).

Called by remote_setup.sh; also runs locally (Windows or Linux) for smoke tests with WS set.
Everything lives under $WS (default /workspace):

  jobs/<Q>/jobs.jsonl           one job per line {"id","notebook","params","expect_complete",...}
  jobs/<Q>/meta.json            queue kind, frozen notebook, base params (used by finalize)
  jobs/<Q>/notebook.ipynb       notebook snapshot: every job and finalize run exactly this file
  jobs/<Q>/claims/<id>/         atomic claim (os.mkdir); owner.json = worker + process identities
  jobs/<Q>/workers/w<N>.json    worker registry (pid + start time)
  jobs/<Q>/done|failed/<id>.json job records (failed/history/ keeps released failures)
  jobs/<Q>/finalize.json        finalize record
  runs/<Q>/<id>/                per-job RESULTS_ROOT: no two processes ever share an output folder
  results/<Q>/study_<ID>/       merged study (fit folders hard-linked, other files copied)
  logs/<Q>/                     worker logs, papermill logs, executed notebooks
  packs/                        results archives with .sha256 and .manifest.sha256 sidecars

Process safety: a job is "live" while its worker, any recorded descendant (the papermill kernel)
or any python process whose working directory is inside the job folder is alive. Papermill kernels
survive a killed worker, so requeue/finalize/launch check all three and never release, merge or
duplicate a live job. PIDs are compared together with their start time (PID reuse after restarts).
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path

try:
    import psutil  # optional but recommended (pinned in requirements-runpod.txt)
except ImportError:  # pragma: no cover
    psutil = None

HOST = socket.gethostname()
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FOLDS_CSV_NAME = "chbmit_lopo_folds.csv"
BONN_FITS_PER_TASK = 11
PROC_MARKERS = ("ipykernel", "rp_queue.py", "papermill")
REDACTED = "***REDACTED***"


class QueueError(RuntimeError):
    pass


def ws() -> Path:
    return Path(os.environ.get("WS", "/workspace"))


def kernel_name() -> str:
    return os.environ.get("KERNEL", "atcnet-venv")


def heartbeat_s() -> float:
    return float(os.environ.get("RP_HEARTBEAT", "5"))


def qdir(q) -> Path:
    return ws() / "jobs" / q


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def atomic_write(path: Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{os.urandom(3).hex()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # Windows: target briefly open elsewhere
            time.sleep(0.05)
    os.replace(tmp, path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=1))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


# =============================================================================== processes

def proc_ident(pid):
    """Start-time identity of a live process ('ps:<t>', 'proc:<ticks>' or 'pid'), None if gone."""
    if not pid or pid <= 0:
        return None
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                return None
            return f"ps:{p.create_time():.2f}"
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return None
        except psutil.AccessDenied:
            return "pid"
    if os.path.isdir("/proc"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            return None
        return None if fields[0] in ("Z", "X") else f"proc:{fields[19]}"
    if os.name == "nt":                  # never os.kill(pid, 0) on Windows: signal 0 is CTRL_C_EVENT
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                             capture_output=True, text=True).stdout
        return "pid" if f'"{pid}"' in out else None
    try:
        os.kill(pid, 0)
        return "pid"
    except ProcessLookupError:
        return None
    except PermissionError:
        return "pid"


def alive(pid, ident=None) -> bool:
    cur = proc_ident(pid)
    if cur is None:
        return False
    if not ident or "pid" in (cur, ident) or cur.split(":")[0] != ident.split(":")[0]:
        return True                      # exists; identities not comparable
    if cur.startswith("ps:"):
        return abs(float(cur[3:]) - float(ident[3:])) < 1.0
    return cur == ident


def _proc_table():
    """pid -> (ppid, cmdline, cwd) without psutil (Linux /proc)."""
    table = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            ppid = int(Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()[1])
            cmd = Path(f"/proc/{d}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            try:
                cwd = os.readlink(f"/proc/{d}/cwd")
            except OSError:
                cwd = ""
            table[int(d)] = (ppid, cmd, cwd)
        except (OSError, IndexError, ValueError):
            continue
    return table


def descendants(pid) -> list:
    if psutil is not None:
        try:
            return [c.pid for c in psutil.Process(pid).children(recursive=True)]
        except psutil.Error:
            return []
    if os.path.isdir("/proc"):
        table = _proc_table()
        out, frontier = [], [pid]
        while frontier:
            kids = [p for p, (pp, _, _) in table.items() if pp in frontier]
            out += kids
            frontier = kids
        return out
    return []


def _norm(p) -> str:
    return os.path.normcase(os.path.realpath(str(p)))


def proc_snapshot() -> list:
    """[(pid, normalized cwd)] of python processes that look like kernels, workers or papermill."""
    me, out = os.getpid(), []
    if psutil is not None:
        for p in psutil.process_iter(["pid", "name"]):
            if p.pid == me or "python" not in (p.info.get("name") or "").lower():
                continue
            try:
                if not any(m in " ".join(p.cmdline()) for m in PROC_MARKERS):
                    continue
                cwd = p.cwd()
            except psutil.Error:
                continue
            if cwd:
                out.append((p.pid, _norm(cwd)))
    elif os.path.isdir("/proc"):
        for pid, (_, cmd, cwd) in _proc_table().items():
            if pid != me and cwd and any(m in cmd for m in PROC_MARKERS):
                out.append((pid, _norm(cwd)))
    return out


def procs_under(roots, snap=None) -> list:
    """Live python processes whose cwd is inside one of `roots`. Papermill runs the kernel with
    cwd = job folder, so this finds kernels that no record mentions."""
    roots = [_norm(r) for r in roots if Path(r).exists()]
    if not roots:
        return []
    snap = proc_snapshot() if snap is None else snap
    return [pid for pid, cwd in snap if any(cwd == r or cwd.startswith(r + os.sep) for r in roots)]


def kill_tree(pid, ident=None) -> bool:
    """Kills a process and all its descendants (children first). Returns True if it was alive."""
    if not alive(pid, ident):
        return False
    if psutil is not None:
        try:
            parent = psutil.Process(pid)
            procs = parent.children(recursive=True) + [parent]
        except psutil.Error:
            return False
        for p in procs:
            try:
                p.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(procs, timeout=5)
        return True
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        return True
    for child in reversed(descendants(pid)):
        try:
            os.kill(child, signal.SIGKILL)
        except OSError:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return True


# =============================================================================== queue files

def load_jobs(q) -> list:
    p = qdir(q) / "jobs.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_meta(q) -> dict:
    m = read_json(qdir(q) / "meta.json")
    if m is None:
        raise QueueError(f"no queue '{q}' under {ws() / 'jobs'} (make-bonn-jobs / make-chbmit-jobs first)")
    return m


def _ensure_dirs(q):
    for sub in ("claims", "done", "failed", "workers"):
        (qdir(q) / sub).mkdir(parents=True, exist_ok=True)


def done_ids(q) -> set:
    return {p.stem for p in (qdir(q) / "done").glob("*.json")}


def failed_ids(q) -> set:
    return {p.stem for p in (qdir(q) / "failed").glob("*.json")} - done_ids(q)


def claimed_ids(q) -> set:
    d = qdir(q) / "claims"
    return {p.name for p in d.iterdir() if p.is_dir()} if d.exists() else set()


def job_root(q, jid) -> Path:
    return ws() / "runs" / q / jid


def count_complete(root) -> int:
    root = Path(root)
    return sum(1 for _ in root.rglob("complete.json")) if root.exists() else 0


# ------------------------------------------------------------------------------- workers

def register_worker(q, wid):
    write_json(qdir(q) / "workers" / f"w{wid}.json",
               dict(wid=wid, pid=os.getpid(), ident=proc_ident(os.getpid()), host=HOST, started=time.time()))


def unregister_worker(q, wid):
    try:
        (qdir(q) / "workers" / f"w{wid}.json").unlink()
    except OSError:
        pass


def live_workers(q) -> list:
    out = []
    for f in sorted((qdir(q) / "workers").glob("w*.json")):
        w = read_json(f) or {}
        if w.get("host") == HOST and alive(w.get("pid"), w.get("ident")):
            out.append(w)
    return out


def next_wid(q) -> int:
    ids = [int(f.stem[1:]) for f in (qdir(q) / "workers").glob("w*.json") if f.stem[1:].isdigit()]
    live = {w["wid"] for w in live_workers(q)}
    return max([i for i in ids if i in live] or [0]) + 1


# ------------------------------------------------------------------------------- liveness

def job_live_procs(q, jid, has_record=None, snap=None) -> list:
    """PIDs keeping job `jid` alive: its worker (while no record exists), recorded descendants
    (kernels) and python processes working inside runs/<q>/<jid>."""
    if has_record is None:
        has_record = jid in done_ids(q) or (qdir(q) / "failed" / f"{jid}.json").exists()
    claim = qdir(q) / "claims" / jid
    owner = read_json(claim / "owner.json")
    pids = set()
    if owner and owner.get("host") == HOST:
        if not has_record and alive(owner.get("pid"), owner.get("ident")):
            pids.add(owner["pid"])
        for pid, ident in owner.get("procs", []):
            if alive(pid, ident):
                pids.add(pid)
    elif owner is None and claim.exists() and time.time() - claim.stat().st_mtime < 60:
        pids.add(-1)                     # just claimed; owner file not written yet
    pids.update(procs_under([job_root(q, jid)], snap))
    return sorted(pids)


def queue_state(q) -> dict:
    jobs = load_jobs(q)
    done, failed, claimed = done_ids(q), failed_ids(q), claimed_ids(q)
    running, stale, orphans = {}, [], {}
    snap = proc_snapshot()
    for jid in sorted(claimed - done - failed):
        live = job_live_procs(q, jid, has_record=False, snap=snap)
        if live:
            running[jid] = live
        else:
            stale.append(jid)
    for jid in sorted((done | failed) & claimed):
        live = job_live_procs(q, jid, has_record=True, snap=snap)
        if live:
            orphans[jid] = live
    extra = procs_under([ws() / "results" / q], snap)
    return dict(jobs=jobs, done=done, failed=failed, claimed=claimed, running=running, stale=stale,
                orphans=orphans, finalize_procs=extra, workers=live_workers(q),
                waiting=[j["id"] for j in jobs if j["id"] not in claimed and j["id"] not in done])


def live_summary(st) -> list:
    out = [f"worker w{w['wid']} pid {w['pid']}" for w in st["workers"]]
    out += [f"job {j} pids {p}" for j, p in st["running"].items()]
    out += [f"orphan kernel(s) of finished job {j} pids {p}" for j, p in st["orphans"].items()]
    out += [f"finalize/aggregate pids {st['finalize_procs']}"] if st["finalize_procs"] else []
    return out


# =============================================================================== notebooks

def nb_source(cell) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def _parse(src: str):
    lines = [ln if not ln.lstrip().startswith(("%", "!")) else "" for ln in src.splitlines()]
    return ast.parse("\n".join(lines))


def notebook_params(nb_path) -> dict:
    """Names assigned in the papermill 'parameters' cell, their literal defaults, and names the cell
    itself computes with (these cannot be overridden: the injected cell runs after it)."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"]) if "parameters" in c.get("metadata", {}).get("tags", [])]
    if not idx:
        raise QueueError(f"{nb_path} has no cell tagged 'parameters': run prep-notebook first "
                         "(without it papermill injects parameters BEFORE the defaults, which then win)")
    tree = _parse(nb_source(nb["cells"][idx[0]]))
    names, defaults, used = set(), {}, set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for t in targets:
            if isinstance(t, ast.Name):
                names.add(t.id)
                try:
                    defaults[t.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError, RecursionError):
                    pass
        if isinstance(node, ast.Assert):
            continue                     # validation of the defaults only
        scope = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else node
        if scope is not None:
            used |= {n.id for n in ast.walk(scope) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return dict(cell=idx[0], names=names, defaults=defaults, computed=used & names)


def check_params(info, params, allow_unknown=False):
    unknown = sorted(set(params) - info["names"])
    if unknown and not allow_unknown:
        raise QueueError(f"parameter(s) {unknown} are not assigned in the notebook's parameters cell "
                         f"(cell {info['cell']}); papermill would inject them silently with no effect")
    early = sorted(set(params) & info["computed"])
    if early:
        raise QueueError(f"parameter(s) {early} are already used inside the parameters cell (e.g. seeding) "
                         "before the injected values take effect; they cannot be overridden safely")


def cmd_prep_notebook(nb, out=None, markers=None):
    """Tags the configuration cell 'parameters' (metadata only; sources byte-identical)."""
    nb, markers = Path(nb), markers or ["BONN_DATA_DIR", "CHBMIT_NPZ", "RESULTS_ROOT"]
    data = json.loads(nb.read_text(encoding="utf-8"))
    tagged = [i for i, c in enumerate(data["cells"]) if "parameters" in c.get("metadata", {}).get("tags", [])]
    if tagged:
        print(f"{nb}: cell {tagged[0]} already tagged 'parameters'; using it unchanged")
        print(nb.resolve().as_posix())
        return nb
    hit = next((i for i, c in enumerate(data["cells"]) if c.get("cell_type") == "code"
                and any(m in nb_source(c) for m in markers)), None)
    if hit is None:
        raise QueueError(f"no code cell in {nb} contains any of {markers}")
    before = [nb_source(c) for c in data["cells"]]
    data["cells"][hit].setdefault("metadata", {}).setdefault("tags", []).append("parameters")
    assert [nb_source(c) for c in data["cells"]] == before
    out = Path(out) if out else ws() / "code" / f"pod_{nb.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"tagged cell {hit} as 'parameters' (0 source changes) -> {out}")
    print(out.resolve().as_posix())
    return out


def bonn_combinations(nb_path) -> list:
    """(id, n_classes, recordings) from the notebook's PREVIOUS_32 literal, in the notebook's own
    order: stable sort by number of output classes, descending (5 -> 4 -> 3 -> 2)."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    groups = None
    for c in nb["cells"]:
        src = nb_source(c)
        if c.get("cell_type") != "code" or "PREVIOUS_32" not in src:
            continue
        try:
            for node in _parse(src).body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PREVIOUS_32" for t in node.targets):
                    groups = ast.literal_eval(node.value)
        except SyntaxError:
            for line in src.splitlines():
                if line.startswith("PREVIOUS_32 = "):
                    groups = ast.literal_eval(line.split("=", 1)[1].strip())
        if groups is not None:
            break
    if groups is None:
        raise QueueError(f"PREVIOUS_32 literal not found in {nb_path}")
    cid = lambda g: "_vs_".join("+".join(x) for x in g)          # noqa: E731 (notebook's combination_id)
    ordered = sorted(groups, key=lambda g: -len(g))
    return [(cid(g), len(g), 100 * sum(map(len, g))) for g in ordered]


# ------------------------------------------------------------------------------- CHB-MIT units

def default_folds_csv():
    env = os.environ.get("FOLDS_CSV")
    if env:
        return Path(env)
    for p in (REPO / "protocol" / FOLDS_CSV_NAME, ws() / "code" / "seizure_study" / "protocol" / FOLDS_CSV_NAME,
              ws() / "code" / "protocol" / FOLDS_CSV_NAME):
        if p.is_file():
            return p
    return None


def chbmit_units(meta_csv, unit="subject", folds_csv=None):
    """Test units in fold order: from the protocol folds CSV when present, else the notebook's rule
    (natural ID order of the unit column: 'subject' for LOPO_UNIT='subject', 'patient' for 'case')."""
    col = {"subject": "subject", "case": "patient"}.get(unit)
    if col is None:
        raise QueueError(f"LOPO_UNIT must be 'subject' or 'case', got {unit!r}")
    with open(meta_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or col not in rows[0]:
        raise QueueError(f"{meta_csv} has no '{col}' column")
    avail = sorted({r[col] for r in rows}, key=natural_key)
    path = Path(folds_csv) if folds_csv else default_folds_csv()
    if path is None or not path.is_file():
        if folds_csv:
            raise QueueError(f"folds CSV not found: {folds_csv}")
        return avail, f"notebook rule (natural order of '{col}' in {Path(meta_csv).name})"
    with open(path, newline="", encoding="utf-8") as f:
        frows = list(csv.DictReader(f))
    if not frows:
        raise QueueError(f"{path} is empty")
    ucol = next((c for c in ("test_unit", "test_units", "test", "test_subject", "unit") if c in frows[0]), None)
    if ucol is None:
        raise QueueError(f"{path}: no test-unit column (expected 'test_unit')")
    kinds = {r.get("lopo_unit") for r in frows if r.get("lopo_unit")}
    if kinds and kinds != {unit}:
        raise QueueError(f"{path} was built for LOPO_UNIT={sorted(kinds)}, the queue uses {unit!r}")
    if "fold" in frows[0]:
        frows.sort(key=lambda r: int(r["fold"]))
    units = []
    for r in frows:
        for u in r[ucol].split(";"):
            if u and u not in units:
                units.append(u)
    unknown = [u for u in units if u not in avail]
    if unknown:
        raise QueueError(f"{path}: units {unknown} do not occur in the '{col}' column of {meta_csv}")
    if len(units) != len(avail):
        log(f"WARNING: {path.name} lists {len(units)} of {len(avail)} units")
    return units, f"protocol folds CSV {path}"


# =============================================================================== job creation

def _freeze_notebook(q, nb) -> Path:
    """Copies the notebook into the queue folder so every job and finalize run the same code."""
    dst = qdir(q) / "notebook.ipynb"
    blob = Path(nb).read_bytes()
    if dst.exists() and dst.read_bytes() != blob:
        raise QueueError(f"queue {q} already exists with a different notebook; use a new queue name")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_bytes(blob)
    return dst


def _init_queue(q, meta):
    _ensure_dirs(q)
    old = read_json(qdir(q) / "meta.json")
    keys = ("kind", "notebook_sha256", "base_params")
    if old and any(old.get(k) != meta.get(k) for k in keys):
        raise QueueError(f"queue {q} exists with a different kind/notebook/base params (would mix STUDY_IDs); "
                         "use a new queue name")
    if not old:
        write_json(qdir(q) / "meta.json", meta)


def add_job(q, job) -> bool:
    if any(j["id"] == job["id"] for j in load_jobs(q)):
        print("exists, skipped:", job["id"])
        return False
    with open(qdir(q) / "jobs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(job) + "\n")
    return True


def _abs(p) -> str:
    return Path(p).resolve().as_posix()


def cmd_make_bonn(q, nb, data_dir, extra_json="{}", allow_unknown=False):
    extra = json.loads(extra_json or "{}")
    info = notebook_params(nb)
    selected = extra.pop("SELECTED_IDS", None) or []
    if not Path(data_dir).is_dir():
        log(f"WARNING: data dir {data_dir} does not exist (yet)")
    base = {"BONN_DATA_DIR": _abs(data_dir), "MAX_CLASSES_TO_RUN": None}
    base.update(extra)
    check_params(info, dict(base, RESULTS_ROOT="", SELECTED_IDS=[]), allow_unknown)
    combos = bonn_combinations(nb)
    ids = [c[0] for c in combos]
    unknown = sorted(set(selected) - set(ids))
    if unknown:
        raise QueueError(f"unknown combination id(s): {unknown}")
    frozen = _freeze_notebook(q, nb)
    _init_queue(q, dict(kind="bonn", notebook=frozen.as_posix(), notebook_source=_abs(nb),
                        notebook_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                        base_params=base, select_key="SELECTED_IDS", root_key="RESULTS_ROOT",
                        param_names=sorted(info["names"]), created=time.time()))
    n = 0
    for cid, k, rec in combos:
        if selected and cid not in selected:
            continue
        params = dict(base, RESULTS_ROOT="{JOB_ROOT}", SELECTED_IDS=[cid])
        n += add_job(q, dict(id=cid, kind="bonn", notebook=frozen.as_posix(), params=params,
                             expect_complete=BONN_FITS_PER_TASK, classes=k, recordings=rec))
    jobs = load_jobs(q)
    print(f"queue {q}: {len(jobs)} jobs ({n} new), expected complete.json = "
          f"{sum(j['expect_complete'] for j in jobs)}; order: " + " -> ".join(
              f"{k}-class x{sum(1 for j in jobs if j.get('classes') == k)}" for k in (5, 4, 3, 2)
              if any(j.get("classes") == k for j in jobs)))


def cmd_make_chbmit(q, nb, npz, meta_csv, per_job=1, extra_json="{}", folds_csv=None, allow_unknown=False):
    extra = json.loads(extra_json or "{}")
    info = notebook_params(nb)
    per_job = max(1, int(per_job))
    unit = extra.get("LOPO_UNIT", info["defaults"].get("LOPO_UNIT", "subject"))
    selected = extra.pop("SELECTED_TEST_UNITS", None) or []
    units, source = chbmit_units(meta_csv, unit, folds_csv)
    unknown = sorted(set(selected) - set(units))
    if unknown:
        raise QueueError(f"SELECTED_TEST_UNITS {unknown} are not LOPO units ({source})")
    if selected:
        units = [u for u in units if u in selected]
    plan = extra.get("LOPO_PLAN_PATH")
    if plan and not Path(plan).is_file():
        raise QueueError(f"LOPO_PLAN_PATH {plan} does not exist")
    base = {"CHBMIT_NPZ": _abs(npz), "CHBMIT_META": _abs(meta_csv)}
    for k, v in (("REUSE_COMPLETED", True), ("AGGREGATE_ONLY", False), ("CREATE_WORD_SUMMARY_REPORT", False)):
        if k in info["names"]:
            base[k] = v
    base.update(extra)
    check_params(info, dict(base, RESULTS_ROOT="", SELECTED_TEST_UNITS=[]), allow_unknown)
    frozen = _freeze_notebook(q, nb)
    _init_queue(q, dict(kind="chbmit", notebook=frozen.as_posix(), notebook_source=_abs(nb),
                        notebook_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                        base_params=base, select_key="SELECTED_TEST_UNITS", root_key="RESULTS_ROOT",
                        units=units, lopo_unit=unit, units_source=source,
                        param_names=sorted(info["names"]), created=time.time()))
    n = 0
    for i in range(0, len(units), per_job):
        chunk = units[i:i + per_job]
        jid = "+".join(chunk)
        params = dict(base, RESULTS_ROOT="{JOB_ROOT}", SELECTED_TEST_UNITS=chunk)
        n += add_job(q, dict(id=jid, kind="chbmit", notebook=frozen.as_posix(), params=params,
                             expect_complete=len(chunk), units=chunk))
    jobs = load_jobs(q)
    print(f"queue {q}: {len(jobs)} jobs ({n} new) covering {len(units)} test {unit}s; "
          f"expected complete.json = {sum(j['expect_complete'] for j in jobs)}; units from {source}")


def cmd_add(q, jid, nb, params_json, expect=0, allow_unknown=False):
    params = json.loads(params_json)
    info = notebook_params(nb)
    check_params(info, params, allow_unknown)
    frozen = _freeze_notebook(q, nb)
    base = {k: v for k, v in params.items() if k not in ("RESULTS_ROOT", "SELECTED_IDS")}
    _ensure_dirs(q)
    if not (qdir(q) / "meta.json").exists():
        write_json(qdir(q) / "meta.json", dict(kind="generic", notebook=frozen.as_posix(), notebook_source=_abs(nb),
                                                notebook_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                                                base_params=base, select_key="SELECTED_IDS",
                                                root_key="RESULTS_ROOT", param_names=sorted(info["names"]),
                                                created=time.time()))
    add_job(q, dict(id=jid, kind="generic", notebook=frozen.as_posix(), params=params, expect_complete=int(expect)))


# =============================================================================== running

def _subst(obj, key, value):
    if isinstance(obj, str):
        return obj.replace(key, value)
    if isinstance(obj, list):
        return [_subst(x, key, value) for x in obj]
    if isinstance(obj, dict):
        return {k: _subst(v, key, value) for k, v in obj.items()}
    return obj


class Heartbeat(threading.Thread):
    """Records this process and its descendants (the kernel) in owner.json while a job runs.
    An optional guard is polled; when it returns a message, the kernel is killed."""

    def __init__(self, path, base, guard=None, interval=None):
        super().__init__(daemon=True)
        self.path, self.base, self.guard = Path(path), dict(base), guard
        self.interval = interval or heartbeat_s()
        self.procs = {}
        self.tripped = None
        self._stop_evt = threading.Event()

    def beat(self):
        for pid in descendants(os.getpid()):
            if pid not in self.procs:
                ident = proc_ident(pid)
                if ident:
                    self.procs[pid] = ident
        write_json(self.path, dict(self.base, heartbeat=time.time(), procs=[[p, i] for p, i in self.procs.items()]))
        if self.guard and not self.tripped:
            msg = self.guard()
            if msg:
                self.tripped = msg
                log("GUARD:", msg, "-> killing the kernel")
                for pid in descendants(os.getpid()):
                    kill_tree(pid)

    def run(self):
        while not self._stop_evt.wait(self.interval):
            try:
                self.beat()
            except Exception as e:  # noqa: BLE001  (never let the heartbeat kill a job)
                log("heartbeat error:", e)

    def stop(self):
        self._stop_evt.set()


def fake_run(job, root: Path):
    """RP_FAKE_RUN=1: simulates a notebook run for tests (no papermill, no GPU)."""
    study = root / "study_fake0000000000"
    study.mkdir(parents=True, exist_ok=True)
    if os.environ.get("RP_FAKE_FAIL") == job["id"]:
        raise RuntimeError("fake failure")
    kernel = None
    if os.environ.get("RP_FAKE_KERNEL"):             # a child that outlives a killed worker
        kernel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "ipykernel_fake"],
                                  cwd=str(root), start_new_session=(os.name != "nt"))
        time.sleep(0.3)
    if job.get("units"):
        folders = [study / f"fold_{u}" for u in job["units"]]
    elif job["id"] == "__finalize__":
        folders = []
    else:
        folders = [study / job["id"] / f"fit_{i:02d}" for i in range(int(job.get("expect_complete") or 1))]
    for f in folders:
        f.mkdir(parents=True, exist_ok=True)
        (f / "history.csv").write_text("epoch,val_macro_f1\n1,0.5\n2,0.9\n")
        (f / "metrics.json").write_text("{}")
        time.sleep(float(os.environ.get("RP_FAKE_SLEEP", "0.02")))
        (f / "complete.json").write_text(json.dumps({"signature": "x", "files": ["metrics.json"]}))
    (study / "study.json").write_text("{}")
    if kernel is not None:
        kernel.wait()


def run_notebook(q, job, root: Path, owner_path: Path, owner_base: dict, guard=None, out_name=None):
    """Runs one papermill job with RESULTS_ROOT = root. Returns (error or None, guard message)."""
    root.mkdir(parents=True, exist_ok=True)
    logdir = ws() / "logs" / q
    logdir.mkdir(parents=True, exist_ok=True)
    out_name = out_name or job["id"]
    params = _subst(job["params"], "{JOB_ROOT}", root.as_posix())
    handler = logging.FileHandler(logdir / f"{out_name}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    rootlog = logging.getLogger()
    rootlog.addHandler(handler)
    rootlog.setLevel(logging.INFO)
    os.environ["RP_JOB_ROOT"] = root.as_posix()
    os.environ["PYTHONPATH"] = os.pathsep.join([str(REPO)] + [p for p in os.environ.get("PYTHONPATH", "").split(
        os.pathsep) if p and p != str(REPO)])     # lets notebooks import the repo's modules
    hb = Heartbeat(owner_path, owner_base, guard=guard)
    hb.beat()
    hb.start()
    err, cwd0 = None, os.getcwd()
    try:
        if os.environ.get("RP_FAKE_RUN"):
            os.chdir(root)                       # papermill also runs with cwd = job folder
            fake_run(job, root)
        else:
            import papermill as pm
            pm.execute_notebook(job["notebook"], str(logdir / f"{out_name}.ipynb"), parameters=params,
                                kernel_name=kernel_name(), log_output=True, progress_bar=False,
                                request_save_on_cell_execute=True, cwd=str(root))
    except KeyboardInterrupt:
        raise
    except BaseException as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[-2000:]}"
    finally:
        os.chdir(cwd0)
        hb.stop()
        try:
            hb.beat()                            # final process record
        except Exception:  # noqa: BLE001
            pass
        rootlog.removeHandler(handler)
        handler.close()
    if hb.tripped:
        err = f"GUARD: {hb.tripped}" + (f" | {err}" if err else "")
    return err, hb.tripped


def run_job(q, job, wid):
    root = job_root(q, job["id"])
    claim = qdir(q) / "claims" / job["id"]
    base = dict(worker=wid, pid=os.getpid(), ident=proc_ident(os.getpid()), host=HOST, job=job["id"],
                claimed=time.time())
    t0 = time.time()
    err, _ = run_notebook(q, job, root, claim / "owner.json", base)
    n = count_complete(root)
    expect = int(job.get("expect_complete") or 0)
    ok = (n >= expect) if expect else err is None
    rec = dict(id=job["id"], worker=wid, host=HOST, start=t0, end=time.time(),
               minutes=round((time.time() - t0) / 60, 2), complete=n, expect=expect,
               status=("done" if ok and not err else "done_with_errors" if ok else "failed"), error=err)
    write_json(qdir(q) / ("done" if ok else "failed") / f"{job['id']}.json", rec)
    return rec


def claim_next(q, wid):
    done = done_ids(q)
    for job in load_jobs(q):
        if job["id"] in done:
            continue
        try:
            os.mkdir(qdir(q) / "claims" / job["id"])          # atomic
        except FileExistsError:
            continue
        write_json(qdir(q) / "claims" / job["id"] / "owner.json",
                   dict(worker=wid, pid=os.getpid(), ident=proc_ident(os.getpid()), host=HOST,
                        job=job["id"], claimed=time.time(), procs=[]))
        return job
    return None


def cmd_work(q, wid, delay=0.0):
    """Worker loop: claim the next job, run it, repeat until the queue is empty or STOP exists.
    `delay` staggers worker starts so kernels do not all load the data at the same moment."""
    _ensure_dirs(q)
    register_worker(q, wid)
    try:
        time.sleep(max(0.0, float(delay)))
        while True:
            if (qdir(q) / "STOP").exists():
                log(f"worker {wid}: STOP file present, exiting")
                return
            job = claim_next(q, wid)
            if job is None:
                log(f"worker {wid}: no more jobs")
                return
            log(f"worker {wid}: START {job['id']}")
            rec = run_job(q, job, wid)
            log(f"worker {wid}: {rec['status'].upper()} {job['id']} ({rec['complete']}/{rec['expect']} fits, "
                f"{rec['minutes']} min) {rec['error'] or ''}")
    finally:
        unregister_worker(q, wid)


# =============================================================================== monitoring

def _last_epoch(root: Path):
    hs = list(root.rglob("history.csv")) if root.exists() else []
    if not hs:
        return ""
    h = max(hs, key=lambda p: p.stat().st_mtime)
    try:
        last = h.read_text(encoding="utf-8").strip().splitlines()[-1].split(",")[0]
    except (OSError, IndexError):
        last = "?"
    return f"{h.parent.name} epoch {last}"


def cmd_progress(q):
    meta, st = load_meta(q), queue_state(q)
    jobs = st["jobs"]
    by_id = {j["id"]: j for j in jobs}
    runs = ws() / "runs" / q
    expect = sum(int(j.get("expect_complete") or 0) for j in jobs)
    stamps = sorted(p.stat().st_mtime for p in runs.rglob("complete.json")) if runs.exists() else []
    n = len(stamps)
    print(f"queue {q} ({meta['kind']}, notebook {Path(meta.get('notebook_source', meta['notebook'])).name})")
    print(f"jobs: {len(st['done'])}/{len(jobs)} done, {len(st['running'])} running, {len(st['failed'])} failed, "
          f"{len(st['stale'])} interrupted, {len(st['waiting'])} waiting | live workers: {len(st['workers'])}")
    print(f"fits complete: {n}/{expect}" + (f" ({100 * n / expect:.1f}%)" if expect else ""))
    recent = [t for t in stamps if t > time.time() - 3600]
    claims = [p.stat().st_mtime for p in (qdir(q) / "claims").iterdir()] if (qdir(q) / "claims").exists() else []
    if len(stamps) >= 2:
        span = max(1e-3, (time.time() - min(claims + stamps[:1])) / 3600)
        rate = len(recent) / min(1.0, span)          # fits per hour over the last hour (or since start)
        if rate > 0:
            print(f"rate: {rate:.1f} fits/h (last hour), {n / span:.1f}/h since the first claim; "
                  f"ETA ~{(expect - n) / rate:.1f} h (jobs differ in size)")
    for jid, pids in st["running"].items():
        owner = read_json(qdir(q) / "claims" / jid / "owner.json") or {}
        exp = by_id.get(jid, {}).get("expect_complete", "?")
        state = "running " if alive(owner.get("pid"), owner.get("ident")) or -1 in pids else "ORPHANED"
        print(f"  {state} w{str(owner.get('worker', '?')):<3} {jid:34s} {count_complete(runs / jid):>3}/{exp} fits  "
              f"{_last_epoch(runs / jid)}" + ("  (worker dead, kernel alive: kill-workers, then requeue)"
                                              if state == "ORPHANED" else ""))
    for jid in sorted(st["failed"]):
        r = read_json(qdir(q) / "failed" / f"{jid}.json") or {}
        print(f"  FAILED   {jid:38s} {r.get('complete')}/{r.get('expect')} fits  {(r.get('error') or '')[:150]}")
    for jid in st["stale"]:
        print(f"  INTERRUPTED {jid:35s} {count_complete(runs / jid)} fits kept; no live process -> requeue")
    for jid, pids in st["orphans"].items():
        print(f"  ORPHAN   kernel(s) {pids} of finished job {jid} still alive -> kill-workers {q}")
    if (qdir(q) / "STOP").exists():
        print("STOP file present: workers exit after their current job")
    fin = read_json(qdir(q) / "finalize.json")
    if fin:
        print(f"finalize: {fin.get('status')} at {time.strftime('%Y-%m-%d %H:%M', time.localtime(fin.get('end', 0)))}"
              f" {fin.get('error') or ''}")


def cmd_live(q) -> int:
    st = queue_state(q)
    lines = live_summary(st)
    for line in lines:
        print(line)
    if not lines:
        print(f"no live processes for queue {q}")
    return 3 if lines else 0


# =============================================================================== recovery

def cmd_requeue(q, force=False, ids=None):
    """Releases claims of interrupted or failed jobs whose processes are all dead."""
    st = queue_state(q)
    if st["workers"] and not force:
        raise QueueError(f"{len(st['workers'])} worker(s) of {q} are running "
                         f"({', '.join(str(w['pid']) for w in st['workers'])}). Live jobs are never released; "
                         f"to retry failed/interrupted jobs while the others run, use --force.")
    released, kept = [], []
    hist = qdir(q) / "failed" / "history"
    for jid in sorted(st["claimed"]):
        if (ids and jid not in ids) or jid in st["done"]:
            continue
        has_record = jid in st["failed"]
        live = job_live_procs(q, jid, has_record=has_record)
        if live:
            kept.append(f"{jid} (live pids {live})")
            continue
        owner = read_json(qdir(q) / "claims" / jid / "owner.json") or {}
        if owner.get("host") not in (None, HOST) and time.time() - float(owner.get("heartbeat", 0)) < 900:
            kept.append(f"{jid} (owned by host {owner['host']}, heartbeat < 15 min ago)")
            continue
        f = qdir(q) / "failed" / f"{jid}.json"
        if f.exists():
            hist.mkdir(parents=True, exist_ok=True)
            os.replace(f, hist / f"{jid}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        shutil.rmtree(qdir(q) / "claims" / jid)
        released.append(jid)
    for k in kept:
        print("  kept    ", k)
    for r in released:
        print("  released", r)
    print(f"released {len(released)} claim(s), kept {len(kept)} live; completed fits inside runs/{q}/<job> "
          "are reused when a worker picks the job up again")
    return released


def cmd_kill(q):
    """Kills this queue's workers, their kernels (recorded or found by folder) and its finalize run.
    Processes of other queues are never touched."""
    st = queue_state(q)
    targets = []
    for jid in list(st["running"]) + list(st["orphans"]):
        owner = read_json(qdir(q) / "claims" / jid / "owner.json") or {}
        targets += [(pid, ident) for pid, ident in owner.get("procs", [])]
        targets += [(pid, None) for pid in (st["running"].get(jid) or st["orphans"].get(jid) or []) if pid > 0]
    targets += [(pid, None) for pid in st["finalize_procs"]]
    targets += [(w["pid"], w.get("ident")) for w in st["workers"]]
    auto = read_json(qdir(q) / "autostop.json") or {}
    if auto.get("host") == HOST and alive(auto.get("pid"), auto.get("ident")):
        targets.append((auto["pid"], auto.get("ident")))
        print("  also stopping the autostop watchdog of this queue (restart it after relaunching)")
    killed = [pid for pid, ident in dict.fromkeys(targets) if kill_tree(pid, ident)]
    time.sleep(0.5)
    left = live_summary(queue_state(q))
    print(f"killed {len(killed)} process tree(s) of queue {q}: {killed}")
    if left:
        print("STILL ALIVE:", *left, sep="\n  ")
        return 1
    print(f"no live processes left for {q}. Next: requeue {q} (completed fits are reused)")
    return 0


# =============================================================================== merge / finalize

def _link_or_copy(src: Path, dst: Path):
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            pass
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_fresh(src: Path, dst: Path):
    """Copy that never writes through a hard link shared with a job folder."""
    if dst.exists():
        s, d = src.stat(), dst.stat()
        if s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime) and not os.path.samefile(src, dst):
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def study_dirs(root: Path) -> list:
    return sorted(p for p in root.glob("study_*") if p.is_dir()) if root.exists() else []


def cmd_merge(q) -> Path:
    """Merges the study folders of DONE jobs into results/<q>/study_<ID>. Files inside folders that
    hold a complete.json (fit/fold results, only read on reuse) are hard-linked; all other files are
    copied, so the finalize run never writes through into a job folder."""
    done = done_ids(q)
    jobs = [j for j in load_jobs(q) if j["id"] in done]
    if not jobs:
        raise QueueError(f"queue {q} has no done jobs to merge")
    per_job = {j["id"]: study_dirs(job_root(q, j["id"])) for j in jobs}
    sids = {d.name for ds in per_job.values() for d in ds}
    if len(sids) != 1:
        detail = "; ".join(f"{k}: {[d.name for d in v]}" for k, v in per_job.items() if len(v) != 1 or
                           (v and v[0].name != sorted(sids)[0]))
        raise QueueError(f"expected one study id across done jobs, found {sorted(sids)} ({detail}). Different "
                         "code/config/package versions/device: do not merge.")
    sid = sids.pop()
    dest = ws() / "results" / q / sid
    linked = copied = 0
    for jid, ds in per_job.items():
        src = ds[0]
        fit_roots = {p.parent for p in src.rglob("complete.json")}
        for p in sorted(src.rglob("*")):
            if not p.is_file() or p.name.endswith(".tmp"):
                continue
            t = dest / p.relative_to(src)
            if any(r == p.parent or r in p.parents for r in fit_roots):
                _link_or_copy(p, t)
                linked += 1
            else:
                _copy_fresh(p, t)
                copied += 1
    print(f"merged {len(jobs)} done job(s) into {dest}: {linked} result files linked, {copied} copied; "
          f"complete.json there: {count_complete(dest)}")
    return dest


def train_guard(results_q: Path, dest: Path):
    """Finalize must never train: trips on a new study folder (STUDY_ID changed) or a removed
    completion marker (the notebook's run_fit deletes it right before re-training)."""
    studies0 = {p.name for p in study_dirs(results_q)}
    markers0 = list(dest.rglob("complete.json"))

    def check():
        new = {p.name for p in study_dirs(results_q)} - studies0
        if new:
            return (f"new study folder {sorted(new)}: STUDY_ID differs from the training runs (environment or "
                    "config changed); finalize would retrain everything")
        gone = [p for p in markers0 if not p.exists()]
        if gone:
            return f"{len(gone)} completion marker(s) removed (e.g. {gone[0]}): the notebook started re-training"
        return None
    return check


def cmd_finalize(q, force=False):
    meta = load_meta(q)
    st = queue_state(q)
    live = [x for x in live_summary(st)]
    if live and not force:
        raise QueueError("queue still has live processes; wait, or stop-workers/kill-workers first "
                         "(--force finalizes the done jobs anyway):\n  " + "\n  ".join(live))
    dest = cmd_merge(q)
    results_q = dest.parent
    jobs = {j["id"]: j for j in load_jobs(q)}
    params = dict(meta["base_params"])
    params[meta["root_key"]] = results_q.as_posix()
    if meta["kind"] == "chbmit":
        params["SELECTED_TEST_UNITS"] = meta["units"]
        for k, v in (("AGGREGATE_ONLY", True), ("CREATE_WORD_SUMMARY_REPORT", True), ("REUSE_COMPLETED", True)):
            if k in meta.get("param_names", [k]):
                params[k] = v
        missing = [u for u in meta["units"] if not (dest / f"fold_{u}" / "complete.json").exists()]
        if missing:
            print(f"WARNING: {len(missing)} fold(s) not complete, reported as missing: {missing}")
    else:
        sel = []
        for jid in st["done"]:
            if jid not in jobs:
                continue
            n = count_complete(dest / jid)
            if n >= int(jobs[jid].get("expect_complete") or 0):
                sel.append(jid)
            else:
                print(f"WARNING: {jid} has {n} complete.json in the merged study; excluded")
        order = [j for j in jobs if j in sel]
        if not order:
            raise QueueError("no complete job to finalize")
        if len(order) < len(jobs):
            print(f"WARNING: finalizing {len(order)}/{len(jobs)} jobs (the others are not complete)")
        params[meta["select_key"]] = order
    job = dict(id="__finalize__", notebook=meta["notebook"], params=params, expect_complete=0)
    print("finalize params:", json.dumps(params))
    before = count_complete(dest)
    t0 = time.time()
    base = dict(worker=0, pid=os.getpid(), ident=proc_ident(os.getpid()), host=HOST, job="__finalize__")
    err, tripped = run_notebook(q, job, results_q, qdir(q) / "finalize_owner.json", base,
                                guard=train_guard(results_q, dest), out_name="finalize")
    after = count_complete(dest)
    rec = dict(status="failed" if err else "done", error=err, start=t0, end=time.time(), study=dest.name,
               complete_before=before, complete_after=after, selected=params.get(meta["select_key"]))
    write_json(qdir(q) / "finalize.json", rec)
    print(json.dumps(rec, indent=1))
    if after != before and not tripped:
        print(f"WARNING: complete.json count changed during finalize ({before} -> {after})")
    for pat in ("selected_master_results.csv", "lopo_*", "*.docx"):
        for f in sorted(dest.glob(pat))[:6]:
            print("  output:", f)
    return 0 if not err else 1


# =============================================================================== pack

def _pack_items(q):
    root = ws()
    items = ([f"results/{q}", f"runs/{q}", f"logs/{q}", f"jobs/{q}"] if q else ["results", "runs", "logs", "jobs"])
    items = [i for i in items if (root / i).exists()]
    extras = ["constraints.txt"] + ([p.relative_to(root).as_posix() for p in (root / "logs").glob("*.*")
                                     if p.is_file()] if (root / "logs").exists() else [])
    return items + [e for e in extras if (root / e).is_file() and e not in items]


def cmd_pack(q=None, out_dir=None, no_checkpoints=False, keep=3):
    """results/runs/logs/jobs of a queue -> packs/results_<q>_<stamp>.tgz, safe while workers run:
    files are read whole (no torn members), vanished files are skipped, hard links stored once.
    Writes <tgz>.sha256 and <tgz>.manifest.sha256; the manifest is also the last member."""
    root = ws()
    items = _pack_items(q)
    if not items:
        raise QueueError(f"nothing to pack under {root}")
    out_dir = Path(out_dir) if out_dir else root / "packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    files, seen, est = [], set(), 0
    for it in items:
        base = root / it
        for p in ([base] if base.is_file() else sorted(base.rglob("*"))):
            if not p.is_file() or p.is_symlink() or p.name.endswith((".tmp", ".part")):
                continue
            if no_checkpoints and p.suffix in (".pt", ".pth"):
                continue
            try:
                s = p.stat()
            except OSError:
                continue
            files.append(p)
            if (s.st_dev, s.st_ino) not in seen or s.st_nlink < 2:
                est += s.st_size
            seen.add((s.st_dev, s.st_ino))
    free = shutil.disk_usage(out_dir).free
    if est > 0.9 * free:
        raise QueueError(f"pack needs up to {est / 1e9:.2f} GB but only {free / 1e9:.2f} GB are free in {out_dir}; "
                         "delete old packs or use --no-checkpoints")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"results_{q or 'all'}_{stamp}.tgz"
    part = out_dir / (name + ".part")
    lines, inodes, skipped = [], {}, []
    with tarfile.open(part, "w:gz", format=tarfile.PAX_FORMAT, compresslevel=6) as tar:
        for p in files:
            arc = p.relative_to(root).as_posix()
            try:
                s = p.stat()
                key = (s.st_dev, s.st_ino)
                if s.st_nlink > 1 and key in inodes:
                    ti = tarfile.TarInfo(arc)
                    ti.type, ti.linkname, ti.mtime, ti.mode = tarfile.LNKTYPE, inodes[key][0], int(s.st_mtime), 0o644
                    tar.addfile(ti)
                    lines.append(f"{inodes[key][1]}  {arc}")
                    continue
                data = p.read_bytes()
            except OSError as e:
                skipped.append(f"{arc} ({e.__class__.__name__})")
                continue
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mtime, ti.mode = len(data), int(s.st_mtime), 0o644
            tar.addfile(ti, io.BytesIO(data))
            h = hashlib.sha256(data).hexdigest()
            if s.st_nlink > 1:
                inodes[key] = (arc, h)
            lines.append(f"{h}  {arc}")
        blob = ("\n".join(lines) + "\n").encode()
        ti = tarfile.TarInfo("MANIFEST.sha256")
        ti.size, ti.mtime, ti.mode = len(blob), int(time.time()), 0o644
        tar.addfile(ti, io.BytesIO(blob))
    out = out_dir / name
    os.replace(part, out)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    (out_dir / (name + ".sha256")).write_bytes(f"{h.hexdigest()}  {name}\n".encode())
    (out_dir / (name + ".manifest.sha256")).write_bytes(blob)
    older = sorted(out_dir.glob(f"results_{q or 'all'}_*.tgz"))[:-keep] if keep > 0 else []
    for o in older:
        for f in (o, o.with_name(o.name + ".sha256"), o.with_name(o.name + ".manifest.sha256")):
            f.unlink(missing_ok=True)
    for s_ in skipped:
        print("  skipped (changed or vanished while packing):", s_)
    print(f"packed {len(lines)} files ({out.stat().st_size / 1e6:.1f} MB) -> {out}")
    print(f"  sha256 {h.hexdigest()}  (sidecars: {name}.sha256, {name}.manifest.sha256)"
          + (f"; removed {len(older)} older pack(s)" if older else ""))
    return out


# =============================================================================== autostop

def stop_this_pod(dry_run=False, logf=print) -> bool:
    """Stops THIS pod through the API with the pod's own RUNPOD_POD_ID / RUNPOD_API_KEY (pod-scoped).
    /workspace is kept; the GPU is released. The key is never printed."""
    pod, key = os.environ.get("RUNPOD_POD_ID", "").strip(), os.environ.get("RUNPOD_API_KEY", "").strip()
    if not pod or not key:
        logf("cannot stop the pod: RUNPOD_POD_ID / RUNPOD_API_KEY not set (not on a RunPod pod?)")
        return False
    attempts = [("POST", f"https://api.runpod.io/v2/pods/{pod}/action", {"action": "stop"}),
                ("POST", f"https://rest.runpod.io/v1/pods/{pod}/stop", None)]
    scrub = lambda s: str(s).replace(key, REDACTED)                     # noqa: E731
    if dry_run:
        for method, url, body in attempts:
            logf(f"DRY RUN: would send {method} {url} body={json.dumps(body)} "
                 f"headers={{Authorization: Bearer {REDACTED} ($RUNPOD_API_KEY)}}")
        return False
    import urllib.error
    import urllib.request
    for method, url, body in attempts:
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else b"",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                logf(f"stop request accepted: {method} {url} -> HTTP {r.status}")
                return True
        except urllib.error.HTTPError as e:
            logf(f"stop request refused: {method} {url} -> HTTP {e.code} {scrub(e.read()[:300])}")
        except Exception as e:  # noqa: BLE001
            logf(f"stop request failed: {method} {url}: {scrub(e)}")
    if shutil.which("runpodctl"):
        r = subprocess.run(["runpodctl", "stop", "pod", pod], capture_output=True, text=True)
        logf(f"runpodctl stop pod -> exit {r.returncode} {scrub(r.stdout + r.stderr)[:300]}")
        return r.returncode == 0
    return False


def cmd_autostop(q, stop_pod=False, interval=300.0, max_hours=None, dry_run=False, no_finalize=False,
                 min_idle_checks=2):
    """Watchdog: once the queue has no live process (twice in a row), finalize + pack, then (only with
    --stop-pod) stop this pod. With --max-hours it packs and stops at the deadline even if jobs run."""
    load_meta(q)
    prev = read_json(qdir(q) / "autostop.json") or {}
    if prev.get("host") == HOST and alive(prev.get("pid"), prev.get("ident")):
        raise QueueError(f"an autostop watchdog for {q} is already running (pid {prev['pid']})")
    logdir = ws() / "logs" / q
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / "autostop.log"

    def logf(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    write_json(qdir(q) / "autostop.json", dict(pid=os.getpid(), ident=proc_ident(os.getpid()), host=HOST,
                                                started=time.time(), stop_pod=stop_pod, max_hours=max_hours))
    logf(f"autostop watching {q}: interval {interval}s, stop pod: {stop_pod}{' (dry run)' if dry_run else ''}, "
         f"max hours: {max_hours}")
    t0, seen_live, idle = time.time(), False, 0
    reason = None
    while reason is None:
        st = queue_state(q)
        live = bool(st["workers"] or st["running"] or st["orphans"] or st["finalize_procs"])
        terminal = not st["waiting"] and not st["stale"] and not st["running"]
        if live:
            seen_live, idle = True, 0
        else:
            idle += 1
        if max_hours and time.time() - t0 > max_hours * 3600:
            reason = f"deadline of {max_hours} h reached"
        elif idle >= min_idle_checks and (seen_live or terminal):
            reason = (f"queue idle: {len(st['done'])} done, {len(st['failed'])} failed, "
                      f"{len(st['stale'])} interrupted, {len(st['waiting'])} waiting")
        else:
            time.sleep(interval)
    logf(f"finishing: {reason}")
    st = queue_state(q)
    if not no_finalize and st["done"] and not live_summary(st):
        try:
            code = cmd_finalize(q)
            logf(f"finalize exit {code}")
        except Exception as e:  # noqa: BLE001
            logf(f"finalize failed: {e}")
    try:
        out = cmd_pack(q)
        logf(f"packed {out}")
    except Exception as e:  # noqa: BLE001
        logf(f"pack failed: {e} (results stay on /workspace)")
    if stop_pod:
        stop_this_pod(dry_run=dry_run, logf=logf)
    else:
        logf("pod NOT stopped (autostop was started without --stop-pod); it keeps billing")
    (qdir(q) / "autostop.json").unlink(missing_ok=True)
    return 0


# =============================================================================== CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep-notebook"); s.add_argument("nb"); s.add_argument("out", nargs="?")
    s = sub.add_parser("make-bonn"); s.add_argument("q"); s.add_argument("nb"); s.add_argument("data_dir")
    s.add_argument("params", nargs="?", default="{}"); s.add_argument("--allow-unknown", action="store_true")
    s = sub.add_parser("make-chbmit"); s.add_argument("q"); s.add_argument("nb"); s.add_argument("npz")
    s.add_argument("meta"); s.add_argument("per_job", nargs="?", default="1")
    s.add_argument("params", nargs="?", default="{}"); s.add_argument("--folds-csv")
    s.add_argument("--allow-unknown", action="store_true")
    s = sub.add_parser("add"); s.add_argument("q"); s.add_argument("id"); s.add_argument("nb")
    s.add_argument("params"); s.add_argument("expect", nargs="?", default="0")
    s.add_argument("--allow-unknown", action="store_true")
    s = sub.add_parser("work"); s.add_argument("q"); s.add_argument("wid", type=int)
    s.add_argument("delay", nargs="?", type=float, default=0.0)
    for name in ("progress", "live", "kill", "merge", "next-wid"):
        s = sub.add_parser(name); s.add_argument("q")
    s = sub.add_parser("requeue"); s.add_argument("q"); s.add_argument("ids", nargs="*")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("finalize"); s.add_argument("q"); s.add_argument("--force", action="store_true")
    s = sub.add_parser("pack"); s.add_argument("q", nargs="?"); s.add_argument("--out-dir")
    s.add_argument("--no-checkpoints", action="store_true"); s.add_argument("--keep", type=int, default=3)
    s = sub.add_parser("autostop"); s.add_argument("q")
    s.add_argument("--stop-pod", action="store_true", help="stop THIS pod via the API when finished")
    s.add_argument("--interval", type=float, default=300.0); s.add_argument("--max-hours", type=float)
    s.add_argument("--dry-run", action="store_true"); s.add_argument("--no-finalize", action="store_true")
    a = ap.parse_args(argv)
    try:
        c = a.cmd
        if c == "prep-notebook":
            cmd_prep_notebook(a.nb, a.out)
        elif c == "make-bonn":
            cmd_make_bonn(a.q, a.nb, a.data_dir, a.params, a.allow_unknown)
        elif c == "make-chbmit":
            cmd_make_chbmit(a.q, a.nb, a.npz, a.meta, a.per_job, a.params, a.folds_csv, a.allow_unknown)
        elif c == "add":
            cmd_add(a.q, a.id, a.nb, a.params, a.expect, a.allow_unknown)
        elif c == "work":
            cmd_work(a.q, a.wid, a.delay)
        elif c == "progress":
            cmd_progress(a.q)
        elif c == "live":
            return cmd_live(a.q)
        elif c == "next-wid":
            print(next_wid(a.q))
        elif c == "requeue":
            cmd_requeue(a.q, a.force, a.ids)
        elif c == "kill":
            return cmd_kill(a.q)
        elif c == "merge":
            cmd_merge(a.q)
        elif c == "finalize":
            return cmd_finalize(a.q, a.force)
        elif c == "pack":
            cmd_pack(a.q, a.out_dir, a.no_checkpoints, a.keep)
        elif c == "autostop":
            return cmd_autostop(a.q, a.stop_pod, a.interval, a.max_hours, a.dry_run, a.no_finalize)
    except QueueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace", line_buffering=True)
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
