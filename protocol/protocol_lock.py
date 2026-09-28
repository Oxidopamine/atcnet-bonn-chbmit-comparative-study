#!/usr/bin/env python
"""Protocol lock: show that two runs of the shared protocol are comparable before any training.

Bonn mode runs the notebook's own configuration, task, loader, fingerprint and split-planning
cells in plan-only mode (no model is built or trained; CPU only, about 30 s) and writes a lock:

  protocol_hash    shared hyper-parameters, the full preset task list, data handling, and the
                   normalised source of every shared cell (imports/env, setup, tasks, loader,
                   fingerprint, training loop, split planning). Operational knobs (paths,
                   SELECTED_IDS, MAX_CLASSES_TO_RUN, PLOT_EVERY, reuse/stop and report flags) and
                   model-specific items (MODEL_OPTIONS, model cell, provenance) are excluded.
  data_order_hash  set + float32 signal sha256 of every recording, in load order.
  split_plan_hash  every fit of every preset task: seed and the ordered (signal, target) lists of
                   train / validation / test. All preset tasks are planned whatever the run scope,
                   so staged (MAX_CLASSES_TO_RUN) and sharded (SELECTED_IDS) runs lock identically.

CHB-MIT mode hashes a leave-one-patient-out plan (lopo_plan.json). With --npz/--meta the folds
are re-keyed on segment signals, the plan is checked for leakage, and it is compared with the
decided fold rule (fold k tests unit k in natural order; validation = next subject, cyclically,
with >= min_val_per_class segments of each class; seed = SEED + k).

usage:
  protocol_lock.py --notebook NB.ipynb [--data BONN_DIR] [--parameters JSON|FILE] --out LOCK [--expect LOCK]
  protocol_lock.py --chbmit-plan PLAN.json [--npz X.npz --meta META.csv] --out LOCK [--expect LOCK]
  protocol_lock.py --chbmit-reference-plan OUT.json --npz X.npz --meta META.csv [--lopo-unit subject]
  protocol_lock.py --compare LOCK_A LOCK_B

exit codes: 0 ok / MATCH, 2 usage or input error, 3 MISMATCH or invalid plan, 4 locks not comparable.
"""
from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

BONN_SCHEMA = 2
CHBMIT_SCHEMA = 1
EXIT_OK, EXIT_USAGE, EXIT_MISMATCH, EXIT_INCOMPARABLE = 0, 2, 3, 4

# Cell roles, found by content (first matching role wins, so extra cells never shift indices).
# kind: protocol = executed and hashed; info = executed, not hashed; other = neither.
ROLES = [
    ("imports", r"^packages = \{", "protocol"),
    ("config", r"^LEARNING_RATE = ", "protocol"),
    ("provenance", r"^MODEL_PROVENANCE = \{", "info"),
    ("tasks", r"^PREVIOUS_32 = \[", "protocol"),
    ("loader", r"^dataset_root = Path\(BONN_DATA_DIR\)", "protocol"),
    ("fingerprint", r"^NOTEBOOK_CODE_SHA256 = ", "protocol"),
    ("training", r"^def run_fit\(", "protocol"),
    ("splits", r"^def plan_splits\(", "protocol"),
    ("charts", r"^completed_chart = ", "other"),
    ("download", r"^DOWNLOAD_GRAPHS_AND_TABLES = ", "other"),
    ("report", r"^CREATE_WORD_SUMMARY_REPORT = ", "other"),
    ("model", r"^class BonnClassifier\b", "other"),
]
REQUIRED_ROLES = [r for r, _, kind in ROLES if kind != "other"]
# Parameters that change how a run is organised, never what it computes.
OPERATIONAL = {"BONN_DATA_DIR", "RESULTS_ROOT", "SELECTED_IDS", "MAX_CLASSES_TO_RUN", "STOP_ON_STAGE_FAILURE",
               "PLOT_EVERY", "REUSE_COMPLETED", "CREATE_WORD_SUMMARY_REPORT", "DOWNLOAD_WORD_SUMMARY_REPORT",
               "DOWNLOAD_GRAPHS_AND_TABLES", "AGGREGATE_ONLY", "NOTEBOOK_FILE"}
PATH_PARAMETERS = {"BONN_DATA_DIR", "RESULTS_ROOT", "NOTEBOOK_FILE"}
MODEL_SPECIFIC = {"MODEL_OPTIONS"}
DEFAULT_METRICS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
                   "kappa", "mcc", "macro_specificity", "roc_auc_summary", "test_loss"]


class LockError(Exception):
    def __init__(self, message, code=EXIT_USAGE):
        super().__init__(message)
        self.code = code


# ----------------------------------------------------------------------------- hashing helpers
def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode("utf-8")).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def normalize_code(src):
    """Line endings, trailing spaces, blank and comment-only lines do not count; code does."""
    lines = src.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(l.rstrip() for l in lines if l.strip() and not l.lstrip().startswith("#"))


def code_sha(src):
    return sha(normalize_code(src))


def cell_source(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _lines(src):
    return src.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def split_parameter_lines(src):
    """Remove top-level literal assignments (the parameter lines) -> (remaining code, {name: value})."""
    lines = _lines(src)
    tree = ast.parse("\n".join(lines))
    drop, params = set(), {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError):
            continue
        found = []
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
            if isinstance(target, ast.Name):
                found.append((target.id, value))
            elif (isinstance(target, ast.Tuple) and all(isinstance(e, ast.Name) for e in target.elts)
                  and isinstance(value, tuple) and len(value) == len(target.elts)):
                found.extend((e.id, v) for e, v in zip(target.elts, value))
            else:
                found = None
                break
        if found:
            params.update(found)
            drop.update(range(node.lineno - 1, node.end_lineno))
    return "\n".join(l for i, l in enumerate(lines) if i not in drop), params


def env_import_lines(src):
    """Import statements and os.environ settings of the dependency cell (not its package list)."""
    lines = _lines(src)
    keep = []
    for node in ast.parse("\n".join(lines)).body:
        segment = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if isinstance(node, (ast.Import, ast.ImportFrom)) or "os.environ" in segment:
            keep.append(segment)
    return "\n".join(keep)


# ----------------------------------------------------------------------------- notebook layout
def classify_cells(cells):
    """Map roles to cell indices; papermill-injected cells are kept apart, unknown code cells listed."""
    layout = dict(roles={}, injected=[], parameters_tag=None, unknown=[])
    for i, cell in enumerate(cells):
        tags = (cell.get("metadata") or {}).get("tags") or []
        if "parameters" in tags and layout["parameters_tag"] is None:
            layout["parameters_tag"] = i
        if cell.get("cell_type") != "code":
            continue
        if "injected-parameters" in tags:
            layout["injected"].append(i)
            continue
        src = cell_source(cell)
        role = next((r for r, pattern, _ in ROLES if re.search(pattern, src, re.M)), None)
        if role is None:
            if normalize_code(src):
                layout["unknown"].append(i)
        elif role in layout["roles"]:
            raise LockError(f"cell role {role!r} matched cells {layout['roles'][role]} and {i}; "
                            "the notebook does not have the shared-protocol layout")
        else:
            layout["roles"][role] = i
    missing = [r for r in REQUIRED_ROLES if r not in layout["roles"]]
    if missing:
        raise LockError(f"cells not found for roles {missing}; is this one of the shared Bonn notebooks?")
    return layout


def shared_code_digests(cells, layout):
    src = {role: cell_source(cells[i]) for role, i in layout["roles"].items()}
    setup, _ = split_parameter_lines(src["config"])
    fingerprint = re.sub(r"^\s*NOTEBOOK_CODE_SHA256\s*=.*$", "", src["fingerprint"], flags=re.M)
    return {
        "env_imports": code_sha(env_import_lines(src["imports"])),
        "setup": code_sha(setup),
        "tasks": code_sha(src["tasks"]),
        "loader": code_sha(src["loader"]),
        "fingerprint": code_sha(fingerprint),
        "training": code_sha(src["training"]),
        "splits": code_sha(src["splits"]),
    }


def load_parameters(value):
    if not value:
        return None
    text = value.strip()
    if not text.startswith("{"):
        path = Path(value)
        if not path.is_file():
            raise LockError(f"--parameters is neither a JSON object nor a file: {value}")
        text = path.read_text(encoding="utf-8")
    try:
        params = json.loads(text)
    except ValueError as error:
        raise LockError(f"--parameters is not valid JSON: {error}")
    if not isinstance(params, dict):
        raise LockError("--parameters must be a JSON object {name: value}")
    return params


def _read_json(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LockError(f"cannot read {what} {path}: {error}")


def _literal_assignments(src):
    try:
        return split_parameter_lines(src)[1]
    except SyntaxError:
        return {}


# ----------------------------------------------------------------------------- Bonn lock
class _Zero(dict):
    def __missing__(self, key):
        return 0.0


@contextlib.contextmanager
def _patched_environment():
    """Headless plotting, no package installs, tolerant version lookup; restored afterwards."""
    import importlib.metadata as md
    import matplotlib
    matplotlib.use("Agg")
    real_version = md.version

    def safe_version(name):
        try:
            return real_version(name)
        except Exception:
            return "not-installed"
    md.version = safe_version
    try:
        yield
    finally:
        md.version = real_version


def _exec(code, ns, label, log):
    try:
        with contextlib.redirect_stdout(log):
            exec(compile(code, f"<{label}>", "exec"), ns)
    except Exception as error:
        tail = "\n".join(log.getvalue().strip().splitlines()[-8:])
        raise LockError(f"notebook cell {label!r} failed: {type(error).__name__}: {error}"
                        + (f"\n--- notebook output ---\n{tail}" if tail else "")) from error


def _exec_imports(src, ns, log):
    lines = _lines(src)
    for node in ast.parse("\n".join(lines)).body:
        segment = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if isinstance(node, ast.If) and "pip" in segment:
            continue  # never install packages from here
        if isinstance(node, (ast.Import, ast.ImportFrom)) and re.search(r"\b(requests|IPython)\b", segment):
            try:
                _exec(segment, ns, "imports", log)
            except LockError:
                pass  # optional in plan-only mode; stubbed below
            continue
        _exec(segment, ns, "imports", log)
    ns["display"] = lambda *a, **k: None
    ns["FileLink"] = str
    ns.setdefault("requests", None)


def _defs_only(src):
    lines = _lines(src)
    keep = [node for node in ast.parse("\n".join(lines)).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    return "\n".join("\n".join(lines[n.lineno - 1:n.end_lineno]) for n in keep)


def build_bonn_lock(notebook, data_dir=None, parameters=None):
    notebook = Path(notebook)
    cells = _read_json(notebook, "notebook").get("cells") or []
    layout = classify_cells(cells)
    roles = layout["roles"]
    notes = []

    # Execution order: role cells in notebook order, papermill parameters where papermill puts them.
    sequence = [(i, role) for role, i in roles.items() if role in REQUIRED_ROLES]
    config_params = _literal_assignments(cell_source(cells[roles["config"]]))
    injected = {}
    if parameters is not None:
        position = layout["parameters_tag"] if layout["parameters_tag"] is not None else -1
        sequence.append((position + 0.5, "cli_parameters"))
        injected.update(parameters)
        source_of_parameters = "command line"
    else:
        for i in layout["injected"]:
            sequence.append((i, f"injected:{i}"))
            injected.update(_literal_assignments(cell_source(cells[i])))
        source_of_parameters = "notebook" if layout["injected"] else "none"
    if injected and layout["parameters_tag"] is None and parameters is not None:
        notes.append("no cell is tagged 'parameters': papermill would inject at the top and the config cell "
                     "would overwrite the injected values (modelled here)")
    elif layout["parameters_tag"] is not None and layout["parameters_tag"] != roles["config"]:
        notes.append(f"the 'parameters' tag is on cell {layout['parameters_tag']}, not on the config cell "
                     f"{roles['config']} (papermill injects after the tagged cell; modelled here)")
    sequence.sort()

    ns = {"__name__": "__protocol_lock__"}
    log = io.StringIO()
    tmp = tempfile.mkdtemp(prefix="plock_")
    fits, scope = [], None
    try:
        with _patched_environment():
            for _, role in sequence:
                if role == "cli_parameters":
                    ns.update(parameters)
                    continue
                if role.startswith("injected:"):
                    _exec(cell_source(cells[int(role.split(":")[1])]), ns, role, log)
                    continue
                src = cell_source(cells[roles[role]])
                if role == "imports":
                    _exec_imports(src, ns, log)
                elif role == "config":
                    _exec(re.sub(r"drive\.mount\([^)]*\)", "pass", src), ns, role, log)
                elif role == "tasks":
                    _exec(src, ns, "tasks (run scope)", log)  # validates SELECTED_IDS / MAX_CLASSES_TO_RUN
                    scope = list(ns["EXPERIMENTS"])
                    knobs = {k: ns[k] for k in ("SELECTED_IDS", "MAX_CLASSES_TO_RUN")}
                    ns.update(SELECTED_IDS=[], MAX_CLASSES_TO_RUN=None)
                    _exec(src, ns, "tasks (all preset tasks)", log)
                    ns.update(knobs)
                elif role == "loader":
                    ns["BONN_DATA_DIR"] = str(data_dir) if data_dir else ns["BONN_DATA_DIR"]
                    _exec(src, ns, role, log)
                elif role == "fingerprint":
                    ns["RESULTS_ROOT"] = tmp
                    _exec(src, ns, role, log)
                elif role == "splits":
                    _exec(_defs_only(src), ns, "splits (definitions only)", log)
                else:
                    _exec(src, ns, role, log)
            fits = _plan_all_fits(ns, cell_source(cells[roles["splits"]]), log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return _assemble_bonn_lock(ns, cells, layout, fits, scope, config_params, injected,
                               source_of_parameters, notebook.name, notes)


def _plan_all_fits(ns, splits_src, log):
    """Call the notebook's own run_combination for every preset task with a plan-only run_fit."""
    metadata = ns["metadata"]
    sig = metadata["signal_sha256"].to_numpy()
    listed = re.search(r"for metric in \[([^\]]*)\]", splits_src)  # summary metrics read by run_combination
    names = [x or y for x, y in re.findall(r"\"(\w+)\"|'(\w+)'", listed.group(1))] if listed else []
    zero = _Zero({m: 0.0 for m in set(DEFAULT_METRICS) | set(names)})
    fits = []

    def plan_only_run_fit(combo_id, groups, global_idx, labels, train_idx, val_idx, test_idx, folder, fit_seed):
        parts = {}
        for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            parts[name] = [f"{s}:{int(t)}" for s, t in zip(sig[global_idx[idx]], labels[idx])]
        fits.append(dict(task=combo_id, fit=Path(folder).name, seed=int(fit_seed), **parts))
        frame = metadata.iloc[global_idx].iloc[test_idx].reset_index(drop=True).copy()
        frame["target"] = labels[test_idx]
        for i in range(len(groups)):
            frame[f"prob_{i}"] = 1.0 / len(groups)
        return _Zero(zero), frame

    def evaluation_stub(truth, probs, names, folder, title):
        Path(folder).mkdir(parents=True, exist_ok=True)
        return _Zero(zero)

    ns.update(run_fit=plan_only_run_fit, evaluation_artifacts=evaluation_stub, aggregate_curves=lambda d: None)
    for task, groups in ns["EXPERIMENTS"].items():
        try:
            with contextlib.redirect_stdout(log):
                ns["run_combination"](task, groups)  # notebook's own index construction and OOF assertions
        except Exception as error:
            raise LockError(f"split planning failed for task {task}: {type(error).__name__}: {error}") from error
    expected = int(ns.get("N_FOLDS", 10)) + 1
    per_task = collections.Counter(f["task"] for f in fits)
    short = sorted(t for t in ns["EXPERIMENTS"] if per_task[t] != expected)
    if short:
        raise LockError(f"tasks without {expected} planned fits: {short}")
    return fits


def _assemble_bonn_lock(ns, cells, layout, fits, scope, config_params, injected, source_of_parameters,
                        notebook_name, notes):
    metadata = ns["metadata"]
    roles = layout["roles"]
    shared_config = {k: v for k, v in ns["CONFIG"].items() if k not in ("model", "model_provenance")}
    protocol_params = {name: ns.get(name) for name in sorted(config_params)
                       if name not in OPERATIONAL | MODEL_SPECIFIC}
    other_injected = {k: v for k, v in injected.items()
                      if k not in OPERATIONAL | MODEL_SPECIFIC and k not in config_params}
    extra_cells = [dict(cell=i, sha256=code_sha(cell_source(cells[i])),
                        first_line=normalize_code(cell_source(cells[i])).split("\n")[0][:80])
                   for i in layout["unknown"]]
    protocol = dict(
        shared_config=shared_config,
        preset=ns["PRESET"],
        tasks=[[task, [list(g) for g in groups]] for task, groups in ns["EXPERIMENTS"].items()],
        raw_data_handling=ns["RAW_DATA_HANDLING"],
        parameters=protocol_params,
        shared_code=shared_code_digests(cells, layout),
        extra_code_cells=[c["sha256"] for c in extra_cells],
        other_injected_parameters=other_injected,
    )

    fit_hash, per_task = {}, {}
    for f in fits:
        fit_hash.setdefault(f["task"], {})[f["fit"]] = sha(canon(f))
    for task in fit_hash:
        per_task[task] = sha(canon([[name, h] for name, h in fit_hash[task].items()]))

    rows = list(zip(metadata["record_id"], metadata["set"], metadata["signal_sha256"], metadata["file_sha256"]))
    data = dict(
        data_order_hash=sha("\n".join(f"{s}:{g}" for _, s, g, _ in rows)),
        signal_set_hash=sha("\n".join(sorted(f"{s}:{g}" for _, s, g, _ in rows))),
        file_order_hash=sha("\n".join(f"{s}:{fh}" for _, s, _, fh in rows)),
        record_order_hash=sha("\n".join(r for r, _, _, _ in rows)),
        notebook_data_hash=ns.get("data_hash"),
        n_recordings=len(rows),
        target_length=ns.get("TARGET_LENGTH"),
    )
    operational = {k.lower(): ns.get(k) for k in sorted(OPERATIONAL - PATH_PARAMETERS) if k in ns}
    injected_view = {k: ("<path>" if k in PATH_PARAMETERS else v) for k, v in injected.items()}
    lock = dict(
        lock_kind="bonn",
        lock_schema=BONN_SCHEMA,
        protocol_hash=sha(canon(protocol)),
        data_order_hash=data["data_order_hash"],
        split_plan_hash=sha(canon([[t, h] for t, h in per_task.items()])),
        n_tasks=len(per_task),
        n_fits=len(fits),
        n_recordings=len(rows),
        split_hash_per_task=per_task,
        split_hash_per_fit={t: {name: h[:16] for name, h in v.items()} for t, v in fit_hash.items()},
        protocol=protocol,
        data=data,
        run_scope=dict(tasks=scope, n_tasks=len(scope), n_fits=len(scope) * (int(ns.get("N_FOLDS", 10)) + 1)),
        operational=operational,
        info_only=dict(
            model=ns.get("MODEL_NAME"),
            model_options=ns.get("MODEL_OPTIONS"),
            model_provenance=ns.get("MODEL_PROVENANCE"),
            study_id=ns.get("STUDY_ID"),
            notebook_code_sha256_constant=ns.get("NOTEBOOK_CODE_SHA256"),
            versions=ns.get("VERSIONS"),
            notebook=notebook_name,
            cells={role: i for role, i in sorted(roles.items(), key=lambda kv: kv[1])},
            other_cells={role: code_sha(cell_source(cells[i]))[:16] for role, i in roles.items()
                         if role not in REQUIRED_ROLES},
            extra_code_cells=extra_cells,
            papermill=dict(parameters_tag_cell=layout["parameters_tag"], injected_cells=layout["injected"],
                           parameters_source=source_of_parameters, injected=injected_view),
            notes=notes,
            tool_sha256=_tool_sha(),
        ),
        recordings=[[r, s, g[:16], fh[:16]] for r, s, g, fh in rows],
    )
    return lock


def _tool_sha():
    try:
        return code_sha(Path(__file__).read_text(encoding="utf-8"))[:16]
    except OSError:
        return None


# ----------------------------------------------------------------------------- CHB-MIT lock
PLAN_FORMAT = "chbmit-lopo-plan/1"
PLAN_KEYS = ("fold", "unit", "test_unit", "test_subject", "val_subject", "excluded_units", "seed")


def natural_key(value):
    """chb2 < chb10 < chb17 < chb17a; type-tagged so digits never compare with letters."""
    return tuple((0, int(t)) if t.isdigit() else (1, t) for t in re.split(r"(\d+)", str(value)) if t)


def plan_sha256_of(body):
    """The plan's own content hash: row-order independent, over units, seeds and record ids per partition."""
    if body.get("format") == PLAN_FORMAT:
        canonical = [dict({k: f[k] for k in PLAN_KEYS}, excluded_units=list(f["excluded_units"]),
                          train=sorted(f["train"]), val=sorted(f["val"]), test=sorted(f["test"])) for f in body["folds"]]
        return sha(json.dumps(canonical, sort_keys=True, separators=(",", ":")))
    return sha(json.dumps(body["folds"], sort_keys=True))  # earlier prototype format


def load_chbmit_table(npz_path, meta_path):
    """Rows of the NPZ with metadata; signal hashes are recomputed from the float32 arrays."""
    import numpy as np
    import pandas as pd
    meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
    with np.load(npz_path, allow_pickle=False) as f:
        if "X" not in f:
            raise LockError(f"{npz_path}: no array 'X'")
        X = f["X"]
        y = f["y"] if "y" in f else None
    if len(X) != len(meta):
        raise LockError(f"NPZ has {len(X)} rows but the metadata has {len(meta)}")
    signals = [hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest() for x in X]
    if "signal_sha256" in meta and list(meta["signal_sha256"].str.lower()) != signals:
        bad = next(i for i, (a, b) in enumerate(zip(meta["signal_sha256"].str.lower(), signals)) if a != b)
        raise LockError(f"metadata signal_sha256 does not match NPZ row {bad}: the files do not belong together")
    if "label" not in meta or "subject" not in meta:
        raise LockError("metadata needs 'label' and 'subject' columns")
    labels = meta["label"].astype(int).to_numpy()
    if y is not None and not np.array_equal(np.asarray(y).astype(int).reshape(-1), labels):
        raise LockError("NPZ y and metadata label disagree")
    return dict(signal=signals, label=labels.tolist(), subject=meta["subject"].tolist(),
                patient=(meta["patient"] if "patient" in meta else meta["subject"]).tolist(),
                columns=list(meta.columns), meta=meta,
                median_abs_value=float(np.median(np.abs(X[:: max(1, len(X) // 64)]))))


def decided_rule_folds(subject, patient, label, unit="subject", min_val_per_class=5, seed=42):
    """The decided LOPO rule, as row indices (an independent implementation used for checking)."""
    n = len(subject)
    subjects = sorted(set(subject), key=natural_key)
    counts = {s: [0, 0] for s in subjects}
    for s, y in zip(subject, label):
        counts[s][int(y)] += 1
    eligible = {s for s in subjects if min(counts[s]) >= max(1, min_val_per_class)}
    unit_of = subject if unit == "subject" else patient
    folds = []
    for k, u in enumerate(sorted(set(unit_of), key=natural_key), 1):
        test = [i for i in range(n) if unit_of[i] == u]
        test_subject = subject[test[0]]
        excluded = sorted({patient[i] for i in range(n) if subject[i] == test_subject and patient[i] != u},
                          key=natural_key) if unit == "case" else []
        start = subjects.index(test_subject)
        ring = [subjects[(start + j) % len(subjects)] for j in range(1, len(subjects))]
        val_subject = next((s for s in ring if s in eligible), None)
        if val_subject is None:
            raise LockError(f"no eligible validation subject for fold {u}")
        folds.append(dict(fold=k, test_units=[u], test_subject=test_subject, val_subject=val_subject,
                          excluded_units=excluded, seed=seed + k,
                          train=[i for i in range(n) if subject[i] not in (test_subject, val_subject)],
                          val=[i for i in range(n) if subject[i] == val_subject], test=test))
    return folds


def _plan_params(body):
    params = body.get("params") or {}
    get = lambda *keys, default=None: next((params[k] for k in keys if k in params), default)  # noqa: E731
    return dict(unit=get("lopo_unit", "test_unit", "LOPO_UNIT", default="subject"),
                min_val_per_class=int(get("min_val_per_class", "MIN_VAL_PER_CLASS", default=5)),
                seed=int(get("seed", "SEED", default=42)),
                id_column=body.get("record_id_column") or get("id_column"))


def _fold_part(fold, *keys):
    for key in keys:
        if key in fold:
            return fold[key]
    raise LockError(f"plan fold {fold.get('fold')} has none of the keys {keys}")


def _pick_id_column(table, plan_ids, preferred):
    candidates = [preferred] if preferred else [c for c in ("record_id", "segment_id") if c in table["columns"]]
    for column in candidates:
        if column in table["columns"]:
            ids = table["meta"][column].tolist()
            if len(set(ids)) == len(ids) and plan_ids <= set(ids):
                return column
    raise LockError(f"no unique metadata id column contains every plan id (tried {candidates}); "
                    "is this the dataset the plan was built on? (--id-col selects the column)")


def build_chbmit_lock(plan_path, npz=None, meta=None, id_col=None):
    body = _read_json(plan_path, "plan")
    if not isinstance(body, dict) or not isinstance(body.get("folds"), list) or not body["folds"]:
        raise LockError("a LOPO plan needs a non-empty 'folds' list")
    params = _plan_params(body)
    folds = sorted(body["folds"], key=lambda f: int(f.get("fold", 0)))
    parts = [{k: [str(r) for r in _fold_part(f, *keys)] for k, keys in
              (("train", ("train",)), ("val", ("val", "validation")), ("test", ("test",)))} for f in folds]
    heads = [dict(fold=int(f.get("fold", k)), test_units=[str(u) for u in (f.get("test_units") or [f.get("test_unit")])],
                  val_subject=str(f.get("val_subject", f.get("validation_subject", ""))),
                  excluded_units=[str(u) for u in (f.get("excluded_units") or [])], seed=int(_fold_part(f, "seed")))
             for k, f in enumerate(folds, 1)]
    try:
        plan_sha = plan_sha256_of(body)
    except (KeyError, TypeError) as error:
        raise LockError(f"plan folds lack a field needed for plan_sha256: {error}")
    by_record = [dict(h, **{f"n_{p}": len(v[p]) for p in v}, **{f"{p}_ids": sha(canon(sorted(v[p]))) for p in v})
                 for h, v in zip(heads, parts)]
    lock = dict(
        lock_kind="chbmit-lopo",
        lock_schema=CHBMIT_SCHEMA,
        plan_sha256=plan_sha,
        plan_sha256_stored=body.get("plan_sha256"),
        fold_table_by_record_hash=sha(canon(by_record)),
        n_folds=len(folds),
        params=params,
        keyed_on="record_id",
        data=None,
        fold_table_hash=None,
        fold_table=None,
        fold_table_by_record=by_record,
        rule_check=None,
        info_only=dict(plan=Path(plan_path).name, plan_format=body.get("format"),
                       fold_names=[f.get("name") for f in folds], plan_params=body.get("params"),
                       tool_sha256=_tool_sha()),
    )
    if lock["plan_sha256_stored"] and lock["plan_sha256_stored"] != plan_sha:
        lock["info_only"]["note"] = "the stored plan_sha256 does not match the fold contents: the plan file was edited"
    if not npz:
        return lock

    table = load_chbmit_table(npz, meta)
    column = _pick_id_column(table, {r for v in parts for p in v.values() for r in p}, id_col or params["id_column"])
    pos = {r: i for i, r in enumerate(table["meta"][column])}
    sig, lab, subj, pat = table["signal"], table["label"], table["subject"], table["patient"]
    rows_of = [{p: [pos[r] for r in v[p]] for p in v} for v in parts]

    signal_rows = []
    for h, v in zip(heads, rows_of):
        row = dict(h)
        for p, idx in v.items():
            row[f"n_{p}"] = len(idx)
            row[f"seizure_{p}"] = sum(lab[i] for i in idx)
            row[f"{p}_signals"] = sha(canon(sorted(sig[i] for i in idx)))
        signal_rows.append(row)
    lock.update(
        keyed_on="signal_sha256",
        fold_table_hash=sha(canon(signal_rows)),
        fold_table=signal_rows,
        data=dict(
            data_signal_hash=sha("\n".join(sorted(f"{s}:{y}" for s, y in zip(sig, lab)))),
            subject_map_hash=sha("\n".join(sorted(f"{s}:{a}:{b}" for s, a, b in zip(sig, subj, pat)))),
            data_row_order_hash=sha("\n".join(f"{s}:{y}" for s, y in zip(sig, lab))),
            n_segments=len(sig), n_seizure=int(sum(lab)), n_nonseizure=int(len(lab) - sum(lab)),
            n_subjects=len(set(subj)), n_patients=len(set(pat)), id_column=column,
            median_abs_value=table["median_abs_value"],
        ),
        rule_check=_check_plan(heads, rows_of, lab, subj, pat, params),
    )
    return lock


def _check_plan(heads, rows_of, lab, subj, pat, params):
    """Leakage and usability invariants, then agreement with the decided rule."""
    problems, n = [], len(lab)
    unit_of = subj if params["unit"] == "subject" else pat
    tested = []
    for h, v in zip(heads, rows_of):
        name = f"fold {h['fold']} ({','.join(h['test_units'])})"
        tr, va, te = (set(v[p]) for p in ("train", "val", "test"))
        if any(len(set(v[p])) != len(v[p]) for p in v):
            problems.append(f"{name}: duplicate segments inside a partition")
        if tr & va or tr & te or va & te:
            problems.append(f"{name}: partitions overlap at segment level")
        s_tr, s_va, s_te = ({subj[i] for i in part} for part in (tr, va, te))
        if s_te & (s_tr | s_va) or s_va & s_tr:
            problems.append(f"{name}: a subject appears in more than one partition")
        if s_va != {h["val_subject"]} or len(va) != sum(1 for s in subj if s == h["val_subject"]):
            problems.append(f"{name}: validation is not exactly subject {h['val_subject']}")
        val_counts = [sum(1 for i in va if lab[i] == c) for c in (0, 1)]
        if min(val_counts) < max(1, params["min_val_per_class"]):
            problems.append(f"{name}: validation has {val_counts} segments per class")
        if len({lab[i] for i in tr}) < 2:
            problems.append(f"{name}: training lacks a class")
        test_rows = {i for i in range(n) if unit_of[i] in h["test_units"]}
        if te != test_rows:
            problems.append(f"{name}: test partition is not exactly the rows of {h['test_units']}")
        elif s_te:
            left_out = set(range(n)) - tr - va - te
            siblings = {i for i in range(n) if subj[i] in s_te} - te
            if left_out != siblings:
                problems.append(f"{name}: {len(left_out)} segments left out; only the test subject's other "
                                f"recording folders ({len(siblings)}) may be")
        tested.extend(te)
    coverage = "full" if sorted(tested) == list(range(n)) else f"partial ({len(set(tested))} of {n} segments tested)"
    if len(tested) != len(set(tested)):
        problems.append("a segment is tested in more than one fold")

    reference = {f["fold"]: f for f in decided_rule_folds(subj, pat, lab, params["unit"],
                                                            params["min_val_per_class"], params["seed"])}
    difference = None
    for h, v in zip(heads, rows_of):
        ref = reference.get(h["fold"])
        if ref is None:
            difference = f"fold {h['fold']} does not exist under the decided rule"
        else:
            differing = [k for k in ("test_units", "val_subject", "excluded_units", "seed") if h[k] != ref[k]]
            differing += [p for p in ("train", "val", "test") if sorted(v[p]) != ref[p]]
            if differing:
                k = differing[0]
                difference = (f"fold {h['fold']}: {k} {h[k]} (decided rule: {ref[k]})" if k in h
                              else f"fold {h['fold']}: {k} membership differs from the decided rule")
        if difference:
            break
    if difference is None and len(heads) != len(reference):
        difference = f"plan has {len(heads)} folds, the decided rule {len(reference)}"
    return dict(invariants_ok=not problems, problems=problems[:20], coverage=coverage,
                matches_decided_rule=difference is None, first_difference=difference)


def write_reference_plan(out, npz, meta, unit="subject", min_val_per_class=5, seed=42, id_col=None):
    """The decided-rule plan in the shared plan format (replayable through LOPO_PLAN_PATH)."""
    table = load_chbmit_table(npz, meta)
    column = id_col or next((c for c in ("record_id", "segment_id") if c in table["columns"]), None)
    if column is None or column not in table["columns"]:
        raise LockError("metadata has no record_id or segment_id column; use --id-col")
    ids = table["meta"][column].tolist()
    if len(set(ids)) != len(ids):
        raise LockError(f"metadata column {column} is not unique")
    lab = table["label"]
    folds = []
    for f in decided_rule_folds(table["subject"], table["patient"], lab, unit, min_val_per_class, seed):
        entry = dict(fold=f["fold"], name=f"fold_{f['test_units'][0]}", unit=unit, test_unit=f["test_units"][0],
                     test_subject=f["test_subject"], val_subject=f["val_subject"], excluded_units=f["excluded_units"],
                     seed=f["seed"])
        entry.update({p: sorted(ids[i] for i in f[p]) for p in ("train", "val", "test")})
        entry["counts"] = {p: dict(nonseizure=sum(1 for i in f[p] if lab[i] == 0),
                                   seizure=sum(1 for i in f[p] if lab[i] == 1)) for p in ("train", "val", "test")}
        folds.append(entry)
    body = dict(format=PLAN_FORMAT, record_id_column=column,
                params=dict(lopo_unit=unit, min_val_per_class=min_val_per_class, seed=seed,
                            validation_rule="first subject after the test subject in cyclic natural id order with "
                                            ">= min_val_per_class segments of each class",
                            fit_seed_rule="seed + fold (fold numbered from 1)"),
                n_records=len(ids), n_folds=len(folds), plan_sha256=None, folds=folds)
    body["plan_sha256"] = plan_sha256_of(body)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    return body


# ----------------------------------------------------------------------------- comparison
def compare_locks(mine, other):
    """-> (status, lines); status is MATCH, MISMATCH or INCOMPARABLE."""
    kinds = (mine.get("lock_kind", "bonn?"), other.get("lock_kind", "bonn?"))
    schemas = (mine.get("lock_schema"), other.get("lock_schema"))
    if kinds[0] != kinds[1] or schemas[0] != schemas[1]:
        return "INCOMPARABLE", [f"cannot compare a {kinds[0]} lock (schema {schemas[0]}) with a {kinds[1]} lock "
                                f"(schema {schemas[1]}); re-run both sides with the same protocol_lock.py"]
    lines = []
    tools = (mine.get("info_only", {}).get("tool_sha256"), other.get("info_only", {}).get("tool_sha256"))
    if None not in tools and tools[0] != tools[1]:
        lines.append(f"warning: the locks were written by different protocol_lock.py versions ({tools[0]} vs {tools[1]})")
    if kinds[0] == "bonn":
        return _compare_bonn(mine, other, lines)
    return _compare_chbmit(mine, other, lines)


def _dict_diff(a, b, prefix):
    out = []
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            out.append(f"  {prefix}{key}: {a.get(key)!r} vs {b.get(key)!r}")
    return out


def _compare_bonn(a, b, lines):
    keys = ("protocol_hash", "data_order_hash", "split_plan_hash")
    bad = [k for k in keys if a[k] != b[k]]
    pa, pb = a["protocol"], b["protocol"]
    for k in keys:
        lines.append(f"  {'MATCH   ' if k not in bad else 'MISMATCH'} {k}")
    if "protocol_hash" in bad:
        lines.append("protocol differences (this lock vs expected):")
        lines += _dict_diff(pa["shared_config"], pb["shared_config"], "config.")
        lines += _dict_diff(pa["parameters"], pb["parameters"], "parameter ")
        lines += _dict_diff(pa["raw_data_handling"], pb["raw_data_handling"], "raw_data_handling.")
        lines += _dict_diff(pa["other_injected_parameters"], pb["other_injected_parameters"], "injected ")
        for role in sorted(set(pa["shared_code"]) | set(pb["shared_code"])):
            if pa["shared_code"].get(role) != pb["shared_code"].get(role):
                lines.append(f"  shared code differs: {role} cell")
        if pa["preset"] != pb["preset"]:
            lines.append(f"  preset: {pa['preset']!r} vs {pb['preset']!r}")
        ta, tb = [t for t, _ in pa["tasks"]], [t for t, _ in pb["tasks"]]
        if pa["tasks"] != pb["tasks"]:
            only_a, only_b = [t for t in ta if t not in tb], [t for t in tb if t not in ta]
            lines.append(f"  task list differs (only here: {only_a[:5]}, only there: {only_b[:5]}"
                         + (", same tasks in another order or grouping" if not only_a and not only_b else "") + ")")
        if pa["extra_code_cells"] != pb["extra_code_cells"]:
            extra = a.get("info_only", {}).get("extra_code_cells", [])
            lines.append(f"  extra code cells: {len(pa['extra_code_cells'])} here vs {len(pb['extra_code_cells'])} there"
                         + "".join(f"\n    cell {c['cell']}: {c['first_line']}" for c in extra[:5]))
    da, db = a["data"], b["data"]
    if "data_order_hash" in bad:
        if da["signal_set_hash"] == db["signal_set_hash"]:
            lines.append("  data: same recordings at signal level but a different load order (file naming?); "
                         "fold membership therefore differs")
        else:
            ra = {(s, g) for _, s, g, _ in a.get("recordings", [])}
            rb = {(s, g) for _, s, g, _ in b.get("recordings", [])}
            per_set = {}
            for s, _ in ra - rb:
                per_set[s] = per_set.get(s, 0) + 1
            lines.append(f"  data: {len(ra - rb)} recording signals here are not in the expected lock"
                         + (f" (per set: {per_set})" if per_set else ""))
    else:
        if da["file_order_hash"] != db["file_order_hash"]:
            lines.append("  note: file bytes differ but signals are identical (line endings, whitespace or number "
                         "format); the data are the same")
        if da["record_order_hash"] != db["record_order_hash"]:
            lines.append("  note: file names or folder nesting differ; irrelevant because the signal order matches")
        if da.get("notebook_data_hash") != db.get("notebook_data_hash"):
            lines.append("  note: the notebooks' own data_hash differs (naming/bytes), so STUDY_IDs differ; expected")
    sa, sb = a["split_hash_per_task"], b["split_hash_per_task"]
    common = [t for t in sa if t in sb]
    differing = [t for t in common if sa[t] != sb[t]]
    lines.append(f"  splits: {len(common) - len(differing)}/{len(common)} common tasks identical"
                 + (f"; differing: {differing[:6]}{' ...' if len(differing) > 6 else ''}" if differing else "")
                 + (f"; {len(set(sa) ^ set(sb))} task(s) in only one lock" if set(sa) ^ set(sb) else ""))
    if differing:
        fa, fb = a["split_hash_per_fit"][differing[0]], b["split_hash_per_fit"][differing[0]]
        lines.append(f"    {differing[0]}: fits differing {[k for k in fa if fa.get(k) != fb.get(k)]}")
    scope_a, scope_b = a["run_scope"], b["run_scope"]
    lines.append(f"  run scope (not compared): {scope_a['n_tasks']} task(s) here, {scope_b['n_tasks']} there")
    ia, ib = a.get("info_only", {}), b.get("info_only", {})
    lines.append(f"  models (not compared): {ia.get('model')} vs {ib.get('model')}")
    va, vb = ia.get("versions") or {}, ib.get("versions") or {}
    vdiff = {k: (va.get(k), vb.get(k)) for k in ("scikit-learn", "numpy", "torch", "python") if va.get(k) != vb.get(k)}
    if vdiff:
        lines.append(f"  environment (not compared): {vdiff}")
    return ("MISMATCH" if bad else "MATCH"), lines


def _compare_chbmit(a, b, lines):
    signal_level = a.get("data") is not None and b.get("data") is not None
    if signal_level:
        keys = [("data_signal_hash", a["data"]["data_signal_hash"], b["data"]["data_signal_hash"]),
                ("subject_map_hash", a["data"]["subject_map_hash"], b["data"]["subject_map_hash"]),
                ("fold_table_hash", a["fold_table_hash"], b["fold_table_hash"])]
    else:
        keys = [("plan_sha256", a["plan_sha256"], b["plan_sha256"]),
                ("fold_table_by_record_hash", a["fold_table_by_record_hash"], b["fold_table_by_record_hash"])]
        lines.append("  note: at least one lock was made without --npz/--meta; comparing by record id only")
    bad = [k for k, x, y in keys if x != y]
    for k, _, _ in keys:
        lines.append(f"  {'MATCH   ' if k not in bad else 'MISMATCH'} {k}")
    if signal_level and a["plan_sha256"] != b["plan_sha256"]:
        lines.append("  note: plan_sha256 differs (record ids or fold names differ) but the comparison is on signals")
    if signal_level and a["data"]["data_row_order_hash"] != b["data"]["data_row_order_hash"]:
        lines.append("  note: segment row order differs between the two datasets (same set of segments)"
                     if "data_signal_hash" not in bad else "  data: the segment sets differ")
    table = "fold_table" if signal_level else "fold_table_by_record"
    rows_a = {r["fold"]: r for r in a[table]}
    rows_b = {r["fold"]: r for r in b[table]}
    shown = 0
    for fold in sorted(set(rows_a) | set(rows_b)):
        ra, rb = rows_a.get(fold), rows_b.get(fold)
        if ra != rb and shown < 5:
            shown += 1
            if ra is None or rb is None:
                lines.append(f"    fold {fold}: present in only one plan")
            else:
                lines.append(f"    fold {fold}: differs in {[k for k in ra if ra.get(k) != rb.get(k)]}")
    for side, lock in (("this", a), ("expected", b)):
        rc = lock.get("rule_check")
        if rc and (not rc["invariants_ok"] or not rc["matches_decided_rule"]):
            lines.append(f"  {side} plan: invariants_ok={rc['invariants_ok']} "
                         f"matches_decided_rule={rc['matches_decided_rule']} {rc['first_difference'] or ''}")
    return ("MISMATCH" if bad else "MATCH"), lines


# ----------------------------------------------------------------------------- CLI
def _summary_bonn(lock):
    info = lock["info_only"]
    print(f"{info['model']} ({info['notebook']})")
    print(f"  protocol_hash   {lock['protocol_hash'][:16]}  ({lock['n_tasks']} preset tasks, {lock['n_fits']} fits "
          f"planned; run scope {lock['run_scope']['n_tasks']} task(s))")
    print(f"  data_order_hash {lock['data_order_hash'][:16]}  ({lock['n_recordings']} recordings, signal level)")
    print(f"  split_plan_hash {lock['split_plan_hash'][:16]}")
    for note in info["notes"]:
        print(f"  note: {note}")
    for cell in info["extra_code_cells"]:
        print(f"  warning: unrecognised code cell {cell['cell']} is part of the protocol hash: {cell['first_line']}")


def _summary_chbmit(lock):
    print(f"CHB-MIT LOPO plan ({lock['info_only']['plan']}): {lock['n_folds']} folds, keyed on {lock['keyed_on']}")
    print(f"  plan_sha256     {lock['plan_sha256'][:16]}")
    if lock["data"]:
        print(f"  data_signal     {lock['data']['data_signal_hash'][:16]}  ({lock['data']['n_segments']} segments, "
              f"{lock['data']['n_subjects']} subjects)")
        print(f"  fold_table_hash {lock['fold_table_hash'][:16]}")
        rc = lock["rule_check"]
        print(f"  rule check: invariants_ok={rc['invariants_ok']} coverage={rc['coverage']} "
              f"matches_decided_rule={rc['matches_decided_rule']}")
        for problem in rc["problems"]:
            print(f"    INVALID: {problem}")
        if rc["first_difference"]:
            print(f"    differs from the decided rule: {rc['first_difference']}")
    else:
        print(f"  fold_table_by_record_hash {lock['fold_table_by_record_hash'][:16]} (no --npz/--meta: no signal check)")
    if lock["info_only"].get("note"):
        print(f"  note: {lock['info_only']['note']}")


def _report(mine, other, other_name):
    status, lines = compare_locks(mine, other)
    print(f"comparison with {other_name}:")
    for line in lines:
        print(line)
    if status == "MATCH":
        if mine.get("lock_kind") == "bonn":
            print("MATCH: same shared protocol, same recordings at signal level, identical split plans")
        elif mine.get("data") and other.get("data"):
            print("MATCH: same segments, labels and subject grouping at signal level; identical LOPO folds")
        else:
            print("MATCH: identical LOPO folds by record id (signals not checked: lock both sides with --npz/--meta)")
        return EXIT_OK
    if status == "INCOMPARABLE":
        print("NOT COMPARABLE")
        return EXIT_INCOMPARABLE
    print("MISMATCH: do not start or merge training until this is resolved")
    return EXIT_MISMATCH


def _read_lock(path):
    return _read_json(path, "lock")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", help="Bonn notebook (.ipynb); original, tagged or papermill-executed")
    ap.add_argument("--data", help="Bonn folder with Z/O/N/F/S (or A-E); default: the notebook's BONN_DATA_DIR")
    ap.add_argument("--parameters", help="papermill parameters to model: JSON object or path to a JSON file")
    ap.add_argument("--chbmit-plan", help="CHB-MIT LOPO plan JSON")
    ap.add_argument("--chbmit-reference-plan", metavar="OUT", help="write the decided-rule LOPO plan to OUT")
    ap.add_argument("--npz", help="CHB-MIT NPZ (X [n, channels, samples], y)")
    ap.add_argument("--meta", help="CHB-MIT metadata CSV, one row per NPZ row")
    ap.add_argument("--id-col", help="metadata column holding the plan's record ids (default: auto)")
    ap.add_argument("--lopo-unit", default="subject", choices=["subject", "case"])
    ap.add_argument("--min-val-per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="lock JSON to write")
    ap.add_argument("--expect", help="lock JSON from the other side; exit 3 on mismatch")
    ap.add_argument("--compare", nargs=2, metavar=("LOCK_A", "LOCK_B"), help="compare two existing locks")
    a = ap.parse_args(argv)
    modes = [m for m in ("notebook", "chbmit_plan", "chbmit_reference_plan", "compare") if getattr(a, m)]
    if len(modes) != 1:
        ap.error("choose exactly one of --notebook, --chbmit-plan, --chbmit-reference-plan, --compare")
    if bool(a.npz) != bool(a.meta):
        ap.error("--npz and --meta go together")
    try:
        if a.compare:
            return _report(_read_lock(a.compare[0]), _read_lock(a.compare[1]), Path(a.compare[1]).name)
        if a.chbmit_reference_plan:
            if not a.npz:
                ap.error("--chbmit-reference-plan needs --npz and --meta")
            body = write_reference_plan(a.chbmit_reference_plan, a.npz, a.meta, a.lopo_unit, a.min_val_per_class,
                                        a.seed, a.id_col)
            print(f"wrote {a.chbmit_reference_plan}: {len(body['folds'])} folds, plan_sha256 {body['plan_sha256'][:16]}")
            return EXIT_OK
        if not a.out:
            ap.error("--out is required")
        expected = _read_lock(a.expect) if a.expect else None
        if a.notebook:
            lock = build_bonn_lock(a.notebook, a.data, load_parameters(a.parameters))
            _summary_bonn(lock)
            invalid = False
        else:
            lock = build_chbmit_lock(a.chbmit_plan, a.npz, a.meta, a.id_col)
            _summary_chbmit(lock)
            invalid = bool(lock["rule_check"]) and not lock["rule_check"]["invariants_ok"]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(lock, indent=1, default=str), encoding="utf-8")
        print(f"  wrote {a.out}")
        code = _report(lock, expected, Path(a.expect).name) if expected is not None else EXIT_OK
        if invalid:
            print("INVALID PLAN: subject leakage or an unusable validation subject (see above)")
            return EXIT_MISMATCH
        return code
    except LockError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return error.code


if __name__ == "__main__":
    sys.exit(main())
