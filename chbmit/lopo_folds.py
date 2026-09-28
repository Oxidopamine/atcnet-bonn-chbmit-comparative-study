"""Leave-one-patient-out (LOPO) fold plan for CHB-MIT, shared by every model in the study.

Rule (deterministic, no random numbers involved):
  unit='subject' (default)  one fold per subject. Recording folders of the same person
                            (chb01/chb21, chb17a/chb17b) form one subject: 23 folds.
  unit='case'               one fold per recording folder (25 folds). The other folders of the
                            test subject are excluded from training and validation of that fold.
  Units are taken in natural id order and fold k (k = 1..K) tests unit k.
  Validation is one whole subject: the first subject after the test subject, cyclically in natural
  id order, that has at least `min_val_per_class` segments of each class.
  Training is every remaining subject. Fit seed = seed + k (the shared notebooks use SEED + fold).

The plan is exchanged as JSON (record ids = segment_id) with a `plan_sha256` over the fold
contents, so every model replays identical folds regardless of row order.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PLAN_FORMAT = "chbmit-lopo-plan/1"
UNITS = ("subject", "case")
UNIT_COLUMN = {"subject": "subject", "case": "patient"}
VALIDATION_RULE = ("validation = first subject after the test subject in cyclic natural id order with "
                   ">= min_val_per_class segments of each class; training = all other subjects; "
                   "in 'case' mode the test subject's other recording folders are excluded")


class FoldPlanError(ValueError):
    """The fold plan cannot be built, is inconsistent, or does not match the data."""


def natural_key(value):
    """chb2 < chb10 < chb17a < chb17b; type-tagged so mixed ids never compare int with str."""
    return tuple((0, int(t)) if t.isdigit() else (1, t) for t in re.split(r"(\d+)", str(value)) if t)


@dataclass(eq=False)
class Fold:
    fold: int                 # 1-based position in the plan
    name: str                 # result folder name, fold_<test unit>
    unit: str                 # 'subject' or 'case'
    test_unit: str
    test_subject: str
    val_subject: str
    excluded_units: tuple     # other recording folders of the test subject ('case' mode only)
    seed: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def as_cell18_tuple(self):
        """(name, train, val, test, fit_seed) in the shape the shared notebooks' split planner returns."""
        return self.name, self.train_idx, self.val_idx, self.test_idx, self.seed


def _columns(meta):
    missing = [c for c in ("subject", "patient", "label") if c not in meta.columns]
    if missing:
        raise FoldPlanError(f"metadata lacks columns {missing}")
    subj = meta["subject"].astype(str).to_numpy()
    pat = meta["patient"].astype(str).to_numpy()
    if meta["subject"].isna().any() or meta["patient"].isna().any() or (subj == "").any() or (pat == "").any():
        raise FoldPlanError("LOPO needs a subject and a recording-folder id for every segment")
    labels = pd.to_numeric(meta["label"], errors="coerce").to_numpy()
    if np.isnan(labels.astype(float)).any() or not set(np.unique(labels)) <= {0, 1}:
        raise FoldPlanError("labels must be 0 (non-seizure) or 1 (seizure)")
    owner = pd.DataFrame(dict(p=pat, s=subj)).drop_duplicates()
    split = owner[owner.p.duplicated(keep=False)]
    if len(split):
        raise FoldPlanError(f"recording folders assigned to several subjects: {sorted(set(split.p))}")
    return subj, pat, labels.astype(np.int64)


def record_ids(meta):
    col = "record_id" if "record_id" in meta.columns else "segment_id"
    if col not in meta.columns:
        raise FoldPlanError("metadata needs a record_id or segment_id column")
    rid = meta[col].astype(str).to_numpy()
    if len(set(rid)) != len(rid):
        raise FoldPlanError(f"{col} values are not unique")
    return rid


def subject_table(meta, min_val_per_class=5):
    """Per-subject class counts, recording folders and validation eligibility (natural order)."""
    subj, pat, labels = _columns(meta)
    rows = []
    for s in sorted(set(subj), key=natural_key):
        m = subj == s
        n0, n1 = int((labels[m] == 0).sum()), int((labels[m] == 1).sum())
        rows.append(dict(subject=s, cases=";".join(sorted(set(pat[m]), key=natural_key)), n_nonseizure=n0,
                         n_seizure=n1, eligible_for_validation=min(n0, n1) >= max(1, int(min_val_per_class))))
    return pd.DataFrame(rows)


def build_lopo_folds(meta, *, unit="subject", min_val_per_class=5, seed=42):
    """All folds of the plan, in order. meta: one row per segment (row i <-> X[i]) with subject,
    patient (recording folder) and label. Never subset here: use select_folds() so fold numbers
    and seeds stay those of the full plan."""
    if unit not in UNITS:
        raise FoldPlanError(f"unit must be one of {UNITS}, got {unit!r}")
    subj, pat, labels = _columns(meta)
    table = subject_table(meta, min_val_per_class)
    subjects = table["subject"].tolist()
    eligible = set(table.loc[table.eligible_for_validation, "subject"])
    if len(eligible) < 2:
        raise FoldPlanError(f"fewer than 2 subjects have >= {min_val_per_class} segments of each class")
    unit_values = subj if unit == "subject" else pat
    folds = []
    for k, test_unit in enumerate(sorted(set(unit_values), key=natural_key), 1):
        test_mask = unit_values == test_unit
        test_subject = subj[test_mask][0]
        start = subjects.index(test_subject)
        val_subject = next((subjects[(start + j) % len(subjects)] for j in range(1, len(subjects))
                            if subjects[(start + j) % len(subjects)] in eligible), None)
        if val_subject is None:
            raise FoldPlanError(f"no eligible validation subject for test unit {test_unit}")
        val_mask = subj == val_subject
        excluded_mask = (subj == test_subject) & ~test_mask
        train_mask = ~(test_mask | val_mask | excluded_mask)
        folds.append(Fold(fold=k, name=f"fold_{test_unit}", unit=unit, test_unit=str(test_unit),
                          test_subject=str(test_subject), val_subject=str(val_subject),
                          excluded_units=tuple(sorted(set(pat[excluded_mask]), key=natural_key)),
                          seed=int(seed) + k, train_idx=np.flatnonzero(train_mask),
                          val_idx=np.flatnonzero(val_mask), test_idx=np.flatnonzero(test_mask)))
    check_plan(folds, meta, min_val_per_class=min_val_per_class)
    return folds


def check_plan(folds, meta, *, min_val_per_class=5, complete=True):
    """Raise FoldPlanError unless every invariant holds.

    Per fold: indices valid and unique; train/validation/test disjoint at segment and subject level;
    test = exactly the rows of the test unit; validation = exactly the rows of one subject with both
    classes and >= min_val_per_class of each; training has both classes; rows left out are exactly
    the test subject's other recording folders. For a complete plan also: folds numbered 1..K with
    seed - fold constant, one fold per unit, and every segment tested exactly once."""
    subj, pat, labels = _columns(meta)
    n = len(meta)
    need = max(1, int(min_val_per_class))
    if not folds:
        raise FoldPlanError("empty fold plan")
    kinds = {f.unit for f in folds}
    if len(kinds) != 1 or not kinds <= set(UNITS):
        raise FoldPlanError(f"folds mix or use unknown units: {sorted(kinds)}")
    unit = kinds.pop()
    unit_values = subj if unit == "subject" else pat
    everything = np.arange(n)
    for f in folds:
        parts = dict(train=np.asarray(f.train_idx), val=np.asarray(f.val_idx), test=np.asarray(f.test_idx))
        for part, idx in parts.items():
            if idx.size == 0:
                raise FoldPlanError(f"{f.name}: empty {part} set")
            if not np.issubdtype(idx.dtype, np.integer) or idx.min() < 0 or idx.max() >= n:
                raise FoldPlanError(f"{f.name}: {part} indices outside 0..{n - 1}")
            if np.unique(idx).size != idx.size:
                raise FoldPlanError(f"{f.name}: repeated indices in {part}")
        tr, va, te = parts["train"], parts["val"], parts["test"]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = np.intersect1d(parts[a], parts[b])
            if shared.size:
                raise FoldPlanError(f"{f.name}: {shared.size} segments in both {a} and {b}")
        s_tr, s_va, s_te = set(subj[tr]), set(subj[va]), set(subj[te])
        for a, b, common in (("train", "test", s_tr & s_te), ("val", "test", s_va & s_te),
                             ("train", "val", s_tr & s_va)):
            if common:
                raise FoldPlanError(f"{f.name}: subjects {sorted(common)} in both {a} and {b}")
        if s_te != {f.test_subject} or s_va != {f.val_subject}:
            raise FoldPlanError(f"{f.name}: test/validation subjects {sorted(s_te)}/{sorted(s_va)} do not match "
                                f"{f.test_subject}/{f.val_subject}")
        if not np.array_equal(np.sort(te), np.flatnonzero(unit_values == f.test_unit)):
            raise FoldPlanError(f"{f.name}: test set is not exactly the rows of {unit} {f.test_unit}")
        if not np.array_equal(np.sort(va), np.flatnonzero(subj == f.val_subject)):
            raise FoldPlanError(f"{f.name}: validation set is not exactly subject {f.val_subject}")
        left_out = np.setdiff1d(everything, np.concatenate([tr, va, te]))
        siblings = np.flatnonzero((subj == f.test_subject) & (unit_values != f.test_unit))
        if not np.array_equal(left_out, siblings):
            raise FoldPlanError(f"{f.name}: rows outside train/val/test must be exactly the test subject's "
                                f"other recording folders ({left_out.size} vs {siblings.size})")
        n0, n1 = int((labels[va] == 0).sum()), int((labels[va] == 1).sum())
        if min(n0, n1) < need:
            raise FoldPlanError(f"{f.name}: validation subject {f.val_subject} has {n0}/{n1} segments per class; "
                                f"needs >= {need} of each")
        if np.unique(labels[tr]).size < 2:
            raise FoldPlanError(f"{f.name}: training set lacks a class")
    if complete:
        if [f.fold for f in folds] != list(range(1, len(folds) + 1)):
            raise FoldPlanError("folds must be numbered 1..K in plan order")
        if len({f.seed - f.fold for f in folds}) != 1:
            raise FoldPlanError("fit seeds must follow seed + fold")
        tested_units = [f.test_unit for f in folds]
        if sorted(tested_units, key=natural_key) != sorted(set(unit_values), key=natural_key):
            raise FoldPlanError(f"a complete plan tests every {unit} exactly once")
        tested = np.sort(np.concatenate([np.asarray(f.test_idx) for f in folds]))
        if not np.array_equal(tested, everything):
            raise FoldPlanError("every segment must be tested exactly once")
    return True


def select_folds(folds, units=None):
    """Folds whose test unit is in `units` (None or empty = all), keeping plan order, numbers and seeds."""
    if not units:
        return list(folds)
    units = [str(u) for u in units]
    if len(set(units)) != len(units):
        raise FoldPlanError(f"repeated test units in selection: {units}")
    known = {f.test_unit for f in folds}
    unknown = [u for u in units if u not in known]
    if unknown:
        raise FoldPlanError(f"unknown test units {unknown}; the plan has {sorted(known, key=natural_key)}")
    return [f for f in folds if f.test_unit in set(units)]


def manifest(folds, meta):
    """Human-readable fold table: units, class counts per partition and fit seed."""
    subj, pat, labels = _columns(meta)
    rows = []
    for f in folds:
        row = dict(fold=f.fold, name=f.name, lopo_unit=f.unit, test_unit=f.test_unit, test_subject=f.test_subject,
                   validation_unit=f.val_subject,
                   validation_cases=";".join(sorted(set(pat[f.val_idx]), key=natural_key)),
                   excluded_units=";".join(f.excluded_units), fit_seed=f.seed)
        for part, idx in (("train", f.train_idx), ("val", f.val_idx), ("test", f.test_idx)):
            row[f"n_{part}"] = int(len(idx))
            row[f"n_{part}_nonseizure"] = int((labels[idx] == 0).sum())
            row[f"n_{part}_seizure"] = int((labels[idx] == 1).sum())
        row["n_excluded"] = len(meta) - row["n_train"] - row["n_val"] - row["n_test"]
        row["n_train_subjects"] = len(set(subj[f.train_idx]))
        row["test_single_class"] = bool(min(row["n_test_nonseizure"], row["n_test_seizure"]) == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_fingerprints(meta):
    """Order-independent hashes of the segment table (ids, groups, labels) and of the signals."""
    rid = record_ids(meta)
    subj, pat, labels = _columns(meta)
    lines = sorted(f"{r}\t{s}\t{p}\t{y}" for r, s, p, y in zip(rid, subj, pat, labels))
    out = dict(records_sha256=hashlib.sha256("\n".join(lines).encode()).hexdigest())
    if "signal_sha256" in meta.columns:
        sig = sorted(f"{r}\t{h}" for r, h in zip(rid, meta["signal_sha256"].astype(str).str.lower()))
        out["signals_sha256"] = hashlib.sha256("\n".join(sig).encode()).hexdigest()
    return out


def _canonical(fold_entries):
    keys = ("fold", "unit", "test_unit", "test_subject", "val_subject", "excluded_units", "seed")
    canon = [dict({k: e[k] for k in keys}, excluded_units=list(e["excluded_units"]),
                  train=sorted(e["train"]), val=sorted(e["val"]), test=sorted(e["test"])) for e in fold_entries]
    return json.dumps(canon, sort_keys=True, separators=(",", ":"))


def _fold_entries(folds, meta):
    rid = record_ids(meta)
    return [dict(fold=f.fold, name=f.name, unit=f.unit, test_unit=f.test_unit, test_subject=f.test_subject,
                 val_subject=f.val_subject, excluded_units=list(f.excluded_units), seed=f.seed,
                 train=sorted(rid[f.train_idx].tolist()), val=sorted(rid[f.val_idx].tolist()),
                 test=sorted(rid[f.test_idx].tolist())) for f in folds]


def plan_sha256(folds, meta):
    """sha256 over the fold contents (record ids per partition, units, seeds); row-order independent."""
    return hashlib.sha256(_canonical(_fold_entries(folds, meta)).encode()).hexdigest()


def plan_to_json(folds, meta, params=None):
    """Serializable plan. params: extra settings to record (unit, min_val_per_class, seed are required)."""
    check_plan(folds, meta, min_val_per_class=(params or {}).get("min_val_per_class", 1))
    entries = _fold_entries(folds, meta)
    table = manifest(folds, meta).set_index("fold")
    for e in entries:
        e["counts"] = {part: dict(nonseizure=int(table.loc[e["fold"], f"n_{part}_nonseizure"]),
                                  seizure=int(table.loc[e["fold"], f"n_{part}_seizure"]))
                       for part in ("train", "val", "test")}
    params = dict(params or {})
    params.setdefault("lopo_unit", folds[0].unit)
    params.setdefault("validation_rule", VALIDATION_RULE)
    params.setdefault("fit_seed_rule", "seed + fold (fold numbered from 1)")
    return dict(format=PLAN_FORMAT, record_id_column="segment_id", params=params, n_records=int(len(meta)),
                n_folds=len(folds), **dataset_fingerprints(meta),
                plan_sha256=hashlib.sha256(_canonical(entries).encode()).hexdigest(), folds=entries)


def plan_from_json(body, meta, *, check=True):
    """Replay a saved plan on (re)loaded metadata: record id -> current row index, any row order.

    Raises if the file was edited (plan_sha256), if the records differ from the data, or (check=True)
    if any invariant fails on this data."""
    if body.get("format") != PLAN_FORMAT:
        raise FoldPlanError(f"unknown plan format {body.get('format')!r}; expected {PLAN_FORMAT}")
    entries = body["folds"]
    if hashlib.sha256(_canonical(entries).encode()).hexdigest() != body.get("plan_sha256"):
        raise FoldPlanError("plan_sha256 does not match the fold contents: the plan file was modified")
    rid = record_ids(meta)
    if int(body.get("n_records", -1)) != len(rid):
        raise FoldPlanError(f"plan covers {body.get('n_records')} records, data has {len(rid)}")
    fp = dataset_fingerprints(meta)
    for key in ("records_sha256", "signals_sha256"):
        if key in body and key in fp and body[key] != fp[key]:
            raise FoldPlanError(f"{key} differs: the data (ids, subjects, labels or signals) is not the "
                                "dataset this plan was built on")
    pos = {r: i for i, r in enumerate(rid)}
    folds = []
    for e in entries:
        idx = []
        for part in ("train", "val", "test"):
            missing = [r for r in e[part] if r not in pos]
            if missing:
                raise FoldPlanError(f"{e['name']}: {len(missing)} {part} records not in the data, e.g. {missing[:3]}")
            idx.append(np.sort(np.fromiter((pos[r] for r in e[part]), dtype=np.int64, count=len(e[part]))))
        folds.append(Fold(fold=int(e["fold"]), name=e["name"], unit=e["unit"], test_unit=e["test_unit"],
                          test_subject=e["test_subject"], val_subject=e["val_subject"],
                          excluded_units=tuple(e["excluded_units"]), seed=int(e["seed"]),
                          train_idx=idx[0], val_idx=idx[1], test_idx=idx[2]))
    if check:
        check_plan(folds, meta, min_val_per_class=body.get("params", {}).get("min_val_per_class", 1))
    return folds


def save_plan(body, path):
    """Write the plan deterministically (same folds -> same bytes)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8", newline="\n")


def load_plan(path, meta, *, check=True):
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    return plan_from_json(body, meta, check=check), body


def plan_or_build(meta, *, plan_path="", unit="subject", min_val_per_class=5, seed=42):
    """Notebook entry point: replay LOPO_PLAN_PATH if given, else build from the data.

    Returns (folds, body). A replayed plan must agree with the requested unit, minimum and seed."""
    if plan_path:
        folds, body = load_plan(plan_path, meta)
        p = body.get("params", {})
        wanted = dict(lopo_unit=unit, min_val_per_class=int(min_val_per_class), seed=int(seed))
        differ = {k: (p.get(k), v) for k, v in wanted.items() if p.get(k) != v}
        if differ:
            raise FoldPlanError(f"plan {Path(plan_path).name} was built with different settings "
                                f"(plan, requested): {differ}")
        return folds, body
    folds = build_lopo_folds(meta, unit=unit, min_val_per_class=min_val_per_class, seed=seed)
    body = plan_to_json(folds, meta, dict(lopo_unit=unit, min_val_per_class=int(min_val_per_class), seed=int(seed)))
    return folds, body
