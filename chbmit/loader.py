"""Strict loader for the 8-channel CHB-MIT segment export (NPZ + per-row metadata CSV).

The NPZ is produced by ``data_prep/chbmit_zip_to_npz.py``: ``X`` float32 [N, 8, 2560]
(channel-major, values in volts as exported) and ``y`` int64 [N]. The metadata CSV has one row
per ``X`` row, including ``signal_sha256`` = sha256 of that row's little-endian float32 bytes.

``load_chbmit_npz`` verifies counts, shape, dtype, finiteness, channel order and row alignment
(hashes are recomputed from the stored float32 rows *before* any scaling), then applies a fixed
units conversion (``unit_scale``; 1e6 turns volts into microvolts). No normalization, filtering
or resampling is applied.
"""

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("FP1-F7", "P3-O1", "P4-O2", "FP2-F8", "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7")
FS = 256
N_SAMPLES = 2560
EXPECTED_COUNTS = dict(total=2053, nonseizure=1039, seizure=1014)
LABEL_NAMES = {0: "nonseizure", 1: "seizure"}
# Recording folders that belong to the same person (PhysioNet CHB-MIT notes).
SUBJECT_OF = {"chb21": "chb01", "chb17a": "chb17", "chb17b": "chb17"}
REQUIRED_META = ("segment_id", "patient", "subject", "label", "signal_sha256")
PLAUSIBLE_MEDIAN_ABS_UV = (1.0, 1000.0)


class ChbmitDataError(ValueError):
    """The NPZ/metadata pair does not match the expected CHB-MIT export."""


def signal_sha256(segment):
    """sha256 of one segment's little-endian float32 bytes (same definition as the converter)."""
    return hashlib.sha256(np.ascontiguousarray(segment, dtype="<f4").tobytes()).hexdigest()


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _preview(items, k=5):
    items = list(items)
    return ", ".join(map(str, items[:k])) + (f", ... ({len(items)} total)" if len(items) > k else "")


def _channel_sources(npz_path, npz_channels):
    """Channel lists declared next to the data: in the NPZ, the converter report, the export config."""
    found = {}
    if npz_channels is not None:
        found["npz:channels"] = [str(c) for c in np.asarray(npz_channels).ravel()]
    stem = npz_path.with_suffix("").name
    report = npz_path.parent / f"{stem}_report.json"
    config = npz_path.parent / f"{stem}_source" / "dataset_config.json"
    for path, key in ((report, "channels"), (config, "selected_eeg_channels")):
        if path.is_file():
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ChbmitDataError(f"cannot read {path.name}: {exc}") from exc
            if key in body:
                found[f"{path.relative_to(npz_path.parent).as_posix()}:{key}"] = [str(c) for c in body[key]]
    return found, (report if report.is_file() else None)


def load_chbmit_npz(npz, meta, unit_scale, *, expected=EXPECTED_COUNTS, channels=CHANNELS,
                    require_channel_names=True, return_info=False):
    """Load and validate the CHB-MIT export; return ``(X, y, meta)`` (plus ``info`` if requested).

    npz, meta     paths to ``chbmit_8ch.npz`` and ``chbmit_8ch_metadata.csv``.
    unit_scale    multiplier applied to X after validation (1e6: volts -> microvolts; 1.0: as stored).
    expected      dict(total, nonseizure, seizure) or None to skip the count check.
    channels      channel order the model expects; must equal every channel list found next to the
                  data (NPZ key ``channels``, ``<stem>_report.json``, ``<stem>_source/dataset_config.json``).
    require_channel_names
                  raise if no channel list is found (the NPZ itself stores no names).

    X is float32 [N, 8, 2560] scaled in place; y is int64 [N]; meta keeps the CSV columns (as text,
    label and segment_index as int) plus ``record_id`` (= segment_id), row i describing X[i].
    """
    npz, meta_path = Path(npz), Path(meta)
    unit_scale = float(unit_scale)
    if not (math.isfinite(unit_scale) and unit_scale > 0):
        raise ChbmitDataError(f"unit_scale must be a positive finite number, got {unit_scale!r}")
    for path in (npz, meta_path):
        if not path.is_file():
            raise ChbmitDataError(f"file not found: {path}")

    with np.load(npz, allow_pickle=False) as data:
        missing = {"X", "y"} - set(data.files)
        if missing:
            raise ChbmitDataError(f"{npz.name}: missing arrays {sorted(missing)} (found {data.files})")
        X, y = data["X"], data["y"]
        npz_channels = data["channels"] if "channels" in data.files else None

    # Array structure.
    n_chans = len(channels)
    if X.ndim != 3 or X.shape[1:] != (n_chans, N_SAMPLES):
        raise ChbmitDataError(f"X has shape {X.shape}; expected [N, {n_chans}, {N_SAMPLES}] (channel-major)")
    if X.dtype != np.float32:
        raise ChbmitDataError(f"X dtype is {X.dtype}; the export stores float32 (signal hashes depend on it)")
    if y.ndim != 1 or len(y) != len(X):
        raise ChbmitDataError(f"y has shape {y.shape}; expected [{len(X)}]")
    if not np.issubdtype(y.dtype, np.integer):
        raise ChbmitDataError(f"y dtype is {y.dtype}; expected integer labels")
    y = y.astype(np.int64, copy=False)
    bad_labels = sorted(set(np.unique(y).tolist()) - set(LABEL_NAMES))
    if bad_labels:
        raise ChbmitDataError(f"labels outside {{0, 1}}: {bad_labels}")
    counts = dict(total=int(len(y)), nonseizure=int((y == 0).sum()), seizure=int((y == 1).sum()))
    if expected is not None and counts != dict(expected):
        raise ChbmitDataError(f"segment counts {counts} != expected {dict(expected)}")

    # Metadata table.
    table = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
    missing_cols = [c for c in REQUIRED_META if c not in table.columns]
    if missing_cols:
        raise ChbmitDataError(f"{meta_path.name}: missing columns {missing_cols}")
    if len(table) != len(X):
        raise ChbmitDataError(f"{meta_path.name} has {len(table)} rows but X has {len(X)} segments")
    for col in ("segment_id", "patient", "subject", "signal_sha256"):
        empty = np.flatnonzero(table[col].str.strip() == "")
        if empty.size:
            raise ChbmitDataError(f"{meta_path.name}: empty '{col}' in rows {_preview(empty)}")
    dup_ids = table.loc[table["segment_id"].duplicated(), "segment_id"]
    if len(dup_ids):
        raise ChbmitDataError(f"duplicate segment_id values: {_preview(dup_ids)}")
    try:
        meta_label = table["label"].astype(int).to_numpy()
    except ValueError as exc:
        raise ChbmitDataError(f"{meta_path.name}: non-integer label values") from exc
    wrong = np.flatnonzero(meta_label != y)
    if wrong.size:
        raise ChbmitDataError(f"metadata label differs from y in rows {_preview(wrong)}")
    y_names = np.array([LABEL_NAMES[v] for v in y.tolist()], dtype=object)
    if "label_name" in table.columns:
        wrong = np.flatnonzero(table["label_name"].to_numpy() != y_names)
        if wrong.size:
            raise ChbmitDataError(f"label_name contradicts label in rows {_preview(wrong)}")
    expected_subject = table["patient"].map(lambda p: SUBJECT_OF.get(p, p))
    wrong = np.flatnonzero(table["subject"].to_numpy() != expected_subject.to_numpy())
    if wrong.size:
        raise ChbmitDataError("subject column disagrees with the same-person mapping "
                              f"{SUBJECT_OF} in rows {_preview(wrong)}")
    parts = table["segment_id"].str.replace("\\", "/", regex=False).str.split("/")
    three = parts.str.len() == 3
    if three.any():
        wrong = np.flatnonzero(three & ((parts.str[0] != y_names) | (parts.str[1] != table["patient"])))
        if wrong.size:
            raise ChbmitDataError(f"segment_id folder contradicts label/patient in rows {_preview(wrong)}")

    # Row alignment and content: recompute each row's hash from the stored float32 values.
    stored = table["signal_sha256"].str.lower().to_numpy()
    computed = np.empty(len(X), dtype=object)
    nonfinite = []
    for i in range(len(X)):
        if not np.isfinite(X[i]).all():
            nonfinite.append(i)
        computed[i] = signal_sha256(X[i])
    if nonfinite:
        raise ChbmitDataError(f"non-finite values in segments {_preview(nonfinite)}")
    mismatch = np.flatnonzero(computed != stored)
    if mismatch.size:
        if set(computed) == set(stored):
            raise ChbmitDataError(f"metadata rows are not aligned with X (same signals, different order); "
                                  f"first misaligned rows {_preview(mismatch)}")
        raise ChbmitDataError(f"signal_sha256 mismatch in {mismatch.size} rows ({_preview(mismatch)}): "
                              "the NPZ and metadata come from different exports or a file is corrupted")
    dup = table[table["signal_sha256"].duplicated(keep=False)]
    cross = [h for h, g in dup.groupby("signal_sha256") if g["subject"].nunique() > 1]
    if cross:
        raise ChbmitDataError(f"identical signals under different subjects (would leak across folds): "
                              f"{_preview(cross)}")

    # Channel order: the NPZ stores no names, so check the lists written next to it.
    sources, report_path = _channel_sources(npz, npz_channels)
    for name, found in sources.items():
        if list(found) != list(channels):
            raise ChbmitDataError(f"channel order from {name} is {found}; expected {list(channels)}")
    if not sources and require_channel_names:
        raise ChbmitDataError(f"cannot verify channel order: no channel list found for {npz.name}. Keep "
                              f"{npz.with_suffix('').name}_report.json next to the NPZ (written by the "
                              "converter) or pass require_channel_names=False")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path else {}
    for key, value in (("fs", FS), ("n_samples", N_SAMPLES)):
        if key in report and report[key] != value:
            raise ChbmitDataError(f"{report_path.name}: {key}={report[key]}, expected {value}")

    # Fixed units conversion, applied only after the raw rows were verified.
    probe = X[:, :, ::8]
    median_raw = float(np.median(np.abs(probe)))
    if unit_scale != 1.0:
        X *= np.float32(unit_scale)
        if not np.isfinite(X).all():
            raise ChbmitDataError(f"unit_scale={unit_scale} overflows float32")
    median_scaled = median_raw * unit_scale
    if unit_scale == 1e6 and not (PLAUSIBLE_MEDIAN_ABS_UV[0] <= median_scaled <= PLAUSIBLE_MEDIAN_ABS_UV[1]):
        warnings.warn(f"median |x| after unit_scale=1e6 is {median_scaled:.3g}; scalp EEG in microvolts "
                      f"is expected within {PLAUSIBLE_MEDIAN_ABS_UV}. Check the stored units.", stacklevel=2)

    meta_out = table.copy()
    meta_out["label"] = y
    if "segment_index" in meta_out.columns:
        meta_out["segment_index"] = pd.to_numeric(meta_out["segment_index"], errors="coerce").astype("Int64")
    meta_out.insert(0, "record_id", meta_out["segment_id"])
    meta_out = meta_out.reset_index(drop=True)

    if not return_info:
        return X, y, meta_out
    npz_hash = file_sha256(npz)
    info = dict(
        npz=npz.name, metadata=meta_path.name, npz_sha256=npz_hash,
        npz_sha256_matches_report=(report.get("npz_sha256") == npz_hash) if "npz_sha256" in report else None,
        counts=counts, shape=list(X.shape), dtype=str(X.dtype), fs=FS, channels=list(channels),
        channel_order_checked_against=sorted(sources), signal_rows_verified=int(len(X)),
        duplicate_signal_rows=int(len(dup)), unit_scale=unit_scale,
        median_abs_stored=median_raw, median_abs_scaled=median_scaled,
        patients=sorted(table["patient"].unique()), subjects=sorted(table["subject"].unique()),
    )
    return X, y, meta_out, info


def data_handling_record(unit_scale, info=None):
    """What the loader did to the signals, for the notebook's DATA_HANDLING block."""
    unit_scale = float(unit_scale)
    units = {1.0: "volts (as stored)", 1e3: "millivolts", 1e6: "microvolts"}.get(unit_scale,
                                                                                 f"volts x {unit_scale:g}")
    record = dict(dataset="CHB-MIT scalp EEG, 10 s segments, 8 bipolar channels", fs=FS, n_samples=N_SAMPLES,
                  channels=list(CHANNELS), input_layout="[N, channels, time]", stored_units="volts",
                  unit_scale=unit_scale, model_input_units=units,
                  unit_scale_note="fixed units conversion applied in the loader; not a normalization",
                  normalization="none", filtering="none (as exported)", resampling="none")
    if info:
        record.update(npz_sha256=info.get("npz_sha256"), signal_rows_verified=info.get("signal_rows_verified"),
                      median_abs_input=info.get("median_abs_scaled"))
    return record
