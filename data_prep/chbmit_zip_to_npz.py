"""Convert the CHBMIT_10sec_8channels_labeled per-segment CSV export into one validated NPZ.

Reads directly from the zip archive (no extraction needed). Values are stored exactly as
exported (volts); no filtering, scaling or resampling is applied here. Unit handling is a
training-time decision recorded in the study configuration.

Output:
  <out>/chbmit_8ch.npz            X: float32 [n_segments, 8, 2560], y: int64 [n_segments]
  <out>/chbmit_8ch_metadata.csv   one row per segment (order matches X), incl. patient/subject ids
  <out>/chbmit_8ch_source/        dataset_config.json and segment_manifest.csv copied verbatim
  <out>/chbmit_8ch_report.json    validation summary and per-patient counts
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

DATASET_DIR = "CHBMIT_10sec_8channels_labeled"
CHANNELS = ["FP1-F7", "P3-O1", "P4-O2", "FP2-F8", "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7"]
N_SAMPLES = 2560
FS = 256
EXPECTED = dict(total=2053, nonseizure=1039, seizure=1014)
# Recording folders that belong to the same person (PhysioNet CHB-MIT notes):
# chb21 was recorded 1.5 years after chb01 from the same subject; chb17a/chb17b are sessions of chb17.
SUBJECT_OF = {"chb21": "chb01", "chb17a": "chb17", "chb17b": "chb17"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "chbmit_8ch_source").mkdir(parents=True, exist_ok=True)

    archive = zipfile.ZipFile(args.zip)
    names = archive.namelist()
    for extra in ("dataset_config.json", "segment_manifest.csv"):
        (out / "chbmit_8ch_source" / extra).write_bytes(archive.read(f"{DATASET_DIR}/{extra}"))
    manifest = pd.read_csv(io.BytesIO(archive.read(f"{DATASET_DIR}/segment_manifest.csv")), dtype=str)
    manifest["segment_file"] = manifest["segment_file"].str.replace("\\", "/", regex=False)

    csv_names = sorted(n for n in names if n.startswith(DATASET_DIR + "/") and n.endswith(".csv")
                       and len(PurePosixPath(n).parts) == 4)
    assert len(csv_names) == EXPECTED["total"], f"expected {EXPECTED['total']} segment CSVs, found {len(csv_names)}"
    assert set(manifest["segment_file"]) == {n.split("/", 1)[1] for n in csv_names}, "manifest and zip contents differ"

    X = np.empty((len(csv_names), len(CHANNELS), N_SAMPLES), dtype=np.float32)
    rows = []
    for i, name in enumerate(csv_names):
        _, class_dir, patient, file_name = PurePosixPath(name).parts
        blob = archive.read(name)
        frame = pd.read_csv(io.BytesIO(blob))
        if list(frame.columns) != CHANNELS + ["label"]:
            raise ValueError(f"{name}: unexpected columns {list(frame.columns)}")
        if frame.shape[0] != N_SAMPLES:
            raise ValueError(f"{name}: {frame.shape[0]} rows, expected {N_SAMPLES}")
        labels = frame["label"].unique()
        if len(labels) != 1:
            raise ValueError(f"{name}: mixed labels {labels}")
        label = int(labels[0])
        if label != {"nonseizure": 0, "seizure": 1}[class_dir]:
            raise ValueError(f"{name}: label {label} contradicts folder {class_dir}")
        values = frame[CHANNELS].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: non-finite values")
        X[i] = values.T.astype(np.float32)  # [channels, time]
        rel = name.split("/", 1)[1]
        m = manifest.loc[manifest["segment_file"] == rel].iloc[0]
        if m["patient"] != patient or int(m["label"]) != label:
            raise ValueError(f"{name}: manifest disagrees (patient={m['patient']}, label={m['label']})")
        rows.append(dict(segment_id=rel, patient=patient, subject=SUBJECT_OF.get(patient, patient),
                         recording=m["recording"], source_file=m["source_file"],
                         segment_index=int(m["segment_index"]), label=label, label_name=class_dir,
                         file_sha256=hashlib.sha256(blob).hexdigest(),
                         signal_sha256=hashlib.sha256(X[i].astype("<f4").tobytes()).hexdigest()))
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(csv_names)}", flush=True)

    meta = pd.DataFrame(rows)
    y = meta["label"].to_numpy(dtype=np.int64)
    counts = dict(total=len(meta), nonseizure=int((y == 0).sum()), seizure=int((y == 1).sum()))
    assert counts == EXPECTED, f"class counts {counts} != {EXPECTED}"
    duplicates = meta[meta["signal_sha256"].duplicated(keep=False)]

    np.savez(out / "chbmit_8ch.npz", X=X, y=y)
    meta.to_csv(out / "chbmit_8ch_metadata.csv", index=False)
    abs_uv = np.abs(X) * 1e6
    per_patient = (meta.groupby(["subject", "patient", "label_name"]).size().unstack(fill_value=0)
                   .reset_index().to_dict(orient="records"))
    report = dict(
        counts=counts, channels=CHANNELS, n_samples=N_SAMPLES, fs=FS, stored_units="volts (as exported)",
        amplitude_uV=dict(median_abs=float(np.median(abs_uv)), p99_abs=float(np.percentile(abs_uv, 99)),
                          max_abs=float(abs_uv.max())),
        duplicate_signal_groups=int(duplicates["signal_sha256"].nunique()),
        duplicate_segments=duplicates["segment_id"].tolist(),
        patients=sorted(meta["patient"].unique()), subjects=sorted(meta["subject"].unique()),
        single_class_patients=sorted(p for p, g in meta.groupby("patient") if g["label"].nunique() < 2),
        single_class_subjects=sorted(s for s, g in meta.groupby("subject") if g["label"].nunique() < 2),
        per_patient=per_patient,
        npz_sha256=hashlib.sha256((out / "chbmit_8ch.npz").read_bytes()).hexdigest(),
    )
    (out / "chbmit_8ch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_patient", "duplicate_segments")}, indent=2))


if __name__ == "__main__":
    main()
