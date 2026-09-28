"""Build the shared CHB-MIT LOPO plan from the validated export and write it as JSON + CSV.

    python -m chbmit.make_plan --npz <dir>/chbmit_8ch.npz --meta <dir>/chbmit_8ch_metadata.csv

The data are loaded through the strict loader first (counts, channel order, row alignment by
signal hash), so a plan is only written for a verified dataset. Output is deterministic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chbmit.loader import load_chbmit_npz  # noqa: E402
from chbmit.lopo_folds import build_lopo_folds, manifest, plan_to_json, save_plan  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CSV_COLUMNS = ["fold", "test_unit", "validation_unit", "fit_seed",
               "n_train_nonseizure", "n_train_seizure", "n_val_nonseizure", "n_val_seizure",
               "n_test_nonseizure", "n_test_seizure", "n_train", "n_val", "n_test", "n_excluded",
               "test_subject", "validation_cases", "excluded_units", "n_train_subjects", "test_single_class",
               "name", "lopo_unit"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--unit", choices=("subject", "case"), default="subject")
    ap.add_argument("--min-val-per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default=str(REPO / "protocol" / "chbmit_lopo_plan.json"))
    ap.add_argument("--out-csv", default=str(REPO / "protocol" / "chbmit_lopo_folds.csv"))
    args = ap.parse_args(argv)

    _, _, meta, info = load_chbmit_npz(args.npz, args.meta, unit_scale=1.0, return_info=True)
    folds = build_lopo_folds(meta, unit=args.unit, min_val_per_class=args.min_val_per_class, seed=args.seed)
    params = dict(lopo_unit=args.unit, min_val_per_class=args.min_val_per_class, seed=args.seed)
    body = plan_to_json(folds, meta, params)
    body["source"] = dict(npz=info["npz"], metadata=info["metadata"], npz_sha256=info["npz_sha256"],
                          signal_rows_verified=info["signal_rows_verified"],
                          channel_order_checked_against=info["channel_order_checked_against"])
    save_plan(body, args.out_json)
    table = manifest(folds, meta)[CSV_COLUMNS]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False, lineterminator="\n")
    print(table[CSV_COLUMNS[:10]].to_string(index=False))
    print(f"\n{len(folds)} folds ({args.unit}), plan_sha256 {body['plan_sha256']}")
    print(f"records_sha256 {body['records_sha256']}\nsignals_sha256 {body['signals_sha256']}")
    print(f"wrote {args.out_json}\nwrote {args.out_csv}")
    return body


if __name__ == "__main__":
    main()
