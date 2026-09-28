# Protocol lock

`protocol_lock.py` shows, before any training, that two runs of the shared protocol are
comparable: the same hyper-parameters, the same shared code, the same data and the same splits.
Each side writes a small JSON lock; the locks are exchanged and compared. A run starts (or its
results are merged) only when the comparison says `MATCH`.

It is a single file with no dependencies beyond what the notebooks already use, so it runs
unchanged in Colab, on a RunPod pod and locally. Nothing is trained and nothing is installed.

## What a Bonn `MATCH` guarantees

The lock executes the notebook's own configuration, task, loader, fingerprint and split-planning
cells in plan-only mode (the training function is replaced by a recorder; about 20–30 s on CPU).
Three hashes are compared:

| Hash | Covers |
|---|---|
| `protocol_hash` | Shared configuration (seed, epochs, batch size, learning rate, weight decay, folds, holdout and inner-validation fractions, sampling rate, record length, checkpoint rule, metrics version); the preset and its full task list in execution order; raw-data handling; the code of every shared cell: imports and environment, setup (cell 4 without its parameter lines), task list, loader, fingerprint (without the code-hash constant), training loop and split planning; any additional code cell. |
| `data_order_hash` | The 500 recordings as float32 signal SHA-256, with their set, in the order the loader reads them. |
| `split_plan_hash` | All 352 fits (32 tasks × 10 CV folds + holdout): fit seed and the ordered train / validation / test lists of (signal, target), exactly as the notebook's own `run_combination` builds them. |

So a `MATCH` means both runs train on the same recordings, with the same targets, the same
partitions in the same order and the same seeds, under the same training and evaluation code.

**Deliberately not compared** (recorded in the lock for information):

- the model: model cell, `MODEL_OPTIONS`, `MODEL_PROVENANCE`, the charts/inference and report cells;
- operational settings: data and results paths, `SELECTED_IDS`, `MAX_CLASSES_TO_RUN`,
  `PLOT_EVERY`, `REUSE_COMPLETED`, `STOP_ON_STAGE_FAILURE`, report and download flags;
- the environment (package versions, device) and `STUDY_ID`, which depend on it.

All preset tasks are planned whatever `SELECTED_IDS` or `MAX_CLASSES_TO_RUN` select, so a staged
run (5 → 4 → 3 → 2 classes) or a run sharded across machines locks exactly like a full run.
The selected tasks are reported as the *run scope*.

**Normalisation.** Code is compared after removing blank lines, comment-only lines, trailing
spaces and line-ending differences; any other edit to a shared cell is a mismatch. Data are
compared by signal, not by file bytes: a copy with different line endings or file names but the
same values matches, with a note (`bytes differ, signals identical`). A copy whose files sort
differently (for example `Z1.txt` instead of `Z001.txt`) does **not** match, because the load
order, and therefore every fold, changes.

**Papermill.** A `parameters` tag and papermill's `injected-parameters` cells are understood:
injected values take effect where papermill puts them (after the tagged cell, or at the top when
no cell is tagged, where the configuration cell then overrides them). `--parameters` models a
planned papermill run without executing it. Injected names that are not known operational
settings count as protocol.

**Not guaranteed:** bit-identical training (GPU kernels are not deterministic across hardware),
identical results, or anything about the model definitions.

## Bonn commands

### Colab (the shared protocol notebooks)

Save the notebook first (File → Save), so the `.ipynb` on Drive holds the current code.

```python
from google.colab import drive
drive.mount("/content/drive")
!curl -sSLO https://raw.githubusercontent.com/Oxidopamine/atcnet-bonn-chbmit-comparative-study/main/protocol/protocol_lock.py
!python protocol_lock.py \
    --notebook "/content/drive/MyDrive/Colab Notebooks/Bonn_EEGNet_5_to_2_Class_Validation.ipynb" \
    --out /content/drive/MyDrive/bonn_lock_eegnet.json
```

`--data` defaults to the notebook's own `BONN_DATA_DIR`. Send `bonn_lock_eegnet.json` to the
other side. To check against a lock received from the other side, add
`--expect /content/drive/MyDrive/bonn_lock_atcnet.json`.

### RunPod (ATCNet)

With the repository at `/workspace/code/seizure_study`, the Bonn data at `/workspace/data/bonn`
and the lock received from Colab at `/workspace/locks/bonn_lock_eegnet.json`:

```bash
cd /workspace/code/seizure_study
python protocol/protocol_lock.py \
    --notebook notebooks/Bonn_ATCNet_5_to_2_Class_Validation.ipynb \
    --data /workspace/data/bonn \
    --out /workspace/locks/bonn_lock_atcnet.json \
    --expect /workspace/locks/bonn_lock_eegnet.json || exit 1
```

Lock the exact notebook file the queue runs (for example its frozen snapshot). A job's papermill
parameters can be checked before launch, and an executed notebook after the run:

```bash
python protocol/protocol_lock.py --notebook NOTEBOOK.ipynb --data /workspace/data/bonn \
    --parameters '{"SELECTED_IDS": ["Z_vs_S"], "PLOT_EVERY": 100, "RESULTS_ROOT": "/workspace/runs/x"}' \
    --out /tmp/job_lock.json --expect /workspace/locks/bonn_lock_eegnet.json
python protocol/protocol_lock.py --notebook /workspace/logs/.../executed.ipynb --data /workspace/data/bonn \
    --out /tmp/executed_lock.json --expect /workspace/locks/bonn_lock_eegnet.json
```

### Comparing saved locks

```bash
python protocol/protocol_lock.py --compare bonn_lock_atcnet.json bonn_lock_eegnet.json
```

The report names every differing configuration field and shared cell, compares split hashes task
by task on the tasks both locks contain (and fit by fit for the first differing task), and
explains data differences (different bytes, different load order, or different recordings).

## CHB-MIT leave-one-patient-out

The CHB-MIT notebooks share one fold plan, `lopo_plan.json` (format `chbmit-lopo-plan/1`, record
ids = `segment_id`), written by `chbmit/make_plan.py` or by the notebook when `LOPO_PLAN_PATH` is
empty, and replayed through `LOPO_PLAN_PATH`. The lock hashes that plan:

| Hash | Covers |
|---|---|
| `plan_sha256` | The plan's own content hash (units, validation subject, seeds, record ids per partition). |
| `data_signal_hash` | With `--npz/--meta`: every segment's float32 signal SHA-256 with its label (order-independent). Signals are recomputed from the NPZ and must agree with the metadata. |
| `subject_map_hash` | With `--npz/--meta`: which subject and recording folder every signal belongs to. |
| `fold_table_hash` | With `--npz/--meta`: per fold, the test unit(s), validation subject, excluded sibling folders, seed, sizes, seizure counts and the train / validation / test membership keyed by signal. |

With data, the comparison is on signals, so renamed record ids or a different row order still
match (both are reported as notes). The plan is also checked, with or without `--expect`:
disjoint partitions at segment and subject level, validation = exactly one subject with at least
`min_val_per_class` segments of each class, test = exactly the rows of the test unit, only the
test subject's other recording folders left out, every segment tested once. A violation exits
with code 3. Agreement with the decided fold rule (fold *k* tests unit *k* in natural id order;
validation = the next subject, cyclically, with enough segments of both classes; seed = SEED + *k*)
is reported as `matches_decided_rule`, computed by an independent implementation.

```bash
# either side, with the shared NPZ + metadata
python protocol/protocol_lock.py --chbmit-plan lopo_plan.json \
    --npz chbmit_8ch.npz --meta chbmit_8ch_metadata.csv \
    --out chbmit_lock_atcnet.json [--expect chbmit_lock_eegnet.json]

# the plan the decided rule gives on this data (same format, replayable via LOPO_PLAN_PATH)
python protocol/protocol_lock.py --chbmit-reference-plan lopo_plan_reference.json \
    --npz chbmit_8ch.npz --meta chbmit_8ch_metadata.csv [--lopo-unit subject|case]
```

In Colab, download the tool as above and pass the Drive paths of the plan, NPZ and metadata. Without
`--npz/--meta` only record ids are compared; lock both sides with data for the signal-level check.
The CHB-MIT lock covers the data and folds; training settings are fixed by the shared notebook.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | lock written; `MATCH` when `--expect`/`--compare` was given |
| 2 | usage or input error (notebook layout not recognised, a notebook cell failed, NPZ and metadata do not belong together) |
| 3 | `MISMATCH`, or an invalid CHB-MIT plan |
| 4 | the two locks cannot be compared (different kind or lock schema: re-run both sides with the same tool) |

Lock files (`protocol/*.json`) are not committed.
