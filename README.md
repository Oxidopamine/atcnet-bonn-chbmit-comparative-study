# ATCNet on Bonn and CHB-MIT: comparative seizure-classification study

ATCNet (attention temporal convolutional network) adapted from motor-imagery EEG to epileptic
seizure classification, trained under a shared protocol so it can be compared fairly with
EEGNet and TCFormer.

> **Status:** in progress. Code and protocol are being finalized; no results are published yet.

## Scope

| Dataset | Input | Tasks | Evaluation |
|---|---|---|---|
| Bonn University EEG (sets Z, O, N, F, S) | 1 channel × 4097 samples @ 173.61 Hz | 32 class combinations, 5 → 4 → 3 → 2 classes | 10-fold stratified CV + separate 70/30 holdout, inner validation for checkpoint selection |
| CHB-MIT Scalp EEG, 10 s segments | 8 bipolar channels × 2560 samples @ 256 Hz | seizure vs non-seizure | leave-one-patient-out: one test patient, one validation patient, remaining patients train |

Metrics: accuracy, balanced accuracy, precision, recall, F1 (macro/weighted), Cohen's kappa,
MCC, per-class sensitivity/specificity, ROC/AUC, precision–recall, confusion matrices, and
training/validation accuracy and loss curves.

On CHB-MIT, a test patient with one class (chb07 has no seizure segments) leaves sensitivity,
AUC, average precision, balanced accuracy, macro-averaged scores, kappa and MCC undefined. The
LOPO tables (`lopo_fold_metrics.csv`, `lopo_per_patient_metrics.csv`) report them as NaN and
are the per-fold source of record; the aggregates in `lopo_summary.json` count only the folds
that define each metric. Each fold's own `metrics.json`
comes from the shared metric code, which nulls only sensitivity, specificity, AUC and average
precision, so it keeps scikit-learn's placeholder values for the rest on such a fold.

## Design principles

- **Adaptation, not redesign.** ATCNet is changed only where the data requires it (number of
  input channels, sampling rate). At one input channel the PyTorch model is bit-identical to the
  shared single-channel port used for the other models.
- **Identical protocol across models.** Splits, seeds, optimizer, epochs, class weighting,
  checkpoint selection and metric code are shared; split identity is verified by hash.
- **Patient-level separation on CHB-MIT.** Recordings from the same person are grouped
  (chb01/chb21, chb17a/chb17b) so no subject appears in both training and testing.

## Layout

```
data_prep/   dataset conversion and validation scripts
```

More components (model, notebooks, RunPod harness) are added as they are finalized.

## Data

Datasets are not included. Obtain them from the original sources:

- **Bonn:** University of Bonn, Department of Epileptology (Andrzejak et al., 2001).
  Expected layout: `bonn/{Z,O,N,F,S}/*.txt`, 100 recordings per set, 4097 samples each.
- **CHB-MIT:** [PhysioNet](https://physionet.org/content/chbmit/1.0.0/), segmented into 10 s
  windows with 8 common bipolar channels. `data_prep/chbmit_zip_to_npz.py` converts a
  per-segment CSV export into a single validated `.npz` plus metadata.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for upstream attribution
(ATCNet by Altaheri et al.).
