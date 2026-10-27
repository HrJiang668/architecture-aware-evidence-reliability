# Architecture-Aware Evidence Reliability Estimation for Medical Object Detection

Minimal research code for the plug-in reliability layer described in the ESWA manuscript. A trained detector remains frozen. Candidate validity is learned from prediction-native evidence through three architecture-specific interfaces:

- **DER:** candidate-aligned dense features and local candidate relations (YOLO, RetinaNet).
- **RER:** ROI-aligned FPN evidence and proposal relations (Faster R-CNN).
- **QER:** decoder-query states and candidate competition (RF-DETR).

The implementation uses `QueryQCC` and `ours_qcc` as legacy *code identifiers* for the QER branch. These names do not denote a fourth method. The evidence-derived score is combined with raw detector confidence using an image-disjoint calibration subset; detector boxes, classes, confidence, and AP are unchanged.

## Contents

| Path | Purpose |
| --- | --- |
| `scripts/architecture_reliability_train.py` | Candidate-validity models, calibration baselines, raw-anchored selection, evaluation |
| `scripts/export_yolo_qcc_cache.py` | Frozen YOLO candidate and multi-scale evidence export |
| `scripts/export_torchvision_qcc_cache.py` | Frozen Faster R-CNN/RetinaNet evidence export |
| `scripts/export_rfdetr_qcc_cache.py` | Frozen RF-DETR query evidence export |
| `scripts/train_*_baseline.py` | Detector training entry points used before evidence export |
| `scripts/bootstrap_image_ci.py` | Paired image-cluster bootstrap of ranking and fixed-TP effects |
| `protocol.json` | Paper-reported split statistics, manifest fingerprints, and training settings |

No medical images, annotations, candidate caches, checkpoints, passwords, or private server files are included. Split statistics in `protocol.json` follow the manuscript table; the archived USE conversion has a different training count and is intentionally **not** published as the final split. Original dataset releases and the matching COCO/YOLO manifests must be obtained separately. The manifest SHA-256 values allow checking whether a locally obtained annotation conversion matches the experimental input; a matching hash is stronger evidence than a count alone.

## Environment

Python 3.10+ and a CUDA-enabled PyTorch environment are recommended. Install the packages listed in `requirements.txt` with versions compatible with your CUDA runtime and detector checkpoints. RF-DETR and Ultralytics interfaces can change across releases; use the versions matching the original detector weights when reproducing those branches. Only load checkpoints from trusted sources, because the exporter uses PyTorch checkpoint deserialization.

## Minimal workflow

1. Train or provide a detector checkpoint using the corresponding `train_*_baseline.py` script and your locally licensed dataset. Keep the detector fixed afterward.
2. Export **train** and **validation** candidate caches with the matching `export_*_qcc_cache.py` script. Pass dataset paths and output paths on the command line; the exporter writes `.npz` arrays plus a metadata `.json` file. Each cache contains `image`, `hidden`, `score`, `cls`, `geometry`, `context`, and `label` arrays. Query/dense caches also include `hidden_group_sizes`.
3. Train the plug-in and evaluate on the frozen validation candidates:

```bash
python scripts/architecture_reliability_train.py \
  --train-cache /path/to/train_candidates.npz \
  --val-cache /path/to/val_candidates.npz \
  --out-dir /path/to/reliability_results \
  --family dense --epochs 8 --seed 20260812
```

Use `--family roi` for Faster R-CNN and `--family query` for RF-DETR. The script writes raw, calibration, score-only, native-evidence, and full-model metrics to `reliability_metrics.json`, plus per-candidate score arrays. `anchor_ours_dense`, `anchor_ours_roi`, and `anchor_ours_qcc` are the raw-anchored full realizations.

For paired image-level uncertainty on a completed run:

```bash
python scripts/bootstrap_image_ci.py \
  --val-cache /path/to/val_candidates.npz \
  --ours-scores /path/to/reliability_results/anchor_ours_dense_scores.npy \
  --out /path/to/bootstrap.json --n 1000 --seed 20260826
```

Candidate labels use class-consistent, one-to-one matching at IoU 0.50. The training cache is split by image for fitting and calibration; final validation labels are not used to select the raw-anchor coefficient. `FP@TP90` and `FP@TP95` count invalid candidates remaining when 90% and 95% of valid candidates are retained. Reported detector AP belongs to the frozen detector, not to the post-hoc reliability score.

## Scope

This compact release includes the primary three-family protocol. Deformable DETR was a detector-convergence control in the supplementary analysis, not a primary reliability branch. The repository is source-available for inspection; reuse permissions require the project owners to add a license.
