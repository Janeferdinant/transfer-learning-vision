# Run log: 2026-10-14, transfer learning vs training from scratch, time-to-target

Status: **Verified (run 2026-10-14, 10 seeds per approach, AMD Ryzen 7 5800X 8-core / 16 logical cores / 31.9 GB RAM, CPU only)**

This is an independent experiment on public data. It is not the project behind the CV line "reported 30% reduction in model training time" and does not reproduce it.

## Command

```
python -m tlv.experiment --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 30
python -m tlv.aggregate
python -m tlv.export_viz --date 2026-10-14
```

Software: Python 3.11.9, TensorFlow 2.21.0 (Keras 3.15.1), OS string as reported by Python: Windows 10 (the machine runs Windows 11). Hardware record: `results/hardware.json`.
The 30 runs took 39 minutes of timed training in total (38 minutes between the first and last finish); every individual run took under 2.5 minutes. Nothing else of mine was running while they ran; the machine is a normal desktop, so background load from other applications was not controlled (this is why per-run times vary by roughly 10 to 20% between seeds).

## Protocol (fixed before the 10-seed run)

- Data: TF Flowers, 5 classes. 3 exact-duplicate files removed from the 3,670 photos, then 240 photos per class sampled with seed 2026 (1,200 images). Centre-cropped to a square and resized to 96 x 96. Stratified 60/20/20 split, seed 2026: 720 train, 240 validation, 240 test (144/48/48 per class). The split is the same for every run. Per-split file names: `results/split_manifest.json`. A pixel-hash check confirmed no image appears in two splits.
- Models: (a) small 4-block CNN with batch norm, random init, 241,829 trainable parameters; (b) MobileNetV2 (alpha 1.0, ImageNet weights, 96 px) with the backbone frozen and a trained head (6,405 trainable parameters); (c) the same backbone with blocks from `block_13_expand` onward unfrozen (BatchNorm layers stay frozen; 1,669,765 trainable parameters).
- Identical across approaches: input scaling to [-1, 1], augmentation (horizontal flip, rotation 0.05, zoom 0.1), head (global average pooling, dropout 0.3, 5-way softmax), Adam, batch size 32, loss, 30-epoch budget, shuffling and initialisation seeds 0 to 9.
- Learning rate is the one per-approach setting. It was chosen on a separate pilot seed (100, not among the reported seeds) from {1e-3, 3e-4, 1e-4} by final validation accuracy after 20 epochs: scratch 3e-4, frozen 1e-3, fine-tune 3e-4 (fine-tune tied at 0.850 between 3e-4 and 1e-4; the larger rate was taken). At 1e-3 fine-tuning collapsed to chance accuracy, which is why one shared rate was not used. Pilot logs: `docs/runs/pilot-lr-sweep/`. This is a single-seed selection and is itself noisy.
- Two further choices were made after looking at the pilot, and are disclosed here because they favour neither side by design but could affect the numbers: the epoch budget was raised from 20 to 30 because the scratch CNN had not plateaued at 20 epochs; and batch-norm momentum in the scratch CNN was lowered to 0.9 because the validation accuracy of a first, unsaved pilot (default momentum 0.99, 23 steps per epoch) was erratic.
- Timer: `time.perf_counter()` (monotonic), started when training starts and read after each epoch. It includes the per-epoch validation pass and any graph tracing in epoch 1. It excludes dataset download and caching, model construction, ImageNet weight download, and the final test evaluation. One untimed one-epoch warm-up per approach runs at process start. Approach order is rotated by seed.
- Time-to-target: wall-clock seconds and epoch count at the end of the first epoch whose validation accuracy is at least the target. Runs that never reach the target are charged their full budget, so the reported time is a lower bound (marked >=).
- Targets (defined before the 10-seed run): primary = 0.95 x the mean final validation accuracy of the from-scratch runs ("match what training from scratch achieves in the full budget"); secondary = 0.90 x the best approach's mean final validation accuracy; plus fixed 60/70/80/90%. The target values themselves are computed once from all 10 runs and are not recomputed inside each bootstrap resample.
- Intervals: percentile bootstrap, 10,000 resamples of the seeds, 95%. The relative reduction vs scratch is 1 - mean(approach) / mean(scratch), resampling each approach's seeds independently.

## Console output of the runs (progress lines only)

```
scratch   seed=0 train=108.5s val=0.692 test=0.654 (wall incl. eval 109.0s)
frozen    seed=0 train=63.8s val=0.817 test=0.792 (wall incl. eval 66.1s)
finetune  seed=0 train=73.8s val=0.796 test=0.817 (wall incl. eval 76.6s)
frozen    seed=1 train=65.7s val=0.804 test=0.808 (wall incl. eval 67.9s)
finetune  seed=1 train=79.5s val=0.854 test=0.854 (wall incl. eval 82.1s)
scratch   seed=1 train=131.1s val=0.733 test=0.688 (wall incl. eval 131.8s)
finetune  seed=2 train=82.1s val=0.804 test=0.800 (wall incl. eval 85.6s)
scratch   seed=2 train=129.1s val=0.625 test=0.650 (wall incl. eval 129.8s)
frozen    seed=2 train=69.3s val=0.812 test=0.821 (wall incl. eval 71.9s)
scratch   seed=3 train=112.5s val=0.650 test=0.658 (wall incl. eval 113.1s)
frozen    seed=3 train=58.3s val=0.833 test=0.804 (wall incl. eval 60.5s)
finetune  seed=3 train=66.5s val=0.858 test=0.846 (wall incl. eval 68.6s)
frozen    seed=4 train=52.4s val=0.800 test=0.804 (wall incl. eval 54.7s)
finetune  seed=4 train=59.3s val=0.863 test=0.854 (wall incl. eval 61.0s)
scratch   seed=4 train=129.2s val=0.733 test=0.696 (wall incl. eval 129.8s)
finetune  seed=5 train=81.1s val=0.850 test=0.833 (wall incl. eval 83.5s)
scratch   seed=5 train=116.8s val=0.708 test=0.712 (wall incl. eval 117.4s)
frozen    seed=5 train=65.1s val=0.817 test=0.804 (wall incl. eval 67.3s)
scratch   seed=6 train=122.1s val=0.708 test=0.704 (wall incl. eval 122.7s)
frozen    seed=6 train=50.5s val=0.804 test=0.800 (wall incl. eval 52.3s)
finetune  seed=6 train=57.9s val=0.812 test=0.783 (wall incl. eval 59.6s)
frozen    seed=7 train=49.1s val=0.829 test=0.796 (wall incl. eval 50.7s)
finetune  seed=7 train=57.3s val=0.842 test=0.850 (wall incl. eval 59.0s)
scratch   seed=7 train=90.7s val=0.654 test=0.688 (wall incl. eval 91.2s)
finetune  seed=8 train=58.3s val=0.854 test=0.817 (wall incl. eval 59.9s)
scratch   seed=8 train=90.1s val=0.654 test=0.658 (wall incl. eval 90.5s)
frozen    seed=8 train=48.9s val=0.812 test=0.804 (wall incl. eval 50.6s)
scratch   seed=9 train=88.8s val=0.637 test=0.688 (wall incl. eval 89.3s)
frozen    seed=9 train=48.8s val=0.817 test=0.796 (wall incl. eval 50.5s)
finetune  seed=9 train=58.0s val=0.858 test=0.829 (wall incl. eval 59.7s)
```

## Hardware

```json
{
  "cpu": "AMD Ryzen 7 5800X 8-Core Processor",
  "logical_cores": 16,
  "ram_gb": 31.9,
  "os": "Windows 10",
  "python": "3.11.9",
  "tensorflow": "2.21.0",
  "accelerator": "none (CPU only, no CUDA)"
}
```

## Results

### Target `primary_0.95_of_scratch_final` = 64.6% validation accuracy

| Approach | Runs reaching target | Mean time-to-target, s (95% CI) | Mean epochs (95% CI) | Time reduction vs scratch (95% CI) | Epoch reduction vs scratch (95% CI) |
|---|---|---|---|---|---|
| From scratch | 10 of 10 | 32.0 (26.2 to 38.3) | 8.1 (6.5 to 9.8) | baseline | baseline |
| Feature extraction (frozen) | 10 of 10 | 11.4 (10.3 to 12.7) | 1.6 (1.3 to 1.9) | 64% (55 to 71%) | 80% (74 to 85%) |
| Fine-tuning | 10 of 10 | 13.3 (11.9 to 14.7) | 1.0 (1.0 to 1.0) | 58% (48 to 66%) | 88% (85 to 90%) |

### Target `secondary_0.90_of_best_final` = 75.5% validation accuracy

| Approach | Runs reaching target | Mean time-to-target, s (95% CI) | Mean epochs (95% CI) | Time reduction vs scratch (95% CI) | Epoch reduction vs scratch (95% CI) |
|---|---|---|---|---|---|
| From scratch | 0 of 10 | >= 111.9 (102.0 to 121.5) | >= 30.0 (30.0 to 30.0) | baseline | baseline |
| Feature extraction (frozen) | 10 of 10 | 13.6 (12.3 to 15.0) | 3.0 (2.6 to 3.4) | 88% (86 to 89%) | 90% (89 to 91%) |
| Fine-tuning | 10 of 10 | 16.0 (13.7 to 18.5) | 2.4 (1.7 to 3.1) | 86% (83 to 88%) | 92% (90 to 94%) |

### Target `fixed_0.60` = 60.0% validation accuracy

| Approach | Runs reaching target | Mean time-to-target, s (95% CI) | Mean epochs (95% CI) | Time reduction vs scratch (95% CI) | Epoch reduction vs scratch (95% CI) |
|---|---|---|---|---|---|
| From scratch | 10 of 10 | 13.5 (12.4 to 14.8) | 2.8 (2.4 to 3.3) | baseline | baseline |
| Feature extraction (frozen) | 10 of 10 | 10.7 (9.7 to 12.1) | 1.2 (1.0 to 1.5) | 20% (8 to 31%) | 57% (44 to 67%) |
| Fine-tuning | 10 of 10 | 13.3 (11.9 to 14.7) | 1.0 (1.0 to 1.0) | 1% (-13 to 14%) | 64% (58 to 70%) |

### Target `fixed_0.70` = 70.0% validation accuracy

| Approach | Runs reaching target | Mean time-to-target, s (95% CI) | Mean epochs (95% CI) | Time reduction vs scratch (95% CI) | Epoch reduction vs scratch (95% CI) |
|---|---|---|---|---|---|
| From scratch | 9 of 10 | >= 89.9 (73.8 to 106.2) | >= 23.9 (20.2 to 27.2) | baseline | baseline |
| Feature extraction (frozen) | 10 of 10 | 12.0 (11.0 to 13.1) | 2.0 (2.0 to 2.0) | 87% (83 to 89%) | 92% (90 to 93%) |
| Fine-tuning | 10 of 10 | 14.4 (12.7 to 16.4) | 1.6 (1.2 to 2.0) | 84% (80 to 87%) | 93% (91 to 95%) |

### Target `fixed_0.80` = 80.0% validation accuracy

| Approach | Runs reaching target | Mean time-to-target, s (95% CI) | Mean epochs (95% CI) | Time reduction vs scratch (95% CI) | Epoch reduction vs scratch (95% CI) |
|---|---|---|---|---|---|
| From scratch | 0 of 10 | >= 111.9 (102.0 to 121.5) | >= 30.0 (30.0 to 30.0) | baseline | baseline |
| Feature extraction (frozen) | 10 of 10 | 21.2 (18.2 to 24.3) | 7.7 (6.3 to 9.1) | 81% (78 to 84%) | 74% (69 to 79%) |
| Fine-tuning | 10 of 10 | 19.0 (16.7 to 21.3) | 4.1 (3.2 to 5.2) | 83% (80 to 85%) | 86% (83 to 89%) |

### Target `fixed_0.90` = 90.0% validation accuracy

| Approach | Runs reaching target | Mean time-to-target, s (95% CI) | Mean epochs (95% CI) | Time reduction vs scratch (95% CI) | Epoch reduction vs scratch (95% CI) |
|---|---|---|---|---|---|
| From scratch | 0 of 10 | >= 111.9 (102.0 to 121.5) | >= 30.0 (30.0 to 30.0) | baseline | baseline |
| Feature extraction (frozen) | 0 of 10 | >= 57.2 (52.4 to 62.0) | >= 30.0 (30.0 to 30.0) | 49% (42 to 55%) | 0% (0 to 0%) |
| Fine-tuning | 0 of 10 | >= 67.4 (61.2 to 73.7) | >= 30.0 (30.0 to 30.0) | 40% (31 to 47%) | 0% (0 to 0%) |

### Final accuracy at the fixed epoch budget

| Approach | Runs | Final val accuracy, % (95% CI) | Test accuracy, % (95% CI) | Total training time, s (95% CI) |
|---|---|---|---|---|
| From scratch | 10 | 68.0 (65.7 to 70.4) | 68.0 (66.6 to 69.2) | 111.9 (102.0 to 121.5) |
| Feature extraction (frozen) | 10 | 81.5 (80.9 to 82.1) | 80.3 (79.8 to 80.8) | 57.2 (52.4 to 62.0) |
| Fine-tuning | 10 | 83.9 (82.3 to 85.3) | 82.8 (81.3 to 84.2) | 67.4 (61.2 to 73.7) |

Note on `fixed_0.90`: no approach reaches 90% in the budget, so every row is censored at the budget and the "reduction" there only reflects the different per-epoch cost, not time-to-target. Do not read it as a time-to-target saving.
