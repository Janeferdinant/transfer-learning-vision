# Run log: second dataset, more backbones, data efficiency, ablations, Grad-CAM (2026-10-14 to 2026-10-15)

This log sits next to `2026-10-14-experiment.md` (the original TF Flowers / MobileNetV2 run), which is unchanged.
Everything below was produced by the commands in `scripts/run_queue.sh` and `python -m tlv.gradcam`; the
per-run CSVs are committed under `results/<suite>/` and the tables are regenerated with `python -m tlv.analysis`.
The progress lines of the queue are in `2026-10-14_15-queue-progress.txt` (console warnings and local paths removed).

## Environment

- Hardware: AMD Ryzen 7 5800X (8 cores, 16 logical), 31.9 GB RAM, CPU only (no CUDA). Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.
- 363 training runs in total (pilot 45, backbones 90, data efficiency 135, ablations 93), plus the Grad-CAM models. The longest timed fit was 271 s (about 4.5 minutes), well inside the 25-minute limit.

## Protocol (fixed before the reported runs; items marked * were decided after the pilot)

- **Datasets.** TF Flowers as before (5 classes, 240 photos per class, 720/240/240 split, identical file list to the 2026-10-14 run: verified by comparing `results/split_manifest.json` with the new pipeline's manifest). EuroSAT RGB (MIT licence): 10 classes, 120 images per class sampled with seed 2026, stratified 60/20/20 split = 720 train / 240 validation / 240 test, 64 px tiles up-scaled to 96 px. Archive SHA-256 values are checked before use.
- **Backbones** (ImageNet weights, same head, augmentation, optimiser, batch size 32, 30-epoch budget): MobileNetV2 (as before), MobileNetV3Small, EfficientNetB0, ResNet50. Fine-tuning unfreezes the last 4 blocks (MobileNetV2: `block_13_expand` onwards, i.e. the legacy setting; the parameter counts of the legacy models are reproduced exactly and asserted in `tests/test_models.py`); batch-norm layers stay frozen. Seeds 0 to 4.
- **Time-to-target** is unchanged: cumulative wall-clock time at the end of the first epoch whose validation accuracy is >= the target, including the per-epoch validation pass; runs that never reach it are charged the full budget (lower bound, `>=`). Primary target per dataset = 95% of the from-scratch final validation accuracy of that dataset's backbone-suite runs (flowers 65.2%, EuroSAT 70.3%).
- **Learning rates.** Per (dataset, backbone, approach), chosen on pilot seed 100 (not a reported seed) from {1e-4, 3e-4, 1e-3} by final validation accuracy after 8 epochs (`results/pilot_*`, `configs/lr_choices.json`). Flowers + MobileNetV2 and flowers + scratch keep the 2026-10-14 choices. On flowers every new backbone picked the same rates as the legacy ones; on EuroSAT scratch and the fine-tuned ResNet50/EfficientNetB0/MobileNetV3Small picked 1e-4, the lowest value of the grid (so the best rate may lie below it). *Pilot rule written after the first pilot attempt crashed (see below), before any reported run.
- **Data-efficiency suite.** MobileNetV2; training images per class 10, 25, 50, 100 and all (144 on flowers; 100 does not exist on EuroSAT, which has 72, so its grid is 10, 25, 50, all=72). Subsets are nested per seed and drawn with the run seed, so intervals include the subset draw. Validation and test sets are fixed. Epoch budget = max(30, ceil(200 / steps per epoch)), capped at 150, so tiny sets still get at least 200 optimiser steps (n=10 on flowers: 100 epochs). Seeds 0 to 4.
- **Ablations** (MobileNetV2, 3 seeds each, so their intervals are wide and only large effects should be read): fine-tune depth k in {1, 2, 4, 8, all} on flowers and EuroSAT; learning rate in {1e-4, 3e-4, 1e-3, 3e-3} for all three approaches on flowers; image size in {64, 96, 128} px for all three approaches on flowers.
- **Grad-CAM.** One scratch and one fine-tuned MobileNetV2 (seed 0, the 30-epoch protocol), Grad-CAM on the last feature map for the predicted class, plus a masking check on all 240 test images (grey out the top 20% of pixels by Grad-CAM vs by a random control map of the same grid size, measure the drop in the predicted-class probability on correctly classified images).

## Shared machine: what this means for the seconds

The runs were made on a desktop that other jobs were using at the same time. Each run records `bg_cpu_fraction`, the share of all CPU time used by processes other than the experiment during the timed fit (`src/tlv/quiet.py`):

| Suite | Runs | Median background CPU share | Max | Runs above 0.5 |
|---|---|---|---|---|
| pilot flowers / EuroSAT | 18 / 27 | 0.66 / 0.41 | 0.77 / 0.75 | 8 / 9 (accuracy-only, timings not used) |
| backbones flowers | 45 | 0.27 | 0.77 | 10 |
| backbones EuroSAT | 45 | 0.21 | 0.35 | 0 |
| data efficiency flowers / EuroSAT | 75 / 60 | 0.18 / 0.35 | 0.29 / 0.45 | 0 / 0 |
| ablations (depth, lr, size; flowers + EuroSAT depth) | 93 | 0.17 to 0.26 | 0.46 | 0 |

The lowest value ever observed on this machine was about 0.15, so a share of 0.15 to 0.3 is the normal level here; the earlier 2026-10-14 run was made under similar conditions and its README already says background load was not controlled. Consequences, stated plainly: seconds carry more noise than epochs (epochs and accuracies are not affected by load); the first 10 flowers backbone runs (seed 0 block) overlapped a heavier load. `python -m tlv.analysis --max-bg 0.5 --suffix _lowbg` repeats the analysis without runs above 0.5 (`results/summary_*_lowbg.*`): every backbone time reduction stays within 6 points of the full analysis (for example flowers frozen MobileNetV2 74% vs 70%, fine-tuned ResNet50 42% vs 47%); the MobileNets stay on top and the ordering of the heavier backbones is unchanged, while the order inside the top four moves within their intervals. The wait-for-a-quiet-machine guard added to the runner gave up after repeated timeouts (the machine was never quiet), so it did not delay or select runs.

## Incidents (all runs are kept)

- First pilot attempt: a run hit a Windows memory shortage (`MemoryError`) after 10 of 18 flowers pilot runs; the queue then moved on to the EuroSAT pilot, which was interrupted when the queue was restarted and was re-run from its start. The runner now retries a run after a memory shortage, and the 8 missing flowers pilot runs were completed with the same settings. One suite invocation was restarted after its guard code was changed (the one completed run before the restart, flowers scratch seed 0 of the backbone suite, is kept and waited 120 s for a quiet machine without effect on its timing).
- `per_run.csv` of the pilot got extra columns mid-way; it was repaired by inserting empty values for the older rows (no values changed).

## Pilot learning-rate choices (seed 100, 8 epochs)

See `configs/lr_choices.json` (committed) and `results/pilot_flowers/`, `results/pilot_eurosat/`.

## Generated tables

The tables below are copied from `results/summary_backbones.md`, `results/summary_data_efficiency.md` and `results/summary_ablations.md` (95% percentile bootstrap intervals over the seeds; reduction intervals resample each group independently).

## Grad-CAM masking check (TF Flowers test set, seed 0 models)

Drop in the predicted-class probability, in percentage points, when 20% of pixels are greyed out, on the test images each model classified correctly (95% bootstrap intervals over images). Source: `results/gradcam/gradcam_summary.json`; figure: `results/gradcam/gradcam_scratch_vs_finetuned.png` (one test image per class, chosen with seed 0, test indices [40, 78, 120, 156, 206]).

| Model | Test accuracy | Heat-map grid | Drop when Grad-CAM top 20% is masked | Drop when a random 20% is masked | Difference |
|---|---|---|---|---|---|
| scratch | 65.4% (157/240) | 6x6 | 19.5 (16.3 to 22.9) | 8.2 (5.5 to 11.0) | +11.3 (+7.0 to +15.6) |
| fine-tuned | 82.9% (199/240) | 3x3 | 23.7 (18.6 to 28.9) | 8.3 (5.3 to 11.7) | +15.4 (+9.4 to +21.5) |

## Backbones

### Backbone comparison (all-data training, 30 epochs)

#### flowers (primary target 65.2% validation accuracy)

| Backbone | Approach | Trainable params | Total params | Test acc, % (95% CI) | Reached | Time to 65.2% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch 1, s | Steady epoch, s |
|---|---|---|---|---|---|---|---|---|---|---|
| small CNN | From scratch | 241,829 | 242,533 | 66.9 (65.3 to 68.6) | 5/5 | 33.6 (22.5 to 45.3) | 8.2 | baseline | 7.4 | 3.38 |
| MobileNetV2 | Frozen | 6,405 | 2,264,389 | 79.7 (78.8 to 80.6) | 5/5 | 10.0 (6.9 to 14.8) | 1.6 | 70% (49 to 82%) | 8.8 | 1.68 |
| MobileNetV2 | Fine-tune | 1,669,765 | 2,264,389 | 82.7 (81.9 to 83.4) | 5/5 | 10.6 (8.4 to 14.5) | 1.0 | 68% (49 to 79%) | 10.6 | 1.80 |
| MobileNetV3Small | Frozen | 2,885 | 942,005 | 79.7 (78.9 to 80.3) | 5/5 | 9.6 (8.7 to 10.4) | 4.0 | 71% (57 to 79%) | 6.9 | 0.83 |
| MobileNetV3Small | Fine-tune | 761,013 | 942,005 | 82.1 (81.2 to 82.9) | 5/5 | 11.3 (10.6 to 12.0) | 2.0 | 67% (50 to 76%) | 10.1 | 1.21 |
| EfficientNetB0 | Frozen | 6,405 | 4,055,976 | 82.2 (81.4 to 82.8) | 5/5 | 15.8 (13.6 to 18.4) | 1.4 | 53% (29 to 67%) | 14.9 | 2.29 |
| EfficientNetB0 | Fine-tune | 2,876,869 | 4,055,976 | 82.5 (81.4 to 83.4) | 5/5 | 18.3 (15.9 to 20.8) | 1.0 | 45% (18 to 61%) | 18.3 | 2.98 |
| ResNet50 | Frozen | 10,245 | 23,597,957 | 83.7 (82.7 to 84.6) | 5/5 | 14.0 (11.5 to 17.4) | 1.0 | 58% (36 to 72%) | 14.0 | 4.17 |
| ResNet50 | Fine-tune | 16,079,365 | 23,597,957 | 85.3 (84.7 to 85.9) | 5/5 | 17.8 (16.3 to 20.0) | 1.0 | 47% (21 to 62%) | 17.8 | 6.79 |

#### flowers, fixed target 60% validation accuracy

| Backbone | Approach | Reached | Time to 60.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 5/5 | 13.4 (8.7 to 20.7) | 2.6 | baseline | baseline |
| MobileNetV2 | Frozen | 5/5 | 9.1 (6.8 to 13.0) | 1.2 | 32% (-24 to 61%) | 54% (30 to 69%) |
| MobileNetV2 | Fine-tune | 5/5 | 10.6 (8.4 to 14.5) | 1.0 | 21% (-41 to 53%) | 62% (50 to 71%) |
| MobileNetV3Small | Frozen | 5/5 | 8.7 (8.1 to 9.4) | 3.0 | 35% (-2 to 58%) | -15% (-50 to 12%) |
| MobileNetV3Small | Fine-tune | 5/5 | 10.5 (9.7 to 11.6) | 1.4 | 21% (-23 to 49%) | 46% (20 to 64%) |
| EfficientNetB0 | Frozen | 5/5 | 14.9 (13.2 to 17.0) | 1.0 | -11% (-76 to 29%) | 62% (50 to 71%) |
| EfficientNetB0 | Fine-tune | 5/5 | 18.3 (15.9 to 20.8) | 1.0 | -37% (-117 to 13%) | 62% (50 to 71%) |
| ResNet50 | Frozen | 5/5 | 14.0 (11.5 to 17.4) | 1.0 | -4% (-73 to 35%) | 62% (50 to 71%) |
| ResNet50 | Fine-tune | 5/5 | 17.8 (16.3 to 20.0) | 1.0 | -33% (-109 to 15%) | 62% (50 to 71%) |

#### flowers, fixed target 80% validation accuracy

| Backbone | Approach | Reached | Time to 80.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 0/5 | >= 108.0 (91.7 to 135.0) | 30.0 | baseline | baseline |
| MobileNetV2 | Frozen | 5/5 | 21.2 (14.0 to 28.7) | 8.8 | >= 80% (72 to 87%) | >= 71% (52 to 81%) |
| MobileNetV2 | Fine-tune | 5/5 | 16.1 (12.6 to 21.0) | 4.0 | >= 85% (79 to 89%) | >= 87% (84 to 89%) |
| MobileNetV3Small | Frozen | 4/5 | >= 28.3 (24.1 to 32.4) | 25.8 | 74% (67 to 80%) (both censored) | 14% (3 to 26%) (both censored) |
| MobileNetV3Small | Fine-tune | 5/5 | 14.8 (13.4 to 16.4) | 5.0 | >= 86% (83 to 89%) | >= 83% (81 to 87%) |
| EfficientNetB0 | Frozen | 5/5 | 21.0 (19.3 to 22.3) | 3.8 | >= 81% (76 to 85%) | >= 87% (85 to 90%) |
| EfficientNetB0 | Fine-tune | 5/5 | 19.5 (16.8 to 22.8) | 1.4 | >= 82% (77 to 86%) | >= 95% (94 to 97%) |
| ResNet50 | Frozen | 5/5 | 29.6 (23.6 to 35.6) | 4.6 | >= 73% (64 to 80%) | >= 85% (80 to 89%) |
| ResNet50 | Fine-tune | 5/5 | 21.5 (19.0 to 23.2) | 1.6 | >= 80% (76 to 84%) | >= 95% (93 to 96%) |

#### eurosat (primary target 70.3% validation accuracy)

| Backbone | Approach | Trainable params | Total params | Test acc, % (95% CI) | Reached | Time to 70.3% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch 1, s | Steady epoch, s |
|---|---|---|---|---|---|---|---|---|---|---|
| small CNN | From scratch | 242,474 | 243,178 | 72.0 (70.2 to 73.8) | 5/5 | 35.2 (29.0 to 42.1) | 10.8 | baseline | 5.7 | 2.96 |
| MobileNetV2 | Frozen | 12,810 | 2,270,794 | 85.4 (84.9 to 85.8) | 5/5 | 8.3 (8.3 to 8.4) | 2.0 | 76% (71 to 80%) | 6.9 | 1.37 |
| MobileNetV2 | Fine-tune | 1,676,170 | 2,270,794 | 85.7 (83.9 to 87.6) | 5/5 | 11.0 (10.2 to 11.7) | 2.6 | 69% (62 to 74%) | 8.4 | 1.59 |
| MobileNetV3Small | Frozen | 5,770 | 944,890 | 84.3 (83.3 to 85.0) | 5/5 | 8.7 (8.3 to 9.1) | 3.4 | 75% (70 to 80%) | 6.7 | 0.81 |
| MobileNetV3Small | Fine-tune | 763,898 | 944,890 | 87.7 (86.6 to 88.7) | 5/5 | 12.8 (12.3 to 13.2) | 3.6 | 64% (56 to 70%) | 9.7 | 1.14 |
| EfficientNetB0 | Frozen | 12,810 | 4,062,381 | 85.1 (84.3 to 85.8) | 5/5 | 15.6 (15.5 to 15.7) | 2.0 | 56% (46 to 63%) | 13.5 | 2.07 |
| EfficientNetB0 | Fine-tune | 2,883,274 | 4,062,381 | 89.1 (88.2 to 90.3) | 5/5 | 18.8 (18.1 to 19.4) | 1.8 | 46% (35 to 56%) | 16.7 | 2.74 |
| ResNet50 | Frozen | 20,490 | 23,608,202 | 89.6 (89.1 to 90.3) | 5/5 | 11.9 (11.7 to 12.2) | 1.0 | 66% (59 to 72%) | 11.9 | 3.88 |
| ResNet50 | Fine-tune | 16,089,610 | 23,608,202 | 90.6 (89.7 to 91.4) | 5/5 | 16.5 (16.1 to 17.0) | 1.0 | 53% (43 to 61%) | 16.5 | 6.17 |

#### eurosat, fixed target 60% validation accuracy

| Backbone | Approach | Reached | Time to 60.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 5/5 | 13.5 (12.2 to 14.6) | 3.6 | baseline | baseline |
| MobileNetV2 | Frozen | 5/5 | 6.9 (6.9 to 7.0) | 1.0 | 48% (43 to 53%) | 72% (69 to 75%) |
| MobileNetV2 | Fine-tune | 5/5 | 9.7 (9.0 to 10.1) | 1.8 | 28% (19 to 36%) | 50% (38 to 61%) |
| MobileNetV3Small | Frozen | 5/5 | 7.5 (7.4 to 7.6) | 2.0 | 44% (38 to 49%) | 44% (38 to 50%) |
| MobileNetV3Small | Fine-tune | 5/5 | 11.6 (11.2 to 12.0) | 2.6 | 14% (4 to 22%) | 28% (12 to 42%) |
| EfficientNetB0 | Frozen | 5/5 | 13.5 (13.5 to 13.6) | 1.0 | -1% (-11 to 8%) | 72% (69 to 75%) |
| EfficientNetB0 | Fine-tune | 5/5 | 16.7 (16.4 to 17.0) | 1.0 | -24% (-37 to -13%) | 72% (69 to 75%) |
| ResNet50 | Frozen | 5/5 | 11.9 (11.7 to 12.2) | 1.0 | 12% (2 to 19%) | 72% (69 to 75%) |
| ResNet50 | Fine-tune | 5/5 | 16.5 (16.1 to 17.0) | 1.0 | -22% (-36 to -12%) | 72% (69 to 75%) |

#### eurosat, fixed target 80% validation accuracy

| Backbone | Approach | Reached | Time to 80.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 0/5 | >= 91.8 (87.9 to 97.4) | 30.0 | baseline | baseline |
| MobileNetV2 | Frozen | 5/5 | 10.3 (9.2 to 11.6) | 3.4 | >= 89% (87 to 90%) | >= 89% (85 to 91%) |
| MobileNetV2 | Fine-tune | 5/5 | 14.2 (12.7 to 15.5) | 4.6 | >= 85% (83 to 86%) | >= 85% (82 to 88%) |
| MobileNetV3Small | Frozen | 5/5 | 11.9 (11.2 to 12.6) | 7.4 | >= 87% (86 to 88%) | >= 75% (72 to 78%) |
| MobileNetV3Small | Fine-tune | 5/5 | 14.6 (14.2 to 15.3) | 5.2 | >= 84% (83 to 85%) | >= 83% (81 to 83%) |
| EfficientNetB0 | Frozen | 5/5 | 18.6 (17.7 to 19.5) | 3.4 | >= 80% (78 to 81%) | >= 89% (87 to 90%) |
| EfficientNetB0 | Fine-tune | 5/5 | 19.9 (19.2 to 20.9) | 2.2 | >= 78% (77 to 80%) | >= 93% (91 to 93%) |
| ResNet50 | Frozen | 5/5 | 15.8 (15.4 to 16.1) | 2.0 | >= 83% (82 to 84%) | >= 93% (93 to 93%) |
| ResNet50 | Fine-tune | 5/5 | 16.5 (16.1 to 17.0) | 1.0 | >= 82% (81 to 83%) | >= 97% (97 to 97%) |

## Data efficiency

### Data efficiency (MobileNetV2; training images per class vary, validation and test sets fixed)

#### flowers

Target `primary` = 65.2% validation accuracy

| Train images/class | Approach | Epoch budget | Test acc, % (95% CI) | Diff vs scratch, pts (95% CI) | Reached | Time to 65.2% val, s (95% CI) | Epochs | Time reduction vs scratch |
|---|---|---|---|---|---|---|---|---|
| 10 | From scratch | 100 | 51.4 (48.3 to 54.5) |  | 0/5 | >= 39.6 (39.0 to 40.4) | 100.0 | baseline |
| 10 | Frozen | 100 | 66.7 (64.7 to 68.8) | +15.3 (+11.7 to +19.0) | 5/5 | 15.7 (11.3 to 20.5) | 26.4 | >= 60% (48 to 71%) |
| 10 | Fine-tune | 100 | 71.0 (69.8 to 72.4) | +19.6 (+16.3 to +22.9) | 5/5 | 9.1 (8.6 to 9.6) | 6.4 | >= 77% (76 to 78%) |
| 25 | From scratch | 50 | 55.8 (53.3 to 58.3) |  | 1/5 | >= 34.1 (31.5 to 35.9) | 48.4 | baseline |
| 25 | Frozen | 50 | 73.5 (70.7 to 76.0) | +17.7 (+13.9 to +21.2) | 5/5 | 8.3 (7.7 to 8.8) | 6.2 | >= 76% (73 to 78%) |
| 25 | Fine-tune | 50 | 76.6 (73.7 to 78.9) | +20.8 (+16.9 to +24.2) | 5/5 | 8.3 (7.6 to 9.1) | 3.8 | >= 76% (73 to 78%) |
| 50 | From scratch | 30 | 59.9 (57.4 to 63.4) |  | 3/5 | >= 27.2 (21.1 to 33.4) | 22.4 | baseline |
| 50 | Frozen | 30 | 76.4 (74.6 to 78.3) | +16.5 (+12.6 to +19.7) | 5/5 | 7.7 (7.3 to 8.1) | 3.8 | >= 72% (63 to 77%) |
| 50 | Fine-tune | 30 | 80.2 (79.0 to 81.7) | +20.3 (+16.5 to +23.2) | 5/5 | 8.4 (7.7 to 9.2) | 2.4 | >= 69% (60 to 76%) |
| 100 | From scratch | 30 | 67.3 (66.0 to 68.7) |  | 5/5 | 36.0 (30.1 to 42.0) | 16.4 | baseline |
| 100 | Frozen | 30 | 79.4 (78.2 to 80.4) | +12.1 (+10.2 to +13.8) | 5/5 | 7.2 (6.9 to 7.6) | 2.0 | 80% (76 to 83%) |
| 100 | Fine-tune | 30 | 82.7 (81.1 to 83.8) | +15.4 (+13.3 to +17.2) | 5/5 | 7.9 (7.5 to 8.4) | 1.2 | 78% (73 to 81%) |
| all | From scratch | 30 | 66.9 (65.3 to 68.6) |  | 5/5 | 25.9 (21.0 to 34.2) | 8.2 | baseline |
| all | Frozen | 30 | 79.7 (78.8 to 80.6) | +12.8 (+10.8 to +14.6) | 5/5 | 7.3 (6.9 to 7.6) | 1.6 | 72% (65 to 79%) |
| all | Fine-tune | 30 | 82.7 (81.9 to 83.4) | +15.8 (+14.0 to +17.5) | 5/5 | 7.9 (7.7 to 8.1) | 1.0 | 70% (62 to 77%) |

Target `fixed_0.50` = 50.0% validation accuracy

| Train images/class | Approach | Epoch budget | Test acc, % (95% CI) | Diff vs scratch, pts (95% CI) | Reached | Time to 50.0% val, s (95% CI) | Epochs | Time reduction vs scratch |
|---|---|---|---|---|---|---|---|---|
| 10 | From scratch | 100 | 51.4 (48.3 to 54.5) |  | 5/5 | 11.2 (10.0 to 12.5) | 23.4 | baseline |
| 10 | Frozen | 100 | 66.7 (64.7 to 68.8) | +15.3 (+11.7 to +19.0) | 5/5 | 8.5 (8.1 to 8.9) | 8.2 | 23% (14 to 32%) |
| 10 | Fine-tune | 100 | 71.0 (69.8 to 72.4) | +19.6 (+16.3 to +22.9) | 5/5 | 7.4 (7.0 to 7.8) | 2.4 | 33% (25 to 41%) |
| 25 | From scratch | 50 | 55.8 (53.3 to 58.3) |  | 5/5 | 8.6 (8.4 to 8.7) | 9.2 | baseline |
| 25 | Frozen | 50 | 73.5 (70.7 to 76.0) | +17.7 (+13.9 to +21.2) | 5/5 | 7.1 (6.7 to 7.6) | 3.8 | 17% (11 to 22%) |
| 25 | Fine-tune | 50 | 76.6 (73.7 to 78.9) | +20.8 (+16.9 to +24.2) | 5/5 | 7.2 (6.6 to 7.5) | 1.6 | 16% (11 to 23%) |
| 50 | From scratch | 30 | 59.9 (57.4 to 63.4) |  | 5/5 | 8.0 (8.0 to 8.1) | 5.0 | baseline |
| 50 | Frozen | 30 | 76.4 (74.6 to 78.3) | +16.5 (+12.6 to +19.7) | 5/5 | 6.5 (6.3 to 6.8) | 2.0 | 19% (16 to 22%) |
| 50 | Fine-tune | 30 | 80.2 (79.0 to 81.7) | +20.3 (+16.5 to +23.2) | 5/5 | 7.3 (7.0 to 7.6) | 1.0 | 9% (5 to 13%) |
| 100 | From scratch | 30 | 67.3 (66.0 to 68.7) |  | 5/5 | 8.8 (8.6 to 9.0) | 3.0 | baseline |
| 100 | Frozen | 30 | 79.4 (78.2 to 80.4) | +12.1 (+10.2 to +13.8) | 5/5 | 6.2 (5.9 to 6.5) | 1.0 | 29% (25 to 33%) |
| 100 | Fine-tune | 30 | 82.7 (81.1 to 83.8) | +15.4 (+13.3 to +17.2) | 5/5 | 7.7 (7.5 to 7.9) | 1.0 | 13% (10 to 15%) |
| all | From scratch | 30 | 66.9 (65.3 to 68.6) |  | 5/5 | 8.4 (8.1 to 8.6) | 2.0 | baseline |
| all | Frozen | 30 | 79.7 (78.8 to 80.6) | +12.8 (+10.8 to +14.6) | 5/5 | 6.5 (6.3 to 6.7) | 1.0 | 22% (19 to 26%) |
| all | Fine-tune | 30 | 82.7 (81.9 to 83.4) | +15.8 (+14.0 to +17.5) | 5/5 | 7.9 (7.7 to 8.1) | 1.0 | 6% (2 to 9%) |

#### eurosat

Target `primary` = 70.3% validation accuracy

| Train images/class | Approach | Epoch budget | Test acc, % (95% CI) | Diff vs scratch, pts (95% CI) | Reached | Time to 70.3% val, s (95% CI) | Epochs | Time reduction vs scratch |
|---|---|---|---|---|---|---|---|---|
| 10 | From scratch | 50 | 57.5 (55.6 to 58.7) |  | 0/5 | >= 32.9 (31.7 to 34.0) | 50.0 | baseline |
| 10 | Frozen | 50 | 74.5 (71.9 to 76.9) | +17.0 (+14.0 to +20.0) | 5/5 | 12.5 (10.1 to 14.5) | 13.6 | >= 62% (56 to 69%) |
| 10 | Fine-tune | 50 | 76.7 (73.4 to 79.9) | +19.2 (+15.5 to +22.8) | 5/5 | 14.1 (11.6 to 16.6) | 13.4 | >= 57% (49 to 65%) |
| 25 | From scratch | 30 | 63.3 (61.8 to 64.8) |  | 1/5 | >= 37.8 (36.9 to 38.7) | 29.6 | baseline |
| 25 | Frozen | 30 | 81.2 (79.0 to 83.5) | +17.9 (+15.2 to +20.6) | 5/5 | 8.8 (8.5 to 9.0) | 4.4 | >= 77% (76 to 78%) |
| 25 | Fine-tune | 30 | 81.3 (79.3 to 83.3) | +18.0 (+15.5 to +20.6) | 5/5 | 10.7 (10.0 to 11.6) | 4.8 | >= 72% (69 to 74%) |
| 50 | From scratch | 30 | 70.6 (69.5 to 71.8) |  | 5/5 | 34.5 (28.6 to 39.3) | 14.2 | baseline |
| 50 | Frozen | 30 | 85.7 (85.1 to 86.6) | +15.1 (+13.6 to +16.5) | 5/5 | 8.1 (7.6 to 8.7) | 2.0 | 76% (71 to 80%) |
| 50 | Fine-tune | 30 | 85.2 (83.7 to 86.7) | +14.6 (+12.7 to +16.6) | 5/5 | 11.6 (10.4 to 13.5) | 3.2 | 66% (58 to 72%) |
| all | From scratch | 30 | 72.0 (70.2 to 73.8) |  | 5/5 | 36.7 (31.9 to 41.5) | 10.8 | baseline |
| all | Frozen | 30 | 85.4 (84.9 to 85.8) | +13.4 (+11.7 to +15.3) | 5/5 | 9.1 (8.3 to 10.3) | 2.0 | 75% (70 to 79%) |
| all | Fine-tune | 30 | 85.7 (83.9 to 87.6) | +13.7 (+11.2 to +16.3) | 5/5 | 11.3 (10.6 to 12.1) | 2.6 | 69% (64 to 73%) |

Target `fixed_0.50` = 50.0% validation accuracy

| Train images/class | Approach | Epoch budget | Test acc, % (95% CI) | Diff vs scratch, pts (95% CI) | Reached | Time to 50.0% val, s (95% CI) | Epochs | Time reduction vs scratch |
|---|---|---|---|---|---|---|---|---|
| 10 | From scratch | 50 | 57.5 (55.6 to 58.7) |  | 5/5 | 12.7 (10.9 to 14.6) | 16.2 | baseline |
| 10 | Frozen | 50 | 74.5 (71.9 to 76.9) | +17.0 (+14.0 to +20.0) | 5/5 | 8.0 (7.3 to 8.8) | 4.6 | 37% (24 to 47%) |
| 10 | Fine-tune | 50 | 76.7 (73.4 to 79.9) | +19.2 (+15.5 to +22.8) | 5/5 | 9.0 (8.6 to 9.4) | 4.0 | 29% (17 to 39%) |
| 25 | From scratch | 30 | 63.3 (61.8 to 64.8) |  | 5/5 | 10.6 (10.3 to 10.8) | 6.6 | baseline |
| 25 | Frozen | 30 | 81.2 (79.0 to 83.5) | +17.9 (+15.2 to +20.6) | 5/5 | 7.1 (6.8 to 7.3) | 2.0 | 33% (30 to 36%) |
| 25 | Fine-tune | 30 | 81.3 (79.3 to 83.3) | +18.0 (+15.5 to +20.6) | 5/5 | 8.1 (7.4 to 8.8) | 1.4 | 24% (17 to 30%) |
| 50 | From scratch | 30 | 70.6 (69.5 to 71.8) |  | 5/5 | 10.3 (9.1 to 11.5) | 3.4 | baseline |
| 50 | Frozen | 30 | 85.7 (85.1 to 86.6) | +15.1 (+13.6 to +16.5) | 5/5 | 7.5 (6.8 to 8.4) | 1.4 | 27% (15 to 38%) |
| 50 | Fine-tune | 30 | 85.2 (83.7 to 86.7) | +14.6 (+12.7 to +16.6) | 5/5 | 9.4 (8.3 to 10.4) | 1.6 | 9% (-7 to 23%) |
| all | From scratch | 30 | 72.0 (70.2 to 73.8) |  | 5/5 | 11.5 (10.9 to 12.1) | 2.8 | baseline |
| all | Frozen | 30 | 85.4 (84.9 to 85.8) | +13.4 (+11.7 to +15.3) | 5/5 | 7.6 (6.9 to 8.7) | 1.0 | 34% (24 to 41%) |
| all | Fine-tune | 30 | 85.7 (83.9 to 87.6) | +13.7 (+11.2 to +16.3) | 5/5 | 8.7 (8.4 to 9.0) | 1.0 | 24% (19 to 29%) |

## Ablations

### Ablations (MobileNetV2; 3 seeds each)

#### Blocks fine-tuned

Primary targets used: flowers 65.2%, eurosat 70.3%.

| Dataset | Approach | Blocks fine-tuned | Trainable params | Final val acc, % (95% CI) | Test acc, % (95% CI) | Reached | Time to primary target, s (95% CI) | Epochs | Total train, s |
|---|---|---|---|---|---|---|---|---|---|
| eurosat | Fine-tune | 1 | 891,850 | 83.7 (81.7 to 86.3) | 85.6 (85.0 to 85.8) | 3/3 | 6.1 (6.0 to 6.1) | 1.0 | 45 |
| eurosat | Fine-tune | 2 | 1,207,690 | 85.0 (80.8 to 87.9) | 84.6 (83.7 to 85.8) | 3/3 | 8.2 (6.3 to 10.0) | 1.7 | 51 |
| eurosat | Fine-tune | 4 | 1,676,170 | 86.8 (83.3 to 88.7) | 86.1 (84.2 to 87.9) | 3/3 | 9.6 (8.6 to 10.4) | 2.7 | 50 |
| eurosat | Fine-tune | 8 | 2,025,226 | 85.1 (77.9 to 89.2) | 84.7 (80.4 to 87.1) | 3/3 | 11.5 (9.6 to 12.7) | 3.0 | 55 |
| eurosat | Fine-tune | all | 2,202,570 | 83.6 (75.8 to 91.7) | 81.7 (73.3 to 88.7) | 3/3 | 24.1 (22.9 to 25.2) | 6.3 | 80 |
| flowers | Fine-tune | 1 | 885,445 | 84.3 (82.5 to 86.3) | 82.4 (82.1 to 82.5) | 3/3 | 7.5 (7.4 to 7.7) | 1.0 | 49 |
| flowers | Fine-tune | 2 | 1,201,285 | 82.6 (80.0 to 85.0) | 82.4 (78.3 to 84.6) | 3/3 | 7.7 (7.5 to 7.8) | 1.0 | 50 |
| flowers | Fine-tune | 4 | 1,669,765 | 83.9 (82.5 to 85.4) | 82.9 (82.9 to 82.9) | 3/3 | 8.2 (7.9 to 8.7) | 1.0 | 53 |
| flowers | Fine-tune | 8 | 2,018,821 | 83.3 (81.7 to 85.4) | 83.3 (82.9 to 83.7) | 3/3 | 11.5 (9.9 to 12.9) | 2.0 | 60 |
| flowers | Fine-tune | all | 2,196,165 | 82.5 (80.0 to 85.4) | 81.9 (78.3 to 84.2) | 3/3 | 18.5 (15.4 to 20.4) | 3.3 | 84 |

#### Learning rate

Primary targets used: flowers 65.2%, eurosat 70.3%.

| Dataset | Approach | Learning rate | Trainable params | Final val acc, % (95% CI) | Test acc, % (95% CI) | Reached | Time to primary target, s (95% CI) | Epochs | Total train, s |
|---|---|---|---|---|---|---|---|---|---|
| flowers | From scratch | 0.0001 | 241,829 | 67.2 (64.2 to 71.2) | 66.5 (64.6 to 69.2) | 3/3 | 33.3 (24.1 to 41.3) | 10.0 | 94 |
| flowers | From scratch | 0.0003 | 241,829 | 68.3 (62.5 to 73.3) | 66.4 (65.0 to 68.8) | 3/3 | 22.1 (19.7 to 23.6) | 6.7 | 89 |
| flowers | From scratch | 0.001 | 241,829 | 66.0 (61.3 to 69.6) | 64.2 (62.5 to 67.5) | 3/3 | 29.5 (23.7 to 32.6) | 9.0 | 90 |
| flowers | From scratch | 0.003 | 241,829 | 66.1 (62.9 to 70.0) | 66.4 (60.4 to 71.7) | 3/3 | 27.0 (19.6 to 37.4) | 8.3 | 90 |
| flowers | Frozen | 0.0001 | 6,405 | 78.1 (76.7 to 79.6) | 72.9 (71.7 to 74.2) | 3/3 | 21.8 (19.1 to 24.7) | 12.0 | 46 |
| flowers | Frozen | 0.0003 | 6,405 | 81.2 (80.4 to 82.5) | 79.6 (78.3 to 80.4) | 3/3 | 11.3 (10.9 to 12.1) | 4.3 | 45 |
| flowers | Frozen | 0.001 | 6,405 | 80.6 (79.6 to 81.2) | 80.0 (78.3 to 81.2) | 3/3 | 8.1 (8.0 to 8.2) | 2.0 | 46 |
| flowers | Frozen | 0.003 | 6,405 | 80.8 (80.0 to 81.7) | 79.2 (78.8 to 79.6) | 3/3 | 6.9 (6.8 to 7.0) | 1.0 | 46 |
| flowers | Fine-tune | 0.0001 | 1,669,765 | 84.3 (83.7 to 84.6) | 83.9 (83.3 to 84.2) | 3/3 | 8.4 (8.4 to 8.5) | 1.0 | 54 |
| flowers | Fine-tune | 0.0003 | 1,669,765 | 83.9 (82.5 to 85.4) | 82.9 (82.9 to 82.9) | 3/3 | 8.4 (8.3 to 8.4) | 1.0 | 53 |
| flowers | Fine-tune | 0.001 | 1,669,765 | 20.0 (20.0 to 20.0) | 20.0 (20.0 to 20.0) | 0/3 | >= 53.0 (52.4 to 53.9) | 30.0 | 53 |
| flowers | Fine-tune | 0.003 | 1,669,765 | 20.0 (20.0 to 20.0) | 20.0 (20.0 to 20.0) | 0/3 | >= 54.0 (53.0 to 54.6) | 30.0 | 54 |

#### Image size, px

Primary targets used: flowers 65.2%, eurosat 70.3%.

| Dataset | Approach | Image size, px | Trainable params | Final val acc, % (95% CI) | Test acc, % (95% CI) | Reached | Time to primary target, s (95% CI) | Epochs | Total train, s |
|---|---|---|---|---|---|---|---|---|---|
| flowers | From scratch | 64 | 241,829 | 66.4 (62.9 to 69.2) | 65.4 (64.6 to 67.1) | 3/3 | 12.9 (7.2 to 19.2) | 7.7 | 40 |
| flowers | From scratch | 96 | 241,829 | 68.3 (62.5 to 73.3) | 66.4 (65.0 to 68.8) | 3/3 | 22.7 (21.2 to 23.5) | 6.7 | 92 |
| flowers | From scratch | 128 | 241,829 | 67.8 (63.7 to 72.9) | 67.1 (63.3 to 69.6) | 3/3 | 42.7 (38.9 to 44.8) | 6.7 | 179 |
| flowers | Frozen | 64 | 6,405 | 74.6 (73.8 to 75.4) | 75.6 (75.4 to 75.8) | 3/3 | 8.6 (7.6 to 9.5) | 3.0 | 31 |
| flowers | Frozen | 96 | 6,405 | 80.6 (79.6 to 81.2) | 80.0 (78.3 to 81.2) | 3/3 | 8.9 (8.4 to 9.4) | 2.0 | 46 |
| flowers | Frozen | 128 | 6,405 | 85.8 (85.4 to 86.3) | 85.1 (85.0 to 85.4) | 3/3 | 8.7 (6.8 to 11.1) | 1.3 | 74 |
| flowers | Fine-tune | 64 | 1,669,765 | 78.1 (76.7 to 79.2) | 80.7 (79.6 to 81.2) | 3/3 | 11.7 (11.2 to 12.3) | 3.3 | 46 |
| flowers | Fine-tune | 96 | 1,669,765 | 83.9 (82.5 to 85.4) | 82.9 (82.9 to 82.9) | 3/3 | 9.1 (8.7 to 9.4) | 1.0 | 56 |
| flowers | Fine-tune | 128 | 1,669,765 | 86.0 (84.2 to 87.5) | 86.9 (83.3 to 89.6) | 3/3 | 10.9 (9.8 to 12.3) | 1.7 | 84 |


## Verification (fresh clone of this branch, new virtual environment)

- `python -m pip install -r requirements.txt`, then `python -m pytest -q`: 69 passed; `ruff check .`: all checks passed; `mypy`: no issues in 19 source files.
- CI configuration (`requirements-ci.txt`, no TensorFlow): ruff clean, mypy clean, `61 passed, 1 skipped` (the TensorFlow model tests are skipped).
- Smoke runs from the README: `python -m tlv.experiment --seeds 0 --epochs 3 --out results/smoke` (split sizes 720/240/240, three runs printed); `python -m tlv.suite --suite backbones --dataset eurosat --seeds 0 --epochs 2 --backbones mobilenetv3small --out results/smoke_eurosat` (three runs); `python -m tlv.gradcam --epochs 2 --out results/smoke_gradcam` (figure and summary written). The two dataset archives were copied into the clone's `data/` folder instead of being downloaded again; their SHA-256 digests were still verified by the code.
- `python -m tlv.analysis` and `python -m tlv.plots` on the committed CSVs reproduced `results/summary_*.json` and `.md`, and the figures, with no difference from the committed files (`git diff` empty); `python -m tlv.export_viz --date 2026-10-15` reproduced `docs/explorer-data.js` byte for byte.
- Explorer checked with Playwright at 1280 px and 390 px: no horizontal page overflow at either width (also after switching to EuroSAT and to the time-to-target view), dataset, backbone, slider and metric controls work, zero console errors or warnings. The flowers and MobileNetV2 view still shows the original 10-seed numbers (60% target: frozen 20%, fine-tuned 1%).
- GitHub Actions (`.github/workflows/ci.yml`: ruff, mypy, fast tests without TensorFlow) on the pull request: the first runs failed because mypy flagged Windows-only calls in `hardware.py` on Linux; after the fix (`unused-ignore` comments) the run succeeded.
