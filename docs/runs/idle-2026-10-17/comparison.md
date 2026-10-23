# Old (shared CPU) versus new (idle-machine rerun) timings

Old: `results/` (run 2026-10-14/15, other jobs on the machine). New: `results/idle_2026-10-17` (rerun 2026-10-17). Same seeds, settings and code; new times are cumulative wall-clock seconds on this machine. 'Intervals' says whether the old and new 95% intervals of the reduction overlap, and flags a change in the sign of the interval.

### Backbones: time to the primary target (95% of scratch's final validation accuracy)

Primary targets, old / new: flowers 65.2% / 65.2%, EuroSAT 70.3% / 70.3%.

| Dataset | Backbone | Approach | Time, s: old | new | Reduction vs scratch: old (95% CI) | new (95% CI) | Intervals | Test acc %: old | new | Mean bg CPU share: old | new |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flowers | small CNN | scratch | 33.6 | 28.0 | baseline | baseline |  | 66.9 | 66.9 | 0.37 | 0.36 |
| flowers | efficientnetb0 | finetune | 18.3 | 21.9 | 45% (18 to 61) | 22% (4 to 40) | overlap | 82.5 | 82.5 | 0.34 | 0.40 |
| flowers | efficientnetb0 | frozen | 15.8 | 18.5 | 53% (29 to 67) | 34% (18 to 50) | overlap | 82.2 | 82.2 | 0.37 | 0.46 |
| flowers | mobilenetv2 | finetune | 10.6 | 11.3 | 68% (49 to 79) | 60% (50 to 69) | overlap | 82.7 | 82.7 | 0.38 | 0.56 |
| flowers | mobilenetv2 | frozen | 10.0 | 10.0 | 70% (49 to 82) | 64% (56 to 73) | overlap | 79.7 | 79.7 | 0.37 | 0.55 |
| flowers | mobilenetv3small | finetune | 11.3 | 14.2 | 67% (50 to 76) | 49% (38 to 61) | overlap | 82.1 | 82.1 | 0.34 | 0.54 |
| flowers | mobilenetv3small | frozen | 9.6 | 11.6 | 71% (57 to 79) | 59% (49 to 68) | overlap | 79.7 | 79.7 | 0.31 | 0.62 |
| flowers | resnet50 | finetune | 17.8 | 20.5 | 47% (21 to 62) | 27% (10 to 44) | overlap | 85.3 | 85.3 | 0.31 | 0.38 |
| flowers | resnet50 | frozen | 14.0 | 15.1 | 58% (36 to 72) | 46% (34 to 59) | overlap | 83.7 | 83.7 | 0.31 | 0.40 |
| eurosat | small CNN | scratch | 35.2 | 35.6 | baseline | baseline |  | 72.0 | 72.0 | 0.27 | 0.35 |
| eurosat | efficientnetb0 | finetune | 18.8 | 23.7 | 46% (35 to 56) | 34% (20 to 45) | overlap | 89.1 | 89.1 | 0.19 | 0.44 |
| eurosat | efficientnetb0 | frozen | 15.6 | 19.0 | 56% (46 to 63) | 47% (37 to 54) | overlap | 85.1 | 85.1 | 0.21 | 0.53 |
| eurosat | mobilenetv2 | finetune | 11.0 | 13.4 | 69% (62 to 74) | 62% (55 to 68) | overlap | 85.7 | 85.7 | 0.20 | 0.54 |
| eurosat | mobilenetv2 | frozen | 8.3 | 10.4 | 76% (71 to 80) | 71% (66 to 75) | overlap | 85.4 | 85.4 | 0.20 | 0.54 |
| eurosat | mobilenetv3small | finetune | 12.8 | 15.4 | 64% (56 to 70) | 57% (49 to 63) | overlap | 87.7 | 87.7 | 0.20 | 0.55 |
| eurosat | mobilenetv3small | frozen | 8.7 | 10.6 | 75% (70 to 80) | 70% (65 to 75) | overlap | 84.3 | 84.3 | 0.20 | 0.64 |
| eurosat | resnet50 | finetune | 16.5 | 19.7 | 53% (43 to 61) | 45% (35 to 52) | overlap | 90.6 | 90.6 | 0.21 | 0.38 |
| eurosat | resnet50 | frozen | 11.9 | 14.1 | 66% (59 to 72) | 60% (53 to 66) | overlap | 89.6 | 89.6 | 0.20 | 0.40 |

### Backbones: fixed 60% target (where the saving disappears)

| Dataset | Backbone | Approach | Reduction: old (95% CI) | new (95% CI) | Intervals |
|---|---|---|---|---|---|
| flowers | efficientnetb0 | finetune | -37% (-117 to 13) | -92% (-124 to -61) | overlap; sign of interval 0->- |
| flowers | efficientnetb0 | frozen | -11% (-76 to 29) | -54% (-80 to -29) | overlap; sign of interval 0->- |
| flowers | mobilenetv2 | finetune | 21% (-41 to 53) | 2% (-16 to 17) | overlap |
| flowers | mobilenetv2 | frozen | 32% (-24 to 61) | 18% (3 to 31) | overlap; sign of interval 0->+ |
| flowers | mobilenetv3small | finetune | 21% (-23 to 49) | -17% (-38 to 1) | overlap |
| flowers | mobilenetv3small | frozen | 35% (-2 to 58) | 7% (-9 to 22) | overlap |
| flowers | resnet50 | finetune | -33% (-109 to 15) | -79% (-110 to -51) | overlap; sign of interval 0->- |
| flowers | resnet50 | frozen | -4% (-73 to 35) | -32% (-55 to -11) | overlap; sign of interval 0->- |
| eurosat | efficientnetb0 | finetune | -24% (-37 to -13) | -43% (-59 to -31) | overlap |
| eurosat | efficientnetb0 | frozen | -1% (-11 to 8) | -17% (-30 to -7) | overlap; sign of interval 0->- |
| eurosat | mobilenetv2 | finetune | 28% (19 to 36) | 16% (5 to 26) | overlap |
| eurosat | mobilenetv2 | frozen | 48% (43 to 53) | 38% (31 to 43) | overlap |
| eurosat | mobilenetv3small | finetune | 14% (4 to 22) | 1% (-11 to 10) | overlap; sign of interval +->0 |
| eurosat | mobilenetv3small | frozen | 44% (38 to 49) | 34% (27 to 40) | overlap |
| eurosat | resnet50 | finetune | -22% (-36 to -12) | -38% (-53 to -27) | overlap |
| eurosat | resnet50 | frozen | 12% (2 to 19) | 1% (-9 to 9) | overlap; sign of interval +->0 |

### Data efficiency: time to the primary target and to 50%

| Dataset | Images/class | Approach | Primary: reached (old / new) | Time, s: old | new | Reduction: old (95% CI) | new (95% CI) | 50% reduction: old | new | Test acc %: old | new |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flowers | 10 | scratch | 0/5 / 0/5 | >= 39.6 | >= 43.9 | baseline | baseline |  |  | 51.4 | 51.4 |
| flowers | 10 | finetune | 5/5 / 5/5 | 9.1 | 11.9 | >= 77% (76 to 78) | >= 73% (71 to 75) | 33% (25 to 41) | 22% (13 to 31) | 71.0 | 71.0 |
| flowers | 10 | frozen | 5/5 / 5/5 | 15.7 | 19.1 | >= 60% (48 to 71) | >= 56% (43 to 68) | 23% (14 to 32) | 14% (4 to 24) | 66.7 | 66.7 |
| flowers | 25 | scratch | 1/5 / 1/5 | >= 34.1 | >= 38.8 | baseline | baseline |  |  | 55.8 | 55.8 |
| flowers | 25 | finetune | 5/5 / 5/5 | 8.3 | 11.2 | 76% (73 to 78) | 71% (68 to 74) | 16% (11 to 23) | 4% (-2 to 10) | 76.6 | 76.6 |
| flowers | 25 | frozen | 5/5 / 5/5 | 8.3 | 10.9 | 76% (73 to 78) | 72% (69 to 75) | 17% (11 to 22) | 8% (1 to 14) | 73.5 | 73.5 |
| flowers | 50 | scratch | 3/5 / 3/5 | >= 27.2 | >= 30.2 | baseline | baseline |  |  | 59.9 | 59.9 |
| flowers | 50 | finetune | 5/5 / 5/5 | 8.4 | 10.7 | 69% (60 to 76) | 65% (54 to 72) | 9% (5 to 13) | -1% (-4 to 2) | 80.2 | 80.2 |
| flowers | 50 | frozen | 5/5 / 5/5 | 7.7 | 10.2 | 72% (63 to 77) | 66% (56 to 73) | 19% (16 to 22) | 7% (4 to 10) | 76.4 | 76.4 |
| flowers | 100 | scratch | 5/5 / 5/5 | 36.0 | 39.2 | baseline | baseline |  |  | 67.3 | 67.3 |
| flowers | 100 | finetune | 5/5 / 5/5 | 7.9 | 10.4 | 78% (73 to 81) | 74% (68 to 77) | 13% (10 to 15) | -0% (-3 to 2) | 82.7 | 82.7 |
| flowers | 100 | frozen | 5/5 / 5/5 | 7.2 | 9.9 | 80% (76 to 83) | 75% (70 to 78) | 29% (25 to 33) | 13% (11 to 15) | 79.4 | 79.4 |
| flowers | all | scratch | 5/5 / 5/5 | 25.9 | 28.5 | baseline | baseline |  |  | 66.9 | 66.9 |
| flowers | all | finetune | 5/5 / 5/5 | 7.9 | 11.0 | 70% (62 to 77) | 61% (53 to 70) | 6% (2 to 9) | -14% (-17 to -10) | 82.7 | 82.7 |
| flowers | all | frozen | 5/5 / 5/5 | 7.3 | 10.5 | 72% (65 to 79) | 63% (53 to 72) | 22% (19 to 26) | 5% (2 to 7) | 79.7 | 79.7 |
| eurosat | 10 | scratch | 0/5 / 0/5 | >= 32.9 | >= 34.4 | baseline | baseline |  |  | 57.5 | 57.5 |
| eurosat | 10 | finetune | 5/5 / 5/5 | 14.1 | 17.0 | >= 57% (49 to 65) | >= 51% (44 to 57) | 29% (17 to 39) | 15% (5 to 23) | 76.7 | 76.7 |
| eurosat | 10 | frozen | 5/5 / 5/5 | 12.5 | 14.8 | >= 62% (56 to 69) | >= 57% (52 to 63) | 37% (24 to 47) | 26% (19 to 33) | 74.5 | 74.5 |
| eurosat | 25 | scratch | 1/5 / 1/5 | >= 37.8 | >= 40.5 | baseline | baseline |  |  | 63.3 | 63.3 |
| eurosat | 25 | finetune | 5/5 / 5/5 | 10.7 | 13.3 | 72% (69 to 74) | 67% (64 to 70) | 24% (17 to 30) | 14% (6 to 20) | 81.3 | 81.3 |
| eurosat | 25 | frozen | 5/5 / 5/5 | 8.8 | 10.9 | 77% (76 to 78) | 73% (70 to 75) | 33% (30 to 36) | 24% (17 to 31) | 81.2 | 81.2 |
| eurosat | 50 | scratch | 5/5 / 5/5 | 34.5 | 35.1 | baseline | baseline |  |  | 70.6 | 70.6 |
| eurosat | 50 | finetune | 5/5 / 5/5 | 11.6 | 13.1 | 66% (58 to 72) | 63% (56 to 68) | 9% (-7 to 23) | 2% (-13 to 14) | 85.2 | 85.2 |
| eurosat | 50 | frozen | 5/5 / 5/5 | 8.1 | 10.0 | 76% (71 to 80) | 71% (67 to 75) | 27% (15 to 38) | 16% (7 to 24) | 85.7 | 85.7 |
| eurosat | all | scratch | 5/5 / 5/5 | 36.7 | 37.4 | baseline | baseline |  |  | 72.0 | 72.0 |
| eurosat | all | finetune | 5/5 / 5/5 | 11.3 | 13.7 | 69% (64 to 73) | 63% (55 to 70) | 24% (19 to 29) | 10% (-3 to 18) | 85.7 | 85.7 |
| eurosat | all | frozen | 5/5 / 5/5 | 9.1 | 10.7 | 75% (70 to 79) | 71% (65 to 76) | 34% (24 to 41) | 25% (14 to 31) | 85.4 | 85.4 |

### Ablations: time to the primary target (3 seeds, wide intervals)

#### depth

| Dataset | Approach | unfreeze_blocks | Reached: old / new | Time, s: old | new | Test acc %: old | new |
|---|---|---|---|---|---|---|---|
| flowers | finetune | 1 | 3/3 / 3/3 | 7.5 | 9.0 | 82.4 | 82.4 |
| flowers | finetune | 2 | 3/3 / 3/3 | 7.7 | 9.4 | 82.4 | 82.4 |
| flowers | finetune | 4 | 3/3 / 3/3 | 8.2 | 10.0 | 82.9 | 82.9 |
| flowers | finetune | 8 | 3/3 / 3/3 | 11.5 | 13.7 | 83.3 | 83.3 |
| flowers | finetune | all | 3/3 / 3/3 | 18.5 | 21.8 | 81.9 | 81.9 |
| eurosat | finetune | 1 | 3/3 / 3/3 | 6.1 | 9.6 | 85.6 | 85.6 |
| eurosat | finetune | 2 | 3/3 / 3/3 | 8.2 | 11.0 | 84.6 | 84.6 |
| eurosat | finetune | 4 | 3/3 / 3/3 | 9.6 | 13.9 | 86.1 | 86.1 |
| eurosat | finetune | 8 | 3/3 / 3/3 | 11.5 | 16.2 | 84.7 | 84.7 |
| eurosat | finetune | all | 3/3 / 3/3 | 24.1 | 31.5 | 81.7 | 81.7 |

#### lr

| Dataset | Approach | learning_rate | Reached: old / new | Time, s: old | new | Test acc %: old | new |
|---|---|---|---|---|---|---|---|
| flowers | finetune | 0.0001 | 3/3 / 3/3 | 8.4 | 10.1 | 83.9 | 83.9 |
| flowers | finetune | 0.0003 | 3/3 / 3/3 | 8.4 | 10.5 | 82.9 | 82.9 |
| flowers | finetune | 0.001 | 0/3 / 0/3 | >= 53.0 | >= 60.5 | 20.0 | 20.0 |
| flowers | finetune | 0.003 | 0/3 / 0/3 | >= 54.0 | >= 60.4 | 20.0 | 20.0 |
| flowers | frozen | 0.0001 | 3/3 / 3/3 | 21.8 | 25.0 | 72.9 | 72.9 |
| flowers | frozen | 0.0003 | 3/3 / 3/3 | 11.3 | 13.6 | 79.6 | 79.6 |
| flowers | frozen | 0.001 | 3/3 / 3/3 | 8.1 | 10.9 | 80.0 | 80.0 |
| flowers | frozen | 0.003 | 3/3 / 3/3 | 6.9 | 8.8 | 79.2 | 79.2 |
| flowers | scratch | 0.0001 | 3/3 / 3/3 | 33.3 | 32.4 | 66.5 | 66.5 |
| flowers | scratch | 0.0003 | 3/3 / 3/3 | 22.1 | 22.7 | 66.4 | 66.4 |
| flowers | scratch | 0.001 | 3/3 / 3/3 | 29.5 | 29.4 | 64.2 | 64.2 |
| flowers | scratch | 0.003 | 3/3 / 3/3 | 27.0 | 27.4 | 66.4 | 66.4 |

#### size

| Dataset | Approach | image_size | Reached: old / new | Time, s: old | new | Test acc %: old | new |
|---|---|---|---|---|---|---|---|
| flowers | finetune | 64 | 3/3 / 3/3 | 11.7 | 13.6 | 80.7 | 80.7 |
| flowers | finetune | 96 | 3/3 / 3/3 | 9.1 | 11.0 | 82.9 | 82.9 |
| flowers | finetune | 128 | 3/3 / 3/3 | 10.9 | 13.8 | 86.9 | 86.9 |
| flowers | frozen | 64 | 3/3 / 3/3 | 8.6 | 10.5 | 75.6 | 75.6 |
| flowers | frozen | 96 | 3/3 / 3/3 | 8.9 | 10.4 | 80.0 | 80.0 |
| flowers | frozen | 128 | 3/3 / 3/3 | 8.7 | 10.8 | 85.1 | 85.1 |
| flowers | scratch | 64 | 3/3 / 3/3 | 12.9 | 13.0 | 65.4 | 65.4 |
| flowers | scratch | 96 | 3/3 / 3/3 | 22.7 | 23.1 | 66.4 | 66.4 |
| flowers | scratch | 128 | 3/3 / 3/3 | 42.7 | 40.1 | 67.1 | 67.1 |

