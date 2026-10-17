### Backbone comparison (all-data training, 30 epochs)

#### flowers (primary target 65.2% validation accuracy)

| Backbone | Approach | Trainable params | Total params | Test acc, % (95% CI) | Reached | Time to 65.2% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch 1, s | Steady epoch, s |
|---|---|---|---|---|---|---|---|---|---|---|
| small CNN | From scratch | 241,829 | 242,533 | 66.9 (65.3 to 68.6) | 5/5 | 28.0 (22.9 to 36.6) | 8.2 | baseline | 6.7 | 2.95 |
| MobileNetV2 | Frozen | 6,405 | 2,264,389 | 79.7 (78.8 to 80.6) | 5/5 | 10.0 (9.4 to 10.5) | 1.6 | 64% (56 to 73%) | 9.1 | 1.47 |
| MobileNetV2 | Fine-tune | 1,669,765 | 2,264,389 | 82.7 (81.9 to 83.4) | 5/5 | 11.3 (10.9 to 11.8) | 1.0 | 60% (50 to 69%) | 11.3 | 1.70 |
| MobileNetV3Small | Frozen | 2,885 | 942,005 | 79.7 (78.9 to 80.3) | 5/5 | 11.6 (11.0 to 12.1) | 4.0 | 59% (49 to 68%) | 8.9 | 0.89 |
| MobileNetV3Small | Fine-tune | 761,013 | 942,005 | 82.1 (81.2 to 82.9) | 5/5 | 14.2 (14.0 to 14.3) | 2.0 | 49% (38 to 61%) | 12.9 | 1.21 |
| EfficientNetB0 | Frozen | 6,405 | 4,055,976 | 82.2 (81.4 to 82.8) | 5/5 | 18.5 (17.5 to 19.5) | 1.4 | 34% (18 to 50%) | 17.6 | 2.24 |
| EfficientNetB0 | Fine-tune | 2,876,869 | 4,055,976 | 82.5 (81.4 to 83.4) | 5/5 | 21.9 (21.9 to 22.1) | 1.0 | 22% (4 to 40%) | 21.9 | 2.96 |
| ResNet50 | Frozen | 10,245 | 23,597,957 | 83.7 (82.7 to 84.6) | 5/5 | 15.1 (15.0 to 15.2) | 1.0 | 46% (34 to 59%) | 15.1 | 4.06 |
| ResNet50 | Fine-tune | 16,079,365 | 23,597,957 | 85.3 (84.7 to 85.9) | 5/5 | 20.5 (20.5 to 20.7) | 1.0 | 27% (10 to 44%) | 20.5 | 6.13 |

#### flowers, fixed target 60% validation accuracy

| Backbone | Approach | Reached | Time to 60.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 5/5 | 11.5 (9.8 to 13.6) | 2.6 | baseline | baseline |
| MobileNetV2 | Frozen | 5/5 | 9.4 (9.0 to 10.0) | 1.2 | 18% (3 to 31%) | 54% (30 to 69%) |
| MobileNetV2 | Fine-tune | 5/5 | 11.3 (10.9 to 11.8) | 1.0 | 2% (-16 to 17%) | 62% (50 to 71%) |
| MobileNetV3Small | Frozen | 5/5 | 10.7 (10.6 to 10.7) | 3.0 | 7% (-9 to 22%) | -15% (-50 to 12%) |
| MobileNetV3Small | Fine-tune | 5/5 | 13.4 (12.9 to 14.0) | 1.4 | -17% (-38 to 1%) | 46% (20 to 64%) |
| EfficientNetB0 | Frozen | 5/5 | 17.6 (17.5 to 17.8) | 1.0 | -54% (-80 to -29%) | 62% (50 to 71%) |
| EfficientNetB0 | Fine-tune | 5/5 | 21.9 (21.9 to 22.1) | 1.0 | -92% (-124 to -61%) | 62% (50 to 71%) |
| ResNet50 | Frozen | 5/5 | 15.1 (15.0 to 15.2) | 1.0 | -32% (-55 to -11%) | 62% (50 to 71%) |
| ResNet50 | Fine-tune | 5/5 | 20.5 (20.5 to 20.7) | 1.0 | -79% (-110 to -51%) | 62% (50 to 71%) |

#### flowers, fixed target 80% validation accuracy

| Backbone | Approach | Reached | Time to 80.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 0/5 | >= 92.3 (91.4 to 93.6) | 30.0 | baseline | baseline |
| MobileNetV2 | Frozen | 5/5 | 20.5 (15.8 to 28.8) | 8.8 | >= 78% (69 to 83%) | >= 71% (52 to 81%) |
| MobileNetV2 | Fine-tune | 5/5 | 16.7 (14.7 to 19.0) | 4.0 | >= 82% (79 to 84%) | >= 87% (84 to 89%) |
| MobileNetV3Small | Frozen | 4/5 | >= 31.1 (27.6 to 34.1) | 25.8 | 66% (63 to 70%) (both censored) | 14% (3 to 26%) (both censored) |
| MobileNetV3Small | Fine-tune | 5/5 | 17.8 (16.6 to 18.8) | 5.0 | >= 81% (80 to 82%) | >= 83% (81 to 87%) |
| EfficientNetB0 | Frozen | 5/5 | 23.8 (21.8 to 25.8) | 3.8 | >= 74% (72 to 76%) | >= 87% (85 to 90%) |
| EfficientNetB0 | Fine-tune | 5/5 | 23.1 (22.0 to 24.2) | 1.4 | >= 75% (74 to 76%) | >= 95% (94 to 97%) |
| ResNet50 | Frozen | 5/5 | 30.0 (23.9 to 36.2) | 4.6 | >= 67% (61 to 74%) | >= 85% (80 to 89%) |
| ResNet50 | Fine-tune | 5/5 | 24.3 (21.8 to 26.7) | 1.6 | >= 74% (71 to 76%) | >= 95% (93 to 96%) |

#### eurosat (primary target 70.3% validation accuracy)

| Backbone | Approach | Trainable params | Total params | Test acc, % (95% CI) | Reached | Time to 70.3% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch 1, s | Steady epoch, s |
|---|---|---|---|---|---|---|---|---|---|---|
| small CNN | From scratch | 242,474 | 243,178 | 72.0 (70.2 to 73.8) | 5/5 | 35.6 (30.2 to 41.3) | 10.8 | baseline | 6.6 | 2.95 |
| MobileNetV2 | Frozen | 12,810 | 2,270,794 | 85.4 (84.9 to 85.8) | 5/5 | 10.4 (10.2 to 10.5) | 2.0 | 71% (66 to 75%) | 8.8 | 1.52 |
| MobileNetV2 | Fine-tune | 1,676,170 | 2,270,794 | 85.7 (83.9 to 87.6) | 5/5 | 13.4 (12.3 to 14.4) | 2.6 | 62% (55 to 68%) | 10.7 | 1.70 |
| MobileNetV3Small | Frozen | 5,770 | 944,890 | 84.3 (83.3 to 85.0) | 5/5 | 10.6 (10.2 to 11.1) | 3.4 | 70% (65 to 75%) | 8.5 | 0.88 |
| MobileNetV3Small | Fine-tune | 763,898 | 944,890 | 87.7 (86.6 to 88.7) | 5/5 | 15.4 (14.8 to 16.0) | 3.6 | 57% (49 to 63%) | 12.2 | 1.22 |
| EfficientNetB0 | Frozen | 12,810 | 4,062,381 | 85.1 (84.3 to 85.8) | 5/5 | 19.0 (18.5 to 19.4) | 2.0 | 47% (37 to 54%) | 16.8 | 2.23 |
| EfficientNetB0 | Fine-tune | 2,883,274 | 4,062,381 | 89.1 (88.2 to 90.3) | 5/5 | 23.7 (21.5 to 26.2) | 1.8 | 34% (20 to 45%) | 20.4 | 3.01 |
| ResNet50 | Frozen | 20,490 | 23,608,202 | 89.6 (89.1 to 90.3) | 5/5 | 14.1 (14.0 to 14.2) | 1.0 | 60% (53 to 66%) | 14.1 | 4.03 |
| ResNet50 | Fine-tune | 16,089,610 | 23,608,202 | 90.6 (89.7 to 91.4) | 5/5 | 19.7 (19.5 to 19.8) | 1.0 | 45% (35 to 52%) | 19.7 | 6.31 |

#### eurosat, fixed target 60% validation accuracy

| Backbone | Approach | Reached | Time to 60.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 5/5 | 14.3 (12.9 to 15.5) | 3.6 | baseline | baseline |
| MobileNetV2 | Frozen | 5/5 | 8.8 (8.7 to 8.9) | 1.0 | 38% (31 to 43%) | 72% (69 to 75%) |
| MobileNetV2 | Fine-tune | 5/5 | 12.0 (11.0 to 12.7) | 1.8 | 16% (5 to 26%) | 50% (38 to 61%) |
| MobileNetV3Small | Frozen | 5/5 | 9.4 (9.2 to 9.5) | 2.0 | 34% (27 to 40%) | 44% (38 to 50%) |
| MobileNetV3Small | Fine-tune | 5/5 | 14.2 (13.5 to 14.8) | 2.6 | 1% (-11 to 10%) | 28% (12 to 42%) |
| EfficientNetB0 | Frozen | 5/5 | 16.8 (16.3 to 17.2) | 1.0 | -17% (-30 to -7%) | 72% (69 to 75%) |
| EfficientNetB0 | Fine-tune | 5/5 | 20.4 (20.1 to 20.8) | 1.0 | -43% (-59 to -31%) | 72% (69 to 75%) |
| ResNet50 | Frozen | 5/5 | 14.1 (14.0 to 14.2) | 1.0 | 1% (-9 to 9%) | 72% (69 to 75%) |
| ResNet50 | Fine-tune | 5/5 | 19.7 (19.5 to 19.8) | 1.0 | -38% (-53 to -27%) | 72% (69 to 75%) |

#### eurosat, fixed target 80% validation accuracy

| Backbone | Approach | Reached | Time to 80.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 0/5 | >= 92.4 (90.2 to 94.6) | 30.0 | baseline | baseline |
| MobileNetV2 | Frozen | 5/5 | 12.5 (11.2 to 14.0) | 3.4 | >= 86% (85 to 88%) | >= 89% (85 to 91%) |
| MobileNetV2 | Fine-tune | 5/5 | 16.8 (15.0 to 18.4) | 4.6 | >= 82% (80 to 84%) | >= 85% (82 to 88%) |
| MobileNetV3Small | Frozen | 5/5 | 14.3 (13.3 to 15.2) | 7.4 | >= 85% (83 to 86%) | >= 75% (72 to 78%) |
| MobileNetV3Small | Fine-tune | 5/5 | 17.4 (16.8 to 18.0) | 5.2 | >= 81% (80 to 82%) | >= 83% (81 to 83%) |
| EfficientNetB0 | Frozen | 5/5 | 22.1 (21.3 to 23.0) | 3.4 | >= 76% (75 to 77%) | >= 89% (87 to 90%) |
| EfficientNetB0 | Fine-tune | 5/5 | 24.8 (23.0 to 28.0) | 2.2 | >= 73% (70 to 75%) | >= 93% (91 to 93%) |
| ResNet50 | Frozen | 5/5 | 18.2 (18.0 to 18.3) | 2.0 | >= 80% (80 to 81%) | >= 93% (93 to 93%) |
| ResNet50 | Fine-tune | 5/5 | 19.7 (19.5 to 19.8) | 1.0 | >= 79% (78 to 79%) | >= 97% (97 to 97%) |
