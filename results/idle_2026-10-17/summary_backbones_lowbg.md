### Backbone comparison (all-data training, 30 epochs)

#### flowers (primary target 65.2% validation accuracy)

| Backbone | Approach | Trainable params | Total params | Test acc, % (95% CI) | Reached | Time to 65.2% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch 1, s | Steady epoch, s |
|---|---|---|---|---|---|---|---|---|---|---|
| small CNN | From scratch | 241,829 | 242,533 | 66.9 (65.3 to 68.6) | 5/5 | 28.0 (22.9 to 36.6) | 8.2 | baseline | 6.7 | 2.95 |
| MobileNetV2 | Frozen | 6,405 | 2,264,389 | 78.3 (78.3 to 78.3) | 1/1 | 10.4 (10.4 to 10.4) | 2.0 | 63% (55 to 72%) | 8.9 | 1.49 |
| MobileNetV2 | Fine-tune | 1,669,765 | 2,264,389 | 82.9 (82.9 to 82.9) | 1/1 | 10.8 (10.8 to 10.8) | 1.0 | 61% (53 to 70%) | 10.8 | 1.69 |
| MobileNetV3Small | Frozen | 2,885 | 942,005 | 80.0 (80.0 to 80.0) | 1/1 | 11.6 (11.6 to 11.6) | 4.0 | 59% (50 to 68%) | 8.8 | 0.90 |
| MobileNetV3Small | Fine-tune | 761,013 | 942,005 | 83.1 (82.9 to 83.3) | 2/2 | 14.0 (13.9 to 14.1) | 2.0 | 50% (39 to 62%) | 12.8 | 1.24 |
| EfficientNetB0 | Frozen | 6,405 | 4,055,976 | 82.1 (81.2 to 82.9) | 4/4 | 18.1 (17.5 to 19.1) | 1.2 | 35% (20 to 51%) | 17.5 | 2.23 |
| EfficientNetB0 | Fine-tune | 2,876,869 | 4,055,976 | 82.5 (81.4 to 83.4) | 5/5 | 21.9 (21.9 to 22.1) | 1.0 | 22% (4 to 40%) | 21.9 | 2.96 |
| ResNet50 | Frozen | 10,245 | 23,597,957 | 83.7 (82.7 to 84.6) | 5/5 | 15.1 (15.0 to 15.2) | 1.0 | 46% (34 to 59%) | 15.1 | 4.06 |
| ResNet50 | Fine-tune | 16,079,365 | 23,597,957 | 85.3 (84.7 to 85.9) | 5/5 | 20.5 (20.5 to 20.7) | 1.0 | 27% (10 to 44%) | 20.5 | 6.13 |

#### flowers, fixed target 60% validation accuracy

| Backbone | Approach | Reached | Time to 60.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 5/5 | 11.5 (9.8 to 13.6) | 2.6 | baseline | baseline |
| MobileNetV2 | Frozen | 1/1 | 8.9 (8.9 to 8.9) | 1.0 | 22% (9 to 35%) | 62% (50 to 71%) |
| MobileNetV2 | Fine-tune | 1/1 | 10.8 (10.8 to 10.8) | 1.0 | 6% (-11 to 21%) | 62% (50 to 71%) |
| MobileNetV3Small | Frozen | 1/1 | 10.6 (10.6 to 10.6) | 3.0 | 7% (-9 to 22%) | -15% (-50 to 12%) |
| MobileNetV3Small | Fine-tune | 2/2 | 13.4 (12.7 to 14.1) | 1.5 | -17% (-38 to 3%) | 42% (9 to 67%) |
| EfficientNetB0 | Frozen | 4/4 | 17.5 (17.5 to 17.6) | 1.0 | -53% (-79 to -29%) | 62% (50 to 71%) |
| EfficientNetB0 | Fine-tune | 5/5 | 21.9 (21.9 to 22.1) | 1.0 | -92% (-124 to -61%) | 62% (50 to 71%) |
| ResNet50 | Frozen | 5/5 | 15.1 (15.0 to 15.2) | 1.0 | -32% (-55 to -11%) | 62% (50 to 71%) |
| ResNet50 | Fine-tune | 5/5 | 20.5 (20.5 to 20.7) | 1.0 | -79% (-110 to -51%) | 62% (50 to 71%) |

#### flowers, fixed target 80% validation accuracy

| Backbone | Approach | Reached | Time to 80.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 0/5 | >= 92.3 (91.4 to 93.6) | 30.0 | baseline | baseline |
| MobileNetV2 | Frozen | 1/1 | 14.8 (14.8 to 14.8) | 5.0 | >= 84% (84 to 84%) | >= 83% (83 to 83%) |
| MobileNetV2 | Fine-tune | 1/1 | 17.5 (17.5 to 17.5) | 5.0 | >= 81% (81 to 81%) | >= 83% (83 to 83%) |
| MobileNetV3Small | Frozen | 0/1 | >= 34.9 (34.9 to 34.9) | 30.0 | 62% (62 to 63%) (both censored) | 0% (0 to 0%) (both censored) |
| MobileNetV3Small | Fine-tune | 2/2 | 18.3 (17.7 to 19.0) | 5.5 | >= 80% (79 to 81%) | >= 82% (80 to 83%) |
| EfficientNetB0 | Frozen | 4/4 | 23.0 (21.8 to 25.4) | 3.5 | >= 75% (72 to 77%) | >= 88% (85 to 90%) |
| EfficientNetB0 | Fine-tune | 5/5 | 23.1 (22.0 to 24.2) | 1.4 | >= 75% (74 to 76%) | >= 95% (94 to 97%) |
| ResNet50 | Frozen | 5/5 | 30.0 (23.9 to 36.2) | 4.6 | >= 67% (61 to 74%) | >= 85% (80 to 89%) |
| ResNet50 | Fine-tune | 5/5 | 24.3 (21.8 to 26.7) | 1.6 | >= 74% (71 to 76%) | >= 95% (93 to 96%) |

#### eurosat (primary target 70.3% validation accuracy)

| Backbone | Approach | Trainable params | Total params | Test acc, % (95% CI) | Reached | Time to 70.3% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch 1, s | Steady epoch, s |
|---|---|---|---|---|---|---|---|---|---|---|
| small CNN | From scratch | 242,474 | 243,178 | 72.0 (70.2 to 73.8) | 5/5 | 35.6 (30.2 to 41.3) | 10.8 | baseline | 6.6 | 2.95 |
| MobileNetV2 | Frozen | 12,810 | 2,270,794 | 84.6 (84.6 to 84.6) | 1/1 | 10.1 (10.1 to 10.1) | 2.0 | 72% (67 to 76%) | 8.5 | 1.54 |
| MobileNetV2 | Fine-tune | 1,676,170 | 2,270,794 | 86.3 (86.3 to 86.3) | 1/1 | 11.7 (11.7 to 11.7) | 2.0 | 67% (61 to 72%) | 9.9 | 1.73 |
| MobileNetV3Small | Frozen | 5,770 | 944,890 | 84.6 (84.6 to 84.6) | 1/1 | 9.9 (9.9 to 9.9) | 3.0 | 72% (67 to 76%) | 8.1 | 0.89 |
| MobileNetV3Small | Fine-tune | 763,898 | 944,890 | 86.7 (86.3 to 87.1) | 2/2 | 14.8 (14.2 to 15.3) | 3.5 | 59% (51 to 65%) | 11.7 | 1.23 |
| EfficientNetB0 | Frozen | 12,810 | 4,062,381 | 84.2 (83.7 to 84.6) | 2/2 | 18.4 (18.2 to 18.5) | 2.0 | 48% (39 to 56%) | 16.1 | 2.22 |
| EfficientNetB0 | Fine-tune | 2,883,274 | 4,062,381 | 89.1 (88.2 to 90.3) | 5/5 | 23.7 (21.5 to 26.2) | 1.8 | 34% (20 to 45%) | 20.4 | 3.01 |
| ResNet50 | Frozen | 20,490 | 23,608,202 | 89.6 (89.1 to 90.3) | 5/5 | 14.1 (14.0 to 14.2) | 1.0 | 60% (53 to 66%) | 14.1 | 4.03 |
| ResNet50 | Fine-tune | 16,089,610 | 23,608,202 | 90.6 (89.7 to 91.4) | 5/5 | 19.7 (19.5 to 19.8) | 1.0 | 45% (35 to 52%) | 19.7 | 6.31 |

#### eurosat, fixed target 60% validation accuracy

| Backbone | Approach | Reached | Time to 60.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 5/5 | 14.3 (12.9 to 15.5) | 3.6 | baseline | baseline |
| MobileNetV2 | Frozen | 1/1 | 8.5 (8.5 to 8.5) | 1.0 | 40% (34 to 45%) | 72% (69 to 75%) |
| MobileNetV2 | Fine-tune | 1/1 | 9.9 (9.9 to 9.9) | 1.0 | 31% (23 to 36%) | 72% (69 to 75%) |
| MobileNetV3Small | Frozen | 1/1 | 9.0 (9.0 to 9.0) | 2.0 | 37% (30 to 42%) | 44% (38 to 50%) |
| MobileNetV3Small | Fine-tune | 2/2 | 13.5 (13.0 to 14.1) | 2.5 | 5% (-6 to 14%) | 31% (12 to 47%) |
| EfficientNetB0 | Frozen | 2/2 | 16.1 (16.0 to 16.2) | 1.0 | -13% (-25 to -4%) | 72% (69 to 75%) |
| EfficientNetB0 | Fine-tune | 5/5 | 20.4 (20.1 to 20.8) | 1.0 | -43% (-59 to -31%) | 72% (69 to 75%) |
| ResNet50 | Frozen | 5/5 | 14.1 (14.0 to 14.2) | 1.0 | 1% (-9 to 9%) | 72% (69 to 75%) |
| ResNet50 | Fine-tune | 5/5 | 19.7 (19.5 to 19.8) | 1.0 | -38% (-53 to -27%) | 72% (69 to 75%) |

#### eurosat, fixed target 80% validation accuracy

| Backbone | Approach | Reached | Time to 80.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 0/5 | >= 92.4 (90.2 to 94.6) | 30.0 | baseline | baseline |
| MobileNetV2 | Frozen | 1/1 | 11.6 (11.6 to 11.6) | 3.0 | >= 87% (87 to 88%) | >= 90% (90 to 90%) |
| MobileNetV2 | Fine-tune | 1/1 | 13.4 (13.4 to 13.4) | 3.0 | >= 86% (85 to 86%) | >= 90% (90 to 90%) |
| MobileNetV3Small | Frozen | 1/1 | 12.5 (12.5 to 12.5) | 6.0 | >= 86% (86 to 87%) | >= 80% (80 to 80%) |
| MobileNetV3Small | Fine-tune | 2/2 | 16.6 (16.5 to 16.7) | 5.0 | >= 82% (82 to 82%) | >= 83% (83 to 83%) |
| EfficientNetB0 | Frozen | 2/2 | 21.6 (20.7 to 22.5) | 3.5 | >= 77% (75 to 78%) | >= 88% (87 to 90%) |
| EfficientNetB0 | Fine-tune | 5/5 | 24.8 (23.0 to 28.0) | 2.2 | >= 73% (70 to 75%) | >= 93% (91 to 93%) |
| ResNet50 | Frozen | 5/5 | 18.2 (18.0 to 18.3) | 2.0 | >= 80% (80 to 81%) | >= 93% (93 to 93%) |
| ResNet50 | Fine-tune | 5/5 | 19.7 (19.5 to 19.8) | 1.0 | >= 79% (78 to 79%) | >= 97% (97 to 97%) |
