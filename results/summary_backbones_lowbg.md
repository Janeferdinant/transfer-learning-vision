### Backbone comparison (all-data training, 30 epochs)

#### flowers (primary target 65.1% validation accuracy)

| Backbone | Approach | Trainable params | Total params | Test acc, % (95% CI) | Reached | Time to 65.1% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch 1, s | Steady epoch, s |
|---|---|---|---|---|---|---|---|---|---|---|
| small CNN | From scratch | 241,829 | 242,533 | 67.3 (65.4 to 69.2) | 4/4 | 28.8 (21.4 to 39.6) | 8.5 | baseline | 5.6 | 3.01 |
| MobileNetV2 | Frozen | 6,405 | 2,264,389 | 79.3 (78.5 to 80.0) | 4/4 | 7.6 (6.7 to 8.5) | 1.5 | 74% (63 to 81%) | 6.9 | 1.46 |
| MobileNetV2 | Fine-tune | 1,669,765 | 2,264,389 | 82.7 (81.7 to 83.5) | 4/4 | 8.7 (8.3 to 9.2) | 1.0 | 70% (59 to 78%) | 8.7 | 1.67 |
| MobileNetV3Small | Frozen | 2,885 | 942,005 | 79.5 (78.8 to 80.2) | 4/4 | 9.3 (8.5 to 10.1) | 4.0 | 68% (55 to 77%) | 6.7 | 0.82 |
| MobileNetV3Small | Fine-tune | 761,013 | 942,005 | 81.9 (81.0 to 82.8) | 4/4 | 10.9 (10.5 to 11.3) | 2.0 | 62% (49 to 73%) | 9.8 | 1.17 |
| EfficientNetB0 | Frozen | 6,405 | 4,055,976 | 82.4 (81.7 to 82.9) | 3/3 | 13.9 (12.5 to 15.5) | 1.3 | 52% (33 to 65%) | 13.3 | 2.14 |
| EfficientNetB0 | Fine-tune | 2,876,869 | 4,055,976 | 82.6 (81.1 to 83.5) | 4/4 | 17.6 (15.4 to 20.6) | 1.0 | 39% (14 to 57%) | 17.6 | 2.88 |
| ResNet50 | Frozen | 10,245 | 23,597,957 | 84.0 (82.9 to 85.0) | 4/4 | 12.3 (11.2 to 13.7) | 1.0 | 57% (41 to 69%) | 12.3 | 4.01 |
| ResNet50 | Fine-tune | 16,079,365 | 23,597,957 | 85.2 (84.5 to 85.9) | 4/4 | 16.8 (15.9 to 17.4) | 1.0 | 42% (21 to 58%) | 16.8 | 6.33 |

#### flowers, fixed target 60% validation accuracy

| Backbone | Approach | Reached | Time to 60.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 4/4 | 10.0 (8.6 to 12.5) | 2.5 | baseline | baseline |
| MobileNetV2 | Frozen | 4/4 | 7.3 (6.5 to 8.0) | 1.2 | 28% (10 to 43%) | 50% (25 to 67%) |
| MobileNetV2 | Fine-tune | 4/4 | 8.7 (8.3 to 9.2) | 1.0 | 13% (-4 to 31%) | 60% (50 to 71%) |
| MobileNetV3Small | Frozen | 4/4 | 8.5 (8.1 to 9.2) | 3.0 | 15% (-3 to 32%) | -20% (-50 to 14%) |
| MobileNetV3Small | Fine-tune | 4/4 | 10.0 (9.4 to 10.6) | 1.2 | -0% (-20 to 20%) | 50% (25 to 67%) |
| EfficientNetB0 | Frozen | 3/3 | 13.3 (12.5 to 13.7) | 1.0 | -33% (-57 to -6%) | 60% (50 to 71%) |
| EfficientNetB0 | Fine-tune | 4/4 | 17.6 (15.4 to 20.6) | 1.0 | -76% (-123 to -37%) | 60% (50 to 71%) |
| ResNet50 | Frozen | 4/4 | 12.3 (11.2 to 13.7) | 1.0 | -23% (-52 to 3%) | 60% (50 to 71%) |
| ResNet50 | Fine-tune | 4/4 | 16.8 (15.9 to 17.4) | 1.0 | -68% (-99 to -33%) | 60% (50 to 71%) |

#### flowers, fixed target 80% validation accuracy

| Backbone | Approach | Reached | Time to 80.0% val, s (95% CI) | Epochs | Time reduction vs scratch | Epoch reduction vs scratch |
|---|---|---|---|---|---|---|
| small CNN | From scratch | 0/4 | >= 94.6 (89.3 to 99.6) | 30.0 | baseline | baseline |
| MobileNetV2 | Frozen | 4/4 | 19.2 (13.5 to 28.5) | 9.5 | >= 80% (70 to 86%) | >= 68% (45 to 82%) |
| MobileNetV2 | Fine-tune | 4/4 | 13.8 (12.0 to 15.6) | 4.0 | >= 85% (83 to 87%) | >= 87% (83 to 90%) |
| MobileNetV3Small | Frozen | 3/4 | >= 26.9 (23.0 to 30.9) | 25.2 | 72% (67 to 76%) (both censored) | 16% (2 to 30%) (both censored) |
| MobileNetV3Small | Fine-tune | 4/4 | 14.1 (13.0 to 15.0) | 4.8 | >= 85% (84 to 86%) | >= 84% (81 to 88%) |
| EfficientNetB0 | Frozen | 3/3 | 20.3 (17.7 to 21.7) | 4.3 | >= 79% (76 to 81%) | >= 86% (83 to 90%) |
| EfficientNetB0 | Fine-tune | 4/4 | 19.0 (16.6 to 23.1) | 1.5 | >= 80% (75 to 83%) | >= 95% (93 to 97%) |
| ResNet50 | Frozen | 4/4 | 27.1 (22.1 to 32.2) | 4.8 | >= 71% (65 to 77%) | >= 84% (78 to 90%) |
| ResNet50 | Fine-tune | 4/4 | 21.3 (18.3 to 23.5) | 1.8 | >= 77% (75 to 81%) | >= 94% (93 to 96%) |

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
