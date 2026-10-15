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
