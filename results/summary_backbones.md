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
