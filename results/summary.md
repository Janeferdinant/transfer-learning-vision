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
