### Ablations (MobileNetV2; 3 seeds each)

#### Blocks fine-tuned

Primary targets used: flowers 65.2%, eurosat 70.3%.

| Dataset | Approach | Blocks fine-tuned | Trainable params | Final val acc, % (95% CI) | Test acc, % (95% CI) | Reached | Time to primary target, s (95% CI) | Epochs | Total train, s |
|---|---|---|---|---|---|---|---|---|---|
| eurosat | Fine-tune | 1 | 891,850 | 83.7 (81.7 to 86.3) | 85.6 (85.0 to 85.8) | 3/3 | 9.6 (8.9 to 10.2) | 1.0 | 58 |
| eurosat | Fine-tune | 2 | 1,207,690 | 85.0 (80.8 to 87.9) | 84.6 (83.7 to 85.8) | 3/3 | 11.0 (10.1 to 11.8) | 1.7 | 58 |
| eurosat | Fine-tune | 4 | 1,676,170 | 86.8 (83.3 to 88.7) | 86.1 (84.2 to 87.9) | 3/3 | 13.9 (12.3 to 14.8) | 2.7 | 62 |
| eurosat | Fine-tune | 8 | 2,025,226 | 85.1 (77.9 to 89.2) | 84.7 (80.4 to 87.1) | 3/3 | 16.2 (14.0 to 18.4) | 3.0 | 69 |
| eurosat | Fine-tune | all | 2,202,570 | 83.6 (75.8 to 91.7) | 81.7 (73.3 to 88.7) | 3/3 | 31.5 (28.0 to 33.3) | 6.3 | 97 |
| flowers | Fine-tune | 1 | 885,445 | 84.3 (82.5 to 86.3) | 82.4 (82.1 to 82.5) | 3/3 | 9.0 (9.0 to 9.1) | 1.0 | 57 |
| flowers | Fine-tune | 2 | 1,201,285 | 82.6 (80.0 to 85.0) | 82.4 (78.3 to 84.6) | 3/3 | 9.4 (9.3 to 9.5) | 1.0 | 59 |
| flowers | Fine-tune | 4 | 1,669,765 | 83.9 (82.5 to 85.4) | 82.9 (82.9 to 82.9) | 3/3 | 10.0 (9.9 to 10.1) | 1.0 | 60 |
| flowers | Fine-tune | 8 | 2,018,821 | 83.3 (81.7 to 85.4) | 83.3 (82.9 to 83.7) | 3/3 | 13.7 (11.7 to 15.9) | 2.0 | 69 |
| flowers | Fine-tune | all | 2,196,165 | 82.5 (80.0 to 85.4) | 81.9 (78.3 to 84.2) | 3/3 | 21.8 (18.2 to 23.7) | 3.3 | 94 |

#### Learning rate

Primary targets used: flowers 65.2%, eurosat 70.3%.

| Dataset | Approach | Learning rate | Trainable params | Final val acc, % (95% CI) | Test acc, % (95% CI) | Reached | Time to primary target, s (95% CI) | Epochs | Total train, s |
|---|---|---|---|---|---|---|---|---|---|
| flowers | From scratch | 0.0001 | 241,829 | 67.2 (64.2 to 71.2) | 66.5 (64.6 to 69.2) | 3/3 | 32.4 (23.8 to 41.6) | 10.0 | 90 |
| flowers | From scratch | 0.0003 | 241,829 | 68.3 (62.5 to 73.3) | 66.4 (65.0 to 68.8) | 3/3 | 22.7 (20.5 to 24.0) | 6.7 | 90 |
| flowers | From scratch | 0.001 | 241,829 | 66.0 (61.3 to 69.6) | 64.2 (62.5 to 67.5) | 3/3 | 29.4 (23.7 to 32.8) | 9.0 | 90 |
| flowers | From scratch | 0.003 | 241,829 | 66.1 (62.9 to 70.0) | 66.4 (60.4 to 71.7) | 3/3 | 27.4 (20.4 to 38.4) | 8.3 | 90 |
| flowers | Frozen | 0.0001 | 6,405 | 78.1 (76.7 to 79.6) | 72.9 (71.7 to 74.2) | 3/3 | 25.0 (21.7 to 27.5) | 12.0 | 52 |
| flowers | Frozen | 0.0003 | 6,405 | 81.2 (80.4 to 82.5) | 79.6 (78.3 to 80.4) | 3/3 | 13.6 (13.1 to 14.4) | 4.3 | 51 |
| flowers | Frozen | 0.001 | 6,405 | 80.6 (79.6 to 81.2) | 80.0 (78.3 to 81.2) | 3/3 | 10.9 (10.2 to 12.2) | 2.0 | 52 |
| flowers | Frozen | 0.003 | 6,405 | 80.8 (80.0 to 81.7) | 79.2 (78.8 to 79.6) | 3/3 | 8.8 (8.7 to 8.9) | 1.0 | 52 |
| flowers | Fine-tune | 0.0001 | 1,669,765 | 84.3 (83.7 to 84.6) | 83.9 (83.3 to 84.2) | 3/3 | 10.1 (10.0 to 10.5) | 1.0 | 60 |
| flowers | Fine-tune | 0.0003 | 1,669,765 | 83.9 (82.5 to 85.4) | 82.9 (82.9 to 82.9) | 3/3 | 10.5 (10.5 to 10.6) | 1.0 | 59 |
| flowers | Fine-tune | 0.001 | 1,669,765 | 20.0 (20.0 to 20.0) | 20.0 (20.0 to 20.0) | 0/3 | >= 60.5 (59.5 to 61.4) | 30.0 | 61 |
| flowers | Fine-tune | 0.003 | 1,669,765 | 20.0 (20.0 to 20.0) | 20.0 (20.0 to 20.0) | 0/3 | >= 60.4 (59.5 to 61.2) | 30.0 | 60 |

#### Image size, px

Primary targets used: flowers 65.2%, eurosat 70.3%.

| Dataset | Approach | Image size, px | Trainable params | Final val acc, % (95% CI) | Test acc, % (95% CI) | Reached | Time to primary target, s (95% CI) | Epochs | Total train, s |
|---|---|---|---|---|---|---|---|---|---|
| flowers | From scratch | 64 | 241,829 | 66.4 (62.9 to 69.2) | 65.4 (64.6 to 67.1) | 3/3 | 13.0 (8.7 to 18.6) | 7.7 | 40 |
| flowers | From scratch | 96 | 241,829 | 68.3 (62.5 to 73.3) | 66.4 (65.0 to 68.8) | 3/3 | 23.1 (21.4 to 24.7) | 6.7 | 91 |
| flowers | From scratch | 128 | 241,829 | 67.8 (63.7 to 72.9) | 67.1 (63.3 to 69.6) | 3/3 | 40.1 (36.9 to 43.4) | 6.7 | 168 |
| flowers | Frozen | 64 | 6,405 | 74.6 (73.8 to 75.4) | 75.6 (75.4 to 75.8) | 3/3 | 10.5 (8.9 to 11.7) | 3.0 | 36 |
| flowers | Frozen | 96 | 6,405 | 80.6 (79.6 to 81.2) | 80.0 (78.3 to 81.2) | 3/3 | 10.4 (10.2 to 10.6) | 2.0 | 51 |
| flowers | Frozen | 128 | 6,405 | 85.8 (85.4 to 86.3) | 85.1 (85.0 to 85.4) | 3/3 | 10.8 (9.7 to 12.5) | 1.3 | 78 |
| flowers | Fine-tune | 64 | 1,669,765 | 78.1 (76.7 to 79.2) | 80.7 (79.6 to 81.2) | 3/3 | 13.6 (12.2 to 15.0) | 3.3 | 50 |
| flowers | Fine-tune | 96 | 1,669,765 | 83.9 (82.5 to 85.4) | 82.9 (82.9 to 82.9) | 3/3 | 11.0 (10.2 to 11.7) | 1.0 | 59 |
| flowers | Fine-tune | 128 | 1,669,765 | 86.0 (84.2 to 87.5) | 86.9 (83.3 to 89.6) | 3/3 | 13.8 (11.9 to 14.9) | 1.7 | 91 |
