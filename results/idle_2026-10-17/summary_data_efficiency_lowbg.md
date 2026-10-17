### Data efficiency (MobileNetV2; training images per class vary, validation and test sets fixed)

#### flowers

Target `primary` = 65.2% validation accuracy

| Train images/class | Approach | Epoch budget | Test acc, % (95% CI) | Diff vs scratch, pts (95% CI) | Reached | Time to 65.2% val, s (95% CI) | Epochs | Time reduction vs scratch |
|---|---|---|---|---|---|---|---|---|
| 10 | From scratch | 100 | 51.4 (48.3 to 54.5) |  | 0/5 | >= 43.9 (42.8 to 44.9) | 100.0 | baseline |
| 10 | Frozen | 100 | 69.2 (69.2 to 69.2) | +17.7 (+14.7 to +20.8) | 1/1 | 16.1 (16.1 to 16.1) | 19.0 | >= 63% (62 to 64%) |
| 10 | Fine-tune | 100 | 69.6 (69.6 to 69.6) | +18.2 (+15.1 to +21.3) | 1/1 | 10.5 (10.5 to 10.5) | 4.0 | >= 76% (75 to 77%) |
| 25 | From scratch | 50 | 55.1 (52.7 to 57.7) |  | 1/4 | >= 37.6 (33.7 to 40.1) | 48.0 | baseline |
| 25 | Frozen | 50 | 68.3 (68.3 to 68.3) | +13.2 (+10.6 to +15.6) | 1/1 | 10.9 (10.9 to 10.9) | 6.0 | >= 71% (68 to 73%) |
| 25 | Fine-tune | 50 | 75.4 (70.8 to 80.0) | +20.3 (+14.0 to +26.7) | 2/2 | 11.4 (10.3 to 12.4) | 4.5 | >= 70% (65 to 74%) |
| 50 | From scratch | 30 | 59.9 (57.4 to 63.4) |  | 3/5 | >= 30.2 (23.5 to 36.9) | 22.4 | baseline |
| 50 | Fine-tune | 30 | 80.8 (78.8 to 82.9) | +20.9 (+17.0 to +24.2) | 3/3 | 11.3 (10.5 to 12.5) | 3.0 | >= 63% (51 to 70%) |
| 100 | From scratch | 30 | 67.3 (66.0 to 68.7) |  | 5/5 | 39.2 (32.8 to 45.6) | 16.4 | baseline |
| all | From scratch | 30 | 66.9 (65.3 to 68.6) |  | 5/5 | 28.5 (23.2 to 37.0) | 8.2 | baseline |

Target `fixed_0.50` = 50.0% validation accuracy

| Train images/class | Approach | Epoch budget | Test acc, % (95% CI) | Diff vs scratch, pts (95% CI) | Reached | Time to 50.0% val, s (95% CI) | Epochs | Time reduction vs scratch |
|---|---|---|---|---|---|---|---|---|
| 10 | From scratch | 100 | 51.4 (48.3 to 54.5) |  | 5/5 | 12.9 (11.6 to 14.5) | 23.4 | baseline |
| 10 | Frozen | 100 | 69.2 (69.2 to 69.2) | +17.7 (+14.7 to +20.8) | 1/1 | 10.9 (10.9 to 10.9) | 8.0 | 16% (6 to 25%) |
| 10 | Fine-tune | 100 | 69.6 (69.6 to 69.6) | +18.2 (+15.1 to +21.3) | 1/1 | 9.6 (9.6 to 9.6) | 2.0 | 26% (17 to 34%) |
| 25 | From scratch | 50 | 55.1 (52.7 to 57.7) |  | 4/4 | 10.1 (9.6 to 10.6) | 9.2 | baseline |
| 25 | Frozen | 50 | 68.3 (68.3 to 68.3) | +13.2 (+10.6 to +15.6) | 1/1 | 9.5 (9.5 to 9.5) | 4.0 | 5% (1 to 10%) |
| 25 | Fine-tune | 50 | 75.4 (70.8 to 80.0) | +20.3 (+14.0 to +26.7) | 2/2 | 9.6 (9.1 to 10.0) | 1.5 | 5% (-3 to 12%) |
| 50 | From scratch | 30 | 59.9 (57.4 to 63.4) |  | 5/5 | 9.5 (9.3 to 9.7) | 5.0 | baseline |
| 50 | Fine-tune | 30 | 80.8 (78.8 to 82.9) | +20.9 (+17.0 to +24.2) | 3/3 | 9.6 (9.3 to 9.8) | 1.0 | -1% (-5 to 2%) |
| 100 | From scratch | 30 | 67.3 (66.0 to 68.7) |  | 5/5 | 10.1 (9.9 to 10.3) | 3.0 | baseline |
| all | From scratch | 30 | 66.9 (65.3 to 68.6) |  | 5/5 | 9.7 (9.4 to 9.9) | 2.0 | baseline |

#### eurosat

Target `primary` = 70.3% validation accuracy

| Train images/class | Approach | Epoch budget | Test acc, % (95% CI) | Diff vs scratch, pts (95% CI) | Reached | Time to 70.3% val, s (95% CI) | Epochs | Time reduction vs scratch |
|---|---|---|---|---|---|---|---|---|
| 10 | From scratch | 50 | 57.5 (55.6 to 58.7) |  | 0/5 | >= 34.4 (32.6 to 36.8) | 50.0 | baseline |
| 10 | Fine-tune | 50 | 81.2 (81.2 to 81.2) | +23.8 (+22.6 to +25.7) | 1/1 | 18.9 (18.9 to 18.9) | 19.0 | >= 45% (42 to 49%) |
| 25 | From scratch | 30 | 64.6 (63.7 to 65.8) |  | 1/3 | >= 37.7 (35.2 to 40.1) | 29.3 | baseline |
| 25 | Frozen | 30 | 80.8 (80.8 to 80.8) | +16.3 (+15.0 to +17.1) | 1/1 | 9.8 (9.8 to 9.8) | 4.0 | >= 74% (72 to 76%) |
| 25 | Fine-tune | 30 | 81.9 (78.8 to 85.0) | +17.3 (+13.5 to +21.1) | 2/2 | 13.2 (13.1 to 13.4) | 5.5 | >= 65% (63 to 67%) |
| 50 | From scratch | 30 | 70.6 (69.5 to 71.8) |  | 5/5 | 35.1 (30.7 to 40.0) | 14.2 | baseline |
| 50 | Fine-tune | 30 | 87.9 (87.9 to 87.9) | +17.3 (+16.1 to +18.4) | 1/1 | 12.3 (12.3 to 12.3) | 3.0 | 65% (60 to 69%) |
| all | From scratch | 30 | 72.0 (70.2 to 73.8) |  | 5/5 | 37.4 (30.9 to 44.8) | 10.8 | baseline |

Target `fixed_0.50` = 50.0% validation accuracy

| Train images/class | Approach | Epoch budget | Test acc, % (95% CI) | Diff vs scratch, pts (95% CI) | Reached | Time to 50.0% val, s (95% CI) | Epochs | Time reduction vs scratch |
|---|---|---|---|---|---|---|---|---|
| 10 | From scratch | 50 | 57.5 (55.6 to 58.7) |  | 5/5 | 13.4 (12.2 to 14.5) | 16.2 | baseline |
| 10 | Fine-tune | 50 | 81.2 (81.2 to 81.2) | +23.8 (+22.6 to +25.7) | 1/1 | 10.5 (10.5 to 10.5) | 4.0 | 21% (14 to 28%) |
| 25 | From scratch | 30 | 64.6 (63.7 to 65.8) |  | 3/3 | 11.4 (10.1 to 12.5) | 6.7 | baseline |
| 25 | Frozen | 30 | 80.8 (80.8 to 80.8) | +16.3 (+15.0 to +17.1) | 1/1 | 8.3 (8.3 to 8.3) | 2.0 | 27% (19 to 34%) |
| 25 | Fine-tune | 30 | 81.9 (78.8 to 85.0) | +17.3 (+13.5 to +21.1) | 2/2 | 10.2 (10.1 to 10.3) | 2.0 | 10% (1 to 19%) |
| 50 | From scratch | 30 | 70.6 (69.5 to 71.8) |  | 5/5 | 11.1 (10.1 to 12.2) | 3.4 | baseline |
| 50 | Fine-tune | 30 | 87.9 (87.9 to 87.9) | +17.3 (+16.1 to +18.4) | 1/1 | 9.7 (9.7 to 9.7) | 1.0 | 13% (4 to 20%) |
| all | From scratch | 30 | 72.0 (70.2 to 73.8) |  | 5/5 | 12.2 (10.7 to 13.3) | 2.8 | baseline |
