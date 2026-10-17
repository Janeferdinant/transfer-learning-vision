# Run log: timed suites rerun on 2026-10-17 (rerun of the shared-CPU timings)

This log sits next to `2026-10-14-experiment.md` and `2026-10-15-extension.md`, which are unchanged. The rerun used the same code, seeds, epochs, learning rates (`configs/lr_choices.json`) and suites as the earlier run (`scripts/run_queue.sh`), through `scripts/run_queue_idle.sh`. Results: `results/idle_2026-10-17/` (CSVs, `summary_*.md/json`, `_lowbg` sensitivity files); console and idle-check logs: `docs/runs/idle-2026-10-17/`; old-versus-new tables: `docs/runs/idle-2026-10-17/comparison.md`; figures: `docs/figures/rerun-2026-10-17/`; explorer data: `docs/explorer-data-rerun-2026-10-17.js`.

## Headline: the machine was not idle, and the rerun was busier than the earlier run

The aim was a rerun on an idle machine. It did not get one. A program that belongs to the machine's owner (SanDisk Dashboard, using about 3 to 12 logical CPUs at below-normal priority; 343,676 CPU seconds in the 27 hours after it started) used CPU all day. It was not started by this work and was not stopped. The pre-run idle checks therefore never reached the 15% criterion, and after the 30-minute waiting period each batch ran anyway, as specified. The background CPU share recorded during the fits (`bg_cpu_fraction`) was **higher** than in the earlier run, so these numbers are a second measurement under load, not an idle-machine measurement. Read the two runs as a range; the earlier run was the quieter one.

## Environment

- Hardware: AMD Ryzen 7 5800X (8 cores, 16 threads), 31.9 GB RAM, Windows 11 Pro (`os` in `hardware.json` says "Windows 10", which is what Python reports for this build), desktop on AC power (no battery), power plan High performance (already active, not changed). Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, pandas 3.0.6, Pillow 12.3.0 (`docs/runs/idle-2026-10-17/machine.txt`).
- 318 timed training runs (backbones 90, data efficiency 135, ablations 93). The pilot (45 runs) is accuracy-only and was not rerun; its learning-rate choices are reused. The original 10-seed `tlv.experiment` run was not rerun either (its 5-seed counterpart in `backbones_flowers` was).
- A network interruption stopped the working session for about five hours on this day, before the first timed run of this repository. The first launch of the queue was stopped by hand during its first idle check and relaunched; no run had started. No run was cut off, retried or discarded: every suite finished on its first attempt (`docs/runs/idle-2026-10-17/*_console.txt`; TensorFlow start-up warnings, which contain local paths, were removed from these console logs).

## Idle checks and background load

Before each batch of runs a 60-second sample of total CPU use was taken (`scripts/idle_gate.ps1`: a mean above 15% means wait 5 minutes and re-check for up to 30 minutes, then run anyway). The top processes by CPU of every check are in `docs/runs/idle-2026-10-17/<suite>_idle_check.txt`: always `Dashboard` first, then `SignalRgb`, an IDE and `UsbMonitor`, with occasional `chrome`, `svchost` or `SearchIndexer`.

| Suite | Runs | Pre-run idle checks (60 s each): count, first start | Mean total CPU in the checks, % | Gate | Background CPU share during the runs: median (max) | Runs above 0.5 | Earlier run: median |
|---|---|---|---|---|---|---|---|
| backbones_flowers | 45 | 6, 08:37 | 28.8 to 30.1 | not met (above 15%), ran anyway after 30 min | 0.40 (0.71) | 16 | 0.27 |
| backbones_eurosat | 45 | 6, 10:26 | 29.7 to 59.6 | not met (above 15%), ran anyway after 30 min | 0.42 (0.71) | 18 | 0.21 |
| data_efficiency_flowers | 75 | 6, 12:16 | 29.9 to 47.9 | not met (above 15%), ran anyway after 30 min | 0.54 (0.67) | 43 | 0.18 |
| data_efficiency_eurosat | 60 | 6, 14:00 | 46.4 to 54.6 | not met (above 15%), ran anyway after 30 min | 0.57 (0.70) | 37 | 0.35 |
| ablation_depth_flowers | 15 | 6, 15:30 | 28.8 to 39.8 | not met (above 15%), ran anyway after 30 min | 0.43 (0.50) | 0 | 0.21 |
| ablation_lr_flowers | 36 | 6, 16:24 | 28.7 to 37.1 | not met (above 15%), ran anyway after 30 min | 0.51 (0.58) | 19 | 0.21 |
| ablation_size_flowers | 27 | 6, 17:43 | 28.0 to 43.8 | not met (above 15%), ran anyway after 30 min | 0.50 (0.70) | 13 | 0.26 |
| ablation_depth_eurosat | 15 | 6, 18:55 | 31.8 to 87.8 | not met (above 15%), ran anyway after 30 min | 0.57 (0.63) | 13 | 0.18 |

The background share is recorded for every fit in `per_run.csv` (`bg_cpu_fraction`). The runner's own wait-for-quiet step (threshold 0.10, at most 120 s per run, given up after two consecutive timeouts) was active as before and gave up at the start of every suite.

## Accuracy

Validation and test accuracies are identical to the earlier run for every (dataset, backbone, approach) and every ablation cell at the precision reported (same seeds; deterministic on this machine): `comparison.md` shows old and new test accuracies side by side. Epoch counts to each target are therefore also unchanged; only the seconds moved.

## Time to the primary target (95% of scratch's final validation accuracy): earlier run versus rerun

Time to target in seconds and reduction versus scratch (1 minus the ratio of mean times, 95% percentile bootstrap interval over 5 seeds), from `results/summary_backbones.json` and `results/idle_2026-10-17/summary_backbones.json`.

| Dataset | Backbone | Approach | Earlier run (shared CPU): time to target s, reduction (95% CI) | Rerun 2026-10-17: time to target s, reduction (95% CI) |
|---|---|---|---|---|
| TF Flowers | MobileNetV2 | frozen | 10.0 s, 70% (49 to 82) | 10.0 s, 64% (56 to 73) |
| TF Flowers | MobileNetV2 | fine-tuned | 10.6 s, 68% (49 to 79) | 11.3 s, 60% (50 to 69) |
| TF Flowers | MobileNetV3Small | frozen | 9.6 s, 71% (57 to 79) | 11.6 s, 59% (49 to 68) |
| TF Flowers | MobileNetV3Small | fine-tuned | 11.3 s, 67% (50 to 76) | 14.2 s, 49% (38 to 61) |
| TF Flowers | EfficientNetB0 | frozen | 15.8 s, 53% (29 to 67) | 18.5 s, 34% (18 to 50) |
| TF Flowers | EfficientNetB0 | fine-tuned | 18.3 s, 45% (18 to 61) | 21.9 s, 22% (4 to 40) |
| TF Flowers | ResNet50 | frozen | 14.0 s, 58% (36 to 72) | 15.1 s, 46% (34 to 59) |
| TF Flowers | ResNet50 | fine-tuned | 17.8 s, 47% (21 to 62) | 20.5 s, 27% (10 to 44) |
| TF Flowers | small CNN | scratch (baseline) | 33.6 s | 28.0 s |
| EuroSAT | MobileNetV2 | frozen | 8.3 s, 76% (71 to 80) | 10.4 s, 71% (66 to 75) |
| EuroSAT | MobileNetV2 | fine-tuned | 11.0 s, 69% (62 to 74) | 13.4 s, 62% (55 to 68) |
| EuroSAT | MobileNetV3Small | frozen | 8.7 s, 75% (70 to 80) | 10.6 s, 70% (65 to 75) |
| EuroSAT | MobileNetV3Small | fine-tuned | 12.8 s, 64% (56 to 70) | 15.4 s, 57% (49 to 63) |
| EuroSAT | EfficientNetB0 | frozen | 15.6 s, 56% (46 to 63) | 19.0 s, 47% (37 to 54) |
| EuroSAT | EfficientNetB0 | fine-tuned | 18.8 s, 46% (35 to 56) | 23.7 s, 34% (20 to 45) |
| EuroSAT | ResNet50 | frozen | 11.9 s, 66% (59 to 72) | 14.1 s, 60% (53 to 66) |
| EuroSAT | ResNet50 | fine-tuned | 16.5 s, 53% (43 to 61) | 19.7 s, 45% (35 to 52) |
| EuroSAT | small CNN | scratch (baseline) | 35.2 s | 35.6 s |

Primary targets are unchanged (flowers 65.2%, EuroSAT 70.3%). Pretrained-model times rose by 0% to 26% (median 16% on flowers, 22% on EuroSAT); the scratch CNN's time fell 17% on flowers and was flat on EuroSAT, so every reduction is smaller in the rerun, by 5 to 23 points. All old and new intervals overlap. The cause was not isolated; the rerun had a higher background CPU share, which is consistent with the pattern but was not tested.

## Did the conclusions change?

- **Direction and ranking: no.** At the primary target every backbone and both approaches on both datasets still reached the target faster than scratch, with intervals above zero (smallest: EfficientNetB0 fine-tuned on flowers, 22% (4 to 40)). Cheap MobileNets save most, heavy backbones least.
- **Size of the saving: lower.** Range over all 16 pairs: 22% to 71% (earlier run 45% to 76%). MobileNetV2: flowers 64% frozen and 60% fine-tuned (earlier 70% and 68%); EuroSAT 71% and 62% (earlier 76% and 69%).
- **Where it does not help: strengthened.** At the 60% target on EuroSAT, fine-tuned ResNet50 and EfficientNetB0 now take 38% and 43% longer than scratch (earlier 22% and 24%). On flowers the same comparison, earlier not different from zero, is now below zero for fine-tuned ResNet50 (-79%, -110 to -51) and EfficientNetB0 (-92%, -124 to -61).
- **Data efficiency:** accuracy statements are unchanged. Time-to-target savings at the primary target stay above zero at every size (rerun 51% to 75%, earlier 57% to 80%; scratch still never reached the target with 10 images per class and once with 25). At the lenient 50% target the saving of fine-tuned flowers is about zero at 25, 50 and 100 images per class and negative with all images (-14%, -17 to -10).
- **Ablations:** the ordering is unchanged (more unfrozen blocks cost more time; learning rates of 1e-3 and above still collapse fine-tuning to chance accuracy in 3 of 3 seeds; larger images cost the scratch CNN far more time than the pretrained models). Seconds for pretrained cells are 1.1x to 1.6x the earlier ones.

## Limitations of this rerun

- Not idle (see above); the earlier run (median background share 0.18 to 0.35) was the quieter of the two.
- One repetition per (configuration, seed), as before; the intervals are over seeds only.
- Seconds are comparable between suites only to the extent that background load was similar. The per-run `bg_cpu_fraction` allows a sensitivity analysis (`results/idle_2026-10-17/summary_*_lowbg.md`, runs above 0.5 dropped), which leaves too few runs in some cells to be useful.
