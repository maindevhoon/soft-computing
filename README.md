# Adaptive Vehicle Suspension Using GA-Optimized Fuzzy Control

Fundamentals of Soft Computing Lab (26B16CS311). This project compares passive suspension, manual fuzzy control, two fixed GA-fuzzy controllers, and an adaptive two-mode controller in a quarter-car simulation. A sixth comparison holds the best training-selected adaptive mode fixed.

## Run the project

Follow [readme.txt](readme.txt) to set up Python 3.13 and install the pinned dependencies. Start the keyboard dashboard with `python -m suspension demo`. The supplied calibration and result files let you demonstrate the experiment immediately after installation. The dashboard supports terminal use on macOS and Linux. Batch mode exports the same metrics without curses.

## Experiment

- **Plant.** Two masses connected by a spring and damper, with a tire spring and bounded actuator force. A SciPy zero-order-hold transition advances the linear model every 20 ms.
- **Controller.** The Mamdani rule base takes delayed body acceleration, body velocity, and relative suspension velocity. It aggregates clipped consequent sets and computes a numerical centroid. The adaptive force blends a comfort and a travel-protection mode using a 300 ms activity window.
- **Training.** Three tuning road profiles at 240, 300, and 360 kg. Four GA objectives use 30 candidates, 40 generations, and seeds 11, 23, 37, 53, and 71. Numba accelerates the repeated simulation loops.
- **Evaluation.** Twelve test cases use new bump and sine settings, a held-out random seed, and a mixed road sequence, each at the same three loads. The controller configuration is frozen before scoring them. Passive and manual results are deterministic and repeated across seed rows to make paired comparisons easy.

## Results included in the package

Across the held-out cases, adaptive mean RMS body acceleration was **0.479 m/s²**, compared with **0.866 m/s²** for passive and **0.463 m/s²** for the fixed controller tuned with the new objective. Adaptation used less actuator effort and had lower RMS tire deflection than that fixed controller. The report explains the trade-offs and the fixed-mode ablation.

| File | Contents |
| --- | --- |
| `results/calibration.json` | Frozen controller parameters, thresholds, seed histories, and demo seed |
| `results/metrics.csv` | One row per controller, seed, and held-out road/load case |
| `results/summary.csv` | Overall controller averages and violation counts |
| `results/seed_summary.csv` | Per-seed averages and violation counts |
| `results/scenario_summary.csv` | Per-case averages across the GA seeds |
| `results/comparison.png` | Summary of comfort, travel, tire deflection, and effort |
| `results/mixed_trace.png` | Mixed-road time traces and adaptive blend factor |
| `results/ga_convergence.png` | Training cost histories across seeds |

The fixed travel-focused GA baseline shares the motivation of the selected paper and uses this project's common model and implementation. [Pekgökgöz et al. (2010)](https://dergipark.org.tr/en/pub/ijeas/article/251137)
