"""Held-out evaluation, tables, and plots."""

import csv
from pathlib import Path

import numpy as np

from .fuzzy import COMFORT_RULES, MANUAL, PROTECTION_RULES
from .model import TIME, TRAVEL_LIMIT, plant, scenarios
from .optimize import COSTS, cost
from .simulate import METHODS, measurements, simulate, simulate_raw

ABLATION = "Fixed mode ablation"
LABELS = ("Passive", "Manual", "GA travel", "GA fixed", "Adaptive", "Fixed mode")
COLORS = ("#596977", "#927440", "#7a69a9", "#387499", "#137d67", "#a56759")
FIELDS = ("rms_accel_m_s2", "peak_accel_m_s2", "peak_travel_m", "rms_tire_m",
          "rms_force_n", "saturation_fraction", "travel_excess_m", "mean_alpha")


def best_fixed_mode(run, thresholds):
    """Pick one of the adaptive controller's modes using tuning cases only."""
    candidates = (("comfort", COMFORT_RULES), ("protection", PROTECTION_RULES))
    scores = []
    for name, rules in candidates:
        genes = run["genes"][name]
        values = [cost(simulate_raw(*plant(case.sprung_mass), case.road, 1, genes, rules,
                                   MANUAL, thresholds), COSTS["fixed"]) for case in scenarios("train")]
        scores.append(0.75 * np.mean(values) + 0.25 * max(values))
    return candidates[int(np.argmin(scores))]


def traces(case, run, thresholds, ablation=None):
    curves = {name: simulate(case, name, run["genes"], thresholds) for name in METHODS}
    if ablation is not None:
        mode_name, rules = ablation
        curves[ABLATION] = simulate_raw(*plant(case.sprung_mass), case.road, 1,
                                        run["genes"][mode_name], rules, MANUAL, thresholds)
    return curves


def write_csv(path, records):
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize(records):
    out = {"runs": len(records)}
    out.update({field: float(np.mean([row[field] for row in records])) for field in FIELDS})
    out["travel_violations"] = sum(row["travel_excess_m"] > 0 for row in records)
    settling = [row["settling_s"] for row in records if row["settling_s"] is not None]
    out["mean_settling_s"] = float(np.mean(settling)) if settling else None
    out["settled_bump_runs"] = len(settling)
    out["bump_runs"] = sum(row["road"] == "bump" for row in records)
    return out


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def trace_figure(case, curves, path):
    plt = pyplot()
    fig, ax = plt.subplots(6, 1, figsize=(10, 10), sharex=True, layout="constrained")
    fig.suptitle(f"Held-out {case.name} road | sprung mass {case.sprung_mass} kg", fontweight="bold")
    ax[0].plot(TIME, 1000 * case.road, color="#455c6b")
    for index, style in ((0, "--"), (3, ":"), (4, "-")):
        name = METHODS[index]
        curve = curves[name]
        for axis, col, scale in zip(ax[1:5], range(4), (1, 1000, 1000, 1)):
            axis.plot(TIME[:-1], scale * curve[:, col], linestyle=style, lw=1.3,
                      color=COLORS[index], label=LABELS[index])
    for limit in (-TRAVEL_LIMIT, TRAVEL_LIMIT):
        ax[2].axhline(1000 * limit, color="#946655", ls="--", lw=.7)
    ax[5].plot(TIME[:-1], curves[METHODS[4]][:, 4], color=COLORS[4])
    ax[5].set_ylim(-.03, 1.03)
    for axis, label in zip(ax, ("Road (mm)", "Body accel. (m/s²)", "Travel (mm)",
                               "Tire defl. (mm)", "Force (N)", "Blend alpha")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.24)
    ax[1].legend(ncol=3, loc="upper right", fontsize=8)
    ax[-1].set_xlabel("Time (s)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summary_figure(rows, path, seed_count):
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.7), layout="constrained")
    fig.suptitle(f"Held-out road/load results | mean over {seed_count} GA seeds", fontweight="bold")
    for axis, (field, scale, title) in zip(axes.flat, (
        ("rms_accel_m_s2", 1, "RMS body acceleration (m/s²)"),
        ("peak_travel_m", 1000, "Peak suspension travel (mm)"),
        ("rms_tire_m", 1000, "RMS tire deflection (mm)"),
        ("rms_force_n", 1, "RMS actuator force (N)"),
    )):
        values = [scale * row[field] for row in rows]
        bars = axis.barh(LABELS, values, color=COLORS)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=8)
        axis.set_xlim(0, max(values) * 1.20)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=10)
        axis.grid(axis="x", alpha=.23)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def convergence_figure(calibration, path):
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    fig.suptitle("Offline GA convergence | training scenarios only", fontweight="bold")
    for axis, name in zip(axes.flat, COSTS):
        histories = np.array([run["history"][name] for run in calibration["runs"]])
        x = np.arange(1, histories.shape[1] + 1)
        axis.fill_between(x, histories.min(axis=0), histories.max(axis=0), color="#cee3dc")
        axis.plot(x, np.median(histories, axis=0), color=COLORS[4])
        axis.set(title=name.capitalize(), xlabel="Generation", ylabel="Best training cost")
        axis.grid(alpha=.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(calibration, output):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    cases = scenarios("test")
    methods = (*METHODS, ABLATION)
    rows, seed_rows, scenario_rows, summary = [], [], [], []
    thresholds = calibration["alpha_thresholds"]
    for run in calibration["runs"]:
        fixed_mode = best_fixed_mode(run, thresholds)
        for case in cases:
            curves = traces(case, run, thresholds, fixed_mode)
            for method, trace in curves.items():
                rows.append({"seed": run["seed"], "road": case.name,
                             "sprung_mass_kg": case.sprung_mass, "controller": method,
                             "ablation_mode": fixed_mode[0] if method == ABLATION else "",
                             **measurements(case, trace)})
            if run["seed"] == calibration["demo_seed"] and case.name == "mixed" and case.sprung_mass == 300:
                trace_figure(case, curves, folder / "mixed_trace.png")

    for method in methods:
        group = [row for row in rows if row["controller"] == method]
        summary.append({"controller": method, **summarize(group)})
        for run in calibration["runs"]:
            seed = run["seed"]
            seed_rows.append({"controller": method, "seed": seed,
                              **summarize([row for row in group if row["seed"] == seed])})
        for case in cases:
            scenario_rows.append({"controller": method, "road": case.name, "sprung_mass_kg": case.sprung_mass,
                                  **summarize([row for row in group if row["road"] == case.name
                                               and row["sprung_mass_kg"] == case.sprung_mass])})
    for file, records in (("metrics", rows), ("summary", summary),
                          ("seed_summary", seed_rows), ("scenario_summary", scenario_rows)):
        write_csv(folder / f"{file}.csv", records)
    summary_figure(summary, folder / "comparison.png", len(calibration["runs"]))
    convergence_figure(calibration, folder / "ga_convergence.png")
    return summary
