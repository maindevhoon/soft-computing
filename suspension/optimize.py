"""Seeded offline genetic search. Test scenarios are never used here."""

import json
from pathlib import Path

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from .fuzzy import COMMON_RULES, COMFORT_RULES, HIGH, LOW, MANUAL, PROTECTION_RULES
from .model import TRAVEL_LIMIT, plant, scenarios
from .simulate import simulate_raw

SEEDS = (11, 23, 37, 53, 71)
POPULATION = 30
GENERATIONS = 40
RULES = {"paper": COMMON_RULES, "fixed": COMMON_RULES,
         "comfort": COMFORT_RULES, "protection": PROTECTION_RULES}
# Weights for RMS acceleration, RMS tire deflection, RMS force, peak travel, squared travel excess.
COSTS = {
    "paper": np.array([0.00, 0.00, 0.00, 1.0, 0.0]),
    "fixed": np.array([0.50, 0.26, 0.10, 0.0, 16.0]),
    "comfort": np.array([0.78, 0.14, 0.04, 0.0, 7.0]),
    "protection": np.array([0.22, 0.29, 0.07, 0.0, 45.0]),
}


@njit(cache=True)
def cost(trace, weights):
    a, tire, force, travel = 0.0, 0.0, 0.0, 0.0
    for row in trace:
        a += row[0] ** 2
        tire += row[2] ** 2
        force += row[3] ** 2
        travel = max(travel, abs(row[1]))
    n = trace.shape[0]
    excess = max(0.0, travel / TRAVEL_LIMIT - 1.0)
    return (weights[0] * np.sqrt(a / n) / 1.0
            + weights[1] * np.sqrt(tire / n) / 0.01
            + weights[2] * np.sqrt(force / n) / 1000.0
            + weights[3] * travel / TRAVEL_LIMIT
            + weights[4] * excess ** 2)


@njit(cache=True, parallel=True)
def evaluate_population(population, roads, a, b, ad, bd, rules, weights, thresholds, mode, second):
    values = np.zeros(population.shape[0])
    for i in prange(population.shape[0]):
        total, worst = 0.0, 0.0
        for j in range(roads.shape[0]):
            trace = simulate_raw(a[j], b[j], ad[j], bd[j], roads[j], mode,
                                 population[i], rules, second, thresholds)
            score = cost(trace, weights)
            total += score
            worst = max(worst, score)
        values[i] = 0.75 * total / roads.shape[0] + 0.25 * worst
    return values


def training_data():
    cases = scenarios("train")
    matrices = [plant(case.sprung_mass) for case in cases]
    roads = np.stack([case.road for case in cases])
    stacks = tuple(np.stack(items) for items in zip(*matrices))
    passive_activity = np.concatenate([
        simulate_raw(*model, case.road, 0, MANUAL, COMMON_RULES, MANUAL, np.array([0.0, 1.0]))[:, 5]
        for model, case in zip(matrices, cases)
    ])
    thresholds = np.quantile(passive_activity, [0.45, 0.85])
    if thresholds[1] <= thresholds[0]:
        raise ValueError("Training scenarios did not produce distinct activity thresholds.")
    return (roads, *stacks), thresholds


def genetic_search(score, seed, population_size, generations):
    rng = np.random.default_rng(seed)
    population = rng.random((population_size, LOW.size))
    population[0] = (MANUAL - LOW) / (HIGH - LOW)
    history = []
    for generation in range(generations):
        scaled = LOW + population * (HIGH - LOW)
        scores = score(scaled)
        order = np.argsort(scores, kind="stable")
        best = int(order[0])
        history.append(float(scores[best]))
        if generation == generations - 1:
            return scaled[best].copy(), history

        next_population = np.empty_like(population)
        next_population[:2] = population[order[:2]]
        # Two three-way tournaments supply parents for each child.
        candidates = rng.integers(0, population_size, (population_size - 2, 2, 3))
        winners = np.take_along_axis(candidates, np.argmin(scores[candidates], axis=-1)[..., None], axis=-1)
        parents = population[winners[..., 0]]
        blend = rng.random((population_size - 2, LOW.size))
        children = parents[:, 0, :] * blend + parents[:, 1, :] * (1 - blend)
        mutate = rng.random(children.shape) < 0.28
        children += mutate * rng.normal(0, 0.11, children.shape)
        next_population[2:] = np.clip(children, 0, 1)
        population = next_population
    raise ValueError("Need at least one generation")


def train(path, quick=False):
    set_num_threads(min(6, get_num_threads()))
    data, thresholds = training_data()
    seeds, size, generations = ((SEEDS[:1], 8, 5) if quick else
                                (SEEDS, POPULATION, GENERATIONS))
    records = []
    for seed in seeds:
        genes, histories = {}, {}
        for name in COSTS:
            def score(rows):
                return evaluate_population(rows, *data, RULES[name], COSTS[name],
                                           thresholds, 1, MANUAL)

            result, history = genetic_search(score, seed, size, generations)
            genes[name], histories[name] = result.tolist(), history
            print(f"Seed {seed:2} | {name:10} | training cost {history[0]:.5f} -> {history[-1]:.5f}",
                  flush=True)

        blended = evaluate_population(np.array([genes["comfort"]]), *data, COMFORT_RULES,
                                      COSTS["fixed"], thresholds, 2, np.array(genes["protection"]))[0]
        records.append({"seed": seed, "genes": genes, "history": histories,
                        "adaptive_training_cost": float(blended)})

    payload = {"population": size, "generations": generations, "quick": quick,
               "alpha_thresholds": thresholds.tolist(), "runs": records,
               "demo_seed": min(records, key=lambda row: row["adaptive_training_cost"])["seed"]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    return payload


def load_calibration(path):
    payload = json.loads(Path(path).read_text())
    thresholds = np.array(payload["alpha_thresholds"], dtype=float)
    if thresholds.shape != (2,) or not np.all(np.isfinite(thresholds)) or not 0 <= thresholds[0] < thresholds[1]:
        raise ValueError("Invalid adaptation thresholds")
    seeds = set()
    for run in payload["runs"]:
        seed = run["seed"]
        if seed in seeds or set(run["genes"]) != set(COSTS):
            raise ValueError("Invalid controller set in calibration")
        seeds.add(seed)
        for name, values in run["genes"].items():
            genes = np.array(values, dtype=float)
            if (genes.shape != LOW.shape or not np.all(np.isfinite(genes)) or
                    np.any(genes < LOW - 1e-12) or np.any(genes > HIGH + 1e-12)):
                raise ValueError(f"Out-of-range genes in seed {seed}, {name}")
            run["genes"][name] = genes
    if payload["demo_seed"] not in seeds:
        raise ValueError("Demo seed missing from calibration")
    payload["alpha_thresholds"] = thresholds
    return payload
