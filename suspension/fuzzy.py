"""Mamdani inference, symmetric fuzzy sets, and a readable rule generator."""

import numpy as np
from numba import njit

# Three zero-set half widths, three input ranges, the inner output center, and force gain.
LOW = np.array([0.35, 0.35, 0.35, 0.50, 0.06, 0.09, 0.27, 250.0])
HIGH = np.array([1.00, 1.00, 1.00, 4.50, 0.55, 0.85, 0.73, 2400.0])
MANUAL = np.array([0.78, 0.78, 0.78, 2.0, 0.25, 0.45, 0.50, 850.0])
GRID = np.linspace(-1.0, 1.0, 33)


def rule_table(weights):
    """A positive weighted linguistic input calls for a negative actuator force."""
    return np.array([
        2 - int(np.clip(round(weights[0] * a + weights[1] * v + weights[2] * d), -2, 2))
        for a in (-1, 0, 1) for v in (-1, 0, 1) for d in (-1, 0, 1)
    ])


COMMON_RULES = rule_table((1.0, 1.0, 1.0))
COMFORT_RULES = rule_table((1.0, 1.4, 0.55))
PROTECTION_RULES = rule_table((0.6, 0.65, 1.7))


@njit(cache=True)
def input_sets(value, range_, width, out):
    normalized = min(1.0, max(-1.0, value / range_))
    out[0] = max(0.0, -normalized)
    out[1] = max(0.0, 1.0 - abs(normalized) / width)
    out[2] = max(0.0, normalized)


@njit(cache=True)
def output_sets(spacing):
    centers = np.array([-1.0, -spacing, 0.0, spacing, 1.0])
    sets = np.zeros((5, GRID.size))
    for label in range(5):
        for i in range(GRID.size):
            point = GRID[i]
            if point <= centers[label]:
                sets[label, i] = (1.0 if label == 0 else
                                  max(0.0, (point - centers[label - 1]) /
                                      (centers[label] - centers[label - 1])))
            else:
                sets[label, i] = (1.0 if label == 4 else
                                  max(0.0, (centers[label + 1] - point) /
                                      (centers[label + 1] - centers[label])))
    return sets


@njit(cache=True)
def command(acceleration, velocity, relative, genes, rules, sets, inputs, strengths):
    input_sets(acceleration, genes[3], genes[0], inputs[0])
    input_sets(velocity, genes[4], genes[1], inputs[1])
    input_sets(relative, genes[5], genes[2], inputs[2])
    strengths[:] = 0.0
    for a in range(3):
        for v in range(3):
            antecedent = min(inputs[0, a], inputs[1, v])
            if antecedent > 0.0:
                for d in range(3):
                    consequent = rules[9 * a + 3 * v + d]
                    strength = min(antecedent, inputs[2, d])
                    strengths[consequent] = max(strengths[consequent], strength)
    area, moment = 0.0, 0.0
    for i in range(GRID.size):
        membership = 0.0
        for label in range(5):
            membership = max(membership, min(sets[label, i], strengths[label]))
        # Equal spacing cancels in the centroid ratio; the endpoints get half weight.
        if i == 0 or i == GRID.size - 1:
            membership *= 0.5
        area += membership
        moment += membership * GRID[i]
    return genes[7] * moment / area if area > 0.0 else 0.0
