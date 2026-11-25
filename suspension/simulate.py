"""Sampled feedback loop and performance measurements."""

import numpy as np
from numba import njit

from .fuzzy import COMMON_RULES, COMFORT_RULES, MANUAL, PROTECTION_RULES, command, output_sets
from .model import DT, FORCE_LIMIT, FORCE_RATE, TIME, TRAVEL_LIMIT, plant

METHODS = ("Passive", "Manual fuzzy", "Fixed GA (paper-inspired)", "Fixed GA (new cost)", "Adaptive GA-fuzzy")
WINDOW = round(0.30 / DT)
ALPHA_TAU = 0.20
ALPHA_RATE = 2.0


@njit(cache=True)
def activity_step(relative, window, index, squares):
    square = relative * relative
    offset = index % window.size
    squares += square - window[offset]
    window[offset] = square
    return np.sqrt(max(0.0, squares) / min(index + 1, window.size)), squares


@njit(cache=True)
def simulate_raw(a, b, ad, bd, road, mode, first, rules, second, thresholds):
    """Columns are acceleration, travel, tire deflection, force, alpha, and activity."""
    trace = np.zeros((road.size - 1, 6))
    state = np.zeros(4)
    next_state = np.zeros(4)
    inputs = np.empty((3, 3))
    strengths = np.empty(5)
    window = np.zeros(WINDOW)
    first_sets = output_sets(first[6])
    second_sets = output_sets(second[6]) if mode == 2 else first_sets
    acceleration, previous_force, alpha, squares = 0.0, 0.0, 0.0, 0.0
    for i in range(trace.shape[0]):
        relative = state[1] - state[3]
        activity, squares = activity_step(relative, window, i, squares)
        force = 0.0
        if mode:
            force = command(acceleration, state[1], relative, first, rules, first_sets, inputs, strengths)
            if mode == 2:
                target = min(1.0, max(0.0, (activity - thresholds[0]) /
                                     (thresholds[1] - thresholds[0])))
                delta = (target - alpha) * DT / (ALPHA_TAU + DT)
                alpha += min(ALPHA_RATE * DT, max(-ALPHA_RATE * DT, delta))
                protection = command(acceleration, state[1], relative, second, PROTECTION_RULES,
                                     second_sets, inputs, strengths)
                force = (1.0 - alpha) * force + alpha * protection
            force = min(FORCE_LIMIT, max(-FORCE_LIMIT, force))
            force = min(previous_force + FORCE_RATE * DT, max(previous_force - FORCE_RATE * DT, force))

        # The acceleration sample here becomes the input at the next control step.
        acceleration = b[1, 0] * force
        for j in range(4):
            acceleration += a[1, j] * state[j]
        trace[i, 0] = acceleration
        trace[i, 1] = state[0] - state[2]
        trace[i, 2] = state[2] - road[i]
        trace[i, 3] = force
        trace[i, 4] = alpha
        trace[i, 5] = activity
        for j in range(4):
            next_state[j] = bd[j, 0] * force + bd[j, 1] * road[i]
            for k in range(4):
                next_state[j] += ad[j, k] * state[k]
        state, next_state = next_state, state
        previous_force = force
    return trace


def simulate(case, method, genes, thresholds):
    first, second, rules, mode = MANUAL, MANUAL, COMMON_RULES, 1
    if method == METHODS[0]:
        mode = 0
    elif method == METHODS[2]:
        first = genes["paper"]
    elif method == METHODS[3]:
        first = genes["fixed"]
    elif method == METHODS[4]:
        mode, rules = 2, COMFORT_RULES
        first, second = genes["comfort"], genes["protection"]
    elif method != METHODS[1]:
        raise ValueError("Unknown controller")
    return simulate_raw(*plant(case.sprung_mass), case.road, mode, first, rules, second, thresholds)


def measurements(case, trace):
    rms = np.sqrt(np.mean(trace ** 2, axis=0))
    peak = np.max(np.abs(trace), axis=0)
    elapsed = None
    if case.bump_end is not None:
        time = TIME[:-1]
        after = time >= case.bump_end
        bad = after & ((np.abs(trace[:, 0]) > 0.10) | (np.abs(trace[:, 1]) > 0.003))
        end = np.flatnonzero(bad)
        start = np.searchsorted(time, case.bump_end)
        first = max(start, end[-1] + 1) if end.size else start
        if first < time.size and time[-1] - time[first] >= 0.30:
            elapsed = round(time[first] - case.bump_end, 3)
    return {
        "rms_accel_m_s2": rms[0], "peak_accel_m_s2": peak[0],
        "peak_travel_m": peak[1], "rms_tire_m": rms[2], "rms_force_n": rms[3],
        "saturation_fraction": np.mean(np.abs(trace[:, 3]) >= FORCE_LIMIT - 1e-8),
        "travel_excess_m": max(0.0, peak[1] - TRAVEL_LIMIT),
        "settling_s": elapsed, "mean_alpha": np.mean(trace[:, 4]),
    }
