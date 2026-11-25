"""Quarter-car dynamics and reproducible road scenarios."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import cont2discrete

DT = 0.02
DURATION = 6.0
TIME = np.arange(round(DURATION / DT) + 1) * DT
LOADS = (240, 300, 360)
WHEEL_MASS = 45.0
SPRING = 16_000.0
DAMPER = 1_000.0
TIRE = 190_000.0
FORCE_LIMIT = 1_600.0
FORCE_RATE = 16_000.0
TRAVEL_LIMIT = 0.045


@lru_cache
def plant(sprung_mass):
    """State is body position/velocity followed by wheel position/velocity."""
    m, w = sprung_mass, WHEEL_MASS
    k, c, kt = SPRING, DAMPER, TIRE
    a = np.array([
        [0, 1, 0, 0],
        [-k / m, -c / m, k / m, c / m],
        [0, 0, 0, 1],
        [k / w, c / w, -(k + kt) / w, -c / w],
    ])
    b = np.array([[0, 0], [1 / m, 0], [0, 0], [-1 / w, kt / w]])
    ad, bd, _, _, _ = cont2discrete((a, b, np.eye(4), np.zeros((4, 2))), DT)
    return a, b, ad, bd


@dataclass(frozen=True)
class Scenario:
    name: str
    sprung_mass: int
    road: np.ndarray
    bump_end: float | None = None


def bump(start, width, height):
    phase = np.clip((TIME - start) / width, 0, 1)
    return height * np.sin(np.pi * phase) ** 2


def envelope(start, end, fade=0.30):
    on = np.clip((TIME - start) / fade, 0, 1)
    off = np.clip((end - TIME) / fade, 0, 1)
    return np.sin(np.pi * on / 2) ** 2 * np.sin(np.pi * off / 2) ** 2


def sine(start, end, amplitude, frequency):
    return envelope(start, end) * amplitude * np.sin(2 * np.pi * frequency * (TIME - start))


def rough(start, end, amplitude, seed):
    """A finite sum of tapered harmonics, with carrier frequencies from 0.7 to 8 Hz."""
    rng = np.random.default_rng(seed)
    frequencies = np.linspace(0.7, 8.0, 16)
    phase = rng.uniform(0, 2 * np.pi, frequencies.size)
    angles = 2 * np.pi * frequencies[:, None] * (TIME - start) + phase[:, None]
    signal = np.sum(np.sin(angles) / frequencies[:, None] ** 0.7, axis=0)
    active = (TIME >= start) & (TIME <= end)
    return envelope(start, end) * amplitude * signal / np.sqrt(np.mean(signal[active] ** 2))


def scenarios(split):
    if split == "train":
        profiles = (
            ("bump", bump(0.80, 0.50, 0.040), 1.30),
            ("sine", sine(0.35, 4.90, 0.011, 1.35), None),
            ("rough", rough(0.35, 5.50, 0.009, 127), None),
        )
    elif split == "test":
        profiles = (
            ("bump", bump(1.00, 0.44, 0.055), 1.44),
            ("sine", sine(0.45, 5.35, 0.012, 1.80), None),
            ("rough", rough(0.40, 5.45, 0.011, 941), None),
            ("mixed", bump(0.55, 0.42, 0.048) + sine(1.50, 3.12, 0.013, 1.55)
             + rough(3.20, 5.55, 0.012, 1709), None),
        )
    else:
        raise ValueError("Unknown scenario split")
    return [Scenario(name, mass, road, end) for name, road, end in profiles for mass in LOADS]
