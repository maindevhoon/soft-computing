"""Keyboard dashboard for inspecting frozen test scenarios."""

import curses
from functools import lru_cache
from pathlib import Path

import numpy as np

from .evaluate import ABLATION, best_fixed_mode, trace_figure, traces
from .model import LOADS, TRAVEL_LIMIT, scenarios
from .simulate import METHODS, measurements


def open_dashboard(calibration):
    curses.wrapper(dashboard, calibration)


def dashboard(screen, calibration):
    try:
        curses.curs_set(0)
    except curses.error:
        pass

    accent = selected = 0
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            accent, selected = curses.color_pair(1), curses.color_pair(2)
        except curses.error:
            pass

    road_index, load_index, focus = 0, 1, 0
    cases = scenarios("test")
    roads = tuple(dict.fromkeys(case.name for case in cases))
    run = next(run for run in calibration["runs"] if run["seed"] == calibration["demo_seed"])
    thresholds = calibration["alpha_thresholds"]
    fixed_mode = best_fixed_mode(run, thresholds)
    names = (*METHODS, ABLATION)
    notice = ""

    @lru_cache(maxsize=len(cases))
    def result(road, load):
        case = cases[road * len(LOADS) + load]
        curves = traces(case, run, thresholds, fixed_mode)
        return case, curves, [measurements(case, curves[name]) for name in names]

    while True:
        height, width = screen.getmaxyx()
        screen.erase()

        def text(row, column, message, attr=0):
            if 0 <= row < height and 0 <= column < width - 1:
                screen.addnstr(row, column, message, width - column - 1, attr)

        text(0, 2, "ADAPTIVE VEHICLE SUSPENSION  /  QUARTER-CAR LAB", curses.A_BOLD | accent)
        text(2, 2, "Held-out scenario comparison", curses.A_BOLD)
        road_name, mass = roads[road_index], LOADS[load_index]
        text(4, 2, ("> " if focus == 0 else "  ") + f"Road     < {road_name.capitalize():8} >",
             curses.A_BOLD | accent if focus == 0 else 0)
        text(5, 2, ("> " if focus == 1 else "  ") + f"Load     < {mass:3} kg   >",
             curses.A_BOLD | accent if focus == 1 else 0)
        text(4, 40, f"Calibration seed {run['seed']}")
        text(5, 40, f"Travel limit {TRAVEL_LIMIT * 1000:.0f} mm")

        case, curves, rows = result(road_index, load_index)
        text(7, 2, f"{'Controller':26} {'RMS accel':>9} {'Travel':>8} {'Tire RMS':>9} {'Force RMS':>10}",
             curses.A_BOLD | accent)
        text(8, 2, f"{'':26} {'m/s^2':>9} {'mm':>8} {'mm':>9} {'N':>10}")
        for i, (name, row) in enumerate(zip(names, rows), start=9):
            data = (f"{name:26} {row['rms_accel_m_s2']:9.3f} {1000 * row['peak_travel_m']:8.2f} "
                    f"{1000 * row['rms_tire_m']:9.2f} {row['rms_force_n']:10.1f}")
            text(i, 2, data, curses.A_BOLD | selected if name == METHODS[4] else 0)
            if row["travel_excess_m"] > 0:
                text(i, 72, "!", curses.A_BOLD)

        alpha = curves[METHODS[4]][:, 4]
        meter = min(20, round(20 * float(np.max(alpha))))
        text(17, 2, f"Adaptive blend peak  [{'#' * meter}{'.' * (20 - meter)}] {np.max(alpha):.2f}")
        text(18, 2, "0 = comfort   1 = travel protection    ! = travel limit exceeded")
        text(20, 2, notice)
        text(22, 2, "[Up/Down] Field   [Left/Right] Change   [P] Save plot   [Q] Exit", curses.A_BOLD)
        screen.refresh()
        key = screen.getch()
        notice = ""
        if key in (ord("q"), ord("Q"), 27):
            return
        if key in (curses.KEY_UP, curses.KEY_DOWN, 9):
            focus = 1 - focus
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT):
            step = 1 if key == curses.KEY_RIGHT else -1
            if focus == 0:
                road_index = (road_index + step) % len(roads)
            else:
                load_index = (load_index + step) % len(LOADS)
        elif key in (ord("p"), ord("P")):
            path = Path(f"suspension-{case.name}-{case.sprung_mass}.png")
            try:
                trace_figure(case, curves, path)
                notice = f"Saved {path}"
            except OSError as error:
                notice = str(error)
