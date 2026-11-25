"""Run the dashboard, train controllers, or reproduce the evaluation."""

import argparse
import sys

from .evaluate import evaluate
from .optimize import load_calibration, train

CALIBRATION = "results/calibration.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    training = sub.add_parser("train", help="Run the offline genetic algorithm")
    training.add_argument("--quick", action="store_true", help="Small teaching run with one seed")
    training.add_argument("--out", help="Calibration JSON to write")
    evaluation = sub.add_parser("evaluate", help="Measure the frozen controllers on test cases")
    evaluation.add_argument("--calibration", default=CALIBRATION)
    evaluation.add_argument("--out", default="results")
    dashboard = sub.add_parser("demo", help="Open the terminal dashboard")
    dashboard.add_argument("--calibration", default=CALIBRATION)
    args = parser.parse_args()
    try:
        if args.command == "train":
            path = args.out or ("results/quick-calibration.json" if args.quick else CALIBRATION)
            train(path, args.quick)
            print(f"Saved {path}")
        elif args.command == "evaluate":
            rows = evaluate(load_calibration(args.calibration), args.out)
            print(f"{'Controller':27} {'RMS accel.':>10} {'Travel (mm)':>12} {'Tire (mm)':>10} {'Force (N)':>10}")
            for row in rows:
                print(f"{row['controller']:27} {row['rms_accel_m_s2']:10.3f} "
                      f"{1000 * row['peak_travel_m']:12.2f} {1000 * row['rms_tire_m']:10.2f} "
                      f"{row['rms_force_n']:10.1f}")
        else:
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                parser.exit(2, "Demo needs an interactive terminal. Use evaluate for batch results.\n")
            from .tui import open_dashboard
            open_dashboard(load_calibration(getattr(args, "calibration", CALIBRATION)))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
