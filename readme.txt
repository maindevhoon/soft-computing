ADAPTIVE VEHICLE SUSPENSION USING GA-OPTIMIZED FUZZY CONTROL
Fundamentals of Soft Computing Lab (26B16CS311)

SETUP
  Open a terminal in the extracted Adaptive_Vehicle_Suspension folder.
  Use Python 3.13. The interactive demo uses curses on macOS or Linux.

      python3.13 -m venv .venv
      source .venv/bin/activate
      python -m pip install -r requirements.txt

RUN
  Start the terminal dashboard with the included five-seed calibration:

      python -m suspension demo

  Use Up/Down to select Road or Load, and Left/Right to change it. Each
  row is a controller tested on the same road and load. An exclamation
  mark flags a peak travel over 45 mm. Press P to save a plot or Q to exit.
  A terminal at least 78 columns wide and 24 rows tall is preferred.

REPRODUCE
  Run the tests and regenerate the results without retraining:

      python -m unittest discover -s tests -v
      python -m suspension evaluate --out reproduced

  Repeat the full offline GA search, then evaluate the new calibration:

      python -m suspension train --out calibration_retrained.json
      python -m suspension evaluate --calibration calibration_retrained.json --out retrained

  The full run uses 30 individuals, 40 generations, and five GA seeds
  per tuning objective. For a shorter teaching run, use train --quick
  and evaluate with --calibration results/quick-calibration.json.

FILES
  suspension/       Plant, roads, Mamdani controller, GA, evaluation and UI
  tests/            Simulation, control and reproducibility tests
  results/          Frozen parameters, measured CSVs and plots
  submission/       Project report and its interpretation of the results

DEMO ORDER
  Show a bump at 300 kg, then change to a rough or mixed road. Compare
  the adaptive controller to the fixed GA (new cost) row. Save a plot
  and point to the blend factor, body acceleration, and actuator effort.
  The report describes a quarter-car simulation with synthetic road inputs.
