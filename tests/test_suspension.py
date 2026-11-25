import unittest

import numpy as np
from scipy.integrate import solve_ivp

from suspension.evaluate import ABLATION, best_fixed_mode, traces
from suspension.fuzzy import COMMON_RULES, HIGH, LOW, MANUAL, command, output_sets
from suspension.model import DT, FORCE_LIMIT, FORCE_RATE, LOADS, TIME, Scenario, plant, rough, scenarios
from suspension.optimize import genetic_search, load_calibration, training_data
from suspension.simulate import ALPHA_RATE, METHODS, activity_step, measurements, simulate


class SuspensionTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = np.array([0.05, 0.20])
        self.genes = {name: MANUAL.copy() for name in ("paper", "fixed", "comfort", "protection")}

    def test_plant_integration(self):
        a, b, ad, bd = plant(300)
        state = np.array([.010, -.04, .005, .03])
        inputs = np.array([150., .012])
        expected = solve_ivp(lambda _, x: a @ x + b @ inputs, (0, DT), state,
                             rtol=1e-11, atol=1e-12).y[:, -1]
        np.testing.assert_allclose(ad @ state + bd @ inputs, expected, atol=1e-10)

    def test_roads_and_load_split(self):
        training, testing = scenarios("train"), scenarios("test")
        self.assertEqual(len(training), 9)
        self.assertEqual(len(testing), 12)
        self.assertEqual({case.sprung_mass for case in testing}, set(LOADS))
        for case in training:
            counterpart = next(c for c in testing if c.name == case.name and c.sprung_mass == case.sprung_mass)
            self.assertFalse(np.array_equal(case.road, counterpart.road))
        np.testing.assert_array_equal(rough(.2, 5., .01, 42), rough(.2, 5., .01, 42))

    def test_zero_motion_on_flat_road(self):
        case = Scenario("flat", 300, np.zeros_like(TIME))
        for method in METHODS:
            trace = simulate(case, method, self.genes, self.thresholds)
            np.testing.assert_allclose(trace, 0.0, atol=1e-8)

    def test_mamdani_output_is_signed_and_symmetric(self):
        inputs, strength = np.empty((3, 3)), np.empty(5)
        sets = output_sets(MANUAL[6])
        positive = command(1., .12, .13, MANUAL, COMMON_RULES, sets, inputs, strength)
        negative = command(-1., -.12, -.13, MANUAL, COMMON_RULES, sets, inputs, strength)
        self.assertLess(positive, 0)
        self.assertAlmostEqual(positive, -negative)

    def test_rolling_rms_expires_old_values(self):
        window = np.zeros(2)
        squares = 0.0
        for i, (value, expected) in enumerate(((3., 3.), (4., np.sqrt(12.5)), (0., np.sqrt(8)))):
            rms, squares = activity_step(value, window, i, squares)
            self.assertAlmostEqual(rms, expected)

    def test_adaptive_limits(self):
        case = Scenario("stress", 240, scenarios("test")[-1].road * 4)
        self.genes["comfort"], self.genes["protection"] = HIGH.copy(), HIGH.copy()
        trace = simulate(case, METHODS[-1], self.genes, self.thresholds)
        self.assertTrue(np.isfinite(trace).all())
        self.assertLessEqual(np.abs(trace[:, 3]).max(), FORCE_LIMIT + 1e-9)
        self.assertLessEqual(np.abs(np.diff(np.r_[0, trace[:, 3]])).max(), FORCE_RATE * DT + 1e-9)
        self.assertGreater(trace[:, 4].max(), 0.1)
        self.assertGreaterEqual(trace[:, 4].min(), 0.0)
        self.assertLessEqual(trace[:, 4].max(), 1.0)
        self.assertLessEqual(np.abs(np.diff(trace[:, 4])).max(), ALPHA_RATE * DT + 1e-9)

    def test_settling_metric_respects_road_type(self):
        bump, mixed = scenarios("test")[0], scenarios("test")[-1]
        for case in (bump, mixed):
            row = measurements(case, simulate(case, METHODS[0], self.genes, self.thresholds))
            self.assertEqual(row["rms_force_n"], 0.0)
            if case is bump:
                self.assertGreater(row["settling_s"], 0)
            else:
                self.assertIsNone(row["settling_s"])

    def test_seeded_ga_and_elitism(self):
        target = (LOW + HIGH) / 2

        def score(genes):
            return np.sum(((genes - target) / (HIGH - LOW)) ** 2, axis=1)

        first, history = genetic_search(score, 11, 10, 8)
        again, repeated = genetic_search(score, 11, 10, 8)
        np.testing.assert_array_equal(first, again)
        self.assertEqual(history, repeated)
        self.assertTrue(np.all(np.diff(history) <= 0.0))
        self.assertTrue(np.all((first >= LOW) & (first <= HIGH)))

    def test_train_thresholds_and_fixed_ablation(self):
        _, thresholds = training_data()
        self.assertGreater(thresholds[1], thresholds[0])
        frozen = load_calibration("results/calibration.json")
        self.assertEqual(len(frozen["runs"]), 5)
        run = frozen["runs"][0]
        mode = best_fixed_mode(run, thresholds)
        self.assertIn(mode[0], ("comfort", "protection"))
        case = scenarios("test")[-1]
        result = traces(case, run, thresholds, mode)
        self.assertEqual(len(result), 6)
        self.assertIn(ABLATION, result)
        self.assertGreater(np.max(result[METHODS[-1]][:, 4]), 0)
        self.assertEqual(np.max(result[ABLATION][:, 4]), 0)


if __name__ == "__main__":
    unittest.main()
