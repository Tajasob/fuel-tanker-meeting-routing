"""Heuristics on small street grids: feasibility, optimality checks and independent validation.

Run from the code folder:  python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import copy
import tempfile
import unittest

from analysis.independent_validation import local_access_site_checker, validate_result
from model.evaluator import CandidateSpace, RouteEvaluator, solve_finite_reference
from runner import run_method
from solver.alns import ALNSSolver
from solver.common import plan_cost
from solver.seeding import profitable_whole_route_seeds
from tests.test_core import grid_instance, with_table

METHODS = ('priority-greedy', 'cheapest-insertion', 'singleton-assignment', 'alns', 'alns-fixed', 'seeding',
           'seeding-alns')


class HeuristicTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def instance(self, seed, **options):
        return with_table(grid_instance(seed, **options), self.folder.name)

    def test_every_method_returns_a_validated_plan_no_better_than_the_optimum(self):
        for seed in range(6):
            instance = self.instance(seed, requests=5, vehicles=2)
            space = CandidateSpace(instance, 2)
            optimum = solve_finite_reference(instance, 2, space=space)['objective']
            for method in METHODS:
                result = run_method(instance, space, method, seed=seed, iterations=300, time_limit=30)
                self.assertGreaterEqual(result['objective'], optimum - 1e-6, (seed, method))
                report = validate_result(instance.network, instance, result)
                self.assertTrue(report['passed'], (seed, method, report['errors']))

    def test_alns_reaches_the_optimum_on_small_grids(self):
        hits = 0
        for seed in range(6):
            instance = self.instance(seed, requests=5, vehicles=2)
            space = CandidateSpace(instance, 2)
            optimum = solve_finite_reference(instance, 2, space=space)['objective']
            result = run_method(instance, space, 'alns', seed=1, iterations=2000, time_limit=30)
            hits += abs(result['objective'] - optimum) <= 1e-6 * max(1.0, optimum)
        self.assertGreaterEqual(hits, 5)

    def test_alns_counts_and_weight_updates(self):
        instance = self.instance(1, requests=4)
        for count in (49, 50, 101):
            for frozen in (False, True):
                out = ALNSSolver(instance, iterations=count, time_limit=60, freeze_weights=frozen).solve()
                self.assertEqual(out['iterations'], count)
                self.assertEqual(out['weight_updates'], 0 if frozen else count // 50)
                self.assertTrue(all(w == 1.0 for kind in out['final_weights'].values() for w in kind.values())
                                or not frozen)

    def test_alns_never_returns_worse_than_its_start(self):
        instance = self.instance(2, requests=5)
        evaluator = RouteEvaluator(instance, 2)
        start = {1: [], 2: []}
        for cid in sorted(instance.customers):
            if evaluator.evaluate(1, [cid]).feasible:
                start[1] = [cid]
                break
        out = ALNSSolver(instance, iterations=200, initial_routes=start, evaluator=evaluator).solve()
        self.assertLessEqual(out['objective'], plan_cost(instance, evaluator, start) + 1e-9)

    def test_whole_route_seeding_opens_a_jointly_profitable_route(self):
        penalties = {1: 3, 2: 3, 3: 3}

        def evaluate(vehicle, route):
            return True, 7.0 + 0.1 * len(route)
        out = profitable_whole_route_seeds({1: []}, penalties, evaluate, max_depth=3, beam_width=8)
        self.assertEqual(sorted(out['routes'][1]), [1, 2, 3])
        out = profitable_whole_route_seeds({1: []}, penalties, evaluate, max_depth=2, beam_width=8)
        self.assertEqual(out['routes'][1], [])

    def test_seeding_keeps_existing_routes(self):
        penalties = {1: 10, 2: 10}
        out = profitable_whole_route_seeds({1: [1], 2: []}, penalties, lambda v, r: (True, 4.0), max_depth=2)
        self.assertEqual(out['routes'][1], [1])
        self.assertEqual(out['routes'][2], [2])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.instance = with_table(grid_instance(3, requests=5), folder.name)
        self.space = CandidateSpace(self.instance, 2)
        self.result = run_method(self.instance, self.space, 'cheapest-insertion')
        self.assertTrue(self.result['schedule'], 'fixture must serve a request')

    def check(self, result):
        return validate_result(self.instance.network, self.instance, result)

    def test_accepts_the_returned_plan(self):
        self.assertTrue(self.check(self.result)['passed'])

    def test_rejects_tampered_schedules(self):
        for field, change in (('tau', 0.5), ('meeting_time', 0.25), ('waiting', 1.0), ('arrive', -0.5)):
            tampered = copy.deepcopy(self.result)
            tampered['schedule'][0][field] += change
            self.assertFalse(self.check(tampered)['passed'], field)

    def test_rejects_a_wrong_objective_and_a_duplicate(self):
        tampered = copy.deepcopy(self.result)
        tampered['objective'] += 1.0
        self.assertFalse(self.check(tampered)['passed'])
        tampered = copy.deepcopy(self.result)
        vid, route = next((v, r) for v, r in tampered['routes'].items() if r)
        other = next(v for v in tampered['routes'] if v != vid)
        tampered['routes'][other] = tampered['routes'][other] + [route[0]]
        self.assertFalse(self.check(tampered)['passed'])

    def test_restricted_sites_need_an_independent_screen(self):
        checker = local_access_site_checker(self.instance.network)
        report = validate_result(self.instance.network, self.instance, self.result, approved_site_checker=checker)
        self.assertFalse(report['passed'])  # grid links carry no road class, so no site is approved


if __name__ == '__main__':
    unittest.main(verbosity=2)
