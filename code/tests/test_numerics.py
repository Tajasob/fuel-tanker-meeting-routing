"""Timing-tolerance cases: oracle cutoffs, service domain and route feasibility at EPS scale.

Run from the code folder:  python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import math
import tempfile
import unittest

from model.evaluator import RouteEvaluator, solve_finite_reference
from model.instance import Customer, Instance, PathLeg, Point, Vehicle
from model.road import EPS, NodeLocation, PhysicalLink, RoadNetwork, TravelOracle
from tests.test_core import with_table


def oracle_for(weight, backend, compact=False):
    graph = RoadNetwork([PhysicalLink('edge', 0, 1, weight)], {0: (0., 0.), 1: (1., 0.)})
    oracle = TravelOracle(graph, backend=backend)
    if compact:
        oracle.restrict_cache_targets([0])
    return oracle


class OracleToleranceTests(unittest.TestCase):
    backends = ('heap', 'scipy')

    def test_cold_warm_cutoff_and_compact_fallback(self):
        for backend in self.backends:
            for compact in (False, True):
                with self.subTest(backend=backend, compact=compact):
                    oracle = oracle_for(1., backend, compact)
                    for cutoff in (1. - EPS / 2, 1., math.inf, 1. - EPS / 2):
                        self.assertEqual(oracle.travel(NodeLocation(0), NodeLocation(1), cutoff=cutoff).minutes, 1.)

    def test_cache_must_expand_for_sub_epsilon_bound_increase(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                oracle = oracle_for(1. + 1.5 * EPS, backend)
                self.assertTrue(math.isinf(oracle.node_time(0, 1, cutoff=1.)))
                self.assertEqual(oracle.node_time(0, 1, cutoff=1. + 0.75 * EPS), 1. + 1.5 * EPS)
                self.assertEqual(oracle.dijkstra_runs, 2)

    def test_batch_scalar_negative_slack_and_zero_arc(self):
        for backend in self.backends:
            for compact in (False, True):
                for warm in (False, True):
                    with self.subTest(backend=backend, compact=compact, warm=warm):
                        oracle = oracle_for(0., backend, compact)
                        if warm:
                            oracle.node_time(0, 1)
                        for cutoff in (-2 * EPS, -EPS / 2, 0.):
                            want = math.inf if cutoff < -EPS else 0.
                            self.assertEqual(oracle.node_time(0, 1, cutoff), want)
                            self.assertEqual(list(oracle.node_times(0, [0, 1], cutoff)), [want, want])
                            self.assertEqual(oracle.travel(NodeLocation(0), NodeLocation(1), cutoff=cutoff).minutes,
                                             want)

    def test_cutoffs_outside_tolerance_remain_infeasible(self):
        for backend in self.backends:
            oracle = oracle_for(1., backend)
            for warm in (False, True):
                if warm:
                    oracle.node_time(0, 1)
                self.assertTrue(math.isinf(oracle.node_time(0, 1, cutoff=1. - 2 * EPS)))


class ServiceDomainTests(unittest.TestCase):
    def test_rejects_zero_sub_epsilon_and_epsilon(self):
        oracle = oracle_for(1., 'heap')
        vehicle = Vehicle(1, NodeLocation(0), 0., 0., 0., 1., 0., 0., 0.)
        for service in (0., EPS / 4, math.nextafter(EPS, 0.), EPS):
            with self.subTest(service=service):
                customer = Customer(1, (PathLeg('edge', True, 1., 100.),), (0, 1),
                                    (Point(0., 0., 0.), Point(1., 0., 1.)), 0., service, 100.)
                with self.assertRaisesRegex(ValueError, 'exceed the timing tolerance'):
                    Instance(oracle.network, {1: customer}, {1: vehicle})


class RouteToleranceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.count = 0

    def ready(self, instance):
        self.count += 1
        instance.name = f'case{self.count}'
        return with_table(instance, self.folder.name)

    def test_travel_within_tolerance_is_feasible(self):
        network = RoadNetwork([PhysicalLink('edge', 0, 1, 1. + EPS / 2)], {0: (0., 0.), 1: (1., 0.)})
        customer = Customer(1, (PathLeg('edge', True, 1., 100.),), (0, 1), (Point(0., 0., 0.), Point(1., 0., 1.)),
                            0., 1., 100.)
        vehicle = Vehicle(1, NodeLocation(0), 0., 0., 0., 1., 0., 0., 0.)
        instance = self.ready(Instance(network, {1: customer}, {1: vehicle},
                                       eligibility=lambda c: c.location == NodeLocation(1),
                                       eligibility_label='endpoint_only'))
        self.assertEqual(RouteEvaluator(instance).evaluate(1, [1]).total, 1. + EPS / 2)
        self.assertEqual(solve_finite_reference(instance)['objective'], 1. + EPS / 2)

    def test_service_just_above_epsilon_is_accepted(self):
        network = RoadNetwork([PhysicalLink('edge', 0, 1, 1.)], {0: (0., 0.), 1: (1., 0.)})
        service = math.nextafter(EPS, math.inf)
        customer = Customer(1, (PathLeg('edge', True, 1., 100.),), (0, 1), (Point(0., 0., 0.), Point(1., 0., 1.)),
                            0., service, 100.)
        vehicle = Vehicle(1, NodeLocation(0), 0., 0., 0., 1., 0., 0., 0.)
        instance = self.ready(Instance(network, {1: customer}, {1: vehicle}))
        self.assertTrue(RouteEvaluator(instance).evaluate(1, [1]).feasible)

    def test_tolerances_of_consecutive_stops_are_kept(self):
        """B can only follow A (each stop admits EPS); the candidate of B must stay usable."""
        for meeting in (1., 10., 40., 1000.):
            for service in (math.nextafter(EPS, math.inf), 1.01 * EPS, 2 * EPS):
                advance = (meeting + service) - meeting
                margin = EPS - advance
                second_travel = margin / 2 if margin > 0 else EPS / 100
                with self.subTest(meeting=meeting, service=service):
                    network = RoadNetwork([PhysicalLink('depot_A', 0, 1, 1. + EPS), PhysicalLink('A_B', 1, 2, second_travel)],
                                          {0: (0., 0.), 1: (1., 0.), 2: (2., 0.)})
                    customers = {cid: Customer(cid, (PathLeg(link, True, 1., 100.),), (cid - 1, cid),
                                               (Point(cid - 1., 0., 0.), Point(float(cid), 0., 1.)),
                                               meeting - 1., service, 100.)
                                 for cid, link in ((1, 'depot_A'), (2, 'A_B'))}
                    vehicle = Vehicle(1, NodeLocation(0), 0., 0., meeting - 1., 0., 0., 0., 0.)
                    instance = self.ready(Instance(network, customers, {1: vehicle},
                                                   eligibility=lambda c: c.location == NodeLocation(c.cid),
                                                   eligibility_label='endpoint_only', add_depot_catchpoints=False))
                    follows_a = second_travel <= EPS - advance
                    evaluator = RouteEvaluator(instance)
                    if follows_a:
                        self.assertEqual(evaluator.space.candidates[2], instance.candidates(2))
                    route = evaluator.evaluate(1, [1, 2])
                    self.assertEqual(route.feasible, follows_a)
                    if follows_a:
                        self.assertEqual(route.total, 0.)
                    self.assertEqual(solve_finite_reference(instance)['objective'], 0. if follows_a else 100.)

    def test_waiting_uses_the_slack_before_travel(self):
        for waiting_rate in (35., 1e15):
            with self.subTest(waiting_rate=waiting_rate):
                network = RoadNetwork([PhysicalLink('depot_A', 0, 1, 1.), PhysicalLink('A_B', 1, 2, .001)],
                                      {0: (0., 0.), 1: (1., 0.), 2: (2., 0.)})
                second_meeting = 41. + .001 + 1e-14
                customers = {cid: Customer(cid, (PathLeg(link, True, 1., 100.),), (cid - 1, cid),
                                           (Point(cid - 1., 0., 0.), Point(float(cid), 0., 1.)), start, 1., 1000.)
                             for cid, link, start in ((1, 'depot_A', 39.), (2, 'A_B', second_meeting - 1.))}
                vehicle = Vehicle(1, NodeLocation(0), 0., 0., 39., 0., waiting_rate, 0., 0.)
                instance = self.ready(Instance(network, customers, {1: vehicle},
                                               eligibility=lambda c: c.location == NodeLocation(c.cid),
                                               eligibility_label='endpoint_only', add_depot_catchpoints=False))
                expected_wait = max(0., (second_meeting - 41.) - .001)
                evaluator = RouteEvaluator(instance)
                self.assertEqual(evaluator.evaluate(1, [1, 2]).total, waiting_rate * expected_wait)
                self.assertEqual(evaluator.detail(1, [1, 2]).schedule[1]['waiting'], expected_wait)
                self.assertEqual(solve_finite_reference(instance)['objective'], waiting_rate * expected_wait)


if __name__ == '__main__':
    unittest.main(verbosity=2)
