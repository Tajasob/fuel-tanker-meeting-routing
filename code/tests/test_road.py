"""Travel kernel tests: directed links, interior points, caches and backends.

Run from the code folder:  python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import itertools
import math
import unittest

from model.instance import Customer, Instance, PathLeg, Point, Vehicle
from model.road import LinkLocation, NodeLocation, PhysicalLink, RoadNetwork, TravelOracle
from model.loader import match_trajectory_arc
from tests.test_core import split_graph_time


def fixture(*, bidirectional=False, driver_time=10, starts=(0, 4, 8), release=0):
    net = RoadNetwork([PhysicalLink('ab', 0, 1, 10, 10 if bidirectional else None),
                       PhysicalLink('return', 1, 2, 10), PhysicalLink('return2', 2, 0, 10)],
                      {0: (0., 0.), 1: (1., 0.), 2: (1., 1.)})
    customers = {i + 1: Customer(i + 1, (PathLeg('ab', True, driver_time, 100),), (0, 1),
                                 (Point(0, 0, 0), Point(1, 0, driver_time)), start, 1, 50000, None)
                 for i, start in enumerate(starts)}
    vehicles = {1: Vehicle(1, NodeLocation(0), 0, 0, release, 25, 35, 4000, 25),
                2: Vehicle(2, NodeLocation(1), 1, 0, 0, 25, 35, 4000, 25)}
    return Instance(net, customers, vehicles)


class RoadTests(unittest.TestCase):
    def test_directed_terminal_node(self):
        inst = fixture(starts=(0,))
        self.assertEqual(inst.oracle.travel(NodeLocation(1), inst.network.location('ab', 1)).minutes, 0)

    def test_near_endpoint_is_not_node(self):
        inst=fixture(starts=(0,))
        location=inst.network.location('ab',1-1e-10)
        self.assertIsInstance(location,LinkLocation)
        self.assertGreater(inst.oracle.travel(NodeLocation(1),location).minutes,29.9)

    def test_zero_time_arc(self):
        net=RoadNetwork([PhysicalLink('zero',0,1,0),PhysicalLink('one',1,2,1)])
        for backend in ('heap','auto'):
            self.assertEqual(TravelOracle(net,backend=backend).travel(NodeLocation(0),NodeLocation(2)).minutes,1)

    def test_shared_link_direct_and_reverse(self):
        inst=fixture()
        a=LinkLocation('ab',.4);b=LinkLocation('ab',.6)
        self.assertAlmostEqual(inst.oracle.travel(a,b).minutes,2)
        self.assertAlmostEqual(inst.oracle.travel(b,a).minutes,28)
        two=fixture(bidirectional=True)
        self.assertAlmostEqual(two.oracle.travel(b,a).minutes,2)

    def test_parallel_links_not_teleportation(self):
        net=RoadNetwork([PhysicalLink('east',0,1,10),PhysicalLink('west',0,1,None,20)])
        self.assertAlmostEqual(TravelOracle(net).travel(LinkLocation('east',.5),LinkLocation('west',.5)).minutes,15)

    def test_interior_junction_access(self):
        net=RoadNetwork([PhysicalLink('aj',0,1,4),PhysicalLink('jb',1,2,6),PhysicalLink('xj',3,1,1)])
        oracle=TravelOracle(net)
        self.assertEqual(oracle.travel(NodeLocation(3),NodeLocation(1)).minutes,1)
        self.assertAlmostEqual(oracle.travel(NodeLocation(3),LinkLocation('jb',.5)).minutes,4)

    def test_unequal_forward_reverse_rates(self):
        oracle=TravelOracle(RoadNetwork([PhysicalLink('ab',0,1,10,15)]))
        self.assertAlmostEqual(oracle.travel(LinkLocation('ab',.6),LinkLocation('ab',.4)).minutes,3)

    def test_oracle_matches_independent_split_graph(self):
        net=RoadNetwork([PhysicalLink('a',0,1,10,15),PhysicalLink('b',1,2,4),
            PhysicalLink('c',2,3,3,7),PhysicalLink('d',3,0,6),PhysicalLink('parallel',0,1,2),
            PhysicalLink('isolated',4,5,1)])
        locs=[NodeLocation(n) for n in net.adjacency]+[LinkLocation(lid,u) for lid in net.links for u in (.2,.5,.8)]
        oracle=TravelOracle(net,backend='heap')
        for a,b in itertools.product(locs,repeat=2):
            got=oracle.travel(a,b).minutes;want=split_graph_time(net,a,b)
            self.assertTrue((math.isinf(got) and math.isinf(want)) or abs(got-want)<1e-8,(a,b,got,want))

    def test_cutoff_and_cache(self):
        inst=fixture()
        a=NodeLocation(0);b=NodeLocation(1)
        self.assertEqual(inst.oracle.travel(a,b,cutoff=10).minutes,10)
        self.assertTrue(math.isinf(inst.oracle.travel(a,b,cutoff=9).minutes))
        inst.oracle.clear_cache();self.assertEqual(inst.oracle.dijkstra_runs,0)
        self.assertEqual(inst.oracle.travel(a,b,cutoff=10).minutes,10)

    def test_vehicle_sources_are_not_evicted(self):
        net=fixture().network
        oracle=TravelOracle(net,cache_sources=2,pinned_sources={0,1},backend='heap')
        oracle.node_time(0,1);oracle.node_time(1,0)
        before=oracle.dijkstra_runs
        oracle.node_time(2,0)
        oracle.node_time(0,1);oracle.node_time(1,0)
        self.assertEqual(oracle.dijkstra_runs,before+1)

    def test_scipy_matches_heap_when_available(self):
        net=fixture(bidirectional=True).network
        try:fast=TravelOracle(net,backend='scipy')
        except ImportError:self.skipTest('SciPy not installed in this runtime')
        slow=TravelOracle(net,backend='heap')
        for a,b in itertools.product([NodeLocation(0),NodeLocation(1),LinkLocation('ab',.3),LinkLocation('ab',.7)],repeat=2):
            self.assertAlmostEqual(fast.travel(a,b).minutes,slow.travel(a,b).minutes)

    def test_compact_rows_and_unknown_target_fallback(self):
        net=fixture(bidirectional=True).network
        try:compact=TravelOracle(net,backend='scipy',pinned_sources={0})
        except ImportError:self.skipTest('SciPy unavailable')
        compact.restrict_cache_targets([0,1],nonpinned_byte_budget=32)
        baseline=TravelOracle(net,backend='heap')
        for source,target in itertools.product(net.adjacency,repeat=2):
            self.assertAlmostEqual(compact.node_time(source,target),baseline.node_time(source,target))
        for source in net.adjacency:
            self.assertEqual(list(compact.node_times(source,[0,1,2])),[baseline.node_time(source,n) for n in [0,1,2]])
        self.assertGreater(compact.outside_target_runs,0)
        self.assertLessEqual(compact.statistics()['cached_row_bytes'],48)



class CandidateTests(unittest.TestCase):
    def test_bidirectional_catch(self):
        inst = fixture(bidirectional=True, starts=(0,))
        self.assertTrue(any(abs(c.u - .5) < 1e-8 and c.provenance == 'depot_catch' for c in inst.candidates(1, 3)))

    def test_unequal_speed_following_catch(self):
        inst = fixture(driver_time=20, starts=(0,), release=5)
        self.assertTrue(any(abs(c.u - .5) < 1e-8 and c.provenance == 'depot_catch' for c in inst.candidates(1, 3)))

    def test_individual_deadline_and_eligibility(self):
        inst = fixture(starts=(0, 0))
        inst.customers[1].latest_time = 3
        inst.customers[2].latest_time = 7
        inst.eligibility = lambda c: c.u > 0
        self.assertTrue(all(c.meet_time <= 3 + 1e-8 and c.u > 0 for c in inst.candidates(1, 2)))
        self.assertTrue(any(abs(c.meet_time - 7) < 1e-8 for c in inst.candidates(2, 2)))

    def test_parallel_trajectory_ambiguity_rejected(self):
        net = fixture().network
        net.driver_arcs = {(0, 1): [('ab', True, 10, 100), ('different_physical_link', True, 10, 100)]}
        with self.assertRaises(ValueError):
            match_trajectory_arc(net, 0, 1, 10, 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
