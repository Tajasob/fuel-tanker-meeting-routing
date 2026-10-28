"""Tests of the travel table, the route evaluator, the integer program and the CPLEX exchange.

Run from the code folder:  python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import bz2
import heapq
import itertools
import math
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from model.cplex_io import collect, read_log, read_solution, write_mps
from model.evaluator import CandidateSpace, EvaluationBudgetExceeded, RouteEvaluator, solve_finite_reference
from model.exact_ip import assemble, build_model, routes_from_columns
from model.instance import Customer, Instance, PathLeg, Point, Vehicle
from model.road import LinkLocation, NodeLocation, PhysicalLink, RoadNetwork
from model.travel_table import compute_table, preload_table
from solver.common import plan_cost

RATES = dict(travel_cost=25.0, waiting_cost=35.0, fixed_cost=4000.0, service_cost=25.0)


def split_graph_time(network, source, target):
    """Reference travel time: split links at both points, then plain Dijkstra (no port formulas)."""
    source, target = network.canonical(source), network.canonical(target)

    def name(location):
        return location.node if isinstance(location, NodeLocation) else ('in', location.link_id, location.fraction)
    graph = {n: {} for n in network.adjacency}
    for link in network.links.values():
        positions = {0.0: link.tail, 1.0: link.head}
        for location in (source, target):
            if isinstance(location, LinkLocation) and location.link_id == link.link_id:
                positions[location.fraction] = name(location)
        ordered = sorted(positions.items())
        for (f, u), (g, v) in zip(ordered, ordered[1:]):
            graph.setdefault(u, {})
            graph.setdefault(v, {})
            if link.forward_min is not None:
                graph[u][v] = min(graph[u].get(v, math.inf), (g - f) * link.forward_min)
            if link.reverse_min is not None:
                graph[v][u] = min(graph[v].get(u, math.inf), (g - f) * link.reverse_min)
    start, goal = name(source), name(target)
    dist, heap = {start: 0.0}, [(0.0, 0, start)]
    serial = 0
    while heap:
        d, _, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == goal:
            return d
        for v, w in graph[u].items():
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w
                serial += 1
                heapq.heappush(heap, (d + w, serial, v))
    return math.inf


def grid_instance(seed, *, requests=4, vehicles=2, size=4, penalties=(4500.0, 50000.0)):
    """Random street grid with one-way and two-way links, driver paths and identical tankers."""
    rng = random.Random(seed)
    node = lambda r, c: r * size + c
    coordinates = {node(r, c): (float(c), float(r)) for r in range(size) for c in range(size)}
    links = []
    for r in range(size):
        for c in range(size):
            if c + 1 < size:
                links.append(PhysicalLink(f'h{r}.{c}', node(r, c), node(r, c + 1), rng.uniform(1, 3),
                                          rng.choice([None, rng.uniform(1, 3)])))
            if r + 1 < size:
                links.append(PhysicalLink(f'v{r}.{c}', node(r, c), node(r + 1, c), rng.uniform(1, 3),
                                          rng.uniform(1, 3)))
    network = RoadNetwork(links, coordinates)
    moves = {}
    for link in links:
        if link.forward_min is not None:
            moves.setdefault(link.tail, []).append((link.head, link, True))
        if link.reverse_min is not None:
            moves.setdefault(link.head, []).append((link.tail, link, False))
    customers = {}
    cid = 1
    while len(customers) < requests:
        current = rng.randrange(size * size)
        visited, legs, nodes = {current}, [], [current]
        for _ in range(rng.randint(2, 4)):
            options = [m for m in moves.get(current, []) if m[0] not in visited]
            if not options:
                break
            nxt, link, forward = rng.choice(options)
            rate = link.forward_min if forward else link.reverse_min
            legs.append(PathLeg(link.link_id, forward, rate * rng.uniform(0.8, 1.5), 100.0))
            nodes.append(nxt)
            visited.add(nxt)
            current = nxt
        if len(legs) < 2:
            continue
        points, elapsed = [], 0.0
        for k, n in enumerate(nodes):
            if k:
                elapsed += legs[k - 1].driver_minutes
            points.append(Point(*coordinates[n], elapsed))
        customers[cid] = Customer(cid, tuple(legs), tuple(nodes), tuple(points), rng.uniform(0, 4),
                                  rng.uniform(0.5, 2.0), rng.choice(penalties), None)
        cid += 1
    fleet = {}
    for vid in range(1, vehicles + 1):
        location = NodeLocation(rng.randrange(size * size))
        fleet[vid] = Vehicle(vid, location, *coordinates[location.node], 0.0, **RATES)
    return Instance(network, customers, fleet, name=f'grid{seed}')


def with_table(instance, folder):
    path = Path(folder) / f'{instance.name}.npy'
    compute_table(instance, path)
    preload_table(instance, path)
    return instance


def brute_route_cost(instance, vid, route):
    """Least cost of a route over all generated candidates, with reference travel times."""
    v = instance.vehicles[vid]
    best = math.inf
    for chosen in itertools.product(*(instance.candidates(cid, 2) for cid in route)):
        cost, location, departure = v.fixed_cost, v.location, v.t0
        for candidate in chosen:
            tau = split_graph_time(instance.network, location, candidate.location)
            if departure + tau > candidate.meet_time + 1e-9:
                cost = math.inf
                break
            customer = instance.customers[candidate.cid]
            cost += (v.travel_cost * tau + v.waiting_cost * (candidate.meet_time - departure - tau)
                     + v.service_cost * customer.service)
            location, departure = candidate.location, candidate.meet_time + customer.service
        best = min(best, cost)
    return best


def solve_ip(instance, model):
    objective, matrix, lower, upper, constant = assemble(instance, model)
    result = milp(objective, constraints=LinearConstraint(matrix, lower, upper), integrality=np.ones(len(objective)),
                  bounds=Bounds(0, 1), options={'mip_rel_gap': 1e-12})
    assert result.success, result.message
    return result.fun + constant, np.flatnonzero(result.x > 0.5)


class TravelTableTests(unittest.TestCase):
    def test_table_matches_reference_travel(self):
        with tempfile.TemporaryDirectory() as folder:
            for seed in range(3):
                instance = with_table(grid_instance(seed), folder)
                space = CandidateSpace(instance, 2)
                every = [c for cid in space.request_ids for c in space.candidates[cid]]
                for a, b in itertools.product(space.request_ids, repeat=2):
                    block = space.travel(a, b)
                    for i, x in enumerate(space.candidates[a]):
                        for j, y in enumerate(space.candidates[b]):
                            want = split_graph_time(instance.network, x.location, y.location)
                            got = block[i, j]
                            if math.isinf(want) or want > instance.oracle.search_limit:
                                self.assertTrue(math.isinf(got) or abs(got - want) < 1e-9)
                            else:
                                self.assertAlmostEqual(got, want, places=9)
                self.assertTrue(every)

    def test_usable_candidates_are_exactly_the_reachable_ones(self):
        with tempfile.TemporaryDirectory() as folder:
            instance = with_table(grid_instance(7, vehicles=1), folder)
            space = CandidateSpace(instance, 2)
            v = instance.vehicles[1]
            for cid in space.request_ids:
                reachable = {(c.location, round(c.meet_time, 9)) for c in instance.candidates(cid, 2)
                             if split_graph_time(instance.network, v.location, c.location)
                             <= c.meet_time - v.t0 + 1e-9 * len(instance.customers)}
                kept = {(c.location, round(c.meet_time, 9)) for c in space.candidates[cid]}
                self.assertEqual(kept, reachable)


class EvaluatorTests(unittest.TestCase):
    def test_route_costs_match_enumeration_over_all_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            for seed in range(4):
                instance = with_table(grid_instance(seed, requests=3), folder)
                evaluator = RouteEvaluator(instance, 2)
                for vid in instance.vehicles:
                    for size in (1, 2, 3):
                        for route in itertools.permutations(instance.customers, size):
                            want = brute_route_cost(instance, vid, route)
                            got = evaluator.evaluate(vid, route)
                            self.assertEqual(got.feasible, math.isfinite(want), (seed, vid, route))
                            if got.feasible:
                                self.assertAlmostEqual(got.total, want, places=7)
                                detail = evaluator.detail(vid, route)
                                self.assertAlmostEqual(detail.total, got.total, places=9)
                                self.assertEqual([row['customer'] for row in detail.schedule], list(route))

    def test_schedule_components_add_up(self):
        with tempfile.TemporaryDirectory() as folder:
            instance = with_table(grid_instance(11, requests=4), folder)
            evaluator = RouteEvaluator(instance, 2)
            for vid in instance.vehicles:
                for route in itertools.permutations(instance.customers, 2):
                    detail = evaluator.detail(vid, route)
                    if not detail.feasible:
                        continue
                    parts = (detail.travel_from_depot + detail.travel_between_customers + detail.service
                             + detail.waiting_first_leg + detail.waiting_between_customers + detail.fixed_vehicle)
                    self.assertAlmostEqual(parts, detail.total, places=7)
                    for row in detail.schedule:
                        self.assertGreaterEqual(row['meeting_time'] - row['arrive'], -1e-9)

    def test_empty_duplicate_and_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            instance = with_table(grid_instance(3), folder)
            evaluator = RouteEvaluator(instance, 2)
            self.assertEqual(evaluator.evaluate(1, []).total, 0.0)
            with self.assertRaises(ValueError):
                evaluator.evaluate(1, [1, 1])
            late = RouteEvaluator(instance, 2, deadline=0.0)
            with self.assertRaises(EvaluationBudgetExceeded):
                late.evaluate(1, [1])
            self.assertNotIn((1, (1,)), late.cache)

    def test_reference_matches_complete_route_partition(self):
        with tempfile.TemporaryDirectory() as folder:
            for seed in range(3):
                instance = with_table(grid_instance(seed, requests=3), folder)
                evaluator = RouteEvaluator(instance, 2)
                routes = [()] + [r for k in (1, 2, 3) for r in itertools.permutations(instance.customers, k)]
                best = math.inf
                for a, b in itertools.product(routes, repeat=2):
                    if set(a) & set(b):
                        continue
                    total = (evaluator.evaluate(1, a).total + evaluator.evaluate(2, b).total
                             + sum(c.penalty for cid, c in instance.customers.items() if cid not in a + b))
                    best = min(best, total)
                self.assertAlmostEqual(solve_finite_reference(instance, 2)['objective'], best, places=6)


class IntegerProgramTests(unittest.TestCase):
    def test_integer_program_matches_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            for seed in range(8):
                instance = with_table(grid_instance(seed, requests=5, vehicles=2), folder)
                evaluator = RouteEvaluator(instance, 2)
                model = build_model(instance, 2, evaluator=evaluator)
                value, ones = solve_ip(instance, model)
                routes = routes_from_columns(model, ones)
                reference = solve_finite_reference(instance, 2)['objective']
                self.assertAlmostEqual(value, reference, places=6)
                self.assertAlmostEqual(plan_cost(instance, evaluator, routes), value, places=6)

    def test_single_candidate_relaxation_is_integral(self):
        """With one candidate per request the constraint matrix is a network matrix."""
        with tempfile.TemporaryDirectory() as folder:
            for seed in range(5):
                instance = with_table(grid_instance(seed, requests=6, vehicles=2), folder)
                evaluator = RouteEvaluator(instance, 2)
                space = evaluator.space
                for cid in space.request_ids:
                    keep = space.candidates[cid][:1]
                    index = [0] if keep else []
                    space.candidates[cid] = keep
                    space.theta[cid] = space.theta[cid][index]
                    for vid in space.vehicle_ids:
                        space.depot_travel[vid, cid] = space.depot_travel[vid, cid][index]
                    ports = space.ports[cid]
                    for name in ('exit_node', 'exit_time', 'entry_node', 'entry_time', 'fraction'):
                        setattr(ports, name, getattr(ports, name)[index])
                    ports.link = ports.link[:len(index)]
                evaluator = RouteEvaluator(instance, 2, space=space)
                model = build_model(instance, 2, evaluator=evaluator)
                objective, matrix, lower, upper, _ = assemble(instance, model)
                if not len(objective):
                    continue
                relaxed = milp(objective, constraints=LinearConstraint(matrix, lower, upper),
                               integrality=np.zeros(len(objective)), bounds=Bounds(0, 1))
                self.assertTrue(relaxed.success)
                integral = milp(objective, constraints=LinearConstraint(matrix, lower, upper),
                                integrality=np.ones(len(objective)), bounds=Bounds(0, 1))
                self.assertAlmostEqual(relaxed.fun, integral.fun, places=6)

    def test_fixed_meeting_keeps_one_candidate(self):
        """keep_one leaves the earliest or the last usable candidate, the relaxation stays integral,
        and no route becomes cheaper than with all candidates."""
        with tempfile.TemporaryDirectory() as folder:
            for seed in range(4):
                for which in ('earliest', 'destination'):
                    instance = with_table(grid_instance(seed, requests=5, vehicles=2), folder)
                    full = CandidateSpace(instance, 2)
                    space = CandidateSpace(instance, 2)
                    space.keep_one(which)
                    for cid in space.request_ids:
                        if not len(full.theta[cid]):
                            self.assertEqual(len(space.candidates[cid]), 0)
                            continue
                        self.assertEqual(len(space.candidates[cid]), 1)
                        pick = min if which == 'earliest' else max
                        self.assertAlmostEqual(space.theta[cid][0], pick(full.theta[cid]), places=12)
                    restricted = RouteEvaluator(instance, 2, space=space)
                    unrestricted = RouteEvaluator(instance, 2, space=full)
                    for vid in instance.vehicles:
                        for route in itertools.permutations(instance.customers, 2):
                            a = restricted.evaluate(vid, route)
                            b = unrestricted.evaluate(vid, route)
                            if a.feasible:
                                self.assertTrue(b.feasible)
                                self.assertGreaterEqual(a.total, b.total - 1e-9)
                    model = build_model(instance, 2, evaluator=restricted)
                    objective, matrix, lower, upper, _ = assemble(instance, model)
                    if not len(objective):
                        continue
                    relaxed = milp(objective, constraints=LinearConstraint(matrix, lower, upper),
                                   integrality=np.zeros(len(objective)), bounds=Bounds(0, 1))
                    integral = milp(objective, constraints=LinearConstraint(matrix, lower, upper),
                                    integrality=np.ones(len(objective)), bounds=Bounds(0, 1))
                    self.assertAlmostEqual(relaxed.fun, integral.fun, places=6)

    def test_routes_from_columns_rejects_malformed_plans(self):
        with tempfile.TemporaryDirectory() as folder:
            instance = with_table(grid_instance(2, requests=4), folder)
            model = build_model(instance, 2)
            sources = np.flatnonzero(model['source_vehicle'] == 0)
            if len(sources) >= 2:
                with self.assertRaises(ValueError):
                    routes_from_columns(model, sources[:2])


def parse_mps(path):
    opener = bz2.open if str(path).endswith('.bz2') else open
    section, senses, objective, entries, rhs, binaries = None, {}, {}, [], {}, set()
    with opener(path, 'rt') as handle:
        for line in handle:
            if not line.startswith(' '):
                section = line.split()[0]
                continue
            parts = line.split()
            if section == 'ROWS':
                senses[parts[1]] = parts[0]
            elif section == 'COLUMNS':
                for name, value in zip(parts[1::2], parts[2::2]):
                    if name == 'obj':
                        objective[parts[0]] = float(value)
                    else:
                        entries.append((name, parts[0], float(value)))
            elif section == 'RHS':
                rhs[parts[1]] = float(parts[2])
            elif section == 'BOUNDS':
                assert parts[0] == 'BV'
                binaries.add(parts[2])
    return senses, objective, entries, rhs, binaries


class CplexExchangeTests(unittest.TestCase):
    def build(self, seed):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        instance = with_table(grid_instance(seed, requests=5), folder)
        evaluator = RouteEvaluator(instance, 2)
        model = build_model(instance, 2, evaluator=evaluator)
        return instance, evaluator, model, assemble(instance, model), Path(folder)

    def test_mps_round_trip(self):
        instance, _, model, assembled, folder = self.build(4)
        objective, matrix, _, upper, _ = assembled
        for name in ('model.mps', 'model.mps.bz2'):
            write_mps(folder / name, model, assembled)
            senses, costs, entries, rhs, binaries = parse_mps(folder / name)
            self.assertEqual(set(senses.values()) - {'N'}, {'L'})
            self.assertEqual(len(binaries), len(objective))
            for j, value in enumerate(objective):
                self.assertEqual(costs[f'x{j}'], float(value))
            rows = {name: q for q, name in enumerate(n for n in senses if senses[n] != 'N')}
            rebuilt = np.zeros(matrix.shape)
            for row, column, value in entries:
                rebuilt[rows[row], int(column[1:])] = value
            self.assertTrue(np.array_equal(rebuilt, matrix.toarray()))
            for row, q in rows.items():
                self.assertEqual(rhs.get(row, 0.0), float(upper[q]))

    def test_collect_reads_a_cplex_solution(self):
        instance, evaluator, model, assembled, folder = self.build(5)
        value, ones = solve_ip(instance, model)
        constant = assembled[4]
        variables = ''.join(f'  <variable name="x{j}" index="{j}" value="1"/>\n' for j in ones)
        (folder / 'm.sol').write_text(
            '<?xml version = "1.0" encoding="UTF-8" standalone="yes"?>\n<CPLEXSolution version="1.2">\n'
            f' <header problemName="m.mps" objectiveValue="{value - constant!r}" solutionTypeValue="3"'
            ' solutionStatusValue="101" solutionStatusString="integer optimal solution" MIPNodes="0"/>\n'
            f' <variables>\n{variables} </variables>\n</CPLEXSolution>\n')
        (folder / 'm.out').write_text(
            'Welcome to IBM(R) ILOG(R) CPLEX(R) Interactive Optimizer 22.1.1.0\n'
            'Parallel mode: deterministic, using up to 8 threads.\n'
            f'MIP - Integer optimal solution:  Objective = {value - constant:.10e}\n'
            'Solution time =    1.25 sec.  Iterations = 10  Nodes = 3\n'
            'Deterministic time = 12.50 ticks  (10.00 ticks/sec)\n'
            'Dual simplex - Optimal:  Objective = -1.0e+05\n'
            'Solution time =    0.10 sec.  Iterations = 4 (0)\n'
            'wall_seconds 2\n')
        result = collect(model, assembled, folder / 'm.sol', folder / 'missing.sol', folder / 'm.out', time_limit=60)
        self.assertAlmostEqual(result['objective'], value, places=6)
        self.assertTrue(result['exact']['optimal'])
        self.assertEqual(result['exact']['version'], '22.1.1.0')
        self.assertEqual(result['exact']['threads'], 8)
        self.assertEqual(result['exact']['nodes'], 3)
        self.assertEqual(result['exact']['solver_seconds'], 1.25)
        self.assertAlmostEqual(plan_cost(instance, evaluator, result['routes']), value, places=6)
        info = read_log(folder / 'm.out')
        self.assertEqual(info['solution_seconds'], [1.25, 0.1])
        self.assertEqual(read_solution(folder / 'm.sol')['ones'].tolist(), sorted(ones.tolist()))

    def test_collect_rejects_infeasible_solution(self):
        instance, _, model, assembled, folder = self.build(6)
        sources = np.flatnonzero(model['source_vehicle'] == 0)[:2]
        if len(sources) < 2:
            self.skipTest('too few source arcs')
        variables = ''.join(f'  <variable name="x{j}" index="{j}" value="1"/>\n' for j in sources)
        (folder / 'bad.sol').write_text(
            '<?xml version = "1.0"?>\n<CPLEXSolution version="1.2">\n'
            ' <header objectiveValue="0" solutionStatusValue="101"/>\n'
            f' <variables>\n{variables} </variables>\n</CPLEXSolution>\n')
        (folder / 'bad.out').write_text('')
        with self.assertRaises(ValueError):
            collect(model, assembled, folder / 'bad.sol', folder / 'none', folder / 'bad.out', time_limit=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
