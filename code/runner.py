#!/usr/bin/env python3
"""Runs one solution method on one instance, then validates the returned plan.

An instance is prepared once per scenario: the node-to-node travel table of its data
seed is preloaded and the usable meeting candidates are generated (CandidateSpace).
Every method then starts with a fresh route evaluator, so no method profits from
route evaluations of another. Preparation times are reported separately.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from model import sites
from model.evaluator import CandidateSpace, RouteEvaluator
from model.instance import load_instance
from model.loader import load_graphml
from model.road import TravelOracle
from model.travel_table import preload_table
from solver.alns import ALNSSolver
from solver.baselines import SingletonAssignment
from solver.common import clone_routes, plan_cost, served_customers, solution_components
from solver.constructors import CheapestInsertion, PriorityGreedy
from solver.seeding import profitable_whole_route_seeds

HERE = Path(__file__).resolve().parent
SAMPLES = 2
HEURISTICS = ('priority-greedy', 'cheapest-insertion', 'singleton-assignment', 'alns', 'alns-fixed', 'alns-plain',
              'seeding', 'seeding-alns')
SEARCHES = ('alns', 'alns-fixed', 'alns-plain', 'seeding-alns')  # alns-plain: no route-pool recombination
SEED_DEPTH, SEED_WIDTH = 3, 64
RECOMBINE_EVERY, RECOMBINE_TIME = 1000, 60.0


@dataclass(frozen=True)
class Scenario:
    name: str = 'base'
    penalty_scale: float = 1.0
    deadline_minutes: float | None = None
    restricted_sites: bool = False
    fixed_meeting: str | None = None  # 'earliest' or 'destination': one meeting candidate per request


SCENARIOS = {
    'base': Scenario(),
    'penalty0.05': Scenario('penalty0.05', penalty_scale=0.05),
    'penalty0.02': Scenario('penalty0.02', penalty_scale=0.02),
    'sites': Scenario('sites', restricted_sites=True),
    'deadline10': Scenario('deadline10', deadline_minutes=10.0),
    'earliest': Scenario('earliest', fixed_meeting='earliest'),
    'destination': Scenario('destination', fixed_meeting='destination'),
}


def load_network(graphml: Path):
    begin = time.monotonic()
    network = load_graphml(graphml)
    TravelOracle(network).prepare_acceleration()
    return network, time.monotonic() - begin


def prepare(directory: Path, network, table: Path, scenario: Scenario, *, site_nodes=None, samples=SAMPLES):
    """Instance of the scenario with its travel table preloaded and its usable candidates."""
    begin = time.monotonic()
    options = {}
    if scenario.restricted_sites:
        if site_nodes is None:
            site_nodes = sites.local_access_nodes(network)
        options = dict(eligibility=sites.candidate_predicate(site_nodes), eligibility_label=sites.LABEL)
    instance = load_instance(directory, network, **options)
    for customer in instance.customers.values():
        customer.penalty *= scenario.penalty_scale
        if scenario.deadline_minutes is not None and customer.urgent:
            customer.latest_time = customer.start + scenario.deadline_minutes
    loaded = time.monotonic()
    table_info = preload_table(instance, table)
    space = CandidateSpace(instance, samples)
    if scenario.fixed_meeting is not None:
        space.keep_one(scenario.fixed_meeting)
    timing = {'instance_load_seconds': loaded - begin, 'candidate_seconds': time.monotonic() - loaded,
              'travel_table_rows': table_info['rows']}
    return instance, space, timing


def construct_best(instance, evaluator):
    """The better of the priority-greedy and cheapest-insertion plans (ties go to priority greedy)."""
    pg = PriorityGreedy(instance, evaluator=evaluator).solve()
    ci = CheapestInsertion(instance, evaluator=evaluator).solve()
    best = pg if pg['objective'] <= ci['objective'] else ci
    return clone_routes(best['routes']), {
        'priority_greedy_objective': pg['objective'], 'cheapest_insertion_objective': ci['objective'],
        'start_plan': best['algorithm'], 'start_objective': best['objective'],
        'construction_seconds': pg['elapsed_seconds'] + ci['elapsed_seconds']}


def seed_whole_routes(instance, evaluator, routes):
    penalties = {cid: customer.penalty for cid, customer in instance.customers.items()}

    def evaluate(vid, route):
        evaluation = evaluator.evaluate(vid, list(route))
        return evaluation.feasible, evaluation.total
    begin = time.monotonic()
    out = profitable_whole_route_seeds(routes, penalties, evaluate, max_depth=SEED_DEPTH, beam_width=SEED_WIDTH)
    out['seconds'] = time.monotonic() - begin
    out['objective'] = plan_cost(instance, evaluator, out['routes'])
    return out


def run_method(instance, space, method, *, seed=1, iterations=5000, time_limit=300.0, exact=None):
    """Solve the prepared instance with one method; returns the result record (not yet validated)."""
    begin = time.monotonic()
    evaluator = RouteEvaluator(instance, space.samples, space=space)
    record = {'algorithm': method, 'algorithm_seed': None}
    if method == 'priority-greedy':
        out = PriorityGreedy(instance, evaluator=evaluator).solve()
    elif method == 'cheapest-insertion':
        out = CheapestInsertion(instance, evaluator=evaluator).solve()
    elif method == 'singleton-assignment':
        out = SingletonAssignment(instance, evaluator=evaluator).solve()
    elif method in SEARCHES + ('seeding',):
        start, record['construction'] = construct_best(instance, evaluator)
        if method.startswith('seeding'):
            seeded = seed_whole_routes(instance, evaluator, start)
            record['seeding'] = {key: seeded[key] for key in ('accepted_routes', 'evaluations', 'time_truncated',
                                                              'max_depth', 'beam_width', 'seconds', 'objective')}
            start = seeded['routes']
        if method == 'seeding':
            out = {'routes': start, 'objective': plan_cost(instance, evaluator, start), 'iterations': None}
        else:
            record['algorithm_seed'] = seed
            search = ALNSSolver(instance, samples=space.samples, iterations=iterations, time_limit=time_limit,
                                seed=seed, initial_routes=start, freeze_weights=method == 'alns-fixed',
                                evaluator=evaluator,
                                recombine_every=None if method == 'alns-plain' else RECOMBINE_EVERY,
                                recombine_time=RECOMBINE_TIME)
            out = search.solve()
            record['search'] = {key: out[key] for key in ('elapsed_seconds', 'iterations', 'weight_updates', 'trace',
                                                          'final_weights', 'freeze_weights', 'start_temperature',
                                                          'recombination', 'pool_routes')}
            record['search'].update(iteration_limit=iterations, time_limit=time_limit,
                                    stopped_by='iterations' if out['iterations'] >= iterations else 'time')
    elif method == 'exact':
        if exact is None:
            raise ValueError('The exact method needs a CPLEX result (campaign.py import).')
        out = exact(instance, evaluator)
        record['exact'] = out['exact']
    else:
        raise ValueError(f'Unknown method {method}')
    method_seconds = time.monotonic() - begin
    routes = {vid: [int(c) for c in out['routes'].get(vid, [])] for vid in sorted(instance.vehicles)}
    objective = plan_cost(instance, evaluator, routes)
    if not math.isfinite(objective):
        raise AssertionError(f'{method} returned an infeasible plan.')
    if method == 'exact':
        if objective > out['objective'] + 1e-6 * max(1.0, abs(objective)):
            raise AssertionError('The route evaluator prices the CPLEX plan above its model value.')
    elif abs(objective - out['objective']) > 1e-6 * max(1.0, abs(objective)):
        raise AssertionError(f'{method} reported {out["objective"]}, the plan costs {objective}.')
    components = solution_components(instance, routes, evaluator)
    if abs(components['total'] - objective) > 1e-6 * max(1.0, abs(objective)):
        raise AssertionError('Cost components do not add up to the objective.')
    schedule = [row for vid in sorted(routes) for row in evaluator.detail(vid, routes[vid]).schedule]
    served = served_customers(routes)
    lengths = collections.Counter(len(route) for route in routes.values() if route)
    record.update(
        objective=objective, components=components, routes={str(v): r for v, r in routes.items()}, schedule=schedule,
        samples=space.samples, requests=len(instance.customers), vehicles=len(instance.vehicles),
        served=len(served), unserved=len(instance.customers) - len(served),
        urgent_total=sum(c.urgent for c in instance.customers.values()),
        urgent_served=sum(instance.customers[c].urgent for c in served),
        used_vehicles=sum(1 for route in routes.values() if route),
        route_lengths={str(k): v for k, v in sorted(lengths.items())},
        method_seconds=method_seconds, iterations=out.get('iterations'),
        route_evaluations=evaluator.evaluations,
        mean_path_minutes=sum(c.total_path for c in instance.customers.values()) / len(instance.customers),
        usable_candidates=space.candidate_count(),
        generated_candidates=sum(len(instance.candidates(c, space.samples)) for c in instance.customers))
    return record


def validate(network, instance, result, scenario: Scenario):
    from analysis.independent_validation import local_access_site_checker, validate_result
    checker = local_access_site_checker(network) if scenario.restricted_sites else None
    begin = time.monotonic()
    report = validate_result(network, instance, result, approved_site_checker=checker)
    report['seconds'] = time.monotonic() - begin
    return report


def save_result(path: Path, result: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance-dir', type=Path, required=True)
    parser.add_argument('--graphml', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True, help='travel table of the data seed (campaign.py table)')
    parser.add_argument('--method', choices=HEURISTICS, required=True)
    parser.add_argument('--scenario', choices=sorted(SCENARIOS), default='base')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--time-limit', type=float, default=300.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    network, graph_seconds = load_network(args.graphml)
    scenario = SCENARIOS[args.scenario]
    instance, space, timing = prepare(args.instance_dir, network, args.table, scenario)
    result = run_method(instance, space, args.method, seed=args.seed, iterations=args.iterations,
                        time_limit=args.time_limit)
    result.update(instance=args.instance_dir.name, scenario=asdict(scenario), graph_seconds=graph_seconds, **timing)
    result['validation'] = validate(network, instance, result, scenario)
    save_result(args.output, result)
    print(json.dumps({key: result[key] for key in ('algorithm', 'objective', 'served', 'method_seconds')}))
    if not result['validation']['passed']:
        raise SystemExit(f"Validation failed: {result['validation']['errors'][:5]}")


if __name__ == '__main__':
    main()
