"""Plan representation and costs shared by all heuristics."""
from __future__ import annotations

import math
from typing import Dict, List

INF = math.inf
EPS = 1e-7  # an insertion or move must lower the objective by more than this (cost units)

Routes = Dict[int, List[int]]


def clone_routes(routes: Routes) -> Routes:
    return {vid: list(route) for vid, route in routes.items()}


def served_customers(routes: Routes) -> set:
    return {cid for route in routes.values() for cid in route}


def latest_time(instance, cid) -> float:
    latest = instance.customers[cid].latest_time
    return INF if latest is None else float(latest)


def priority_key(instance, cid):
    """Deadline, then larger penalty, then earlier path start, then identifier."""
    customer = instance.customers[cid]
    return (latest_time(instance, cid), -customer.penalty, customer.start, cid)


def plan_cost(instance, evaluator, routes: Routes) -> float:
    """Route costs plus penalties of unserved requests; +inf for an infeasible plan."""
    seen = set()
    total = 0.0
    for vid in sorted(instance.vehicles):
        route = routes.get(vid, [])
        for cid in route:
            if cid in seen:
                return INF
            seen.add(cid)
        evaluation = evaluator.evaluate(vid, route)
        if not evaluation.feasible:
            return INF
        total += evaluation.total
    for cid, customer in instance.customers.items():
        if cid not in seen:
            total += customer.penalty
    return total


def solution_components(instance, routes: Routes, evaluator) -> dict:
    components = dict(travel_from_depot=0.0, travel_between_customers=0.0, service=0.0, waiting_first_leg=0.0,
                      waiting_between_customers=0.0, fixed_vehicle=0.0, unserved_penalty=0.0)
    served = set()
    for vid in sorted(instance.vehicles):
        route = routes.get(vid, [])
        served.update(route)
        evaluation = evaluator.detail(vid, route)
        for key in ('travel_from_depot', 'travel_between_customers', 'service', 'waiting_first_leg',
                    'waiting_between_customers', 'fixed_vehicle'):
            components[key] += getattr(evaluation, key)
    for cid, customer in instance.customers.items():
        if cid not in served:
            components['unserved_penalty'] += customer.penalty
    components['total'] = sum(components.values())
    return components


def best_insertion(evaluator, instance, routes: Routes, cid, vid):
    """Cheapest position for request cid in the route of vid: (delta, position) or (inf, None).

    delta is the change in route cost minus the request's penalty, so a negative
    value lowers the plan objective.
    """
    route = routes[vid]
    base = evaluator.evaluate(vid, route)
    if not base.feasible:
        return INF, None
    penalty = instance.customers[cid].penalty
    best, where = INF, None
    for position in evaluator.insertion_positions(vid, route, cid):
        trial = evaluator.evaluate(vid, route[:position] + [cid] + route[position:])
        if trial.feasible:
            delta = trial.total - base.total - penalty
            if delta < best:
                best, where = delta, position
    return best, where


def two_best_insertions(evaluator, instance, routes: Routes, cid, vid):
    """The two cheapest positions for cid in the route of vid, as [(delta, position), ...]."""
    route = routes[vid]
    base = evaluator.evaluate(vid, route)
    if not base.feasible:
        return []
    penalty = instance.customers[cid].penalty
    moves = []
    for position in evaluator.insertion_positions(vid, route, cid):
        trial = evaluator.evaluate(vid, route[:position] + [cid] + route[position:])
        if trial.feasible:
            moves.append((trial.total - base.total - penalty, position))
    moves.sort()
    return moves[:2]
