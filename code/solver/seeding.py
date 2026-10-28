"""Whole-route seeding: opens idle tankers with routes that no single insertion can open.

An insertion rule adds one request at a time and keeps it only if the objective
falls. When no single request covers a tanker's activation cost, such a rule never
opens that tanker, although a group of requests may be worth serving together.
For each idle tanker, the search extends feasible request prefixes up to max_depth
requests in a beam of width beam_width, keeping prefixes that do not yet pay for
themselves. It accepts the most improving complete route over all idle tankers,
removes its requests from the pool and repeats until no route improves the objective.
"""
from __future__ import annotations

import math
import time


def profitable_whole_route_seeds(routes, penalties, evaluate, *, max_depth=3, beam_width=64,
                                 time_limit=math.inf, clock=time.monotonic):
    """evaluate(vehicle, route) returns (feasible, operational route cost including activation)."""
    if max_depth < 1 or beam_width < 1:
        raise ValueError('Invalid search limits')
    result = {v: list(route) for v, route in routes.items()}
    assigned = [c for route in result.values() for c in route]
    if len(assigned) != len(set(assigned)) or not set(assigned) <= set(penalties):
        raise ValueError('Invalid incoming assignments')
    remaining = set(penalties) - set(assigned)
    deadline = clock() + time_limit
    accepted, evaluations, truncated = [], 0, False
    while remaining and any(not r for r in result.values()):
        best = None
        for vehicle in sorted(result):
            if result[vehicle]:
                continue
            frontier = [()]
            for depth in range(1, min(max_depth, len(remaining)) + 1):
                extended = []
                for prefix in frontier:
                    for customer in sorted(remaining - set(prefix), key=lambda c: (-penalties[c], c)):
                        if clock() >= deadline:
                            truncated = True
                            break
                        route = prefix + (customer,)
                        feasible, cost = evaluate(vehicle, route)
                        evaluations += 1
                        if not feasible:
                            continue
                        delta = cost - sum(penalties[c] for c in route)
                        extended.append((delta, route))
                        key = (delta, vehicle, route)
                        if delta < -1e-9 and (best is None or key < best[0]):
                            best = (key, vehicle, route, cost, delta)
                    if truncated:
                        break
                extended.sort()
                frontier = [route for _, route in extended[:beam_width]]
                if truncated or not frontier:
                    break
            if truncated:
                break
        if best is None:
            break
        _, vehicle, route, cost, delta = best
        result[vehicle] = list(route)
        remaining.difference_update(route)
        accepted.append(dict(vehicle=vehicle, route=list(route), operational_cost=cost,
                             avoided_penalty=sum(penalties[c] for c in route), objective_delta=delta))
        if truncated:
            break
    return dict(routes=result, accepted_routes=accepted, evaluations=evaluations, time_truncated=truncated,
                max_depth=max_depth, beam_width=beam_width)
