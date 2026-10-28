"""Construction rules: priority greedy and cheapest insertion (run to completion)."""
from __future__ import annotations

import time

from model.evaluator import RouteEvaluator
from .common import EPS, INF, best_insertion, plan_cost, priority_key, served_customers


class _Constructor:
    name = 'constructor'

    def __init__(self, instance, *, samples=2, evaluator=None):
        self.instance = instance
        self.samples = samples
        self.evaluator = evaluator if evaluator is not None else RouteEvaluator(instance, samples)
        self.vehicle_ids = sorted(instance.vehicles)
        self.customer_ids = sorted(instance.customers)

    def result(self, routes, begin, steps):
        return {'algorithm': self.name, 'routes': routes, 'objective': plan_cost(self.instance, self.evaluator, routes),
                'elapsed_seconds': time.monotonic() - begin, 'iterations': steps,
                'served': len(served_customers(routes))}


class PriorityGreedy(_Constructor):
    """Requests in priority order; each is inserted at its cheapest position if that lowers the objective."""
    name = 'priority-greedy'

    def solve(self):
        begin = time.monotonic()
        routes = {vid: [] for vid in self.vehicle_ids}
        steps = 0
        for cid in sorted(self.customer_ids, key=lambda c: priority_key(self.instance, c)):
            best = (INF, None, None)
            for vid in self.vehicle_ids:
                delta, position = best_insertion(self.evaluator, self.instance, routes, cid, vid)
                if position is not None and (delta, vid, position) < best:
                    best = (delta, vid, position)
            if best[1] is not None and best[0] < -EPS:
                routes[best[1]].insert(best[2], cid)
                steps += 1
        return self.result(routes, begin, steps)


class CheapestInsertion(_Constructor):
    """Repeatedly inserts the request whose cheapest insertion lowers the objective most.

    Ties go to the larger penalty, then the smaller request, tanker and position. The
    best insertion of every remaining request is kept per tanker and recomputed only
    for the route that has just changed.
    """
    name = 'cheapest-insertion'

    def solve(self):
        begin = time.monotonic()
        routes = {vid: [] for vid in self.vehicle_ids}
        remaining = set(self.customer_ids)
        table = {(cid, vid): best_insertion(self.evaluator, self.instance, routes, cid, vid)
                 for cid in remaining for vid in self.vehicle_ids}
        steps = 0
        while remaining:
            choice = None
            for cid in remaining:
                penalty = self.instance.customers[cid].penalty
                for vid in self.vehicle_ids:
                    delta, position = table[cid, vid]
                    if position is None:
                        continue
                    key = (delta, -penalty, cid, vid, position)
                    if choice is None or key < choice:
                        choice = key
            if choice is None or choice[0] >= -EPS:
                break
            _, _, cid, vid, position = choice
            routes[vid].insert(position, cid)
            remaining.discard(cid)
            steps += 1
            for other in remaining:
                table[other, vid] = best_insertion(self.evaluator, self.instance, routes, other, vid)
        return self.result(routes, begin, steps)
