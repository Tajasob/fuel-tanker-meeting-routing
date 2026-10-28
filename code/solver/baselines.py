"""Singleton assignment: the best plan in which no tanker serves more than one request."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from model.evaluator import RouteEvaluator
from .common import plan_cost, served_customers


class SingletonAssignment:
    """Prices every tanker-request pair as a one-request route and matches the profitable
    pairs by a maximum-saving assignment (saving = penalty minus route cost)."""
    name = 'singleton-assignment'

    def __init__(self, instance, *, samples=2, evaluator=None):
        self.instance = instance
        self.samples = samples
        self.evaluator = evaluator if evaluator is not None else RouteEvaluator(instance, samples)
        self.vehicle_ids = sorted(instance.vehicles)
        self.customer_ids = sorted(instance.customers)

    def solve(self):
        begin = time.monotonic()
        saving = np.zeros((len(self.vehicle_ids), len(self.customer_ids)))
        for a, vid in enumerate(self.vehicle_ids):
            for b, cid in enumerate(self.customer_ids):
                evaluation = self.evaluator.evaluate(vid, [cid])
                if evaluation.feasible:
                    saving[a, b] = max(0.0, self.instance.customers[cid].penalty - evaluation.total)
        rows, cols = linear_sum_assignment(saving, maximize=True)
        routes = {vid: [] for vid in self.vehicle_ids}
        for a, b in zip(rows, cols):
            if saving[a, b] > 0:
                routes[self.vehicle_ids[a]] = [self.customer_ids[b]]
        return {'algorithm': self.name, 'routes': routes,
                'objective': plan_cost(self.instance, self.evaluator, routes),
                'elapsed_seconds': time.monotonic() - begin, 'iterations': None,
                'served': len(served_customers(routes))}
