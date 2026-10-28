"""Route-pool recombination: the best plan assembled from routes of earlier plans.

The pool keeps, for every tanker and request set met during the search, the
cheapest feasible order. The set-packing model picks at most one pooled route per
tanker and serves every request at most once:

    min  sum_r (c_r - sum_{i in r} p_i) z_r + sum_i p_i
    s.t. sum_{r containing i} z_r <= 1   for every request i
         sum_{r of tanker m}  z_r <= 1   for every tanker m,   z binary.

The model is solved with HiGHS through scipy.optimize.milp. The current plan is in
the pool, so the packing is never worse than it.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


class RoutePool:
    def __init__(self, instance, *, capacity=200_000):
        self.instance = instance
        self.capacity = capacity
        self.routes = {}
        self.penalty = {cid: c.penalty for cid, c in instance.customers.items()}

    def add_plan(self, evaluator, routes):
        for vid, route in routes.items():
            if not route:
                continue
            key = (vid, frozenset(route))
            evaluation = evaluator.evaluate(vid, route)
            if not evaluation.feasible:
                continue
            known = self.routes.get(key)
            if known is None or evaluation.total < known[0]:
                self.routes[key] = (evaluation.total, tuple(route))
        if len(self.routes) > self.capacity:
            ranked = sorted(self.routes.items(), key=lambda kv: kv[1][0] - sum(self.penalty[i] for i in kv[0][1]))
            self.routes = dict(ranked[:self.capacity // 2])

    def __len__(self):
        return len(self.routes)

    def pack(self, *, time_limit=60.0):
        """Best plan from pooled routes: (routes, objective, seconds, status) or None for an empty pool."""
        if not self.routes:
            return None
        begin = time.monotonic()
        requests = sorted(self.instance.customers)
        vehicles = sorted(self.instance.vehicles)
        row_of = {cid: q for q, cid in enumerate(requests)}
        vehicle_row = {vid: len(requests) + q for q, vid in enumerate(vehicles)}
        items = list(self.routes.items())
        cost = np.empty(len(items))
        rows, cols = [], []
        for j, ((vid, members), (route_cost, _)) in enumerate(items):
            cost[j] = route_cost - sum(self.penalty[i] for i in members)
            for i in members:
                rows.append(row_of[i])
                cols.append(j)
            rows.append(vehicle_row[vid])
            cols.append(j)
        matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(requests) + len(vehicles), len(items)))
        result = milp(cost, constraints=LinearConstraint(matrix.tocsr(), -np.inf, 1.0), integrality=np.ones(len(items)),
                      bounds=Bounds(0, 1), options={'time_limit': time_limit})
        if result.x is None:
            return None
        plan = {vid: [] for vid in vehicles}
        for j in np.flatnonzero(result.x > 0.5):
            (vid, _), (_, route) = items[j]
            plan[vid] = list(route)
        return {'routes': plan, 'seconds': time.monotonic() - begin, 'status': int(result.status),
                'columns': len(items)}
