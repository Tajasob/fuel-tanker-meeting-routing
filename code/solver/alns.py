"""Adaptive large neighborhood search over request sets and orders.

Removal operators (random, worst net contribution, related, whole route) and repair
operators (greedy, regret-2, randomized) work on complete tanker plans priced by the
fixed-route evaluator. Unserved requests form a request bank from which every repair
may insert. Plans are accepted by simulated annealing with a temperature that starts
at min_i p_i / ln 2 and falls geometrically to 0.2% of that value by the end of the
search. Operator weights adapt every 50 iterations; the fixed-weight variant keeps
them at one.
"""
from __future__ import annotations

import math
import random
import time

from model.evaluator import RouteEvaluator
from .common import EPS, INF, clone_routes, latest_time, plan_cost, served_customers, two_best_insertions
from .recombination import RoutePool

DESTROY = ('random', 'worst', 'related', 'route')
REPAIR = ('greedy', 'regret2', 'randomized')
SCORES = {'best': 6.0, 'better': 2.0, 'accepted': 0.5}
FINAL_TEMPERATURE_SHARE = 0.002
REACTION = 0.2
UPDATE_PERIOD = 50
WEIGHT_FLOOR = 0.01
REMOVAL_SHARE = 0.30
FULL_BANK_REQUESTS = 80
INSERTION_CACHE = 500_000  # (tanker, route, request) -> two cheapest insertions; routes recur across iterations


class _InsertionTable:
    """The two cheapest insertions of every pending request into every route."""

    def __init__(self, solver, routes, pending):
        self.solver = solver
        self.routes = routes
        self.pending = list(pending)
        self.moves = {}
        for vid in solver.vehicle_ids:
            self.refresh(vid)

    def refresh(self, vid):
        route = tuple(self.routes[vid])
        cache = self.solver.insertion_cache
        for cid in self.pending:
            key = (vid, route, cid)
            moves = cache.get(key)
            if moves is None:
                if len(cache) >= INSERTION_CACHE:
                    cache.clear()
                moves = cache[key] = two_best_insertions(self.solver.evaluator, self.solver.instance, self.routes,
                                                         cid, vid)
            self.moves[cid, vid] = moves

    def candidates(self, cid):
        """All (delta, vid, position) insertions of cid kept in the table, cheapest first."""
        out = []
        for vid in self.solver.vehicle_ids:
            for delta, position in self.moves.get((cid, vid), ()):
                out.append((delta, vid, position))
        out.sort()
        return out

    def insert(self, cid, vid, position):
        self.routes[vid].insert(position, cid)
        self.pending.remove(cid)
        for v in self.solver.vehicle_ids:
            self.moves.pop((cid, v), None)
        self.refresh(vid)


class ALNSSolver:
    def __init__(self, instance, *, samples=2, iterations=5000, time_limit=300.0, seed=1, initial_routes=None,
                 freeze_weights=False, evaluator=None, recombine_every=None, recombine_time=60.0):
        self.instance = instance
        self.samples = samples
        self.iterations = iterations
        self.time_limit = time_limit
        self.rng = random.Random(seed)
        self.initial_routes = clone_routes(initial_routes) if initial_routes is not None else None
        self.freeze_weights = freeze_weights
        self.recombine_every = recombine_every
        self.recombine_time = recombine_time
        self.evaluator = evaluator if evaluator is not None else RouteEvaluator(instance, samples)
        self.vehicle_ids = sorted(instance.vehicles)
        self.customer_ids = sorted(instance.customers)
        self.start_temperature = min(c.penalty for c in instance.customers.values()) / math.log(2.0)
        self.insertion_cache = {}
        self.centroid = {}
        for cid, customer in instance.customers.items():
            xs = [p.x for p in customer.breakpoints]
            ys = [p.y for p in customer.breakpoints]
            self.centroid[cid] = (sum(xs) / len(xs), sum(ys) / len(ys))
        self.km_per_degree = (111.32 * math.cos(math.radians(sum(c[1] for c in self.centroid.values())
                                                             / max(1, len(self.centroid)))), 110.57)

    # ------------------------------------------------------------------ search
    def cost(self, routes):
        return plan_cost(self.instance, self.evaluator, routes)

    def solve(self):
        begin = time.monotonic()
        current = clone_routes(self.initial_routes) if self.initial_routes is not None else \
            {vid: [] for vid in self.vehicle_ids}
        current_cost = self.cost(current)
        best, best_cost = clone_routes(current), current_cost
        weights = {'destroy': {name: 1.0 for name in DESTROY}, 'repair': {name: 1.0 for name in REPAIR}}
        scores = {kind: {name: 0.0 for name in names} for kind, names in weights.items()}
        uses = {kind: {name: 0 for name in names} for kind, names in weights.items()}
        trace = [(0, time.monotonic() - begin, best_cost)]
        completed = updates = 0
        pool = RoutePool(self.instance) if self.recombine_every else None
        recombination = {'runs': 0, 'improvements': 0, 'seconds': 0.0, 'columns': 0}
        if pool is not None:
            pool.add_plan(self.evaluator, current)

        def recombine(iteration):
            nonlocal current, current_cost, best, best_cost
            remaining = self.time_limit - (time.monotonic() - begin)
            packed = pool.pack(time_limit=max(1.0, min(self.recombine_time, remaining)))
            recombination['runs'] += 1
            if packed is None:
                return
            recombination['seconds'] += packed['seconds']
            recombination['columns'] = packed['columns']
            value = self.cost(packed['routes'])
            if value < best_cost - EPS:
                recombination['improvements'] += 1
                best, best_cost = clone_routes(packed['routes']), value
                current, current_cost = clone_routes(packed['routes']), value
                trace.append((iteration, time.monotonic() - begin, best_cost))
        for iteration in range(1, self.iterations + 1):
            elapsed = time.monotonic() - begin
            if elapsed >= self.time_limit:
                break
            progress = max((iteration - 1) / self.iterations, elapsed / self.time_limit)
            temperature = self.start_temperature * FINAL_TEMPERATURE_SHARE ** progress
            destroy = self._roulette(weights['destroy'])
            repair = self._roulette(weights['repair'])
            partial, removed = self._destroy(current, destroy)
            candidate = self._repair(partial, removed, repair)
            candidate_cost = self.cost(candidate)
            if pool is not None:
                pool.add_plan(self.evaluator, candidate)
            score = 0.0
            if candidate_cost < current_cost - EPS:
                current, current_cost, score = candidate, candidate_cost, SCORES['better']
                if candidate_cost < best_cost - EPS:
                    best, best_cost, score = clone_routes(candidate), candidate_cost, SCORES['best']
                    trace.append((iteration, time.monotonic() - begin, best_cost))
            elif math.isfinite(candidate_cost):
                if self.rng.random() < math.exp(-(candidate_cost - current_cost) / temperature):
                    current, current_cost, score = candidate, candidate_cost, SCORES['accepted']
            for kind, name in (('destroy', destroy), ('repair', repair)):
                scores[kind][name] += score
                uses[kind][name] += 1
            completed = iteration
            if not self.freeze_weights and iteration % UPDATE_PERIOD == 0:
                updates += 1
                for kind in weights:
                    for name in weights[kind]:
                        if uses[kind][name]:
                            mean = scores[kind][name] / uses[kind][name]
                            weights[kind][name] = (1 - REACTION) * weights[kind][name] + REACTION * max(WEIGHT_FLOOR, mean)
                        scores[kind][name] = 0.0
                        uses[kind][name] = 0
            if pool is not None and iteration % self.recombine_every == 0:
                recombine(iteration)
        if pool is not None and completed % self.recombine_every != 0:
            recombine(completed)
        return {'routes': best, 'objective': best_cost, 'elapsed_seconds': time.monotonic() - begin,
                'iterations': completed, 'weight_updates': updates, 'trace': trace,
                'final_weights': weights, 'freeze_weights': self.freeze_weights,
                'start_temperature': self.start_temperature, 'recombination': recombination,
                'pool_routes': len(pool) if pool is not None else None}

    def _roulette(self, weights):
        total = sum(max(WEIGHT_FLOOR, w) for w in weights.values())
        pick = self.rng.random() * total
        running = 0.0
        for name, w in weights.items():
            running += max(WEIGHT_FLOOR, w)
            if running >= pick:
                return name
        return next(iter(weights))

    # ------------------------------------------------------------------ removal
    def _destroy(self, routes, name):
        served = sorted(served_customers(routes))
        partial = clone_routes(routes)
        if not served:
            return partial, []
        if name == 'route':
            vid = self.rng.choice([v for v in self.vehicle_ids if partial[v]])
            removed, partial[vid] = list(partial[vid]), []
            return partial, removed
        k = self.rng.randint(1, max(1, math.ceil(REMOVAL_SHARE * len(served))))
        if name == 'worst':
            removed = self._worst(routes, served, k)
        elif name == 'related':
            removed = self._related(served, k)
        else:
            removed = self.rng.sample(served, k)
        drop = set(removed)
        for vid in self.vehicle_ids:
            partial[vid] = [c for c in partial[vid] if c not in drop]
        return partial, removed

    def _rank_pick(self, ordered, k, power):
        """k distinct items of an ordered list, drawn with a bias to its head (index = floor(n * r^power))."""
        pool, picked = list(ordered), []
        while pool and len(picked) < k:
            picked.append(pool.pop(int(len(pool) * self.rng.random() ** power)))
        return picked

    def _worst(self, routes, served, k):
        """Requests whose service saves least: route-cost contribution minus penalty, largest first."""
        where = {cid: vid for vid, route in routes.items() for cid in route}
        values = []
        for cid in served:
            vid = where[cid]
            route = routes[vid]
            without = [c for c in route if c != cid]
            saving = (self.evaluator.evaluate(vid, route).total - self.evaluator.evaluate(vid, without).total
                      - self.instance.customers[cid].penalty)
            values.append((-saving, cid))
        values.sort()
        return self._rank_pick([cid for _, cid in values], k, 3)

    def _distance(self, a, b):
        (xa, ya), (xb, yb) = self.centroid[a], self.centroid[b]
        return math.hypot((xa - xb) * self.km_per_degree[0], (ya - yb) * self.km_per_degree[1])

    def _related(self, served, k):
        """Shaw removal: requests whose driver paths lie close to those already removed."""
        removed = [self.rng.choice(served)]
        while len(removed) < k:
            anchor = self.rng.choice(removed)
            rest = sorted((c for c in served if c not in removed), key=lambda c: (self._distance(anchor, c), c))
            if not rest:
                break
            removed.append(rest[int(len(rest) * self.rng.random() ** 6)])
        return removed

    # ------------------------------------------------------------------ repair
    def _bank(self, routes, removed):
        pool = list(dict.fromkeys(removed))
        present = served_customers(routes) | set(pool)
        rejected = sorted((c for c in self.customer_ids if c not in present),
                          key=lambda c: (latest_time(self.instance, c), -self.instance.customers[c].penalty, c))
        limit = max(10, 3 * len(pool), 2 * len(self.vehicle_ids))
        if len(self.customer_ids) <= FULL_BANK_REQUESTS or len(rejected) <= limit:
            return pool + rejected
        return pool + self.rng.sample(rejected, limit)

    def _repair(self, routes, removed, name):
        pending = self._bank(routes, removed)
        routes = clone_routes(routes)
        if not pending:
            return routes
        table = _InsertionTable(self, routes, pending)
        while table.pending:
            if name == 'greedy':
                move = self._greedy(table)
            elif name == 'regret2':
                move = self._regret(table)
            else:
                move = self._randomized(table)
            if move is None:
                break
            table.insert(*move)
        return routes

    def _greedy(self, table):
        order = list(table.pending)
        self.rng.shuffle(order)
        best = None
        for cid in order:
            moves = table.candidates(cid)
            if moves and moves[0][0] < 0 and (best is None or moves[0][0] < best[0]):
                best = (moves[0][0], cid, moves[0][1], moves[0][2])
        return None if best is None else best[1:]

    def _regret(self, table):
        """Largest regret: the best insertion into another route (or 0, leaving the request
        unserved, if that is better) minus the best insertion."""
        best = None
        for cid in table.pending:
            per_route = sorted((moves[0][0], vid, moves[0][1]) for vid in self.vehicle_ids
                               for moves in [table.moves.get((cid, vid), ())] if moves)
            if not per_route or per_route[0][0] >= 0:
                continue
            alternative = min(0.0, per_route[1][0]) if len(per_route) > 1 else 0.0
            key = (alternative - per_route[0][0], -per_route[0][0], -cid)
            if best is None or key > best[0]:
                best = (key, cid, per_route[0][1], per_route[0][2])
        return None if best is None else best[1:]

    def _randomized(self, table):
        options = []
        for cid in table.pending:
            moves = table.candidates(cid)
            if moves and moves[0][0] < 0:
                options.append((moves[0][0], cid, moves[0][1], moves[0][2]))
        if not options:
            return None
        options.sort()
        shortlist = options[:5]
        weights = [1.0 / (rank + 1) for rank in range(len(shortlist))]
        pick = self.rng.random() * sum(weights)
        running = 0.0
        for option, weight in zip(shortlist, weights):
            running += weight
            if running >= pick:
                return option[1:]
        return shortlist[0][1:]
