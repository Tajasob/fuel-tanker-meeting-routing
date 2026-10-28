"""Meeting candidates, travel between them, and the fixed-route dynamic program.

CandidateSpace holds, for every request, the candidates that at least one tanker can
reach directly from its release point by their meeting time ("usable" candidates;
no route can use any other candidate, because travel through earlier services is
never shorter than direct travel; the test admits the timing tolerance of a route of
n stops). Travel times between candidates are computed by
the kernel of Section 3 from the node-to-node travel table, in vectorized blocks.
The integer program (model/exact_ip.py) uses the same candidate sets and the same
travel blocks, so its arc costs and the route evaluator agree exactly.

RouteEvaluator evaluates a tanker route with the layered dynamic program: the first
layer prices travel from the release point, every later layer the transition from
the previous request's candidates, and the route cost is the least final value.
A transition between two requests is stored as the smallest block of its cost matrix
that holds every feasible entry (candidates are sorted by meeting time, so feasible
transitions fill a band); a pair without any feasible transition is recorded as
incompatible, and no route that serves the pair in that order is evaluated further.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .road import EPS, LinkLocation, NodeLocation


class EvaluationBudgetExceeded(RuntimeError):
    """A computation was interrupted, not proved infeasible; it is never cached."""


@dataclass
class RouteEvaluation:
    feasible: bool
    total: float
    travel_from_depot: float = 0.0
    travel_between_customers: float = 0.0
    service: float = 0.0
    waiting_first_leg: float = 0.0
    waiting_between_customers: float = 0.0
    fixed_vehicle: float = 0.0
    schedule: list | None = None


def location_record(location):
    return {'node': location.node} if isinstance(location, NodeLocation) else \
        {'link_id': location.link_id, 'fraction': location.fraction}


class _Ports:
    """Exit and entry ports of a list of locations, as table indices and times."""

    def __init__(self, oracle, network, locations):
        index = oracle.table_index
        n = len(locations)
        self.exit_node = np.zeros((n, 2), dtype=np.int64)
        self.exit_time = np.full((n, 2), math.inf)
        self.entry_node = np.zeros((n, 2), dtype=np.int64)
        self.entry_time = np.full((n, 2), math.inf)
        self.link = []
        self.fraction = np.full(n, math.nan)
        for j, location in enumerate(locations):
            location = network.canonical(location)
            for q, (node, duration) in enumerate(oracle._ports(location, True)):
                self.exit_node[j, q] = index[node]
                self.exit_time[j, q] = duration
            for q, (node, duration) in enumerate(oracle._ports(location, False)):
                self.entry_node[j, q] = index[node]
                self.entry_time[j, q] = duration
            if isinstance(location, LinkLocation):
                self.link.append(location.link_id)
                self.fraction[j] = location.fraction
            else:
                self.link.append(None)


def travel_block(network, table, source: _Ports, target: _Ports, rows=None, cols=None):
    """Shortest travel times D[a, b] from source locations to target locations.

    D = min over exit port p of a and entry port q of b of
    (exit time + node-to-node time) + entry time, and direct movement when a and b
    lie inside the same physical link. rows/cols select subsets of the two lists.
    """
    rows = np.arange(len(source.exit_node)) if rows is None else np.asarray(rows)
    cols = np.arange(len(target.entry_node)) if cols is None else np.asarray(cols)
    best = np.full((len(rows), len(cols)), math.inf)
    if not len(rows) or not len(cols):
        return best
    for p in range(2):
        exit_time = source.exit_time[rows, p]
        if not np.isfinite(exit_time).any():
            continue
        exit_node = source.exit_node[rows, p]
        unique_exit, exit_pos = np.unique(exit_node, return_inverse=True)
        for q in range(2):
            entry_time = target.entry_time[cols, q]
            if not np.isfinite(entry_time).any():
                continue
            entry_node = target.entry_node[cols, q]
            unique_entry, entry_pos = np.unique(entry_node, return_inverse=True)
            middle = table[np.ix_(unique_exit, unique_entry)][np.ix_(exit_pos, entry_pos)]
            np.minimum(best, (exit_time[:, None] + middle) + entry_time[None, :], out=best)
    # direct movement inside a shared physical link
    source_links = [source.link[a] for a in rows]
    if any(link is not None for link in source_links):
        target_links = {}
        for jb, b in enumerate(cols):
            link = target.link[b]
            if link is not None:
                target_links.setdefault(link, []).append(jb)
        for ia, link_id in enumerate(source_links):
            if link_id is None or link_id not in target_links:
                continue
            link = network.links[link_id]
            f = source.fraction[rows[ia]]
            for jb in target_links[link_id]:
                g = target.fraction[cols[jb]]
                delta = g - f
                if delta == 0:
                    best[ia, jb] = 0.0
                    continue
                rate = link.forward_min if delta >= 0 else link.reverse_min
                if rate is not None:
                    value = abs(delta) * rate
                    if value < best[ia, jb]:
                        best[ia, jb] = value
    return best


class CandidateSpace:
    """Usable meeting candidates of every request and travel between them."""

    def __init__(self, instance, samples=2):
        oracle = instance.oracle
        if getattr(oracle, 'table_array', None) is None:
            raise ValueError('Preload a travel table (model/travel_table.py) before evaluating routes.')
        self.instance = instance
        self.samples = samples
        self.network = instance.network
        self.table = oracle.table_array
        self.vehicle_ids = sorted(instance.vehicles)
        self.request_ids = sorted(instance.customers)
        vehicles = [instance.vehicles[v] for v in self.vehicle_ids]
        self._depot_ports = _Ports(oracle, self.network, [v.location for v in vehicles])
        self.candidates, self.ports, self.theta = {}, {}, {}
        self.depot_travel = {}
        slack_tolerance = EPS * max(1, len(self.request_ids))
        for cid in self.request_ids:
            generated = instance.candidates(cid, samples)
            ports = _Ports(oracle, self.network, [c.location for c in generated])
            theta = np.asarray([c.meet_time for c in generated])
            travel = travel_block(self.network, self.table, self._depot_ports, ports)
            release = np.asarray([v.t0 for v in vehicles])[:, None]
            reachable = np.isfinite(travel) & (travel <= (theta[None, :] - release) + EPS)
            # A route of k stops accumulates at most k timing tolerances, so a candidate is kept
            # when some tanker reaches it directly within n * EPS; the first layer itself
            # still uses the single tolerance of a direct trip.
            usable = np.isfinite(travel) & (travel <= (theta[None, :] - release) + slack_tolerance)
            keep = np.flatnonzero(usable.any(axis=0))
            self.candidates[cid] = tuple(generated[k] for k in keep)
            self.ports[cid] = _Ports(oracle, self.network, [generated[k].location for k in keep])
            self.theta[cid] = theta[keep]
            for row, vid in enumerate(self.vehicle_ids):
                d = travel[row, keep].copy()
                d[~reachable[row, keep]] = math.inf
                self.depot_travel[vid, cid] = d

    def keep_one(self, which):
        """Fix the meeting of every request in advance: keep its earliest usable candidate
        ('earliest') or its last usable candidate ('destination', the end of the path when a
        tanker can reach the driver there in time)."""
        if which not in ('earliest', 'destination'):
            raise ValueError(f'Unknown fixed meeting rule {which}')
        oracle = self.instance.oracle
        for cid in self.request_ids:
            theta = self.theta[cid]
            if len(theta) == 0:
                continue
            k = int(np.argmin(theta)) if which == 'earliest' else int(np.argmax(theta))
            candidate = self.candidates[cid][k]
            self.candidates[cid] = (candidate,)
            self.ports[cid] = _Ports(oracle, self.network, [candidate.location])
            self.theta[cid] = theta[k:k + 1]
            for vid in self.vehicle_ids:
                self.depot_travel[vid, cid] = self.depot_travel[vid, cid][k:k + 1]

    def travel(self, previous_cid, current_cid):
        return travel_block(self.network, self.table, self.ports[previous_cid], self.ports[current_cid])

    def candidate_count(self):
        return sum(len(c) for c in self.candidates.values())


class RouteEvaluator:
    """Exact cost of a tanker route on the usable candidates (fixed request order)."""

    def __init__(self, instance, samples=2, *, space=None, max_cache=200000, max_pair_bytes=512 * 2**20,
                 deadline=None):
        self.instance = instance
        self.samples = samples
        self.space = space if space is not None else CandidateSpace(instance, samples)
        self.deadline = deadline
        self.max_cache = max_cache
        self.cache = {}
        self._prefix = {}
        self._pairs = OrderedDict()
        self._pair_bytes = 0
        self.max_pair_bytes = max_pair_bytes
        self._first = {}
        self._compatible = {}
        self._reachable = {}
        self._any_rates = self._rates(instance.vehicles[self.space.vehicle_ids[0]]) if instance.vehicles else None
        self.evaluations = 0

    # ------------------------------------------------------------------ helpers
    def _check_budget(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise EvaluationBudgetExceeded('Route evaluation reached its deadline.')

    @staticmethod
    def _rates(vehicle):
        return (vehicle.travel_cost, vehicle.waiting_cost, vehicle.service_cost)

    def candidates(self, cid):
        return self.space.candidates[cid]

    def first_layer(self, vid, cid):
        """Activation, travel from the release point, waiting and service at each candidate."""
        key = (vid, cid)
        value = self._first.get(key)
        if value is None:
            v = self.instance.vehicles[vid]
            d = self.space.depot_travel[vid, cid]
            slack = self.space.theta[cid] - v.t0
            value = np.full(len(d), math.inf)
            ok = np.isfinite(d)
            service = v.service_cost * self.instance.customers[cid].service
            value[ok] = v.fixed_cost + ((v.travel_cost * d[ok] + v.waiting_cost * np.maximum(0.0, slack[ok] - d[ok]))
                                        + service)
            self._first[key] = value
        return value

    def pair(self, rates, previous_cid, current_cid, cache=True):
        """Transition block (row offset, column offset, costs) between the candidates of two
        consecutive requests, or None when no transition is feasible."""
        key = (rates, previous_cid, current_cid)
        if key in self._pairs:
            self._pairs.move_to_end(key)
            return self._pairs[key]
        known = self._compatible.get((previous_cid, current_cid))
        if known is False:
            return None
        self._check_budget()
        block = self._transition(rates, previous_cid, current_cid)
        self._compatible[previous_cid, current_cid] = block is not None
        if block is None or not cache:
            return block
        size = block[2].nbytes
        while self._pairs and self._pair_bytes + size > self.max_pair_bytes:
            _, removed = self._pairs.popitem(last=False)
            self._pair_bytes -= removed[2].nbytes
        self._pairs[key] = block
        self._pair_bytes += size
        return block

    def _transition(self, rates, previous_cid, current_cid):
        c, k, sc = rates
        theta_a, theta_b = self.space.theta[previous_cid], self.space.theta[current_cid]
        if not len(theta_a) or not len(theta_b):
            return None
        departure = theta_a + self.instance.customers[previous_cid].service
        # candidates are sorted by meeting time: only these rows and columns can hold a feasible entry
        r1 = int(np.searchsorted(departure, theta_b[-1] + EPS, side='right'))
        c0 = int(np.searchsorted(theta_b, departure[0] - EPS, side='left'))
        if r1 == 0 or c0 == len(theta_b):
            return None
        rows, cols = np.arange(r1), np.arange(c0, len(theta_b))
        d = travel_block(self.space.network, self.space.table, self.space.ports[previous_cid],
                         self.space.ports[current_cid], rows, cols)
        slack = theta_b[c0:][None, :] - departure[:r1, None]
        feasible = np.isfinite(d) & (d <= slack + EPS)
        if not feasible.any():
            return None
        used_rows = np.flatnonzero(feasible.any(axis=1))
        used_cols = np.flatnonzero(feasible.any(axis=0))
        y0, y1 = int(used_rows[0]), int(used_rows[-1]) + 1
        z0, z1 = int(used_cols[0]), int(used_cols[-1]) + 1
        d, slack, feasible = d[y0:y1, z0:z1], slack[y0:y1, z0:z1], feasible[y0:y1, z0:z1]
        costs = np.full(d.shape, math.inf)
        service = sc * self.instance.customers[current_cid].service
        costs[feasible] = (c * d[feasible] + k * np.maximum(0.0, slack[feasible] - d[feasible])) + service
        return (y0, c0 + z0, costs)

    def pair_dense(self, rates, previous_cid, current_cid, cache=True):
        """The full transition cost matrix (+inf where infeasible)."""
        shape = (len(self.space.theta[previous_cid]), len(self.space.theta[current_cid]))
        matrix = np.full(shape, math.inf)
        block = self.pair(rates, previous_cid, current_cid, cache=cache)
        if block is not None:
            y0, z0, costs = block
            matrix[y0:y0 + costs.shape[0], z0:z0 + costs.shape[1]] = costs
        return matrix

    def step(self, previous, rates, previous_cid, current_cid):
        """Next DP layer: least cost of reaching each candidate of current_cid."""
        out = np.full(len(self.space.theta[current_cid]), math.inf)
        block = self.pair(rates, previous_cid, current_cid)
        if block is not None:
            y0, z0, costs = block
            out[z0:z0 + costs.shape[1]] = (previous[y0:y0 + costs.shape[0], None] + costs).min(axis=0)
        return out

    def compatible(self, previous_cid, current_cid):
        """Whether some candidate of current_cid can follow some candidate of previous_cid."""
        known = self._compatible.get((previous_cid, current_cid))
        if known is None:
            self.pair(self._any_rates, previous_cid, current_cid)
            known = self._compatible[previous_cid, current_cid]
        return known

    def reachable(self, vid, cid):
        """Whether tanker vid can reach some candidate of cid from its release point."""
        key = (vid, cid)
        known = self._reachable.get(key)
        if known is None:
            known = self._reachable[key] = bool(np.isfinite(self.first_layer(vid, cid)).any())
        return known

    def insertion_positions(self, vid, route, cid):
        """Positions where inserting cid passes the pairwise necessary conditions."""
        positions = []
        for position in range(len(route) + 1):
            if position == 0:
                ok = self.reachable(vid, cid)
            else:
                ok = self.compatible(route[position - 1], cid)
            if ok and position < len(route):
                ok = self.compatible(cid, route[position])
            if ok:
                positions.append(position)
        return positions

    def _layers(self, vid, route):
        """Final DP layer of the route, reusing cached prefixes."""
        key = (vid, route)
        layer = self._prefix.get(key)
        if layer is not None:
            return layer
        if len(route) == 1:
            layer = self.first_layer(vid, route[0])
        else:
            previous = self._layers(vid, route[:-1])
            if not np.isfinite(previous).any():
                layer = np.full(len(self.space.candidates[route[-1]]), math.inf)
            else:
                layer = self.step(previous, self._rates(self.instance.vehicles[vid]), route[-2], route[-1])
        if len(self._prefix) >= self.max_cache:
            self._prefix.clear()
        self._prefix[key] = layer
        return layer

    # ------------------------------------------------------------------ public
    def evaluate(self, vid, route):
        """Feasibility and cost of the route (no schedule)."""
        route = tuple(route)
        key = (vid, route)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if len(set(route)) != len(route):
            raise ValueError('A route may not serve a request twice.')
        self.evaluations += 1
        if not route:
            result = RouteEvaluation(True, 0.0, schedule=[])
        else:
            self._check_budget()
            final = self._layers(vid, route)
            total = float(final.min()) if len(final) else math.inf
            result = RouteEvaluation(bool(math.isfinite(total)), total)
        if len(self.cache) >= self.max_cache:
            self.cache.clear()
        self.cache[key] = result
        return result

    def detail(self, vid, route):
        """Cost components and the schedule of the cheapest meetings of a feasible route."""
        route = tuple(route)
        if not route:
            return RouteEvaluation(True, 0.0, schedule=[])
        v = self.instance.vehicles[vid]
        rates = self._rates(v)
        layers = [self.first_layer(vid, route[0])]
        back = []
        for a, b in zip(route, route[1:]):
            if not len(layers[-1]) or not len(self.space.candidates[b]):
                return RouteEvaluation(False, math.inf, schedule=[])
            values = layers[-1][:, None] + self.pair_dense(rates, a, b)
            arg = values.argmin(axis=0)
            back.append(arg)
            layers.append(values[arg, np.arange(values.shape[1])])
        if not len(layers[-1]):
            return RouteEvaluation(False, math.inf, schedule=[])
        index = int(np.argmin(layers[-1]))
        total = float(layers[-1][index])
        if not math.isfinite(total):
            return RouteEvaluation(False, math.inf, schedule=[])
        chosen = [index]
        for arg in reversed(back):
            chosen.append(int(arg[chosen[-1]]))
        chosen.reverse()
        result = RouteEvaluation(True, total, fixed_vehicle=v.fixed_cost, schedule=[])
        departure, source_ports, source_row = v.t0, self.space._depot_ports, self.space.vehicle_ids.index(vid)
        previous_cid = None
        for position, (cid, k) in enumerate(zip(route, chosen)):
            candidate = self.space.candidates[cid][k]
            customer = self.instance.customers[cid]
            if previous_cid is None:
                tau = float(self.space.depot_travel[vid, cid][k])
            else:
                tau = float(travel_block(self.space.network, self.space.table, source_ports,
                                         self.space.ports[cid], [source_row], [k])[0, 0])
            slack = candidate.meet_time - departure
            waiting = max(0.0, slack - tau)
            if position == 0:
                result.travel_from_depot += v.travel_cost * tau
                result.waiting_first_leg += v.waiting_cost * waiting
            else:
                result.travel_between_customers += v.travel_cost * tau
                result.waiting_between_customers += v.waiting_cost * waiting
            result.service += v.service_cost * customer.service
            result.schedule.append({
                'vehicle': vid, 'position': position + 1, 'customer': cid,
                'from': 'depot' if position == 0 else route[position - 1],
                'segment': candidate.segment, 'u': candidate.u, 'x': candidate.x, 'y': candidate.y,
                'meeting_time': candidate.meet_time, 'tau': tau, 'waiting': waiting,
                'arrive': departure + tau, 'service': customer.service, 'urgent': int(customer.urgent),
                'road_location': location_record(candidate.location), 'candidate_source': candidate.provenance})
            departure = candidate.meet_time + customer.service
            source_ports, source_row, previous_cid = self.space.ports[cid], k, cid
        check = sum(getattr(result, name) for name in ('travel_from_depot', 'travel_between_customers', 'service',
                                                       'waiting_first_leg', 'waiting_between_customers',
                                                       'fixed_vehicle'))
        if abs(check - result.total) > 1e-7 * max(1.0, abs(result.total)):
            raise AssertionError('Route cost reconstruction failed.')
        return result


def solve_finite_reference(instance, samples=2, *, max_requests=8, space=None):
    """Optimal plan on the usable candidates by complete subset enumeration (small instances).

    For each tanker, the label of (served subset, last request) is the vector of least
    route costs ending at each candidate of the last request. Tanker plans are then
    combined over disjoint request subsets, and each chosen route is recovered by
    searching the orders of its subset for one that attains the label value.
    """
    import itertools
    ids = tuple(sorted(instance.customers))
    n = len(ids)
    if n > max_requests:
        raise ValueError(f'The enumeration reference handles at most {max_requests} requests; got {n}.')
    evaluator = RouteEvaluator(instance, samples, space=space)
    bit = {cid: 1 << q for q, cid in enumerate(ids)}
    pools = {}
    for vid in sorted(instance.vehicles):
        rates = evaluator._rates(instance.vehicles[vid])
        labels = {}
        for cid in ids:
            first = evaluator.first_layer(vid, cid)
            if np.isfinite(first).any():
                labels[bit[cid], cid] = first
        best = {0: 0.0}
        for mask in range(1, 1 << n):
            for cid in ids:
                values = labels.get((mask, cid))
                if values is None:
                    continue
                best[mask] = min(best.get(mask, math.inf), float(values.min()))
                for nxt in ids:
                    if mask & bit[nxt]:
                        continue
                    layer = evaluator.step(values, rates, cid, nxt)
                    if not np.isfinite(layer).any():
                        continue
                    key = (mask | bit[nxt], nxt)
                    labels[key] = layer if key not in labels else np.minimum(labels[key], layer)
        pools[vid] = best
    combined = {0: (0.0, {})}
    for vid in sorted(instance.vehicles):
        updated = {}
        for used, (cost, masks) in combined.items():
            for mask, route_cost in pools[vid].items():
                if used & mask:
                    continue
                joined = used | mask
                value = cost + route_cost
                if joined not in updated or value < updated[joined][0]:
                    updated[joined] = (value, {**masks, vid: mask})
        combined = updated

    def objective(mask):
        return combined[mask][0] + sum(instance.customers[c].penalty for c in ids if not mask & bit[c])
    best_mask = min(combined, key=objective)
    routes = {}
    for vid in sorted(instance.vehicles):
        mask = combined[best_mask][1].get(vid, 0)
        members = [c for c in ids if mask & bit[c]]
        target = pools[vid][mask]
        routes[vid] = []
        if not members:
            continue
        for order in itertools.permutations(members):
            value = evaluator.evaluate(vid, order).total
            if abs(value - target) <= 1e-9 * max(1.0, abs(target)):
                routes[vid] = list(order)
                break
        else:
            raise AssertionError('No order attains the enumerated route cost.')
    return {'objective': objective(best_mask), 'routes': routes}
