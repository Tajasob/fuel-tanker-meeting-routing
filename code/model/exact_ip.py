"""Candidate-graph integer program and its aggregated single-commodity form.

With identical tanker cost rates, the transition costs a_yz do not depend on the
tanker, so the arc formulation aggregates into one flow on the time-ordered
candidate graph with one source arc per tanker and candidate:

    min  sum_src (g_m + a0_my - p_i(y)) x0_my + sum_arcs (a_yz - p_i(z)) x_yz + sum_i p_i
    s.t. outflow(y) - inflow(y) <= 0          for every candidate y
         sum_{y in Y_i} inflow(y) <= 1        for every request i
         sum_y x0_my <= 1                     for every tanker m
         x binary.

Inflow is at most one at every candidate, so the selected arcs form vertex-disjoint
paths, each starting with one source arc (the graph is acyclic because every arc
increases the meeting time by at least a positive service duration). Each path is
the route of the tanker that owns its source arc. Candidates, travel times and arc
costs come from model/evaluator.py, so the model prices every route exactly as the
route evaluator does. The model is solved with IBM ILOG CPLEX (model/cplex_io.py).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.sparse import coo_matrix

from .evaluator import RouteEvaluator


def identical_rates(instance):
    rates = {(v.travel_cost, v.waiting_cost, v.service_cost) for v in instance.vehicles.values()}
    if len(rates) != 1:
        raise ValueError('The aggregated model needs identical tanker cost rates.')
    return next(iter(rates))


def build_model(instance, samples: int = 2, *, evaluator=None) -> dict:
    """Source arcs and candidate arcs of the finite candidate graph, with their costs."""
    start = time.monotonic()
    rates = identical_rates(instance)
    evaluator = evaluator if evaluator is not None else RouteEvaluator(instance, samples)
    space = evaluator.space
    requests = space.request_ids
    offset, U = {}, []
    for cid in requests:
        offset[cid] = len(U)
        U.extend(space.candidates[cid])
    vehicle_ids = space.vehicle_ids
    s_vehicle, s_target, s_cost = [], [], []
    for m, vid in enumerate(vehicle_ids):
        g = instance.vehicles[vid].fixed_cost
        for cid in requests:
            first = evaluator.first_layer(vid, cid)  # includes the activation cost g
            ok = np.flatnonzero(np.isfinite(first))
            s_vehicle.append(np.full(len(ok), m, dtype=np.int64))
            s_target.append(offset[cid] + ok)
            s_cost.append(first[ok] - g)
    a_from, a_to, a_cost = [], [], []
    for i in requests:
        for j in requests:
            if i == j:
                continue
            block = evaluator.pair(rates, i, j, cache=False)
            if block is None:
                continue
            y0, z0, costs = block
            y, z = np.nonzero(np.isfinite(costs))
            a_from.append(offset[i] + y0 + y)
            a_to.append(offset[j] + z0 + z)
            a_cost.append(costs[y, z])

    def stack(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)
    return {'U': U, 'request_of': np.asarray([requests.index(c.cid) for c in U], dtype=np.int64),
            'requests': requests, 'vehicle_ids': vehicle_ids,
            'source_vehicle': stack(s_vehicle, np.int64), 'source_target': stack(s_target, np.int64),
            'source_cost': stack(s_cost, float),
            'arc_from': stack(a_from, np.int64), 'arc_to': stack(a_to, np.int64), 'arc_cost': stack(a_cost, float),
            'activation': np.asarray([instance.vehicles[v].fixed_cost for v in vehicle_ids], dtype=float),
            'candidates_total': sum(len(instance.candidates(c, samples)) for c in requests),
            'build_seconds': time.monotonic() - start}


def assemble(instance, model: dict):
    """Objective, constraint matrix and bounds; variables are source arcs, then arcs."""
    n = len(model['U'])
    request_of = model['request_of']
    penalty = np.asarray([instance.customers[c].penalty for c in model['requests']], dtype=float)
    sv, st, sc = model['source_vehicle'], model['source_target'], model['source_cost']
    af, at, ac = model['arc_from'], model['arc_to'], model['arc_cost']
    ns, na = len(st), len(af)
    objective = np.empty(ns + na)
    objective[:ns] = (sc + model['activation'][sv]) - penalty[request_of[st]]
    objective[ns:] = ac - penalty[request_of[at]]
    arcs = ns + np.arange(na)
    sources = np.arange(ns)
    nq, nm = len(penalty), len(model['vehicle_ids'])
    rows = np.concatenate([af, at, st, n + request_of[at], n + request_of[st], n + nq + sv])
    cols = np.concatenate([arcs, arcs, sources, arcs, sources, sources])
    vals = np.concatenate([np.ones(na), -np.ones(na), -np.ones(ns), np.ones(na), np.ones(ns), np.ones(ns)])
    matrix = coo_matrix((vals, (rows, cols)), shape=(n + nq + nm, ns + na)).tocsr()
    upper = np.concatenate([np.zeros(n), np.ones(nq), np.ones(nm)])
    lower = np.full(n + nq + nm, -np.inf)
    return objective, matrix, lower, upper, float(penalty.sum())


def routes_from_columns(model: dict, ones) -> dict:
    """Tanker routes of the plan whose columns (source arcs, then arcs) are at one."""
    U = model['U']
    ns = len(model['source_target'])
    ones = np.asarray(ones, dtype=np.int64)
    chosen = ones[ones >= ns] - ns
    nxt = {}
    for j in chosen:
        y, z = int(model['arc_from'][j]), int(model['arc_to'][j])
        if y in nxt:
            raise ValueError('Two arcs leave the same candidate.')
        nxt[y] = z
    routes = {vid: [] for vid in model['vehicle_ids']}
    reached = set()
    for q in ones[ones < ns]:
        vid = model['vehicle_ids'][int(model['source_vehicle'][q])]
        if routes[vid]:
            raise ValueError(f'Tanker {vid} has two source arcs.')
        current = int(model['source_target'][q])
        route = [U[current].cid]
        reached.add(current)
        while current in nxt:
            current = nxt[current]
            route.append(U[current].cid)
            reached.add(current)
        routes[vid] = route
    if {int(model['arc_from'][j]) for j in chosen} - reached:
        raise ValueError('An arc of the plan is not reached from a source arc.')
    return routes
