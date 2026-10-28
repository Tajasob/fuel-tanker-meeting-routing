"""Node-to-node travel table computed once per data seed and shared by all methods.

The travel kernel (model/road.py) needs shortest times between the ports of meeting
candidates and tanker release points. Every such port is a node of a driver path or a
tanker release node. The table holds, for every node of that set, one bounded
one-to-all Dijkstra search restricted to the set, so that every later query is a
lookup. Entries beyond the search bound are +inf, exactly as for the lazy oracle,
whose rows the table reproduces.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .road import EPS, NodeLocation


def oracle_targets(instance) -> list:
    """Tanker release ports and every node of every driver path, in the oracle's order."""
    network = instance.network
    sources = set()
    for vehicle in instance.vehicles.values():
        location = network.canonical(vehicle.location)
        if isinstance(location, NodeLocation):
            sources.add(location.node)
        else:
            link = network.links[location.link_id]
            if link.forward_min is not None:
                sources.add(link.head)
            if link.reverse_min is not None:
                sources.add(link.tail)
    nodes = sources | {node for customer in instance.customers.values() for node in customer.nodes}
    return sorted(nodes, key=str)


def table_limit(instance) -> float:
    """Search bound: latest path end minus earliest release, plus the timing tolerance of
    every stop of a route (a usable candidate may be reached that much later)."""
    return float(instance.oracle.search_limit) + EPS * len(instance.customers)


def compute_rows(network, nodes, rows, limit, table, *, chunk=32, progress=None):
    """Fill table[rows] with bounded Dijkstra rows restricted to the columns of `nodes`."""
    from scipy.sparse.csgraph import dijkstra
    index = network._node_index
    columns = np.asarray([index[n] for n in nodes], dtype=np.int64)
    rows = list(rows)
    for begin in range(0, len(rows), chunk):
        block = rows[begin:begin + chunk]
        sources = [index[nodes[r]] for r in block]
        dist = dijkstra(network._csr, directed=True, indices=sources, limit=limit + EPS)
        table[block[0]:block[-1] + 1] = dist[:, columns]
        if progress is not None:
            progress(begin + len(block))
    return table


def compute_table(instance, path: Path) -> dict:
    """Table of one instance (rows and columns in oracle_targets order)."""
    instance.oracle.prepare_acceleration()
    nodes = oracle_targets(instance)
    path = Path(path)
    start = time.monotonic()
    table = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(len(nodes), len(nodes)))
    compute_rows(instance.network, nodes, range(len(nodes)), table_limit(instance), table)
    table.flush()
    del table
    np.save(path.with_suffix('.nodes.npy'), np.asarray(nodes, dtype=np.int64))
    info = {'rows': len(nodes), 'limit_minutes': table_limit(instance), 'seconds': time.monotonic() - start}
    path.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


_FORK_NETWORK = None


def _table_part(path, nodes, first, last, limit):
    table = np.load(path, mmap_mode='r+')
    compute_rows(_FORK_NETWORK, nodes, range(first, last), limit, table)
    table.flush()


def compute_union_table(instances, path: Path, workers: int = 2) -> dict:
    """One table for several instances on the same network (for example all classes of a data seed).

    Rows and columns are the union of the instances' port nodes; the bound is the
    largest instance search limit. Rows are split across `workers` forked processes.
    """
    global _FORK_NETWORK
    import multiprocessing
    network = instances[0].network
    if any(inst.network is not network for inst in instances):
        raise ValueError('All instances must share one network object.')
    instances[0].oracle.prepare_acceleration()
    start = time.monotonic()
    nodes = sorted({n for inst in instances for n in oracle_targets(inst)}, key=str)
    limit = max(table_limit(inst) for inst in instances)
    path = Path(path)
    partial = path.with_name(path.stem + '.partial.npy')
    table = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float64, shape=(len(nodes), len(nodes)))
    del table
    bounds = np.linspace(0, len(nodes), workers + 1).astype(int)
    _FORK_NETWORK = network
    context = multiprocessing.get_context('fork')
    processes = [context.Process(target=_table_part, args=(partial, nodes, int(a), int(b), limit))
                 for a, b in zip(bounds, bounds[1:])]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    _FORK_NETWORK = None
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError('Travel-table computation failed.')
    partial.replace(path)
    np.save(path.with_suffix('.nodes.npy'), np.asarray(nodes, dtype=np.int64))
    info = {'rows': len(nodes), 'limit_minutes': float(limit), 'seconds': time.monotonic() - start,
            'workers': workers, 'instances': [inst.name for inst in instances]}
    path.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def preload_table(instance, path: Path) -> dict:
    """Serve every oracle row of the instance from a stored table (memory-mapped, read only).

    The table may cover more nodes than the instance needs (a union table of several
    instances); its node list is stored next to it. Rows are views of the mapped file.
    """
    oracle = instance.oracle
    oracle.prepare_acceleration()
    path = Path(path)
    table = np.load(path, mmap_mode='r').view(np.ndarray)
    nodes = [int(x) for x in np.load(path.with_suffix('.nodes.npy'))]
    info = json.loads(path.with_suffix('.json').read_text())
    bound = float(info['limit_minutes'])
    if bound + 1e-9 < table_limit(instance):
        raise ValueError(f'Table bound {bound} is below the instance search limit {table_limit(instance)}.')
    index = {node: q for q, node in enumerate(nodes)}
    missing = [n for n in oracle_targets(instance) if n not in index]
    if missing:
        raise ValueError(f'Table {path} lacks {len(missing)} nodes of the instance.')
    if table.shape != (len(nodes), len(nodes)):
        raise ValueError(f'Table {path} has shape {table.shape}, expected {(len(nodes),) * 2}.')
    oracle._target_index = index
    oracle._target_columns = np.asarray([oracle.network._node_index[n] for n in nodes], dtype=np.int64)
    oracle.clear_cache()
    oracle.cache_sources = len(nodes) + 64
    oracle.cache_byte_budget = None
    for row, node in enumerate(nodes):
        oracle._cache[node] = (bound, table[row])
    oracle.table_array = table
    oracle.table_index = index
    oracle.table_bound = bound
    return info
