"""Directed travel on physical road links between road nodes and points inside links.

Locations at link endpoints are road nodes, independent of trajectory identity.
Each link is a geometric road section, with independently weighted legal arcs.
The oracle returns SHORTEST travel time, not a waiting-cost-dependent detour.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import heapq
import math
from typing import Hashable

EPS = 1e-9
Node = Hashable


@dataclass(frozen=True)
class NodeLocation:
    node: Node


@dataclass(frozen=True)
class LinkLocation:
    link_id: str
    fraction: float  # measured from PhysicalLink.tail to PhysicalLink.head


Location = NodeLocation | LinkLocation


@dataclass(frozen=True)
class PhysicalLink:
    link_id: str
    tail: Node
    head: Node
    forward_min: float | None
    reverse_min: float | None = None
    length_m: float = 0.0
    coordinates: tuple[tuple[float, float], ...] = ()  # lon/lat, tail to head
    road_classes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.tail == self.head:
            raise ValueError('A loop needs explicit ordered geometry states, not a two-end link.')
        for value in (self.forward_min, self.reverse_min):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError('Travel times must be finite and nonnegative.')
        if self.forward_min is None and self.reverse_min is None:
            raise ValueError('A physical link needs at least one legal directed arc.')


@dataclass(frozen=True)
class Travel:
    minutes: float
    mode: str
    exit_node: Node | None = None
    entry_node: Node | None = None


class RoadNetwork:
    def __init__(self, links: list[PhysicalLink], node_coordinates=None, extra_arcs=()):
        self.links = {link.link_id: link for link in links}
        if len(self.links) != len(links):
            raise ValueError('Physical-link IDs must be unique.')
        self.node_coordinates = dict(node_coordinates or {})
        self.adjacency: dict[Node, dict[Node, float]] = {}
        self.directed_links: dict[tuple[Node, Node], list[tuple[str, bool]]] = {}
        for link in links:
            self.adjacency.setdefault(link.tail, {})
            self.adjacency.setdefault(link.head, {})
            for u, v, t, forward in ((link.tail, link.head, link.forward_min, True),
                                      (link.head, link.tail, link.reverse_min, False)):
                if t is not None:
                    self.adjacency[u][v] = min(t, self.adjacency[u].get(v, math.inf))
                    self.directed_links.setdefault((u, v), []).append((link.link_id, forward))
        # Directed self-loops do not improve nonnegative shortest distances. They
        # remain in provenance; trajectory traversal of one must be rejected.
        for u, v, t in extra_arcs:
            if not math.isfinite(t) or t < 0:
                raise ValueError('Invalid extra arc.')
            self.adjacency.setdefault(u, {})[v] = min(t, self.adjacency.get(u, {}).get(v, math.inf))
            self.adjacency.setdefault(v, {})
        for node in self.node_coordinates:
            self.adjacency.setdefault(node, {})

    def location(self, link_id: str, fraction: float) -> Location:
        if not math.isfinite(fraction) or not -EPS <= fraction <= 1 + EPS:
            raise ValueError('Location fraction must lie in [0,1].')
        link = self.links[link_id]
        # Never snap a positive interior fraction to a node: directed access can
        # be discontinuous there, even if the geometric displacement is tiny.
        if fraction <= 0:
            return NodeLocation(link.tail)
        if fraction >= 1:
            return NodeLocation(link.head)
        return LinkLocation(link_id, float(fraction))

    def canonical(self, location: Location) -> Location:
        if isinstance(location, NodeLocation):
            if location.node not in self.adjacency:
                raise KeyError(f'Unknown road node {location.node}')
            return location
        return self.location(location.link_id, location.fraction)

    def coordinates(self, location: Location) -> tuple[float, float]:
        location = self.canonical(location)
        if isinstance(location, NodeLocation):
            return self.node_coordinates[location.node]
        link = self.links[location.link_id]
        coords = link.coordinates or (self.node_coordinates[link.tail], self.node_coordinates[link.head])
        distances = [0.0]
        for a, b in zip(coords, coords[1:]):
            # Spherical distances preserve curved polyline geometry; fraction is
            # a length fraction under the constant speed on this physical link.
            lon1, lat1, lon2, lat2 = map(math.radians, (*a, *b))
            h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
            distances.append(distances[-1]+2*6371000*math.asin(min(1,math.sqrt(h))))
        target = location.fraction*distances[-1]
        for k in range(1,len(coords)):
            if distances[k]+EPS >= target:
                span = distances[k]-distances[k-1]
                u = 0 if span <= EPS else (target-distances[k-1])/span
                return tuple((1-u)*a+u*b for a,b in zip(coords[k-1],coords[k]))
        return coords[-1]


class TravelOracle:
    """Exact shortest travel for node/interior-link locations on a fixed graph.

    An interior source can exit only in a legal direction; an interior target
    can be entered only in a legal direction. Same-link direct travel is added
    before endpoint paths. This handles all network junctions automatically.
    Turn restrictions and traffic dynamics absent from the source graph are not
    inferred. search_limit affects caching only, never the returned optimum.
    """
    def __init__(self, network: RoadNetwork, *, search_limit: float = 0, cache_sources: int = 64, backend='auto',pinned_sources=()):
        if cache_sources<1:raise ValueError('cache_sources must be at least one.')
        if backend not in ('auto','heap','scipy'):raise ValueError('Unknown shortest-path backend.')
        self.network = network
        self.search_limit = max(0,search_limit)
        self.cache_sources = cache_sources
        self.pinned_sources=frozenset(pinned_sources)
        if len(self.pinned_sources)>cache_sources:raise ValueError('Cache must hold all pinned vehicle sources.')
        self._cache = OrderedDict()
        self.dijkstra_runs = 0
        self._target_index=None
        self.cache_byte_budget=None
        self.outside_target_runs=0
        self.backend='heap'
        if backend in ('auto','scipy'):
            try:
                from scipy.sparse.csgraph import dijkstra
                self._scipy_dijkstra=dijkstra
                self.backend='scipy'
            except ImportError:
                if backend=='scipy':raise

    def clear_cache(self):
        """Reset per-method search state; immutable graph/CSR can remain shared."""
        self._cache.clear()
        self.dijkstra_runs=0
        self.outside_target_runs=0

    def restrict_cache_targets(self,targets,*,nonpinned_byte_budget=64*1024*1024):
        """Store only known instance port columns; unknown target queries remain
        exact via an uncached full-graph computation. The nonpinned row budget is
        bounded separately from the pinned vehicle rows and reported in stats.
        """
        if self.backend!='scipy':return
        self.prepare_acceleration()
        targets=tuple(dict.fromkeys(targets))
        if not targets:raise ValueError('At least one target is required for compact rows.')
        import numpy as np
        self._target_index={n:i for i,n in enumerate(targets)}
        self._target_columns=np.asarray([self.network._node_index[n] for n in targets],dtype=int)
        row_bytes=8*len(targets)
        self.cache_byte_budget=nonpinned_byte_budget
        self.cache_sources=len(self.pinned_sources)+max(2,nonpinned_byte_budget//row_bytes)
        self.clear_cache()

    def statistics(self):
        return {'backend':self.backend,'dijkstra_runs':self.dijkstra_runs,'outside_target_runs':self.outside_target_runs,
            'cached_sources':len(self._cache),'cache_capacity':self.cache_sources,
            'pinned_sources':len(self.pinned_sources),'compact_target_count':len(self._target_index) if self._target_index is not None else None,
            'cached_row_bytes':sum(getattr(row,'nbytes',0) for _,row in self._cache.values()),
            'nonpinned_row_byte_budget':self.cache_byte_budget,
            'pin_minimum_bytes':len(self.pinned_sources)*8*(len(self._target_index) if self._target_index is not None else len(self.network.adjacency))}

    def prepare_acceleration(self):
        if self.backend!='scipy' or hasattr(self.network,'_csr'):return
        import numpy as np
        from scipy.sparse import csr_matrix
        nodes=list(self.network.adjacency)
        index={n:i for i,n in enumerate(nodes)}
        rows=[];cols=[];values=[]
        for u,arcs in self.network.adjacency.items():
            for v,w in arcs.items():
                rows.append(index[u]);cols.append(index[v]);values.append(w)
        # Explicit zero-weight stored entries are valid scipy graph arcs.
        self.network._csr=csr_matrix((np.asarray(values,dtype=float),(rows,cols)),shape=(len(nodes),len(nodes)))
        self.network._node_index=index

    def node_time(self, source: Node, target: Node, cutoff=math.inf) -> float:
        if source == target:
            return 0.0 if cutoff >= -EPS else math.inf
        if cutoff < -EPS:
            return math.inf
        if self._target_index is not None and target not in self._target_index:
            # Do not mistake a column omitted for storage for an unreachable node.
            row=self._scipy_dijkstra(self.network._csr,directed=True,
                indices=self.network._node_index[source],limit=cutoff+EPS)
            self.dijkstra_runs+=1;self.outside_target_runs+=1
            result=float(row[self.network._node_index[target]])
            return result if result<=cutoff+EPS else math.inf
        cached = self._cache.get(source)
        bound = max(cutoff, self.search_limit)
        # Each row covers exactly its recorded bound plus the common timing
        # tolerance. A later larger bound needs a new search even if the bound
        # increment itself is smaller than EPS.
        if cached is None or cached[0] < bound:
            if self.backend=='scipy':
                self.prepare_acceleration()
                dist=self._scipy_dijkstra(self.network._csr,directed=True,
                    indices=self.network._node_index[source],limit=bound+EPS)
                if self._target_index is not None:dist=dist[self._target_columns]
            else:
                dist = {source: 0.0}
                serial = 0
                heap = [(0.0, serial, source)]
                while heap:
                    value, _, node = heapq.heappop(heap)
                    if value > dist[node]+EPS:
                        continue
                    for nxt, step in self.network.adjacency[node].items():
                        value2 = value+step
                        if value2 <= bound+EPS and value2 < dist.get(nxt, math.inf):
                            dist[nxt] = value2
                            serial += 1
                            heapq.heappush(heap,(value2,serial,nxt))
            cached = (bound,dist)
            self._cache[source] = cached
            self.dijkstra_runs += 1
            if len(self._cache)>self.cache_sources:
                victim=next((n for n in self._cache if n not in self.pinned_sources),None)
                if victim is not None and victim!=source:del self._cache[victim]
                elif victim==source:
                    del self._cache[source]
        if source in self._cache:self._cache.move_to_end(source)
        result = cached[1].get(target,math.inf) if self.backend=='heap' else float(cached[1][(self._target_index if self._target_index is not None else self.network._node_index)[target]])
        return result if result <= cutoff+EPS else math.inf

    def _ports(self, location: Location, departure: bool):
        if isinstance(location,NodeLocation):
            return [(location.node,0.0)]
        link = self.network.links[location.link_id]
        f = location.fraction
        ports=[]
        if link.forward_min is not None:
            ports.append((link.head,(1-f)*link.forward_min) if departure else (link.tail,f*link.forward_min))
        if link.reverse_min is not None:
            ports.append((link.tail,f*link.reverse_min) if departure else (link.head,(1-f)*link.reverse_min))
        return ports

    def node_times(self,source,targets,cutoff=math.inf):
        """Batch a row lookup; caches one shortest-path row, not all node pairs."""
        targets=tuple(targets)
        if self._target_index is not None and any(n not in self._target_index for n in targets):
            return [self.node_time(source,n,cutoff=cutoff) for n in targets]
        representative=next((n for n in targets if n!=source),None)
        if representative is not None:self.node_time(source,representative,cutoff=cutoff)
        cached=self._cache.get(source)
        if cached is None:
            return [self.node_time(source,n,cutoff=cutoff) for n in targets]
        if self.backend=='scipy':
            import numpy as np
            index=self._target_index if self._target_index is not None else self.network._node_index
            values=cached[1][[index[n] for n in targets]]
            return np.where(values<=cutoff+EPS,values,math.inf)
        return [cached[1].get(n,math.inf) if cached[1].get(n,math.inf)<=cutoff+EPS else math.inf for n in targets]

    def travel(self, source: Location, target: Location, *, cutoff=math.inf) -> Travel:
        if cutoff < -EPS:
            return Travel(math.inf,'unreachable')
        source = self.network.canonical(source)
        target = self.network.canonical(target)
        if source == target:
            return Travel(0.0,'same_location')
        best = Travel(math.inf,'unreachable')
        if isinstance(source,LinkLocation) and isinstance(target,LinkLocation) and source.link_id==target.link_id:
            link = self.network.links[source.link_id]
            diff = target.fraction-source.fraction
            rate = link.forward_min if diff >= 0 else link.reverse_min
            if rate is not None and abs(diff)*rate<=cutoff+EPS:
                best=Travel(abs(diff)*rate,'direct_same_link')
        for exit_node,exit_time in self._ports(source,True):
            for entry_node,entry_time in self._ports(target,False):
                remaining=min(cutoff,best.minutes)-exit_time-entry_time
                if remaining < -EPS:
                    continue
                middle=self.node_time(exit_node,entry_node,cutoff=max(0,remaining))
                value=exit_time+middle+entry_time
                if value<best.minutes and value<=cutoff+EPS:
                    best=Travel(value,'network',exit_node,entry_node)
        return best
