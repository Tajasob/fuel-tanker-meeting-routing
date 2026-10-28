"""Restricted meeting sites: junctions of a driver path with a local access street.

A road node is an eligible site when at least one incident physical link is a local
access street (OpenStreetMap classes residential, living_street, unclassified) and no
incident link carries through traffic (motorway, trunk, primary, their links, or a
busway). Link interiors are never eligible in this scenario.
"""
from __future__ import annotations

from .road import NodeLocation

LOCAL_ACCESS = frozenset({'residential', 'living_street', 'unclassified'})
THROUGH_TRAFFIC = frozenset({'motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link',
                             'busway'})
LABEL = 'local_access_junctions'


def local_access_nodes(network) -> frozenset:
    local, through = set(), set()
    for link in network.links.values():
        classes = set(link.road_classes)
        for node in (link.tail, link.head):
            if classes & LOCAL_ACCESS:
                local.add(node)
            if classes & THROUGH_TRAFFIC:
                through.add(node)
    return frozenset(local - through)


def candidate_predicate(nodes):
    """Eligibility predicate for Instance(..., eligibility=...)."""
    def eligible(candidate):
        return isinstance(candidate.location, NodeLocation) and candidate.location.node in nodes
    return eligible
