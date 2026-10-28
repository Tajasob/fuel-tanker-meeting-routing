"""GraphML road network and driver-path loader.

Link travel times follow the speed rule of the benchmark generator: the posted
maximum speed when OpenStreetMap records one, otherwise a speed by road class.
No road or instance is edited. Reverse arcs are merged into one physical link only
when geometry and OpenStreetMap identity both match.
"""
from __future__ import annotations

import ast
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .road import PhysicalLink, RoadNetwork

SPEEDS = {'motorway':80,'motorway_link':45,'trunk':65,'trunk_link':40,
          'primary':50,'primary_link':35,'secondary':40,'secondary_link':30,
          'tertiary':35,'tertiary_link':25,'residential':25,'living_street':15,
          'unclassified':25,'service':15}


def _value(text):
    if text and text.startswith('[') and text.endswith(']'):
        return ast.literal_eval(text)
    return text


def benchmark_speed(maxspeed, highway):
    values=[]
    for item in maxspeed if isinstance(maxspeed,list) else ([] if maxspeed is None else [maxspeed]):
        text=str(item).lower().replace('kph','').replace('km/h','').strip()
        token=''.join(c for c in text if c.isdigit() or c=='.')
        if token:
            try:values.append(float(token))
            except ValueError:pass
    if values:
        return max(5.0,min(110.0,sum(values)/len(values)))
    if isinstance(highway,list):highway=highway[0] if highway else 'residential'
    return float(SPEEDS.get(str(highway),25))


def load_graphml(path: Path, *, tanker_time_factor=1.0) -> RoadNetwork:
    if not math.isfinite(tanker_time_factor) or tanker_time_factor<=0:
        raise ValueError('Tanker time factor must be positive.')
    keys={}; nodes={}; groups={}; fingerprint={}; driver_arcs={}; loops=[]
    edge_count=0
    for _,element in ET.iterparse(path,events=('end',)):
        kind=element.tag.rsplit('}',1)[-1]
        if kind=='key':
            keys[element.attrib['id']]=element.attrib.get('attr.name',element.attrib['id'])
        elif kind=='node':
            attrs={keys[x.attrib['key']]:x.text for x in element if x.tag.endswith('data')}
            nodes[int(element.attrib['id'])]=(float(attrs['x']),float(attrs['y']))
            element.clear()
        elif kind=='edge':
            edge_count+=1
            u,v=int(element.attrib['source']),int(element.attrib['target'])
            attrs={keys[x.attrib['key']]:_value(x.text) for x in element if x.tag.endswith('data')}
            length=float(attrs.get('length',1) or 1)
            base_time=length/(benchmark_speed(attrs.get('maxspeed'),attrs.get('highway'))*1000/60)
            if u==v:
                loops.append((u,v,base_time*tanker_time_factor))
                element.clear()
                continue
            geometry=attrs.get('geometry')
            if geometry:
                if not geometry.upper().startswith('LINESTRING '):
                    raise ValueError(f'Unsupported geometry for {u}->{v}: {geometry[:30]}')
                coords=tuple(tuple(float(x) for x in part.strip().split()) for part in geometry[geometry.index('(')+1:geometry.rindex(')')].split(','))
                if any(len(point)!=2 for point in coords):raise ValueError('Expected 2-D coordinates.')
                # OSMnx geometries are normally oriented u->v; enforce this.
                near=lambda a,b:sum((x-y)**2 for x,y in zip(a,b))
                if near(coords[-1],nodes[u])<near(coords[0],nodes[u]):coords=coords[::-1]
            else:
                coords=(nodes[u],nodes[v])
            tail,head=min(u,v),max(u,v)
            canonical_coords=coords if u==tail else coords[::-1]
            osm=attrs.get('osmid')
            osm_ids=tuple(sorted(str(x) for x in (osm if isinstance(osm,list) else [osm])))
            if osm is None:
                # Do not infer shared pavement for unidentified reciprocal arcs.
                osm_ids=(f'unknown-directed:{u}:{v}:{element.attrib.get("id",edge_count)}',)
            identity=(tail,head,osm_ids,canonical_coords)
            encoded=json.dumps(identity,separators=(',',':'))
            link_id=hashlib.sha256(encoded.encode()).hexdigest()[:24]
            if link_id in fingerprint and fingerprint[link_id]!=identity:
                raise ValueError('Physical identity hash collision.')
            fingerprint[link_id]=identity
            highway=attrs.get('highway','unclassified')
            classes=tuple(sorted(str(x) for x in (highway if isinstance(highway,list) else [highway])))
            direction='forward_min' if u==tail else 'reverse_min'
            value=base_time*tanker_time_factor
            if link_id not in groups:
                groups[link_id]=PhysicalLink(link_id,tail,head,value if u==tail else None,
                                             value if u==head else None,length,canonical_coords,classes)
            else:
                old=groups[link_id]
                if abs(old.length_m-length)>max(1e-3,1e-5*length):
                    raise ValueError(f'Conflicting lengths on matched physical link {link_id}')
                old_rate=getattr(old,direction)
                groups[link_id]=replace(old,**{direction:min(value,old_rate if old_rate is not None else math.inf)},
                                        road_classes=tuple(sorted(set(old.road_classes)|set(classes))))
            driver_arcs.setdefault((u,v),[]).append((link_id,u==tail,base_time,length))
            element.clear()
    network=RoadNetwork(list(groups.values()),nodes,loops)
    network.driver_arcs=driver_arcs
    network.provenance={'graphml':str(path.resolve()),'nodes':len(nodes),'directed_edges':edge_count,
                        'physical_links':len(groups),'self_loops':len(loops),
                        'tanker_time_factor':tanker_time_factor,
                        'speed_rule':'posted maximum speed, otherwise a speed by road class',
                        'reverse_identity':'same endpoints, OSM IDs and complete oriented geometry',
                        'turn_restrictions':'not supplied by original graph'}
    return network


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def match_trajectory_arc(network,u,v,recorded_time,recorded_length, *, tolerance=2.1e-6):
    options=getattr(network,'driver_arcs',{}).get((u,v),[])
    matches=[x for x in options if abs(x[2]-recorded_time)<=tolerance and abs(x[3]-recorded_length)<=tolerance]
    # Multiple copies of exactly the same physical road are harmless; different
    # parallel geometries with indistinguishable records require source repair.
    identities={(x[0],x[1]) for x in matches}
    if len(identities)!=1:
        raise ValueError(f'Trajectory {u}->{v}: {len(identities)} physical matches; time={recorded_time}, length={recorded_length}')
    return min(matches,key=lambda x:abs(x[2]-recorded_time)+abs(x[3]-recorded_length))
