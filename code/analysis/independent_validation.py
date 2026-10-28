"""Independent schedule replay using an augmented NetworkX road graph.

Does not call the route evaluator, TravelOracle, its caches, candidate generator,
or its endpoint-port formula. All scheduled interior points split physical links
into directed graph arcs, then NetworkX computes shortest travel independently.
Loader/raw road provenance remains a separate validation responsibility.
"""
from __future__ import annotations
import math
import networkx as nx

EPS=1e-9

def location_key(network,record):
    if not isinstance(record,dict):
        record={'node':record.node} if hasattr(record,'node') else {'link_id':record.link_id,'fraction':record.fraction}
    if 'node' in record:
        node=record['node']
        if node not in network.node_coordinates:raise ValueError('Unknown road node')
        return ('node',node)
    lid=record['link_id'];fraction=float(record['fraction']);link=network.links[lid]
    if not math.isfinite(fraction) or not -EPS<=fraction<=1+EPS:raise ValueError('Invalid road fraction')
    if fraction<=EPS:return ('node',link.tail)
    if fraction>=1-EPS:return ('node',link.head)
    return ('interior',lid,fraction)

def split_graph(network,locations):
    interior={}
    for rec in locations:
        key=location_key(network,rec)
        if key[0]=='interior':interior.setdefault(key[1],set()).add(key[2])
    graph=nx.DiGraph()
    graph.add_nodes_from(('node',n) for n in network.node_coordinates)
    def arc(a,b,w):
        if not graph.has_edge(a,b) or w<graph[a][b]['minutes']:graph.add_edge(a,b,minutes=w)
    for lid,link in network.links.items():
        points=[(0.,('node',link.tail))]+[(f,('interior',lid,f)) for f in sorted(interior.get(lid,()))]+[(1.,('node',link.head))]
        for (f,a),(g,b) in zip(points,points[1:]):
            if link.forward_min is not None:arc(a,b,(g-f)*link.forward_min)
            if link.reverse_min is not None:arc(b,a,(g-f)*link.reverse_min)
    return graph

def point_coordinates(network,key):
    if key[0]=='node':return network.node_coordinates[key[1]]
    link=network.links[key[1]]
    coordinates=getattr(link,'coordinates',()) or (network.node_coordinates[link.tail],network.node_coordinates[link.head])
    lengths=[]
    for (lon1,lat1),(lon2,lat2) in zip(coordinates,coordinates[1:]):
        dlat=math.radians(lat2-lat1);dlon=math.radians(lon2-lon1)
        a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        lengths.append(2*math.atan2(math.sqrt(max(0,a)),math.sqrt(max(0,1-a))))
    target=sum(lengths)*key[2];traveled=0.
    for index,length in enumerate(lengths):
        if traveled+length>=target-1e-15:
            fraction=0. if length==0 else min(1,max(0,(target-traveled)/length))
            return tuple(a+fraction*(b-a) for a,b in zip(coordinates[index],coordinates[index+1]))
        traveled+=length
    return coordinates[-1]

def validate_schedule(instance,routes,schedule,reported_objective,*,approved_site_checker=None,tolerance=1e-5,samples=None):
    errors=[];checks=0;travel_cache={}
    reported_objective=float(reported_objective)
    if not math.isfinite(float(reported_objective)):errors.append('Nonfinite reported objective')
    rows=[dict(row) for row in schedule]
    assignments=[cid for route in routes.values() for cid in route]
    if len(assignments)!=len(set(assignments)):errors.append('Duplicate request assignment')
    if not set(assignments)<=set(instance.customers):errors.append('Unknown assigned request')
    if not set(routes)<=set(instance.vehicles):errors.append('Unknown vehicle')
    if len(rows)!=len(assignments):errors.append('Schedule and assignment lengths differ')
    if errors:return dict(passed=False,errors=errors,independent=True)
    try:
        graph=split_graph(instance.network,[v.location for v in instance.vehicles.values()]+[r['road_location'] for r in rows])
    except (ValueError,KeyError,TypeError) as e:
        return dict(passed=False,errors=errors+[f'Invalid location: {e}'],independent=True)
    components=dict(travel=0.,waiting=0.,service=0.,activation=0.,unserved_penalty=0.)
    detailed=dict(travel_from_depot=0.,travel_between_customers=0.,waiting_first_leg=0.,waiting_between_customers=0.,service=0.,fixed_vehicle=0.)
    def distance(a,b):
        key=(a,b)
        if key not in travel_cache:
            try:travel_cache[key]=nx.shortest_path_length(graph,a,b,weight='minutes')
            except (nx.NetworkXNoPath,nx.NodeNotFound):travel_cache[key]=math.inf
        return travel_cache[key]
    for vid,vehicle in instance.vehicles.items():
        route=list(routes.get(vid,[]));vr=sorted([r for r in rows if int(r['vehicle'])==vid],key=lambda r:int(r['position']))
        if [int(r['customer']) for r in vr]!=route or [int(r['position']) for r in vr]!=list(range(1,len(route)+1)):
            errors.append(f'Route/schedule identity mismatch for vehicle {vid}');continue
        if route:components['activation']+=vehicle.fixed_cost;detailed['fixed_vehicle']+=vehicle.fixed_cost
        source=location_key(instance.network,vehicle.location);departure=vehicle.t0
        for row in vr:
            cid=int(row['customer']);customer=instance.customers[cid];checks+=1
            try:
                if any(not math.isfinite(float(row[k])) for k in ['u','meeting_time','tau','arrive','waiting','service']):
                    raise ValueError('Nonfinite timing or path coordinate')
                segment=int(row['segment']);u=float(row['u'])
                if not 1<=segment<=len(customer.legs) or not -EPS<=u<=1+EPS:raise ValueError('Invalid path coordinate')
                leg=customer.legs[segment-1]
                expected_record={'link_id':leg.link_id,'fraction':u if leg.forward else 1-u}
                actual=location_key(instance.network,row['road_location']);expected=location_key(instance.network,expected_record)
                same=actual==expected or (actual[0]==expected[0]=='interior' and actual[1]==expected[1] and abs(actual[2]-expected[2])<1e-9)
                if not same:errors.append(f'Request {cid}: location differs from path coordinate')
                if 'x' in row and 'y' in row:
                    xy=point_coordinates(instance.network,actual)
                    if any(not math.isfinite(float(row[k])) or abs(float(row[k])-xy[j])>1e-8 for j,k in enumerate(['x','y'])):
                        errors.append(f'Request {cid}: reported coordinate differs from physical road geometry')
                meeting=customer.start+sum(x.driver_minutes for x in customer.legs[:segment-1])+u*leg.driver_minutes
                if abs(float(row['meeting_time'])-meeting)>tolerance:errors.append(f'Request {cid}: driver passage time mismatch')
                if customer.latest_time is not None and meeting>customer.latest_time+tolerance:errors.append(f'Request {cid}: deadline exceeded')
                if samples is not None:
                    origin=row.get('candidate_source','')
                    admitted=False
                    if origin=='grid':admitted=abs(u*samples-round(u*samples))<=1e-7
                    elif origin=='deadline':admitted=customer.latest_time is not None and abs(meeting-customer.latest_time)<=tolerance
                    elif origin=='depot_catch':
                        physical=instance.network.links[leg.link_id]
                        first=physical.tail if leg.forward else physical.head
                        second=physical.head if leg.forward else physical.tail
                        along=physical.forward_min if leg.forward else physical.reverse_min
                        against=physical.reverse_min if leg.forward else physical.forward_min
                        # the serving tanker's own release point first; stop at the first generator found
                        for depot in sorted(instance.vehicles.values(),key=lambda d:(d.vid!=vid,d.vid)):
                            depkey=location_key(instance.network,depot.location)
                            for entry,rate,fraction in [(first,along,u),(second,against,1-u)]:
                                if rate is None or admitted:continue
                                if depot.t0+fraction*rate>meeting+tolerance:continue
                                arrival_at_candidate=depot.t0+distance(depkey,('node',entry))+fraction*rate
                                if abs(arrival_at_candidate-meeting)<=tolerance:admitted=True
                            if admitted:break
                    if not admitted:errors.append(f'Request {cid}: candidate source/grid membership not reproduced')
                if approved_site_checker is not None and not approved_site_checker(cid,expected_record,meeting):errors.append(f'Request {cid}: site screening failed')
                travel=distance(source,actual);arrival=departure+travel
                # Keep independent graph travel, but honor the declared time
                # arithmetic: subtract departure before travel. Subtracting a
                # rounded absolute arrival can alter high-rate waiting costs.
                available=meeting-departure;waiting_residual=available-travel
                if not math.isfinite(travel) or waiting_residual < -tolerance:errors.append(f'Request {cid}: unreachable meeting')
                waiting=max(0.,waiting_residual)
                if not math.isfinite(float(row['tau'])) or abs(float(row['tau'])-travel)>tolerance:errors.append(f'Request {cid}: shortest travel mismatch')
                if abs(float(row['arrive'])-arrival)>tolerance:errors.append(f'Request {cid}: arrival mismatch')
                if abs(float(row['waiting'])-waiting)>tolerance:errors.append(f'Request {cid}: waiting mismatch')
                if abs(float(row['service'])-customer.service)>tolerance:errors.append(f'Request {cid}: service mismatch')
                components['travel']+=vehicle.travel_cost*travel
                components['waiting']+=vehicle.waiting_cost*waiting
                components['service']+=vehicle.service_cost*customer.service
                first_stop=int(row['position'])==1
                detailed['travel_from_depot' if first_stop else 'travel_between_customers']+=vehicle.travel_cost*travel
                detailed['waiting_first_leg' if first_stop else 'waiting_between_customers']+=vehicle.waiting_cost*waiting
                detailed['service']+=vehicle.service_cost*customer.service
                departure=meeting+customer.service;source=actual
            except (ValueError,KeyError,TypeError,IndexError) as e:errors.append(f'Request {cid}: invalid schedule row: {e}')
    components['unserved_penalty']=sum(c.penalty for cid,c in instance.customers.items() if cid not in set(assignments))
    total=sum(components.values());residual=abs(total-reported_objective)
    detailed.update(unserved_penalty=components['unserved_penalty'],total=total)
    if not math.isfinite(total) or residual>max(tolerance,1e-9*abs(total)):errors.append('Objective reconstruction mismatch')
    site_status='independent_predicate' if approved_site_checker is not None else 'all_locations'
    if approved_site_checker is None and instance.eligibility is not None:
        errors.append('Restricted eligibility requires an independent site predicate');site_status='not_independently_checked'
    return dict(passed=not errors,errors=errors,independent=True,method='augmented_link_split_graph_networkx',
                stops_checked=checks,graph_nodes=len(graph),graph_arcs=graph.number_of_edges(),components=components,
                detailed_components=detailed,candidate_membership_checked=samples is not None,
                objective_recomputed=total,objective_abs_error=residual,site_status=site_status,
                scope='feasibility and cost of the returned plan on the physical-link data')

LOCAL_ACCESS_CLASSES={'residential','living_street','unclassified'}
THROUGH_TRAFFIC_CLASSES={'motorway','motorway_link','trunk','trunk_link','primary','primary_link','busway'}

def local_access_site_checker(network):
    """Independent screen for the restricted-site scenario: the meeting point must be a
    road node with an incident local access street and no incident through-traffic road."""
    incident={}
    for link in network.links.values():
        for node in (link.tail,link.head):incident.setdefault(node,set()).update(link.road_classes)
    def check(cid,record,meeting):
        key=location_key(network,record)
        if key[0]!='node':return False
        classes=incident.get(key[1],set())
        return bool(classes&LOCAL_ACCESS_CLASSES) and not classes&THROUGH_TRAFFIC_CLASSES
    return check

def validate_result(network,instance,result,*,approved_site_checker=None):
    """Runner-compatible API; validates saved result.json dictionaries directly."""
    if network is not instance.network:
        raise ValueError('Network/instance mismatch')
    routes={int(v):[int(c) for c in route] for v,route in result['routes'].items()}
    report=validate_schedule(instance,routes,result['schedule'],float(result['objective']),samples=int(result['samples']),approved_site_checker=approved_site_checker)
    if 'detailed_components' in report:
        for key,value in report['detailed_components'].items():
            raw=result.get('components',{}).get(key)
            if raw is None or not math.isfinite(float(raw)) or abs(float(raw)-value)>max(1e-5,1e-9*abs(value)):
                report['errors'].append(f'Reported component {key} does not match independent replay')
    report['passed']=not report['errors']
    return report
