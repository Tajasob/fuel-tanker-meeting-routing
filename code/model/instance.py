"""Requests with full road-link driver paths, tankers, and the finite meeting candidates."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math

from .road import Location,NodeLocation,LinkLocation,RoadNetwork,TravelOracle,EPS
from .loader import read_csv,match_trajectory_arc


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    cum_time: float


@dataclass(frozen=True)
class PathLeg:
    link_id: str
    forward: bool
    driver_minutes: float
    length_m: float


@dataclass
class Customer:
    cid: int
    legs: tuple[PathLeg,...]
    nodes: tuple[int,...]
    breakpoints: tuple[Point,...]
    start: float
    service: float
    penalty: float
    latest_time: float | None = None

    @property
    def urgent(self):return self.latest_time is not None
    @property
    def seg_times(self):return (0.0,)+tuple(leg.driver_minutes for leg in self.legs)
    @property
    def seg_l1(self):return (0.0,)+tuple(leg.length_m for leg in self.legs)
    @property
    def total_path(self):return sum(leg.driver_minutes for leg in self.legs)
    @property
    def total_path_raw(self):return self.total_path


@dataclass(frozen=True)
class Vehicle:
    vid: int
    location: Location
    depot_x: float
    depot_y: float
    t0: float
    travel_cost: float
    waiting_cost: float
    fixed_cost: float
    service_cost: float


@dataclass(frozen=True)
class Candidate:
    cid: int
    segment: int  # full road-link index, NOT the original coarse segment index
    u: float      # fraction in driver's direction
    meet_time: float
    x: float
    y: float
    location: Location
    provenance: str = 'grid'


class Instance:
    def __init__(self,network:RoadNetwork,customers:dict,vehicles:dict,*,name='instance',
                 eligibility=None,eligibility_label='all_locations',add_depot_catchpoints=True):
        self.network=network
        self.customers=customers
        self.vehicles=vehicles
        for cid,c in customers.items():
            if cid!=c.cid or len(c.nodes)!=len(c.legs)+1 or len(c.breakpoints)!=len(c.nodes):
                raise ValueError('Inconsistent request/path indexing.')
            if any(not math.isfinite(x) or x<0 for x in (c.start,c.service,c.penalty)):
                raise ValueError('Request time, service, and penalty must be finite and nonnegative.')
            # Positive service is required by the real-arithmetic model. Its
            # numerical implementation additionally excludes services swallowed
            # by the absolute feasibility tolerance; this is not a guarantee
            # against all roundoff at arbitrary input scales.
            if c.service<=EPS:
                raise ValueError('Service duration must exceed the timing tolerance EPS.')
            if c.latest_time is not None and (not math.isfinite(c.latest_time) or c.latest_time<0):
                raise ValueError('Invalid individual deadline.')
            if c.breakpoints and abs(c.breakpoints[0].cum_time)>EPS:raise ValueError('Path time must start at zero.')
            for s,leg in enumerate(c.legs):
                if leg.driver_minutes<=0 or not math.isfinite(leg.driver_minutes):raise ValueError('Invalid driver link duration.')
                link=network.links[leg.link_id]
                if (c.nodes[s],c.nodes[s+1])!=((link.tail,link.head) if leg.forward else (link.head,link.tail)):
                    raise ValueError('Driver link does not match trajectory node order.')
                if (link.forward_min if leg.forward else link.reverse_min) is None:
                    raise ValueError('Driver path runs against the supplied road direction.')
                if abs(c.breakpoints[s+1].cum_time-c.breakpoints[s].cum_time-leg.driver_minutes)>1e-7:
                    raise ValueError('Inconsistent driver passage times.')
        for vid,v in vehicles.items():
            if vid!=v.vid:raise ValueError('Inconsistent vehicle ID.')
            network.canonical(v.location)
            if any(not math.isfinite(x) or x<0 for x in (v.t0,v.travel_cost,v.waiting_cost,v.fixed_cost,v.service_cost)):
                raise ValueError('Release and vehicle cost rates must be finite and nonnegative.')
        self.name=name
        self.eligibility=eligibility
        self.eligibility_label=eligibility_label
        if eligibility is None and eligibility_label!='all_locations':
            raise ValueError('A restricted eligibility label needs a screening predicate.')
        self.add_depot_catchpoints=add_depot_catchpoints
        self._candidate_cache={}
        self.urgent_fuel_range=math.inf  # individual latest_time is authoritative
        horizon=max((c.start+c.total_path for c in customers.values()),default=0)
        earliest=min((v.t0 for v in vehicles.values()),default=0)
        vehicle_sources=set()
        for v in vehicles.values():
            location=network.canonical(v.location)
            if isinstance(location,NodeLocation):vehicle_sources.add(location.node)
            else:
                link=network.links[location.link_id]
                if link.forward_min is not None:vehicle_sources.add(link.head)
                if link.reverse_min is not None:vehicle_sources.add(link.tail)
        # Preserve every depot row while generating all vehicle catch points;
        # otherwise >64 vehicles cause a full-cache miss at every road link.
        self.oracle=TravelOracle(network,search_limit=max(0,horizon-earliest),
            cache_sources=max(64,len(vehicle_sources)+32),pinned_sources=vehicle_sources)
        ports=vehicle_sources|{node for c in customers.values() for node in c.nodes}
        if ports:self.oracle.restrict_cache_targets(sorted(ports,key=str))
        self.provenance={'travel_policy':'directed shortest road travel',
                         'departure_policy':'fixed release, with all arrival waiting charged',
                         'eligibility':eligibility_label,
                         'meeting_domain':'continuous road trajectories, evaluated on explicit finite candidates'}

    def reset_caches(self,*,candidates=True,travel=True):
        """Reset instance query state between methods; excludes immutable graph/CSR.

        Candidate generation (including catch-point shortest paths) must either
        be timed for every method or prepared identically before every run.
        A newly constructed RouteEvaluator has separate route/solo caches.
        """
        if candidates:self._candidate_cache.clear()
        if travel:self.oracle.clear_cache()

    def cdb(self,vid,cid,b):
        return self.oracle.travel(self.vehicles[vid].location,NodeLocation(self.customers[cid].nodes[b])).minutes

    def cbb(self,i1,b1,i2,b2):
        return self.oracle.travel(NodeLocation(self.customers[i1].nodes[b1]),NodeLocation(self.customers[i2].nodes[b2])).minutes

    def _candidate(self,cid,segment,u,provenance):
        customer=self.customers[cid]
        leg=customer.legs[segment-1]
        location=self.network.location(leg.link_id,u if leg.forward else 1-u)
        when=customer.start+customer.breakpoints[segment-1].cum_time+u*leg.driver_minutes
        x,y=self.network.coordinates(location)
        return Candidate(cid,segment,u,when,x,y,location,provenance)

    def candidates(self,cid,samples=2):
        if not isinstance(samples,int) or samples<1:raise ValueError('samples must be a positive integer.')
        key=(cid,samples,self.add_depot_catchpoints)
        if key in self._candidate_cache:return self._candidate_cache[key]
        c=self.customers[cid]
        latest=c.latest_time if c.latest_time is not None else math.inf
        values={}
        for s,leg in enumerate(c.legs,1):
            fractions=[(q/samples,'grid') for q in range(samples+1)]
            at_start=c.start+c.breakpoints[s-1].cum_time
            if leg.driver_minutes>0 and math.isfinite(latest):
                fractions.append(((latest-at_start)/leg.driver_minutes,'deadline'))
            if self.add_depot_catchpoints:
                link=self.network.links[leg.link_id]
                along=link.forward_min if leg.forward else link.reverse_min
                against=link.reverse_min if leg.forward else link.forward_min
                first=c.nodes[s-1]; second=c.nodes[s]
                for v in self.vehicles.values():
                    for entry,rate,head_on in ((first,along,False),(second,against,True)):
                        if rate is None:continue
                        denominator=(leg.driver_minutes+rate) if head_on else (rate-leg.driver_minutes)
                        if abs(denominator)<=EPS:continue
                        d=self.oracle.travel(v.location,NodeLocation(entry),cutoff=max(0,min(latest,at_start+leg.driver_minutes)-v.t0)).minutes
                        if not math.isfinite(d):continue
                        numerator=(v.t0+d+rate-at_start) if head_on else (at_start-v.t0-d)
                        fractions.append((numerator/denominator,'depot_catch'))
            for u,source in fractions:
                if not -EPS<=u<=1+EPS:continue
                cand=self._candidate(cid,s,min(1,max(0,u)),source)
                if cand.meet_time>latest+EPS:continue
                if self.eligibility is not None and not self.eligibility(cand):continue
                marker=(cand.location,round(cand.meet_time,10))
                values.setdefault(marker,cand)
        result=tuple(sorted(values.values(),key=lambda x:(x.meet_time,x.segment,x.u)))
        self._candidate_cache[key]=result
        return result

    def tau_depot_options(self,vid,cand):
        return [self.oracle.travel(self.vehicles[vid].location,cand.location).minutes]

    def tau_candidate_options(self,first,second):
        return [self.oracle.travel(first.location,second.location).minutes]

    def tau_depot_options_feasible(self,vid,cand,max_tau):
        t=self.oracle.travel(self.vehicles[vid].location,cand.location,cutoff=max_tau).minutes
        return [t] if math.isfinite(t) else []

    def tau_candidate_options_feasible(self,first,second,max_tau):
        t=self.oracle.travel(first.location,second.location,cutoff=max_tau).minutes
        return [t] if math.isfinite(t) else []


def load_instance(instance_dir:Path,network:RoadNetwork,*,remaining_from_epoch=True,
                  eligibility=None,eligibility_label='all_locations',add_depot_catchpoints=True):
    trajectories={}
    for row in read_csv(instance_dir/'trajectory_nodes.csv'):
        trajectories.setdefault(int(row['customer_id']),[]).append(row)
    customers={}; maximum_time_residual=maximum_length_residual=0.0
    maximum_cumulative_time_residual=maximum_cumulative_length_residual=0.0
    for row in read_csv(instance_dir/'requests.csv'):
        cid=int(row['customer_id'])
        path=sorted(trajectories[cid],key=lambda x:int(x['seq']))
        if [int(x['seq']) for x in path]!=list(range(len(path))):raise ValueError('Trajectory sequence is incomplete.')
        if len(path)<2:raise ValueError('A parked path needs an explicit time-window state; not inferred.')
        nodes=tuple(int(x['node']) for x in path)
        legs=[];points=[]
        for r in path:
            node=int(r['node']); x,y=network.node_coordinates[node]
            if abs(x-float(r['lon']))>1e-7 or abs(y-float(r['lat']))>1e-7:raise ValueError('Trajectory coordinate mismatch.')
        if abs(float(path[0]['cum_time_min']))>EPS or abs(float(path[0]['cum_length_m']))>EPS:
            raise ValueError('Recorded trajectory must begin at cumulative time/length zero.')
        x,y=network.node_coordinates[nodes[0]]
        points.append(Point(x,y,0.0))
        cumulative_time=cumulative_length=0.0
        for a,b in zip(path,path[1:]):
            u,v=int(a['node']),int(b['node'])
            dt=float(b['cum_time_min'])-float(a['cum_time_min'])
            dl=float(b['cum_length_m'])-float(a['cum_length_m'])
            if dt<=0 or dl<=0:raise ValueError('Nonpositive trajectory link duration/length.')
            lid,forward,base_time,length=match_trajectory_arc(network,u,v,dt,dl)
            maximum_time_residual=max(maximum_time_residual,abs(base_time-dt))
            maximum_length_residual=max(maximum_length_residual,abs(length-dl))
            # The generator computed these full-precision edge sums, then wrote
            # cumulative CSV values to six decimals. Restore those original sums
            # only after verifying every recorded cumulative value. Differences
            # of rounded cumulative times would otherwise create tiny artificial
            # driver/tanker speed differences and unstable following catch roots.
            cumulative_time+=base_time;cumulative_length+=length
            time_error=abs(cumulative_time-float(b['cum_time_min']))
            length_error=abs(cumulative_length-float(b['cum_length_m']))
            maximum_cumulative_time_residual=max(maximum_cumulative_time_residual,time_error)
            maximum_cumulative_length_residual=max(maximum_cumulative_length_residual,length_error)
            if time_error>5.1e-7 or length_error>5.1e-7:
                raise ValueError(f'Reconstructed trajectory {cid} at node {v} exceeds CSV rounding tolerance: time={time_error}, length={length_error}')
            legs.append(PathLeg(lid,forward,base_time,length))
            x,y=network.node_coordinates[v]
            points.append(Point(x,y,cumulative_time))
        start=0.0 if remaining_from_epoch else float(row['start_time_min'])
        latest=None if not row.get('urgent_range_min') else start+float(row['urgent_range_min'])
        customers[cid]=Customer(cid,tuple(legs),nodes,tuple(points),start,float(row['service_time_min']),float(row['penalty']),latest)
    vehicles={}
    for r in read_csv(instance_dir/'vehicles.csv'):
        vid=int(r['vehicle_id']);loc=NodeLocation(int(r['node']))
        x,y=network.coordinates(loc)
        vehicles[vid]=Vehicle(vid,loc,x,y,float(r['available_time_min']),float(r['travel_cost']),
                              float(r['waiting_cost']),float(r['fixed_cost']),float(r['service_cost']))
    inst=Instance(network,customers,vehicles,name=instance_dir.name,eligibility=eligibility,
                  eligibility_label=eligibility_label,add_depot_catchpoints=add_depot_catchpoints)
    inst.provenance.update({'instance_dir':str(instance_dir.resolve()),'remaining_from_epoch':remaining_from_epoch,
                            'maximum_link_time_rounding_residual':maximum_time_residual,
                            'maximum_link_length_rounding_residual':maximum_length_residual,
                            'driver_time_source':'full-precision matched graph edge sums, verified against every six-decimal cumulative CSV timestamp',
                            'maximum_cumulative_time_rounding_residual':maximum_cumulative_time_residual,
                            'maximum_cumulative_length_rounding_residual':maximum_cumulative_length_residual})
    return inst
