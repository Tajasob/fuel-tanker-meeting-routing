#!/usr/bin/env python3
"""Tables and text macros of the computational study, computed from the result records.

    python analysis/summarize.py [--results ../results] [--out ../generated]

The manuscript carries these tables and numbers inline; analysis/check_inline.py compares them.

Writes tables/*.tex (one file per table) and macros.tex (one command per number
quoted in the text). Records that did not pass the independent validation stop the
script, and so does any exact record whose plan is not proven optimal.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy.stats import wilcoxon

CLASSES = ('S01', 'S03', 'M02', 'M05', 'M10', 'L01', 'L06')
EXACT_CLASSES = ('S01', 'S03', 'M02', 'M05')
LARGE_CLASSES = ('M10', 'L01', 'L06')
BASE_METHODS = ('priority-greedy', 'cheapest-insertion', 'singleton-assignment', 'alns')
PENALTY_METHODS = ('priority-greedy', 'cheapest-insertion', 'singleton-assignment', 'alns', 'seeding',
                   'seeding-alns', 'exact')
SHORT = {'priority-greedy': 'PG', 'cheapest-insertion': 'CI', 'singleton-assignment': 'SA', 'alns': 'ALNS',
         'alns-fixed': 'ALNS-F', 'alns-plain': 'ALNS-P', 'seeding': 'WS', 'seeding-alns': 'WS+ALNS', 'exact': 'IP'}
SCENARIO_NAME = {'base': 'Base', 'sites': 'Local sites', 'deadline10': 'Deadlines',
                 'penalty0.05': 'Penalties $\\times0.05$', 'penalty0.02': 'Penalties $\\times0.02$',
                 'destination': 'Destination', 'earliest': 'Earliest point'}
MAIN_SCENARIOS = ('base', 'sites', 'deadline10', 'penalty0.05', 'penalty0.02')
FIXED_SCENARIOS = ('destination', 'earliest')  # one meeting candidate per request
SEARCHES = ('alns', 'alns-fixed', 'alns-plain', 'seeding-alns')
TOL = 1e-6  # relative objective tolerance for "equal to the reference"
TAG = {'S01': 'Sone', 'S03': 'Sthree', 'M02': 'Mtwo', 'M05': 'Mfive', 'M10': 'Mten', 'L01': 'Lone', 'L06': 'Lsix'}


def load(results: Path):
    records = []
    for path in sorted(results.glob('*/*.json')):
        if path.name == 'machine.json':
            continue
        record = json.loads(path.read_text())
        if not record.get('validation', {}).get('passed'):
            raise SystemExit(f'Record without a passed validation: {path}')
        records.append(record)
    return records


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def fmt(value, digits=2):
    """Nature style: thousands separators only for numbers of five or more digits."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '--'
    if isinstance(value, int):
        return (f'{value:,}' if abs(value) >= 10000 else f'{value}').replace(',', '{,}')
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0
    big = abs(round(value, digits)) >= 10000
    return (f'{value:,.{digits}f}' if big else f'{value:.{digits}f}').replace(',', '{,}')


def pct_drop(before, after):
    return 100.0 * (before - after) / before


class Study:
    def __init__(self, records):
        self.records = records
        self.by = {}
        for r in records:
            key = (r['scenario']['name'], r['instance_class'], r['data_seed'], r['algorithm'], r.get('algorithm_seed'))
            if key in self.by:
                raise SystemExit(f'Duplicate record {key}')
            self.by[key] = r
        self.seeds = sorted({r['data_seed'] for r in records})
        exact = [r for r in records if r['algorithm'] == 'exact']
        not_optimal = [r['job'] for r in exact if not r['exact']['optimal']]
        if not_optimal:
            raise SystemExit(f'Exact records without proven optimality: {not_optimal}')
        for r in exact:  # no method may beat a proven optimum
            for other in self.instance_records(r['scenario']['name'], r['instance_class'], r['data_seed']):
                if other['objective'] < r['objective'] - TOL * abs(r['objective']):
                    raise SystemExit(f"{other['job']} ({other['data_seed']}) is below the optimum of {r['job']}")

    def get(self, scenario, cls, seed, method, algorithm_seed=None):
        if method in SEARCHES and algorithm_seed is None:
            algorithm_seed = 1
        return self.by.get((scenario, cls, seed, method, algorithm_seed))

    def instance_records(self, scenario, cls, seed):
        return [r for r in self.records
                if (r['scenario']['name'], r['instance_class'], r['data_seed']) == (scenario, cls, seed)]

    def optimum(self, scenario, cls, seed):
        r = self.get(scenario, cls, seed, 'exact')
        return None if r is None else r['objective']

    def best_known(self, scenario, cls, seed):
        return min(r['objective'] for r in self.instance_records(scenario, cls, seed))

    def reference(self, scenario, cls, seed):
        value = self.optimum(scenario, cls, seed)
        return self.best_known(scenario, cls, seed) if value is None else value

    def gap(self, record):
        ref = self.reference(record['scenario']['name'], record['instance_class'], record['data_seed'])
        return max(0.0, 100.0 * (record['objective'] - ref) / abs(ref))

    def hits(self, records):
        return sum(1 for r in records if self.gap(r) <= 100 * TOL)

    def runs(self, scenario, cls, method, algorithm_seed=None):
        out = [self.get(scenario, cls, seed, method, algorithm_seed) for seed in self.seeds]
        return [r for r in out if r is not None]


def multi_share(record):
    lengths = {int(k): v for k, v in record['route_lengths'].items()}
    used = sum(lengths.values())
    return 100.0 * sum(v for k, v in lengths.items() if k >= 2) / used if used else math.nan


def lp_gap(r):
    return max(0.0, 100.0 * (r['objective'] - r['exact']['lp_objective']) / abs(r['objective']))


def table(columns, header, body, rules=()):
    lines = [r'\begin{tabular}{' + columns + '}', r'\toprule'] + list(header) + list(rules) + [r'\midrule']
    lines += body + [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def table_instances(s: Study):
    body = []
    for cls in CLASSES:
        rs = s.runs('base', cls, 'priority-greedy')
        ex = s.runs('base', cls, 'exact')
        cols = fmt(round(mean(r['exact']['columns'] for r in ex))) if ex else '--'
        rows = fmt(round(mean(r['exact']['rows'] for r in ex))) if ex else '--'
        body.append(f"{cls} & {rs[0]['requests']} & {rs[0]['vehicles']} & {rs[0]['urgent_total']} & "
                    f"{fmt(mean(r['mean_path_minutes'] for r in rs), 1)} & "
                    f"{fmt(mean(r['generated_candidates'] / r['requests'] for r in rs), 1)} & "
                    f"{fmt(mean(r['usable_candidates'] / r['requests'] for r in rs), 1)} & {cols} & {rows}\\\\")
    header = [r'& & & & Mean path & \multicolumn{2}{c}{Candidates per request} & \multicolumn{2}{c}{Integer program}\\',
              r'\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
              r'Class & Requests & Tankers & Urgent & (min) & Generated & Usable & Columns & Rows\\']
    return table('lrrrrrrrr', header, body)


def table_exact(s: Study):
    body = []
    for scenario in ('base', 'sites', 'deadline10', 'penalty0.05', 'penalty0.02'):
        for cls in EXACT_CLASSES:
            ex = s.runs(scenario, cls, 'exact')
            if not ex:
                continue
            body.append(f"{SCENARIO_NAME[scenario]} & {cls} & "
                        f"{fmt(mean(r['exact']['solver_seconds'] for r in ex), 1)} & "
                        f"{fmt(max(r['exact']['solver_seconds'] for r in ex), 1)} & "
                        f"{fmt(mean(lp_gap(r) for r in ex), 3)} & {sum(1 for r in ex if lp_gap(r) <= 1e-4)} & "
                        f"{fmt(mean(r['served'] for r in ex), 1)} & {fmt(mean(r['used_vehicles'] for r in ex), 1)} & "
                        f"{fmt(mean(multi_share(r) for r in ex), 1)}\\\\")
    header = [r'& & \multicolumn{2}{c}{CPLEX time (s)} & \multicolumn{2}{c}{Linear relaxation} & '
              r'\multicolumn{3}{c}{Optimal plan}\\',
              r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}',
              r'Scenario & Class & Mean & Max & Gap (\%) & Tight & Served & Tankers & Shared (\%)\\']
    return table('llrrrrrrr', header, body)


def gap_cell(value):
    """Mean gap with two decimals; a mean above the optimum tolerance that would print as 0.00
    prints as <0.01."""
    return '$<$0.01' if 100 * TOL < value < 0.005 else fmt(value, 2)


def table_quality(s: Study):
    body = []
    for scenario in ('base', 'sites', 'deadline10'):
        for cls in EXACT_CLASSES:
            if not s.runs(scenario, cls, 'exact'):
                continue
            cells = []
            for method in BASE_METHODS:
                rs = s.runs(scenario, cls, method)
                cells.append(f'{gap_cell(mean(s.gap(r) for r in rs))} & {s.hits(rs)}')
            body.append(f"{SCENARIO_NAME[scenario]} & {cls} & " + ' & '.join(cells) + r'\\')
    head = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{SHORT[m]}}}' for m in BASE_METHODS)
    rule = ''.join(f'\\cmidrule(lr){{{3 + 2 * q}-{4 + 2 * q}}}' for q in range(len(BASE_METHODS)))
    sub = ' & '.join(['Gap (\\%) & Opt.'] * len(BASE_METHODS))
    header = [f'& & {head}\\\\', rule, f'Scenario & Class & {sub}\\\\']
    return table('ll' + 'rr' * len(BASE_METHODS), header, body)


def improvement(s: Study, cls, rule):
    """Mean relative decrease of the objective from a construction rule to ALNS (algorithm seed 1)."""
    rows = []
    for seed in s.seeds:
        a = s.get('base', cls, seed, 'alns')['objective']
        if rule == 'best':
            z = min(s.get('base', cls, seed, m)['objective'] for m in ('priority-greedy', 'cheapest-insertion'))
        else:
            z = s.get('base', cls, seed, rule)['objective']
        rows.append(100.0 * (z - a) / z)
    return mean(rows)


def table_large(s: Study):
    body = []
    for cls in LARGE_CLASSES:
        better = [fmt(improvement(s, cls, rule), 1) for rule in ('priority-greedy', 'cheapest-insertion',
                                                                  'singleton-assignment')]
        served = [fmt(mean(r['served'] for r in s.runs('base', cls, m)), 1) for m in BASE_METHODS]
        alns = s.runs('base', cls, 'alns')
        body.append(f"{cls} & " + ' & '.join(better + served) +
                    f" & {fmt(mean(r['method_seconds'] for r in alns), 1)} & "
                    f"{fmt(max(r['method_seconds'] for r in alns), 1)}\\\\")
    names = ' & '.join(SHORT[m] for m in BASE_METHODS)
    header = [r'& \multicolumn{3}{c}{Gain of ALNS (\%)} & \multicolumn{4}{c}{Served requests} & '
              r'\multicolumn{2}{c}{ALNS time (s)}\\',
              r'\cmidrule(lr){2-4}\cmidrule(lr){5-8}\cmidrule(lr){9-10}',
              f'Class & PG & CI & SA & {names} & Mean & Max\\\\']
    return table('l' + 'r' * 9, header, body)


def ablation_runs(s: Study, cls, method):
    return [r for seed in s.seeds for a in range(1, 6) for r in [s.get('base', cls, seed, method, a)] if r is not None]


def ablation_means(s: Study, cls, a, b):
    """Per-instance mean gaps over five algorithm seeds of two search variants."""
    xa, xb = [], []
    for seed in s.seeds:
        ga = [s.gap(r) for alg in range(1, 6) for r in [s.get('base', cls, seed, a, alg)] if r is not None]
        gb = [s.gap(r) for alg in range(1, 6) for r in [s.get('base', cls, seed, b, alg)] if r is not None]
        if ga and gb:
            xa.append(mean(ga))
            xb.append(mean(gb))
    return xa, xb


def ablation_test(s: Study, cls, a, b):
    """Exact two-sided Wilcoxon signed-rank test over the instances on which the mean gaps of five
    algorithm seeds differ. Returns the number n of these instances, the mean difference over all
    instances and the unadjusted P value."""
    xa, xb = ablation_means(s, cls, a, b)
    diffs = [q - p for p, q in zip(xa, xb)]
    nonzero = [d for d in diffs if abs(d) > 100 * TOL]
    if not nonzero:
        return 0, 0.0, 1.0
    return len(nonzero), mean(diffs), float(wilcoxon(nonzero, zero_method='wilcox', method='exact').pvalue)


ABLATION_TESTS = tuple((cls, other) for cls in ('M05', 'M10') for other in ('alns-fixed', 'alns-plain'))


def ablation_holm(s: Study):
    """P values of the four comparisons with ALNS, adjusted by Holm's step-down method."""
    raw = {key: ablation_test(s, key[0], 'alns', key[1])[2] for key in ABLATION_TESTS}
    adjusted, running = {}, 0.0
    for rank, key in enumerate(sorted(raw, key=raw.get)):
        running = max(running, min(1.0, (len(raw) - rank) * raw[key]))
        adjusted[key] = running
    return adjusted


def table_ablation(s: Study):
    body = []
    holm = ablation_holm(s)
    for cls in ('M05', 'M10'):
        for method in ('alns', 'alns-fixed', 'alns-plain'):
            rs = ablation_runs(s, cls, method)
            if method == 'alns':
                diff, n, p = '--', '--', '--'
            else:
                n, d, _ = ablation_test(s, cls, 'alns', method)
                diff, n, p = fmt(d, 3), fmt(n), fmt(holm[(cls, method)], 3)
            body.append(f"{cls} & {SHORT[method]} & {len(rs)} & {fmt(mean(s.gap(r) for r in rs), 3)} & "
                        f"{fmt(max(s.gap(r) for r in rs), 3)} & {s.hits(rs)} & {diff} & {n} & {p} & "
                        f"{fmt(mean(r['method_seconds'] for r in rs), 1)}\\\\")
    header = [r'& & & \multicolumn{2}{c}{Gap (\%)} & At & \multicolumn{3}{c}{Against ALNS} & Time\\',
              r'\cmidrule(lr){4-5}\cmidrule(lr){7-9}',
              r'Class & Variant & Runs & Mean & Max & reference & Diff. (pp) & $n$ & $P$ & (s)\\']
    return table('llrrrrrrrr', header, body)


def table_scenarios(s: Study):
    body = []
    for scenario in ('base', 'sites', 'deadline10') + FIXED_SCENARIOS:
        for cls in ('M02', 'M05'):
            ex = s.runs(scenario, cls, 'exact')
            if not ex:
                continue
            change = 0.0 if scenario == 'base' else mean(
                100.0 * (r['objective'] - s.optimum('base', cls, r['data_seed'])) / s.optimum('base', cls, r['data_seed'])
                for r in ex)
            body.append(f"{SCENARIO_NAME[scenario]} & {cls} & {fmt(mean(r['served'] for r in ex), 1)} & "
                        f"{fmt(mean(r['urgent_served'] for r in ex), 1)} & "
                        f"{fmt(mean(r['used_vehicles'] for r in ex), 1)} & "
                        f"{fmt(mean(r['served'] / max(1, r['used_vehicles']) for r in ex), 2)} & "
                        f"{fmt(mean(r['objective'] for r in ex), 0)} & {fmt(change, 1)}\\\\")
    header = [r'& & \multicolumn{2}{c}{Served requests} & Tankers & Requests & Optimum & Change\\',
              r'\cmidrule(lr){3-4}',
              r'Scenario & Class & All & Urgent & used & per tanker & (CU) & (\%)\\']
    return table('llrrrrrr', header, body)


def table_penalty(s: Study):
    body = []
    for method in PENALTY_METHODS:
        cells = []
        for scenario in ('penalty0.05', 'penalty0.02'):
            rs = s.runs(scenario, 'M02', method)
            cells += [fmt(mean(r['served'] for r in rs), 1), fmt(mean(r['used_vehicles'] for r in rs), 1),
                      fmt(mean(s.gap(r) for r in rs), 2), str(s.hits(rs))]
        body.append(f'{SHORT[method]} & ' + ' & '.join(cells) + r'\\')
    sub = 'Served & Tankers & Gap (\\%) & Opt.'
    header = [r'& \multicolumn{4}{c}{Penalties $\times0.05$} & \multicolumn{4}{c}{Penalties $\times0.02$}\\',
              r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
              f'Method & {sub} & {sub}\\\\']
    return table('l' + 'r' * 8, header, body)


def macros(s: Study):
    m = {}

    def put(name, value):
        m[name] = value
    exact = [r for r in s.records if r['algorithm'] == 'exact' and r['scenario']['name'] in MAIN_SCENARIOS]
    fixed = [r for r in s.records if r['algorithm'] == 'exact' and r['scenario']['name'] in FIXED_SCENARIOS]
    put('NumExact', fmt(len(exact)))
    every = exact + fixed
    put('NumExactAll', fmt(len(every)))
    put('MaxCplexSecondsAll', fmt(max(r['exact']['solver_seconds'] for r in every), 1))
    if any(r['exact']['nodes'] != 0 for r in every):
        raise SystemExit('An instance needed branching; revise the root-node statements.')
    put('NumExactBase', fmt(sum(1 for r in exact if r['scenario']['name'] == 'base')))
    put('NumExactVariants', fmt(sum(1 for r in exact if r['scenario']['name'] != 'base')))
    put('MfiveColumnsMillions', fmt(mean(r['exact']['columns'] for r in s.runs('base', 'M05', 'exact')) / 1e6, 1))

    def usable(scenario):
        return mean(r['usable_candidates'] / r['requests'] for c in ('M02', 'M05')
                    for r in s.runs(scenario, c, 'priority-greedy'))
    put('UsableMtwoMfive', fmt(round(usable('base'))))
    put('SitesUsableShare', fmt(100.0 * usable('sites') / usable('base'), 0))
    put('NumSeeds', fmt(len(s.seeds)))
    put('NumRecords', fmt(len(s.records)))
    put('MaxCplexSeconds', fmt(max(r['exact']['solver_seconds'] for r in exact), 1))
    put('MaxColumnsMillions', fmt(max(r['exact']['columns'] for r in exact) / 1e6, 1))
    put('LpIntegral', fmt(sum(1 for r in exact if lp_gap(r) <= 1e-4)))
    put('MaxLpGap', fmt(max(lp_gap(r) for r in exact), 2))
    put('RootSolved', fmt(sum(1 for r in exact if r['exact']['nodes'] == 0)))
    put('maxRouteLength', fmt(max(int(k) for r in s.records for k in r['route_lengths'])))
    searches = [r for r in s.records if r['algorithm'] in SEARCHES]
    put('MaxAlnsSeconds', fmt(max(r['method_seconds'] for r in searches), 1))
    if max((r['search'].get('recombination') or {}).get('seconds', 0.0) for r in searches) >= 60.0:
        raise SystemExit('Recombination took 60 s in one run; a set-packing call may have hit its limit.')
    if any(r['search']['stopped_by'] != 'iterations' for r in searches):
        raise SystemExit('A search stopped at the time limit.')

    def alns_on(scenarios):
        rows = [s.get(r['scenario']['name'], r['instance_class'], r['data_seed'], 'alns') for r in exact
                if r['scenario']['name'] in scenarios]
        return [r for r in rows if r is not None]
    base = alns_on(('base',))
    put('AlnsBaseHits', fmt(s.hits(base)))
    put('AlnsBaseRuns', fmt(len(base)))
    put('AlnsBaseMaxGap', fmt(max(s.gap(r) for r in base), 3))
    variants = alns_on(('sites', 'deadline10', 'penalty0.05'))
    put('AlnsVariantHits', fmt(s.hits(variants)))
    put('AlnsVariantRuns', fmt(len(variants)))
    put('AlnsVariantMaxGap', fmt(max(s.gap(r) for r in variants), 2))
    full = alns_on(('base', 'sites', 'deadline10', 'penalty0.05'))
    put('AlnsFullHits', fmt(s.hits(full)))
    put('AlnsFullRuns', fmt(len(full)))
    put('AlnsFullMeanGap', fmt(mean(s.gap(r) for r in full), 3))
    low = s.runs('penalty0.02', 'M02', 'seeding-alns')
    put('WsAlnsLowGap', fmt(mean(s.gap(r) for r in low), 2))
    put('WsAlnsLowMaxGap', fmt(max(s.gap(r) for r in low), 2))
    put('WsAlnsLowHits', fmt(s.hits(low)))
    put('WsAlnsFiveHits', fmt(s.hits(s.runs('penalty0.05', 'M02', 'seeding-alns'))))
    put('AlnsFiveHits', fmt(s.hits(s.runs('penalty0.05', 'M02', 'alns'))))
    put('InsertionLowServed', fmt(max(r['served'] for meth in ('priority-greedy', 'cheapest-insertion',
                                                               'singleton-assignment', 'alns')
                                      for r in s.runs('penalty0.02', 'M02', meth))))
    opt_low = s.runs('penalty0.02', 'M02', 'exact')
    put('PenaltyOptServed', fmt(mean(r['served'] for r in opt_low), 1))
    put('PenaltyOptTankers', fmt(mean(r['used_vehicles'] for r in opt_low), 1))
    put('PenaltyLowValue', fmt(round(0.02 * 50000)))
    put('PenaltyLowUrgentValue', fmt(round(0.02 * 100000)))
    improve = {}
    for cls in LARGE_CLASSES:
        improve[cls] = improvement(s, cls, 'best')
        put(f'AlnsImprove{TAG[cls]}', fmt(improve[cls], 1))
        put(f'AlnsSeconds{TAG[cls]}', fmt(mean(r['method_seconds'] for r in s.runs('base', cls, 'alns')), 0))
        put(f'AlnsServed{TAG[cls]}', fmt(mean(r['served'] for r in s.runs('base', cls, 'alns')), 1))
        put(f'CiServed{TAG[cls]}', fmt(mean(r['served'] for r in s.runs('base', cls, 'cheapest-insertion')), 1))
    put('CiBetterLarge', fmt(sum(1 for cls in LARGE_CLASSES for d in s.seeds
                                 if s.get('base', cls, d, 'cheapest-insertion')['objective']
                                 <= min(s.get('base', cls, d, m)['objective']
                                        for m in ('priority-greedy', 'singleton-assignment')))))
    large = len(LARGE_CLASSES) * len(s.seeds)
    if m['CiBetterLarge'] != fmt(large):
        raise SystemExit('CI is not the best construction rule on every large instance; revise the text.')
    # ALNS serves more requests than every construction rule, instance by instance ('in all ... instances')
    more = sum(1 for cls in LARGE_CLASSES for d in s.seeds
               if s.get('base', cls, d, 'alns')['served']
               > max(s.get('base', cls, d, meth)['served'] for meth in BASE_METHODS[:3]))
    if more != large:
        raise SystemExit('ALNS does not serve more requests than every rule on every large instance; revise the text.')
    put('AlnsServedMoreLarge', fmt(more))
    put('AlnsImproveMin', fmt(min(improve.values()), 1))
    put('AlnsImproveMax', fmt(max(improve.values()), 1))
    # value of multi-request routes: optimum against the best singleton plan
    pairs = [(s.get('base', cls, seed, 'singleton-assignment'), s.get('base', cls, seed, 'exact'))
             for cls in ('M02', 'M05') for seed in s.seeds]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    put('SaServedM', fmt(mean(a['served'] for a, _ in pairs), 1))
    put('OptServedM', fmt(mean(b['served'] for _, b in pairs), 1))
    put('SaGapM', fmt(mean(100.0 * (a['objective'] - b['objective']) / b['objective'] for a, b in pairs), 1))
    put('OptSharedM', fmt(mean(multi_share(b) for _, b in pairs), 1))
    put('OptUsedCount', fmt(sum(b['used_vehicles'] for _, b in pairs)))
    put('OptSharedCount', fmt(sum(v for _, b in pairs for k, v in b['route_lengths'].items() if int(k) >= 2)))
    # scenario effects on the optimum, pooled over the classes M02 and M05
    for scenario, tag in (('sites', 'Sites'), ('deadline10', 'Deadline')):
        before = mean(s.get('base', c, d, 'exact')['served'] for c in ('M02', 'M05') for d in s.seeds)
        after = mean(s.get(scenario, c, d, 'exact')['served'] for c in ('M02', 'M05') for d in s.seeds)
        put(f'{tag}ServedDrop', fmt(pct_drop(before, after), 1))
    before = mean(s.get('base', c, d, 'exact')['urgent_served'] for c in ('M02', 'M05') for d in s.seeds)
    after = mean(s.get('deadline10', c, d, 'exact')['urgent_served'] for c in ('M02', 'M05') for d in s.seeds)
    put('DeadlineUrgentDrop', fmt(pct_drop(before, after), 1))
    fixed_weights = []
    holm = ablation_holm(s)
    for cls, other in ABLATION_TESTS:
        _, diff, _ = ablation_test(s, cls, 'alns', other)
        p = holm[(cls, other)]
        tag = TAG[cls] + ('Fixed' if other == 'alns-fixed' else 'Plain')
        put(f'AblDiff{tag}', fmt(diff, 2))
        put(f'AblP{tag}', fmt(p, 3))
        if other == 'alns-fixed':
            fixed_weights.append((abs(diff), p))
    xa, xb = ablation_means(s, 'M10', 'alns', 'alns-fixed')
    d10 = [q - p for p, q in zip(xa, xb)]
    put('AblFixedUpMten', fmt(sum(1 for x in d10 if x > 100 * TOL)))
    put('AblFixedDownMten', fmt(sum(1 for x in d10 if x < -100 * TOL)))
    put('AblFixedMaxAbsMten', fmt(max(abs(x) for x in d10), 2))

    def cost_mean(cls, seed, method):
        return mean(s.get('base', cls, seed, method, a)['objective'] for a in range(1, 6))

    def optimal_penalty(cls, seed):
        return s.get('base', cls, seed, 'exact')['components']['unserved_penalty']
    # on M05 the full search and the fixed-weight search end at the optimal penalty in every run
    for method in ('alns', 'alns-fixed'):
        if any(r['components']['unserved_penalty'] > optimal_penalty('M05', r['data_seed']) + 1e-6
               for r in ablation_runs(s, 'M05', method)):
            raise SystemExit(f'A run of {method} on M05 misses the optimal penalty; revise the ablation text.')
    put('AblFixedMaxCostMfive', fmt(round(max(abs(cost_mean('M05', d, 'alns-fixed') - cost_mean('M05', d, 'alns'))
                                              for d in s.seeds))))
    excess = [(r['data_seed'], r['components']['unserved_penalty'] - optimal_penalty('M05', r['data_seed']))
              for r in ablation_runs(s, 'M05', 'alns-plain')]
    excess = [(d, x) for d, x in excess if x > 1e-6]
    if len({round(x, 6) for _, x in excess}) != 1:
        raise SystemExit('Runs of ALNS-P on M05 miss the optimal penalty by different amounts; revise the text.')
    put('AblPlainExcessMfive', fmt(round(excess[0][1])))
    put('AblPlainPenaltyRunsMfive', fmt(len(excess)))
    worse = sorted({d for d, _ in excess})
    put('AblPlainPenaltyInstMfive', fmt(len(worse)))
    xa, xb = ablation_means(s, 'M05', 'alns', 'alns-plain')
    dm05 = dict(zip(s.seeds, [q - p for p, q in zip(xa, xb)]))
    put('AblPlainPenaltyShareMfive', fmt(100.0 * sum(dm05[d] for d in worse) / sum(dm05.values()), 1))
    put('AblPlainOtherCostMfive', fmt(round(max(abs(cost_mean('M05', d, 'alns-plain') - cost_mean('M05', d, 'alns'))
                                                for d in s.seeds if d not in worse))))
    # the text gives one adjusted P value for fixed weights in both classes
    if len({fmt(p, 3) for _, p in fixed_weights}) != 1:
        raise SystemExit('The adjusted P values of fixed weights differ between M05 and M10; revise the ablation text.')
    put('AblFixedP', fmt(fixed_weights[0][1], 3))
    rise = [mean(100.0 * (s.get('deadline10', c, d, 'exact')['objective'] - s.optimum('base', c, d))
                 / s.optimum('base', c, d) for d in s.seeds) for c in ('M02', 'M05')]
    put('DeadlineCostRiseMin', fmt(min(rise), 1))
    put('DeadlineCostRiseMtwo', fmt(rise[0], 1))
    put('DeadlineCostRiseMfive', fmt(rise[1], 1))
    # gaps mainly count rejected requests: share of penalties in the optimum
    shares = []
    for scenario in ('base', 'sites', 'deadline10'):
        for cls in EXACT_CLASSES:
            ex = s.runs(scenario, cls, 'exact')
            if ex:
                shares.append(mean(100.0 * r['components']['unserved_penalty'] / r['objective'] for r in ex))
    put('PenaltyShareMin', fmt(min(shares), 0))
    put('PenaltyShareMax', fmt(max(shares), 0))
    for cls in ('M02', 'M05'):
        ex = s.runs('base', cls, 'exact')
        put(f'OneRequestShare{TAG[cls]}', fmt(100.0 * 50000 / mean(r['objective'] for r in ex), 1))
        put(f'OptPerTanker{TAG[cls]}', fmt(mean(r['served'] / max(1, r['used_vehicles']) for r in ex), 1))
    for cls in ('S01', 'S03'):
        put(f'PgMiss{TAG[cls]}', fmt(len(s.seeds) - s.hits(s.runs('base', cls, 'priority-greedy'))))
    ci_pg = [(s.gap(s.get(sc, c, d, 'cheapest-insertion')), s.gap(s.get(sc, c, d, 'priority-greedy')))
             for sc in ('base', 'sites', 'deadline10') for c in ('M02', 'M05') for d in s.seeds]
    put('CiBeatsPg', fmt(sum(1 for a, b in ci_pg if a < b)))
    put('CiPgInstances', fmt(len(ci_pg)))
    m05 = ablation_runs(s, 'M05', 'alns')
    put('AlnsSeedsMfiveHits', fmt(s.hits(m05)))
    put('AlnsSeedsMfiveRuns', fmt(len(m05)))
    put('AlnsSeedsMfiveMaxGap', fmt(max(s.gap(r) for r in m05), 3))
    put('AlnsBaseAllMaxGap', fmt(max(s.gap(r) for cls in EXACT_CLASSES for d in s.seeds for a in range(1, 6)
                                     for r in [s.get('base', cls, d, 'alns', a)] if r is not None), 3))
    put('CandidateSecondsMax', fmt(max(r['candidate_seconds'] for r in s.records if r['algorithm'] != 'exact'), 1))
    put('SaSameTankers', fmt(sum(1 for a, b in pairs if a['used_vehicles'] == b['used_vehicles'])))
    # claims of Section 5.2: SA uses the tankers of the optimum and has the lower operating cost
    if any(a['used_vehicles'] != b['used_vehicles'] for a, b in pairs):
        raise SystemExit('SA and the optimum differ in the number of tankers; revise Section 5.2.')
    if any(a['objective'] - a['components']['unserved_penalty'] >= b['objective'] - b['components']['unserved_penalty']
           for a, b in pairs):
        raise SystemExit('SA does not have the lower operating cost everywhere; revise Section 5.2.')
    put('SaPairs', fmt(len(pairs)))
    put('SaPenaltyShare', fmt(100.0 * sum(a['components']['unserved_penalty'] - b['components']['unserved_penalty']
                                         for a, b in pairs) / sum(a['objective'] - b['objective'] for a, b in pairs), 0))
    for cls in ('M05', 'L06'):
        put(f'UsableTotal{TAG[cls]}', fmt(int(round(mean(r['usable_candidates']
                                                          for r in s.runs('base', cls, 'priority-greedy')), -2))))
    for cls in ('M05', 'M10'):
        xa, xb = ablation_means(s, cls, 'alns', 'alns-plain')
        d = [q - p for p, q in zip(xa, xb)]
        put(f'AblPlainWorse{TAG[cls]}', fmt(sum(1 for x in d if x > 100 * TOL)))
        put(f'AblPlainDiffer{TAG[cls]}', fmt(sum(1 for x in d if abs(x) > 100 * TOL)))
        if cls == 'M10' and sum(1 for x in d if x > 100 * TOL) != len(s.seeds):
            raise SystemExit('Removing recombination does not worsen every M10 instance; revise the text.')
    put('AllRejectedLowGap', fmt(mean(s.gap(r) for r in s.runs('penalty0.02', 'M02', 'priority-greedy')), 2))
    m10 = ablation_runs(s, 'M10', 'alns')
    put('AlnsSeedsMtenGap', fmt(mean(s.gap(r) for r in m10), 2))
    put('AlnsSeedsMtenHits', fmt(s.hits(m10)))
    put('AlnsSeedsMtenRuns', fmt(len(m10)))
    put('OneRequestShareMten', fmt(100.0 * 50000 / mean(s.best_known('base', 'M10', d) for d in s.seeds), 1))
    half = mean(s.gap(r) for r in m10) / (100.0 * 50000 / mean(s.best_known('base', 'M10', d) for d in s.seeds))
    if not 0.5 < half < 0.7:
        raise SystemExit('The M10 five-seed gap is no longer a little over half a request; revise the text.')
    totals = {c: mean(r['usable_candidates'] for r in s.runs('base', c, 'priority-greedy')) for c in ('M05', 'M10')}
    put('UsableTotalMten', fmt(int(round(totals['M10'], -2))))
    put('PairGrowthMfiveMten', fmt(round((totals['M10'] / totals['M05']) ** 2)))
    for c in ('M02', 'M05'):
        ex = s.runs('sites', c, 'exact')
        put(f'SitesPerTanker{TAG[c]}', fmt(mean(r['served'] / max(1, r['used_vehicles']) for r in ex), 1))
    for method, tag in (('priority-greedy', 'Pg'), ('cheapest-insertion', 'Ci')):
        put(f'{tag}SecondsMaxLsix', fmt(math.ceil(max(r['method_seconds'] for r in s.runs('base', 'L06', method)))))
    ratios = [s.runs('base', c, 'priority-greedy')[0]['requests'] / s.runs('base', c, 'priority-greedy')[0]['vehicles']
              for c in CLASSES]
    put('RatioMin', fmt(min(ratios), 1))
    put('RatioMax', fmt(max(ratios), 0))

    def direction(scenario):
        out = [s.get(scenario, c, d, 'exact')['served'] - s.get('base', c, d, 'exact')['served']
               for c in ('M02', 'M05') for d in s.seeds]
        return sum(1 for x in out if x < 0), sum(1 for x in out if x == 0), sum(1 for x in out if x > 0)
    fewer, same, more = direction('sites')
    if fewer != 2 * len(s.seeds):
        raise SystemExit('The local-site rule does not lower the served count everywhere; revise Section 5.6.')
    fewer, same, more = direction('deadline10')
    # 'the number of served urgent requests [falls] in all 20 instances'
    if any(s.get('deadline10', c, d, 'exact')['urgent_served'] >= s.get('base', c, d, 'exact')['urgent_served']
           for c in ('M02', 'M05') for d in s.seeds):
        raise SystemExit('Deadlines do not lower the served urgent requests everywhere; revise the text.')
    put('DeadlineFewer', fmt(fewer))
    put('DeadlineSame', fmt(same))
    put('DeadlineMore', fmt(more))
    def drops(scenario):
        return [pct_drop(s.get('base', c, d, 'exact')['served'], s.get(scenario, c, d, 'exact')['served'])
                for c in ('M02', 'M05') for d in s.seeds]
    put('SitesDropInstMin', fmt(min(drops('sites')), 1))
    put('SitesDropInstMax', fmt(max(drops('sites')), 1))
    if fixed:
        put('DestDropInstMin', fmt(min(drops('destination')), 1))
        put('DestDropInstMax', fmt(max(drops('destination')), 1))
        pairs_sd = [(s.get('sites', c, d, 'exact')['served'], s.get('destination', c, d, 'exact')['served'])
                    for c in ('M02', 'M05') for d in s.seeds]
        put('SitesBeatDest', fmt(sum(1 for a, b in pairs_sd if a > b)))
        if any(a < b for a, b in pairs_sd):
            raise SystemExit('Destination meetings serve more than local sites somewhere; revise the text.')
        put('SitesServedTotal', fmt(sum(a for a, _ in pairs_sd)))
        put('DestServedTotal', fmt(sum(b for _, b in pairs_sd)))
        # the instance of the map figures, M02 with data seed 11223: the text says that both plans use all
        # five tankers and that the instance has 30 requests
        chosen, dest = s.get('base', 'M02', 11223, 'exact'), s.get('destination', 'M02', 11223, 'exact')
        if (chosen['requests'], chosen['used_vehicles'], dest['used_vehicles']) != (30, 5, 5):
            raise SystemExit('The map instance changed; revise the text on its plans.')
        put('MapServedChosen', fmt(chosen['served']))
        put('MapServedDestination', fmt(dest['served']))
        base_served = mean(s.get('base', c, d, 'exact')['served'] for c in ('M02', 'M05') for d in s.seeds)
        for scenario, tag in (('destination', 'Dest'), ('earliest', 'Earliest')):
            rs = [s.get(scenario, c, d, 'exact') for c in ('M02', 'M05') for d in s.seeds]
            put(f'{tag}ServedM', fmt(mean(r['served'] for r in rs), 1))
            put(f'{tag}ServedDrop', fmt(pct_drop(base_served, mean(r['served'] for r in rs)), 1))
            rises = [mean(100.0 * (s.get(scenario, c, d, 'exact')['objective'] - s.optimum('base', c, d))
                          / s.optimum('base', c, d) for d in s.seeds) for c in ('M02', 'M05')]
            put(f'{tag}CostRiseMin', fmt(min(rises), 1))
            put(f'{tag}CostRiseMax', fmt(max(rises), 1))
        per = [mean(r['served'] / max(1, r['used_vehicles']) for r in s.runs('destination', c, 'exact'))
               for c in ('M02', 'M05')]
        put('DestPerTankerMtwo', fmt(per[0], 1))
        put('DestPerTankerMfive', fmt(per[1], 1))
        put('DestPerTankerMin', fmt(min(per), 1))
        put('DestPerTankerMax', fmt(max(per), 1))
        put('NumExactFixed', fmt(len(fixed)))
        if direction('destination')[0] != 2 * len(s.seeds):
            raise SystemExit('Destination meetings do not lower the served count everywhere; revise Section 5.6.')
        if direction('earliest')[0] != 2 * len(s.seeds):
            raise SystemExit('Earliest-point meetings do not lower the served count everywhere; revise the text.')
        if sum(1 for r in fixed if lp_gap(r) <= 1e-4) != len(fixed):
            raise SystemExit('A relaxation with one candidate per request is not tight; check Proposition 5.')
        put('FixedLpTight', fmt(sum(1 for r in fixed if lp_gap(r) <= 1e-4)))
        put('FixedMaxCplexSeconds', fmt(max(r['exact']['solver_seconds'] for r in fixed), 2))
        put('FixedMaxColumns', fmt(max(r['exact']['columns'] for r in fixed)))
    machine = next((r['exact'].get('machine') for r in exact if r['exact'].get('machine')), None) or {}
    put('MacCpu', machine.get('cpu', '--'))
    put('MacCores', machine.get('cores', '--'))
    return m


def write_macros(path: Path, m: dict):
    lines = ['% Generated by analysis/summarize.py from the result records; do not edit.']
    for name, value in sorted(m.items()):
        if not name.isalpha():
            raise SystemExit(f'Macro names must consist of letters: {name}')
        lines.append(f'\\newcommand{{\\{name}}}{{{value}}}')
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[2] / 'results')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'generated')
    args = parser.parse_args()
    study = Study(load(args.results))
    tables = args.out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    for name, build in (('instances', table_instances), ('exact', table_exact), ('quality', table_quality),
                        ('large', table_large), ('ablation', table_ablation), ('scenarios', table_scenarios),
                        ('penalty', table_penalty)):
        (tables / f'{name}.tex').write_text('% Generated by analysis/summarize.py; do not edit.\n' + build(study) + '\n')
    write_macros(args.out / 'macros.tex', macros(study))
    print(f'{len(study.records)} records from {len(study.seeds)} data seeds')


if __name__ == '__main__':
    main()
