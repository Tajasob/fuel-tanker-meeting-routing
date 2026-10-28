#!/usr/bin/env python3
"""Computational study: travel tables, heuristic runs, integer-program export and import.

    python campaign.py table  --seed 11223           node-to-node travel table of a data seed
    python campaign.py run    --seed 11223 [--worker 0 --workers 2]   heuristic runs (resumable)
    python campaign.py export --seed 11223           integer programs of the exact jobs (MPS, bzip2)
    python campaign.py import --seed 11223 --cplex-results DIR        CPLEX solutions to result records
    python campaign.py jobs   --seed 11223           list the jobs of a data seed

Paths default to the layout of the study folder: ../real_city holds the instances and
the road network, ../results the records, ../models the exported programs and
../cplex_results the CPLEX output.

Each result is a JSON record in results/<seed>/<job>.json. A run is skipped when its
record exists and passed the independent validation. Exported models carry a digest
of their coefficients; the import rebuilds every model and refuses a solution whose
model differs from the one that was solved.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from runner import SCENARIOS, load_network, prepare, run_method, save_result, validate  # noqa: E402

BENCHMARK = 'real_city/tehran_controlled_scaling_benchmark'
CLASSES = ('S01', 'S03', 'M02', 'M05', 'M10', 'L01', 'L06')
SEEDS = (11223, 12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123)
BASE_METHODS = ('priority-greedy', 'cheapest-insertion', 'singleton-assignment', 'alns')
EXACT = {'base': ('S01', 'S03', 'M02', 'M05'), 'penalty0.05': ('M02',), 'penalty0.02': ('M02',),
         'sites': ('M02', 'M05'), 'deadline10': ('M02', 'M05'),
         'earliest': ('M02', 'M05'), 'destination': ('M02', 'M05')}
ABLATION_CLASSES = ('M05', 'M10')
ABLATION_SEEDS = (1, 2, 3, 4, 5)
ITERATIONS, TIME_LIMIT = 5000, 1800.0
CPLEX_TIME_LIMIT = 3600.0


def job(scenario, cls, method, algorithm_seed=None):
    key = f'{scenario}__{cls}__{method}' + ('' if algorithm_seed is None else f'__s{algorithm_seed}')
    return {'key': key, 'scenario': scenario, 'class': cls, 'method': method, 'algorithm_seed': algorithm_seed}


def heuristic_jobs():
    out = []
    for cls in CLASSES:
        for method in BASE_METHODS:
            out.append(job('base', cls, method, 1 if method == 'alns' else None))
    for cls in ABLATION_CLASSES:
        for s in ABLATION_SEEDS:
            if s != 1:
                out.append(job('base', cls, 'alns', s))
            out.append(job('base', cls, 'alns-fixed', s))
            out.append(job('base', cls, 'alns-plain', s))
    for scenario in ('penalty0.05', 'penalty0.02'):
        for method in BASE_METHODS + ('seeding', 'seeding-alns'):
            out.append(job(scenario, 'M02', method, 1 if 'alns' in method else None))
    for scenario in ('sites', 'deadline10'):
        for cls in ('M02', 'M05'):
            for method in BASE_METHODS:
                out.append(job(scenario, cls, method, 1 if method == 'alns' else None))
    return out


def exact_jobs():
    return [job(scenario, cls, 'exact') for scenario, classes in EXACT.items() for cls in classes]


def model_name(seed, spec):
    return f"{spec['scenario']}__{spec['class']}__{seed}"


def instance_dir(data: Path, cls: str, seed: int) -> Path:
    found = sorted((data / BENCHMARK / cls).glob(f'*/*_seed{seed}'))
    if len(found) != 1:
        raise FileNotFoundError(f'{cls} seed {seed}: {found}')
    return found[0]


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_digest() -> dict:
    files = sorted(p for p in HERE.rglob('*.py') if '__pycache__' not in p.parts and 'tests' not in p.parts)
    return {str(p.relative_to(HERE)): file_digest(p) for p in files}


def machine() -> dict:
    cpu = platform.processor()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    memory = None
    try:
        memory = int(Path('/proc/meminfo').read_text().split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return {'cpu': cpu, 'cores': os.cpu_count(), 'memory_bytes': memory, 'python': sys.version.split()[0],
            'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'networkx')}}


def model_digest(assembled) -> str:
    objective, matrix, _, upper, _ = assembled
    h = hashlib.sha256()
    for array in (objective, matrix.indptr, matrix.indices, matrix.data, upper):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


class Study:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.data = args.data
        self.table = args.tables / f'seed{self.seed}.npy'
        self.results = args.results / str(self.seed)
        self.network = None
        self.graph_seconds = None

    def network_ready(self):
        if self.network is None:
            self.network, self.graph_seconds = load_network(self.args.graphml)
        return self.network

    def prepare(self, scenario, cls):
        directory = instance_dir(self.data, cls, self.seed)
        instance, space, timing = prepare(directory, self.network_ready(), self.table, SCENARIOS[scenario])
        return directory, instance, space, timing

    # ---------------------------------------------------------------- table
    def build_table(self):
        from model.instance import load_instance
        from model.travel_table import compute_union_table
        network = self.network_ready()
        instances = [load_instance(instance_dir(self.data, cls, self.seed), network) for cls in CLASSES]
        self.table.parent.mkdir(parents=True, exist_ok=True)
        info = compute_union_table(instances, self.table, workers=self.args.workers)
        print(json.dumps({'seed': self.seed, **info}), flush=True)

    # ---------------------------------------------------------------- heuristics
    def finished(self, path: Path) -> bool:
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        return bool(record.get('validation', {}).get('passed'))

    def record(self, spec, directory, scenario, timing, result, instance):
        result.update(job=spec['key'], data_seed=self.seed, instance_class=spec['class'], instance=directory.name,
                      scenario=asdict(SCENARIOS[scenario]), graph_seconds=self.graph_seconds,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                      **timing)
        result['validation'] = validate(self.network, instance, result, SCENARIOS[scenario])
        path = self.results / f"{spec['key']}.json"
        save_result(path, result)
        print(json.dumps({'job': spec['key'], 'objective': round(result['objective'], 4), 'served': result['served'],
                          'seconds': round(result['method_seconds'], 2), 'iterations': result['iterations'],
                          'valid': result['validation']['passed']}), flush=True)
        if not result['validation']['passed']:
            raise AssertionError(f"{spec['key']}: {result['validation']['errors'][:5]}")

    def run(self):
        if not self.table.exists():
            raise FileNotFoundError(f'Build the travel table first: {self.table}')
        self.results.mkdir(parents=True, exist_ok=True)
        info = self.results / 'machine.json'
        if not info.exists():
            info.write_text(json.dumps({**machine(), 'code_sha256': code_digest()}, indent=1) + '\n')
        for (scenario, cls), specs in assign(self.args.workers)[self.args.worker]:
            pending = [s for s in specs if not self.finished(self.results / f"{s['key']}.json")]
            if not pending:
                continue
            directory, instance, space, timing = self.prepare(scenario, cls)
            for spec in pending:
                result = run_method(instance, space, spec['method'], seed=spec['algorithm_seed'] or 1,
                                    iterations=ITERATIONS, time_limit=TIME_LIMIT)
                self.record(spec, directory, scenario, timing, result, instance)
            del instance, space

    # ---------------------------------------------------------------- exact
    def export(self):
        from model.exact_ip import assemble, build_model
        from model.cplex_io import write_mps
        out = self.args.models
        out.mkdir(parents=True, exist_ok=True)
        for spec in exact_jobs():
            name = model_name(self.seed, spec)
            target = out / f'{name}.mps.bz2'
            if target.exists() and target.with_name(f'{name}.json').exists():
                continue
            directory, instance, space, timing = self.prepare(spec['scenario'], spec['class'])
            from model.evaluator import RouteEvaluator
            model = build_model(instance, space.samples, evaluator=RouteEvaluator(instance, space.samples, space=space))
            assembled = assemble(instance, model)
            begin = time.monotonic()
            info = write_mps(target.with_name(target.name + '.writing'), model, assembled, compress=True)
            target.with_name(target.name + '.writing').replace(target)
            sidecar = {'name': name, 'job': spec['key'], 'model_sha256': model_digest(assembled),
                       'file_md5': hashlib.md5(target.read_bytes()).hexdigest(), **info,
                       'build_seconds': model['build_seconds'], 'write_seconds': time.monotonic() - begin,
                       'usable_candidates': len(model['U']), 'candidates_total': model['candidates_total']}
            target.with_name(f'{name}.json').write_text(json.dumps(sidecar, indent=1) + '\n')
            print(json.dumps({k: sidecar[k] for k in ('name', 'columns', 'rows', 'bytes')}), flush=True)
            del instance, space, model, assembled

    def import_results(self):
        from model.cplex_io import collect
        from model.exact_ip import assemble, build_model
        from model.evaluator import RouteEvaluator
        source = self.args.cplex_results
        machine_file = source / 'machine.txt'
        mac = dict(line.split(': ', 1) for line in machine_file.read_text().splitlines() if ': ' in line) \
            if machine_file.exists() else None
        for spec in exact_jobs():
            name = model_name(self.seed, spec)
            path = self.results / f"{spec['key']}.json"
            solution = source / f'{name}.sol'
            if self.finished(path) or not solution.exists():
                continue
            sidecar = json.loads((self.args.models / f'{name}.json').read_text())
            directory, instance, space, timing = self.prepare(spec['scenario'], spec['class'])

            def exact(instance, evaluator):
                model = build_model(instance, space.samples, evaluator=evaluator)
                assembled = assemble(instance, model)
                if model_digest(assembled) != sidecar['model_sha256']:
                    raise ValueError(f'{name}: the rebuilt model differs from the exported model.')
                return collect(model, assembled, solution, source / f'{name}.relax.sol', source / f'{name}.out',
                               time_limit=CPLEX_TIME_LIMIT, machine=mac)
            result = run_method(instance, space, 'exact', exact=exact)
            result['exact']['model_file_md5'] = sidecar['file_md5']
            self.record(spec, directory, spec['scenario'], timing, result, instance)
            del instance, space


WEIGHT = {'S01': 1, 'S03': 1, 'M02': 3, 'M05': 8, 'M10': 30, 'L01': 40, 'L06': 300}


def assign(workers):
    """Groups of jobs sharing one prepared instance, spread over the workers by longest
    estimated work first (ALNS runs dominate the work)."""
    groups = {}
    for spec in heuristic_jobs():
        groups.setdefault((spec['scenario'], spec['class']), []).append(spec)

    def work(item):
        (_, cls), specs = item
        return WEIGHT[cls] * (1 + sum('alns' in s['method'] for s in specs))
    load = [0.0] * workers
    out = [[] for _ in range(workers)]
    for item in sorted(groups.items(), key=lambda kv: (-work(kv), kv[0])):
        w = load.index(min(load))
        out[w].append(item)
        load[w] += work(item)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('table', 'run', 'export', 'import', 'jobs'))
    parser.add_argument('--seed', type=int, required=True, choices=SEEDS)
    parser.add_argument('--data', type=Path, default=ROOT, help='folder that holds real_city/')
    parser.add_argument('--graphml', type=Path, default=ROOT / 'real_city' / 'data' / 'tehran_drive.graphml')
    parser.add_argument('--tables', type=Path, default=ROOT / 'tables')
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    parser.add_argument('--cplex-results', type=Path, default=ROOT / 'cplex_results')
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(variable) != '1':
            raise SystemExit(f'Set {variable}=1 so that every run uses one core.')
    study = Study(args)
    if args.mode == 'jobs':
        for spec in heuristic_jobs() + exact_jobs():
            print(spec['key'])
    elif args.mode == 'table':
        study.build_table()
    elif args.mode == 'run':
        study.run()
    elif args.mode == 'export':
        study.export()
    else:
        study.import_results()


if __name__ == '__main__':
    main()
