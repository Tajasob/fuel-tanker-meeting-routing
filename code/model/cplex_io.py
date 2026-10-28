"""Exchange of the aggregated integer program with IBM ILOG CPLEX.

write_mps writes the model of exact_ip.assemble in free MPS format. Column j is named
x<j>, the rows are f<y> (flow at candidate y), r<i> (request i) and v<m> (tanker m),
and every column is binary. The objective omits the constant sum_i p_i, which is
added back when the solution is read. The CPLEX Interactive Optimizer solves the
integer program and then its linear relaxation (commands); collect reads the two
solution files and the log, checks the solution against the model, rebuilds the
routes and returns the result record.
"""
from __future__ import annotations

import math
import re
import subprocess
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np

OPTIMAL_STATUS = {101, 102}  # integer optimal, and integer optimal within the relative gap tolerance
RELATIVE_GAP = 1e-9


def row_names(model: dict) -> list[str]:
    return ([f'f{i}' for i in range(len(model['U']))] + [f'r{q}' for q in range(len(model['requests']))]
            + [f'v{m}' for m in range(len(model['vehicle_ids']))])


def write_mps(path: Path, model: dict, assembled, *, chunk: int = 200000, compress=None) -> dict:
    """Write the model to path (compressed with bzip2 when compress is set or the name ends in .bz2)."""
    objective, matrix, _, upper, constant = assembled
    names = row_names(model)
    csc = matrix.tocsc()
    csc.sort_indices()
    counts = np.diff(csc.indptr)
    if len(counts) and (counts.min() != 3 or counts.max() != 3):
        raise ValueError('Every column of the aggregated model has three nonzeros.')
    rows = csc.indices.reshape(-1, 3)
    values = csc.data.reshape(-1, 3)
    path = Path(path)
    compress = path.suffix == '.bz2' if compress is None else compress
    target = open(path, 'wb')
    sink = subprocess.Popen(['bzip2', '-9', '-c'], stdin=subprocess.PIPE, stdout=target) if compress else None
    handle = sink.stdin if compress else target

    def emit(text: str):
        handle.write(text.encode('ascii'))

    emit('NAME P1\nROWS\n N obj\n')
    emit(''.join(f' L {name}\n' for name in names))
    emit('COLUMNS\n')
    for begin in range(0, len(objective), chunk):
        end = min(begin + chunk, len(objective))
        costs = objective[begin:end].tolist()  # Python floats: repr gives the shortest exact decimal
        block = []
        for j, (a, b, c), (va, vb, vc), cost in zip(range(begin, end), rows[begin:end].tolist(),
                                                    values[begin:end].tolist(), costs):
            block.append(f' x{j} obj {cost!r} {names[a]} {va:g}\n x{j} {names[b]} {vb:g} {names[c]} {vc:g}\n')
        emit(''.join(block))
    emit('RHS\n')
    emit(''.join(f' rhs {names[q]} {float(upper[q])!r}\n' for q in np.flatnonzero(upper != 0).tolist()))
    emit('BOUNDS\n')
    for begin in range(0, len(objective), chunk):
        emit(''.join(f' BV bnd x{j}\n' for j in range(begin, min(begin + chunk, len(objective)))))
    emit('ENDATA\n')
    handle.close()
    if compress:
        if sink.wait() != 0:
            raise RuntimeError('bzip2 failed')
        target.close()
    return {'columns': int(len(objective)), 'rows': int(matrix.shape[0]), 'nonzeros': int(matrix.nnz),
            'constant': float(constant), 'bytes': path.stat().st_size}


def commands(mps, solution, relaxation, workdir, time_limit: float) -> list[str]:
    """Interactive Optimizer commands: the integer program, then its linear relaxation."""
    return ['set parallel 1', f'set timelimit {time_limit:g}', f'set mip tolerances mipgap {RELATIVE_GAP:g}',
            'set output writelevel 4', f'set workdir {workdir}', f'read {mps}', 'optimize', f'write {solution}',
            'change problem lp', 'optimize', f'write {relaxation}', 'quit']


def run_cplex(binary: str, mps: Path, workdir: Path, time_limit: float) -> dict:
    """Solve a model file with the Interactive Optimizer; returns the paths of its outputs."""
    workdir = Path(workdir)
    stem = Path(mps).name.split('.')[0]
    paths = {'solution': workdir / f'{stem}.sol', 'relaxation': workdir / f'{stem}.relax.sol',
             'log': workdir / f'{stem}.out'}
    for path in paths.values():
        path.unlink(missing_ok=True)
    begin = time.monotonic()
    with open(paths['log'], 'w') as log:
        subprocess.run([binary, '-c', *commands(mps, paths['solution'], paths['relaxation'], workdir, time_limit)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    with open(paths['log'], 'a') as log:
        log.write(f'wall_seconds {time.monotonic() - begin:.0f}\n')
    return paths


def read_solution(path: Path) -> dict:
    """Header of a CPLEX solution file and the indices of the columns at one."""
    root = ElementTree.parse(path).getroot()
    header = dict(root.find('header').attrib)
    ones = []
    for variable in root.iter('variable'):
        if float(variable.get('value')) > 0.5:
            name = variable.get('name')
            if not name.startswith('x'):
                raise ValueError(f'Unexpected variable {name}')
            ones.append(int(name[1:]))
    return {'header': header, 'ones': np.asarray(sorted(ones), dtype=np.int64)}


_NUMBER = r'([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)'


def read_log(path: Path) -> dict:
    """Version, threads, solve times, nodes and the final bound reported in a CPLEX log."""
    text = Path(path).read_text(errors='replace')
    version = re.search(r'Interactive Optimizer (\d+(?:\.\d+)+)', text)
    times = [float(x) for x in re.findall(r'Solution time =\s*' + _NUMBER + r' sec', text)]
    ticks = [float(x) for x in re.findall(r'Deterministic time =\s*' + _NUMBER + r' ticks', text)]
    nodes = re.search(r'Solution time =.*?Nodes = (\d+)', text)
    bound = re.search(r'Current MIP best bound =\s*' + _NUMBER, text)
    status = re.findall(r'^((?:MIP|Dual simplex|Primal simplex|Barrier|Sifting|Network) - [^:]*(?::[^:]*)?):\s+Objective',
                        text, re.M)
    threads = re.search(r'using up to (\d+) threads', text)
    wall = re.search(r'wall_seconds (\d+)', text)
    errors = re.findall(r'^CPLEX Error.*$', text, re.M)
    return {'version': version.group(1) if version else None, 'solution_seconds': times,
            'deterministic_ticks': ticks, 'nodes': int(nodes.group(1)) if nodes else None,
            'best_bound': float(bound.group(1)) if bound else None, 'status_lines': status,
            'threads': int(threads.group(1)) if threads else None,
            'wall_seconds': float(wall.group(1)) if wall else None, 'errors': errors}


def collect(model: dict, assembled, solution: Path, relaxation: Path, log: Path, *, time_limit: float,
            machine: dict | None = None) -> dict:
    """Routes, objective and solver record of a CPLEX run on the model."""
    from .exact_ip import routes_from_columns
    objective, matrix, _, upper, constant = assembled
    sol = read_solution(solution)
    ones = sol['ones']
    x = np.zeros(len(objective))
    x[ones] = 1.0
    if np.any(matrix.dot(x) > upper + 1e-9):
        raise ValueError('The CPLEX solution violates a constraint of the model.')
    value = math.fsum(objective[ones].tolist()) + constant
    reported = float(sol['header']['objectiveValue']) + constant
    if abs(value - reported) > 1e-6 * max(1.0, abs(value)):
        raise ValueError(f'CPLEX objective {reported} differs from the recomputed value {value}.')
    status = int(sol['header']['solutionStatusValue'])
    info = read_log(log)
    relax = read_solution(relaxation) if Path(relaxation).exists() else None
    lp_value = None if relax is None else float(relax['header']['objectiveValue']) + constant
    best_bound = None if info['best_bound'] is None else info['best_bound'] + constant
    times = info['solution_seconds']
    record = dict(solver='CPLEX', version=info['version'], status=status,
                  status_string=sol['header'].get('solutionStatusString'), optimal=status in OPTIMAL_STATUS,
                  objective=value, best_bound=best_bound,
                  mip_gap=None if best_bound is None else (value - best_bound) / max(1e-10, abs(value)),
                  solver_seconds=times[0] if times else None, nodes=info['nodes'],
                  deterministic_ticks=info['deterministic_ticks'][0] if info['deterministic_ticks'] else None,
                  lp_objective=lp_value, lp_seconds=times[1] if len(times) > 1 else None,
                  threads=info['threads'], wall_seconds=info['wall_seconds'], time_limit=time_limit,
                  relative_gap_tolerance=RELATIVE_GAP, cplex_errors=info['errors'],
                  columns=int(len(objective)), rows=int(matrix.shape[0]), nonzeros=int(matrix.nnz),
                  usable_candidates=len(model['U']), candidates_total=model['candidates_total'],
                  source_arcs=int(len(model['source_target'])), candidate_arcs=int(len(model['arc_from'])),
                  build_seconds=model['build_seconds'], machine=machine)
    return {'routes': routes_from_columns(model, ones), 'objective': value, 'exact': record}


def solve(instance, samples: int, *, binary: str, workdir: Path, time_limit: float, evaluator=None) -> dict:
    """Build the model of an instance, solve it with a local CPLEX and collect the result."""
    from .exact_ip import assemble, build_model
    model = build_model(instance, samples, evaluator=evaluator)
    assembled = assemble(instance, model)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mps = workdir / 'model.mps'
    write_mps(mps, model, assembled)
    paths = run_cplex(binary, mps, workdir, time_limit)
    return collect(model, assembled, paths['solution'], paths['relaxation'], paths['log'], time_limit=time_limit)
