# Routing fuel tankers to meet drivers along their routes on a city road network

Code and data of the computational study in

> Tajasob, P. & Mirzapour Al-e-Hashem, S. M. J. Routing fuel tankers to meet drivers along their routes on a city road network. Manuscript under review (2026).

A platform receives refuelling requests from drivers who keep following their road paths, and decides which requests to serve, by which tanker, in which order and where each tanker meets its driver. The code reads the instances, builds their finite sets of meeting candidates on the OpenStreetMap road network of Tehran, runs the construction rules and the adaptive large neighbourhood search, exports the integer programs solved with CPLEX, replays every plan on a separately built graph, and produces the tables, numbers and figures of the paper.

## Contents

| Path | Content |
|---|---|
| `code/` | Python code of the study (MIT License) |
| `data/road-network.zip` | `real_city/data/tehran_drive.graphml`, the drivable road network of Tehran, extracted from OpenStreetMap on 12 June 2026 with OSMnx 2.0.1 (113,870 nodes and 227,731 directed edges) |
| `data/instances.zip` | `real_city/tehran_controlled_scaling_benchmark/`, the 70 instances of classes S01, S03, M02, M05, M10, L01 and L06, each for the ten data seeds |
| `data/results.zip` | `results/<seed>/`, the 980 result records, one JSON file per run or exact solution, and `machine.json` with the computer and the SHA-256 of the code files at the first run of the data seed |
| `data/cplex-results.zip` | `cplex_results/`, the CPLEX logs (`.out`), integer solutions (`.sol`), linear relaxations (`.relax.sol`) and finishing times (`.done`) of the 140 integer programs, the run log `status.txt`, and `machine.txt`, a description of the computer that solved them |

Each instance folder holds `requests.csv`, `vehicles.csv` with the tanker rates, `trajectory_nodes.csv` and `metadata.json`, the generator's record of its settings. The data seeds are 11223, 12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012 and 90123. The instances were generated as described in the Methods of the article, which also give the zones and the path filter. The `files` entry of `metadata.json` also names `breakpoints.csv`, `segments.csv` and `trajectories.geojson`, generator outputs that the study does not use and that are not included; `matrix_mode` and `cplex_enabled` are generator options that the study does not use.

## Set-up

Unzip the four archives in the top folder of the repository, so that `real_city/`, `results/` and `cplex_results/` lie next to `code/`, which is where the scripts look for them:

```bash
unzip data/road-network.zip
unzip data/instances.zip
unzip data/results.zip
unzip data/cplex-results.zip
python -m pip install -r code/requirements.txt matplotlib
```

Python 3.11 or later with numpy, scipy and networkx is needed, and matplotlib for the figures. The heuristic runs used Python 3.13.16, numpy 2.5.3, scipy 1.18.1 and networkx 3.6.1, as each `results/<seed>/machine.json` records; the exact records were imported later, in an environment that is not recorded. The figures of the paper use the Liberation Sans font; without it the scripts fall back to another sans-serif font. `campaign.py` needs Linux or macOS, because it forks its worker processes and compresses the programs with `bzip2`. The exact solutions use the IBM ILOG CPLEX 22.1.1 Interactive Optimizer, which is not needed to rebuild the tables and figures from the records.

## Tests

```bash
cd code
python -m unittest discover -s tests -t .
```

## Tables, numbers and figures from the records

Run from `code/`:

```bash
python analysis/summarize.py        # tables and macros.tex in ../generated
mkdir -p ../figures
python analysis/concept.py ../figures/refuelling-concept.pdf
python analysis/figure_schematic.py ../figures/meeting-service-schematic.pdf
python analysis/charts.py ../real_city/tehran_controlled_scaling_benchmark/M02/medium/tehran_TMixed_U30_n30_m5_seed11223 \
    ../real_city/data/tehran_drive.graphml ../figures
python analysis/figure_map.py ../results/11223/base__M02__exact.json \
    ../real_city/tehran_controlled_scaling_benchmark/M02/medium/tehran_TMixed_U30_n30_m5_seed11223 \
    ../real_city/data/tehran_drive.graphml ../figures/optimal-plan-map.pdf
```

`summarize.py` writes the main and supplementary tables and the numbers quoted in the text. Figures 1 and 2 are drawings, Figures 3, 4 and 6 are computed from the records with the functions of `summarize.py`, and Figure 5 shows the optimal plan of the M02 instance of data seed 11223.

## The whole study

Run from `code/`, with one numerical thread per process. A run skips every job whose record already exists, so with the records of `results.zip` unpacked, send the new records to another folder:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python campaign.py table  --seed 11223 --workers 2                         # travel table (2 GB) in ../tables
python campaign.py run    --seed 11223 --worker 0 --workers 2 --results ../rerun &
python campaign.py run    --seed 11223 --worker 1 --workers 2 --results ../rerun
wait
python campaign.py export --seed 11223                                     # integer programs in ../models
```

`cplex/run.sh` solves every program placed in `cplex/inbox/` as `<name>.mps.bz2.part*` together with `<name>.ready`, whose first line is the md5 of the joined file, for example `mkdir -p cplex/inbox && cp ../models/NAME.mps.bz2 cplex/inbox/NAME.mps.bz2.part0 && md5 -q ../models/NAME.mps.bz2 > cplex/inbox/NAME.ready`. It writes the log and both solutions to `cplex/results/` and a description of the computer to `cplex/machine.txt`; the CPLEX settings are on one line of the script. Copy `cplex/results/*` and `cplex/machine.txt` to `../cplex_results/`, then

```bash
python campaign.py import --seed 11223 --results ../rerun
```

`export` and `import` do not need CPLEX, so the exact records can also be rebuilt from the CPLEX output in `cplex_results/`. `summarize.py` needs the records of all ten data seeds: repeat the steps above for every seed, or copy the stored records of the other seeds first with `cp -Rn ../results/* ../rerun/`, then run

```bash
python analysis/summarize.py --results ../rerun
```

A single job can be re-run with `runner.py`, for example ALNS on the base scenario of M02 for data seed 11223, with the iteration and time limits of the study:

```bash
python runner.py --instance-dir ../real_city/tehran_controlled_scaling_benchmark/M02/medium/tehran_TMixed_U30_n30_m5_seed11223 \
    --graphml ../real_city/data/tehran_drive.graphml --table ../tables/seed11223.npy \
    --method alns --seed 1 --iterations 5000 --time-limit 1800 --output ../rerun/base__M02__alns__s1.json
```

Every result record carries the independent validation of its plan, and the import refuses a solution whose rebuilt program differs from the exported one.

## Code

| Path | Content |
|---|---|
| `model/road.py` | Directed travel between road nodes and points inside links |
| `model/loader.py` | GraphML road network, link speeds and driver paths |
| `model/instance.py` | Requests, tankers and the finite meeting candidates (grid, deadline and catch points) |
| `model/travel_table.py` | Node-to-node travel table of a data seed |
| `model/evaluator.py` | Usable candidates, travel between them, fixed-route dynamic program |
| `model/exact_ip.py` | Aggregated candidate-graph integer program |
| `model/cplex_io.py` | MPS export, CPLEX solution and log import |
| `model/sites.py` | Local-access junctions of the local-site scenario |
| `solver/common.py` | Plan representation and costs shared by all heuristics |
| `solver/constructors.py` | Priority greedy (PG) and cheapest insertion (CI) |
| `solver/baselines.py` | Singleton assignment (SA) |
| `solver/seeding.py` | Whole-route seeding (WS) |
| `solver/alns.py` | Adaptive large neighbourhood search |
| `solver/recombination.py` | Route pool and set-packing recombination (HiGHS through SciPy) |
| `analysis/independent_validation.py` | Schedule replay on a separately built NetworkX graph |
| `analysis/summarize.py` | Tables and numbers of the paper |
| `analysis/check_inline.py` | Check that the LaTeX source of the article carries the current tables and numbers (`check_inline.py MAIN.tex [SUPPLEMENT.tex ...] GENERATED_DIR`) |
| `analysis/concept.py` | Figure 1, refuelling at a station and on the route |
| `analysis/figure_schematic.py` | Figure 2, meeting and service timing |
| `analysis/charts.py` | Figures 3, 4 and 6 |
| `analysis/figure_map.py` | Figure 5, an optimal plan on the road network |
| `runner.py` | Scenarios and one method on one instance |
| `campaign.py` | The whole study: travel tables, runs, model export and import |
| `cplex/run.sh` | CPLEX runner used for the exact solutions (macOS) |
| `tests/` | Unit tests |

Scenarios: `base`; `sites` (meetings at local-access junctions); `deadline10` (urgent requests met within ten minutes); `penalty0.05` and `penalty0.02` (scaled penalties); `destination` and `earliest` (one meeting candidate per request, at the end of the path or at the earliest usable candidate).

The result records were produced with earlier revisions of part of the code. Each `results/<seed>/machine.json` lists the SHA-256 of the code files present when the first run of that data seed started. Against this repository, `model/evaluator.py`, `runner.py`, `campaign.py` and `analysis/summarize.py` differ for every data seed, `model/cplex_io.py` for 11223, 12345 and 23456, `analysis/figure_schematic.py` for 78901, 89012 and 90123, and `analysis/figure_map.py` for 89012 and 90123, whose lists also include a helper for the bibliography of the manuscript that is not part of this repository. The exact records were imported after the heuristic runs, and the code version of that import is not recorded. Eleven recorded jobs of data seed 11223 were re-run with the code in this repository (PG, CI, SA and ALNS on base M02; ALNS, ALNS-F and ALNS-P on base M05; ALNS on local-site M02 and on deadline M05; WS+ALNS on the ×0.02 penalty scenario of M02; ALNS on base M10), and all eleven reproduced the stored objective and routes exactly. Re-running all 84 heuristic jobs of data seed 11223 on another computer (an Apple M5 under macOS) reproduced 79 records exactly. The others are ALNS runs on M10 and L06, which ended within 215 CU of their records (one of them serving one request fewer at almost the same cost), and CI on L06, whose plan differs in the last digit of one coordinate.

## Licences

- Code (`code/`): MIT License, see `LICENSE`.
- Data (`data/`): road data © OpenStreetMap contributors. The road network and everything built on it are available under the Open Data Commons Open Database License (ODbL) 1.0, see `LICENSE-DATA.md`.

## Citation

Please cite the article above and this repository (`CITATION.cff`). Version 1.0.0 is archived in Zenodo, https://doi.org/10.5281/zenodo.23743781.

## Contact

S. Mohammad J. Mirzapour Al-e-Hashem, Mirzapour@aut.ac.ir, and Pouria Tajasob, Tajasob@aut.ac.ir, Department of Industrial Engineering and Management Systems, Amirkabir University of Technology (Tehran Polytechnic).
