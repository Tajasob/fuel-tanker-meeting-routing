#!/usr/bin/env python3
"""Result figures: heuristics against the optimum, the larger classes with the components of the
search, and the value of choosing the meeting point.

    python analysis/charts.py INSTANCE_DIR GRAPHML OUT_DIR [--results ../results]

INSTANCE_DIR and GRAPHML are the M02 instance of data seed 11223 and the road network, as for
figure_map.py. Every plotted value is computed with the functions of summarize.py from the result
records that also give the Supplementary Tables, so a figure and its table cannot disagree. The
figures share the settings of figure_map.py: printed width 16 cm, one type size, no line thinner
than one point and no grey. Series colours come from a palette checked for colour-vision
deficiency, and every series also has its own marker shape. Markers are open, so that markers of
close values stay visible where they overlap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.figure_map import (FONT, INK, KEY, LINE, REJECTED, REJECTED_DOTS, TANKER_COLOURS, WIDTH, ASPECT,  # noqa: E402
                                 Geometry, boxes, frame, map_layer, place_labels, plan, read_paths,
                                 read_tankers, scale_bar)
from analysis.summarize import (EXACT_CLASSES, LARGE_CLASSES, SHORT, Study, ablation_means, load,  # noqa: E402
                                mean)
from model.loader import load_graphml  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
                     'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold'})
SLOTS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')  # categorical slots 1-4, in this order
MARKERS = ('o', 's', '^', 'D')
METHODS = ('alns', 'cheapest-insertion', 'priority-greedy', 'singleton-assignment')  # best to worst mean gap
ABLATION = {'alns-fixed': ('#000000', 'v'), 'alns-plain': ('#6250d6', 'h')}  # no hue of a construction rule
VARIANTS = (('base', 'Chosen meeting points'), ('sites', 'Local-street junctions'),
            ('destination', 'Arrival at the destination'), ('earliest', 'Earliest usable point'))
SCENARIOS = (('base', 'Base'), ('sites', 'Local sites'), ('deadline10', 'Deadlines'))
SIZE = 4.0  # marker size in points of a circle; other shapes are scaled to look as large
SHAPE = {'o': 1.0, 's': 0.9, '^': 1.15, 'D': 0.85, 'v': 1.15, 'h': 1.05}
CM = 1 / 2.54
HEIGHT = {'gaps': 7.0, 'larger': 8.0, 'chart': 5.0}  # figure heights in cm (chart: panel a of the last figure)
GROUP_GAP = 0.6  # extra space between the scenarios of the gap figure, in class widths


def style(ax, ylabel):
    ax.set_ylabel(ylabel, labelpad=3)
    ax.tick_params(length=2.5, width=LINE, pad=2, color=INK)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def panel(ax, letter, title, rise=20, left=-30):
    """Bold lower-case panel letter followed by the panel title, as in figure_schematic.py, placed
    rise points above the axes and left points from its left edge."""
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(left, rise), textcoords='offset points',
                ha='left', va='bottom', fontweight='bold')
    ax.annotate(title, (0, 1), xycoords='axes fraction', xytext=(left + 11, rise), textcoords='offset points',
                ha='left', va='bottom')


def swarm(ax, x0, ys, half_width):
    """Horizontal offsets that keep markers of one column apart: points are placed from the lowest
    value up, each at the nearest free offset, in display points, from the column centre."""
    fig = ax.figure
    box = ax.get_position()
    (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
    w_pts = box.width * fig.get_figwidth() * 72
    h_pts = box.height * fig.get_figheight() * 72
    sx, sy = w_pts / (x_hi - x_lo), h_pts / (y_hi - y_lo)
    limit = half_width * sx
    placed, xs = [], {}
    for i in sorted(range(len(ys)), key=lambda k: (ys[k], k)):
        py = ys[i] * sy
        best = None
        for step in range(0, 40):
            off = (step + 1) // 2 * (0.62 * SIZE) * (1 if step % 2 else -1)
            if abs(off) > limit:
                continue
            if all((off - qx) ** 2 + (py - qy) ** 2 >= (0.9 * SIZE) ** 2 for qx, qy in placed):
                best = off
                break
        if best is None:  # column full: overlap at the centre
            best = 0.0
        placed.append((best, py))
        xs[i] = x0 + best / sx
    return [xs[i] for i in range(len(ys))]


def strip(ax, groups, series, values, colours, markers, width=0.9, mean_line=True, positions=None):
    """One column of instance markers per series within each group, with the mean as a short line;
    the groups are centred at the given positions, by default 0, 1, 2 and so on."""
    step = width / len(series)
    for gi, group in zip(positions or range(len(groups)), groups):
        for si, key in enumerate(series):
            x0 = gi - width / 2 + step * (si + 0.5)
            ys = values[(group, key)]
            xs = swarm(ax, x0, ys, 0.42 * step)
            if mean_line:  # drawn over the markers, so that it stays whole
                m = mean(ys)
                ax.plot([x0 - 0.42 * step, x0 + 0.42 * step], [m, m], color=INK, lw=1.6, solid_capstyle='butt',
                        zorder=4)
            ax.plot(xs, ys, ls='', marker=markers[key], ms=SIZE * SHAPE[markers[key]], mfc='none', mec=colours[key],
                    mew=LINE, zorder=3)


def legend(ax, entries, y=1.0, ncol=None, mean_entry=True, x=0.5):
    handles = [Line2D([], [], ls='', marker=marker, ms=SIZE * SHAPE[marker], mfc='none', mec=colour, mew=LINE,
                      label=label) for label, colour, marker in entries]
    if mean_entry:
        handles.append(Line2D([], [], color=INK, lw=1.6, label='Mean of ten instances'))
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(x, y), ncol=ncol or len(handles), frameon=False,
              handlelength=1.4, handletextpad=0.35, columnspacing=1.1, borderaxespad=0.2)


def heuristic_gaps(s: Study, out: Path):
    rows = [(scenario, cls) for scenario, _ in SCENARIOS for cls in EXACT_CLASSES if s.runs(scenario, cls, 'exact')]
    values = {(row, m): [s.gap(r) for r in s.runs(row[0], row[1], m)] for row in rows for m in METHODS}
    fig = plt.figure(figsize=(WIDTH, HEIGHT['gaps'] * CM))
    bottom, top_room = 1.4 / HEIGHT['gaps'], 1.25 / HEIGHT['gaps']
    ax = fig.add_axes([0.075, bottom, 0.915, 1 - bottom - top_room])
    # one class width per class and GROUP_GAP more between scenarios, which are named below the classes
    positions = [k + GROUP_GAP * [sc for sc, _ in SCENARIOS].index(row[0]) for k, row in enumerate(rows)]
    ax.set_xlim(positions[0] - 0.55, positions[-1] + 0.55)
    top = max(v for vs in values.values() for v in vs)
    ax.set_ylim(-3, 10 * (int(top // 10) + 1))
    style(ax, 'Gap to the optimum (%)')
    colours = dict(zip(METHODS, SLOTS))
    markers = dict(zip(METHODS, MARKERS))
    strip(ax, rows, METHODS, values, colours, markers, positions=positions)
    ax.set_xticks(positions, [cls for _, cls in rows])
    ax.tick_params(axis='x', length=0)
    below = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for scenario, name in SCENARIOS:
        xs = [x for x, row in zip(positions, rows) if row[0] == scenario]
        ax.plot([xs[0] - 0.45, xs[-1] + 0.45], [0, 0], color=INK, lw=LINE, solid_capstyle='butt', clip_on=False,
                transform=transforms.offset_copy(below, fig=fig, y=-14, units='points'))
        ax.annotate(name, ((xs[0] + xs[-1]) / 2, 0), xycoords=('data', 'axes fraction'), xytext=(0, -17),
                    textcoords='offset points', ha='center', va='top')
    legend(ax, [(SHORT[m], colours[m], markers[m]) for m in METHODS], y=1.03)
    fig.savefig(out, metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
    plt.close(fig)
    return {f'{row[0]} {row[1]} {SHORT[m]}': round(mean(values[(row, m)]), 2) for row in rows for m in METHODS}


def larger_and_ablation(s: Study, out: Path):
    fig = plt.figure(figsize=(WIDTH, HEIGHT['larger'] * CM))
    bottom, top_room = 0.95 / HEIGHT['larger'], 1.65 / HEIGHT['larger']
    ax_a = fig.add_axes([0.075, bottom, 0.5, 1 - bottom - top_room])
    ax_b = fig.add_axes([0.70, bottom, 0.29, 1 - bottom - top_room])
    # a: share of requests served on the larger classes
    values = {(cls, m): [100.0 * r['served'] / r['requests'] for r in s.runs('base', cls, m)]
              for cls in LARGE_CLASSES for m in METHODS}
    ax_a.set_xlim(-0.55, len(LARGE_CLASSES) - 0.45)
    ax_a.set_ylim(0, 10 * (int(max(v for vs in values.values() for v in vs) // 10) + 1))
    style(ax_a, 'Requests served (%)')
    colours = dict(zip(METHODS, SLOTS))
    markers = dict(zip(METHODS, MARKERS))
    strip(ax_a, LARGE_CLASSES, METHODS, values, colours, markers)
    requests = {cls: s.runs('base', cls, 'alns')[0]['requests'] for cls in LARGE_CLASSES}
    ax_a.set_xticks(range(len(LARGE_CLASSES)), [f'{cls}\n{requests[cls]} requests' for cls in LARGE_CLASSES])
    ax_a.tick_params(axis='x', length=0)
    legend(ax_a, [(SHORT[m], colours[m], markers[m]) for m in METHODS], y=1.0)
    panel(ax_a, 'a', 'Larger classes')
    # b: mean gap of a variant minus that of ALNS, per instance, over five algorithm seeds
    diffs = {}
    for cls in ('M05', 'M10'):
        for variant in ABLATION:
            xa, xb = ablation_means(s, cls, 'alns', variant)
            diffs[(cls, variant)] = [q - p for p, q in zip(xa, xb)]
    low = min(v for vs in diffs.values() for v in vs)
    high = max(v for vs in diffs.values() for v in vs)
    ax_b.set_xlim(-0.6, 1.6)
    ax_b.set_ylim(min(-1.0, low - 0.4), high + 0.5)
    style(ax_b, 'Mean gap minus that of ALNS (pp)')
    ax_b.axhline(0, color=INK, lw=LINE, ls=(0, (4, 3)), zorder=1)  # dashed, so that means on it stay visible
    strip(ax_b, ('M05', 'M10'), tuple(ABLATION), diffs, {k: v[0] for k, v in ABLATION.items()},
          {k: v[1] for k, v in ABLATION.items()})
    ax_b.set_xticks([0, 1], ['M05', 'M10'])
    ax_b.tick_params(axis='x', length=0)
    legend(ax_b, [(SHORT[v], ABLATION[v][0], ABLATION[v][1]) for v in ABLATION], y=1.0, mean_entry=False)
    panel(ax_b, 'b', 'Components of the search', left=-34)
    fig.savefig(out, metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
    plt.close(fig)
    return ({f'{cls} {SHORT[m]}': round(mean(values[(cls, m)]), 1) for cls in LARGE_CLASSES for m in METHODS},
            {f'{cls} {SHORT[v]}': [round(x, 3) for x in diffs[(cls, v)]] for cls in ('M05', 'M10') for v in ABLATION})


def meeting_point_value(s: Study, instance_dir: Path, graphml: Path, out: Path, network=None):
    share = {(cls, sc): {r['data_seed']: 100.0 * r['served'] / r['requests'] for r in s.runs(sc, cls, 'exact')}
             for cls in ('M02', 'M05') for sc, _ in VARIANTS}
    map_gap = 0.16  # inches between the two maps
    map_w = (WIDTH - 0.04 - map_gap) / 2
    map_h = 0.0  # set from the map extent below
    network = network or load_graphml(graphml)
    geometry = Geometry(network)
    paths, tankers = read_paths(instance_dir), read_tankers(instance_dir)
    paths_xy = {cid: geometry.nodes(seq) for cid, seq in paths.items()}
    releases = {vid: network.node_coordinates[node] for vid, (node, _) in tankers.items()}
    box, segments = frame(network, paths_xy, releases)
    map_h = map_w * (box[3] - box[2]) * ASPECT / (box[1] - box[0])
    chart_h, top_h, mid_h, legend_h, bottom_h = HEIGHT['chart'] * CM, 1.3 * CM, 1.45 * CM, 0.75 * CM, 0.1 * CM
    height = top_h + chart_h + mid_h + map_h + legend_h + bottom_h
    fig = plt.figure(figsize=(WIDTH, height))
    y_chart = (height - top_h - chart_h) / height
    left = 0.075
    gap = 0.035
    facet_w = (0.99 - left - gap) / 2
    axes = [fig.add_axes([left, y_chart, facet_w, chart_h / height]),
            fig.add_axes([left + facet_w + gap, y_chart, facet_w, chart_h / height])]
    top = max(v for d in share.values() for v in d.values())
    offsets = (-0.375, -0.125, 0.125, 0.375)  # markers of the four rules side by side
    for ax, cls in zip(axes, ('M02', 'M05')):
        order = sorted(share[(cls, 'base')], key=lambda seed: (-share[(cls, 'base')][seed], seed))
        if cls == 'M02' and order.index(11223) != 3:
            raise SystemExit('The map instance is no longer instance 4 of M02 in panel a; revise the legend.')
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(0, 10 * (int(top // 10) + 1))
        style(ax, 'Requests served (%)' if cls == 'M02' else '')
        if cls != 'M02':
            ax.set_yticklabels([])
        for (sc, _), colour, marker, off in zip(VARIANTS, SLOTS, MARKERS, offsets):
            ax.plot([i + off for i in range(len(order))], [share[(cls, sc)][seed] for seed in order], ls='',
                    marker=marker, ms=SIZE * SHAPE[marker], mfc='none', mec=colour, mew=LINE, zorder=3)
        requests = s.runs('base', cls, 'exact')[0]['requests']
        ax.set_xticks(range(len(order)), [str(i + 1) for i in range(len(order))])
        ax.tick_params(axis='x', length=0)
        ax.set_xlabel(f'Instances of {cls} ({requests} requests)', labelpad=3)
    legend(axes[0], [(label, colour, marker) for (_, label), colour, marker in zip(VARIANTS, SLOTS, MARKERS)],
           y=1.0, mean_entry=False, x=1 + gap / facet_w / 2)
    panel(axes[0], 'a', 'All instances of M02 and M05')
    # b, c: the M02 instance of data seed 11223 with chosen meeting points and with meetings at the destinations
    y_map = (legend_h + bottom_h) / height
    records = [s.get('base', 'M02', 11223, 'exact'), s.get('destination', 'M02', 11223, 'exact')]
    if any(r['instance'] != instance_dir.name for r in records):
        raise SystemExit(f'{instance_dir} is not the instance of the plans ({records[0]["instance"]}).')
    colour = {vid: TANKER_COLOURS[q % len(TANKER_COLOURS)] for q, vid in enumerate(sorted(tankers))}
    titles = ('Chosen meeting points', 'Arrival at the destination')  # as in the legend of panel a
    summary = []
    for k, (record, title) in enumerate(zip(records, titles)):
        ax = fig.add_axes([(0.02 + k * (map_w + map_gap)) / WIDTH, y_map, map_w / WIDTH, map_h / height])
        rows, served, rejected, used, tanker_of, legs, meetings, _ = plan(record, network, geometry, paths, tankers)
        _, lines, spots = map_layer(ax, box, segments, paths_xy, served, rejected, legs, releases, meetings, colour,
                                    tanker_of, label_meetings=lambda vid: False, arrow_size=5.5)
        bar = scale_bar(ax, 5, box, '5 km')
        place_labels(ax, [(x, y, f'T{vid}', colour[vid], FONT, spots.get(vid, [])) for vid, (x, y) in releases.items()],
                     [p for v in spots.values() for p in v] + list(releases.values()), distance=6.0, lines=lines,
                     placed=boxes(ax, bar))
        letter = 'bc'[k]
        ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(0, 4), textcoords='offset points',
                    ha='left', va='bottom', fontweight='bold')
        ax.annotate(f'{title}, {len(served)} of {len(paths)} requests served',
                    (0, 1), xycoords='axes fraction', xytext=(11, 4), textcoords='offset points', ha='left',
                    va='bottom')
        summary.append({'served': len(served), 'tankers': len(used),
                        'meetings_per_tanker': {f'T{v}': len(meetings[v]) for v in used}})
    handles = [Line2D([], [], color=KEY, lw=1.8, label='Tanker route'),
               Line2D([], [], color=KEY, lw=LINE, label='Path of a served driver'),
               Line2D([], [], color=REJECTED, lw=LINE, ls=REJECTED_DOTS, label='Path of a rejected driver'),
               Line2D([], [], marker='s', ls='', ms=5.5, mfc='white', mec=KEY, mew=1.4, label='Tanker release'),
               Line2D([], [], marker='o', ls='', ms=4.6, mfc=KEY, mec='white', mew=LINE, label='Meeting')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, bottom_h / height), ncol=5, frameon=False,
               handlelength=2.0, columnspacing=1.3, handletextpad=0.5)
    fig.savefig(out, dpi=400, metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
    plt.close(fig)
    return {f'{cls} {sc}': round(mean(share[(cls, sc)].values()), 1) for cls in ('M02', 'M05') for sc, _ in VARIANTS}, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instance_dir', type=Path)
    parser.add_argument('graphml', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[2] / 'results')
    args = parser.parse_args()
    records = load(args.results)
    s = Study(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {'heuristic_gaps': heuristic_gaps(s, args.out_dir / 'heuristic-gaps.pdf')}
    report['larger'], report['ablation'] = larger_and_ablation(s, args.out_dir / 'larger-instances.pdf')
    report['meeting_value'], report['maps'] = meeting_point_value(s, args.instance_dir, args.graphml,
                                                                  args.out_dir / 'meeting-point-value.pdf')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
