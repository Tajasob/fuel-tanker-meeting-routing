#!/usr/bin/env python3
"""Figure: a plan on the Tehran road network (deterministic drawing of a result record).

    python analysis/figure_map.py RESULT.json INSTANCE_DIR GRAPHML OUT.pdf

Panel (a) maps the plan, panel (b) gives the schedule of every tanker over time and panel (c)
enlarges the densest route. Tanker legs follow shortest paths on the road graph split at the
meeting points, as in the independent replay; driver paths are the recorded node sequences.
Both follow the geometry of the road links. The figure is drawn at its printed width, 16 cm, with
one type size for all lettering, no line thinner than one point and no grey, as in
figure_schematic.py.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

WIDTH = 160 / 25.4  # printed width in inches, the text width of the manuscript
FONT = 8  # points, every label of both figures
LINE = 1.0  # points, the thinnest line
BOTTOM = 1.85  # inches, height of panels b and c
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica'],
                     'font.size': FONT, 'axes.linewidth': LINE, 'xtick.major.width': LINE,
                     'ytick.major.width': LINE, 'patch.linewidth': LINE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.prop_cycle': plt.cycler(color=['#000000'])})  # nothing falls back to default colours
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.independent_validation import location_key, point_coordinates, split_graph  # noqa: E402
from model.loader import load_graphml  # noqa: E402

INK = '#000000'
ROAD = '#f2e7cf'  # the road network in light sand
REJECTED = KEY = INK  # paths of rejected drivers, and the symbols of the legends
TANKER_COLOURS = ('#1f5fa8', '#c0392b', '#1e8449', '#b9770e', '#7d3c98', '#117a8b', '#a04000', '#283593',
                  '#b03a6f', '#827717')
REJECTED_DOTS = (0, (1.0, 2.6))  # dotted line of a rejected driver's path, in multiples of its width
ARROW_HEAD = '-|>,head_length=0.75,head_width=0.42'  # arrowheads about three times as wide as a route
TRAVEL_W, WAIT_W, WAIT_DASH = 2.0, 1.0, (0, (2.5, 1.5))  # schedule lines, as in Figure 2
MEETING_MS, MEETING_GAP = 4.6, 6.0  # meeting marker size, and the least distance of two markers, in points
COS = math.cos(math.radians(35.7))
ASPECT = 1 / COS  # equal ground distances on both axes
KM_LON = 1 / (111.32 * COS)  # degrees of longitude per kilometre
HALO = [pe.withStroke(linewidth=2.2, foreground='white')]


def read_paths(instance_dir: Path):
    paths = {}
    with (instance_dir / 'trajectory_nodes.csv').open() as stream:
        for row in csv.DictReader(stream):
            paths.setdefault(int(row['customer_id']), []).append((int(row['seq']), int(row['node'])))
    return {cid: [node for _, node in sorted(rows)] for cid, rows in paths.items()}


def read_tankers(instance_dir: Path):
    with (instance_dir / 'vehicles.csv').open() as stream:
        return {int(r['vehicle_id']): (int(r['node']), float(r['available_time_min'])) for r in csv.DictReader(stream)}


class Geometry:
    """Coordinates of node and interior-point sequences along the link geometry."""

    def __init__(self, network):
        self.network = network
        self.link = {}
        for link in network.links.values():
            coords = list(link.coordinates or (network.node_coordinates[link.tail], network.node_coordinates[link.head]))
            self.link.setdefault((link.tail, link.head), coords)
            self.link.setdefault((link.head, link.tail), coords[::-1])

    def nodes(self, sequence):
        out = [self.network.node_coordinates[sequence[0]]]
        for a, b in zip(sequence, sequence[1:]):
            out.extend(self.link.get((a, b), [self.network.node_coordinates[a], self.network.node_coordinates[b]])[1:])
        return out

    def keys(self, keys):
        out = [point_coordinates(self.network, keys[0])]
        for a, b in zip(keys, keys[1:]):
            if a[0] == 'node' and b[0] == 'node' and (a[1], b[1]) in self.link:
                out.extend(self.link[(a[1], b[1])][1:])
            else:
                out.append(point_coordinates(self.network, b))
        return out


def arrow_on(ax, line, colour, where=0.55, size=7, zorder=5):
    """Arrowhead at a fraction of the length of a polyline, pointing along it."""
    lengths = [math.hypot((x2 - x1) * COS, y2 - y1) for (x1, y1), (x2, y2) in zip(line, line[1:])]
    total = sum(lengths)
    if total <= 0:
        return
    target, run = where * total, 0.0
    for (p, q), length in zip(zip(line, line[1:]), lengths):
        if run + length >= target and length > 0:
            f = (target - run) / length
            tip = (p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1]))
            back = max(0.0, f - 0.02 * total / length)
            tail = (p[0] + back * (q[0] - p[0]), p[1] + back * (q[1] - p[1]))
            if tail == tip:
                tail = p
            ax.annotate('', xy=tip, xytext=tail, zorder=zorder,
                        arrowprops=dict(arrowstyle=ARROW_HEAD, fc=colour, ec='none', lw=0, mutation_scale=size,
                                        shrinkA=0, shrinkB=0))
            return
        run += length


DIRECTIONS = ((1, 0.6), (-1, 0.6), (1, -0.6), (-1, -0.6), (0, 1), (0, -1), (1, 0), (-1, 0))
MARKER_REACH = 3.5  # points from the centre of a meeting or release marker to the outside of its rim


def place_labels(ax, items, markers, distance=7.0, lines=(), placed=None, directions=DIRECTIONS):
    """Annotate each (x, y, text, colour, size[, kin]) in bold beside its point, kin being markers
    that may lie nearer to the label than its point, such as the meetings of the tanker a tanker
    name stands for. Candidate boxes lie in the
    given directions at the given distance, or at 1.5 or 2.1 times it when no nearer box is clear.
    A box is clear when it stays inside the frame of the axes, 2 points from the labels placed
    before it, 1 point from every marker rim and 1 point from the lines, which are sampled every
    point of their length together with the frame, and when every other marker is at least 2 points
    farther from it than its own point, so that it reads as belonging to that point. Labels are
    compared first by their distance to other labels, then by that margin, then by their distance
    to markers, then to lines; the boxes are added to placed, so that later calls avoid them as
    well."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 72.0 / fig.dpi
    frame = [v * scale for v in ax.get_window_extent(renderer).extents]  # x0, y0, x1, y1 in points
    border = [(frame[0], frame[1]), (frame[2], frame[1]), (frame[2], frame[3]), (frame[0], frame[3]), (frame[0], frame[1])]
    marks = np.array([ax.transData.transform(m) * scale for m in markers]).reshape(-1, 2)
    samples = []
    for line in [ax.transData.transform(line) * scale for line in lines] + [np.array(border)]:
        for (x1, y1), (x2, y2) in zip(line, line[1:]):
            steps = max(1, int(math.hypot(x2 - x1, y2 - y1)))
            samples.extend((x1 + (x2 - x1) * k / steps, y1 + (y2 - y1) * k / steps) for k in range(steps + 1))
    samples = np.array(samples).reshape(-1, 2)
    placed = [] if placed is None else placed

    def gap(box, pts):
        if not len(pts):
            return 1e9
        x0, y0, x1, y1 = box
        dx = np.maximum(np.maximum(x0 - pts[:, 0], pts[:, 0] - x1), 0)
        dy = np.maximum(np.maximum(y0 - pts[:, 1], pts[:, 1] - y1), 0)
        return float(np.hypot(dx, dy).min())

    def score(box):
        x0, y0, x1, y1 = box
        if x0 < frame[0] + 1 or y0 < frame[1] + 1 or x1 > frame[2] - 1 or y1 > frame[3] - 1:
            return (-1.0, -1.0, -1.0, -1.0)
        to_labels = min((math.hypot(max(x0 - b[2], 0, b[0] - x1), max(y0 - b[3], 0, b[1] - y1)) for b in placed),
                        default=1e9)
        own = gap(box, np.array([[px, py]]))
        others = marks[np.hypot(marks[:, 0] - px, marks[:, 1] - py) > 0.5] if len(marks) else marks
        for kx, ky in kin:  # markers of the same tanker do not compete with its name
            others = others[np.hypot(others[:, 0] - kx, others[:, 1] - ky) > 0.5]
        # the label must read as belonging to its own point: other markers at least 2 points farther away
        belongs = min(gap(box, others) - own, 2.0) if len(others) else 2.0
        return (min(to_labels, 2.0), belongs, min(gap(box, marks) - MARKER_REACH, 1.0), min(gap(box, samples), 3.0))

    for item in items:
        x, y, text, colour, size = item[:5]
        kin = [tuple(ax.transData.transform(k) * scale) for k in (item[5] if len(item) > 5 else ())]
        font = FontProperties(family=plt.rcParams['font.sans-serif'][0], size=size, weight='bold')
        width, height, _ = (v * scale for v in renderer.get_text_width_height_descent(text, font, ismath=False))
        px, py = ax.transData.transform((x, y)) * scale
        best = None
        for reach in (distance, 1.5 * distance, 2.1 * distance):
            for dx, dy in directions:
                norm = math.hypot(dx, dy)
                ox, oy = reach * dx / norm, reach * dy / norm
                ha = 'left' if ox > 0.5 else 'right' if ox < -0.5 else 'center'
                va = 'bottom' if oy > 0.5 else 'top' if oy < -0.5 else 'center'
                left = px + ox - {'left': 0, 'right': width, 'center': width / 2}[ha]
                bottom = py + oy - {'bottom': 0, 'top': height, 'center': height / 2}[va]
                box = (left, bottom, left + width, bottom + height)
                value = score(box)
                if best is None or value > best[0]:
                    best = (value, (ox, oy), ha, va, box)
            if best[0][0] >= 2.0 and best[0][1] >= 2.0 and best[0][2] >= 1.0 and best[0][3] >= 1.0:
                break  # a clear box at this distance: no need to go farther
        _, offset, ha, va, box = best
        ax.annotate(text, (x, y), xytext=offset, textcoords='offset points', ha=ha, va=va, fontsize=size,
                    fontweight='bold', color=colour, zorder=9, path_effects=HALO)
        placed.append(box)
    return placed


def scale_bar(ax, km, box, label):
    """Scale bar in the lower left corner, with its length written on it; returns its artists."""
    x0 = box[0] + 0.04 * (box[1] - box[0])
    y0 = box[2] + 0.05 * (box[3] - box[2])
    bar, = ax.plot([x0, x0 + km * KM_LON], [y0, y0], color=INK, lw=1.6, solid_capstyle='butt', zorder=8)
    text = ax.text(x0 + km * KM_LON / 2, y0 + 0.015 * (box[3] - box[2]), label, ha='center', va='bottom',
                   fontsize=FONT, zorder=8, path_effects=HALO)
    return [bar, text]


def boxes(ax, artists):
    """Boxes of drawn artists in points, for place_labels to keep labels off them."""
    fig = ax.figure
    fig.canvas.draw()
    scale = 72.0 / fig.dpi
    return [tuple(v * scale for v in a.get_window_extent(fig.canvas.get_renderer()).extents) for a in artists]


def map_layer(ax, box, segments, paths_xy, served, rejected, legs, releases, meetings, colour, tanker_of,
              label_meetings, arrow_size):
    # the road network alone is drawn as a 400 dpi image, which keeps the files within the 3 MB that
    # Scientific Reports allows for a first submission; everything else is vector
    ax.add_collection(LineCollection(segments, colors=ROAD, linewidths=LINE, zorder=1, rasterized=True))
    ax.add_collection(LineCollection([paths_xy[c] for c in rejected], colors=REJECTED, linewidths=LINE,
                                     linestyles=REJECTED_DOTS, zorder=2))
    for cid in served:
        line = paths_xy[cid]
        ax.add_collection(LineCollection([line], colors=colour[tanker_of[cid]], linewidths=LINE, zorder=3))
        arrow_on(ax, line, colour[tanker_of[cid]], where=0.9, size=arrow_size * 0.8, zorder=3)
    for vid, vlegs in legs.items():
        ax.add_collection(LineCollection(vlegs, colors=colour[vid], linewidths=1.8, zorder=4))
        for leg in vlegs:
            arrow_on(ax, leg, colour[vid], where=0.55, size=arrow_size, zorder=5)
    for vid, (x, y) in releases.items():
        ax.plot(x, y, ls='', marker='s', ms=5.5, mfc='white', mec=colour.get(vid, INK), mew=1.4, zorder=6)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect(ASPECT)
    ax.apply_aspect()
    sites = []  # meetings of one tanker less than 50 m apart share a marker and a label, e.g. "1, 2"
    for vid, stops in meetings.items():
        for order, (x, y) in enumerate(stops, start=1):
            for site in sites:
                if site[0] == vid and math.hypot((site[1] - x) * COS, site[2] - y) * 110.57 < 0.05:
                    site[3].append(order)
                    break
            else:
                sites.append((vid, x, y, [order]))
    shown = spread(ax, [(x, y) for _, x, y, _ in sites], MEETING_GAP)
    labels, spots = [], {vid: [] for vid in meetings}  # spots: where each tanker's meeting markers are drawn
    for (vid, _, _, orders), (x, y) in zip(sites, shown):
        ax.plot(x, y, ls='', marker='o', ms=MEETING_MS, mfc=colour[vid], mec='white', mew=LINE, zorder=7)
        spots[vid].append((x, y))
        if label_meetings(vid):
            labels.append((x, y, ', '.join(map(str, orders)), colour[vid], FONT))
    lines = [paths_xy[c] for c in rejected] + [paths_xy[c] for c in served] + [leg for v in legs.values() for leg in v]
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(LINE)
        spine.set_color(INK)
    return labels, lines, spots


def spread(ax, points, gap, rounds=200):
    """Positions at which markers of the given points are drawn: markers closer than gap points are
    pushed apart along the line joining them, or sideways when they coincide, until none overlap.
    Markers that do not overlap stay where they are."""
    scale = 72.0 / ax.figure.dpi
    xy = [list(ax.transData.transform(p) * scale) for p in points]
    for _ in range(rounds):
        moved = False
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                dx, dy = xy[j][0] - xy[i][0], xy[j][1] - xy[i][1]
                d = math.hypot(dx, dy)
                if d < gap - 1e-6:
                    ux, uy = (dx / d, dy / d) if d > 1e-6 else (1.0, 0.0)
                    push = (gap - d) / 2
                    xy[i][0] -= ux * push
                    xy[i][1] -= uy * push
                    xy[j][0] += ux * push
                    xy[j][1] += uy * push
                    moved = True
        if not moved:
            break
    inverse = ax.transData.inverted()
    return [tuple(inverse.transform((x / scale, y / scale))) for x, y in xy]


def plan(record, network, geometry, paths, tankers):
    """Schedule rows, served and rejected requests, tankers used, and the legs and meeting points of
    every tanker route of the plan of a result record."""
    rows = sorted(record['schedule'], key=lambda r: (int(r['vehicle']), int(r['position'])))
    graph = split_graph(network, [r['road_location'] for r in rows])
    served = sorted({int(r['customer']) for r in rows})
    rejected = sorted(c for c in paths if c not in served)
    used = sorted({int(r['vehicle']) for r in rows})
    tanker_of = {int(r['customer']): int(r['vehicle']) for r in rows}
    legs, meetings, first_leg = {vid: [] for vid in used}, {vid: [] for vid in used}, {}
    for vid in used:
        source = ('node', tankers[vid][0])
        for r in (r for r in rows if int(r['vehicle']) == vid):
            target = location_key(network, r['road_location'])
            legs[vid].append(geometry.keys(nx.shortest_path(graph, source, target, weight='minutes')))
            meetings[vid].append(point_coordinates(network, target))
            source = target
        first_leg[vid] = next(float(r['tau']) for r in rows if int(r['vehicle']) == vid)
    return rows, served, rejected, used, tanker_of, legs, meetings, first_leg


def frame(network, paths_xy, releases):
    """Map extent around all driver paths and tanker releases, and the road links inside it."""
    xs = [x for p in paths_xy.values() for x, _ in p] + [x for x, _ in releases.values()]
    ys = [y for p in paths_xy.values() for _, y in p] + [y for _, y in releases.values()]
    pad_x, pad_y = 0.03 * (max(xs) - min(xs)), 0.03 * (max(ys) - min(ys))
    box = (min(xs) - pad_x, max(xs) + pad_x, min(ys) - pad_y, max(ys) + pad_y)
    segments = []
    for link in network.links.values():
        coords = link.coordinates or (network.node_coordinates[link.tail], network.node_coordinates[link.head])
        if any(box[0] <= x <= box[1] and box[2] <= y <= box[3] for x, y in coords):
            segments.append(coords)
    return box, segments


def main(result: Path, instance_dir: Path, graphml: Path, out: Path):
    record = json.loads(result.read_text())
    network = load_graphml(graphml)
    geometry = Geometry(network)
    paths, tankers = read_paths(instance_dir), read_tankers(instance_dir)
    rows, served, rejected, used, tanker_of, legs, meetings, first_leg = plan(record, network, geometry, paths, tankers)
    colour = {vid: TANKER_COLOURS[q % len(TANKER_COLOURS)] for q, vid in enumerate(used)}
    paths_xy = {cid: geometry.nodes(seq) for cid, seq in paths.items()}
    releases = {vid: network.node_coordinates[node] for vid, (node, _) in tankers.items()}
    box, segments = frame(network, paths_xy, releases)

    # The densest route, enlarged in panel (c): the tanker whose release and meetings span the smallest area
    def span(vid):
        pts = meetings[vid] + [releases[vid]]
        return max((max(p[0] for p in pts) - min(p[0] for p in pts)) * COS, max(p[1] for p in pts) - min(p[1] for p in pts))
    dense = min((v for v in used if len(meetings[v]) >= 3), key=span)
    pts = meetings[dense] + [releases[dense]]
    cx = (max(p[0] for p in pts) + min(p[0] for p in pts)) / 2
    cy = (max(p[1] for p in pts) + min(p[1] for p in pts)) / 2
    half_y = 0.75 * max(span(dense), 0.006)
    half_x = half_y / COS * 1.25
    zoom = (cx - half_x, cx + half_x, cy - half_y, cy + half_y)

    width, legend_h, gap, bottom_h, axis_h = WIDTH, 0.26, 0.3, BOTTOM, 0.36
    map_w = width - 0.04
    map_h = map_w * (box[3] - box[2]) * ASPECT / (box[1] - box[0])
    height = legend_h + map_h + gap + bottom_h + axis_h
    fig = plt.figure(figsize=(width, height))
    ax_map = fig.add_axes([0.02 / width, (axis_h + bottom_h + gap) / height, map_w / width, map_h / height])
    zoom_w = bottom_h * (zoom[1] - zoom[0]) / (ASPECT * (zoom[3] - zoom[2]))
    ax_zoom = fig.add_axes([(width - 0.02 - zoom_w) / width, axis_h / height, zoom_w / width, bottom_h / height])
    time_left = 0.34
    ax_time = fig.add_axes([time_left / width, axis_h / height, (width - 0.02 - zoom_w - 0.3 - time_left) / width,
                            bottom_h / height])

    map_layer_labels, map_lines, map_spots = map_layer(ax_map, box, segments, paths_xy, served, rejected, legs,
                                                       releases, meetings, colour, tanker_of,
                                                       label_meetings=lambda vid: vid != dense, arrow_size=7)
    ax_map.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2], fill=False, lw=LINE,
                               ec=INK, zorder=8))
    zoom_frame = [(zoom[0], zoom[2]), (zoom[1], zoom[2]), (zoom[1], zoom[3]), (zoom[0], zoom[3]), (zoom[0], zoom[2])]
    fixed = scale_bar(ax_map, 5, box, '5 km')
    fixed.append(ax_map.text(0.012, 0.975, 'a', transform=ax_map.transAxes, fontsize=FONT, fontweight='bold', va='top',
                             zorder=9, path_effects=HALO))
    obstacles = [p for v in map_spots.values() for p in v] + list(releases.values())
    placed = boxes(ax_map, fixed)
    # the letter of the enlarged area outside its frame, at the upper right corner
    place_labels(ax_map, [(zoom[1], zoom[3], 'c', INK, FONT)], obstacles, distance=2.5, lines=map_lines + [zoom_frame],
                 placed=placed, directions=((1, 0.6), (0, 1), (-1, 0.6), (1, 0)))
    place_labels(ax_map, [(x, y, f'T{vid}', colour[vid], FONT, map_spots.get(vid, [])) for vid, (x, y) in releases.items()
                          if vid in colour]
                 + map_layer_labels, obstacles, distance=6.5, lines=map_lines + [zoom_frame], placed=placed)

    zoom_labels, zoom_lines, zoom_spots = map_layer(ax_zoom, zoom, [s for s in segments
                                                                    if any(zoom[0] <= x <= zoom[1] and
                                                                           zoom[2] <= y <= zoom[3] for x, y in s)],
                                                    paths_xy, served, rejected, {dense: legs[dense]},
                                                    {dense: releases[dense]}, {dense: meetings[dense]}, colour,
                                                    tanker_of, label_meetings=lambda vid: True, arrow_size=8)
    fixed = scale_bar(ax_zoom, 0.5, zoom, '500 m')
    fixed.append(ax_zoom.text(0.03, 0.96, 'c', transform=ax_zoom.transAxes, fontsize=FONT, fontweight='bold', va='top',
                              zorder=9, path_effects=HALO))
    place_labels(ax_zoom, [(releases[dense][0], releases[dense][1], f'T{dense}', colour[dense], FONT, zoom_spots[dense])]
                 + zoom_labels, zoom_spots[dense] + [releases[dense]], distance=6.5, lines=zoom_lines,
                 placed=boxes(ax_zoom, fixed))

    # b: schedule of every tanker: travel, waiting before the meeting, service
    rows_of = {vid: [r for r in rows if int(r['vehicle']) == vid] for vid in used}
    end = 0.0
    for level, vid in enumerate(used):
        y = len(used) - 1 - level
        clock = tankers[vid][1]
        for order, r in enumerate(rows_of[vid], start=1):
            arrive, meet, service = float(r['arrive']), float(r['meeting_time']), float(r['service'])
            ax_time.plot([clock, arrive], [y, y], color=colour[vid], lw=TRAVEL_W, solid_capstyle='butt', zorder=2)
            if meet > arrive + 1e-9:
                ax_time.plot([arrive, meet], [y, y], color=colour[vid], lw=WAIT_W, ls=WAIT_DASH, zorder=2)
            ax_time.add_patch(Rectangle((meet, y - 0.26), service, 0.52, fc=colour[vid], ec='none', zorder=3))
            ax_time.text(meet + service / 2, y, str(order), ha='center', va='center', fontsize=FONT, color='white',
                         fontweight='bold', zorder=4)
            clock = meet + service
        end = max(end, clock)
    ax_time.set_yticks(range(len(used)))
    ax_time.set_yticklabels([f'T{vid}' for vid in reversed(used)])
    for tick, vid in zip(ax_time.get_yticklabels(), reversed(used)):
        tick.set_color(colour[vid])
        tick.set_fontweight('bold')
    ax_time.set_ylim(-0.7, len(used) - 0.3)
    ax_time.set_xlim(0, 5 * math.ceil(end / 5))
    ax_time.set_xlabel('Time after the release of the tankers (min)', fontsize=FONT, labelpad=2)
    ax_time.tick_params(labelsize=FONT, length=2.5, width=LINE, pad=2)
    for side in ('top', 'right'):
        ax_time.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax_time.spines[side].set_linewidth(LINE)
    ax_time.text(0.0, 1.03, 'b', transform=ax_time.transAxes, fontsize=FONT, fontweight='bold', va='bottom', ha='left')

    handles = [Line2D([], [], color=KEY, lw=1.8, label='Tanker route'),
               Line2D([], [], color=KEY, lw=LINE, label='Path of a served driver'),
               Line2D([], [], color=REJECTED, lw=LINE, ls=REJECTED_DOTS, label='Path of a rejected driver'),
               Line2D([], [], marker='s', ls='', ms=5.5, mfc='white', mec=KEY, mew=1.4, label='Tanker release'),
               Line2D([], [], marker='o', ls='', ms=4.6, mfc=KEY, mec='white', mew=LINE, label='Meeting')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0 - 0.04 / height), ncol=5, frameon=False,
               fontsize=FONT, handlelength=2.0, columnspacing=1.3, handletextpad=0.5)
    steps = [Line2D([], [], color=KEY, lw=TRAVEL_W, label='Travel'),
             Line2D([], [], color=KEY, lw=WAIT_W, ls=WAIT_DASH, label='Waiting'),
             Patch(fc=KEY, ec='none', label='Service')]
    ax_time.legend(handles=steps, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=FONT,
                   handlelength=1.8, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.1)
    fig.savefig(out, dpi=400, metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
    print(json.dumps({'served': len(served), 'requests': len(paths), 'tankers_used': len(used),
                      'tankers': len(tankers), 'enlarged_tanker': dense,
                      'first_leg_minutes': {f'T{v}': round(first_leg[v], 2) for v in used},
                      'max_waiting': round(max(float(r['waiting']) for r in rows), 2),
                      'route_lengths': {f'T{v}': len(meetings[v]) for v in used}}))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
