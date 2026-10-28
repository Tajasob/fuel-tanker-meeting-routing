#!/usr/bin/env python3
"""Figure 1: refuelling at a station against a meeting on the driver's route (vector drawing).

    python analysis/concept.py OUT.pdf

Every shape is drawn here from geometric primitives; no image is embedded. The figure uses the
settings of figure_schematic.py: printed width 16 cm, one type size, no line thinner than one
point, and black, white or a hue for every shape and label, with no grey.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

WIDTH = 160 / 25.4
FONT = 8
LINE = 1.0
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica'],
                     'font.size': FONT, 'axes.linewidth': LINE, 'patch.linewidth': LINE,
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.prop_cycle': plt.cycler(color=['#000000'])})  # nothing falls back to default colours
INK = '#000000'
BLOCK, ROAD = '#f6e9cc', '#ffffff'  # city blocks in light sand, streets white
DRIVER, TANKER, PUMP = '#2a78d6', '#eb6834', '#e34948'
GLASS, LAMP, TYRE, HUB, CAB = '#c6dcf7', '#fff4c2', '#000000', '#ffffff', '#b4461c'
BAND = '#cde0f8'  # opaque light blue under the driver's planned route
QUEUE = '#1baf7a'  # the two cars waiting at the station
ROAD_W = 9.0  # points
XS, YS = (0.9, 3.6, 6.3, 9.0), (0.85, 2.9, 5.0)  # road axes in panel units
CAR_X = 1.75  # the driver's car on the first street of both panels
UNDER_CAR = [(CAR_X - 0.3, YS[0]), (CAR_X, YS[0]), (CAR_X + 0.3, YS[0])]  # no route arrow under the car
TOP = 5.75  # height of each panel in the same units; the panels are 10 units wide
GAP = 0.03  # space between the panels, as a share of the figure width
PT_PER_UNIT = (1 - GAP) / 2 * WIDTH * 72 / 10  # points per panel unit at the printed size


def poly(ax, points, **kw):
    ax.add_patch(Polygon(points, closed=True, **kw))


def car(ax, x, y, w, body, flip=False, z=6):
    """Side view of a car standing on (x, y), w units long, facing right unless flipped."""
    s = -1 if flip else 1
    X = lambda u: x + s * u * w  # noqa: E731
    ax.add_patch(FancyBboxPatch((x - 0.5 * w, y + 0.09 * w), w, 0.2 * w,
                                boxstyle=f'round,pad=0,rounding_size={0.08 * w}', fc=body, ec='none', zorder=z))
    poly(ax, [(X(-0.33), y + 0.27 * w), (X(-0.2), y + 0.47 * w), (X(0.13), y + 0.47 * w), (X(0.3), y + 0.27 * w)],
         fc=body, ec='none', zorder=z)
    poly(ax, [(X(-0.27), y + 0.29 * w), (X(-0.18), y + 0.42 * w), (X(-0.04), y + 0.42 * w), (X(-0.04), y + 0.29 * w)],
         fc=GLASS, ec='none', zorder=z + 0.1)
    poly(ax, [(X(0.01), y + 0.29 * w), (X(0.01), y + 0.42 * w), (X(0.11), y + 0.42 * w), (X(0.22), y + 0.29 * w)],
         fc=GLASS, ec='none', zorder=z + 0.1)
    ax.add_patch(Circle((X(0.46), y + 0.2 * w), 0.03 * w, fc=LAMP, ec='none', zorder=z + 0.1))
    for u in (-0.29, 0.29):
        ax.add_patch(Circle((X(u), y + 0.09 * w), 0.1 * w, fc=TYRE, ec='none', zorder=z + 0.2))
        ax.add_patch(Circle((X(u), y + 0.09 * w), 0.045 * w, fc=HUB, ec='none', zorder=z + 0.3))


def tanker(ax, x, y, w, flip=False, z=6):
    """Side view of a fuel tanker standing on (x, y), w units long, cab on the right unless flipped."""
    s = -1 if flip else 1
    X = lambda u: x + s * u * w  # noqa: E731

    def rect(u0, u1, v0, v1, **kw):
        ax.add_patch(Rectangle((min(X(u0), X(u1)), y + v0 * w), abs(u1 - u0) * w, (v1 - v0) * w, **kw))
    rect(-0.5, 0.42, 0.07, 0.125, fc=TYRE, ec='none', zorder=z)
    ax.add_patch(FancyBboxPatch((min(X(-0.5), X(0.17)), y + 0.125 * w), 0.67 * w, 0.29 * w,
                                boxstyle=f'round,pad=0,rounding_size={0.145 * w}', fc=TANKER, ec='none', zorder=z))
    cab = [(X(0.2), y + 0.125 * w), (X(0.2), y + 0.46 * w), (X(0.37), y + 0.46 * w), (X(0.5), y + 0.3 * w),
           (X(0.5), y + 0.125 * w)]
    poly(ax, cab, fc=CAB, ec='none', zorder=z)
    poly(ax, [(X(0.28), y + 0.3 * w), (X(0.28), y + 0.42 * w), (X(0.355), y + 0.42 * w), (X(0.46), y + 0.3 * w)],
         fc=GLASS, ec='none', zorder=z + 0.1)
    rect(0.475, 0.5, 0.17, 0.215, fc=LAMP, ec='none', zorder=z + 0.1)
    for u in (-0.37, -0.215, 0.36):
        ax.add_patch(Circle((X(u), y + 0.075 * w), 0.075 * w, fc=TYRE, ec='none', zorder=z + 0.2))
        ax.add_patch(Circle((X(u), y + 0.075 * w), 0.033 * w, fc=HUB, ec='none', zorder=z + 0.3))
    drop(ax, X(-0.165), y + 0.27 * w, 0.19 * w, 'white', z=z + 0.2)


def drop(ax, x, y, h, colour, z=7):
    """A fuel drop of height h with its tip at the top, centred at (x, y): a circle joined to the tip
    by its two tangents."""
    r = 0.34 * h
    cy = y - 0.5 * h + r
    d = h - r  # distance from the circle centre to the tip
    a = math.acos(r / d)
    start, end = math.pi / 2 - a, -3 * math.pi / 2 + a  # tangent points, going round the bottom
    arc = [(x + r * math.cos(start + (end - start) * k / 48), cy + r * math.sin(start + (end - start) * k / 48))
           for k in range(49)]
    poly(ax, [(x, cy + d)] + arc, fc=colour, ec='none', zorder=z)


def station(ax, x, y, s, z=5):
    """A fuel station: canopy on two posts over a pump, base centre at (x, y), s units wide."""
    ax.add_patch(Rectangle((x - 0.5 * s, y + 0.78 * s), s, 0.12 * s, fc=PUMP, ec='none', zorder=z))
    for u in (-0.42, 0.36):
        ax.add_patch(Rectangle((x + u * s, y), 0.06 * s, 0.78 * s, fc=INK, ec='none', zorder=z))
    ax.add_patch(FancyBboxPatch((x - 0.17 * s, y), 0.3 * s, 0.52 * s, boxstyle=f'round,pad=0,rounding_size={0.04 * s}',
                                fc=PUMP, ec='none', zorder=z + 0.1))
    ax.add_patch(Rectangle((x - 0.11 * s, y + 0.3 * s), 0.18 * s, 0.13 * s, fc='white', ec='none', zorder=z + 0.2))
    ax.plot([x + 0.13 * s, x + 0.24 * s, x + 0.24 * s], [y + 0.36 * s, y + 0.36 * s, y + 0.12 * s], color=INK,
            lw=LINE, solid_capstyle='round', zorder=z + 0.2)


def flag(ax, x, y, h, z=7):
    ax.plot([x, x], [y, y + h], color=INK, lw=1.2, solid_capstyle='round', zorder=z)
    poly(ax, [(x, y + h), (x + 0.55 * h, y + 0.8 * h), (x, y + 0.6 * h)], fc=INK, ec='none', zorder=z)


def streets(ax):
    ax.add_patch(Rectangle((0, 0), 10, TOP, fc=BLOCK, ec='none', zorder=0))
    for y in YS:
        ax.plot([0, 10], [y, y], color=ROAD, lw=ROAD_W, solid_capstyle='butt', zorder=1)
    for x in XS:
        ax.plot([x, x], [0, TOP], color=ROAD, lw=ROAD_W, solid_capstyle='butt', zorder=1)


def chevrons(ax, points, colour, every=1.3, first=0.75, size=0.15, z=4, skip=()):
    """Arrowheads along a polyline, every so many units, pointing along it; positions within 0.3
    units of a point in skip are left out."""
    run, target = 0.0, first
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        while run + length >= target:
            f = (target - run) / length
            px, py = x1 + f * (x2 - x1), y1 + f * (y2 - y1)
            if all(math.hypot(px - qx, py - qy) > 0.3 for qx, qy in skip):
                poly(ax, [(px + ux * size, py + uy * size), (px - ux * 0.6 * size - uy * 0.75 * size,
                                                             py - uy * 0.6 * size + ux * 0.75 * size),
                          (px - ux * 0.6 * size + uy * 0.75 * size, py - uy * 0.6 * size - ux * 0.75 * size)],
                     fc=colour, ec='none', zorder=z)
            target += every
        run += length


def route(ax, points, colour, band=None, dashed=False, z=3, arrows=True, skip=(), first=0.75, every=1.3):
    """A route along the given points, with arrowheads; band, if given, is the colour of a road-wide
    band under it."""
    xs, ys = zip(*points)
    if band:
        ax.plot(xs, ys, color=band, lw=ROAD_W, solid_capstyle='butt', solid_joinstyle='miter', zorder=z - 0.5)
    style = '-'
    if dashed:  # dashes of 3 and gaps of 2 line widths, shifted so that the route ends on a whole dash
        length = sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:]))
        style = ((3 - length * PT_PER_UNIT / 1.6) % 5, (3, 2))
    ax.plot(xs, ys, color=colour, lw=1.6, ls=style, solid_joinstyle='miter', dash_joinstyle='miter', zorder=z)
    if arrows:
        chevrons(ax, points, colour, z=z + 0.2, skip=skip, first=first, every=every)


def label(ax, x, y, text, ha='left', va='center', colour=INK, z=9, **kw):
    ax.text(x, y, text, ha=ha, va=va, color=colour, zorder=z, **kw)


def pointer(ax, start, end, z=8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-', lw=LINE, color=INK, shrinkA=1, shrinkB=3,
                                 zorder=z))


def base_scene(ax, letter, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, TOP)
    ax.set_aspect('equal')
    ax.axis('off')
    streets(ax)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(0, 4), textcoords='offset points', ha='left',
                va='bottom', fontweight='bold')
    ax.annotate(title, (0, 1), xycoords='axes fraction', xytext=(11, 4), textcoords='offset points', ha='left',
                va='bottom')
    planned = [(0.25, YS[0]), (XS[2], YS[0]), (XS[2], YS[2]), (9.55, YS[2])]  # from the origin marker
    return planned


def origin_destination(ax):
    ax.add_patch(Circle((0.25, YS[0]), 0.13, fc='white', ec=INK, lw=1.2, zorder=7))
    label(ax, 0.2, YS[0] + 0.33, 'Origin', ha='left', va='bottom')
    flag(ax, 9.6, YS[2], 0.62)
    label(ax, 9.75, YS[2] - 0.32, 'Destination', ha='right', va='top')


def main(out: Path):
    gap = GAP
    w = (1 - gap) / 2
    axes_h = w * WIDTH * TOP / 10
    title_h = 0.2
    height = axes_h + title_h
    fig = plt.figure(figsize=(WIDTH, height))
    ax_a = fig.add_axes([0.0, 0.0, w, axes_h / height])
    ax_b = fig.add_axes([w + gap, 0.0, w, axes_h / height])

    # a: the driver leaves the planned route for a station, joins the back of the queue there and
    # rejoins the route beyond the station; with right-hand traffic it drives north on the east lane
    planned = base_scene(ax_a, 'a', 'Refuelling at a station')
    lot_y = YS[1] + 0.62  # the station stands in the block between the second and third streets
    base, queue_w = lot_y - 0.18, 0.52
    lane = base + 0.26 * queue_w  # height of the middle of the queue cars
    # no arrowhead of the planned route where the detour leaves it or rejoins it
    route(ax_a, planned, DRIVER, band=BAND, skip=UNDER_CAR + [(XS[1], YS[0]), (XS[2], lane)])
    station(ax_a, XS[2] - 0.85, base, 1.0)
    for x in (XS[1] + 0.52, XS[1] + 1.12):  # the queue faces the pump, its back towards the street
        car(ax_a, x, base, queue_w, QUEUE, z=6)
    route(ax_a, [(XS[1] + 0.11, YS[0]), (XS[1] + 0.11, lane)], DRIVER, dashed=True, z=4, first=0.75, every=1.05)
    route(ax_a, [(XS[2] - 0.55, lane), (XS[2], lane)], DRIVER, dashed=True, z=4, first=0.3, every=1.05)
    label(ax_a, XS[2] - 0.35, lot_y + 0.82, 'Station', va='bottom', ha='right')
    label(ax_a, XS[1] + 0.78, lot_y + 0.26, 'Queue', va='bottom', ha='center')
    label(ax_a, XS[1] + 0.32, (YS[0] + YS[1]) / 2, 'Detour', ha='left')
    car(ax_a, CAR_X, YS[0] + 0.02, 0.9, DRIVER)
    label(ax_a, 4.55, YS[0] - 0.36, 'Planned route', va='top')
    origin_destination(ax_a)

    # b: a tanker meets the driver at a point of the planned route
    planned = base_scene(ax_b, 'b', 'Mobile fuel delivery on the route')
    meet = (XS[2], YS[0] + 0.62 * (YS[1] - YS[0]))
    # candidates lie ahead of the driver only: a tanker cannot meet the driver at a point already passed
    candidates = [(2.45, YS[0]), (XS[1], YS[0]), (5.0, YS[0]), (XS[2], YS[0]), (XS[2], YS[1]),
                  (XS[2], (YS[1] + YS[2]) / 2), (XS[2], YS[2]), (7.65, YS[2]), (XS[3], YS[2])]
    route(ax_b, planned, DRIVER, band=BAND, skip=candidates + [meet] + UNDER_CAR)
    release = (XS[3] - 0.45, YS[1])
    # the tanker drives south in the west lane, beside the driver's route, to the meeting point
    lane = XS[2] - 0.18
    route(ax_b, [release, (lane, YS[1]), (lane, meet[1])], TANKER, z=4.5, first=1.1, every=1.0, skip=[(lane, YS[1])])
    for x, y in candidates:
        ax_b.add_patch(Circle((x, y), 0.1, fc='white', ec=DRIVER, lw=1.2, zorder=6))
    ax_b.add_patch(Circle(meet, 0.17, fc=INK, ec='white', lw=1.2, zorder=7.5))
    drop(ax_b, meet[0] - 0.42, meet[1] + 0.03, 0.4, TANKER, z=7.6)
    tanker(ax_b, release[0], release[1] + 0.03, 1.55, flip=True)
    car(ax_b, CAR_X, YS[0] + 0.02, 0.9, DRIVER)
    label(ax_b, meet[0] + 0.3, meet[1], 'Meeting point', va='center')
    label(ax_b, release[0], release[1] + 0.85, 'Tanker', va='bottom', ha='center')
    label(ax_b, 4.55, YS[0] - 0.36, 'Planned route', va='top')
    label(ax_b, 0.35, YS[1] + 0.95, 'Candidate meeting\npoints on the route', va='bottom', linespacing=1.1)
    pointer(ax_b, (1.6, YS[1] + 0.9), (2.45, YS[0] + 0.12))
    origin_destination(ax_b)
    fig.savefig(out, metadata={'Creator': None, 'Producer': None, 'CreationDate': None})


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('refuelling-concept.pdf'))
