#!/usr/bin/env python3
"""Figure 2: meeting and service timing on a three-link example (deterministic drawing).

    python analysis/figure_schematic.py OUT.pdf

The figure is drawn at its printed width, 16 cm, with one type size for all lettering, no line
thinner than one point and no grey, as the figure guidelines of Scientific Reports ask; the other
figure scripts use the same settings. Drivers are blue and the tanker is orange, as in Figure 1,
and panel b draws travel, waiting and service as the schedule of Figure 5 does. Node names are
italic, as in the text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402


WIDTH = 160 / 25.4  # printed width in inches, the text width of the manuscript
FONT = 8  # points, every label of all figures
LINE = 1.0  # points, the thinnest line
HEIGHT = WIDTH * 4.6 / 6.8  # inches
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica'],
                     'font.size': FONT, 'axes.linewidth': LINE, 'xtick.major.width': LINE,
                     'ytick.major.width': LINE, 'patch.linewidth': LINE,
                     'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
                     'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold',
                     'mathtext.sf': 'Liberation Sans', 'mathtext.tt': 'Liberation Sans', 'mathtext.cal': 'Liberation Sans',
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.prop_cycle': plt.cycler(color=['#000000'])})  # nothing falls back to default colours
INK = '#000000'
DRIVER, TANKER = '#2a78d6', '#eb6834'  # the colours of the driver and the tanker in Figure 1
TRAVEL_W, WAIT_W, WAIT_DASH = 2.0, 1.0, (0, (2.5, 1.5))  # as in the schedule of Figure 5


def panel(fig, y, letter, title):
    """Bold lower-case panel letter at the left edge of the figure, followed by the panel title,
    both at the figure's type size, with their baseline at height y (figure fraction)."""
    fig.text(0.0, y, letter, ha='left', va='baseline', fontweight='bold')
    fig.text(11 / (WIDTH * 72), y, title, ha='left', va='baseline')


def dimension(ax, x0, x1, y, label, tick=0.035):
    """Travel time of a link as a dimension line below it, with end ticks and the label under it."""
    ax.plot([x0, x1], [y, y], color=INK, lw=LINE, solid_capstyle='butt')
    for x in (x0, x1):
        ax.plot([x, x], [y - tick, y + tick], color=INK, lw=LINE, solid_capstyle='butt')
    ax.text((x0 + x1) / 2, y - 0.05, label, ha='center', va='top')


def main(out: Path):
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    pts = HEIGHT * 72  # figure height in points

    def frac(p):
        return p / pts

    # a: directed links, the release of the tanker, the predicted passages and the meeting points
    top = fig.add_axes([0.0, frac(176), 1.0, frac(112)])
    nodes = {'S': (0.0, 0.62), 'A': (1.0, 1.05), 'B': (4.0, 1.05), 'C': (6.0, 1.05)}
    for a, b in (('S', 'A'), ('A', 'B'), ('B', 'C')):
        top.add_patch(FancyArrowPatch(nodes[a], nodes[b], arrowstyle='-|>', mutation_scale=12, lw=1.4, color=INK,
                                      shrinkA=9, shrinkB=9))
    top.text(0.6, 0.76, '2 min', ha='left', va='center')
    for name, (x, y) in nodes.items():
        top.plot(x, y, ls='', marker='o', ms=13, mfc='white', mec=INK, mew=1.2, zorder=3)
        top.text(x, y, f'${name}$', ha='center', va='center', zorder=4)
    for name, x, t in (('M1', 2.5, '7 min'), ('M2', 5.0, '14 min')):
        top.plot(x, 1.05, ls='', marker='o', ms=9, mfc=INK, mec=INK, zorder=3)
        top.text(x, 1.2, f'{name} at {t}', ha='center', va='bottom')
    dimension(top, 1.0, 4.0, 0.86, '6 min')
    dimension(top, 4.0, 6.0, 0.86, '4 min')
    top.text(1.0, 0.56, 'Driver 1 predicted at $A$ at 4 min and at $B$ at 10 min', ha='left', va='top')
    top.text(1.0, 0.38, 'Driver 2 predicted at $B$ at 12 min and at $C$ at 16 min', ha='left', va='top')
    top.text(-0.35, 0.42, 'Tanker released\nat $S$ at 0 min', ha='left', va='top', linespacing=1.15)
    top.set_xlim(-0.4, 6.5)
    top.set_ylim(0.08, 1.45)
    top.axis('off')
    panel(fig, frac(295), 'a', 'Directed road links and meeting points')

    # b: travel, waiting and service over time
    bottom = fig.add_axes([0.085, frac(54), 0.905, frac(90)])
    rows = {'Driver 1': 2, 'Driver 2': 1, 'Tanker': 0}
    steps = {'Driver 1': [('travel', 4.0, 7.0), ('service', 7.0, 8.0)],
             'Driver 2': [('travel', 12.0, 14.0), ('service', 14.0, 15.5)],
             'Tanker': [('travel', 0.0, 5.0), ('wait', 5.0, 7.0), ('service', 7.0, 8.0),
                        ('travel', 8.0, 13.0), ('wait', 13.0, 14.0), ('service', 14.0, 15.5)]}
    for who, items in steps.items():
        y, colour = rows[who], TANKER if who == 'Tanker' else DRIVER
        for kind, start, end in items:
            if kind == 'travel':
                bottom.plot([start, end], [y, y], color=colour, lw=TRAVEL_W, solid_capstyle='butt', zorder=2)
            elif kind == 'wait':
                bottom.plot([start, end], [y, y], color=colour, lw=WAIT_W, ls=WAIT_DASH, zorder=2)
            else:
                bottom.add_patch(Rectangle((start, y - 0.26), end - start, 0.52, fc=colour, ec='none', zorder=3))
    bottom.set_yticks(list(rows.values()), list(rows.keys()))
    bottom.set_xticks([0, 4, 5, 7, 8, 12, 13, 14, 15.5], ['0', '4', '5', '7', '8', '12', '13', '14', '15.5'])
    bottom.set_xlim(-0.3, 16.3)
    bottom.set_ylim(-0.6, 2.6)
    bottom.set_xlabel('Time after the release of the tanker (min)', labelpad=3)
    bottom.tick_params(length=2.5, width=LINE, pad=2)
    for side in ('top', 'right'):
        bottom.spines[side].set_visible(False)
    panel(fig, frac(158), 'b', 'Travel, waiting and service')
    handles = [Line2D([], [], color=INK, lw=TRAVEL_W, label='Travel'),
               Line2D([], [], color=INK, lw=WAIT_W, ls=WAIT_DASH, label='Waiting'),
               Patch(fc=INK, ec='none', label='Service')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, frac(4)), ncol=3, frameon=False,
               handlelength=1.8, columnspacing=1.6, handletextpad=0.5, borderaxespad=0.1)
    fig.savefig(out, metadata={'Creator': None, 'Producer': None, 'CreationDate': None})


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('meeting-service-schematic.pdf'))
