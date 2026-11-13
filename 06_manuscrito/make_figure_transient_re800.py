"""Plot the finite-time Re=800 disturbance on the matched medium grids."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data', type=Path,
                     default=ROOT / '07_estudio_3d' / 'series_transitorias_re800_l60.csv')
    cli.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    table = pd.read_csv(args.data)
    primary = table.query("mesh == 'media' and dt_star == .5 and amplitude_max_U > 0")
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.0, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.15, 2.7), layout='constrained')
    for ax, lateral, heading in zip(axes, ('solido', 'periodico_z'),
                                   ('(a) Solid sidewalls', '(b) Transverse periodicity')):
        subset = primary.loc[primary.lateral.eq(lateral)]
        for case, label, color, style in [
            ('H15_Plana','Flat','#006b86','o-'), ('H15_A150_L60','60 mm wave','#aa492f','s--')]:
            rows = subset.loc[subset.geometry.eq(case)].sort_values('t_transitos')
            assert len(rows)>3 and rows.relative_rms.gt(0).all()
            ax.semilogy(rows.t_transitos, rows.relative_rms, style, color=color, label=label,
                        ms=3.0, lw=1.15)
            if 'relative_rms_against_control' in rows:
                paired = rows.dropna(subset=['relative_rms_against_control'])
                ax.semilogy(paired.t_transitos, paired.relative_rms_against_control, ':',
                            color=color, lw=1.15, alpha=.9)
        ax.set_title(heading, loc='left', fontweight='normal')
        ax.set_xticks([0, .5, 1, 1.5, 2])
        ax.set_xlim(-.03, 2.06)
        ax.set_xlabel('Time / duct transit time')
        ax.grid(axis='y', which='major', linewidth=.55, alpha=.3)
    axes[0].set_ylabel('Velocity RMS / initial RMS')
    handles, _ = axes[1].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#555555', linestyle=':', label='Relative to unseeded run'))
    axes[1].legend(handles=handles, frameon=False)
    fig.savefig(args.out, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
