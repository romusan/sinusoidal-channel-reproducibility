"""Plot the phase-resolved change in physical wall heat flux from Supplementary Data S13."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    folder = Path(__file__).resolve().parent / 'ijts'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=folder / 'data/S13_L60_axial_diagnostics.csv')
    parser.add_argument('--out', type=Path, default=folder / 'figures/wall_flux.pdf')
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    data = data[data['wave'].isin([1, 5])]
    index = ['Re_ref', 'wave', 'phase', 'lateral', 'stencil']
    flux = data.pivot(index=index, columns='case', values='physical_Q1_wall_flux_W_m2')
    contrast = (flux['H15_A150_L60'] - flux['H15_Plana']).rename('difference').reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.25), sharex=True, sharey=True,
                            layout='constrained')
    for row, reynolds in enumerate((100, 800)):
        for column, wave in enumerate((1, 5)):
            ax = axes[row, column]
            cases = contrast[(contrast.Re_ref == reynolds) & (contrast.wave == wave)]
            bands = cases.groupby('phase').difference.agg(['min', 'mean', 'max'])
            ax.axhline(0, color='0.55', lw=.65)
            for phase in (.25, .75):
                ax.axvline(phase, color='0.65', ls=':', lw=.75)
            ax.fill_between(bands.index, bands['min'], bands['max'], color='#a8c9d8', alpha=.8)
            ax.plot(bands.index, bands['mean'], color='#175373', lw=1.45)
            ax.set_title(f'$Re_0={reynolds}$, period {wave}', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, .25, .5, .75, 1])
    fig.supxlabel('Phase within wave', fontsize=10)
    fig.supylabel(r'Wavy minus flat wall heat flux (W m$^{-2}$)', fontsize=10)
    fig.savefig(args.out)


if __name__ == '__main__':
    main()
