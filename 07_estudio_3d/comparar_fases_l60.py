"""Compare crest and trough velocities and period-mean temperatures on the fourth grid."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostico_axial_l60 import CASES, MESHES, PERIOD, ROOT


DISTANCE = 0.0015


def phase_rows(mesh, folder, re, case, lateral):
    field = pd.read_csv(folder / 'restart_air.csv').sort_values('PointID')
    xyz = mesh['air_xyz']
    np.testing.assert_allclose(field[['x', 'y', 'z']], xyz.reshape(-1, 3), rtol=0, atol=1e-12)
    speed = field.Velocity_x.to_numpy().reshape(xyz.shape[:-1])
    u0 = json.loads((folder / 'case.json').read_text(encoding='utf8'))['U_m_s']
    z = mesh['z']
    span_mean = np.trapezoid(speed, z, axis=-1) / (z[-1] - z[0])
    x = mesh['x']
    y = mesh['y_air']
    velocity = np.array([np.interp(DISTANCE, section - section[0], ux) for section, ux in zip(y, span_mean)])
    heated = mesh['heated_i']
    length = np.hypot(np.diff(x[heated]), np.diff(y[heated, 0]))
    period = np.floor((x[heated[1:]] + x[heated[:-1]]) / (2 * PERIOD)).astype(int) + 1
    for stencil in ('baseline', 'expanded'):
        thermal = pd.read_csv(folder / stencil / 'secciones.csv')
        np.testing.assert_allclose(thermal.x_m, x, rtol=0, atol=1e-14)
        temperatures = thermal[['wall_K', 'bulk_K']].to_numpy()[heated]
        at_segments = 0.5 * (temperatures[:-1] + temperatures[1:])
        for wave in (1, 5):
            mask = period == wave
            tw, tb = np.average(at_segments[mask], axis=0, weights=length[mask])
            for phase, location in ((0.25, 'crest'), (0.75, 'trough')):
                xpos = PERIOD * (wave - 1 + phase)
                yield dict(Re_ref=re, case=case, lateral=lateral, stencil=stencil, wave=wave,
                           location=location, x_m=xpos, sample_height_m=DISTANCE,
                           gap_m=y[0, -1] - np.interp(xpos, x, y[:, 0]),
                           span_mean_ux_over_U0=np.interp(xpos, x, velocity) / u0,
                           period_mean_wall_K=tw, period_mean_bulk_K=tb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case in CASES:
        with np.load(MESHES / (case + '.npz')) as mesh:
            for re in (100, 800):
                for lateral in ('solido', 'periodico_z'):
                    folder = ROOT / f'Re{re}' / 'pared' / case / lateral
                    rows.extend(phase_rows(mesh, folder, re, case, lateral))
    frame = pd.DataFrame(rows)
    keys = ['Re_ref', 'lateral', 'stencil', 'wave', 'location']
    values = ['span_mean_ux_over_U0', 'period_mean_wall_K', 'period_mean_bulk_K']
    flat, wavy = (frame[frame.case == case].set_index(keys) for case in CASES)
    paired = (wavy[values] - flat[values]).add_prefix('wavy_minus_flat_')
    frame.merge(paired.reset_index(), on=keys, validate='many_to_one').to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
