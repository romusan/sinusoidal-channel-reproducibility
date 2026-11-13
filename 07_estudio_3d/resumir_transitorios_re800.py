"""Validate the Re=800 finite-time runs and write compact publication data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from energia_q1 import locate
from medir_su2 import fields, load_meta
from su2_caso import weights


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = HERE / 'transitorios' / 're800_l60'


def evaluate(folder):
    info = json.loads((folder / 'transient.json').read_text(encoding='utf-8'))
    base = PROJECT / info['baseline']
    meta = load_meta(base)
    grid = locate(meta)
    table = pd.read_csv(folder / 'perturbacion.csv')
    hist = pd.read_csv(folder / 'history_transient_00002.csv')
    hist.columns = hist.columns.str.strip().str.strip('"')
    log = (folder / 'su2.log').read_text(encoding='utf-8', errors='replace')
    if 'Exit Success' not in log[-3000:]:
        raise ValueError(f'Run not complete: {folder}')
    if not table.step.is_monotonic_increasing or not table.step.is_unique:
        raise ValueError(f'Non-monotone snapshots: {folder}')
    assert info['Re_ref'] == 800 and table.step.iloc[0] == 1
    assert table.step.iloc[-1] == info['steps'] + 1
    assert list(hist.Time_Iter) == list(range(2, info['steps'] + 2))
    assert np.isfinite(table.to_numpy()).all() and np.isfinite(hist.to_numpy()).all()

    geometry = meta['mesh']['caso']
    boundary = meta.get('lateral', 'solido')
    end_field = fields(folder / f'transient_{int(table.step.iloc[-1]):05d}.csv').sort_values('pointid')
    with np.load(grid) as g:
        points = g['air_xyz']
        np.testing.assert_array_equal(end_field.pointid, np.arange(points.size//3))
        np.testing.assert_allclose(end_field[['x','y','z']], points.reshape(-1,3), rtol=1e-6, atol=5e-10)
        seam = float('nan')
        if boundary == 'periodico_z':
            v = end_field[['velocityx','velocityy','velocityz']].to_numpy().reshape(points.shape)
            seam = float(np.abs(v[:,:,0] - v[:,:,-1]).max())
            assert seam < 5e-6
        cells = int(np.prod(np.array(points.shape[:3])-1))

    amplitude = info['amplitude_max_U']
    np.testing.assert_allclose(table.max_U.iloc[0], amplitude, rtol=0., atol=1e-10)
    duration = table.t_transitos.iloc[-1]
    assert info['transits'] <= duration + 1e-12
    assert duration <= info['transits'] + info['dt_star']/36. + 1e-12
    if amplitude:
        table['relative_rms'] = table.rms_U / table.rms_U.iloc[0]
        norms = table.relative_rms
        late = table.t_transitos >= .75*duration
        preceding = table.t_transitos.between(.50*duration, .75*duration, inclusive='left')
        late_peak, prior_peak = norms[late].max(), norms[preceding].max()
    else:
        table['relative_rms'] = np.nan
        norms = table.relative_rms
        late_peak = prior_peak = float('nan')

    setup = dict(mesh=folder.parent.name, cells_air=cells, geometry=geometry,
                 lateral=boundary, dt_star=info['dt_star'], amplitude_max_U=amplitude,
                 window_nominal_transits=info['transits'])
    nominal = np.isclose(table.t_transitos, info['transits'], rtol=0., atol=1.e-10)
    assert nominal.sum() == 1
    results = dict(**setup, run=folder.name, dt_s=info['dt_s'], physical_steps=info['steps'],
                   time_transits=duration, time_s=table.t_s.iloc[-1], snapshots=len(table),
                   inner_max=info['inner'], inner_mean=(hist.Inner_Iter + 1).mean(),
                   pressure_rms_log10_max=hist['rms[P]'].max(),
                   periodic_velocity_mismatch_m_s=seam, initial_rms_U=table.rms_U.iloc[0],
                   final_rms_U=table.rms_U.iloc[-1], final_relative_rms=norms.iloc[-1],
                   nominal_relative_rms=norms.loc[nominal].iloc[0],
                   max_relative_after_seed=norms.iloc[1:].max(),
                   preceding_quarter_peak_relative=prior_peak, last_quarter_peak_relative=late_peak)
    return results, table.assign(**setup, run=folder.name)


def compare_unseeded(seed, control, series):
    seed_info, control_info = (json.loads((p / 'transient.json').read_text(encoding='utf-8'))
                               for p in (seed, control))
    for key in ('baseline', 'dt_s', 'steps'):
        assert seed_info[key] == control_info[key]
    assert control_info['amplitude_max_U'] == 0
    meta = load_meta(PROJECT / seed_info['baseline'])
    with np.load(locate(meta)) as grid:
        weight = (weights(grid['x'])[:,None,None] *
                  np.stack([weights(y) for y in grid['y_air']])[:,:,None] *
                  weights(grid['z'])).ravel()
    weight /= weight.sum()
    components = ['velocityx', 'velocityy', 'velocityz']
    shared = {1} | {int(p.stem.rsplit('_', 1)[1]) for p in control.glob('transient_*.csv')}
    for index, row in series.iterrows():
        step = int(row.step)
        if step not in shared:
            continue
        name = 'initial_00001.csv' if step == 1 else f'transient_{step:05d}.csv'
        a, b = (fields(d / name).sort_values('pointid') for d in (seed, control))
        np.testing.assert_array_equal(a.pointid, b.pointid)
        delta = a[components].to_numpy() - b[components].to_numpy()
        rms = np.sqrt(weight @ (delta**2).sum(axis=1)) / meta['U_m_s']
        series.loc[index, 'rms_U_against_control'] = rms
    series['relative_rms_against_control'] = series.rms_U_against_control / series.rms_U.iloc[0]
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    rows, series = [], []
    folders = {p.parent.name: p.parent for p in args.root.glob('media/*/perturbacion.csv')}
    for source in sorted(args.root.glob('*/*/perturbacion.csv')):
        folder = source.parent
        row, curve = evaluate(folder)
        control = folders.get(folder.name + '_unseeded') if folder.parent.name == 'media' else None
        if control is not None:
            curve = compare_unseeded(folder, control, curve)
            row['final_relative_rms_against_control'] = curve.relative_rms_against_control.iloc[-1]
        rows.append(row)
        series.append(curve)
    summary = pd.DataFrame(rows)
    primary = summary.query("mesh == 'media' and dt_star == .5 and amplitude_max_U > 0")
    assert len(primary) == 4 and not primary.duplicated(['geometry','lateral']).any()
    summary.to_csv(HERE / 'transitorios_re800_l60.csv', index=False, encoding='utf-8-sig')
    pd.concat(series, ignore_index=True).to_csv(HERE / 'series_transitorias_re800_l60.csv',
                                              index=False, encoding='utf-8-sig')
    print(summary[['mesh','geometry','lateral','dt_star','amplitude_max_U','time_transits','final_rms_U',
                   'final_relative_rms','last_quarter_peak_relative','pressure_rms_log10_max']].to_string(index=False))


if __name__ == '__main__':
    main()
