"""Solve the 60 mm wave on the mesh family previously used for the 30 mm wave."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import pandas as pd

from comparar_contornos_re100 import compare
from energia_acotada import thermal_case
from flujo_periodico_re800 import finished, run_periodic_case
from flujo_re800 import HERE, seed_restart
from generar_malla_su2 import GEOMETRY, generate_mesh
from prueba_re100_contornos import paths, pressure_residual
from verificar_malla_su2 import verify


FAMILY = HERE / 'mallas' / 'prueba_re800'
RUNS = HERE / 'corridas' / 'prueba_re800_l60_malla_comun'
WAVE = 'H15_A150_L60'
GRIDS = ('fina', 'media', 'gruesa')


def common_mesh(grid):
    directory = FAMILY / grid
    short = json.loads((directory / 'H15_A150_L30.json').read_text(encoding='utf-8'))
    counts = [short[key] for key in (
        'celdas_calentadas', 'celdas_altura_aire', 'celdas_ancho', 'celdas_espesor_placa')]
    mesh = directory / f'{WAVE}.su2'
    if not mesh.is_file():
        generate_mesh(WAVE, GEOMETRY, mesh, *counts, .75, 1.25, heat_flux_strips=True)
    with np.load(mesh.with_suffix('.npz')) as long, np.load(directory / 'H15_A150_L30.npz') as short:
        for key in ('x', 'z', 'air_ids', 'solid_ids', 'heated_i'):
            np.testing.assert_array_equal(long[key], short[key])
        for arrays in (long, short):
            assert arrays['y_air'].shape == (len(arrays['x']), counts[1] + 1)
        np.testing.assert_allclose(long['y_air'][0], short['y_air'][0], rtol=0., atol=1e-14)
    print(grid, verify(mesh), flush=True)
    return mesh


def solid_case(grid, mesh, folder, solver):
    donor_mesh, donor, _ = paths(grid, WAVE, 800)
    if not finished(folder):
        seed = folder / 'initial_guess'
        seed_restart(donor, donor_mesh.with_suffix('.npz'), mesh.with_suffix('.npz'), seed)
        subprocess.run([sys.executable, str(HERE / 'su2_caso.py'), '--mesh', str(mesh),
                        '--out', str(folder), '--re', '800', '--cold', '--fluid-only', '--threads', '1',
                        '--restart', str(seed), '--iterations', '900', '--cfl', '50', '--su2', str(solver)],
                       check=True)
    if pressure_residual(folder) > -11.:
        raise RuntimeError(f'Solid residual target not met in {folder}')


def long_wave(grid, solver):
    mesh = common_mesh(grid)
    solid = RUNS / grid / 'solido'
    periodic = RUNS / grid / 'periodico_z'
    solid_case(grid, mesh, solid, solver)
    previous_mesh, _, previous_periodic = paths(grid, WAVE, 800)
    run_periodic_case(periodic, solid, mesh.with_suffix('.npz'), solver,
                      donor=(previous_periodic, previous_mesh.with_suffix('.npz')))
    if pressure_residual(periodic) > -11.:
        raise RuntimeError(f'Periodic residual target not met in {periodic}')
    assert (solid / 'inlet.dat').read_bytes() == (periodic / 'inlet.dat').read_bytes()
    solid_mesh, periodic_mesh = ((case / 'mesh.su2').read_text(encoding='ascii').split('NMARK=', 1)[0]
                                 for case in (solid, periodic))
    assert solid_mesh == periodic_mesh

    rows = []
    for boundary, case in [('solido', solid), ('periodico_z', periodic)]:
        row = thermal_case(mesh, 800., case=case, tolerance=1.e-6,
                           first_ring=False, archive=case / 'termico')
        row.update(malla=grid, lateral=boundary, log10_rms_presion=pressure_residual(case))
        print(grid, boundary, f"h={row['h_W_m2K']:.8f} dp={row['dp_Pa']:.8f}", flush=True)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--mallas', nargs='+', choices=GRIDS, default=list(GRIDS))
    cli.add_argument('--su2', type=Path,
                     default=Path(os.environ['TEMP']) / 'colector_su2_860/bin/SU2_CFD.exe')
    args = cli.parse_args()
    solid_flat = pd.read_csv(HERE / 'resultados_re800_acotados.csv')
    solid_flat['lateral'] = 'solido'
    periodic_flat = pd.read_csv(HERE / 'resultados_re800_periodicos.csv')
    flat = pd.concat([solid_flat, periodic_flat], ignore_index=True).query("caso == 'H15_Plana'")
    for i, row in flat.iterrows():
        run_family = 'prueba_re800' if row.lateral == 'solido' else 'prueba_re800_periodica'
        archive = HERE / 'corridas' / run_family / row.malla / 'H15_Plana_Re800'
        flat.loc[i, 'log10_rms_presion'] = pressure_residual(archive)

    raw = HERE / 'resultados_re800_l60_malla_comun.csv'
    paired = HERE / 'comparacion_re800_l60_malla_comun.csv'
    previous = pd.read_csv(raw) if raw.exists() else None
    for grid in args.mallas:
        if previous is not None and previous.malla.eq(grid).sum() == 4:
            print(grid, 'already tabulated', flush=True)
            continue
        result = pd.concat([flat.loc[flat.malla.eq(grid)], long_wave(grid, args.su2.resolve())],
                           ignore_index=True)
        previous = result if previous is None else pd.concat([previous, result], ignore_index=True)
        previous.to_csv(raw, index=False, encoding='utf-8-sig')
        completed = [g for g in reversed(GRIDS) if previous.malla.eq(g).any()]
        summary = compare(raw, completed, False, 800)
        summary.to_csv(paired, index=False, encoding='utf-8-sig')
        print(summary[['malla', 'celdas_aire', 'cambio_h_solido_pct', 'cambio_h_periodico_pct',
                       'cambio_dp_solido_pct', 'cambio_dp_periodico_pct']].to_string(index=False))


if __name__ == '__main__':
    main()
