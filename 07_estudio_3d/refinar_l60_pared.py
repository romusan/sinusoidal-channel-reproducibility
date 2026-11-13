"""Matched Reynolds comparison on the common family and a near-wall fourth mesh."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd

from energia_acotada import thermal_case
from flujo_periodico_re800 import finished, run_periodic_case, verify_flow
from flujo_re800 import HERE, seed_restart
from generar_malla_su2 import GEOMETRY, generate_mesh
from prueba_re100_contornos import CASES, paths, pressure_residual
from refinar_re800_l60 import FAMILY, RUNS
from su2_caso import options
from verificar_malla_su2 import verify


GRIDS = ('pared', 'fina', 'media', 'gruesa')
REYNOLDS = (100, 800)
ROOT = HERE / 'corridas' / 'prueba_l60_refinamiento_pared'
BOUNDARIES = ('solido', 'periodico_z')


def mesh_path(grid, name):
    mesh = FAMILY / grid / (name + '.su2')
    if grid == 'pared' and not all(mesh.with_suffix(ext).is_file() for ext in ('.su2', '.npz', '.json')):
        generate_mesh(name, GEOMETRY, mesh, 135, 14, 27, 4, .75, 1.25,
                      heat_flux_strips=True, split_bottom=4)
        fine = FAMILY / 'fina' / (name + '.npz')
        with np.load(fine) as parent, np.load(mesh.with_suffix('.npz')) as child:
            for key in ('x', 'z', 'heated_i', 'solid_xyz'):
                np.testing.assert_array_equal(child[key], parent[key])
            np.testing.assert_allclose(child['y_air'][:, :9:2], parent['y_air'][:, :5], rtol=0, atol=1e-14)
            np.testing.assert_array_equal(child['y_air'][:, 8:], parent['y_air'][:, 4:])
            np.testing.assert_allclose(np.diff(child['y_air'][:, :2], axis=1),
                                       np.diff(parent['y_air'][:, :2], axis=1) / 2, rtol=0, atol=1e-14)
        print(grid, verify(mesh), flush=True)
    return mesh


def common_high(grid, name, boundary):
    if name == CASES[1]:
        return RUNS / grid / boundary
    label = 'prueba_re800' if boundary == 'solido' else 'prueba_re800_periodica'
    return HERE / 'corridas' / label / grid / f'{name}_Re800'


def archive(grid, name, re, boundary):
    if re == 800 and grid != 'pared':
        return common_high(grid, name, boundary)
    return ROOT / f'Re{re}' / grid / name / boundary


def initial_guess(grid, name, re, boundary):
    if grid == 'pared' and re == 800:
        return FAMILY / 'fina' / (name + '.npz'), common_high('fina', name, boundary)
    base = 'fina' if grid == 'pared' else grid
    mesh, solid, periodic = paths(base, name, re)
    return mesh.with_suffix('.npz'), solid if boundary == 'solido' else periodic


def converge(case, solver):
    for attempt in range(3):
        if finished(case) and pressure_residual(case) <= -11:
            return
        if not (case / 'restart_air.csv').is_file():
            raise RuntimeError(f'No usable restart in {case}')
        config = dict(tuple(v.strip() for v in line.split('=', 1))
                      for line in (case / 'air.cfg').read_text(encoding='ascii').splitlines()
                      if '=' in line and not line.lstrip().startswith('%'))
        for name in ('history_air.csv', 'su2.log'):
            shutil.copyfile(case/name, case/f'{name}.retry{attempt}')
        shutil.copyfile(case/'restart_air.csv', case/'solution_air.csv')
        config.update(RESTART_SOL='YES', CFL_NUMBER=('50' if attempt == 0 else '35'),
                      INNER_ITER='1100')
        options(case/'air.cfg', config)
        print('Continuing', case, flush=True)
        with (case/'su2.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(solver), 'air.cfg', '-t', '1'], cwd=case,
                           env=dict(os.environ, OMP_DYNAMIC='FALSE'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    if not finished(case) or pressure_residual(case) > -11:
        raise RuntimeError(f'Pressure residual target not reached: {case}')


def prepare(grid, name, re, mesh, solver):
    solid, periodic = (archive(grid, name, re, boundary) for boundary in BOUNDARIES)
    if re == 100 or grid == 'pared':
        if not finished(solid):
            donor_mesh, donor = initial_guess(grid, name, re, 'solido')
            seed = solid/'initial_guess'
            seed_restart(donor, donor_mesh, mesh.with_suffix('.npz'), seed)
            subprocess.run([sys.executable, str(HERE/'su2_caso.py'), '--mesh', str(mesh),
                            '--out', str(solid), '--re', str(re), '--cold', '--fluid-only', '--threads', '1',
                            '--restart', str(seed), '--iterations', '1050', '--cfl', '50', '--su2', str(solver)],
                           check=True)
        converge(solid, solver)
        donor_mesh, donor = initial_guess(grid, name, re, 'periodico_z')
        run_periodic_case(periodic, solid, mesh.with_suffix('.npz'), solver, donor=(donor, donor_mesh))
        needs_retry = pressure_residual(periodic) > -11
        converge(periodic, solver)
        if needs_retry:
            with np.load(mesh.with_suffix('.npz')) as data:
                check = verify_flow(periodic, data['air_xyz'])
            (periodic/'periodic_check.json').write_text(json.dumps(check, indent=2)+'\n', encoding='utf-8')
    for case in (solid, periodic):
        if not finished(case) or pressure_residual(case) > -11:
            raise RuntimeError(f'Incomplete momentum case: {case}')
    assert (solid/'inlet.dat').read_bytes() == (periodic/'inlet.dat').read_bytes()
    def geometry(case):
        return (case/'mesh.su2').read_text(encoding='ascii').split('NMARK=', 1)[0]
    assert geometry(solid) == geometry(periodic)
    print(re, grid, name, 'momentum verified', flush=True)


def solve_thermal(grid, name, re, mesh):
    for boundary in BOUNDARIES:
        case = archive(grid, name, re, boundary)
        for expanded in (False, True):
            out = ROOT / f'Re{re}' / grid / name / boundary / ('expanded' if expanded else 'baseline')
            report = out / 'metrics.csv'
            if report.is_file():
                continue
            if re == 800 and grid != 'pared' and not expanded:
                filename = ('resultados_re800_acotados.csv' if boundary == 'solido'
                            else 'resultados_re800_periodicos.csv') if name == CASES[0] else 'resultados_re800_l60_malla_comun.csv'
                archived = pd.read_csv(HERE/filename)
                matches = archived.loc[archived.caso.eq(name) & archived.malla.eq(grid)]
                if name == CASES[1]:
                    matches = matches.loc[matches.lateral.eq(boundary)]
                assert len(matches) == 1 and not matches.iloc[0].first_ring
                row = matches.iloc[0].to_dict()
                out.mkdir(parents=True, exist_ok=True)
            else:
                row = thermal_case(mesh, float(re), case=case, tolerance=1e-6, first_ring=expanded, archive=out)
            row.update(malla=grid, lateral=boundary, log10_rms_presion=pressure_residual(case))
            staged = report.with_suffix('.tmp')
            pd.DataFrame([row]).to_csv(staged, index=False, encoding='utf-8-sig')
            staged.replace(report)
            print(re, grid, name, boundary, expanded, f"h={row['h_W_m2K']:.8f} dp={row['dp_Pa']:.8f}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--phase', choices=('momentum','thermal','all'), default='all')
    cli.add_argument('--re', type=int, nargs='+', choices=REYNOLDS, default=REYNOLDS)
    cli.add_argument('--grids', nargs='+', choices=GRIDS, default=GRIDS)
    cli.add_argument('--workers', type=int, choices=(1,2,3), default=2)
    cli.add_argument('--su2', type=Path, default=Path(os.environ['TEMP'])/'colector_su2_860/bin/SU2_CFD.exe')
    args = cli.parse_args()
    meshes = {(g,n):mesh_path(g,n) for g in args.grids for n in CASES}
    jobs = [(g,n,r,meshes[g,n]) for g in args.grids for r in reversed(args.re) for n in CASES]
    if args.phase in ('all','momentum'):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda j: prepare(*j,args.su2.resolve()), jobs))
    if args.phase in ('all','thermal'):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda j: solve_thermal(*j), jobs))


if __name__ == '__main__':
    main()
