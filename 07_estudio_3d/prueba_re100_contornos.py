"""Compare solid and transverse-periodic sides for the 60 mm wave."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

# Limit the thermal linear algebra before importing NumPy or SciPy.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import pandas as pd

from energia_acotada import thermal_case
from flujo_periodico_re800 import finished, run_periodic_case
from flujo_re800 import HERE, seed_restart


GRIDS = {'gruesa': 'gruesas', 'media': 'campana', 'fina': 'refinadas'}
CASES = ('H15_Plana', 'H15_A150_L60')


def output_root(reynolds):
    label = 'prueba_re100_periodica' if reynolds == 100 else 'prueba_re800_l60'
    return HERE / 'corridas' / label


def paths(grid, name, reynolds):
    family = GRIDS[grid]
    case = f'{name}_Re{reynolds}'
    root = output_root(reynolds)
    solid = HERE / 'corridas' / family / case
    if reynolds == 800 and grid != 'media':
        solid = root / grid / (case + '_solido')
    return HERE / 'mallas' / family / (name + '.su2'), solid, root / grid / case


def pressure_residual(case):
    history = pd.read_csv(case / 'history_air.csv')
    key = next(c for c in history if 'rms[P]' in c)
    return float(history[key].iloc[-1])


def prepare(grid, name, solver, reynolds):
    mesh, solid, periodic = paths(grid, name, reynolds)
    if reynolds == 800 and grid != 'media' and not finished(solid):
        archived = HERE/'corridas'/'campana'/f'{name}_Re800'
        if not finished(archived) or pressure_residual(archived) > -11.:
            raise RuntimeError(f'Incomplete momentum seed: {archived}')
        seed = solid/'initial_guess'
        seed_restart(archived, HERE/'mallas'/'campana'/(name+'.npz'), mesh.with_suffix('.npz'), seed)
        subprocess.run([sys.executable, str(HERE/'su2_caso.py'), '--mesh', str(mesh),
                        '--out', str(solid), '--re', str(reynolds), '--cold', '--fluid-only',
                        '--threads', '1', '--restart', str(seed), '--iterations', '700',
                        '--cfl', '50', '--su2', str(solver.resolve())], check=True)
    if not finished(solid) or pressure_residual(solid) > -11.:
        raise RuntimeError(f'Incomplete archived solid control: {solid}')
    run_periodic_case(periodic, solid, mesh.with_suffix('.npz'), solver)
    if pressure_residual(periodic) > -11.:
        raise RuntimeError(f'Periodic momentum convergence not reached: {periodic}')
    if (solid / 'inlet.dat').read_bytes() != (periodic / 'inlet.dat').read_bytes():
        raise ValueError(f'Inlet differs for {grid} / {name}')
    def geometry(case):
        return (case / 'mesh.su2').read_text(encoding='ascii').split('NMARK=', 1)[0]
    if geometry(solid) != geometry(periodic):
        raise ValueError(f'Mesh connectivity or coordinates differ for {grid} / {name}')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--mallas', nargs='+', choices=GRIDS, default=list(GRIDS))
    cli.add_argument('--re', type=int, choices=(100, 800), default=100)
    cli.add_argument('--workers', type=int, choices=(1, 2), default=2)
    cli.add_argument('--first-ring', action='store_true')
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--su2', type=Path,
                     default=Path(os.environ['TEMP'])/'colector_su2_860/bin/SU2_CFD.exe')
    args = cli.parse_args()
    rows = []
    for grid in args.mallas:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda name: prepare(grid, name, args.su2, args.re), CASES))
        for name in CASES:
            mesh, solid, periodic = paths(grid, name, args.re)
            for boundary, case in [('solido', solid), ('periodico_z', periodic)]:
                archive = periodic / ('termico_' + boundary)
                if args.first_ring:
                    archive = archive.with_name(archive.name + '_first_ring')
                print(grid, name, boundary, flush=True)
                row = thermal_case(mesh, float(args.re), case=case, tolerance=1.e-6,
                                   first_ring=args.first_ring, archive=archive)
                row.update(malla=grid, lateral=boundary, log10_rms_presion=pressure_residual(case))
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out, index=False, encoding='utf-8-sig')
                print(f"h={row['h_W_m2K']:.8f} dp={row['dp_Pa']:.8f} "
                      f"energy={row['cierre_energia_rel']:.2e}", flush=True)


if __name__ == '__main__':
    main()
