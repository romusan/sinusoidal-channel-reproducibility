"""Solve the Re=800 momentum pairs with the archived fields as initial guesses."""

import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
GRIDS = ('gruesa', 'media', 'fina')
CASES = ('H15_Plana', 'H15_A150_L30')
ROOT = HERE / 'corridas' / 'prueba_re800'


def computational_coordinates(grid):
    y = grid['y_air'][0]
    return grid['x'], (y - y[0]) / (y[-1] - y[0]), grid['z']


def seed_restart(case, source, target, out):
    fields = pd.read_csv(case / 'restart_air.csv').sort_values('PointID')
    names = ['Pressure', 'Velocity_x', 'Velocity_y', 'Velocity_z']
    with np.load(source) as old, np.load(target) as new:
        interp = RegularGridInterpolator(computational_coordinates(old),
                                        fields[names].to_numpy().reshape((*old['air_ids'].shape, 4)),
                                        bounds_error=False, fill_value=None)
        points = np.stack(np.meshgrid(*computational_coordinates(new), indexing='ij'), axis=-1)
        xyz = new['air_xyz'].reshape(-1, 3)
        values = interp(points).reshape(-1, 4)
    table = pd.DataFrame(np.column_stack((xyz, values)), columns=['x', 'y', 'z', *names])
    table.insert(0, 'PointID', np.arange(len(table)))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'restart_air.csv', index=False, float_format='%.15e')


def main():
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for grid, iterations in zip(GRIDS, (550, 600, 690)):
        for name in CASES:
            path = ROOT / grid / (name + '_Re800')
            log = path / 'su2.log'
            if (path / 'restart_air.csv').is_file() and log.is_file() and 'Exit Success' in log.read_text(encoding='utf-8', errors='replace')[-3000:]:
                print(grid, name, 'already solved', flush=True)
                continue
            mesh = HERE / 'mallas' / 'prueba_re800' / grid / (name + '.su2')
            old = HERE / 'mallas' / 'campana' / (name + '.npz')
            archive = HERE / 'corridas' / 'campana' / (name + '_Re800')
            seed = path / 'initial_guess'
            seed_restart(archive, old, mesh.with_suffix('.npz'), seed)
            begin = perf_counter()
            subprocess.run([sys.executable, str(HERE / 'su2_caso.py'), '--mesh', str(mesh),
                            '--out', str(path), '--re', '800', '--cold', '--fluid-only', '--threads', '1',
                            '--restart', str(seed), '--iterations', str(iterations), '--cfl', '50'],
                           check=True, env=env)
            print(grid, name, f'{perf_counter()-begin:.1f} s', flush=True)


if __name__ == '__main__':
    main()
