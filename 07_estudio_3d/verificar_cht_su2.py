"""Cross-check the bounded Q1 pair at Re=800 with native SU2 CHT on the common coarse mesh."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from medir_su2 import measure
from su2_caso import CP, options


HERE = Path(__file__).resolve().parent
CASES = {
    'H15_Plana': 'prueba_re800/gruesa/H15_Plana_Re800',
    'H15_A150_L60': 'prueba_re800_l60_malla_comun/gruesa/solido',
}


def configuration(path):
    return dict(tuple(part.strip() for part in line.split('=', 1))
                for line in path.read_text(encoding='ascii').splitlines() if '=' in line)


def seed(case, donor, thermal, grid):
    air = pd.read_csv(donor / 'restart_air.csv').sort_values('PointID')
    np.testing.assert_allclose(air[['x', 'y', 'z']], grid['air_xyz'].reshape(-1, 3), atol=1e-12, rtol=0)
    with np.load(thermal / 'temperaturas.npz') as temperatures:
        air_t = temperatures['air_temperature_K']
        solid_shape = grid['solid_xyz'].shape[:-1]
        solid_t = np.empty(solid_shape)
        solid_t[:, :-1, :] = temperatures['solid_noninterface_K'].reshape(solid_shape[0], -1, solid_shape[2])
        solid_t[:, -1, :] = air_t[grid['heated_i'], 0, :]
    air['Enthalpy'] = CP * (air_t.ravel() - 298.15)
    air.to_csv(case / 'solution_air_0.csv', index=False)
    solid = pd.DataFrame(grid['solid_xyz'].reshape(-1, 3), columns=['x', 'y', 'z'])
    solid.insert(0, 'PointID', np.arange(len(solid)))
    solid['Temperature'] = solid_t.ravel()
    solid.to_csv(case / 'solution_solid_1.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--su2', required=True, type=Path)
    parser.add_argument('--outer', type=int, default=500)
    args = parser.parse_args()
    target = HERE / 'corridas/verificacion_cht_su2'
    rows = []
    for name, path in CASES.items():
        mesh = HERE / 'mallas/prueba_re800/gruesa' / (name + '.su2')
        case = target / name
        donor = HERE / 'corridas' / path
        thermal = HERE / 'corridas/prueba_l60_refinamiento_pared/Re800/gruesa' / name / 'solido/baseline'
        subprocess.run([sys.executable, str(HERE/'su2_caso.py'), '--mesh', str(mesh),
                        '--out', str(case), '--re', '800', '--prepare-only', '--iterations', str(args.outer),
                        '--inner', '4', '--solid-inner', '15', '--cfl', '25', '--linear', '20'], check=True)
        with np.load(mesh.with_suffix('.npz')) as grid:
            seed(case, donor, thermal, grid)
        values = {'cht.cfg': {'CONV_FIELD': '(BGS_ENTHALPY[0], BGS_TEMPERATURE[1])',
                              'HISTORY_OUTPUT': '(OUTER_ITER, BGS_RES[0], BGS_RES[1])',
                              'CONV_RESIDUAL_MINVAL': '-9', 'WRT_ZONE_HIST': 'YES'},
                  'air.cfg': {}, 'solid.cfg': {'CFL_NUMBER': '1000000'}}
        for file, updates in values.items():
            cfg = case / file
            options(cfg, {**configuration(cfg), 'RESTART_SOL': 'YES', **updates})
        with (case / 'su2.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(args.su2.resolve()), 'cht.cfg', '-t', '1'], cwd=case,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        row = measure(case)
        log = (case/'su2.log').read_text(encoding='utf-8', errors='replace')
        row['outer_limit_reached'] = 'before convergence' in log
        history = pd.read_csv(case / 'cht.csv', skipinitialspace=True)
        history.columns = history.columns.str.strip()
        row['log_enthalpy_coupling_residual'] = history['bgs[h][0]'].iloc[-1]
        row['log_solid_temperature_coupling_residual'] = history['bgs[T][1]'].iloc[-1]
        rows.append(row)
        pd.DataFrame(rows).to_csv(HERE/'verificacion_cht_su2.csv', index=False)
        print(name, json.dumps({key: row[key] for key in ('h_W_m2K', 'cierre_energia_rel',
                                                           'T_aire_min_K', 'outer_limit_reached')}), flush=True)


if __name__ == '__main__':
    main()
