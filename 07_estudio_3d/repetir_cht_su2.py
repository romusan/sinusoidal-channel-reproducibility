"""Repeat a native SU2 heat-transfer case and audit the port heat rates."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
import subprocess
import sys

import numpy as np
import pandas as pd

from medir_su2 import fields, measure, sectional
from su2_caso import CP, K, RHO, TIN, options, read_su2, weights
from verificar_cht_su2 import CASES, configuration, seed


HERE = Path(__file__).resolve().parent


def port_heat_from_history(row):
    return CP * sum(-row[f'Avg_Massflow({name})'] * (row[f'Avg_Temp({name})'] - TIN)
                    for name in ('inlet', 'outlet'))


def heat_rates(case):
    air, solid = read_su2(case / 'mesh.su2')
    coords = air['points']
    flow = fields(case / 'restart_air_0.csv').sort_values('pointid')
    section, (ux, _, _, temperature) = sectional(coords, flow)
    xyz = coords.reshape(ux.shape + (3,))
    x = xyz[:, 0, 0, 0]
    z_weights = weights(xyz[0, 0, :, 2])
    port_weights = weights(xyz[0, :, 0, 1])[:, None] * z_weights
    power = RHO * CP * np.sum(port_weights * (ux[-1] * (temperature[-1] - TIN) -
                                              ux[0] * (temperature[0] - TIN)))
    gradient = ((temperature[1]-temperature[0])/(x[1]-x[0]) -
                (temperature[-1]-temperature[-2])/(x[-1]-x[-2]))
    solid_t = fields(case / 'restart_solid_1.csv').sort_values('pointid').temperature.to_numpy()
    fluid_ids = np.unique(air['markers']['interface_air'])
    solid_ids = np.unique(solid['markers']['interface_solid'])
    np.testing.assert_allclose(coords[fluid_ids], solid['points'][solid_ids], atol=1e-12, rtol=0)
    # The finite-volume transfer uses the first off-wall point. Retain the
    # projected vector face areas and sum them at each boundary vertex.
    points = air['markers']['interface_air']
    quad = coords[points]
    vector_area = 0.5 * np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1])
    vertex_area = np.zeros_like(coords)
    np.add.at(vertex_area, points.ravel(), np.repeat(vector_area / 4, 4, axis=0))
    ids = np.arange(len(coords)).reshape(ux.shape)
    wall = ids[:, 0, :].ravel()
    next_normal = ids[:, 1, :].ravel()
    first_layer = np.linalg.norm(coords[wall] - coords[next_normal], axis=1)
    gradient_flux = K * (temperature.ravel()[wall] - temperature.ravel()[next_normal]) / first_layer
    q_interface = np.dot(gradient_flux, np.linalg.norm(vertex_area[wall], axis=1))
    rates = {'Q_port_nodal_W': power, 'Q_port_gradient_estimate_W': K * np.sum(port_weights * gradient),
             'Q_interface_air_normal_estimate_W': q_interface,
             'interface_temperature_jump_K': np.max(np.abs(temperature.ravel()[fluid_ids] - solid_t[solid_ids])),
             'inlet_bulk_K': section.bulk_K.iloc[0]}
    history = pd.read_csv(case / 'history_air_0.csv', skipinitialspace=True)
    history.columns = history.columns.str.strip()
    if 'Avg_Massflow(outlet)' in history:
        rates['Q_port_SU2_history_W'] = port_heat_from_history(history.iloc[-1])
    return rates


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--su2', required=True, type=Path)
    cli.add_argument('--case', choices=CASES, required=True)
    cli.add_argument('--re', type=int, choices=(100, 800), default=800)
    cli.add_argument('--outer', type=int, default=1200)
    cli.add_argument('--coupling', choices=('NEUMANN', 'ROBIN'), default='NEUMANN')
    cli.add_argument('--first-order', action='store_true')
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--measure-only', action='store_true')
    args = cli.parse_args()
    case = args.out.resolve()
    if not args.measure_only:
        mesh = HERE / 'mallas/prueba_re800/gruesa' / f'{args.case}.su2'
        thermal = HERE / 'corridas/prueba_l60_refinamiento_pared' / f'Re{args.re}/gruesa' / args.case / 'solido/baseline'
        subprocess.run([sys.executable, str(HERE / 'su2_caso.py'), '--mesh', str(mesh), '--out', str(case),
                        '--re', str(args.re), '--prepare-only', '--iterations', str(args.outer),
                        '--inner', '8', '--solid-inner', '8', '--cfl', '75', '--linear', '35'], check=True)
        previous = HERE / 'corridas/verificacion_cht_su2' / args.case
        if args.re == 800 and (previous / 'restart_air_0.csv').is_file():
            for name in ('air_0', 'solid_1'):
                copyfile(previous / f'restart_{name}.csv', case / f'solution_{name}.csv')
        else:
            donor = HERE / 'corridas' / CASES[args.case] if args.re == 800 else thermal.parent
            with np.load(mesh.with_suffix('.npz')) as grid:
                seed(case, donor, thermal, grid)
        changes = {'cht.cfg': {'CHT_COUPLING_METHOD': f'DIRECT_TEMPERATURE_{args.coupling}_HEATFLUX',
                               'CONV_FIELD': '(BGS_ENTHALPY[0], BGS_TEMPERATURE[1])',
                               'CONV_RESIDUAL_MINVAL': '-9', 'HISTORY_OUTPUT': '(OUTER_ITER, BGS_RES[0], BGS_RES[1])',
                               'WRT_ZONE_HIST': 'YES', 'OUTPUT_WRT_FREQ': '300'},
                   'air.cfg': {'MARKER_ANALYZE': '(inlet, outlet)',
                               'HISTORY_OUTPUT': '(ITER, RMS_RES, FLOW_COEFF_SURF)'},
                   'solid.cfg': {'CFL_NUMBER': '100000', 'MARKER_MONITORING': '(interface_solid)',
                                 'HISTORY_OUTPUT': '(ITER, RMS_RES, HEAT)'}}
        if args.first_order:
            changes['air.cfg']['MUSCL_FLOW'] = 'NO'
        for name, update in changes.items():
            path = case / name
            options(path, {**configuration(path), 'RESTART_SOL': 'YES', **update})
        with (case / 'su2.log').open('w') as log:
            subprocess.run([str(args.su2.resolve()), 'cht.cfg', '-t', '1'], cwd=case,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    row = measure(case) | heat_rates(case)
    history = pd.read_csv(case / 'cht.csv', skipinitialspace=True)
    history.columns = history.columns.str.strip()
    row.update(outer_iterations=int(history.Outer_Iter.iloc[-1])+1,
               log_enthalpy_coupling_residual=history['bgs[h][0]'].iloc[-1],
               log_solid_temperature_coupling_residual=history['bgs[T][1]'].iloc[-1],
               port_heat_deviation_pct=100 * (row['Q_port_nodal_W'] / row['Q_input_W'] - 1),
               inlet_temperature_undershoot_K=TIN - row['T_aire_min_K'])
    pd.DataFrame([row]).to_csv(case / 'balance.csv', index=False)
    print(json.dumps({k: row[k] for k in ('h_W_m2K', 'Q_aire_W', 'Q_port_nodal_W',
                                         'interface_temperature_jump_K', 'outer_iterations',
                                         'log_enthalpy_coupling_residual')}, default=float), flush=True)


if __name__ == '__main__':
    main()
