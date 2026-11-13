"""Apply 9 W directly to the flat air channel as a control for native conjugate coupling."""

import argparse
from pathlib import Path
from shutil import copyfile
import subprocess

import pandas as pd

from repetir_cht_su2 import configuration, port_heat_from_history
from su2_caso import options


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--su2', type=Path, required=True)
    cli.add_argument('--donor', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.donor / 'mesh.su2').open() as source, (args.out / 'mesh.su2').open('w') as dest:
        for line in source:
            if line.startswith('IZONE= 2'):
                break
            if not line.startswith(('NZONE', 'IZONE')):
                dest.write(line)
    copyfile(args.donor / 'inlet.dat', args.out / 'inlet.dat')
    copyfile(args.donor / 'restart_air_0.csv', args.out / 'solution_air.csv')
    values = configuration(args.donor / 'air.cfg')
    values.update(MARKER_HEATFLUX='(roof, 0, sidewalls, 0, floor_adiabatic, 0, interface_air, 100)',
                  INNER_ITER=600, CONV_FIELD='(RMS_ENTHALPY)', CONV_RESIDUAL_MINVAL=-10,
                  MARKER_ANALYZE='(inlet, outlet)', HISTORY_OUTPUT='(ITER, RMS_RES, FLOW_COEFF_SURF)')
    options(args.out / 'air.cfg', values)
    with (args.out / 'su2.log').open('w') as log:
        subprocess.run([str(args.su2.resolve()), 'air.cfg', '-t', '1'], cwd=args.out,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    history = pd.read_csv(args.out / 'history_air.csv', skipinitialspace=True)
    history.columns = history.columns.str.strip()
    print(f'Port heat = {port_heat_from_history(history.iloc[-1]):.8f} W', flush=True)


if __name__ == '__main__':
    main()
