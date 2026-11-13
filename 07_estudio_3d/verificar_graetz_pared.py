"""Compare the common fine and wall-refined flat meshes with the modal Graetz solution."""

import argparse
import os
from pathlib import Path
from time import perf_counter

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import pandas as pd

from energia_acotada import thermal_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reynolds', nargs='+', type=int, choices=(100, 800), default=(800,))
    parser.add_argument('--grids', nargs='+', choices=('pared', 'fina'), default=('pared',))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    reference = pd.read_csv(root / 'benchmark_rectangular_3d.csv').set_index('Re_ref')
    rows = []
    for re in args.reynolds:
        for grid in args.grids:
            mesh = root / 'mallas/prueba_re800' / grid / 'H15_Plana.su2'
            for expanded in (False, True):
                start = perf_counter()
                row = thermal_case(mesh, re, tolerance=1e-6, first_ring=expanded)
                h_ref = reference.loc[re, 'h_Graetz_3D_W_m2K']
                error = 100 * (row['h_W_m2K'] / h_ref - 1)
                row.update(mesh=grid, modal_h_W_m2K=h_ref, error_h_pct=error)
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out or root / 'graetz_malla_pared.csv', index=False)
                print(re, grid, expanded, row['h_W_m2K'], error, round(perf_counter()-start, 1), flush=True)


if __name__ == '__main__':
    main()
