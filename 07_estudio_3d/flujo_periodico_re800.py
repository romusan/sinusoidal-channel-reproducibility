"""Repeat the Re=800 momentum pairs on the identical grids with transverse periodicity."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd

from flujo_re800 import CASES, GRIDS, HERE, ROOT as SOLID
from flujo_re800 import seed_restart
from su2_caso import options

ROOT = HERE / 'corridas' / 'prueba_re800_periodica'
MESHES = HERE / 'mallas' / 'prueba_re800'


def finished(case):
    log = case / 'su2.log'
    return ((case / 'restart_air.csv').is_file() and log.is_file() and
            'Exit Success' in log.read_text(encoding='utf-8', errors='replace')[-3000:])


def periodic_mesh(solid_mesh, output, xyz, width):
    """Rename the two lateral face sets without changing cells or point IDs."""
    content = solid_mesh.read_text(encoding='ascii')
    marker = re.search(r'^MARKER_TAG=\s*sidewalls\r?\nMARKER_ELEMS=\s*(\d+)\r?\n', content, re.M)
    if marker is None:
        raise ValueError(f'Lateral marker unavailable in {solid_mesh}')
    n = int(marker[1])
    tail = content[marker.end():].splitlines(keepends=True)
    faces = np.array([[int(i) for i in row.split()[1:5]] for row in tail[:n]])
    z = xyz[faces, 2]
    lower = np.all(np.isclose(z, 0., atol=1.e-12), axis=1)
    upper = np.all(np.isclose(z, width, atol=1.e-12), axis=1)
    if not (np.all(lower ^ upper) and lower.sum() == upper.sum()):
        raise ValueError('Side faces do not form a matched periodic pair')
    lo = np.sort(faces[lower], axis=1)
    hi = np.sort(faces[upper], axis=1)
    translated = xyz[hi] - np.array([0., 0., width])
    if not np.allclose(xyz[lo], translated, rtol=0, atol=1.e-12):
        raise ValueError('Periodic face coordinates do not coincide under translation')
    header = re.sub(r'^NMARK=\s*(\d+)', lambda m: f'NMARK= {int(m[1])+1}', content[:marker.start()],
                    flags=re.M, count=1)
    groups = ''.join(f'MARKER_TAG= {tag}\nMARKER_ELEMS= {int(mask.sum())}\n' +
                     ''.join(row for row, use in zip(tail[:n], mask) if use)
                     for tag, mask in (('span_min', lower), ('span_max', upper)))
    output.write_text(header + groups + ''.join(tail[n:]), encoding='ascii')
    return int(lower.sum())


def verify_flow(case, points):
    table = pd.read_csv(case / 'restart_air.csv').sort_values('PointID')
    if len(table) != points.size // 3 or not np.allclose(table[['x','y','z']], points.reshape(-1,3), atol=5e-10):
        raise ValueError(f'Periodic restart coordinates do not match {case.name}')
    fields = table[['Pressure','Velocity_x','Velocity_y','Velocity_z']].to_numpy().reshape((*points.shape[:-1],4))
    delta = np.max(np.abs(fields[:, :, -1] - fields[:, :, 0]), axis=(0,1))
    if not np.isfinite(fields).all() or np.max(delta) > 5e-6:
        raise ValueError(f'Periodic flow mismatch at the two planes: {delta}')
    return {'pressure_Pa': float(delta[0]), 'velocity_m_s': float(delta[1:].max())}


def run_periodic_case(path, control, npz, solver, *, donor=None):
    if finished(path) and (path / 'periodic_check.json').is_file():
        print(path.parent.name, path.name, 'already solved', flush=True)
        return
    tick = perf_counter()
    path.mkdir(parents=True, exist_ok=True)
    meta = json.loads((control / 'case.json').read_text(encoding='utf-8'))
    width = meta['W_m']
    with np.load(npz) as f:
        points = f['air_xyz']
    nfaces = periodic_mesh(control / 'mesh.su2', path / 'mesh.su2', points.reshape(-1,3), width)
    shutil.copyfile(control / 'inlet.dat', path / 'inlet.dat')
    if donor is None:
        shutil.copyfile(control / 'restart_air.csv', path / 'solution_air.csv')
    else:
        previous_case, previous_mesh = donor
        if not finished(previous_case):
            raise RuntimeError(f'Periodic initial field is missing: {previous_case}')
        seed = path / 'initial_guess'
        seed_restart(previous_case, previous_mesh, npz, seed)
        shutil.copyfile(seed / 'restart_air.csv', path / 'solution_air.csv')
    cfg = dict(tuple(value.strip() for value in line.split('=', 1))
               for line in (control / 'air.cfg').read_text(encoding='ascii').splitlines()
               if '=' in line and not line.lstrip().startswith('%'))
    cfg.update(RESTART_SOL='YES',
               MARKER_HEATFLUX=re.sub(r'\bsidewalls\s*,\s*0\s*,\s*', '', cfg['MARKER_HEATFLUX']),
               MARKER_PERIODIC=f'(span_min, span_max, 0., 0., 0., 0., 0., 0., 0., 0., {width:.15g})')
    if 'sidewalls' in cfg['MARKER_HEATFLUX']:
        raise ValueError('No-slip lateral marker still present in the periodic configuration')
    options(path / 'air.cfg', cfg)
    meta.update(mesh_sha256=hashlib.sha256((path/'mesh.su2').read_bytes()).hexdigest(),
                lateral='periodico_z', periodic_span_m=width, periodic_faces_per_side=nfaces,
                control=str(control.relative_to(HERE)))
    (path / 'case.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE')
    print(path.parent.name, path.name, f'SU2 periodic with {nfaces} faces per side', flush=True)
    with (path / 'su2.log').open('w', encoding='utf-8') as log:
        subprocess.run([str(solver.resolve()), 'air.cfg', '-t', '1'], cwd=path, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    check = verify_flow(path, points)
    (path/'periodic_check.json').write_text(json.dumps(check, indent=2)+'\n', encoding='utf-8')
    print(path.parent.name, path.name, f'{perf_counter()-tick:.1f} s', check, flush=True)


def run_case(grid, name, solver):
    path = ROOT / grid / (name + '_Re800')
    mesh = MESHES / grid / (name + '.npz')
    index = GRIDS.index(grid)
    donor = None
    if index:
        previous = GRIDS[index-1]
        donor = ROOT / previous / path.name, MESHES / previous / mesh.name
    run_periodic_case(path, SOLID / grid / path.name, mesh, solver, donor=donor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mallas', nargs='+', choices=GRIDS, default=list(GRIDS))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--su2', type=Path,
                        default=Path(os.environ['TEMP'])/'colector_su2_860/bin/SU2_CFD.exe')
    args = parser.parse_args()
    for grid in args.mallas:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda name: run_case(grid, name, args.su2), CASES))


if __name__ == '__main__':
    main()
