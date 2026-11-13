"""Conservative Q1 heat transport with a posteriori temperature bound checks."""

import argparse
import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import LinearOperator, lgmres, splu

from energia_q1 import assemble, thermo_metrics
from flujo_re800 import CASES, GRIDS, HERE, ROOT
from flujo_periodico_re800 import ROOT as PERIODIC_ROOT
from medir_su2 import fields, load_meta, measure, sectional
from referencia_rectangular import rectangular_velocity_profile
from su2_caso import DREF, MU, QPROJ, RHO, TIN, weights


def solid_interior_ids(grid):
    sx, sy, sz = grid['solid_ids'].shape
    return grid['air_ids'].size + np.arange(sx*(sy-1)*sz).reshape(sx, sy-1, sz)


def identify_span_nodes(grid, n, graetz):
    """Build the map from full Q1 nodes to unique transverse-periodic nodes."""
    ids = [grid['air_ids']]
    if not graetz:
        ids.append(solid_interior_ids(grid))
    mapping = np.arange(n)
    for material in ids:
        mapping[material[..., -1]] = material[..., 0]
    _, index = np.unique(mapping, return_inverse=True)
    return index


def axial_blocks(grid, free, graetz, identification=None):
    air = grid['air_ids']
    planes = [plane.ravel() for plane in air]
    if not graetz:
        for i, solid_plane in zip(grid['heated_i'], solid_interior_ids(grid)):
            planes[i] = np.r_[planes[i], solid_plane.ravel()]
    if identification is not None:
        planes = [np.unique(identification[p]) for p in planes]
    reverse = np.full(sum(map(len, planes)), -1)
    reverse[free] = np.arange(len(free))
    return [ids[ids >= 0] for plane in planes if np.any((ids := reverse[plane]) >= 0)]


def solve_axial_blocks(matrix, rhs, planes):
    """Block Thomas elimination in the physical streamwise order.

    A plane contains air and plate nodes at the same x. Hexahedra, their
    boundary facets, and graph corrections connect at most adjacent planes.
    """
    matrix = matrix.tocsr()
    plane_index = np.empty(len(rhs), dtype=int)
    for i, ids in enumerate(planes):
        plane_index[ids] = i
    coo = matrix.tocoo(copy=False)
    if np.any(np.abs(plane_index[coo.row] - plane_index[coo.col]) > 1):
        raise ValueError('The assembled matrix is not block tridiagonal in x')
    forward = []
    for i, ids in enumerate(planes):
        rows = matrix[ids]
        diagonal = rows[:, ids].toarray()
        load = rhs[ids].copy()
        if i:
            lower = rows[:, planes[i-1]]
            reduced_load, transfer = forward[-1]
            diagonal -= lower @ transfer
            load -= lower @ reduced_load
        upper = rows[:, planes[i+1]].toarray() if i+1 < len(planes) else np.empty((len(ids), 0))
        block = np.asfortranarray(np.column_stack((load, upper)))
        lu = lu_factor(np.asfortranarray(diagonal), overwrite_a=True, check_finite=False)
        reduced = lu_solve(lu, block, overwrite_b=True, check_finite=False)
        forward.append((reduced[:, 0], reduced[:, 1:]))
    answer = np.empty_like(rhs)
    for i in range(len(planes)-1, -1, -1):
        load, transfer = forward[i]
        answer[planes[i]] = load if i+1 == len(planes) else load - transfer @ answer[planes[i+1]]
    residual = np.linalg.norm(matrix @ answer - rhs) / np.linalg.norm(rhs)
    if not np.isfinite(residual) or residual > 1e-8:
        raise RuntimeError(f'Axial block solver relative residual {residual:.3e}')
    return answer


def correct(meta, grid, velocity, *, graetz=False, tolerance=1e-6, first_ring=False, periodic=False):
    """Cancel positive off-diagonals near inadmissible nodes, then re-solve.

    No temperatures are clipped. Edge diffusion is symmetric and has zero
    row and column sums. The admissibility check is on the final field and
    applies only to the source-free air and the heated, insulated solid here.
    """
    A, rhs, fixed, free, scale = assemble(meta, grid, velocity, graetz)
    nf = grid['air_ids'].size
    boundary = grid['air_ids'][grid['heated_i'], 0, :].ravel()
    identification = None
    if periodic:
        identification = identify_span_nodes(grid, len(rhs), graetz)
        nperiodic = int(identification.max()) + 1
        expand = coo_matrix((np.ones(len(rhs)), (np.arange(len(rhs)), identification)),
                            shape=(len(rhs), nperiodic)).tocsr()
        load = float(rhs.sum())
        A = (expand.T @ A @ expand).tocsr()
        rhs = expand.T @ rhs
        if not np.isclose(rhs.sum(), load, rtol=1e-13):
            raise RuntimeError('Periodic condensation changed the applied heat')
        fixed = np.unique(identification[fixed])
        free = np.setdiff1d(np.arange(nperiodic), fixed)
        boundary = np.unique(identification[boundary])
        nf = int(identification[:nf].max()) + 1
    positive = A.maximum(A.T).maximum(0).tocsr()
    positive.setdiag(0.)
    positive.eliminate_zeros()
    edges = positive.tocoo()
    active = np.zeros(A.shape[0], dtype=bool)
    theta = np.zeros_like(rhs)
    strength = 0.
    factor = None
    planes = axial_blocks(grid, free, graetz, identification) if len(free) >= 70000 else None
    if graetz and planes is not None:
        # The advective outlet term provides a better starting block when the
        # Graetz operator omits axial conduction.
        planes.reverse()
    for step in range(21):
        if active.any():
            selected = active[edges.row] | active[edges.col]
            added = coo_matrix((edges.data[selected], (edges.row[selected], edges.col[selected])), shape=A.shape).tocsr()
            diagonal = np.asarray(added.sum(axis=1)).ravel()
            strength = diagonal.sum()
            system = A + diags(diagonal) - added
        else:
            system = A
        reduced = system[free][:, free].tocsc()
        status = 1
        if planes is not None:
            print(f'  solving axial blocks step={step} n={len(free)}', flush=True)
            solution = solve_axial_blocks(reduced, rhs[free], planes)
            status = 0
        elif factor is not None and step >= 2:
            operator = LinearOperator(reduced.shape, factor.solve)
            solution, status = lgmres(reduced, rhs[free], x0=theta[free], M=operator,
                                      rtol=1e-10, atol=1e-12, inner_m=14, maxiter=6)
            del operator
        if status != 0:
            factor = None
            print(f'  factoring step={step} n={len(free)}', flush=True)
            factor = splu(reduced, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=.05)
            solution = factor.solve(rhs[free])
        theta[free] = solution
        del reduced
        temperature = TIN + scale * theta
        maximum_wall = max(TIN, temperature[boundary].max())
        bad = temperature < TIN - tolerance
        bad[:nf] |= temperature[:nf] > maximum_wall + tolerance
        print(f'  step={step} marked={active.sum()} violations={bad.sum()} '
              f'min={temperature.min():.8f} max_air-wall={temperature[:nf].max()-maximum_wall:.2e}', flush=True)
        if not bad.any():
            break
        if step == 20:
            raise RuntimeError('Final thermal field failed the physical bounds')
        new = active | bad
        if first_ring or np.array_equal(new, active):
            new |= (positive @ bad.astype(float)) > 0
        if np.array_equal(new, active):
            new |= (positive @ active.astype(float)) > 0
        if step == 18 or np.array_equal(new, active):
            new[:] = True
        active = new
    residual = system @ theta - rhs
    H = meta['mesh']['dimensiones_m']['H']
    q_inlet = -residual[fixed].sum() * QPROJ * H * H
    stats = {'Q1_residual_relative': np.linalg.norm(residual[free]) / np.linalg.norm(rhs[free]),
             'correction_passes': step, 'marked_nodes': int(active.sum()), 'total_nodes': len(theta),
             'graph_strength': float(strength), 'bound_tolerance_K': tolerance,
             'max_interface_K': float(maximum_wall), 'max_air_minus_wall_K': float(temperature[:nf].max()-maximum_wall),
             'min_temperature_K': float(temperature.min())}
    if identification is not None:
        temperature = temperature[identification]
    return temperature, q_inlet, stats


def thermal_case(mesh, re, *, case=None, tolerance, first_ring, archive=None):
    graetz = case is None
    spec = json.loads(mesh.with_suffix('.json').read_text(encoding='utf-8'))
    geo = spec['dimensiones_m']
    with np.load(mesh.with_suffix('.npz')) as npz:
        grid = {name: npz[name] for name in npz.files}
    xyz = grid['air_xyz'].reshape(-1, 3)
    x, y, z = xyz.T
    if graetz:
        U = re * MU / (RHO * DREF)
        meta = {'mesh': spec, 'U_m_s': U}
        ux = rectangular_velocity_profile(y, z, geo['H'], geo['W'], U).reshape(grid['air_ids'].shape)
        w = weights(grid['y_air'][0])[:, None] * weights(grid['z'])
        ux *= U * geo['W'] * geo['H'] / np.sum(w * ux[0])
        table = pd.DataFrame(dict(x=x, y=y, z=z, velocityx=ux.ravel(), velocityy=0., velocityz=0., pressure=0.))
        hyd = {}
    else:
        meta = load_meta(case)
        table = fields(case/'restart_air.csv').sort_values('pointid')
        if not np.allclose(table[['x','y','z']].to_numpy(), xyz, rtol=1e-6, atol=5e-10):
            raise ValueError('Velocity coordinates do not match the thermal mesh')
        hyd = measure(case)
    vel = table[['velocityx','velocityy','velocityz']].to_numpy()
    periodic = meta.get('lateral') == 'periodico_z'
    temp, q_inlet, stats = correct(meta, grid, vel, graetz=graetz, tolerance=tolerance,
                                   first_ring=first_ring, periodic=periodic)
    if periodic:
        air_t = temp[:len(xyz)].reshape(grid['air_ids'].shape)
        solid_t = temp[len(xyz):].reshape(solid_interior_ids(grid).shape)
        flow_check = json.loads((case/'periodic_check.json').read_text(encoding='utf-8'))
        stats.update(lateral='periodico_z',
                     periodic_temperature_air_K=float(np.max(np.abs(air_t[...,0]-air_t[...,-1]))),
                     periodic_temperature_solid_K=float(np.max(np.abs(solid_t[...,0]-solid_t[...,-1]))),
                     periodic_pressure_Pa=flow_check['pressure_Pa'],
                     periodic_velocity_m_s=flow_check['velocity_m_s'])
        hyd.update(dp_rect_exacto_Pa=np.nan, error_dp_rect_rel=np.nan)
    table['temperature'] = temp[:len(xyz)]
    section, _ = sectional(xyz, table)
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)
        section.to_csv(archive/'secciones.csv', index=False)
        np.savez_compressed(archive/'temperaturas.npz', air_temperature_K=temp[:len(xyz)].reshape(grid['air_ids'].shape),
                            solid_noninterface_K=temp[len(xyz):])
    metrics = thermo_metrics(meta, section, geo, temp[:len(xyz)], q_inlet, stats['Q1_residual_relative'], 0.)
    return {**hyd, **metrics, **stats, 'caso': mesh.stem, 'Re_ref': re, 'celdas_aire': spec['celdas_aire'],
            'celdas_placa': 0 if graetz else spec['celdas_placa'], 'graetz': graetz, 'first_ring': first_ring}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--mallas', nargs='+', default=list(GRIDS))
    cli.add_argument('--casos', nargs='+', default=list(CASES))
    cli.add_argument('--graetz', action='store_true')
    cli.add_argument('--periodic', action='store_true', help='Read the transverse-periodic momentum cases')
    cli.add_argument('--tolerance', type=float, default=1e-6)
    cli.add_argument('--first-ring', action='store_true')
    cli.add_argument('--wait', type=float, default=0., help='Seconds to wait for each momentum case')
    cli.add_argument('--out', required=True, type=Path)
    args = cli.parse_args()
    if args.periodic and args.graetz:
        cli.error('The archived rectangular Graetz profile uses solid lateral walls')
    root = PERIODIC_ROOT if args.periodic else ROOT
    rows = []
    for grid in args.mallas:
        for name in (('H15_Plana',) if args.graetz else args.casos):
            mesh = HERE/'mallas'/'prueba_re800'/grid/(name + '.su2')
            case = None if args.graetz else root/grid/(name + '_Re800')
            if case:
                log = case/'su2.log'
                deadline = perf_counter() + args.wait
                while not (log.is_file() and (case/'restart_air.csv').is_file() and
                           (not args.periodic or (case/'periodic_check.json').is_file()) and
                           'Exit Success' in log.read_text(encoding='utf-8', errors='replace')[-3000:]):
                    if perf_counter() >= deadline:
                        raise RuntimeError(f'Unfinished momentum solve in {case}')
                    sleep(5)
            archive = None if args.graetz or args.first_ring else case/'termico_acotado'
            tick = perf_counter()
            print(grid, name, 'Graetz' if args.graetz else 'CHT', flush=True)
            row = thermal_case(mesh, 800., case=case, tolerance=args.tolerance,
                               first_ring=args.first_ring, archive=archive)
            rows.append(dict(malla=grid, **row))
            pd.DataFrame(rows).to_csv(args.out, index=False, encoding='utf-8-sig')
            print(f"h={row['h_W_m2K']:.7f} balance={row['cierre_energia_rel']:.2e} "
                  f"residual={row['Q1_residual_relative']:.2e} elapsed={perf_counter()-tick:.1f}s", flush=True)


if __name__ == '__main__':
    main()
