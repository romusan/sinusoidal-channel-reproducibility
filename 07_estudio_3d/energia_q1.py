"""Conservative conjugate heat transfer on the full 3D mesh with a given velocity."""

import argparse
import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, gmres, splu
from scipy.sparse import coo_matrix, diags
from skfem import Basis, BilinearForm, ElementHex1, FacetBasis, LinearForm, MeshHex, asm
from skfem.helpers import dot, grad

from medir_su2 import average_difference, load_meta, measure, restart, sectional
from referencia_rectangular import rectangular_velocity_profile
from su2_caso import CP, DREF, K, MU, QPROJ, RHO, TIN, KS, weights


HERE = Path(__file__).resolve().parent


@BilinearForm
def diffusion(u, v, w):
    return w.ratio * (w.axial * grad(u)[0] * grad(v)[0] +
                      grad(u)[1] * grad(v)[1] + grad(u)[2] * grad(v)[2])


@BilinearForm
def transport(u, v, w):
    return -w.pe * u * dot(w.velocity, grad(v))


@BilinearForm
def streamline(u, v, w):
    return w.diffusion * dot(w.velocity, grad(u)) * dot(w.velocity, grad(v))


@BilinearForm
def outlet(u, v, w):
    return w.pe * u * v * dot(w.velocity, w.n)


@LinearForm
def load(v, w):
    return v * abs(w.n[1])


def connectivity(ids):
    axes = tuple(np.arange(n, dtype=float) for n in ids.shape)
    ref = MeshHex.init_tensor(*axes)
    source = ids[tuple(ref.p.astype(int))]
    return source[ref.t]


def assemble(meta, grid, velocity, graetz=False, sigma=0.):
    geo = meta["mesh"]["dimensiones_m"]
    H, W = geo["H"], geo["W"]
    air_ids = grid["air_ids"]
    air_xyz = grid["air_xyz"].reshape(-1, 3)
    nf = air_xyz.shape[0]
    hex_air = connectivity(air_ids)
    xyz, elements = air_xyz, hex_air
    if not graetz:
        solid_xyz = grid["solid_xyz"]
        solid_ids = np.empty(solid_xyz.shape[:3], dtype=int)
        solid_ids[:, -1, :] = air_ids[grid["heated_i"], 0, :]
        remaining = np.arange(solid_ids[:, :-1, :].size) + nf
        solid_ids[:, :-1, :] = remaining.reshape(solid_ids[:, :-1, :].shape)
        xyz = np.r_[air_xyz, solid_xyz[:, :-1, :, :].reshape(-1, 3)]
        elements = np.c_[hex_air, connectivity(solid_ids)]
        if not np.array_equal(xyz[solid_ids], solid_xyz):
            raise ValueError("Thermal grid interface does not match")
    mesh = MeshHex(np.ascontiguousarray(xyz.T / H), np.ascontiguousarray(elements))
    element = ElementHex1()
    all_basis = Basis(mesh, element, intorder=4)
    fluid = all_basis.with_elements(np.arange(hex_air.shape[1]))
    height_scale = QPROJ * H / K
    U = meta["U_m_s"]
    pe = RHO * CP * U * H / K
    velocity_global = np.zeros((3, len(xyz)))
    velocity_global[:, :nf] = velocity.T / U
    coeff = np.array([fluid.interpolate(v) for v in velocity_global])
    A = asm(diffusion, fluid, ratio=1., axial=0. if graetz else 1.)
    A += asm(transport, fluid, pe=pe, velocity=coeff)
    if sigma:
        # Directional diffusion uses the axial cell length, preserving the
        # conservative test function with a constant value.
        dx = np.ptp(xyz[elements[:, :hex_air.shape[1]], 0], axis=0)[:, None] / H
        speed = np.sqrt((coeff ** 2).sum(axis=0))
        local_diffusion = pe * sigma * dx / np.maximum(speed, 1e-10)
        A += asm(streamline, fluid, diffusion=local_diffusion, velocity=coeff)
    if not graetz:
        solid = all_basis.with_elements(np.arange(hex_air.shape[1], elements.shape[1]))
        A += asm(diffusion, solid, ratio=KS / K, axial=1.)
    xlim = grid["x"][[0, -1]] / H
    boundary = mesh.boundary_facets()
    vertices = mesh.p[:, mesh.facets[:, boundary]]
    exit_faces = boundary[np.all(np.isclose(vertices[0], xlim[1]), axis=0)]
    exit_basis = FacetBasis(mesh, element, facets=exit_faces, intorder=4)
    exit_vel = np.array([exit_basis.interpolate(v) for v in velocity_global])
    A += asm(outlet, exit_basis, pe=pe, velocity=exit_vel)
    floor_offset = 0. if graetz else geo["espesor"] / H
    floor_nodes = np.interp(vertices[0], grid["x"] / H, grid["y_air"][:, 0] / H) - floor_offset
    solar_faces = boundary[(np.all(np.isclose(vertices[1], floor_nodes), axis=0)) &
                           (np.min(vertices[0], axis=0) >= -1e-10) &
                           (np.max(vertices[0], axis=0) <= geo["L"] / H + 1e-10)]
    rear = FacetBasis(mesh, element, facets=solar_faces, intorder=4)
    rhs = asm(load, rear)
    expected = geo["L"] * W / H ** 2
    if not np.isclose(rhs.sum(), expected, rtol=2e-10):
        raise ValueError(f"Integrated projected load {rhs.sum()} vs {expected}")
    fixed = all_basis.get_dofs(lambda X: np.isclose(X[0], xlim[0])).all()
    free = np.setdiff1d(np.arange(len(xyz)), fixed)
    return A.tocsr(), rhs, fixed, free, height_scale


def solve(meta, grid, velocity, graetz=False, sigma=0., monotone=False):
    A, rhs, fixed, free, height_scale = assemble(meta, grid, velocity, graetz, sigma)
    H = meta["mesh"]["dimensiones_m"]["H"]
    temperature = np.zeros(A.shape[0])
    system = A.tocsr()
    initial = splu(system[free][:, free].tocsc())
    preconditioner = LinearOperator((len(free),len(free)), matvec=initial.solve)
    temperature[free] = initial.solve(rhs[free])
    active = np.zeros(A.shape[0], dtype=bool)
    if monotone:
        # Cancel positive off-diagonal coefficients in the neighborhood of
        # nodes with a nonphysical undershoot. Added edge diffusion is
        # symmetric with zero row and column sums.
        positive = A.maximum(A.T).maximum(0).tocsr()
        positive.setdiag(0.)
        positive.eliminate_zeros()
        edges = positive.tocoo()
        tolerance = .03 / height_scale
        for _ in range(8):
            bad = temperature < -tolerance
            if not np.any(bad):
                break
            grow = bad | (np.asarray(positive @ bad.astype(float)).ravel() > 0)
            if np.all(active[grow]):
                grow = grow | (np.asarray(positive @ active.astype(float)).ravel() > 0)
            active |= grow
            near = active[edges.row] | active[edges.col]
            additions = coo_matrix((edges.data[near], (edges.row[near], edges.col[near])),
                                   shape=A.shape).tocsr()
            system = A + diags(np.asarray(additions.sum(axis=1)).ravel()) - additions
            reduced = system[free][:, free].tocsc()
            approximation, status = gmres(reduced, rhs[free], x0=temperature[free],
                                           M=preconditioner, rtol=1e-10, atol=1e-11,
                                           restart=35, maxiter=4)
            temperature[free] = approximation if status == 0 else splu(reduced).solve(rhs[free])
    residual = system @ temperature - rhs
    q_inlet = -residual[fixed].sum() * QPROJ * H * H
    return TIN + temperature * height_scale, q_inlet, np.linalg.norm(residual[free]), int(active.sum())


def locate(meta):
    stem = Path(meta["mesh_source"]).stem
    for f in (HERE / "mallas").rglob(stem + ".json"):
        spec = json.loads(f.read_text(encoding="utf-8"))
        if spec["celdas_aire"] == meta["mesh"]["celdas_aire"]:
            return f.with_suffix(".npz")
    raise FileNotFoundError(f"Structured mesh data unavailable: {stem}")


def thermo_metrics(meta, section, g, temperature, q_inlet, residual, sigma):
    delta, area = average_difference(section, g)
    heat = QPROJ * g["L"] * g["W"]
    q = CP * (section.mass_kg_s.iloc[-1] * (section.bulk_K.iloc[-1] - TIN) -
              section.mass_kg_s.iloc[0] * (section.bulk_K.iloc[0] - TIN))
    return dict(Q_input_W=heat, Q_aire_W=q, Q_dif_entrada_W=q_inlet,
                cierre_energia_rel=(q+q_inlet)/heat-1, area_real_m2=area,
                DT_pared_mezcla_K=delta, h_W_m2K=heat/(area*delta),
                Nu_ref=heat*DREF/(area*delta*K), T_salida_K=section.bulk_K.iloc[-1],
                T_aire_min_K=temperature.min(), T_aire_max_K=temperature.max(),
                residual_termico_Q1=residual, estabilizacion_Q1=sigma)


def from_case(case, sigma, monotone, tag):
    meta = load_meta(case)
    grid = np.load(locate(meta))
    xyz = grid["air_xyz"].reshape(-1, 3)
    _, data = restart(case, len(xyz), True)
    data = data.sort_values("pointid")
    vel = data[["velocityx", "velocityy", "velocityz"]].to_numpy()
    result = measure(case)
    temperature, q_in, residual, active = solve(meta, grid, vel, sigma=sigma, monotone=monotone)
    data["temperature"] = temperature[:len(xyz)]
    section, _ = sectional(xyz, data)
    g = meta["mesh"]["dimensiones_m"]
    result.update(thermo_metrics(meta, section, g, temperature[:len(xyz)], q_in, residual, sigma),
                  modo="SU2 + Q1 CHT", T_placa_min_K=temperature[len(xyz):].min(),
                  T_placa_max_K=temperature[len(xyz):].max(), nodos_difusion_local=active,
                  celdas_placa=meta["mesh"]["celdas_placa"])
    dg = 4*g["W"]*g["L"]*g["H"] / (result["area_real_m2"] + g["W"]*g["L"] + 2*g["H"]*g["L"])
    result.update(Dh_geom_m=dg, Nu_geom=result["Nu_ref"]*dg/DREF,
                  Re_geom=result["Re_ref_entrada"]*dg/DREF, f_D_geom=result["f_D_ref"]*dg/DREF)
    section.to_csv(case / f"secciones_cht_{tag}.csv", index=False)
    np.savez_compressed(case / f"termico_3d_{tag}.npz", air_temperature_K=temperature[:len(xyz)],
                        solid_temperature_K=np.concatenate((
                            temperature[len(xyz):].reshape(grid["solid_ids"].shape[0], -1, grid["solid_ids"].shape[2]),
                            temperature[grid["air_ids"][grid["heated_i"], 0, :]][:, None, :]
                        ), axis=1).ravel())
    return result


def saved_or_compute(case, sigma, monotone, tag, reuse):
    stamp = (case / "restart_air.csv").stat()
    signature = [stamp.st_mtime_ns, stamp.st_size, sigma, monotone]
    saved = case / f"metricas_{tag}.json"
    if reuse and saved.is_file():
        cached = json.loads(saved.read_text(encoding="utf-8"))
        if cached["signature"] == signature and (case / f"termico_3d_{tag}.npz").is_file():
            return cached["metrics"]
    row = from_case(case, sigma, monotone, tag)
    saved.write_text(json.dumps({"signature": signature, "metrics": row}, indent=2)+"\n",
                     encoding="utf-8")
    return row


def graetz_benchmark(gridfile, re):
    spec = json.loads(gridfile.with_suffix(".json").read_text(encoding="utf-8"))
    g = spec["dimensiones_m"]
    if g["A"] != 0:
        raise ValueError("The rectangular Graetz benchmark requires a flat duct")
    grid = np.load(gridfile)
    x, y, z = grid["air_xyz"].reshape(-1, 3).T
    U = re * MU / (RHO * DREF)
    meta = {"mesh": spec, "U_m_s": U}
    ux = rectangular_velocity_profile(y, z, g["H"], g["W"], U)
    iy, iz = np.searchsorted(grid["y_air"][0], y), np.searchsorted(grid["z"], z)
    inlet = x == x.min()
    a = weights(grid["y_air"][0])[iy[inlet]] * weights(grid["z"])[iz[inlet]]
    ux *= U * g["W"] * g["H"] / (a @ ux[inlet])
    vel = np.c_[ux, np.zeros_like(ux), np.zeros_like(ux)]
    temperature, inlet_loss, residual, _ = solve(meta, grid, vel, graetz=True)
    data = pd.DataFrame(dict(x=x, y=y, z=z, velocityx=ux, velocityy=0., velocityz=0.,
                             pressure=0., temperature=temperature))
    section, _ = sectional(np.c_[x,y,z], data)
    return dict(mesh=gridfile.stem, celdas_aire=spec["celdas_aire"], Re_ref=re,
                **thermo_metrics(meta, section, g, temperature, inlet_loss, residual, 0.))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", type=Path, nargs="+")
    p.add_argument("--graetz", nargs="+", type=float)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sigma", type=float, default=0.)
    p.add_argument("--monotone", action="store_true")
    p.add_argument("--tag", default="q1")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--expected", type=int, default=0,
                   help="Continue processing completed cases until this many are available")
    p.add_argument("--timeout", type=float, default=1800.)
    args = p.parse_args()
    rows = []
    seen = set()
    started = perf_counter()
    while True:
        items = []
        for path in args.paths:
            if args.graetz:
                items.extend((path, re) for re in args.graetz)
            else:
                for case in [path] if (path / "case.json").is_file() else sorted(path.glob("*/")):
                    if ((case / "case.json").is_file() and (case / "restart_air.csv").is_file() and
                            (not args.expected or "Exit Success" in
                             (case/"su2.log").read_text(encoding="utf-8", errors="replace")[-3000:])):
                        items.append((case, None))
        for case, re in items:
            if (case, re) in seen:
                continue
            tic = perf_counter()
            row = (graetz_benchmark(case, re) if re is not None else
                   saved_or_compute(case, args.sigma, args.monotone, args.tag, args.reuse))
            print(case.name, f"h={row['h_W_m2K']:.6f}", f"E={row['cierre_energia_rel']:.2e}",
                  f"{perf_counter()-tic:.1f} s", flush=True)
            rows.append(row)
            seen.add((case, re))
            pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
        if len(rows) >= args.expected:
            break
        if perf_counter() - started > args.timeout:
            raise SystemExit(f"Only {len(rows)} of {args.expected} cases available")
        sleep(4)


if __name__ == "__main__":
    main()
