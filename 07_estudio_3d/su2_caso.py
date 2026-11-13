"""Prepare and run dimensional SU2 cases for the parametric solar-air duct."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from referencia_rectangular import rectangular_velocity_profile


RHO, MU, CP, K = 1.177, 1.846e-5, 1006.0, .02624
TIN, QPROJ, KS = 300., 100., 205.
DREF = 2 * .3 * .015 / (.3 + .015)
HERE = Path(__file__).resolve().parent


def read_su2(path):
    """Read only mesh nodes and boundary markers, preserving zone order."""
    zones = []
    current = None
    with path.open(encoding="ascii") as file:
        for line in file:
            if line.startswith("IZONE") or (line.startswith("NDIME") and current is None):
                current = {"markers": {}}
                zones.append(current)
            if line.startswith("NPOIN"):
                n = int(line.split("=")[1].split()[0])
                current["points"] = np.array([[float(v) for v in next(file).split()[:3]] for _ in range(n)])
            if line.startswith("MARKER_TAG"):
                marker = line.split("=", 1)[1].strip()
                n = int(next(file).split("=")[1])
                current["markers"][marker] = np.array([[int(v) for v in next(file).split()[1:5]]
                                                       for _ in range(n)], dtype=int)
    return zones


def fmt(value):
    return f"{value:.15g}"


def options(path, values):
    path.write_text("".join(f"{k}= {v}\n" for k, v in values.items()), encoding="ascii")


def pairs(mapping):
    return "(" + ", ".join(item for key, value in mapping.items() for item in (key, fmt(value))) + ")"


def weights(axis):
    dx = np.diff(axis)
    return np.r_[dx[0], dx[:-1] + dx[1:], dx[-1]] / 2


def setup_inlet(path, zone, velocity, uniform=False):
    nodes = np.unique(zone["markers"]["inlet"])
    points = zone["points"][nodes]
    span = np.unique(points[:, 2])
    height = np.unique(points[:, 1])
    H, W = height[-1] - height[0], span[-1] - span[0]
    if uniform:
        local_u = np.full(len(points), velocity)
    else:
        local_u = rectangular_velocity_profile(points[:, 1] - height[0],
                                               points[:, 2] - span[0], H, W, velocity)
        yy, zz = np.searchsorted(height, points[:, 1]), np.searchsorted(span, points[:, 2])
        area_weights = weights(height)[yy] * weights(span)[zz]
        local_u *= H * W * velocity / (area_weights @ local_u)
    values = np.c_[points, np.full(len(points), TIN), local_u, np.tile([1., 0., 0.], (len(points), 1))]
    with path.open("w", encoding="ascii") as f:
        f.write(f"NMARK= 1\nMARKER_TAG= inlet\nNROW= {len(nodes)}\nNCOL= 8\n")
        np.savetxt(f, values, fmt="%.16g")
    return H, W


def heat_rates(solid, width, reference_length=.3):
    areas = {}
    xyz = solid["points"]
    for tag, faces in solid["markers"].items():
        if tag.startswith("flux_") or tag == "absorbed_flux":
            # Project each quad onto the x-z plane, avoiding sensitivity to winding.
            planar = xyz[faces][:, :, [0, 2]]
            x, z = planar.transpose(2, 0, 1)
            projected = .5 * np.abs((x * np.roll(z, -1, axis=1) -
                                     z * np.roll(x, -1, axis=1)).sum(axis=1))
            areas[tag] = projected.sum() * QPROJ
    if not np.isclose(sum(areas.values()), QPROJ * width * reference_length, rtol=1e-9):
        raise ValueError(f"Projected heat rate mismatch: {sum(areas.values()):g} W")
    return areas


def prepare(args):
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mesh = args.mesh.resolve()
    zones = read_su2(mesh)
    cht = len(zones) == 2 and not args.fluid_only
    u = args.re * MU / (RHO * DREF)
    H, W = setup_inlet(directory / "inlet.dat", zones[0], u, args.uniform)
    if cht:
        shutil.copyfile(directory / "inlet.dat", directory / "inlet_0.dat")
    fluid_tags = zones[0]["markers"]
    args.case_mesh = directory / "mesh.su2"
    if args.fluid_only:
        # Strip the multizone header and the solid zone, keeping original node IDs.
        with mesh.open(encoding="ascii") as src, args.case_mesh.open("w", encoding="ascii") as dest:
            for line in src:
                if line.startswith("IZONE") and int(line.split("=")[1]) == 2:
                    break
                if not line.startswith(("NZONE", "IZONE")):
                    dest.write(line)
    elif mesh != args.case_mesh:
        shutil.copyfile(mesh, args.case_mesh)
    interfaces = [tag for tag in fluid_tags if tag == "interface_air"]
    fluid_wall = {tag: 0. for tag in fluid_tags if tag not in ["inlet", "outlet", *interfaces]}
    if not cht:
        fluid_wall.update({tag: 0. if args.cold else QPROJ for tag in interfaces})
    common = {
        "INC_NONDIM": "DIMENSIONAL",
        "NUM_METHOD_GRAD": "GREEN_GAUSS",
        "NUM_METHOD_GRAD_RECON": "WEIGHTED_LEAST_SQUARES",
        "CFL_ADAPT": "NO", "LINEAR_SOLVER": "FGMRES", "LINEAR_SOLVER_PREC": "ILU",
        "LINEAR_SOLVER_ERROR": "1E-7", "LINEAR_SOLVER_ITER": args.linear,
        "CONV_RESIDUAL_MINVAL": "-11", "CONV_STARTITER": "10",
        "TABULAR_FORMAT": "CSV", "HISTORY_OUTPUT": "(ITER, RMS_RES)",
        "VOLUME_OUTPUT": "(COORDINATES, SOLUTION, PRIMITIVE)",
        "OUTPUT_FILES": "(RESTART_ASCII)",
        "OUTPUT_WRT_FREQ": "1000000",
        "READ_BINARY_RESTART": "NO",
    }
    flow = {
        **common, "SOLVER": "INC_NAVIER_STOKES", "KIND_TURB_MODEL": "NONE",
        "RESTART_SOL": "YES" if args.restart else "NO",
        "MESH_FILENAME": "mesh.su2", "MESH_FORMAT": "SU2",
        "INC_ENERGY_EQUATION": "NO" if args.cold else "YES",
        "INC_DENSITY_MODEL": "CONSTANT",
        "INC_DENSITY_INIT": fmt(RHO), "INC_TEMPERATURE_INIT": fmt(TIN),
        "INC_VELOCITY_INIT": f"({fmt(u)}, 0., 0.)", "SPECIFIC_HEAT_CP": fmt(CP),
        "VISCOSITY_MODEL": "CONSTANT_VISCOSITY", "MU_CONSTANT": fmt(MU),
        "CONDUCTIVITY_MODEL": "CONSTANT_CONDUCTIVITY",
        "THERMAL_CONDUCTIVITY_CONSTANT": fmt(K),
        "INC_INLET_TYPE": "VELOCITY_INLET", "INC_OUTLET_TYPE": "PRESSURE_OUTLET",
        "MARKER_INLET": f"(inlet, {fmt(TIN)}, {fmt(u)}, 1., 0., 0.)",
        "MARKER_OUTLET": "(outlet, 0.)",
        "SPECIFIED_INLET_PROFILE": "YES", "INLET_FILENAME": "inlet.dat",
        "MARKER_HEATFLUX": pairs(fluid_wall),
        "CONV_NUM_METHOD_FLOW": "FDS", "MUSCL_FLOW": "YES",
        "SLOPE_LIMITER_FLOW": "NONE", "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
        "CFL_NUMBER": fmt(args.cfl), "INNER_ITER": args.inner if cht else args.iterations,
        "RESTART_FILENAME": "restart_air", "SOLUTION_FILENAME": "solution_air",
        "CONV_FILENAME": "history_air",
    }
    if args.restart:
        source = args.restart.resolve()
        for zone, name in ((0, "air"), (1, "solid")) if cht else ((None, "air"),):
            suffix = "" if zone is None else f"_{zone}"
            saved = source / f"restart_{name}{suffix}.csv"
            target = directory / f"solution_{name}{suffix}.csv"
            if args.restart_scale and args.cold and not cht:
                base = json.loads((source / "case.json").read_text(encoding="utf-8"))
                scale = u / base["U_m_s"]
                with saved.open(encoding="utf-8-sig", newline="") as f, target.open("w", encoding="ascii", newline="") as out:
                    reader = csv.DictReader(f)
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames, quoting=csv.QUOTE_NONNUMERIC)
                    writer.writeheader()
                    for row in reader:
                        for key in ("Pressure", "Velocity_x", "Velocity_y", "Velocity_z"):
                            row[key] = fmt(float(row[key]) * scale)
                        writer.writerow({key: float(v) for key, v in row.items()})
            else:
                shutil.copyfile(saved, target)
    if not cht:
        options(directory / "air.cfg", flow)
        return directory, "air.cfg", {"Re_ref": args.re, "U_m_s": u, "H_m": H, "W_m": W, "cht": False}

    rates = heat_rates(zones[1], W)
    rates["solid_adiabatic"] = 0.
    solid = {
        **common, "SOLVER": "HEAT_EQUATION",
        "RESTART_SOL": "YES" if args.restart else "NO",
        "FREESTREAM_TEMPERATURE": fmt(TIN + min(20., 1600 / args.re)),
        "MATERIAL_DENSITY": "2700.", "SPECIFIC_HEAT_CP": "900.",
        "THERMAL_CONDUCTIVITY_CONSTANT": fmt(KS),
        "MARKER_HEATFLUX": pairs(rates), "INTEGRATED_HEATFLUX": "YES",
        "TIME_DISCRE_HEAT": "EULER_IMPLICIT", "CFL_NUMBER": "5000",
        "INNER_ITER": args.solid_inner, "LINEAR_SOLVER_ITER": max(args.linear, 15),
        "RESTART_FILENAME": "restart_solid", "SOLUTION_FILENAME": "solution_solid",
        "CONV_FILENAME": "history_solid",
    }
    main = {
        **common, "SOLVER": "MULTIPHYSICS", "MATH_PROBLEM": "DIRECT",
        "TIME_DOMAIN": "NO", "OUTER_ITER": args.iterations,
        "MESH_FILENAME": "mesh.su2", "MESH_FORMAT": "SU2",
        "CONFIG_LIST": "(air.cfg, solid.cfg)",
        "MARKER_ZONE_INTERFACE": "(interface_air, interface_solid)",
        "MARKER_CHT_INTERFACE": "(interface_air, interface_solid)",
        "CHT_COUPLING_METHOD": "DIRECT_TEMPERATURE_ROBIN_HEATFLUX",
        "CONV_FILENAME": "history",
        "RESTART_SOL": "YES" if args.restart else "NO",
    }
    options(directory / "air.cfg", flow)
    options(directory / "solid.cfg", solid)
    options(directory / "cht.cfg", main)
    meta = {"Re_ref": args.re, "U_m_s": u, "H_m": H, "W_m": W, "cht": True, "Q_input_W": sum(rates.values())}
    return directory, "cht.cfg", meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--re", type=float, required=True)
    p.add_argument("--iterations", type=int, default=220)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--solid-inner", type=int, default=10)
    p.add_argument("--linear", type=int, default=10)
    p.add_argument("--cfl", type=float, default=20.)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--fluid-only", action="store_true")
    p.add_argument("--cold", action="store_true")
    p.add_argument("--uniform", action="store_true")
    p.add_argument("--restart", type=Path)
    p.add_argument("--restart-scale", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--su2", type=Path,
                   default=Path(os.environ.get("TEMP", ".")) / "colector_su2_860/bin/SU2_CFD.exe")
    args = p.parse_args()
    work, config, meta = prepare(args)
    geometry_metadata = args.mesh.with_suffix(".json")
    if geometry_metadata.is_file():
        meta["mesh"] = json.loads(geometry_metadata.read_text(encoding="utf-8"))
    meta.update(inlet="uniform" if args.uniform else "developed", cold=args.cold,
                mesh_source=args.mesh.name, mesh_sha256=hashlib.sha256(args.case_mesh.read_bytes()).hexdigest(),
                out_iter=args.iterations, inner_iter=args.inner, solid_inner=args.solid_inner)
    (work / "case.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        return
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE")
    with (work / "su2.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([str(args.su2.resolve()), config, "-t", str(args.threads)],
                                cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"SU2 exit {result.returncode}. See {work / 'su2.log'}")
    print(work)


if __name__ == "__main__":
    main()
