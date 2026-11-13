"""Measure sectional and real-area quantities from archived SU2 3D nodal fields."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from referencia_rectangular import conductance_factor
from su2_caso import DREF, K, CP, MU, QPROJ, RHO, TIN, weights, read_su2


def mass_matrix(axis):
    spacing = np.diff(axis)
    diagonal = (np.r_[spacing, 0.] + np.r_[0., spacing]) / 3
    return np.diag(diagonal) + np.diag(spacing / 6, 1) + np.diag(spacing / 6, -1)


def fields(file):
    table = pd.read_csv(file)
    table.columns = [re.sub(r"[^a-z0-9]", "", c.lower()) for c in table.columns]
    return table


def restart(case, npoints, fluid):
    for file in sorted(case.glob("restart*.csv"), reverse=True):
        head = file.open(encoding="utf-8-sig").readline().lower()
        if "pointid" not in head or ("velocity" in head) != fluid:
            continue
        data = fields(file)
        if len(data) == npoints:
            return file, data
    raise FileNotFoundError(f"No matching {'fluid' if fluid else 'solid'} restart in {case}")


def sectional(xyz, data):
    if "pointid" in data:
        data = data.sort_values("pointid")
    points = data[["x", "y", "z"]].to_numpy()
    if not np.allclose(points, xyz, rtol=1e-6, atol=5e-10):
        raise ValueError("Restart coordinates do not match archived mesh ordering")
    x, span = np.unique(xyz[:, 0]), np.unique(xyz[:, 2])
    nx, nz = len(x), len(span)
    ny = len(xyz) // (nx * nz)
    perm = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    table = data.iloc[perm]
    arr = lambda key: table[key].to_numpy().reshape(nx, ny, nz)
    y = arr("y")[:, :, 0]
    ux, vy, wz, pressure = (arr(key) for key in ("velocityx", "velocityy", "velocityz", "pressure"))
    if "temperature" in table:
        temperature = arr("temperature")
    elif "enthalpy" in table:
        temperature = 298.15 + arr("enthalpy") / CP
    else:
        temperature = np.zeros_like(ux) + TIN
    mass, bulk, pmean, wall, ke, mirror = [np.empty(nx) for _ in range(6)]
    mass_z, w_z = mass_matrix(span), weights(span)
    for i in range(nx):
        w = weights(y[i])[:, None] * w_z
        volume = np.sum(w * ux[i])
        mass[i] = RHO * volume
        pmean[i] = np.sum(w * pressure[i]) / w.sum()
        k_energy = .5 * RHO * (ux[i] ** 2 + vy[i] ** 2 + wz[i] ** 2)
        ke[i] = np.sum(w * ux[i] * k_energy) / volume
        bulk[i] = np.sum(ux[i] * (mass_matrix(y[i]) @ temperature[i] @ mass_z)) / volume
        wall[i] = temperature[i, 0] @ w_z / w_z.sum()
        mirror[i] = np.sqrt(np.sum(w * (ux[i] - ux[i, :, ::-1]) ** 2) / w.sum())
    data = pd.DataFrame({"x_m": x, "floor_m": y[:, 0], "mass_kg_s": mass, "bulk_K": bulk,
                         "pressure_Pa": pmean, "kinetic_Pa": ke, "wall_K": wall,
                         "symmetry_rms_Ux_m_s": mirror})
    return data, (ux, vy, wz, temperature)


def average_difference(section, geometry):
    L, A, wave = geometry["L"], geometry["A"], geometry["lambda"]
    x = section.x_m.to_numpy()
    sl = (x >= -1e-12) & (x <= L + 1e-12)
    x = x[sl]
    delta = (section.wall_K - section.bulk_K).to_numpy()[sl]
    xi, w = np.polynomial.legendre.leggauss(8)
    xq = (x[:-1, None] + np.diff(x)[:, None] * (xi + 1) / 2).ravel()
    delta_q = np.interp(xq, x, delta)
    area_w = (np.diff(x)[:, None] * w / 2).ravel()
    if A > 0:
        area_w *= np.hypot(1, 2 * np.pi * A / wave * np.cos(2 * np.pi * xq / wave))
    return np.sum(area_w * delta_q) / area_w.sum(), area_w.sum() * geometry["W"]


def load_meta(case):
    meta = json.loads((case / "case.json").read_text(encoding="utf-8"))
    if "mesh" not in meta:
        spec = Path(meta["mesh_source"]).with_suffix(".json").name
        source = next(Path(__file__).parent.joinpath("mallas").rglob(spec))
        meta["mesh"] = json.loads(source.read_text(encoding="utf-8"))
    return meta


def measure(case):
    meta = load_meta(case)
    rotated_mesh = meta.get("mesh_rotation") == "x,-z,y"
    zones = read_su2(case / ("gravity_mesh.su2" if rotated_mesh else "mesh.su2"))
    airfile, air = restart(case, len(zones[0]["points"]), True)
    if rotated_mesh:
        coords = zones[0]["points"].copy()
        zones[0]["points"][:, 1] = coords[:, 2]
        zones[0]["points"][:, 2] = -coords[:, 1]
        rotated = air.copy()
        air["y"], air["z"] = rotated.z, -rotated.y
        air["velocityy"], air["velocityz"] = rotated.velocityz, -rotated.velocityy
    section, fields_3d = sectional(zones[0]["points"], air)
    section.to_csv(case / "secciones.csv", index=False)
    g = meta["mesh"]["dimensiones_m"]
    u, h, w, re = (meta[k] for k in ("U_m_s", "H_m", "W_m", "Re_ref"))
    length_taps = g["L"] + (g["L_entrada"] + g["L_salida"]) / 2
    p = section.pressure_Pa.to_numpy()
    p1, p2 = np.interp([-g["L_entrada"] / 2, g["L"] + g["L_salida"] / 2], section.x_m, p)
    dp = p1 - p2
    mass = section.mass_kg_s.to_numpy()
    q = CP * (mass[-1] * (section.bulk_K.iloc[-1] - TIN) -
              mass[0] * (section.bulk_K.iloc[0] - TIN))
    exact_pressure = 12 * MU * (mass[0] / (RHO * h * w)) * length_taps / (h * h * conductance_factor(h/w))
    row = {
        "caso": meta["mesh"]["caso"], "corrida": case.name, "modo": ("CHT Boussinesq" if meta.get("boussinesq") else "CHT") if meta["cht"] else
        ("frio" if meta["cold"] else "flujo_directo"),
        "celdas_aire": meta["mesh"]["celdas_aire"],
        "celdas_placa": meta["mesh"]["celdas_placa"] if meta["cht"] else 0,
        "Re_ref_objetivo": re, "Re_ref_entrada": DREF * mass[0] / (h * w * MU),
        "H_mm": h * 1000, "A_mm": g["A"] * 1000, "lambda_mm": g["lambda"] * 1000,
        "dp_Pa": dp, "dp_completa_Pa": p[0] - p[-1],
        "dp_total_completa_Pa": p[0] + section.kinetic_Pa.iloc[0] - p[-1] - section.kinetic_Pa.iloc[-1],
        "f_D_ref": 2 * dp * DREF / (RHO * u * u * length_taps),
        "dp_rect_exacto_Pa": exact_pressure if g["A"] == 0 else np.nan,
        "error_dp_rect_rel": dp / exact_pressure - 1 if g["A"] == 0 else np.nan,
        "mass_in_kg_s": mass[0], "mass_out_kg_s": mass[-1],
        "error_masa_rel": mass[-1] / mass[0] - 1,
        "max_error_flujo_seccional_rel": np.max(np.abs(mass / mass[0] - 1)),
        "bomba_W": (p[0] - p[-1]) * mass[0] / RHO,
        "T_salida_K": section.bulk_K.iloc[-1], "Q_aire_W": q,
        "symmetry_rms_Ux_U": section.symmetry_rms_Ux_m_s.max() / u,
        "Uy_max_U": np.max(np.abs(fields_3d[1])) / u, "Uz_max_U": np.max(np.abs(fields_3d[2])) / u,
        "T_aire_min_K": fields_3d[3].min(), "T_aire_max_K": fields_3d[3].max(),
        "restart_air": airfile.name,
    }
    if not meta["cold"]:
        delta, area = average_difference(section, g)
        power = meta.get("Q_input_W", QPROJ * w * g["L"])
        row.update(Q_input_W=power, cierre_energia_rel=q/power-1, area_real_m2=area,
                   DT_pared_mezcla_K=delta, h_W_m2K=power / (area * delta),
                   Nu_ref=power * DREF / (area * delta * K))
    if meta["cht"]:
        solidfile, solid = restart(case, len(zones[1]["points"]), False)
        row.update(T_placa_min_K=solid.temperature.min(), T_placa_max_K=solid.temperature.max(),
                   restart_solid=solidfile.name)
    return row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cases", type=Path, nargs="+")
    cli.add_argument("--out", type=Path, default=Path(__file__).parent / "resultados_su2.csv")
    args = cli.parse_args()
    rows = []
    for path in args.cases:
        candidates = [path] if (path / "case.json").exists() else sorted(path.glob("*/"))
        for candidate in candidates:
            if not (candidate / "case.json").exists():
                continue
            try:
                rows.append(measure(candidate))
            except (ValueError, FileNotFoundError, KeyError) as exc:
                print(f"{candidate.name}: {exc}")
    if not rows:
        raise SystemExit("No complete SU2 fields to measure")
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(table[["corrida", "Re_ref_entrada", "dp_Pa", "error_masa_rel", "Q_aire_W"] +
                (["h_W_m2K", "cierre_energia_rel"] if "h_W_m2K" in table else [])].to_string(index=False))


if __name__ == "__main__":
    main()
