"""Malla hexaédrica del canal y la placa, con interfaz coincidente para SU2."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


GEOMETRY = Path(__file__).resolve().parents[1] / "02_cad" / "parametros.csv"


def read_geometry(path, case):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["caso"] == case]
    if len(rows) != 1:
        raise ValueError(f"Se requiere una fila para {case} en {path}")
    keys = ("W_mm", "L_mm", "H_mm", "A_mm", "lambda_mm", "espesor_mm", "L_entrada_mm", "L_salida_mm")
    dimensions = np.array([float(rows[0][key]) for key in keys]) * 1e-3
    width, length, height, amplitude, wavelength, thickness, inlet, outlet = dimensions
    if not np.isfinite(dimensions).all() or min(width, length, height, wavelength, thickness, inlet, outlet) <= 0:
        raise ValueError("Las dimensiones deben ser finitas y positivas")
    if not 0 <= amplitude < height:
        raise ValueError("La geometría requiere 0 ≤ A < H")
    if amplitude and not np.isclose(length / wavelength, round(length / wavelength), rtol=0, atol=1e-10):
        raise ValueError("Se necesitan ondas completas para unir el piso ondulado a los tramos rectos")
    return dimensions


def spacing(cells, stretch):
    if cells < 1 or not np.isfinite(stretch) or not 0 <= stretch <= 6:
        raise ValueError("El número de celdas debe ser positivo y 0 ≤ estiramiento ≤ 6")
    coord = np.linspace(0, 1, cells + 1)
    if stretch == 0:
        return coord
    return 0.5 * (1 + np.tanh(stretch * (2 * coord - 1)) / np.tanh(stretch))


def vertices(x, y, z):
    xg, yg, zg = np.broadcast_arrays(x[:, None, None], y[:, :, None], z[None, None, :])
    return np.stack((xg, yg, zg), axis=-1)


def hexahedra(ids):
    return np.stack(
        (ids[:-1, :-1, :-1], ids[1:, :-1, :-1], ids[1:, 1:, :-1], ids[:-1, 1:, :-1],
         ids[:-1, :-1, 1:], ids[1:, :-1, 1:], ids[1:, 1:, 1:], ids[:-1, 1:, 1:]),
        axis=-1,
    ).reshape(-1, 8)


def quadrilaterals(ids, reverse=False):
    faces = np.stack((ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]), axis=-1).reshape(-1, 4)
    return faces[:, [0, 3, 2, 1]] if reverse else faces


def write_zone(stream, index, points, ids, markers):
    elements = hexahedra(ids)
    if index is not None:
        stream.write(f"IZONE= {index}\n")
    stream.write(f"NDIME= 3\nNELEM= {len(elements)}\n")
    element_rows = np.column_stack((elements, np.arange(len(elements))))
    np.savetxt(stream, element_rows, fmt="12 %d %d %d %d %d %d %d %d %d")
    coordinates = points.reshape(-1, 3)
    stream.write(f"NPOIN= {len(coordinates)}\n")
    np.savetxt(stream, coordinates, fmt="%.15e %.15e %.15e")
    stream.write(f"NMARK= {len(markers)}\n")
    for name, faces in markers.items():
        stream.write(f"MARKER_TAG= {name}\nMARKER_ELEMS= {len(faces)}\n")
        np.savetxt(stream, faces, fmt="9 %d %d %d %d")


def generate_mesh(case, geometry, output, nx, ny, nz, ns, stretch_y, stretch_z,
                  fluid_only=False, heat_flux_strips=False, split_bottom=0):
    width, length, height, amplitude, wavelength, thickness, lin, lout = read_geometry(geometry, case)
    if min(nx, ny, nz, ns) < 1:
        raise ValueError("Todos los números de celdas deben ser positivos")
    n_in = max(1, round(nx * lin / length))
    n_out = max(1, round(nx * lout / length))
    heated = slice(n_in, n_in + nx + 1)
    x = np.concatenate((
        np.linspace(-lin, 0, n_in, endpoint=False),
        np.linspace(0, length, nx, endpoint=False),
        np.linspace(length, length + lout, n_out + 1),
    ))
    base = np.zeros_like(x)
    base[heated] = amplitude * np.sin(2 * np.pi * x[heated] / wavelength)
    base[n_in] = base[n_in + nx] = 0
    eta = spacing(ny, stretch_y)
    if not 0 <= split_bottom <= ny:
        raise ValueError("Las capas inferiores a subdividir deben estar dentro de la malla")
    if split_bottom:
        midpoints = (eta[:split_bottom] + eta[1:split_bottom + 1]) / 2
        eta = np.sort(np.r_[eta, midpoints])
        ny = len(eta) - 1
    z = width * spacing(nz, stretch_z)
    y_air = base[:, None] + (height - base[:, None]) * eta
    air = vertices(x, y_air, z)
    air_ids = np.arange(np.prod(air.shape[:-1])).reshape(air.shape[:-1])

    air_boundaries = {
        "inlet": quadrilaterals(air_ids[0, :, :], reverse=True),
        "outlet": quadrilaterals(air_ids[-1, :, :]),
        "roof": quadrilaterals(air_ids[:, -1, :], reverse=True),
        "sidewalls": np.concatenate((
            quadrilaterals(air_ids[:, :, 0], reverse=True),
            quadrilaterals(air_ids[:, :, -1]),
        )),
        "floor_adiabatic": np.concatenate((
            quadrilaterals(air_ids[:n_in + 1, 0, :]),
            quadrilaterals(air_ids[n_in + nx:, 0, :]),
        )),
        "interface_air": quadrilaterals(air_ids[heated, 0, :]),
    }
    y_solid = base[heated, None] - thickness * np.linspace(1, 0, ns + 1)
    solid = vertices(x[heated], y_solid, z)
    solid_ids = np.arange(np.prod(solid.shape[:-1])).reshape(solid.shape[:-1])
    rear = quadrilaterals(solid_ids[:, 0, :])
    solid_boundaries = {
        "interface_solid": quadrilaterals(solid_ids[:, -1, :], reverse=True),
        "solid_adiabatic": np.concatenate((
            quadrilaterals(solid_ids[0, :, :], reverse=True),
            quadrilaterals(solid_ids[-1, :, :]),
            quadrilaterals(solid_ids[:, :, 0], reverse=True),
            quadrilaterals(solid_ids[:, :, -1]),
        )),
    }
    strips = []
    if heat_flux_strips and not fluid_only:
        for i, (left, right) in enumerate(zip(x[heated][:-1], x[heated][1:])):
            name = f"flux_{i:03d}"
            solid_boundaries[name] = rear[i * nz:(i + 1) * nz]
            dy = base[n_in + i + 1] - base[n_in + i]
            strips.append({
                "marcador": name, "x_min_m": left, "x_max_m": right, "dx_m": right - left,
                "area_proyectada_m2": width * (right - left),
                "area_real_malla_m2": width * np.hypot(right - left, dy),
            })
    else:
        solid_boundaries["absorbed_flux"] = rear
    if not np.array_equal(air[heated, 0, :], solid[:, -1, :]):
        raise ValueError("Los nodos de la interfaz placa-aire no son coincidentes")
    if not (np.diff(x) > 0).all() or not (np.diff(z) > 0).all():
        raise ValueError("Hay celdas de anchura o longitud nula")
    # Para este mapeo, dx/dxi > 0 y dz/dzeta > 0. El signo del Jacobiano
    # depende del espesor vertical, que se interpola linealmente entre nodos en x.
    if not (np.diff(y_air, axis=1) > 0).all() or not (np.diff(y_solid, axis=1) > 0).all():
        raise ValueError("Hay celdas de altura nula o negativa")

    area_projected = width * length
    area_mesh = width * np.hypot(np.diff(x[heated]), np.diff(base[heated])).sum()
    integrand = np.hypot(1, 2 * np.pi * amplitude / wavelength *
                         np.cos(2 * np.pi * np.linspace(0, length, 8193) / wavelength))
    area_exact = width * length / 8192 * (
        integrand[0] + integrand[-1] + 4 * integrand[1:-1:2].sum() + 2 * integrand[2:-1:2].sum()
    ) / 3
    volumes_air = np.diff(x) * (height - (base[:-1] + base[1:]) / 2)
    meta = {
        "caso": case,
        "unidades": "m",
        "dimensiones_m": dict(zip(("W", "L", "H", "A", "lambda", "espesor", "L_entrada", "L_salida"),
                                  map(float, (width, length, height, amplitude, wavelength, thickness, lin, lout)))),
        "celdas_entrada": n_in,
        "celdas_calentadas": nx,
        "celdas_salida": n_out,
        "celdas_altura_aire": ny,
        "celdas_ancho": nz,
        "celdas_espesor_placa": ns if not fluid_only else 0,
        "celdas_aire": (n_in + nx + n_out) * ny * nz,
        "celdas_placa": nx * ns * nz if not fluid_only else 0,
        "area_proyectada_m2": area_projected,
        "area_interfaz_malla_m2": area_mesh,
        "area_interfaz_seno_m2": area_exact,
        "error_relativo_area_interfaz": area_mesh / area_exact - 1,
        "volumen_aire_m3": width * volumes_air.sum(),
        "volumen_placa_m3": area_projected * thickness if not fluid_only else 0,
        "luz_minima_nodal_m": (height - base).min(),
        "luz_minima_seno_m": height - amplitude,
        "marcadores_aire": {name: len(faces) for name, faces in air_boundaries.items()},
        "marcadores_placa": {name: len(faces) for name, faces in solid_boundaries.items()} if not fluid_only else {},
        "franjas_calentadas": strips,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as stream:
        if not fluid_only:
            stream.write("NZONE= 2\n")
        write_zone(stream, None if fluid_only else 1, air, air_ids, air_boundaries)
        if not fluid_only:
            write_zone(stream, 2, solid, solid_ids, solid_boundaries)
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    arrays = {"x": x, "z": z, "y_air": y_air, "air_xyz": air,
              "air_ids": air_ids, "heated_i": np.arange(n_in, n_in + nx + 1)}
    if not fluid_only:
        arrays.update(solid_xyz=solid, solid_ids=solid_ids, y_solid=y_solid)
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caso", required=True, help="Nombre de la configuración en parametros.csv")
    parser.add_argument("--salida", type=Path, required=True, help="Ruta de la malla .su2")
    parser.add_argument("--parametros", type=Path, default=GEOMETRY)
    parser.add_argument("--nx", type=int, default=80, help="Celdas axiales del tramo calentado")
    parser.add_argument("--ny", type=int, default=12, help="Celdas en altura del aire")
    parser.add_argument("--nz", type=int, default=20, help="Celdas en ancho")
    parser.add_argument("--ns", type=int, default=3, help="Celdas en el espesor de la placa")
    parser.add_argument("--estirar-y", type=float, default=0.75)
    parser.add_argument("--estirar-z", type=float, default=1.25)
    parser.add_argument("--solo-aire", action="store_true", help="Malla de una zona para aislar la hidráulica")
    parser.add_argument("--franjas-flujo", action="store_true", help="Usa un marcador de carga por celda axial")
    args = parser.parse_args()
    if args.salida.suffix.lower() != ".su2":
        parser.error("La ruta de salida debe terminar en .su2")
    meta = generate_mesh(args.caso, args.parametros, args.salida, args.nx, args.ny, args.nz, args.ns,
                         args.estirar_y, args.estirar_z, args.solo_aire, args.franjas_flujo)
    print(f'{args.caso}: {meta["celdas_aire"]} celdas de aire, {meta["celdas_placa"]} de placa')


if __name__ == "__main__":
    main()
