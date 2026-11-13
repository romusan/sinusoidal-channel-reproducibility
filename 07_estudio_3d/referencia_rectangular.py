"""Referencias independientes del ducto plano 3D a una pared con flujo térmico."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


RHO = 1.177
MU = 1.846e-5
K = 0.02624
CP = 1006.0
TIN = 300.0


def hydraulic_diameter(width, height):
    return 2 * width * height / (width + height)


def conductance_factor(aspect, terms=512):
    """Caudal rectangular dividido por el de placas al mismo gradiente."""
    odd = 2 * np.arange(terms) + 1
    return 1 - 192 * aspect / np.pi**5 * np.sum(
        np.tanh(np.pi * odd / (2 * aspect)) / odd**5
    )


def rectangular_velocity_profile(y, z, height, width, speed, terms=128):
    """Velocidad axial desarrollada en m/s, con y,z desde esquinas del ducto."""
    if min(height, width) <= 0 or terms < 1:
        raise ValueError("Se requieren dimensiones y número de términos positivos.")
    eta, span = np.broadcast_arrays(np.asarray(y) / height, np.asarray(z) / width)
    tolerance = 1e-12
    if (not np.isfinite(eta).all() or not np.isfinite(span).all() or
            np.any((eta < -tolerance) | (eta > 1 + tolerance) |
                   (span < -tolerance) | (span > 1 + tolerance))):
        raise ValueError("Las coordenadas (y,z) deben quedar dentro del ducto.")
    eta, span = np.clip(eta, 0, 1), np.clip(span, 0, 1)
    aspect = height / width
    odd_pi = np.pi * (2 * np.arange(terms) + 1)
    b = odd_pi / aspect
    # Cociente de cosenos hiperbólicos, evaluado con exponenciales decrecientes.
    factor = (np.exp(-span[..., None] * b) + np.exp(-(1 - span[..., None]) * b)) / (1 + np.exp(-b))
    correction = (factor * np.sin(eta[..., None] * odd_pi)) @ (8 / odd_pi**3)
    profile = 6 * speed * (eta * (1 - eta) - correction) / conductance_factor(aspect)
    return np.where((eta == 0) | (eta == 1) | (span == 0) | (span == 1), 0, profile)


def velocity_moments(aspect, y_max, z_max):
    """Momentos exactos en cosenos del perfil desarrollado normalizado u/U."""
    y_even = np.arange(0, y_max + 1, 2)
    z_even = np.arange(0, z_max + 1, 2)
    yp, zp = np.pi * y_even, np.pi * z_even
    odd_pi = np.pi * (2 * np.arange(max(512, 2 * max(y_max, z_max))) + 1)
    # Integrales de sin(j*pi*eta) cos(m*pi*eta) y del cociente cosh por cos(n*pi*s).
    sin_cos = 2 * odd_pi[:, None] / (odd_pi[:, None] ** 2 - yp**2)
    b = odd_pi / aspect
    span_cos = 2 * b[:, None] * np.tanh(b[:, None] / 2) / (b[:, None] ** 2 + zp**2)
    moments = -(sin_cos * (8 / odd_pi**3)[:, None]).T @ span_cos
    moments[0, 0] += 1 / 6
    moments[1:, 0] -= 2 / yp[1:] ** 2
    moments /= moments[0, 0]
    all_orders = np.zeros((y_max + 1, z_max + 1))
    all_orders[::2, ::2] = moments
    return all_orders


def developed_wall_difference(aspect, modes=192):
    """Diferencia pared-media en unidades qH/k para desarrollo térmico completo."""
    even = 2 * np.arange(modes)
    moments = velocity_moments(aspect, int(even[-1]), int(even[-1]))[::2, ::2]
    cosine_norm = np.full(modes, 0.5)
    cosine_norm[0] = 1
    diffusion = np.pi**2 * (even[:, None] ** 2 + (aspect * even) ** 2)
    diffusion[0, 0] = np.inf
    source = moments.copy()
    source[:, 0] -= 2
    # La contribución fuente-pared al cuadrado se suma analíticamente (1/3).
    norm = cosine_norm[:, None] * cosine_norm
    return 1 / 3 + np.sum(moments * source / (norm * diffusion))


def graetz_modes(aspect, ny, nz):
    """Pares de autovalor y peso térmico de Galerkin para la sección 3D."""
    y_mode = np.repeat(np.arange(ny), nz)
    z_mode = np.tile(2 * np.arange(nz), ny)
    moments = velocity_moments(aspect, 2 * int(y_mode[-1]), 2 * int(z_mode[-1]))
    y_sum = y_mode[:, None] + y_mode
    z_sum = z_mode[:, None] + z_mode
    y_diff = np.abs(y_mode[:, None] - y_mode)
    z_diff = np.abs(z_mode[:, None] - z_mode)
    mass = (moments[y_sum, z_sum] + moments[y_sum, z_diff]
            + moments[y_diff, z_sum] + moments[y_diff, z_diff]) / 4
    norm = np.where(y_mode == 0, 1., .5) * np.where(z_mode == 0, 1., .5)
    diffusion = np.diag(np.pi**2 * (y_mode**2 + (aspect * z_mode)**2) * norm)
    rates, vectors = eigh(diffusion, mass, overwrite_a=True, overwrite_b=True, check_finite=False)
    forcing = (z_mode == 0).astype(float)
    projection = vectors[:, 1:].T @ forcing
    return rates[1:], projection**2 / rates[1:]


def mean_delta(xi, developed, modes):
    rates, strengths = modes
    argument = rates * xi
    return developed + strengths @ (np.expm1(-argument) / argument)


def local_delta(xi, developed, modes):
    if xi == 0:
        return 0.0
    rates, strengths = modes
    return developed - strengths @ np.exp(-rates * xi)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=float, default=.30)
    parser.add_argument("--height", type=float, default=.015)
    parser.add_argument("--reference-width", type=float, default=.30)
    parser.add_argument("--reference-height", type=float, default=.015)
    parser.add_argument("--length", type=float, default=.30)
    parser.add_argument("--pressure-length", type=float, default=.42)
    parser.add_argument("--domain-length", type=float, default=.54)
    parser.add_argument("--heat-flux", type=float, default=100.)
    parser.add_argument("--reynolds", type=float, nargs="+", default=[75, 100, 125, 150, 200, 400, 800, 1200])
    parser.add_argument("--modes-y", type=int, default=40)
    parser.add_argument("--modes-z", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if min(args.width, args.height, args.reference_width, args.reference_height, args.length,
           args.pressure_length, args.domain_length, args.heat_flux, *args.reynolds) <= 0:
        parser.error("Las dimensiones, el flujo de calor y Reynolds deben ser positivos.")
    if args.height > args.width or min(args.modes_y, args.modes_z) < 12:
        parser.error("Se requiere una pared calentada de ancho W >= H y al menos 12 modos por dirección.")

    W, H, L, q = args.width, args.height, args.length, args.heat_flux
    diameter = hydraulic_diameter(W, H)
    ref = hydraulic_diameter(args.reference_width, args.reference_height)
    aspect = H / W
    conductance = conductance_factor(aspect)
    developed = developed_wall_difference(aspect)
    modes_coarse = graetz_modes(aspect, args.modes_y * 2 // 3, args.modes_z * 2 // 3)
    modes_fine = graetz_modes(aspect, args.modes_y, args.modes_z)
    alpha = K / (RHO * CP)
    power = q * W * L
    summary, stations, inlet = [], [], []
    for re in args.reynolds:
        speed = re * MU / (RHO * ref)
        mass = RHO * speed * W * H
        dpdx = 12 * MU * speed / (H**2 * conductance)
        t_scale = q * H / K
        axial_scale = alpha / (speed * H**2)
        xi = axial_scale * L
        delta_mean = t_scale * mean_delta(xi, developed, modes_fine)
        delta_coarse = t_scale * mean_delta(xi, developed, modes_coarse)
        h = q / delta_mean
        summary.append({
            "Re_ref": re, "D_ref_m": ref, "Dh_m": diameter, "W_m": W, "H_m": H, "L_m": L,
            "U_m_s": speed, "Re_Dh": RHO * speed * diameter / MU, "mass_flow_kg_s": mass,
            "P_heating_W": power, "T_out_K": TIN + power / (mass * CP), "dpdx_Pa_m": dpdx,
            "dp_heated_Pa": dpdx * L, "L_pressure_m": args.pressure_length,
            "dp_pressure_taps_Pa": dpdx * args.pressure_length,
            "L_domain_m": args.domain_length, "dp_domain_Pa": dpdx * args.domain_length,
            "f_D_Re_Dh": 24 * (diameter / H)**2 / conductance,
            "h_Graetz_3D_W_m2K": h, "Nu_ref_Graetz_3D": h * ref / K, "Nu_Dh_Graetz_3D": h * diameter / K,
            "Nu_Dh_3D_fully_developed": diameter / (H * developed),
            "area_averaged_wall_minus_bulk_K": delta_mean,
            "h_modal_sensitivity_pct": 100 * abs(delta_mean / delta_coarse - 1),
        })
        for x in L * np.array([1/30, 1/12, 1/6, 1/3, .5, 2/3, 5/6, 1.]):
            t_bulk = TIN + q * W * x / (mass * CP)
            t_delta = t_scale * local_delta(x * axial_scale, developed, modes_fine)
            stations.append({
                "Re_ref": re, "x_from_heat_start_m": x, "T_bulk_K": t_bulk,
                "T_wall_width_mean_K": t_bulk + t_delta,
                "T_wall_width_mean_minus_bulk_K": t_delta, "Nu_Dh_section": q * diameter / (K * t_delta),
            })
        eta, span = np.meshgrid(np.linspace(0, 1, 21), np.array([
            0, .0025, .005, .01, .02, .025, .05, .1, .15, .25, .5, .75, .85,
            .9, .95, .975, .98, .99, .995, .9975, 1,
        ]), indexing="ij")
        profile = rectangular_velocity_profile(eta * H, span * W, H, W, speed)
        inlet.extend({"Re_ref": re, "y_m": yi, "z_m": zi, "u_m_s": u, "u_over_U": u / speed}
                     for yi, zi, u in zip((eta * H).ravel(), (span * W).ravel(), profile.ravel()))

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "benchmark_rectangular_3d.csv", summary)
    write_csv(args.out / "graetz_3d_estaciones.csv", stations)
    write_csv(args.out / "perfil_desarrollado_3d.csv", inlet)


if __name__ == "__main__":
    main()
