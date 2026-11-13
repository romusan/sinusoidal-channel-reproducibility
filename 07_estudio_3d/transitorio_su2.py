"""Perturb a converged full-width SU2 solution and measure finite-time decay."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from energia_q1 import locate
from medir_su2 import fields, load_meta
from su2_caso import options, weights


def set_seed(base, grid, meta, amplitude):
    H, W = meta["H_m"], meta["W_m"]
    U = meta["U_m_s"]
    g = meta["mesh"]["dimensiones_m"]
    xyz = grid["air_xyz"]
    x, y, z = xyz.transpose(3, 0, 1, 2)
    eta = (y-grid["y_air"][:, 0, None, None]) / (H-grid["y_air"][:, 0, None, None])
    span = z/W
    extent = np.sin(np.pi * np.clip(x/g["L"], 0., 1.)) ** 2
    wave = 2*np.pi/(4*H)
    lateral = np.sin(np.pi*span)**2
    psi_y = np.pi*np.sin(2*np.pi*eta)/(H-grid["y_air"][:, 0, None, None])
    psi_z = np.pi/W*np.sin(2*np.pi*span)*np.sin(wave*z)+lateral*wave*np.cos(wave*z)
    vy = H*extent*np.sin(np.pi*eta)**2*psi_z
    vz = -H*extent*psi_y*lateral*np.sin(wave*z)
    peak = np.hypot(vy, vz).max()
    seeded = base.copy()
    seeded["Velocity_y"] += amplitude*U*vy.ravel()/peak
    seeded["Velocity_z"] += amplitude*U*vz.ravel()/peak
    return seeded


def measure_series(folder, baseline, grid, dt, U, start, meta):
    ref = baseline[["Velocity_x", "Velocity_y", "Velocity_z"]].to_numpy()
    x = grid["x"]
    volume_weights = (weights(x)[:, None, None] *
                      np.stack([weights(y) for y in grid["y_air"]])[:, :, None] *
                      weights(grid["z"]))
    wp = volume_weights.ravel()
    g = meta["mesh"]["dimensiones_m"]
    L = x[-1]-x[0]
    probes = np.array([[g["L"]*.37, g["H"]*.45, g["W"]*.25],
                       [g["L"]*.64, g["H"]*.55, g["W"]*.50],
                       [g["L"]*.81, g["H"]*.45, g["W"]*.75]])
    coordinates = grid["air_xyz"].reshape(-1, 3)
    nearest = [np.argmin(((coordinates-p)**2).sum(axis=1)) for p in probes]
    rows = []
    frames = [(start-1, folder/f"initial_{start-1:05d}.csv")]
    frames.extend((int(p.stem.rsplit("_", 1)[1]), p)
                  for p in sorted(folder.glob("transient_*.csv")))
    for step, path in frames:
        data = fields(path).sort_values("pointid")
        vel = data[["velocityx", "velocityy", "velocityz"]].to_numpy()
        delta = (vel-ref)/U
        rms = np.sqrt(wp @ (delta**2).sum(axis=1)/wp.sum())
        t = (step-start+1)*dt
        rows.append(dict(step=step, t_s=t, t_transitos=t*U/L, rms_U=rms,
                         max_U=np.sqrt((delta**2).sum(axis=1)).max(),
                         **{f"w_sonda_{j+1}_U": delta[i,2] for j,i in enumerate(nearest)}))
    table = pd.DataFrame(rows)
    table.to_csv(folder / "perturbacion.csv", index=False)
    if len(table) < 3:
        raise RuntimeError("Too few transient fields for decay measurement")
    if table.rms_U.iloc[0] > 1.e-12:
        final = f"RMS final/initial={table.rms_U.iloc[-1]/table.rms_U.iloc[0]:.4g}"
    else:
        final = f"unseeded RMS final/U0={table.rms_U.iloc[-1]:.4g}"
    print(f"{folder.name}: {len(table)} frames, {final}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dt-star", type=float, default=.5)
    p.add_argument("--transits", type=float, default=2.)
    p.add_argument("--sample", type=int, default=18)
    p.add_argument("--amplitude", type=float, default=.01)
    p.add_argument("--inner", type=int, default=6)
    p.add_argument("--min-inner", type=int, help="Enable the residual test from this inner iteration")
    p.add_argument("--su2", type=Path,
                   default=Path(os.environ.get("TEMP", "."))/"colector_su2_860/bin/SU2_CFD.exe")
    args = p.parse_args()
    if args.min_inner is not None and not 0 <= args.min_inner < args.inner:
        p.error("--min-inner must be less than --inner and nonnegative")
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.baseline)
    grid = np.load(locate(meta))
    data = pd.read_csv(args.baseline/"restart_air.csv").sort_values("PointID")
    data.columns = data.columns.str.strip()
    seeded = set_seed(data, grid, meta, args.amplitude)
    start = 2
    for index in (0, 1):
        seeded.to_csv(folder/f"initial_{index:05d}.csv", index=False)
    for file in ("mesh.su2", "inlet.dat"):
        shutil.copyfile(args.baseline/file, folder/file)
    shutil.copyfile(folder/"inlet.dat", folder/f"inlet_{start:05d}.dat")
    config = dict(tuple(k.strip() for k in line.split("=", 1))
                  for line in (args.baseline/"air.cfg").read_text(encoding="ascii").splitlines()
                  if "=" in line and not line.startswith("%"))
    U, H = meta["U_m_s"], meta["H_m"]
    dt = args.dt_star*H/U
    n = int(np.ceil(args.transits*(grid["x"][-1]-grid["x"][0])/(U*dt)))
    config.update(RESTART_SOL="YES", TIME_DOMAIN="YES",
                  TIME_MARCHING="DUAL_TIME_STEPPING-2ND_ORDER", RESTART_ITER=start,
                  TIME_STEP=f"{dt:.15g}", TIME_ITER=start+n, MAX_TIME="10000.",
                  INNER_ITER=args.inner, UNST_CFL_NUMBER="0.",
                  RESTART_FILENAME="transient", SOLUTION_FILENAME="initial",
                  OUTPUT_WRT_FREQ=args.sample, CONV_FILENAME="history_transient",
                  SCREEN_OUTPUT="(TIME_ITER, INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X)")
    if args.min_inner is not None:
        config["CONV_STARTITER"] = args.min_inner
    options(folder/"transient.cfg", config)
    info = dict(baseline=str(args.baseline), Re_ref=meta["Re_ref"], dt_star=args.dt_star,
                dt_s=dt, steps=n, transits=args.transits, amplitude_max_U=args.amplitude,
                inner=args.inner, min_inner=args.min_inner)
    (folder/"transient.json").write_text(json.dumps(info, indent=2)+"\n", encoding="utf-8")
    with (folder/"su2.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([str(args.su2.resolve()), "transient.cfg", "-t", "1"],
                                cwd=folder, env=dict(os.environ, OMP_NUM_THREADS="1"),
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"SU2 exit {result.returncode}: {folder/'su2.log'}")
    measure_series(folder, data, grid, dt, U, start, meta)


if __name__ == "__main__":
    main()
