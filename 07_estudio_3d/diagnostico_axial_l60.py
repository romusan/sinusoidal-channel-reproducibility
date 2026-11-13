"""Extract phase-resolved diagnostics from the wall-refined, bounded 60 mm runs."""

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE / 'corridas/prueba_l60_refinamiento_pared'
MESHES = HERE / 'mallas/prueba_re800/pared'
CASES = ('H15_Plana', 'H15_A150_L60')
PERIOD = .060
CONDUCTIVITY = .02624


def segment_metrics(re, case, lateral, stencil, mesh):
    folder = ROOT / f'Re{re}' / 'pared' / case / lateral
    solution = folder / stencil
    section = pd.read_csv(solution / 'secciones.csv')
    np.testing.assert_allclose(section.x_m, mesh['x'], atol=1e-14, rtol=0)
    with np.load(solution / 'temperaturas.npz') as state:
        temperature = state['air_temperature_K']
    velocity = pd.read_csv(folder / 'restart_air.csv').sort_values('PointID')
    xyz = mesh['air_xyz']
    np.testing.assert_allclose(velocity[['x','y','z']], xyz.reshape(-1,3), atol=1e-12, rtol=0)
    u = velocity.Velocity_x.to_numpy().reshape(xyz.shape[:-1])
    # The first-layer face-average at each axial node is a nodal diagnostic.
    z = mesh['z']
    dz = np.diff(z)
    layer_mid = (u[:,1,:-1] + u[:,1,1:]) / 2
    first_layer_backflow = ((layer_mid < 0) @ dz) / dz.sum()
    x = mesh['x']
    heated = mesh['heated_i']
    x0, x1 = x[heated[:-1]], x[heated[1:]]
    dx = x1-x0
    xmid = .5*(x0+x1)
    y = mesh['y_air']
    slope = np.diff(y[heated,0])/dx
    real_length = dx * np.hypot(1., slope)
    delta = (section.wall_K-section.bulk_K).to_numpy()
    dmid = (delta[heated[:-1]]+delta[heated[1:]])/2
    # Integrate the physical Q1 conductive flux on each wall face with two
    # Gauss points axially and exact trapezoidal integration in span.
    at = temperature[heated,0,:]
    dt = temperature[heated,1,:]-at
    gap = y[heated,1]-y[heated,0]
    dtdx = np.diff(at,axis=0)/dx[:,None]
    q = np.zeros_like(dtdx)
    for xi in ((1-1/np.sqrt(3))/2,(1+1/np.sqrt(3))/2):
        dt_dy = ((1-xi)*dt[:-1]+xi*dt[1:]) / ((1-xi)*gap[:-1]+xi*gap[1:])[:,None]
        q -= .5*CONDUCTIVITY*((1+slope[:,None]**2)*dt_dy - slope[:,None]*dtdx) / np.hypot(1.,slope[:,None])
    qspan = ((q[:,:-1]+q[:,1:])/2) @ dz / dz.sum()
    back = (first_layer_backflow[heated[:-1]]+first_layer_backflow[heated[1:]])/2
    return pd.DataFrame(dict(Re_ref=re, case=case, lateral=lateral, stencil=stencil,
                             x_mid_m=xmid, wave=np.minimum((xmid/PERIOD).astype(int)+1,5),
                             phase=(xmid/PERIOD)%1, dx_m=dx, surface_length_m=real_length,
                             wall_bulk_difference_K=dmid, physical_Q1_wall_flux_W_m2=qspan,
                             first_off_wall_backflow_span_fraction=back))


def main():
    groups = []
    for case in CASES:
        with np.load(MESHES/(case+'.npz')) as mesh:
            for re in (100,800):
                for lateral in ('solido','periodico_z'):
                    for stencil in ('baseline','expanded'):
                        groups.append(segment_metrics(re,case,lateral,stencil,mesh))
    result=pd.concat(groups,ignore_index=True)
    result.to_csv(HERE/'diagnostico_axial_l60.csv',index=False,encoding='utf-8-sig')
    result['area_delta_K_m'] = result.surface_length_m * result.wall_bulk_difference_K
    result['conductive_power_W'] = result.surface_length_m * .3 * result.physical_Q1_wall_flux_W_m2
    result['backflow_length_m'] = result.dx_m * result.first_off_wall_backflow_span_fraction
    phase = result.groupby(['Re_ref','lateral','stencil','wave','case'])[
        ['area_delta_K_m','surface_length_m']].sum()
    delta = (phase.area_delta_K_m/phase.surface_length_m).unstack('case')
    (delta[CASES[1]]-delta[CASES[0]]).rename('wave_minus_flat_K').reset_index().to_csv(
        HERE/'balance_por_onda_l60.csv', index=False)
    for (re, lateral, stencil), group in result.groupby(['Re_ref','lateral','stencil']):
        phase = group.groupby(['case','wave'])[['area_delta_K_m','surface_length_m','conductive_power_W','backflow_length_m']].sum()
        delta = (phase.area_delta_K_m/phase.surface_length_m).unstack(0)
        print(re,lateral,stencil,'wave-mean delta differences',np.round(delta.H15_A150_L60-delta.H15_Plana,3).tolist(),
              'conductive W',phase.conductive_power_W.groupby('case').sum().round(3).to_dict(),
              'backflow fraction',(phase.backflow_length_m.groupby('case').sum()/.3).round(5).to_dict())


if __name__ == '__main__':
    main()
