"""Compare the corrugation response under solid and transverse-periodic sides."""

import numpy as np
import pandas as pd

from flujo_periodico_re800 import ROOT as PERIODIC_ROOT
from flujo_re800 import CASES, GRIDS, HERE, ROOT as SOLID_ROOT


def paired_metrics(block, label, case_names=CASES):
    flat, wave = (block.loc[c] for c in case_names)
    return {
        f'h_plano_{label}_W_m2K': flat.h_W_m2K,
        f'h_onda_{label}_W_m2K': wave.h_W_m2K,
        f'dp_plano_{label}_Pa': flat.dp_Pa,
        f'dp_onda_{label}_Pa': wave.dp_Pa,
        f'cambio_h_{label}_pct': 100*(wave.h_W_m2K/flat.h_W_m2K - 1),
        f'cambio_dp_{label}_pct': 100*(wave.dp_Pa/flat.dp_Pa - 1),
    }


def main():
    solid = pd.read_csv(HERE/'resultados_re800_acotados.csv').set_index(['malla','caso'])
    periodic = pd.read_csv(HERE/'resultados_re800_periodicos.csv').set_index(['malla','caso'])
    expected = pd.MultiIndex.from_product([GRIDS, CASES], names=['malla','caso'])
    assert solid.index.is_unique and periodic.index.is_unique
    assert set(solid.index) == set(periodic.index) == set(expected)
    solid, periodic = solid.reindex(expected), periodic.reindex(expected)
    assert periodic.Re_ref.eq(800.).all()
    assert periodic.lateral.eq('periodico_z').all()
    for column in ('celdas_aire','celdas_placa','H_mm','A_mm','lambda_mm','area_real_m2','Q_input_W'):
        np.testing.assert_allclose(solid[column], periodic[column], rtol=1.e-12, atol=1.e-12)
    assert (periodic.Re_ref_entrada - 800.).abs().max() < .5
    inlet_delta = (periodic.Re_ref_entrada-solid.Re_ref_entrada).abs().max()
    assert inlet_delta < .5
    assert periodic.error_masa_rel.abs().max() < .005
    assert periodic.max_error_flujo_seccional_rel.abs().max() < .015
    assert periodic.cierre_energia_rel.abs().max() < 1.e-7
    assert periodic.Q1_residual_relative.abs().max() < 2.e-8
    assert (periodic.min_temperature_K >= 300.-periodic.bound_tolerance_K).all()
    assert (periodic.max_air_minus_wall_K <= periodic.bound_tolerance_K).all()
    assert periodic[['periodic_temperature_air_K','periodic_temperature_solid_K']].eq(0.).all().all()
    assert periodic[['periodic_pressure_Pa','periodic_velocity_m_s']].abs().max().max() <= 5.e-6
    for grid, name in expected:
        path = PERIODIC_ROOT/grid/(name+'_Re800')
        control = SOLID_ROOT/grid/path.name
        assert (path/'inlet.dat').read_bytes() == (control/'inlet.dat').read_bytes()
        periodic_geometry = (path/'mesh.su2').read_text(encoding='ascii').split('NMARK=', 1)[0]
        solid_geometry = (control/'mesh.su2').read_text(encoding='ascii').split('NMARK=', 1)[0]
        assert periodic_geometry == solid_geometry, (grid, name, 'mesh geometry or connectivity changed')
        last = pd.read_csv(path/'history_air.csv').iloc[-1]
        rp = next(key for key in last.index if 'rms[P]' in key)
        assert float(last[rp]) <= -11., (grid, name, float(last[rp]))
    rows = []
    for grid in GRIDS:
        ns, ps = solid.loc[grid], periodic.loc[grid]
        pair = dict(malla=grid, celdas_aire=int(ns.celdas_aire.iloc[0]),
                    **paired_metrics(ns, 'solido'), **paired_metrics(ps, 'periodico'))
        for key in ('h','dp'):
            pair[f'efecto_lateral_{key}_pp'] = pair[f'cambio_{key}_solido_pct']-pair[f'cambio_{key}_periodico_pct']
        for label, case in zip(('plano', 'onda'), CASES):
            pair[f'h_{label}_periodico_vs_solido_pct'] = 100*(ps.loc[case].h_W_m2K/ns.loc[case].h_W_m2K-1)
            pair[f'dp_{label}_periodico_vs_solido_pct'] = 100*(ps.loc[case].dp_Pa/ns.loc[case].dp_Pa-1)
        rows.append(pair)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(HERE/'comparacion_contornos_re800.csv', index=False, encoding='utf-8-sig')
    keys = ['malla', 'cambio_h_solido_pct', 'cambio_h_periodico_pct', 'efecto_lateral_h_pp',
            'cambio_dp_solido_pct', 'cambio_dp_periodico_pct', 'efecto_lateral_dp_pp']
    print(comparison[keys].to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    for key in ('h', 'dp'):
        values = comparison[f'efecto_lateral_{key}_pp'].to_numpy()
        print(f'{key}: medium-to-fine contrast difference {values[-1]-values[-2]:+.5f} pp; '
              f'1.25 times the three-grid spread {1.25*np.ptp(values):.5f} pp')
    print(f'Maximum relative energy imbalance {periodic.cierre_energia_rel.abs().max():.2e}; '
          f'maximum sectional mass deviation {periodic.max_error_flujo_seccional_rel.abs().max():.2e}')
    print(f'Maximum measured inlet Reynolds difference {inlet_delta:.6f}')


if __name__ == '__main__':
    main()
