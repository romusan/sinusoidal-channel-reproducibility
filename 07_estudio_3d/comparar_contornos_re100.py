"""Check and summarize the 60 mm paired forced-convection results."""

import argparse

from pathlib import Path

import numpy as np
import pandas as pd

from comparar_contornos_re800 import paired_metrics
from prueba_re100_contornos import CASES, GRIDS, HERE


def compare(path, grids, first_ring, reynolds=100):
    data = pd.read_csv(path).set_index(['malla', 'lateral', 'caso'])
    index = pd.MultiIndex.from_product([grids, ('solido', 'periodico_z'), CASES])
    assert data.index.is_unique and set(data.index) == set(index)
    assert data.Re_ref.eq(reynolds).all() and data.first_ring.eq(first_ring).all()
    assert data.log10_rms_presion.le(-11.).all()
    assert data.error_masa_rel.abs().max() < .005
    assert data.max_error_flujo_seccional_rel.abs().max() < .015
    assert data.cierre_energia_rel.abs().max() < 1.e-7
    assert data.Q1_residual_relative.abs().max() < 2.e-8
    assert (data.min_temperature_K >= 300.-data.bound_tolerance_K).all()
    assert (data.max_air_minus_wall_K <= data.bound_tolerance_K).all()
    solid = data.xs('solido', level='lateral')
    periodic = data.xs('periodico_z', level='lateral').reindex(solid.index)
    for column in ('celdas_aire', 'celdas_placa', 'H_mm', 'A_mm', 'lambda_mm', 'area_real_m2', 'Q_input_W'):
        np.testing.assert_allclose(solid[column], periodic[column], rtol=1.e-12, atol=1.e-12)
    inlet_delta = (periodic.Re_ref_entrada - solid.Re_ref_entrada).abs().max()
    assert inlet_delta < min(.001*reynolds, .5)
    assert periodic[['periodic_temperature_air_K', 'periodic_temperature_solid_K']].eq(0.).all().all()
    assert periodic[['periodic_pressure_Pa', 'periodic_velocity_m_s']].abs().max().max() <= 5.e-6
    rows = []
    for grid in grids:
        no_slip, periodic_z = solid.loc[grid], periodic.loc[grid]
        row = dict(malla=grid, celdas_aire=int(no_slip.celdas_aire.iloc[0]),
                   celdas_placa=int(no_slip.celdas_placa.iloc[0]), first_ring=first_ring,
                   **paired_metrics(no_slip, 'solido', CASES),
                   **paired_metrics(periodic_z, 'periodico', CASES))
        for key in ('h', 'dp'):
            row[f'efecto_lateral_{key}_pp'] = row[f'cambio_{key}_solido_pct'] - row[f'cambio_{key}_periodico_pct']
        for geometry in ('plano', 'onda'):
            for metric, units in [('h', 'W_m2K'), ('dp', 'Pa')]:
                periodic_value = row[f'{metric}_{geometry}_periodico_{units}']
                solid_value = row[f'{metric}_{geometry}_solido_{units}']
                row[f'{metric}_{geometry}_periodico_vs_solido_pct'] = 100*(periodic_value/solid_value-1)
        rows.append(row)
    print(f'{Path(path).name}: max inlet Re difference {inlet_delta:.6f}, '
          f'max energy imbalance {data.cierre_energia_rel.abs().max():.3e}, '
          f'max sectional mass deviation {data.max_error_flujo_seccional_rel.abs().max():.3e}')
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--re', type=int, choices=(100, 800), default=100)
    parser.add_argument('--fine-sensitivity', action='store_true',
                        help='Include the fine-grid correction variant')
    args = parser.parse_args()
    reynolds = args.re
    results_file, sensitivity_file, output_file = {
        100: ('resultados_re100_contornos.csv', 're100_contornos_sensibilidad_acotada.csv',
              'comparacion_contornos_re100.csv'),
        800: ('resultados_re800_l60_contornos.csv', 're800_l60_sensibilidad_acotada.csv',
              'comparacion_contornos_re800_l60.csv'),
    }[reynolds]
    main_results = compare(HERE/results_file, GRIDS, False, reynolds)
    sensitivity = compare(HERE/sensitivity_file, ('media',), True, reynolds)
    if args.fine_sensitivity:
        fine_file = HERE / sensitivity_file.replace('.csv', '_fina.csv')
        fine_variant = compare(fine_file, ('fina',), True, reynolds)
        sensitivity = pd.concat([sensitivity, fine_variant], ignore_index=True)
    results = pd.concat([main_results, sensitivity], ignore_index=True)
    results.to_csv(HERE/output_file, index=False, encoding='utf-8-sig')
    keys = ['malla', 'first_ring', 'cambio_h_solido_pct', 'cambio_h_periodico_pct',
            'efecto_lateral_h_pp', 'cambio_dp_solido_pct', 'cambio_dp_periodico_pct', 'efecto_lateral_dp_pp']
    print(results[keys].to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    for key in keys[2:5]:
        series = main_results.set_index('malla')[key]
        grid_margin = 1.25 * (series.max() - series.min())
        thermal_shift = (sensitivity.set_index('malla')[key] - series).abs().max()
        print(f'{key}: sampled grid margin {grid_margin:.6f} pp, '
              f'thermal shift {thermal_shift:.6f} pp, sum {grid_margin + thermal_shift:.6f} pp')
    if reynolds == 800:
        low = pd.read_csv(HERE/'comparacion_contornos_re100.csv').set_index(['malla', 'first_ring'])
        high = results.set_index(['malla', 'first_ring'])
        shared = low.index.intersection(high.index, sort=False)
        high, low = high.loc[shared], low.loc[shared]
        assert set(high.xs(False, level='first_ring').index) == set(GRIDS)
        np.testing.assert_array_equal(high[['celdas_aire', 'celdas_placa']], low[['celdas_aire', 'celdas_placa']])
        metrics = [column for column in keys if column.startswith(('cambio_', 'efecto_'))]
        differences = (high[metrics] - low[metrics]).add_suffix('_800_menos_100_pp')
        differences.to_csv(HERE/'comparacion_reynolds_l60.csv', encoding='utf-8-sig')
        print(differences.to_string(float_format=lambda x: f'{x:.6f}'))
        base = differences.xs(False, level='first_ring')
        expanded = differences.xs(True, level='first_ring')
        for metric in ('cambio_h_solido_pct', 'cambio_h_periodico_pct'):
            column = metric + '_800_menos_100_pp'
            spread = 1.25 * (base[column].max() - base[column].min())
            thermal = (expanded[column] - base[column]).abs().max()
            print(f'{metric} Reynolds contrast: grid margin {spread:.6f} pp, '
                  f'thermal shift {thermal:.6f} pp, sum {spread + thermal:.6f} pp')


if __name__ == '__main__':
    main()
