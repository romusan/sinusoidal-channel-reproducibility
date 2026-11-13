"""Verify and compare the common-grid 60 mm steady thermal calculations."""

import argparse
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from comparar_contornos_re100 import compare
from refinar_l60_pared import BOUNDARIES, CASES, GRIDS, HERE, REYNOLDS, ROOT, archive


ORDER = list(reversed(GRIDS))
METRICS = ['cambio_h_solido_pct','cambio_h_periodico_pct','efecto_lateral_h_pp',
           'cambio_dp_solido_pct','cambio_dp_periodico_pct','efecto_lateral_dp_pp']


def verify_matching_inlets():
    for grid in ORDER:
        for name in CASES:
            for boundary in BOUNDARIES:
                low, high = (archive(grid, name, re, boundary) for re in REYNOLDS)
                assert (low/'mesh.su2').read_bytes() == (high/'mesh.su2').read_bytes()
                a, b = [np.loadtxt(case/'inlet.dat', skiprows=4) for case in (low, high)]
                fixed = [0, 1, 2, 3, 5, 6, 7]
                np.testing.assert_array_equal(a[:,fixed], b[:,fixed])
                np.testing.assert_allclose(b[:,4], 8*a[:,4], rtol=2e-14, atol=2e-16)


def read_runs():
    verify_matching_inlets()
    inputs = [ROOT/f'Re{re}'/g/n/b/stencil/'metrics.csv'
              for re in REYNOLDS for stencil in ('baseline','expanded')
              for g in ORDER for n in CASES for b in BOUNDARIES]
    raw = pd.concat([pd.read_csv(p) for p in inputs], ignore_index=True)
    assert len(raw) == len(inputs)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-csv', type=Path, help='Recompute from the archived case-level data')
    args = parser.parse_args()
    raw = pd.read_csv(args.from_csv) if args.from_csv else read_runs()
    assert len(raw) == len(REYNOLDS)*2*len(ORDER)*len(CASES)*len(BOUNDARIES)
    summaries = []
    with tempfile.TemporaryDirectory(prefix='l60_pared_') as tmp:
        for re in REYNOLDS:
            for expanded in (False, True):
                data = raw.loc[raw.Re_ref.eq(re) & raw.first_ring.eq(expanded)]
                file = Path(tmp)/f're{re}_{expanded}.csv'
                data.to_csv(file, index=False)
                pairs = compare(file, ORDER, expanded, re)
                summaries.append(pairs.assign(Re_ref=re))
    pair = pd.concat(summaries, ignore_index=True)
    low, high = (pair.loc[pair.Re_ref.eq(re)].set_index(['malla','first_ring']) for re in REYNOLDS)
    np.testing.assert_array_equal(low[['celdas_aire','celdas_placa']],high[['celdas_aire','celdas_placa']])
    contrasts = (high[METRICS] - low[METRICS]).add_suffix('_800_menos_100_pp')
    raw.to_csv(HERE/'resultados_l60_pared.csv',index=False,encoding='utf-8-sig')
    pair.to_csv(HERE/'comparacion_l60_pared.csv',index=False,encoding='utf-8-sig')
    contrasts.to_csv(HERE/'contraste_reynolds_l60_pared.csv',encoding='utf-8-sig')
    subset_rows = []
    for boundary in ('solid','periodic'):
        label = 'solido' if boundary == 'solid' else 'periodico'
        key = f'cambio_h_{label}_pct_800_menos_100_pp'
        for scope, grids in [('all_four', ORDER), ('medium_fine_wall_refined', ORDER[1:])]:
            base, expanded = (contrasts.xs(flag,level='first_ring').loc[grids,key]
                              for flag in (False, True))
            spread = 1.25*(base.max()-base.min())
            shift = (expanded-base).abs()
            subset_rows.append(dict(lateral=boundary, grids=scope, scaled_baseline_range_pp=spread,
                                    largest_stencil_shift_pp=shift.max(), grid_of_largest_stencil_shift=shift.idxmax(),
                                    sampled_sensitivity_pp=spread+shift.max(),
                                    wall_refined_baseline_pp=base.loc['pared'],
                                    wall_refined_expanded_pp=expanded.loc['pared']))
    pd.DataFrame(subset_rows).to_csv(HERE/'sensibilidad_contraste_subconjuntos_l60.csv',index=False)
    print(pair[['Re_ref','first_ring','malla',*METRICS]].round(5).to_string(index=False))
    print(contrasts.round(5).to_string())
    metrics = []
    for signal, table in [('Re100',low),('Re800',high),('Re800-Re100',contrasts)]:
        base = table.xs(False,level='first_ring')
        expanded = table.xs(True,level='first_ring')
        columns = list(base.columns) if signal == 'Re800-Re100' else METRICS
        for column in columns:
            b,e = base[column].reindex(ORDER),expanded[column].reindex(ORDER)
            spread = 1.25*(b.max()-b.min())
            stencil_shift = (e-b).abs().max()
            metrics.append(dict(signal=signal, quantity=column, spatial_spread_pp=spread,
                                largest_stencil_shift_pp=stencil_shift,
                                sampled_sensitivity_pp=spread+stencil_shift,
                                fourth_baseline=b.loc['pared'], fourth_expanded=e.loc['pared'],
                                fourth_grid_shift_baseline_pp=b.loc['pared']-b.loc['fina'],
                                fourth_grid_shift_expanded_pp=e.loc['pared']-e.loc['fina']))
    sensitivity = pd.DataFrame(metrics)
    sensitivity.to_csv(HERE/'sensibilidad_l60_pared.csv',index=False,encoding='utf-8-sig')
    print(sensitivity.round(5).to_string(index=False))
    changes = []
    for (re, boundary, geom, expanded), group in raw.groupby(['Re_ref','lateral','caso','first_ring']):
        h = group.set_index('malla').h_W_m2K.reindex(ORDER)
        jumps = 100 * (h.iloc[1:].to_numpy()/h.iloc[:-1].to_numpy()-1)
        changes.append(dict(Re_ref=re, lateral=boundary, caso=geom, first_ring=expanded,
                            h_coarse_W_m2K=h.iloc[0], h_medium_W_m2K=h.iloc[1],
                            h_fine_W_m2K=h.iloc[2], h_fourth_W_m2K=h.iloc[3],
                            coarse_to_medium_pct=jumps[0], medium_to_fine_pct=jumps[1],
                            fine_to_fourth_pct=jumps[2]))
    pd.DataFrame(changes).to_csv(HERE/'refinamiento_h_l60_pared.csv',index=False,encoding='utf-8-sig')


if __name__ == '__main__':
    main()
