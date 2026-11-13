from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project=Path(__file__).resolve().parents[1]
d=pd.read_csv(project/'07_estudio_3d/balance_por_onda_l60.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,sharey=True,figsize=(7.2,2.7),layout='constrained')
for ax,re in zip(axes,[100,800]):
    ax.axhline(0,color='#777777',lw=.65)
    s=d.loc[d.Re_ref.eq(re)]
    for lateral,color,label in [('solido','#006b86','Solid sides'),('periodico_z','#aa492f','Periodic sides')]:
        for stencil,style in [('baseline','o-'),('expanded','s--')]:
            rows=s.loc[s.lateral.eq(lateral)&s.stencil.eq(stencil)].sort_values('wave')
            ax.plot(rows.wave,rows.wave_minus_flat_K,style,color=color,ms=3.6,
                    lw=1.15,label=f'{label}, {stencil}')
    ax.set(title=rf'$Re_0={re}$',xlabel='Wave number from heated inlet',xticks=np.arange(1,6))
    ax.grid(axis='y',alpha=.25)
axes[0].set_ylabel('Wavy minus flat wall–bulk difference (K)')
axes[1].legend(loc='lower right',frameon=False,fontsize=7.6)
fig.savefig(project/'06_manuscrito/ijts/figures/wave_balance.pdf')
