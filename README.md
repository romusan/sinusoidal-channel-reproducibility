# Conjugate heat transfer in a sinusoidal air channel

This repository supports the study *Conjugate heat transfer and pressure drop in a finite-width sinusoidal air channel*. It contains the Python solvers and analysis scripts for
a flat reference and a 60 mm sinusoidal wall at reference Reynolds numbers 100
and 800. The model uses a finite air duct, fixed projected heat input and either
solid sidewalls or transverse periodicity.

Use Python 3.12 and the versions in requirements.txt. Momentum fields were
computed with SU2 8.6.0. The thermal scripts use the archived momentum fields
to solve passive conjugate heat transfer. The verification bundle documents
how to regenerate the tabulated results and which files are needed for full
CFD reruns.


## Verification data

The data directory holds the fifteen manuscript data supplements. To regenerate the
four-grid comparison and sensitivity table, run:

    python -m pip install -r requirements.txt
    python 07_estudio_3d/comparar_l60_pared.py --from-csv data/S10_L60_four_grid_absolute.csv

The script writes the paired responses, contrasts, absolute-coefficient grid
increments and sensitivity to the study directory. The fourth mesh bisects the
four lowest air layers of the 91,854-cell grid while preserving the axial,
spanwise and solid meshes.

The verification ZIP also contains momentum residual histories, solver
configurations, geometry metadata and the underlying steady and transient
tables. Full numerical fields and native CAD remain in the project archive.

S11 reports sensitivity with and without the coarse mesh. S12 reports wall-to-bulk
temperature differences by wave period. S13 contains segment diagnostics from
the wall-refined fields. The wall flux in S13 is a physical-gradient diagnostic.

## Thermal verification

S14 compares the common fine and wall-refined flat grids against the modal
Graetz result with both stencils at Reynolds numbers 100 and 800. S15 records
native SU2 CHT runs on the coarse common grid. Both native runs reached the
iteration limit. The enthalpy imbalance and temperature undershoot exclude
these estimates from the coefficient comparison.

To regenerate the wall-flux figure from the checkout:

    python 06_manuscrito/make_figure_wall_flux.py --data data/S13_L60_axial_diagnostics.csv --out wall_flux.pdf

Extract `IJTS_verification_data.zip` into a separate directory. From its root,
run the following with the packages listed in `requirements.txt` in the checkout:

    python 07_estudio_3d/verificar_graetz_pared.py --reynolds 800 100 --grids pared fina
    python 07_estudio_3d/verificar_cht_su2.py --su2 /path/to/SU2_CFD --outer 600

The archive includes the mesh arrays and modal reference needed by the first
command, and the two coarse meshes and starting fields needed by the second.
