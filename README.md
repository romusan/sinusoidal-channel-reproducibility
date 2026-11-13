# Conjugate heat transfer in a sinusoidal air channel

This repository supports the study *Verification of conjugate heat transfer in a
sinusoidal air channel*. It contains the Python solvers and analysis scripts for
a flat reference and a 60 mm sinusoidal wall at reference Reynolds numbers 100
and 800. The model uses a finite air duct, fixed projected heat input and either
solid sidewalls or transverse periodicity.

Use Python 3.12 and the versions in requirements.txt. Momentum fields were
computed with SU2 8.6.0. The thermal scripts use the archived momentum fields
to solve passive conjugate heat transfer. The verification bundle documents
how to regenerate the tabulated results and which files are needed for full
CFD reruns.
