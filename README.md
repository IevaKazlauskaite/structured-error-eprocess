# Sequential Structure-Sensitive Residual Diagnostics

Code accompanying the paper *Sequential Structure-Sensitive Residual Diagnostics for PDE Inverse Problems*.

The package implements e-process diagnostics for detecting structured residual error, together with reproducible Poisson and Stokes experiments. The Poisson experiment uses direct inversion while the Stokes experiment additionally requires an FEM solver (specifically, DOLFINx, PETSc, MPI, and UFL).

Run the Poisson experiments:
`python scripts/run_poisson_power.py`
`python scripts/run_poisson_pipeline.py`

Run the Stokes experiment:
`python scripts/run_stokes.py`
Generated results are written to `results/`.
