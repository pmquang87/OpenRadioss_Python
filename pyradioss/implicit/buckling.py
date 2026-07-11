"""
Linearized (eigenvalue) buckling analysis — M9.

Fortran origin: ``engine/source/implicit/imp_buck.F`` — the /IMPL/BUCKL
branch of the OpenRadioss implicit solver, which extracts the lowest
eigenvalues of the tangent pencil built from the material stiffness and the
geometric (initial-stress) stiffness of a pre-stressed state. The port
exposes the same computation as a library function on an already-solved
implicit model (the /IMPL/BUCKL engine card itself is deferred — see
PORTING_GUIDE M9).

Theory
------
Take an equilibrium state with stress field sigma_0 produced by a reference
load P_0 (solved by the implicit static driver). For a proportional load
mu * P_0, the linearized-buckling assumption is that the pre-stress scales
with the load while the geometry change stays negligible, so the tangent
stiffness along the fundamental path is

    K_T(mu) ~= K_mat + mu * K_geo(sigma_0)

with K_mat the material + hourglass tangent (``assembly.assemble``) and
K_geo the initial-stress matrix assembled from sigma_0
(``assembly.assemble_kgeo``). Buckling = the fundamental path losing
uniqueness = K_T singular:

    (K_mat + mu * K_geo) phi = 0
        <=>  (-K_geo) phi = theta * K_mat phi,   mu = 1 / theta

a symmetric generalized eigenproblem with K_mat positive definite (a
well-posed constrained static model), solved here with the dense
``scipy.linalg.eigh``. The LARGEST positive theta gives the SMALLEST
positive load multiplier — the classical critical load P_cr = mu_1 * P_0
(Euler's column formula pi^2 EI/(kL)^2 is the closed-form special case the
M9 validation checks against). Negative theta correspond to buckling under
the REVERSED load and are reported separately by sign.

The dense solve is O(ndof^3): perfectly fine for the validation-scale models
this port targets (10^2-10^4 DOFs); a shift-invert sparse eigensolver is the
documented upgrade path for bigger models.
"""

from __future__ import annotations

import numpy as np

from . import require_scipy
from .assembly import assemble, assemble_kgeo
from .dofmap import DofMap


def buckling_factors(model, nev=4, log=None):
    """Lowest positive linearized-buckling load multipliers of the model's
    CURRENT (pre-stressed) state.

    Run an implicit static analysis first (it leaves the committed stress in
    the element buffers and the deformed geometry in ``model.x``); then this
    returns ``(factors, modes)``:

    * ``factors`` — up to ``nev`` positive multipliers mu, ascending: the
      structure buckles at mu times the CURRENT load;
    * ``modes``   — matching list of ``(du, dur)`` nodal mode shapes
      ((numnod, 3) each, fixed DOFs zero), normalized to unit equation-space
      length.

    The eigenproblem is stated at the current committed geometry with the
    current stress: for the classical "small preload" use (Euler column) that
    is indistinguishable from x0; after a large-displacement /IMPL/NONLIN run
    it is the tangent buckling estimate about the deformed state."""
    _, _ = require_scipy()             # fail early with the clear message
    import scipy.linalg as sla

    dof = DofMap(model, log)
    x_geom = model.x if model.x.size else model.x0
    Km = assemble(model, dof, x_geom).toarray()
    Kg = assemble_kgeo(model, dof, x_geom).toarray()

    # (-K_geo) phi = theta K_mat phi ; K_mat PD (scipy checks via Cholesky)
    theta, vecs = sla.eigh(-Kg, Km)
    order = np.argsort(-theta)                    # largest theta first
    factors = []
    modes = []
    for idx in order:
        th = theta[idx]
        if th <= 1e-12 or len(factors) >= nev:
            break
        factors.append(1.0 / th)
        du, dur = dof.scatter_solution(vecs[:, idx])
        modes.append((du, dur))
    if log is not None and factors:
        log.info(f" BUCKLING FACTORS (LOWEST) . . . . . . : "
                 f"{', '.join(f'{f:.6E}' for f in factors)}")
    return np.array(factors), modes
