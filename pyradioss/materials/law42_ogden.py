"""
LAW42 — Ogden / Mooney-Rivlin hyperelasticity (/MAT/LAW42, /MAT/OGDEN).
Solids only in this port.

Fortran origin: ``engine/source/materials/mat/mat042/sigeps42.F``; deck
reading in ``starter/source/materials/mat/mat042/hm_read_mat42.F``.

Theory (Ogden 1972; Holzapfel "Nonlinear Solid Mechanics" §6.5)
---------------------------------------------------------------
A hyperelastic material defines the stress from a strain-energy density
W of the principal stretches lambda_i of the deformation gradient F —
NOT from a stress rate: the stress is a pure function of the current
deformation, so a closed strain cycle returns exactly to zero stress
(unlike the hypoelastic LAW1/2 laws). The Ogden series with a volumetric
penalty (Radioss' nearly-incompressible formulation):

    W = sum_p  mu_p / alpha_p * (lb1^alpha_p + lb2^alpha_p + lb3^alpha_p - 3)
        + K/2 * (J - 1)^2

with J = det F (volume ratio) and the *deviatoric* stretches
lb_i = lambda_i * J^(-1/3) (so the first sum is purely distortional).
mu_p may be negative when alpha_p < 0 — the stability requirement is
mu_p * alpha_p > 0 for every term (each term then contributes
mu_p*alpha_p/2 > 0 to the ground-state shear modulus G0 = sum mu_p*alpha_p/2;
checked by the Starter). Mooney-Rivlin is the special case
(mu1, alpha1=2; mu2, alpha2=-2). The principal Cauchy stresses are

    sigma_i = 1/J * ( S_i - (S_1+S_2+S_3)/3 ) + K*(J - 1),
    S_i     = sum_p mu_p * lb_i^alpha_p

and the stress tensor is reassembled from the principal directions of the
left Cauchy-Green tensor B = F F^T (which shares eigenvectors with the
Cauchy stress for isotropic materials).

Interface with the element kernels
----------------------------------
The kernels compute the deformation gradient at the integration point
EXACTLY from the stored initial shape-function gradients,

    F_ab = sum_i x_ia * dN_i/dX_b        (no rate integration, no drift)

and pass it in ``extra["F"]``. The incoming (Jaumann-rotated) stress and
the strain increment are ignored — sig is overwritten with the fresh
hyperelastic stress. deps is still used by the kernel for the energy
bookkeeping, which remains consistent (the force is V * sigma * gradN).

Sound speed / time step (the M2 lesson, applied here by construction)
---------------------------------------------------------------------
An Ogden material STIFFENS with stretch: the tangent modulus grows like
mu_p * alpha_p * lambda^alpha_p, so at lambda = 2 with alpha = 3 the wave
speed is ~2x the ground-state one and a time step frozen at the initial
sound speed is UNSTABLE. This law therefore returns its own per-element
sound speed each cycle,

    c^2 = ( K_t + 4/3 * G_t ) / rho_current
    G_t = max_i sum_p mu_p * alpha_p * lb_i^alpha_p / (2 J)
    K_t = K * max(J, 1)

which reduces to the ground-state c at F = I and grows monotonically with
the largest deviatoric stretch (both alpha > 0 terms at lb > 1 and
alpha < 0 terms at lb < 1 grow — tension and compression both stiffen).
The kernels feed this c into the Courant limit in place of the constant
elastic estimate; the long-run /DT 0.9 divergence test in
tests/test_m3_integration.py exercises exactly this path at large
stretch.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20


def _principal_kirchhoff_dev(mat, lb: np.ndarray):
    """S_i = sum_p mu_p lb_i^alpha_p per principal direction; lb (m, 3)."""
    S = np.zeros_like(lb)
    for mu, al in zip(mat.params["mu"], mat.params["alpha"]):
        if mu != 0.0:
            S += mu * lb ** al
    return S


def solid_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray, dt: float, extra: dict):
    """Hyperelastic stress from the deformation gradient (module doc).

    Returns (sig, epsp, c) with c the per-element nonlinear sound speed
    used by the kernel for the time step."""
    F = extra["F"]                                # (m, 3, 3)
    K = mat.K
    rho0 = mat.rho0

    J = np.linalg.det(F)
    J = np.maximum(J, EM20)                       # inverted elements guard
    B = np.einsum("nab,ncb->nac", F, F)           # left Cauchy-Green FF^T
    lam2, N = np.linalg.eigh(B)                   # ascending eigenvalues
    lam = np.sqrt(np.maximum(lam2, EM20))         # principal stretches
    lb = lam * (J ** (-1.0 / 3.0))[:, None]       # deviatoric stretches

    # principal Cauchy stresses: deviatoric Ogden part + volumetric penalty
    S = _principal_kirchhoff_dev(mat, lb)
    smean = S.mean(axis=1)
    sigp = (S - smean[:, None]) / J[:, None] + (K * (J - 1.0))[:, None]

    # reassemble the tensor from the principal directions of B
    T = np.einsum("nak,nk,nbk->nab", N, sigp, N)
    sig[:, 0] = T[:, 0, 0]
    sig[:, 1] = T[:, 1, 1]
    sig[:, 2] = T[:, 2, 2]
    sig[:, 3] = T[:, 0, 1]
    sig[:, 4] = T[:, 1, 2]
    sig[:, 5] = T[:, 0, 2]

    # nonlinear sound speed (module docstring): tangent shear bound from
    # the stiffest principal direction, volumetric tangent K*J in tension
    Gt = np.zeros_like(lb)
    for mu, al in zip(mat.params["mu"], mat.params["alpha"]):
        if mu != 0.0:
            Gt += 0.5 * (mu * al) * lb ** al
    Gt = Gt.max(axis=1) / J
    Kt = K * np.maximum(J, 1.0)
    rho = rho0 / J                                # current density
    c = np.sqrt((Kt + 4.0 * Gt / 3.0) / rho)
    return sig, epsp, c
