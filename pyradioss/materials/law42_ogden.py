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

Implicit CONSISTENT tangent (M14) — the spectral spatial elasticity
--------------------------------------------------------------------
``consistent_solid_tangent`` returns the (m, 6, 6) SPATIAL tangent
modulus c for the implicit element stiffness K_c = int B^T c B dV at the
CURRENT configuration, paired with the geometric term K_geo =
int gradN . sigma . gradN dV the assembler already adds — the standard
updated-Lagrangian split (Bonet & Wood, *Nonlinear Continuum Mechanics
for Finite Element Analysis*, ch. 8: K = K_c + K_sigma with c the
elasticity tensor for the Lie (Truesdell) rate of Kirchhoff stress / J).

What the ORIGINAL does instead (verified in sigeps42.F): its implicit
branch (IMPL_S > 0) computes only the SCALAR stiffness ratio ET =
max_i(tangent shear ratio) that the element KE routines use to scale a
LINEAR-elastic modulus matrix — a secant-stiffness modified Newton. The
port deviates deliberately (the M13 IMP_KPRES pattern: Newton quality
comes from consistency with the residual actually iterated) and builds
the EXACT spectral tangent of its own stress expression.

Derivation. The port's stress (above) is the exact derivative of

    W = sum_p mu_p/alpha_p (lb1^a + lb2^a + lb3^a - 3) + K/2 (J-1)^2

in principal logarithmic stretches: tau_a = dW/d ln(lambda_a) =
S_a - mean(S) + K J (J-1), sigma_a = tau_a / J (checked term by term
against solid_update — the tangent is the derivative of exactly what the
residual assembles). For an ISOTROPIC hyperelastic material the spatial
tensor in the principal frame {n_a} of B is (Bonet & Wood §6.6, the
principal-direction elasticity):

    c_aabb = (1/J) beta_ab - 2 sigma_a delta_ab,
    beta_ab = d tau_a / d ln(lambda_b)
            = A_a delta_ab - (A_a + A_b)/3 + sum_c A_c / 9 + K J (2J - 1),
    A_a = sum_p mu_p alpha_p lb_a^alpha_p,

    c_abab = (sigma_a lambda_b^2 - sigma_b lambda_a^2)
             / (lambda_a^2 - lambda_b^2)          (a != b, distinct)

with all minor symmetries (c_abab = c_abba = c_baab). EQUAL-STRETCH
LIMIT (the coalescent-eigenvalue case every hydrostatic/uniaxial state
hits): the shear coefficient is 0/0; differentiating the quotient
(L'Hopital in lambda_b^2, using isotropy sigma_b = f(lambda_b; ...) =
the same principal function with arguments swapped) gives

    c_abab -> (beta_aa - beta_ab)/(2J) - sigma_a
            = (c_aaaa - c_aabb)/2              (lambda_a = lambda_b)

in THIS convention — the c_aabb entries above already carry the
-2 sigma_a diagonal shift, which is why the familiar textbook form
"(c_aaaa - c_aabb)/2 - sigma_a" (stated for the UNSHIFTED beta/J
entries — Bonet & Wood; Simo & Taylor 1991) must not be applied
verbatim: doing so subtracts sigma_a twice (caught by the coalescent
FD Truesdell-identity validation, which probes a hydrostatic state on
purpose). The port switches to the limit when |lambda_a^2 - lambda_b^2|
falls below 1e-6 of their sum — an ANALYTIC treatment, not an
eigenvalue perturbation, so the tangent stays exact and symmetric
through the coalescence. The tensor is
rotated to global axes through the eigenvectors of B and packed in the
port's Voigt order [xx, yy, zz, xy, yz, zx] with ENGINEERING shear
columns (the factor-2 of gamma cancels the minor-symmetry pair sum, so
D[I, J>=3] = c_ijkl verbatim).

The tangent needs the deformation gradient of the TRIAL configuration —
the element kernels compute it exactly from the stored initial gradients
(``needs_defgrad``) and pass it through the ``extra`` hook of
``materials.solid_tangent``. Under implicit, LAW42 REQUIRES /IMPL/NONLIN
(checked loudly by the drivers): on the frozen small-strain frame the
total-form stress never sees the trial displacement at all — F is a
function of the reference coordinates there — so a "small-displacement
hyperelastic" run would iterate a residual that cannot move.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20


def _principal_kirchhoff_dev(mat, lb: np.ndarray):
    """S_i = sum_p mu_p lb_i^alpha_p per principal direction; lb (m, 3)."""
    S = np.zeros_like(lb)
    mus = mat.params.get("mu", [])
    alphas = mat.params.get("alpha", [])
    for mu, al in zip(mus, alphas):
        if mu != 0.0:
            S += mu * (lb ** al)
    return S


def solid_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray, dt: float, extra: dict | None = None):
    """Hyperelastic stress from the deformation gradient (module doc).

    Returns (sig, epsp, c) with c the per-element nonlinear sound speed
    used by the kernel for the time step."""
    m = sig.shape[0]
    if m == 0:
        return sig, epsp, np.empty(0)

    if extra is not None and "F" in extra:
        F = extra["F"]
    else:
        # Fallback: construct F from deps if available, else identity
        F = np.tile(np.eye(3), (m, 1, 1))
        if deps is not None and deps.shape[0] == m:
            F[:, 0, 0] += deps[:, 0]
            F[:, 1, 1] += deps[:, 1]
            F[:, 2, 2] += deps[:, 2]
            F[:, 0, 1] += 0.5 * deps[:, 3]
            F[:, 1, 0] += 0.5 * deps[:, 3]
            F[:, 1, 2] += 0.5 * deps[:, 4]
            F[:, 2, 1] += 0.5 * deps[:, 4]
            F[:, 0, 2] += 0.5 * deps[:, 5]
            F[:, 2, 0] += 0.5 * deps[:, 5]

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
    mus = mat.params.get("mu", [])
    alphas = mat.params.get("alpha", [])
    for mu, al in zip(mus, alphas):
        if mu != 0.0:
            Gt += 0.5 * (mu * al) * (lb ** al)
    Gt = Gt.max(axis=1) / J
    Kt = K * np.maximum(J, 1.0)
    rho = rho0 / J                                # current density
    c = np.sqrt(np.maximum((Kt + 4.0 * Gt / 3.0) / rho, EM20))
    return sig, epsp, c


def shell_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp: np.ndarray, dt: float, extra: dict | None = None):
    """LAW42 Ogden hyperelasticity is defined for 3D solid continuum elements
    only (Fortran origin: engine/source/materials/mat/mat042/sigeps42.F)."""
    raise NotImplementedError(
        f"Material /MAT/LAW42 (Ogden hyperelasticity) is currently supported for "
        f"3D solid elements only (Fortran origin: sigeps42.F)."
    )


# ----------------------------------------------------------------------------
# Implicit consistent tangent (M14) — see the module docstring for the
# derivation, the equal-stretch limit and the documented deviation from
# sigeps42.F's scalar-ET modified Newton.
# ----------------------------------------------------------------------------

#: Voigt order of the port's solid B-operator: rows [xx, yy, zz, xy, yz, zx]
_VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


def consistent_solid_tangent(mat, F):
    """(m, 6, 6) spatial tangent modulus c at deformation gradients ``F``
    (m, 3, 3) — the exact linearization of ``solid_update``'s Cauchy
    stress for the K_c = int B^T c B dV + K_geo pairing (module
    docstring). Symmetric; engineering-shear Voigt convention."""
    m = F.shape[0]
    if m == 0:
        return np.empty((0, 6, 6))
    K = mat.K
    J = np.maximum(np.linalg.det(F), EM20)
    B = np.einsum("nab,ncb->nac", F, F)
    lam2, N = np.linalg.eigh(B)                   # ascending, orthonormal
    lam2 = np.maximum(lam2, EM20)
    lam = np.sqrt(lam2)
    lb = lam * (J ** (-1.0 / 3.0))[:, None]

    # principal Cauchy stresses + the hardening functions A_a (docstring)
    S = _principal_kirchhoff_dev(mat, lb)
    A = np.zeros_like(lb)
    mus = mat.params.get("mu", [])
    alphas = mat.params.get("alpha", [])
    for mu, al in zip(mus, alphas):
        if mu != 0.0:
            A += (mu * al) * (lb ** al)
    tau = S - S.mean(axis=1)[:, None] + (K * J * (J - 1.0))[:, None]
    sigp = tau / J[:, None]

    # beta_ab = d tau_a / d ln(lambda_b) (symmetric), then the (aa,bb)
    # block of c in the principal frame
    Asum = A.sum(axis=1)
    beta = (-(A[:, :, None] + A[:, None, :]) / 3.0
            + (Asum / 9.0 + K * J * (2.0 * J - 1.0))[:, None, None])
    ii = np.arange(3)
    beta[:, ii, ii] += A
    c_diag = beta / J[:, None, None]              # c_aabb before the -2sig
    c_diag[:, ii, ii] -= 2.0 * sigp

    # shear coefficients c_abab, with the equal-stretch L'Hopital limit
    shear = np.zeros((m, 3, 3))
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            num = sigp[:, a] * lam2[:, b] - sigp[:, b] * lam2[:, a]
            den = lam2[:, a] - lam2[:, b]
            close = np.abs(den) <= 1e-6 * (lam2[:, a] + lam2[:, b])
            # L'Hopital limit in THIS c convention: differentiating the
            # quotient gives lim = (beta_aa - beta_ab)/(2J) - sigma_a,
            # which in the stored entries (c_diag already carries the
            # -2 sigma_a diagonal shift) is exactly
            # (c_aaaa - c_aabb)/2 — NO further -sigma_a (textbook forms
            # quoting "... - sigma_a" define c_aabb WITHOUT the shift;
            # verified against the FD Truesdell identity at coalescent
            # stretches in tests/test_m14_implgen.py)
            lim = 0.5 * (c_diag[:, a, a] - c_diag[:, a, b])
            with np.errstate(divide="ignore", invalid="ignore"):
                gen = num / np.where(close, 1.0, den)
            shear[:, a, b] = np.where(close, lim, gen)

    # assemble the (m,3,3,3,3) tensor in the principal frame and rotate:
    # c = sum_ab c_aabb Na Na Nb Nb + sum_{a!=b} c_abab (NaNbNaNb + NaNbNbNa)
    n_a = N.transpose(0, 2, 1)                    # (m, 3(principal), 3)
    c4 = np.zeros((m, 3, 3, 3, 3))
    for a in range(3):
        for b in range(3):
            Pa = np.einsum("mi,mj->mij", n_a[:, a], n_a[:, a])
            Pb = np.einsum("mi,mj->mij", n_a[:, b], n_a[:, b])
            c4 += c_diag[:, a, b, None, None, None, None] \
                * np.einsum("mij,mkl->mijkl", Pa, Pb)
            if a != b:
                Qab = np.einsum("mi,mj->mij", n_a[:, a], n_a[:, b])
                Qba = np.einsum("mi,mj->mij", n_a[:, b], n_a[:, a])
                c4 += shear[:, a, b, None, None, None, None] * (
                    np.einsum("mij,mkl->mijkl", Qab, Qab)
                    + np.einsum("mij,mkl->mijkl", Qab, Qba))

    # Voigt 6x6, engineering shear (docstring: D[I,J] = c_ijkl verbatim)
    D = np.empty((m, 6, 6))
    for I, (i, j) in enumerate(_VOIGT):
        for Jc, (k, ell) in enumerate(_VOIGT):
            D[:, I, Jc] = c4[:, i, j, k, ell]
    return D
