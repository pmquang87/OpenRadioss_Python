"""
/DAMP/FREQ_RANGE — Range Damping module for solid and shell elements (M613).

Fortran origin:
  * Parameter computation:
    ``starter/source/general_controls/damping/damping_range_compute_param.F90``
  * Initialization:
    ``starter/source/general_controls/damping/damping_range_init.F90``
  * Solid elements damping:
    ``engine/source/general_controls/damping/damping_range_solid.F90``
  * Shell elements damping (in-plane / membrane):
    ``engine/source/general_controls/damping/damping_range_shell.F90``
  * Shell elements damping (bending moments):
    ``engine/source/general_controls/damping/damping_range_shell_mom.F90``

Theory:
Frequency-range damping models a constant (or nearly constant) damping ratio
over a specified frequency band [f_low, f_high] by superposing 3 Maxwell
viscoelastic elements.

For each Maxwell component j in {1, 2, 3}:
  gv_j = alpha_j * G
  beta_j = 1 / tau_j
  kv_j = alpha_j * E
  betak_j = 1 / tau_j

The stress update follows a recursive exponential decay:
  A_j = exp(-beta_j * dt)
  B_j = 2 * dt * gv_j * exp(-0.5 * beta_j * dt)
  Ak_j = exp(-betak_j * dt)
  Bk_j = dt * kv_j * exp(-0.5 * betak_j * dt)

For shells, the out-of-plane strain rate epsp_zz is determined dynamically
such that the normal viscous stress sigma_zz = 0 (plane stress condition).
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def damping_range_compute_param(
    damp_ratio: float,
    f_low: float,
    f_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 3 Maxwell component parameters (alpha, tau) for frequency range damping.

    Matches ``starter/source/general_controls/damping/damping_range_compute_param.F90``
    subroutine ``damping_range_compute_param`` (lines 47-119).

    Parameters
    ----------
    damp_ratio : float
        Target damping ratio in frequency range (e.g. 0.05 for 5% damping).
    f_low : float
        Lower bound of frequency range in Hz (must be > 0).
    f_high : float
        Upper bound of frequency range in Hz (must be > f_low).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (maxwell_alpha, maxwell_tau): each a 1D float array of length 3.
        alpha gives the dimensionless stiffness ratios, and tau gives the
        relaxation time constants in seconds.
    """
    if f_low <= 0.0:
        raise ValueError(f"f_low must be positive, got {f_low}")
    if f_high <= f_low:
        raise ValueError(f"f_high ({f_high}) must be strictly greater than f_low ({f_low})")
    if damp_ratio <= 0.0:
        raise ValueError(f"damp_ratio must be positive, got {damp_ratio}")

    # Sample points: f_low, geometric mean f_mid, f_high
    # Factor vector [0.965, 1.0, 0.965] matching Fortran lines 82-89
    f_mid = np.sqrt(f_low * f_high)
    freq_sample = np.array([f_low, f_mid, f_high], dtype=np.float64)
    factor = np.array([0.965, 1.0, 0.965], dtype=np.float64)

    # Matrix of single Maxwell component damping ratios at sample points
    # Matrix(i, j) = 4 * damp_ratio * (f_i / f_max) / (2 + 2 * (f_i / f_max)**2)
    # matching Fortran lines 91-97
    matrix = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            f_max = freq_sample[j]
            ratio = freq_sample[i] / f_max
            matrix[i, j] = (4.0 * damp_ratio * ratio) / (2.0 + 2.0 * ratio**2)

    # Invert matrix matching Fortran line 99
    inv_mat = np.linalg.inv(matrix)

    # Solve for e_max matching Fortran lines 103-109
    e_max = np.zeros(3, dtype=np.float64)
    for i in range(3):
        e_fac = 0.0
        for j in range(3):
            e_fac += inv_mat[i, j] * damp_ratio * factor[j]
        e_max[i] = e_fac * damp_ratio

    # Maxwell alpha and tau for each component matching Fortran lines 113-116:
    # alpha_i = 8 * e_max_i^2 + 4 * e_max_i * sqrt(4 * e_max_i^2 + 1)
    # tau_i   = 1 / (sqrt(alpha_i + 1) * 2 * pi * freq_sample_i)
    maxwell_alpha = np.zeros(3, dtype=np.float64)
    maxwell_tau = np.zeros(3, dtype=np.float64)
    for i in range(3):
        e2 = e_max[i] ** 2
        maxwell_alpha[i] = 8.0 * e2 + 4.0 * e_max[i] * np.sqrt(4.0 * e2 + 1.0)
        maxwell_tau[i] = 1.0 / (np.sqrt(maxwell_alpha[i] + 1.0) * 2.0 * np.pi * freq_sample[i])

    return maxwell_alpha, maxwell_tau


def damping_range_solid_subroutine(
    alpha: np.ndarray,
    tau: np.ndarray,
    uvarvis: np.ndarray,
    et: np.ndarray,
    timestep: float,
    young: float,
    shear_modulus: float,
    epspxx: np.ndarray,
    epspyy: np.ndarray,
    epspzz: np.ndarray,
    epspxy: np.ndarray,
    epspyz: np.ndarray,
    epspzx: np.ndarray,
    sv1: np.ndarray,
    sv2: np.ndarray,
    sv3: np.ndarray,
    sv4: np.ndarray,
    sv5: np.ndarray,
    sv6: np.ndarray,
    rho: np.ndarray,
    soundsp: np.ndarray,
) -> None:
    """Direct port of Fortran ``damping_range_solid.F90``.

    Computes viscous damping stresses for solid elements. Updates uvarvis,
    sv1..sv6, and soundsp in place.
    """
    nel = len(epspxx)
    g = 0.0
    rbulk = 0.0

    aa = np.zeros(3, dtype=np.float64)
    bb = np.zeros(3, dtype=np.float64)
    aak = np.zeros(3, dtype=np.float64)
    bbk = np.zeros(3, dtype=np.float64)

    for j in range(3):
        gv_j = alpha[j] * shear_modulus
        beta_j = 1.0 / tau[j]
        kv_j = alpha[j] * young
        betak_j = 1.0 / tau[j]

        g += gv_j
        rbulk += kv_j

        aa[j] = np.exp(-beta_j * timestep)
        bb[j] = 2.0 * timestep * gv_j * np.exp(-0.5 * beta_j * timestep)
        aak[j] = np.exp(-betak_j * timestep)
        bbk[j] = timestep * kv_j * np.exp(-0.5 * betak_j * timestep)

    # Spheric and deviatoric decomposition matching Fortran lines 119-136
    # trace = -(epspxx + epspyy + epspzz)
    trace = -(epspxx + epspyy + epspzz)
    dav = trace / 3.0
    epxx = epspxx + dav
    epyy = epspyy + dav
    epzz = epspzz + dav

    sv1[:] = 0.0
    sv2[:] = 0.0
    sv3[:] = 0.0
    sv4[:] = 0.0
    sv5[:] = 0.0
    sv6[:] = 0.0
    p = np.zeros(nel, dtype=np.float64)

    for j in range(3):
        ii = 7 * j
        h0_1 = uvarvis[:, ii + 0]
        h0_2 = uvarvis[:, ii + 1]
        h0_3 = uvarvis[:, ii + 2]
        h0_4 = uvarvis[:, ii + 3]
        h0_5 = uvarvis[:, ii + 4]
        h0_6 = uvarvis[:, ii + 5]
        hp0 = uvarvis[:, ii + 6]

        h1 = aa[j] * h0_1 + bb[j] * epxx * et
        h2 = aa[j] * h0_2 + bb[j] * epyy * et
        h3 = aa[j] * h0_3 + bb[j] * epzz * et
        h4 = aa[j] * h0_4 + 0.5 * bb[j] * epspxy * et
        h5 = aa[j] * h0_5 + 0.5 * bb[j] * epspyz * et
        h6 = aa[j] * h0_6 + 0.5 * bb[j] * epspzx * et
        hp = aak[j] * hp0 + bbk[j] * trace * et

        uvarvis[:, ii + 0] = h1
        uvarvis[:, ii + 1] = h2
        uvarvis[:, ii + 2] = h3
        uvarvis[:, ii + 3] = h4
        uvarvis[:, ii + 4] = h5
        uvarvis[:, ii + 5] = h6
        uvarvis[:, ii + 6] = hp

        sv1 += h1
        sv2 += h2
        sv3 += h3
        sv4 += h4
        sv5 += h5
        sv6 += h6
        p += hp

    sv1 -= p
    sv2 -= p
    sv3 -= p

    four_over_3 = 4.0 / 3.0
    soundsp[:] = np.sqrt(soundsp**2 + (four_over_3 * g + rbulk) * et / rho)


class DampingRangeSolid:
    """Frequency-range damping manager for 3D solid elements.

    Matches ``engine/source/general_controls/damping/damping_range_solid.F90``.
    Maintains 21 internal state variables per element (3 Maxwell components x 7).
    """

    def __init__(
        self,
        alpha: np.ndarray,
        tau: np.ndarray,
        young: float,
        shear_modulus: float,
        nel: int = 1,
    ):
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.tau = np.asarray(tau, dtype=np.float64)
        if len(self.alpha) != 3 or len(self.tau) != 3:
            raise ValueError("alpha and tau must have length 3")
        self.young = float(young)
        self.shear_modulus = float(shear_modulus)
        self.nel = nel
        self.uvarvis = np.zeros((nel, 21), dtype=np.float64)

    def reset(self, nel: Optional[int] = None) -> None:
        """Reset internal state buffer."""
        if nel is not None:
            self.nel = nel
        self.uvarvis = np.zeros((self.nel, 21), dtype=np.float64)

    def update(
        self,
        timestep: float,
        epsp: np.ndarray,
        rho: np.ndarray,
        soundsp: np.ndarray,
        et: Optional[np.ndarray] = None,
        uvarvis: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute viscous stresses for solid elements.

        Parameters
        ----------
        timestep : float
            Current time step dt.
        epsp : np.ndarray
            Strain rates (nel, 6) or (6,): [xx, yy, zz, xy, yz, zx].
        rho : np.ndarray
            Element density (nel,) or scalar.
        soundsp : np.ndarray
            Element sound speed (nel,) or scalar.
        et : np.ndarray, optional
            Tangent modulus coefficient (nel,), default 1.0.
        uvarvis : np.ndarray, optional
            Custom external buffer (nel, 21). If None, uses internal self.uvarvis.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (sv, soundsp_updated):
            sv is array of shape (nel, 6) containing viscous stresses
            [sv_xx, sv_yy, sv_zz, sv_xy, sv_yz, sv_zx].
        """
        is_1d = (np.ndim(epsp) == 1)
        epsp_arr = np.atleast_2d(epsp)
        nel = epsp_arr.shape[0]

        if uvarvis is None:
            if self.uvarvis.shape[0] != nel:
                self.uvarvis = np.zeros((nel, 21), dtype=np.float64)
            buf = self.uvarvis
        else:
            buf = uvarvis

        rho_arr = np.broadcast_to(np.asarray(rho, dtype=np.float64), (nel,)).copy()
        c_arr = np.broadcast_to(np.asarray(soundsp, dtype=np.float64), (nel,)).copy()
        if et is None:
            et_arr = np.ones(nel, dtype=np.float64)
        else:
            et_arr = np.broadcast_to(np.asarray(et, dtype=np.float64), (nel,))

        sv1 = np.zeros(nel, dtype=np.float64)
        sv2 = np.zeros(nel, dtype=np.float64)
        sv3 = np.zeros(nel, dtype=np.float64)
        sv4 = np.zeros(nel, dtype=np.float64)
        sv5 = np.zeros(nel, dtype=np.float64)
        sv6 = np.zeros(nel, dtype=np.float64)

        damping_range_solid_subroutine(
            alpha=self.alpha,
            tau=self.tau,
            uvarvis=buf,
            et=et_arr,
            timestep=timestep,
            young=self.young,
            shear_modulus=self.shear_modulus,
            epspxx=epsp_arr[:, 0],
            epspyy=epsp_arr[:, 1],
            epspzz=epsp_arr[:, 2],
            epspxy=epsp_arr[:, 3],
            epspyz=epsp_arr[:, 4],
            epspzx=epsp_arr[:, 5],
            sv1=sv1,
            sv2=sv2,
            sv3=sv3,
            sv4=sv4,
            sv5=sv5,
            sv6=sv6,
            rho=rho_arr,
            soundsp=c_arr,
        )

        sv = np.column_stack([sv1, sv2, sv3, sv4, sv5, sv6])
        if is_1d:
            return sv[0], c_arr[0]
        return sv, c_arr


def damping_range_shell_subroutine(
    alpha: np.ndarray,
    tau: np.ndarray,
    uvarv: np.ndarray,
    etse: np.ndarray,
    timestep: float,
    young: float,
    shear_mod: float,
    epspxx: np.ndarray,
    epspyy: np.ndarray,
    epspxy: np.ndarray,
    epspyz: np.ndarray,
    epspzx: np.ndarray,
    sigvxx: np.ndarray,
    sigvyy: np.ndarray,
    sigvxy: np.ndarray,
    sigvyz: np.ndarray,
    sigvzx: np.ndarray,
    rho0: np.ndarray,
    soundsp: np.ndarray,
    off: np.ndarray,
    flag_incr: int = 0,
) -> None:
    """Direct port of Fortran ``damping_range_shell.F90``.

    Computes membrane viscous stresses for shell elements under plane stress
    (sigzz = 0). Updates uvarv, sigvxx..sigvzx, and soundsp in place.
    """
    nel = len(epspxx)
    g = 0.0
    rbulk = 0.0

    aa = np.zeros(3, dtype=np.float64)
    bb = np.zeros(3, dtype=np.float64)
    aak = np.zeros(3, dtype=np.float64)
    bbk = np.zeros(3, dtype=np.float64)

    for j in range(3):
        gv_j = alpha[j] * shear_mod
        beta_j = 1.0 / tau[j]
        kv_j = alpha[j] * young
        betak_j = 1.0 / tau[j]

        g += gv_j
        rbulk += kv_j

        aa[j] = np.exp(-beta_j * timestep)
        bb[j] = 2.0 * timestep * gv_j * np.exp(-0.5 * beta_j * timestep)
        aak[j] = np.exp(-betak_j * timestep)
        bbk[j] = timestep * kv_j * np.exp(-0.5 * betak_j * timestep)

    if flag_incr == 0:
        et = etse.copy()
    else:
        et = 1.0 + etse

    # Compute parameters a1, a2, a3 for epszz_dot under sigzz = 0
    # matching Fortran lines 132-146
    a1 = np.zeros(nel, dtype=np.float64)
    a2 = np.zeros(nel, dtype=np.float64)
    a3 = np.zeros(nel, dtype=np.float64)

    for j in range(3):
        ii = 7 * j
        h0_3 = uvarv[:, ii + 2]
        hp0 = uvarv[:, ii + 6]
        a1 += aa[j] * h0_3 - aak[j] * hp0
        a2 += bb[j] * et
        a3 += bbk[j] * et

    # Out-of-plane strain rate enforcing sig33 = 0 matching Fortran lines 150-154
    two_third = 2.0 / 3.0
    third = 1.0 / 3.0
    denom = np.maximum(1e-20, two_third * a2 + a3)
    fac = 1.0 / denom
    epspzz = fac * (-a1 + (third * a2 - a3) * (epspxx + epspyy))

    # Deviatoric and spherical decomposition matching Fortran lines 157-162
    dav = third * (epspxx + epspyy + epspzz)
    epxx = epspxx - dav
    epyy = epspyy - dav
    epzz = epspzz - dav

    s1 = np.zeros(nel, dtype=np.float64)
    s2 = np.zeros(nel, dtype=np.float64)
    s4 = np.zeros(nel, dtype=np.float64)
    s5 = np.zeros(nel, dtype=np.float64)
    s6 = np.zeros(nel, dtype=np.float64)
    p = np.zeros(nel, dtype=np.float64)

    for j in range(3):
        ii = 7 * j
        h0_1 = uvarv[:, ii + 0]
        h0_2 = uvarv[:, ii + 1]
        h0_3 = uvarv[:, ii + 2]
        h0_4 = uvarv[:, ii + 3]
        h0_5 = uvarv[:, ii + 4]
        h0_6 = uvarv[:, ii + 5]
        hp0 = uvarv[:, ii + 6]

        h1 = aa[j] * h0_1 + bb[j] * epxx * et
        h2 = aa[j] * h0_2 + bb[j] * epyy * et
        h3 = aa[j] * h0_3 + bb[j] * epzz * et
        h4 = aa[j] * h0_4 + 0.5 * bb[j] * epspxy * et
        h5 = aa[j] * h0_5 + 0.5 * bb[j] * epspyz * et
        h6 = aa[j] * h0_6 + 0.5 * bb[j] * epspzx * et
        hp = aak[j] * hp0 + bbk[j] * (-3.0 * dav) * et

        uvarv[:, ii + 0] = h1
        uvarv[:, ii + 1] = h2
        uvarv[:, ii + 2] = h3
        uvarv[:, ii + 3] = h4
        uvarv[:, ii + 4] = h5
        uvarv[:, ii + 5] = h6
        uvarv[:, ii + 6] = hp

        s1 += h1
        s2 += h2
        s4 += h4
        s5 += h5
        s6 += h6
        p += hp

    if flag_incr == 0:
        # Total viscous stress increment (mulawc) matching Fortran lines 206-213
        sigvxx[:] = sigvxx + (s1 - p) * off
        sigvyy[:] = sigvyy + (s2 - p) * off
        sigvxy[:] = sigvxy + s4 * off
        sigvyz[:] = sigvyz + s5 * off
        sigvzx[:] = sigvzx + s6 * off
    else:
        # Incremental formulation (mulawglc) matching Fortran lines 214-236
        s_old1 = uvarv[:, 21].copy()
        s_old2 = uvarv[:, 22].copy()
        s_old4 = uvarv[:, 24].copy()
        s_old5 = uvarv[:, 25].copy()
        s_old6 = uvarv[:, 26].copy()
        p_old = uvarv[:, 27].copy()

        uvarv[:, 21] = s1
        uvarv[:, 22] = s2
        uvarv[:, 24] = s4
        uvarv[:, 25] = s5
        uvarv[:, 26] = s6
        uvarv[:, 27] = p

        sigvxx[:] = sigvxx + (s1 - s_old1 - p + p_old) * off
        sigvyy[:] = sigvyy + (s2 - s_old2 - p + p_old) * off
        sigvxy[:] = sigvxy + (s4 - s_old4) * off
        sigvyz[:] = sigvyz + (s5 - s_old5) * off
        sigvzx[:] = sigvzx + (s6 - s_old6) * off

    soundsp[:] = np.sqrt(soundsp**2 + (g + rbulk) * et / rho0)


def damping_range_shell_mom_subroutine(
    alpha: np.ndarray,
    tau: np.ndarray,
    uvarv: np.ndarray,
    timestep: float,
    young: float,
    shear_mod: float,
    depbxx: np.ndarray,
    depbyy: np.ndarray,
    depbxy: np.ndarray,
    momnxx: np.ndarray,
    momnyy: np.ndarray,
    momnxy: np.ndarray,
    thk0: np.ndarray,
    off: np.ndarray,
    etse: np.ndarray,
) -> None:
    """Direct port of Fortran ``damping_range_shell_mom.F90``.

    Computes viscous damping bending moments for shell elements.
    Updates uvarv and momnxx..momnxy in place.
    """
    nel = len(depbxx)
    dtinv = 1.0 / timestep

    aa = np.zeros(3, dtype=np.float64)
    bb = np.zeros(3, dtype=np.float64)
    aak = np.zeros(3, dtype=np.float64)
    bbk = np.zeros(3, dtype=np.float64)

    for j in range(3):
        gv_j = alpha[j] * shear_mod
        beta_j = 1.0 / tau[j]
        kv_j = alpha[j] * young
        betak_j = 1.0 / tau[j]

        aa[j] = np.exp(-beta_j * timestep)
        bb[j] = 2.0 * timestep * gv_j * np.exp(-0.5 * beta_j * timestep)
        aak[j] = np.exp(-betak_j * timestep)
        bbk[j] = timestep * kv_j * np.exp(-0.5 * betak_j * timestep)

    et = 1.0 + etse

    # Bending transverse contraction parameter matching Fortran lines 113-124
    a2 = np.zeros(nel, dtype=np.float64)
    a3 = np.zeros(nel, dtype=np.float64)
    for j in range(3):
        a2 += bb[j] * et
        a3 += bbk[j] * et

    two_third = 2.0 / 3.0
    third = 1.0 / 3.0
    fac = 1.0 / np.maximum(1e-20, two_third * a2 + a3)
    fac_nu = fac * (third * a2 - a3)

    epspbxx = depbxx * dtinv
    epspbyy = depbyy * dtinv
    epbxy = depbxy * dtinv

    dav = ((1.0 + fac_nu) / 3.0) * (epspbxx + epspbyy)
    epbxx = ((2.0 - fac_nu) / 3.0) * epspbxx - ((1.0 + fac_nu) / 3.0) * epspbyy
    epbyy = ((2.0 - fac_nu) / 3.0) * epspbyy - ((1.0 + fac_nu) / 3.0) * epspbxx

    s1 = np.zeros(nel, dtype=np.float64)
    s2 = np.zeros(nel, dtype=np.float64)
    s3 = np.zeros(nel, dtype=np.float64)
    p = np.zeros(nel, dtype=np.float64)

    # 3 Maxwell components x 4 state vars (h1, h2, h3, hp)
    for j in range(3):
        ii = 4 * j
        h0_1 = uvarv[:, ii + 0]
        h0_2 = uvarv[:, ii + 1]
        h0_3 = uvarv[:, ii + 2]
        hp0 = uvarv[:, ii + 3]

        h1 = aa[j] * h0_1 + bb[j] * epbxx * et
        h2 = aa[j] * h0_2 + bb[j] * epbyy * et
        h3 = aa[j] * h0_3 + 0.5 * bb[j] * epbxy * et
        hp = aak[j] * hp0 + bbk[j] * (-3.0 * dav) * et

        uvarv[:, ii + 0] = h1
        uvarv[:, ii + 1] = h2
        uvarv[:, ii + 2] = h3
        uvarv[:, ii + 3] = hp

        s1 += h1
        s2 += h2
        s3 += h3
        p += hp

    # Incremental formulation for bending moments matching Fortran lines 164-179
    # Old stresses stored at offset 12..15
    s_old1 = uvarv[:, 12].copy()
    s_old2 = uvarv[:, 13].copy()
    s_old3 = uvarv[:, 14].copy()
    p_old = uvarv[:, 15].copy()

    uvarv[:, 12] = s1
    uvarv[:, 13] = s2
    uvarv[:, 14] = s3
    uvarv[:, 15] = p

    thk08 = thk0 / 12.0
    momnxx[:] = momnxx + thk08 * (s1 - s_old1 - p + p_old) * off
    momnyy[:] = momnyy + thk08 * (s2 - s_old2 - p + p_old) * off
    momnxy[:] = momnxy + thk08 * (s3 - s_old3) * off


class DampingRangeShell:
    """Frequency-range damping manager for shell elements.

    Matches ``damping_range_shell.F90`` (membrane stresses) and
    ``damping_range_shell_mom.F90`` (bending moments).
    """

    def __init__(
        self,
        alpha: np.ndarray,
        tau: np.ndarray,
        young: float,
        shear_modulus: float,
        nel: int = 1,
        flag_incr: int = 0,
    ):
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.tau = np.asarray(tau, dtype=np.float64)
        if len(self.alpha) != 3 or len(self.tau) != 3:
            raise ValueError("alpha and tau must have length 3")
        self.young = float(young)
        self.shear_modulus = float(shear_modulus)
        self.nel = nel
        self.flag_incr = flag_incr

        n_mem = 28 if flag_incr != 0 else 21
        self.uvarv_mem = np.zeros((nel, n_mem), dtype=np.float64)
        self.uvarv_mom = np.zeros((nel, 16), dtype=np.float64)

    def reset(self, nel: Optional[int] = None) -> None:
        """Reset internal membrane and bending buffers."""
        if nel is not None:
            self.nel = nel
        n_mem = 28 if self.flag_incr != 0 else 21
        self.uvarv_mem = np.zeros((self.nel, n_mem), dtype=np.float64)
        self.uvarv_mom = np.zeros((self.nel, 16), dtype=np.float64)

    def update_membrane(
        self,
        timestep: float,
        epsp: np.ndarray,
        rho: np.ndarray,
        soundsp: np.ndarray,
        etse: Optional[np.ndarray] = None,
        off: Optional[np.ndarray] = None,
        flag_incr: Optional[int] = None,
        uvarv: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute membrane viscous stresses for shell elements under plane stress.

        Parameters
        ----------
        timestep : float
            Time step dt.
        epsp : np.ndarray
            Membrane strain rates: (nel, 5) or (5,) [xx, yy, xy, yz, zx].
        rho : np.ndarray
            Density (nel,) or scalar.
        soundsp : np.ndarray
            Sound speed (nel,) or scalar.
        etse : np.ndarray, optional
            Tangent modulus coefficient (nel,), default 1.0 (for flag_incr=0)
            or 0.0 (for flag_incr=1).
        off : np.ndarray, optional
            On/off flag (nel,), default 1.0.
        flag_incr : int, optional
            0 for total (mulawc), 1 for incremental (mulawglc).
        uvarv : np.ndarray, optional
            External state buffer.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (sigv, soundsp_updated):
            sigv is array of shape (nel, 5) containing [xx, yy, xy, yz, zx].
        """
        is_1d = (np.ndim(epsp) == 1)
        epsp_arr = np.atleast_2d(epsp)
        nel = epsp_arr.shape[0]

        f_incr = self.flag_incr if flag_incr is None else flag_incr
        n_mem = 28 if f_incr != 0 else 21

        if uvarv is None:
            if self.uvarv_mem.shape[0] != nel or self.uvarv_mem.shape[1] != n_mem:
                self.uvarv_mem = np.zeros((nel, n_mem), dtype=np.float64)
            buf = self.uvarv_mem
        else:
            buf = uvarv

        rho_arr = np.broadcast_to(np.asarray(rho, dtype=np.float64), (nel,)).copy()
        c_arr = np.broadcast_to(np.asarray(soundsp, dtype=np.float64), (nel,)).copy()

        if etse is None:
            etse_arr = np.ones(nel, dtype=np.float64) if f_incr == 0 else np.zeros(nel, dtype=np.float64)
        else:
            etse_arr = np.broadcast_to(np.asarray(etse, dtype=np.float64), (nel,))

        if off is None:
            off_arr = np.ones(nel, dtype=np.float64)
        else:
            off_arr = np.broadcast_to(np.asarray(off, dtype=np.float64), (nel,))

        sigvxx = np.zeros(nel, dtype=np.float64)
        sigvyy = np.zeros(nel, dtype=np.float64)
        sigvxy = np.zeros(nel, dtype=np.float64)
        sigvyz = np.zeros(nel, dtype=np.float64)
        sigvzx = np.zeros(nel, dtype=np.float64)

        damping_range_shell_subroutine(
            alpha=self.alpha,
            tau=self.tau,
            uvarv=buf,
            etse=etse_arr,
            timestep=timestep,
            young=self.young,
            shear_mod=self.shear_modulus,
            epspxx=epsp_arr[:, 0],
            epspyy=epsp_arr[:, 1],
            epspxy=epsp_arr[:, 2],
            epspyz=epsp_arr[:, 3],
            epspzx=epsp_arr[:, 4],
            sigvxx=sigvxx,
            sigvyy=sigvyy,
            sigvxy=sigvxy,
            sigvyz=sigvyz,
            sigvzx=sigvzx,
            rho0=rho_arr,
            soundsp=c_arr,
            off=off_arr,
            flag_incr=f_incr,
        )

        sigv = np.column_stack([sigvxx, sigvyy, sigvxy, sigvyz, sigvzx])
        if is_1d:
            return sigv[0], c_arr[0]
        return sigv, c_arr

    def update_bending(
        self,
        timestep: float,
        depb: np.ndarray,
        thk0: np.ndarray,
        etse: Optional[np.ndarray] = None,
        off: Optional[np.ndarray] = None,
        uvarv_mom: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute viscous bending moments for shell elements.

        Parameters
        ----------
        timestep : float
            Time step dt.
        depb : np.ndarray
            Bending strain increments: (nel, 3) or (3,) [xx, yy, xy].
        thk0 : np.ndarray
            Element thickness parameter (nel,) or scalar.
        etse : np.ndarray, optional
            Tangent modulus coefficient (nel,), default 0.0.
        off : np.ndarray, optional
            On/off flag (nel,), default 1.0.
        uvarv_mom : np.ndarray, optional
            External bending buffer (nel, 16).

        Returns
        -------
        np.ndarray
            Viscous moments: (nel, 3) containing [mom_xx, mom_yy, mom_xy].
        """
        is_1d = (np.ndim(depb) == 1)
        depb_arr = np.atleast_2d(depb)
        nel = depb_arr.shape[0]

        if uvarv_mom is None:
            if self.uvarv_mom.shape[0] != nel:
                self.uvarv_mom = np.zeros((nel, 16), dtype=np.float64)
            buf = self.uvarv_mom
        else:
            buf = uvarv_mom

        thk_arr = np.broadcast_to(np.asarray(thk0, dtype=np.float64), (nel,))
        if etse is None:
            etse_arr = np.zeros(nel, dtype=np.float64)
        else:
            etse_arr = np.broadcast_to(np.asarray(etse, dtype=np.float64), (nel,))

        if off is None:
            off_arr = np.ones(nel, dtype=np.float64)
        else:
            off_arr = np.broadcast_to(np.asarray(off, dtype=np.float64), (nel,))

        momnxx = np.zeros(nel, dtype=np.float64)
        momnyy = np.zeros(nel, dtype=np.float64)
        momnxy = np.zeros(nel, dtype=np.float64)

        damping_range_shell_mom_subroutine(
            alpha=self.alpha,
            tau=self.tau,
            uvarv=buf,
            timestep=timestep,
            young=self.young,
            shear_mod=self.shear_modulus,
            depbxx=depb_arr[:, 0],
            depbyy=depb_arr[:, 1],
            depbxy=depb_arr[:, 2],
            momnxx=momnxx,
            momnyy=momnyy,
            momnxy=momnxy,
            thk0=thk_arr,
            off=off_arr,
            etse=etse_arr,
        )

        mom = np.column_stack([momnxx, momnyy, momnxy])
        if is_1d:
            return mom[0]
        return mom
