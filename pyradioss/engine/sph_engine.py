# pyradioss/engine/sph_engine.py
# Port of engine/source/elements/sph/*.F
"""
Smoothed Particle Hydrodynamics (SPH) Physics Engine.

Faithful port of OpenRadioss Fortran source:
  - weight.F  (lines 33-66: WEIGHT0, lines 78-119: WEIGHT1): cubic B-spline kernel and gradient
  - spdens.F  (lines 101-167, 373-376): density summation rho_a = sum_b m_b W_ab
  - spdefo3.F (lines 63-70) & spdens.F (lines 188-202): rate of deformation D_ab
  - sppro3.F  (lines 72-118) & spforcp.F (lines 240-304): SPH forces (pressure + Monaghan 1992 artificial viscosity)
  - spstab.F  (lines 80-129, 212-250) & spforcp.F (lines 259-277): Monaghan-Gray tensile instability stabilization
  - spcompl.F (lines 156-202, 266-325): zeroth-order Shepard and first-order MLS kernel gradient corrections
  - sphreq.F  (lines 34-40) & mdtsph.F (lines 96-135): critical time step dt = CFL * h / c_s
"""

import numpy as np
from scipy.spatial import cKDTree


def cubic_bspline_kernel(r, h):
    """Cubic B-spline SPH kernel W(r, h) in 3D.

    Port of OpenRadioss engine/source/elements/sph/weight.F lines 33-66 (WEIGHT0).

    In 3D with compact support radius 2h:
      W(r, h) = (1 / (pi * h^3)) * (1 - 1.5*(r/h)^2 + 0.75*(r/h)^3)   if 0 <= r <= h
              = (1 / (4 * pi * h^3)) * (2 - r/h)^3                    if h < r <= 2h
              = 0                                                     if r > 2h

    Args:
        r: Distance between particles (float or np.ndarray >= 0).
        h: Smoothing length (float or np.ndarray > 0).

    Returns:
        Kernel value W(r, h) (float or np.ndarray).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    q = r_arr / h_arr
    inv_pi = 1.0 / np.pi
    h3 = h_arr ** 3

    w = np.zeros_like(q, dtype=np.float64)

    # Branch 1: 0 <= q <= 1 (weight.F lines 52-56)
    m1 = (q >= 0.0) & (q <= 1.0)
    if np.any(m1):
        q1 = q[m1]
        h3_1 = h3[m1] if np.ndim(h_arr) > 0 else h3
        w[m1] = (1.0 - 1.5 * q1 * q1 + 0.75 * q1 * q1 * q1) * inv_pi / h3_1

    # Branch 2: 1 < q <= 2 (weight.F lines 57-62)
    m2 = (q > 1.0) & (q <= 2.0)
    if np.any(m2):
        q2 = q[m2]
        h3_2 = h3[m2] if np.ndim(h_arr) > 0 else h3
        w[m2] = 0.25 * (2.0 - q2) ** 3 * inv_pi / h3_2

    # Branch 3: q > 2 -> 0.0 (weight.F line 63)
    if np.ndim(r) == 0 and np.ndim(h) == 0:
        return float(w)
    return w


def cubic_bspline_grad(rij, h):
    """Gradient of cubic B-spline kernel grad_a W_ab with respect to r_a.

    Port of OpenRadioss engine/source/elements/sph/weight.F lines 78-119 (WEIGHT1).

    grad_a W_ab = (dW/dr / r) * (x_a - x_b) = WPRIMR * rij
    where WPRIMR is:
      (-3 + 2.25 * (r/h)) / (pi * h^5)             if 0 <= r <= h
      -0.75 * (2 - r/h)^2 / (pi * h^4 * r)         if h < r <= 2h
      0                                            if r > 2h

    Args:
        rij: Vector x_i - x_j (shape (3,) or (N, 3)).
        h: Smoothing length (float or np.ndarray of shape (N,)).

    Returns:
        grad_i W_ij (shape matching rij: (3,) or (N, 3)).
    """
    rij_arr = np.asarray(rij, dtype=np.float64)
    is_1d = (rij_arr.ndim == 1)
    if is_1d:
        rij_arr = rij_arr.reshape(1, 3)

    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 0:
        h_arr = np.full(len(rij_arr), float(h_arr))

    r = np.linalg.norm(rij_arr, axis=1)
    wprimr = np.zeros(len(rij_arr), dtype=np.float64)
    inv_pi = 1.0 / np.pi

    # Branch 1: r <= h (weight.F lines 97-103)
    m1 = (r <= h_arr)
    if np.any(m1):
        r1 = r[m1]
        h1 = h_arr[m1]
        rh = r1 / h1
        ih3 = inv_pi / (h1 ** 3)
        wprimr[m1] = (-3.0 + 2.25 * rh) * ih3 / (h1 * h1)

    # Branch 2: h < r <= 2h (weight.F lines 104-110)
    m2 = (r > h_arr) & (r <= 2.0 * h_arr)
    if np.any(m2):
        r2 = r[m2]
        h2 = h_arr[m2]
        rhm = (2.0 - r2 / h2) / h2
        rhm2 = rhm * rhm * inv_pi
        wprimr[m2] = -0.75 * rhm2 / (h2 * h2 * np.maximum(r2, 1e-30))

    # WGRAD = WPRIMR * rij (weight.F lines 115-117)
    grad = wprimr[:, None] * rij_arr
    if is_1d:
        return grad[0]
    return grad


def wendland_c2_kernel(r, h):
    """Wendland C2 smoothing kernel in 3D with compact support radius 2h.

    Normalisation in 3D:
      alpha_3D = 21 / (16 * pi * h^3)
    For q = r / h:
      W(r, h) = alpha_3D * (1 - q/2)^4 * (1 + 2q)    if 0 <= q <= 2
              = 0                                    if q > 2

    Args:
        r: Inter-particle distance (float or np.ndarray >= 0).
        h: Smoothing length (float or np.ndarray > 0).

    Returns:
        Kernel value W(r, h).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    q = r_arr / h_arr
    alpha = 21.0 / (16.0 * np.pi * (h_arr ** 3))

    w = np.zeros_like(q, dtype=np.float64)
    mask = (q >= 0.0) & (q <= 2.0)
    if np.any(mask):
        qm = q[mask]
        alpha_m = alpha[mask] if np.ndim(h_arr) > 0 else alpha
        term1 = (1.0 - 0.5 * qm) ** 4
        term2 = 1.0 + 2.0 * qm
        w[mask] = alpha_m * term1 * term2

    if np.ndim(r) == 0 and np.ndim(h) == 0:
        return float(w)
    return w


def wendland_c2_grad(rij, h):
    """Gradient of Wendland C2 kernel in 3D with respect to r_i.

    grad_i W_ij = (1/r * dW/dr) * rij
    where dW/dr = - (105 / (16 * pi * h^4)) * q * (1 - q/2)^3
    and 1/r * dW/dr = - (105 / (16 * pi * h^5)) * (1 - q/2)^3   for 0 <= q <= 2

    Args:
        rij: Vector x_i - x_j (shape (3,) or (N, 3)).
        h: Smoothing length (float or np.ndarray of shape (N,)).

    Returns:
        grad_i W_ij (shape matching rij: (3,) or (N, 3)).
    """
    rij_arr = np.asarray(rij, dtype=np.float64)
    is_1d = (rij_arr.ndim == 1)
    if is_1d:
        rij_arr = rij_arr.reshape(1, 3)

    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 0:
        h_arr = np.full(len(rij_arr), float(h_arr))

    r = np.linalg.norm(rij_arr, axis=1)
    q = r / h_arr
    coeff = np.zeros(len(rij_arr), dtype=np.float64)

    mask = (q >= 0.0) & (q <= 2.0)
    if np.any(mask):
        qm = q[mask]
        hm = h_arr[mask]
        factor = -105.0 / (16.0 * np.pi * (hm ** 5))
        coeff[mask] = factor * ((1.0 - 0.5 * qm) ** 3)

    grad = coeff[:, None] * rij_arr
    if is_1d:
        return grad[0]
    return grad


def quintic_spline_kernel(r, h):
    """Quintic B-spline SPH kernel in 3D with compact support radius 3h.

    Normalisation in 3D:
      alpha_3D = 1 / (120 * pi * h^3)
    For q = r / h:
      W(r, h) = alpha_3D * ((3-q)^5 - 6*(2-q)^5 + 15*(1-q)^5)  if 0 <= q <= 1
              = alpha_3D * ((3-q)^5 - 6*(2-q)^5)               if 1 < q <= 2
              = alpha_3D * (3-q)^5                             if 2 < q <= 3
              = 0                                              if q > 3

    Args:
        r: Inter-particle distance (float or np.ndarray >= 0).
        h: Smoothing length (float or np.ndarray > 0).

    Returns:
        Kernel value W(r, h).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    q = r_arr / h_arr
    alpha = 1.0 / (120.0 * np.pi * (h_arr ** 3))

    w = np.zeros_like(q, dtype=np.float64)

    m1 = (q >= 0.0) & (q <= 1.0)
    if np.any(m1):
        q1 = q[m1]
        a1 = alpha[m1] if np.ndim(h_arr) > 0 else alpha
        w[m1] = a1 * ((3.0 - q1)**5 - 6.0 * (2.0 - q1)**5 + 15.0 * (1.0 - q1)**5)

    m2 = (q > 1.0) & (q <= 2.0)
    if np.any(m2):
        q2 = q[m2]
        a2 = alpha[m2] if np.ndim(h_arr) > 0 else alpha
        w[m2] = a2 * ((3.0 - q2)**5 - 6.0 * (2.0 - q2)**5)

    m3 = (q > 2.0) & (q <= 3.0)
    if np.any(m3):
        q3 = q[m3]
        a3 = alpha[m3] if np.ndim(h_arr) > 0 else alpha
        w[m3] = a3 * ((3.0 - q3)**5)

    if np.ndim(r) == 0 and np.ndim(h) == 0:
        return float(w)
    return w


def quintic_spline_grad(rij, h):
    """Gradient of Quintic B-spline SPH kernel in 3D with respect to r_i.

    Args:
        rij: Vector x_i - x_j (shape (3,) or (N, 3)).
        h: Smoothing length (float or np.ndarray of shape (N,)).

    Returns:
        grad_i W_ij (shape matching rij: (3,) or (N, 3)).
    """
    rij_arr = np.asarray(rij, dtype=np.float64)
    is_1d = (rij_arr.ndim == 1)
    if is_1d:
        rij_arr = rij_arr.reshape(1, 3)

    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 0:
        h_arr = np.full(len(rij_arr), float(h_arr))

    r = np.linalg.norm(rij_arr, axis=1)
    q = r / h_arr
    alpha = 1.0 / (120.0 * np.pi * (h_arr ** 3))

    dw_dq = np.zeros(len(rij_arr), dtype=np.float64)

    m1 = (q >= 0.0) & (q <= 1.0)
    if np.any(m1):
        q1 = q[m1]
        dw_dq[m1] = -5.0 * (3.0 - q1)**4 + 30.0 * (2.0 - q1)**4 - 75.0 * (1.0 - q1)**4

    m2 = (q > 1.0) & (q <= 2.0)
    if np.any(m2):
        q2 = q[m2]
        dw_dq[m2] = -5.0 * (3.0 - q2)**4 + 30.0 * (2.0 - q2)**4

    m3 = (q > 2.0) & (q <= 3.0)
    if np.any(m3):
        q3 = q[m3]
        dw_dq[m3] = -5.0 * (3.0 - q3)**4

    r_safe = np.maximum(r, 1e-30)
    coeff = (alpha / (h_arr * r_safe)) * dw_dq
    coeff[r < 1e-30] = 0.0

    grad = coeff[:, None] * rij_arr
    if is_1d:
        return grad[0]
    return grad


def sph_shepard_correction(pos, h_arr, mass, rho):
    """Compute Shepard (order 0) kernel correction factors c_i = sum_j V_j W_ij.

    Port of OpenRadioss engine/source/elements/sph/spcompl.F lines 156-202 (NZERO).
    Restores zeroth-order consistency: sum_j V_j W_ij^corr = 1.0.

    Args:
        pos: Particle positions, shape (N, 3).
        h_arr: Smoothing lengths, shape (N,) or float.
        mass: Particle masses, shape (N,) or float.
        rho: Particle densities, shape (N,) or float.

    Returns:
        c_i: Shepard normalisation factor for each particle, shape (N,).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    n = len(pos_arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))
    r_rho = np.asarray(rho, dtype=np.float64)
    if r_rho.ndim == 0:
        r_rho = np.full(n, float(r_rho))
    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    vols = m / np.maximum(r_rho, 1e-20)
    # Self-contribution
    c = vols * (1.0 / (np.pi * (h ** 3)))

    if n <= 1:
        return c

    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=2.0 * float(np.max(h)), output_type='ndarray')
    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)
        hij = 0.5 * (h[i_idx] + h[j_idx])
        valid = r <= 2.0 * hij
        if np.any(valid):
            iv = i_idx[valid]
            jv = j_idx[valid]
            w = cubic_bspline_kernel(r[valid], hij[valid])
            np.add.at(c, iv, vols[jv] * w)
            np.add.at(c, jv, vols[iv] * w)

    return np.maximum(c, 1e-12)


def sph_mls_gradient_correction(pos, h_arr, mass, rho):
    """Compute Moving Least Squares (order 1) gradient correction matrix M_i.

    Port of OpenRadioss engine/source/elements/sph/spcompl.F lines 266-325 (NUN).
    Correction matrix:
      M_i = - sum_j V_j (grad_i W_ij) (x) (x_i - x_j)
    Corrected gradient:
      grad_tilde_i W_ij = M_i^{-1} grad_i W_ij
    Restores linear consistency: sum_j V_j (x_j - x_i) . grad_tilde_i W_ij = I (3x3 identity).

    Args:
        pos: Particle positions, shape (N, 3).
        h_arr: Smoothing lengths, shape (N,) or float.
        mass: Particle masses, shape (N,) or float.
        rho: Particle densities, shape (N,) or float.

    Returns:
        M_inv: Inverse correction matrix array, shape (N, 3, 3).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    n = len(pos_arr)
    if n == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))
    r_rho = np.asarray(rho, dtype=np.float64)
    if r_rho.ndim == 0:
        r_rho = np.full(n, float(r_rho))
    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    vols = m / np.maximum(r_rho, 1e-20)
    M = np.zeros((n, 3, 3), dtype=np.float64)

    if n <= 1:
        return np.repeat(np.eye(3)[None, :, :], n, axis=0)

    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=2.0 * float(np.max(h)), output_type='ndarray')
    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)
        hij = 0.5 * (h[i_idx] + h[j_idx])
        valid = (r <= 2.0 * hij) & (r > 1e-30)
        if np.any(valid):
            iv = i_idx[valid]
            jv = j_idx[valid]
            rij_v = rij[valid]
            hij_v = hij[valid]
            grad_w = cubic_bspline_grad(rij_v, hij_v)

            outer_i = grad_w[:, :, None] * rij_v[:, None, :]
            term_i = -vols[jv, None, None] * outer_i
            term_j = -vols[iv, None, None] * outer_i

            np.add.at(M, iv, term_i)
            np.add.at(M, jv, term_j)

    M_inv = np.zeros_like(M)
    for i in range(n):
        mat = M[i]
        det = np.linalg.det(mat)
        if abs(det) > 1e-4:
            M_inv[i] = np.linalg.inv(mat)
        else:
            reg = mat + 1e-3 * np.eye(3)
            det_reg = np.linalg.det(reg)
            if abs(det_reg) > 1e-6:
                M_inv[i] = np.linalg.inv(reg)
            else:
                M_inv[i] = np.eye(3)

    return M_inv


def sph_tensile_stabilization(pos, vel, mass, rho, stress, h_arr, zstab=1.0, dp=None, dt=1e-6):
    """Compute Monaghan-Gray tensile instability stabilization forces.

    Faithful port of OpenRadioss Fortran:
      - spstab.F lines 80-129 (SPSTABW): STAB(7, N) = zstab / max(1e-30, W0(dp)^4)
      - spstab.F lines 212-250 (SPSTABS): Principal tensile stress projection C_ij
      - spforcp.F lines 259-277: Stabilization pair forces T_ij

    When particles are under tensile stress (sigma_k > 0), standard SPH exhibits
    unphysical particle clumping / clustering (tensile instability). This algorithm
    introduces an artificial repulsive force along the tensile principal directions
    scaled by (W(r)/W(dp))^4.

    Args:
        pos: Particle positions, shape (N, 3).
        vel: Particle velocities, shape (N, 3).
        mass: Particle masses, shape (N,) or float.
        rho: Particle densities, shape (N,) or float.
        stress: Particle Cauchy stress tensors, shape (N, 3, 3) or (N, 6) [xx, yy, zz, xy, yz, zx].
        h_arr: Smoothing lengths, shape (N,) or float.
        zstab: Tensile stabilization parameter (default 1.0, from /PROP/SPH ZSTAB).
        dp: Initial particle spacing (if None, defaults to 2/3 * h as in spstab.F line 123).
        dt: Time step increment for energy accounting.

    Returns:
        f_stab: Stabilization force array, shape (N, 3).
        w_stab: Numerical work / energy done by stabilization forces.
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    n = len(pos_arr)
    if n <= 1 or zstab <= 0.0:
        return np.zeros((n, 3), dtype=np.float64), 0.0

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))
    r_rho = np.asarray(rho, dtype=np.float64)
    if r_rho.ndim == 0:
        r_rho = np.full(n, float(r_rho))
    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    # Convert stress to (N, 3, 3)
    stress_arr = np.asarray(stress, dtype=np.float64)
    if stress_arr.ndim == 2 and stress_arr.shape[1] == 6:
        sig_tensor = np.zeros((n, 3, 3), dtype=np.float64)
        sig_tensor[:, 0, 0] = stress_arr[:, 0]
        sig_tensor[:, 1, 1] = stress_arr[:, 1]
        sig_tensor[:, 2, 2] = stress_arr[:, 2]
        sig_tensor[:, 0, 1] = sig_tensor[:, 1, 0] = stress_arr[:, 3]
        sig_tensor[:, 1, 2] = sig_tensor[:, 2, 1] = stress_arr[:, 4]
        sig_tensor[:, 0, 2] = sig_tensor[:, 2, 0] = stress_arr[:, 5]
    elif stress_arr.ndim == 3 and stress_arr.shape[1:] == (3, 3):
        sig_tensor = stress_arr
    else:
        sig_tensor = np.zeros((n, 3, 3), dtype=np.float64)

    # 1. Compute principal stresses and tensile projection tensor C (spstab.F lines 234-250)
    C_tensor = np.zeros((n, 3, 3), dtype=np.float64)
    stab_active = np.zeros(n, dtype=bool)

    for i in range(n):
        evals, evecs = np.linalg.eigh(sig_tensor[i])
        if np.any(evals > 0.0):
            stab_active[i] = True
            for k in range(3):
                if evals[k] > 0.0:
                    rk = -evals[k]
                    vk = evecs[:, k]
                    C_tensor[i] += rk * np.outer(vk, vk)

    if not np.any(stab_active):
        return np.zeros((n, 3), dtype=np.float64), 0.0

    # 2. Reference kernel value at initial interparticle distance W0(dp) (spstab.F lines 121-126)
    stab7 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if stab_active[i]:
            dd = (dp / h[i]) if (dp is not None and dp > 1e-30) else (2.0 / 3.0)
            w0 = cubic_bspline_kernel(dd, 1.0)
            stab7[i] = zstab / max(1e-30, w0 ** 4)

    # 3. Inter-particle stabilization forces (spforcp.F lines 259-277)
    forces = np.zeros((n, 3), dtype=np.float64)
    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=2.0 * float(np.max(h)), output_type='ndarray')

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        active_pairs = (stab7[i_idx] > 0.0) & (stab7[j_idx] > 0.0)
        if np.any(active_pairs):
            iv = i_idx[active_pairs]
            jv = j_idx[active_pairs]
            rij = pos_arr[iv] - pos_arr[jv]
            r = np.linalg.norm(rij, axis=1)
            hij = 0.5 * (h[iv] + h[jv])
            valid = (r <= 2.0 * hij) & (r > 1e-30)
            if np.any(valid):
                iv = iv[valid]
                jv = jv[valid]
                rij_v = rij[valid]
                hij_v = hij[valid]
                r_v = r[valid]

                grad_w = cubic_bspline_grad(rij_v, hij_v)
                w_val = cubic_bspline_kernel(r_v, hij_v)

                vol_i = m[iv] / np.maximum(r_rho[iv], 1e-20)
                vol_j = m[jv] / np.maximum(r_rho[jv], 1e-20)
                vij_vol = vol_i * vol_j

                ww = w_val * (hij_v ** 3)
                wr = 0.5 * (stab7[iv] + stab7[jv])
                wi = (ww ** 4) * wr

                cx = np.einsum('ijk,ik->ij', C_tensor[iv], grad_w)
                dx = np.einsum('ijk,ik->ij', C_tensor[jv], grad_w)

                t_force = (vij_vol * wi)[:, None] * (cx + dx)

                np.add.at(forces, iv, t_force)
                np.subtract.at(forces, jv, t_force)

    vel_arr = np.asarray(vel, dtype=np.float64)
    w_stab = float(np.sum(forces * vel_arr) * dt)
    return forces, w_stab


def sph_density_sum(pos, mass, h_arr):
    """Compute SPH density by summation: rho_a = sum_b m_b * W(r_ab, h_ab).

    Port of OpenRadioss engine/source/elements/sph/spdens.F (lines 101-167, 373-376).
    Uses scipy.spatial.cKDTree for efficient neighbor searching within 2h.

    Args:
        pos: Particle positions, shape (N, 3).
        mass: Particle masses, shape (N,) or float.
        h_arr: Smoothing lengths, shape (N,) or float.

    Returns:
        rho: Densities, shape (N,).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    n = len(pos_arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))

    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    # Self-contribution: W(0, h_a) = 1 / (pi * h_a^3) (weight.F lines 56, 101-102)
    rho = m * (1.0 / (np.pi * h ** 3))

    if n == 1:
        return rho

    max_h = float(np.max(h))
    search_radius = 2.0 * max_h

    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=search_radius, output_type='ndarray')

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)

        # Average smoothing length: DIJ = 0.5 * (DI + DJ) (spdens.F line 147)
        hij = 0.5 * (h[i_idx] + h[j_idx])
        valid = r <= 2.0 * hij
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            r_v = r[valid]
            hij_v = hij[valid]
            w = cubic_bspline_kernel(r_v, hij_v)
            np.add.at(rho, i_v, m[j_v] * w)
            np.add.at(rho, j_v, m[i_v] * w)

    return rho


def sph_defo_rate(pos, vel, mass, rho, h_arr):
    """Compute SPH strain rate tensor D_ab (rate of deformation).

    Port of OpenRadioss engine/source/elements/sph/spdefo3.F (lines 63-70)
    and spdens.F (lines 188-202).

    Velocity gradient tensor:
      L_a = - sum_b (m_b / rho_b) * (v_a - v_b) (x) grad_a W_ab
    Rate of deformation (symmetric strain rate):
      D_a = 0.5 * (L_a + L_a^T)

    Args:
        pos: Particle positions, shape (N, 3).
        vel: Particle velocities, shape (N, 3).
        mass: Particle masses, shape (N,) or float.
        rho: Particle densities, shape (N,) or float.
        h_arr: Smoothing lengths, shape (N,) or float.

    Returns:
        D: Rate of deformation tensor array, shape (N, 3, 3).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    vel_arr = np.asarray(vel, dtype=np.float64)
    n = len(pos_arr)
    if n == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))

    r_rho = np.asarray(rho, dtype=np.float64)
    if r_rho.ndim == 0:
        r_rho = np.full(n, float(r_rho))

    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    L = np.zeros((n, 3, 3), dtype=np.float64)
    if n == 1:
        return L

    max_h = float(np.max(h))
    search_radius = 2.0 * max_h

    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=search_radius, output_type='ndarray')

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)
        hij = 0.5 * (h[i_idx] + h[j_idx])

        valid = (r <= 2.0 * hij) & (r > 1e-30)
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            rij_v = rij[valid]
            hij_v = hij[valid]

            grad_w = cubic_bspline_grad(rij_v, hij_v)
            vij = vel_arr[i_v] - vel_arr[j_v]

            # outer product: v_ij (x) grad_i W_ij (shape (P, 3, 3))
            outer = vij[:, :, None] * grad_w[:, None, :]

            # spdens.F lines 188-202:
            # VJ = m_j / rho_j
            # L_i += - (m_j / rho_j) * (v_i - v_j) (x) grad_i W_ij
            vol_j = m[j_v] / np.maximum(r_rho[j_v], 1e-20)
            vol_i = m[i_v] / np.maximum(r_rho[i_v], 1e-20)

            term_i = -vol_j[:, None, None] * outer
            # For particle j: (v_j - v_i) (x) grad_j W_ji = (-vij) (x) (-grad_w) = vij (x) grad_w
            term_j = -vol_i[:, None, None] * outer

            np.add.at(L, i_v, term_i)
            np.add.at(L, j_v, term_j)

    # Symmetric deformation rate tensor: D = 0.5 * (L + L^T) (spdefo3.F lines 63-70)
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    return D


def sph_forces(pos, vel, mass, rho, pressure, h_arr, alpha_visc=1.0, beta_visc=2.0, c_s=None, gamma=1.4, stress=None, zstab=0.0, dp=None):
    """Compute SPH force array (pressure gradient + Monaghan 1992 artificial viscosity + optional tensile stabilization).

    Port of OpenRadioss engine/source/elements/sph/sppro3.F (lines 72-118),
    spforcp.F (lines 240-304), and spstab.F (lines 234-277).

    Pressure force (spforcp.F lines 251-256):
      F_p,ij = - (m_i * m_j / (rho_i * rho_j)) * (p_i + p_j) * grad_i W_ij

    Monaghan 1992 artificial viscosity (spforcp.F lines 280-304):
      mu_ij = h_ij * (v_ij . x_ij) / (|x_ij|^2 + 0.01 * h_ij^2)   if v_ij . x_ij < 0 else 0
      Pi_ij = (beta_visc * mu_ij^2 - alpha_visc * c_s_bar * mu_ij) * 2 / (rho_i + rho_j)
      F_v,ij = - m_i * m_j * Pi_ij * grad_i W_ij

    Tensile stabilization (spstab.F lines 234-277, spforcp.F lines 259-277):
      T_ij = V_i * V_j * W_I * (C_i + C_j) . grad_i W_ij  (if zstab > 0 and stress tensile)

    Total force:
      F_i = sum_j (F_p,ij + F_v,ij + T_ij)
      By Newton's third law, F_ji = - F_ij, guaranteeing exact momentum conservation.

    Args:
        pos: Particle positions, shape (N, 3).
        vel: Particle velocities, shape (N, 3).
        mass: Particle masses, shape (N,) or float.
        rho: Particle densities, shape (N,) or float.
        pressure: Particle pressures, shape (N,) or float.
        h_arr: Smoothing lengths, shape (N,) or float.
        alpha_visc: Monaghan linear viscosity coefficient (default 1.0).
        beta_visc: Monaghan quadratic viscosity coefficient (default 2.0).
        c_s: Sound speed array, shape (N,) or None (computed from EOS if None).
        gamma: Ratio of specific heats for ideal gas (default 1.4).
        stress: Cauchy stress tensor array (N, 3, 3) or (N, 6) for tensile stabilization.
        zstab: Tensile stabilization parameter (default 0.0, disabled).
        dp: Initial particle spacing (default 2/3 * h).

    Returns:
        forces: Force array, shape (N, 3).
    """
    pos_arr = np.asarray(pos, dtype=np.float64)
    vel_arr = np.asarray(vel, dtype=np.float64)
    n = len(pos_arr)
    forces = np.zeros((n, 3), dtype=np.float64)
    if n <= 1:
        return forces

    m = np.asarray(mass, dtype=np.float64)
    if m.ndim == 0:
        m = np.full(n, float(m))

    r_rho = np.asarray(rho, dtype=np.float64)
    if r_rho.ndim == 0:
        r_rho = np.full(n, float(r_rho))

    p = np.asarray(pressure, dtype=np.float64)
    if p.ndim == 0:
        p = np.full(n, float(p))

    h = np.asarray(h_arr, dtype=np.float64)
    if h.ndim == 0:
        h = np.full(n, float(h))

    # Sound speed: c_s = sqrt(gamma * max(p, 0) / rho) (mdtsph.F line 98)
    if c_s is None:
        c_s = np.sqrt(np.maximum(gamma * p / np.maximum(r_rho, 1e-20), 1e-20))
    else:
        c_s = np.asarray(c_s, dtype=np.float64)
        if c_s.ndim == 0:
            c_s = np.full(n, float(c_s))

    max_h = float(np.max(h))
    search_radius = 2.0 * max_h

    tree = cKDTree(pos_arr)
    pairs = tree.query_pairs(r=search_radius, output_type='ndarray')

    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        rij = pos_arr[i_idx] - pos_arr[j_idx]
        r = np.linalg.norm(rij, axis=1)
        hij = 0.5 * (h[i_idx] + h[j_idx])

        valid = (r <= 2.0 * hij) & (r > 1e-30)
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            rij_v = rij[valid]
            r_v = r[valid]
            hij_v = hij[valid]

            grad_w = cubic_bspline_grad(rij_v, hij_v)

            # 1. Pressure forces (spforcp.F lines 251-256):
            # VIJ = (m_i / rho_i) * (m_j / rho_j)
            vol_i = m[i_v] / np.maximum(r_rho[i_v], 1e-20)
            vol_j = m[j_v] / np.maximum(r_rho[j_v], 1e-20)
            vij_vol = vol_i * vol_j
            # AX + BX = - (p_i + p_j) * grad_W
            f_p_coeff = -vij_vol * (p[i_v] + p[j_v])
            f_p = f_p_coeff[:, None] * grad_w

            # 2. Monaghan 1992 artificial viscosity (spforcp.F lines 280-304):
            vij_vel = vel_arr[i_v] - vel_arr[j_v]
            vr_dot = np.sum(vij_vel * rij_v, axis=1)

            # Viscosity active only under compression: vr_dot < 0
            compress = vr_dot < 0.0
            f_v = np.zeros_like(f_p)
            if np.any(compress):
                i_c = i_v[compress]
                j_c = j_v[compress]
                hij_c = hij_v[compress]
                rij_c = rij_v[compress]
                r_c = r_v[compress]
                vr_c = vr_dot[compress]
                grad_w_c = grad_w[compress]

                # mu_ij = hij * (v_ij . r_ij) / (|r_ij|^2 + 0.01 * hij^2) (spforcp.F lines 286, 290)
                mu_ij = hij_c * vr_c / (r_c * r_c + 0.01 * hij_c * hij_c)
                ssp_bar = 0.5 * (c_s[i_c] + c_s[j_c])
                rho_bar = 0.5 * (r_rho[i_c] + r_rho[j_c])

                # PIJ = (QA * mu^2 - QB * ssp * mu) / rho_bar (spforcp.F line 294)
                # QA = beta_visc, QB = alpha_visc
                pij = (beta_visc * mu_ij * mu_ij - alpha_visc * ssp_bar * mu_ij) / np.maximum(rho_bar, 1e-20)

                # FACT = m_i * m_j * PIJ (spforcp.F line 295)
                # FV = -FACT * grad_W (spforcp.F lines 301-303)
                f_v_coeff = - (m[i_c] * m[j_c] * pij)
                f_v[compress] = f_v_coeff[:, None] * grad_w_c

            f_total = f_p + f_v
            np.add.at(forces, i_v, f_total)
            np.subtract.at(forces, j_v, f_total)

    # 3. Monaghan-Gray tensile instability stabilization (spstab.F lines 234-277, spforcp.F lines 259-277)
    if zstab > 0.0 and stress is not None:
        f_stab, _ = sph_tensile_stabilization(pos, vel, mass, rho, stress, h_arr, zstab=zstab, dp=dp, dt=0.0)
        forces += f_stab

    return forces


def sph_critical_dt(h_arr, rho, pressure, gamma, cfl=0.6):
    """Compute SPH critical time step from CFL stability condition.

    Port of OpenRadioss engine/source/elements/sph/sphreq.F and mdtsph.F (lines 96-135).
    dt = cfl * h / c_s, where c_s = sqrt(gamma * p / rho).

    Args:
        h_arr: Smoothing lengths (float or np.ndarray).
        rho: Densities (float or np.ndarray).
        pressure: Pressures (float or np.ndarray).
        gamma: Ratio of specific heats for ideal gas EOS (LAW6).
        cfl: Courant-Friedrichs-Lewy safety coefficient (default 0.6).

    Returns:
        dt: Critical time step (float or np.ndarray).
    """
    h = np.asarray(h_arr, dtype=np.float64)
    r = np.asarray(rho, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)

    # Sound speed: c_s = sqrt(gamma * max(p, 0) / max(rho, 1e-20)) (mdtsph.F line 98)
    c_s = np.sqrt(np.maximum(gamma * p / np.maximum(r, 1e-20), 1e-20))
    dt = cfl * h / c_s

    if np.ndim(h_arr) == 0 and np.ndim(rho) == 0 and np.ndim(pressure) == 0:
        return float(dt)
    return dt


def sph_step(model, dt, state, fint=None):
    """Execute one SPH time step and assemble forces into the model.

    Integrates SPH physics into the explicit time loop of pyradioss/engine/engine.py:
      1. Evaluates density by summation (spdens.F)
      2. Computes pressure from EOS / material state (LAW6 / perfect gas)
      3. Computes internal and artificial viscous forces (sppro3.F, spforcp.F)
      4. Assembles forces into fint
      5. Books numerical dissipation / energy into state (EN ledger)

    Args:
        model: Radioss Model instance with sph_cells.
        dt: Current time step increment.
        state: EngineState instance for energy tracking.
        fint: Optional global internal force array (if None, taken from model.fint).
    """
    if not hasattr(model, 'sph_cells') or not model.sph_cells:
        return

    cells = model.sph_cells

    # Identify node indices for SPH particles
    if hasattr(cells, 'conn'):
        node_idx = cells.conn.reshape(-1)
    elif hasattr(cells, 'node_ids'):
        node_idx = np.asarray(cells.node_ids, dtype=np.int64)
    elif hasattr(cells, 'ids'):
        node_idx = np.asarray(cells.ids, dtype=np.int64)
    else:
        node_idx = np.arange(len(cells))

    n = len(node_idx)
    if n == 0:
        return

    # Extract positions, velocities, masses
    if hasattr(cells, 'pos'):
        pos = cells.pos
    elif hasattr(model, 'x'):
        pos = model.x[node_idx]
    else:
        return

    if hasattr(cells, 'vel'):
        vel = cells.vel
    elif hasattr(model, 'v'):
        vel = model.v[node_idx]
    else:
        vel = np.zeros_like(pos)

    if hasattr(cells, 'mass'):
        mass = cells.mass
    elif hasattr(model, 'mass'):
        mass = model.mass[node_idx]
    else:
        mass = np.ones(n, dtype=np.float64)

    # Smoothing lengths
    if hasattr(cells, 'h'):
        h_arr = cells.h
    elif hasattr(cells, 'state') and "h" in cells.state:
        h_arr = cells.state["h"]
    else:
        h_arr = np.full(n, 0.1, dtype=np.float64)

    # Density computation: summation
    rho = sph_density_sum(pos, mass, h_arr)
    if hasattr(cells, 'state'):
        cells.state["rho"] = rho

    # Pressure from EOS (LAW6: p = (gamma - 1) * rho * u)
    gamma = getattr(cells, 'gamma', 1.4)
    if hasattr(cells, 'pressure'):
        pressure = cells.pressure
    elif hasattr(cells, 'state') and "pressure" in cells.state:
        pressure = cells.state["pressure"]
    elif hasattr(cells, 'state') and "u" in cells.state:
        pressure = (gamma - 1.0) * rho * cells.state["u"]
    else:
        pressure = np.zeros(n, dtype=np.float64)

    # Viscosity parameters
    alpha = getattr(cells, 'alpha_visc', 1.0)
    beta = getattr(cells, 'beta_visc', 2.0)

    # Tensile stabilization parameters (spstab.F)
    stress = getattr(cells, 'stress', None)
    if stress is None and hasattr(cells, 'state') and "stress" in cells.state:
        stress = cells.state["stress"]
    zstab = getattr(cells, 'zstab', 0.0)
    if zstab == 0.0 and hasattr(cells, 'state') and "zstab" in cells.state:
        zstab = float(cells.state["zstab"])
    dp = getattr(cells, 'dp', None)

    # Compute SPH forces
    f_sph = sph_forces(
        pos, vel, mass, rho, pressure, h_arr,
        alpha_visc=alpha, beta_visc=beta, gamma=gamma,
        stress=stress, zstab=zstab, dp=dp,
    )

    # Assemble into fint
    if fint is None:
        fint = getattr(model, 'fint', None)
    if fint is not None:
        fint[node_idx] += f_sph

    # Book artificial viscosity work into numerical dissipation ledger (EN)
    # Energy accounting: spforcp.F lines 519-521: WVIS = 0.5 * (FV . (VI - VJ))
    if state is not None:
        # Rate of work on particles: W_dot = sum(F_sph . v)
        w_sph = float(np.sum(f_sph * vel) * dt)
        if hasattr(state, 'e_num'):
            state.e_num = getattr(state, 'e_num', 0.0) + w_sph
