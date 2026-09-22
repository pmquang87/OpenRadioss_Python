"""
Isogeometric Analysis (IGA) 3D Solid Elements (NURBS basis functions).

Fortran origin: ``engine/source/elements/ige3d/`` & ``starter/source/elements/ige3d/``
- ``ig3duforc3.F``       Driver routine for 3D IGA elements (Gauss loop, kinematics, dt)
- ``ig3dfint.F``         Internal force integration at control points: F_a = - int B^T sigma dV
- ``ig3donederiv.F``     3D NURBS basis functions, derivatives, and physical Jacobian
- ``dersonebasisfun.F``  Cox-de Boor 1D B-spline basis function and derivative
- ``dersbasisfuns.F``    Piegl & Tiller Algorithm A2.2: all active basis functions on knot span
- ``onebasisfun.F``      1D B-spline basis function evaluation
- ``ige3ddefo.F``        Velocity strain rate tensor D and spin tensor W with large-rotation correction
- ``ig3dderishap.F``     B-matrix assembly from NURBS Cartesian derivatives
- ``ig3daire.F``         Face areas from boundary surface parameter evaluations
- ``ig3dcumu3.F``        Nodal/control-point force accumulation
- ``ig3daverage.F``      Element-average stress, density, and plastic strain
- ``ig3dinit3.F``        Starter initialization: geometry, volumes, and control-point masses
- ``ig3dmass3.F``        Row-sum lumped mass matrix at control points: M_a = int rho R_a dV

Formulation:
* Tensor-product NURBS solid elements with polynomial degrees (p_x, p_y, p_z)
  and N_ctrl = (p_x + 1) * (p_y + 1) * (p_z + 1) control points per element.
* Rational basis functions:
      R_a(xi, eta, zeta) = (N_a(xi) * M_a(eta) * L_a(zeta) * w_a) / W(xi, eta, zeta)
  where W = sum_b N_b M_b L_b w_b is the weight function.
* Spatial gradients:
      grad_x R_a = J^{-T} grad_xi R_a,   with J = d x / d xi
* Kinematics (ige3ddefo.F):
      L_ij = sum_a v_{a, i} dR_a/dx_j
      Rate of deformation D = sym(L) - dt/2 * (L^T L)
      Spin increment W = dt/2 * skew(L)
* Stress update:
      Jaumann objective rate rotation (srota3.F): sigma <- sigma + (W.sigma - sigma.W)
      Hypoelastic 3D Hooke's law: sigma <- sigma + C : D dt
* Internal force (ig3dfint.F):
      F_{a, i} = - sum_g dR_a/dx_j * sigma_{ij} * dV_g
* Lumped mass (ig3dmass3.F):
      M_a = sum_g rho * R_a(xi_g) * dV_g,   sum_a M_a = rho * V
* Energy accounting:
      Internal energy increment de = sum_g (sigma : D)_g * dV_g * dt
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import scatter_add3


# ----------------------------------------------------------------------------
# 1D B-spline Basis Functions (dersonebasisfun.F & dersbasisfuns.F)
# ----------------------------------------------------------------------------

def ders_one_basis_fun(idxi: int, p: int, xi: float, knot: np.ndarray) -> tuple[float, float]:
    """Calculate 1D B-spline function and its first derivative via Cox-de Boor.

    # Ported from engine/source/elements/ige3d/dersonebasisfun.F lines 29-133
    idxi : 0-based index of the basis function in knot vector
    p    : polynomial degree
    xi   : parameter value
    knot : local or full knot vector array
    Returns: (N, dN/dxi)
    """
    m = p + 1
    andu = np.zeros((m, m), dtype=float)

    # Degree 0 basis functions
    for j in range(m):
        k_left = knot[idxi + j]
        k_right = knot[idxi + j + 1]
        if k_left <= xi < k_right or (xi == k_right and k_right == knot[-1] and k_left < k_right):
            andu[j, 0] = 1.0
        else:
            andu[j, 0] = 0.0

    # Triangular table for basis functions
    for k in range(1, m):
        if andu[0, k - 1] == 0.0:
            saved = 0.0
        else:
            denom = knot[idxi + k] - knot[idxi]
            saved = ((xi - knot[idxi]) * andu[0, k - 1]) / denom if denom > EM20 else 0.0

        for j in range(m - k):
            aleft = knot[idxi + j + 1]
            right = knot[idxi + j + k + 1]
            if andu[j + 1, k - 1] == 0.0:
                andu[j, k] = saved
                saved = 0.0
            else:
                denom = right - aleft
                temp = andu[j + 1, k - 1] / denom if denom > EM20 else 0.0
                andu[j, k] = saved + (right - xi) * temp
                saved = (xi - aleft) * temp

    ders1 = andu[0, p]

    # Derivative calculation (dersonebasisfun.F lines 102-127)
    nd = np.zeros(m, dtype=float)
    for j in range(2):
        nd[j] = andu[j, p - 1]

    if nd[0] == 0.0:
        saved = 0.0
    else:
        denom = knot[idxi + p] - knot[idxi]
        saved = nd[0] / denom if denom > EM20 else 0.0

    aleft = knot[idxi + 1]
    right = knot[idxi + p + 1]
    if nd[1] == 0.0:
        ders2 = float(p) * saved
    else:
        denom = right - aleft
        temp = nd[1] / denom if denom > EM20 else 0.0
        ders2 = float(p) * (saved - temp)

    return ders1, ders2


def ders_basis_funs(span_idx: int, p: int, xi: float, knot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piegl & Tiller Algorithm A2.2 (all p+1 active B-spline functions and 1st derivatives).

    # Ported from engine/source/elements/ige3d/dersbasisfuns.F lines 28-142
    span_idx : knot span index such that knot[span_idx] <= xi < knot[span_idx+1]
    p        : polynomial degree
    xi       : parameter evaluation point
    knot     : knot vector
    Returns: (N (p+1,), dN (p+1,))
    """
    m = p + 1
    ders1 = np.zeros(m, dtype=float)
    ders2 = np.zeros(m, dtype=float)

    andu = np.zeros((m, m), dtype=float)
    aleft = np.zeros(m, dtype=float)
    right = np.zeros(m, dtype=float)

    andu[0, 0] = 1.0
    for j in range(1, m):
        aleft[j] = xi - knot[span_idx + 1 - j]
        right[j] = knot[span_idx + j] - xi
        saved = 0.0
        for r in range(j):
            andu[j, r] = right[r + 1] + aleft[j - r]
            denom = andu[j, r]
            temp = andu[r, j - 1] / denom if denom > EM20 else 0.0
            andu[r, j] = saved + right[r + 1] * temp
            saved = aleft[j - r] * temp
        andu[j, j] = saved

    # Load basis functions
    for j in range(m):
        ders1[j] = andu[j, p]

    # Compute first derivatives (dersbasisfuns.F lines 88-135)
    a = np.zeros((2, m), dtype=float)
    a[0, 0] = 1.0
    for r in range(m):
        s1, s2 = 0, 1
        d = 0.0
        kr = r - 1
        kp = p - 1
        if r >= 1:
            denom = andu[kp + 1, kr]
            a[s2, 0] = a[s1, 0] / denom if denom > EM20 else 0.0
            d = a[s2, 0] * andu[kr, kp]
        j1 = 1 if kr >= -1 else -kr
        j2 = 0 if (r - 1) <= kp else p - r
        for j in range(j1, j2 + 1):
            denom = andu[kp + 1, kr + j]
            a[s2, j] = (a[s1, j] - a[s1, j - 1]) / denom if denom > EM20 else 0.0
            d += a[s2, j] * andu[kr + j, kp]
        if r <= kp:
            denom = andu[kp + 1, r]
            a[s2, 1] = -a[s1, 0] / denom if denom > EM20 else 0.0
            d += a[s2, 1] * andu[r, kp]
        ders2[r] = d * p

    return ders1, ders2


# ----------------------------------------------------------------------------
# 3D NURBS Basis Functions & Gradients (ig3donederiv.F)
# ----------------------------------------------------------------------------

def nurbs_3d_basis_and_derivs(
    xi: np.ndarray,
    degrees: tuple[int, int, int],
    knot_span: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    weights: np.ndarray,
    cp_coords: np.ndarray,
    knots_local: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate 3D NURBS rational basis functions, Cartesian derivatives, and Jacobian.

    # Ported from engine/source/elements/ige3d/ig3donederiv.F lines 33-259
    xi         : (3,) parameter in parent space [-1, 1]^3
    degrees    : (px, py, pz)
    knot_span  : ((xi_min, xi_max), (eta_min, eta_max), (zeta_min, zeta_max))
    weights    : (n_ctrl,) control point weights
    cp_coords  : (n_ctrl, 3) control point Cartesian coordinates
    knots_local: optional tuple of (knot_x, knot_y, knot_z) local knot vectors
    Returns:
        R      : (n_ctrl,) rational basis function values
        dndx   : (n_ctrl, 3) Cartesian spatial gradients dR_a/dx_i
        Jmat   : (3, 3) Jacobian matrix d x / d xi_parent
        detJ   : float, Jacobian determinant
    """
    px, py, pz = degrees
    n_ctrl = len(weights)

    # 1. Map parent coordinates [-1, 1] to knot span parameter space (lines 124-126)
    (x0, x1), (y0, y1), (z0, z1) = knot_span
    xi_param = np.array([
        0.5 * ((x1 - x0) * xi[0] + (x1 + x0)),
        0.5 * ((y1 - y0) * xi[1] + (y1 + y0)),
        0.5 * ((z1 - z0) * xi[2] + (z1 + z0)),
    ])
    dxi_dparent = np.diag([0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * (z1 - z0)])

    # 2. Evaluate univariate B-spline functions in each direction
    fn = np.zeros(n_ctrl)
    fm = np.zeros(n_ctrl)
    fl = np.zeros(n_ctrl)
    dndxi = np.zeros(n_ctrl)
    dmdxi = np.zeros(n_ctrl)
    dldxi = np.zeros(n_ctrl)

    if knots_local is not None:
        kx_loc, ky_loc, kz_loc = knots_local
        for a in range(n_ctrl):
            fn[a], dndxi[a] = ders_one_basis_fun(0, px, xi_param[0], kx_loc[a])
            fm[a], dmdxi[a] = ders_one_basis_fun(0, py, xi_param[1], ky_loc[a])
            fl[a], dldxi[a] = ders_one_basis_fun(0, pz, xi_param[2], kz_loc[a])
    else:
        # Standard open uniform knot vectors spanning [x0, x1], [y0, y1], [z0, z1]
        kx_std = np.array([x0] * (px + 1) + [x1] * (px + 1))
        ky_std = np.array([y0] * (py + 1) + [y1] * (py + 1))
        kz_std = np.array([z0] * (pz + 1) + [z1] * (pz + 1))
        nx, dnx = ders_basis_funs(px, px, xi_param[0], kx_std)
        ny, dny = ders_basis_funs(py, py, xi_param[1], ky_std)
        nz, dnz = ders_basis_funs(pz, pz, xi_param[2], kz_std)

        # Tensor product ordering: node a = k*(px+1)*(py+1) + j*(px+1) + i
        idx = 0
        for k in range(pz + 1):
            for j in range(py + 1):
                for i in range(px + 1):
                    fn[idx] = nx[i]
                    dndxi[idx] = dnx[i]
                    fm[idx] = ny[j]
                    dmdxi[idx] = dny[j]
                    fl[idx] = nz[k]
                    dldxi[idx] = dnz[k]
                    idx += 1

    # 3. Build numerators and denominator for rational NURBS (lines 159-190)
    R_num = fn * fm * fl * weights
    sum_tot = float(np.sum(R_num))
    if abs(sum_tot) < EM20:
        sum_tot = 1.0

    dR_dxi = np.empty((n_ctrl, 3))
    dR_dxi[:, 0] = dndxi * fm * fl * weights
    dR_dxi[:, 1] = fn * dmdxi * fl * weights
    dR_dxi[:, 2] = fn * fm * dldxi * weights

    sum_xi = np.sum(dR_dxi, axis=0)

    # Rational basis function: R = B / W
    R = R_num / sum_tot

    # Quotient rule: dR/dxi = (dB/dxi - R * dW/dxi) / W (lines 185-189)
    for c in range(3):
        dR_dxi[:, c] = (dR_dxi[:, c] - R * sum_xi[c]) / sum_tot

    # 4. Gradient of mapping from parameter space to physical space: dx/dxi = sum_a x_a (x) dR_a/dxi (lines 193-199)
    # dx_dxi[i, j] = d x_i / d xi_j
    dx_dxi = cp_coords.T @ dR_dxi  # (3, 3)

    # 5. Invert dx_dxi to get dxi/dx (lines 203-220)
    det_dx_dxi = float(np.linalg.det(dx_dxi))
    if abs(det_dx_dxi) > EM20:
        dxi_dx = np.linalg.inv(dx_dxi)
    else:
        dxi_dx = np.eye(3)
        det_dx_dxi = 1.0

    # 6. Combined Jacobian to parent space: J_parent = dx/dxi @ dxi/dparent (lines 243-246)
    Jmat = dx_dxi @ dxi_dparent
    detJ = float(np.linalg.det(Jmat))

    # 7. Spatial gradients of shape functions: dR_a/dx_i = sum_k dR_a/dxi_k * dxi_k/dx_i (line 240)
    dndx = dR_dxi @ dxi_dx  # (n_ctrl, 3)

    return R, dndx, Jmat, detJ


# ----------------------------------------------------------------------------
# Gauss Quadrature (ig3duforc3.F lines 253-310)
# ----------------------------------------------------------------------------

def gauss_points_3d(ngp: tuple[int, int, int] = (2, 2, 2)) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss-Legendre quadrature points and weights in [-1, 1]^3.

    # Ported from engine/source/elements/ige3d/ig3duforc3.F DATA W_GAUSS, A_GAUSS
    ngp : (ngx, ngy, ngz) number of points in each parametric direction
    Returns:
        points : (N_g, 3) quadrature points in [-1, 1]^3
        weights: (N_g,) quadrature weights
    """
    gx, wx = np.polynomial.legendre.leggauss(ngp[0])
    gy, wy = np.polynomial.legendre.leggauss(ngp[1])
    gz, wz = np.polynomial.legendre.leggauss(ngp[2])

    pts = []
    wts = []
    for k in range(ngp[2]):
        for j in range(ngp[1]):
            for i in range(ngp[0]):
                pts.append([gx[i], gy[j], gz[k]])
                wts.append(wx[i] * wy[j] * wz[k])

    return np.array(pts, dtype=float), np.array(wts, dtype=float)


# ----------------------------------------------------------------------------
# Kinematics & Strain Rates (ige3ddefo.F)
# ----------------------------------------------------------------------------

def iga3d_defo(v: np.ndarray, dndx: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Velocity strain rates D and spin increment W with large-rotation correction.

    # Ported from engine/source/elements/ige3d/ige3ddefo.F lines 28-137
    v    : (n_ctrl, 3) control point velocities
    dndx : (n_ctrl, 3) Cartesian gradients dR_a/dx_j
    dt   : time step
    Returns:
        D6   : (6,) strain rates [Dxx, Dyy, Dzz, 2*Dxy, 2*Dyz, 2*Dzx]
        spin : (3,) spin vector increments [Wxx, Wyy, Wzz] = [W_yz, W_zx, W_xy]
        L    : (3, 3) velocity gradient dv_i/dx_j
    """
    # L_ij = sum_a v_{a, i} * dR_a/dx_j (ige3ddefo.F lines 91-104)
    L = v.T @ dndx  # L[i, j] = dv_i/dx_j

    dxx = L[0, 0]
    dyy = L[1, 1]
    dzz = L[2, 2]
    dxy = L[0, 1]
    dyx = L[1, 0]
    dyz = L[1, 2]
    dzy = L[2, 1]
    dzx = L[2, 0]
    dxz = L[0, 2]

    # Large rotation second-order correction (lines 106-130)
    dt1d2 = 0.5 * dt
    dxx_c = dxx - dt1d2 * (dxx * dxx + dyx * dyx + dzx * dzx)
    dyy_c = dyy - dt1d2 * (dyy * dyy + dzy * dzy + dxy * dxy)
    dzz_c = dzz - dt1d2 * (dzz * dzz + dxz * dxz + dyz * dyz)

    aaa = dt1d2 * (dxx * dxy + dyx * dyy + dzx * dzy)
    dxy_c = dxy - aaa
    dyx_c = dyx - aaa
    d4 = dxy_c + dyx_c  # engineering shear rate 2*Dxy

    aaa = dt1d2 * (dyy * dyz + dzy * dzz + dxy * dxz)
    dyz_c = dyz - aaa
    dzy_c = dzy - aaa
    d5 = dyz_c + dzy_c  # engineering shear rate 2*Dyz

    aaa = dt1d2 * (dzz * dzx + dxz * dxx + dyz * dyx)
    dxz_c = dxz - aaa
    dzx_c = dzx - aaa
    d6 = dxz_c + dzx_c  # engineering shear rate 2*Dzx

    wxx = dt1d2 * (dzy_c - dyz_c)
    wyy = dt1d2 * (dxz_c - dzx_c)
    wzz = dt1d2 * (dyx_c - dxy_c)

    D6 = np.array([dxx_c, dyy_c, dzz_c, d4, d5, d6])
    spin = np.array([wxx, wyy, wzz])
    return D6, spin, L


# ----------------------------------------------------------------------------
# Internal Forces (ig3dfint.F)
# ----------------------------------------------------------------------------

def iga3d_fint(sig: np.ndarray, dndx: np.ndarray, vol: float) -> np.ndarray:
    """Internal force contribution from one Gauss point to all control points.

    # Ported from engine/source/elements/ige3d/ig3dfint.F lines 28-115
    sig  : (6,) Cauchy stress [sxx, syy, szz, sxy, syz, szx]
    dndx : (n_ctrl, 3) Cartesian shape function gradients
    vol  : float, Gauss point volume weight dV_g
    Returns:
        f_int: (n_ctrl, 3) internal forces accumulated NEGATED (-B^T sigma dV)
    """
    n_ctrl = len(dndx)
    f_int = np.zeros((n_ctrl, 3))

    sxx, syy, szz, sxy, syz, szx = sig[0], sig[1], sig[2], sig[3], sig[4], sig[5]

    # Matrix multiplication: F_a = - vol * [dNa/dx*sxx + dNa/dy*sxy + dNa/dz*szx, ...]
    # Matches ig3dfint.F lines 90-102:
    f_int[:, 0] = -vol * (dndx[:, 0] * sxx + dndx[:, 1] * sxy + dndx[:, 2] * szx)
    f_int[:, 1] = -vol * (dndx[:, 0] * sxy + dndx[:, 1] * syy + dndx[:, 2] * syz)
    f_int[:, 2] = -vol * (dndx[:, 0] * szx + dndx[:, 1] * syz + dndx[:, 2] * szz)

    return f_int


# ----------------------------------------------------------------------------
# Jaumann Stress Rotation (srota3.F)
# ----------------------------------------------------------------------------

def srota3_stress(sig: np.ndarray, spin: np.ndarray) -> np.ndarray:
    """Jaumann objective rate stress rotation: sigma <- sigma + (W.sigma - sigma.W).

    # Ported from engine/source/elements/solid/solide/srota3.F lines 65-105
    sig : (6,) Cauchy stress [sxx, syy, szz, sxy, syz, szx]
    spin: (3,) spin increments [Wxx, Wyy, Wzz] where
          W_yz = Wxx, W_zx = Wyy, W_xy = Wzz
    """
    sxx, syy, szz, sxy, syz, szx = sig
    wx, wy, wz = spin

    # 3x3 symmetric tensor representation
    S = np.array([
        [sxx, sxy, szx],
        [sxy, syy, syz],
        [szx, syz, szz]
    ])
    # Skew spin matrix
    W = np.array([
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0]
    ])
    # Jaumann correction: dS = W @ S - S @ W
    dS = W @ S - S @ W
    S_rot = S + dS
    return np.array([S_rot[0, 0], S_rot[1, 1], S_rot[2, 2], S_rot[0, 1], S_rot[1, 2], S_rot[2, 0]])


# ----------------------------------------------------------------------------
# Starter-side initialization (ig3dinit3.F & ig3dmass3.F)
# ----------------------------------------------------------------------------

def init_group(group, model, log):
    """Initialize IGA 3D solid element group buffer and lumped mass matrix.

    # Ported from starter/source/elements/ige3d/ig3dinit3.F & ig3dmass3.F
    """
    n = group.n
    if n == 0 or len(group.conn) == 0:
        group.state.update(
            vol=np.empty(0),
            rho=np.empty(0),
            eint=np.empty(0),
            sig=np.empty((0, 6)),
        )
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float), None

    conn = group.conn  # (n, n_ctrl)
    n_ctrl = conn.shape[1]

    # Determine polynomial degrees from n_ctrl
    # Default: cubic (4x4x4=64), quadratic (3x3x3=27), or linear (2x2x2=8)
    if n_ctrl == 8:
        deg = (1, 1, 1)
        ngp = (2, 2, 2)
    elif n_ctrl == 27:
        deg = (2, 2, 2)
        ngp = (3, 3, 3)
    elif n_ctrl == 64:
        deg = (3, 3, 3)
        ngp = (4, 4, 4)
    else:
        # Fallback to general cube root
        p = int(round(n_ctrl ** (1.0 / 3.0))) - 1
        deg = (p, p, p)
        ngp = (p + 1, p + 1, p + 1)

    degrees = getattr(group, "degrees", deg)
    ngp_eval = getattr(group, "ngp", ngp)
    n_gp = ngp_eval[0] * ngp_eval[1] * ngp_eval[2]

    # Gauss points in parent space [-1, 1]^3
    gp_pts, gp_wts = gauss_points_3d(ngp_eval)

    # Parametric knot span (default [0, 1]^3 per element)
    knot_span = getattr(group, "knot_span", ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))

    # Weights
    weights = getattr(group, "weights", np.ones(n_ctrl))

    vol0 = np.zeros(n)
    rho0 = np.zeros(n)
    ssp0 = np.zeros(n)
    mass_nodes = np.zeros((n, n_ctrl))

    # Evaluate geometry at Gauss points for each element
    gp_dndx = np.zeros((n, n_gp, n_ctrl, 3))
    gp_vol = np.zeros((n, n_gp))
    gp_R = np.zeros((n, n_gp, n_ctrl))

    for ie in range(n):
        cp_xe = model.x0[conn[ie]]  # (n_ctrl, 3)
        sl_mat = None
        for sl, mat, prop in group.state.get("slices", []):
            if ie in np.arange(sl.start or 0, sl.stop or n):
                sl_mat = mat
                break
        if sl_mat is None:
            # Fallback material properties
            mat_rho = getattr(group, "rho", 1000.0)
            mat_E = getattr(group, "E", 2.1e11)
            mat_nu = getattr(group, "nu", 0.3)
        else:
            mat_rho = getattr(sl_mat, "rho0", getattr(sl_mat, "rho", 1000.0))
            mat_E = getattr(sl_mat, "E", getattr(sl_mat, "young", 2.1e11))
            mat_nu = getattr(sl_mat, "nu", 0.3)

        rho0[ie] = mat_rho
        # P-wave sound speed c = sqrt(E*(1-nu) / ((1+nu)*(1-2nu)*rho))
        c_sound = np.sqrt(max(mat_E * (1.0 - mat_nu) / ((1.0 + mat_nu) * (1.0 - 2.0 * mat_nu) * mat_rho), EM20))
        ssp0[ie] = c_sound

        elem_vol = 0.0
        for ig in range(n_gp):
            R, dndx, Jmat, detJ = nurbs_3d_basis_and_derivs(
                gp_pts[ig], degrees, knot_span, weights, cp_xe
            )
            dV = gp_wts[ig] * detJ
            gp_vol[ie, ig] = dV
            gp_dndx[ie, ig] = dndx
            gp_R[ie, ig] = R
            elem_vol += dV

            # Row-sum lumped mass contribution: M_a += rho * R_a * dV (ig3dmass3.F line 88)
            mass_nodes[ie] += mat_rho * R * dV

        vol0[ie] = elem_vol

    group.state.update(
        degrees=degrees,
        ngp=ngp_eval,
        n_gp=n_gp,
        gp_pts=gp_pts,
        gp_wts=gp_wts,
        knot_span=knot_span,
        weights=weights,
        vol0=vol0,
        vol=vol0.copy(),
        rho0=rho0,
        ssp0=ssp0,
        sig=np.zeros((n, n_gp, 6)),  # per-GP Cauchy stress
        eint=np.zeros(n),            # internal energy
        gp_vol=gp_vol,
        gp_dndx=gp_dndx,
        gp_R=gp_R,
        off=np.ones(n),
    )

    node_idx = conn.reshape(-1)
    mass_c = mass_nodes.reshape(-1)
    return node_idx, mass_c, None


# ----------------------------------------------------------------------------
# Engine-side Force Computation (ig3duforc3.F)
# ----------------------------------------------------------------------------

def forces(group, x: np.ndarray, v: np.ndarray, vr: np.ndarray | None, dt: float, fint: np.ndarray, mint: np.ndarray | None) -> np.ndarray:
    """Compute 3D IGA NURBS internal forces, update stresses, energy, and return dt.

    # Ported from engine/source/elements/ige3d/ig3duforc3.F lines 434-620
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)

    n_ctrl = conn.shape[1]
    degrees = st["degrees"]
    knot_span = st["knot_span"]
    weights = st["weights"]
    n_gp = st["n_gp"]
    gp_pts = st["gp_pts"]
    gp_wts = st["gp_wts"]

    off = st.get("off", np.ones(n))
    alive = off > 0.0

    sig = st["sig"]
    eint = st["eint"]
    vol = st["vol"]
    dt_e = np.full(n, EP30)

    # Elastic material properties
    E_mod = getattr(group, "E", 2.1e11)
    nu = getattr(group, "nu", 0.3)
    lam = E_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E_mod / (2.0 * (1.0 + nu))
    rho = st["rho0"]
    c_sound = st["ssp0"]

    if dt is None or dt <= 0.0 or v is None:
        # Cycle 0: compute initial volumes and dt estimate
        for ie in range(n):
            if not alive[ie]:
                continue
            v_elem = vol[ie]
            # Characteristic length lc ~ V^(1/3)
            lc = max(v_elem ** (1.0 / 3.0), EM20)
            dt_e[ie] = lc / max(c_sound[ie], EM20)
        return dt_e

    # Cycle integration
    elem_forces = np.zeros((n, n_ctrl, 3))

    for ie in range(n):
        if not alive[ie]:
            continue

        cp_xe = x[conn[ie]]  # current control point coordinates (n_ctrl, 3)
        cp_ve = v[conn[ie]]  # current control point velocities (n_ctrl, 3)

        curr_vol = 0.0
        de_elem = 0.0

        for ig in range(n_gp):
            # 1. Evaluate current basis functions and Cartesian gradients
            R, dndx, Jmat, detJ = nurbs_3d_basis_and_derivs(
                gp_pts[ig], degrees, knot_span, weights, cp_xe
            )
            dV = gp_wts[ig] * detJ
            curr_vol += dV

            # 2. Kinematics: strain rates D and spin increment W (ige3ddefo.F)
            D6, spin, _ = iga3d_defo(cp_ve, dndx, dt)

            # 3. Jaumann stress rotation (srota3.F)
            s_rot = srota3_stress(sig[ie, ig], spin)

            # 4. Constitutive law (3D Hooke's law)
            trD = D6[0] + D6[1] + D6[2]
            dsxx = (lam * trD + 2.0 * mu * D6[0]) * dt
            dsyy = (lam * trD + 2.0 * mu * D6[1]) * dt
            dszz = (lam * trD + 2.0 * mu * D6[2]) * dt
            dsxy = (mu * D6[3]) * dt
            dsyz = (mu * D6[4]) * dt
            dszx = (mu * D6[5]) * dt

            s_new = s_rot + np.array([dsxx, dsyy, dszz, dsxy, dsyz, dszx])
            sig[ie, ig] = s_new

            # 5. Internal energy increment: de = (sigma_mid : D) * dV * dt
            s_mid = 0.5 * (s_rot + s_new)
            work_rate = (
                s_mid[0] * D6[0] + s_mid[1] * D6[1] + s_mid[2] * D6[2]
                + s_mid[3] * D6[3] + s_mid[4] * D6[4] + s_mid[5] * D6[5]
            )
            de_elem += work_rate * dV * dt

            # 6. Internal forces integration: F_a = - B^T sigma dV (ig3dfint.F)
            f_g = iga3d_fint(s_new, dndx, dV)
            elem_forces[ie] += f_g

        vol[ie] = curr_vol
        eint[ie] += de_elem

        # Critical time step: lc / c (ig3duforc3.F lines 778-820)
        lc = max(curr_vol ** (1.0 / 3.0), EM20)
        dt_e[ie] = lc / max(c_sound[ie], EM20)

    # Accumulate internal forces into global fint (negated sign convention)
    flat_idx = conn.reshape(-1)
    scatter_add3(fint, flat_idx, elem_forces.reshape(-1, 3))

    return dt_e
