"""
numba mirrors of the hot kernel blocks (the M7 accelerated backend).

Importing this module requires numba (an OPTIONAL dependency — the
import error is caught by ``accel.select_backend`` which then falls back
to NumPy). Each function here mirrors, operation for operation, one
NumPy block of the reference kernels:

    hexa_pre      <-> elements/solid_hexa8._pre     (srcoor3/sdefo3/srota3)
    hexa_post     <-> elements/solid_hexa8._post    (sbulk3/sfint3/shour3/sdlen3)
    tetra10_pre   <-> elements/solid_tetra10._pre   (geometry/kinematics/Jaumann)
    tetra10_post  <-> elements/solid_tetra10._post  (viscosity/forces/energies/dt)
    shell_pre     <-> elements/shell_bt4._pre       (ccoor3/cdefo3)
    shell_post    <-> elements/shell_bt4._post      (czforc3/chour3)
    t7_narrow     <-> contact/inter_type7._narrow   (i7dst3)

READ THE NUMPY REFERENCE FIRST. The functions here are deliberately
comment-light on physics: every formula is documented in the reference
implementation, and the porting guide's M7 notes explain why these five
blocks (they were the measured hotspots — models of O(100) elements are
dominated by NumPy per-call overhead, which a single compiled loop per
block eliminates).

The parity contract (enforced by tests/test_m7_backends.py)
-----------------------------------------------------------
* same formulas, same branch structure, same evaluation order as the
  NumPy reference — element-wise expressions agree BITWISE;
* short reductions (3/4/5/6/8-term dot products) are written as
  sequential left-to-right sums; NumPy's einsum/matmul may associate
  them differently (SIMD partial sums), which bounds the per-call
  difference at machine precision (~1e-15 relative) — the documented
  per-kernel tolerance;
* NO ``fastmath`` (it licenses reassociation and breaks the contract),
  NO ``parallel`` (scatter order must be deterministic — the restart
  chaining acceptance tests are the canary);
* ``cache=True`` everywhere: the one-time JIT compilation (~10 s for
  this module) is cached on disk, so runs after the first pay nothing.

The scalar-heavy style (explicit 3x3 unrolls instead of small ndarray
temporaries) is intentional: numba allocates real heap arrays for
np.empty inside a loop, and at O(100)-element groups that allocation
would eat the win.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ...common.constants import EM20, EP30

# ---- module-level constants (numba freezes global ndarrays) ----------------
# hexa: uniform-gradient operator signs and FB hourglass vectors — the
# same tables as elements/solid_hexa8 (_DN_DXI = _XI/8, _H, _FACES)
_XI8 = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64) / 8.0
_H4 = np.array([
    [1, 1, -1, -1, -1, -1, 1, 1],
    [1, -1, -1, 1, -1, 1, 1, -1],
    [1, -1, 1, -1, 1, -1, 1, -1],
    [-1, 1, -1, 1, 1, -1, 1, -1],
], dtype=np.float64)
_FACES6 = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
], dtype=np.int64)


# ============================================================================
# solid_hexa8 mirrors
# ============================================================================

@njit(cache=True, parallel=True)
def hexa_pre(xe, ve, sig, dt, off, lc_scale):
    """Mirror of solid_hexa8._pre — geometry, velocity gradient, Jaumann
    rotation (in place on sig). Returns (dndx, vol, lc, deps, trD)."""
    n = xe.shape[0]
    dndx = np.empty((n, 8, 3))
    vol = np.empty(n)
    lc = np.empty(n)
    deps = np.empty((n, 6))
    trD = np.empty(n)

    for e in prange(n):
        # Jacobian J[a,b] = sum_i dN[i,a] xe[i,b]
        j00 = 0.0; j01 = 0.0; j02 = 0.0
        j10 = 0.0; j11 = 0.0; j12 = 0.0
        j20 = 0.0; j21 = 0.0; j22 = 0.0
        for i in range(8):
            d0 = _XI8[i, 0]; d1 = _XI8[i, 1]; d2 = _XI8[i, 2]
            x0 = xe[e, i, 0]; x1 = xe[e, i, 1]; x2 = xe[e, i, 2]
            j00 += d0 * x0; j01 += d0 * x1; j02 += d0 * x2
            j10 += d1 * x0; j11 += d1 * x1; j12 += d1 * x2
            j20 += d2 * x0; j21 += d2 * x1; j22 += d2 * x2
        # cofactor determinant/inverse (same formulas as fastmath.det_inv33)
        A = j11 * j22 - j12 * j21
        B = j12 * j20 - j10 * j22
        C = j10 * j21 - j11 * j20
        det = j00 * A + j01 * B + j02 * C
        v8 = 8.0 * det
        vol[e] = v8 if v8 > EM20 else EM20
        # np.divide, not '/': a collapsed element (det = 0, produced only
        # by an already-diverged run) must yield inf like the NumPy path,
        # not raise ZeroDivisionError (numba gives scalar '/' Python
        # semantics) — the Engine's NaN/dt guards then stop the run the
        # same way under both backends
        idet = np.divide(1.0, det)
        i00 = A * idet
        i01 = (j02 * j21 - j01 * j22) * idet
        i02 = (j01 * j12 - j02 * j11) * idet
        i10 = B * idet
        i11 = (j00 * j22 - j02 * j20) * idet
        i12 = (j02 * j10 - j00 * j12) * idet
        i20 = C * idet
        i21 = (j01 * j20 - j00 * j21) * idet
        i22 = (j00 * j11 - j01 * j10) * idet
        # dndx[i,b] = sum_a dN[i,a] * Jinv[b,a]
        for i in range(8):
            d0 = _XI8[i, 0]; d1 = _XI8[i, 1]; d2 = _XI8[i, 2]
            dndx[e, i, 0] = d0 * i00 + d1 * i01 + d2 * i02
            dndx[e, i, 1] = d0 * i10 + d1 * i11 + d2 * i12
            dndx[e, i, 2] = d0 * i20 + d1 * i21 + d2 * i22

        # characteristic length: vol / max diagonal-cross face area
        amax = 0.0
        for f in range(6):
            n0 = _FACES6[f, 0]; n1 = _FACES6[f, 1]
            n2 = _FACES6[f, 2]; n3 = _FACES6[f, 3]
            d1x = xe[e, n2, 0] - xe[e, n0, 0]
            d1y = xe[e, n2, 1] - xe[e, n0, 1]
            d1z = xe[e, n2, 2] - xe[e, n0, 2]
            d2x = xe[e, n3, 0] - xe[e, n1, 0]
            d2y = xe[e, n3, 1] - xe[e, n1, 1]
            d2z = xe[e, n3, 2] - xe[e, n1, 2]
            cx = d1y * d2z - d1z * d2y
            cy = d1z * d2x - d1x * d2z
            cz = d1x * d2y - d1y * d2x
            a = 0.5 * np.sqrt(cx * cx + cy * cy + cz * cz)
            if a > amax:
                amax = a
        lc[e] = (vol[e] / (amax if amax > EM20 else EM20)) * lc_scale[e]

        # velocity gradient L[b,c] = sum_i ve[i,b] dndx[i,c]
        l00 = 0.0; l01 = 0.0; l02 = 0.0
        l10 = 0.0; l11 = 0.0; l12 = 0.0
        l20 = 0.0; l21 = 0.0; l22 = 0.0
        for i in range(8):
            v0 = ve[e, i, 0]; v1 = ve[e, i, 1]; v2 = ve[e, i, 2]
            g0 = dndx[e, i, 0]; g1 = dndx[e, i, 1]; g2 = dndx[e, i, 2]
            l00 += v0 * g0; l01 += v0 * g1; l02 += v0 * g2
            l10 += v1 * g0; l11 += v1 * g1; l12 += v1 * g2
            l20 += v2 * g0; l21 += v2 * g1; l22 += v2 * g2

        alive = off[e] > 0.0
        if alive:
            # round-off trace flush — mirrors the reference exactly (see
            # the vgm/1e-14 comment in solid_hexa8._pre)
            tr = l00 + l11 + l22
            vmax = 0.0
            gmax = 0.0
            for i in range(8):
                for b in range(3):
                    av = abs(ve[e, i, b])
                    if av > vmax:
                        vmax = av
                    ag = abs(dndx[e, i, b])
                    if ag > gmax:
                        gmax = ag
            if abs(tr) <= 1e-14 * (vmax * gmax):
                tr = 0.0
            trD[e] = tr
            deps[e, 0] = l00 * dt
            deps[e, 1] = l11 * dt
            deps[e, 2] = l22 * dt
            deps[e, 3] = (l01 + l10) * dt
            deps[e, 4] = (l12 + l21) * dt
            deps[e, 5] = (l02 + l20) * dt
        else:                        # deleted: frozen, no straining
            trD[e] = 0.0
            for k in range(6):
                deps[e, k] = 0.0

        # Jaumann rotation of the old stress by the spin increment
        wxy = 0.5 * (l01 - l10) * dt
        wyz = 0.5 * (l12 - l21) * dt
        wxz = 0.5 * (l02 - l20) * dt
        sxx = sig[e, 0]; syy = sig[e, 1]; szz = sig[e, 2]
        sxy = sig[e, 3]; syz = sig[e, 4]; szx = sig[e, 5]
        sig[e, 0] = sxx + 2.0 * (wxy * sxy + wxz * szx)
        sig[e, 1] = syy + 2.0 * (-wxy * sxy + wyz * syz)
        sig[e, 2] = szz + 2.0 * (-wxz * szx - wyz * syz)
        sig[e, 3] = sxy + wxy * (syy - sxx) + wxz * syz + wyz * szx
        sig[e, 4] = syz + wyz * (szz - syy) - wxy * szx - wxz * sxy
        sig[e, 5] = szx + wxz * (szz - sxx) + wxy * syz - wyz * sxy
    return dndx, vol, lc, deps, trD


@njit(cache=True, parallel=True)
def hexa_post(xe, ve, dndx, vol, lc, rho, trD, deps, sig, sig_old,
              qa, qb, c, hcoef, alive, qvw_pend, dt, dtfac):
    """Mirror of solid_hexa8._post — bulk viscosity, internal + hourglass
    forces, energy increments, critical dt. Returns
    (fe, dt_crit, w_visc, qvw_new, deint0, dehour)."""
    n = xe.shape[0]
    fe = np.empty((n, 8, 3))
    dt_crit = np.empty(n)
    w_visc = np.empty(n)
    qvw_new = np.empty(n)
    deint0 = np.empty(n)
    dehour = np.empty(n)
    gam = np.empty((n, 4, 8))
    qd = np.empty((n, 4, 3))

    for e in prange(n):
        live = alive[e]
        compressing = (trD[e] < 0.0) and live
        if compressing:
            qv = rho[e] * lc[e] * (qa[e] * qa[e] * lc[e] * trD[e] * trD[e]
                                   - qb[e] * c[e] * trD[e])
        else:
            qv = 0.0

        # stress matrix with the viscous pressure on the diagonal
        s00 = sig[e, 0] - qv
        s11 = sig[e, 1] - qv
        s22 = sig[e, 2] - qv
        s01 = sig[e, 3]
        s12 = sig[e, 4]
        s02 = sig[e, 5]
        mv = -vol[e]
        for i in range(8):
            g0 = dndx[e, i, 0]; g1 = dndx[e, i, 1]; g2 = dndx[e, i, 2]
            fe[e, i, 0] = mv * (g0 * s00 + g1 * s01 + g2 * s02)
            fe[e, i, 1] = mv * (g0 * s01 + g1 * s11 + g2 * s12)
            fe[e, i, 2] = mv * (g0 * s02 + g1 * s12 + g2 * s22)

        # hourglass: gamma = H - (H x) . dndx, modal velocities, forces
        for a in range(4):
            hx0 = 0.0; hx1 = 0.0; hx2 = 0.0
            for i in range(8):
                h = _H4[a, i]
                hx0 += h * xe[e, i, 0]
                hx1 += h * xe[e, i, 1]
                hx2 += h * xe[e, i, 2]
            for i in range(8):
                gam[e, a, i] = _H4[a, i] - (hx0 * dndx[e, i, 0]
                                         + hx1 * dndx[e, i, 1]
                                         + hx2 * dndx[e, i, 2])
            q0 = 0.0; q1 = 0.0; q2 = 0.0
            for i in range(8):
                g = gam[e, a, i]
                q0 += g * ve[e, i, 0]
                q1 += g * ve[e, i, 1]
                q2 += g * ve[e, i, 2]
            qd[e, a, 0] = q0; qd[e, a, 1] = q1; qd[e, a, 2] = q2
        ah = hcoef[e] * rho[e] * c[e] * vol[e] ** (2.0 / 3.0) / 4.0
        if not live:
            ah = 0.0
        dh = 0.0
        for i in range(8):
            f0 = 0.0; f1 = 0.0; f2 = 0.0
            for a in range(4):
                g = gam[e, a, i]
                f0 += qd[e, a, 0] * g
                f1 += qd[e, a, 1] * g
                f2 += qd[e, a, 2] * g
            f0 *= -ah; f1 *= -ah; f2 *= -ah
            fe[e, i, 0] += f0
            fe[e, i, 1] += f1
            fe[e, i, 2] += f2
            dh += f0 * ve[e, i, 0] + f1 * ve[e, i, 1] + f2 * ve[e, i, 2]
        dehour[e] = -dh * dt

        # energies (see the reference for the trapezoidal qb booking)
        w_visc[e] = (0.5 * vol[e] * qv * (-trD[e] * dt)
                     + qvw_pend[e] * (-trD[e]))
        qvw_new[e] = 0.5 * vol[e] * qv * dt
        de = 0.0
        for k in range(6):
            de += 0.5 * (sig_old[e, k] + sig[e, k]) * deps[e, k]
        deint0[e] = vol[e] * de

        # critical time step
        if compressing:
            Q = qb[e] * c[e] + qa[e] * lc[e] * abs(trD[e])
        else:
            Q = 0.0
        if live:
            dt_crit[e] = dtfac[e] * lc[e] / (Q + np.sqrt(Q * Q
                                                         + c[e] * c[e]))
        else:
            dt_crit[e] = EP30
    return fe, dt_crit, w_visc, qvw_new, deint0, dehour


# ============================================================================
# shell_bt4 mirrors
# ============================================================================

@njit(cache=True, parallel=True)
def shell_pre(xe, ve, vre, off):
    """Mirror of shell_bt4._pre — corotational frame, local geometry and
    rate kinematics. Returns (E, area, lc, B1, B2, bb, gam, V, dm, kap,
    gs) — see the reference docstring for each output."""
    n = xe.shape[0]
    E = np.empty((n, 3, 3))
    area = np.empty(n)
    lc = np.empty(n)
    B1 = np.empty((n, 4))
    B2 = np.empty((n, 4))
    bb = np.empty(n)
    gam = np.empty((n, 4))
    V = np.empty((n, 4, 5))
    dm = np.empty((n, 3))
    kap = np.empty((n, 3))
    gs = np.empty((n, 2))
    xl = np.empty((n, 4, 2))

    for e in prange(n):
        # frame: e3 from the diagonals, e1 = side 1-2 projected, e2 = e3xe1
        r31x = xe[e, 2, 0] - xe[e, 0, 0]
        r31y = xe[e, 2, 1] - xe[e, 0, 1]
        r31z = xe[e, 2, 2] - xe[e, 0, 2]
        r42x = xe[e, 3, 0] - xe[e, 1, 0]
        r42y = xe[e, 3, 1] - xe[e, 1, 1]
        r42z = xe[e, 3, 2] - xe[e, 1, 2]
        e3x = r31y * r42z - r31z * r42y
        e3y = r31z * r42x - r31x * r42z
        e3z = r31x * r42y - r31y * r42x
        nrm = np.sqrt(e3x * e3x + e3y * e3y + e3z * e3z)
        if nrm < EM20:
            nrm = EM20
        e3x /= nrm; e3y /= nrm; e3z /= nrm
        s1x = xe[e, 1, 0] - xe[e, 0, 0]
        s1y = xe[e, 1, 1] - xe[e, 0, 1]
        s1z = xe[e, 1, 2] - xe[e, 0, 2]
        dot = s1x * e3x + s1y * e3y + s1z * e3z
        e1x = s1x - dot * e3x
        e1y = s1y - dot * e3y
        e1z = s1z - dot * e3z
        nrm = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        if nrm < EM20:
            nrm = EM20
        e1x /= nrm; e1y /= nrm; e1z /= nrm
        e2x = e3y * e1z - e3z * e1y
        e2y = e3z * e1x - e3x * e1z
        e2z = e3x * e1y - e3y * e1x
        E[e, 0, 0] = e1x; E[e, 1, 0] = e1y; E[e, 2, 0] = e1z
        E[e, 0, 1] = e2x; E[e, 1, 1] = e2y; E[e, 2, 1] = e2z
        E[e, 0, 2] = e3x; E[e, 1, 2] = e3y; E[e, 2, 2] = e3z

        # local corner coordinates about the center (z never used)
        cx = 0.25 * (xe[e, 0, 0] + xe[e, 1, 0] + xe[e, 2, 0] + xe[e, 3, 0])
        cy = 0.25 * (xe[e, 0, 1] + xe[e, 1, 1] + xe[e, 2, 1] + xe[e, 3, 1])
        cz = 0.25 * (xe[e, 0, 2] + xe[e, 1, 2] + xe[e, 2, 2] + xe[e, 3, 2])
        for i in range(4):
            dx = xe[e, i, 0] - cx
            dy = xe[e, i, 1] - cy
            dz = xe[e, i, 2] - cz
            xl[e, i, 0] = dx * e1x + dy * e1y + dz * e1z
            xl[e, i, 1] = dx * e2x + dy * e2y + dz * e2z

        A = 0.5 * ((xl[e, 2, 0] - xl[e, 0, 0]) * (xl[e, 3, 1] - xl[e, 1, 1])
                   + (xl[e, 1, 0] - xl[e, 3, 0]) * (xl[e, 2, 1] - xl[e, 0, 1]))
        twoA = 2.0 * A
        if twoA < EM20:
            twoA = EM20
        inv2A = 1.0 / twoA
        B1[e, 0] = (xl[e, 1, 1] - xl[e, 3, 1]) * inv2A
        B1[e, 1] = (xl[e, 2, 1] - xl[e, 0, 1]) * inv2A
        B1[e, 2] = (xl[e, 3, 1] - xl[e, 1, 1]) * inv2A
        B1[e, 3] = (xl[e, 0, 1] - xl[e, 2, 1]) * inv2A
        B2[e, 0] = (xl[e, 3, 0] - xl[e, 1, 0]) * inv2A
        B2[e, 1] = (xl[e, 0, 0] - xl[e, 2, 0]) * inv2A
        B2[e, 2] = (xl[e, 1, 0] - xl[e, 3, 0]) * inv2A
        B2[e, 3] = (xl[e, 2, 0] - xl[e, 0, 0]) * inv2A
        area[e] = A if A > EM20 else EM20

        # lc = A / longest side (local 2D)
        lmax = 0.0
        for i in range(4):
            j = i + 1 if i < 3 else 0
            dx = xl[e, j, 0] - xl[e, i, 0]
            dy = xl[e, j, 1] - xl[e, i, 1]
            l2 = dx * dx + dy * dy
            if l2 > lmax:
                lmax = l2
        lden = np.sqrt(lmax)
        if lden < EM20:
            lden = EM20
        lc[e] = area[e] / lden

        # local nodal rates [vx, vy, vz, thx, thy]
        for i in range(4):
            vx = ve[e, i, 0]; vy = ve[e, i, 1]; vz = ve[e, i, 2]
            V[e, i, 0] = vx * e1x + vy * e1y + vz * e1z
            V[e, i, 1] = vx * e2x + vy * e2y + vz * e2z
            V[e, i, 2] = vx * e3x + vy * e3y + vz * e3z
            wx = vre[e, i, 0]; wy = vre[e, i, 1]; wz = vre[e, i, 2]
            V[e, i, 3] = wx * e1x + wy * e1y + wz * e1z
            V[e, i, 4] = wx * e2x + wy * e2y + wz * e2z

        # the ten B.v dot products (M[a,k] = B_a . field_k)
        m00 = 0.0; m01 = 0.0; m02 = 0.0; m03 = 0.0; m04 = 0.0
        m10 = 0.0; m11 = 0.0; m12 = 0.0; m13 = 0.0; m14 = 0.0
        thx_m = 0.0; thy_m = 0.0
        for i in range(4):
            b1 = B1[e, i]; b2 = B2[e, i]
            m00 += b1 * V[e, i, 0]; m10 += b2 * V[e, i, 0]
            m01 += b1 * V[e, i, 1]; m11 += b2 * V[e, i, 1]
            m02 += b1 * V[e, i, 2]; m12 += b2 * V[e, i, 2]
            m03 += b1 * V[e, i, 3]; m13 += b2 * V[e, i, 3]
            m04 += b1 * V[e, i, 4]; m14 += b2 * V[e, i, 4]
            thx_m += V[e, i, 3]
            thy_m += V[e, i, 4]
        thx_m *= 0.25
        thy_m *= 0.25

        alive = off[e] > 0.0
        if alive:
            dm[e, 0] = m00
            dm[e, 1] = m11
            dm[e, 2] = m01 + m10
            kap[e, 0] = m04
            kap[e, 1] = -m13
            kap[e, 2] = m14 - m03
            gs[e, 0] = m02 + thy_m
            gs[e, 1] = m12 - thx_m
        else:                        # deleted: frozen, no straining
            dm[e, 0] = 0.0; dm[e, 1] = 0.0; dm[e, 2] = 0.0
            kap[e, 0] = 0.0; kap[e, 1] = 0.0; kap[e, 2] = 0.0
            gs[e, 0] = 0.0; gs[e, 1] = 0.0

        # hourglass shape vector and B.B stiffness factor
        hx = xl[e, 0, 0] - xl[e, 1, 0] + xl[e, 2, 0] - xl[e, 3, 0]
        hy = xl[e, 0, 1] - xl[e, 1, 1] + xl[e, 2, 1] - xl[e, 3, 1]
        gam[e, 0] = 1.0 - hx * B1[e, 0] - hy * B2[e, 0]
        gam[e, 1] = -1.0 - hx * B1[e, 1] - hy * B2[e, 1]
        gam[e, 2] = 1.0 - hx * B1[e, 2] - hy * B2[e, 2]
        gam[e, 3] = -1.0 - hx * B1[e, 3] - hy * B2[e, 3]
        s = 0.0
        for i in range(4):
            s += B1[e, i] * B1[e, i] + B2[e, i] * B2[e, i]
        bb[e] = s
    return E, area, lc, B1, B2, bb, gam, V, dm, kap, gs


@njit(cache=True, parallel=True)
def shell_post(E, area, B1, B2, gam, V, Nres, Mres, qres, Q,
               k_m, k_w, hqm, hqb, hqr, dt):
    """Mirror of shell_bt4._post — resultant nodal forces, chvis3.F
    hourglass (elastic Q updated in place + quadratic viscous dampers),
    back-transform to global axes. Returns (fg, mg, dehg)."""
    n = area.shape[0]
    fg = np.empty((n, 4, 3))
    mg = np.empty((n, 4, 3))
    dehg = np.empty(n)
    qd = np.empty((n, 5))
    F = np.empty((n, 5))

    for e in prange(n):
        A = area[e]
        # modal velocities: translations 0-2 on gamma, rotations 3-4 on the
        # RAW h = (1,-1,1,-1) pattern (chvis3.F lines 327-330)
        for k in range(3):
            s = 0.0
            for i in range(4):
                s += gam[e, i] * V[e, i, k]
            qd[e, k] = s
        for k in range(3, 5):
            qd[e, k] = V[e, 0, k] - V[e, 1, k] + V[e, 2, k] - V[e, 3, k]
        # elastic branch (modes 0-2 only); rotation carries no elastic state
        Q[e, 0] = Q[e, 0] + k_m[e] * qd[e, 0] * dt
        Q[e, 1] = Q[e, 1] + k_m[e] * qd[e, 1] * dt
        Q[e, 2] = Q[e, 2] + k_w[e] * qd[e, 2] * dt
        Q[e, 3] = 0.0
        Q[e, 4] = 0.0
        # total modal force = elastic + quadratic viscous damper
        F[e, 0] = Q[e, 0] + qd[e, 0] * hqm[e] * abs(qd[e, 0])
        F[e, 1] = Q[e, 1] + qd[e, 1] * hqm[e] * abs(qd[e, 1])
        F[e, 2] = Q[e, 2] + qd[e, 2] * hqb[e] * abs(qd[e, 2])
        F[e, 3] = qd[e, 3] * hqr[e] * abs(qd[e, 3])
        F[e, 4] = qd[e, 4] * hqr[e] * abs(qd[e, 4])
        # modal energy increment
        de = 0.0
        for k in range(5):
            de += F[e, k] * qd[e, k] * dt
        dehg[e] = de

        # distribute back to the 4 nodes
        for i in range(4):
            b1 = B1[e, i]; b2 = B2[e, i]; g = gam[e, i]
            h = 1.0 if (i % 2) == 0 else -1.0        # raw h = (1,-1,1,-1)
            # local total force = -(internal) - F*gamma (hourglass)
            flx = -A * (b1 * Nres[e, 0] + b2 * Nres[e, 2]) - g * F[e, 0]
            fly = -A * (b2 * Nres[e, 1] + b1 * Nres[e, 2]) - g * F[e, 1]
            flz = -A * (b1 * qres[e, 0] + b2 * qres[e, 1]) - g * F[e, 2]
            mlx = -A * (-b2 * Mres[e, 1] - b1 * Mres[e, 2]
                        - 0.25 * qres[e, 1]) - h * F[e, 3]
            mly = -A * (b1 * Mres[e, 0] + b2 * Mres[e, 2]
                        + 0.25 * qres[e, 0]) - h * F[e, 4]
            # back to global axes: fg[b] = sum_a fl[a] E[b,a]
            fg[e, i, 0] = flx * E[e, 0, 0] + fly * E[e, 0, 1] + flz * E[e, 0, 2]
            fg[e, i, 1] = flx * E[e, 1, 0] + fly * E[e, 1, 1] + flz * E[e, 1, 2]
            fg[e, i, 2] = flx * E[e, 2, 0] + fly * E[e, 2, 1] + flz * E[e, 2, 2]
            mg[e, i, 0] = mlx * E[e, 0, 0] + mly * E[e, 0, 1]
            mg[e, i, 1] = mlx * E[e, 1, 0] + mly * E[e, 1, 1]
            mg[e, i, 2] = mlx * E[e, 2, 0] + mly * E[e, 2, 1]
    return fg, mg, dehg


# ============================================================================
# TYPE7 narrow phase mirror
# ============================================================================

@njit(cache=True)
def _tri_closest(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """Scalar Ericson closest-point-on-triangle; the sequential
    region-overwrite order matches the vectorized reference exactly
    (later region tests overwrite earlier ones, so plain sequential
    ``if`` blocks — NOT elif — reproduce the same winner)."""
    abx = bx - ax; aby = by - ay; abz = bz - az
    acx = cx - ax; acy = cy - ay; acz = cz - az
    apx = px - ax; apy = py - ay; apz = pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    bpx = px - bx; bpy = py - by; bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    cpx = px - cx; cpy = py - cy; cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom = va + vb + vc
    if denom < EM20:
        denom = EM20
    v = vb / denom
    w = vc / denom
    if (vc <= 0.0) and (d1 >= 0.0) and (d3 <= 0.0):     # edge AB
        den = d1 - d3
        if den < EM20:
            den = EM20
        v = d1 / den
        w = 0.0
    if (vb <= 0.0) and (d2 >= 0.0) and (d6 <= 0.0):     # edge AC
        den = d2 - d6
        if den < EM20:
            den = EM20
        v = 0.0
        w = d2 / den
    if (va <= 0.0) and (d4 - d3 >= 0.0) and (d5 - d6 >= 0.0):   # edge BC
        den = (d4 - d3) + (d5 - d6)
        if den < EM20:
            den = EM20
        t = (d4 - d3) / den
        v = 1.0 - t
        w = t
    if (d1 <= 0.0) and (d2 <= 0.0):                     # vertex A
        v = 0.0
        w = 0.0
    if (d3 >= 0.0) and (d4 <= d3):                      # vertex B
        v = 1.0
        w = 0.0
    if (d6 >= 0.0) and (d5 <= d6):                      # vertex C
        v = 0.0
        w = 1.0

    u = 1.0 - v - w
    qx = u * ax + v * bx + w * cx
    qy = u * ay + v * by + w * cy
    qz = u * az + v * bz + w * cz
    return qx, qy, qz, u, v, w


@njit(cache=True)
def t7_narrow(x, ni, seg):
    """Mirror of inter_type7._narrow — closest point of each candidate
    node on its quad segment (two triangles, best wins). Returns
    (best_d, best_pt, best_w)."""
    m = ni.shape[0]
    best_d = np.empty(m)
    best_pt = np.empty((m, 3))
    best_w = np.empty((m, 4))

    for k in range(m):
        node = ni[k]
        px = x[node, 0]; py = x[node, 1]; pz = x[node, 2]
        i0 = seg[k, 0]; i1 = seg[k, 1]; i2 = seg[k, 2]; i3 = seg[k, 3]

        # triangle (0, 1, 2)
        qx, qy, qz, u, v, w = _tri_closest(
            px, py, pz,
            x[i0, 0], x[i0, 1], x[i0, 2],
            x[i1, 0], x[i1, 1], x[i1, 2],
            x[i2, 0], x[i2, 1], x[i2, 2])
        dx = px - qx; dy = py - qy; dz = pz - qz
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        best_d[k] = d
        best_pt[k, 0] = qx; best_pt[k, 1] = qy; best_pt[k, 2] = qz
        best_w[k, 0] = u; best_w[k, 1] = v
        best_w[k, 2] = w; best_w[k, 3] = 0.0

        # triangle (0, 2, 3)
        qx, qy, qz, u, v, w = _tri_closest(
            px, py, pz,
            x[i0, 0], x[i0, 1], x[i0, 2],
            x[i2, 0], x[i2, 1], x[i2, 2],
            x[i3, 0], x[i3, 1], x[i3, 2])
        dx = px - qx; dy = py - qy; dz = pz - qz
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        if d < best_d[k]:
            best_d[k] = d
            best_pt[k, 0] = qx; best_pt[k, 1] = qy; best_pt[k, 2] = qz
            best_w[k, 0] = u; best_w[k, 1] = 0.0
            best_w[k, 2] = v; best_w[k, 3] = w
    return best_d, best_pt, best_w


# ============================================================================
# solid_hexa8 physical (Belytschko-Bindeman) hourglass STIFFNESS — LAW70 foam
# ============================================================================
# Mirror of solid_hexa8._phys_hourglass_law70 (M38): the HEPH/Isolid=24
# stiffness-hourglass restoring force the /MAT/LAW70 foam asks for. Under numba
# this was the c46 regime's #1 self-time block (a per-cycle einsum/matmul stack
# over the whole brick group). READ solid_hexa8._phys_hourglass_law70 for the
# physics — every formula is documented there. These two constants MUST match
# solid_hexa8.HG_PHYS / DT_HG_SF (the numba backend freezes globals at compile
# time, so they are duplicated here as literals exactly like _H4/_XI8; the M39
# parity test asserts the two stay equal).
_HG_PHYS = 0.03      # == solid_hexa8.HG_PHYS
_DT_HG_SF = 0.9      # == solid_hexa8.DT_HG_SF


@njit(cache=True)
def hexa_hgphys(xe, ve, dndx, vol, c, mask, mass, vol0, q, dt):
    """Mirror of solid_hexa8._phys_hourglass_law70 — the Flanagan-Belytschko
    (HEPH) physical hourglass STIFFNESS for the LAW70 slices of a brick group.
    Accumulates the hourglass MODAL displacement ``q`` (== st['hgqex'], shape
    (n,4,3)) IN PLACE, exactly as hexa_pre mutates ``sig``: masked (LAW70)
    elements only. Returns (f_hg, dehour, dt_hg) — the (n,8,3) nodal restoring
    force to add to the internal force, the (n,) hourglass-energy increment,
    and the (n,) element time-step cap. Non-LAW70 elements get zero force /
    EP30 dt. Parity: element-wise expressions bitwise; the 8- and 4-term dot
    products are sequential (NumPy matmul/einsum may reassociate) — the
    documented ~1e-15 per-call reduction tolerance (rtol 1e-12 kernel test)."""
    n = xe.shape[0]
    f_hg = np.empty((n, 8, 3))
    dehour = np.empty(n)
    dt_hg = np.empty(n)
    gamma = np.empty((n, 4, 8))

    for e in prange(n):
        # hourglass base vectors: hx[a,b] = sum_i H[a,i] xe[i,b]
        # gamma[a,i] = H[a,i] - (sum_b hx[a,b] gradN[i,b]) : orthogonalized
        for a in range(4):
            hx0 = 0.0; hx1 = 0.0; hx2 = 0.0
            for i in range(8):
                h = _H4[a, i]
                hx0 += h * xe[e, i, 0]
                hx1 += h * xe[e, i, 1]
                hx2 += h * xe[e, i, 2]
            for i in range(8):
                gamma[e, a, i] = _H4[a, i] - (hx0 * dndx[e, i, 0]
                                           + hx1 * dndx[e, i, 1]
                                           + hx2 * dndx[e, i, 2])

        # current P-wave modulus AA1 = rho0 c^2 (rho0 = mass/vol0)
        v0 = vol0[e]
        if v0 < EM20:
            v0 = EM20
        aa1 = (mass[e] / v0) * c[e] * c[e]
        # traceS = sum_{i,a} gradN[i,a]^2  (einsum "nia,nia->n" on dndx)
        traceS = 0.0
        for i in range(8):
            for a in range(3):
                g = dndx[e, i, a]
                traceS += g * g
        if mask[e]:
            kstiff = _HG_PHYS * aa1 * vol[e] * traceS
        else:
            kstiff = 0.0

        # accumulate the hourglass modal displacement (rate form, LAW70 only):
        # q[a,b] += (sum_i gamma[a,i] ve[i,b]) dt
        if mask[e]:
            for a in range(4):
                s0 = 0.0; s1 = 0.0; s2 = 0.0
                for i in range(8):
                    g = gamma[e, a, i]
                    s0 += g * ve[e, i, 0]
                    s1 += g * ve[e, i, 1]
                    s2 += g * ve[e, i, 2]
                q[e, a, 0] += s0 * dt
                q[e, a, 1] += s1 * dt
                q[e, a, 2] += s2 * dt

        # f_hg[i,b] = -kstiff * sum_a gamma[a,i] q[a,b]; dehour = -f_hg.ve dt
        nk = -kstiff
        dh = 0.0
        for i in range(8):
            f0 = 0.0; f1 = 0.0; f2 = 0.0
            for a in range(4):
                g = gamma[e, a, i]
                f0 += g * q[e, a, 0]
                f1 += g * q[e, a, 1]
                f2 += g * q[e, a, 2]
            f0 *= nk; f1 *= nk; f2 *= nk
            f_hg[e, i, 0] = f0
            f_hg[e, i, 1] = f1
            f_hg[e, i, 2] = f2
            dh += f0 * ve[e, i, 0] + f1 * ve[e, i, 1] + f2 * ve[e, i, 2]
        dehour[e] = -dh * dt

        # dt cap: omega_max^2 = 2 kstiff gnorm / mass, gnorm = sum|gamma|^2
        gnorm = 0.0
        for a in range(4):
            for i in range(8):
                g = gamma[e, a, i]
                gnorm += g * g
        if kstiff > 0.0:
            mm = mass[e]
            if mm < EM20:
                mm = EM20
            den = 2.0 * kstiff * gnorm
            if den < EM20:
                den = EM20
            dt_hg[e] = _DT_HG_SF * np.sqrt(mm / den)
        else:
            dt_hg[e] = EP30
    return f_hg, dehour, dt_hg


# ============================================================================
# force scatter-assembly (Fortran asspar) — the shared force-scatter primitive
# ============================================================================
# Mirror of common.fastmath.scatter_add3 (the NumPy reference). Every element
# kernel scatters its (n_elem*nodes, 3) nodal force block into the global
# (N, 3) fint/mint via this one primitive; on the 65 439-brick cliff the
# reference's three np.bincount passes (one per component, each a full pass
# over the ~0.5 M index list) were the measured assembly cost. This fuses the
# three components into ONE pass over the index list.
#
# The M7 parity contract: this MUST reproduce the bincount reference BITWISE
# (fastmath.scatter_add3 is what the parity tests assert against). It does, by
# construction — bincount accumulates weights[k] into bin[idx[k]] in INPUT (k)
# order, then the reference adds the full-length bin array to the target. This
# kernel accumulates the three components into a zeroed (N, 3) scratch in the
# SAME k order, then adds the scratch to the target with the SAME elementwise
# add — identical operations in an identical order, so the result is
# bit-for-bit equal to the reference for ANY starting target (a node already
# carrying force from another group included; the direct-scatter shortcut
# ((f+v1)+v2) would NOT match the reference's f+(v1+v2), so the scratch is
# load-bearing, not an optimisation to skip). Verified against the reference on
# both zero and non-zero targets by tests/test_m7_backends. No parallel (the
# scatter order must stay deterministic — the restart-chaining canary), no
# fastmath (it would license reassociation of the per-bin sums).
@njit(cache=True)
def scatter3(target, idx, values):
    """Bitwise mirror of fastmath.scatter_add3: accumulate the (m, 3)
    ``values`` into ``target`` (N, 3) at rows ``idx`` (m,), in input order,
    via a zeroed scratch (so cross-group additions associate exactly as the
    bincount reference does). In place on ``target``; returns nothing."""
    n = target.shape[0]
    m = idx.shape[0]
    acc = np.zeros((n, 3))
    for k in range(m):
        j = idx[k]
        acc[j, 0] += values[k, 0]
        acc[j, 1] += values[k, 1]
        acc[j, 2] += values[k, 2]
    target += acc

@njit(cache=True)
def scatter3_colored(target, idx, values, color_indices, color_offsets, npe):
    """Node-colored accumulation (serial fallback without parallel overhead)."""
    num_colors = len(color_offsets) - 1
    for c in range(num_colors):
        for k in range(color_offsets[c], color_offsets[c+1]):
            e = color_indices[k]
            for i in range(npe):
                flat_idx = e * npe + i
                row = idx[flat_idx]
                for comp in range(3):
                    target[row, comp] += values[flat_idx, comp]


# ============================================================================
# LAW70 tabulated-foam numeric leaves (M39 — the "LAW70 curve lookups")
# ============================================================================
# Mirrors of the hot leaf helpers of materials/law70_tabfoam.solid_update
# (sigeps70.F): the (strain, rate) table interpolation and the Voigt
# strain/stress norms and elastic map. Each is a SINGLE scalar expression per
# element with NO reduction, so — unlike the reduction-bearing hexa/shell
# mirrors — these are BITWISE-identical to the NumPy reference (verified 0-ulp
# in tests/test_m39_accel_law70). They are the c46-regime material hotspot
# once the hexa blocks are compiled. READ the law70_tabfoam reference for the
# physics; dispatched by solid_update via accel.get, NumPy path is reference.

@njit(cache=True)
def law70_tab2d(xg, rates, Y, x, r):
    """Mirror of law70_tabfoam._tab2d — bilinear (strain, rate) lookup with
    end-slope extrapolation in both dims (TABLE_MAT_VINTERP). Bitwise: same
    searchsorted index, same unclamped interpolation expression per element.
    ``xg`` is strictly increasing (np.unique in _build_table) so the
    denominators are never zero."""
    m = xg.shape[0]
    nr = rates.shape[0]
    n = x.shape[0]
    out = np.empty(n)
    for k in range(n):
        xk = x[k]
        i = np.searchsorted(xg, xk, side="right")
        if i < 1:
            i = 1
        elif i > m - 1:
            i = m - 1
        t = (xk - xg[i - 1]) / (xg[i] - xg[i - 1])       # unclamped
        if nr == 1:
            out[k] = Y[i - 1, 0] + t * (Y[i, 0] - Y[i - 1, 0])
        else:
            rk = r[k]
            j = np.searchsorted(rates, rk, side="right")
            if j < 1:
                j = 1
            elif j > nr - 1:
                j = nr - 1
            u = (rk - rates[j - 1]) / (rates[j] - rates[j - 1])   # unclamped
            y0 = Y[i - 1, j - 1] + t * (Y[i, j - 1] - Y[i - 1, j - 1])
            y1 = Y[i - 1, j] + t * (Y[i, j] - Y[i - 1, j])
            out[k] = y0 + u * (y1 - y0)
    return out


@njit(cache=True)
def law70_enorm(v):
    """Mirror of law70_tabfoam._enorm — tensor norm of a Voigt STRAIN
    (0.5 on the engineering shears). Bitwise (per-element scalar sqrt)."""
    n = v.shape[0]
    out = np.empty(n)
    for k in range(n):
        out[k] = np.sqrt(v[k, 0] ** 2 + v[k, 1] ** 2 + v[k, 2] ** 2
                         + 0.5 * (v[k, 3] ** 2 + v[k, 4] ** 2 + v[k, 5] ** 2))
    return out


@njit(cache=True)
def law70_snorm(v):
    """Mirror of law70_tabfoam._snorm — Frobenius norm of a Voigt STRESS
    (2.0 on the shears). Bitwise (per-element scalar sqrt)."""
    n = v.shape[0]
    out = np.empty(n)
    for k in range(n):
        out[k] = np.sqrt(v[k, 0] ** 2 + v[k, 1] ** 2 + v[k, 2] ** 2
                         + 2.0 * (v[k, 3] ** 2 + v[k, 4] ** 2 + v[k, 5] ** 2))
    return out


@njit(cache=True)
def law70_elastic_stress(aa1, aa2, g, e):
    """Mirror of law70_tabfoam._elastic_stress — C(E):eps for per-element
    moduli (Voigt, engineering shear). Bitwise (per-element, per-component
    scalar expression identical to the NumPy columns)."""
    n = e.shape[0]
    out = np.empty((n, 6))
    for k in range(n):
        a1 = aa1[k]; a2 = aa2[k]; gk = g[k]
        e0 = e[k, 0]; e1 = e[k, 1]; e2 = e[k, 2]
        out[k, 0] = a1 * e0 + a2 * (e1 + e2)
        out[k, 1] = a1 * e1 + a2 * (e0 + e2)
        out[k, 2] = a1 * e2 + a2 * (e0 + e1)
        out[k, 3] = gk * e[k, 3]
        out[k, 4] = gk * e[k, 4]
        out[k, 5] = gk * e[k, 5]
    return out

# ============================================================================
# solid_tetra10 mirrors (10-node tet, 4-point Gauss integration)
# ============================================================================
# Mirror of solid_tetra10._pre/_post — the same parity contract as hexa_pre/
# hexa_post: element-wise expressions bitwise, short reductions sequential.
# READ solid_tetra10 for the physics (every formula documented there).

# Shape-function derivatives at 4 Gauss points: _DN_DXI_T10[k, i, a] =
# dN_i/dxi_a at Gauss point k.  (4 Gauss pts × 10 nodes × 3 parent dims)
_DN_DXI_T10 = np.array([[[-0.44721360, 0.00000000, 0.00000000],
  [ 0.00000000,-0.44721360, 0.00000000],
  [ 0.00000000, 0.00000000,-0.44721360],
  [-1.34164079,-1.34164079,-1.34164079],
  [ 0.55278640, 0.55278640, 0.00000000],
  [ 0.00000000, 0.55278640, 0.55278640],
  [ 0.55278640, 0.00000000, 0.55278640],
  [ 1.78885438,-0.55278640,-0.55278640],
  [-0.55278640, 1.78885438,-0.55278640],
  [-0.55278640,-0.55278640, 1.78885438]],
 [[ 1.34164079, 0.00000000, 0.00000000],
  [ 0.00000000,-0.44721360, 0.00000000],
  [ 0.00000000, 0.00000000,-0.44721360],
  [ 0.44721360, 0.44721360, 0.44721360],
  [ 0.55278640, 2.34164079, 0.00000000],
  [ 0.00000000, 0.55278640, 0.55278640],
  [ 0.55278640, 0.00000000, 2.34164079],
  [-1.78885438,-2.34164079,-2.34164079],
  [-0.55278640, 0.00000000,-0.55278640],
  [-0.55278640,-0.55278640, 0.00000000]],
 [[-0.44721360, 0.00000000, 0.00000000],
  [ 0.00000000, 1.34164079, 0.00000000],
  [ 0.00000000, 0.00000000,-0.44721360],
  [ 0.44721360, 0.44721360, 0.44721360],
  [ 2.34164079, 0.55278640, 0.00000000],
  [ 0.00000000, 0.55278640, 2.34164079],
  [ 0.55278640, 0.00000000, 0.55278640],
  [ 0.00000000,-0.55278640,-0.55278640],
  [-2.34164079,-1.78885438,-2.34164079],
  [-0.55278640,-0.55278640, 0.00000000]],
 [[-0.44721360, 0.00000000, 0.00000000],
  [ 0.00000000,-0.44721360, 0.00000000],
  [ 0.00000000, 0.00000000, 1.34164079],
  [ 0.44721360, 0.44721360, 0.44721360],
  [ 0.55278640, 0.55278640, 0.00000000],
  [ 0.00000000, 2.34164079, 0.55278640],
  [ 2.34164079, 0.00000000, 0.55278640],
  [ 0.00000000,-0.55278640,-0.55278640],
  [-0.55278640, 0.00000000,-0.55278640],
  [-2.34164079,-2.34164079,-1.78885438]]])

_WIP_T10 = np.array([0.25, 0.25, 0.25, 0.25])

# 4 triangular faces of the corner tetrahedron (corner node indices only)
_FACES_T10 = np.array([
    [0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2],
], dtype=np.int64)


@njit(cache=True, parallel=True)
def tetra10_pre(xe, ve, sig, dt, off):
    """Mirror of solid_tetra10._pre — geometry at 4 Gauss points, velocity
    gradient, Jaumann rotation (in place on sig).  Returns
    (dndx, vol, vol_tot, lc, deps, trD).

    dndx: (n, 4, 10, 3)  —  shape-function derivatives in physical space
    vol:  (n, 4)          —  sub-volumes at each Gauss point (detJ / 6)
    vol_tot: (n,)         —  total volume (weighted sum)
    lc:   (n,)            —  characteristic length
    deps: (n, 4, 6)       —  strain increment (Voigt, engineering shear)
    trD:  (n, 4)          —  trace of rate-of-deformation
    """
    n = xe.shape[0]
    dndx = np.empty((n, 4, 10, 3))
    vol = np.empty((n, 4))
    vol_tot = np.empty(n)
    lc = np.empty(n)
    deps = np.empty((n, 4, 6))
    trD = np.empty((n, 4))

    for e in prange(n):
        # ---- geometry: Jacobian + inverse at each Gauss point ----------
        vtot = 0.0
        for k in range(4):
            # J[a,b] = sum_i dN_dxi[k,i,a] * xe[i,b]
            j00 = 0.0; j01 = 0.0; j02 = 0.0
            j10 = 0.0; j11 = 0.0; j12 = 0.0
            j20 = 0.0; j21 = 0.0; j22 = 0.0
            for i in range(10):
                d0 = _DN_DXI_T10[k, i, 0]
                d1 = _DN_DXI_T10[k, i, 1]
                d2 = _DN_DXI_T10[k, i, 2]
                x0 = xe[e, i, 0]; x1 = xe[e, i, 1]; x2 = xe[e, i, 2]
                j00 += d0 * x0; j01 += d0 * x1; j02 += d0 * x2
                j10 += d1 * x0; j11 += d1 * x1; j12 += d1 * x2
                j20 += d2 * x0; j21 += d2 * x1; j22 += d2 * x2
            # cofactor determinant / inverse
            A = j11 * j22 - j12 * j21
            B = j12 * j20 - j10 * j22
            C = j10 * j21 - j11 * j20
            det = j00 * A + j01 * B + j02 * C
            vol[e, k] = det / 6.0
            idet = np.divide(1.0, det)
            i00 = A * idet
            i01 = (j02 * j21 - j01 * j22) * idet
            i02 = (j01 * j12 - j02 * j11) * idet
            i10 = B * idet
            i11 = (j00 * j22 - j02 * j20) * idet
            i12 = (j02 * j10 - j00 * j12) * idet
            i20 = C * idet
            i21 = (j01 * j20 - j00 * j21) * idet
            i22 = (j00 * j11 - j01 * j10) * idet
            # dndx[k,i,b] = sum_a dN_dxi[k,i,a] * Jinv[b,a]
            for i in range(10):
                d0 = _DN_DXI_T10[k, i, 0]
                d1 = _DN_DXI_T10[k, i, 1]
                d2 = _DN_DXI_T10[k, i, 2]
                dndx[e, k, i, 0] = d0 * i00 + d1 * i01 + d2 * i02
                dndx[e, k, i, 1] = d0 * i10 + d1 * i11 + d2 * i12
                dndx[e, k, i, 2] = d0 * i20 + d1 * i21 + d2 * i22
            vtot += _WIP_T10[k] * vol[e, k]

        if vtot < EM20:
            vtot = EM20
        vol_tot[e] = vtot

        # characteristic length: 3 * V / max face area (corner faces only)
        amax = 0.0
        for f in range(4):
            f0 = _FACES_T10[f, 0]; f1 = _FACES_T10[f, 1]; f2 = _FACES_T10[f, 2]
            e1x = xe[e, f1, 0] - xe[e, f0, 0]
            e1y = xe[e, f1, 1] - xe[e, f0, 1]
            e1z = xe[e, f1, 2] - xe[e, f0, 2]
            e2x = xe[e, f2, 0] - xe[e, f0, 0]
            e2y = xe[e, f2, 1] - xe[e, f0, 1]
            e2z = xe[e, f2, 2] - xe[e, f0, 2]
            cx = e1y * e2z - e1z * e2y
            cy = e1z * e2x - e1x * e2z
            cz = e1x * e2y - e1y * e2x
            a = 0.5 * np.sqrt(cx * cx + cy * cy + cz * cz)
            if a > amax:
                amax = a
        if amax < EM20:
            amax = EM20
        lc[e] = 3.0 * vtot / amax

        # ---- velocity gradient at each Gauss point --------------------
        alive = off[e] > 0.0
        for k in range(4):
            # L[b,c] = sum_i ve[i,b] * dndx[k,i,c]
            l00 = 0.0; l01 = 0.0; l02 = 0.0
            l10 = 0.0; l11 = 0.0; l12 = 0.0
            l20 = 0.0; l21 = 0.0; l22 = 0.0
            for i in range(10):
                v0 = ve[e, i, 0]; v1 = ve[e, i, 1]; v2 = ve[e, i, 2]
                g0 = dndx[e, k, i, 0]
                g1 = dndx[e, k, i, 1]
                g2 = dndx[e, k, i, 2]
                l00 += v0 * g0; l01 += v0 * g1; l02 += v0 * g2
                l10 += v1 * g0; l11 += v1 * g1; l12 += v1 * g2
                l20 += v2 * g0; l21 += v2 * g1; l22 += v2 * g2

            if alive:
                tr = l00 + l11 + l22
                # round-off trace flush
                vmax = 0.0
                gmax = 0.0
                for i in range(10):
                    for b in range(3):
                        av = abs(ve[e, i, b])
                        if av > vmax:
                            vmax = av
                        ag = abs(dndx[e, k, i, b])
                        if ag > gmax:
                            gmax = ag
                if abs(tr) <= 1e-14 * (vmax * gmax):
                    tr = 0.0
                trD[e, k] = tr
                deps[e, k, 0] = l00 * dt
                deps[e, k, 1] = l11 * dt
                deps[e, k, 2] = l22 * dt
                deps[e, k, 3] = (l01 + l10) * dt
                deps[e, k, 4] = (l12 + l21) * dt
                deps[e, k, 5] = (l02 + l20) * dt
            else:
                trD[e, k] = 0.0
                for j in range(6):
                    deps[e, k, j] = 0.0

            # Jaumann rotation of stress at this Gauss point
            wxy = 0.5 * (l01 - l10) * dt
            wyz = 0.5 * (l12 - l21) * dt
            wxz = 0.5 * (l02 - l20) * dt
            sxx = sig[e, k, 0]; syy = sig[e, k, 1]; szz = sig[e, k, 2]
            sxy = sig[e, k, 3]; syz_s = sig[e, k, 4]; szx = sig[e, k, 5]
            sig[e, k, 0] = sxx + 2.0 * (wxy * sxy + wxz * szx)
            sig[e, k, 1] = syy + 2.0 * (-wxy * sxy + wyz * syz_s)
            sig[e, k, 2] = szz + 2.0 * (-wxz * szx - wyz * syz_s)
            sig[e, k, 3] = sxy + wxy * (syy - sxx) + wxz * syz_s + wyz * szx
            sig[e, k, 4] = syz_s + wyz * (szz - syy) - wxy * szx - wxz * sxy
            sig[e, k, 5] = szx + wxz * (szz - sxx) + wxy * syz_s - wyz * sxy

    return dndx, vol, vol_tot, lc, deps, trD


@njit(cache=True, parallel=True)
def tetra10_post(xe, dndx, vol, vol_tot, lc, rho, trD, deps, sig, sig_old,
                 qa, qb, c, alive, qvw_pend, dt, dtfac):
    """Mirror of solid_tetra10._post — bulk viscosity, internal forces,
    energies, critical dt.  Returns (fe, dt_crit, w_visc, qvw_new, deint0).

    fe:      (n, 10, 3) nodal forces (negated for fint accumulation)
    dt_crit: (n,)       element critical time step
    w_visc:  (n,)       viscous energy increment
    qvw_new: (n,)       pending viscous work
    deint0:  (n,)       internal energy increment
    """
    n = xe.shape[0]
    fe = np.empty((n, 10, 3))
    dt_crit = np.empty(n)
    w_visc = np.empty(n)
    qvw_new = np.empty(n)
    deint0 = np.empty(n)

    for e in prange(n):
        live = alive[e]

        # zero the nodal forces — accumulated over 4 Gauss points
        for i in range(10):
            fe[e, i, 0] = 0.0
            fe[e, i, 1] = 0.0
            fe[e, i, 2] = 0.0

        wv = 0.0       # viscous energy accumulator
        qvn = 0.0      # pending viscous work accumulator
        de = 0.0        # internal energy accumulator
        trD_min = trD[e, 0]

        for k in range(4):
            # bulk viscosity at this Gauss point
            compressing = (trD[e, k] < 0.0) and live
            if compressing:
                qv = (rho[e] * lc[e]
                      * (qa[e] * qa[e] * lc[e] * trD[e, k] * trD[e, k]
                         - qb[e] * c[e] * trD[e, k]))
            else:
                qv = 0.0

            # total stress with viscous pressure
            s00 = sig[e, k, 0] - qv
            s11 = sig[e, k, 1] - qv
            s22 = sig[e, k, 2] - qv
            s01 = sig[e, k, 3]
            s12 = sig[e, k, 4]
            s02 = sig[e, k, 5]

            # internal force: fe[i,b] += -w_k * vol_k * S[b,c] * dndx[k,i,c]
            wv_k = -_WIP_T10[k] * vol[e, k]
            for i in range(10):
                g0 = dndx[e, k, i, 0]
                g1 = dndx[e, k, i, 1]
                g2 = dndx[e, k, i, 2]
                fe[e, i, 0] += wv_k * (g0 * s00 + g1 * s01 + g2 * s02)
                fe[e, i, 1] += wv_k * (g0 * s01 + g1 * s11 + g2 * s12)
                fe[e, i, 2] += wv_k * (g0 * s02 + g1 * s12 + g2 * s22)

            # energy contributions at this Gauss point
            wk_vol = _WIP_T10[k] * vol[e, k]
            wv += wk_vol * 0.5 * qv * (-trD[e, k] * dt)
            qvn += wk_vol * 0.5 * qv * dt

            # internal energy: sig_mid . deps
            de_k = 0.0
            for j in range(6):
                de_k += 0.5 * (sig_old[e, k, j] + sig[e, k, j]) * deps[e, k, j]
            de += wk_vol * de_k

            # track minimum trD for dt calculation
            if trD[e, k] < trD_min:
                trD_min = trD[e, k]

        # add pending viscous work from previous cycle
        sum_neg_trD = 0.0
        for k in range(4):
            sum_neg_trD += -trD[e, k]
        wv += qvw_pend[e] * sum_neg_trD / 4.0

        w_visc[e] = wv
        qvw_new[e] = qvn
        deint0[e] = de + wv

        # critical time step
        if trD_min < 0.0:
            Q = qb[e] * c[e] + qa[e] * lc[e] * abs(trD_min)
        else:
            Q = 0.0
        if live:
            dt_crit[e] = dtfac[e] * lc[e] / (Q + np.sqrt(Q * Q + c[e] * c[e]))
        else:
            dt_crit[e] = EP30

    return fe, dt_crit, w_visc, qvw_new, deint0


from .shells_qbat import qbat_pre_flat, qbat_post_flat, qbat_pre, qbat_post

