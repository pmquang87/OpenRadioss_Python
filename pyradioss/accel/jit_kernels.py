"""
numba mirrors of the hot kernel blocks (the M7 accelerated backend).

Importing this module requires numba (an OPTIONAL dependency — the
import error is caught by ``accel.select_backend`` which then falls back
to NumPy). Each function here mirrors, operation for operation, one
NumPy block of the reference kernels:

    hexa_pre   <-> elements/solid_hexa8._pre     (srcoor3/sdefo3/srota3)
    hexa_post  <-> elements/solid_hexa8._post    (sbulk3/sfint3/shour3/sdlen3)
    shell_pre  <-> elements/shell_bt4._pre       (ccoor3/cdefo3)
    shell_post <-> elements/shell_bt4._post      (czforc3/chour3)
    t7_narrow  <-> contact/inter_type7._narrow   (i7dst3)

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
from numba import njit

from ..common.constants import EM20, EP30

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

@njit(cache=True)
def hexa_pre(xe, ve, sig, dt, off):
    """Mirror of solid_hexa8._pre — geometry, velocity gradient, Jaumann
    rotation (in place on sig). Returns (dndx, vol, lc, deps, trD)."""
    n = xe.shape[0]
    dndx = np.empty((n, 8, 3))
    vol = np.empty(n)
    lc = np.empty(n)
    deps = np.empty((n, 6))
    trD = np.empty(n)

    for e in range(n):
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
        lc[e] = vol[e] / (amax if amax > EM20 else EM20)

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


@njit(cache=True)
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
    gam = np.empty((4, 8))
    qd = np.empty((4, 3))

    for e in range(n):
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
                gam[a, i] = _H4[a, i] - (hx0 * dndx[e, i, 0]
                                         + hx1 * dndx[e, i, 1]
                                         + hx2 * dndx[e, i, 2])
            q0 = 0.0; q1 = 0.0; q2 = 0.0
            for i in range(8):
                g = gam[a, i]
                q0 += g * ve[e, i, 0]
                q1 += g * ve[e, i, 1]
                q2 += g * ve[e, i, 2]
            qd[a, 0] = q0; qd[a, 1] = q1; qd[a, 2] = q2
        ah = hcoef[e] * rho[e] * c[e] * vol[e] ** (2.0 / 3.0) / 4.0
        if not live:
            ah = 0.0
        dh = 0.0
        for i in range(8):
            f0 = 0.0; f1 = 0.0; f2 = 0.0
            for a in range(4):
                g = gam[a, i]
                f0 += qd[a, 0] * g
                f1 += qd[a, 1] * g
                f2 += qd[a, 2] * g
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

@njit(cache=True)
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
    xl = np.empty((4, 2))

    for e in range(n):
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
            xl[i, 0] = dx * e1x + dy * e1y + dz * e1z
            xl[i, 1] = dx * e2x + dy * e2y + dz * e2z

        A = 0.5 * ((xl[2, 0] - xl[0, 0]) * (xl[3, 1] - xl[1, 1])
                   + (xl[1, 0] - xl[3, 0]) * (xl[2, 1] - xl[0, 1]))
        twoA = 2.0 * A
        if twoA < EM20:
            twoA = EM20
        inv2A = 1.0 / twoA
        B1[e, 0] = (xl[1, 1] - xl[3, 1]) * inv2A
        B1[e, 1] = (xl[2, 1] - xl[0, 1]) * inv2A
        B1[e, 2] = (xl[3, 1] - xl[1, 1]) * inv2A
        B1[e, 3] = (xl[0, 1] - xl[2, 1]) * inv2A
        B2[e, 0] = (xl[3, 0] - xl[1, 0]) * inv2A
        B2[e, 1] = (xl[0, 0] - xl[2, 0]) * inv2A
        B2[e, 2] = (xl[1, 0] - xl[3, 0]) * inv2A
        B2[e, 3] = (xl[2, 0] - xl[0, 0]) * inv2A
        area[e] = A if A > EM20 else EM20

        # lc = A / longest side (local 2D)
        lmax = 0.0
        for i in range(4):
            j = i + 1 if i < 3 else 0
            dx = xl[j, 0] - xl[i, 0]
            dy = xl[j, 1] - xl[i, 1]
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
        hx = xl[0, 0] - xl[1, 0] + xl[2, 0] - xl[3, 0]
        hy = xl[0, 1] - xl[1, 1] + xl[2, 1] - xl[3, 1]
        gam[e, 0] = 1.0 - hx * B1[e, 0] - hy * B2[e, 0]
        gam[e, 1] = -1.0 - hx * B1[e, 1] - hy * B2[e, 1]
        gam[e, 2] = 1.0 - hx * B1[e, 2] - hy * B2[e, 2]
        gam[e, 3] = -1.0 - hx * B1[e, 3] - hy * B2[e, 3]
        s = 0.0
        for i in range(4):
            s += B1[e, i] * B1[e, i] + B2[e, i] * B2[e, i]
        bb[e] = s
    return E, area, lc, B1, B2, bb, gam, V, dm, kap, gs


@njit(cache=True)
def shell_post(E, area, B1, B2, gam, V, Nres, Mres, qres, Q,
               k_m, k_w, k_r, dt):
    """Mirror of shell_bt4._post — resultant nodal forces, BLT84
    stiffness hourglass (Q updated in place), back-transform to global
    axes. Returns (fg, mg, dehg)."""
    n = area.shape[0]
    fg = np.empty((n, 4, 3))
    mg = np.empty((n, 4, 3))
    dehg = np.empty(n)
    kv = np.empty(5)
    qd = np.empty(5)

    for e in range(n):
        A = area[e]
        # modal velocities of the 5 hourglass modes
        for k in range(5):
            s = 0.0
            for i in range(4):
                s += gam[e, i] * V[e, i, k]
            qd[k] = s
        kv[0] = k_m[e]; kv[1] = k_m[e]; kv[2] = k_w[e]
        kv[3] = k_r[e]; kv[4] = k_r[e]
        de = 0.0
        for k in range(5):
            q_old = Q[e, k]
            Q[e, k] = q_old + kv[k] * qd[k] * dt
            de += 0.5 * (q_old + Q[e, k]) * qd[k] * dt
        dehg[e] = de

        for i in range(4):
            b1 = B1[e, i]; b2 = B2[e, i]; g = gam[e, i]
            # local total force = -(internal) - Q*gamma (hourglass)
            flx = -A * (b1 * Nres[e, 0] + b2 * Nres[e, 2]) - g * Q[e, 0]
            fly = -A * (b2 * Nres[e, 1] + b1 * Nres[e, 2]) - g * Q[e, 1]
            flz = -A * (b1 * qres[e, 0] + b2 * qres[e, 1]) - g * Q[e, 2]
            mlx = -A * (-b2 * Mres[e, 1] - b1 * Mres[e, 2]
                        - 0.25 * qres[e, 1]) - g * Q[e, 3]
            mly = -A * (b1 * Mres[e, 0] + b2 * Mres[e, 2]
                        + 0.25 * qres[e, 0]) - g * Q[e, 4]
            # back to global axes: fg[b] = sum_a fl[a] E[b,a]  (mlz = 0)
            fg[e, i, 0] = flx * E[e, 0, 0] + fly * E[e, 0, 1] \
                + flz * E[e, 0, 2]
            fg[e, i, 1] = flx * E[e, 1, 0] + fly * E[e, 1, 1] \
                + flz * E[e, 1, 2]
            fg[e, i, 2] = flx * E[e, 2, 0] + fly * E[e, 2, 1] \
                + flz * E[e, 2, 2]
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
