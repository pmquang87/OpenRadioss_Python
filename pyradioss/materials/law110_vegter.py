"""/MAT/LAW110 — /MAT/VEGTER (/MAT/PLAS_VEGTER) Vegter Anisotropic Yield Model.

Plane-stress shell elastoplastic material law based on the Corus-Vegter 2006/2017
Bezier spline yield surface formulation for sheet metal forming.

Fortran source references:
  - Starter reader & initialization:
    ``starter/source/materials/mat/mat110/hm_read_mat110.F``
    ``hm_cfg_files/config/CFG/radioss2021/MAT/matl110_vegter.cfg``
  - Engine plane stress shell kernels:
    ``engine/source/materials/mat/mat110/sigeps110c.F``
    ``engine/source/materials/mat/mat110/sigeps110c_newton.F``
    ``engine/source/materials/mat/mat110/sigeps110c_nice.F``
    ``engine/source/materials/mat/mat110/sigeps110c_lite_newton.F``
    ``engine/source/materials/mat/mat110/sigeps110c_lite_nice.F``
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Chebyshev polynomial coefficient matrix COS2 (10x10) from hm_read_mat110.F / sigeps110c_newton.F
# Column I (0-based) represents cos(2*I*theta) expanded in powers of cos(2*theta).
COS2_DATA = np.array([
    [1.0,  0.0, -1.0,  0.0,   1.0,   0.0,  -1.0,   0.0,    1.0,    0.0],
    [0.0,  1.0,  0.0, -3.0,   0.0,   5.0,   0.0,  -7.0,    0.0,    9.0],
    [0.0,  0.0,  2.0,  0.0,  -8.0,   0.0,  18.0,   0.0,  -32.0,    0.0],
    [0.0,  0.0,  0.0,  4.0,   0.0, -20.0,   0.0,  56.0,    0.0, -120.0],
    [0.0,  0.0,  0.0,  0.0,   8.0,   0.0, -48.0,   0.0,  160.0,    0.0],
    [0.0,  0.0,  0.0,  0.0,   0.0,  16.0,   0.0, -112.0,   0.0,  432.0],
    [0.0,  0.0,  0.0,  0.0,   0.0,   0.0,  32.0,   0.0, -256.0,    0.0],
    [0.0,  0.0,  0.0,  0.0,   0.0,   0.0,   0.0,  64.0,    0.0, -576.0],
    [0.0,  0.0,  0.0,  0.0,   0.0,   0.0,   0.0,   0.0,  128.0,    0.0],
    [0.0,  0.0,  0.0,  0.0,   0.0,   0.0,   0.0,   0.0,    0.0,  256.0],
], dtype=np.float64)

EM20 = 1.0e-20
EM10 = 1.0e-10
EM08 = 1.0e-8
TOL = 1.0e-6


def _eval_curve_1d(curve: Any, x: float) -> Tuple[float, float]:
    """Evaluate a 1D curve or tabular function at point x, returning (val, slope)."""
    if curve is None:
        return 0.0, 0.0
    if hasattr(curve, "evaluate_with_derivative"):
        return curve.evaluate_with_derivative(x)
    if hasattr(curve, "evaluate"):
        v = float(curve.evaluate(x))
        eps = 1.0e-6 * max(1.0, abs(x))
        v2 = float(curve.evaluate(x + eps))
        return v, (v2 - v) / eps
    if callable(curve):
        v = float(curve(x))
        eps = 1.0e-6 * max(1.0, abs(x))
        v2 = float(curve(x + eps))
        return v, (v2 - v) / eps
    if isinstance(curve, (tuple, list)) and len(curve) == 2:
        try:
            x_arr = np.asarray(curve[0], dtype=np.float64)
            y_arr = np.asarray(curve[1], dtype=np.float64)
            if len(x_arr) > 1:
                v = float(np.interp(x, x_arr, y_arr))
                idx = np.searchsorted(x_arr, x)
                idx = max(1, min(idx, len(x_arr) - 1))
                dx = x_arr[idx] - x_arr[idx - 1]
                slope = float((y_arr[idx] - y_arr[idx - 1]) / dx) if abs(dx) > 1.0e-14 else 0.0
                return v, slope
        except Exception:
            pass
    return 0.0, 0.0


def _eval_curve_or_table_yield(
    mat: Any,
    epsp: float,
    rate: float,
    temp: float,
) -> Tuple[float, float]:
    """Evaluate tabulated yield stress and hardening modulus from TAB_YLD and TAB_TEMP."""
    tab_yld = getattr(mat, "curve_yld", None)
    if tab_yld is None and hasattr(mat, "params") and isinstance(mat.params, dict):
        tab_yld = mat.params.get("curve_yld")

    xscale = getattr(mat, "xscale", 1.0) or 1.0
    yscale = getattr(mat, "yscale", 1.0) or 1.0

    sig_y, h = 0.0, 0.0
    if tab_yld is not None:
        if hasattr(tab_yld, "evaluate_2d"):
            sig_y, h = tab_yld.evaluate_2d(epsp, rate * xscale)
        elif callable(tab_yld):
            try:
                res = tab_yld(epsp, rate * xscale)
                if isinstance(res, tuple):
                    sig_y, h = res[0], res[1]
                else:
                    sig_y = float(res)
            except TypeError:
                sig_y, h = _eval_curve_1d(tab_yld, epsp)
        else:
            sig_y, h = _eval_curve_1d(tab_yld, epsp)

        sig_y *= yscale
        h *= yscale

    # Temperature scaling from TAB_TEMP
    tab_temp = getattr(mat, "curve_temp", None)
    if tab_temp is not None:
        tini = getattr(mat, "tini", 293.0) or 293.0
        v_ref, _ = _eval_curve_1d(tab_temp, tini)
        v_cur, _ = _eval_curve_1d(tab_temp, temp)
        if abs(v_ref) > 1.0e-12:
            tfac = v_cur / v_ref
            sig_y *= tfac
            h *= tfac

    return sig_y, h


class VegterModelParams:
    """Preprocessed Vegter yield locus constants and interpolation factors."""

    def __init__(self, mat: Any = None) -> None:
        if mat is None:
            class _DefaultMat:
                pass
            mat = _DefaultMat()
        mat_src = getattr(mat, "m110", mat)
        pdict = getattr(mat, "params", {}) or {}
        if not isinstance(pdict, dict):
            pdict = {}

        self.mat = mat_src
        self.icrit = int(getattr(mat_src, "icrit", pdict.get("MAT_Icrit", getattr(mat, "icrit", 1))) or 1)
        self.ires = int(getattr(mat_src, "ires", pdict.get("MAT_Ires", getattr(mat, "ires", 2))) or 2)
        self.young = float(getattr(mat_src, "young", getattr(mat_src, "e", getattr(mat, "E", pdict.get("MAT_E", pdict.get("E", 0.0))))))
        self.nu = float(getattr(mat_src, "nu", getattr(mat, "nu", pdict.get("MAT_NU", pdict.get("nu", 0.0)))))
        self.rho0 = float(getattr(mat_src, "rho0", getattr(mat_src, "rho_i", getattr(mat_src, "rho", getattr(mat, "rho0", pdict.get("MAT_RHO", 0.0))))))

        # Hardening parameters
        self.sigma_r = float(getattr(mat_src, "sigma_r", getattr(mat_src, "sig0", pdict.get("SIGMA_r", 0.0))))
        self.dsigm = float(getattr(mat_src, "dsigm", getattr(mat_src, "b_swift", pdict.get("MAT_DSIGM", 0.0))))
        self.beta = float(getattr(mat_src, "beta", pdict.get("MAT_BETA", 0.0)))
        self.omega = float(getattr(mat_src, "omega", pdict.get("Omega", 0.0)))
        self.hard_n = float(getattr(mat_src, "hard_n", getattr(mat_src, "n_swift", getattr(mat_src, "n", pdict.get("MAT_HARD", 0.0)))))
        if self.beta == 0.0 and self.omega == 0.0 and self.dsigm > 0.0:
            self.beta = 1.0

        # Viscoplastic parameters (Bergstrom-van Liempt)
        self.eps0 = float(getattr(mat_src, "eps0", getattr(mat_src, "eps_0", pdict.get("Epsilon_0", 0.0))))
        self.sigs = float(getattr(mat_src, "sigs", pdict.get("MAT_SIGS", 0.0)))
        self.dg0 = float(getattr(mat_src, "dg0", pdict.get("MAT_DG0", 0.0)))
        self.deps0 = float(getattr(mat_src, "deps0", pdict.get("MAT_Deps0", 0.0)))
        self.m = float(getattr(mat_src, "m", pdict.get("MAT_StrainRate_m", 0.0)))
        self.tini = float(getattr(mat_src, "tini", pdict.get("T_Initial", 293.0)) or 293.0)
        self.chard = float(getattr(mat_src, "chard", pdict.get("MAT_CHARD", 0.0)))  # kinematic factor FISOKIN
        self.fcut = float(getattr(mat_src, "fcut", pdict.get("Fcut", 1.0e20)) or 1.0e20)
        self.vp = int(getattr(mat_src, "vp", pdict.get("Vflag", 2)) or 2)

        # Tabulated options
        self.tab_yld = int(getattr(mat_src, "tab_yld", pdict.get("MAT_TAB_YLD", 0)) or 0)
        self.tab_temp = int(getattr(mat_src, "tab_temp", pdict.get("MAT_TAB_TEMP", 0)) or 0)

        # Elastic plane-stress moduli
        self.a11 = self.young / max(1.0e-12, (1.0 - self.nu * self.nu))
        self.a12 = self.a11 * self.nu
        self.g = self.young / max(1.0e-12, 2.0 * (1.0 + self.nu))
        self.g_shear = 5.0 / 6.0 * self.g  # Reissner-Mindlin transverse shear

        # Biaxial values
        self.fbi = float(getattr(mat_src, "fbi", pdict.get("MAT_fBI", 1.0)) or 1.0)
        self.rhobi = float(getattr(mat_src, "rhobi", pdict.get("MAT_rhoBI", 1.0)) or 1.0)

        # Angle tables setup
        self._setup_yield_locus()

    def _setup_yield_locus(self) -> None:
        """Construct the reference and hinge points and solve for Chebyshev coefficients Q."""
        angles_data = getattr(self.mat, "angles_data", None)
        if not angles_data and hasattr(self, "mat") and hasattr(self.mat, "m110"):
            angles_data = getattr(self.mat.m110, "angles_data", None)
        angles_data = angles_data or []
        nangle = len(angles_data)

        if self.icrit == 3:
            nangle = 3
            rm = [float(getattr(self.mat, "rm_0", 1.0)), float(getattr(self.mat, "rm_45", 1.0)), float(getattr(self.mat, "rm_90", 1.0))]
            ag = [float(getattr(self.mat, "ag_0", 0.1)), float(getattr(self.mat, "ag_45", 0.1)), float(getattr(self.mat, "ag_90", 0.1))]
            rlank = [float(getattr(self.mat, "r_0", 1.0)), float(getattr(self.mat, "r_45", 1.0)), float(getattr(self.mat, "r_90", 1.0))]
            for i in range(3):
                if rlank[i] == 0.0:
                    rlank[i] = 1.0

            epsag = [math.log(1.0 + max(1.0e-8, ag[i] / 100.0)) for i in range(3)]
            sigag = [rm[i] * (1.0 + ag[i] / 100.0) for i in range(3)]
            cag = [sigag[i] / max(1.0e-12, epsag[i] ** epsag[i]) for i in range(3)]
            w0 = (cag[0] / (epsag[0] + 1.0)) * (epsag[0] ** (epsag[0] + 1.0))
            epseq = [((epsag[i] + 1.0) * (w0 / cag[i])) ** (1.0 / (epsag[i] + 1.0)) for i in range(3)]
            fun = np.zeros((3, 2), dtype=np.float64)
            for i in range(3):
                fun[i, 0] = epsag[i] / max(1.0e-12, epseq[i])

            fun_av = (fun[0, 0] + 2.0 * fun[1, 0] + fun[2, 0]) / 4.0
            rlank_av = (rlank[0] + 2.0 * rlank[1] + rlank[2]) / 4.0
            fdirac = math.exp(3.4 * (rlank_av - 1.22))
            self.fbi = fun_av * (0.97 / (1.0 + fdirac) + 1.14 / (1.0 + 1.0 / fdirac))
            if self.rhobi == 0.0:
                self.rhobi = rlank[0] / rlank[2]

            fps = np.zeros((3, 2), dtype=np.float64)
            for j in range(3):
                fdir = math.exp(1.2 * (rlank[j] - 0.5))
                fps[j, 0] = fun[j, 0] * (0.827 / (1.0 + fdir) + 1.315 / (1.0 + 1.0 / fdir))

            fsh = np.zeros((3, 2), dtype=np.float64)
            fun_13 = (fun[0, 0] + fun[2, 0]) / 2.0
            rlank_13 = (rlank[0] + rlank[2]) / 2.0
            fdir_13 = math.exp(1.6 * rlank_13)
            fsh[0, 0] = fun_13 * (0.757 / (1.0 + fdir_13) + 0.525 / (1.0 + 1.0 / fdir_13))
            fsh[2, 0] = fsh[0, 0]
            fdir_2 = math.exp(1.6 * rlank[1])
            fsh[1, 0] = fun[1, 0] * (0.757 / (1.0 + fdir_2) + 0.525 / (1.0 + 1.0 / fdir_2))

            # Plane strain second coordinate via Newton iteration for weight
            for j in range(3):
                theta_rad = (j * 45.0) * (math.pi / 180.0)
                cos2t = math.cos(2.0 * theta_rad)
                num = (self.rhobi + 1.0) + (self.rhobi - 1.0) * cos2t
                den = (self.rhobi + 1.0) - (self.rhobi - 1.0) * cos2t
                rhobi_theta = num / max(1.0e-12, den)
                rhoone_theta = -rlank[j] / (rlank[j] + 1.0)

                fh2 = (self.fbi + rhobi_theta * self.fbi - fun[j, 0]) / max(1.0e-12, (rhobi_theta - rhoone_theta))
                fh1 = self.fbi - rhobi_theta * (fh2 - self.fbi)

                mu_val = 0.5
                weight = 0.0
                for _ in range(10):
                    u = fun[j, 0] * ((1.0 - mu_val) ** 2) + 2.0 * fh1 * mu_val * weight * (1.0 - mu_val) + self.fbi * (mu_val ** 2)
                    v = (1.0 - mu_val) ** 2 + 2.0 * weight * mu_val * (1.0 - mu_val) + mu_val ** 2
                    fps1 = u / max(1.0e-12, v)
                    phi = fps1 - fps[j, 0]
                    uprim = 2.0 * mu_val * (1.0 - mu_val) * fh1
                    vprim = 2.0 * mu_val * (1.0 - mu_val)
                    dfps1_dw = (uprim * v - u * vprim) / max(v ** 2, EM20)
                    if abs(dfps1_dw) > 1.0e-14:
                        weight -= phi / dfps1_dw

                fps[j, 1] = (2.0 * fh2 * mu_val * weight * (1.0 - mu_val) + self.fbi * (mu_val ** 2)) / max(
                    1.0e-12, (1.0 - mu_val) ** 2 + 2.0 * weight * mu_val * (1.0 - mu_val) + mu_val ** 2
                )
        elif self.icrit in (1, 2):
            if nangle == 0:
                nangle = 1
                angles_data = [[1.0, 1.0, 1.0, 0.5, 0.5]]
            fun = np.zeros((nangle, 2), dtype=np.float64)
            fsh = np.zeros((nangle, 2), dtype=np.float64)
            fps = np.zeros((nangle, 2), dtype=np.float64)
            rlank = np.ones(nangle, dtype=np.float64)
            alps = np.full(nangle, 0.5, dtype=np.float64)

            for j, row in enumerate(angles_data):
                fun[j, 0] = float(row[0]) if len(row) > 0 else 1.0
                rlank[j] = float(row[1]) if len(row) > 1 and float(row[1]) > 0.0 else 1.0
                fps[j, 0] = float(row[2]) if len(row) > 2 else 1.0
                if self.icrit == 1:
                    fps[j, 1] = float(row[3]) if len(row) > 3 else 0.0
                    fsh[j, 0] = float(row[4]) if len(row) > 4 else 0.5
                else:  # icrit == 2
                    alps[j] = float(row[3]) if len(row) > 3 and float(row[3]) > 0.0 else 0.5
                    fsh[j, 0] = float(row[4]) if len(row) > 4 else 0.5
        else:  # icrit == 4 (Vegter-Lite)
            if nangle == 0:
                nangle = 1
                angles_data = [[1.0, 1.0, 0.5, 0.5]]
            fun = np.zeros((nangle, 2), dtype=np.float64)
            rlank = np.ones(nangle, dtype=np.float64)
            wps = np.full(nangle, 0.5, dtype=np.float64)
            wsh = np.full(nangle, 0.5, dtype=np.float64)
            for j, row in enumerate(angles_data):
                fun[j, 0] = float(row[0]) if len(row) > 0 else 1.0
                rlank[j] = float(row[1]) if len(row) > 1 and float(row[1]) > 0.0 else 1.0
                wps[j] = float(row[2]) if len(row) > 2 else 0.5
                wsh[j] = float(row[3]) if len(row) > 3 else 0.5

        self.nangle = nangle
        self.rlank = rlank

        # Compute theta angles in radians
        theta_rad = np.zeros(nangle, dtype=np.float64)
        for j in range(nangle):
            if nangle > 1:
                theta_deg = j * (90.0 / (nangle - 1))
                theta_rad[j] = theta_deg * (math.pi / 180.0)
            else:
                theta_rad[j] = 0.0
        self.theta_rad = theta_rad

        # Compute hinge points
        hips = np.zeros((nangle, 2), dtype=np.float64)
        hiun = np.zeros((nangle, 2), dtype=np.float64)
        hish = np.zeros((nangle, 2), dtype=np.float64)

        for j in range(nangle):
            cos2t = math.cos(2.0 * theta_rad[j])
            num = (self.rhobi + 1.0) + (self.rhobi - 1.0) * cos2t
            den = (self.rhobi + 1.0) - (self.rhobi - 1.0) * cos2t
            rhobi_theta = num / max(1.0e-12, den)

            if self.icrit in (1, 2, 3):
                # HIPS: between biaxial and plane strain
                a1 = fps[j, 0]
                a2 = fps[j, 1]
                n1 = 1.0
                n2 = 0.0
                c1 = self.fbi
                c2 = self.fbi
                m1 = 1.0
                m2 = rhobi_theta
                det = n1 * m2 - m1 * n2
                hips[j, 0] = (m2 * (n1 * a1 + n2 * a2) - n2 * (m1 * c1 + m2 * c2)) / max(1.0e-12, det)
                hips[j, 1] = (n1 * (m1 * c1 + m2 * c2) - m1 * (n1 * a1 + n2 * a2)) / max(1.0e-12, det)

                # HIUN: between plane strain and uniaxial
                a1 = fun[j, 0]
                fun[j, 1] = 0.0
                a2 = 0.0
                n1 = 1.0
                n2 = -rlank[j] / (rlank[j] + 1.0)
                c1 = fps[j, 0]
                c2 = fps[j, 1]
                m1 = 1.0
                m2 = 0.0
                det = n1 * m2 - m1 * n2
                hiun[j, 0] = (m2 * (n1 * a1 + n2 * a2) - n2 * (m1 * c1 + m2 * c2)) / max(1.0e-12, det)
                hiun[j, 1] = (n1 * (m1 * c1 + m2 * c2) - m1 * (n1 * a1 + n2 * a2)) / max(1.0e-12, det)

                if self.icrit == 1 and fps[j, 1] == 0.0:
                    fps[j, 1] = hiun[j, 1] + 0.5 * (hips[j, 1] - hiun[j, 1])
                elif self.icrit == 2:
                    fps[j, 1] = hiun[j, 1] + alps[j] * (hips[j, 1] - hiun[j, 1])

                # HISH: between uniaxial and shear
                a1 = fsh[j, 0]
                fsh[j, 1] = -fsh[nangle - 1 - j, 0]
                a2 = fsh[j, 1]
                n1 = 1.0
                n2 = -1.0
                c1 = fun[j, 0]
                c2 = 0.0
                m1 = 1.0
                m2 = -rlank[j] / (rlank[j] + 1.0)
                det = n1 * m2 - m1 * n2
                hish[j, 0] = (m2 * (n1 * a1 + n2 * a2) - n2 * (m1 * c1 + m2 * c2)) / max(1.0e-12, det)
                hish[j, 1] = (n1 * (m1 * c1 + m2 * c2) - m1 * (n1 * a1 + n2 * a2)) / max(1.0e-12, det)
            else:  # icrit == 4
                # HIPS: between biaxial and uniaxial
                a1 = fun[j, 0]
                fun[j, 1] = 0.0
                a2 = 0.0
                n1 = 1.0
                n2 = -rlank[j] / (rlank[j] + 1.0)
                c1 = self.fbi
                c2 = self.fbi
                m1 = 1.0
                m2 = rhobi_theta
                det = n1 * m2 - m1 * n2
                hips[j, 0] = (m2 * (n1 * a1 + n2 * a2) - n2 * (m1 * c1 + m2 * c2)) / max(1.0e-12, det)
                hips[j, 1] = (n1 * (m1 * c1 + m2 * c2) - m1 * (n1 * a1 + n2 * a2)) / max(1.0e-12, det)

                # HISH: between uniaxial tension and uniaxial compression
                a1 = fun[j, 1]
                a2 = -fun[j, 0]
                n1 = rlank[j] / (rlank[j] + 1.0)
                n2 = -1.0
                c1 = fun[j, 0]
                c2 = fun[j, 1]
                m1 = 1.0
                m2 = -rlank[j] / (rlank[j] + 1.0)
                det = n1 * m2 - m1 * n2
                hish[j, 0] = (m2 * (n1 * a1 + n2 * a2) - n2 * (m1 * c1 + m2 * c2)) / max(1.0e-12, det)
                hish[j, 1] = (n1 * (m1 * c1 + m2 * c2) - m1 * (n1 * a1 + n2 * a2)) / max(1.0e-12, det)

        # Build A-matrix: AMAT(J, I) = sum_{K=1}^I COS2(K-1, I-1) * cos(2*theta_J)^(K-1)
        amat = np.zeros((nangle, nangle), dtype=np.float64)
        for j in range(nangle):
            cos2t = math.cos(2.0 * theta_rad[j])
            for i in range(nangle):
                for k in range(i + 1):
                    amat[j, i] += COS2_DATA[k, i] * (cos2t ** k)

        # Solve for interpolation coefficients Q
        if self.icrit in (1, 2, 3):
            bvec = np.zeros((nangle, 12), dtype=np.float64)
            bvec[:, 0:2] = fsh[:, 0:2]
            bvec[:, 2:4] = fun[:, 0:2]
            bvec[:, 4:6] = fps[:, 0:2]
            bvec[:, 6:8] = hish[:, 0:2]
            bvec[:, 8:10] = hiun[:, 0:2]
            bvec[:, 10:12] = hips[:, 0:2]

            try:
                q_sol = np.linalg.solve(amat, bvec)
            except np.linalg.LinAlgError:
                q_sol = np.linalg.pinv(amat) @ bvec

            self.q_fsh = q_sol[:, 0:2]
            self.q_fun = q_sol[:, 2]  # fun[j, 1] is zero
            self.q_fps = q_sol[:, 4:6]
            self.q_hish = q_sol[:, 6:8]
            self.q_hiun = q_sol[:, 8:10]
            self.q_hips = q_sol[:, 10:12]
        else:  # icrit == 4
            bvec = np.zeros((nangle, 8), dtype=np.float64)
            bvec[:, 0:2] = fun[:, 0:2]
            bvec[:, 2:4] = hish[:, 0:2]
            bvec[:, 4:6] = hips[:, 0:2]
            bvec[:, 6] = wsh
            bvec[:, 7] = wps

            try:
                q_sol = np.linalg.solve(amat, bvec)
            except np.linalg.LinAlgError:
                q_sol = np.linalg.pinv(amat) @ bvec

            self.q_fun = q_sol[:, 0]
            self.q_hish = q_sol[:, 2:4]
            self.q_hips = q_sol[:, 4:6]
            self.q_wsh = q_sol[:, 6]
            self.q_wps = q_sol[:, 7]


def get_vegter_params(mat: Any) -> VegterModelParams:
    """Retrieve or compute cached Vegter parameters for material."""
    cached = getattr(mat, "_vegter_params", None)
    if cached is None:
        cached = VegterModelParams(mat)
        try:
            mat._vegter_params = cached
        except Exception:
            pass
    return cached


def eval_vegter_equivalent_stress_and_normal(
    sigxx: float,
    sigyy: float,
    sigxy: float,
    params: VegterModelParams,
) -> Tuple[float, np.ndarray]:
    """Compute Vegter equivalent stress sigma_vg and normal gradient dPhi/dsigma.

    Returns:
        (sigma_vg, normal_vector [N_xx, N_yy, N_xy])
    """
    # Mohr's circle decomposition in sheet plane
    center = 0.5 * (sigxx + sigyy)
    rad = math.sqrt(max(0.0, (0.5 * (sigxx - sigyy)) ** 2 + sigxy ** 2))
    sig1 = center + rad
    sig2 = center - rad

    if rad > EM20:
        cos2t = 0.5 * (sigxx - sigyy) / rad
        sin2t = sigxy / rad
    else:
        cos2t = 1.0
        sin2t = 0.0

    nangle = params.nangle

    def _eval_poly(q_mat: np.ndarray, c2: float) -> np.ndarray:
        res = np.zeros(q_mat.shape[1] if q_mat.ndim > 1 else 1, dtype=np.float64)
        for i in range(nangle):
            c_factor = 0.0
            for k in range(i + 1):
                c_factor += COS2_DATA[k, i] * (c2 ** k)
            if q_mat.ndim > 1:
                res += q_mat[i] * c_factor
            else:
                res += q_mat[i] * c_factor
        return res

    def _eval_poly_dcos2(q_mat: np.ndarray, c2: float) -> np.ndarray:
        res = np.zeros(q_mat.shape[1] if q_mat.ndim > 1 else 1, dtype=np.float64)
        if nangle <= 1:
            return res
        for i in range(1, nangle):
            for k in range(1, i + 1):
                deriv = k * COS2_DATA[k, i] * (c2 ** (k - 1))
                if q_mat.ndim > 1:
                    res += q_mat[i] * deriv
                else:
                    res += q_mat[i] * deriv
        return res

    if params.icrit in (1, 2, 3):
        fsh_t = _eval_poly(params.q_fsh, cos2t)
        fps_t = _eval_poly(params.q_fps, cos2t)

        sign_changed = False
        if sig1 < 0.0 or (sig2 < 0.0 and sig2 * fsh_t[0] < sig1 * fsh_t[1]):
            cos2t = -cos2t
            sin2t = -sin2t
            sig1, sig2 = -sig2, -sig1
            sign_changed = True
            fsh_t = _eval_poly(params.q_fsh, cos2t)
            fps_t = _eval_poly(params.q_fps, cos2t)

        if sig2 < 0.0:  # Region 1: shear to uniaxial
            fun_t1 = _eval_poly(params.q_fun, cos2t)[0]
            fun_t = np.array([fun_t1, 0.0], dtype=np.float64)
            hish_t = _eval_poly(params.q_hish, cos2t)
            a = fsh_t
            b = hish_t
            c = fun_t
            da_dcos2 = _eval_poly_dcos2(params.q_fsh, cos2t)
            db_dcos2 = _eval_poly_dcos2(params.q_hish, cos2t)
            dc_dcos2 = np.array([_eval_poly_dcos2(params.q_fun, cos2t)[0], 0.0], dtype=np.float64)
        elif sig2 / max(EM20, sig1) < fps_t[1] / max(EM20, fps_t[0]):  # Region 2: uniaxial to plane strain
            fun_t1 = _eval_poly(params.q_fun, cos2t)[0]
            fun_t = np.array([fun_t1, 0.0], dtype=np.float64)
            hiun_t = _eval_poly(params.q_hiun, cos2t)
            a = fun_t
            b = hiun_t
            c = fps_t
            da_dcos2 = np.array([_eval_poly_dcos2(params.q_fun, cos2t)[0], 0.0], dtype=np.float64)
            db_dcos2 = _eval_poly_dcos2(params.q_hiun, cos2t)
            dc_dcos2 = _eval_poly_dcos2(params.q_fps, cos2t)
        else:  # Region 3: plane strain to biaxial
            hips_t = _eval_poly(params.q_hips, cos2t)
            a = fps_t
            b = hips_t
            c = np.array([params.fbi, params.fbi], dtype=np.float64)
            da_dcos2 = _eval_poly_dcos2(params.q_fps, cos2t)
            db_dcos2 = _eval_poly_dcos2(params.q_hips, cos2t)
            dc_dcos2 = np.zeros(2, dtype=np.float64)

        if sig1 < EM20:
            return 0.0, np.zeros(3, dtype=np.float64)

        sig_ratio = sig2 / sig1
        var_a = (c[1] + a[1] - 2.0 * b[1]) - sig_ratio * (c[0] + a[0] - 2.0 * b[0])
        var_b = 2.0 * ((b[1] - a[1]) - sig_ratio * (b[0] - a[0]))
        var_c = a[1] - sig_ratio * a[0]

        if abs(var_a) < EM08:
            mu = -var_c / max(1.0e-12, var_b)
        else:
            disc = max(0.0, var_b * var_b - 4.0 * var_a * var_c)
            mu = (-var_b + math.sqrt(disc)) / (2.0 * var_a)
        mu = max(0.0, min(1.0, mu))

        f1 = ((1.0 - mu) ** 2) * a[0] + 2.0 * mu * (1.0 - mu) * b[0] + (mu ** 2) * c[0]
        f2 = ((1.0 - mu) ** 2) * a[1] + 2.0 * mu * (1.0 - mu) * b[1] + (mu ** 2) * c[1]
        sigma_vg = sig1 / max(1.0e-12, f1)

        # Derivatives
        df1_dmu = 2.0 * (b[0] - a[0]) + 2.0 * mu * (a[0] + c[0] - 2.0 * b[0])
        df2_dmu = 2.0 * (b[1] - a[1]) + 2.0 * mu * (a[1] + c[1] - 2.0 * b[1])
        df1_dcos2 = da_dcos2[0] + 2.0 * mu * (db_dcos2[0] - da_dcos2[0]) + (mu ** 2) * (da_dcos2[0] + dc_dcos2[0] - 2.0 * db_dcos2[0])
        df2_dcos2 = da_dcos2[1] + 2.0 * mu * (db_dcos2[1] - da_dcos2[1]) + (mu ** 2) * (da_dcos2[1] + dc_dcos2[1] - 2.0 * db_dcos2[1])

        det_mu = f1 * df2_dmu - f2 * df1_dmu
        if abs(det_mu) > 1.0e-14:
            dphi_dsig1 = df2_dmu / det_mu
            dphi_dsig2 = -df1_dmu / det_mu
            if abs(sig1 - sig2) > TOL:
                dphi_dcos2 = (sigma_vg / (sig1 - sig2)) * (df2_dcos2 * df1_dmu - df1_dcos2 * df2_dmu) / det_mu
            else:
                dphi_dcos2 = 0.0
        else:
            dphi_dsig1 = 0.0
            dphi_dsig2 = 0.0
            dphi_dcos2 = 0.0

    else:  # icrit == 4 (Vegter-Lite)
        hish_t = _eval_poly(params.q_hish, cos2t)
        sign_changed = False
        if sig1 < 0.0 or (sig2 < 0.0 and sig2 * hish_t[0] < sig1 * hish_t[1]):
            cos2t = -cos2t
            sin2t = -sin2t
            sig1, sig2 = -sig2, -sig1
            sign_changed = True

        fun_t1 = _eval_poly(params.q_fun, cos2t)[0]
        fun_t = np.array([fun_t1, 0.0], dtype=np.float64)

        if sig2 < 0.0:  # Region 1: tension to compression
            hish_t = _eval_poly(params.q_hish, cos2t)
            weight = _eval_poly(params.q_wsh, cos2t)[0]
            a = np.array([fun_t[1], -fun_t[0]], dtype=np.float64)
            b = hish_t
            c = fun_t
        else:  # Region 2: uniaxial to equibiaxial
            hips_t = _eval_poly(params.q_hips, cos2t)
            weight = _eval_poly(params.q_wps, cos2t)[0]
            a = fun_t
            b = hips_t
            c = np.array([params.fbi, params.fbi], dtype=np.float64)

        if sig1 < EM20:
            return 0.0, np.zeros(3, dtype=np.float64)

        sig_ratio = sig2 / sig1
        var_a = (c[1] + a[1] - 2.0 * weight * b[1]) - sig_ratio * (c[0] + a[0] - 2.0 * weight * b[0])
        var_b = 2.0 * ((weight * b[1] - a[1]) - sig_ratio * (weight * b[0] - a[0]))
        var_c = a[1] - sig_ratio * a[0]

        if abs(var_a) < EM08:
            mu = -var_c / max(1.0e-12, var_b)
        else:
            disc = max(0.0, var_b * var_b - 4.0 * var_a * var_c)
            mu = (-var_b + math.sqrt(disc)) / (2.0 * var_a)
        mu = max(0.0, min(1.0, mu))

        f1_denom = a[0] * ((1.0 - mu) ** 2) + 2.0 * b[0] * mu * weight * (1.0 - mu) + c[0] * (mu ** 2)
        f_num = (1.0 - mu) ** 2 + 2.0 * mu * weight * (1.0 - mu) + (mu ** 2)
        sigma_vg = sig1 * f_num / max(1.0e-12, f1_denom)

        # Approximate normal for lite
        dphi_dsig1 = 1.0 / max(1.0e-12, f1_denom)
        dphi_dsig2 = 0.0
        dphi_dcos2 = 0.0

    # Rotate derivatives to Cartesian components via Mohr's circle
    norm_xx = 0.5 * (1.0 + cos2t) * dphi_dsig1 + 0.5 * (1.0 - cos2t) * dphi_dsig2 + (sin2t ** 2) * dphi_dcos2
    norm_yy = 0.5 * (1.0 - cos2t) * dphi_dsig1 + 0.5 * (1.0 + cos2t) * dphi_dsig2 - (sin2t ** 2) * dphi_dcos2
    norm_xy = sin2t * dphi_dsig1 - sin2t * dphi_dsig2 - (2.0 * sin2t * cos2t) * dphi_dcos2

    if sign_changed:
        norm_xx = -norm_xx
        norm_yy = -norm_yy
        norm_xy = -norm_xy

    return sigma_vg, np.array([norm_xx, norm_yy, norm_xy], dtype=np.float64)


def eval_flow_stress_and_hardening(
    params: VegterModelParams,
    epsp: float,
    rate: float,
    temp: float,
) -> Tuple[float, float]:
    """Compute yield stress sigma_y and plastic hardening modulus H = dsigma_y/depsp."""
    if params.tab_yld > 0:
        sig_y, h = _eval_curve_or_table_yield(params.mat, epsp, rate, temp)
    else:
        # Continuous Swift / Voce hardening:
        # SIGHARD = SIGMA_r + DSIGM * (BETA * epsp + (1 - exp(-OMEGA * epsp))^n)
        p = max(0.0, epsp)
        exp_term = 1.0 - math.exp(-params.omega * p) if params.omega > 0.0 else 0.0
        pow_term = (exp_term ** params.hard_n) if (exp_term > 0.0 and params.hard_n > 0.0) else 0.0
        sighard = params.sigma_r + params.dsigm * (params.beta * p + pow_term)

        # Viscoplastic strain rate dependency (Bergstrom-van Liempt model)
        sigrate = 0.0
        if params.sigs > 0.0 and params.deps0 > 0.0 and params.dg0 > 0.0:
            kboltz = 8.6173303e-5  # eV/K
            rate_ratio = max(0.0, rate / params.deps0)
            arg = 1.0 + (kboltz * temp / params.dg0) * math.log(1.0 + rate_ratio)
            if arg > 0.0 and params.m != 0.0:
                sigrate = params.sigs * (arg ** params.m)

        sig_y = sighard + sigrate

        # Derivative H = d(SIGHARD)/dp
        h = params.dsigm * params.beta
        if p > 0.0 and params.omega > 0.0 and params.hard_n > 0.0 and exp_term > 0.0:
            h += params.dsigm * params.hard_n * (exp_term ** (params.hard_n - 1.0)) * (params.omega * math.exp(-params.omega * p))

    sig_y = max(EM10, sig_y)
    return sig_y, h


def _single_law110_shell_update(
    params: VegterModelParams,
    dxx: float,
    dyy: float,
    dxy: float,
    dyz: float,
    dzx: float,
    so_xx: float,
    so_yy: float,
    so_xy: float,
    so_yz: float,
    so_zx: float,
    pla: float,
    temp: float,
    thk: float,
    thkly: float,
    rate: float,
    dt: float,
    has_shear: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    # Elastic trial stresses
    sig_tr_xx = so_xx + params.a11 * dxx + params.a12 * dyy
    sig_tr_yy = so_yy + params.a11 * dyy + params.a12 * dxx
    sig_tr_xy = so_xy + params.g * dxy

    # Transverse shear (elastic)
    sig_new_yz = so_yz + params.g_shear * dyz
    sig_new_zx = so_zx + params.g_shear * dzx

    # Strain rate estimate
    if rate == 0.0 and dt > 1.0e-20:
        rate = math.sqrt(dxx * dxx + dyy * dyy + 0.5 * dxy * dxy) / dt

    # Initial flow stress
    sig_y, h = eval_flow_stress_and_hardening(params, pla, rate, temp)

    # Evaluate yield condition
    sig_vg_tr, norm_tr = eval_vegter_equivalent_stress_and_normal(sig_tr_xx, sig_tr_yy, sig_tr_xy, params)
    phi_tr = sig_vg_tr - sig_y

    sign_xx = sig_tr_xx
    sign_yy = sig_tr_yy
    sign_xy = sig_tr_xy
    dpla = 0.0
    dezz_pl = 0.0

    if phi_tr > 0.0:
        if params.ires == 1:
            # Nice explicit projection
            norm_xx, norm_yy, norm_xy = norm_tr[0], norm_tr[1], norm_tr[2]
            dsig2 = norm_xx * (params.a11 * norm_xx + params.a12 * norm_yy) + \
                    norm_yy * (params.a11 * norm_yy + params.a12 * norm_xx) + \
                    norm_xy * norm_xy * params.g
            denom = dsig2 + h
            dlam = max(0.0, phi_tr / max(1.0e-12, denom))

            dpxx = dlam * norm_xx
            dpyy = dlam * norm_yy
            dpxy = dlam * norm_xy

            sign_xx -= (params.a11 * dpxx + params.a12 * dpyy)
            sign_yy -= (params.a11 * dpyy + params.a12 * dpxx)
            sign_xy -= params.g * dpxy

            sig_dfdsig = sign_xx * norm_xx + sign_yy * norm_yy + sign_xy * norm_xy
            dpla = max(0.0, dlam * sig_dfdsig / max(1.0e-12, sig_y))
            pla += dpla
            dezz_pl = -(dpxx + dpyy)
            sig_y, h = eval_flow_stress_and_hardening(params, pla, rate, temp)
            sig_vg, _ = eval_vegter_equivalent_stress_and_normal(sign_xx, sign_yy, sign_xy, params)
        else:
            # Newton-Raphson semi-implicit cutting plane (NITER = 3)
            niter = 3
            sign_xx = sig_tr_xx
            sign_yy = sig_tr_yy
            sign_xy = sig_tr_xy

            for _ in range(niter):
                sig_vg, norm = eval_vegter_equivalent_stress_and_normal(sign_xx, sign_yy, sign_xy, params)
                phi = sig_vg - sig_y
                if phi <= 1.0e-7 * sig_y:
                    break

                norm_xx, norm_yy, norm_xy = norm[0], norm[1], norm[2]
                dfdsig2 = norm_xx * (params.a11 * norm_xx + params.a12 * norm_yy) + \
                          norm_yy * (params.a11 * norm_yy + params.a12 * norm_xx) + \
                          norm_xy * norm_xy * params.g

                sig_dfdsig = sign_xx * norm_xx + sign_yy * norm_yy + sign_xy * norm_xy
                dpla_dlam = sig_dfdsig / max(1.0e-12, sig_y)
                dphi_dlam = -dfdsig2 - h * dpla_dlam

                dlam = -phi / (dphi_dlam if abs(dphi_dlam) > 1.0e-12 else -1.0e-12)
                dlam = max(0.0, dlam)

                dpxx = dlam * norm_xx
                dpyy = dlam * norm_yy
                dpxy = dlam * norm_xy

                sign_xx -= (params.a11 * dpxx + params.a12 * dpyy)
                sign_yy -= (params.a11 * dpyy + params.a12 * dpxx)
                sign_xy -= params.g * dpxy

                step_dp = dlam * sig_dfdsig / max(1.0e-12, sig_y)
                dpla += step_dp
                pla += step_dp
                dezz_pl -= (dpxx + dpyy)

                sig_y, h = eval_flow_stress_and_hardening(params, pla, rate, temp)

            sig_vg, _ = eval_vegter_equivalent_stress_and_normal(sign_xx, sign_yy, sign_xy, params)
    else:
        sig_vg = sig_vg_tr

    # Elastic through-thickness strain
    deelzz = -params.nu * ((sign_xx - so_xx) + (sign_yy - so_yy)) / max(1.0e-12, params.young)
    dezz_total = deelzz + dezz_pl
    thk_new = thk * (1.0 + dezz_total * thkly)

    # Sound speed in sheet plane
    soundsp = math.sqrt(params.a11 / max(1.0e-12, params.rho0))

    # Hourglass stiffness tangent
    et = (h / (h + params.young)) if dpla > 0.0 else 1.0

    if has_shear:
        res_sig = np.array([sign_xx, sign_yy, sign_xy, sig_new_yz, sig_new_zx], dtype=np.float64)
    else:
        res_sig = np.array([sign_xx, sign_yy, sign_xy], dtype=np.float64)

    res_ex = {
        "pla": pla,
        "dpla": dpla,
        "sigy": sig_y,
        "seq": sig_vg,
        "thk": thk_new,
        "dezz": dezz_total,
        "soundsp": soundsp,
        "et": et,
        "temp": temp,
    }
    return res_sig, res_ex


def law110_shell_update(
    mat: Any,
    deps: np.ndarray,
    sigo: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
    dt: float = 0.0,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Plane-stress shell elasto-plastic stress update for /MAT/LAW110 (Vegter).

    Args:
        mat: MaterialLaw110 model entity or wrapper.
        deps: Strain increment tensor [deps_xx, deps_yy, deps_xy, (deps_yz, deps_zx)].
        sigo: Previous stress tensor [sig_xx, sig_yy, sig_xy, (sig_yz, sig_zx)].
        extra: State dictionary (pla, temp, thk, soundsp, etc.).
        dt: Current time increment.

    Returns:
        (sig_new, extra_out)
    """
    params = get_vegter_params(mat)
    ex = dict(extra) if extra is not None else {}

    deps_arr = np.asarray(deps, dtype=np.float64)
    sigo_arr = np.asarray(sigo, dtype=np.float64)
    is_1d = (sigo_arr.ndim == 1 and deps_arr.ndim == 1)

    if is_1d:
        pla = float(ex.get("pla", ex.get("epsp", 0.0)))
        temp = float(ex.get("temp", params.tini))
        thk = float(ex.get("thk", 1.0))
        thkly = float(ex.get("thkly", 1.0))
        rate = float(ex.get("rate", 0.0))

        dxx = float(deps_arr[0])
        dyy = float(deps_arr[1])
        dxy = float(deps_arr[2])
        dyz = float(deps_arr[3]) if len(deps_arr) > 3 else 0.0
        dzx = float(deps_arr[4]) if len(deps_arr) > 4 else 0.0

        so_xx = float(sigo_arr[0])
        so_yy = float(sigo_arr[1])
        so_xy = float(sigo_arr[2])
        so_yz = float(sigo_arr[3]) if len(sigo_arr) > 3 else 0.0
        so_zx = float(sigo_arr[4]) if len(sigo_arr) > 4 else 0.0

        has_shear = (len(sigo_arr) > 3 or len(deps_arr) > 3)
        res_sig, res_ex = _single_law110_shell_update(
            params, dxx, dyy, dxy, dyz, dzx, so_xx, so_yy, so_xy, so_yz, so_zx,
            pla, temp, thk, thkly, rate, dt, has_shear=has_shear
        )
        ex.update(res_ex)
        return res_sig, ex
    else:
        deps_2d = np.atleast_2d(deps_arr)
        sigo_2d = np.atleast_2d(sigo_arr)
        n = sigo_2d.shape[0]
        ncomp = sigo_2d.shape[1]
        has_shear = (ncomp > 3 or deps_2d.shape[1] > 3)

        def _get_arr(key: str, default_val: float) -> np.ndarray:
            val = ex.get(key)
            if val is None:
                return np.full(n, default_val, dtype=np.float64)
            arr = np.asarray(val, dtype=np.float64).reshape(-1)
            if len(arr) == n:
                return arr.copy()
            elif len(arr) == 1:
                return np.full(n, float(arr[0]), dtype=np.float64)
            return np.full(n, default_val, dtype=np.float64)

        pla_arr = _get_arr("pla", 0.0)
        temp_arr = _get_arr("temp", params.tini)
        thk_arr = _get_arr("thk", 1.0)
        thkly_arr = _get_arr("thkly", 1.0)
        rate_arr = _get_arr("rate", 0.0)

        res_sig_all = np.zeros((n, ncomp), dtype=np.float64)
        dpla_arr = np.zeros(n, dtype=np.float64)
        sigy_arr = np.zeros(n, dtype=np.float64)
        seq_arr = np.zeros(n, dtype=np.float64)
        dezz_arr = np.zeros(n, dtype=np.float64)
        soundsp_arr = np.zeros(n, dtype=np.float64)
        et_arr = np.ones(n, dtype=np.float64)

        for i in range(n):
            dxx = float(deps_2d[i, 0])
            dyy = float(deps_2d[i, 1])
            dxy = float(deps_2d[i, 2])
            dyz = float(deps_2d[i, 3]) if deps_2d.shape[1] > 3 else 0.0
            dzx = float(deps_2d[i, 4]) if deps_2d.shape[1] > 4 else 0.0

            so_xx = float(sigo_2d[i, 0])
            so_yy = float(sigo_2d[i, 1])
            so_xy = float(sigo_2d[i, 2])
            so_yz = float(sigo_2d[i, 3]) if sigo_2d.shape[1] > 3 else 0.0
            so_zx = float(sigo_2d[i, 4]) if sigo_2d.shape[1] > 4 else 0.0

            sig_i, ex_i = _single_law110_shell_update(
                params, dxx, dyy, dxy, dyz, dzx, so_xx, so_yy, so_xy, so_yz, so_zx,
                pla_arr[i], temp_arr[i], thk_arr[i], thkly_arr[i], rate_arr[i], dt, has_shear=has_shear
            )
            res_sig_all[i, :min(ncomp, len(sig_i))] = sig_i[:ncomp]
            pla_arr[i] = ex_i["pla"]
            dpla_arr[i] = ex_i["dpla"]
            sigy_arr[i] = ex_i["sigy"]
            seq_arr[i] = ex_i["seq"]
            thk_arr[i] = ex_i["thk"]
            dezz_arr[i] = ex_i["dezz"]
            soundsp_arr[i] = ex_i["soundsp"]
            et_arr[i] = ex_i["et"]

        ex["pla"] = pla_arr
        ex["dpla"] = dpla_arr
        ex["sigy"] = sigy_arr
        ex["seq"] = seq_arr
        ex["thk"] = thk_arr
        ex["dezz"] = dezz_arr
        ex["soundsp"] = soundsp_arr
        ex["et"] = et_arr
        ex["temp"] = temp_arr

        return res_sig_all, ex


def law110_consistent_shell_tangent(
    mat: Any,
    stress: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 3x3 plane stress algorithmic tangent matrix for /MAT/LAW110.

    D = C_el - (C_el : N (x) N : C_el) / (N : C_el : N + H)
    """
    params = get_vegter_params(mat)
    c_el = np.array([
        [params.a11, params.a12, 0.0],
        [params.a12, params.a11, 0.0],
        [0.0, 0.0, params.g],
    ], dtype=np.float64)

    if stress is None:
        return c_el

    s_arr = np.asarray(stress, dtype=np.float64).flatten()
    sxx, syy, sxy = s_arr[0], s_arr[1], s_arr[2]

    ex = extra or {}
    pla = float(ex.get("pla", 0.0))
    rate = float(ex.get("rate", 0.0))
    temp = float(ex.get("temp", params.tini))

    sig_y, h = eval_flow_stress_and_hardening(params, pla, rate, temp)
    sig_vg, norm = eval_vegter_equivalent_stress_and_normal(sxx, syy, sxy, params)

    if sig_vg < sig_y - 1.0e-5 * sig_y:
        return c_el

    cn = c_el @ norm
    denom = float(norm @ cn + max(0.0, h))
    if denom <= 1.0e-12:
        return c_el

    c_ep = c_el - np.outer(cn, cn) / denom
    return c_ep


def law110_sound_speed(mat: Any, rho: Optional[float] = None, is_shell: bool = True, **kwargs: Any) -> float:
    """Calculate sound speed for /MAT/LAW110.

    c = sqrt(A_11 / rho) where A_11 = E / (1 - nu^2).
    """
    params = get_vegter_params(mat)
    r = rho if rho is not None and rho > 0.0 else params.rho0
    return math.sqrt(params.a11 / max(1.0e-12, r))


def resolve(mat: Any, model: Any, log: Any = None) -> None:
    """Hook to resolve /FUNCT references in model for tabulated hardening curves."""
    curves = getattr(model, "curves", {}) or {}
    tab_yld = getattr(mat, "tab_yld", None)
    if tab_yld is None and hasattr(mat, "params") and isinstance(mat.params, dict):
        tab_yld = mat.params.get("MAT_TAB_YLD", mat.params.get("tab_yld"))
    if tab_yld and tab_yld in curves:
        mat.curve_yld = curves[tab_yld]

    tab_temp = getattr(mat, "tab_temp", None)
    if tab_temp is None and hasattr(mat, "params") and isinstance(mat.params, dict):
        tab_temp = mat.params.get("MAT_TAB_TEMP", mat.params.get("tab_temp"))
    if tab_temp and tab_temp in curves:
        mat.curve_temp = curves[tab_temp]


def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return required persistent extra states for /MAT/LAW110."""
    shapes: Dict[str, Tuple[int, ...]] = {}
    if nip:
        shapes["pla"] = (nip,)
        shapes["thk"] = (nip,)
        shapes["sigb"] = (nip, 3)
    else:
        shapes["pla"] = ()
        shapes["thk"] = ()
        shapes["sigb"] = (3,)
    return shapes


def build_law110(mat: Any = None, **kwargs: Any) -> VegterModelParams:
    """Construct VegterModelParams from material entity or keyword arguments."""
    p = VegterModelParams(mat)
    for k, v in kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    return p


VegterParams = VegterModelParams
law110_extra_shapes = extra_shapes
shell_update = law110_shell_update
sound_speed = law110_sound_speed
consistent_shell_tangent = law110_consistent_shell_tangent
