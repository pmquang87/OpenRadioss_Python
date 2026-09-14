"""
LAW94 — Yeoh 3rd-order polynomial hyperelastic material (/MAT/LAW94, /MAT/YEOH).

Implements the Yeoh hyperelastic constitutive law for 3D solid continuum
elements and 2D shell / membrane elements in pure Python / NumPy.

Fortran reference sources:
- engine/source/materials/mat/mat094/sigeps94.F   (3D continuum solid kernel)
- starter/source/materials/mat/mat094/hm_read_mat94.F (starter reader & parameters)
- hm_cfg_files/config/CFG/radioss2017/MAT/matl94_94.cfg (card layout)

Theory
------
The Yeoh model represents a reduced polynomial strain energy function for
incompressible or nearly-incompressible elastomers:
    W = sum_{i=1}^3 C_i0 * (I_1^* - 3)^i + sum_{k=1}^3 (1 / D_k) * (J - 1)^(2k)

where:
    I_1^* = (lambda_1^*)^2 + (lambda_2^*)^2 + (lambda_3^*)^2 (first isochoric invariant)
    lambda_i^* = lambda_i * J^(-1/3) (isochoric principal stretches)
    J = lambda_1 * lambda_2 * lambda_3 = rho0 / rho (relative volume)
    C_10, C_20, C_30: Yeoh polynomial material parameters
    D_1, D_2, D_3: volumetric compressibility parameters

Ground-state initial shear modulus G_0 (hm_read_mat94.F:143):
    G_0 = 2 * C_10

Bulk modulus K (hm_read_mat94.F:148-153):
    If D_1 == 0:
        nu = 0.495
        K = (2/3) * G_0 * (1 + nu) / (1 - 2*nu)
        1/D_1 = K / 2
        E = 2 * G_0 * (1 + nu)
    Else:
        1/D_1 = 1 / D_1_input
        K = 2 * (1 / D_1_input)
        nu = (3*K - 2*G_0) / (2 * (3*K + G_0))
        E = 9*K*G_0 / (3*K + G_0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

_EM10 = 1e-10
_EM20 = 1e-20
_EM30 = 1e-30


@dataclass
class YeohParams:
    """Parameters for /MAT/LAW94 (/MAT/YEOH).

    Attributes
    ----------
    id : int
        Material ID.
    title : str
        Material title.
    rho0 : float
        Initial density.
    ref_rho : float
        Reference density.
    c10 : float
        Yeoh parameter C10 (shear term).
    c20 : float
        Yeoh parameter C20 (moderate-strain curvature).
    c30 : float
        Yeoh parameter C30 (large-strain upturn).
    d1 : float
        Stored inverse parameter 1/D1 (or K/2 if D1_input==0).
    d2 : float
        Stored inverse parameter 1/D2 (or 0 if D2_input==0).
    d3 : float
        Stored inverse parameter 1/D3 (or 0 if D3_input==0).
    d1_raw : float
        Original un-inverted input D1.
    d2_raw : float
        Original un-inverted input D2.
    d3_raw : float
        Original un-inverted input D3.
    g0 : float
        Initial ground-state shear modulus G_0 = 2 * C10.
    rbulk : float
        Bulk modulus K = 2 * d1.
    nu : float
        Equivalent Poisson's ratio.
    e : float
        Equivalent Young's modulus.
    """

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    ref_rho: float = 0.0
    c10: float = 0.0
    c20: float = 0.0
    c30: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    d1_raw: float = 0.0
    d2_raw: float = 0.0
    d3_raw: float = 0.0
    g0: float = 0.0
    rbulk: float = 0.0
    nu: float = 0.495
    e: float = 0.0

    @property
    def K(self) -> float:
        """Bulk modulus alias."""
        return self.rbulk

    @property
    def bulk(self) -> float:
        """Bulk modulus alias."""
        return self.rbulk

    @property
    def G(self) -> float:
        """Initial shear modulus alias."""
        return self.g0

    @property
    def G0(self) -> float:
        """Initial shear modulus alias."""
        return self.g0

    @property
    def E(self) -> float:
        """Equivalent Young's modulus."""
        return self.e

    @property
    def young(self) -> float:
        """Equivalent Young's modulus alias."""
        return self.e

    @property
    def rho(self) -> float:
        """Density alias."""
        return self.ref_rho if self.ref_rho > 0.0 else self.rho0

    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "rho0": self.rho0,
            "ref_rho": self.ref_rho,
            "c10": self.c10,
            "c20": self.c20,
            "c30": self.c30,
            "d1": self.d1,
            "d2": self.d2,
            "d3": self.d3,
            "d1_raw": self.d1_raw,
            "d2_raw": self.d2_raw,
            "d3_raw": self.d3_raw,
            "g0": self.g0,
            "G": self.g0,
            "rbulk": self.rbulk,
            "K": self.rbulk,
            "nu": self.nu,
            "e": self.e,
            "E": self.e,
            "LAW94_C01": self.c10,
            "LAW94_C02": self.c20,
            "LAW94_C03": self.c30,
            "LAW94_D1": self.d1_raw,
            "LAW94_D2": self.d2_raw,
            "LAW94_D3": self.d3_raw,
        }


def build_law94(rec: Any = None, **kwargs: Any) -> YeohParams:
    """Build and initialize YeohParams from material entity or kwargs.

    Implements exact starter conversion logic from hm_read_mat94.F:136-174.
    """
    mid = 1
    title = ""
    rho0 = 1.0
    ref_rho = 0.0
    c10 = 0.0
    c20 = 0.0
    c30 = 0.0
    d1_in = 0.0
    d2_in = 0.0
    d3_in = 0.0

    if rec is not None:
        mid = int(getattr(rec, "id", getattr(rec, "mat_id", 1)))
        rho0 = float(getattr(rec, "rho0", getattr(rec, "density", 1.0)))
        title = str(getattr(rec, "title", ""))
        p = getattr(rec, "params", {})
        if p and isinstance(p, dict):
            c10 = float(p.get("LAW94_C01", p.get("c10", p.get("C10", getattr(rec, "c10", 0.0)))))
            c20 = float(p.get("LAW94_C02", p.get("c20", p.get("C20", getattr(rec, "c20", 0.0)))))
            c30 = float(p.get("LAW94_C03", p.get("c30", p.get("C30", getattr(rec, "c30", 0.0)))))
            d1_in = float(p.get("LAW94_D1", p.get("d1", p.get("D1", getattr(rec, "d1", 0.0)))))
            d2_in = float(p.get("LAW94_D2", p.get("d2", p.get("D2", getattr(rec, "d2", 0.0)))))
            d3_in = float(p.get("LAW94_D3", p.get("d3", p.get("D3", getattr(rec, "d3", 0.0)))))
            ref_rho = float(p.get("refer_rho", getattr(rec, "ref_rho", 0.0)))
        else:
            c10 = float(getattr(rec, "c10", 0.0))
            c20 = float(getattr(rec, "c20", 0.0))
            c30 = float(getattr(rec, "c30", 0.0))
            d1_in = float(getattr(rec, "d1", 0.0))
            d2_in = float(getattr(rec, "d2", 0.0))
            d3_in = float(getattr(rec, "d3", 0.0))
            ref_rho = float(getattr(rec, "ref_rho", 0.0))

    # Kwargs overrides
    if "c10" in kwargs:
        c10 = float(kwargs["c10"])
    if "C10" in kwargs:
        c10 = float(kwargs["C10"])
    if "c20" in kwargs:
        c20 = float(kwargs["c20"])
    if "C20" in kwargs:
        c20 = float(kwargs["C20"])
    if "c30" in kwargs:
        c30 = float(kwargs["c30"])
    if "C30" in kwargs:
        c30 = float(kwargs["C30"])
    if "d1" in kwargs:
        d1_in = float(kwargs["d1"])
    if "D1" in kwargs:
        d1_in = float(kwargs["D1"])
    if "d2" in kwargs:
        d2_in = float(kwargs["d2"])
    if "D2" in kwargs:
        d2_in = float(kwargs["D2"])
    if "d3" in kwargs:
        d3_in = float(kwargs["d3"])
    if "D3" in kwargs:
        d3_in = float(kwargs["D3"])
    if "rho" in kwargs:
        rho0 = float(kwargs["rho"])
    if "rho0" in kwargs:
        rho0 = float(kwargs["rho0"])
    if "ref_rho" in kwargs:
        ref_rho = float(kwargs["ref_rho"])
    if "id" in kwargs:
        mid = int(kwargs["id"])

    # Upstream starter logic (hm_read_mat94.F:140-156)
    d2 = (1.0 / d2_in) if d2_in != 0.0 else 0.0
    d3 = (1.0 / d3_in) if d3_in != 0.0 else 0.0

    g = 2.0 * c10

    if d1_in == 0.0:
        d2 = 0.0
        d3 = 0.0
        nu = 0.495
        rbulk = (2.0 / 3.0) * g * (1.0 + nu) / max(_EM30, (1.0 - 2.0 * nu))
        d1 = rbulk / 2.0  # Represents 1/D1 stored in UPARAM(7)
        e = 2.0 * g * (1.0 + nu)
    else:
        d1 = 1.0 / d1_in  # 1/D1 stored in UPARAM(7)
        rbulk = 2.0 * d1  # 2 / D1_in
        denom = 3.0 * rbulk + g
        if denom > _EM30:
            nu = (3.0 * rbulk - 2.0 * g) / (2.0 * denom)
            e = 9.0 * rbulk * g / denom
        else:
            nu = 0.495
            e = 2.0 * g * (1.0 + nu)

    return YeohParams(
        id=int(mid),
        title=str(title),
        rho0=float(rho0),
        ref_rho=float(ref_rho),
        c10=float(c10),
        c20=float(c20),
        c30=float(c30),
        d1=float(d1),
        d2=float(d2),
        d3=float(d3),
        d1_raw=float(d1_in),
        d2_raw=float(d2_in),
        d3_raw=float(d3_in),
        g0=float(g),
        rbulk=float(rbulk),
        nu=float(nu),
        e=float(e),
    )


def yeoh_analytical_stress(
    stretch: float,
    c10: float,
    c20: float = 0.0,
    c30: float = 0.0,
    itype: int = 1,
) -> float:
    """Calculate nominal (engineering) tensile stress for incompressible Yeoh model.

    Parameters
    ----------
    stretch : float
        Principal stretch ratio lambda.
    c10, c20, c30 : float
        Yeoh polynomial parameters.
    itype : int
        1 = Uniaxial, 2 = Equibiaxial, 3 = Planar shear.

    Returns
    -------
    float
        Engineering/nominal stress sigma_nom.
    """
    if stretch <= 0.0:
        return 0.0

    if itype == 1:
        # Uniaxial: lambda_1 = lambda, lambda_2 = lambda_3 = lambda^(-1/2)
        i1 = stretch * stretch + 2.0 / stretch
        fac = 2.0 * (stretch - 1.0 / (stretch * stretch))
    elif itype == 2:
        # Equibiaxial: lambda_1 = lambda_2 = lambda, lambda_3 = lambda^(-2)
        i1 = 2.0 * stretch * stretch + 1.0 / (stretch ** 4)
        fac = 2.0 * (stretch - 1.0 / (stretch ** 5))
    elif itype == 3:
        # Planar: lambda_1 = lambda, lambda_2 = 1, lambda_3 = lambda^(-1)
        i1 = stretch * stretch + 1.0 + 1.0 / (stretch * stretch)
        fac = 2.0 * (stretch - 1.0 / (stretch ** 3))
    else:
        i1 = stretch * stretch + 2.0 / stretch
        fac = 2.0 * (stretch - 1.0 / (stretch * stretch))

    aa = i1 - 3.0
    dW_dI1 = c10 + 2.0 * c20 * aa + 3.0 * c30 * (aa * aa)
    return float(fac * dW_dI1)


def solid_update(
    mat: Any = None,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    ismstr: int = 0,
    *,
    epsp: np.ndarray | None = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, dict, float]:
    """3D solid continuum Yeoh hyperelastic stress update.

    Fortran reference: engine/source/materials/mat/mat094/sigeps94.F
    Returns (sig_out, epsp, sound_speed) or (sig_out, hist_dict, sound_speed).
    """
    return_constitutive_dict = False
    if isinstance(mat, np.ndarray) and (sig is None or not isinstance(sig, np.ndarray)):
        eps_in = mat
        params_kw = kwargs.copy()
        if isinstance(sig, dict):
            params_kw.update(sig)
        mat = build_law94(**params_kw)
        sig_arr = np.zeros_like(eps_in) if eps_in.ndim > 1 else np.zeros((1, 6))
        deps_arr = np.zeros_like(sig_arr)
        eps_arr = eps_in[np.newaxis, :] if eps_in.ndim == 1 else eps_in
        epsp_arr = np.zeros(len(eps_arr))
        is_1d = (eps_in.ndim == 1)
        return_constitutive_dict = True
        if "rho" in kwargs and extra is None:
            extra = {"rho": kwargs["rho"]}
    else:
        if not isinstance(mat, YeohParams):
            mat = build_law94(mat, **kwargs)
        if sig is None:
            sig = np.zeros(6, dtype=np.float64)
        is_1d = (sig.ndim == 1)
        if is_1d:
            sig_arr = sig[np.newaxis, :]
            deps_arr = deps[np.newaxis, :] if deps is not None else np.zeros((1, 6))
            eps_arr = eps[np.newaxis, :] if eps is not None else np.zeros((1, 6))
            epsp_arr = epsp[np.newaxis] if epsp is not None else np.zeros(1)
        else:
            sig_arr = sig
            deps_arr = deps if deps is not None else np.zeros_like(sig)
            eps_arr = eps if eps is not None else np.zeros_like(sig)
            epsp_arr = epsp if epsp is not None else np.zeros(len(sig))

    nel = len(sig_arr)
    sig_out = np.zeros_like(sig_arr)
    sound_sp = np.zeros(nel, dtype=np.float64)

    # Current strain tensor: preserve total strain via extra['eps94']
    if extra is not None and "eps94" in extra and extra["eps94"] is not None:
        if np.asarray(extra["eps94"]).ndim == 1 and deps_arr.shape[0] == 1:
            extra["eps94"] = np.asarray(extra["eps94"]) + deps_arr[0]
            tot_eps = np.asarray(extra["eps94"])[np.newaxis, :]
        else:
            extra["eps94"] = np.asarray(extra["eps94"]) + deps_arr
            tot_eps = np.asarray(extra["eps94"])
    elif eps is not None:
        tot_eps = eps_arr + deps_arr if deps is not None else eps_arr.copy()
        if extra is not None and isinstance(extra, dict):
            extra["eps94"] = tot_eps[0].copy() if is_1d else tot_eps.copy()
    else:
        tot_eps = deps_arr.copy()
        if extra is not None and isinstance(extra, dict):
            extra["eps94"] = tot_eps[0].copy() if is_1d else tot_eps.copy()

    c10 = mat.c10
    c20 = mat.c20
    c30 = mat.c30
    c0 = np.array([c10, c20, c30], dtype=np.float64)
    d1 = mat.d1
    d2 = mat.d2
    d3 = mat.d3
    g = mat.g0
    rbulk = mat.rbulk

    if extra is not None and "rho" in extra and extra["rho"] is not None:
        rho_val = np.asarray(extra["rho"])
        if rho_val.ndim == 0:
            rho_arr = np.full(nel, float(rho_val))
        else:
            rho_arr = rho_val
    else:
        rho_arr = np.full(nel, mat.rho0)

    mullins_w = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        exx = tot_eps[i, 0]
        eyy = tot_eps[i, 1]
        ezz = tot_eps[i, 2]
        exy = 0.5 * tot_eps[i, 3]
        eyz = 0.5 * tot_eps[i, 4]
        ezx = 0.5 * tot_eps[i, 5]

        mat_eps = np.array([
            [exx, exy, ezx],
            [exy, eyy, eyz],
            [ezx, eyz, ezz]
        ], dtype=np.float64)

        evals, evecs = np.linalg.eigh(mat_eps)

        # Stretches (sigeps94.F:148-174)
        if ismstr in (0, 2, 4):
            ev = np.exp(evals)
        else:
            ev = evals + 1.0
        ev = np.maximum(ev, 1e-12)

        # Relative volume J = lambda_1 * lambda_2 * lambda_3
        rv = ev[0] * ev[1] * ev[2]
        rv = max(rv, _EM20)
        invr = 1.0 / rv
        rvd = np.exp((-1.0 / 3.0) * np.log(rv))

        # Isochoric stretches lambda_k^*
        evd = ev * rvd
        trace = np.sum(evd * evd)

        l1di1lam = 2.0 * (evd * evd - (1.0 / 3.0) * trace)

        aa = trace - 3.0
        bb = aa * aa
        cc = (c0[0] + 2.0 * c0[1] * aa + 3.0 * c0[2] * bb) * invr

        # Deviatoric stresses (sigeps94.F:234-236)
        t_dev = l1di1lam * cc

        # Volumetric pressure P (sigeps94.F:238)
        # Note: in sigeps94.F, P = RBULK*(RV - 1) + 4*D(2)*(RV - 1)**3 + 6*D(3)*(RV - 1)**5
        j_minus_1 = rv - 1.0
        p = rbulk * j_minus_1 + 4.0 * d2 * (j_minus_1 ** 3) + 6.0 * d3 * (j_minus_1 ** 5)

        # Total principal Cauchy stress (sigeps94.F:240-242)
        t_cauchy = t_dev + p

        # Mullins / internal strain energy density
        w_iso = c10 * aa + c20 * bb + c30 * (aa * bb)
        w_vol = d1 * (j_minus_1 ** 2) + d2 * (j_minus_1 ** 4) + d3 * (j_minus_1 ** 6)
        mullins_w[i] = w_iso + w_vol

        # Sound speed computation (sigeps94.F:245-272)
        cii = np.zeros(3, dtype=np.float64)
        if abs(aa) >= _EM10:
            for ii in range(1, 4):
                clp = 4.0 * ii * c0[ii - 1]
                lam_2 = evd * evd
                lam_4 = lam_2 * lam_2
                aa_c = (1.0 / 9.0) * ii * (aa ** ii)
                bb_c = (1.0 / 3.0) * (3.0 - ii) * (aa ** (ii - 1)) if ii > 1 else 0.0
                cc_c = (ii - 1.0) * (aa ** (ii - 2)) if ii > 2 else 0.0
                cii += clp * (aa_c + bb_c * lam_2 + cc_c * lam_4)

        amax = float(np.max(cii))
        eti = max(1.0, amax * 0.81)
        gtmax = g * eti
        rkmax = rbulk + 12.0 * d2 * (j_minus_1 ** 2) + 30.0 * d3 * (j_minus_1 ** 4)
        rkmax = max(rbulk, rkmax)

        # Global Cauchy stress tensor reconstruction (sigeps94.F:275-297)
        diag_t = np.diag(t_cauchy)
        sig_tensor = evecs @ diag_t @ evecs.T

        sig_out[i, 0] = sig_tensor[0, 0]
        sig_out[i, 1] = sig_tensor[1, 1]
        sig_out[i, 2] = sig_tensor[2, 2]
        sig_out[i, 3] = sig_tensor[0, 1]
        sig_out[i, 4] = sig_tensor[1, 2]
        sig_out[i, 5] = sig_tensor[2, 0]

        cur_rho = max(rho_arr[i], 1e-12)
        sound_sp[i] = np.sqrt(((4.0 / 3.0) * gtmax + rkmax) / cur_rho)

    if extra is not None:
        extra["mullins_w"] = mullins_w

    sig_res = sig_out[0] if is_1d else sig_out
    epsp_res = epsp_arr[0] if is_1d else epsp_arr
    sound_res = sound_sp[0] if is_1d else sound_sp

    if return_constitutive_dict:
        hist_dict = {"w_mullins": float(mullins_w[0])}
        return sig_res, hist_dict, float(sound_res)

    if return_tuple:
        return sig_res, epsp_res, sound_res
    return sig_res, epsp_res


def shell_update(
    mat: Any = None,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    *,
    eps: np.ndarray | None = None,
    **kwargs: Any,
) -> Any:
    """2D shell plane-stress Yeoh hyperelastic stress update.

    Enforces plane-stress condition sigma_zz = 0 iteratively via Newton-Raphson
    on out-of-plane stretch lambda_3, analogous to LAW82/LAW69/LAW92.
    """
    return_constitutive_shell = False
    if isinstance(mat, np.ndarray) and (sig is None or not isinstance(sig, np.ndarray)):
        eps_in = mat
        params_kw = kwargs.copy()
        if isinstance(sig, dict):
            params_kw.update(sig)
        mat = build_law94(**params_kw)
        is_1d = (eps_in.ndim == 1)
        sig_arr = np.zeros((1, 3) if is_1d else (len(eps_in), 3), dtype=np.float64)
        deps_arr = eps_in[np.newaxis, :3] if is_1d else eps_in[:, :3]
        epsp_arr = np.zeros(1 if is_1d else len(eps_in))
        return_constitutive_shell = True
        if "rho" in kwargs and extra is None:
            extra = {"rho": kwargs["rho"]}
    else:
        if not isinstance(mat, YeohParams):
            mat = build_law94(mat, **kwargs)
        if sig is None:
            sig = np.zeros(3, dtype=np.float64)
        is_1d = (sig.ndim == 1)
        if is_1d:
            sig_arr = sig[np.newaxis, :]
            deps_arr = deps[np.newaxis, :] if deps is not None else np.zeros((1, len(sig)))
            epsp_arr = epsp[np.newaxis] if epsp is not None else np.zeros(1)
        else:
            sig_arr = sig
            deps_arr = deps if deps is not None else np.zeros_like(sig)
            epsp_arr = epsp if epsp is not None else np.zeros(len(sig))

    nel = len(sig_arr)
    ncomp = sig_arr.shape[1]
    sig_out = np.zeros_like(sig_arr)

    # Accumulated in-plane total strain
    if return_constitutive_shell:
        eps_tot = deps_arr[:, :3]
    elif eps is not None:
        eps_in = np.asarray(eps, dtype=np.float64)
        eps_tot = eps_in[np.newaxis, :3] if eps_in.ndim == 1 else eps_in[:, :3]
    elif extra is not None and "eps94" in extra:
        extra["eps94"] += deps_arr[:, :3]
        eps_tot = extra["eps94"]
    elif extra is not None and "eps" in extra:
        eps_in = np.asarray(extra["eps"], dtype=np.float64)
        eps_tot = eps_in[np.newaxis, :3] if eps_in.ndim == 1 else eps_in[:, :3]
    elif "eps" in kwargs and kwargs["eps"] is not None:
        eps_in = np.asarray(kwargs["eps"], dtype=np.float64)
        eps_tot = eps_in[np.newaxis, :3] if eps_in.ndim == 1 else eps_in[:, :3]
    else:
        if extra is not None and isinstance(extra, dict):
            extra["eps94"] = deps_arr[:, :3].copy()
            eps_tot = extra["eps94"]
        else:
            eps_tot = deps_arr[:, :3]

    # History variable for out-of-plane stretch lambda_3
    if extra is not None and "uvar_lam3" in extra:
        lam3_old = np.asarray(extra["uvar_lam3"]).copy()
    else:
        lam3_old = np.ones(nel, dtype=np.float64)

    lam3_new = np.zeros(nel, dtype=np.float64)

    c10 = mat.c10
    c20 = mat.c20
    c30 = mat.c30
    d2 = mat.d2
    d3 = mat.d3
    g0 = mat.g0
    rbulk = mat.rbulk

    for i in range(nel):
        exx = float(eps_tot[i, 0])
        eyy = float(eps_tot[i, 1])
        exy = float(eps_tot[i, 2]) if ncomp > 2 else 0.0

        mat_2d = np.array([
            [exx, 0.5 * exy],
            [0.5 * exy, eyy]
        ], dtype=np.float64)

        evals_2d, evecs_2d = np.linalg.eigh(mat_2d)
        evals_2d = np.clip(evals_2d, -10.0, 10.0)
        lam1 = max(1e-4, min(100.0, float(np.exp(evals_2d[0]))))
        lam2 = max(1e-4, min(100.0, float(np.exp(evals_2d[1]))))

        # Newton-Raphson solver for lambda_3 such that sigma_3(lambda_3) = 0
        l3 = max(0.1, min(10.0, float(lam3_old[i])))
        for _ in range(15):
            rv = lam1 * lam2 * l3
            invr = 1.0 / max(rv, _EM20)
            rvd = np.exp((-1.0 / 3.0) * np.log(max(rv, _EM20)))
            evd = np.array([lam1, lam2, l3]) * rvd
            trace = np.sum(evd * evd)

            di1lam3 = 2.0 * (evd[2] ** 2 - (1.0 / 3.0) * trace)
            aa = trace - 3.0
            bb = aa * aa
            cc = (c10 + 2.0 * c20 * aa + 3.0 * c30 * bb) * invr

            t_dev3 = di1lam3 * cc
            j_minus_1 = rv - 1.0
            p = rbulk * j_minus_1 + 4.0 * d2 * (j_minus_1 ** 3) + 6.0 * d3 * (j_minus_1 ** 5)
            sig3 = t_dev3 + p

            if abs(sig3) < 1e-6 * max(1.0, g0):
                break

            # Numerical derivative d(sig3)/d(l3)
            dl3 = 1e-6 * l3
            l3_p = l3 + dl3
            rv_p = lam1 * lam2 * l3_p
            invr_p = 1.0 / max(rv_p, _EM20)
            rvd_p = np.exp((-1.0 / 3.0) * np.log(max(rv_p, _EM20)))
            evd_p = np.array([lam1, lam2, l3_p]) * rvd_p
            trace_p = np.sum(evd_p * evd_p)
            di1lam3_p = 2.0 * (evd_p[2] ** 2 - (1.0 / 3.0) * trace_p)
            aa_p = trace_p - 3.0
            bb_p = aa_p * aa_p
            cc_p = (c10 + 2.0 * c20 * aa_p + 3.0 * c30 * bb_p) * invr_p
            t_dev3_p = di1lam3_p * cc_p
            jp_minus_1 = rv_p - 1.0
            p_p = rbulk * jp_minus_1 + 4.0 * d2 * (jp_minus_1 ** 3) + 6.0 * d3 * (jp_minus_1 ** 5)
            sig3_p = t_dev3_p + p_p

            dsig3 = (sig3_p - sig3) / dl3
            if abs(dsig3) < 1e-12:
                break
            step = sig3 / dsig3
            step = np.clip(step, -0.5 * l3, 0.5 * l3)
            l3 = max(0.01, min(100.0, l3 - step))

        lam3_new[i] = l3

        # Compute in-plane principal Cauchy stresses
        rv = lam1 * lam2 * l3
        invr = 1.0 / max(rv, _EM20)
        rvd = np.exp((-1.0 / 3.0) * np.log(max(rv, _EM20)))
        evd = np.array([lam1, lam2, l3]) * rvd
        trace = np.sum(evd * evd)

        di1lam = 2.0 * (evd[:2] ** 2 - (1.0 / 3.0) * trace)
        aa = trace - 3.0
        bb = aa * aa
        cc = (c10 + 2.0 * c20 * aa + 3.0 * c30 * bb) * invr
        t_dev2 = di1lam * cc
        j_minus_1 = rv - 1.0
        p = rbulk * j_minus_1 + 4.0 * d2 * (j_minus_1 ** 3) + 6.0 * d3 * (j_minus_1 ** 5)
        sig_princ_2d = t_dev2 + p

        # Rotate back to shell reference frame
        sig_tensor_2d = evecs_2d @ np.diag(sig_princ_2d) @ evecs_2d.T
        sig_out[i, 0] = sig_tensor_2d[0, 0]
        sig_out[i, 1] = sig_tensor_2d[1, 1]
        if ncomp > 2:
            sig_out[i, 2] = sig_tensor_2d[0, 1]

        # Transverse shear stresses if present
        if ncomp >= 5:
            sig_out[i, 3] = sig_arr[i, 3] + 2.0 * g0 * deps_arr[i, 3]
            sig_out[i, 4] = sig_arr[i, 4] + 2.0 * g0 * deps_arr[i, 4]

    if extra is not None:
        extra["uvar_lam3"] = lam3_new
        if "thk" in extra:
            extra["thk"] = extra["thk"] * (lam3_new / np.maximum(1e-12, lam3_old))

    r = mat.rho0
    if extra is not None and "rho" in extra and extra["rho"] is not None:
        r = float(np.asarray(extra["rho"]).flat[0])
    c_shell = float(np.sqrt(mat.E / (max(1e-4, 1.0 - mat.nu ** 2) * max(r, _EM20))))

    if return_constitutive_shell:
        sig_ret = sig_out[0, :3] if is_1d else sig_out[:, :3]
        eps_zz = float(np.log(lam3_new[0])) if is_1d else np.log(lam3_new)
        hist = {"lam3": float(lam3_new[0]), "w_mullins": 0.0}
        return sig_ret, eps_zz, hist, c_shell

    if kwargs.get("return_sound_speed", False) or kwargs.get("return_tuple", False):
        if is_1d:
            return sig_out[0], epsp_arr[0], c_shell
        return sig_out, epsp_arr, c_shell

    if is_1d:
        return sig_out[0], epsp_arr[0]
    return sig_out, epsp_arr


def sound_speed(
    mat: Any = None,
    eps: np.ndarray | None = None,
    *,
    rho: float | None = None,
    is_shell: bool = False,
    extra: dict | None = None,
    **kwargs: Any,
) -> float:
    """Solid or shell acoustic wave sound speed for /MAT/LAW94."""
    if isinstance(mat, np.ndarray):
        eps = mat
        mat = build_law94(**kwargs)
    elif not isinstance(mat, YeohParams):
        mat = build_law94(mat, **kwargs)

    r = rho if rho is not None else kwargs.get("rho", None)
    if is_shell:
        return sound_speed_shell(mat, rho=r, extra=extra)

    if eps is not None:
        eps_arr = np.asarray(eps, dtype=np.float64)
        if eps_arr.size >= 6:
            r_val = r if r is not None else mat.rho0
            _, _, c_arr = solid_update(mat, np.zeros(6), eps=eps_arr, extra={"rho": r_val})
            return float(c_arr if np.isscalar(c_arr) else c_arr[0])

    r_val = r if r is not None else mat.rho0
    r_val = max(float(r_val), _EM20)
    return float(np.sqrt(((4.0 / 3.0) * mat.g0 + mat.rbulk) / r_val))


def sound_speed_shell(mat: Any, rho: float | None = None, extra: dict | None = None) -> float:
    """Shell acoustic wave sound speed for /MAT/LAW94."""
    if not isinstance(mat, YeohParams):
        mat = build_law94(mat)
    r = rho if rho is not None else mat.rho0
    r = max(float(r), _EM20)
    nu = mat.nu
    e = mat.E
    return float(np.sqrt(e / (max(1e-4, 1.0 - nu * nu) * r)))


def consistent_tangent(
    mat: Any = None,
    eps: np.ndarray | None = None,
    sig: np.ndarray | None = None,
    dt: float = 0.0,
    ismstr: int = 0,
    is_shell: bool = False,
    perturb: float = 1e-7,
    **kwargs: Any,
) -> np.ndarray:
    """Compute algorithmic consistent tangent stiffness tensor via central finite differences.

    Returns (6, 6) for solids or (3, 3) for shells.
    """
    if isinstance(mat, np.ndarray):
        eps = mat
        mat = build_law94(**kwargs)
    elif not isinstance(mat, YeohParams):
        mat = build_law94(mat, **kwargs)

    if eps is None:
        eps = np.zeros(3 if is_shell else 6, dtype=np.float64)

    if is_shell:
        ncomp = 3
        c_mat = np.zeros((ncomp, ncomp), dtype=np.float64)
        base_eps = eps[:3].copy()
        for j in range(ncomp):
            eps_pos = base_eps.copy()
            eps_neg = base_eps.copy()
            eps_pos[j] += perturb
            eps_neg[j] -= perturb
            s_pos, _ = shell_update(mat, np.zeros(ncomp), deps=eps_pos)
            s_neg, _ = shell_update(mat, np.zeros(ncomp), deps=eps_neg)
            c_mat[:, j] = (s_pos[:ncomp] - s_neg[:ncomp]) / (2.0 * perturb)
        return c_mat
    else:
        ncomp = 6
        c_mat = np.zeros((ncomp, ncomp), dtype=np.float64)
        base_eps = eps[:6].copy()
        for j in range(ncomp):
            eps_pos = base_eps.copy()
            eps_neg = base_eps.copy()
            eps_pos[j] += perturb
            eps_neg[j] -= perturb
            s_pos, _, _ = solid_update(mat, np.zeros(ncomp), deps=eps_pos, ismstr=ismstr)
            s_neg, _, _ = solid_update(mat, np.zeros(ncomp), deps=eps_neg, ismstr=ismstr)
            c_mat[:, j] = (s_pos[:ncomp] - s_neg[:ncomp]) / (2.0 * perturb)
        return c_mat


def resolve(mat: Any, model: Any, log: Any = None) -> Any:
    """Starter hook for /MAT/LAW94."""
    if not isinstance(mat, YeohParams):
        return build_law94(mat)
    return mat


def extra_shapes(mat: Any = None, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Extra history shapes needed for LAW94 Yeoh."""
    if nip:
        return {"eps94": (nip, 3), "uvar_lam3": (nip,)}
    return {"eps94": (6,)}

