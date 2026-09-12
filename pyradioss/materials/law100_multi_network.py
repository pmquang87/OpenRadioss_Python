"""
LAW100 — Multi-Network Visco-Hyperelastic Polymer Model (/MAT/LAW100, /MAT/VISC_HYP, /MAT/MNF).

Implements the general parallel rheological framework (PRF) constitutive law for 3D solid continuum
elements in pure Python / NumPy, matching upstream OpenRadioss.

Fortran reference sources:
- engine/source/materials/mat/mat100/sigeps100.F90    (3D continuum solid kernel)
- starter/source/materials/mat/mat100/hm_read_mat100.F (starter reader & parameters)
- starter/source/materials/mat/mat100/law100_upd.F    (temperature & curve update)
- engine/source/materials/mat/mat100/calcmatb.F      (multiplicative kinematic split)
- engine/source/materials/mat/mat100/viscbb.F        (Bergstrom-Boyce creep flow rule)
- engine/source/materials/mat/mat100/viscsinh.F      (Hyperbolic sine creep flow rule)
- engine/source/materials/mat/mat100/viscpower.F     (Power law creep flow rule)
- engine/source/materials/mat/mat100/sigpoly.F       (Polynomial strain energy potential)
- engine/source/materials/mat/mat100/sigaboyce.F     (Arruda-Boyce 8-chain potential)
- engine/source/materials/mat/mat100/neo_hook_t.F    (Thermal Neo-Hookean potential)
- hm_cfg_files/config/CFG/radioss2020/MAT/LAW100.cfg (card layout definitions)

Theory & Kinematics
-------------------
The PRF decomposes total stress into an equilibrium network (Network A) plus N_net parallel
time-dependent secondary networks (Networks B_k):
    sigma = sigma_A + sum_{k=1}^{N_net} sigma_B_k

1. Primary / Equilibrium Network (A):
    - Pure hyperelasticity (Flag_Cr = 0) or creep/plasticity (Flag_Cr = 1).
    - Strain energy potential W_A selected by Flag_HE:
        * Flag_HE = 1: Polynomial form (C10..C03, D1..D3)
        * Flag_HE = 2: Arruda-Boyce 8-chain model (mu, D, lambda_M, itype, nu)
        * Flag_HE = 3: Neo-Hookean (C10, D1) -> polynomial reduction
        * Flag_HE = 4: Mooney-Rivlin (C10, C01, D1) -> polynomial reduction
        * Flag_HE = 5: Yeoh (C10, C20, C30, D1) -> polynomial reduction
        * Flag_HE = 13: Neo-Hookean with temperature-dependent curves (fct_ID_SM, fct_ID_BM)
    - If Flag_Cr = 1:
        F = F_e * F_p_eq => F_e = F * F_p_eq^(-1), b_e = F_e * F_e^T
        tau_A = ||dev(sigma_A)||
        p_dot = A_pl * (tau_A / tau_y)^N_pl
        tau_y = tau_y0 * [F_pl + (1 - F_pl) * exp(-p / eps_pl)]

2. Secondary Networks (B_k, k = 1..N_net):
    - Each network has stiffness scale factor S_k, and viscous flow rule Flag_visc:
        * Flag_visc = 1: Bergstrom-Boyce:
            dgamma_k = A_1k * dt * (lambda_p - 1 + xi_k)^C_k * (tau_k / tau_ref_k)^M_k
        * Flag_visc = 2: Hyperbolic Sine:
            dgamma_k = A_2k * dt * [sinh(B_k * tau_k)]^n_2k
        * Flag_visc = 3: Power Law:
            dgamma_k = A_3k * dt * [tau_k^n_3k * ((M_3k + 1) * gamma_old_k)^M_3k]^(1 / (1 + M_3k))
    - Multiplicative split:
        F = F_ek * F_pk => F_ek = F * F_pk^(-1), b_ek = F_ek * F_ek^T
        sigma_B_k,trial = S_k * sigma_HE(b_ek)
        s_k = dev(sigma_B_k,trial), tau_k = ||s_k||
        L_k = (dgamma_k / tau_k) * s_k
        D_Fpk = F_ek^(-1) * L_k * F_ek
        F_pk^(n+1) = (I + D_Fpk) * F_pk^n
        Recompute elastic left Cauchy-Green tensor with F_pk^(n+1) and update sigma_B_k.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

_EM10 = 1e-10
_EM20 = 1e-20
_EM30 = 1e-30


@dataclass
class SecondaryNetworkParams:
    """Parameters for a secondary time-dependent relaxation network in LAW100.

    Attributes
    ----------
    network_id : int
        Network index (1-based).
    flag_visc : int
        Viscous flow formulation:
        1: Bergstrom-Boyce
        2: Hyperbolic sine
        3: Power law
    stiffness : float
        Stiffness weight factor S_k.
    a : float
        Creep strain rate parameter A (A1, A2, or A3).
    expc : float
        Creep strain exponent C for Bergstrom-Boyce (-1 < C < 0, default -0.7).
    expm : float
        Stress exponent M for Bergstrom-Boyce (default 1.0) or M3 for Power law.
    ksi : float
        Regularization constant xi near undeformed state (default 0.01).
    tauref : float
        Reference stress tau_ref for Bergstrom-Boyce (default 1.0).
    b0 : float
        Multiplier B in Hyperbolic sine model.
    expn : float
        Exponent n2 (Hyperbolic sine) or n3 (Power law).
    """

    network_id: int = 1
    flag_visc: int = 1
    stiffness: float = 1.0
    a: float = 0.0
    expc: float = -0.7
    expm: float = 1.0
    ksi: float = 0.01
    tauref: float = 1.0
    b0: float = 0.0
    expn: float = 1.0

    def __init__(
        self,
        network_id: int = 1,
        flag_visc: int = 1,
        stiffness: float = 1.0,
        a: float = 0.0,
        expc: float = -0.7,
        expm: float = 1.0,
        ksi: float = 0.01,
        tauref: float = 1.0,
        b0: float = 0.0,
        expn: float = 1.0,
        **kwargs: Any,
    ):
        for k, v in kwargs.items():
            kl = k.lower()
            if kl in ("net_id", "networkid", "id"): network_id = int(v)
            elif kl in ("flag_visc", "flagvisc"): flag_visc = int(v)
            elif kl in ("stiffness", "stiff", "sb"): stiffness = float(v)
            elif kl in ("a", "a1", "a2", "a3"): a = float(v)
            elif kl in ("c", "expc"): expc = float(v)
            elif kl in ("m", "expm", "m3"): expm = float(v)
            elif kl == "ksi": ksi = float(v)
            elif kl in ("tau_ref", "tauref"): tauref = float(v)
            elif kl in ("b", "b0"): b0 = float(v)
            elif kl in ("n", "expn", "n2", "n3"): expn = float(v)

        self.network_id = network_id
        self.flag_visc = flag_visc
        self.stiffness = stiffness
        self.a = a
        self.expc = expc
        self.expm = expm
        self.ksi = ksi
        self.tauref = tauref
        self.b0 = b0
        self.expn = expn


@dataclass
class MultiNetworkParams:
    """Constitutive parameters for /MAT/LAW100 (/MAT/VISC_HYP, /MAT/MNF).

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
    n_net : int
        Number of secondary networks (0 <= N_net <= 50).
    flag_he : int
        Hyperelastic model flag:
        1: Polynomial form
        2: Arruda-Boyce
        3: Neo-Hookean
        4: Mooney-Rivlin
        5: Yeoh
        13: Thermal Neo-Hookean
    flag_cr : int
        Equilibrium network creep/plasticity flag (0: none, 1: creep active).
    c10, c01, c20, c11, c02, c30, c21, c12, c03 : float
        Coefficients of polynomial strain energy potential.
    d1_raw, d2_raw, d3_raw : float
        Input volumetric parameters D_1, D_2, D_3.
    d1, d2, d3 : float
        Stored inverted volumetric parameters (1/D_1, 1/D_2, 1/D_3).
    mu : float
        Shear modulus for Arruda-Boyce / Neo-Hookean.
    d_raw : float
        Volumetric parameter D for Arruda-Boyce.
    d_inv : float
        Inverted parameter 1/D for Arruda-Boyce.
    lambda_m : float
        Limit stretch parameter for Arruda-Boyce (default 7.0).
    itype : int
        Test data type for Arruda-Boyce (1: Uniaxial, 2: Equibiaxial, 3: Planar).
    fct_id_ab : int
        Function ID for stress-strain curve in Arruda-Boyce.
    nu_val : float
        Input Poisson's ratio.
    fscale_ab : float
        Scale factor for fct_id_ab.
    fct_id_sm, fct_id_bm : int
        Function IDs for Shear and Bulk modulus vs temperature.
    fscale_sm, fscale_bm : float
        Scale factors for temperature curves.
    a_pl : float
        Scaling factor for plastic flow rule in equilibrium network (default 1.0).
    sigma_pl : float
        Flow resistance / initial yield stress tau_y0 (default 1.0).
    f_pl : float
        Weight factor for flow resistance (default 1.0).
    epsilon_f : float
        Characteristic strain eps_pl (default 1.0).
    n_pl : int
        Exponent for plastic flow rule (default 1).
    networks : list of SecondaryNetworkParams
        List of secondary relaxation networks.
    iform : int
        Formulation flag for volumetric energy (1: standard, 2: modified).
    sb : float
        Sum of secondary network stiffnesses sum(S_k).
    g0 : float
        Ground-state shear modulus G_0.
    rbulk : float
        Ground-state bulk modulus K.
    nu : float
        Effective Poisson's ratio.
    e : float
        Equivalent Young's modulus E.
    sound_speed : float
        Small-strain solid sound speed.
    """

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    ref_rho: float = 0.0
    n_net: int = 0
    flag_he: int = 1
    flag_cr: int = 0

    # Polynomial parameters
    c10: float = 0.0
    c01: float = 0.0
    c20: float = 0.0
    c11: float = 0.0
    c02: float = 0.0
    c30: float = 0.0
    c21: float = 0.0
    c12: float = 0.0
    c03: float = 0.0
    d1_raw: float = 0.0
    d2_raw: float = 0.0
    d3_raw: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0

    # Arruda-Boyce parameters
    mu: float = 0.0
    d_raw: float = 0.0
    d_inv: float = 0.0
    lambda_m: float = 7.0
    itype: int = 1
    fct_id_ab: int = 0
    nu_val: float = 0.0
    fscale_ab: float = 1.0

    # Thermal Neo-Hookean parameters
    fct_id_sm: int = 0
    fct_id_bm: int = 0
    fscale_sm: float = 1.0
    fscale_bm: float = 1.0

    # Equilibrium creep parameters
    a_pl: float = 1.0
    sigma_pl: float = 1.0
    f_pl: float = 1.0
    epsilon_f: float = 1.0
    n_pl: int = 1

    # Secondary networks
    networks: List[SecondaryNetworkParams] = field(default_factory=list)

    iform: int = 1
    sb: float = 0.0
    g0: float = 0.0
    rbulk: float = 0.0
    nu: float = 0.495
    e: float = 0.0
    sound_speed: float = 0.0


# Arruda-Boyce series coefficients (hm_read_mat100.F:364-368, sigaboyce.F)
_AB_C1 = 0.5
_AB_C2 = 1.0 / 20.0
_AB_C3 = 11.0 / 1050.0
_AB_C4 = 19.0 / 7000.0
_AB_C5 = 519.0 / 673750.0


def build_law100(id_or_dict: Any = 1, **kwargs: Any) -> MultiNetworkParams:
    """Build a MultiNetworkParams instance from dict, object, or keyword arguments.

    Matches starter/source/materials/mat/mat100/hm_read_mat100.F initialization.
    """
    rec: Dict[str, Any] = {}
    if isinstance(id_or_dict, dict):
        rec.update(id_or_dict)
    elif hasattr(id_or_dict, "params") and isinstance(id_or_dict.params, dict):
        rec.update(id_or_dict.params)
        for attr in ("id", "title", "rho0", "rho", "flag_he", "flag_cr", "n_net"):
            if hasattr(id_or_dict, attr):
                rec[attr] = getattr(id_or_dict, attr)
    elif hasattr(id_or_dict, "__dict__"):
        rec.update(id_or_dict.__dict__)
    elif isinstance(id_or_dict, (int, str)):
        try:
            rec["id"] = int(id_or_dict)
        except ValueError:
            pass

    rec.update(kwargs)

    def _get(keys: tuple[str, ...], default: Any = 0.0) -> Any:
        for k in keys:
            if k in rec and rec[k] is not None:
                return rec[k]
            k_low = k.lower()
            if k_low in rec and rec[k_low] is not None:
                return rec[k_low]
            k_up = k.upper()
            if k_up in rec and rec[k_up] is not None:
                return rec[k_up]
        return default

    mat_id = int(_get(("id", "mat_id"), 1))
    title = str(_get(("title", "name", "mat_title"), ""))
    rho0 = float(_get(("rho0", "rho", "mat_rho", "rhor"), 1.0))
    ref_rho = float(_get(("ref_rho", "rhor"), 0.0))

    flag_he = int(_get(("flag_he", "mat_flag_he"), 1))
    flag_cr = int(_get(("flag_cr", "mat_flag_cr", "flag_pl"), 0))
    n_net = int(_get(("n_net", "mat_n_net", "n_network"), 0))

    # Polynomial / hyperelastic parameters
    c10 = float(_get(("c10", "c_10", "mat_c_10"), 0.0))
    c01 = float(_get(("c01", "c_01", "mat_c_01"), 0.0))
    c20 = float(_get(("c20", "c_20", "mat_c_20"), 0.0))
    c11 = float(_get(("c11", "c_11", "mat_c_11"), 0.0))
    c02 = float(_get(("c02", "c_02", "mat_c_02"), 0.0))
    c30 = float(_get(("c30", "c_30", "mat_c_30"), 0.0))
    c21 = float(_get(("c21", "c_21", "mat_c_21"), 0.0))
    c12 = float(_get(("c12", "c_12", "mat_c_12"), 0.0))
    c03 = float(_get(("c03", "c_03", "mat_c_03"), 0.0))

    d1_raw = float(_get(("d1", "d_1", "mat_d_1"), 0.0))
    d2_raw = float(_get(("d2", "d_2", "mat_d_2"), 0.0))
    d3_raw = float(_get(("d3", "d_3", "mat_d_3"), 0.0))

    # Arruda-Boyce parameters
    mu = float(_get(("mu", "mue1", "mat_mue1"), 0.0))
    d_raw = float(_get(("d", "mat_d"), 0.0))
    lambda_m = float(_get(("lambda_m", "lambda", "lm"), 7.0))
    if lambda_m == 0.0:
        lambda_m = 7.0
    itype = int(_get(("itype",), 1))
    fct_id_ab = int(_get(("fct_id_ab", "mat_fct_id_ab"), 0))
    nu_val = float(_get(("nu", "mat_nu", "nu_val"), 0.0))
    fscale_ab = float(_get(("fscale_ab",), 1.0))

    # Thermal Neo-Hookean
    fct_id_sm = int(_get(("fct_id_sm", "mat_fct_id_sm"), 0))
    fct_id_bm = int(_get(("fct_id_bm", "mat_fct_id_bm"), 0))
    fscale_sm = float(_get(("fscale_sm", "mat_fscale_sm"), 1.0))
    fscale_bm = float(_get(("fscale_bm", "mat_fscale_bm"), 1.0))

    # Equilibrium creep parameters
    a_pl = float(_get(("a_pl", "mat_a_pl", "facpl"), 1.0))
    sigma_pl = float(_get(("sigma_pl", "mat_sigma_pl", "tauy0", "tauy"), 1.0))
    f_pl = float(_get(("f_pl", "mat_f_pl", "ff"), 1.0))
    epsilon_f = float(_get(("epsilon_f", "epsilon_pl", "mat_epsilon_f", "epshat"), 1.0))
    n_pl = int(_get(("n_pl", "mat_n_pl", "exppl"), 1))

    # Secondary networks parsing
    networks: List[SecondaryNetworkParams] = []
    networks_in = _get(("networks", "secondary_networks", "mat_arr_networks"), None)

    if networks_in is not None and isinstance(networks_in, list):
        for idx, net_item in enumerate(networks_in):
            net_dict = net_item if isinstance(net_item, dict) else (
                net_item.__dict__ if hasattr(net_item, "__dict__") else {}
            )
            net_id = int(net_dict.get("network_id", idx + 1))
            flag_visc = int(net_dict.get("flag_visc", 1))
            stiffness = float(net_dict.get("stiffness", net_dict.get("stiffn", 1.0)))
            a = float(net_dict.get("a", net_dict.get("a1", net_dict.get("a2", net_dict.get("a3", 0.0)))))
            expc = float(net_dict.get("expc", net_dict.get("c", -0.7)))
            expm = float(net_dict.get("expm", net_dict.get("m", net_dict.get("m3", 1.0))))
            ksi = float(net_dict.get("ksi", 0.01))
            tauref = float(net_dict.get("tauref", net_dict.get("tau_ref", 1.0)))
            b0 = float(net_dict.get("b0", net_dict.get("b", 0.0)))
            expn = float(net_dict.get("expn", net_dict.get("n2", net_dict.get("n3", 1.0))))
            networks.append(SecondaryNetworkParams(
                network_id=net_id, flag_visc=flag_visc, stiffness=stiffness,
                a=a, expc=expc, expm=expm, ksi=ksi, tauref=tauref, b0=b0, expn=expn,
            ))
    elif n_net > 0:
        # Check for array attributes (e.g. MAT_ARR_stiffness, MAT_ARR_Flag_visc, etc.)
        stiff_arr = _get(("mat_arr_stiffness", "stiffness_arr"), [])
        flag_arr = _get(("mat_arr_flag_visc", "flag_visc_arr"), [])
        a1_arr = _get(("mat_arr_a1", "mat_arr_a2", "mat_arr_a3", "a_arr"), [])
        c_arr = _get(("mat_arr_c", "c_arr"), [])
        m_arr = _get(("mat_arr_m", "mat_arr_m3", "m_arr"), [])
        ksi_arr = _get(("mat_arr_ksi", "ksi_arr"), [])
        tauref_arr = _get(("tau_ref", "tauref_arr"), [])
        b_arr = _get(("mat_arr_b", "b_arr"), [])
        n_arr = _get(("mat_arr_n2", "mat_arr_n3", "n_arr"), [])

        for i in range(n_net):
            stiff_i = float(stiff_arr[i]) if i < len(stiff_arr) else float(_get((f"stiffness_{i+1}", f"sb_{i+1}"), 1.0))
            fvisc_i = int(flag_arr[i]) if i < len(flag_arr) else int(_get((f"flag_visc_{i+1}",), 1))
            a_i = float(a1_arr[i]) if i < len(a1_arr) else float(_get((f"a_{i+1}", f"a1_{i+1}", f"a2_{i+1}", f"a3_{i+1}"), 0.0))
            c_i = float(c_arr[i]) if i < len(c_arr) else float(_get((f"c_{i+1}", f"expc_{i+1}"), -0.7))
            m_i = float(m_arr[i]) if i < len(m_arr) else float(_get((f"m_{i+1}", f"expm_{i+1}"), 1.0))
            ksi_i = float(ksi_arr[i]) if i < len(ksi_arr) else float(_get((f"ksi_{i+1}",), 0.01))
            tref_i = float(tauref_arr[i]) if i < len(tauref_arr) else float(_get((f"tauref_{i+1}", f"tau_ref_{i+1}"), 1.0))
            b0_i = float(b_arr[i]) if i < len(b_arr) else float(_get((f"b0_{i+1}", f"b_{i+1}"), 0.0))
            expn_i = float(n_arr[i]) if i < len(n_arr) else float(_get((f"expn_{i+1}", f"n2_{i+1}", f"n3_{i+1}"), 1.0))
            networks.append(SecondaryNetworkParams(
                network_id=i + 1, flag_visc=fvisc_i, stiffness=stiff_i,
                a=a_i, expc=c_i, expm=m_i, ksi=ksi_i, tauref=tref_i, b0=b0_i, expn=expn_i,
            ))
    elif "sb" in rec or "a" in rec or "expc" in rec or "expm" in rec:
        # Fallback single secondary network from flat kwargs (legacy / LAW95 format compatibility)
        sb_val = float(_get(("sb", "stiffness"), 0.0))
        a_val = float(_get(("a", "a1"), 0.0))
        c_val = float(_get(("expc", "c"), -0.7))
        m_val = float(_get(("expm", "m"), 1.0))
        ksi_val = float(_get(("ksi",), 0.01))
        tref_val = float(_get(("tauref", "tau_ref"), 1.0))
        if sb_val > 0.0 or a_val > 0.0:
            networks.append(SecondaryNetworkParams(
                network_id=1, flag_visc=1, stiffness=sb_val if sb_val > 0.0 else 1.0,
                a=a_val, expc=c_val, expm=m_val, ksi=ksi_val, tauref=tref_val,
            ))

    n_net_actual = len(networks)

    # Invert D parameters matching hm_read_mat100.F:345-350
    d1 = (1.0 / d1_raw) if d1_raw != 0.0 else 0.0
    d2 = (1.0 / d2_raw) if d2_raw != 0.0 else 0.0
    d3 = (1.0 / d3_raw) if d3_raw != 0.0 else 0.0
    d_inv = (1.0 / d_raw) if d_raw != 0.0 else (1.0 / 1e-20)

    sb = sum(net.stiffness for net in networks)

    # Calculate small-strain shear G_0, bulk K, Poisson's ratio nu, Young's E
    # hm_read_mat100.F:343-382
    iform = int(_get(("iform",), 1))
    if flag_he in (1, 3, 4, 5):
        g0 = 2.0 * (c10 + c01) * (1.0 + sb)
        if d1 != 0.0:
            rbulk = 2.0 * d1 * (1.0 + sb)
            nu = (3.0 * rbulk - 2.0 * g0) / (2.0 * (3.0 * rbulk + g0))
            e = 9.0 * rbulk * g0 / (3.0 * rbulk + g0)
        else:
            d2 = 0.0
            d3 = 0.0
            nu = nu_val if (0.0 < nu_val < 0.5) else 0.495
            rbulk = (2.0 / 3.0) * g0 * (1.0 + nu) / max(_EM30, (1.0 - 2.0 * nu))
            d1 = rbulk / 2.0
            e = 2.0 * g0 * (1.0 + nu)
    elif flag_he == 2:  # Arruda-Boyce
        beta = 1.0 / (lambda_m ** 2)
        arruda_poly = (
            1.0
            + (3.0 / 5.0) * beta
            + (99.0 / 175.0) * (beta ** 2)
            + (513.0 / 875.0) * (beta ** 3)
            + (42039.0 / 67375.0) * (beta ** 4)
        )
        g0 = mu * arruda_poly * (1.0 + sb)
        rbulk = 2.0 * (1.0 + sb) * d_inv
        e = 9.0 * rbulk * g0 / max(_EM30, 3.0 * rbulk + g0)
        if fct_id_ab == 0:
            nu = (3.0 * rbulk - 2.0 * g0) / max(_EM30, 2.0 * (3.0 * rbulk + g0))
        else:
            nu = nu_val if (0.0 < nu_val < 0.5) else 0.495
    elif flag_he == 13:  # Thermal Neo-Hookean
        g0 = fscale_sm * (1.0 + sb)
        rbulk = fscale_bm * (1.0 + sb)
        nu = (3.0 * rbulk - 2.0 * g0) / max(_EM30, 2.0 * (3.0 * rbulk + g0))
        e = 9.0 * rbulk * g0 / max(_EM30, 3.0 * rbulk + g0)
    else:
        g0 = 2.0 * (c10 + c01) * (1.0 + sb)
        rbulk = 2.0 * d1 * (1.0 + sb) if d1 != 0.0 else g0 * 100.0
        nu = 0.495
        e = 2.0 * g0 * (1.0 + nu)

    sound_speed = float(np.sqrt(max(0.0, (rbulk + (4.0 / 3.0) * g0) / max(_EM20, rho0))))

    return MultiNetworkParams(
        id=mat_id, title=title, rho0=rho0, ref_rho=ref_rho,
        n_net=n_net_actual, flag_he=flag_he, flag_cr=flag_cr,
        c10=c10, c01=c01, c20=c20, c11=c11, c02=c02,
        c30=c30, c21=c21, c12=c12, c03=c03,
        d1_raw=d1_raw, d2_raw=d2_raw, d3_raw=d3_raw,
        d1=d1, d2=d2, d3=d3,
        mu=mu, d_raw=d_raw, d_inv=d_inv, lambda_m=lambda_m,
        itype=itype, fct_id_ab=fct_id_ab, nu_val=nu_val, fscale_ab=fscale_ab,
        fct_id_sm=fct_id_sm, fct_id_bm=fct_id_bm,
        fscale_sm=fscale_sm, fscale_bm=fscale_bm,
        a_pl=a_pl, sigma_pl=sigma_pl, f_pl=f_pl,
        epsilon_f=epsilon_f, n_pl=n_pl,
        networks=networks, iform=iform,
        sb=sb, g0=g0, rbulk=rbulk, nu=nu, e=e,
        sound_speed=sound_speed,
    )


# ============================================================================
# Kinematics and Viscoelastic Flow Kernels
# ============================================================================

def calc_mat_b(f: np.ndarray, fp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute elastic left Cauchy-Green tensor b_e and F_e = F * F_p^(-1).

    Mirrors engine/source/materials/mat/mat100/calcmatb.F.
    """
    if fp is None:
        fp = np.eye(3, dtype=f.dtype if hasattr(f, "dtype") else np.float64)
    inv_fp = np.linalg.inv(fp)
    fe = f @ inv_fp
    matb = fe @ fe.T
    return matb, fe


def visc_bb(
    fp: np.ndarray,
    tbnorm: float,
    a1: float,
    expc: float,
    expm: float,
    ksi: float,
    tauref: float,
) -> float:
    """Bergstrom-Boyce nonlinear creep rate model.

    Mirrors engine/source/materials/mat/mat100/viscbb.F.
    """
    ip1 = fp[0, 0] ** 2 + fp[1, 1] ** 2 + fp[2, 2] ** 2
    lpchain = np.sqrt(max(0.0, ip1 / 3.0))
    temp = max(_EM20, lpchain - 1.0 + ksi)
    stress_ratio = (tbnorm ** expm) / max(_EM20, tauref ** expm)
    dgamma = a1 * np.exp(expc * np.log(temp)) * stress_ratio
    return float(dgamma)


def visc_sinh(
    tbnorm: float,
    a1: float,
    b0: float,
    expn: float,
) -> float:
    """Hyperbolic sine nonlinear creep rate model.

    Mirrors engine/source/materials/mat/mat100/viscsinh.F.
    """
    val = np.sinh(b0 * tbnorm)
    val_safe = max(_EM20, val) if val > 0.0 else _EM20
    dgamma = a1 * (val_safe ** expn)
    return float(dgamma)


def visc_power(
    tbnorm: float,
    a1: float,
    expm: float,
    expn: float,
    gammaold: float,
) -> float:
    """Power law creep rate model.

    Mirrors engine/source/materials/mat/mat100/viscpower.F.
    """
    g_old = max(_EM20, gammaold)
    t_norm = max(_EM20, tbnorm)
    temp1 = (expm + 1.0) * g_old
    temp2 = np.exp(expn * np.log(t_norm))
    temp3 = np.exp(expm * np.log(temp1))
    dgamma = a1 * np.exp((1.0 / (1.0 + expm)) * np.log(temp2 * temp3))
    return float(dgamma)


def poly_stress(
    b: np.ndarray,
    c10: float,
    c01: float,
    c20: float,
    c11: float,
    c02: float,
    c30: float,
    c21: float,
    c12: float,
    c03: float,
    d1: float,
    d2: float,
    d3: float,
    rbulk: float = 0.0,
    iform: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Cauchy stress and directional stiffness for polynomial hyperelastic model.

    Mirrors engine/source/materials/mat/mat100/sigpoly.F (polystress2).
    """
    b2 = b @ b

    det_b = (
        b[0, 0] * (b[1, 1] * b[2, 2] - b[1, 2] * b[2, 1])
        - b[0, 1] * (b[1, 0] * b[2, 2] - b[1, 2] * b[2, 0])
        + b[0, 2] * (b[1, 0] * b[2, 1] - b[1, 1] * b[2, 0])
    )
    jdet = np.sqrt(max(_EM20, det_b))

    i1 = b[0, 0] + b[1, 1] + b[2, 2]
    tr_b2 = i1 ** 2
    tr_b_sq = b2[0, 0] + b2[1, 1] + b2[2, 2]
    i2 = 0.5 * (tr_b2 - tr_b_sq)

    jthird = np.exp((-1.0 / 3.0) * np.log(jdet)) if jdet > 0.0 else 0.0
    j2third = jthird ** 2
    j4third = jthird ** 4

    bi1 = i1 * j2third
    bi2 = i2 * j4third

    di1 = bi1 - 3.0
    di2 = bi2 - 3.0

    dphi_di1 = (
        c10
        + 2.0 * c20 * di1
        + 3.0 * c30 * (di1 ** 2)
        + c11 * di2
        + c12 * (di2 ** 2)
        + 2.0 * c21 * di1 * di2
    )
    dphi_di2 = (
        c01
        + 2.0 * c02 * di2
        + 3.0 * c03 * (di2 ** 2)
        + c11 * di1
        + c21 * (di1 ** 2)
        + 2.0 * c12 * di1 * di2
    )

    inv2j = 2.0 / max(_EM20, jdet)

    j_minus_1 = jdet - 1.0
    if iform == 1:
        dphi_dj = 2.0 * d1 * j_minus_1 + 4.0 * d2 * (j_minus_1 ** 3) + 6.0 * d3 * (j_minus_1 ** 5)
        dphi2_dj = 2.0 * d1 + 12.0 * d2 * (j_minus_1 ** 2) + 30.0 * d3 * (j_minus_1 ** 4)
    else:
        dphi_dj = rbulk * (1.0 - 1.0 / jdet)
        dphi2_dj = rbulk / (jdet ** 2)

    aa = (dphi_di1 + dphi_di2 * bi1) * inv2j * j2third
    bb = dphi_di2 * inv2j * j4third
    cc = (1.0 / 3.0) * inv2j * (bi1 * dphi_di1 + 2.0 * bi2 * dphi_di2)

    sig = aa * b - bb * b2
    sig[0, 0] += -cc + dphi_dj
    sig[1, 1] += -cc + dphi_dj
    sig[2, 2] += -cc + dphi_dj

    # Directional stiffness
    dphi2_di1 = 2.0 * (c20 + 3.0 * c30 * di1 + c21 * di2)
    dphi2_di2 = 2.0 * (c02 + 3.0 * c03 * di2 + c12 * di1)

    lam_b = np.array([b[0, 0], b[1, 1], b[2, 2]], dtype=np.float64) * j2third
    lam_b_safe = np.maximum(lam_b, 1e-12)
    lam_b_inv = 1.0 / lam_b_safe

    bi1_3 = bi1 / 3.0
    bi2_3 = bi2 / 3.0

    term1 = (2.0 / 3.0) * dphi_di1 * (lam_b + bi1_3) + dphi2_di1 * (lam_b - bi1_3)
    term2 = (2.0 / 3.0) * dphi_di2 * (lam_b_inv + bi2_3) + dphi2_di2 * (lam_b_inv - bi2_3)
    cii = 2.0 * (term1 + term2) + dphi2_dj

    return sig, cii


def arruda_boyce_stress(
    b: np.ndarray,
    mu: float,
    lambda_m: float,
    d_inv: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Cauchy stress for Arruda-Boyce 8-chain hyperelastic model.

    Mirrors engine/source/materials/mat/mat100/sigaboyce.F.
    """
    det_b = (
        b[0, 0] * (b[1, 1] * b[2, 2] - b[1, 2] * b[2, 1])
        - b[0, 1] * (b[1, 0] * b[2, 2] - b[1, 2] * b[2, 0])
        + b[0, 2] * (b[1, 0] * b[2, 1] - b[1, 1] * b[2, 0])
    )
    jdet = np.sqrt(max(_EM20, det_b))

    i1 = b[0, 0] + b[1, 1] + b[2, 2]
    j2third = np.exp((-2.0 / 3.0) * np.log(jdet)) if jdet > 0.0 else 0.0
    bi1 = i1 * j2third

    beta = 1.0 / (lambda_m ** 2)

    # dPhi/dI1 (sigaboyce.F:105-108)
    dphi_di1 = (
        2.0
        * mu
        * (
            _AB_C1
            + 2.0 * _AB_C2 * beta * bi1
            + 3.0 * _AB_C3 * ((beta * bi1) ** 2)
            + 4.0 * _AB_C4 * ((beta * bi1) ** 3)
            + 5.0 * _AB_C5 * ((beta * bi1) ** 4)
        )
        / max(_EM20, jdet)
    )

    # dPhi/dJ (sigaboyce.F:110)
    dphi_dj = d_inv * (jdet - 1.0 / max(_EM20, jdet))

    # Cauchy stress (sigaboyce.F:113-125)
    sig = np.zeros((3, 3), dtype=np.float64)
    sig[0, 0] = dphi_di1 * (b[0, 0] - (1.0 / 3.0) * bi1) + dphi_dj
    sig[1, 1] = dphi_di1 * (b[1, 1] - (1.0 / 3.0) * bi1) + dphi_dj
    sig[2, 2] = dphi_di1 * (b[2, 2] - (1.0 / 3.0) * bi1) + dphi_dj
    sig[0, 1] = dphi_di1 * b[0, 1]
    sig[1, 2] = dphi_di1 * b[1, 2]
    sig[2, 0] = dphi_di1 * b[2, 0]
    sig[1, 0] = sig[0, 1]
    sig[2, 1] = sig[1, 2]
    sig[0, 2] = sig[2, 0]

    cii = np.full(3, 2.0 * mu * (1.0 + beta) + d_inv)
    return sig, cii


def neo_hook_t_stress(
    b: np.ndarray,
    mu: float,
    d_bulk: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Cauchy stress for thermal / basic Neo-Hookean model.

    Mirrors engine/source/materials/mat/mat100/neo_hook_t.F.
    """
    det_b = (
        b[0, 0] * (b[1, 1] * b[2, 2] - b[1, 2] * b[2, 1])
        - b[0, 1] * (b[1, 0] * b[2, 2] - b[1, 2] * b[2, 0])
        + b[0, 2] * (b[1, 0] * b[2, 1] - b[1, 1] * b[2, 0])
    )
    jdet = np.sqrt(max(_EM20, det_b))
    i1 = b[0, 0] + b[1, 1] + b[2, 2]

    dphi_dj = d_bulk * (jdet - 1.0)
    aa = mu / max(_EM20, jdet)

    sig = np.zeros((3, 3), dtype=np.float64)
    sig[0, 0] = aa * (b[0, 0] - (1.0 / 3.0) * i1) + dphi_dj
    sig[1, 1] = aa * (b[1, 1] - (1.0 / 3.0) * i1) + dphi_dj
    sig[2, 2] = aa * (b[2, 2] - (1.0 / 3.0) * i1) + dphi_dj
    sig[0, 1] = aa * b[0, 1]
    sig[1, 2] = aa * b[1, 2]
    sig[2, 0] = aa * b[2, 0]
    sig[1, 0] = sig[0, 1]
    sig[2, 1] = sig[1, 2]
    sig[0, 2] = sig[2, 0]

    cii = np.full(3, aa + d_bulk)
    return sig, cii


def compute_he_stress(
    b: np.ndarray,
    mat: MultiNetworkParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch hyperelastic stress calculation based on mat.flag_he."""
    fhe = mat.flag_he
    if fhe in (3, 4, 5):
        fhe = 1  # OpenRadioss reduction (sigeps100.F90:189)

    if fhe == 1:
        return poly_stress(
            b, mat.c10, mat.c01, mat.c20, mat.c11, mat.c02,
            mat.c30, mat.c21, mat.c12, mat.c03,
            mat.d1, mat.d2, mat.d3, rbulk=mat.rbulk, iform=mat.iform,
        )
    elif fhe == 2:
        return arruda_boyce_stress(b, mat.mu, mat.lambda_m, mat.d_inv)
    elif fhe == 13:
        return neo_hook_t_stress(b, mat.fscale_sm, mat.fscale_bm)
    else:
        return poly_stress(
            b, mat.c10, mat.c01, mat.c20, mat.c11, mat.c02,
            mat.c30, mat.c21, mat.c12, mat.c03,
            mat.d1, mat.d2, mat.d3, rbulk=mat.rbulk, iform=mat.iform,
        )


# ============================================================================
# 3D Continuum Solid Kernel: solid_update
# ============================================================================

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
    """3D solid continuum Multi-Network Visco-Hyperelastic stress update.

    Fortran reference: engine/source/materials/mat/mat100/sigeps100.F90.

    Parameters
    ----------
    mat : MultiNetworkParams or Material or dict
        Material definition.
    sig : ndarray of shape (6,) or (n, 6)
        Current Cauchy stress (xx, yy, zz, xy, yz, zx).
    deps : ndarray of shape (6,) or (n, 6)
        Strain increment tensor (engineering shear: 2*eps_ij).
    eps : ndarray of shape (6,) or (n, 6), optional
        Total strain tensor.
    dt : float
        Time step increment.
    extra : dict, optional
        Extra state storage containing:
        - "uvar": history variables array storing F_p_eq (if Flag_Cr=1) and F_p_k for each network.
        - "F": optional deformation gradient array of shape (n, 3, 3) or (3, 3).
        - "rho": optional current density array.
    ismstr : int
        Strain formulation flag (0: log, 1: eng).
    epsp : ndarray, optional
        Accumulated plastic / creep strain.

    Returns
    -------
    sig_out, epsp_out, sound_speed (or dict / tuple depending on call mode).
    """
    return_constitutive_dict = False
    if "state" in kwargs and extra is None:
        extra = kwargs["state"]
    if "F" in kwargs:
        if extra is None:
            extra = {}
        extra["F"] = kwargs["F"]

    if sig is not None and isinstance(sig, np.ndarray) and sig.ndim == 2 and sig.shape == (3, 3):
        if extra is None:
            extra = {}
        extra["F"] = sig
        sig = np.zeros(6, dtype=np.float64)

    if isinstance(mat, np.ndarray) and (sig is None or not isinstance(sig, np.ndarray)):
        # Direct constitutive call: solid_update(strain_tensor, c10=..., ...)
        eps_in = mat
        params_kw = kwargs.copy()
        if isinstance(sig, dict):
            params_kw.update(sig)
        mat = build_law100(**params_kw)
        sig_arr = np.zeros_like(eps_in) if eps_in.ndim > 1 else np.zeros((1, 6))
        deps_arr = np.zeros_like(sig_arr)
        eps_arr = eps_in[np.newaxis, :] if eps_in.ndim == 1 else eps_in
        epsp_arr = np.zeros(len(eps_arr))
        is_1d = (eps_in.ndim == 1)
        return_constitutive_dict = True
        if "rho" in kwargs and extra is None:
            extra = {"rho": kwargs["rho"]}
    else:
        if not isinstance(mat, MultiNetworkParams):
            mat = build_law100(mat, **kwargs)
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
    epsp_out = epsp_arr.copy()

    # Total strain tensor
    tot_eps = eps_arr + deps_arr if deps is not None else eps_arr.copy()

    stiff0 = (4.0 / 3.0) * mat.g0 + mat.rbulk

    # Density handling
    if extra is not None and "rho" in extra and extra["rho"] is not None:
        rho_val = np.asarray(extra["rho"])
        rho_arr = np.full(nel, float(rho_val)) if rho_val.ndim == 0 else rho_val
    else:
        rho_arr = np.full(nel, mat.rho0)

    # History sizing:
    # If flag_cr == 1: 13 variables for equilibrium network (tauy, dummy, dummy, pla, Fp(9))
    # Each network k:
    #   if flag_visc == 1 (BB): 11 vars (Fp(9), dgamma, tbnorm)
    #   if flag_visc == 2 (Sinh): 11 vars (Fp(9), dgamma, tbnorm)
    #   if flag_visc == 3 (Power): 12 vars (Fp(9), gammaold, dgamma, tbnorm)
    eq_nvars = 13 if mat.flag_cr == 1 else 0
    net_offsets: List[int] = []
    total_uvar = eq_nvars
    for net in mat.networks:
        net_offsets.append(total_uvar)
        shift = 12 if net.flag_visc == 3 else 11
        total_uvar += shift

    if total_uvar == 0:
        total_uvar = 1  # allocate at least dummy array if no internal states

    uvar_hist = None
    if extra is not None:
        for uvar_key in ("uvar", "uvar100", "mat_uvar", "uvar95"):
            if uvar_key in extra and extra[uvar_key] is not None:
                uvar_hist = extra[uvar_key]
                break

    if uvar_hist is None or uvar_hist.shape[-1] < total_uvar:
        uvar_hist = np.zeros((nel, total_uvar), dtype=np.float64)
        # Initialize equilibrium network if active
        if mat.flag_cr == 1:
            uvar_hist[:, 0] = mat.sigma_pl  # tauy0
            uvar_hist[:, 4] = 1.0  # Fp_11
            uvar_hist[:, 5] = 1.0  # Fp_22
            uvar_hist[:, 6] = 1.0  # Fp_33
        # Initialize secondary networks
        for k, net in enumerate(mat.networks):
            off = net_offsets[k]
            uvar_hist[:, off + 0] = 1.0  # Fp_11
            uvar_hist[:, off + 1] = 1.0  # Fp_22
            uvar_hist[:, off + 2] = 1.0  # Fp_33
            if net.flag_visc == 3:
                uvar_hist[:, off + 9] = _EM20  # gammaold
    else:
        if uvar_hist.ndim == 1:
            uvar_hist = uvar_hist[np.newaxis, :]

    # Element-by-element integration loop
    for i in range(nel):
        e_vec = tot_eps[i]
        # Deformation gradient F
        if extra is not None and "F" in extra and extra["F"] is not None:
            f_in = np.asarray(extra["F"])
            f_mat = f_in[i] if f_in.ndim == 3 else f_in
        else:
            # Reconstruct from total strain: eps = 0.5 * (F + F^T) - I (or small strain approximation)
            f_mat = np.array([
                [1.0 + e_vec[0], 0.5 * e_vec[3], 0.5 * e_vec[5]],
                [0.5 * e_vec[3], 1.0 + e_vec[1], 0.5 * e_vec[4]],
                [0.5 * e_vec[5], 0.5 * e_vec[4], 1.0 + e_vec[2]],
            ], dtype=np.float64)

        # -------------------------------------------------------------
        # 1. Equilibrium Network (A) Stress Update
        # -------------------------------------------------------------
        if mat.flag_cr == 1:
            tauy = uvar_hist[i, 0]
            pla = uvar_hist[i, 3]
            fpeqo = np.array([
                [uvar_hist[i, 4], uvar_hist[i, 7], uvar_hist[i, 9]],
                [uvar_hist[i, 10], uvar_hist[i, 5], uvar_hist[i, 8]],
                [uvar_hist[i, 11], uvar_hist[i, 12], uvar_hist[i, 6]],
            ], dtype=np.float64)

            # b_trial = Fe * Fe^T with Fe = F * fpeqo^(-1)
            b_a, fe_a = calc_mat_b(f_mat, fpeqo)
            sig_a, _ = compute_he_stress(b_a, mat)

            # Deviatoric stress and norm
            tr_a = np.trace(sig_a) / 3.0
            s_a = sig_a - tr_a * np.eye(3)
            tanorm = np.sqrt(max(_EM20, np.sum(s_a ** 2)))

            # Creep evolution
            if dt > 0.0 and tauy > 0.0:
                temp1 = tanorm / tauy
                plap = mat.a_pl * (temp1 ** mat.n_pl)
                dpla = plap * dt
                pla += dpla
                factor = plap * dt / tanorm
                s_mat = np.eye(3) + factor * s_a
                fpeq = s_mat @ fpeqo
                # Recompute stress with updated fpeq
                b_a, _ = calc_mat_b(f_mat, fpeq)
                sig_a, _ = compute_he_stress(b_a, mat)
            else:
                fpeq = fpeqo

            # Update yield resistance softening (sigeps100.F90:560)
            tauy = mat.sigma_pl * (mat.f_pl + (1.0 - mat.f_pl) * np.exp(-pla / max(_EM20, mat.epsilon_f)))

            # Store history
            uvar_hist[i, 0] = tauy
            uvar_hist[i, 3] = pla
            uvar_hist[i, 4] = fpeq[0, 0]
            uvar_hist[i, 5] = fpeq[1, 1]
            uvar_hist[i, 6] = fpeq[2, 2]
            uvar_hist[i, 7] = fpeq[0, 1]
            uvar_hist[i, 8] = fpeq[1, 2]
            uvar_hist[i, 9] = fpeq[0, 2]
            uvar_hist[i, 10] = fpeq[1, 0]
            uvar_hist[i, 11] = fpeq[2, 0]
            uvar_hist[i, 12] = fpeq[2, 1]
            epsp_out[i] = pla
        else:
            # Pure hyperelastic equilibrium network: b = F * F^T
            b_a = f_mat @ f_mat.T
            sig_a, _ = compute_he_stress(b_a, mat)

        total_sig = sig_a.copy()

        # -------------------------------------------------------------
        # 2. Secondary Networks (B_k) Stress Update
        # -------------------------------------------------------------
        for k, net in enumerate(mat.networks):
            off = net_offsets[k]
            fpo = np.array([
                [uvar_hist[i, off + 0], uvar_hist[i, off + 3], uvar_hist[i, off + 5]],
                [uvar_hist[i, off + 6], uvar_hist[i, off + 1], uvar_hist[i, off + 4]],
                [uvar_hist[i, off + 7], uvar_hist[i, off + 8], uvar_hist[i, off + 2]],
            ], dtype=np.float64)

            # Fe = F * Fp^(-1), b_e = Fe * Fe^T
            matb, fe = calc_mat_b(f_mat, fpo)

            # Trial Cauchy stress scaled by stiffness S_k
            sig_trial, _ = compute_he_stress(matb, mat)
            sig_b = net.stiffness * sig_trial

            # Deviatoric stress
            tr_b = np.trace(sig_b) / 3.0
            sb_mat = sig_b - tr_b * np.eye(3)
            # Frobenius norm: sqrt(sum(s_ij^2))
            tbnorm = np.sqrt(max(_EM20, np.sum(sb_mat ** 2)))

            # Flow rule rate factor
            a1_dt = net.a * dt
            if net.flag_visc == 1:
                dgamma = visc_bb(fpo, tbnorm, a1_dt, net.expc, net.expm, net.ksi, net.tauref)
                uvar_hist[i, off + 9] = dgamma
                uvar_hist[i, off + 10] = tbnorm
            elif net.flag_visc == 2:
                dgamma = visc_sinh(tbnorm, a1_dt, net.b0, net.expn)
                uvar_hist[i, off + 9] = dgamma
                uvar_hist[i, off + 10] = tbnorm
            elif net.flag_visc == 3:
                gammaold = uvar_hist[i, off + 9]
                dgamma = visc_power(tbnorm, a1_dt, net.expm, net.expn, gammaold)
                uvar_hist[i, off + 9] = gammaold + dgamma
                uvar_hist[i, off + 10] = dgamma
                uvar_hist[i, off + 11] = tbnorm
            else:
                dgamma = 0.0

            # Kinematic evolution of F_p
            if dt > 0.0 and dgamma > 0.0:
                factor = dgamma / tbnorm
                lb = factor * sb_mat
                inv_fe = np.linalg.inv(fe)
                dfp = inv_fe @ lb @ fe
                sn = np.eye(3) + dfp
                fp_new = sn @ fpo
                # Recompute stress with fp_new
                matb_new, _ = calc_mat_b(f_mat, fp_new)
                sig_new, _ = compute_he_stress(matb_new, mat)
                sig_b = net.stiffness * sig_new
            else:
                fp_new = fpo

            # Store updated F_p
            uvar_hist[i, off + 0] = fp_new[0, 0]
            uvar_hist[i, off + 1] = fp_new[1, 1]
            uvar_hist[i, off + 2] = fp_new[2, 2]
            uvar_hist[i, off + 3] = fp_new[0, 1]
            uvar_hist[i, off + 4] = fp_new[1, 2]
            uvar_hist[i, off + 5] = fp_new[0, 2]
            uvar_hist[i, off + 6] = fp_new[1, 0]
            uvar_hist[i, off + 7] = fp_new[2, 0]
            uvar_hist[i, off + 8] = fp_new[2, 1]

            total_sig += sig_b

        # Convert 3x3 Cauchy stress to Voigt (xx, yy, zz, xy, yz, zx)
        sig_out[i, 0] = total_sig[0, 0]
        sig_out[i, 1] = total_sig[1, 1]
        sig_out[i, 2] = total_sig[2, 2]
        sig_out[i, 3] = total_sig[0, 1]
        sig_out[i, 4] = total_sig[1, 2]
        sig_out[i, 5] = total_sig[2, 0]

        # Sound speed calculation (sigeps100.F90:816-829)
        dsig = np.sqrt(
            (sig_out[i, 0] - sig_arr[i, 0]) ** 2
            + (sig_out[i, 1] - sig_arr[i, 1]) ** 2
            + (sig_out[i, 2] - sig_arr[i, 2]) ** 2
            + 2.0 * (
                (sig_out[i, 3] - sig_arr[i, 3]) ** 2
                + (sig_out[i, 4] - sig_arr[i, 4]) ** 2
                + (sig_out[i, 5] - sig_arr[i, 5]) ** 2
            )
        )
        d_eps = deps_arr[i]
        deps_norm = np.sqrt(
            d_eps[0] ** 2 + d_eps[1] ** 2 + d_eps[2] ** 2
            + 2.0 * (d_eps[3] ** 2 + d_eps[4] ** 2 + d_eps[5] ** 2)
        )
        if deps_norm == 0.0:
            stiff_i = stiff0
        else:
            stiff_i = max(stiff0, dsig / max(_EM20, deps_norm))

        sound_sp[i] = np.sqrt(max(0.0, stiff_i / max(_EM20, rho_arr[i])))

    # Persist history back into extra
    if extra is not None:
        extra["uvar"] = uvar_hist[0] if is_1d else uvar_hist
        extra["uvar100"] = extra["uvar"]

    res_sig = sig_out[0] if is_1d else sig_out
    res_epsp = epsp_out[0] if is_1d else epsp_out
    res_sound = sound_sp[0] if is_1d else sound_sp

    if return_constitutive_dict:
        state_dict = {
            "uvar": uvar_hist[0] if is_1d else uvar_hist,
            "epsp": res_epsp,
            "sound_speed": res_sound,
        }
        return res_sig, state_dict, float(np.mean(res_sound))

    if return_tuple:
        return res_sig, res_epsp, res_sound
    return res_sig, res_epsp


def sound_speed(
    mat: Any,
    rho: Optional[float | np.ndarray] = None,
    extra: Optional[dict] = None,
) -> float | np.ndarray:
    """Calculate small-strain solid sound speed for /MAT/LAW100.

    c = sqrt((K + 4/3 * G) / rho0).
    """
    if not isinstance(mat, MultiNetworkParams):
        mat = build_law100(mat)

    g0 = mat.g0
    k = mat.rbulk
    stiff0 = k + (4.0 / 3.0) * g0

    if rho is not None:
        rho_arr = np.asarray(rho, dtype=np.float64)
        return np.sqrt(np.maximum(0.0, stiff0 / np.maximum(_EM20, rho_arr)))
    return float(mat.sound_speed)


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    epsp: float = 0.0,
    dt: float = 0.0,
    extra: Optional[dict] = None,
) -> np.ndarray:
    """Numerical tangent stiffness matrix (6x6) for 3D solid continuum elements."""
    if not isinstance(mat, MultiNetworkParams):
        mat = build_law100(mat)

    c = np.zeros((6, 6), dtype=np.float64)
    g0 = mat.g0
    k = mat.rbulk

    # Lamé parameters
    lambda_lame = k - (2.0 / 3.0) * g0
    c11 = lambda_lame + 2.0 * g0
    c12 = lambda_lame

    # Isotropic ground-state elasticity baseline
    c[0, 0] = c[1, 1] = c[2, 2] = c11
    c[0, 1] = c[1, 0] = c[0, 2] = c[2, 0] = c[1, 2] = c[2, 1] = c12
    c[3, 3] = c[4, 4] = c[5, 5] = g0

    # If stress and strain are provided, perform numerical perturbation
    if deps is not None and np.any(deps != 0.0):
        h = 1e-7
        base_sig, _, _ = solid_update(mat, sig=sig, deps=deps, dt=dt, extra=extra)
        for j in range(6):
            deps_pert = deps.copy()
            deps_pert[j] += h
            sig_pert, _, _ = solid_update(mat, sig=sig, deps=deps_pert, dt=dt, extra=extra)
            c[:, j] = (sig_pert - base_sig) / h

    return c
