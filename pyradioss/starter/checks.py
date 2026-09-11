"""
Starter model checks (cross references, sanity).

Fortran origin: scattered through the Starter (every hm_read does its own
id checks; contchk/prelecdt do global ones). Centralized here: after
finalization, verify every reference between options resolves, so the
Engine can index blindly.
"""

from __future__ import annotations

from ..common.messages import MessageLog
from ..model.model import Model


# multi-material ALE/Euler laws (LAW51, LAW151/MULTIFLUID): their initial
# density is NOT a top-level RHO0 — it is carried by the submaterial
# references (mat_ID_ii) weighted by their volume fractions (Vfrac_ii), so
# the material card's own RHO0 field is legitimately blank/zero.  The
# reference Starter reads the submaterial densities to build the element
# mass and does NOT fatal-error the empty top-level density, so the port's
# null-RHO0 MAT CHECK must exempt the family (M38 / M37-BUG-2).
_MULTIMAT_ALE_LAWS = {51, 151}


# Laws whose RHO0 may legally be zero — the null-density MAT CHECK exempts
# them (M39 / M38-NEW-2).  Two distinct reasons:
#
# * LAW0 (/MAT/VOID) is MASSLESS BY DESIGN.  The upstream reader
#   ``starter/source/materials/mat/mat000/hm_read_mat00.F`` applies NO
#   positivity check to RHO0 and explicitly guards the only place the
#   density is divided by —  ``SDSP = SQRT(YOUNG/MAX(RHOR,EM20))`` — so a
#   void card with RHO0 = 0 is legal and produces a zero sound speed, not
#   an error.  The cfg agrees and is the sharpest evidence: every load-
#   bearing law's cfg CHECK block demands ``MAT_RHO > 0`` (e.g.
#   matl2_plas_johns.cfg) while ``MAT/matl_void0.cfg`` demands only
#   ``MAT_RHO >= 0``.  A void element contributes zero mass and zero
#   stress: dummy contact skins, airbag reference geometry, parts replaced
#   by a rigid body.  The port's fatal here was a false positive on the
#   RD-E-2700 Football decks (BAT_CIR / BAT_SQR), whose /MAT/VOID/12 skin
#   shells carry RHO0 = 0 verbatim;
# * the multimaterial ALE family carries its density on the submaterials
#   (see _MULTIMAT_ALE_LAWS above).
_NULL_RHO0_OK_LAWS = frozenset({0} | _MULTIMAT_ALE_LAWS)


# element family -> material laws its kernels implement (see the
# materials package dispatch; extending a kernel means extending this map).
#
# LAW0 (/MAT/VOID) is legal on EVERY family here, matching the upstream
# compatibility declaration in hm_read_mat00.F, which tags the void law
# SOLID_ISOTROPIC / SHELL_ISOTROPIC / SPRING_MATERIAL / BEAM_ALL / TRUSS /
# SPH — i.e. all of them (M39 / M38-NEW-2, closing the M38 prop-pack OPEN
# item: /PROP/VOID was already made universally family-compatible by
# ``prop_reader.prop_type_ok``, but the MATERIAL half still rejected LAW0
# on /BEAM and /TRUSS, so a void beam passed the property check and failed
# the material one).  The truss/beam kernels honour it: a void material's
# E = G = 0 makes every resultant identically zero (the void semantics)
# and their density divisions are guarded exactly as hm_read_mat00.F
# guards its own — see elements/truss.py and elements/beam_type3.py.
_ALLOWED_LAWS = {
    "bricks": {0, 1, 2, 3, 4, 5, "5", "LAW5", "JWL", 10, "10", "LAW10", "SOIL", "DPRAG", "DPRAG1", 12, "12", "LAW12", "3D_COMP", "COMP_3D", "3PARBI", "RAGAB", 14, "14", "LAW14", "COMPSO", "COMP_SOL", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 24, 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 28, "28", "LAW28", "HONEYCOMB", 33, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 35, 36, 37, "37", "LAW37", "BIPHAS", "BIPHASIC", 38, "38", "LAW38", "VISC_TAB", 40, 42, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 62, 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 70, 81, 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN", 83, 999},
    "tetras": {0, 1, 2, 3, 4, 5, "5", "LAW5", "JWL", 10, "10", "LAW10", "SOIL", "DPRAG", "DPRAG1", 12, "12", "LAW12", "3D_COMP", "COMP_3D", "3PARBI", "RAGAB", 14, "14", "LAW14", "COMPSO", "COMP_SOL", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 24, 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 28, "28", "LAW28", "HONEYCOMB", 33, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 35, 36, 37, "37", "LAW37", "BIPHAS", "BIPHASIC", 38, "38", "LAW38", "VISC_TAB", 40, 42, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 62, 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 70, 81, 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN", 999},
    "penta6": {0, 1, 2, 3, 4, 5, "5", "LAW5", "JWL", 10, "10", "LAW10", "SOIL", "DPRAG", "DPRAG1", 12, "12", "LAW12", "3D_COMP", "COMP_3D", "3PARBI", "RAGAB", 14, "14", "LAW14", "COMPSO", "COMP_SOL", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 24, 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 28, "28", "LAW28", "HONEYCOMB", 33, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 35, 36, 37, "37", "LAW37", "BIPHAS", "BIPHASIC", 38, "38", "LAW38", "VISC_TAB", 40, 42, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 62, 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 70, 81, 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN", 83, 999},
    "pyra5": {0, 1, 2, 3, 4, 5, "5", "LAW5", "JWL", 10, "10", "LAW10", "SOIL", "DPRAG", "DPRAG1", 12, "12", "LAW12", "3D_COMP", "COMP_3D", "3PARBI", "RAGAB", 14, "14", "LAW14", "COMPSO", "COMP_SOL", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 24, 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 28, "28", "LAW28", "HONEYCOMB", 33, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 35, 36, 37, "37", "LAW37", "BIPHAS", "BIPHASIC", 38, "38", "LAW38", "VISC_TAB", 40, 42, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 62, 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 70, 81, 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN", 83, 999},
    "shells": {0, 1, 2, 3, 15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", 19, 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 27, 32, "32", "LAW32", "HILL", 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 36, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN"},
    # QBAT (Ishell=12, M41): the laws the layered kernel reuses from the
    # BT plumbing; no orthotropic (LAW19) shell_ortho wiring yet
    "shells_qbat": {0, 1, 2, 3, 15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 27, 32, "32", "LAW32", "HILL", 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 36, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN"},
    # QEPH (Ishell=24, M41): shares the BT layer plumbing INCLUDING the
    # shell_ortho fiber rotation (LAW19); the czfintn.F stabilization
    # runs isotropic moduli (czfintn_or orthotropic HM/HF deferred)
    "shells_qeph": {0, 1, 2, 3, 15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", 19, 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 27, 32, "32", "LAW32", "HILL", 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 36, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN"},
    "sh3n": {0, 1, 2, 3, 15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", 19, 22, "22", "LAW22", "DAMA", "PLAS_DAMA", 25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", 27, 32, "32", "LAW32", "HILL", 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW", 36, 43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB", 44, 60, "60", "LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60", 69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS", 82, "82", "LAW82", "OGDEN", "LAW82_OGDEN"},
    "trusses": {0, 1, 2, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW"},
    "springs": None,          # springs ignore their material entirely
    "beams": {0, 1, 2, 34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW"},
}
_ALLOWED_LAWS["quads"] = _ALLOWED_LAWS["shells"]



def check_mat_law34(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW34 (/MAT/BOLTZMAN, /MAT/VISC_MAXW) parameter bounds (M539).

    Required checks:
      - rho > 0
      - bulk > 0
      - g0 >= 0
      - gi >= 0
      - beta >= 0
    """
    mid = getattr(mat, "id", 0)
    if hasattr(mat, "k") and hasattr(mat, "rho0"):
        rho = getattr(mat, "rho0", 0.0)
        bulk = getattr(mat, "k", 0.0)
        g0 = getattr(mat, "g0", 0.0)
        gi = getattr(mat, "gl", 0.0)
        beta = getattr(mat, "beta", 0.0)
    else:
        params = getattr(mat, "params", {}) or {}
        rho = getattr(mat, "rho0", 0.0) or params.get("rho", 0.0) or params.get("MAT_RHO", 0.0)
        bulk = params.get("bulk", params.get("k", params.get("MAT_BULK", 0.0)))
        g0 = params.get("g0", params.get("MAT_G0", 0.0))
        gi = params.get("gi", params.get("gl", params.get("MAT_GI", 0.0)))
        beta = params.get("beta", params.get("decay", params.get("MAT_DECAY", 0.0)))

    try:
        rho = float(rho)
    except (TypeError, ValueError):
        rho = 0.0
    try:
        bulk = float(bulk)
    except (TypeError, ValueError):
        bulk = 0.0
    try:
        g0 = float(g0)
    except (TypeError, ValueError):
        g0 = 0.0
    try:
        gi = float(gi)
    except (TypeError, ValueError):
        gi = 0.0
    try:
        beta = float(beta)
    except (TypeError, ValueError):
        beta = 0.0

    if rho <= 0.0:
        log.error(f"/MAT/LAW34/{mid}: initial density RHO must be > 0 (got {rho:g})", "MAT CHECK")
    if bulk <= 0.0:
        log.error(f"/MAT/LAW34/{mid}: bulk modulus BULK must be > 0 (got {bulk:g})", "MAT CHECK")
    if g0 < 0.0:
        log.error(f"/MAT/LAW34/{mid}: short-term shear modulus G0 must be >= 0 (got {g0:g})", "MAT CHECK")
    if gi < 0.0:
        log.error(f"/MAT/LAW34/{mid}: long-term shear modulus GI must be >= 0 (got {gi:g})", "MAT CHECK")
    if beta < 0.0:
        log.error(f"/MAT/LAW34/{mid}: decay constant BETA must be >= 0 (got {beta:g})", "MAT CHECK")
    if g0 > 0.0 and gi > g0:
        log.warning(f"/MAT/LAW34/{mid}: long-term shear modulus GI ({gi:g}) exceeds short-term shear modulus G0 ({g0:g})", "MAT CHECK")


def check_mat_law37(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW37 (/MAT/BIPHAS, /MAT/BIPHASIC) parameter bounds (M540).

    Required checks:
      - rho_l0 > 0
      - rho_g0 > 0
      - c_l > 0
      - gamma_g > 0
      - 0 <= alpha1 <= 1
      - nu_l >= 0
      - nu_g >= 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    rho_l0 = _extract(["rho_l0", "RHO_l0", "Lqud_Rho_l", "rho_l", "RHO_L"])
    rho_g0 = _extract(["rho_g0", "RHO_G0", "Lqud_Rho_g", "rho_g", "RHO_G"], default=1.0)
    c_l = _extract(["c_l", "C_l", "bulk_l", "c1", "C1"])
    gamma_g = _extract(["gamma_g", "gamma", "GAMMA", "Lqud_Gamma_bulk", "gam"], default=1.4)
    alpha1 = _extract(["alpha1", "ALPHA1", "alpha_l", "Alpha_l", "a1", "A1"])
    nu_l = _extract(["nu_l", "Nu_l", "NU_l", "vis_l"])
    nu_g = _extract(["nu_g", "Nu_g", "NU_g", "vis_g"])

    if rho_l0 <= 0.0:
        log.error(f"/MAT/LAW37/{mid}: liquid reference density rho_l0 must be > 0 (got {rho_l0:g})", "MAT CHECK")
    if rho_g0 <= 0.0:
        log.error(f"/MAT/LAW37/{mid}: gas reference density rho_g0 must be > 0 (got {rho_g0:g})", "MAT CHECK")
    if c_l <= 0.0:
        log.error(f"/MAT/LAW37/{mid}: liquid bulk modulus c_l must be > 0 (got {c_l:g})", "MAT CHECK")
    if gamma_g <= 0.0:
        log.error(f"/MAT/LAW37/{mid}: gas constant gamma must be > 0 (got {gamma_g:g})", "MAT CHECK")
    if alpha1 < 0.0 or alpha1 > 1.0:
        log.error(f"/MAT/LAW37/{mid}: initial liquid massic fraction alpha1 must be between 0 and 1 (got {alpha1:g})", "MAT CHECK")
    if nu_l < 0.0:
        log.error(f"/MAT/LAW37/{mid}: liquid shear viscosity nu_l must be >= 0 (got {nu_l:g})", "MAT CHECK")
    if nu_g < 0.0:
        log.error(f"/MAT/LAW37/{mid}: gas shear viscosity nu_g must be >= 0 (got {nu_g:g})", "MAT CHECK")


def check_mat_law38(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW38 (/MAT/VISC_TAB) parameter bounds (M541).

    Required checks:
      - rho0 > 0
      - e > 0
      - 0 <= nu_t < 0.5 and 0 <= nu_c < 0.5
      - 1 <= nfunc <= 5
      - If air content active (kcompair == 1): 0 <= phi < 1 and P0 >= 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW38/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus e > 0
    e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW38/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratios: 0 <= nu_t < 0.5 and 0 <= nu_c < 0.5
    nu_t = _extract(["nu_t", "nu", "NU_t", "MAT_NU", "MAT_NUt", "vt", "VT"], default=0.0)
    nu_c = _extract(["nu_c", "NU_c", "MAT_NUc", "vc", "VC"], default=nu_t)
    for k in ["nu_c", "NU_c", "MAT_NUc", "vc", "VC"]:
        if hasattr(mat, k) and getattr(mat, k) is not None:
            try:
                nu_c = float(getattr(mat, k))
                break
            except (TypeError, ValueError):
                pass
        if isinstance(params, dict) and k in params and params[k] is not None:
            try:
                nu_c = float(params[k])
                break
            except (TypeError, ValueError):
                pass

    if nu_t < 0.0 or nu_t >= 0.5:
        log.error(f"/MAT/LAW38/{mid}: tensile Poisson's ratio nu_t must be in [0, 0.5) (got {nu_t:g})", "MAT CHECK")
    if nu_c < 0.0 or nu_c >= 0.5:
        log.error(f"/MAT/LAW38/{mid}: compressive Poisson's ratio nu_c must be in [0, 0.5) (got {nu_c:g})", "MAT CHECK")

    # 4. Number of functions: 1 <= nfunc <= 5
    nfunc = None
    for k in ["nfunc", "NFUNC", "mfunc", "MFUNC", "m_func", "n_func", "num_curves"]:
        if hasattr(mat, k) and getattr(mat, k) is not None:
            try:
                nfunc = int(getattr(mat, k))
                break
            except (TypeError, ValueError):
                pass
        if isinstance(params, dict) and k in params and params[k] is not None:
            try:
                nfunc = int(params[k])
                break
            except (TypeError, ValueError):
                pass
    if nfunc is None:
        curves = getattr(mat, "curves", None) or (params.get("curves", None) if isinstance(params, dict) else None) or (params.get("funct_ids", None) if isinstance(params, dict) else None)
        if isinstance(curves, (list, tuple)):
            nfunc = len(curves)
        else:
            nfunc = 1

    if nfunc < 1 or nfunc > 5:
        log.error(f"/MAT/LAW38/{mid}: number of functions nfunc must be between 1 and 5 (got {nfunc})", "MAT CHECK")

    # 5. Air content active (kcompair == 1): 0 <= phi < 1 and P0 >= 0
    kcompair = 0
    for k in ["kcompair", "KCOMPAIR", "MAT_Kair", "kair", "KAIR"]:
        if hasattr(mat, k) and getattr(mat, k) is not None:
            try:
                kcompair = int(getattr(mat, k))
                break
            except (TypeError, ValueError):
                pass
        if isinstance(params, dict) and k in params and params[k] is not None:
            try:
                kcompair = int(params[k])
                break
            except (TypeError, ValueError):
                pass

    if kcompair == 1:
        phi = _extract(["phi", "PHI", "poros", "porosity", "MAT_POROS"], default=0.0)
        p0 = _extract(["p0", "P0", "p_atm", "MAT_P0"], default=0.0)
        if phi < 0.0 or phi >= 1.0:
            log.error(f"/MAT/LAW38/{mid}: porosity phi must be in [0, 1) when air content is active (got {phi:g})", "MAT CHECK")
        if p0 < 0.0:
            log.error(f"/MAT/LAW38/{mid}: initial air pressure P0 must be >= 0 (got {p0:g})", "MAT CHECK")


def check_mat_law32(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW32 (/MAT/HILL) parameter bounds (M542).

    Required checks:
      - rho0 > 0
      - e > 0
      - 0 <= nu < 0.5
      - a > 0 (sigy)
      - n <= 1.0 (error if hard > 1.0 per hm_read_mat32.F line 219)
      - eps0 > 0 (error if srp <= 0 per hm_read_mat32.F line 224)
      - r00 > 0, r45 > 0, r90 > 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus e > 0
    e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0 <= nu < 0.5
    nu = _extract(["nu", "NU", "MAT_NU", "anu"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW32/{mid}: Poisson's ratio NU must be in [0, 0.5) (got {nu:g})", "MAT CHECK")

    # 4. Yield stress a (sigy) > 0
    a = _extract(["a", "A", "sigy", "SIGY", "MAT_SIGY", "ca", "CA"], default=0.0)
    if a <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: yield stress A (SIGY) must be > 0 (got {a:g})", "MAT CHECK")

    # 5. Hardening exponent n <= 1.0 (error if hard > 1.0 per hm_read_mat32.F line 219)
    n = _extract(["n", "N", "hard", "HARD", "MAT_HARD", "cn", "CN"], default=1.0)
    if n > 1.0:
        log.error(f"/MAT/LAW32/{mid}: hardening exponent n must be <= 1.0 (got {n:g})", "MAT CHECK")

    # 6. Reference strain rate eps0 > 0 (error if srp <= 0 per hm_read_mat32.F line 224)
    eps0 = _extract(["eps0", "EPS0", "srp", "SRP", "MAT_SRP", "MAT_SRP_MIN", "eps_dot_0", "EPS_DOT_0"], default=1.0)
    if eps0 <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: reference strain rate EPS_DOT_0 (srp) must be > 0 (got {eps0:g})", "MAT CHECK")

    # 7. Lankford parameters: r00 > 0, r45 > 0, r90 > 0
    r00 = _extract(["r00", "R00", "MAT_R00"], default=1.0)
    if r00 <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: Lankford parameter r00 must be > 0 (got {r00:g})", "MAT CHECK")

    r45 = _extract(["r45", "R45", "MAT_R45"], default=1.0)
    if r45 <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: Lankford parameter r45 must be > 0 (got {r45:g})", "MAT CHECK")

    r90 = _extract(["r90", "R90", "MAT_R90"], default=1.0)
    if r90 <= 0.0:
        log.error(f"/MAT/LAW32/{mid}: Lankford parameter r90 must be > 0 (got {r90:g})", "MAT CHECK")


def check_mat_law15(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW15 (/MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG) parameter bounds (M544).

    Required checks:
      - rho0 > 0
      - E1 > 0, E2 > 0
      - detc = 1 - nu12*nu21 > 0
      - G12, G23, G31 > 0
      - sig_1yt, sig_2yt, sig_1yc, sig_2yc, sig_12yc, sig_12yt > 0
      - b <= 1.0 (and n <= 1.0)
      - strengths (S1, S2, C1, C2, S12) > 0 (if specified or failure active)
      - tmax > 0 (if specified or failure active)
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's moduli: E1 > 0, E2 > 0
    e11 = _extract(["e11", "E11", "e1", "E1", "MAT_EA", "EA", "MAT_E1"], default=0.0)
    if e11 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: Young's modulus E1 must be > 0 (got {e11:g})", "MAT CHECK")

    e22 = _extract(["e22", "E22", "e2", "E2", "MAT_EB", "EB", "MAT_E2"], default=0.0)
    if e22 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: Young's modulus E2 must be > 0 (got {e22:g})", "MAT CHECK")

    # 3. Poisson's ratio determinant: detc = 1 - nu12*nu21 > 0
    nu12 = _extract(["nu12", "NU12", "MAT_PRAB", "nu", "NU"], default=0.0)
    nu21 = nu12 * e22 / e11 if e11 > 0.0 else 0.0
    detc = 1.0 - nu12 * nu21
    if detc <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: invalid Poisson ratio, detc = 1 - nu12*nu21 must be > 0 (got {detc:g})", "MAT CHECK")

    # 4. Shear moduli: G12, G23, G31 > 0
    g12 = _extract(["g12", "G12", "MAT_GAB", "GAB"], default=0.0)
    if g12 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: shear modulus G12 must be > 0 (got {g12:g})", "MAT CHECK")

    g23 = _extract(["g23", "G23", "MAT_GBC", "GBC"], default=0.0)
    if g23 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: shear modulus G23 must be > 0 (got {g23:g})", "MAT CHECK")

    g31 = _extract(["g31", "G31", "MAT_GCA", "GCA"], default=0.0)
    if g31 <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: shear modulus G31 must be > 0 (got {g31:g})", "MAT CHECK")

    # 5. Yield stresses: sig_1yt, sig_2yt, sig_1yc, sig_2yc, sig_12yc, sig_12yt > 0
    sig_1yt = _extract(["sig_1yt", "sig1yt", "sigyt1", "SIG_1YT", "MAT_SIGYT1", "SIGYT1"], default=0.0)
    sig_2yt = _extract(["sig_2yt", "sig2yt", "sigyt2", "SIG_2YT", "MAT_SIGYT2", "SIGYT2"], default=0.0)
    sig_1yc = _extract(["sig_1yc", "sig1yc", "sigyc1", "SIG_1YC", "MAT_SIGYC1", "SIGYC1"], default=0.0)
    sig_2yc = _extract(["sig_2yc", "sig2yc", "sigyc2", "SIG_2YC", "MAT_SIGYC2", "SIGYC2"], default=0.0)
    sig_12yc = _extract(["sig_12yc", "sig12yc", "sigc12", "SIG_12YC", "MAT_SIGC12", "SIGC12"], default=0.0)
    sig_12yt = _extract(["sig_12yt", "sig12yt", "sigt12", "SIG_12YT", "MAT_SIGT12", "SIGT12"], default=0.0)

    if sig_1yt <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: tensile yield stress in dir 1 (sig_1yt) must be > 0 (got {sig_1yt:g})", "MAT CHECK")
    if sig_1yc <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: compressive yield stress in dir 1 (sig_1yc) must be > 0 (got {sig_1yc:g})", "MAT CHECK")
    if sig_2yt <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: tensile yield stress in dir 2 (sig_2yt) must be > 0 (got {sig_2yt:g})", "MAT CHECK")
    if sig_2yc <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: compressive yield stress in dir 2 (sig_2yc) must be > 0 (got {sig_2yc:g})", "MAT CHECK")
    if sig_12yc <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: compressive shear yield stress in dir 12 (sig_12yc) must be > 0 (got {sig_12yc:g})", "MAT CHECK")
    if sig_12yt <= 0.0:
        log.error(f"/MAT/LAW15/{mid}: tensile shear yield stress in dir 12 (sig_12yt) must be > 0 (got {sig_12yt:g})", "MAT CHECK")

    # 6. Hardening parameter b <= 1.0, n <= 1.0
    b = _extract(["b", "B", "MAT_BETA", "cb", "CB"], default=0.0)
    if b > 1.0:
        log.error(f"/MAT/LAW15/{mid}: hardening parameter b must be <= 1.0 (got {b:g})", "MAT CHECK")

    n = _extract(["n", "N", "MAT_HARD", "cn", "CN"], default=1.0)
    if n > 1.0:
        log.error(f"/MAT/LAW15/{mid}: hardening exponent n must be <= 1.0 (got {n:g})", "MAT CHECK")

    # 7. Strengths S1, S2, C1, C2, S12 > 0 and relaxation time tmax > 0
    s1 = _extract(["s1", "S1", "MCHANG_S1"], default=0.0)
    s2 = _extract(["s2", "S2", "MCHANG_S2"], default=0.0)
    s12 = _extract(["s12", "S12", "MCHANG_S12"], default=0.0)
    c1 = _extract(["c1", "C1", "MCHANG_C1"], default=0.0)
    c2 = _extract(["c2", "C2", "MCHANG_C2"], default=0.0)
    tmax = _extract(["tmax", "TMAX", "MAT_TMAX"], default=0.0)

    if s1 < 0.0:
        log.error(f"/MAT/LAW15/{mid}: longitudinal tensile strength S1 must be > 0 (got {s1:g})", "MAT CHECK")
    if s2 < 0.0:
        log.error(f"/MAT/LAW15/{mid}: transverse tensile strength S2 must be > 0 (got {s2:g})", "MAT CHECK")
    if s12 < 0.0:
        log.error(f"/MAT/LAW15/{mid}: shear strength S12 must be > 0 (got {s12:g})", "MAT CHECK")
    if c1 < 0.0:
        log.error(f"/MAT/LAW15/{mid}: longitudinal compressive strength C1 must be > 0 (got {c1:g})", "MAT CHECK")
    if c2 < 0.0:
        log.error(f"/MAT/LAW15/{mid}: transverse compressive strength C2 must be > 0 (got {c2:g})", "MAT CHECK")
    if tmax < 0.0:
        log.error(f"/MAT/LAW15/{mid}: stress relaxation time tmax must be > 0 (got {tmax:g})", "MAT CHECK")

    failure_active = any(v > 0.0 for v in (s1, s2, s12, c1, c2, tmax))
    if failure_active:
        if s1 <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: longitudinal tensile strength S1 must be > 0 (got {s1:g})", "MAT CHECK")
        if s2 <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: transverse tensile strength S2 must be > 0 (got {s2:g})", "MAT CHECK")
        if s12 <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: shear strength S12 must be > 0 (got {s12:g})", "MAT CHECK")
        if c1 <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: longitudinal compressive strength C1 must be > 0 (got {c1:g})", "MAT CHECK")
        if c2 <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: transverse compressive strength C2 must be > 0 (got {c2:g})", "MAT CHECK")
        if tmax <= 0.0:
            log.error(f"/MAT/LAW15/{mid}: stress relaxation time tmax must be > 0 (got {tmax:g})", "MAT CHECK")


def check_mat_law22(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW22 (/MAT/DAMA, /MAT/PLAS_DAMA) parameter bounds (M545).

    Required checks:
      - rho0 > 0
      - E > 0
      - 0 <= nu < 0.5
      - a > 0 (sigy)
      - n <= 1.0 (error if n > 1.0 per hm_read_mat22.F line 213)
      - eps_dot_0 > 0 (error if srp <= 0 per hm_read_mat22.F line 221)
      - e_tan <= 0.0 (error if e_tan > 0 per hm_read_mat22.F line 229 / matl22_dama.cfg line 109)
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW22/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW22/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0 <= nu < 0.5
    nu = _extract(["nu", "NU", "MAT_NU", "anu"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW22/{mid}: Poisson's ratio NU must be in [0, 0.5) (got {nu:g})", "MAT CHECK")

    # 4. Yield stress a (sigy) > 0
    a = _extract(["a", "A", "sigy", "SIGY", "MAT_SIGY", "ca", "CA"], default=0.0)
    if a <= 0.0:
        log.error(f"/MAT/LAW22/{mid}: yield stress a (SIGY) must be > 0 (got {a:g})", "MAT CHECK")

    # 5. Hardening exponent n <= 1.0
    n = _extract(["n", "N", "hard", "HARD", "MAT_HARD", "cn", "CN"], default=1.0)
    if n > 1.0:
        log.error(f"/MAT/LAW22/{mid}: hardening exponent n must be <= 1.0 (got {n:g})", "MAT CHECK")

    # 6. Reference strain rate eps_dot_0 > 0
    eps_dot_0 = _extract(["eps_dot_0", "eps0", "EPS0", "srp", "SRP", "MAT_SRP", "EPS_DOT_0"], default=1.0)
    if eps_dot_0 <= 0.0:
        log.error(f"/MAT/LAW22/{mid}: reference strain rate EPS_DOT_0 (srp) must be > 0 (got {eps_dot_0:g})", "MAT CHECK")

    # 7. Softening damage slope e_tan <= 0.0
    e_tan = _extract(["e_tan", "etan", "e_t", "MAT_ETAN", "EL", "E_TAN"], default=0.0)
    if e_tan > 0.0:
        log.error(f"/MAT/LAW22/{mid}: softening damage slope E_tan must be <= 0.0 (got {e_tan:g})", "MAT CHECK")


def check_mat_law12(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW12 (/MAT/3D_COMP, /MAT/COMP_3D) parameter bounds (M546).

    Required checks (hm_read_mat12.F):
      - rho0 > 0
      - E11, E22, E33 > 0
      - detc > 0 (compliance matrix determinant)
      - G12, G23, G31 >= 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW12/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's moduli E11, E22, E33 > 0
    e11 = _extract(["e11", "E11", "MAT_EA", "ea", "EA"], default=0.0)
    e22 = _extract(["e22", "E22", "MAT_EB", "eb", "EB"], default=0.0)
    e33 = _extract(["e33", "E33", "MAT_EC", "ec", "EC"], default=0.0)

    if e11 <= 0.0 or e22 <= 0.0 or e33 <= 0.0:
        log.error(
            f"/MAT/LAW12/{mid}: Young's moduli E11, E22, E33 must be > 0 (got E11={e11:g}, E22={e22:g}, E33={e33:g}) (ANCMSG 306)",
            "MAT CHECK",
        )
    else:
        # 3. Determinant of compliance matrix DETC > 0
        nu12 = _extract(["nu12", "NU12", "MAT_PRAB", "prab", "PRAB"], default=0.0)
        nu23 = _extract(["nu23", "NU23", "MAT_PRBC", "prbc", "PRBC"], default=0.0)
        nu31 = _extract(["nu31", "NU31", "MAT_PRCA", "prca", "PRCA"], default=0.0)

        c11 = 1.0 / e11
        c22 = 1.0 / e22
        c33 = 1.0 / e33
        c12 = -nu12 / e11
        c13 = -nu31 / e33
        c23 = -nu23 / e22

        detc = (
            c11 * c22 * c33
            - c11 * (c23**2)
            - (c12**2) * c33
            + 2.0 * c12 * c13 * c23
            - (c13**2) * c22
        )
        if detc <= 0.0:
            log.error(f"/MAT/LAW12/{mid}: compliance matrix determinant DETC must be > 0 (got {detc:g}) (ANCMSG 307)", "MAT CHECK")

    # 4. Shear moduli
    g12 = _extract(["g12", "G12", "MAT_GAB", "gab", "GAB"], default=0.0)
    g23 = _extract(["g23", "G23", "MAT_GBC", "gbc", "GBC"], default=0.0)
    g31 = _extract(["g31", "G31", "MAT_GCA", "gca", "GCA"], default=0.0)
    if g12 < 0.0 or g23 < 0.0 or g31 < 0.0:
        log.error(f"/MAT/LAW12/{mid}: shear moduli G12, G23, G31 must be >= 0", "MAT CHECK")


def check_mat_law14(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL) parameter bounds (M547).

    Required checks (hm_read_mat14.F):
      - rho0 > 0
      - E11, E22, E33 > 0
      - detc > 0 (compliance matrix determinant)
      - G12, G23, G31 >= 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    # Cam-Clay check bypass (M187)
    is_cam_clay = (
        (hasattr(mat, "kappa") or "kappa" in params or "KAPPA" in params)
        and not hasattr(mat, "ea")
        and "e11" not in params
        and "E11" not in params
        and "MAT_EA" not in params
    )
    if is_cam_clay:
        rho0 = getattr(mat, "rho0", None)
        if rho0 is None:
            rho0 = getattr(mat, "rho", params.get("rho", params.get("MAT_RHO", 0.0)))
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0
        if rho0 <= 0.0:
            log.error(f"/MAT/LAW14/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")
        return

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW14/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's moduli E11, E22, E33 > 0
    e11 = _extract(["e11", "E11", "MAT_EA", "ea", "EA"], default=0.0)
    e22 = _extract(["e22", "E22", "MAT_EB", "eb", "EB"], default=0.0)
    e33 = _extract(["e33", "E33", "MAT_EC", "ec", "EC"], default=0.0)

    if e11 <= 0.0 or e22 <= 0.0 or e33 <= 0.0:
        log.error(
            f"/MAT/LAW14/{mid}: Young's moduli E11, E22, E33 must be > 0 (got E11={e11:g}, E22={e22:g}, E33={e33:g}) (ANCMSG 306)",
            "MAT CHECK",
        )
    else:
        # 3. Determinant of compliance matrix DETC > 0
        nu12 = _extract(["nu12", "NU12", "MAT_PRAB", "prab", "PRAB"], default=0.0)
        nu23 = _extract(["nu23", "NU23", "MAT_PRBC", "prbc", "PRBC"], default=0.0)
        nu31 = _extract(["nu31", "NU31", "MAT_PRCA", "prca", "PRCA"], default=0.0)

        c11 = 1.0 / e11
        c22 = 1.0 / e22
        c33 = 1.0 / e33
        c12 = -nu12 / e11
        c13 = -nu31 / e33
        c23 = -nu23 / e22

        detc = (
            c11 * c22 * c33
            - c11 * (c23**2)
            - (c12**2) * c33
            + 2.0 * c12 * c13 * c23
            - (c13**2) * c22
        )
        if detc <= 0.0:
            log.error(f"/MAT/LAW14/{mid}: compliance matrix determinant DETC must be > 0 (got {detc:g}) (ANCMSG 307)", "MAT CHECK")

    # 4. Shear moduli
    g12 = _extract(["g12", "G12", "MAT_GAB", "gab", "GAB"], default=0.0)
    g23 = _extract(["g23", "G23", "MAT_GBC", "gbc", "GBC"], default=0.0)
    g31 = _extract(["g31", "G31", "MAT_GCA", "gca", "GCA"], default=0.0)
    if g12 < 0.0 or g23 < 0.0 or g31 < 0.0:
        log.error(f"/MAT/LAW14/{mid}: shear moduli G12, G23, G31 must be >= 0", "MAT CHECK")


def check_mat_law25(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW25 (/MAT/COMP_PLAS, /MAT/COMPSH, /MAT/TSAI_WU, /MAT/CRASURV) parameter bounds (M543).

    Required checks:
      - rho0 > 0
      - E1 > 0, E2 > 0
      - detc = 1 - nu12*nu21 > 0
      - G12, G23, G31 > 0
      - sigyt1, sigyc1, sigyt2, sigyc2, sigt12, sigc12 > 0
      - n <= 1.0
      - 0 <= dmax <= 1.0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's moduli: E1 > 0, E2 > 0
    e11 = _extract(["e11", "E11", "e1", "E1", "MAT_EA", "EA", "MAT_E1"], default=0.0)
    if e11 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: Young's modulus E1 must be > 0 (got {e11:g})", "MAT CHECK")

    e22 = _extract(["e22", "E22", "e2", "E2", "MAT_EB", "EB", "MAT_E2"], default=0.0)
    if e22 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: Young's modulus E2 must be > 0 (got {e22:g})", "MAT CHECK")

    # 3. Poisson's ratio determinant: detc = 1 - nu12*nu21 > 0
    nu12 = _extract(["nu12", "NU12", "MAT_PRAB", "nu", "NU"], default=0.0)
    nu21 = nu12 * e22 / e11 if e11 > 0.0 else 0.0
    detc = 1.0 - nu12 * nu21
    if detc <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: invalid Poisson ratio, detc = 1 - nu12*nu21 must be > 0 (got {detc:g})", "MAT CHECK")

    # 4. Shear moduli: G12, G23, G31 > 0
    g12 = _extract(["g12", "G12", "MAT_GAB", "GAB"], default=0.0)
    if g12 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: shear modulus G12 must be > 0 (got {g12:g})", "MAT CHECK")

    g23 = _extract(["g23", "G23", "MAT_GBC", "GBC"], default=0.0)
    if g23 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: shear modulus G23 must be > 0 (got {g23:g})", "MAT CHECK")

    g31 = _extract(["g31", "G31", "MAT_GCA", "GCA"], default=0.0)
    if g31 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: shear modulus G31 must be > 0 (got {g31:g})", "MAT CHECK")

    # 5. Yield stresses: sigyt1, sigyc1, sigyt2, sigyc2, sigt12, sigc12 > 0
    sigyt1 = _extract(["sigyt1", "sig_1yt", "MAT_SIGYT1", "SIGYT1", "sigyt_1"], default=0.0)
    sigyc1 = _extract(["sigyc1", "sig_1yc", "MAT_SIGYC1", "SIGYC1", "MAT_SIG1_yc", "sigyc_1"], default=0.0)
    sigyt2 = _extract(["sigyt2", "sig_2yt", "MAT_SIGYT2", "SIGYT2", "sigyt_2"], default=0.0)
    sigyc2 = _extract(["sigyc2", "sig_2yc", "MAT_SIGYC2", "SIGYC2", "MAT_SIG2_yc", "sigyc_2"], default=0.0)
    sigt12 = _extract(["sigt12", "sigyt12", "sig_12yt", "MAT_SIGT12", "SIGT12", "MAT_SIG12_yt"], default=0.0)
    sigc12 = _extract(["sigc12", "sigyc12", "sig_12yc", "MAT_SIGC12", "SIGC12"], default=sigt12)

    if sigyt1 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: tensile yield stress in dir 1 (sigyt1) must be > 0 (got {sigyt1:g})", "MAT CHECK")
    if sigyc1 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: compressive yield stress in dir 1 (sigyc1) must be > 0 (got {sigyc1:g})", "MAT CHECK")
    if sigyt2 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: tensile yield stress in dir 2 (sigyt2) must be > 0 (got {sigyt2:g})", "MAT CHECK")
    if sigyc2 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: compressive yield stress in dir 2 (sigyc2) must be > 0 (got {sigyc2:g})", "MAT CHECK")
    if sigt12 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: tensile shear yield stress in dir 12 (sigt12) must be > 0 (got {sigt12:g})", "MAT CHECK")
    if sigc12 <= 0.0:
        log.error(f"/MAT/LAW25/{mid}: compressive shear yield stress in dir 12 (sigc12) must be > 0 (got {sigc12:g})", "MAT CHECK")

    # 6. Hardening exponent: n <= 1.0
    iform = 0
    for k in ["iform", "MAT_Iflag", "iflag", "IFORM", "IFLAG"]:
        if hasattr(mat, k) and getattr(mat, k) is not None:
            try:
                iform = int(getattr(mat, k))
                break
            except (TypeError, ValueError):
                pass
        if isinstance(params, dict) and k in params and params[k] is not None:
            try:
                iform = int(params[k])
                break
            except (TypeError, ValueError):
                pass

    if iform == 0:
        n = _extract(["n", "N", "MAT_HARD", "hard", "cn"], default=1.0)
        if n > 1.0:
            log.error(f"/MAT/LAW25/{mid}: hardening exponent n must be <= 1.0 (got {n:g})", "MAT CHECK")
    else:
        n1_t = _extract(["n1_t", "n_1t", "MAT_n1_t", "cnt1"], default=1.0)
        n1_c = _extract(["n1_c", "n_1c", "MAT_n1_c", "cnc1"], default=1.0)
        n2_t = _extract(["n2_t", "n_2t", "MAT_n2_t", "cnt2"], default=1.0)
        n2_c = _extract(["n2_c", "n_2c", "MAT_n2_c", "cnc2"], default=1.0)
        n12_t = _extract(["n12_t", "n_12t", "MAT_n12_t", "cnt12"], default=1.0)
        if n1_t > 1.0 or n1_c > 1.0 or n2_t > 1.0 or n2_c > 1.0 or n12_t > 1.0:
            log.error(f"/MAT/LAW25/{mid}: hardening exponent n must be <= 1.0", "MAT CHECK")

    # 7. Maximum damage: 0 <= dmax <= 1.0
    dmax = _extract(["dmax", "MAT_DAMAGE", "d_max", "damage"], default=0.0)
    if dmax < 0.0 or dmax > 1.0:
        log.error(f"/MAT/LAW25/{mid}: maximum damage dmax must be in [0, 1] (got {dmax:g})", "MAT CHECK")

    # 8. Damage strains: epsm >= epst
    epst1 = _extract(["eps_t1", "epst1", "MAT_EPST1", "EPST1", "eps_t", "epst"], default=0.0)
    epsm1 = _extract(["eps_m1", "epsm1", "MAT_EPSM1", "EPSM1", "eps_m", "epsm"], default=0.0)
    epst2 = _extract(["eps_t2", "epst2", "MAT_EPST2", "EPST2"], default=0.0)
    epsm2 = _extract(["eps_m2", "epsm2", "MAT_EPSM2", "EPSM2"], default=0.0)

    if (epst1 > 0.0 or epsm1 > 0.0) and epsm1 < epst1:
        log.error(f"/MAT/LAW25/{mid}: maximum damage strain eps_m1 must be >= damage initiation strain eps_t1 (got eps_m1={epsm1:g}, eps_t1={epst1:g})", "MAT CHECK")
    if (epst2 > 0.0 or epsm2 > 0.0) and epsm2 < epst2:
        log.error(f"/MAT/LAW25/{mid}: maximum damage strain eps_m2 must be >= damage initiation strain eps_t2 (got eps_m2={epsm2:g}, eps_t2={epst2:g})", "MAT CHECK")

    if iform != 0:
        eps_1t1 = _extract(["eps_1t1", "MAT_EPS1_t1"], default=0.0)
        eps_2t1 = _extract(["eps_2t1", "MAT_EPS2_t1"], default=0.0)
        if (eps_1t1 > 0.0 or eps_2t1 > 0.0) and eps_2t1 < eps_1t1:
            log.error(f"/MAT/LAW25/{mid}: CRASURV maximum damage strain eps_2t1 must be >= eps_1t1 (got eps_2t1={eps_2t1:g}, eps_1t1={eps_1t1:g})", "MAT CHECK")
        eps_1t2 = _extract(["eps_1t2", "MAT_EPS1_t2"], default=0.0)
        eps_2t2 = _extract(["eps_2t2", "MAT_EPS2_t2"], default=0.0)
        if (eps_1t2 > 0.0 or eps_2t2 > 0.0) and eps_2t2 < eps_1t2:
            log.error(f"/MAT/LAW25/{mid}: CRASURV maximum damage strain eps_2t2 must be >= eps_1t2 (got eps_2t2={eps_2t2:g}, eps_1t2={eps_1t2:g})", "MAT CHECK")
        eps_1c1 = _extract(["eps_1c1", "MAT_EPS1_c1"], default=0.0)
        eps_2c1 = _extract(["eps_2c1", "MAT_EPS2_c1"], default=0.0)
        if (eps_1c1 > 0.0 or eps_2c1 > 0.0) and eps_2c1 < eps_1c1:
            log.error(f"/MAT/LAW25/{mid}: CRASURV maximum damage strain eps_2c1 must be >= eps_1c1 (got eps_2c1={eps_2c1:g}, eps_1c1={eps_1c1:g})", "MAT CHECK")
        eps_1c2 = _extract(["eps_1c2", "MAT_EPS1_c2"], default=0.0)
        eps_2c2 = _extract(["eps_2c2", "MAT_EPS2_c2"], default=0.0)
        if (eps_1c2 > 0.0 or eps_2c2 > 0.0) and eps_2c2 < eps_1c2:
            log.error(f"/MAT/LAW25/{mid}: CRASURV maximum damage strain eps_2c2 must be >= eps_1c2 (got eps_2c2={eps_2c2:g}, eps_1c2={eps_1c2:g})", "MAT CHECK")
        eps_1t12 = _extract(["eps_1t12", "MAT_EPS1_t12"], default=0.0)
        eps_2t12 = _extract(["eps_2t12", "MAT_EPS2_t12"], default=0.0)
        if (eps_1t12 > 0.0 or eps_2t12 > 0.0) and eps_2t12 < eps_1t12:
            log.error(f"/MAT/LAW25/{mid}: CRASURV maximum damage strain eps_2t12 must be >= eps_1t12 (got eps_2t12={eps_2t12:g}, eps_1t12={eps_1t12:g})", "MAT CHECK")



def check_mat_law43(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW43 (/MAT/HILL_TAB, /MAT/HILL_PLAS_TAB) parameter bounds (M548).

    Citing hm_read_mat43.F and radioss140/MAT/matl43_HILL_TAB.cfg:
      - rho0 > 0
      - e > 0
      - 0 <= nu < 0.5
      - r00 > 0, r45 > 0, r90 > 0
      - 0 <= fisokin <= 1 (ANCMSG 913)
      - at least one plasticity curve defined with non-zero ID (ANCMSG 366)
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0", "Refer_Rho", "rhor"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW43/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = _extract(["e", "E", "MAT_E", "young", "E0"], default=0.0)
    if e <= 0.0:
        log.error(f"/MAT/LAW43/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0 <= nu < 0.5
    nu = _extract(["nu", "NU", "MAT_NU", "anu"], default=0.0)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW43/{mid}: Poisson's ratio NU must satisfy 0 <= NU < 0.5 (got {nu:g})", "MAT CHECK")

    # 4. Lankford coefficients: R00 > 0, R45 > 0, R90 > 0
    r00 = _extract(["r00", "R00", "MAT_R00", "r0", "R0"], default=1.0)
    if r00 <= 0.0:
        log.error(f"/MAT/LAW43/{mid}: Lankford coefficient R00 must be > 0 (got {r00:g})", "MAT CHECK")

    r45 = _extract(["r45", "R45", "MAT_R45", "r_45"], default=1.0)
    if r45 <= 0.0:
        log.error(f"/MAT/LAW43/{mid}: Lankford coefficient R45 must be > 0 (got {r45:g})", "MAT CHECK")

    r90 = _extract(["r90", "R90", "MAT_R90", "r_90"], default=1.0)
    if r90 <= 0.0:
        log.error(f"/MAT/LAW43/{mid}: Lankford coefficient R90 must be > 0 (got {r90:g})", "MAT CHECK")

    # 5. Iso-kinematic hardening factor: 0 <= FISOKIN <= 1 (ANCMSG 913)
    fisokin = _extract(["fisokin", "FISOKIN", "chard", "CHARD", "MAT_CHARD", "c_hard"], default=0.0)
    if fisokin < 0.0 or fisokin > 1.0:
        log.error(
            f"/MAT/LAW43/{mid}: iso-kinematic hardening factor FISOKIN must be in [0, 1] (got {fisokin:g}) (ANCMSG 913)",
            "MAT CHECK",
        )

    # 6. Plasticity hardening curves: must have at least one valid curve (ANCMSG 366)
    curves = getattr(mat, "curves", None)
    if curves is None and isinstance(params, dict):
        curves = params.get("curves", None)

    has_curve = False
    if curves:
        for c in curves:
            if isinstance(c, dict):
                fid = c.get("fct_id", c.get("funct_id", c.get("id", 0)))
                pts = c.get("points", c.get("pts", None))
                if fid or pts is not None:
                    has_curve = True
                    break
            elif isinstance(c, (list, tuple)):
                if len(c) > 0 and c[0]:
                    has_curve = True
                    break
            elif isinstance(c, (int, float)) and c != 0:
                has_curve = True
                break
    elif isinstance(params, dict):
        cxs = params.get("curve_x", [])
        if len(cxs) > 0:
            has_curve = True

    if not has_curve:
        log.error(f"/MAT/LAW43/{mid}: no plasticity hardening curve defined (ANCMSG 366)", "MAT CHECK")


def check_mat_law82(mat: Any, log: MessageLog) -> None:
    """Validate /MAT/LAW82 (/MAT/OGDEN, /MAT/LAW82_OGDEN) parameter bounds (M549).

    Citing hm_read_mat82.F and radioss110/MAT/matl82_ogden.cfg:
      - rho0 > 0 (MSGERROR if <= 0)
      - nordre >= 1 (MSGERROR if < 1, MSGID 559)
      - sum(mu) > 0 (MSGERROR if <= 0, MSGID 846)
      - 0.0 <= nu < 0.5
      - for each term i, alpha[i] != 0
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW82/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Order of Ogden model: nordre >= 1 (MSGID 559)
    nordre = getattr(mat, "nordre", None)
    if nordre is None:
        nordre = getattr(mat, "order", None)
    if nordre is None and isinstance(params, dict):
        nordre = params.get("ORDER", params.get("nordre", params.get("order", 1)))
    try:
        nordre = int(nordre)
    except (TypeError, ValueError):
        nordre = 0

    if nordre < 1:
        log.error(f"/MAT/LAW82/{mid}: order of Ogden model must be >= 1 (got {nordre}) (MSGID 559)", "MAT CHECK")

    # Extract mu, alpha arrays
    mu = getattr(mat, "mu", None)
    if mu is None:
        mu = getattr(mat, "mu_arr", None)
    if mu is None and isinstance(params, dict):
        mu = params.get("mu", params.get("Mu_arr", []))
    if mu is None:
        mu = []
    mu_vals = [float(x) for x in mu]

    alpha = getattr(mat, "alpha", None)
    if alpha is None:
        alpha = getattr(mat, "alpha_arr", None)
    if alpha is None and isinstance(params, dict):
        alpha = params.get("alpha", params.get("Alpha_arr", []))
    if alpha is None:
        alpha = []
    alpha_vals = [float(x) for x in alpha]

    # 3. sum(mu) > 0 (MSGID 846)
    gs = sum(mu_vals[:nordre]) if nordre > 0 else sum(mu_vals)
    if gs <= 0.0:
        log.error(f"/MAT/LAW82/{mid}: sum of shear moduli mu must be > 0 (got {gs:g}) (MSGID 846)", "MAT CHECK")

    # 4. Poisson's ratio: 0.0 <= nu < 0.5
    nu = _extract(["nu", "MAT_NU", "NU"], default=0.475)
    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW82/{mid}: Poisson ratio nu must satisfy 0.0 <= nu < 0.5 (got {nu:g})", "MAT CHECK")

    # 5. For each term i, alpha[i] != 0
    for i in range(min(nordre, len(alpha_vals))):
        if alpha_vals[i] == 0.0:
            log.error(f"/MAT/LAW82/{mid}: alpha parameter at term {i+1} must be non-zero", "MAT CHECK")


def check_mat_law69(mat: Any, log: MessageLog, functions: Optional[Dict[int, Any]] = None) -> None:
    """Validate /MAT/LAW69 (/MAT/HYP_ELAS, /MAT/HYPERELASTIC) parameter bounds (M550).

    Required checks (citing hm_read_mat69.F and matl69_69.cfg):
      - rho0 > 0
      - 0.0 <= nu < 0.5
      - FCT_ID1 must exist in functions dictionary if specified
      - Monotonicity check of FCT_ID1 curve (abscissae and ordinates)
      - Compatible elements: Solids and Shells supported; reject trusses, beams, springs.
    """
    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW69/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Poisson's ratio: 0.0 <= nu < 0.5
    nu = getattr(mat, "nu", None)
    if nu is None:
        nu = _extract(["nu", "MAT_NU", "NU"], default=0.495)
    else:
        try:
            nu = float(nu)
        except (TypeError, ValueError):
            nu = 0.495

    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW69/{mid}: Poisson ratio nu must satisfy 0.0 <= nu < 0.5 (got {nu:g})", "MAT CHECK")

    # 3. Test curve FCT_ID1
    fct_id1 = getattr(mat, "fct_id1", None)
    if fct_id1 is None:
        fct_id1 = getattr(mat, "fct_id_data", None)
    if fct_id1 is None and isinstance(params, dict):
        fct_id1 = params.get("fct_id1", params.get("fct_id_data", params.get("FUN_B1", 0)))
    try:
        fct_id1 = int(fct_id1) if fct_id1 is not None else 0
    except (TypeError, ValueError):
        fct_id1 = 0

    funcs = functions
    if funcs is None and hasattr(mat, "functions"):
        funcs = mat.functions

    if fct_id1 != 0:
        if funcs is not None:
            if fct_id1 not in funcs:
                log.error(f"/MAT/LAW69/{mid}: test curve FCT_ID1={fct_id1} not found in functions dictionary", "MAT CHECK")
            else:
                fn = funcs[fct_id1]
                x = getattr(fn, "x", None)
                y = getattr(fn, "y", None)
                if x is None and isinstance(fn, dict):
                    x = fn.get("x", None)
                    y = fn.get("y", None)
                if x is not None and y is not None:
                    import numpy as np
                    xa = np.asarray(x, dtype=float)
                    ya = np.asarray(y, dtype=float)
                    if xa.size >= 2 and np.any(np.diff(xa) <= 0):
                        log.error(f"/MAT/LAW69/{mid}: test curve FCT_ID1={fct_id1} abscissae must be strictly increasing (monotonicity violation)", "MAT CHECK")
                    if ya.size >= 2 and np.any(np.diff(ya) < 0):
                        log.error(f"/MAT/LAW69/{mid}: test curve FCT_ID1={fct_id1} ordinates must be monotonically increasing (monotonicity violation)", "MAT CHECK")


def check_mat_law60(
    mat: Any,
    arg1: Any = None,
    arg2: Any = None,
    model: Any = None,
    log: Any = None,
    functions: Any = None,
    **kwargs,
) -> None:
    """Validate /MAT/LAW60 (/MAT/PLAS_T3, /MAT/FABRIC) parameter bounds (M551).

    Required checks (citing hm_read_mat60.F and matl60_PLAS_T3.cfg):
      - rho0 > 0
      - E > 0
      - 0.0 <= nu < 0.5
      - If eps_t1 > 0 and eps_t2 > 0: eps_t1 < eps_t2
      - Monotonic rates: rates[i] < rates[i+1]
      - Curve existence in model.tables / model.functions / functions
      - Compatible elements: Solids and Shells supported; reject trusses, beams, springs.
    """
    actual_log = log
    actual_model = model

    if isinstance(arg1, MessageLog):
        actual_log = arg1
        if arg2 is not None and not isinstance(arg2, MessageLog):
            actual_model = arg2
    elif isinstance(arg2, MessageLog):
        actual_log = arg2
        if arg1 is not None:
            actual_model = arg1
    else:
        if arg1 is not None and actual_model is None:
            actual_model = arg1
        if arg2 is not None and actual_log is None:
            actual_log = arg2

    if actual_log is None:
        actual_log = MessageLog()
    log = actual_log
    model = actual_model

    mid = getattr(mat, "id", 0)
    params = getattr(mat, "params", {}) or {}

    def _extract(keys: list[str], default: float = 0.0) -> float:
        for k in keys:
            if hasattr(mat, k):
                val = getattr(mat, k)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            if isinstance(params, dict) and k in params:
                val = params[k]
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
        return default

    # 1. Density rho0 > 0
    rho0 = getattr(mat, "rho0", None)
    if rho0 is None:
        rho0 = getattr(mat, "rho", None)
    if rho0 is None:
        rho0 = _extract(["rho0", "rho", "MAT_RHO", "RHO", "RHO0"], default=0.0)
    else:
        try:
            rho0 = float(rho0)
        except (TypeError, ValueError):
            rho0 = 0.0

    if rho0 <= 0.0:
        log.error(f"/MAT/LAW60/{mid}: initial density RHO must be > 0 (got {rho0:g})", "MAT CHECK")

    # 2. Young's modulus E > 0
    e = getattr(mat, "e", None)
    if e is None:
        e = getattr(mat, "E", None)
    if e is None:
        e = _extract(["e", "E", "MAT_E", "young"], default=0.0)
    else:
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = 0.0

    if e <= 0.0:
        log.error(f"/MAT/LAW60/{mid}: Young's modulus E must be > 0 (got {e:g})", "MAT CHECK")

    # 3. Poisson's ratio: 0.0 <= nu < 0.5
    nu = getattr(mat, "nu", None)
    if nu is None:
        nu = _extract(["nu", "MAT_NU", "NU", "poisson"], default=0.0)
    else:
        try:
            nu = float(nu)
        except (TypeError, ValueError):
            nu = 0.0

    if nu < 0.0 or nu >= 0.5:
        log.error(f"/MAT/LAW60/{mid}: Poisson's ratio nu must satisfy 0.0 <= nu < 0.5 (got {nu:g})", "MAT CHECK")

    # 4. Tensile failure strains: if eps_t1 > 0 and eps_t2 > 0: eps_t1 < eps_t2
    eps_t1 = getattr(mat, "eps_t1", None)
    if eps_t1 is None:
        eps_t1 = _extract(["eps_t1", "EPST1", "epst1"], default=1.0e30)
    else:
        try:
            eps_t1 = float(eps_t1)
        except (TypeError, ValueError):
            eps_t1 = 1.0e30

    eps_t2 = getattr(mat, "eps_t2", None)
    if eps_t2 is None:
        eps_t2 = _extract(["eps_t2", "EPST2", "epst2"], default=2.0e30)
    else:
        try:
            eps_t2 = float(eps_t2)
        except (TypeError, ValueError):
            eps_t2 = 2.0e30

    if eps_t1 > 0.0 and eps_t2 > 0.0 and eps_t1 >= eps_t2:
        log.error(
            f"/MAT/LAW60/{mid}: tensile failure strains must satisfy eps_t1 < eps_t2 (got eps_t1={eps_t1:g}, eps_t2={eps_t2:g})",
            "MAT CHECK",
        )

    # 5. Monotonic rates: rates[i] < rates[i+1]
    rates = getattr(mat, "rates", None)
    if rates is None:
        rates = getattr(mat, "eps_rates", None)
    if rates is None and isinstance(params, dict):
        rates = params.get("rates", params.get("eps_rates", []))
    if rates is not None and len(rates) > 0:
        rates_list = []
        for r in rates:
            try:
                rates_list.append(float(r))
            except (TypeError, ValueError):
                pass
        for i in range(len(rates_list) - 1):
            if rates_list[i] >= rates_list[i + 1]:
                log.error(
                    f"/MAT/LAW60/{mid}: strain rates must be strictly increasing (got rates[{i}]={rates_list[i]:g} >= rates[{i+1}]={rates_list[i+1]:g})",
                    "MAT CHECK",
                )
                break

    # 6. Curve existence
    funcs_list = getattr(mat, "funcs", None)
    if funcs_list is None:
        funcs_list = getattr(mat, "fun_ids", None)
    if funcs_list is None and isinstance(params, dict):
        funcs_list = params.get("funcs", params.get("fun_ids", []))

    avail_curves = set()
    if model is not None:
        if hasattr(model, "tables") and model.tables:
            avail_curves.update(model.tables.keys())
        if hasattr(model, "functions") and model.functions:
            avail_curves.update(model.functions.keys())
    if functions is not None:
        avail_curves.update(functions.keys())
    if hasattr(mat, "functions") and mat.functions:
        avail_curves.update(mat.functions.keys())
    if hasattr(mat, "tables") and mat.tables:
        avail_curves.update(mat.tables.keys())

    if avail_curves and funcs_list is not None and len(funcs_list) > 0:
        for fid in funcs_list:
            try:
                ifid = int(fid)
            except (TypeError, ValueError):
                continue
            if ifid != 0 and ifid not in avail_curves:
                log.error(
                    f"/MAT/LAW60/{mid}: function curve {ifid} not found in model tables or functions",
                    "MAT CHECK",
                )

    # 7. Reject trusses, beams, springs if attached to element groups
    if model is not None and hasattr(model, "element_groups"):
        for name, el_group in model.element_groups():
            if name in ("trusses", "beams", "springs"):
                for el in el_group.values():
                    el_mid = getattr(el, "mat_id", getattr(el, "mid", None))
                    if el_mid == mid:
                        log.error(
                            f"/MAT/LAW60/{mid} (/MAT/PLAS_T3) is not supported for {name} elements "
                            f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                            "MAT CHECK",
                        )
                        break


_check_mat_law60 = check_mat_law60


def check_materials(model: Model, log: MessageLog) -> None:
    """Validate all material parameters across model."""
    # M539: Material LAW34 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (34, "34", "LAW34", "BOLTZMAN", "BOLTZMANN", "VISC_MAXW"):
            check_mat_law34(mat, log)
    for mid, mat34 in getattr(model, "mat_law34s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law34(mat34, log)

    # M540: Material LAW37 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (37, "37", "LAW37", "BIPHAS", "BIPHASIC"):
            params = getattr(mat, "params", {}) or {}
            is_biquad = hasattr(mat, "a") and hasattr(mat, "c1") and not any(
                k in params for k in ("Lqud_Rho_l", "RHO_l0", "rho_l0", "alpha1", "ALPHA1", "C_l", "c_l")
            )
            if not is_biquad:
                check_mat_law37(mat, log)
    for mid, mat37 in getattr(model, "mat_law37s", {}).items():
        if mid not in getattr(model, "materials", {}):
            params = getattr(mat37, "params", {}) or {}
            is_biquad = hasattr(mat37, "a") and hasattr(mat37, "c1") and not any(
                k in params for k in ("Lqud_Rho_l", "RHO_l0", "rho_l0", "alpha1", "ALPHA1", "C_l", "c_l")
            )
            if not is_biquad:
                check_mat_law37(mat37, log)

    # M541: Material LAW38 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (38, "38", "LAW38", "VISC_TAB") or getattr(mat, "law_name", None) in ("LAW38", "VISC_TAB"):
            check_mat_law38(mat, log)
    for mid, mat38 in getattr(model, "mat_law38s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law38(mat38, log)

    # M542: Material LAW32 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (32, "32", "LAW32", "HILL") or getattr(mat, "law_name", None) in ("32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL"):
            check_mat_law32(mat, log)
    for mid, mat32 in getattr(model, "mat_law32s", {}).items():
        if mid not in getattr(model, "materials", {}):
            if not hasattr(mat32, "fct_id11"):
                check_mat_law32(mat32, log)

    # M543: Material LAW25 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS") or getattr(mat, "law_name", None) in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", "MAT_LAW25", "MAT_COMP_PLAS", "MAT_COMPSH", "MAT_TSAI_WU", "MAT_CRASURV", "MAT_COMPOSITE_PLAS"):
            check_mat_law25(mat, log)
    for mid, mat25 in getattr(model, "mat_law25s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law25(mat25, log)

    # M544: Material LAW15 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG") or getattr(mat, "law_name", None) in ("15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", "MAT_LAW15", "MAT_CHANG", "MAT_PLAS_ANISO", "MAT_COMP_CHANG"):
            check_mat_law15(mat, log)
    for mid, mat15 in getattr(model, "mat_law15s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law15(mat15, log)

    # M545: Material LAW22 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (22, "22", "LAW22", "DAMA", "PLAS_DAMA") or getattr(mat, "law_name", None) in ("22", "LAW22", "DAMA", "PLAS_DAMA", "MAT_LAW22", "MAT_DAMA", "MAT_PLAS_DAMA"):
            check_mat_law22(mat, log)
    for mid, mat22 in getattr(model, "mat_law22s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law22(mat22, log)

    # M546: Material LAW12 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (12, "12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB") or getattr(mat, "law_name", None) in ("12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB", "MAT_LAW12", "MAT_3D_COMP", "MAT_COMP_3D", "MAT_3PARBI", "MAT_RAGAB"):
            check_mat_law12(mat, log)
    for mid, mat12 in getattr(model, "mat_law12s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law12(mat12, log)

    # M547: Material LAW14 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (14, "14", "LAW14", "COMPSO", "COMP_SOL") or getattr(mat, "law_name", None) in ("14", "LAW14", "COMPSO", "COMP_SOL", "MAT_LAW14", "MAT_COMPSO", "MAT_COMP_SOL"):
            check_mat_law14(mat, log)
    for mid, mat14 in getattr(model, "mat_law14s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law14(mat14, log)

    # M548: Material LAW43 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB") or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB"):
            check_mat_law43(mat, log)
    for mid, mat43 in getattr(model, "mat_law43s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law43(mat43, log)

    # M549: Material LAW82 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (82, "82", "LAW82", "OGDEN", "LAW82_OGDEN") or getattr(mat, "law_name", None) in ("82", "LAW82", "OGDEN", "LAW82_OGDEN", "MAT_LAW82", "MAT_OGDEN", "MAT_LAW82_OGDEN"):
            check_mat_law82(mat, log)
    for mid, mat82 in getattr(model, "mat_law82s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law82(mat82, log)

    # M550: Material LAW69 parameter validation
    funcs = getattr(model, "functions", None)
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYP_ELAS") or getattr(mat, "law_name", None) in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYP_ELAS", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "MAT_LAW69_HYP_ELAS"):
            check_mat_law69(mat, log, funcs)
    for mid, mat69 in getattr(model, "mat_law69s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law69(mat69, log, funcs)

    # M551: Material LAW60 parameter validation
    for mid, mat in getattr(model, "materials", {}).items():
        if getattr(mat, "law", None) in (60, "60", "LAW60", "PLAS_T3", "FABRIC", "MAT_PLAS_T3", "MAT_FABRIC") or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "FABRIC", "MAT_PLAS_T3", "MAT_FABRIC", "MAT_LAW60"):
            check_mat_law60(mat, log=log, model=model, functions=funcs)
    for mid, mat60 in getattr(model, "mat_law60s", {}).items():
        if mid not in getattr(model, "materials", {}):
            check_mat_law60(mat60, log=log, model=model, functions=funcs)


def check_model(model: Model, log: MessageLog) -> None:
    if model.numnod == 0:
        log.error("model has no nodes", "MODEL CHECK")
    if not any(True for _ in model.element_groups()):
        log.warning("model has no elements (deck may use only unported "
                    "element types)", "MODEL CHECK")

    # M546: LAW12 is 3D solid only, invalid for 2D formulations (hm_read_mat12.F:167)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (12, "12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB") or getattr(mat, "law_name", None) in ("12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB", "MAT_LAW12", "MAT_3D_COMP", "MAT_COMP_3D"):
                log.error(f"/MAT/LAW12/{mid}: LAW12 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat12 in getattr(model, "mat_law12s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW12/{mid}: LAW12 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # M547: LAW14 is 3D solid only, invalid for 2D formulations (hm_read_mat14.F:151)
    if getattr(model, "n2d", 0) > 0:
        for mid, mat in getattr(model, "materials", {}).items():
            if getattr(mat, "law", None) in (14, "14", "LAW14", "COMPSO", "COMP_SOL") or getattr(mat, "law_name", None) in ("14", "LAW14", "COMPSO", "COMP_SOL", "MAT_LAW14", "MAT_COMPSO", "MAT_COMP_SOL"):
                log.error(f"/MAT/LAW14/{mid}: LAW14 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")
        for mid, mat14 in getattr(model, "mat_law14s", {}).items():
            if mid not in getattr(model, "materials", {}):
                log.error(f"/MAT/LAW14/{mid}: LAW14 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)", "MAT CHECK")

    # material law vs element family compatibility (fail in the Starter
    # with a clear message instead of a NotImplementedError mid-run)
    for name, group in model.element_groups():
        allowed = _ALLOWED_LAWS.get(name)
        if allowed is None and name != "springs":
            continue
        for sl, mat, prop in group.state["slices"]:
            if name == "springs" and (getattr(mat, "id", 0) == 0 or getattr(mat, "law", 0) in (0, -1)):
                continue
            mat_law = getattr(mat, "law", None)
            if mat_law == 999 and getattr(mat, "eos", None) is None:
                # M37 pack 1: a /MAT/GAS on elements has no pressure or
                # stiffness of its own — the initial state must come
                # from an /EOS/IDEAL-GAS card (or the programmatic
                # P0/T0/RHO0 params); see materials/mat_gas.py.  Checked
                # before the density (a bare gas card has neither).
                log.error(f"/MAT/GAS/{getattr(mat, 'id', 0)} on {name} elements needs "
                          f"an /EOS/IDEAL-GAS card for its initial "
                          f"state (P0, gamma) — a bare gas card has no "
                          f"element pressure", "MAT CHECK")
            rho0 = getattr(mat, "rho0", 0.0) or 0.0
            if rho0 <= 0.0 and mat_law not in _NULL_RHO0_OK_LAWS:
                law_name = getattr(mat, "law_name", None) \
                    or f"LAW{mat_law}"
                log.error(f"/MAT/{law_name}/{mat.id} on {name} elements: "
                          f"zero or missing initial density "
                          f"(RHO0={rho0:g}) — element masses cannot be "
                          f"initialized", "MAT CHECK")
                continue
            if getattr(mat, "inactive", False):
                law_name = getattr(mat, "law_name", f"LAW{mat.law}")
                log.warning(f"/MAT/{law_name}/{mat.id} on {name} "
                            f"elements: parsed, physics not implemented "
                            f"(M37) — the Engine will refuse to run "
                            f"this model", "MAT CHECK")
                continue
            if (mat.law in (38, "38", "LAW38", "VISC_TAB")
                    or getattr(mat, "law_name", None) in ("LAW38", "VISC_TAB")):
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "trusses", "beams"):
                    log.error(
                        f"/MAT/LAW38/{mat.id} (/MAT/VISC_TAB) is not supported for {name} elements "
                        f"(solids only: bricks, tetras, penta6, pyra5)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (32, "32", "LAW32", "HILL")
                    or getattr(mat, "law_name", None) in ("32", "LAW32", "HILL", "MAT_LAW32", "MAT_HILL")):
                if name in ("bricks", "tetras", "penta6", "pyra5", "trusses", "beams"):
                    log.error(
                        f"/MAT/LAW32/{mat.id} (/MAT/HILL) is not supported for {name} elements "
                        f"(shells only: shells, shells_qbat, shells_qeph, sh3n)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (25, "25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS")
                    or getattr(mat, "law_name", None) in ("25", "LAW25", "COMP_PLAS", "COMPSH", "TSAI_WU", "CRASURV", "COMPOSITE_PLAS", "MAT_LAW25", "MAT_COMP_PLAS", "MAT_COMPSH", "MAT_TSAI_WU", "MAT_CRASURV", "MAT_COMPOSITE_PLAS")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW25/{mat.id} (/MAT/COMP_PLAS) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (15, "15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG")
                    or getattr(mat, "law_name", None) in ("15", "LAW15", "CHANG", "PLAS_ANISO", "COMP_CHANG", "MAT_LAW15", "MAT_CHANG", "MAT_PLAS_ANISO", "MAT_COMP_CHANG")):
                if name in ("bricks", "tetras", "penta6", "pyra5", "trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW15/{mat.id} (/MAT/CHANG) is not supported for {name} elements "
                        f"(shells only: shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (22, "22", "LAW22", "DAMA", "PLAS_DAMA")
                    or getattr(mat, "law_name", None) in ("22", "LAW22", "DAMA", "PLAS_DAMA", "MAT_LAW22", "MAT_DAMA", "MAT_PLAS_DAMA")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW22/{mat.id} (/MAT/DAMA) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (12, "12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB")
                    or getattr(mat, "law_name", None) in ("12", "LAW12", "3PARBI", "3D_COMP", "COMP_3D", "RAGAB", "MAT_LAW12", "MAT_3D_COMP", "MAT_COMP_3D", "MAT_3PARBI", "MAT_RAGAB")):
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW12/{mat.id} (/MAT/3D_COMP) is not supported for {name} elements "
                        f"(solids only: bricks, tetras, penta6, pyra5)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (14, "14", "LAW14", "COMPSO", "COMP_SOL")
                    or getattr(mat, "law_name", None) in ("14", "LAW14", "COMPSO", "COMP_SOL", "MAT_LAW14", "MAT_COMPSO", "MAT_COMP_SOL")):
                if name in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW14/{mat.id} (/MAT/COMPSO) is not supported for {name} elements "
                        f"(solids only: bricks, tetras, penta6, pyra5)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (43, "43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB")
                    or getattr(mat, "law_name", None) in ("43", "LAW43", "HILL_TAB", "HILL_PLAS_TAB", "LAW43_HILL_TAB", "MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW43/{mat.id} (/MAT/HILL_TAB) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (82, "82", "LAW82", "OGDEN", "LAW82_OGDEN")
                    or getattr(mat, "law_name", None) in ("82", "LAW82", "OGDEN", "LAW82_OGDEN", "MAT_LAW82", "MAT_OGDEN", "MAT_LAW82_OGDEN")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW82/{mat.id} (/MAT/OGDEN) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYP_ELAS")
                    or getattr(mat, "law_name", None) in ("69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYP_ELAS", "MAT_LAW69", "MAT_HYP_ELAS", "MAT_HYPERELASTIC", "MAT_LAW69_HYP_ELAS")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW69/{mat.id} (/MAT/HYP_ELAS) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if (mat.law in (60, "60", "LAW60", "PLAS_T3", "FABRIC", "MAT_PLAS_T3", "MAT_FABRIC")
                    or getattr(mat, "law_name", None) in ("60", "LAW60", "PLAS_T3", "FABRIC", "MAT_PLAS_T3", "MAT_FABRIC", "MAT_LAW60")):
                if name in ("trusses", "beams", "springs"):
                    log.error(
                        f"/MAT/LAW60/{mat.id} (/MAT/PLAS_T3) is not supported for {name} elements "
                        f"(solids and shells only: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads)",
                        "MAT CHECK",
                    )
                    continue
            if allowed is None:
                continue
            if mat.law not in allowed:
                log.error(f"material LAW{mat.law} (/MAT {mat.id}) is not "
                          f"ported for {name} elements (supported: "
                          f"{sorted(allowed, key=str)})", "MAT CHECK")
            if getattr(mat, "fail", None) is not None and name in ("trusses", "springs",
                                                 "beams"):
                log.warning(f"/FAIL on /MAT {getattr(mat, 'id', 0)} is ignored for {name} "
                            f"(failure is ported for solids and shells)",
                            "MAT CHECK")

    check_materials(model, log)



    # M38: element groups that reference a parsed-but-not-implemented
    # PROPERTY (InactiveProperty) — the Starter accepts them (params +
    # safe geometry read, mass init works); the Engine refuses to run the
    # model (see prop_reader.refuse_inactive_properties), mirroring the
    # inactive-material warning above.
    for name, group in model.element_groups():
        for sl, mat, prop in group.state["slices"]:
            if getattr(prop, "inactive", False):
                pn = getattr(prop, "prop_name", None) or f"TYPE{prop.type}"
                log.warning(f"/PROP/{pn}/{prop.id} on {name} elements: "
                            f"parsed, physics not implemented (M38) — the "
                            f"Engine will refuse to run this model",
                            "PROP CHECK")

    def need_group(gid, who):
        if gid is not None and gid != 0 and gid not in model.node_groups:
            log.error(f"{who}: node group {gid} not defined", "CROSS REF")

    def need_funct(fid, who):
        if fid not in model.functions:
            log.error(f"{who}: function {fid} not defined", "CROSS REF")

    for bc in model.bcs:
        need_group(bc.grnod_id, f"/BCS/{bc.id}")
    for iv in model.inivel:
        need_group(iv.grnod_id, f"/INIVEL/{iv.id}")
    for gv in model.gravity:
        need_group(gv.grnod_id, f"/GRAV/{gv.id}")
        need_funct(gv.funct_id, f"/GRAV/{gv.id}")
    for cl in model.cloads:
        need_group(cl.grnod_id, f"/CLOAD/{cl.id}")
        need_funct(cl.funct_id, f"/CLOAD/{cl.id}")
    for imp in model.impvel:
        need_group(imp.grnod_id, f"/IMPVEL/{imp.id}")
        need_funct(imp.funct_id, f"/IMPVEL/{imp.id}")
    for imp in model.impdisp:
        need_group(imp.grnod_id, f"/IMPDISP/{imp.id}")
        need_funct(imp.funct_id, f"/IMPDISP/{imp.id}")
    for imp in model.impacc:
        need_group(imp.grnod_id, f"/IMPACC/{imp.id}")
        need_funct(imp.funct_id, f"/IMPACC/{imp.id}")
    for it in model.imptemp:
        need_group(it.grnod_id, f"/IMPTEMP/{it.id}")
        need_funct(it.funct_id, f"/IMPTEMP/{it.id}")
    for cl in model.centri_loads:
        need_group(cl.grnod_id, f"/LOAD/CENTRI/{cl.id}")
        need_funct(cl.funct_id, f"/LOAD/CENTRI/{cl.id}")
    for pl in model.ploads:
        need_funct(pl.funct_id, f"/PLOAD/{pl.id}")
        if pl.surf_id not in model.surfaces:
            log.error(f"/PLOAD/{pl.id}: surface {pl.surf_id} not defined",
                      "CROSS REF")
    for conv in model.convec_loads:
        need_funct(conv.funct_id, f"/CONVEC/{conv.id}")
        if conv.surf_id not in model.surfaces:
            log.error(f"/CONVEC/{conv.id}: surface {conv.surf_id} not defined",
                      "CROSS REF")
    for rad in model.radiation_loads:
        if rad.funct_id:
            need_funct(rad.funct_id, f"/RADIATION/{rad.id}")
        if rad.surf_id not in model.surfaces:
            log.error(f"/RADIATION/{rad.id}: surface {rad.surf_id} not defined",
                      "CROSS REF")
    for fl in model.impflux_loads:
        if fl.funct_id:
            need_funct(fl.funct_id, f"/IMPFLUX/{fl.id}")
        if fl.surf_id and fl.surf_id not in model.surfaces:
            log.error(f"/IMPFLUX/{fl.id}: surface {fl.surf_id} not defined",
                      "CROSS REF")
        if fl.grbric_id and ("bricks", fl.grbric_id) not in model.element_groups:
            log.error(f"/IMPFLUX/{fl.id}: brick group {fl.grbric_id} not defined",
                      "CROSS REF")
    for itemp in model.initemp:
        if itemp.grnod_id:
            need_group(itemp.grnod_id, f"/INITEMP/{itemp.id}")
    for iv in model.inivol:
        if iv.part_id and iv.part_id not in model.parts:
            log.error(f"/INIVOL/{iv.id}: part {iv.part_id} not defined",
                      "CROSS REF")
        for c in iv.containers:
            if c.surf_id not in model.surfaces and c.surf_id not in model.node_groups:
                log.error(f"/INIVOL/{iv.id}: container surface/group {c.surf_id} not defined",
                          "CROSS REF")
    solid_ids = set()
    for name in ("bricks", "tetra4", "tetra10"):
        grp = getattr(model, name, None)
        if grp is not None and hasattr(grp, "ids") and len(grp.ids) > 0:
            solid_ids.update(grp.ids.tolist())
    for elem_id in model.ini_bricks:
        if elem_id not in solid_ids:
            log.error(f"/INIBRI: solid element {elem_id} not defined",
                      "CROSS REF")

    shell_ids = set()
    for name in ("shells", "sh3n", "quads"):
        grp = getattr(model, name, None)
        if grp is not None and hasattr(grp, "ids") and len(grp.ids) > 0:
            shell_ids.update(grp.ids.tolist())
    for elem_id in model.ini_shells:
        if elem_id not in shell_ids:
            log.error(f"/INISHE: shell element {elem_id} not defined",
                      "CROSS REF")

    truss_ids = set(model.trusses.ids.tolist()) if model.trusses is not None and hasattr(model.trusses, "ids") and len(model.trusses.ids) > 0 else set()
    for elem_id in model.ini_trusses:
        if elem_id not in truss_ids:
            log.error(f"/INITRU: truss element {elem_id} not defined",
                      "CROSS REF")

    beam_ids = set(model.beams.ids.tolist()) if model.beams is not None and hasattr(model.beams, "ids") and len(model.beams.ids) > 0 else set()
    for elem_id in model.ini_beams:
        if elem_id not in beam_ids:
            log.error(f"/INIBEA: beam element {elem_id} not defined",
                      "CROSS REF")

    spring_ids = set(model.springs.ids.tolist()) if model.springs is not None and hasattr(model.springs, "ids") and len(model.springs.ids) > 0 else set()
    for elem_id in model.ini_springs:
        if elem_id not in spring_ids:
            log.error(f"/INISPR: spring element {elem_id} not defined",
                      "CROSS REF")

    for sens in model.sensors:
        who = f"/SENSOR/{sens.kind}/{sens.id}"
        if sens.kind in ("DISP", "VEL") and sens.node_id and sens.node_id not in model._id2idx:
            log.error(f"{who}: unknown node {sens.node_id}", "CROSS REF")
        elif sens.kind == "DIST":
            if sens.node_id1 and sens.node_id1 not in model._id2idx:
                log.error(f"{who}: unknown node 1 {sens.node_id1}", "CROSS REF")
            if sens.node_id2 and sens.node_id2 not in model._id2idx:
                log.error(f"{who}: unknown node 2 {sens.node_id2}", "CROSS REF")
        elif sens.kind == "ENERGY" and sens.part_id and sens.part_id not in model.parts:
            log.error(f"{who}: unknown part {sens.part_id}", "CROSS REF")
        elif sens.kind == "TEMP" and sens.grnod_id:
            need_group(sens.grnod_id, who)
        elif sens.kind == "GAUGE":
            for gid, _, _ in sens.gauge_entries:
                if gid > 0 and gid not in model.gauge_points and gid not in model.gauges:
                    log.error(f"{who}: unknown gauge {gid}", "CROSS REF")
        elif sens.kind == "HIC":
            if sens.accel_id > 0 and sens.accel_id not in model.accelerometers:
                log.error(f"{who}: unknown accelerometer {sens.accel_id}", "CROSS REF")
        elif sens.kind == "WORK":
            if sens.node_id1 and sens.node_id1 not in model._id2idx:
                log.error(f"{who}: unknown node 1 {sens.node_id1}", "CROSS REF")
            if sens.node_id2 and sens.node_id2 not in model._id2idx:
                log.error(f"{who}: unknown node 2 {sens.node_id2}", "CROSS REF")
        elif sens.kind == "RWALL":
            rw_ids = {rw.id for rw in model.rwalls}
            if sens.rwall_id > 0 and sens.rwall_id not in rw_ids:
                log.error(f"{who}: unknown rigid wall {sens.rwall_id}", "CROSS REF")
        elif sens.kind in ("XSECTION", "CROSSSECTION", "SECT"):
            sect_ids = {s.id for s in model.sections}
            if sens.sect_id > 0 and sens.sect_id not in sect_ids:
                log.error(f"{who}: unknown section {sens.sect_id}", "CROSS REF")
        elif sens.kind == "DIST_SURF":
            if sens.node_id1 and sens.node_id1 not in model._id2idx:
                log.error(f"{who}: unknown node 1 {sens.node_id1}", "CROSS REF")

    for am in model.admas:
        # mass_type 0/1: grnod_id is a node group; 2: surface; 3/4: part
        # group; 6/7: single part.  Only types 0/1 cross-ref node_groups.
        if am.mass_type in (0, 1):
            need_group(am.grnod_id, f"/ADMAS/{am.id}")
    for rb in model.rbodies:
        need_group(rb.grnod_id, f"/{rb.kind}/{rb.id}")
        if rb.master_id not in model._id2idx:
            log.error(f"/{rb.kind}/{rb.id}: unknown master node "
                      f"{rb.master_id}", "CROSS REF")
    for r3 in model.rbe3:
        need_group(r3.grnod_id, f"/RBE3/{r3.id}")
        if r3.ref_id not in model._id2idx:
            log.error(f"/RBE3/{r3.id}: unknown reference node {r3.ref_id}",
                      "CROSS REF")
    for sc in model.sections:
        need_group(sc.grnod_id, f"/SECT/{sc.id}")
        if sc.node_id_ref and sc.node_id_ref not in model._id2idx:
            log.error(f"/SECT/{sc.id}: unknown reference node "
                      f"{sc.node_id_ref}", "CROSS REF")
    for rw in model.rwalls:
        need_group(rw.grnod_id, f"/RWALL/{rw.id}")
        if rw.grnod_id2:
            need_group(rw.grnod_id2, f"/RWALL/{rw.id}")
        if rw.node_id and rw.node_id not in model._id2idx:
            log.error(f"/RWALL/{rw.id}: unknown wall node {rw.node_id}",
                      "CROSS REF")
    for itf in model.interfaces:
        who = f"/INTER/TYPE{itf.type}/{itf.id}"
        if itf.type in (7, 2):
            # TYPE7 allows grnod_id = 0 (self-impact: secondary side
            # defaults to the main surface's own nodes); TYPE2 does not.
            if itf.type == 2 or itf.grnod_id != 0:
                need_group(itf.grnod_id, who)
            if itf.type == 2 and itf.grnod_id == 0:
                log.error(f"{who}: a tied interface needs a secondary "
                          f"node group", "CROSS REF")
            if itf.surf_id not in model.surfaces:
                log.error(f"{who}: surface {itf.surf_id} not defined",
                          "CROSS REF")
        elif itf.type == 11:
            for lid in (itf.line_id1, itf.line_id2):
                if lid not in model.lines:
                    log.error(f"{who}: line {lid} not defined", "CROSS REF")
        elif itf.type == 24:
            if itf.grnod_id != 0:
                need_group(itf.grnod_id, who)
            for sid in (itf.surf_id1, itf.surf_id):
                if sid != 0 and sid not in model.surfaces:
                    log.error(f"{who}: surface {sid} not defined", "CROSS REF")
        elif itf.type in (1, 3, 6, 12, 15, 20, 21, 23):
            for sid in (itf.surf_id, itf.surf_id1):
                if sid > 0 and sid not in model.surfaces:
                    log.error(f"{who}: surface {sid} not defined", "CROSS REF")
            if itf.grnod_id > 0 and itf.grnod_id not in model.node_groups:
                log.error(f"{who}: node group {itf.grnod_id} not defined", "CROSS REF")
        elif itf.type in (5, 14):
            if itf.grnod_id > 0 and itf.grnod_id not in model.node_groups:
                log.error(f"{who}: node group {itf.grnod_id} not defined", "CROSS REF")
            if itf.surf_id > 0 and itf.surf_id not in model.surfaces:
                log.error(f"{who}: surface {itf.surf_id} not defined", "CROSS REF")
        elif itf.type == 22:
            if itf.surf_id > 0 and itf.surf_id not in model.surfaces:
                log.error(f"{who}: surface {itf.surf_id} not defined", "CROSS REF")
    for th in model.th_requests:
        if th.kind == "NODE":
            for nid in th.ids:
                if nid not in model._id2idx:
                    log.error(f"/TH/NODE/{th.id}: unknown node {nid}",
                              "CROSS REF")
        elif th.kind == "PART":
            for pid in th.ids:
                if pid not in model.parts:
                    log.error(f"/TH/PART/{th.id}: unknown part {pid}",
                              "CROSS REF")
        elif th.kind == "SECT":
            defined = {s.id for s in model.sections}
            for sid in th.ids:
                if sid not in defined:
                    log.error(f"/TH/SECT/{th.id}: unknown section {sid}",
                              "CROSS REF")

    # Cyclic boundary conditions (M99)
    for cid, cb in getattr(model, "cyclic_bcs", {}).items():
        if cb.grnod1_id not in model.node_groups:
            log.error(f"/BCS/CYCLIC/{cid}: node group 1 {cb.grnod1_id} not defined", "CROSS REF")
        if cb.grnod2_id not in model.node_groups:
            log.error(f"/BCS/CYCLIC/{cid}: node group 2 {cb.grnod2_id} not defined", "CROSS REF")
        if cb.skew_id > 0 and cb.skew_id not in model.skews:
            log.error(f"/BCS/CYCLIC/{cid}: skew {cb.skew_id} not defined", "CROSS REF")

    # Solid part perturbations (M99)
    part_groups = model.egroups.get("PART", {})
    for pid, pt in getattr(model, "perturbations", {}).items():
        if pt.grpart_id > 0 and pt.grpart_id not in part_groups and pt.grpart_id not in model.parts:
            log.error(f"/PERTURB/PART/SOLID/{pid}: part group {pt.grpart_id} not defined", "CROSS REF")

    # Blast loads (M99)
    for bid, pb in getattr(model, "pblast_loads", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{bid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.node_id > 0 and pb.node_id not in model._id2idx:
            log.error(f"/LOAD/PBLAST/{bid}: detonation node {pb.node_id} not defined", "CROSS REF")

    # Plies & Laminates (M100)
    for ply_id, ply in getattr(model, "plies", {}).items():
        if ply.mat_id > 0 and ply.mat_id not in model.materials:
            log.error(f"/PLY/{ply_id}: material {ply.mat_id} not defined", "CROSS REF")
        if ply.skew_id > 0 and ply.skew_id not in model.skews:
            log.error(f"/PLY/{ply_id}: skew {ply.skew_id} not defined", "CROSS REF")

    for lam_id, lam in getattr(model, "laminates", {}).items():
        for lp in lam.plies:
            if lp.ply_id not in model.plies:
                log.error(f"/LAMINATE/{lam_id}: ply {lp.ply_id} not defined", "CROSS REF")
            if lp.mat_interply > 0 and lp.mat_interply not in model.materials:
                log.error(f"/LAMINATE/{lam_id}: interply material {lp.mat_interply} not defined", "CROSS REF")

    # Sub-interfaces (M100)
    inter_ids = {itf.id for itf in model.interfaces}
    for sub in getattr(model, "sub_interfaces", []):
        if sub.inter_id not in inter_ids:
            log.error(f"/INTER/SUB/{sub.id}: main interface {sub.inter_id} not defined", "CROSS REF")
        if sub.main_id1 > 0 and sub.main_id1 not in model.surfaces and sub.main_id1 not in model.lines:
            log.error(f"/INTER/SUB/{sub.id}: main entity 1 {sub.main_id1} not defined", "CROSS REF")
        if sub.main_id2 > 0 and sub.main_id2 not in model.surfaces:
            log.error(f"/INTER/SUB/{sub.id}: main entity 2 {sub.main_id2} not defined", "CROSS REF")

    # Guided cables (M149)
    for gcid, gc in getattr(model, "guided_cables", {}).items():
        if gc.grnod_id > 0 and gc.grnod_id not in model.node_groups:
            log.error(f"/INTER/GUIDED_CABLE/{gcid}: node group {gc.grnod_id} not defined", "CROSS REF")
        if gc.grpart_id > 0 and gc.grpart_id not in model.egroups.get("PART", {}) and gc.grpart_id not in model.parts:
            log.error(f"/INTER/GUIDED_CABLE/{gcid}: part group {gc.grpart_id} not defined", "CROSS REF")

    # Composite properties (M100)
    for prop_id, prop in model.properties.items():
        if prop.type in (10, 11, 16, 6) and hasattr(prop, "params"):
            sk = prop.params.get("skew_id", 0)
            if sk > 0 and sk not in model.skews:
                log.error(f"/PROP/TYPE{prop.type}/{prop_id}: skew {sk} not defined", "CROSS REF")
        if prop.type == 11 and hasattr(prop, "params"):
            for ly in prop.params.get("layers", []):
                mid = ly.get("mat_id", 0)
                if mid > 0 and mid not in model.materials:
                    log.error(f"/PROP/TYPE11/{prop_id}: layer material {mid} not defined", "CROSS REF")

    # Shell and Failure Perturbations & SMS (M101)
    part_groups = model.egroups.get("PART", {})
    for pid, ps in getattr(model, "perturb_shells", {}).items():
        if ps.grpart_id > 0 and ps.grpart_id not in part_groups and ps.grpart_id not in model.parts:
            log.error(f"/PERTURB/PART/SHELL/{pid}: part group/part {ps.grpart_id} not defined", "CROSS REF")

    fail_ids = {mat_id for mat_id, _, _ in getattr(model, "raw_fails", [])}
    for m in model.materials.values():
        if getattr(m, "failure", None) is not None:
            fail_ids.add(m.id)
    for pid, pf in getattr(model, "perturb_fails", {}).items():
        if pf.fail_id > 0 and pf.fail_id not in fail_ids:
            log.error(f"/PERTURB/FAIL/{pf.fail_type}/{pid}: failure criterion {pf.fail_id} not defined", "CROSS REF")

    if getattr(model, "sms_global", None) is not None:
        sms = model.sms_global
        if sms.grpart_id > 0 and sms.grpart_id not in part_groups and sms.grpart_id not in model.parts:
            log.error(f"/SMS: part group/part {sms.grpart_id} not defined", "CROSS REF")

    # Boundary conditions, Joints, Merge, Inicrack, Laser (M102)
    for bid, bcs in getattr(model, "bcs_nrf", {}).items():
        if bcs.grnod_id > 0 and bcs.grnod_id not in model.node_groups:
            log.error(f"/BCS/NRF/{bid}: node group {bcs.grnod_id} not defined", "CROSS REF")

    sensor_ids = {s.id for s in model.sensors}
    for bid, bcs in getattr(model, "bcs_walls", {}).items():
        if bcs.grnod_id > 0 and bcs.grnod_id not in model.node_groups:
            log.error(f"/BCS/WALL/{bid}: node group {bcs.grnod_id} not defined", "CROSS REF")
        if bcs.sensor_id > 0 and bcs.sensor_id not in sensor_ids:
            log.error(f"/BCS/WALL/{bid}: sensor {bcs.sensor_id} not defined", "CROSS REF")

    for rid, rl in getattr(model, "rlinks", {}).items():
        if rl.grnod_id > 0 and rl.grnod_id not in model.node_groups:
            log.error(f"/RLINK/{rid}: node group {rl.grnod_id} not defined", "CROSS REF")
        if rl.skew_id > 0 and rl.skew_id not in model.skews:
            log.error(f"/RLINK/{rid}: skew {rl.skew_id} not defined", "CROSS REF")

    for cid, cj in getattr(model, "cyl_joints", {}).items():
        if cj.node_id1 > 0 and cj.node_id1 not in model._id2idx:
            log.error(f"/CYL_JOINT/{cid}: node {cj.node_id1} not defined", "CROSS REF")
        if cj.node_id2 > 0 and cj.node_id2 not in model._id2idx:
            log.error(f"/CYL_JOINT/{cid}: node {cj.node_id2} not defined", "CROSS REF")
        if cj.grnod_id > 0 and cj.grnod_id not in model.node_groups:
            log.error(f"/CYL_JOINT/{cid}: node group {cj.grnod_id} not defined", "CROSS REF")

    for gid, gj in getattr(model, "gjoints", {}).items():
        for nid in (gj.node_id0, gj.node_id1, gj.node_id2, gj.node_id3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/GJOINT/{gid}: node {nid} not defined", "CROSS REF")

    for mid, mn in getattr(model, "node_merges", {}).items():
        if mn.grnod_id > 0 and mn.grnod_id not in model.node_groups:
            log.error(f"/MERGE/NODE/{mid}: node group {mn.grnod_id} not defined", "CROSS REF")

    for iid, ic in getattr(model, "inicracks", {}).items():
        for seg in ic.segments:
            if seg.node_id1 > 0 and seg.node_id1 not in model._id2idx:
                log.error(f"/INICRACK/{iid}: node {seg.node_id1} not defined", "CROSS REF")
            if seg.node_id2 > 0 and seg.node_id2 not in model._id2idx:
                log.error(f"/INICRACK/{iid}: node {seg.node_id2} not defined", "CROSS REF")

    for lid, las in getattr(model, "laser_loads", {}).items():
        if las.curve_id > 0 and las.curve_id not in model.functions:
            log.error(f"/LASER/{lid}: function {las.curve_id} not defined", "CROSS REF")
        if las.fct_id_target > 0 and las.fct_id_target not in model.functions:
            log.error(f"/LASER/{lid}: function {las.fct_id_target} not defined", "CROSS REF")

    # Specialized loads, Preload, Damping & Controls (M103)
    for lid, pcyl in getattr(model, "pcyl_loads", {}).items():
        if pcyl.surf_id > 0 and pcyl.surf_id not in model.surfaces:
            log.error(f"/LOAD/PCYL/{lid}: surface {pcyl.surf_id} not defined", "CROSS REF")
        if pcyl.sens_id > 0 and pcyl.sens_id not in sensor_ids:
            log.error(f"/LOAD/PCYL/{lid}: sensor {pcyl.sens_id} not defined", "CROSS REF")
        if pcyl.frame_id > 0 and pcyl.frame_id not in model.skews:
            log.error(f"/LOAD/PCYL/{lid}: skew {pcyl.frame_id} not defined", "CROSS REF")
        if pcyl.table_id > 0 and pcyl.table_id not in model.tables:
            log.error(f"/LOAD/PCYL/{lid}: table {pcyl.table_id} not defined", "CROSS REF")

    for lid, pf in getattr(model, "pfluid_loads", {}).items():
        if pf.surf_id > 0 and pf.surf_id not in model.surfaces:
            log.error(f"/LOAD/PFLUID/{lid}: surface {pf.surf_id} not defined", "CROSS REF")
        if pf.sens_id > 0 and pf.sens_id not in sensor_ids:
            log.error(f"/LOAD/PFLUID/{lid}: sensor {pf.sens_id} not defined", "CROSS REF")
        if pf.fct_id_t > 0 and pf.fct_id_t not in model.functions:
            log.error(f"/LOAD/PFLUID/{lid}: function {pf.fct_id_t} not defined", "CROSS REF")
        if pf.fct_id_pc > 0 and pf.fct_id_pc not in model.functions:
            log.error(f"/LOAD/PFLUID/{lid}: function {pf.fct_id_pc} not defined", "CROSS REF")
        if pf.fct_id_vel > 0 and pf.fct_id_vel not in model.functions:
            log.error(f"/LOAD/PFLUID/{lid}: function {pf.fct_id_vel} not defined", "CROSS REF")
        if pf.frame_id > 0 and pf.frame_id not in model.skews:
            log.error(f"/LOAD/PFLUID/{lid}: skew {pf.frame_id} not defined", "CROSS REF")
        if pf.frame_id_vel > 0 and pf.frame_id_vel not in model.skews:
            log.error(f"/LOAD/PFLUID/{lid}: skew {pf.frame_id_vel} not defined", "CROSS REF")

    for pid, pr in getattr(model, "preloads", {}).items():
        if pr.sect_id > 0 and pr.sect_id not in model.sections and pr.sect_id not in model.properties:
            log.error(f"/PRELOAD/{pid}: section {pr.sect_id} not defined", "CROSS REF")
        if pr.sens_id > 0 and pr.sens_id not in sensor_ids:
            log.error(f"/PRELOAD/{pid}: sensor {pr.sens_id} not defined", "CROSS REF")
        if pr.fct_id > 0 and pr.fct_id not in model.functions:
            log.error(f"/PRELOAD/{pid}: function {pr.fct_id} not defined", "CROSS REF")

    for pid, pra in getattr(model, "preload_axials", {}).items():
        if pra.set_id > 0 and pra.set_id not in part_groups and pra.set_id not in model.parts:
            log.error(f"/PRELOAD/AXIAL/{pid}: part group/part {pra.set_id} not defined", "CROSS REF")
        if pra.sens_id > 0 and pra.sens_id not in sensor_ids:
            log.error(f"/PRELOAD/AXIAL/{pid}: sensor {pra.sens_id} not defined", "CROSS REF")
        if pra.fun_id > 0 and pra.fun_id not in model.functions:
            log.error(f"/PRELOAD/AXIAL/{pid}: function {pra.fun_id} not defined", "CROSS REF")

    for did, di in getattr(model, "damp_inters", {}).items():
        if di.grnod_id > 0 and di.grnod_id not in model.node_groups:
            log.error(f"/DAMP/INTER/{did}: node group {di.grnod_id} not defined", "CROSS REF")
        if di.skew_id > 0 and di.skew_id not in model.skews:
            log.error(f"/DAMP/INTER/{did}: skew {di.skew_id} not defined", "CROSS REF")

    for did, dr in getattr(model, "damp_ranges", {}).items():
        if dr.grpart_id > 0 and dr.grpart_id not in part_groups and dr.grpart_id not in model.parts:
            log.error(f"/DAMP/RANGE/{did}: part group/part {dr.grpart_id} not defined", "CROSS REF")

    for cid, caa in getattr(model, "caa_controls", {}).items():
        if caa.surf_id > 0 and caa.surf_id not in model.surfaces:
            log.error(f"/CAA/{cid}: surface {caa.surf_id} not defined", "CROSS REF")
        if caa.grnod_id > 0 and caa.grnod_id not in model.node_groups:
            log.error(f"/CAA/{cid}: node group {caa.grnod_id} not defined", "CROSS REF")
        if caa.sens_id > 0 and caa.sens_id not in sensor_ids:
            log.error(f"/CAA/{cid}: sensor {caa.sens_id} not defined", "CROSS REF")

    # Gauges, Clusters, Ext Links, Flexible Bodies & Initial Fields (M104)
    for gid, g in getattr(model, "gauges", {}).items():
        if g.node_id > 0 and g.node_id not in model._id2idx:
            log.error(f"/GAUGE/{gid}: node {g.node_id} not defined", "CROSS REF")

    for cid, c in getattr(model, "clusters", {}).items():
        if c.skew_id > 0 and c.skew_id not in model.skews:
            log.error(f"/CLUSTER/{cid}: skew {c.skew_id} not defined", "CROSS REF")

    for lid, el in getattr(model, "ext_links", {}).items():
        if el.grnod_id > 0 and el.grnod_id not in model.node_groups:
            log.error(f"/EXTLNK/{lid}: node group {el.grnod_id} not defined", "CROSS REF")

    for fid, fx in getattr(model, "fxbodies", {}).items():
        if fx.node_id > 0 and fx.node_id not in model._id2idx:
            log.error(f"/FXBODY/{fid}: node {fx.node_id} not defined", "CROSS REF")

    grav_ids = {g.id for g in model.gravity}
    for iid, ig in getattr(model, "ini_gravs", {}).items():
        if ig.grpart_id > 0 and ig.grpart_id not in part_groups and ig.grpart_id not in model.parts:
            log.error(f"/INIGRAV/{iid}: part group/part {ig.grpart_id} not defined", "CROSS REF")
        if ig.surf_id > 0 and ig.surf_id not in model.surfaces:
            log.error(f"/INIGRAV/{iid}: surface {ig.surf_id} not defined", "CROSS REF")
        if ig.grav_id > 0 and ig.grav_id not in grav_ids:
            log.error(f"/INIGRAV/{iid}: gravity {ig.grav_id} not defined", "CROSS REF")

    for mid, m1 in getattr(model, "ini_map1ds", {}).items():
        for nid in (m1.node_id1, m1.node_id2):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/INIMAP1D/{mid}: node {nid} not defined", "CROSS REF")

    for mid, m2 in getattr(model, "ini_map2ds", {}).items():
        for nid in (m2.node_id1, m2.node_id2, m2.node_id3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/INIMAP2D/{mid}: node {nid} not defined", "CROSS REF")

    # Monitored Volumes, Airbag Leakage & ALE Controls (M105)
    for mid, mp in getattr(model, "monvol_pres", {}).items():
        if mp.surf_id > 0 and mp.surf_id not in model.surfaces:
            log.error(f"/MONVOL/PRES/{mid}: surface {mp.surf_id} not defined", "CROSS REF")
        if mp.fct_id > 0 and mp.fct_id not in model.functions:
            log.error(f"/MONVOL/PRES/{mid}: function {mp.fct_id} not defined", "CROSS REF")

    for mid, mg in getattr(model, "monvol_gases", {}).items():
        if mg.surf_id > 0 and mg.surf_id not in model.surfaces:
            log.error(f"/MONVOL/GAS/{mid}: surface {mg.surf_id} not defined", "CROSS REF")

    for mid, mc in getattr(model, "monvol_commus", {}).items():
        if mc.surf_id > 0 and mc.surf_id not in model.surfaces:
            log.error(f"/MONVOL/COMMU1/{mid}: surface {mc.surf_id} not defined", "CROSS REF")
        if mc.mat_id > 0 and mc.mat_id not in model.materials:
            log.error(f"/MONVOL/COMMU1/{mid}: material {mc.mat_id} not defined", "CROSS REF")

    for mid, ml in getattr(model, "monvol_lfluids", {}).items():
        if ml.surf_id > 0 and ml.surf_id not in model.surfaces:
            log.error(f"/MONVOL/LFLUID/{mid}: surface {ml.surf_id} not defined", "CROSS REF")
        for fid in (ml.fct_k, ml.fct_mtin, ml.fct_mtout, ml.fct_mpout, ml.fct_padd, ml.fct_pmax):
            if fid > 0 and fid not in model.functions:
                log.error(f"/MONVOL/LFLUID/{mid}: function {fid} not defined", "CROSS REF")

    for lid, lm in getattr(model, "leak_mats", {}).items():
        for fid in (lm.fct_id_e, lm.fct_id_lc, lm.fct_id_ac):
            if fid > 0 and fid not in model.functions:
                log.error(f"/LEAK/{lid}: function {fid} not defined", "CROSS REF")

    for lid, al in getattr(model, "ale_links", {}).items():
        if al.grnod_id > 0 and al.grnod_id not in model.node_groups:
            log.error(f"/ALE/LINK/{lid}: node group {al.grnod_id} not defined", "CROSS REF")
        if al.fct_id > 0 and al.fct_id not in model.functions:
            log.error(f"/ALE/LINK/{lid}: function {al.fct_id} not defined", "CROSS REF")

    # Seatbelts Suite: /RETRACTOR & /SLIPRING (M106)
    sensor_ids = {s.id for s in model.sensors}
    for rid, ret in getattr(model, "retractors", {}).items():
        if ret.node_id > 0 and ret.node_id not in model._id2idx:
            log.error(f"/RETRACTOR/{rid}: node {ret.node_id} not defined", "CROSS REF")
        if ret.sens_id1 > 0 and ret.sens_id1 not in sensor_ids:
            log.error(f"/RETRACTOR/{rid}: sensor {ret.sens_id1} not defined", "CROSS REF")
        if ret.sens_id2 > 0 and ret.sens_id2 not in sensor_ids:
            log.error(f"/RETRACTOR/{rid}: sensor {ret.sens_id2} not defined", "CROSS REF")
        for fid in (ret.fct_id1, ret.fct_id2, ret.fct_id3):
            if fid > 0 and fid not in model.functions:
                log.error(f"/RETRACTOR/{rid}: function {fid} not defined", "CROSS REF")

    for sid, sr in getattr(model, "sliprings", {}).items():
        if sr.subtype == "SPRING":
            for nid in (sr.node_id, sr.node_id2):
                if nid > 0 and nid not in model._id2idx:
                    log.error(f"/SLIPRING/{sid}: node {nid} not defined", "CROSS REF")
        elif sr.subtype == "SHELL":
            if sr.node_id > 0 and sr.node_id not in model.node_groups:
                log.error(f"/SLIPRING/{sid}: node group {sr.node_id} not defined", "CROSS REF")
        if sr.sens_id > 0 and sr.sens_id not in sensor_ids:
            log.error(f"/SLIPRING/{sid}: sensor {sr.sens_id} not defined", "CROSS REF")
        for fid in (sr.fct_id1, sr.fct_id2, sr.fct_id3, sr.fct_id4):
            if fid > 0 and fid not in model.functions:
                log.error(f"/SLIPRING/{sid}: function {fid} not defined", "CROSS REF")

    # M107: Advanced Failure Criteria & SENSOR/NIC
    for mat_id, fm, src in getattr(model, "raw_fails", []):
        if fm.type == "SAHRAEI":
            for fid_key in ("fct_ratio", "fct_elsize"):
                fid = fm.params.get(fid_key, 0)
                if fid > 0 and fid not in model.functions:
                    log.error(f"/FAIL/SAHRAEI on MAT/{mat_id}: function {fid} not defined", "CROSS REF")
        elif fm.type == "TAB2":
            for fid_key in ("epsf_id", "fct_exp"):
                fid = fm.params.get(fid_key, 0)
                if fid > 0 and fid not in model.functions:
                    log.error(f"/FAIL/TAB2 on MAT/{mat_id}: function {fid} not defined", "CROSS REF")
        elif fm.type == "GENE1":
            for fid_key in ("fct_idsm", "fct_idps"):
                fid = fm.params.get(fid_key, 0)
                if fid > 0 and fid not in model.functions:
                    log.error(f"/FAIL/GENE1 on MAT/{mat_id}: function {fid} not defined", "CROSS REF")

    spring_ids = set(model.springs.ids) if (getattr(model, "springs", None) is not None and model.springs.n > 0) else set()
    for sens in getattr(model, "sensors", []):
        if sens.kind == "NIC":
            if sens.spring_id > 0 and sens.spring_id not in spring_ids:
                log.error(f"/SENSOR/NIC/{sens.id}: spring {sens.spring_id} not defined", "CROSS REF")
            if sens.skew_id > 0 and sens.skew_id not in model.skews:
                log.error(f"/SENSOR/NIC/{sens.id}: skew {sens.skew_id} not defined", "CROSS REF")

    # M108: Classical Failure Models, Relative/Function Damping, FVM Airbags, Extended Contacts
    for mat_id, fm, src in getattr(model, "raw_fails", []):
        if fm.type == "ENERGY":
            fid = fm.params.get("fct_id", 0)
            if fid > 0 and fid not in model.functions:
                log.error(f"/FAIL/ENERGY on MAT/{mat_id}: function {fid} not defined", "CROSS REF")

    for d in getattr(model, "damps", []):
        if getattr(d, "kind", "GLOBAL") == "VREL":
            if d.grnod_id > 0 and d.grnod_id not in model.node_groups:
                log.error(f"/DAMP/VREL/{d.id}: node group {d.grnod_id} not defined", "CROSS REF")
            if d.skew_id > 0 and d.skew_id not in model.skews:
                log.error(f"/DAMP/VREL/{d.id}: skew {d.skew_id} not defined", "CROSS REF")
        elif getattr(d, "kind", "GLOBAL") == "FUNCT":
            if d.grnod_id > 0 and d.grnod_id not in model.node_groups:
                log.error(f"/DAMP/FUNCT/{d.id}: node group {d.grnod_id} not defined", "CROSS REF")
            if d.fct_id > 0 and d.fct_id not in model.functions:
                log.error(f"/DAMP/FUNCT/{d.id}: function {d.fct_id} not defined", "CROSS REF")

    for mid, fb in getattr(model, "monvol_fvmbags", {}).items():
        if fb.surf_id > 0 and fb.surf_id not in model.surfaces:
            log.error(f"/MONVOL/FVMBAG1/{mid}: surface {fb.surf_id} not defined", "CROSS REF")
        if fb.mat_id > 0 and fb.mat_id not in model.materials:
            log.error(f"/MONVOL/FVMBAG1/{mid}: material {fb.mat_id} not defined", "CROSS REF")

    for inter in getattr(model, "interfaces", []):
        if inter.type == 19:
            if inter.grnod_id > 0 and inter.grnod_id not in model.node_groups:
                log.error(f"/INTER/TYPE19/{inter.id}: node group {inter.grnod_id} not defined", "CROSS REF")
            if inter.surf_id > 0 and inter.surf_id not in model.surfaces:
                log.error(f"/INTER/TYPE19/{inter.id}: surface {inter.surf_id} not defined", "CROSS REF")
        elif inter.type == 21:
            if inter.surf_id > 0 and inter.surf_id not in model.surfaces:
                log.error(f"/INTER/TYPE21/{inter.id}: surface {inter.surf_id} not defined", "CROSS REF")
            if inter.surf_id1 > 0 and inter.surf_id1 not in model.surfaces:
                log.error(f"/INTER/TYPE21/{inter.id}: surface {inter.surf_id1} not defined", "CROSS REF")
        elif inter.type == 29:  # GUIDED_CABLE
            if inter.grnod_id > 0 and inter.grnod_id not in model.node_groups:
                log.error(f"/INTER/GUIDED_CABLE/{inter.id}: node group {inter.grnod_id} not defined", "CROSS REF")
            if inter.grpart_id > 0 and inter.grpart_id not in model.egroups.get("PART", {}) and inter.grpart_id not in model.parts:
                log.error(f"/INTER/GUIDED_CABLE/{inter.id}: part group {inter.grpart_id} not defined", "CROSS REF")

    # M109: Extended Multi-Physics Sensors & Properties
    for sens in getattr(model, "sensors", []):
        if sens.kind == "ENERGY":
            if sens.part_id > 0 and sens.part_id not in model.parts:
                log.error(f"/SENSOR/ENERGY/{sens.id}: part {sens.part_id} not defined", "CROSS REF")
            if sens.subset_id > 0 and sens.subset_id not in model.subsets:
                log.error(f"/SENSOR/ENERGY/{sens.id}: subset {sens.subset_id} not defined", "CROSS REF")
        elif sens.kind == "TEMP":
            if sens.grnod_id > 0 and sens.grnod_id not in model.node_groups:
                log.error(f"/SENSOR/TEMP/{sens.id}: node group {sens.grnod_id} not defined", "CROSS REF")

    for pid, prop in getattr(model, "properties", {}).items():
        if getattr(prop, "type", 0) in (21, 22):
            skew_id = prop.params.get("skew_id", 0)
            if skew_id > 0 and skew_id not in model.skews:
                log.error(f"/PROP/{pid}: skew {skew_id} not defined", "CROSS REF")
            for layer in prop.params.get("layers", []):
                mid = layer.get("mat_id", 0)
                if mid > 0 and mid not in model.materials:
                    log.error(f"/PROP/{pid}: material {mid} not defined in layer", "CROSS REF")

    # M110: Detonation Wavefronts, Air Blast Loading & Dynamic Element Activation
    for det in getattr(model, "detonations", []):
        if det.mat_id > 0 and det.mat_id not in model.materials:
            log.error(f"/INIT/DET_{det.kind}/{det.id}: material {det.mat_id} not defined", "CROSS REF")

    for pbid, pb in getattr(model, "pblast_loads", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.surf_ground_id > 0 and pb.surf_ground_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: ground surface {pb.surf_ground_id} not defined", "CROSS REF")
        if pb.node_id > 0 and pb.node_id not in model._id2idx:
            log.error(f"/LOAD/PBLAST/{pbid}: node {pb.node_id} not defined", "CROSS REF")

    for act in getattr(model, "activations", []):
        if act.sens_id > 0 and act.sens_id not in sensor_ids:
            log.error(f"/ACTIV/{act.id}: sensor {act.sens_id} not defined", "CROSS REF")

    # M111: Extended Interfaces, Dual-Chamber Airbags & Autopositioning
    for mvid, mv in getattr(model, "monvol_fvmbag2s", {}).items():
        if mv.surf_id_ex > 0 and mv.surf_id_ex not in model.surfaces:
            log.error(f"/MONVOL/FVMBAG2/{mvid}: external surface {mv.surf_id_ex} not defined", "CROSS REF")
        if mv.surf_id_in > 0 and mv.surf_id_in not in model.surfaces:
            log.error(f"/MONVOL/FVMBAG2/{mvid}: internal surface {mv.surf_id_in} not defined", "CROSS REF")
        if mv.mat_id > 0 and mv.mat_id not in model.materials:
            log.error(f"/MONVOL/FVMBAG2/{mvid}: material {mv.mat_id} not defined", "CROSS REF")

    for ap in getattr(model, "autopositions", []):
        if ap.grnod_id > 0 and ap.grnod_id not in model.node_groups:
            log.error(f"/TRANSFORM/AUTOPOSITION/{ap.id}: node group {ap.grnod_id} not defined", "CROSS REF")
        if ap.surf_id > 0 and ap.surf_id not in model.surfaces:
            log.error(f"/TRANSFORM/AUTOPOSITION/{ap.id}: surface {ap.surf_id} not defined", "CROSS REF")
        if ap.skew_id > 0 and ap.skew_id not in model.skews:
            log.error(f"/TRANSFORM/AUTOPOSITION/{ap.id}: skew {ap.skew_id} not defined", "CROSS REF")

    # M112: Centrifugal & Pressure Loads, Advanced Initial Velocities, Final Geometry Imposed Fields, Thermal Rigid Walls, and SPH Boundary Suite
    for lcid, lc in getattr(model, "load_centris", {}).items():
        if lc.fct_id > 0 and lc.fct_id not in model.functions:
            log.error(f"/LOAD/CENTRI/{lcid}: function {lc.fct_id} not defined", "CROSS REF")
        if lc.sens_id > 0 and lc.sens_id not in sensor_ids:
            log.error(f"/LOAD/CENTRI/{lcid}: sensor {lc.sens_id} not defined", "CROSS REF")
        if lc.grnod_id > 0 and lc.grnod_id not in model.node_groups:
            log.error(f"/LOAD/CENTRI/{lcid}: node group {lc.grnod_id} not defined", "CROSS REF")
        if lc.frame_id > 0 and lc.frame_id not in model.skews:
            log.error(f"/LOAD/CENTRI/{lcid}: skew {lc.frame_id} not defined", "CROSS REF")

    for lpid, lp in getattr(model, "load_pfluids", {}).items():
        if lp.surf_id > 0 and lp.surf_id not in model.surfaces:
            log.error(f"/LOAD/PFLUID/{lpid}: surface {lp.surf_id} not defined", "CROSS REF")
        if lp.sens_id > 0 and lp.sens_id not in sensor_ids:
            log.error(f"/LOAD/PFLUID/{lpid}: sensor {lp.sens_id} not defined", "CROSS REF")
        for fid in (lp.fct_id_t, lp.fct_id_pc, lp.fct_id_vel):
            if fid > 0 and fid not in model.functions:
                log.error(f"/LOAD/PFLUID/{lpid}: function {fid} not defined", "CROSS REF")
        for fid in (lp.frame_id, lp.frame_id_vel):
            if fid > 0 and fid not in model.skews:
                log.error(f"/LOAD/PFLUID/{lpid}: skew {fid} not defined", "CROSS REF")

    for lpid, lp in getattr(model, "load_pressures", {}).items():
        if lp.surf_id > 0 and lp.surf_id not in model.surfaces:
            log.error(f"/LOAD/PRESSURE/{lpid}: surface {lp.surf_id} not defined", "CROSS REF")
        if lp.fct_id > 0 and lp.fct_id not in model.functions:
            log.error(f"/LOAD/PRESSURE/{lpid}: function {lp.fct_id} not defined", "CROSS REF")
        if lp.sens_id > 0 and lp.sens_id not in sensor_ids:
            log.error(f"/LOAD/PRESSURE/{lpid}: sensor {lp.sens_id} not defined", "CROSS REF")

    for iaid, ia in getattr(model, "inivel_axes", {}).items():
        if ia.grnod_id > 0 and ia.grnod_id not in model.node_groups:
            log.error(f"/INIVEL/AXIS/{iaid}: node group {ia.grnod_id} not defined", "CROSS REF")
        if ia.frame_id > 0 and ia.frame_id not in model.skews:
            log.error(f"/INIVEL/AXIS/{iaid}: skew {ia.frame_id} not defined", "CROSS REF")
        if ia.sens_id > 0 and ia.sens_id not in sensor_ids:
            log.error(f"/INIVEL/AXIS/{iaid}: sensor {ia.sens_id} not defined", "CROSS REF")

    for ivid, iv in getattr(model, "inivel_fvms", {}).items():
        if iv.skew_id > 0 and iv.skew_id not in model.skews:
            log.error(f"/INIVEL/FVM/{ivid}: skew {iv.skew_id} not defined", "CROSS REF")
        if iv.sens_id > 0 and iv.sens_id not in sensor_ids:
            log.error(f"/INIVEL/FVM/{ivid}: sensor {iv.sens_id} not defined", "CROSS REF")

    for inid, in_obj in getattr(model, "inivel_nodes", {}).items():
        for itm in in_obj.items:
            if itm.node_id > 0 and itm.node_id not in model._id2idx:
                log.error(f"/INIVEL/NODE/{inid}: node {itm.node_id} not defined", "CROSS REF")
            if itm.skew_id > 0 and itm.skew_id not in model.skews:
                log.error(f"/INIVEL/NODE/{inid}: skew {itm.skew_id} not defined", "CROSS REF")

    for idfid, idf in getattr(model, "impdisp_fgeos", {}).items():
        if idf.fct_id > 0 and idf.fct_id not in model.functions:
            log.error(f"/IMPDISP/FGEO/{idfid}: function {idf.fct_id} not defined", "CROSS REF")
        if idf.part_id > 0 and idf.part_id not in model.parts:
            log.error(f"/IMPDISP/FGEO/{idfid}: part {idf.part_id} not defined", "CROSS REF")
        if idf.sens_id > 0 and idf.sens_id not in sensor_ids:
            log.error(f"/IMPDISP/FGEO/{idfid}: sensor {idf.sens_id} not defined", "CROSS REF")
        for nd in idf.nodes:
            if nd["node_id"] > 0 and nd["node_id"] not in model._id2idx:
                log.error(f"/IMPDISP/FGEO/{idfid}: node {nd['node_id']} not defined", "CROSS REF")

    for ivfid, ivf in getattr(model, "impvel_fgeos", {}).items():
        if ivf.fct_id > 0 and ivf.fct_id not in model.functions:
            log.error(f"/IMPVEL/FGEO/{ivfid}: function {ivf.fct_id} not defined", "CROSS REF")
        if ivf.fct_l_id > 0 and ivf.fct_l_id not in model.functions:
            log.error(f"/IMPVEL/FGEO/{ivfid}: function {ivf.fct_l_id} not defined", "CROSS REF")
        if ivf.part_id > 0 and ivf.part_id not in model.parts:
            log.error(f"/IMPVEL/FGEO/{ivfid}: part {ivf.part_id} not defined", "CROSS REF")
        if ivf.sens_id > 0 and ivf.sens_id not in sensor_ids:
            log.error(f"/IMPVEL/FGEO/{ivfid}: sensor {ivf.sens_id} not defined", "CROSS REF")
        for n1, n2 in ivf.pairs:
            if n1 > 0 and n1 not in model._id2idx:
                log.error(f"/IMPVEL/FGEO/{ivfid}: node {n1} not defined", "CROSS REF")
            if n2 > 0 and n2 not in model._id2idx:
                log.error(f"/IMPVEL/FGEO/{ivfid}: node {n2} not defined", "CROSS REF")

    for rtid, rt in getattr(model, "rwall_therms", {}).items():
        if rt.node_id > 0 and rt.node_id not in model._id2idx:
            log.error(f"/RWALL/THERM/{rtid}: node {rt.node_id} not defined", "CROSS REF")
        if rt.grnod_id1 > 0 and rt.grnod_id1 not in model.node_groups:
            log.error(f"/RWALL/THERM/{rtid}: node group 1 {rt.grnod_id1} not defined", "CROSS REF")
        if rt.grnod_id2 > 0 and rt.grnod_id2 not in model.node_groups:
            log.error(f"/RWALL/THERM/{rtid}: node group 2 {rt.grnod_id2} not defined", "CROSS REF")
        if rt.fct_id > 0 and rt.fct_id not in model.functions:
            log.error(f"/RWALL/THERM/{rtid}: function {rt.fct_id} not defined", "CROSS REF")

    for sioid, sio in getattr(model, "sph_inouts", {}).items():
        if sio.surf_id > 0 and sio.surf_id not in model.surfaces:
            log.error(f"/SPH/INOUT/{sioid}: surface {sio.surf_id} not defined", "CROSS REF")
        if sio.part_id > 0 and sio.part_id not in model.parts:
            log.error(f"/SPH/INOUT/{sioid}: part {sio.part_id} not defined", "CROSS REF")
        if sio.fct_id > 0 and sio.fct_id not in model.functions:
            log.error(f"/SPH/INOUT/{sioid}: function {sio.fct_id} not defined", "CROSS REF")

    # M113: SPH Symmetry, Madymo Links/EXFEM, Random Noise, Accelerometers
    for sbid, sb in getattr(model, "sph_bcs", {}).items():
        if sb.frame_id > 0 and sb.frame_id not in model.skews:
            log.error(f"/SPHBCS/{sbid}: skew/frame {sb.frame_id} not defined", "CROSS REF")
        if sb.grnod_id > 0 and sb.grnod_id not in model.node_groups:
            log.error(f"/SPHBCS/{sbid}: node group {sb.grnod_id} not defined", "CROSS REF")

    for mlid, ml in getattr(model, "madymo_links", {}).items():
        if ml.node_id > 0 and ml.node_id not in model._id2idx:
            log.error(f"/MADYMO/LINK/{mlid}: node {ml.node_id} not defined", "CROSS REF")

    for meid, me in getattr(model, "madymo_exfems", {}).items():
        for pid in me.part_ids:
            if pid > 0 and pid not in model.parts:
                log.error(f"/MADYMO/EXFEM/{meid}: part {pid} not defined", "CROSS REF")

    for rn in getattr(model, "random_noises", []):
        if rn.grnod_id > 0 and rn.grnod_id not in model.node_groups:
            log.error(f"/RANDOM/GRNOD/{rn.grnod_id}: node group {rn.grnod_id} not defined", "CROSS REF")

    for aid, acc in getattr(model, "accelerometers", {}).items():
        if acc.node_id > 0 and acc.node_id not in model._id2idx:
            log.error(f"/ACCEL/{aid}: node {acc.node_id} not defined", "CROSS REF")
        if acc.skew_id > 0 and acc.skew_id not in model.skews:
            log.error(f"/ACCEL/{aid}: skew {acc.skew_id} not defined", "CROSS REF")

    # M114: Composite Failure, Propellant Combustion, Non-Uniform Added Mass, Extended Sections
    for mat_id, fc in getattr(model, "fail_composites", {}).items():
        if mat_id > 0 and mat_id not in model.materials:
            log.error(f"/FAIL/COMPOSITE/{mat_id}: material {mat_id} not defined", "CROSS REF")

    for pbid, pb in getattr(model, "ebcs_propellants", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/EBCS/PROPELLANT/{pbid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.sens_id > 0 and pb.sens_id not in sensor_ids:
            log.error(f"/EBCS/PROPELLANT/{pbid}: sensor {pb.sens_id} not defined", "CROSS REF")
        for fid in (pb.f_func_id, pb.g_func_id, pb.h_func_id):
            if fid > 0 and fid not in model.functions:
                log.error(f"/EBCS/PROPELLANT/{pbid}: function {fid} not defined", "CROSS REF")

    for anid, an in getattr(model, "admas_non_uniforms", {}).items():
        for item in an.items:
            if an.kind == "NODE" and item.entity_id > 0 and item.entity_id not in model._id2idx:
                log.error(f"/ADMAS/NON_UNIFORM/{anid}: node {item.entity_id} not defined", "CROSS REF")
            elif an.kind == "PART" and item.entity_id > 0 and item.entity_id not in model.parts:
                log.error(f"/ADMAS/NON_UNIFORM_PART/{anid}: part {item.entity_id} not defined", "CROSS REF")

    egroups_shel = {**getattr(model, "egroups", {}).get("GRSHEL", {}), **getattr(model, "egroups", {}).get("SHEL", {})}
    egroups_bric = {**getattr(model, "egroups", {}).get("GRBRIC", {}), **getattr(model, "egroups", {}).get("BRIC", {})}

    for scid, sc in getattr(model, "sect_circles", {}).items():
        for nid in (sc.n1, sc.n2, sc.n3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/SECT/CIRCLE/{scid}: node {nid} not defined", "CROSS REF")
        if sc.grshel_id > 0 and sc.grshel_id not in egroups_shel:
            log.error(f"/SECT/CIRCLE/{scid}: shell group {sc.grshel_id} not defined", "CROSS REF")
        if sc.grbric_id > 0 and sc.grbric_id not in egroups_bric:
            log.error(f"/SECT/CIRCLE/{scid}: brick group {sc.grbric_id} not defined", "CROSS REF")

    for spid, sp in getattr(model, "sect_parals", {}).items():
        for nid in (sp.n1, sp.n2, sp.n3):
            if nid > 0 and nid not in model._id2idx:
                log.error(f"/SECT/PARAL/{spid}: node {nid} not defined", "CROSS REF")
        if sp.grshel_id > 0 and sp.grshel_id not in egroups_shel:
            log.error(f"/SECT/PARAL/{spid}: shell group {sp.grshel_id} not defined", "CROSS REF")
        if sp.grbric_id > 0 and sp.grbric_id not in egroups_bric:
            log.error(f"/SECT/PARAL/{spid}: brick group {sp.grbric_id} not defined", "CROSS REF")

    for maid, ma in getattr(model, "monvol_areas", {}).items():
        if ma.surf_id_ext > 0 and ma.surf_id_ext not in model.surfaces:
            log.error(f"/MONVOL/AREA/{maid}: surface {ma.surf_id_ext} not defined", "CROSS REF")

    for sid, s in getattr(model, "surfaces", {}).items():
        for mid in getattr(s, "mat_ids", []):
            if mid > 0 and mid not in model.materials:
                log.error(f"/SURF/{sid}: material {mid} not defined", "CROSS REF")
        for pid in getattr(s, "prop_ids", []):
            if pid > 0 and pid not in model.properties:
                log.error(f"/SURF/{sid}: property {pid} not defined", "CROSS REF")
        for bid in getattr(s, "box_ids", []):
            if bid > 0 and bid not in model.boxes:
                log.error(f"/SURF/{sid}: box {bid} not defined", "CROSS REF")

    for pid, sr in getattr(model, "sph_reserves", {}).items():
        if sr.part_id > 0 and sr.part_id not in model.parts:
            log.error(f"/SPH/RESERVE/{pid}: part {sr.part_id} not defined", "CROSS REF")

    for item in getattr(model, "move_functs", []):
        fid = item[0] if isinstance(item, (tuple, list)) else getattr(item, "id", 0)
        if fid > 0 and fid not in model.functions:
            log.error(f"/MOVE_FUNCT/{fid}: function {fid} not defined", "CROSS REF")

    for eid, em in getattr(model, "eigen_modes", {}).items():
        if em.grnod_id > 0 and em.grnod_id not in model.node_groups:
            log.error(f"/EIG/{eid}: node group {em.grnod_id} not defined", "CROSS REF")
        if em.grnod_bc > 0 and em.grnod_bc not in model.node_groups:
            log.error(f"/EIG/{eid}: node group {em.grnod_bc} not defined", "CROSS REF")

    for mid, ff in getattr(model, "fail_fractals", {}).items():
        if ff.mat_id > 0 and ff.mat_id not in model.materials:
            log.error(f"/FAIL/FRACTAL/{mid}: material {ff.mat_id} not defined", "CROSS REF")

    for tid, tp in getattr(model, "transform_positions", {}).items():
        if tp.grnod_id > 0 and tp.grnod_id not in model.node_groups:
            log.error(f"/TRANSFORM/{tid}: node group {tp.grnod_id} not defined", "CROSS REF")

    for lid, el in getattr(model, "external_links", {}).items():
        if el.grnod_id > 0 and el.grnod_id not in model.node_groups:
            log.error(f"/EXTERN/LINK/{lid}: node group {el.grnod_id} not defined", "CROSS REF")

    for fid, fm in getattr(model, "friction_models", {}).items():
        for p in fm.pairs:
            if p.part_id1 > 0 and p.part_id1 not in model.parts:
                log.error(f"/FRICTION/{fid}: part {p.part_id1} not defined", "CROSS REF")
            if p.part_id2 > 0 and p.part_id2 not in model.parts:
                log.error(f"/FRICTION/{fid}: part {p.part_id2} not defined", "CROSS REF")
            if p.grpart_id1 > 0 and p.grpart_id1 not in getattr(model, "part_groups", {}):
                log.error(f"/FRICTION/{fid}: part group {p.grpart_id1} not defined", "CROSS REF")
            if p.grpart_id2 > 0 and p.grpart_id2 not in getattr(model, "part_groups", {}):
                log.error(f"/FRICTION/{fid}: part group {p.grpart_id2} not defined", "CROSS REF")

    for bid, nb in getattr(model, "nbcs_blocks", {}).items():
        for n in nb.nodes:
            if n.node_id > 0 and n.node_id not in model._id2idx:
                log.error(f"/NBCS/{bid}: node {n.node_id} not defined", "CROSS REF")
            if n.skew_id > 0 and n.skew_id not in model.skews:
                log.error(f"/NBCS/{bid}: skew {n.skew_id} not defined", "CROSS REF")

    for nid in getattr(model, "refsta_nodes", {}):
        if nid > 0 and nid not in model._id2idx:
            log.error(f"/REFSTA: node {nid} not defined", "CROSS REF")

    # M150: EBCS, AMS, and Seatbelt Systems
    for pid, eb in getattr(model, "ebcs_pres", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/PRES/{pid}: surface {eb.surf_id} not defined", "CROSS REF")
        for fid in (eb.fct_pres, eb.fct_rho, eb.fct_en):
            if fid > 0 and fid not in model.functions:
                log.error(f"/EBCS/PRES/{pid}: function {fid} not defined", "CROSS REF")

    for vid, eb in getattr(model, "ebcs_vel", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/VEL/{vid}: surface {eb.surf_id} not defined", "CROSS REF")
        for fid in (eb.fct_vx, eb.fct_vy, eb.fct_vz, eb.fct_rho, eb.fct_en):
            if fid > 0 and fid not in model.functions:
                log.error(f"/EBCS/VEL/{vid}: function {fid} not defined", "CROSS REF")

    for iid, eb in getattr(model, "ebcs_inlets", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/INLET/{iid}: surface {eb.surf_id} not defined", "CROSS REF")
        if eb.funct_id > 0 and eb.funct_id not in model.functions:
            log.error(f"/EBCS/INLET/{iid}: function {eb.funct_id} not defined", "CROSS REF")

    for fid, eb in getattr(model, "ebcs_fluxouts", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/FLUXOUT/{fid}: surface {eb.surf_id} not defined", "CROSS REF")

    for gid, eb in getattr(model, "ebcs_gradp0", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/GRADP0/{gid}: surface {eb.surf_id} not defined", "CROSS REF")

    for nid, eb in getattr(model, "ebcs_normv", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/NORMV/{nid}: surface {eb.surf_id} not defined", "CROSS REF")
        if eb.funct_id > 0 and eb.funct_id not in model.functions:
            log.error(f"/EBCS/NORMV/{nid}: function {eb.funct_id} not defined", "CROSS REF")

    for vid, eb in getattr(model, "ebcs_valves", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/{eb.kind}/{vid}: surface {eb.surf_id} not defined", "CROSS REF")

    for mid, eb in getattr(model, "ebcs_monvols", {}).items():
        if eb.surf_id > 0 and eb.surf_id not in model.surfaces:
            log.error(f"/EBCS/MONVOL/{mid}: surface {eb.surf_id} not defined", "CROSS REF")
        if eb.monvol_id > 0 and eb.monvol_id not in getattr(model, "monitored_volumes", {}) and eb.monvol_id not in getattr(model, "airbags", {}):
            log.error(f"/EBCS/MONVOL/{mid}: monvol {eb.monvol_id} not defined", "CROSS REF")

    if getattr(model, "ams_control", None) is not None:
        ams = model.ams_control
        if ams.grpart_id > 0 and ams.grpart_id not in part_groups and ams.grpart_id not in model.parts:
            log.error(f"/AMS: part group {ams.grpart_id} not defined", "CROSS REF")

    for sbid, sb in getattr(model, "seatbelt_systems", {}).items():
        for rid in sb.retractor_ids:
            if rid > 0 and rid not in getattr(model, "retractors", {}):
                log.error(f"/SEATBELT/{sbid}: retractor {rid} not defined", "CROSS REF")
        for sid in sb.slipring_ids:
            if sid > 0 and sid not in getattr(model, "sliprings", {}) and sid not in getattr(model, "slipring_shells", {}):
                log.error(f"/SEATBELT/{sbid}: slipring {sid} not defined", "CROSS REF")

    for bwid, bw in getattr(model, "bcs_walls", {}).items():
        if bw.grnod_id > 0 and bw.grnod_id not in model.node_groups and bw.grnod_id not in getattr(model, "node_sets", {}):
            log.error(f"/BCS/WALL/{bwid}: node group {bw.grnod_id} not defined", "CROSS REF")
        if bw.sensor_id > 0 and bw.sensor_id not in sensor_ids:
            log.error(f"/BCS/WALL/{bwid}: sensor {bw.sensor_id} not defined", "CROSS REF")

    # M151: PBLAST, INIVOL, INIGRAV, INISTA, BEM, PERTURB
    for pbid, pb in getattr(model, "pblast_loads", {}).items():
        if pb.surf_id > 0 and pb.surf_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: surface {pb.surf_id} not defined", "CROSS REF")
        if pb.surf_ground_id > 0 and pb.surf_ground_id not in model.surfaces:
            log.error(f"/LOAD/PBLAST/{pbid}: ground surface {pb.surf_ground_id} not defined", "CROSS REF")
        if pb.node_id > 0 and pb.node_id not in model._id2idx:
            log.error(f"/LOAD/PBLAST/{pbid}: node {pb.node_id} not defined", "CROSS REF")

    for ivid, iv in getattr(model, "inivols", {}).items():
        if iv.part_id > 0 and iv.part_id not in model.parts and iv.part_id not in part_groups:
            log.error(f"/INIVOL/{ivid}: part {iv.part_id} not defined", "CROSS REF")
        for c in iv.containers:
            if c.surf_id > 0 and c.surf_id not in model.surfaces:
                log.error(f"/INIVOL/{ivid}: container surface {c.surf_id} not defined", "CROSS REF")

    for igid, ig in getattr(model, "inigrav_loads", {}).items():
        if ig.grpart_id > 0 and ig.grpart_id not in model.parts and ig.grpart_id not in part_groups:
            log.error(f"/INIGRAV/{igid}: part group {ig.grpart_id} not defined", "CROSS REF")
        if ig.surf_id > 0 and ig.surf_id not in model.surfaces:
            log.error(f"/INIGRAV/{igid}: surface {ig.surf_id} not defined", "CROSS REF")

    for bemid, bem in getattr(model, "bem_controls", {}).items():
        if bem.surf_id > 0 and bem.surf_id not in model.surfaces:
            log.error(f"/BEM/{bem.subtype}/{bemid}: surface {bem.surf_id} not defined", "CROSS REF")
        if bem.grnod_aux_id > 0 and bem.grnod_aux_id not in model.node_groups and bem.grnod_aux_id not in getattr(model, "node_sets", {}):
            log.error(f"/BEM/{bem.subtype}/{bemid}: node group {bem.grnod_aux_id} not defined", "CROSS REF")

    for ptid, pt in getattr(model, "perturb_controls", {}).items():
        if pt.grpart_id > 0 and pt.grpart_id not in model.parts and pt.grpart_id not in part_groups:
            log.error(f"/PERTURB/{pt.subtype}/{ptid}: part group {pt.grpart_id} not defined", "CROSS REF")
        if pt.fct_id > 0 and pt.fct_id not in model.functions:
            log.error(f"/PERTURB/{pt.subtype}/{ptid}: function {pt.fct_id} not defined", "CROSS REF")


