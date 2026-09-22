"""OpenRadioss /FRICTION and /FRIC_ORIENT — Advanced Contact Friction Models.

Upstream Fortran reference:
- Starter Friction Models Card Reader:
  `starter/source/interfaces/friction/reader/hm_read_friction_models.F`
  Subroutine: HM_READ_FRICTION_MODELS
- Starter Friction Table & Formulation Reader:
  `starter/source/interfaces/friction/reader/hm_read_friction.F`
  Subroutine: HM_READ_FRICTION
- Starter Orthotropic Friction Orientation Reader:
  `starter/source/interfaces/friction/reader/hm_read_friction_orientations.F`
  Subroutine: HM_READ_FRICTION_ORIENTATIONS
- Engine 3D Interface Contact Force Evaluation:
  `engine/source/interfaces/int07/i7for3.F`
  Lines 1860-2295 (Isotropic & Orthotropic Friction Coefficient Updates)
  Lines 2475-2600 (Orthotropic Tangential Force & Friction Ellipse Projection)
- Engine Friction Parts Model:
  `engine/source/interfaces/int07/frictionparts_model.F`
  Subroutines: FRICTIONPARTS_MODEL_ORTHO, FRICTIONPARTS_MODEL_ISOT

Formulation & Physics Overview:
--------------------------------
1. Static / Dynamic Velocity Decay Model:
   Models transition from static sticking coefficient (mu_s) to dynamic sliding
   coefficient (mu_d) with exponential decay factor c:
       mu(v_rel) = mu_d + (mu_s - mu_d) * exp(-c * |v_rel|)
   Matches OpenRadioss MFROT = 4 (i7for3.F:1941-1953, 2238-2252) where:
       mu(v) = C1 + (mu_0 - C1) * exp(-C2 * v)

2. Contact Pressure Dependence (Power-Law Scaling):
   Normal contact pressure P = |F_n| / Area alters real contact area of asperities:
       mu(P) = mu_0 * (P / P_0) ** n_p
   where P_0 is reference pressure and n_p is pressure exponent (n_p < 0 represents
   asperity saturation softening at high pressures).

3. Temperature Dependence (Thermal Softening):
   Frictional heating and ambient temperature T soften contact asperities:
       mu(T) = mu_0 * max(0.0, 1.0 - a_T * (T - T_0))
   where T_0 is reference temperature and a_T is temperature sensitivity.

4. Anisotropic / Orthotropic Friction (Elliptical Friction Limit):
   For textured, rolled, or composite surfaces, friction varies with the angle theta
   relative to the primary orthotropic orientation d_ortho:
       mu(theta) = sqrt((mu_1 * cos(theta))**2 + (mu_2 * sin(theta))**2)
   In the local tangent plane:
       cos(theta) = dot(d_tangent, d_ortho)
       sin(theta)**2 = 1.0 - cos(theta)**2
   Matches OpenRadioss /FRIC_ORIENT and i7for3.F:1965-1985, 2551-2600.

5. OpenRadioss MFROT Laws (Complete Compatibility):
   - MFROT = 0: Classical Coulomb friction: mu = mu_0
   - MFROT = 1: Viscous polynomial:
       mu = mu_0 + (C1 + C4 * P) * P + (C2 + C3 * P) * v + C5 * v**2
   - MFROT = 2: Darmstadt law:
       mu = mu_0 + C1 * exp(C2 * v) * P**2 + C3 * exp(C4 * v) * P + C5 * exp(C6 * v)
   - MFROT = 3: Renard piecewise polynomial law with velocity thresholds C5, C6
   - MFROT = 4: Exponential velocity decay
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class FrictionModel:
    """Advanced Contact Friction Model (/FRICTION, /FRIC_ORIENT).

    Parameters
    ----------
    id : int
        Friction model identifier (NOINTF in hm_read_friction.F).
    title : str
        User title / card descriptor.
    mu_0 : float
        Base isotropic friction coefficient (FRIC).
    mu_s : Optional[float]
        Static friction coefficient at v_rel = 0. If None, defaults to mu_0.
    mu_d : Optional[float]
        Dynamic friction coefficient at high sliding velocity. If None, defaults to mu_0.
    decay_coef : float
        Velocity exponential decay rate c >= 0 in exp(-c * |v_rel|).
    p_0 : Optional[float]
        Reference contact pressure P_0 > 0 for pressure power-law scaling.
    n_p : float
        Pressure power-law exponent n_p.
    t_0 : float
        Reference temperature T_0 (default: 293.15 K).
    a_t : float
        Linear temperature softening coefficient a_T >= 0.
    is_orthotropic : bool
        Whether friction is anisotropic/orthotropic (IORTHFRIC > 0).
    mu_1 : Optional[float]
        Longitudinal friction coefficient along orthotropic direction.
    mu_2 : Optional[float]
        Transverse friction coefficient perpendicular to orthotropic direction.
    ortho_dir : Optional[np.ndarray]
        Unit direction vector for the longitudinal axis in global coordinates.
    mfrot : int
        OpenRadioss MFROT model index (0=Coulomb, 1=Viscous, 2=Darmstadt, 3=Renard, 4=ExpDecay).
    c1 : float
        OpenRadioss model coefficient C1 (dynamic friction for MFROT=4).
    c2 : float
        OpenRadioss model coefficient C2 (decay coefficient for MFROT=4).
    c3 : float
        OpenRadioss model coefficient C3.
    c4 : float
        OpenRadioss model coefficient C4.
    c5 : float
        OpenRadioss model coefficient C5.
    c6 : float
        OpenRadioss model coefficient C6.
    vis_f : float
        Friction critical damping factor (VISCF in hm_read_friction.F).
    ifq : int
        Friction filtering flag (IFILTR in hm_read_friction.F).
    xfreq : float
        Filtering frequency / period parameter.
    iform : int
        Tangential formulation (1=viscous total, 2=incremental stiffness).
    mu_min : float
        Absolute minimum friction floor (default: 1.0e-30, matching EM30 in i7for3.F).
    """

    id: int = 1
    title: str = ""

    # Base isotropic friction
    mu_0: float = 0.0

    # Static / Dynamic velocity decay model
    # mu(v_rel) = mu_d + (mu_s - mu_d) * exp(-c * |v_rel|)
    mu_s: Optional[float] = None
    mu_d: Optional[float] = None
    decay_coef: float = 0.0

    # Pressure dependence model
    # mu(P) = mu_0 * (P / P_0) ** n_p
    p_0: Optional[float] = None
    n_p: float = 0.0

    # Temperature dependence model
    # mu(T) = mu_0 * (1.0 - a_T * (T - T_0))
    t_0: float = 293.15
    a_t: float = 0.0

    # Anisotropic (orthotropic) friction model
    # mu(theta) = sqrt((mu_1 * cos(theta))**2 + (mu_2 * sin(theta))**2)
    is_orthotropic: bool = False
    mu_1: Optional[float] = None
    mu_2: Optional[float] = None
    ortho_dir: Optional[np.ndarray] = None

    # OpenRadioss MFROT specific parameters (0..4)
    mfrot: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    vis_f: float = 1.0
    ifq: int = 0
    xfreq: float = 0.0
    iform: int = 1

    # Minimum friction floor (matches OpenRadioss EM30 in i7for3.F)
    mu_min: float = 1.0e-30

    def __post_init__(self) -> None:
        """Sanitize and synchronize model parameters."""
        # Handle OpenRadioss MFROT = 4 mapping to mu_s, mu_d, decay_coef
        if self.mfrot == 4:
            if self.mu_s is None:
                self.mu_s = self.mu_0
            if self.mu_d is None and self.c1 != 0.0:
                self.mu_d = self.c1
            if self.decay_coef == 0.0 and self.c2 != 0.0:
                self.decay_coef = self.c2

        # Sync mu_s and mu_d with mu_0 if velocity decay is requested but only one is given
        if self.decay_coef > 0.0:
            if self.mu_s is None:
                self.mu_s = self.mu_0
            if self.mu_d is None:
                self.mu_d = self.mu_0

        # Anisotropic synchronization
        if self.mu_1 is not None or self.mu_2 is not None:
            self.is_orthotropic = True
            if self.mu_1 is None:
                self.mu_1 = self.mu_0
            if self.mu_2 is None:
                self.mu_2 = self.mu_0

        # Normalize reference orthotropic direction vector
        if self.ortho_dir is not None:
            arr = np.asarray(self.ortho_dir, dtype=np.float64)
            norm = np.linalg.norm(arr)
            if norm > 1.0e-14:
                self.ortho_dir = arr / norm
            else:
                self.ortho_dir = arr

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FrictionModel:
        """Build FrictionModel from a card dictionary or deck parser output."""
        m = cls(
            id=int(d.get("id", d.get("fric_id", 1))),
            title=str(d.get("title", d.get("titr", ""))),
            mu_0=float(d.get("mu_0", d.get("fric", d.get("fricc", 0.0)))),
            mu_s=float(d["mu_s"]) if "mu_s" in d and d["mu_s"] is not None else None,
            mu_d=float(d["mu_d"]) if "mu_d" in d and d["mu_d"] is not None else None,
            decay_coef=float(d.get("decay_coef", d.get("c", 0.0))),
            p_0=float(d["p_0"]) if "p_0" in d and d["p_0"] is not None else None,
            n_p=float(d.get("n_p", 0.0)),
            t_0=float(d.get("t_0", 293.15)),
            a_t=float(d.get("a_t", 0.0)),
            is_orthotropic=bool(d.get("is_orthotropic", d.get("iorthfric", 0) > 0)),
            mu_1=float(d["mu_1"]) if "mu_1" in d and d["mu_1"] is not None else None,
            mu_2=float(d["mu_2"]) if "mu_2" in d and d["mu_2"] is not None else None,
            ortho_dir=np.asarray(d["ortho_dir"], dtype=np.float64) if "ortho_dir" in d and d["ortho_dir"] is not None else None,
            mfrot=int(d.get("mfrot", d.get("ifric", 0))),
            c1=float(d.get("c1", 0.0)),
            c2=float(d.get("c2", 0.0)),
            c3=float(d.get("c3", 0.0)),
            c4=float(d.get("c4", 0.0)),
            c5=float(d.get("c5", 0.0)),
            c6=float(d.get("c6", 0.0)),
            vis_f=float(d.get("vis_f", d.get("viscf", 1.0))),
            ifq=int(d.get("ifq", d.get("ifiltr", 0))),
            xfreq=float(d.get("xfreq", 0.0)),
            iform=int(d.get("iform", 1)),
        )
        return m


def compute_friction_coefficient(
    model: FrictionModel,
    v_rel: Union[float, np.ndarray] = 0.0,
    P: Union[float, np.ndarray] = 0.0,
    T: Union[float, np.ndarray] = 293.15,
    dir_tangent: Optional[np.ndarray] = None,
    dir_ortho: Optional[np.ndarray] = None,
    theta: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """Compute friction coefficient under velocity, pressure, temperature, and anisotropy.

    Formulation:
    ------------
    1. Base / Anisotropic Friction:
       - If orthotropic:
           mu_base(theta) = sqrt((mu_1 * cos(theta))**2 + (mu_2 * sin(theta))**2)
       - If isotropic:
           mu_base = mu_0

    2. Velocity Decay Transition:
       If static/dynamic decay is active:
           mu_v = mu_d + (mu_s - mu_d) * exp(-c * |v_rel|)
       In isotropic mode, mu_v serves as the velocity-dependent friction.
       In orthotropic mode with mu_s > 0, the directional friction scales by (mu_v / mu_s).

    3. OpenRadioss MFROT Models:
       - MFROT = 1: mu = mu_0 + (C1 + C4 * P) * P + (C2 + C3 * P) * v + C5 * v**2
       - MFROT = 2: mu = mu_0 + C1 * exp(C2 * v) * P**2 + C3 * exp(C4 * v) * P + C5 * exp(C6 * v)
       - MFROT = 3: Renard piecewise law

    4. Contact Pressure Power-Law Scaling:
       If P_0 > 0 and n_p != 0:
           factor_P = (P / P_0) ** n_p

    5. Temperature Softening:
       If a_T != 0:
           factor_T = max(0.0, 1.0 - a_T * (T - T_0))

    Parameters
    ----------
    model : FrictionModel
        Configured friction model parameters.
    v_rel : float or np.ndarray
        Relative tangential sliding speed (scalar or array).
    P : float or np.ndarray
        Contact pressure (scalar or array).
    T : float or np.ndarray
        Contact surface temperature (scalar or array).
    dir_tangent : Optional[np.ndarray]
        Tangential sliding direction vector.
    dir_ortho : Optional[np.ndarray]
        Orthotropic reference longitudinal direction vector.
    theta : Optional[float or np.ndarray]
        Direct sliding angle in radians relative to longitudinal direction.

    Returns
    -------
    float or np.ndarray
        Evaluated friction coefficient, clamped above model.mu_min.
    """
    is_scalar_input = np.ndim(v_rel) == 0 and np.ndim(P) == 0 and np.ndim(T) == 0 and (theta is None or np.ndim(theta) == 0)

    # 1. Base / Anisotropic friction calculation
    if model.is_orthotropic:
        mu_1 = model.mu_1 if model.mu_1 is not None else model.mu_0
        mu_2 = model.mu_2 if model.mu_2 is not None else model.mu_0

        if theta is not None:
            # Angle passed directly
            th = np.asarray(theta, dtype=np.float64)
            cos_th = np.cos(th)
            sin_th = np.sin(th)
            mu_base = np.sqrt((mu_1 * cos_th) ** 2 + (mu_2 * sin_th) ** 2)
        elif dir_tangent is not None:
            # Compute angle from tangent direction and orthotropic orientation
            ref_dir = dir_ortho if dir_ortho is not None else model.ortho_dir
            if ref_dir is not None:
                t_vec = np.asarray(dir_tangent, dtype=np.float64)
                d_vec = np.asarray(ref_dir, dtype=np.float64)

                # Normalize vectors
                t_norm = np.linalg.norm(t_vec, axis=-1, keepdims=True)
                d_norm = np.linalg.norm(d_vec, axis=-1, keepdims=True)
                t_unit = np.where(t_norm > 1.0e-14, t_vec / np.maximum(t_norm, 1.0e-14), 0.0)
                d_unit = np.where(d_norm > 1.0e-14, d_vec / np.maximum(d_norm, 1.0e-14), 0.0)

                # Dot product gives cos(theta)
                cos_th = np.sum(t_unit * d_unit, axis=-1)
                cos_th = np.clip(cos_th, -1.0, 1.0)
                sin2_th = np.maximum(0.0, 1.0 - cos_th ** 2)
                mu_base = np.sqrt((mu_1 * cos_th) ** 2 + (mu_2 ** 2) * sin2_th)
            else:
                mu_base = mu_1
        else:
            mu_base = mu_1
    else:
        mu_base = model.mu_0

    # 2. Velocity decay model
    has_decay = (
        model.decay_coef > 0.0
        or (model.mu_s is not None and model.mu_d is not None and model.mu_s != model.mu_d)
    )

    if has_decay:
        mu_s = model.mu_s if model.mu_s is not None else model.mu_0
        mu_d = model.mu_d if model.mu_d is not None else model.mu_0
        c = model.decay_coef
        v = np.abs(np.asarray(v_rel, dtype=np.float64))

        # mu(v_rel) = mu_d + (mu_s - mu_d) * exp(-c * |v_rel|)
        mu_v = mu_d + (mu_s - mu_d) * np.exp(-c * v)

        if model.is_orthotropic:
            if mu_s > 0.0:
                mu_eff = mu_base * (mu_v / mu_s)
            else:
                mu_eff = mu_v
        else:
            mu_eff = mu_v
    else:
        mu_eff = mu_base

    # 3. OpenRadioss MFROT Laws (polynomial, Darmstadt, Renard)
    v_arr = np.abs(np.asarray(v_rel, dtype=np.float64))
    p_arr = np.asarray(P, dtype=np.float64)

    if model.mfrot == 1:
        # Viscous polynomial: mu = mu_0 + (c1 + c4*p)*p + (c2 + c3*p)*v + c5*v^2
        mu_eff = (
            model.mu_0
            + (model.c1 + model.c4 * p_arr) * p_arr
            + (model.c2 + model.c3 * p_arr) * v_arr
            + model.c5 * (v_arr ** 2)
        )
    elif model.mfrot == 2:
        # Darmstadt Law: mu = mu_0 + c1*exp(c2*v)*p^2 + c3*exp(c4*v)*p + c5*exp(c6*v)
        mu_eff = (
            model.mu_0
            + model.c1 * np.exp(model.c2 * v_arr) * (p_arr ** 2)
            + model.c3 * np.exp(model.c4 * v_arr) * p_arr
            + model.c5 * np.exp(model.c6 * v_arr)
        )
    elif model.mfrot == 3:
        # Renard Law (piecewise velocity thresholds C5, C6)
        c1, c2, c3, c4, c5, c6 = (
            model.c1,
            model.c2,
            model.c3,
            model.c4,
            model.c5,
            model.c6,
        )
        if np.ndim(v_arr) == 0:
            v_val = float(v_arr)
            if v_val <= c5:
                dmu = c3 - c1
                xi = v_val / max(c5, 1.0e-30)
                mu_eff = c1 + dmu * xi * (2.0 - xi)
            elif v_val < c6:
                dmu = c4 - c3
                xi = (v_val - c5) / max(c6 - c5, 1.0e-30)
                mu_eff = c3 + dmu * (3.0 - 2.0 * xi) * (xi ** 2)
            else:
                dmu = c2 - c4
                v2 = (v_val - c6) ** 2
                mu_eff = c2 - dmu / (1.0 + dmu * v2)
        else:
            cond1 = v_arr <= c5
            cond2 = (v_arr > c5) & (v_arr < c6)
            cond3 = v_arr >= c6

            xi1 = v_arr / max(c5, 1.0e-30)
            res1 = c1 + (c3 - c1) * xi1 * (2.0 - xi1)

            xi2 = (v_arr - c5) / max(c6 - c5, 1.0e-30)
            res2 = c3 + (c4 - c3) * (3.0 - 2.0 * xi2) * (xi2 ** 2)

            dmu3 = c2 - c4
            v2 = (v_arr - c6) ** 2
            res3 = c2 - dmu3 / (1.0 + dmu3 * v2)

            mu_eff = np.where(cond1, res1, np.where(cond2, res2, res3))

    # 4. Pressure power-law scaling
    if model.p_0 is not None and model.p_0 > 0.0 and model.n_p != 0.0:
        p_safe = np.maximum(0.0, np.asarray(P, dtype=np.float64))
        if model.n_p > 0.0:
            factor_p = (p_safe / model.p_0) ** model.n_p
        else:
            # Avoid division by zero when n_p < 0 and P=0
            p_nonzero = np.maximum(p_safe, 1.0e-12)
            factor_p = (p_nonzero / model.p_0) ** model.n_p
        mu_eff = mu_eff * factor_p

    # 5. Temperature dependence
    if model.a_t != 0.0:
        t_arr = np.asarray(T, dtype=np.float64)
        factor_t = np.maximum(0.0, 1.0 - model.a_t * (t_arr - model.t_0))
        mu_eff = mu_eff * factor_t

    # 6. Minimum floor
    mu_eff = np.maximum(model.mu_min, mu_eff)

    if is_scalar_input:
        return float(mu_eff)
    return mu_eff


def compute_friction_force(
    model: FrictionModel,
    F_normal: Union[float, np.ndarray],
    v_rel_vec: np.ndarray,
    P: Union[float, np.ndarray] = 0.0,
    T: Union[float, np.ndarray] = 293.15,
    dir_ortho: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute friction force vector opposing relative sliding velocity.

    Friction force acts along the tangential slip direction opposing velocity:
        F_tangent = - mu * |F_normal| * (v_rel / |v_rel|)

    Parameters
    ----------
    model : FrictionModel
        Configured friction model.
    F_normal : float or np.ndarray
        Magnitude of normal contact force (scalar or array).
    v_rel_vec : np.ndarray
        Relative velocity vector(s) in tangent plane (1D [vx, vy, vz] or 2D [N, 3]).
    P : float or np.ndarray
        Contact pressure (default: 0.0).
    T : float or np.ndarray
        Contact temperature (default: 293.15 K).
    dir_ortho : Optional[np.ndarray]
        Optional custom orthotropic orientation vector.

    Returns
    -------
    np.ndarray
        Friction force vector(s) of identical shape as v_rel_vec.
    """
    v_arr = np.asarray(v_rel_vec, dtype=np.float64)
    fn = np.abs(np.asarray(F_normal, dtype=np.float64))

    if v_arr.ndim == 1:
        v_norm = float(np.linalg.norm(v_arr))
        if v_norm < 1.0e-15 or float(fn) == 0.0:
            return np.zeros_like(v_arr)

        t_unit = v_arr / v_norm
        mu = compute_friction_coefficient(
            model=model,
            v_rel=v_norm,
            P=P,
            T=T,
            dir_tangent=t_unit,
            dir_ortho=dir_ortho,
        )
        return -float(mu) * float(fn) * t_unit

    else:
        # Multi-vector / batch array
        v_norm = np.linalg.norm(v_arr, axis=-1, keepdims=True)
        zero_mask = (v_norm < 1.0e-15) | (np.expand_dims(fn, axis=-1) == 0.0)
        safe_norm = np.where(zero_mask, 1.0, v_norm)
        t_unit = v_arr / safe_norm

        mu = compute_friction_coefficient(
            model=model,
            v_rel=np.squeeze(v_norm, axis=-1),
            P=P,
            T=T,
            dir_tangent=t_unit,
            dir_ortho=dir_ortho,
        )
        if np.ndim(mu) < v_arr.ndim:
            mu_expanded = np.expand_dims(mu, axis=-1)
        else:
            mu_expanded = mu

        fn_expanded = np.expand_dims(fn, axis=-1) if np.ndim(fn) < v_arr.ndim else fn
        f_fric = -mu_expanded * fn_expanded * t_unit
        f_fric = np.where(zero_mask, 0.0, f_fric)
        return f_fric
