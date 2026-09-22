"""Thick Shell (Solid-Shell) Properties (/PROP/TYPE20 TSHELL and /PROP/TYPE21 TSHELL_COMP).

Upstream OpenRadioss Fortran references:
- Starter Card Reader for Standard Solid-Shell Property (/PROP/TYPE20, /PROP/TSHELL):
  `starter/source/properties/thickshell/hm_read_prop20.F` (SUBROUTINE HM_READ_PROP20, lines 38-389)
- Starter Card Reader for Orthotropic/Composite Thick Shell Property (/PROP/TYPE21, /PROP/TSH_ORTH):
  `starter/source/properties/thickshell/hm_read_prop21.F` (SUBROUTINE HM_PROP_READ21, lines 39-422)
- Engine Kinematics and Internal Forces for 8-node Thick Shells:
  `engine/source/elements/thickshell/solidec/scforc3.F` (SUBROUTINE SCFORC3)
  `engine/source/elements/thickshell/solidec/scderi3.F` (SUBROUTINE SCDERI3)
  `engine/source/elements/thickshell/solidec/scdefc3.F` (SUBROUTINE SCDEFC3)
  `engine/source/elements/thickshell/solidec/scfint3.F` (SUBROUTINE SCFINT3)

Theory & Formulations:
----------------------
1. 8-Node Prismatic Solid-Shell Topology:
   Solid-shell elements (TSHELL) use an 8-node hexahedral brick topology with 3 displacement DOFs
   per node (no rotational DOFs).
   - In-plane coordinates: (xi, eta) in [-1, +1] x [-1, +1]
   - Normalized thickness coordinate: zeta in [-1, +1]
   - Nodes 1..4 lie on the bottom face (zeta = -1)
   - Nodes 5..8 lie on the top face (zeta = +1)

2. Locking Alleviation Mechanisms:
   - ANS (Assumed Natural Strain, e.g. Bathe-Dvorkin MITC):
     Interpolates natural transverse shear strains (gamma_xz, gamma_yz) from edge tying points
     to eliminate shear locking in thin limits (h/L << 1).
     Interpolates normal thickness strain epsilon_zz from the mid-surface to prevent
     curvature thickness locking (trapezoidal locking).
   - EAS (Enhanced Assumed Strain, Simo-Rifai):
     Enriches the strain field with incompatible modes:
       epsilon = B * d + M * alpha
     where alpha are internal parameters statically condensed at the element level,
     preventing Poisson thickness locking and volumetric locking.
   - Reduced Integration (1-point in-plane) with physical hourglass control (dn / cvis).

3. Transverse Shear Correction:
   Parabolic shear stress distribution through homogeneous isotropic plate thickness yields
   an effective shear correction factor:
       kappa = 5 / 6 ~= 0.8333333333333334

4. Through-Thickness Layup for Layered Composites (/PROP/TYPE21):
   Composite stack with K plies of thickness t_k and orientation angle theta_k:
       Total thickness H = sum_{k=1}^K t_k
   Each ply k occupies [z_k_bot, z_k_top] mapped into normalized global thickness [-1, +1]:
       Delta_zeta_k = 2 * t_k / H
   Within each ply, local Gauss points map to global coordinates with weighted quadrature
   preserving exact integration over each layer and the entire stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ============================================================================
# Gauss-Legendre Quadrature Utilities
# ============================================================================

def legendre_gauss_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return 1D Gauss-Legendre quadrature points xi in [-1, +1] and weights w.

    Sum of weights equals 2.0 (length of interval [-1, +1]).

    Parameters
    ----------
    n : int
        Number of integration points (1 <= n <= 15).

    Returns
    -------
    coords : np.ndarray (n,)
        Quadrature point coordinates along [-1, +1].
    weights : np.ndarray (n,)
        Quadrature weights summing to 2.0.
    """
    n_pts = max(1, int(n))
    coords, weights = np.polynomial.legendre.leggauss(n_pts)
    return coords, weights


# ============================================================================
# Composite Ply Definition
# ============================================================================

@dataclass
class ThickShellPly:
    """Single layer/ply in a layered thick shell composite (/PROP/TYPE21).

    Parameters
    ----------
    thick : float
        Physical layer thickness (length unit, e.g. mm or m). Must be > 0.
    angle : float
        Fiber orientation angle theta in degrees (relative to element or reference frame).
    mat_id : int
        Material identification number for this ply.
    nip : int
        Number of through-thickness integration points in this ply (default 1).
    """
    thick: float
    angle: float = 0.0
    mat_id: int = 0
    nip: int = 1

    def __post_init__(self) -> None:
        self.thick = max(float(self.thick), 1.0e-12)
        self.angle = float(self.angle)
        self.mat_id = int(self.mat_id)
        self.nip = max(1, int(self.nip))


# ============================================================================
# /PROP/TYPE20: Standard Thick Shell (TSHELL)
# ============================================================================

@dataclass
class Prop20ThickShell:
    """OpenRadioss /PROP/TYPE20 (or /PROP/TSHELL) solid-shell property.

    Upstream Fortran reference:
    `starter/source/properties/thickshell/hm_read_prop20.F` (SUBROUTINE HM_READ_PROP20)

    Attributes
    ----------
    id : int
        Property identifier number.
    title : str
        Property title or label.
    isolid : int
        Formulation flag (hm_read_prop20.F lines 123, 153-162):
        - 14: Reduced or selective integration thick shell
        - 15: Standard ANS / MITC thick shell (default)
        - 16: Enhanced Assumed Strain (EAS) thick shell
    formulation : str
        Formulation name ("ANS", "EAS", "STANDARD", "REDUCED").
    n_ip : int
        In-plane integration rule:
        - 4: 2x2 Gauss integration (default)
        - 1: 1-point reduced integration with physical hourglass control
    n_thick : int
        Number of through-thickness integration points (NPT, hm_read_prop20.F lines 202-215).
        Default is 3 (can be 1, 2, 3, 5, 7, 9).
    shear_corr : float
        Transverse shear correction factor (default 5/6 ~= 0.8333333333333334).
    inpts_r : int
        Integration points along local r (xi) direction (default 2).
    inpts_s : int
        Integration points along local s (eta) direction (default 2).
    inpts_t : int
        Integration points along local t (zeta) thickness direction (default 2 or n_thick).
    cvis : float
        Hourglass numerical damping parameter (dn, default 0.1 for ISOLID=15).
    qa : float
        Quadratic bulk viscosity coefficient (default 1.1).
    qb : float
        Linear bulk viscosity coefficient (default 0.05).
    icstr : int
        Thickness direction flag (1=T / zeta, 10=S / eta, 100=R / xi; default 10 or 1).
    icontrol : int
        Solid distortion control flag (default 0).
    deltat_min : float
        Minimum time step for thick shell elements (default 0.0).
    h : float
        Reference physical thickness if provided (default 1.0).
    """
    id: int = 1
    title: str = ""
    isolid: int = 15
    formulation: str = "ANS"
    n_ip: int = 4
    n_thick: int = 3
    shear_corr: float = 5.0 / 6.0
    inpts_r: int = 2
    inpts_s: int = 2
    inpts_t: int = 3
    cvis: float = 0.1
    qa: float = 1.1
    qb: float = 0.05
    icstr: int = 1
    icontrol: int = 0
    deltat_min: float = 0.0
    h: float = 1.0

    def __post_init__(self) -> None:
        # Harmonize formulation string and isolid flag
        form_upper = str(self.formulation).strip().upper()
        if form_upper in ("EAS", "ENHANCED", "16"):
            self.isolid = 16
            self.formulation = "EAS"
        elif form_upper in ("REDUCED", "14"):
            self.isolid = 14
            self.formulation = "REDUCED"
        elif form_upper in ("STANDARD", "FULL"):
            self.isolid = 15
            self.formulation = "STANDARD"
        else:
            if self.isolid == 16:
                self.formulation = "EAS"
            elif self.isolid == 14:
                self.formulation = "REDUCED"
            else:
                self.isolid = 15
                self.formulation = "ANS"

        # In-plane integration points
        if self.isolid == 14 or form_upper == "REDUCED":
            self.n_ip = 1
        elif self.n_ip not in (1, 4):
            self.n_ip = 4

        # Through-thickness integration points (hm_read_prop20.F lines 204-205)
        self.n_thick = max(1, min(9, int(self.n_thick)))

        # Transverse shear factor
        if self.shear_corr <= 0.0:
            self.shear_corr = 5.0 / 6.0

        # Hourglass damping default (hm_read_prop20.F lines 267-271)
        if self.isolid != 15 and self.cvis == 0.1:
            self.cvis = 0.0

    def gauss_points_thickness(self, n: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return 1D Gauss-Legendre quadrature coordinates and weights along thickness [-1, +1].

        Parameters
        ----------
        n : int, optional
            Number of points (defaults to self.n_thick).

        Returns
        -------
        zeta : np.ndarray (n,)
            Through-thickness coordinates in [-1, +1].
        weights : np.ndarray (n,)
            Quadrature weights summing to 2.0.
        """
        n_pts = self.n_thick if n is None else max(1, int(n))
        return legendre_gauss_1d(n_pts)

    def in_plane_gauss_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 2D in-plane integration point coordinates and weights on [-1, +1]^2.

        Returns
        -------
        xi : np.ndarray
            Natural coordinates in xi direction.
        eta : np.ndarray
            Natural coordinates in eta direction.
        weights : np.ndarray
            Quadrature weights summing to 4.0 (area of [-1, +1]^2).
        """
        if self.n_ip == 1:
            return np.array([0.0]), np.array([0.0]), np.array([4.0])

        # 2x2 Gauss integration
        c = 1.0 / math.sqrt(3.0)
        xi = np.array([-c, c, -c, c], dtype=float)
        eta = np.array([-c, -c, c, c], dtype=float)
        w = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
        return xi, eta, w

    def all_integration_points(self) -> List[Tuple[float, float, float, float]]:
        """Return full 3D integration point coordinates (xi, eta, zeta, weight).

        Total sum of weights is 8.0 (volume of reference cube [-1, +1]^3).
        """
        xi_ip, eta_ip, w_ip = self.in_plane_gauss_points()
        zeta_th, w_th = self.gauss_points_thickness()

        pts = []
        for i in range(len(xi_ip)):
            for j in range(len(zeta_th)):
                pts.append((
                    float(xi_ip[i]),
                    float(eta_ip[i]),
                    float(zeta_th[j]),
                    float(w_ip[i] * w_th[j]),
                ))
        return pts

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Prop20ThickShell:
        """Construct Prop20ThickShell from dictionary of parameters."""
        prop_id = int(data.get("id", data.get("prop_id", 1)))
        title = str(data.get("title", f"PROP20_{prop_id}"))
        isolid = int(data.get("isolid", data.get("ISOLID", 15)))
        formulation = str(data.get("formulation", "ANS"))

        nbp = int(data.get("nbp", data.get("NBP", 0)))
        n_thick = int(data.get("n_thick", data.get("npts_t", 3)))
        inpts_r = int(data.get("inpts_r", data.get("npts_r", 2)))
        inpts_s = int(data.get("inpts_s", data.get("npts_s", 2)))
        inpts_t = int(data.get("inpts_t", data.get("npts_t", 3)))

        if nbp > 200:
            inpts_r = nbp // 100
            rem = nbp % 100
            inpts_s = rem // 10
            inpts_t = rem % 10
            n_thick = inpts_t
        elif nbp > 0:
            n_thick = nbp

        n_ip = 1 if (inpts_r == 1 and inpts_s == 1) else 4

        shear_corr = float(data.get("shear_corr", data.get("Ashear", 5.0 / 6.0)))
        cvis = float(data.get("cvis", data.get("dn", 0.1)))
        qa = float(data.get("qa", 1.1))
        qb = float(data.get("qb", 0.05))
        icstr = int(data.get("icstr", data.get("Icstr", 1)))
        icontrol = int(data.get("icontrol", data.get("Icontrol", 0)))
        deltat_min = float(data.get("deltat_min", data.get("deltaT_min", 0.0)))
        h = float(data.get("h", data.get("thick", 1.0)))

        return cls(
            id=prop_id,
            title=title,
            isolid=isolid,
            formulation=formulation,
            n_ip=n_ip,
            n_thick=n_thick,
            shear_corr=shear_corr,
            inpts_r=inpts_r,
            inpts_s=inpts_s,
            inpts_t=inpts_t,
            cvis=cvis,
            qa=qa,
            qb=qb,
            icstr=icstr,
            icontrol=icontrol,
            deltat_min=deltat_min,
            h=h,
        )


# ============================================================================
# /PROP/TYPE21: Layered Composite Thick Shell (TSHELL_COMP)
# ============================================================================

@dataclass
class Prop21ThickShellComposite:
    """OpenRadioss /PROP/TYPE21 (or /PROP/TSH_ORTH / /PROP/TSH_COMP) composite thick shell.

    Upstream Fortran reference:
    `starter/source/properties/thickshell/hm_read_prop21.F` (SUBROUTINE HM_PROP_READ21)

    Attributes
    ----------
    id : int
        Property identifier number.
    title : str
        Property title.
    isolid : int
        Formulation flag (14=reduced/standard, 15=ANS/MITC, 16=EAS).
    formulation : str
        Formulation name ("ANS", "EAS", "STANDARD", "REDUCED").
    n_ip : int
        In-plane integration rule (4=2x2 Gauss, 1=reduced).
    shear_corr : float
        Transverse shear correction factor (default 5/6).
    cvis : float
        Hourglass numerical damping (default 0.1).
    qa : float
        Quadratic bulk viscosity (default 1.1).
    qb : float
        Linear bulk viscosity (default 0.05).
    skew_id : int
        Optional skew coordinate system ID for orthotropic material orientation.
    iorth : int
        Orthotropy definition flag (hm_read_prop21.F line 126).
    vx, vy, vz : float
        Reference vector V defining local fiber orientation frame.
    angle : float
        Global reference orientation angle beta in degrees.
    deltat_min : float
        Minimum time step.
    plies : List[ThickShellPly]
        List of composite plies through the thickness.
    """
    id: int = 1
    title: str = ""
    isolid: int = 15
    formulation: str = "ANS"
    n_ip: int = 4
    shear_corr: float = 5.0 / 6.0
    cvis: float = 0.1
    qa: float = 1.1
    qb: float = 0.05
    skew_id: int = 0
    iorth: int = 0
    vx: float = 1.0
    vy: float = 0.0
    vz: float = 0.0
    angle: float = 0.0
    deltat_min: float = 0.0
    plies: List[ThickShellPly] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.isolid not in (14, 15, 16):
            self.isolid = 15
        if self.shear_corr <= 0.0:
            self.shear_corr = 5.0 / 6.0
        if self.vx == 0.0 and self.vy == 0.0 and self.vz == 0.0:
            self.vx = 1.0

    def add_ply(
        self,
        thick: float,
        angle: float = 0.0,
        mat_id: int = 0,
        nip: int = 1,
    ) -> None:
        """Add a composite ply to the thick shell stack."""
        self.plies.append(ThickShellPly(
            thick=thick,
            angle=angle,
            mat_id=mat_id,
            nip=nip,
        ))

    def total_thickness(self) -> float:
        """Return total physical thickness H = sum_{k=1}^K t_k of the composite layup."""
        if not self.plies:
            return 1.0
        return sum(ply.thick for ply in self.plies)

    def num_plies(self) -> int:
        """Return number of plies in the composite layup."""
        return len(self.plies)

    def total_integration_points(self) -> int:
        """Return total number of through-thickness integration points across all plies."""
        if not self.plies:
            return 1
        return sum(ply.nip for ply in self.plies)

    def thickness_mapping(self) -> List[Dict[str, Any]]:
        """Compute exact through-thickness Gauss integration mapping for each ply.

        Maps each ply's local Gauss points [-1, +1] into the global normalized
        thickness interval [-1, +1] and physical coordinates [-H/2, +H/2].

        Returns
        -------
        points : List[Dict[str, Any]]
            List of integration point dictionaries with keys:
            - "zeta": global normalized coordinate in [-1, +1]
            - "z": physical thickness coordinate in [-H/2, +H/2]
            - "weight": global quadrature weight (sum of all weights is 2.0)
            - "ply_idx": 0-based ply index
            - "angle": ply fiber angle in degrees
            - "mat_id": ply material ID
            - "thick_ply": thickness of this ply
        """
        if not self.plies:
            return [{
                "zeta": 0.0,
                "z": 0.0,
                "weight": 2.0,
                "ply_idx": 0,
                "angle": 0.0,
                "mat_id": 0,
                "thick_ply": 1.0,
            }]

        H = self.total_thickness()
        z_curr = -0.5 * H
        mapping = []

        for ply_idx, ply in enumerate(self.plies):
            t_k = ply.thick
            z_bot = z_curr
            z_top = z_curr + t_k
            z_mid = 0.5 * (z_bot + z_top)
            z_curr = z_top

            # Normalized interval [zeta_bot, zeta_top] in [-1, +1]
            zeta_bot = 2.0 * z_bot / H
            zeta_top = 2.0 * z_top / H
            delta_zeta = zeta_top - zeta_bot
            zeta_mid = 0.5 * (zeta_bot + zeta_top)

            # Local Gauss points within this ply
            local_zeta, local_w = legendre_gauss_1d(ply.nip)

            for j in range(ply.nip):
                zeta_global = zeta_mid + 0.5 * delta_zeta * local_zeta[j]
                z_phys = 0.5 * H * zeta_global
                # Quadrature weight scaled so total sum over all plies equals 2.0
                weight_global = 0.5 * delta_zeta * local_w[j]

                mapping.append({
                    "zeta": float(zeta_global),
                    "z": float(z_phys),
                    "weight": float(weight_global),
                    "ply_idx": ply_idx,
                    "angle": float(ply.angle),
                    "mat_id": int(ply.mat_id),
                    "thick_ply": float(t_k),
                })

        return mapping

    def gauss_points_thickness(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return array of through-thickness coordinates zeta and weights across the layup."""
        mp = self.thickness_mapping()
        zetas = np.array([pt["zeta"] for pt in mp], dtype=float)
        weights = np.array([pt["weight"] for pt in mp], dtype=float)
        return zetas, weights

    def in_plane_gauss_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 2D in-plane integration point coordinates and weights on [-1, +1]^2."""
        if self.n_ip == 1:
            return np.array([0.0]), np.array([0.0]), np.array([4.0])

        c = 1.0 / math.sqrt(3.0)
        xi = np.array([-c, c, -c, c], dtype=float)
        eta = np.array([-c, -c, c, c], dtype=float)
        w = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
        return xi, eta, w

    def all_integration_points(self) -> List[Dict[str, Any]]:
        """Return full 3D integration points combining in-plane and thickness points."""
        xi_ip, eta_ip, w_ip = self.in_plane_gauss_points()
        th_pts = self.thickness_mapping()

        all_pts = []
        for i in range(len(xi_ip)):
            for pt in th_pts:
                w_3d = float(w_ip[i] * pt["weight"])
                all_pts.append({
                    "xi": float(xi_ip[i]),
                    "eta": float(eta_ip[i]),
                    "zeta": pt["zeta"],
                    "z": pt["z"],
                    "weight": w_3d,
                    "ply_idx": pt["ply_idx"],
                    "angle": pt["angle"],
                    "mat_id": pt["mat_id"],
                })
        return all_pts

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Prop21ThickShellComposite:
        """Construct Prop21ThickShellComposite from dictionary."""
        prop_id = int(data.get("id", data.get("prop_id", 1)))
        title = str(data.get("title", f"PROP21_{prop_id}"))
        isolid = int(data.get("isolid", data.get("ISOLID", 15)))
        formulation = str(data.get("formulation", "ANS"))
        n_ip = int(data.get("n_ip", 4))
        shear_corr = float(data.get("shear_corr", 5.0 / 6.0))
        cvis = float(data.get("cvis", data.get("dn", 0.1)))
        qa = float(data.get("qa", 1.1))
        qb = float(data.get("qb", 0.05))
        skew_id = int(data.get("skew_id", data.get("SKEW_CSID", 0)))
        iorth = int(data.get("iorth", data.get("Iorth", 0)))
        vx = float(data.get("vx", data.get("VECTOR_X", 1.0)))
        vy = float(data.get("vy", data.get("VECTOR_Y", 0.0)))
        vz = float(data.get("vz", data.get("VECTOR_Z", 0.0)))
        angle = float(data.get("angle", data.get("MAT_BETA", 0.0)))
        deltat_min = float(data.get("deltat_min", data.get("deltaT_min", 0.0)))

        obj = cls(
            id=prop_id,
            title=title,
            isolid=isolid,
            formulation=formulation,
            n_ip=n_ip,
            shear_corr=shear_corr,
            cvis=cvis,
            qa=qa,
            qb=qb,
            skew_id=skew_id,
            iorth=iorth,
            vx=vx,
            vy=vy,
            vz=vz,
            angle=angle,
            deltat_min=deltat_min,
        )

        # Parse layers if provided in list
        layers = data.get("plies", data.get("layers", []))
        for layer in layers:
            if isinstance(layer, ThickShellPly):
                obj.plies.append(layer)
            elif isinstance(layer, dict):
                obj.add_ply(
                    thick=float(layer.get("thick", layer.get("thickness", 1.0))),
                    angle=float(layer.get("angle", layer.get("phi", 0.0))),
                    mat_id=int(layer.get("mat_id", layer.get("mid", 0))),
                    nip=int(layer.get("nip", layer.get("npt", 1))),
                )
            elif isinstance(layer, (tuple, list)):
                t = float(layer[0])
                ang = float(layer[1]) if len(layer) > 1 else 0.0
                mid = int(layer[2]) if len(layer) > 2 else 0
                nip = int(layer[3]) if len(layer) > 3 else 1
                obj.add_ply(thick=t, angle=ang, mat_id=mid, nip=nip)

        return obj
