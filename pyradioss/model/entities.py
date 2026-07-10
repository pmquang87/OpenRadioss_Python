"""
Model entities: materials, properties, parts, groups, loads, constraints.

Fortran origin: in OpenRadioss these are rows of big 2-D arrays —
``PM(NPROPM, NUMMAT)`` for materials, ``GEO(NPROPG, NUMGEO)`` for
properties, ``IPART`` for parts — with *positional* meaning documented only
in scattered comments (e.g. PM(20) = Young modulus, PM(22) = shear
modulus...). The port replaces them with small dataclasses whose fields are
named and documented; the Starter still converts user IDs to dense indices
exactly like the Fortran ``USR2SYS`` machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ============================================================================
# Materials  (Fortran: PM array, filled by starter/source/materials/mat/...)
# ============================================================================

@dataclass
class Material:
    """One /MAT law. Only the fields common to all laws live here; law
    parameters are in ``params``, interpreted by the material kernel.

    Attributes
    ----------
    id, title : user id and title
    law       : integer law number (1 = elastic, 2 = Johnson-Cook, ...)
    rho0      : initial density (PM(1) 'RHO0' in the Fortran)
    params    : law-specific constants, e.g. E, nu, A, B, n, c, eps0...
    """

    id: int
    law: int
    rho0: float
    title: str = ""
    params: Dict[str, float] = field(default_factory=dict)

    # Convenience elastic constants (every implemented law defines these;
    # they drive the sound speed / time step and contact stiffness).
    @property
    def E(self) -> float:
        return self.params["E"]

    @property
    def nu(self) -> float:
        return self.params["nu"]

    @property
    def G(self) -> float:
        """Shear modulus G = E / 2(1+nu)."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def K(self) -> float:
        """Bulk modulus K = E / 3(1-2nu)."""
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    def sound_speed_solid(self) -> float:
        """3-D dilatational wave speed c = sqrt((K + 4G/3)/rho).

        This is the speed of the fastest (P-) wave in the continuum and is
        what the Courant stability condition of an explicit solid element
        must resolve (Fortran: computed in each material's ini routine,
        e.g. starter/source/materials/mat/mat001/ini... -> PM(27) 'SSP').
        """
        return float(np.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0))

    def sound_speed_shell(self) -> float:
        """Plane-stress wave speed c = sqrt(E / (rho (1 - nu^2))).

        Shells use the plane-stress modulus because the through-thickness
        stress is zero, which softens the response relative to 3-D.
        """
        return float(np.sqrt(self.E / (self.rho0 * (1.0 - self.nu ** 2))))


# ============================================================================
# Properties  (Fortran: GEO array, starter/source/properties/...)
# ============================================================================

@dataclass
class Property:
    """One /PROP geometry set.

    ``type`` follows the Radioss numbering:
      1  SHELL   (thickness, n. of integration points, hourglass coeffs)
      2  TRUSS   (cross-section area)
      4  SPRING  (mass, stiffness, damping)
      14 SOLID   (quadratic/linear bulk viscosity, hourglass coeff)
    """

    id: int
    type: int
    title: str = ""
    params: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Parts  (Fortran: IPART, starter/source/model/sets/../hm_read_part.F)
# ============================================================================

@dataclass
class Part:
    """A /PART ties elements to one material + one property (like the
    Fortran IPART(1:4,:) = prop, mat, ... mapping). Elements reference the
    part; the part references material and property."""

    id: int
    prop_id: int
    mat_id: int
    title: str = ""


# ============================================================================
# Groups & surfaces (Fortran: groupdef_mod.F, IGRNOD/IGRSURF)
# ============================================================================

@dataclass
class Box:
    """A /BOX/RECTA rectangular box, used by /GRNOD/BOX to select nodes."""

    id: int
    corner_min: np.ndarray  # (3,)
    corner_max: np.ndarray  # (3,)
    title: str = ""


@dataclass
class NodeGroup:
    """A /GRNOD node group. After Starter resolution, ``node_idx`` holds
    dense 0-based node indices (Fortran IGRNOD(IGR)%ENTITY)."""

    id: int
    title: str = ""
    # Unresolved content, as read from the deck:
    node_ids: List[int] = field(default_factory=list)   # /GRNOD/NODE
    part_ids: List[int] = field(default_factory=list)   # /GRNOD/PART
    box_ids: List[int] = field(default_factory=list)    # /GRNOD/BOX
    # Resolved by the Starter:
    node_idx: Optional[np.ndarray] = None


@dataclass
class Surface:
    """A /SURF contact surface: a set of 3/4-node segments.

    Fortran: IGRSURF(ISU)%NODES(NSEG,4). Segments from /SURF/PART are the
    free (outer) faces of the part's elements — extracted by the Starter,
    like the Fortran surface-from-part builder in starter/source/model/sets.
    """

    id: int
    title: str = ""
    part_ids: List[int] = field(default_factory=list)         # /SURF/PART
    seg_nodes: List[List[int]] = field(default_factory=list)  # /SURF/SEG (user ids)
    # Resolved by the Starter: (nseg, 4) 0-based node indices; triangles
    # repeat the 3rd node in the 4th slot (Radioss convention).
    segments: Optional[np.ndarray] = None


# ============================================================================
# Loads & constraints
# ============================================================================

@dataclass
class BoundaryCondition:
    """/BCS: fixes translational/rotational DOFs of a node group.

    ``trarot`` is the classic Radioss 6-character flag string 'XYZ XYZ'
    (e.g. '111 000' fixes all translations); stored as two boolean triples.
    """

    id: int
    grnod_id: int
    fix_tra: np.ndarray  # (3,) bool
    fix_rot: np.ndarray  # (3,) bool
    title: str = ""


@dataclass
class InitialVelocity:
    """/INIVEL/TRA: initial translational velocity on a node group."""

    id: int
    grnod_id: int
    v: np.ndarray  # (3,)
    title: str = ""


@dataclass
class Gravity:
    """/GRAV: body acceleration a(t) = scale * funct(t) * direction."""

    id: int
    grnod_id: Optional[int]  # None = all nodes
    funct_id: int
    direction: np.ndarray    # (3,) unit vector
    scale: float = 1.0
    title: str = ""


@dataclass
class ConcentratedLoad:
    """/CLOAD: nodal force F(t) = scale * funct(t) along a fixed direction,
    applied to every node of the group."""

    id: int
    grnod_id: int
    funct_id: int
    direction: np.ndarray  # (3,) unit vector
    scale: float = 1.0
    title: str = ""


@dataclass
class ImposedVelocity:
    """/IMPVEL: imposed velocity v(t) = scale * funct(t) on one DOF of a
    node group (kinematic condition: overrides the solution, does not add a
    force; the reaction is recovered from the mass * acceleration)."""

    id: int
    grnod_id: int
    funct_id: int
    dof: int              # 0=x,1=y,2=z
    scale: float = 1.0
    title: str = ""


@dataclass
class RigidWall:
    """/RWALL/PLANE: infinite rigid plane, kinematic treatment.

    Fortran: engine/source/constraints/general/rwall/. A node that ends the
    cycle on the wrong side of the plane is projected back and its normal
    velocity is removed (slide=0) or its full velocity zeroed (tied).
    """

    id: int
    point: np.ndarray    # (3,) a point on the plane (M)
    normal: np.ndarray   # (3,) outward unit normal (side where nodes live)
    slide: int = 0       # 0=sliding, 1=tied, 2=sliding with friction
    fric: float = 0.0
    grnod_id: Optional[int] = None  # None = all nodes are candidates
    dist: float = 0.0    # activation distance (search band), 0 = auto
    title: str = ""


@dataclass
class Interface7:
    """/INTER/TYPE7 penalty contact: candidate *secondary* nodes (a group)
    against a *main* surface. See pyradioss/contact/inter_type7.py for the
    ported mechanics and simplifications."""

    id: int
    grnod_id: int        # secondary nodes
    surf_id: int         # main surface
    stfac: float = 1.0   # stiffness scale factor (Istf default variant)
    fric: float = 0.0    # Coulomb friction coefficient
    gap: float = 0.0     # contact gap (0 = auto from element sizes)
    title: str = ""


@dataclass
class THRequest:
    """/TH/NODE or /TH/PART: time-history output request."""

    id: int
    kind: str            # 'NODE' | 'PART'
    ids: List[int] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)  # e.g. DX, VX, IE
    title: str = ""
