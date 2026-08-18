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
from typing import Dict, List, Optional, Union

import numpy as np


# ============================================================================
# Materials  (Fortran: PM array, filled by starter/source/materials/mat/...)
# ============================================================================

@dataclass
class FailureModel:
    """One /FAIL option, attached to a material (the keyword carries the
    material id: ``/FAIL/JOHNSON/mat_ID``).

    Fortran origin: the FAIL_PARAM structures of ``fail_param_mod.F``
    filled by ``starter/source/materials/fail/hm_read_fail*.F``.

    Attributes
    ----------
    type     : 'JOHNSON' | 'BIQUAD' (see pyradioss.failure)
    params   : criterion constants (D1..D4 / c1..c5 + derived fits)
    ifail_sh : shell deletion rule — 1 = delete when ONE integration
               layer is broken (default), 2 = when ALL layers are broken
    """

    type: str
    params: Dict[str, float] = field(default_factory=dict)
    ifail_sh: int = 1


@dataclass
class EquationOfState:
    """One /EOS option, attached to a material like /FAIL is (the keyword
    carries the material id: ``/EOS/POLYNOMIAL/mat_ID``).

    Fortran origin: the EOS_PARAM structures filled by
    ``starter/source/materials/eos/hm_read_eos.F``. Ported kinds:

    * ``POLYNOMIAL`` — p = C0 + C1 mu + C2 mubar^2 + C3 mu^3
      + (C4 + C5 mu) E  (params c0..c5, e0 = initial energy per V0);
    * ``IDEAL-GAS``   — stored as the equivalent polynomial
      (C4 = C5 = gamma - 1, e0 = P0/(gamma - 1)).

    ``rho0`` is copied from the host material at resolve time (the sound
    speed needs it). See pyradioss/materials/eos.py for the theory.
    """

    kind: str
    params: Dict[str, float] = field(default_factory=dict)
    rho0: float = 0.0


@dataclass
class Material:
    """One /MAT law. Only the fields common to all laws live here; law
    parameters are in ``params``, interpreted by the material kernel.

    Attributes
    ----------
    id, title : user id and title
    law       : integer law number (1 = elastic, 2 = Johnson-Cook,
                27 = brittle, 36 = tabulated, 42 = Ogden, ...)
    rho0      : initial density (PM(1) 'RHO0' in the Fortran)
    params    : law-specific constants, e.g. E, nu, A, B, n, c, eps0...
                (for LAW42 the parse stores the DERIVED E from
                G0 = sum(mu_p*alpha_p)/2 and nu, so the generic elastic
                properties below work for every law)
    fail      : optional /FAIL criterion attached to this material
    eos       : optional /EOS attached to this material (M6): the EOS
                pressure replaces the law's own for solid elements
    """

    id: int
    law: int
    rho0: float
    title: str = ""
    params: Dict[str, float] = field(default_factory=dict)
    fail: Optional[FailureModel] = None
    eos: Optional["EquationOfState"] = None

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
    """A /BOX volume, used by /GRNOD/BOX to select nodes (M37: the three
    real geometries of ``starter/source/model/box/rdbox.F``).

    ``kind`` selects the geometry and which fields are meaningful:

    * ``'RECTA'`` — axis-aligned box between two diagonal corners.
      Corners come either from the coordinate cards (``corner_min``/
      ``corner_max`` filled at read time) or from two NODES (``node1``/
      ``node2`` > 0 — resolved against the mesh at group-resolution
      time, the cfg recta.cfg N1/N2 fields); ``iskew`` > 0 evaluates
      limits in the local skew frame;
    * ``'CYLIN'`` — finite cylinder: axis segment ``p1``->``p2``,
      ``diameter``; a node is inside when its axis projection falls
      between the caps and its distance from the axis is <= D/2
      (rdbox.F INSIDE_CYLINDER, boundaries inclusive);
    * ``'SPHER'`` — sphere: center ``p1``, ``diameter``.
    """

    id: int
    corner_min: Optional[np.ndarray] = None  # (3,) RECTA
    corner_max: Optional[np.ndarray] = None  # (3,) RECTA
    title: str = ""
    kind: str = "RECTA"                      # 'RECTA' | 'CYLIN' | 'SPHER'
    p1: Optional[np.ndarray] = None          # (3,) CYLIN base / SPHER center
    p2: Optional[np.ndarray] = None          # (3,) CYLIN axis end
    diameter: float = 0.0                    # CYLIN / SPHER
    iskew: int = 0                           # CYLIN / SPHER
    node1: int = 0                           # RECTA/CYLIN corner/axis node
    node2: int = 0
    box_ids: List[int] = field(default_factory=list) # /BOX/BOX (M132): positive for union, negative for subtraction


@dataclass
class NodeGroup:
    """A /GRNOD node group. After Starter resolution, ``node_idx`` holds
    dense 0-based node indices (Fortran IGRNOD(IGR)%ENTITY).

    M37 adds the remaining real-deck subtypes (48 % of the official
    corpus): nodes of surfaces (/GRNOD/SURF — hm_surfnod.F), recursive
    group-of-groups (/GRNOD/GRNOD — hm_grogronod.F, iterative fixpoint
    with cycle detection; NEGATIVE ids REMOVE the referenced group's
    nodes, and removal wins over addition whatever the order — the
    BUFTMP = -1 convention), nodes of element groups (/GRNOD/GRSHEL|
    GRSH3N|GRBRIC|... — hm_elngr*.F) and generated id ranges
    (/GRNOD/GENE first..last [+ GEN_INCR increment])."""

    id: int
    title: str = ""
    # Unresolved content, as read from the deck:
    node_ids: List[int] = field(default_factory=list)   # /GRNOD/NODE
    part_ids: List[int] = field(default_factory=list)   # /GRNOD/PART
    box_ids: List[int] = field(default_factory=list)    # /GRNOD/BOX
    surf_ids: List[int] = field(default_factory=list)   # /GRNOD/SURF (M37)
    line_ids: List[int] = field(default_factory=list)   # /GRNOD/LINE (M136)
    grnod_ids: List[int] = field(default_factory=list)  # /GRNOD/GRNOD, signed
    # /GRNOD/GRSHEL|GRSH3N|GRBRIC|GRTRUS|GRBEAM|GRSPRI: (family, group id)
    # pairs — family is the canonical element-group key ('SHEL', 'SH3N',
    # 'BRIC', ...), see Model.egroups (M37)
    egroup_refs: List[tuple] = field(default_factory=list)
    # /GRNOD/GENE (+ GEN_INCR): (first_id, last_id, incr) user-id ranges
    gene_ranges: List[tuple] = field(default_factory=list)
    # Resolved by the Starter:
    node_idx: Optional[np.ndarray] = None


@dataclass
class Surface:
    """A /SURF contact surface: a set of 3/4-node segments.

    Fortran: IGRSURF(ISU)%NODES(NSEG,4). Segments from /SURF/PART are the
    free (outer) faces of the part's elements — extracted by the Starter,
    like the Fortran surface-from-part builder in starter/source/model/sets.

    Since M4 every segment also records its *provenance* — which element it
    is a face of (Fortran IGRSURF%ELTYP/ELEM). Contact needs this twice:

    * the Radioss penalty stiffness and variable-gap formulas are written
      in terms of the parent element (shell thickness, solid volume...);
    * element deletion (/FAIL, M3): a segment whose parent element has
      GBUF%OFF = 0 must drop out of the main surface, so freshly created
      crack faces stop carrying contact forces (the IDEL treatment of the
      original interfaces).
    """

    id: int
    title: str = ""
    part_ids: List[int] = field(default_factory=list)         # /SURF/PART
    seg_nodes: List[List[int]] = field(default_factory=list)  # /SURF/SEG (user ids)
    # M37 subtypes:
    # /SURF/SURF — surface-of-surfaces (hm_read_surfsurf.F): the listed
    # surfaces' segments are CONCATENATED, resolved by iterative fixpoint
    # with cycle detection; a NEGATIVE id includes the surface with its
    # segment node order REVERSED (n4 n3 n2 n1 — the normal flips).
    surf_ids: List[int] = field(default_factory=list)
    # /SURF/GRSHEL | /SURF/GRSH3N — every element of the element group
    # becomes a segment (hm_surfgr2/surftage): (family, group id) pairs.
    egroup_refs: List[tuple] = field(default_factory=list)
    # Resolved by the Starter: (nseg, 4) 0-based node indices; triangles
    # repeat the 3rd node in the 4th slot (Radioss convention).
    segments: Optional[np.ndarray] = None
    # Provenance, parallel to ``segments`` (resolved by the Starter):
    # seg_gtype[i] = element-group attribute on Model ('shells', 'bricks',
    # 'tetras', 'sh3n') or '' for explicit /SURF/SEG segments;
    # seg_elem[i]  = row in that group (-1 for explicit segments).
    seg_gtype: Optional[np.ndarray] = None    # (nseg,) dtype '<U8'
    seg_elem: Optional[np.ndarray] = None     # (nseg,) int64
    # /SURF/PLANE (M92): infinite plane defined by point P1 and normal point P2
    plane_p1: Optional[np.ndarray] = None     # (3,) float [X_A, Y_A, Z_A]
    plane_p2: Optional[np.ndarray] = None     # (3,) float [X_B, Y_B, Z_B]
    # M115 extensions:
    mat_ids: List[int] = field(default_factory=list)   # /SURF/MAT
    prop_ids: List[int] = field(default_factory=list)  # /SURF/PROP
    box_ids: List[int] = field(default_factory=list)   # /SURF/BOX
    modifier: str = ""                                 # 'EXT', 'ALL', 'FREE'
    # /SURF/ELLIPSE (M132): ellipsoidal quadric surface
    ellipse_center: Optional[np.ndarray] = None        # (3,) [Xc, Yc, Zc]
    ellipse_semiaxes: Optional[np.ndarray] = None      # (3,) [a, b, c]
    ellipse_skew: int = 0
    # /SURF/CYL & /SURF/SPHER & /SURF/SUB (M134):
    cyl_center: Optional[np.ndarray] = None            # (3,) [X0, Y0, Z0]
    cyl_axis: Optional[np.ndarray] = None              # (3,) [Ax, Ay, Az]
    cyl_radius: float = 0.0
    cyl_length: float = 0.0
    spher_center: Optional[np.ndarray] = None          # (3,) [Xc, Yc, Zc]
    spher_radius: float = 0.0
    subset_surf_ids: List[int] = field(default_factory=list)


@dataclass
class Line:
    """A /LINE edge set: 2-node segments, the sides of /INTER/TYPE11
    edge-to-edge contact.

    Fortran: IGRSLIN(ISL)%NODES(NSEG,2) built by
    ``starter/source/model/sets/hm_read_lines.F``. The port supports

    * ``/LINE/SURF`` — every unique edge of the segments of the listed
      surfaces (with element provenance carried over from the surface, so
      edges of deleted elements drop out, exactly like surface segments);
    * ``/LINE/SEG``  — explicit node pairs;
    * ``/LINE/EDGE`` (M37) — only the BORDER edges of the listed
      surfaces: edges used by exactly ONE segment (``linedge.F``
      'REMOVAL OF INTERNAL SEGMENTS (EXCEPT BORDERS)' — interior edges,
      shared by two segments, are removed entirely, which turns the
      free boundary of a shell patch into a line);
    * ``/LINE/LINE`` (M37) — line-of-lines: the listed lines' edges
      concatenated (hm_lines_of_lines.F, fixpoint + cycle detection);
    * ``/LINE/PART`` (M37) — every 1-D element (truss/beam/spring) of
      the listed parts becomes an edge (elem_1D_line_buffer.F);
    * ``/LINE/BEAM``, ``/LINE/TRUSS``, ``/LINE/SPRING`` (M134) — direct 1D sets;
    * ``/LINE/BOX``, ``/LINE/CIRC``, ``/LINE/ALL`` (M134) — geometric/boundary lines.
    """

    id: int
    title: str = ""
    surf_ids: List[int] = field(default_factory=list)         # /LINE/SURF
    seg_nodes: List[List[int]] = field(default_factory=list)  # /LINE/SEG (user ids)
    edge_surf_ids: List[int] = field(default_factory=list)    # /LINE/EDGE (M37)
    line_ids: List[int] = field(default_factory=list)         # /LINE/LINE (M37)
    part_ids: List[int] = field(default_factory=list)         # /LINE/PART (M37)
    # M134 extensions:
    beam_ids: List[int] = field(default_factory=list)         # /LINE/BEAM
    truss_ids: List[int] = field(default_factory=list)        # /LINE/TRUSS
    spring_ids: List[int] = field(default_factory=list)       # /LINE/SPRING
    box_ids: List[int] = field(default_factory=list)          # /LINE/BOX
    circ_center: Optional[np.ndarray] = None                  # /LINE/CIRC center
    circ_radius: float = 0.0                                  # /LINE/CIRC radius
    circ_axis: Optional[np.ndarray] = None                    # /LINE/CIRC normal
    all_boundary: bool = False                                # /LINE/ALL
    # Resolved by the Starter: (nseg, 2) node indices + provenance
    # (same convention as Surface.seg_gtype/seg_elem).
    segments: Optional[np.ndarray] = None
    seg_gtype: Optional[np.ndarray] = None
    seg_elem: Optional[np.ndarray] = None


@dataclass
class EntityGroup:
    """An ELEMENT (or part) group: /GRSHEL, /GRSH3N, /GRBRIC, /GRTRUS,
    /GRBEAM, /GRSPRI, /GRQUAD and /GRPART (M37).

    Fortran origin: the IGRSH4N/IGRSH3N/IGRBRIC/... GROUP_ structures of
    ``groupdef_mod.F`` read by ``starter/source/groups/hm_lecgre.F``
    (direct element lists and parts) and ``hm_grogro.F`` (recursive
    group-of-groups with the same negative-id removal convention and
    iterative-fixpoint cycle detection as /GRNOD/GRNOD).

    ``family`` is the canonical element-family key ('SHEL', 'SH3N',
    'BRIC', 'QUAD', 'TRUS', 'BEAM', 'SPRI', 'PART' — GRBRIC covers ALL
    solids, bricks and tetras alike, exactly like IGRBRIC spans IXS).
    After resolution ``members`` lists (model element-group attribute,
    row indices) pairs — the port's dense equivalent of GROUP%ENTITY —
    and for family 'PART' ``part_ids_resolved`` holds the part ids.
    """

    id: int
    family: str
    title: str = ""
    # Unresolved content:
    elem_ids: List[int] = field(default_factory=list)   # direct element ids
    part_ids: List[int] = field(default_factory=list)   # /GR*/PART
    group_ids: List[int] = field(default_factory=list)  # group-of-groups, signed
    box_ids: List[int] = field(default_factory=list)    # /GR*/BOX (M136)
    surf_ids: List[int] = field(default_factory=list)   # /GR*/SURF (M136)
    # Resolved by the Starter:
    members: Optional[list] = None            # [(gtype attr, rows ndarray)]
    part_ids_resolved: Optional[list] = None  # family 'PART' only


# ============================================================================
# Loads & constraints
# ============================================================================

@dataclass
class BoundaryCondition:
    """/BCS: fixes translational/rotational DOFs of a node group.

    ``trarot`` is the classic Radioss 6-character flag string 'XYZ XYZ'
    (e.g. '111 000' fixes all translations); stored as two boolean triples.

    ``skew_id`` (M39): the DOFs are fixed in the axes of that /SKEW, not
    the global ones — the constraint condensation ROTATES with the skew.
    ``bcs1v`` (``engine/source/constraints/general/bcs/bcs1.F``) projects
    the component along each constrained skew axis out of both the
    acceleration and the velocity; with a /SKEW/MOV the axes are rebuilt
    every cycle, so the constraint plane turns with the nodes.  0 = the
    global system.
    """

    id: int
    grnod_id: int
    fix_tra: np.ndarray  # (3,) bool
    fix_rot: np.ndarray  # (3,) bool
    title: str = ""
    skew_id: int = 0
    skew_row: int = 0    # resolved SkewSet row (0 = global)


@dataclass
class AleBoundaryCondition:
    """/ALE/BCS: grid boundary conditions for ALE solver.
    
    Fixes the grid velocity (wx, wy, wz) and/or Lagrange conditions
    (lx, ly, lz) of a node group.
    """
    id: int
    grnod_id: int
    fix_w: np.ndarray  # (3,) bool for WX WY WZ
    fix_l: np.ndarray  # (3,) bool for LX LY LZ
    title: str = ""
    skew_id: int = 0
    skew_row: int = 0


@dataclass
class InitialVelocity:
    """/INIVEL/TRA: initial translational velocity on a node group.
    /INIVEL/AXIS (M5): initial *rotational* velocity field about an axis,
    v += omega * (d x (x0 - P)) — how a spinning body is initialized
    (Fortran: starter/source/initial_conditions/inivel/hm_read_inivel.F).

    kind='TRA' uses ``v``; kind='AXIS' uses ``omega``, ``axis`` (unit
    direction d) and ``origin`` (point P on the axis).

    ``frame_id`` / ``dir`` (M39): with a /FRAME the AXIS card's rotation
    runs about the frame's ``dir`` axis THROUGH THE FRAME ORIGIN, and its
    Vxt/Vyt/Vzt are components IN the frame — the Starter resolves both
    into ``axis``/``origin``/``v`` (hm_read_inivel.F 415-453 rotates Vt by
    the frame, 581-621 builds ``V = Vt + VR * (d x (X - O))``).  Without a
    frame the axis passes through the GLOBAL origin along the global
    ``dir``, which is that code's IFRA == 0 branch."""

    id: int
    grnod_id: int
    v: np.ndarray  # (3,)  (TRA)
    title: str = ""
    kind: str = "TRA"
    omega: float = 0.0
    axis: Optional[np.ndarray] = None    # (3,) unit vector (AXIS)
    origin: Optional[np.ndarray] = None  # (3,) point on the axis (AXIS)
    frame_id: int = 0                    # /FRAME (AXIS); 0 = global
    dir: int = 1                         # IDIR 1/2/3 = the frame's X'/Y'/Z'


@dataclass
class Gravity:
    """/GRAV: body acceleration a(t) = scale * funct(t) * direction."""

    id: int
    grnod_id: Optional[int]  # None = all nodes
    funct_id: int
    direction: np.ndarray    # (3,) unit vector
    scale: float = 1.0
    title: str = ""
    skew_id: int = 0
    sens_id: int = 0
    scale_x: float = 1.0


@dataclass
class ConcentratedLoad:
    """/CLOAD: nodal force F(t) = scale * funct(t) along a fixed direction,
    applied to every node of the group. ``sens_id`` (M6): the load is
    inactive until /SENSOR sens_id fires, then evaluates the curve with
    the shifted time f(t - t_fire)."""

    id: int
    grnod_id: int
    funct_id: int
    direction: np.ndarray  # (3,) unit vector
    scale: float = 1.0
    time_scale: float = 1.0
    sens_id: int = 0
    title: str = ""


@dataclass
class ImposedVelocity:
    """/IMPVEL: imposed velocity v(t) = scale * funct(t / xscale) on one
    DOF of a node group, active inside [tstart, tstop] (kinematic
    condition: overrides the solution, does not add a force; the reaction
    is recovered from the mass * acceleration).

    scale/xscale/tstart/tstop are the official card-2 fields Fscale_Y /
    Ascale_x / Tstart / Tstop (``starter/source/constraints/general/
    impvel/read_impvel.F``: zero Ascale_x or Fscale_Y defaults to 1, zero
    Tstop to infinity; ``engine/.../fixvel.F`` evaluates the curve at
    t * (1/Ascale_x) and skips the condition outside the time window)."""

    id: int
    grnod_id: int
    funct_id: int
    dof: int              # 0/1/2 = tra X/Y/Z, 3/4/5 = rot XX/YY/ZZ (M39)
    scale: float = 1.0    # Fscale_Y (curve ordinate scale)
    xscale: float = 1.0   # Ascale_x (curve abscissa scale, never 0)
    tstart: float = 0.0   # activation window
    tstop: float = 1.0e30
    sens_id: int = 0      # /SENSOR gate (parsed; engine gating not ported)
    title: str = ""
    skew_id: int = 0      # /SKEW: dof is the skew's axis, not the global one
    skew_row: int = 0     # resolved SkewSet row (0 = global)


@dataclass
class ImposedDisplacement:
    """/IMPDISP (M5): imposed displacement d(t) = scale * funct(t) on one
    DOF of a node group. Kinematic like /IMPVEL, but enforced at the
    *position* level: each cycle the velocity is set so the node lands
    exactly at x0 + d(t+dt) — no drift accumulation, unlike integrating an
    equivalent velocity curve.

    Fortran origin: ``engine/source/constraints/general/impvel/fixvel.F``
    (the same routine serves /IMPVEL and /IMPDISP through IFLAG).
    Card-2 fields as for :class:`ImposedVelocity`:
    d(t) = scale * funct(t / xscale) inside [tstart, tstop].

    ``skew_id`` (M39): the imposed component is the one along that /SKEW's
    ``dof`` axis (fixvel.F 390-418 projects the current velocity onto the
    skew axis, imposes the curve there and adds the correction back along
    the SAME axis, leaving the other two components free).
    """

    id: int
    grnod_id: int
    funct_id: int
    dof: int              # 0/1/2 = tra X/Y/Z, 3/4/5 = rot XX/YY/ZZ (M39)
    scale: float = 1.0    # Fscale_Y (curve ordinate scale)
    xscale: float = 1.0   # Ascale_x (curve abscissa scale, never 0)
    tstart: float = 0.0   # activation window
    tstop: float = 1.0e30
    sens_id: int = 0      # /SENSOR gate (parsed; engine gating not ported)
    title: str = ""
    skew_id: int = 0      # /SKEW: dof is the skew's axis, not the global one
    skew_row: int = 0     # resolved SkewSet row (0 = global)


@dataclass
class ImposedAcceleration:
    """/IMPACC (M92): imposed acceleration a(t) = scale * funct(t) on one
    DOF of a node group. Same card structure and fields as :class:`ImposedVelocity`.
    """

    id: int
    grnod_id: int
    funct_id: int
    dof: int              # 0/1/2 = tra X/Y/Z, 3/4/5 = rot XX/YY/ZZ (M39)
    scale: float = 1.0    # Fscale_Y (curve ordinate scale)
    xscale: float = 1.0   # Ascale_x (curve abscissa scale, never 0)
    tstart: float = 0.0   # activation window
    tstop: float = 1.0e30
    sens_id: int = 0      # /SENSOR gate (parsed; engine gating not ported)
    title: str = ""
    skew_id: int = 0      # /SKEW: dof is the skew's axis, not the global one
    skew_row: int = 0     # resolved SkewSet row (0 = global)


@dataclass
class PressureLoad:
    """/PLOAD (M5): follower pressure p(t) = scale * funct(t) on a /SURF.

    Fortran origin: ``engine/source/loads/general/pload/pload.F``. The
    pressure acts along the *current* segment normal (follower load, the
    normal is defined by the segment node ordering n1-n2-n3-n4, right-hand
    rule) and the resultant p*A is lumped to the corners (A/4 per quad
    corner, A/3 per triangle corner). Segments of /FAIL-deleted elements
    stop carrying pressure (a torn face is an open boundary).
    """

    id: int
    surf_id: int
    funct_id: int
    scale: float = 1.0
    sens_id: int = 0       # M6: /SENSOR gating (same semantics as /CLOAD)
    title: str = ""


@dataclass
class CentrifugalLoad:
    """/LOAD/CENTRI (M93) and /CENTRI (M198): Centrifugal rotational load / field.

    Fortran origin: ``starter/source/loads/general/load_centri/hm_read_load_centri.F``.
    """
    id: int
    funct_id: int = 0
    dir: str = "XX"         # rotation axis: X, Y, Z, XX, YY, ZZ
    frame_id: int = 0       # reference frame
    sens_id: int = 0        # /SENSOR gating
    grnod_id: int = 0       # node group
    ivar: int = 1           # 1 = ignore d_omega/dt, 2 = account for d_omega/dt
    scale_x: float = 1.0    # Ascalex (time scale)
    scale_y: float = 1.0    # Fscaley (rotational velocity scale)
    title: str = ""
    grnd_id: int = 0
    fct_id: int = 0
    node_orig: int = 0
    node_axis: int = 0
    omega: float = 0.0
    scale_z: float = 1.0


@dataclass
class ImposedTemperature:
    """/IMPTEMP (M93): imposed nodal temperature T(t) = scale * funct(t / xscale)
    on a node group.

    Fortran origin: ``starter/source/loads/thermal/imptemp/read_imptemp.F``.
    """
    id: int
    funct_id: int
    grnod_id: int
    sens_id: int = 0
    scale: float = 1.0      # Fscale_y (temperature ordinate scale)
    xscale: float = 1.0     # Ascale_x (time scale)
    tstart: float = 0.0     # T_start
    tstop: float = 1.0e30   # T_stop
    title: str = ""


@dataclass
class Damping:
    """/DAMP (M6, M108): Rayleigh MASS damping or relative/function damping —
    force f = -alpha m v on every node of the group, active in the [tstart, tstop] window.

    Fortran origin: ``engine/source/assembly/damping*.F``. The port
    integrates the mass-damping ODE exactly per cycle (integrating
    factor, see engine/damping.py) and books the removed kinetic energy
    into the DE ledger. The stiffness-proportional beta branch of full
    Rayleigh damping is not ported (needs K*v products)."""

    id: int
    grnod_id: int
    alpha: float
    tstart: float = 0.0
    tstop: float = 1e30
    title: str = ""
    kind: str = "GLOBAL"   # 'GLOBAL' | 'VREL' | 'FUNCT'
    skew_id: int = 0
    fct_id: int = 0
    alpha_x: float = 0.0
    alpha_y: float = 0.0
    alpha_z: float = 0.0



@dataclass
class Sensor:
    """/SENSOR (M6, M84, M97): an event source gating loads and interfaces.

    Fortran origin: ``starter/source/tools/sensor/hm_read_sensor.F`` +
    ``engine/source/tools/sensor/``. Ported types:
    * ``kind='TIME'``   — fires at tdelay
    * ``kind='DISP'``   — fires when displacement magnitude of node_id exceeds dmin
    * ``kind='VEL'``    — fires when velocity magnitude of node_id exceeds vmax
    * ``kind='NOT'``    — active when sens_id1 is not active
    * ``kind='AND'``    — active when both sens_id1 and sens_id2 are active
    * ``kind='OR'``     — active when either sens_id1 or sens_id2 is active
    * ``kind='DIST'``   — fires based on distance between node_id1 and node_id2 (M97)
    * ``kind='ENERGY'`` — fires on part/system internal or kinetic energy thresholds (M97)
    * ``kind='INTER'``  — fires on contact interface force thresholds (M97)
    * ``kind='RBODY'``  — fires on rigid body force/moment thresholds (M97)
    * ``kind='TEMP'``   — fires on nodal/group temperature thresholds (M97)
    Sensors latch or update dynamically — see engine/sensors.py."""

    id: int
    kind: str              # 'TIME' | 'DISP' | 'VEL' | 'NOT' | 'AND' | 'OR' | 'DIST' | 'ENERGY' | 'INTER' | 'RBODY' | 'TEMP'
    tdelay: float = 0.0    # Time delay before activation
    node_id: int = 0       # DISP, VEL
    dmin: float = 0.0      # DISP, DIST
    dmax: float = 0.0      # DIST
    vmax: float = 0.0      # VEL
    fcut: float = 0.0      # VEL, INTER
    sens_id1: int = 0      # NOT, AND, OR
    sens_id2: int = 0      # AND, OR
    # DIST (M97)
    node_id1: int = 0
    node_id2: int = 0
    # ENERGY (M97)
    part_id: int = 0
    subset_id: int = 0
    iselect: int = 1
    iemin: float = -1e30
    iemax: float = 1e30
    kemin: float = -1e30
    kemax: float = 1e30
    # INTER (M97)
    int_id: int = 0
    # RBODY (M97)
    rbody_id: int = 0
    # Common force/moment thresholds & direction (M97)
    dir: str = ""
    fmin: float = 0.0
    fmax: float = 0.0
    # TEMP (M97)
    grnod_id: int = 0
    tempmax: float = 1e30
    tempmin: float = 0.0
    tempmean: float = 1e30
    # NIC (M107)
    nij_max: float = 0.0
    fint_tens: float = 0.0
    fint_comp: float = 0.0
    mint_flex: float = 0.0
    mint_ext: float = 0.0
    spring_id: int = 0
    # M121: GAUGE, HIC, WORK, RWALL, XSECTION, DIST_SURF
    gauge_entries: List[Tuple[int, float, float]] = field(default_factory=list) # (gauge_id, fporp, fport)
    accel_id: int = 0
    hic_period: float = 0.0
    hic_val: float = 0.0
    gravity: float = 9.81
    work_max: float = 0.0
    sect_id: int = 0
    rwall_id: int = 0
    node_id3: int = 0
    node_id4: int = 0
    surf_id: int = 0
    skew_id: int = 0
    ax_dir: str = ""
    bend_dir: str = ""
    alpha: float = 0.0
    cfc: float = 0.0
    # Duration limit (M97)
    tmin: float = 0.0
    title: str = ""
    # M131: ACCE, PYTHON
    acc_entries: List[Tuple[int, str, float, float]] = field(default_factory=list) # (acc_id, dir, tomin, tmin)
    script_name: str = ""
    func_name: str = ""
    target_id: int = 0   # SPH, AIRBAG, MONVOL, SHELL, SOLID (M136)
    dflag: int = 0       # DIST deactivation flag (M165)


@dataclass
class Mpc:
    """/MPC (M6): one general linear multi-point constraint row,

        sum_k  coef_k * u(node_k, dof_k) = 0        (dof 1-3 = X,Y,Z
                                                     translations,
                                                     4-6 = rotations)

    imposed on velocities/accelerations (its time derivative — exact for
    the homogeneous row, see engine/mpc.py for the Lagrange treatment).

    Fortran origin: ``starter/source/constraints/general/mpc/
    hm_read_mpc.F`` + ``engine/source/constraints/general/mpc/``.
    """

    id: int
    node_ids: List[int] = field(default_factory=list)
    dofs: List[int] = field(default_factory=list)      # 1..6 (user input)
    coefs: List[float] = field(default_factory=list)
    title: str = ""


@dataclass
class AddedMass:
    """/ADMAS (M5, M139): concentrated or distributed non-structural mass.

    Fortran origin: ``starter/source/tools/admas/hm_read_admas.F``. Beyond
    its normal use (payload, joints), this is how a *moving rigid wall
    with a mass* is built in this port: the wall is tied to a node whose
    inertia comes from /ADMAS (see RigidWall.node_id).
    """

    id: int
    grnod_id: int
    mass: float
    title: str = ""
    mass_type: int = 0  # 0: per node, 1: total on nodes, 2: total on surface, 3: total on elements


@dataclass
class Section:
    """/SECT (M5): section-force output — the time history of the resultant
    force/moment transmitted through a cut of the mesh.

    Fortran origin: ``engine/source/tools/sect/`` (section.F, forint.F):
    the original accumulates the internal forces of the elements of one
    side at the section nodes. The port uses the equivalent *side-sum*
    identity, which needs only a node set: since the internal force vector
    of any element in equilibrium sums to zero over its own nodes (and its
    moments balance), summing the assembled internal forces over ALL nodes
    of one side leaves exactly the force the OTHER side's elements exert
    through the cut:

        F_sect = sum_{n in side} fint_n
        M_sect = sum_{n in side} [(x_n - x_ref) x fint_n + mint_n]

    ``grnod_id`` must therefore contain every node of one side of the cut,
    including the cut nodes themselves (a /GRNOD/PART of the side parts is
    the natural way to write it). The sign convention: the reported force
    is the force the excluded side applies to the included side.

    ``node_id_ref`` (optional): moment reference point = that node's
    current position (it rides the deformation); 0 = the fixed initial
    centroid of the side node set. Output goes to the T01 file via
    /TH/SECT (FX FY FZ MX MY MZ).
    """

    id: int
    grnod_id: int
    node_id_ref: int = 0
    title: str = ""


@dataclass
class RigidBody:
    """/RBODY and /RBE2 (M5): a set of slave nodes moving as one rigid
    body, represented by a master node.

    Fortran origin: ``starter/source/constraints/general/rbody/hm_read_rbody.F``
    (input + mass/inertia assembly in ``rbyini.F``) and the engine update
    ``engine/source/constraints/general/rbody/rbyfor.F`` / ``rbycor.F``;
    /RBE2 is ``constraints/general/rbe2``. Both are the same mechanics —
    a 6-DOF rigid equation of motion fed by the gathered slave forces —
    and share this entity:

    * ``kind='RBODY'``: the classic rigid body. The master node is usually
      a standalone (massless) node; with ``icog=1`` (default, the Radioss
      ICoG behaviour) the Starter MOVES it to the computed center of
      gravity. ``added_mass``/``jadd`` are extra mass/inertia lumped at
      the COG (the Radioss Mass and Jxx/Jyy/Jzz fields).
    * ``kind='RBE2'``: a rigid link. The master is a structural node kept
      at its own position (never relocated); its own mass and the forces
      of the elements attached to it enter the body EOM, so a deformable
      structure can hang off the master. Only the full 6-DOF tie is
      ported (the per-DOF flags of the Radioss card are not).

    The Starter fills the resolved fields: dense indices, the total mass
    (slaves + master-if-structural + added), the COG and the 3x3 inertia
    tensor about it (from the slave point masses + nodal shell inertias +
    jadd). Element deletion does NOT change any of this: a deleted
    element's mass stays on its nodes (the Radioss convention, see
    initialization.py), so the rigid-body inertia is constant for the
    whole run — computed once here, never updated.
    """

    id: int
    kind: str                 # 'RBODY' | 'RBE2'
    master_id: int            # user node id of the master node
    grnod_id: int             # slave node group
    added_mass: float = 0.0   # /RBODY Mass field (at the COG)
    jadd: Optional[np.ndarray] = None   # (3,) added Jxx Jyy Jzz (at the COG)
    ispher: int = 0           # 1 = spherical inertia tensor (average of diagonals)
    icog: int = 1             # 1 = move master to COG (RBODY default)
    # /RBODY sens_ID: 0 = the body is ACTIVE from t=0; nonzero = a /SENSOR
    # gates it, so it starts INACTIVE.  This mirrors the reference's
    # NPBY(7,N) ON/OFF flag, set by hm_read_rbody.F exactly this way
    # ("IF(ISENS == 0) THEN NPBY(7,NRB)=1 ELSE NPBY(7,NRB)=0").  The port
    # does NOT gate the rigid-body kinematics by sensor (the field is
    # warned as ignored by read_rbody and engine/rigid_body.py never reads
    # it); it is carried because the Starter's shared-node check needs the
    # ACTIVE/INACTIVE distinction to match checkrby.F — see
    # starter/initialization.initialize_rigid_bodies (M39 / M38-NEW-4).
    sens_id: int = 0
    title: str = ""
    #: /RBODY Skew_ID (M39): the axes the card's Jxx/Jyy/Jzz are written
    #: in.  ``inirby.F`` calls CHBAS(SKEW(1,NOSKEW), RBY) ONCE, at Starter
    #: time, to rotate that tensor into the global frame — so only the
    #: skew's INITIAL orientation matters even for a /SKEW/MOV (the body
    #: then carries its own rotation).  0 = the global system.
    skew_id: int = 0
    skew_row: int = 0         # resolved SkewSet row (0 = global)
    lagmul: bool = False      # /RBODY/LAGMUL: Lagrange multiplier formulation (M131)
    # Resolved by the Starter (initialize_rigid_bodies):
    master: int = -1                      # dense node index
    slaves: Optional[np.ndarray] = None   # dense node indices (no master)
    mass_total: float = 0.0               # incl. added mass
    xg: Optional[np.ndarray] = None       # (3,) center of gravity
    J: Optional[np.ndarray] = None        # (3,3) inertia tensor about xg


@dataclass
class Rbe3:
    """/RBE3 (M5): interpolation constraint — the motion of one dependent
    (reference) node is the weighted average of a set of independent
    (master) nodes, and a force applied at the reference node is
    distributed to the masters *without adding any stiffness*.

    Fortran origin: ``starter/source/constraints/general/rbe3/hm_read_rbe3.F``
    + ``engine/source/constraints/general/rbe3/rbe3f.F`` (force
    distribution) / ``rbe3v.F`` (kinematic update). The port supports one
    master node group with uniform unit weights (the per-set weights and
    per-DOF flags of the full card are not ported); see
    pyradioss/engine/rbe3.py for the interpolation/distribution math.
    """

    id: int
    ref_id: int               # user node id of the dependent node
    grnod_id: int             # independent (master) nodes
    title: str = ""


@dataclass
class RigidWall:
    """/RWALL — rigid wall, kinematic treatment. Since M5 three geometries
    and moving walls are ported.

    Fortran: engine/source/constraints/general/rwall/ (``rgwal0.F`` plane,
    ``rgwals.F`` sphere, ``rgwalc.F`` cylinder, ``rgwalt.F`` the moving
    variants). A node that would end the cycle behind the wall surface has
    its normal velocity replaced so it lands exactly ON the surface
    (relative to the wall's own motion); slide=0 keeps the tangential
    velocity, slide=1 ties it to the wall, slide=2 applies Coulomb
    friction.

    Geometry (``geom``):

    * 'PLANE' — infinite plane through ``point`` with outward unit
      ``normal`` (nodes live on the +normal side);
    * 'SPHER' — sphere of ``radius`` centered at ``point`` (nodes live
      outside; per-node normal = radial direction);
    * 'CYL'   — infinite cylinder of ``radius`` about the axis through
      ``point`` along ``normal`` (nodes outside).

    Moving walls (``node_id`` > 0): the wall geometry is tied to that
    node — it translates with the node's displacement and pushes with the
    node's velocity, and the contact impulses REACT on the node (Radioss
    moving-wall convention: the node's mass, e.g. from /ADMAS, and its
    /INIVEL make a free flying wall; an /IMPVEL on the node makes a
    velocity-driven wall, in which case the reaction is absorbed by the
    drive instead). The wall does not rotate (the axis/normal direction
    is constant), like the original.
    """

    id: int
    point: np.ndarray    # (3,) plane point / sphere center / cyl axis point
    normal: np.ndarray   # (3,) plane outward normal / cylinder axis / paral normal
    slide: int = 0       # 0=sliding, 1=tied, 2=sliding with friction
    fric: float = 0.0
    grnod_id: Optional[int] = None  # None = all nodes are candidates
    grnod_id2: Optional[int] = None # excluded nodes
    dist: float = 0.0    # activation distance (search band), 0 = auto
    title: str = ""
    geom: str = "PLANE"  # 'PLANE' | 'SPHER' | 'CYL' | 'PARAL'
    radius: float = 0.0  # SPHER / CYL
    node_id: int = 0     # > 0: wall tied to this (user id) node — moving
    axis1: Optional[np.ndarray] = None  # (3,) PARAL first edge vector
    axis2: Optional[np.ndarray] = None  # (3,) PARAL second edge vector


@dataclass
class RwallBox:
    """/RWALL/BOX (M136): Rigid bounding box wall."""
    id: int
    title: str = ""
    node_id: int = 0
    slide: int = 0
    fric: float = 0.0
    grnod_id: int = 0
    grnod_id2: int = 0
    dist: float = 0.0
    p1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    p2: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class RwallCone:
    """/RWALL/CONE (M136): Rigid conical wall."""
    id: int
    title: str = ""
    node_id: int = 0
    slide: int = 0
    fric: float = 0.0
    grnod_id: int = 0
    grnod_id2: int = 0
    dist: float = 0.0
    apex: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    angle: float = 0.0


@dataclass
class SectBox:
    """/SECT/BOX (M136): Section cutting defined by bounding box."""
    id: int
    title: str = ""
    box_id: int = 0
    grnod_id: int = 0
    frame_id: int = 0


@dataclass
class SectCut:
    """/SECT/CUT (M136): Section defined by cutting plane."""
    id: int
    title: str = ""
    orig: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    grnod_id: int = 0
    frame_id: int = 0


@dataclass
class Interface:
    """One /INTER contact interface. ``type`` selects the mechanics:

    * **7**  — penalty node-to-surface (pyradioss/contact/inter_type7.py):
      secondary node group vs main surface; ``grnod_id = 0`` means
      *self-impact* — the secondary side defaults to the nodes of the main
      surface itself, the Radioss single-surface convention;
    * **2**  — tied/kinematic (inter_type2.py): the secondary nodes are
      glued to their main segment for the whole run;
    * **11** — penalty edge-to-edge (inter_type11.py): secondary /LINE
      edges vs main /LINE edges.

    Penalty options (types 7 and 11), following the Radioss cards:

    sens_id (M6): types 7/11 only — the interface is inactive (no
    forces, no dt claim) until /SENSOR sens_id fires.

    istf  : stiffness definition flag —
            0 = main-side element stiffness scaled by ``stfac`` (default),
            1 = ``stfac`` IS the stiffness (a constant spring value),
            2/3/4/5 = combine main-segment and secondary-node stiffness as
            average / max / min / series (K_m*K_s/(K_m+K_s)).
    igap  : 0 = constant gap (``gap``, auto-computed when 0),
            1 = variable gap per pair from element sizes:
            g = g_s(node) + g_m(segment), floored by ``gap`` (Gap_min)
            and optionally capped by ``gap_max``.
            2 = scaled variable gap (scaled by ``fscale_gap``).
            3 = mesh-size limited variable gap.
    stfac : stiffness scale factor — or the stiffness itself for istf=1.
    fric  : Coulomb friction coefficient.
    gap   : constant gap / Gap_min (0 = auto from main element sizes).

    Fortran origin: ``starter/source/interfaces/int07|02|11/hm_read_*.F``
    (the option cards) and the ``INTBUF_TAB`` interface buffers.
    """

    id: int
    type: int = 7
    grnod_id: int = 0     # secondary nodes (7: 0 = self-impact; 2: required; 24: node-to-surface; 16: secondary nodes)
    surf_id: int = 0      # main surface (types 7, 2, 24)
    surf_id1: int = 0     # secondary surface (type 24 surface-to-surface)
    line_id1: int = 0     # secondary edges (type 11)
    line_id2: int = 0     # main edges (type 11)
    grbric_id1: int = 0   # secondary brick group (type 17) or main brick group (type 16)
    grbric_id2: int = 0   # main brick group (type 17)
    istf: int = 0
    itied: int = 0        # tied option flag (16, 17)
    lagmul: bool = False  # True if /INTER/LAGMUL
    hertz: bool = False   # True if /INTER/HERTZ
    igap: int = 0
    stfac: float = 1.0
    fric: float = 0.0
    gap: float = 0.0
    gap_max: float = 0.0  # igap=1 cap, 0 = no cap
    fscale_gap: float = 1.0   # Igap 2/3: gap scale factor (Fscale_gap, default 1.0)
    percent_mesh_size: float = 0.4  # Igap 3: mesh-size gap fraction (default 0.4)
    gap_max_m: float = 0.0 # gap_max_m for TYPE24
    dsearch: float = 0.0  # type 2: projection search distance (0 = auto)
    spotflag: int = 0     # type 2: tied rotational kinematics flag (1: tie rotations, 2: shell rotations)
    sens_id: int = 0      # M6: /SENSOR gating (types 7/11)
    multimp: int = 4      # type 10: max average number of impacted main segments
    idel10: int = 0       # type 10: node and segment deletion flag
    tstart: float = 0.0   # type 10: activation start time
    tstop: float = 1e30   # type 10: deactivation stop time
    inactiv: int = 0      # type 10: initial penetration deactivation flag
    stiff_dc: float = 0.05 # type 10: critical damping coefficient on interface stiffness (VISC)
    sort_fact: float = 0.20 # type 10: bucket sorting search factor
    # ---- friction MODELS (M15, Ifric > 0 — contact/friction.py) ----------
    # mfrot = Ifric (the MFROT law), fric_c = C1..C6, ifq = Ifiltr and
    # xfiltr the reader-derived filter coefficient (hm_read_inter_type07.F:
    # IFQ=1 -> Xfreq, IFQ=2 -> 2*pi/Xfreq, IFQ=3 -> 2*pi*Xfreq). On TYPE11
    # these fields are a documented port EXTENSION (the original TYPE11
    # card carries none — see contact/friction.py).
    mfrot: int = 0
    ifq: int = 0
    xfiltr: float = 0.0
    fric_c: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    title: str = ""
    # ---- TYPE18 (M60) / TYPE10 / TYPE19 / TYPE21 / GUIDED_CABLE ---------
    ibag: int = 0
    multimp: int = 4
    idel18: int = 0
    idel10: int = 0       # type 10: segment deletion flag
    idel: int = 0         # type 19: deletion flag
    icurv: int = 0        # type 19: curve geometry flag
    iadm: int = 0         # type 21: admission flag
    grpart_id: int = 0    # guided cable: part group
    istiff: int = 1       # guided cable: stiffness formulation flag
    gap_scale: float = 1.0 # type 19/21/25: scale factor for gap
    gap_min: float = 0.0   # type 19: min gap
    tstart: float = 0.0   # type 10: activation time
    tstop: float = 1e30   # type 10: deactivation time
    inactiv: int = 0      # type 10: initial penetration treatment
    stiff_dc: float = 0.0
    sort_fact: float = 0.2
    isym: int = 0         # type 20: symmetric contact flag
    iedge: int = 0        # type 20: edge contact flag
    edge_angle: float = 0.0 # type 20: edge angle
    iload: int = 0        # type 14: load formulation flag
    fun_id1: int = 0      # type 14: fct_ID1
    fun_id2: int = 0      # type 14: fct_ID2
    fscale_gap: float = 1.0 # type 23: gap scale
    idel: int = 0         # type 23: element deletion flag
    tol: float = 0.0      # type 12: tolerance
    visc: float = 0.0     # type 9: damping viscosity
    radius: float = 0.0   # type 17: contact radius
    stmin: float = 0.0    # type 19/25: min stiffness
    stmax: float = 0.0    # type 19/25: max stiffness
    ifric: int = 0        # type 25: friction flag
    ifiltr: int = 0       # type 25: filter flag
    xfreq: float = 0.0    # type 25: filter cutoff frequency
    isensor: int = 0      # type 25: sensor ID
    fric_id: int = 0      # type 25: friction law ID
    c1: float = 0.0       # type 25: friction constant 1
    c2: float = 0.0       # type 25: friction constant 2
    c3: float = 0.0       # type 25: friction constant 3
    c4: float = 0.0       # type 25: friction constant 4
    c5: float = 0.0       # type 25: friction constant 5


@dataclass
class THRequest:
    """/TH/NODE, /TH/PART, /TH/SECT and element/entity variants (M68):
    time-history output request."""

    id: int
    kind: str            # 'NODE' | 'PART' | 'SECT' | 'RBODY' | 'SHEL' |
                         # 'SH3N' | 'SPRING' | 'BRIC' | 'RWALL' | 'INTER'
    ids: List[Union[int, str]] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)  # e.g. DX, VX, IE
    title: str = ""

    @property
    def obj_ids(self) -> List[Union[int, str]]:
        return self.ids


@dataclass
class MonitoredVolume:
    """A /MONVOL monitored volume (e.g., AIRBAG1)."""
    id: int
    title: str = ""
    vol_type: str = "AIRBAG1"  # "AIRBAG1", "COMMU", etc.
    surf_id: int = 0
    hconv: float = 0.0
    
    # Material properties
    matid: int = 0
    mu: float = 0.0
    pext: float = 0.0
    t_initial: float = 293.0
    iequil: int = 0
    ittf: int = 0
    
    # Scaling factors (typically 1.0)
    scale_t: float = 1.0
    scale_p: float = 1.0
    scale_s: float = 1.0
    scale_a: float = 1.0
    scale_d: float = 1.0

    # Injectors (jets)
    injectors: List[Dict] = field(default_factory=list)
    # Ventholes and porous surfaces
    vents: List[Dict] = field(default_factory=list)
    porous_surfaces: List[Dict] = field(default_factory=list)


@dataclass
class Table:
    """``/TABLE/dim/table_ID`` (1D, 2D, 3D tabular functions; M139)."""
    id: int
    dim: int
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z: Optional[np.ndarray] = None
    curves: List[Tuple[float, np.ndarray, np.ndarray]] = field(default_factory=list)
    title: str = ""


@dataclass
class Random:
    """``/RANDOM/random_ID`` (Stochastic / random fields)."""
    id: int
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class Submodel:
    """``/SUBMODEL/submodel_ID`` container block (Fortran lecsubmod.F)."""
    id: int
    title: str = ""
    unit_id: int = 0
    off_def: int = 0
    off_nod: int = 0
    off_ele: int = 0
    off_part: int = 0
    off_mat: int = 0
    off_type: int = 0
    off_sub: int = 0


@dataclass
class Subdomain:
    """`/SUBDOMAIN/sub_id` domain partition for Rad2Rad coupling (lecextlnk.F).

    In the Fortran Starter, ``ISUBDOM(1,I)`` stores the count of parts,
    ``ISUBDOM(2,I)`` the user subdomain ID, and ``ISUBDOM_PART`` the
    flat list of internal part indices.  Here we store user-facing part
    IDs directly.
    """
    id: int
    title: str = ""
    part_ids: List[int] = field(default_factory=list)
    neg_part_ids: List[int] = field(default_factory=list)


@dataclass
class Xref:
    """`/XREF/part_id` reference geometry (hm_read_xref.F).

    Stores the initial reference configuration (undeformed coordinates)
    for elements of a given part — used for springback, pre-straining,
    or metric tensor initialization.
    """
    part_id: int
    title: str = ""
    nitrs: int = 100          # steps from reference to initial state
    node_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class DetonatorPoint:
    """`/DFS/DETPOINT/det_id` — Point-source detonation (read_dfs_detpoint.F).

    Ignites explosive elements from a point source; lighting time for each
    element is ``tdet + dist / D_det`` where D_det is the material's
    detonation velocity.
    """
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    grnod_id: int = 0
    node_id: int = 0


@dataclass
class DetonatorPlane:
    """`/DFS/DETPLAN/det_id` — Planar detonation front (read_dfs_detplan.F).

    Ignites explosive elements from a planar wave; the plane passes
    through point (x, y, z) with propagation normal (nx, ny, nz).
    """
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    nx: float = 0.0
    ny: float = 0.0
    nz: float = 0.0
    p_id: int = 0
    n_id: int = 0


@dataclass
class ConvectionLoad:
    """/CONVEC (M94): convection heat flux boundary condition on a /SURF.

    Fortran origin: ``starter/source/loads/thermic/hm_read_convec.F``.
    q = h * (T_surf - T_inf(t))
    """
    id: int
    surf_id: int
    funct_id: int
    sens_id: int = 0
    xscale: float = 1.0     # ASCALE (time scale for T_inf curve)
    scale: float = 1.0      # FSCALE (temperature scale)
    tstart: float = 0.0     # TSTART
    tstop: float = 1.0e30   # TSTOP
    h: float = 0.0          # H (convection coefficient)
    title: str = ""


@dataclass
class InivolContainer:
    """Container surface entry for /INIVOL (M94/M151)."""
    surf_id: int
    ale_phase: int = 1
    fill_opt: int = 0       # 0 = along normal, 1 = against normal (reversed)
    icumu: int = 0          # 0 = erase, 1 = additive, -1 = subtractive
    fill_ratio: float = 1.0 # filling volume fraction in [0, 1]

    @property
    def submat_id(self) -> int:
        return self.ale_phase

    @submat_id.setter
    def submat_id(self, val: int) -> None:
        self.ale_phase = val

    @property
    def ireversed(self) -> int:
        return self.fill_opt

    @ireversed.setter
    def ireversed(self, val: int) -> None:
        self.fill_opt = val

    @property
    def vfrac(self) -> float:
        return self.fill_ratio

    @vfrac.setter
    def vfrac(self, val: float) -> None:
        self.fill_ratio = val


@dataclass
class InitialVolume:
    """/INIVOL (M94/M151): initial volume fraction for multi-material fluid / ALE.

    Fortran origin: ``starter/source/initial_conditions/inivol/hm_read_inivol.F90``.
    """
    id: int
    part_id: int = 0
    title: str = ""
    containers: List[InivolContainer] = field(default_factory=list)


Inivol = InitialVolume


@dataclass
class RadiationLoad:
    """/RADIATION (M95): radiation heat flux boundary condition on a /SURF.

    Fortran origin: ``starter/source/loads/thermic/hm_read_radiation.F``.
    q = epsilon * sigma * (T_surf^4 - T_inf(t)^4)
    """
    id: int
    surf_id: int
    funct_id: int = 0       # time function for T_inf
    sens_id: int = 0
    xscale: float = 1.0     # ASCALE (time scale)
    scale: float = 1.0      # FSCALE (temperature scale)
    tstart: float = 0.0     # TSTART
    tstop: float = 1.0e30   # TSTOP
    emissivity: float = 0.0 # E (surface emissivity)
    title: str = ""


@dataclass
class ImposedFlux:
    """/IMPFLUX (M95/M152): imposed surface or volumetric heat flux.

    Fortran origin: ``starter/source/constraints/thermic/hm_read_impflux.F``.
    """
    id: int
    surf_id: int = 0        # surface ID for surfacic flux
    funct_id: int = 0       # time function for flux density
    sens_id: int = 0
    grbric_id: int = 0      # brick group ID for volumetric flux
    xscale: float = 1.0     # ASCALE
    scale: float = 1.0      # FSCALE
    tstart: float = 0.0     # TSTART
    tstop: float = 1.0e30   # TSTOP
    title: str = ""

    @property
    def fct_id(self) -> int:
        return self.funct_id

    @fct_id.setter
    def fct_id(self, val: int) -> None:
        self.funct_id = val

    @property
    def sensor_id(self) -> int:
        return self.sens_id

    @sensor_id.setter
    def sensor_id(self, val: int) -> None:
        self.sens_id = val

    @property
    def grbrick_id(self) -> int:
        return self.grbric_id

    @grbrick_id.setter
    def grbrick_id(self, val: int) -> None:
        self.grbric_id = val

    @property
    def scale_x(self) -> float:
        return self.xscale

    @scale_x.setter
    def scale_x(self, val: float) -> None:
        self.xscale = val

    @property
    def scale_y(self) -> float:
        return self.scale

    @scale_y.setter
    def scale_y(self, val: float) -> None:
        self.scale = val


ImpFlux = ImposedFlux


@dataclass
class InitialTemperature:
    """/INITEMP (M95): initial nodal temperature.

    Fortran origin: ``starter/source/initial_conditions/thermic/hm_read_initemp.F``.
    """
    id: int
    t0: float = 0.0
    grnod_id: int = 0
    fld_type: int = 0       # 0 = uniform on group, 1 = nodal table
    nodal_temps: Dict[int, float] = field(default_factory=dict)  # node_id -> temp
    title: str = ""


@dataclass
class InitialBrickState:
    """/INIBRI (M96, M138, M142): initial state for solid/brick elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F``.
    """
    elem_id: int
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [sxx, syy, szz, sxy, syz, sxz]
    eps: np.ndarray = field(default_factory=lambda: np.zeros(6))    # [exx, eyy, ezz, exy, eyz, exz] (M142)
    epsp: float = 0.0      # plastic strain
    rho: float = 0.0       # initial density
    ener: float = 0.0      # internal energy
    temp: float = 0.0      # initial temperature (M138)
    pres: float = 0.0      # initial hydrostatic pressure (M138)
    void: float = 0.0      # initial void fraction (M138)
    fail_flag: float = 0.0 # initial failure flag (M142)
    aux: float = 0.0       # auxiliary state variable (M142)
    scale_yld: float = 1.0 # yield stress scale factor (M142)


@dataclass
class InitialShellState:
    """/INISHE and /INISH3 (M96, M138, M142): initial state for shell elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F``.
    """
    elem_id: int
    thick: float = 0.0     # initial thickness override
    epsp: float = 0.0      # plastic strain
    epsp_layers: List[float] = field(default_factory=list) # per-layer plastic strain (M142)
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(6))  # membrane stress
    sigma_b: np.ndarray = field(default_factory=lambda: np.zeros(6)) # bending stress
    eps: np.ndarray = field(default_factory=lambda: np.zeros(6))    # strain tensor (M142)
    em: float = 0.0        # membrane energy
    eb: float = 0.0        # bending energy
    h_energy: np.ndarray = field(default_factory=lambda: np.zeros(3)) # H1, H2, H3
    temp: float = 0.0      # initial temperature (M138)
    rho: float = 0.0       # initial density (M178)
    fail_flag: float = 0.0 # initial failure flag (M142)
    aux: float = 0.0       # auxiliary state variable (M142)
    scale_yld: float = 1.0 # yield stress scale factor (M142)
    orth_angles: List[Tuple[float, float]] = field(default_factory=list) # orthotropy angles per layer (M199)
    orth_phi: List[float] = field(default_factory=list) # phi_i angles per layer (M199)
    orth_alpha: List[float] = field(default_factory=list) # alpha_i angles per layer (M199)



@dataclass
class InitialTrussState:
    """/INITRU (M97, M138): initial state for truss elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F`` and
    ``starter/source/elements/truss/tsigini.F``.
    """
    elem_id: int
    prop_type: int = 2
    eint: float = 0.0      # initial internal energy
    force: float = 0.0     # initial axial force / tension
    area: float = 0.0      # initial area override
    epsp: float = 0.0      # plastic strain
    temp: float = 0.0      # initial temperature (M138)


@dataclass
class InitialBeamState:
    """/INIBEA (M97, M138): initial state for beam elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F`` and
    ``starter/source/elements/beam/bsigini.F``.
    """
    elem_id: int
    prop_type: int = 3
    nb_integr: int = 0
    eint_memb: float = 0.0 # membrane internal energy
    eint_bend: float = 0.0 # bending internal energy
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))   # [Fx, Fy, Fz]
    moment: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [Mx, My, Mz]
    epsp: float = 0.0      # plastic strain
    temp: float = 0.0      # initial temperature (M138)


@dataclass
class InitialSpringState:
    """/INISPR (M97, M138): initial state for spring elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F`` and
    ``starter/source/elements/spring/rinit3.F``.
    """
    elem_id: int
    prop_type: int = 4
    force: float = 0.0     # initial force
    disp: float = 0.0      # initial displacement
    fep: float = 0.0       # elasto-plastic limit force
    dpl_pos: float = 0.0   # positive plastic displacement
    dpl_neg: float = 0.0   # negative plastic displacement
    length: float = 0.0    # initial length
    eint: float = 0.0      # internal energy
    temp: float = 0.0      # initial temperature (M138)


@dataclass
class CyclicBoundaryCondition:
    """/BCS/CYCLIC (M99): cyclic boundary condition linking two node groups.

    Fortran origin: ``starter/source/model/bcs/hm_read_bcscyclic.F`` and
    ``engine/source/assembly/bcs/``.
    """
    id: int
    title: str = ""
    skew_id: int = 0
    grnod1_id: int = 0
    grnod2_id: int = 0


@dataclass
class SolidPartPerturbation:
    """/PERTURB/PART/SOLID (M99): material/geometric perturbation on solid parts.

    Fortran origin: ``starter/source/model/perturbation/hm_read_perturb_solid.F``.
    """
    id: int
    title: str = ""
    f_mean: float = 0.0
    deviation: float = 0.0
    min_cut: float = 0.0
    max_cut: float = 0.0
    seed: int = 0
    idistri: int = 2
    grpart_id: int = 0
    var_name: str = ""


@dataclass
class PBlastLoad:
    """/LOAD/PBLAST (M99/M151): air/ground blast pressure load.

    Fortran origin: ``starter/source/model/loads/hm_read_pblast.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    exp_data: int = 1
    i_tshift: int = 1
    ndt: int = 0
    iz: int = 2
    imodel: int = 0
    node_id: int = 0
    xdet: float = 0.0
    ydet: float = 0.0
    zdet: float = 0.0
    tdet: float = 0.0
    wtnt: float = 0.0
    pmin: float = 0.0
    tstop: float = 1.0e30
    surf_ground_id: int = 0
    ishape: int = 0

    @property
    def iabac(self) -> int:
        return self.exp_data

    @iabac.setter
    def iabac(self, val: int) -> None:
        self.exp_data = val

    @property
    def ita_shift(self) -> int:
        return self.i_tshift

    @ita_shift.setter
    def ita_shift(self, val: int) -> None:
        self.i_tshift = val

    @property
    def iz_update(self) -> int:
        return self.iz

    @iz_update.setter
    def iz_update(self, val: int) -> None:
        self.iz = val


PblastLoad = PBlastLoad


@dataclass
class DetonationWave:
    """/INIT/DET_POINT, /INIT/DET_LINE, /INIT/DET_PLAN, /INIT/DET_CORD (M110):
    High-explosive detonation wavefront initialization.
    
    Fortran origin: ``starter/source/initial_conditions/detonation/``.
    """
    id: int
    kind: str = "POINT"  # 'POINT' | 'LINE' | 'PLAN' | 'CORD'
    title: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    z2: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    ddet: float = 0.0
    iopt: int = 0


@dataclass
class WaveShaper:
    """/DFS/WAVE_SHAPER, /INIT/DET/WAVE_SHAPER (M132): Explosive detonation wave shaper barrier."""
    id: int
    title: str = ""
    surf_id: int = 0
    mat_id: int = 0
    thick: float = 0.0
    delay: float = 0.0


@dataclass
class ElementActivation:
    """/ACTIV (M110): Dynamic activation/deactivation of element groups.
    
    Fortran origin: ``starter/source/tools/activ/hm_read_activ.F``.
    """
    id: int
    title: str = ""
    sens_id: int = 0
    grbric_id: int = 0
    grquad_id: int = 0
    grshel_id: int = 0
    grtrus_id: int = 0
    grbeam_id: int = 0
    grspri_id: int = 0
    iform: int = 1
    tstart: float = 0.0
    tstop: float = 1.0e30


Activ = ElementActivation


@dataclass
class PCylLoad:
    """/LOAD/PCYL, /PLOAD/PCYL (M130): Cylindrical coordinate pressure load."""
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    skew_id: int = 0
    table_id: int = 0
    xscale_r: float = 1.0
    xscale_t: float = 1.0
    yscale_p: float = 1.0


@dataclass
class PreloadAxial:
    """/PRELOAD/AXIAL, /LOAD/PRELOAD_AXIAL (M130): Axial spring/beam preload."""
    id: int
    title: str = ""
    set_id: int = 0
    sens_id: int = 0
    fun_id: int = 0
    preload: float = 1.0
    damp: float = 0.0


@dataclass
class LaserLoad:
    """/LOAD/LASER, /LASER, /DFS/LASER (M130): Laser beam impact load."""
    id: int
    title: str = ""
    slas: float = 0.0
    fct_idlas: int = 0
    star: float = 0.0
    fct_idtar: int = 0
    hn: float = 0.0
    vcp: float = 0.0
    k0: float = 0.0
    rd: float = 0.0
    ks: float = 0.0
    np: int = 0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeMergeOption:
    """/MERGE/NODE, /MERGE (M130): Node group merge tolerance option."""
    id: int
    title: str = ""
    tol: float = 0.0
    grnod_id: int = 0
    merge_type: int = 0


@dataclass
class LagmulGlobal:
    """/LAGMUL, /LAGMUL/OPTION (M131): Global Lagrange multiplier solver parameters."""
    lagmod: int = 1
    lagopt: int = 1
    tol: float = 1e-11
    alpha: float = 5e-4
    alpha_s: float = 0.0


@dataclass
class GearConstraint:
    """/GEAR, /LAGMUL/GEAR (M131): Rotational gear kinematic constraint."""
    id: int
    title: str = ""
    node1: int = 0
    node2: int = 0
    ratio: float = 1.0
    dir1: int = 1
    dir2: int = 1
    skew1: int = 0
    skew2: int = 0


@dataclass
class RackConstraint:
    """/RACK, /LAGMUL/RACK (M131): Rack-and-pinion kinematic constraint."""
    id: int
    title: str = ""
    node1: int = 0
    node2: int = 0
    pitch_radius: float = 1.0
    dir1: int = 1
    dir2: int = 1
    skew1: int = 0
    skew2: int = 0


@dataclass
class DiffConstraint:
    """/DIFF, /LAGMUL/DIFF (M131): Differential rotational kinematic constraint."""
    id: int
    title: str = ""
    node0: int = 0
    node1: int = 0
    node2: int = 0
    ratio: float = 1.0


@dataclass
class Ply:
    """/PLY/ply_id (M100, M156): Composite ply definition.

    Fortran origin: ``starter/source/model/laminate/leclamply.F``.
    """
    id: int
    mat_id: int
    thick: float
    title: str = ""
    skew_id: int = 0
    orientangle: float = 0.0
    grsh4n_id: int = 0
    grsh3n_id: int = 0
    nip: int = 1
    orientangle2: float = 0.0


@dataclass
class LaminatePly:
    """Layer definition inside a /LAMINATE stack."""
    ply_id: int
    phi: float = 0.0
    zi: float = 0.0
    mat_interply: int = 0
    f_weight: float = 1.0


@dataclass
class Laminate:
    """/LAMINATE/laminate_id (M100): Composite laminate stack definition.

    Fortran origin: ``starter/source/model/laminate/leclam.F``.
    """
    id: int
    title: str = ""
    plies: List[LaminatePly] = field(default_factory=list)


@dataclass
class SubInterface:
    """/INTER/SUB/sub_inter_ID (M100): Contact sub-interface.

    Fortran origin: ``starter/source/interfaces/sub/hm_read_inter_sub.F``.
    """
    id: int
    inter_id: int
    main_id1: int
    second_id: int
    main_id2: int = 0
    title: str = ""


@dataclass
class GuidedCable:
    """/INTER/GUIDED_CABLE/cable_ID or /GUIDED_CABLE/cable_ID (M149): Guided cable sliding interface.

    Fortran origin: ``starter/source/tools/seatbelts/hm_read_guided_cable.F90`` / CFG ``inter_guided_cable.cfg``.
    """
    id: int
    grnod_id: int = 0
    grpart_id: int = 0
    istiff: int = 1
    stfac: float = 1.0
    fric: float = 0.0
    title: str = ""

    @property
    def grnod_main(self) -> int:
        return self.grnod_id

    @property
    def grnod_sub(self) -> int:
        return self.grpart_id

    @property
    def frad(self) -> float:
        return self.stfac


@dataclass
class ShellPartPerturbation:
    """/PERTURB/PART/SHELL/perturb_ID (M101): Shell part perturbation.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_perturb_part_shell.F``.
    """
    id: int
    title: str = ""
    grpart_id: int = 0
    chvar: str = "THICK"
    f_mean: float = 0.0
    deviation: float = 0.0
    min_cut: float = 0.0
    max_cut: float = 0.0
    seed: int = 0
    idistri: int = 2


@dataclass
class FailurePerturbation:
    """/PERTURB/FAIL/BIQUAD/perturb_ID (M101): Failure parameter perturbation.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_perturb_fail.F``.
    """
    id: int
    title: str = ""
    fail_id: int = 0
    parameter: str = "C3"
    fail_type: str = "BIQUAD"
    f_mean: float = 0.0
    deviation: float = 0.0
    min_cut: float = 0.0
    max_cut: float = 0.0
    seed: int = 0
    idistri: int = 2


@dataclass
class SphGlobal:
    """/SPHGLO (M101): SPH global computation controls.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_sphglo.F``.
    """
    spasort: float = 0.25
    ale_maxsph: int = 0
    lvoisph: int = 120
    kvoisph: int = 240
    isol2sph: int = 1


@dataclass
class SmsGlobal:
    """/SMS or /AMS (M101): Selective Mass Scaling global parameters.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_sms.F``.
    """
    grpart_id: int = 0
    dt_target: float = 0.0


# ----------------------------------------------------------------------------
# Boundary conditions & joints & special initial states (M102)
# ----------------------------------------------------------------------------

@dataclass
class BcsNrf:
    """/BCS/NRF (M102, M200): Non-reflecting boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_nrf.F90``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0
    set_id: int = 0
    iskep: int = 0
    frame_id: int = 0
    isurf: int = 0
    ivel: int = 0
    isub: int = 0
    ityp: int = 0
    factor: float = 0.0

    def __post_init__(self):
        if not self.grnod_id and self.set_id:
            self.grnod_id = self.set_id
        elif not self.set_id and self.grnod_id:
            self.set_id = self.grnod_id


@dataclass
class BcsWall:
    """/BCS/WALL (M102): Sliding wall boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_wall.F90``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0
    sensor_id: int = 0
    tstart: float = 0.0
    tstop: float = 0.0


@dataclass
class RigidLink:
    """/RLINK (M102/M152): Standard rigid link definition between node group and main/skew frame.

    Fortran origin: ``starter/source/constraints/rigidlink/hm_read_rlink.F``.
    """
    id: int
    title: str = ""
    dofs: Tuple[int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1)
    skew_id: int = 0
    grnod_id: int = 0
    ipol: int = 0

    @property
    def tx(self) -> int:
        return self.dofs[0] if len(self.dofs) > 0 else 1

    @property
    def ty(self) -> int:
        return self.dofs[1] if len(self.dofs) > 1 else 1

    @property
    def tz(self) -> int:
        return self.dofs[2] if len(self.dofs) > 2 else 1

    @property
    def rx(self) -> int:
        return self.dofs[3] if len(self.dofs) > 3 else 1

    @property
    def ry(self) -> int:
        return self.dofs[4] if len(self.dofs) > 4 else 1

    @property
    def rz(self) -> int:
        return self.dofs[5] if len(self.dofs) > 5 else 1


@dataclass
class ExternLink:
    """/EXTERN/LINK or /EXTLNK (M167): External process coupling link.

    Fortran origin: ``starter/source/coupling/rad2rad/lecextlnk.F`` and
    ``hm_cfg_files/config/CFG/radioss2022/RAD2R/extlnk.cfg``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0


@dataclass
class CylJoint:
    """/CYL_JOINT (M102): Cylindrical joint constraint between independent and dependent nodes.

    Fortran origin: ``starter/source/constraints/general/cyl_joint/hm_read_cyljoint.F``.
    """
    id: int
    title: str = ""
    node_id1: int = 0
    node_id2: int = 0
    grnod_id: int = 0


@dataclass
class GeneralJoint:
    """/GJOINT (M102): General kinematic joint (GEAR, RACK, DIFF).

    Fortran origin: ``starter/source/constraints/general/gjoint/hm_read_gjoint.F``.
    """
    id: int
    title: str = ""
    subtype: str = "DEFAULT"  # DEFAULT, GEAR, RACK, DIFF
    node_id0: int = 0
    fscale: float = 1.0
    mass0: float = 0.0
    inertia0: float = 0.0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    mass1: float = 0.0
    inertia1: float = 0.0
    r1: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    mass2: float = 0.0
    inertia2: float = 0.0
    r2: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    mass3: float = 0.0
    inertia3: float = 0.0
    r3: Tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass
class MergeNode:
    """/MERGE/NODE (M102): Merge nodes in node group within tolerance.

    Fortran origin: ``starter/source/constraints/general/merge/hm_read_merge.F``.
    """
    id: int
    title: str = ""
    tol: float = 0.0
    grnod_id: int = 0
    merge_type: int = 0


@dataclass
class MergeRbody:
    """/MERGE/RBODY (M102, M196): Merge rigid bodies.

    Fortran origin: ``starter/source/constraints/general/merge/hm_read_merge.F``.
    """
    id: int
    title: str = ""
    items: List[Tuple[int, int, int, int, int]] = field(default_factory=list)  # (main_id, m_type, secon_id, s_type, iflag)
    rbody_master_id: int = 0
    rbody_slave_ids: List[int] = field(default_factory=list)
    params: dict = field(default_factory=dict)


@dataclass
class IniCrackSegment:
    """Segment definition for /INICRACK."""
    node_id1: int
    node_id2: int
    ratio: float = 0.0


@dataclass
class IniCrack:
    """/INICRACK (M102, M143): Initial crack geometric definition for X-FEM / cohesive elements.

    Fortran origin: ``starter/source/initial_conditions/inicrack/hm_read_inicrack.F``.
    """
    id: int
    title: str = ""
    segments: List[IniCrackSegment] = field(default_factory=list)
    grsh_id: int = 0
    p1: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    p2: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    norm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    open_flag: int = 0


@dataclass
class LaserLoad:
    """/LASER or /DFS/LASER (M102): Laser beam impact load.

    Fortran origin: ``starter/source/loads/laser/leclas.F``.
    """
    id: int
    title: str = ""
    magnitude: float = 0.0
    curve_id: int = 0
    s_target: float = 0.0
    fct_id_target: int = 0
    hn: float = 0.0
    vcp: float = 0.0
    k0: float = 0.0
    rd: float = 0.0
    ks: float = 0.0
    np: int = 0
    nc: int = 0
    plasma_elements: List[int] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Specialized loads, preload, extended damping & solver modes (M103)
# ----------------------------------------------------------------------------

@dataclass
class PcylLoad:
    """/LOAD/PCYL (M103): Cylindrical pressure load.

    Fortran origin: ``starter/source/loads/general/load_pcyl/hm_read_pcyl.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    frame_id: int = 0
    table_id: int = 0
    xscale_r: float = 1.0
    xscale_t: float = 1.0
    yscale_p: float = 1.0


@dataclass
class PfluidLoad:
    """/LOAD/PFLUID (M103): Hydrostatic / fluid surface pressure load.

    Fortran origin: ``starter/source/loads/general/pfluid/hm_read_pfluid.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    fct_id_t: int = 0
    ascalex: float = 1.0
    fscaley: float = 1.0
    dir_p: str = "Z"
    frame_id: int = 0
    fct_id_pc: int = 0
    ascalex_pc: float = 1.0
    fscaley_pc: float = 1.0
    fct_id_vel: int = 0
    ascalex_vel: float = 1.0
    fscaley_vel: float = 1.0
    dir_vel: str = "Z"
    frame_id_vel: int = 0


@dataclass
class Preload:
    """/PRELOAD (M103): Bolt cross-section preload.

    Fortran origin: ``starter/source/loads/general/preload/hm_read_preload.F``.
    """
    id: int
    title: str = ""
    sect_id: int = 0
    sens_id: int = 0
    itype: int = 0
    fct_id: int = 0
    preload: float = 0.0
    tstart: float = 0.0
    tstop: float = 0.0


@dataclass
class PreloadAxial:
    """/PRELOAD/AXIAL (M103): Axial preload on 1D/solid part groups.

    Fortran origin: ``starter/source/loads/general/preload/hm_read_preload_axial.F90``.
    """
    id: int
    title: str = ""
    grpart_id: int = 0
    sens_id: int = 0
    fct_id: int = 0
    preload: float = 0.0
    damp: float = 0.0


@dataclass
class DampInter:
    """/DAMP/INTER (M103/M196): Interface / relative velocity damping.

    Fortran origin: ``starter/source/general_controls/damping/hm_read_damp.F``.
    """
    id: int = 0
    title: str = ""
    nb_time_step: int = 0
    damp_range: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    grnod_id: int = 0
    skew_id: int = 0
    tstart: float = 0.0
    tstop: float = 1.0e30
    alpha_yy: float = 0.0
    beta_yy: float = 0.0
    alpha_zz: float = 0.0
    beta_zz: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def range_val(self) -> int:
        return self.damp_range

    @range_val.setter
    def range_val(self, v: int) -> None:
        self.damp_range = v


@dataclass
class DampRange:
    """/DAMP/RANGE or /DAMP/FREQUENCY_RANGE (M103): Frequency range damping.

    Fortran origin: ``starter/source/general_controls/damping/hm_read_damp.F``.
    """
    id: int
    title: str = ""
    cdamp: float = 0.0
    grpart_id: int = 0
    tstart: float = 0.0
    tstop: float = 0.0
    freq_low: float = 0.0
    freq_high: float = 0.0


@dataclass
class DampGlobal:
    """/DAMP/GLOBAL (M134): Global mass/stiffness Rayleigh damping."""
    id: int = 1
    title: str = ""
    alpha: float = 0.0
    beta: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class DampPart:
    """/DAMP/PART (M134): Per-part Rayleigh damping factor."""
    id: int
    title: str = ""
    part_id: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class AnalyGlobal:
    """/ANALY (M103): Global analysis type options.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_analy.F``.
    """
    n2d3d: int = 0
    analy_temp: int = 0
    iparith: int = 0


@dataclass
class UpwindGlobal:
    """/UPWIND (M103): Upwind advection factors for ALE.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_upwind.F``.
    """
    eta1: float = 0.0
    eta2: float = 0.0
    eta3: float = 0.0


@dataclass
class CaaControl:
    """/CAA (M103): Computational Aeroacoustics control.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_caa.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    grnod_id: int = 0
    sens_id: int = 0


# ----------------------------------------------------------------------------
# Virtual sensors, clusters, flexible bodies & advanced initial states (M104)
# ----------------------------------------------------------------------------

@dataclass
class Gauge:
    """/GAUGE (M104): Numerical strain/stress gauge virtual sensor.

    Fortran origin: ``starter/source/output/gauge/hm_read_gauge.F``.
    """
    id: int
    subtype: str = ""
    title: str = ""
    node_id: int = 0
    elem_id: int = 0
    dist: float = 0.0
    fcut: float = 0.0


@dataclass
class Cluster:
    """/CLUSTER (M104): Element failure/grouping cluster.

    Fortran origin: ``starter/source/output/cluster/hm_read_cluster.F``.
    """
    id: int
    subtype: str = ""
    title: str = ""
    group_id: int = 0
    skew_id: int = 0
    ifail: int = 0
    fn_fail: float = 0.0
    sca_a1: float = 0.0
    sca_b1: float = 0.0
    fs_fail: float = 0.0
    sca_a2: float = 0.0
    sca_b2: float = 0.0
    mt_fail: float = 0.0
    sca_a3: float = 0.0
    sca_b3: float = 0.0
    mb_fail: float = 0.0
    sca_a4: float = 0.0
    sca_b4: float = 0.0


@dataclass
class ExtLink:
    """/EXTLNK (M104): Multi-code external link coupling.

    Fortran origin: ``starter/source/coupling/rad2rad/lecextlnk.F``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0


@dataclass
class FxBody:
    """/FXBODY (M104): Component mode synthesis (CMS) flexible body.

    Fortran origin: ``starter/source/constraints/fxbody/hm_read_fxb.F``.
    """
    id: int
    title: str = ""
    node_id: int = 0
    ianim: int = 0
    imin: int = 0
    imax: int = 0
    filename: str = ""


@dataclass
class IniGrav:
    """/INIGRAV (M104/M151): Initial gravity equilibrium state.

    Fortran origin: ``starter/source/initial_conditions/inigrav/hm_read_inigrav.F``.
    """
    id: int
    title: str = ""
    grpart_id: int = 0
    surf_id: int = 0
    grav_id: int = 0
    pref: float = 0.0
    bx: float = 0.0
    by: float = 0.0
    bz: float = 0.0


InigravLoad = IniGrav


@dataclass
class IniMap1D:
    """/INIMAP1D (M104/M167): 1D mapped field initial condition.

    Fortran origin: ``starter/source/initial_conditions/inimap/hm_read_inimap1d.F``.
    """
    id: int
    title: str = ""
    formulation: str = "FILE"  # VP, VE, FILE
    map_type: int = 0  # 1: Planar, 2: Cylindrical, 3: Spherical
    node_id1: int = 0
    node_id2: int = 0
    grbric_id: int = 0
    grquad_id: int = 0
    grsh3n_id: int = 0
    fscale_v: float = 1.0
    func_vel: int = 0
    fac_vel: float = 1.0
    nb_mat: int = 0
    func_alpha: List[int] = field(default_factory=list)
    func_rho: List[int] = field(default_factory=list)
    func_pres_ener: List[int] = field(default_factory=list)
    fac_rho: List[float] = field(default_factory=list)
    fac_pres_ener: List[float] = field(default_factory=list)
    filename: str = ""


@dataclass
class IniMap2D:
    """/INIMAP2D (M104/M167): 2D mapped field initial condition.

    Fortran origin: ``starter/source/initial_conditions/inimap/hm_read_inimap2d.F``.
    """
    id: int
    title: str = ""
    formulation: str = "FILE"  # VP, VE, FILE
    map_type: int = 0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    grbric_id: int = 0
    grquad_id: int = 0
    grsh3n_id: int = 0
    fscale_v: float = 1.0
    func_vel: int = 0
    fac_vel: float = 1.0
    nb_mat: int = 0
    func_alpha: List[int] = field(default_factory=list)
    func_rho: List[int] = field(default_factory=list)
    func_pres_ener: List[int] = field(default_factory=list)
    fac_rho: List[float] = field(default_factory=list)
    fac_pres_ener: List[float] = field(default_factory=list)
    filename: str = ""


@dataclass
class IniStateFile:
    """/INISTATE or /INISTATE/FILE (M104): External initial state file.

    Fortran origin: ``starter/source/initial_conditions/inista/hm_read_inista.F``.
    """
    filename: str = ""
    isigi: int = 0
    ioutp_fmt: int = 0


# ----------------------------------------------------------------------------
# Extended Control Volumes, Airbag Leakage & ALE Grid Controls (M105)
# ----------------------------------------------------------------------------

@dataclass
class MonvolPres:
    """/MONVOL/PRES (M105): Pressure monitored volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type1.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    fscale: float = 1.0
    p_ext: float = 0.0
    fct_id: int = 0


@dataclass
class MonvolGas:
    """/MONVOL/GAS (M105): Monitored gas control volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type10.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    heat_t0: float = 0.0
    scal_t: float = 1.0
    scal_p: float = 1.0
    scal_s: float = 1.0
    scal_a: float = 1.0
    scal_d: float = 1.0
    gamma: float = 1.4
    mu: float = 0.0
    trelax: float = 0.0
    tini: float = 293.15
    rho_gas: float = 1.2
    pext: float = 0.0
    pini: float = 0.0
    pmax: float = 0.0
    vinc: float = 0.0
    mini: float = 0.0


@dataclass
class MonvolCommu1:
    """/MONVOL/COMMU1 (M105): Communicating multi-chamber airbag volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type2.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    heat_t0: float = 0.0
    scal_t: float = 1.0
    scal_p: float = 1.0
    scal_s: float = 1.0
    scal_a: float = 1.0
    scal_d: float = 1.0
    mat_id: int = 0
    mu: float = 0.0
    pext: float = 0.0
    t_initial: float = 293.15
    iequil: int = 0
    ittf: int = 0


@dataclass
class MonvolLFluid:
    """/MONVOL/LFLUID (M105): Liquid fluid control volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type11.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    scal_t: float = 1.0
    scal_p: float = 1.0
    rho_fluid: float = 1000.0
    fct_k: int = 0
    fct_mtin: int = 0
    fscale_k: float = 1.0
    fscale_mtin: float = 1.0
    fct_mtout: int = 0
    fct_mpout: int = 0
    fscale_mtout: float = 1.0
    fscale_mpout: float = 1.0
    fct_padd: int = 0
    fct_pmax: int = 0
    fscale_padd: float = 1.0
    fscale_pmax: float = 1.0


@dataclass
class MonvolAirbagJet:
    """Injector jet specification for /MONVOL/AIRBAG or /MONVOL/COMMU."""
    gamma: float = 1.4
    cpa: float = 0.0
    cpb: float = 0.0
    cpc: float = 0.0
    fct_id_mass: int = 0
    iflow: int = 0
    fscale_mass: float = 1.0
    fct_id_t: int = 0
    fscale_t: float = 1.0
    sens_id: int = 0
    ijet: int = 0
    n1: int = 0
    n2: int = 0
    n3: int = 0


@dataclass
class MonvolAirbagVent:
    """Vent hole or porous surface specification for /MONVOL/AIRBAG or /MONVOL/COMMU."""
    surf_id_v: int = 0
    avent: float = 0.0
    bvent: float = 0.0
    tstop: float = 0.0
    tvent: float = 0.0
    dpdef: float = 0.0
    dtpdef: float = 0.0
    fct_id_v: int = 0
    fscale_v: float = 1.0


@dataclass
class MonvolAirbag:
    """/MONVOL/AIRBAG or /MONVOL/TYPE4 (M133): Multi-gas airbag control volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type4.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    scal_t: float = 1.0
    scal_p: float = 1.0
    scal_s: float = 1.0
    scal_a: float = 1.0
    scal_d: float = 1.0
    mu: float = 0.0
    pext: float = 0.0
    t0: float = 293.15
    iequi: int = 0
    ittf: int = 0
    gammai: float = 1.4
    cpai: float = 0.0
    cpbi: float = 0.0
    cpci: float = 0.0
    njet: int = 0
    jets: List[MonvolAirbagJet] = field(default_factory=list)
    nvent: int = 0
    vents: List[MonvolAirbagVent] = field(default_factory=list)


@dataclass
class MonvolCommu:
    """/MONVOL/COMMU or /MONVOL/TYPE5 (M133): Multi-chamber communicating volume.

    Fortran origin: ``starter/source/airbag/hm_read_monvol_type5.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    scal_t: float = 1.0
    scal_p: float = 1.0
    scal_s: float = 1.0
    scal_a: float = 1.0
    scal_d: float = 1.0
    mu: float = 0.0
    pext: float = 0.0
    t0: float = 293.15
    iequi: int = 0
    ittf: int = 0
    gammai: float = 1.4
    cpai: float = 0.0
    cpbi: float = 0.0
    cpci: float = 0.0
    njet: int = 0
    jets: List[MonvolAirbagJet] = field(default_factory=list)
    nvent: int = 0
    vents: List[MonvolAirbagVent] = field(default_factory=list)
    comm_ids: List[int] = field(default_factory=list)


@dataclass
class MonvolPart:
    """/MONVOL/PART (M133): Monitored volume defined by part ID / group."""
    id: int
    title: str = ""
    part_id: int = 0
    grpart_id: int = 0
    monvol_type: str = "PRES"
    pini: float = 0.0


@dataclass
class LeakMat:
    """/LEAK/MAT, /LEAK/PART, /LEAK/AREA (M105, M133): Airbag fabric leakage model.

    Fortran origin: ``starter/source/airbag/hm_read_leak.F``.
    """
    id: int
    subtype: str = ""
    title: str = ""
    ileakage: int = 0
    scale_t: float = 1.0
    scale_p: float = 1.0
    acoeft1: float = 0.0
    fct_id_e: int = 0
    fscale_e: float = 1.0
    bcoeft1: float = 0.0
    acoeft2: float = 0.0
    fct_id_lc: int = 0
    fct_id_ac: int = 0
    fscale_lc: float = 1.0
    fscale_ac: float = 1.0
    # Ileakage == 5 micromechanical formulation:
    length: float = 1.0
    thick: float = 1.0
    c1: float = 0.0
    c2: float = 1.0
    c3: float = 0.0


@dataclass
class AleGrid:
    """/ALE/GRID (M105): ALE grid formulation and damping controls.

    Fortran origin: ``starter/source/ale/alelec.F``.
    """
    id: int = 1
    subtype: str = "STANDARD"
    title: str = ""
    dt_min: float = 0.0
    gamma: float = 0.0
    damp: float = 0.0
    nu_g: float = 0.0


@dataclass
class AleLink:
    """/ALE/LINK (M105): ALE grid link velocity condition.

    Fortran origin: ``starter/source/ale/alelec.F``.
    """
    id: int
    subtype: str = "VEL"
    title: str = ""
    grnod_id: int = 0
    fct_id: int = 0
    scale: float = 1.0


@dataclass
class AleSolver:
    """/ALE/SOLVER (M105): Global ALE momentum/interface solver.

    Fortran origin: ``starter/source/ale/alelec.F``.
    """
    imom: int = 0
    isfint: int = 0


@dataclass
class AleClose:
    """/ALE/CLOS or /ALE/CLOSE (M105): ALE mesh closing boundary distance.

    Fortran origin: ``starter/source/ale/hm_read_ale_close.F``.
    """
    htest: float = 0.0
    hclose: float = 0.0


@dataclass
class Retractor:
    """/RETRACTOR (M106): Seatbelt retractor mechanism.

    Fortran origin: ``starter/source/seatbelts/retractor.F`` / CFG ``retractor.cfg``.
    """
    id: int
    title: str = ""
    subtype: str = "SPRING"
    el_id: int = 0
    node_id: int = 0
    elem_size: float = 0.0
    sens_id1: int = 0
    pullout: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    yscale1: float = 1.0
    xscale1: float = 1.0
    sens_id2: int = 0
    tens_typ: int = 0
    force: float = 0.0
    fct_id3: int = 0
    yscale2: float = 1.0
    xscale2: float = 1.0


@dataclass
class Slipring:
    """/SLIPRING (M106): Seatbelt slipring friction element.

    Fortran origin: ``starter/source/seatbelts/slipring.F`` / CFG ``slipring.cfg``, ``slipring_shell.cfg``.
    """
    id: int
    title: str = ""
    subtype: str = "SPRING"  # SPRING or SHELL
    el_id1: int = 0          # or EL_SET1 for SHELL
    el_id2: int = 0          # or EL_SET2 for SHELL
    node_id: int = 0         # or Node_SET for SHELL
    node_id2: int = 0
    sens_id: int = 0
    flow_flag: int = 0
    a: float = 0.0
    ed_factor: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fricd: float = 0.0
    xscale1: float = 1.0
    yscale2: float = 1.0
    xscale2: float = 1.0
    fct_id3: int = 0
    fct_id4: int = 0
    frics: float = 0.0
    xscale3: float = 1.0
    yscale4: float = 1.0
    xscale4: float = 1.0


@dataclass
class Pretensioner:
    """``/PRETENSIONER`` or ``/SEATBELT/PRETENSIONER`` (M197): Seatbelt pretensioner element.

    Fortran origin: ``starter/source/seatbelts/pretensioner.F`` / CFG ``pretensioner.cfg``.
    """
    id: int
    title: str = ""
    sens_id: int = 0
    fct_id: int = 0
    fscale: float = 1.0
    tstart: float = 0.0
    vmax: float = 0.0
    amax: float = 0.0
    reinf: float = 0.0
    i_type: int = 0
    retractor_id: int = 0
    slipring_id: int = 0
    element_ids: List[int] = field(default_factory=list)
    params: dict = field(default_factory=dict)


@dataclass
class UserWindow:
    """/USERWI (M106): User window / data card lines.

    Fortran origin: ``starter/source/starter/userwi.F`` / CFG ``userwi.cfg``.
    """
    lines: List[str] = field(default_factory=list)


@dataclass
class Drape:
    """/DRAPE (M107): Composite fabric draping definition.

    Fortran origin: ``starter/source/properties/drape.F`` / CFG ``drape.cfg``.
    """
    id: int
    title: str = ""
    slices: List[Dict] = field(default_factory=list)


@dataclass
class IniBriEref:
    """/INIBRI/EREF (M107): Initial brick element reference state.

    Fortran origin: ``starter/source/elements/inibri_eref.F`` / CFG ``inibri_eref.cfg``.
    """
    elem_id: int = 0
    ref_elem_id: int = 0
    sub_objects: List[Dict] = field(default_factory=list)


@dataclass
class IncludeDyna:
    """/INCLUDE_DYNA (M107): LS-DYNA include file directive.

    Fortran origin: ``starter/source/starter/includedyna.F`` / CFG ``includedyna.cfg``.
    """
    filename: str = ""


@dataclass
class MonvolFvmBag1:
    """/MONVOL/FVMBAG1 (M108): Finite Volume Method Airbag model.

    Fortran origin: ``starter/source/control_volume/fvmbag1.F`` / CFG ``monvol_fvmbag1.cfg``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    scale_t: float = 1.0
    scale_p: float = 1.0
    scale_s: float = 1.0
    scale_a: float = 1.0
    scale_d: float = 1.0
    mat_id: int = 0
    pext: float = 0.0
    ttot: float = 0.0
    params: Dict = field(default_factory=dict)


@dataclass
class MonvolFvmBag2:
    """/MONVOL/FVMBAG2 (M111): Dual-Chamber Finite Volume Method Airbag model.

    Fortran origin: ``starter/source/control_volume/fvmbag2.F`` / CFG ``monvol_fvmbag2.cfg``.
    """
    id: int
    title: str = ""
    surf_id_ex: int = 0
    surf_id_in: int = 0
    hconv: float = 0.0
    ih3d: int = 0
    mat_id: int = 0
    pext: float = 0.0
    t0: float = 0.0
    i_ttf: int = 0
    params: Dict = field(default_factory=dict)


@dataclass
class Autoposition:
    """/TRANSFORM/AUTOPOSITION (M111): Automated nodal repositioning.

    Fortran origin: ``starter/source/model/transformation/lectrans.F`` / CFG ``autoposition.cfg``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0
    surf_id: int = 0
    skew_id: int = 0
    dir: str = ""
    gap: float = 0.0
    pflag: int = 0
    xpos: float = 0.0
    ypos: float = 0.0
    zpos: float = 0.0
    xflag: int = 0
    yflag: int = 0
    zflag: int = 0


@dataclass
class LoadCentri:
    """/LOAD/CENTRI (M112): Centrifugal body force loading.

    Fortran origin: ``starter/source/loads/general/load_centri/hm_read_load_centri.F`` / CFG ``centri.cfg``.
    """
    id: int
    title: str = ""
    fct_id: int = 0
    dir: str = ""
    frame_id: int = 0
    sens_id: int = 0
    grnod_id: int = 0
    ivar: int = 0
    ascalex: float = 1.0
    fscaley: float = 1.0


@dataclass
class LoadPfluid:
    """/LOAD/PFLUID (M112): Hydrostatic/fluid pressure on surfaces.

    Fortran origin: ``starter/source/loads/general/pfluid/hm_read_pfluid.F`` / CFG ``pfluid.cfg``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    fct_hsp: int = 0
    ascalex_hsp: float = 1.0
    fscaley_hsp: float = 1.0
    dir_hsp: str = "Z"
    frame_hsp: int = 0
    fct_pc: int = 0
    ascalex_pc: float = 1.0
    fscaley_pc: float = 1.0
    fct_vel: int = 0
    ascalex_vel: float = 1.0
    fscaley_vel: float = 1.0
    dir_vel: str = ""
    frame_vel: int = 0


@dataclass
class LoadPressure:
    """/LOAD/PRESSURE & /LOAD/PFLUID (M112/M163): Hydroforming / directional pressure load.

    Fortran origin: ``starter/source/loads/general/load_pressure/hm_read_load_pressure.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    iload: int = 1
    sens_id: int = 0
    inorm: int = 1
    direction: str = ""
    skew_id: int = 0
    fct_id: int = 0
    xscale_p: float = 1.0
    yscale_p: float = 1.0
    inter_ids: List[int] = field(default_factory=list)
    gap_shifts: List[float] = field(default_factory=list)
    # Legacy fields
    scale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class LoadGravity:
    """/LOAD/GRAV or /LOAD/GRAVITY (M135): Gravitational field acceleration loading."""
    id: int
    title: str = ""
    grnod_id: int = 0
    dir_vector: tuple[float, float, float] = (0.0, 0.0, -1.0)
    funct_id: int = 0
    scale: float = 1.0
    sens_id: int = 0


@dataclass
class LoadBody:
    """/LOAD/BODY (M135): Volumetric body force loading."""
    id: int
    title: str = ""
    grpart_id: int = 0
    dir_vector: tuple[float, float, float] = (0.0, 0.0, -1.0)
    funct_id: int = 0
    scale: float = 1.0
    sens_id: int = 0


@dataclass
class LoadTherm:
    """/LOAD/HEAT or /LOAD/THERM (M135): Thermal heat flux loading."""
    id: int
    title: str = ""
    group_id: int = 0
    flux: float = 0.0
    funct_id: int = 0
    scale: float = 1.0
    sens_id: int = 0


@dataclass
class InivelAxis:
    """/INIVEL/AXIS (M112): Axisymmetric initial velocity around frame axis.

    Fortran origin: ``starter/source/initial_conditions/general/inivel/hm_read_inivel.F`` / CFG ``inivel_axis.cfg``.
    """
    id: int
    title: str = ""
    dir: str = "Z"
    frame_id: int = 0
    grnod_id: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    vr: float = 0.0
    tstart: float = 0.0
    sens_id: int = 0


@dataclass
class InivelFvm:
    """/INIVEL/FVM (M112): FVM airbag initial velocity on brick/quad/tria groups.

    Fortran origin: ``starter/source/initial_conditions/general/inivel/hm_read_inivel.F`` / CFG ``inivel_fvm.cfg``.
    """
    id: int
    title: str = ""
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    grbric_id: int = 0
    grquad_id: int = 0
    grsh3n_id: int = 0
    skew_id: int = 0
    tstart: float = 0.0
    sens_id: int = 0


@dataclass
class InivelNodeItem:
    """Single node entry for /INIVEL/NODE (M112)."""
    node_id: int
    skew_id: int = 0
    vxt: float = 0.0
    vyt: float = 0.0
    vzt: float = 0.0
    vxr: float = 0.0
    vyr: float = 0.0
    vzr: float = 0.0


@dataclass
class InivelNode:
    """/INIVEL/NODE (M112): Nodal vector initial velocities.

    Fortran origin: ``starter/source/initial_conditions/general/inivel/hm_read_inivel.F`` / CFG ``inivel_node.cfg``.
    """
    id: int
    title: str = ""
    items: List[InivelNodeItem] = field(default_factory=list)


@dataclass
class ImpdispFgeo:
    """/IMPDISP/FGEO (M112): Imposed final geometry displacement.

    Fortran origin: ``starter/source/constraints/general/impvel/read_impdisp_fgeo.F`` / CFG ``impdisp_fgeo.cfg``.
    """
    id: int
    title: str = ""
    fct_id: int = 0
    part_id: int = 0
    sens_id: int = 0
    ascale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    nodes: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class ImpvelFgeo:
    """/IMPVEL/FGEO (M112): Imposed final geometry velocity.

    Fortran origin: ``starter/source/constraints/general/impvel/read_impvel_fgeo.F`` / CFG ``impvel_fgeo.cfg``.
    """
    id: int
    title: str = ""
    fct_id: int = 0
    part_id: int = 0
    fct_l_id: int = 0
    sens_id: int = 0
    ascale: float = 1.0
    t0: float = 0.0
    tstart: float = 0.0
    fscale_l: float = 1.0
    dmin: float = 0.0
    pairs: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class RwallTherm:
    """/RWALL/THERM (M112): Thermal rigid wall.

    Fortran origin: ``starter/source/constraints/general/rwall/hm_read_rwall_therm.F``.
    """
    id: int
    title: str = ""
    typ: int = 1
    tied: int = 0
    node_id: int = 0
    grnod_id1: int = 0
    grnod_id2: int = 0
    fct_id: int = 0
    temp: float = 0.0
    tstif: float = 0.0
    fric: float = 0.0


@dataclass
class SphInOut:
    """/SPH/INOUT or /SPH/IO (M112/M202): SPH particle inlet/outlet boundary condition.

    Fortran origin: ``starter/source/loads/sph/hm_read_sphio.F``.
    """
    id: int
    title: str = ""
    ityp: int = 1  # 1: Inlet, 2: Outlet, 3: NRF, 4: Control section
    surf_id: int = 0
    part_id: int = 0
    pid: int = 0
    dist: float = 0.0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    fcut: float = 0.0
    coords: list[tuple[float, float, float]] = field(default_factory=list)
    # Inlet fields (ityp=1)
    fct_id_r: int = 0
    fscale_r: float = 1.0
    fct_id_e: int = 0
    fscale_e: float = 1.0
    fct_id_vn: int = 0
    # Outlet fields (ityp=2) & NRF (ityp=3)
    fct_id_p: int = 0
    fscale_p: float = 1.0
    lc: float = 0.0
    # Legacy fields
    fct_id: int = 0
    rho_in: float = 0.0
    p_in: float = 0.0
    e_in: float = 0.0


@dataclass
class SphBcs:
    """/SPHBCS (M113): SPH symmetry boundary condition.

    Fortran origin: ``starter/source/loads/sph/hm_read_sphbcs.F``.
    """
    id: int
    bcs_type: str = "SYM"  # 'SYM', 'CYCL', 'PERIOD'
    title: str = ""
    dir: str = "X"
    frame_id: int = 0
    grnod_id: int = 0
    ilevel: int = 0


@dataclass
class EulerBcs:
    """/EULER/BCS (M135): Eulerian domain boundary condition."""
    id: int
    title: str = ""
    grnod_id: int = 0
    bcs_type: str = "INFLOW"
    val1: float = 0.0
    val2: float = 0.0
    val3: float = 0.0


@dataclass
class HeatBcs:
    """/HEAT/BCS (M135): Thermal boundary condition."""
    id: int
    title: str = ""
    group_id: int = 0
    bcs_type: str = "TEMP"
    tval: float = 0.0
    funct_id: int = 0
    scale: float = 1.0
    sens_id: int = 0


@dataclass
class MadymoLink:
    """/MADYMO/LINK (M113): Madymo coupling link.

    Fortran origin: ``starter/source/madymo/hm_read_madymo_link.F``.
    """
    id: int
    title: str = ""
    mdref: int = 0
    node_id: int = 0


@dataclass
class MadymoExfem:
    """/MADYMO/EXFEM (M113): Madymo sub-model part exchange.

    Fortran origin: ``starter/source/madymo/hm_read_madymo_exfem.F``.
    """
    id: int
    title: str = ""
    part_ids: List[int] = field(default_factory=list)


@dataclass
class AleGridDonea:
    """/ALE/GRID/DONEA (M113): Donea ALE grid solver."""
    alpha: float = 0.0
    gamma: float = 100.0
    vel_x: float = 1.0
    vel_y: float = 1.0
    vel_z: float = 1.0
    v_min: float = -1e30


@dataclass
class AleGridSpring:
    """/ALE/GRID/SPRING (M113): Spring analogy ALE grid solver."""
    dt: float = 0.0
    gamma: float = 0.0
    damp: float = 0.5
    nu: float = 1.0
    v_min: float = -1e30


@dataclass
class AleGridStandard:
    """/ALE/GRID/STANDARD (M113): Standard ALE grid solver."""
    alpha: float = 0.0
    gamma: float = 0.0
    damp: float = 0.5
    l_c: float = 1.0


@dataclass
class AleGridDisp:
    """/ALE/GRID/DISP (M113): Displacement-based ALE grid solver."""
    u_max: float = -1e30
    v_min: float = -1e30


@dataclass
class AleGridLaplacian:
    """/ALE/GRID/LAPLACIAN (M113): Laplacian smoothing ALE grid solver."""
    alpha: float = 0.0
    gamma: float = 0.0
    damp: float = 0.5


@dataclass
class AleGridVolume:
    """/ALE/GRID/VOLUME (M113): Volume-preserving ALE grid solver."""
    alpha: float = 0.0
    gamma: float = 0.0


@dataclass
class AdmeshGlobal:
    """/ADMESH/GLOBAL, /ADGLOB, /ADGLOB/MESH (M113): Adaptive meshing global parameters."""
    level_max: int = 0
    iadm_rule: int = 0
    t_delay: float = 0.0
    istat_cnd: int = 0


@dataclass
class StampingInit:
    """/STAMPING (M113): Sheet metal forming stamping history input."""
    time_scale: float = 1.0
    data_lines: List[str] = field(default_factory=list)


@dataclass
class RandomNoise:
    """/RANDOM, /RANDOM/GRNOD (M113): Random vibration and stochastic input noise."""
    grnod_id: int = 0
    xalea: float = 0.0
    seed: float = 0.0


@dataclass
class Accelerometer:
    """/ACCEL (M113): Accelerometer measurement sensor."""
    id: int
    title: str = ""
    node_id: int = 0
    skew_id: int = 0
    cutoff: float = 0.0


@dataclass
class Subset:
    """/SUBSET (M113): Hierarchical model component subset."""
    id: int
    title: str = ""
    assembly_ids: List[int] = field(default_factory=list)


@dataclass
class FailComposite:
    """/FAIL/COMPOSITE (M114): 3D anisotropic composite failure model.

    Fortran origin: ``starter/source/materials/failure/fail_composite.F``.
    """
    mat_id: int
    sig_1t: float = 0.0
    sig_1c: float = 0.0
    sig_2t: float = 0.0
    sig_2c: float = 0.0
    sig_12: float = 0.0
    sig_3t: float = 0.0
    sig_3c: float = 0.0
    sig_23: float = 0.0
    sig_31: float = 0.0
    beta: float = 0.0
    tau_max: float = 0.0
    expn: float = 0.0
    ifail_sh: int = 0
    ifail_so: int = 0
    fail_id: int = 0


@dataclass
class EbcsPropellant:
    """/EBCS/PROPELLANT or /BCS/PROPELLANT (M114, M200): Solid propellant combustion boundary condition.

    Fortran origin: ``starter/source/loads/ebcs/hm_read_ebcs_propellant.F90``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    sens_id: int = 0
    sensor_id: int = 0
    submat_id: int = 1
    ienthalpy: int = 1
    rho0s: float = 0.0
    tburn: float = 300.0
    param_t: float = 300.0
    param_a: float = 0.0
    param_n: float = 0.0
    f_func_id: int = 0
    ffunc_id: int = 0
    f_scale_x: float = 1.0
    fscale_x: float = 1.0
    f_scale_y: float = 1.0
    fscale_y: float = 1.0
    g_func_id: int = 0
    gfunc_id: int = 0
    g_scale_x: float = 1.0
    gscale_x: float = 1.0
    g_scale_y: float = 1.0
    gscale_y: float = 1.0
    h_func_id: int = 0
    hfunc_id: int = 0
    h_scale_x: float = 1.0
    hscale_x: float = 1.0
    h_scale_y: float = 1.0
    hscale_y: float = 1.0

    def __post_init__(self):
        if not self.sensor_id and self.sens_id:
            self.sensor_id = self.sens_id
        elif not self.sens_id and self.sensor_id:
            self.sens_id = self.sensor_id
        if self.tburn != 300.0 and self.param_t == 300.0:
            self.param_t = self.tburn
        elif self.param_t != 300.0 and self.tburn == 300.0:
            self.tburn = self.param_t
        if not self.ffunc_id and self.f_func_id:
            self.ffunc_id = self.f_func_id
        elif not self.f_func_id and self.ffunc_id:
            self.f_func_id = self.ffunc_id
        if not self.gfunc_id and self.g_func_id:
            self.gfunc_id = self.g_func_id
        elif not self.g_func_id and self.gfunc_id:
            self.g_func_id = self.gfunc_id
        if not self.hfunc_id and self.h_func_id:
            self.hfunc_id = self.h_func_id
        elif not self.h_func_id and self.hfunc_id:
            self.h_func_id = self.hfunc_id
        if self.f_scale_x != 1.0 and self.fscale_x == 1.0:
            self.fscale_x = self.f_scale_x
        elif self.fscale_x != 1.0 and self.f_scale_x == 1.0:
            self.f_scale_x = self.fscale_x
        if self.f_scale_y != 1.0 and self.fscale_y == 1.0:
            self.fscale_y = self.f_scale_y
        elif self.fscale_y != 1.0 and self.f_scale_y == 1.0:
            self.f_scale_y = self.fscale_y
        if self.g_scale_x != 1.0 and self.gscale_x == 1.0:
            self.gscale_x = self.g_scale_x
        elif self.gscale_x != 1.0 and self.g_scale_x == 1.0:
            self.g_scale_x = self.gscale_x
        if self.g_scale_y != 1.0 and self.gscale_y == 1.0:
            self.gscale_y = self.g_scale_y
        elif self.gscale_y != 1.0 and self.g_scale_y == 1.0:
            self.g_scale_y = self.gscale_y
        if self.h_scale_x != 1.0 and self.hscale_x == 1.0:
            self.hscale_x = self.h_scale_x
        elif self.hscale_x != 1.0 and self.h_scale_x == 1.0:
            self.h_scale_x = self.hscale_x
        if self.h_scale_y != 1.0 and self.hscale_y == 1.0:
            self.hscale_y = self.h_scale_y
        elif self.hscale_y != 1.0 and self.h_scale_y == 1.0:
            self.h_scale_y = self.hscale_y



@dataclass
class AdmasNonUniformItem:
    """Item for /ADMAS/NON_UNIFORM (M114)."""
    mass: float = 0.0
    entity_id: int = 0
    iflag: int = 0


@dataclass
class AdmasNonUniform:
    """/ADMAS/NON_UNIFORM or /ADMAS/NON_UNIFORM_PART (M114): Non-uniform added mass list."""
    id: int
    kind: str = "NODE"  # 'NODE' | 'PART'
    items: List[AdmasNonUniformItem] = field(default_factory=list)


@dataclass
class SectCircle:
    """/SECT/CIRCLE (M114): Circular cross-section cut.

    Fortran origin: ``starter/source/tools/sect/hm_read_sect_circle.F``.
    """
    id: int
    title: str = ""
    n1: int = 0
    n2: int = 0
    n3: int = 0
    isave: int = 0
    delta_t: float = 0.0
    alpha: float = 0.0
    file_name: str = ""
    grbric_id: int = 0
    grshel_id: int = 0
    grtrus_id: int = 0
    grbeam_id: int = 0
    grsprg_id: int = 0
    grtria_id: int = 0
    int_ids: List[int] = field(default_factory=list)
    iframe: int = 0
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.0


@dataclass
class SectParal:
    """/SECT/PARAL (M114): Parallelogram cross-section cut.

    Fortran origin: ``starter/source/tools/sect/hm_read_sect_paral.F``.
    """
    id: int
    title: str = ""
    n1: int = 0
    n2: int = 0
    n3: int = 0
    isave: int = 0
    delta_t: float = 0.0
    alpha: float = 0.0
    file_name: str = ""
    grbric_id: int = 0
    grshel_id: int = 0
    grtrus_id: int = 0
    grbeam_id: int = 0
    grsprg_id: int = 0
    grtria_id: int = 0
    int_ids: List[int] = field(default_factory=list)
    iframe: int = 0
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    corner1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    corner2: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class DynainShell:
    """/DYNAIN/SHELL (M114): LS-DYNA shell history initialization."""
    option: str = "AUX/FULL"  # 'AUX/FULL' | 'STRES/FULL' | 'STRAIN/FULL'


@dataclass
class MonvolArea:
    """/MONVOL/AREA (M115): Monitored volume surface area monitoring."""
    id: int
    title: str = ""
    surf_id_ext: int = 0
    scale_t: float = 1.0
    scale_p: float = 1.0
    scale_s: float = 1.0
    scale_a: float = 1.0
    scale_d: float = 1.0

    @property
    def surf_id(self) -> int:
        return self.surf_id_ext


@dataclass
class StateDt:
    """/STATE/DT or /DYNAIN/DT (M115): State output time-step controls."""
    tstart: float = 0.0
    tfreq: float = 0.0
    is_all: bool = False
    component_ids: List[int] = field(default_factory=list)


@dataclass
class SphReserve:
    """/SPH/RESERVE (M116): SPH reserve particle buffer allocation."""
    part_id: int
    np_particles: int = 0


@dataclass
class MoveFunct:
    """/MOVE_FUNCT (M116): Function curve scale and shift transformation."""
    id: int
    title: str = ""
    a_scale_x: float = 1.0
    f_scale_y: float = 1.0
    a_shift_x: float = 0.0
    f_shift_y: float = 0.0


@dataclass
class EigenMode:
    """/EIG (M117): Eigenvalue extraction & modal analysis configuration."""
    id: int
    title: str = ""
    grnod_id: int = 0
    grnod_bc: int = 0
    trarot: str = ""
    ifile: int = 0
    imls: int = 0
    nmod: int = 0
    inorm: int = 0
    cutfreq: float = 0.0
    freqmin: float = 0.0
    nbloc: int = 0
    incv: int = 0
    niter: int = 0
    ipri: int = 0
    tol: float = 0.0
    filename: str = ""


@dataclass
class StressFile:
    """/STATE/STR_FILE or /STR_FILE (M117): Stress output file specification."""
    izip: int = 0
    filename: str = ""


@dataclass
class MemoryRequest:
    """/MEMORY (M117): Memory allocation request."""
    nmots: int = 0
    rate: float = 0.66


@dataclass
class FailFractal:
    """/FAIL/FRACTAL_DMG or /FAIL/FRACTAL (M118): Fractal damage failure model."""
    mat_id: int
    grsh4n_1: int = 0
    grsh3n_1: int = 0
    grsh4n_2: int = 0
    grsh3n_2: int = 0
    damage: float = 0.0
    probability: float = 0.0
    seed: int = 0
    num_walk: int = 0
    printout: int = 0
    fail_id: int = 0


@dataclass
class TransformPosition:
    """/TRANSFORM/POSITION or /TRANSFORM/POS (M118): Spatial positioning transform."""
    id: int
    title: str = ""
    grnod_id: int = 0
    node_ids: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    submodel: int = 0
    points: tuple[tuple[float, float, float], ...] = ()


@dataclass
class TransformProjection:
    """/TRANSFORM/PROJ (M134): Node group projection transformation."""
    id: int
    title: str = ""
    grnod_id: int = 0
    proj_type: str = "PLANE"
    target_id: int = 0
    dir_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)
    dist: float = 0.0


@dataclass
class TransformFrame:
    """/TRANSFORM/FRAME (M134): Coordinate frame transformation."""
    id: int
    title: str = ""
    grnod_id: int = 0
    frame_orig: int = 0
    frame_dest: int = 0


@dataclass
class ExternalLink:
    """/EXTERN/LINK or /EXTLINK (M118): External interface link."""
    id: int
    title: str = ""
    grnod_id: int = 0


@dataclass
class ArchSpec:
    """/ARCH (M118): Architecture specification card."""
    mach: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0)


@dataclass
class FunctPython:
    """/FUNCT_PYTHON/id (M119): Python mathematical function definition."""
    id: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FrictionPartPair:
    """Connected part pair friction specification for /FRICTION."""
    grpart_id1: int = 0
    grpart_id2: int = 0
    part_id1: int = 0
    part_id2: int = 0
    idir: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    fric: float = 0.0
    vis_f: float = 1.0
    c1_dir2: float = 0.0
    c2_dir2: float = 0.0
    c3_dir2: float = 0.0
    c4_dir2: float = 0.0
    c5_dir2: float = 0.0
    c6_dir2: float = 0.0
    fric_dir2: float = 0.0
    vis_f_dir2: float = 1.0


@dataclass
class FrictionModel:
    """/FRICTION/fric_id (M119): Generalized multi-part and orthotropic friction model."""
    id: int
    title: str = ""
    ifric: int = 0
    ifiltr: int = 0
    xfreq: float = 0.0
    iform: int = 1
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    fric: float = 0.0
    vis_f: float = 1.0
    pairs: list[FrictionPartPair] = field(default_factory=list)


@dataclass
class RefstaNode:
    """/REFSTA (M119): Reference state node coordinate."""
    node_id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class ErefSpec:
    """/EREF (M119): Element reference configuration."""
    id: int
    title: str = ""
    part_id: int = 0
    subtype: str = ""
    elem_coords: list[tuple[int, tuple[tuple[float, float, float], ...]]] = field(default_factory=list)


@dataclass
class NbcsNode:
    """Single node DOF constraint entry in /NBCS."""
    tx: int = 0
    ty: int = 0
    tz: int = 0
    wx: int = 0
    wy: int = 0
    wz: int = 0
    skew_id: int = 0
    node_id: int = 0


@dataclass
class NbcsBlock:
    """/NBCS/id (M119): Non-linear boundary conditions block."""
    id: int
    title: str = ""
    nodes: list[NbcsNode] = field(default_factory=list)


@dataclass
class AleMuscl:
    """/ALE/MUSCL (M119): MUSCL advection compression factor."""
    beta: float = 2.0


@dataclass
class BemModel:
    """/BEM (M119): Boundary element method container."""
    id: int = 1
    title: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GaugePoint:
    """/GAUGE/POINT (M121): Point gauge definition for spatial measurement."""
    id: int
    title: str = ""
    subtitle: str = ""
    points: list[tuple[float, float, float, float, str]] = field(default_factory=list) # (x, y, z, dist, subtitle)


@dataclass
class SphGlo:
    """/SPHGLO (M121): Global SPH particle formulation settings."""
    alpha_sort: float = 0.25
    maxsph: int = 0
    lneigh: int = 120
    nneigh: int = 120
    isol2sph: int = 0


@dataclass
class AnalyOptions:
    """/ANALY (M121): Global analysis dimension and arithmetic options."""
    n2d3d: int = 0         # 0=3D, 1=axisymmetric, 2=plane strain
    iparith: int = 1       # 1=ON, 2=OFF
    isubcyc: int = 0       # 0=none, 2=subcycling n2


@dataclass
class AleCfdSph:
    """/ALECFDSPH (M122): Coupled ALE / CFD / SPH fluid-structure interaction parameters."""
    title: str = ""
    icfd: int = 0
    isph: int = 0
    tstart: float = 0.0
    tstop: float = 1e30
    fscale_c: float = 1.0
    fscale_s: float = 1.0


@dataclass
class FailOrthBiquad:
    """/FAIL/ORTHBIQUAD (M123): Orthotropic Biquadratic failure model for shells."""
    id: int
    mat_id: int = 0
    p_thickfail: float = 1.0
    m_flag: int = 0
    s_flag: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    inst_start: float = 0.0
    eps_dot0: float = 0.0
    c_jc: float = 0.0
    fct_id_rate: int = 0
    fct_id_el: int = 0
    ei_ref: float = 0.0
    r1: float = 1.0
    r2: float = 1.0
    r4: float = 1.0
    r5: float = 1.0
    title: str = ""


@dataclass
class SlipringShell:
    """/SLIPRING/SHELL (M123): Shell-to-shell seatbelt slipring connector."""
    id: int
    el_set1: int = 0
    el_set2: int = 0
    node_set: int = 0
    sens_id: int = 0
    flow_flag: int = 0
    a: float = 0.0
    ed_factor: float = 0.0
    fric_d: float = 0.0
    fric_s: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fct_id3: int = 0
    fct_id4: int = 0
    xscale1: float = 1.0
    xscale2: float = 1.0
    yscale2: float = 1.0
    xscale3: float = 1.0
    xscale4: float = 1.0
    yscale4: float = 1.0
    title: str = ""


@dataclass
class EbcsNrf:
    """/EBCS/NRF or /EBCS/NON_REFLECT (M125): Non-reflecting frontier boundary condition.

    Fortran origin: ``starter/source/loads/ebcs/hm_read_ebcs_nrf.F`` / CFG ``ebcs_nrf.cfg``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    tcar_p: float = 0.0
    tcar_vf: float = 0.0


@dataclass
class EbcsPeriodic:
    """/EBCS/PERIODIC (M138): Eulerian periodic boundary condition.

    Fortran origin: ``starter/source/loads/ebcs/hm_read_ebcs_perio.F``.
    """
    id: int
    title: str = ""
    surf1_id: int = 0
    surf2_id: int = 0
    skew_id: int = 0
    grpart_id: int = 0


@dataclass
class EbcsCyclic:
    """/EBCS/CYCLIC (M138, M200): Eulerian cyclic boundary condition.

    Fortran origin: ``starter/source/loads/ebcs/hm_read_ebcs_cyclic.F90``.
    """
    id: int
    title: str = ""
    surf1_id: int = 0
    surf2_id: int = 0
    skew_id: int = 0
    grpart_id: int = 0
    surf_id1: int = 0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    surf_id2: int = 0
    node_id4: int = 0
    node_id5: int = 0
    node_id6: int = 0

    def __post_init__(self):
        if not self.surf_id1 and self.surf1_id:
            self.surf_id1 = self.surf1_id
        elif not self.surf1_id and self.surf_id1:
            self.surf1_id = self.surf_id1
        if not self.surf_id2 and self.surf2_id:
            self.surf_id2 = self.surf2_id
        elif not self.surf2_id and self.surf_id2:
            self.surf2_id = self.surf_id2



@dataclass
class FailRtcl:
    """/FAIL/RTCL (M125): RTCL ductile failure model.

    Fortran origin: ``starter/source/materials/fail/fail_rtcl.F`` / CFG ``fail_rtcl.cfg``.
    """
    mat_id: int
    epscal: float = 0.0
    inst: int = 0
    n: float = 0.0
    fail_id: int = 0


@dataclass
class FailGurson:
    """/FAIL/GURSON (M125): Gurson-Tvergaard-Needleman porous plasticity failure model.

    Fortran origin: ``starter/source/materials/fail/fail_gurson.F`` / CFG ``fail_gurson.cfg``.
    """
    mat_id: int
    q1: float = 0.0
    q2: float = 0.0
    iloc: int = 1
    eps_n: float = 0.0
    a_s: float = 0.0
    k_w: float = 0.0
    f_c: float = 0.0
    f_r: float = 0.0
    f_0: float = 0.0
    r_len: float = 0.0
    h_chi: float = 0.0
    le_max: float = 0.0
    fail_id: int = 0


@dataclass
class FailPuck:
    """/FAIL/PUCK (M126/M189): Puck composite failure model.

    Fortran origin: ``starter/source/materials/fail/fail_puck.F`` / CFG ``fail_puck.cfg``.
    """
    id: int = 0
    mat_id: int = 0
    sigma_1t: float = 0.0
    sigma_2t: float = 0.0
    sigma_12: float = 0.0
    sigma_1c: float = 0.0
    sigma_2c: float = 0.0
    p12_pos: float = 0.0
    p12_neg: float = 0.0
    p22_neg: float = 0.0
    tau_max: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 1
    fcut: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailSahraei:
    """/FAIL/SAHRAEI (M126): Sahraei failure model.

    Fortran origin: ``starter/source/materials/fail/fail_sahraei.F`` / CFG ``fail_sahraei.cfg``.
    """
    mat_id: int
    fct_ratio: int = 0
    num: int = 0
    den: int = 0
    ordi: int = 0
    vol_strain: float = 0.0
    fct_elsize: int = 0
    el_ref: float = 0.0
    comp_dir: int = 0
    idel: int = 0
    max_comp_strain: float = 0.0
    ratio: float = 0.0
    fail_id: int = 0


@dataclass
class FailSyazwan:
    """/FAIL/SYAZWAN (M126): Syazwan fracture and damage failure model.

    Fortran origin: ``starter/source/materials/fail/fail_syazwan.F`` / CFG ``fail_syazwan.cfg``.
    """
    mat_id: int
    icard: int = 0
    epfmin: float = 0.0
    coeffs: List[float] = field(default_factory=list)
    fail_id: int = 0


@dataclass
class FailTab2:
    """/FAIL/TAB2 (M126): Tabulated failure model Version 2.

    Fortran origin: ``starter/source/materials/fail/fail_tab2.F`` / CFG ``fail_tab2.cfg``.
    """
    mat_id: int
    epsf_id: int = 0
    fcrit: float = 0.0
    failip: int = 0
    pthk: float = 0.0
    n: float = 0.0
    dcrit: float = 0.0
    inst_id: int = 0
    ecrit: float = 0.0
    fct_exp: int = 0
    exp_ref: float = 0.0
    exp: float = 0.0
    fail_id: int = 0


@dataclass
class FailGene1:
    """/FAIL/GENE1 (M126/M193): General multi-criteria failure model.

    Fortran origin: ``starter/source/materials/fail/fail_gene1.F`` / CFG ``fail_gene1.cfg``.
    """
    mat_id: int = 0
    pmin: float = 0.0
    pmax: float = 0.0
    sigp1_max: float = 0.0
    tmax: float = 0.0
    time_max: float = 0.0
    dtmin: float = 0.0
    fct_idsm: int = 0
    eps_dot_sm: float = 0.0
    sig_max: float = 0.0
    sigr: float = 0.0
    kf: float = 0.0
    k: float = 0.0
    fct_idps: int = 0
    eps_dot_ps: float = 0.0
    eps_max: float = 0.0
    eps_eff: float = 0.0
    eps_vol: float = 0.0
    eps_min: float = 0.0
    eps_sh: float = 0.0
    fct_idg12: int = 0
    fct_idg13: int = 0
    fct_ide1c: int = 0
    tab_idfld: int = 0
    itab: int = 0
    eps_dot_fld: float = 0.0
    nstep: int = 0
    ismooth: int = 0
    istrain: int = 0
    thinning: float = 0.0
    volfrac: float = 0.0
    pthk: float = 0.0
    ncs: int = 0
    temp_max: float = 0.0
    failip: int = 0
    fct_idel: int = 0
    fscale_el: float = 1.0
    el_ref: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class StackPly:
    """Layer definition inside a /STACK (M127)."""
    ply_id: int
    phi: float = 0.0
    zi: float = 0.0
    p_thick_fail: float = 0.0
    f_weight: float = 1.0


@dataclass
class Stack:
    """/STACK/stack_id (M127): Composite laminate stack definition.

    Fortran origin: ``starter/source/model/laminate/leclamply.F`` / CFG ``stack.cfg``.
    """
    id: int
    title: str = ""
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    z0: float = 0.0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    istrain: int = 0
    ashear: float = 0.833333
    iint: int = 0
    ithick: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0
    ip: int = 0
    plies: List[StackPly] = field(default_factory=list)


@dataclass
class InivelPart:
    """/INIVEL/PART (M137): Initial velocity on a part."""
    id: int
    title: str = ""
    part_id: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    vr: float = 0.0
    skew_id: int = 0
    tstart: float = 0.0
    sens_id: int = 0


@dataclass
class InivelSph:
    """/INIVEL/SPH (M137): Initial velocity on SPH particles."""
    id: int
    title: str = ""
    grsph_id: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0


@dataclass
class DetLine:
    """/DFS/DETLINE (M137): Detonation along a line segment."""
    id: int
    title: str = ""
    p1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    p2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    t0: float = 0.0
    dvel: float = 0.0
    node1: int = 0
    node2: int = 0
    mat_id: int = 0


@dataclass
class DetCirc:
    """/DFS/DETCIRC (M137): Detonation along a circle."""
    id: int
    title: str = ""
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius: float = 0.0
    t0: float = 0.0
    dvel: float = 0.0


@dataclass
class IniMap3D:
    """/INIMAP/3D (M137): 3D solution mapping descriptor."""
    id: int
    title: str = ""
    map_type: int = 0
    grbric_id: int = 0
    grquad_id: int = 0
    grsh3n_id: int = 0
    filename: str = ""
    fscale_v: float = 1.0


@dataclass
class SetGeneric:
    """/SET (M137): Generic ID collection set."""
    id: int
    set_type: str
    title: str = ""
    ids: List[int] = field(default_factory=list)


@dataclass
class MaterialPlasZeril:
    """/MAT/PLAS_ZERIL (M141): Zerilli-Armstrong plasticity modifier.

    Fortran origin: ``starter/source/materials/mat/matl2_plas_zeril.F``.
    """
    mat_id: int
    title: str = ""
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    n: float = 0.0
    fcut: float = 0.0


@dataclass
class MaterialPlasBodne:
    """/MAT/PLAS_BODNE (M141): Bodner-Partom viscoplasticity modifier.

    Fortran origin: ``starter/source/materials/mat/matl2_plas_bodne.F``.
    """
    mat_id: int
    title: str = ""
    z0: float = 0.0
    z1: float = 0.0
    m: float = 0.0
    n: float = 0.0
    d0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0


@dataclass
class MaterialViscProny:
    """/MAT/VISC_PRONY or /VISC/LPRONY (M141): Viscoelastic Prony relaxation series.

    Fortran origin: ``starter/source/materials/mat/mat_VISC_LPRONY.F``.
    """
    mat_id: int
    title: str = ""
    order: int = 0
    form: int = 0
    flag_visc: int = 0
    gammas: List[float] = field(default_factory=list)
    taus: List[float] = field(default_factory=list)


@dataclass
class MaterialThermStress:
    """/MAT/THERM_STRESS (M141): Thermal stress expansion modifier.

    Fortran origin: ``starter/source/materials/mat/mat_therm_stress.F``.
    """
    mat_id: int
    title: str = ""
    alpha: float = 0.0
    t0: float = 293.15
    alpha_y: float = 0.0
    alpha_z: float = 0.0


@dataclass
class DampStiff:
    """/DAMP/STIFF (M141): Stiffness proportional damping.

    Fortran origin: ``starter/source/loads/damp/read_damp_stiff.F``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0
    beta: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class AirbagInjector:
    """/AIRBAG/INJECTOR or /INJECTOR (M142): Airbag jetting injector.

    Fortran origin: ``starter/source/airbag/hm_read_injector.F``.
    """
    id: int
    title: str = ""
    sensor_id: int = 0
    ijet: int = 0
    node1: int = 0
    node2: int = 0
    node3: int = 0
    fct_pt: int = 0
    fct_theta: int = 0
    fct_delta: int = 0
    fscale_pt: float = 1.0
    fscale_ptheta: float = 1.0
    fscale_pdelta: float = 1.0


@dataclass
class AirbagVenthole:
    """/AIRBAG/VENTHOLE or /VENTHOLE (M142): Airbag vent hole and membrane burst model.

    Fortran origin: ``starter/source/airbag/hm_read_venthole.F``.
    """
    id: int
    title: str = ""
    surf_vent: int = 0
    iform: int = 1
    avent: float = 0.0
    bvent: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    dpdef: float = 0.0
    dtpdef: float = 0.0
    idtpdef: int = 0
    fct_id_t: int = 0
    fct_id_p: int = 0
    fct_id_a: int = 0
    fscale_t: float = 1.0
    fscale_p: float = 1.0
    fscale_a: float = 1.0


@dataclass
class ErefElement:
    """/EREF/{SHELL|SH3N|BRICK|TETRA4} (M143): Element reference geometry.

    Fortran origin: ``starter/source/initial_conditions/general/hm_read_eref.F``.
    """
    id: int
    title: str = ""
    elem_type: str = "SHELL"
    part_id: int = 0
    node_coords: Dict[int, List[float]] = field(default_factory=dict)


@dataclass
class PropRivet:
    """/PROP/TYPE5 or /PROP/RIVET (M143): Fastener / Rivet connector property.

    Fortran origin: ``starter/source/properties/rivet/hm_read_prop05.F``.
    """
    id: int
    title: str = ""
    mass: float = 0.0
    stiffness: float = 0.0
    fn_fail: float = 0.0
    ft_fail: float = 0.0


@dataclass
class PropXelem:
    """/PROP/TYPE28 or /PROP/XELEM (M143): X-FEM / cohesive element property.

    Fortran origin: ``starter/source/properties/xelem/hm_read_prop28.F``.
    """
    id: int
    title: str = ""
    itip: int = 0
    isurf: int = 0
    alpha: float = 0.0


@dataclass
class AdmeshControl:
    """/ADMESH/{GLOBAL|PART|STATE|BCS|SET} (M143): Adaptive mesh refinement control.

    Fortran origin: ``starter/source/model/remesh/build_admesh.F``.
    """
    id: int
    title: str = ""
    subtype: str = "GLOBAL"
    crit_level: int = 0
    h_min: float = 0.0
    h_max: float = 0.0
    part_id: int = 0


@dataclass
class PreloadBolt:
    """/PRELOAD/BOLT or /SECT/BOLT (M144): Bolt section pretensioning model.

    Fortran origin: ``starter/source/loads/bolt/sboltini.F``.
    """
    id: int
    title: str = ""
    sect_id: int = 0
    sens_id: int = 0
    fct_id: int = 0
    preload: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    torque: float = 0.0
    speed: float = 0.0


@dataclass
class LoadHydro:
    """/LOAD/HYDRO or /LOAD/HYDROSTATIC (M144): Hydrostatic surface pressure loading.

    Fortran origin: ``starter/source/loads/general/hm_read_load.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    density: float = 0.0
    z_free: float = 0.0
    gravity: float = 9.81
    sens_id: int = 0


@dataclass
class Func2DTable:
    """/FUNC_2D or /FUNC2D (M145): 2D bivariate function table f(x, y).

    Fortran origin: ``starter/source/tools/curve/hm_read_func2d.F``.
    """
    id: int
    title: str = ""
    dim: int = 1
    x_vals: List[float] = field(default_factory=list)
    y_vals: List[float] = field(default_factory=list)
    z_vals: List[float] = field(default_factory=list)


@dataclass
class NonlocalModel:
    """/NONLOCAL/mat_id (M145): Non-local damage regularization model.

    Fortran origin: ``starter/source/materials/nonlocal/hm_read_nonlocal.F``.
    """
    mat_id: int
    title: str = ""
    length: float = 0.0
    le_max: float = 0.0
    dens: float = 0.0
    damp: float = 0.0


@dataclass
class FricOrient:
    """/FRIC_ORIENT/id (M145): Friction orientation & anisotropic contact directions.

    Fortran origin: ``starter/source/interfaces/friction/reader/hm_read_friction_orientations.F``.
    """
    id: int
    title: str = ""
    grpart_id: int = 0
    skew_id: int = 0
    phi: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ifric: int = 0


@dataclass
class IniSphCel:
    """/INISPHCEL/part_id (M145): SPH cell initial state.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F``.
    """
    part_id: int
    p: float = 0.0
    rho: float = 0.0
    e: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclass
class MaterialViscPlas:
    """/MAT/VISC_PLAS or /VISC/PLAS (M147): Frequency independent damping model.

    Fortran origin: ``starter/source/materials/visc/hm_read_visc_plas.F90``.
    """
    mat_id: int
    title: str = ""
    lsd_g: float = 0.0
    lsdyna_sigf: float = 0.0


@dataclass
class EbcsPres:
    """/EBCS/PRES/id (M150): Eulerian imposed pressure boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_pres.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    c: float = 0.0
    fct_pres: int = 0
    scale_pres: float = 1.0
    fct_rho: int = 0
    scale_rho: float = 1.0
    fct_en: int = 0
    scale_en: float = 1.0
    lcar: float = 0.0
    r1: float = 0.0
    r2: float = 0.0


@dataclass
class EbcsVel:
    """/EBCS/VEL/id (M150): Eulerian imposed velocity boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_vel.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    c: float = 0.0
    fct_vx: int = 0
    scale_vx: float = 0.0
    fct_vy: int = 0
    scale_vy: float = 0.0
    fct_vz: int = 0
    scale_vz: float = 0.0
    fct_rho: int = 0
    scale_rho: float = 1.0
    fct_en: int = 0
    scale_en: float = 1.0
    lcar: float = 0.0
    r1: float = 0.0
    r2: float = 0.0


@dataclass
class EbcsInlet:
    """/EBCS/INLET/id (M150): Eulerian inflow boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_inlet.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    density: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    energy: float = 0.0
    fct_id: int = 0


@dataclass
class EbcsFluxout:
    """/EBCS/FLUXOUT/id (M150): Eulerian mass outflow boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_fluxout.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    p_ext: float = 0.0


@dataclass
class EbcsGradp0:
    """/EBCS/GRADP0/id (M150): Eulerian zero pressure gradient boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_gradp0.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0


@dataclass
class EbcsNormv:
    """/EBCS/NORMV/id (M150): Eulerian normal velocity constraint.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_normv.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    vn: float = 0.0
    fct_id: int = 0


@dataclass
class EbcsValv:
    """/EBCS/VALVIN or /EBCS/VALVOUT (M150): Eulerian valve boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_valvin.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    is_out: bool = False
    p_open: float = 0.0
    p_close: float = 0.0


@dataclass
class EbcsMonvol:
    """/EBCS/MONVOL/id (M150): Eulerian monitored volume boundary connection.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_monvol.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    monvol_id: int = 0


@dataclass
class BcsWall:
    """/BCS/WALL/id (M150): Eulerian sliding wall boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_wall.F90``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0
    sens_id: int = 0
    tstart: float = 0.0
    tstop: float = 1.0e20

    @property
    def sensor_id(self) -> int:
        return self.sens_id

    @sensor_id.setter
    def sensor_id(self, val: int) -> None:
        self.sens_id = val


@dataclass
class SeatbeltSystem:
    """/SEATBELT/id (M150): Complete seatbelt system assembly.

    Fortran origin: ``starter/source/tools/seatbelts/create_seatbelt.F``.
    """
    id: int
    title: str = ""
    retractor_ids: list[int] = field(default_factory=list)
    slipring_ids: list[int] = field(default_factory=list)
    element_ids: list[int] = field(default_factory=list)


@dataclass
class AmsControl:
    """/AMS (M150): Advanced Mass Scaling starter control.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_sms.F``.
    """
    id: int = 0
    title: str = ""
    grpart_id: int = 0
    dt_target: float = 0.0
    i_ams: int = 1






@dataclass
class Inista:
    """/INISTA or /INISTATE (M151): Initial state file input.

    Fortran origin: ``starter/source/initial_conditions/inista/hm_read_inista.F``.
    """
    id: int = 0
    title: str = ""
    filename: str = ""
    ibal: int = 1
    ioutyy: int = 0
    ioutynn: int = 0


@dataclass
class BemControl:
    """/BEM/FLOW or /BEM/DAA (M151): Boundary element method controls.

    Fortran origin: ``starter/source/loads/bem/hm_read_bem.F``.
    """
    id: int
    title: str = ""
    subtype: str = "FLOW"
    surf_id: int = 0
    nio: int = 0
    grnod_aux_id: int = 0
    freesurf: int = 1


@dataclass
class PerturbControl:
    """/PERTURB/PART/SHELL, /PERTURB/PART/SOLID, /PERTURB/FAIL (M151): Perturbation controls.

    Fortran origin: ``starter/source/general_controls/computation/hm_read_perturb*.F``.
    """
    id: int
    title: str = ""
    subtype: str = "SHELL"
    grpart_id: int = 0
    ityp: int = 1
    fct_id: int = 0
    scale: float = 0.0
    seed: int = 0


@dataclass
class EbcsInip:
    """/EBCS/INIP (M152): Eulerian initial pressure boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_inip.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    rho: float = 0.0
    c: float = 0.0
    lcar: float = 0.0


@dataclass
class EbcsIniv:
    """/EBCS/INIV (M152): Eulerian initial velocity boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_iniv.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    rho: float = 0.0
    c: float = 0.0
    lcar: float = 0.0


@dataclass
class PropInject1Gas:
    """Gas component entry for /PROP/INJECT1."""
    mat_id: int = 0
    fun_id_m: int = 0
    fun_id_t: int = 0
    fscale_m: float = 1.0
    fscale_t: float = 1.0


@dataclass
class PropInject1:
    """/PROP/INJECT1 or /INJECT1 (M152): Gas injector property definition.

    Fortran origin: ``starter/source/properties/injector/hm_read_inject1.F``.
    """
    id: int
    title: str = ""
    n_gases: int = 1
    iflow: int = 0
    ascale_t: float = 1.0
    gases: List[PropInject1Gas] = field(default_factory=list)


@dataclass
class PropInject2Gas:
    """Gas mixture molar fraction entry for /PROP/INJECT2."""
    mat_id: int = 0
    molar_fraction: float = 1.0
    fun_id_mf: int = 0


@dataclass
class PropInject2:
    """/PROP/INJECT2 or /INJECT2 (M152): Multi-gas mixture injector property definition.

    Fortran origin: ``starter/source/properties/injector/hm_read_inject2.F``.
    """
    id: int
    title: str = ""
    n_gases: int = 1
    iflow: int = 0
    fun_id_m: int = 0
    fun_id_t: int = 0
    fscale_m: float = 1.0
    fscale_t: float = 1.0
    ascale_t: float = 1.0
    gases: List[PropInject2Gas] = field(default_factory=list)


@dataclass
class PropJoint:
    """/PROP/TYPE33 or /PROP/JOINT (M152): Specialized kinematic joints.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop33*.F``.
    """
    id: int
    title: str = ""
    joint_type: str = "SPH"
    skew_id: int = 0
    params: Dict = field(default_factory=dict)


@dataclass
class PropTorsion:
    """/PROP/TYPE35 or /PROP/TORSION (M152): Torsion bar spring property.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop35.F``.
    """
    id: int
    title: str = ""
    mass: float = 0.0
    k_elas: float = 0.0
    x_lim1: float = 0.0
    x_lim2: float = 0.0
    k_post: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    r_load: float = 0.0
    f_scal: float = 1.0
    fct_id1: int = 0
    fct_id2: int = 0
    fct_id3: int = 0
    fct_id4: int = 0


@dataclass
class PropSpringElasPlas:
    """/PROP/SPR_ELAS_PLAS (M152): Elastic-plastic spring with kinematic hardening.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop_spr_ep.F``.
    """
    id: int
    title: str = ""
    skew_id: int = 0
    i_utyp: int = 1
    pid1: int = 0
    pid2: int = 0
    mid1: int = 0
    k_stiff: float = 0.0
    area: float = 0.0
    ixx: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0
    params: Dict = field(default_factory=dict)


@dataclass
class PropSpringBeam:
    """/PROP/TYPE44 (M152): Non-linear beam-spring connector.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop44.F``.
    """
    id: int
    title: str = ""
    skew_id: int = 0
    idamp: int = 0
    nc_filter: int = 0
    params: Dict = field(default_factory=dict)


@dataclass
class PropSpotweld:
    """/PROP/TYPE45 (M152): Spotweld connector beam.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop45.F``.
    """
    id: int
    title: str = ""
    skew_id: int = 0
    sensor_id: int = 0
    knn: float = 0.0
    cr: float = 0.0
    scf: float = 1.0
    params: Dict = field(default_factory=dict)


@dataclass
class PropBushing:
    """/PROP/TYPE46 (M152): Bushing connector spring.

    Fortran origin: ``starter/source/properties/spring/hm_read_prop46.F``.
    """
    id: int
    title: str = ""
    mass: float = 0.0
    k_elas: float = 0.0
    x_lim1: float = 0.0
    x_lim2: float = 0.0
    k_post: float = 0.0
    damp: float = 0.0
    epsi: int = 0
    idens: int = 0
    params: Dict = field(default_factory=dict)


@dataclass
class FailNxt:
    """/FAIL/NXT (M159): Strain-rate dependent failure model.

    Fortran origin: ``starter/source/materials/fail/hm_read_fail_nxt.F`` / CFG ``fail_nxt.cfg``.
    """
    mat_id: int
    fct_id1: int = 0
    fct_id2: int = 0
    ifail_sh: int = 1
    fail_id: int = 0


@dataclass
class FailLadDama:
    """/FAIL/LAD_DAMA (M159/M189): Ladevèze damage failure model.

    Fortran origin: ``starter/source/materials/fail/hm_read_fail_lad_dama.F`` / CFG ``fail_lad_dama.cfg``.
    """
    id: int = 0
    mat_id: int = 0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    gamma1: float = 0.0
    gamma2: float = 0.0
    y0: float = 0.0
    yc: float = 0.0
    k: float = 0.0
    k_lad: float = 0.0
    a: float = 0.0
    a_dama: float = 0.0
    tau_max: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 1
    fail_id: int = 0
    title: str = ""


FailLadeveze = FailLadDama


@dataclass
class FailInievo:
    """/FAIL/INIEVO (M159): Multi-criterion damage initiation & evolution model.

    Fortran origin: ``starter/source/materials/fail/hm_read_fail_inievo.F`` / CFG ``fail_inievo.cfg``.
    """
    mat_id: int
    ninievo: int = 1
    ishear: int = 0
    ilen: int = 0
    failip: int = 0
    pthk: float = 0.0
    models: List[Dict] = field(default_factory=list)
    fail_id: int = 0


@dataclass
class MaterialSprSeatbelt:
    """/MAT/LAW114 or /MAT/SPR_SEATBELT (M161): Seatbelt spring material model.

    Fortran origin: ``starter/source/materials/mat/hm_read_mat114.F`` / CFG ``mat114_spr_seatbelt.cfg``.
    """
    id: int
    title: str = ""
    rho: float = 0.0
    lmin: float = 0.0
    k: float = 0.0
    c: float = 0.0
    fun_l: int = 0
    fun_ul: int = 0
    xscale: float = 1.0
    fscale: float = 1.0
    e: float = 0.0
    i: float = 0.0
    j: float = 0.0
    fmax: float = 0.0
    mmax: float = 0.0
    as_: float = 0.0
    r: float = 0.0
    law: int = 114
    rho0: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rho0 and self.rho:
            self.rho0 = self.rho
        elif not self.rho and self.rho0:
            self.rho = self.rho0
        self.params = {
            "rho": self.rho, "rho0": self.rho0, "lmin": self.lmin, "k": self.k, "stiff1": self.k,
            "c": self.c, "damp1": self.c, "fun_l": self.fun_l, "fun_ul": self.fun_ul,
            "xscale": self.xscale, "fscale": self.fscale, "e": self.e, "E": self.e,
            "i": self.i, "j": self.j, "fmax": self.fmax, "mmax": self.mmax,
            "as": self.as_, "r": self.r
        }


@dataclass
class MaterialShSeatbelt:
    """/MAT/LAW119 or /MAT/SH_SEATBELT (M161): 2D shell seatbelt fabric material model.

    Fortran origin: ``starter/source/materials/mat/hm_read_mat119.F`` / CFG ``mat119_sh_seatbelt.cfg``.
    """
    id: int
    title: str = ""
    rho: float = 0.0
    lmin: float = 0.0
    k: float = 0.0
    c: float = 0.0
    re: float = 0.0
    fun_l: int = 0
    fun_ul: int = 0
    fscale1: float = 1.0
    fscale2: float = 1.0
    ireload: int = 0
    e22: float = 0.0
    nu12: float = 0.0
    g12: float = 0.0
    fscale22: float = 1.0
    ecoat: float = 0.0
    nucoat: float = 0.0
    tcoat: float = 0.0
    law: int = 119
    rho0: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rho0 and self.rho:
            self.rho0 = self.rho
        elif not self.rho and self.rho0:
            self.rho = self.rho0
        self.params = {
            "rho": self.rho, "rho0": self.rho0, "lmin": self.lmin, "k": self.k, "stiff1": self.k,
            "c": self.c, "damp1": self.c, "re": self.re, "fun_l": self.fun_l, "fun_ul": self.fun_ul,
            "fscale1": self.fscale1, "fscale2": self.fscale2, "ireload": self.ireload,
            "e22": self.e22, "nu12": self.nu12, "g12": self.g12, "fscale22": self.fscale22,
            "ecoat": self.ecoat, "nucoat": self.nucoat, "tcoat": self.tcoat
        }


@dataclass
class MaterialTapo:
    """/MAT/LAW120 or /MAT/TAPO (M161): Tape/woven fabric material model with plasticity and damage.

    Fortran origin: ``starter/source/materials/mat/hm_read_mat120.F`` / CFG ``mat120_tapo.cfg``.
    """
    id: int
    title: str = ""
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iform: int = 1
    itrx: int = 0
    idam: int = 0
    thick: float = 0.0
    tab_id: int = 0
    xscale: float = 1.0
    yscale: float = 1.0
    tau: float = 0.0
    q: float = 0.0
    beta: float = 1.0
    h: float = 0.0
    af1: float = 0.0
    af2: float = 0.0
    ah1: float = 0.0
    ah2: float = 0.0
    as_: float = 0.0
    cc: float = 1e21
    gam0: float = 0.0
    gamf: float = 0.0
    d1c: float = 0.0
    d2c: float = 0.0
    d1f: float = 0.0
    d2f: float = 0.0
    d_trx: float = 0.0
    d_jc: float = 0.0
    exp_n: float = 0.0
    law: int = 120
    rho0: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rho0 and self.rho:
            self.rho0 = self.rho
        elif not self.rho and self.rho0:
            self.rho = self.rho0
        self.params = {
            "rho": self.rho, "rho0": self.rho0, "refer_rho": self.refer_rho, "e": self.e, "E": self.e,
            "nu": self.nu, "iform": self.iform, "itrx": self.itrx, "idam": self.idam,
            "thick": self.thick, "tab_id": self.tab_id, "xscale": self.xscale, "yscale": self.yscale,
            "tau": self.tau, "q": self.q, "beta": self.beta, "h": self.h,
            "af1": self.af1, "af2": self.af2, "ah1": self.ah1, "ah2": self.ah2, "as": self.as_,
            "cc": self.cc, "gam0": self.gam0, "gamf": self.gamf,
            "d1c": self.d1c, "d2c": self.d2c, "d1f": self.d1f, "d2f": self.d2f,
            "d_trx": self.d_trx, "d_jc": self.d_jc, "exp_n": self.exp_n
        }


@dataclass
class MaterialPlasRate:
    """/MAT/LAW121 or /MAT/PLAS_RATE (M161): Strain-rate dependent elastoplastic material model.

    Fortran origin: ``starter/source/materials/mat/hm_read_mat121.F`` / CFG ``matl121_plasrate.cfg``.
    """
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    ires: int = 0
    ivisc: int = 0
    fcut: float = 0.0
    tdel: float = 0.0
    fct_sig0: int = 0
    xscale_sig0: float = 1.0
    yscale_sig0: float = 1.0
    fct_youn: int = 0
    xscale_youn: float = 1.0
    yscale_youn: float = 1.0
    fct_tang: int = 0
    xscale_tang: float = 1.0
    tang: float = 0.0
    fct_fail: int = 0
    ifail: int = 0
    xscale_fail: float = 1.0
    yscale_fail: float = 1.0
    law: int = 121
    rho0: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rho0 and self.rho:
            self.rho0 = self.rho
        elif not self.rho and self.rho0:
            self.rho = self.rho0
        self.params = {
            "rho": self.rho, "rho0": self.rho0, "e": self.e, "E": self.e, "nu": self.nu,
            "ires": self.ires, "ivisc": self.ivisc, "fcut": self.fcut, "tdel": self.tdel,
            "fct_sig0": self.fct_sig0, "xscale_sig0": self.xscale_sig0, "yscale_sig0": self.yscale_sig0,
            "fct_youn": self.fct_youn, "xscale_youn": self.xscale_youn, "yscale_youn": self.yscale_youn,
            "fct_tang": self.fct_tang, "xscale_tang": self.xscale_tang, "tang": self.tang,
            "fct_fail": self.fct_fail, "ifail": self.ifail, "xscale_fail": self.xscale_fail, "yscale_fail": self.yscale_fail
        }


@dataclass
class MaterialCdpm2:
    """/MAT/LAW124 or /MAT/CDPM2 (M161): Concrete damage plasticity model 2.

    Fortran origin: ``starter/source/materials/mat/hm_read_mat124.F`` / CFG ``matl124_cdpm2.cfg``.
    """
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    irate: int = 0
    fcut: float = 0.0
    ecc: float = 0.0
    qh0: float = 0.0
    ft: float = 0.0
    fc: float = 0.0
    hp: float = 0.0
    ah: float = 0.0
    bh: float = 0.0
    ch: float = 0.0
    dh: float = 0.0
    as_: float = 0.0
    bs: float = 0.0
    df: float = 0.0
    dflag: int = 0
    dtype: int = 0
    ireg: int = 0
    wf: float = 0.0
    wf1: float = 0.0
    ft1: float = 0.0
    efc: float = 0.0
    law: int = 124
    rho0: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.rho0 and self.rho:
            self.rho0 = self.rho
        elif not self.rho and self.rho0:
            self.rho = self.rho0
        self.params = {
            "rho": self.rho, "rho0": self.rho0, "e": self.e, "E": self.e, "nu": self.nu,
            "irate": self.irate, "fcut": self.fcut, "ecc": self.ecc, "qh0": self.qh0,
            "ft": self.ft, "fc": self.fc, "hp": self.hp, "ah": self.ah, "bh": self.bh,
            "ch": self.ch, "dh": self.dh, "as": self.as_, "bs": self.bs, "df": self.df,
            "dflag": self.dflag, "dtype": self.dtype, "ireg": self.ireg,
            "wf": self.wf, "wf1": self.wf1, "ft1": self.ft1, "efc": self.efc
        }


@dataclass
class FailHcDsse:
    """/FAIL/HC_DSSE (M162): Hosford-Coulomb & DSSE ductile failure and fracture model.

    Fortran origin: ``starter/source/materials/fail/hc_dsse/hm_read_fail_hc_dsse.F``.
    """
    mat_id: int
    ifail_sh: int = 1
    pthkf: float = 0.0
    iflag: int = 0
    a_hc_dsse: float = 0.0
    b_hc_dsse: float = 0.0
    c_hc_dsse: float = 0.0
    d_hc_dsse: float = 0.0
    n_f: float = 1.0
    fail_id: int = 0


@dataclass
class FailMullins:
    """/FAIL/MULLINS_OR or /FAIL/MULLINS (M162): Mullins effect hyperelastic damage model.

    Fortran origin: ``starter/source/materials/fail/mullins_or/hm_read_fail_mullins_or.F``.
    """
    mat_id: int
    coefr: float = 1.0
    beta: float = 0.0
    coefm: float = 0.0
    fail_id: int = 0


@dataclass
class FailSnconnect:
    """/FAIL/SNCONNECT (M162): S-N curve based connector fatigue failure model.

    Fortran origin: ``starter/source/materials/fail/snconnect/hm_read_fail_snconnect.F``.
    """
    mat_id: int
    alpha_0: float = 0.0
    beta_0: float = 0.0
    alpha_f: float = 0.0
    beta_f: float = 0.0
    ifail_so: int = 0
    isym: int = 0
    fct_idon: int = 0
    fct_idos: int = 0
    fct_idfn: int = 0
    fct_idfs: int = 0
    xscale_0: float = 1.0
    xscale_f: float = 1.0
    area_scale: float = 1.0
    fail_id: int = 0


@dataclass
class FailSpalling:
    """/FAIL/SPALLING (M162/M189): Spalling / hydrodynamic tensile cutoff failure model.

    Fortran origin: ``starter/source/materials/fail/spalling/hm_read_fail_spalling.F90``.
    """
    id: int = 0
    mat_id: int = 0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    d4: float = 0.0
    d5: float = 0.0
    eps_dot_0: float = 1.0e-20
    epsilon_dot_0: float = 1.0e-20
    p_min: float = -1.0e20
    ifail_so: int = 1
    fail_id: int = 0
    title: str = ""


FailSpall = FailSpalling


@dataclass
class DfsDetcord:
    """/DFS/DETCORD (M163): Detonation cord ignition model.

    Fortran origin: ``starter/source/initial_conditions/detonation/read_dfs_detcord.F``.
    """
    id: int
    title: str = ""
    grnd_id: int = 0
    t_det: float = 0.0
    v_cj: float = 0.0
    iopt: int = 3
    mat_id: int = 0
    nodes: List[int] = field(default_factory=list)


@dataclass
class AleMat:
    """/ALE/MAT (M163): ALE material volume fraction and formulation directives.

    Fortran origin: ``starter/source/materials/ale/read_ale_mat.F``.
    """
    mat_id: int
    ale_flrd: float = 0.0


@dataclass
class EulerMat:
    """/EULER/MAT (M163): Euler material volume fraction and formulation directives.

    Fortran origin: ``starter/source/materials/ale/read_euler_mat.F``.
    """
    mat_id: int
    euler_flrd: float = 0.0


@dataclass
class EbcsNrf:
    """/EBCS/NRF or /BCS/NRF (M164): Non-reflecting frontier boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/hm_read_ebcs_nrf.F``.
    """
    id: int
    title: str = ""
    surf_id: int = 0
    tcar_p: float = 0.0
    tcar_vf: float = 0.0


@dataclass
class BcsWall:
    """/BCS/WALL (M164): Sliding wall boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_wall.F90``.
    """
    id: int
    title: str = ""
    set_id: int = 0
    sensor_id: int = 0
    tstart: float = 0.0
    tstop: float = 1.0e30

    @property
    def grnod_id(self) -> int:
        return self.set_id

    @property
    def sens_id(self) -> int:
        return self.sensor_id


@dataclass
class EbcsLoad:
    """/EBCS/{PRES|VEL|INLET} (M164): Eulerian boundary condition loading directive.

    Fortran origin: ``starter/source/boundary_conditions/ebcs/read_ebcs.F``.
    """
    id: int
    kind: str = ""
    title: str = ""
    surf_id: int = 0
    fct_id: int = 0
    sens_id: int = 0
    dir: str = ""
    v0: float = 0.0
    p0: float = 0.0
    scale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    imat: int = 0
    rho: float = 0.0
    ener: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclass
class SlipringShell:
    """/SLIPRING/SHELL (M164): 2D shell slipring seatbelt element.

    Fortran origin: ``starter/source/elements/seatbelts/hm_read_slipring_shell.F``.
    """
    id: int
    title: str = ""
    el_set1: int = 0
    el_set2: int = 0
    node_set: int = 0
    sens_id: int = 0
    flow_flag: int = 0
    a: float = 0.0
    ed_factor: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fricd: float = 0.0
    fric_d: float = 0.0
    xscale1: float = 1.0
    yscale2: float = 1.0
    xscale2: float = 1.0
    fct_id3: int = 0
    fct_id4: int = 0
    frics: float = 0.0
    fric_s: float = 0.0
    xscale3: float = 1.0
    yscale4: float = 1.0
    xscale4: float = 1.0

    def __post_init__(self):
        if self.fricd != 0.0 and self.fric_d == 0.0:
            self.fric_d = self.fricd
        elif self.fric_d != 0.0 and self.fricd == 0.0:
            self.fricd = self.fric_d
        if self.frics != 0.0 and self.fric_s == 0.0:
            self.fric_s = self.frics
        elif self.fric_s != 0.0 and self.frics == 0.0:
            self.frics = self.fric_s


@dataclass
class SubLaminatePly:
    """Ply layer within a sub-laminate stack."""
    ply_id: int
    phi: float = 0.0
    zi: float = 0.0
    p_thick_fail: float = 0.0
    f_weight: float = 1.0


@dataclass
class SubLaminate:
    """/SUBLAMINATE or /STACK/SUB_LAMINATE (M165): Sub-laminate composite ply stack definition.

    Fortran origin: ``LAMINATE/sub_laminate_p51.cfg`` and ``LAMINATE/stack_sub_laminate.cfg``.
    """
    id: int
    title: str = ""
    plies: List[SubLaminatePly] = field(default_factory=list)


@dataclass
class InertiaPart:
    """/INERTIA/PART (M169): Part inertia modifier definition.

    Fortran origin: ``starter/source/nodes_elements/inertia/hm_read_inertia.F``.
    """
    id: int
    title: str = ""
    part_id: int = 0
    skew_id: int = 0
    iflag: int = 0
    mass: float = 0.0
    xg: float = 0.0
    yg: float = 0.0
    zg: float = 0.0
    ixx: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0
    ixy: float = 0.0
    iyz: float = 0.0
    izx: float = 0.0


@dataclass
class AleGridConstraint:
    """/ALE/GRID/DISP or /ALE/GRID/VEL (M169): ALE grid nodal boundary constraint.

    Fortran origin: ``starter/source/constraints/ale/hm_read_ale_grid.F``.
    """
    id: int
    kind: str = "DISP"  # 'DISP' | 'VEL'
    title: str = ""
    grnod_id: int = 0
    fun_id: int = 0
    skew_id: int = 0
    tra_code: str = ""
    scale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class MaterialConc:
    """/MAT/LAW24 or /MAT/CONC (M170): Concrete material model.

    Fortran origin: ``starter/source/materials/mat24/hm_read_mat24.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e_c: float = 0.0
    nu: float = 0.0
    f_c: float = 0.0
    ft_on_fc: float = 0.0
    fb_on_fc: float = 0.0
    f2_on_fc: float = 0.0
    s0_on_fc: float = 0.0
    h_t: float = 0.0
    d_sup: float = 0.0
    eps_max: float = 0.0
    k_y: float = 0.0
    r_t: float = 0.0
    r_c: float = 0.0
    h_bp: float = 0.0
    alpha_y: float = 0.0
    alpha_f: float = 0.0
    v_max: float = 0.0
    f_k: float = 0.0
    f0: float = 0.0
    h_v0: float = 0.0
    e2: float = 0.0
    ssig: float = 0.0
    setan: float = 0.0
    alpha1: float = 0.0
    alpha2: float = 0.0
    alpha3: float = 0.0


@dataclass
class MaterialBarlat:
    """/MAT/LAW87 or /MAT/BARLAT (M170): Barlat 2000 anisotropic plasticity model.

    Fortran origin: ``starter/source/materials/mat87/hm_read_mat87.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iflag: int = 0
    vflag: int = 0
    strain1: float = 0.0
    exp1: float = 0.0
    ifit: int = 0
    alphas: List[float] = field(default_factory=lambda: [1.0] * 8)
    sigma_00: float = 0.0
    sigma_45: float = 0.0
    sigma_90: float = 0.0
    sigma_b: float = 0.0
    r_00: float = 0.0
    r_45: float = 0.0
    r_90: float = 0.0
    r_b: float = 0.0
    a_exp: int = 6
    alpha_vol: float = 1.0
    n_hard: float = 0.0
    fcut: float = 0.0
    fsmooth: int = 0
    a_swift: float = 0.0
    eps0: float = 0.0
    q_voce: float = 0.0
    beta: float = 0.0
    k0: float = 0.0


@dataclass
class MaterialLaw83:
    """/MAT/LAW83 or /MAT/SPR_JOU (M170): Non-linear spring/joint material model.

    Fortran origin: ``starter/source/materials/mat83/hm_read_mat83.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    imass: int = 0
    fun_a1: int = 0
    fscale11: float = 1.0
    fscale22: float = 1.0
    alpha: float = 0.0
    beta: float = 0.0
    rn: float = 0.0
    rs: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    fun_a2: int = 0
    fun_a3: int = 0
    fscale33: float = 1.0


@dataclass
class MaterialLaw80:
    """/MAT/LAW80 or /MAT/TRANSFO (M170): Metallurgical phase transformation steel model.

    Fortran origin: ``starter/source/materials/mat80/hm_read_mat80.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    fct_ide: int = 0
    scale_e: float = 1.0
    time_unit: float = 3600.0
    fsmooth: int = 0
    fcut: float = 0.0
    ceps: float = 0.0
    peps: float = 0.0
    fun_a: List[int] = field(default_factory=lambda: [0] * 5)
    fscale_y: List[float] = field(default_factory=lambda: [1.0] * 5)
    scale_x: List[float] = field(default_factory=lambda: [1.0] * 5)
    theta: List[float] = field(default_factory=lambda: [0.0] * 4)
    alpha1: float = 0.0
    alpha2: float = 0.0


@dataclass
class MaterialLaw117:
    """/MAT/LAW117 or /MAT/COH_MC (M171): Cohesive element material model.

    Fortran origin: ``starter/source/materials/mat117/hm_read_mat117.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e_elas_n: float = 0.0
    e_elas_s: float = 0.0
    imass: int = 0
    idel: int = 0
    irupt: int = 0
    fct_tn: int = 0
    fct_tt: int = 0
    tmax_n: float = 0.0
    tmax_s: float = 0.0
    fscale_x: float = 1.0
    gic: float = 0.0
    giic: float = 0.0
    exp_g: float = 1.0
    exp_bk: float = 1.0
    gamma: float = 0.0


@dataclass
class MaterialLaw90:
    """/MAT/LAW90 or /MAT/PLAS_TAB (M171): Strain-rate dependent tabular foam/plasticity material model.

    Fortran origin: ``starter/source/materials/mat90/hm_read_mat90.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e0: float = 0.0
    nu: float = 0.0
    nl: int = 0
    ismooth: int = 0
    fcut: float = 0.0
    shape: float = 0.0
    hys: float = 0.0
    fct_ids: List[int] = field(default_factory=list)
    eps_dots: List[float] = field(default_factory=list)
    fscales: List[float] = field(default_factory=list)


@dataclass
class MaterialLaw33:
    """/MAT/LAW33 or /MAT/FOAM_PLAS (M171): Crushable foam plasticity material model.

    Fortran origin: ``starter/source/materials/mat33/hm_read_mat33.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    itype: int = 0
    fun_a1: int = 0
    ifscale: float = 1.0
    p0: float = 0.0
    phi: float = 0.0
    gama0: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    etan: float = 0.0
    eta1: float = 0.0
    eta2: float = 0.0


@dataclass
class MatHeatModifier:
    """/MAT/HEAT or /HEAT/MAT (M171): Material thermal property modifier.

    Fortran origin: ``starter/source/materials/heat/hm_read_heat.F``.
    """
    id: int
    mat_id: int = 0
    t0: float = 0.0
    rho0_cp: float = 0.0
    as_solid: float = 0.0
    bs_solid: float = 0.0
    t1: float = 1.0e30
    al_liquid: float = 0.0
    bl_liquid: float = 0.0
    efrac: float = 1.0


@dataclass
class MatNonlocalModifier:
    """/MAT/NONLOCAL or /NONLOCAL/MAT (M171): Non-local regularized damage material modifier.

    Fortran origin: ``starter/source/materials/nonlocal/hm_read_nonlocal.F``.
    """
    id: int
    mat_id: int = 0
    length: float = 0.0
    le_max: float = 0.0


# ----------------------------------------------------------------------------
# Advanced Tabular Foam, Viscoelastic Foam, Visco-Hyperelastic, Honeycomb & Cowper-Symonds Material Models (M172)
# ----------------------------------------------------------------------------

@dataclass
class MaterialLaw66:
    """/MAT/LAW66 or /MAT/FOAM_TAB (M172): Tabular foam material model.

    Fortran origin: ``starter/source/materials/mat/mat066/hm_read_mat66.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    c_hard: float = 0.0
    f_cut: float = 0.0
    fsmooth: int = 0
    israte: int = 0
    p_c: float = 0.0
    p_t: float = 0.0
    ec: float = 0.0
    rpct: float = 0.0
    funct_idc: int = 0
    funct_idt: int = 0
    fscalec: float = 1.0
    fscalet: float = 1.0
    epsilon_0: float = 0.0
    c: float = 0.0
    sigma_y0: float = 0.0
    vp: int = 0
    fnyrt_idc: int = 0
    fnyrt_idt: int = 0
    yrate_fscalec: float = 1.0
    yrate_fscalet: float = 1.0
    nfunc: int = 0
    tfunc: int = 0
    func_c_list: List[int] = field(default_factory=list)
    eps_c_list: List[float] = field(default_factory=list)
    fscale_c_list: List[float] = field(default_factory=list)
    func_t_list: List[int] = field(default_factory=list)
    eps_t_list: List[float] = field(default_factory=list)
    fscale_t_list: List[float] = field(default_factory=list)


@dataclass
class MaterialLaw35:
    """/MAT/LAW35 or /MAT/FOAM_VISC (M172): Viscoelastic foam material model.

    Fortran origin: ``starter/source/materials/mat/mat035/hm_read_mat35.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    n: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    itype: int = 0
    pmin: float = 0.0
    func_idf: int = 0
    fscalepres: float = 1.0
    fsmooth: int = 0
    fcut: float = 0.0
    et: float = 0.0
    nu_t: float = 0.0
    eta_0: float = 0.0
    lamda: float = 0.0
    p0: float = 0.0
    phi: float = 0.0
    gama0: float = 0.0


@dataclass
class MaterialLaw62:
    """/MAT/LAW62 or /MAT/VISC_HYP (M172): Viscoelastic hyperelastic Ogden material model.

    Fortran origin: ``starter/source/materials/mat/mat062/hm_read_mat62.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    nu: float = 0.0
    order_n: int = 0
    order_m: int = 0
    mu_max: float = 0.0
    mu_arr: List[float] = field(default_factory=list)
    alpha_arr: List[float] = field(default_factory=list)
    gamma_arr: List[float] = field(default_factory=list)
    tau_arr: List[float] = field(default_factory=list)


@dataclass
class MaterialLaw28:
    """/MAT/LAW28 or /MAT/HONEYCOMB (M172): Orthotropic honeycomb crushable material model.

    Fortran origin: ``starter/source/materials/mat/mat028/hm_read_mat28.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    e33: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    fun_a2: int = 0
    gflag: int = 0
    fscale11: float = 1.0
    fscale22: float = 1.0
    fscale33: float = 1.0
    epsr1: float = 0.0
    epsr2: float = 0.0
    epsr3: float = 0.0
    fun_a3: int = 0
    fun_b3: int = 0
    fun_a4: int = 0
    vflag: int = 0
    fscale12: float = 1.0
    fscale23: float = 1.0
    fscale13: float = 1.0
    epsr4: float = 0.0
    epsr5: float = 0.0
    epsr6: float = 0.0


@dataclass
class MaterialLaw44:
    """/MAT/LAW44 or /MAT/COWPER_SYMONDS (M172): Cowper-Symonds strain-rate dependent elastoplastic material model.

    Fortran origin: ``starter/source/materials/mat/mat044/hm_read_mat44.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iflag: int = 0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    hard: float = 0.0
    sig_max: float = 0.0
    src: float = 0.0
    sre: float = 0.0
    strflag: int = 0
    fsmooth: int = 0
    fcut: float = 0.0
    vflag: int = 0
    eps_max: float = 0.0
    eta1: float = 0.0
    eta2: float = 0.0
    yld_func: int = 0
    yld_scale: float = 1.0


@dataclass
class MaterialLaw88:
    """/MAT/LAW88 or /MAT/HYPER_ELAS or /MAT/TABULATED_HYPERELASTIC (M173):
    Tabulated hyperelastic Ogden material model with strain-rate unloading and damage.

    Fortran origin: ``starter/source/materials/mat/mat088/hm_read_mat88.F90``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    nu: float = 0.495
    bulk: float = 0.0
    fcut: float = 0.0
    fsmooth: int = 0
    nl: int = 0
    ifunc_unload: int = 0
    fscale_unload: float = 1.0
    hys: float = 0.0
    shape: float = 1.0
    tension: int = 0
    rtype: int = 0
    func_load_list: list = field(default_factory=list)
    fscale_load_list: list = field(default_factory=list)
    rate_load_list: list = field(default_factory=list)
    lamfit_list: list = field(default_factory=list)
    sgl: float = 0.0
    sw: float = 0.0
    st: float = 0.0
    g: float = 0.0
    sigf: float = 0.0
    kfail: float = 0.0
    gam1: float = 0.0
    gam2: float = 0.0
    eh: float = 0.0
    failip: int = 0


@dataclass
class MaterialLaw92:
    """/MAT/LAW92 or /MAT/ARRUDA_BOYCE (M173): Arruda-Boyce 8-chain hyperelastic polymer model.

    Fortran origin: ``starter/source/materials/mat/mat092/hm_read_mat92.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    mu: float = 0.0
    d: float = 0.0
    lam: float = 7.0
    itype: int = 1
    fct_id: int = 0
    nu: float = 0.0
    fscale: float = 1.0


@dataclass
class MaterialLaw94:
    """/MAT/LAW94 or /MAT/YEOH (M173): Yeoh 3rd-order polynomial hyperelastic model.

    Fortran origin: ``starter/source/materials/mat/mat094/hm_read_mat94.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    c10: float = 0.0
    c20: float = 0.0
    c30: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0


@dataclass
class MaterialLaw46:
    """/MAT/LAW46 or /MAT/HYD_VISC or /MAT/LES_FLUID (M173): Hydrodynamic viscous fluid model with Smagorinsky turbulence.

    Fortran origin: ``starter/source/materials/mat/mat046/hm_read_mat46.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    c: float = 0.0
    nu: float = 0.0
    istf: int = 1
    smag: float = 1.0
    cps: float = 0.0


@dataclass
class MaterialLaw69:
    """/MAT/LAW69 or /MAT/HYP_EXT_COMP (M173): Hyperelastic material model extended to compression.

    Fortran origin: ``starter/source/materials/mat/mat069/hm_read_mat69.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ref_rho: float = 0.0
    iflag: int = 1
    fct_id_bulk: int = 0
    nu: float = 0.495
    fscale: float = 1.0
    nip: int = 2
    icheck: int = -3
    fct_id_data: int = 0


@dataclass
class MaterialLaw124:
    """/MAT/LAW124 or /MAT/CDPM2 (M174): Concrete Damage Plastic Model 2 (CDPM2).

    Fortran origin: ``starter/source/materials/mat/mat124/hm_read_mat124.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    irate: int = 0
    fcut: float = 0.0
    ecc: float = 0.0
    qh0: float = 0.0
    ft: float = 0.0
    fc: float = 0.0
    hp: float = 0.0
    ah: float = 0.0
    bh: float = 0.0
    ch: float = 0.0
    dh: float = 0.0
    as_: float = 0.0
    bs: float = 0.0
    df: float = 0.0
    dflag: int = 0
    dtype: int = 0
    ireg: int = 0
    wf: float = 0.0
    wf1: float = 0.0
    ft1: float = 0.0
    efc: float = 0.0


@dataclass
class MaterialLaw126:
    """/MAT/LAW126 or /MAT/JOHNSON_HOLMQUIST_CONCRETE (M174): Johnson-Holmquist concrete damage model.

    Fortran origin: ``starter/source/materials/mat/mat126/hm_read_mat126.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    g: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    fc: float = 0.0
    t0: float = 0.0
    c: float = 0.0
    eps0: float = 0.0
    fcut: float = 0.0
    sfmax: float = 0.0
    efmin: float = 0.0
    pc: float = 0.0
    muc: float = 0.0
    pl: float = 0.0
    mul: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    idel: int = 0
    eps_max: float = 0.0
    ifailso: int = 0
    ct: float = 0.0
    powt: float = 0.0
    cc: float = 0.0
    powc: float = 0.0


@dataclass
class MaterialLaw125:
    """/MAT/LAW125 or /MAT/LAMINATED_COMPOSITE (M174): Multi-layered laminated composite model.

    Fortran origin: ``starter/source/materials/mat/mat125/hm_read_mat125.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    ifail: int = 0
    gab: float = 0.0
    gca: float = 0.0
    gbc: float = 0.0
    prba: float = 0.0
    prca: float = 0.0
    prcb: float = 0.0
    lce11t: int = 0
    e11t: float = 0.0
    fct_t11: int = 0
    t11: float = 0.0
    slimt11: float = 0.0
    lce11c: int = 0
    e11c: float = 0.0
    fct_c11: int = 0
    c11: float = 0.0
    slimc11: float = 0.0
    lce22t: int = 0
    e22t: float = 0.0
    fct_t22: int = 0
    t22: float = 0.0
    slimt22: float = 0.0
    lce22c: int = 0
    e22c: float = 0.0
    fct_c22: int = 0
    c22: float = 0.0
    slimc22: float = 0.0
    lce33t: int = 0
    e33t: float = 0.0
    fct_t33: int = 0
    t33: float = 0.0
    slimt33: float = 0.0
    lce33c: int = 0
    e33c: float = 0.0
    fct_c33: int = 0
    c33: float = 0.0
    slimc33: float = 0.0
    g12a: float = 0.0
    t12a: float = 0.0
    g12b: float = 0.0
    t12b: float = 0.0
    slims12: float = 0.0
    fct_g12a: int = 0
    fct_t12a: int = 0
    fct_g12b: int = 0
    fct_t12b: int = 0
    g31a: float = 0.0
    t31a: float = 0.0
    g31b: float = 0.0
    t31b: float = 0.0
    slims31: float = 0.0
    fct_g31a: int = 0
    fct_t31a: int = 0
    fct_g31b: int = 0
    fct_t31b: int = 0
    g23a: float = 0.0
    t23a: float = 0.0
    g23b: float = 0.0
    t23b: float = 0.0
    slims23: float = 0.0
    fct_g23a: int = 0
    fct_t23a: int = 0
    fct_g23b: int = 0
    fct_t23b: int = 0
    epsf: float = 0.0
    epsr: float = 0.0
    dmax: float = 0.0
    fct_fail: int = 0
    fail: float = 0.0
    fcut: float = 0.0


@dataclass
class MaterialLaw127:
    """/MAT/LAW127 or /MAT/ENHANCED_COMPOSITE (M174): Enhanced orthotropic composite model.

    Fortran origin: ``starter/source/materials/mat/mat127/hm_read_mat127.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gca: float = 0.0
    gbc: float = 0.0
    prba: float = 0.0
    prca: float = 0.0
    prcb: float = 0.0
    xt: float = 0.0
    slimt1: float = 0.0
    lcxt: int = 0
    scalcxt: float = 1.0
    yt: float = 0.0
    slimt2: float = 0.0
    lcyt: int = 0
    scalcyt: float = 1.0
    sc: float = 0.0
    slimsc: float = 0.0
    lcsc: int = 0
    scalcsc: float = 1.0
    xc: float = 0.0
    slimc1: float = 0.0
    lcxc: int = 0
    scalcxc: float = 1.0
    yc: float = 0.0
    slimc2: float = 0.0
    lcyc: int = 0
    scalcyc: float = 1.0
    fcut: float = 0.0
    alph: float = 0.0
    beta: float = 0.0
    two_way: int = 0
    ti: int = 0
    dfailt: float = 0.0
    dfailc: float = 0.0
    dfails: float = 0.0
    dfailm: float = 0.0
    ratio: float = 0.0
    ncyred: int = 0
    tfail: float = 0.0
    fbrt: float = 0.0
    ycfac: float = 0.0
    efs: float = 0.0
    epsf: float = 0.0
    epsr: float = 0.0
    tsmd: float = 0.0


@dataclass
class MaterialLaw130:
    """/MAT/LAW130 or /MAT/MODIFIED_HONEYCOMB (M174): Modified crushable honeycomb material model.

    Fortran origin: ``starter/source/materials/mat/mat130/hm_read_mat130.F``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    vf: float = 0.0
    mu: float = 0.0
    iform: int = 0
    shdflg: int = 0
    lca: int = 0
    lcb: int = 0
    lcc: int = 0
    lcs: int = 0
    lcab: int = 0
    lcbc: int = 0
    lcca: int = 0
    lcsr: int = 0
    eaau: float = 0.0
    ebbu: float = 0.0
    eccu: float = 0.0
    gabu: float = 0.0
    gbcu: float = 0.0
    gcau: float = 0.0
    rfac: float = 0.0
    tsef: float = 0.0
    ssef: float = 0.0
    pru: int = 0
    lcsra: int = 0
    lcsrb: int = 0
    lcsrc: int = 0
    lcsrab: int = 0
    lcsrbc: int = 0
    lcsrca: int = 0
    pruab: float = 0.0
    pruac: float = 0.0
    prubc: float = 0.0
    pruba: float = 0.0
    pruca: float = 0.0
    prucb: float = 0.0


@dataclass
class MaterialLaw128:
    """/MAT/LAW128 or /MAT/HILL_VISC_PLAST (M175): Hill anisotropic viscoplastic material model.

    Fortran origin: ``starter/source/materials/mat/mat128/hm_read_mat128.F90`` / CFG ``Law128_hill_visc_plast.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    kin: float = 0.0
    tab_id: int = 0
    facy: float = 0.0
    facx: float = 0.0
    qr1: float = 0.0
    cr1: float = 0.0
    qr2: float = 0.0
    cr2: float = 0.0
    qx1: float = 0.0
    cx1: float = 0.0
    qx2: float = 0.0
    cx2: float = 0.0
    epsp0: float = 0.0
    cp: float = 0.0
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    f: float = 0.0
    g: float = 0.0
    h: float = 0.0
    l: float = 0.0
    m: float = 0.0
    n: float = 0.0


@dataclass
class MaterialLaw129:
    """/MAT/LAW129 or /MAT/THERM_CREEP (M175): Thermo-elasto-viscoplastic creep material model.

    Fortran origin: ``starter/source/materials/mat/mat129/hm_read_mat129.F90`` / CFG ``Law129_therm_creep.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    alpha: float = 0.0
    tref: float = 0.0
    f_young: int = 0
    f_nu: int = 0
    f_yld: int = 0
    f_alpha: int = 0
    isensor: int = 0
    itab: int = 0
    facy: float = 0.0
    qr1: float = 0.0
    cr1: float = 0.0
    qr2: float = 0.0
    cr2: float = 0.0
    f_qr: int = 0
    f_cr: int = 0
    qx1: float = 0.0
    cx1: float = 0.0
    qx2: float = 0.0
    cx2: float = 0.0
    f_qx: int = 0
    f_cx: int = 0
    epsp0: float = 0.0
    cp: float = 0.0
    f_cc: int = 0
    f_cp: int = 0
    crpa: float = 0.0
    crpn: float = 0.0
    crpm: float = 0.0
    f_a: int = 0
    f_n: int = 0
    f_m: int = 0
    crp_law: int = 0
    crsig: float = 0.0
    crt: float = 0.0
    crpq: float = 0.0
    eps0: float = 0.0
    f_q: int = 0
    f_sig: int = 0


@dataclass
class MaterialLaw123:
    """/MAT/LAW123 or /MAT/DAIMLER_PINHO (M175): Daimler-Pinho 3D composite damage model.

    Fortran origin: ``starter/source/materials/mat/mat123/hm_read_mat123.F90`` / CFG ``matl123_daimler_pinho.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gca: float = 0.0
    gbc: float = 0.0
    prba: float = 0.0
    prca: float = 0.0
    prcb: float = 0.0
    enkink: float = 0.0
    ena: float = 0.0
    enb: float = 0.0
    ent: float = 0.0
    enl: float = 0.0
    xc: float = 0.0
    xt: float = 0.0
    yc: float = 0.0
    yt: float = 0.0
    sl: float = 0.0
    fio: float = 53.0
    sigy: float = 0.0
    lcss: int = 0
    beta: float = 0.0
    efs: float = 0.0
    ratio: float = 0.0
    fcut: float = 0.0


@dataclass
class MaterialLaw132:
    """/MAT/LAW132 or /MAT/DAIMLER_CAMANHO (M175): Daimler-Camanho composite failure model.

    Fortran origin: ``starter/source/materials/mat/mat132/hm_read_mat132.F90`` / CFG ``matl132_daimler_camanho.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gca: float = 0.0
    gbc: float = 0.0
    prba: float = 0.0
    prca: float = 0.0
    prcb: float = 0.0
    gxc: float = 0.0
    gxt: float = 0.0
    gyc: float = 0.0
    gyt: float = 0.0
    gsl: float = 0.0
    xc: float = 0.0
    xt: float = 0.0
    yc: float = 0.0
    yt: float = 0.0
    sl: float = 0.0
    gxc0: float = 0.0
    gxt0: float = 0.0
    xc0: float = 0.0
    xt0: float = 0.0
    fio: float = 53.0
    sigy: float = 0.0
    etan: float = 0.0
    beta: float = 0.0
    lcss: int = 0
    epsf23: float = 0.0
    epsr23: float = 0.0
    tsmd23: float = 0.0
    epsf31: float = 0.0
    epsr31: float = 0.0
    tsmd31: float = 0.0
    ef11t: float = 0.0
    ef11c: float = 0.0
    ef22t: float = 0.0
    ef22c: float = 0.0
    ef12: float = 0.0
    ef23: float = 0.0
    ef31: float = 0.0
    cf12: float = 0.0
    cf23: float = 0.0
    cf31: float = 0.0
    ratio: float = 0.0
    fcut: float = 0.0


@dataclass
class MaterialLaw134:
    """/MAT/LAW134 or /MAT/VISCOUS_FOAM (M175): Viscous foam material model.

    Fortran origin: ``starter/source/materials/mat/mat134/hm_read_mat134.F90`` / CFG ``matl134_viscous_foam.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    e1: float = 0.0
    n1: float = 0.0
    nu: float = 0.0
    e2: float = 0.0
    v2: float = 0.0
    n2: float = 0.0


@dataclass
class MaterialLaw104:
    """/MAT/LAW104 or /MAT/JOHNS_VOCE_DRUCKER (M176): Combined Drucker-Prager and Voce hardening model.

    Fortran origin: ``starter/source/materials/mat/mat104/hm_read_mat104.F`` / CFG ``matl104_drucker.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    ires: int = 1
    sigma_r: float = 1e20
    h: float = 0.0
    qv: float = 0.0
    bv: float = 0.0
    cdr: float = 0.0
    cjc: float = 0.0
    epsp0: float = 1.0
    fcut: float = 1e4
    tss: float = 0.0
    tref: float = 20.0
    tini: float = 20.0
    eta: float = 0.0
    cp: float = 0.0
    eps_iso: float = 1e20
    eps_ad: float = 2e20


@dataclass
class MaterialLaw105:
    """/MAT/LAW105 or /MAT/POWDER_BURN (M176): Powder burn propellant material model.

    Fortran origin: ``starter/source/materials/mat/mat105/hm_read_mat105.F90`` / CFG ``matl105.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    bulk: float = 0.0
    p0: float = 0.0
    psh: float = 0.0
    gas_d: float = 0.0
    gas_eg: float = 0.0
    gr: float = 0.0
    c: float = 0.0
    alpha: float = 0.0
    func_b: int = 0
    scale_b: float = 1.0
    scale_p: float = 1.0
    func_gam: int = 0
    scale_gam: float = 1.0
    scale_rho: float = 1.0
    c1: float = 0.0
    c2: float = 0.0


@dataclass
class MaterialLaw106:
    """/MAT/LAW106 or /MAT/JCOOK_ALM (M176): Johnson-Cook additive manufacturing phase transformation model.

    Fortran origin: ``starter/source/materials/mat/mat106/hm_read_mat106.F90`` / CFG ``mat_law106.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fct_id3: int = 0
    sigy: float = 0.0
    beta: float = 0.0
    hard_n: float = 1.0
    ep_max: float = 1e30
    sig_max: float = 1e30
    fcut: float = 0.0
    vp: int = 2
    nmax: int = 3
    tol: float = 1e-7
    cjc: float = 0.0
    deps0: float = 0.0
    m: float = 1.0
    tmelt: float = 1e30
    spheat: float = 0.0
    eta: float = 0.0
    t0: float = 300.0
    tr: float = 300.0


@dataclass
class MaterialLaw107:
    """/MAT/LAW107 or /MAT/PAPER_LIGHT (M176): Orthotropic elastoplastic paper model.

    Fortran origin: ``starter/source/materials/mat/mat107/hm_read_mat107.F`` / CFG ``matl107_paper_light.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    e3: float = 0.0
    ires: int = 0
    itab: int = 0
    ismooth: int = 0
    nu21: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g13: float = 0.0
    xi1: float = 0.0
    xi2: float = 0.0
    g1c: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    sigy1: float = 0.0
    cini1: float = 0.0
    s1: float = 0.0
    sigy2: float = 0.0
    cini2: float = 0.0
    s2: float = 0.0
    sigy1c: float = 0.0
    cini1c: float = 0.0
    s1c: float = 0.0
    sigy2c: float = 0.0
    cini2c: float = 0.0
    s2c: float = 0.0
    sigyt: float = 0.0
    cinit: float = 0.0
    st: float = 0.0
    tab_yld1: int = 0
    xscale1: float = 1.0
    yscale1: float = 1.0
    tab_yld2: int = 0
    xscale2: float = 1.0
    yscale2: float = 1.0
    tab_yld1c: int = 0
    xscale1c: float = 1.0
    yscale1c: float = 1.0
    tab_yld2c: int = 0
    xscale2c: float = 1.0
    yscale2c: float = 1.0
    tab_yldt: int = 0
    xscale_t: float = 1.0
    yscale_t: float = 1.0


@dataclass
class MaterialLaw110:
    """/MAT/LAW110 or /MAT/VEGTER (M176): Vegter anisotropic yield locus material model.

    Fortran origin: ``starter/source/materials/mat/mat110/hm_read_mat110.F`` / CFG ``matl110_vegter.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    ires: int = 0
    icrit: int = 1
    tab_yld: int = 0
    xscale: float = 1.0
    yscale: float = 1.0
    fbi: float = 0.0
    rhobi: float = 0.0
    sigma_r: float = 0.0
    dsigm: float = 0.0
    beta: float = 0.0
    omega: float = 0.0
    hard_n: float = 0.0
    eps0: float = 0.0
    sigs: float = 0.0
    dg0: float = 0.0
    deps0: float = 0.0
    m: float = 0.0
    tini: float = 0.0
    chard: float = 0.0
    fcut: float = 0.0
    vp: int = 0
    ismooth: int = 0
    tab_temp: int = 0
    rm_0: float = 0.0
    rm_45: float = 0.0
    rm_90: float = 0.0
    ag_0: float = 0.0
    ag_45: float = 0.0
    ag_90: float = 0.0
    r_0: float = 1.0
    r_45: float = 1.0
    r_90: float = 1.0
    angles_data: list = field(default_factory=list)


@dataclass
class MaterialLaw115:
    """/MAT/LAW115 or /MAT/DESHPANDE_FLECK (M176): Deshpande-Fleck metallic foam model.

    Fortran origin: ``starter/source/materials/mat/mat115/hm_read_mat115.F`` / CFG ``matl115_deshfleck.cfg``.
    """
    id: int
    title: str = ""
    rho0: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    ires: int = 2
    istat: int = 0
    alpha: float = 0.0
    cfail: float = 0.0
    pfail: float = 0.0
    sigp: float = 0.0
    gamma: float = 0.0
    epsd: float = 0.0
    alpha2: float = 0.0
    beta: float = 0.0
    rhof0: float = 0.0
    sigp_c0: float = 0.0
    sigp_c1: float = 0.0
    sigp_n: float = 0.0
    alpha2_c0: float = 0.0
    alpha2_c1: float = 0.0
    alpha2_n: float = 0.0
    gamma_c0: float = 0.0
    gamma_c1: float = 0.0
    gamma_n: float = 0.0
    beta_c0: float = 0.0
    beta_c1: float = 0.0
    beta_n: float = 0.0


# M177 Material Dataclasses: LAW109, LAW111, LAW112, LAW116, LAW122, LAW158
@dataclass
class MaterialLaw109:
    """/MAT/LAW109: Thermo-viscoplastic with tabular yield, temperature, and Taylor-Quinney conversion."""
    id: int
    title: str = ""
    rho0: float = 0.0
    young: float = 0.0
    nu: float = 0.0
    cp: float = 0.0
    eta: float = 1.0
    tref: float = 293.0
    tini: float = 293.0
    tab_yld: int = 0
    tab_temp: int = 0
    xscale_h: float = 1.0
    yscale_h: float = 1.0
    ismooth: int = 1
    tab_eta: int = 0
    xscale_eta: float = 1.0


@dataclass
class MaterialLaw111:
    """/MAT/LAW111 & /MAT/MARLOW: Marlow hyperelastic model constructed from test data."""
    id: int
    title: str = ""
    rho0: float = 0.0
    itype: int = 1
    fct_id: int = 0
    fscale: float = 1.0
    nu: float = 0.495


@dataclass
class MaterialLaw112:
    """/MAT/LAW112 & /MAT/PAPER / /MAT/PLAS_PAPER: Comprehensive 3D orthotropic paper plasticity model."""
    id: int
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    e3: float = 0.0
    ires: int = 0
    itab: int = 0
    ismooth: int = 0
    nu21: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g13: float = 0.0
    k: float = 0.0
    e3c: float = 0.0
    cc: float = 0.0
    nu1p: float = 0.0
    nu2p: float = 0.0
    nu4p: float = 0.0
    nu5p: float = 0.0
    # Analytic (itab = 0)
    s01: float = 0.0
    a01: float = 0.0
    b01: float = 0.0
    c01: float = 0.0
    s02: float = 0.0
    a02: float = 0.0
    b02: float = 0.0
    c02: float = 0.0
    s03: float = 0.0
    a03: float = 0.0
    b03: float = 0.0
    c03: float = 0.0
    s04: float = 0.0
    a04: float = 0.0
    b04: float = 0.0
    c04: float = 0.0
    s05: float = 0.0
    a05: float = 0.0
    b05: float = 0.0
    c05: float = 0.0
    asig: float = 0.0
    bsig: float = 0.0
    csig: float = 0.0
    tau0: float = 0.0
    atau: float = 0.0
    btau: float = 0.0
    # Tabulated (itab = 1)
    tab_yld1: int = 0
    xscale1: float = 1.0
    yscale1: float = 1.0
    tab_yld2: int = 0
    xscale2: float = 1.0
    yscale2: float = 1.0
    tab_yld3: int = 0
    xscale3: float = 1.0
    yscale3: float = 1.0
    tab_yld4: int = 0
    xscale4: float = 1.0
    yscale4: float = 1.0
    tab_yld5: int = 0
    xscale5: float = 1.0
    yscale5: float = 1.0
    tab_yldc: int = 0
    xscalec: float = 1.0
    yscalec: float = 1.0
    tab_ylds: int = 0
    xscales: float = 1.0
    yscales: float = 1.0


@dataclass
class MaterialLaw116:
    """/MAT/LAW116 & /MAT/COH_HYST / /MAT/COHESIVE_HYSTERETIC: Cohesive zone hysteretic damage model."""
    id: int
    title: str = ""
    rho0: float = 0.0
    young: float = 0.0
    g: float = 0.0
    thick: float = 0.0
    imass: int = 1
    idel: int = 1
    icrit: int = 1
    gc1_ini: float = 0.0
    gc1_inf: float = 0.0
    sratg1: float = 0.0
    fg1: float = 0.0
    gc2_ini: float = 0.0
    gc2_inf: float = 0.0
    sratg2: float = 0.0
    fg2: float = 0.0
    siga1: float = 0.0
    sigb1: float = 0.0
    srate1: float = 0.0
    order1: int = 1
    fail1: int = 1
    siga2: float = 0.0
    sigb2: float = 0.0
    srate2: float = 0.0
    order2: int = 1
    fail2: int = 1


@dataclass
class MaterialLaw122:
    """/MAT/LAW122 & /MAT/MODIFIED_LADEVEZE / /MAT/LADEVEZE_DELAM: Modified Ladevèze Delamination & Composite Damage Model."""
    id: int
    title: str = ""
    rho0: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    e3: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    nu12: float = 0.0
    nu23: float = 0.0
    nu31: float = 0.0
    e1c: float = 0.0
    gamma: float = 0.0
    ish: int = 0
    itr: int = 0
    ires: int = 0
    sigy0: float = 0.0
    beta: float = 0.0
    hard_m: float = 0.0
    hard_a: float = 0.0
    eps_fti: float = 0.0
    eps_ftu: float = 0.0
    dftu: float = 0.0
    eps_fci: float = 0.0
    eps_fcu: float = 0.0
    dfcu: float = 0.0
    ibuck: int = 0
    ifuncd1: int = 0
    dsat1: float = 0.0
    y0: float = 0.0
    yc: float = 0.0
    b: float = 0.0
    dmax: float = 0.0
    yr: float = 0.0
    ysp: float = 0.0
    ifuncd2: int = 0
    dsat2: float = 0.0
    y0p: float = 0.0
    ycp: float = 0.0
    ifuncd2c: int = 0
    dsat2c: float = 0.0
    y0pc: float = 0.0
    ycpc: float = 0.0
    epsd11: float = 0.0
    d11: float = 0.0
    n11: float = 0.0
    d11u: float = 0.0
    n11u: float = 0.0
    epsd12: float = 0.0
    d22: float = 0.0
    n22: float = 0.0
    d12: float = 0.0
    n12: float = 0.0
    epsdr0: float = 0.0
    dr0: float = 0.0
    nr0: float = 0.0
    ltype11: int = 0
    ltype12: int = 0
    ltyper0: int = 0
    fcut: float = 0.0


@dataclass
class MaterialLaw158:
    """/MAT/LAW158 & /MAT/FABR_NL / /MAT/FABRIC_NL: Nonlinear Anisotropic Fabric Material."""
    id: int
    title: str = ""
    rho0: float = 0.0
    s1: float = 0.1
    s2: float = 0.1
    flex: float = 0.0
    flex1: float = 0.0
    flex2: float = 0.0
    zerostress: float = 0.0
    sensor_id: int = 0
    fun_a1: int = 0
    c1: float = 1.0
    fun_a2: int = 0
    c2: float = 1.0
    fun_a3: int = 0
    c3: float = 1.0
    fun_a4: int = 0
    fun_a5: int = 0


# ============================================================================
# M178: BCS_CYCLIC, LOAD_PCYL, EBCS_MONVOL
# ============================================================================

@dataclass
class BcsCyclic:
    """/BCS/CYCLIC (M178): Cyclic symmetry boundary condition coupling two node groups in a skew coordinate system."""
    id: int
    skew_id: int = 0
    grnd_id1: int = 0
    grnd_id2: int = 0
    title: str = ""


@dataclass
class PcylLoad:
    """/LOAD/PCYL (M178): Pressure load in cylindrical coordinates with radius-time table."""
    id: int
    surf_id: int = 0
    sens_id: int = 0
    frame_id: int = 0
    table_id: int = 0
    xscale_r: float = 1.0
    xscale_t: float = 1.0
    yscale_p: float = 1.0
    title: str = ""


@dataclass
class EbcsMonvol:
    """/EBCS/MONVOL (M178): Monitored volume Eulerian boundary condition linking surface fluxes to monitored gas bags."""
    id: int
    surf_id: int = 0
    sens_id: int = 0
    monvol_id: int = 0
    fscale: float = 1.0
    title: str = ""


@dataclass
class DampVrel:
    """/DAMP/VREL (M179): Relative velocity damping in skew coordinate system."""
    id: int
    title: str = ""
    grnod_id: int = 0
    skew_id: int = 0
    alpha_x: float = 0.0
    alpha_y: float = 0.0
    alpha_z: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class FailSyazwan:
    """/FAIL/SYAZWAN (M179): Syazwan ductile fracture criterion for metals."""
    id: int = 0
    mat_id: int = 0
    icard: int = 1
    epfmin: float = 0.0
    failip: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    epf_comp: float = 0.0
    epf_shear: float = 0.0
    epf_tens: float = 0.0
    epf_plstrn: float = 0.0
    epf_biax: float = 0.0
    dinit: int = 0
    dam_sf: float = 1.0
    max_dam: float = 1.0
    inst: int = 0
    iform: int = 0
    n_val: float = 0.0
    softexp: float = 0.0
    reg_func: int = 0
    ref_len: float = 0.0
    reg_scale: float = 1.0
    coeffs: List[float] = field(default_factory=list)
    fail_id: int = 0


@dataclass
class MatLaw113Dof:
    """DOF parameters for /MAT/LAW113 (M179)."""
    stiff: float = 0.0
    damp: float = 0.0
    acoeft: float = 1.0
    bcoeft: float = 1.0
    dcoeft: float = 1.0
    fun_a: int = 0
    hflag: int = 0
    fun_b: int = 0
    fun_c: int = 0
    fun_d: int = 0
    min_rup: float = -1.0e30
    max_rup: float = 1.0e30
    prop_f: float = 0.0
    prop_e: float = 0.0
    scale: float = 1.0
    prop_h: float = 1.0
    fun_k: int = 0


@dataclass
class MatLaw113:
    """/MAT/LAW113 / /MAT/SPR_BEAM (M179): Nonlinear spring-beam material model."""
    id: int
    rho: float = 0.0
    ifail: int = 0
    ileng: int = 0
    ifail2: int = 0
    dofs: List[MatLaw113Dof] = field(default_factory=list)
    trans_vel0: float = 1.0
    rot_vel0: float = 1.0
    asrate: float = 1.0e30
    israte: int = 0
    dir_fails: List[List[float]] = field(default_factory=list)
    title: str = ""


@dataclass
class MatLaw79:
    """/MAT/LAW79 / /MAT/JOHN_HOLM (M179): Johnson-Holmquist ceramic material model."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    g: float = 0.0
    a: float = 0.0
    b: float = 0.0
    m: float = 1.0
    n: float = 1.0
    c: float = 0.0
    eps0: float = 1.0
    sigma_fmax: float = 1.0e30
    fcut: float = 0.0
    t0: float = 0.0
    hel: float = 0.0
    phel: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    idel: int = 0
    epsmax: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    beta: float = 0.0
    title: str = ""


@dataclass
class MatViscLprony:
    """/MAT/VISC_LPRONY / /VISC/LPRONY (M179): Viscoelastic large Prony series model."""
    id: int
    m: int = 0
    form: int = 0
    flag_visc: int = 0
    gamai: List[float] = field(default_factory=list)
    taui: List[float] = field(default_factory=list)
    title: str = ""


# M180: MAT_LAW190, MAT_LAW41, FAIL_CHANG, PROP_TYPE20, PROP_TYPE21, PROP_TYPE22
@dataclass
class MatLaw190:
    """/MAT/LAW190 / /MAT/FOAM_DUBOIS (M180): Du Bois foam model with 3D table."""
    id: int
    rho: float = 0.0
    e0: float = 0.0
    nu: float = 0.0
    hu: float = 0.0
    shape: float = 1.0
    fun_1: int = 0
    xscale_1: float = 1.0
    scale_1: float = 1.0
    title: str = ""


@dataclass
class MatLaw41:
    """/MAT/LAW41 / /MAT/LEE_T (M180): Lee-Tarver explosive reaction kinetics and JWL EOS."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    ireac: int = 0
    a_r: float = 0.0
    b_r: float = 0.0
    r_1r: float = 0.0
    r_2r: float = 0.0
    r_3r: float = 0.0
    a_p: float = 0.0
    b_p: float = 0.0
    r_1p: float = 0.0
    r_2p: float = 0.0
    r_3p: float = 0.0
    c_vr: float = 0.0
    c_vp: float = 0.0
    enq: float = 0.0
    nitrs: int = 0
    epsilon_0: float = 0.0
    ftol: float = 0.0
    i_coeff: float = 0.0
    b_coeff: float = 0.0
    x_coeff: float = 0.0
    g1: float = 0.0
    d_coeff: float = 0.0
    y_coeff: float = 0.0
    c_coeff: float = 0.0
    kn: float = 0.0
    chi: float = 0.0
    tol: float = 0.0
    g2: float = 0.0
    e_coeff: float = 0.0
    g_coeff: float = 0.0
    z_coeff: float = 0.0
    ccrit: float = 0.0
    figmax: float = 0.0
    fg1max: float = 0.0
    fg2min: float = 0.0
    g0: float = 0.0
    t_initial: float = 293.15
    title: str = ""


@dataclass
class FailChang:
    """/FAIL/CHANG (M180): Chang-Chang composite failure model."""
    id: int = 0
    mat_id: int = 0
    sigma_1t: float = 0.0
    sigma_2t: float = 0.0
    sigma_12: float = 0.0
    sigma_1c: float = 0.0
    sigma_2c: float = 0.0
    beta: float = 0.0
    tau_max: float = 0.0
    ifail_sh: int = 1
    failip: int = 0
    fail_id: int = 0


@dataclass
class PropType20:
    """/PROP/TYPE20 / /PROP/TSHELL (M180): Thick shell property."""
    id: int = 0
    isolid: int = 15
    ismstr: int = 0
    icpre: int = 0
    icstr: int = 0
    inpts_r: int = 2
    inpts_s: int = 2
    inpts_t: int = 2
    iint: int = 1
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    h: float = 0.1
    deltat_min: float = 0.0
    nbp: int = 0
    title: str = ""


PropTshell = PropType20
PropThickShell = PropType20


@dataclass
class PropType21:
    """/PROP/TYPE21 / /PROP/TSH_ORTH (M180): Orthotropic thick shell property."""
    id: int
    isolid: int = 15
    ismstr: int = 0
    icstr: int = 0
    inpts_r: int = 2
    inpts_s: int = 2
    inpts_t: int = 2
    iint: int = 1
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    phi: float = 0.0
    deltat_min: float = 0.0
    title: str = ""


@dataclass
class PropType22Layer:
    """Layer definition for /PROP/TYPE22 (/PROP/TSH_COMP)."""
    phi: float = 0.0
    thick: float = 0.0
    zi: float = 0.0
    mat_id: int = 0


@dataclass
class PropType22:
    """/PROP/TYPE22 / /PROP/TSH_COMP (M180): Composite layered thick shell property."""
    id: int
    isolid: int = 15
    ismstr: int = 0
    icstr: int = 0
    inpts_r: int = 2
    inpts_s: int = 2
    inpts_t: int = 2
    iint: int = 1
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0
    ashear: float = 0.0
    layers: List[PropType22Layer] = field(default_factory=list)
    deltat_min: float = 0.0
    title: str = ""


@dataclass
class FailFabric:
    """/FAIL/FABRIC or /FAIL/FABR (M181): Fabric failure criterion."""
    id: int = 0
    mat_id: int = 0
    epsilon_f1: float = 0.0
    epsilon_r1: float = 0.0
    epsilon_f2: float = 0.0
    epsilon_r2: float = 0.0
    ndir: int = 0
    fct_id: int = 0
    fail_id: int = 0


@dataclass
class FailHoffman:
    """/FAIL/HOFFMAN (M181): Hoffman 3D orthotropic failure criterion."""
    id: int = 0
    mat_id: int = 0
    sigma_1t: float = 0.0
    sigma_2t: float = 0.0
    sigma_1c: float = 0.0
    sigma_2c: float = 0.0
    sigma_12: float = 0.0
    tau_max: float = 0.0
    fcut: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 0
    fail_id: int = 0


@dataclass
class FailMaxStrain:
    """/FAIL/MAX_STRAIN or /FAIL/MAXSTRAIN (M181): Maximum strain failure model."""
    id: int = 0
    mat_id: int = 0
    eps1_max: float = 0.0
    eps2_max: float = 0.0
    gam12_max: float = 0.0
    tau_max: float = 0.0
    fcut: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 0
    fail_id: int = 0


@dataclass
class FailTsaiHill:
    """/FAIL/TSAI_HILL or /FAIL/TSAIHILL (M181): Tsai-Hill composite failure criterion."""
    id: int = 0
    mat_id: int = 0
    x11: float = 0.0
    x22: float = 0.0
    s12: float = 0.0
    tau_max: float = 0.0
    fcut: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 0
    fail_id: int = 0


@dataclass
class FailTsaiWu:
    """/FAIL/TSAI_WU or /FAIL/TSAIWU (M181): Tsai-Wu quadratic composite failure criterion."""
    id: int = 0
    mat_id: int = 0
    sigma_1t: float = 0.0
    sigma_2t: float = 0.0
    sigma_1c: float = 0.0
    sigma_2c: float = 0.0
    sigma_12: float = 0.0
    alpha: float = 0.0
    tau_max: float = 0.0
    fcut: float = 0.0
    ifail_sh: int = 1
    ifail_so: int = 0
    fail_id: int = 0


@dataclass
class PropType6:
    """/PROP/TYPE6 or /PROP/SOL_ORTH (M181): Solid orthotropic property."""
    id: int = 0
    isolid: int = 14
    ismstr: int = 0
    icpre: int = 0
    itetra10: int = 0
    inpts_r: int = 1
    inpts_s: int = 1
    inpts_t: int = 1
    itetra4: int = 0
    iframe: int = 0
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    h: float = 0.1
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    skew_csid: int = 0
    refplane: int = 0
    orthtrop: int = 0
    mat_beta: float = 0.0
    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0
    deltat_min: float = 0.0
    vdef_min: float = 0.0
    vdef_max: float = 0.0
    asp_max: float = 0.0
    col_min: float = 0.0
    ndir: int = 0
    sphpart_id: int = 0
    istrain: int = 0
    ihkt: int = 0
    nbp: int = 0
    title: str = ""


PropSolOrth = PropType6
PropSolidOrth = PropType6


@dataclass
class LoadCload:
    """/LOAD/CLOAD or /CLOAD (M181): Concentrated nodal load."""
    id: int
    curve_id: int = 0
    dir: str = "X"
    skew_id: int = 0
    sens_id: int = 0
    grnod_id: int = 0
    xscale: float = 1.0
    magnitude: float = 1.0
    title: str = ""


@dataclass
class LoadPload:
    """/LOAD/PLOAD or /PLOAD (M181): Surface pressure load."""
    id: int
    surf_id: int = 0
    curve_id: int = 0
    sens_id: int = 0
    ipinch: int = 0
    idel: int = 1
    functype: int = 1
    xscale: float = 1.0
    magnitude: float = 1.0
    title: str = ""


# M182: MAT_LAW114, MAT_LAW117, MAT_LAW119, MAT_LAW120, MAT_LAW121, PROP_TYPE26, PROP_TYPE27

@dataclass
class MatLaw114:
    """/MAT/LAW114 or /MAT/SPR_SEATBELT (M182): 1D seatbelt spring material."""
    id: int
    rho: float = 0.0
    lmin: float = 0.0
    stiff1: float = 0.0
    damp1: float = 0.0
    fun_l: int = 0
    fun_ul: int = 0
    xcoeft1: float = 1.0
    fcoeft1: float = 1.0
    young: float = 0.0
    ibend: float = 0.0
    itors: float = 0.0
    fmax: float = 0.0
    mmax: float = 0.0
    shear_area: float = 0.0
    rfac: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def k(self) -> float:
        return self.stiff1

    @property
    def c(self) -> float:
        return self.damp1

    @property
    def xscale(self) -> float:
        return self.xcoeft1

    @property
    def fscale(self) -> float:
        return self.fcoeft1

    @property
    def e(self) -> float:
        return self.young

    @property
    def i(self) -> float:
        return self.ibend

    @property
    def j(self) -> float:
        return self.itors

    @property
    def as_(self) -> float:
        return self.shear_area

    @property
    def r(self) -> float:
        return self.rfac


MatSprSeatbelt = MatLaw114


@dataclass
class MatLaw117:
    """/MAT/LAW117 or /MAT/COH_TAB (M182): Tabulated cohesive zone material."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    en: float = 0.0
    es: float = 0.0
    imass: int = 0
    idel: int = 0
    irupt: int = 0
    fct_tn: int = 0
    fct_tt: int = 0
    tn: float = 0.0
    ts: float = 0.0
    fscale_x: float = 1.0
    gic: float = 0.0
    giic: float = 0.0
    exp_g: float = 0.0
    exp_bk: float = 0.0
    gamma: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def e_elas_n(self) -> float:
        return self.en

    @property
    def e_elas_s(self) -> float:
        return self.es

    @property
    def e(self) -> float:
        return self.en

    @property
    def tmax_n(self) -> float:
        return self.tn

    @property
    def tmax_s(self) -> float:
        return self.ts


MatCohTab = MatLaw117


@dataclass
class MatLaw119:
    """/MAT/LAW119 or /MAT/SH_SEATBELT (M182): 2D shell seatbelt material."""
    id: int
    rho: float = 0.0
    lmin: float = 0.0
    stiff1: float = 0.0
    damp1: float = 0.0
    re: float = 0.0
    fun_l: int = 0
    fun_ul: int = 0
    fcoeft1: float = 1.0
    fcoeft2: float = 1.0
    ireload: int = 0
    e22: float = 0.0
    nu12: float = 0.0
    g12: float = 0.0
    fcoeft22: float = 1.0
    ecoat: float = 0.0
    nucoat: float = 0.0
    tcoat: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def k(self) -> float:
        return self.stiff1

    @property
    def c(self) -> float:
        return self.damp1

    @property
    def fscale1(self) -> float:
        return self.fcoeft1

    @property
    def fscale2(self) -> float:
        return self.fcoeft2

    @property
    def fscale22(self) -> float:
        return self.fcoeft22


MatShSeatbelt = MatLaw119


@dataclass
class MatLaw120:
    """/MAT/LAW120 or /MAT/TAPO (M182): Tabulated orthotropic Pont-Pack material."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iform: int = 0
    itrx: int = 0
    idam: int = 0
    thick: float = 0.0
    tab_id: int = 0
    xscale: float = 1.0
    yscale: float = 1.0
    tau0: float = 0.0
    q: float = 0.0
    beta: float = 0.0
    h: float = 0.0
    af1: float = 0.0
    af2: float = 0.0
    ah1: float = 0.0
    ah2: float = 0.0
    as_: float = 0.0
    cc: float = 0.0
    gam0: float = 0.0
    gamf: float = 0.0
    d1c: float = 0.0
    d2c: float = 0.0
    d1f: float = 0.0
    d2f: float = 0.0
    dtrx: float = 0.0
    djc: float = 0.0
    exp_n: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def tau(self) -> float:
        return self.tau0

    @property
    def d_trx(self) -> float:
        return self.dtrx

    @property
    def d_jc(self) -> float:
        return self.djc


MatTapo = MatLaw120


@dataclass
class MatLaw121:
    """/MAT/LAW121 or /MAT/PLAS_RATE (M182): Tabulated rate-dependent elastoplastic material."""
    id: int
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    ires: int = 2
    ivisc: int = 0
    fcut: float = 0.0
    dtmin: float = 0.0
    fct_sig0: int = 0
    xscale_sig0: float = 1.0
    yscale_sig0: float = 1.0
    fct_youn: int = 0
    xscale_youn: float = 1.0
    yscale_youn: float = 1.0
    fct_tang: int = 0
    xscale_tang: float = 1.0
    tang: float = 0.0
    fct_fail: int = 0
    ifail: int = 0
    xscale_fail: float = 1.0
    yscale_fail: float = 1.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def tdel(self) -> float:
        return self.dtmin


MatPlasRate = MatLaw121


@dataclass
class PropType26Curve:
    fct_id: int = 0
    fscale: float = 1.0
    strain_rate: float = 0.0


@dataclass
class PropType26:
    """/PROP/TYPE26 or /PROP/SPR_TAB (M182): Tabulated nonlinear spring property."""
    id: int
    mass: float = 0.0
    sens_id: int = 0
    isflag: int = 0
    ileng: int = 0
    dmin: float = 0.0
    nfunc: int = 1
    nfund: int = 1
    lscale: float = 1.0
    kmax: float = 1.0
    dmax: float = 0.0
    alpha: float = 1.0
    loading_curves: list[PropType26Curve] = field(default_factory=list)
    unloading_curves: list[PropType26Curve] = field(default_factory=list)
    title: str = ""


PropSprTab = PropType26


@dataclass
class PropType27:
    """/PROP/TYPE27 or /PROP/SPR_BDAMP (M182): Spring with bilinear/barycentric damping."""
    id: int
    mass: float = 0.0
    sens_id: int = 0
    isflag: int = 0
    ileng: int = 0
    itens: int = 0
    ifail: int = 0
    stiff: float = 0.0
    damp: float = 0.0
    nexp: float = 1.0
    delta_min: float = 0.0
    delta_max: float = 0.0
    gap: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    ascale1: float = 1.0
    fscale1: float = 1.0
    ascale2: float = 1.0
    fscale2: float = 1.0
    title: str = ""


PropSprBdamp = PropType27


# ============================================================================
# M183 Materials: LAW50, LAW57, LAW87, LAW95, LAW163, LAW169
# ============================================================================

@dataclass
class MatLaw50:
    """/MAT/LAW50 or /MAT/VISC_HONEY (M183): Rate-dependent honeycomb material."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gbc: float = 0.0
    gca: float = 0.0
    asrate: float = 0.0
    gflag: int = 0
    eps_max11: float = 0.0
    eps_max22: float = 0.0
    eps_max33: float = 0.0
    yfun11: list[int] = field(default_factory=list)
    sfac11: list[float] = field(default_factory=list)
    eps11: list[float] = field(default_factory=list)
    yfun22: list[int] = field(default_factory=list)
    sfac22: list[float] = field(default_factory=list)
    eps22: list[float] = field(default_factory=list)
    yfun33: list[int] = field(default_factory=list)
    sfac33: list[float] = field(default_factory=list)
    eps33: list[float] = field(default_factory=list)
    vflag: int = 0
    eps_max12: float = 0.0
    eps_max23: float = 0.0
    eps_max31: float = 0.0
    yfun12: list[int] = field(default_factory=list)
    sfac12: list[float] = field(default_factory=list)
    eps12: list[float] = field(default_factory=list)
    yfun23: list[int] = field(default_factory=list)
    sfac23: list[float] = field(default_factory=list)
    eps23: list[float] = field(default_factory=list)
    yfun31: list[int] = field(default_factory=list)
    sfac31: list[float] = field(default_factory=list)
    eps31: list[float] = field(default_factory=list)
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def e11(self) -> float:
        return self.ea

    @property
    def e22(self) -> float:
        return self.eb

    @property
    def e33(self) -> float:
        return self.ec

    @property
    def g12(self) -> float:
        return self.gab

    @property
    def g23(self) -> float:
        return self.gbc

    @property
    def g31(self) -> float:
        return self.gca


MatViscHoney = MatLaw50
MatHypFoam = MatLaw50


@dataclass
class MatLaw57Curve:
    fct_id: int = 0
    fscale: float = 1.0
    eps: float = 0.0


@dataclass
class MatLaw57:
    """/MAT/LAW57 or /MAT/BARLAT3 (M183): Barlat 3-parameter anisotropic plasticity."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    r00: float = 0.0
    r45: float = 0.0
    r90: float = 0.0
    chard: float = 0.0
    m: float = 2.0
    epsp_max: float = 0.0
    eps_t1: float = 0.0
    eps_t2: float = 0.0
    curves: list[MatLaw57Curve] = field(default_factory=list)
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho


MatBarlat3 = MatLaw57


@dataclass
class MatLaw87Curve:
    fct_id: int = 0
    fscale: float = 1.0
    epsp: float = 0.0


@dataclass
class MatLaw87:
    """/MAT/LAW87 or /MAT/BARLAT_YLD2000 (M183): Barlat Yld2000 anisotropic plasticity."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iflag: int = 0
    vp: int = 0
    strain1: float = 0.0
    exp1: float = 0.0
    ifit: int = 0
    alpha: list[float] = field(default_factory=lambda: [1.0]*8)
    sigma_00: float = 0.0
    sigma_45: float = 0.0
    sigma_90: float = 0.0
    sigma_b: float = 0.0
    r_00: float = 1.0
    r_45: float = 1.0
    r_90: float = 1.0
    r_b: float = 1.0
    chard: float = 0.0
    ikin: int = 0
    exp_a: float = 6.0
    alpha_vol: float = 1.0
    n_hard: float = 0.0
    fcut: float = 0.0
    fsmooth: int = 0
    nrate: int = 0
    curves: list[MatLaw87Curve] = field(default_factory=list)
    # Swift-Voce parameters (iflag=1)
    aswift: float = 0.0
    eps0: float = 0.0
    qvoce: float = 0.0
    beta: float = 0.0
    k0: float = 0.0
    # Tabulated (iflag=3)
    tab_id0: int = 0
    fscale0: float = 1.0
    epsd0: float = 0.0
    tab_id45: int = 0
    fscale45: float = 1.0
    epsd45: float = 0.0
    tab_id90: int = 0
    fscale90: float = 1.0
    epsd90: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def alphas(self) -> list[float]:
        return self.alpha

    @property
    def a_exp(self) -> int:
        return int(self.exp_a)

    @property
    def a_swift(self) -> float:
        return self.aswift

    @property
    def q_voce(self) -> float:
        return self.qvoce

    @property
    def vflag(self) -> int:
        return self.vp


MatBarlatYld2000 = MatLaw87
MatBarlat2000 = MatLaw87


@dataclass
class MatLaw95:
    """/MAT/LAW95 or /MAT/BERGSTROM_BOYCE (M183): Bergstrom-Boyce hyperelastic viscoplastic polymer model."""
    id: int
    rho: float = 0.0
    c10: float = 0.0
    c01: float = 0.0
    c20: float = 0.0
    c11: float = 0.0
    c02: float = 0.0
    c30: float = 0.0
    c21: float = 0.0
    c12: float = 0.0
    c03: float = 0.0
    sb: float = 1.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    nu: float = 0.49
    iform: int = 0
    a: float = 0.0
    c: float = 0.0
    m: float = 1.0
    ksi: float = 0.0
    tau_ref: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho


MatBergstromBoyce = MatLaw95
MatHypViscPlas = MatLaw95
MatFoamTab = MatLaw95


@dataclass
class MatLaw163:
    """/MAT/LAW163 or /MAT/CRUSHABLE_FOAM (M183): Crushable foam material model."""
    id: int
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    tsc: float = 0.0
    damp: float = 0.0
    ncycle: int = 0
    tab_id: int = 0
    epsd_ref: float = 0.0
    fscale: float = 1.0
    srclmt: float = 0.0
    nrs: int = 0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho


MatCrushableFoam = MatLaw163
MatCrushFoam = MatLaw163


@dataclass
class MatLaw169:
    """/MAT/LAW169 or /MAT/ARUP_ADHESIVE (M183): 3D cohesive adhesive material model."""
    id: int
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sht_sl: float = 0.0
    tenmax: float = 0.0
    gcten: float = 0.0
    shrmax: float = 0.0
    gcshr: float = 0.0
    pwrt: int = 1
    pwrs: int = 1
    shrp: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def pr(self) -> float:
        return self.nu


MatArupAdhesive = MatLaw169
MatCohTab3D = MatLaw169
MatCoh3D = MatLaw169


# =========================================================================
# M184: Steinberg-Guinan Plasticity, SAMP Plasticity, Sandwich Shell,
#       Fabric Shell, Composite Stack & Crushing Spring Suite
# =========================================================================

@dataclass
class MatLaw49:
    """/MAT/LAW49 or /MAT/STEINB (M184): Steinberg-Guinan high-pressure plasticity model."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    e0: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    beta: float = 0.0
    n: float = 0.0
    eps_max: float = 0.0
    sigma_max: float = 0.0
    t0: float = 0.0
    tmelt: float = 0.0
    rhoc_p: float = 0.0
    pmin: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    h: float = 0.0
    f: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def e(self) -> float:
        return self.e0

    @property
    def sigma_0(self) -> float:
        return self.sigy

    @property
    def hard(self) -> float:
        return self.n


MatSteinb = MatLaw49
MatSteinberg = MatLaw49
MatSteinbergGuinan = MatLaw49


@dataclass
class MatLaw76:
    """/MAT/LAW76 or /MAT/SAMP (M184): Semi-Analytical Model for Plastics (SAMP)."""
    id: int
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    fun_d1: int = 0
    fun_d2: int = 0
    fun_d3: int = 0
    fun_d4: int = 0
    fscale11: float = 1.0
    fscale22: float = 1.0
    fscale33: float = 1.0
    fscale12: float = 1.0
    facx: float = 1.0
    mat_nut: float = 0.0
    fun_b5: int = 0
    mat_pscale: float = 1.0
    israte: int = 0
    asrate: float = 0.0
    epsilon_f: float = 0.0
    epsilon_0: float = 0.0
    dc: float = 0.0
    fun_a1: int = 0
    fun_a2: int = 0
    fun_a3: int = 0
    scale: float = 1.0
    iform: int = 0
    iflag: int = 0
    gflag: int = 0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.rho

    @property
    def nu_p(self) -> float:
        return self.mat_nut


MatSamp = MatLaw76
MatPlasSamp = MatLaw76
MatSampPlas = MatLaw76


@dataclass
class PropSandwLayer:
    """Layer definition for /PROP/TYPE11 (SH_SANDW)."""
    phi: float = 0.0
    thick: float = 0.0
    z: float = 0.0
    mat_id: int = 0
    f_weight: float = 0.0


@dataclass
class PropType11:
    """/PROP/TYPE11 or /PROP/SH_SANDW (M184): Sandwich shell property."""
    id: int
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    p_thick_fail: float = 0.0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    nip: int = 0
    istrain: int = 0
    thick: float = 0.0
    ashear: float = 0.0
    ithick: int = 0
    iplas: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_csid: int = 0
    iorth: int = 0
    ipos: int = 0
    ip: int = 0
    layers: list[PropSandwLayer] = field(default_factory=list)
    title: str = ""


PropShSandw = PropType11
PropSandwich = PropType11


@dataclass
class PropFabricLayer:
    """Layer definition for /PROP/TYPE16 (SH_FABR)."""
    phi: float = 0.0
    alpha: float = 0.0
    thick: float = 0.0
    z: float = 0.0
    mat_id: int = 0


@dataclass
class PropType16:
    """/PROP/TYPE16 or /PROP/SH_FABR (M184): Fabric shell property."""
    id: int
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    p_thick_fail: float = 0.0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    nip: int = 0
    istrain: int = 0
    thick: float = 0.0
    ashear: float = 0.0
    ithick: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    ipos: int = 0
    ip: int = 0
    layers: list[PropFabricLayer] = field(default_factory=list)
    title: str = ""


PropShFabr = PropType16
PropFabricShell = PropType16
PropFabric = PropType16


@dataclass
class PropType17:
    """/PROP/TYPE17 or /PROP/STACK (M184): Composite ply stack property."""
    id: int
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    plyxfem: int = 0
    z0: float = 0.0
    vinterply: float = 0.0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    thick: float = 0.0
    ashear: float = 0.0
    ithick: int = 0
    iplas: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0
    refplane: int = 0
    title: str = ""


PropStack = PropType17
PropCompStack = PropType17


@dataclass
class PropType44:
    """/PROP/TYPE44 or /PROP/SPR_CRUS (M184): Crushing frame spring property."""
    id: int
    mass: float = 0.0
    inertia: float = 0.0
    stiff1: float = 0.0
    skew_csid: int = 0
    icoupling: int = 0
    ifiltr: int = 0
    k11: float = 0.0
    k44: float = 0.0
    k55: float = 0.0
    k66: float = 0.0
    idamp: int = 0
    k5b: float = 0.0
    k6c: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    fun_a2: int = 0
    fscale11: float = 1.0
    fun_b2: int = 0
    fun_a3: int = 0
    fun_b3: int = 0
    fun_a4: int = 0
    fscale22: float = 1.0
    fun_b4: int = 0
    fun_a5: int = 0
    fun_b5: int = 0
    fun_a6: int = 0
    fscale33: float = 1.0
    fun_b6: int = 0
    fun_c1: int = 0
    fun_c2: int = 0
    fun_c3: int = 0
    fscale12: float = 1.0
    fun_c4: int = 0
    fun_c5: int = 0
    fun_c6: int = 0
    fun_d1: int = 0
    fscale23: float = 1.0
    fun_d2: int = 0
    fun_d3: int = 0
    fun_d4: int = 0
    fun_d5: int = 0
    fscale13: float = 1.0
    strain1: float = 0.0
    strain2: float = 0.0
    strain3: float = 0.0
    strain4: float = 0.0
    strain5: float = 0.0
    strain6: float = 0.0
    strain7: float = 0.0
    fct_d_x: int = 0
    dscale_x: float = 0.0
    f_x: float = 0.0
    fct_d_y: int = 0
    dscale_y: float = 0.0
    f_y: float = 0.0
    fct_d_z: int = 0
    dscale_z: float = 0.0
    f_z: float = 0.0
    fct_d_xx: int = 0
    dscale_xx: float = 0.0
    f_xx: float = 0.0
    fct_d_yy: int = 0
    dscale_yy: float = 0.0
    f_yy: float = 0.0
    fct_d_zz: int = 0
    dscale_zz: float = 0.0
    f_zz: float = 0.0
    title: str = ""


PropSprCrus = PropType44
PropCrushSpring = PropType44
PropSpringCrush = PropType44


# ============================================================================
# M185: LAW60 (PLAS_T3), LAW63 (HANSEL), LAW48 (ZHAO), LAW26 (SESAM),
#       PROP TYPE12 (SPR_PUL), PROP TYPE15 (POROUS), PROP TYPE28 (NSTRAND)
# ============================================================================

@dataclass
class MatLaw60:
    """``/MAT/LAW60`` or ``/MAT/PLAS_T3``: Tabulated temperature/rate plasticity."""
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    eps_p_max: float = 0.0
    eps_t1: float = 0.0
    eps_t2: float = 0.0
    nfunc: int = 5
    fsmooth: int = 0
    chard: float = 0.0
    fcut: float = 0.0
    xr_fun: int = 0
    fpscale: float = 1.0
    fun_ids: list[int] = field(default_factory=list)
    fscales: list[float] = field(default_factory=list)
    eps_rates: list[float] = field(default_factory=list)
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @property
    def hard(self) -> float:
        return self.chard


MatPlasT3 = MatLaw60
MatPlastT3 = MatLaw60
MatMaxwell = MatLaw60


@dataclass
class MatLaw63:
    """``/MAT/LAW63`` or ``/MAT/HANSEL``: Hänsel transformation plasticity."""
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    cp: float = 0.0
    a: float = 0.0
    b: float = 0.0
    q: float = 0.0
    c: float = 0.0
    d: float = 0.0
    p: float = 0.0
    ahs: float = 0.0
    bhs: float = 0.0
    m: float = 0.0
    n: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    dh: float = 0.0
    vm0: float = 0.0
    eps0: float = 0.0
    t0: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @property
    def mat_t0(self) -> float:
        return self.t0


MatHansel = MatLaw63
MatPlasHansel = MatLaw63
MatTransfoPlas = MatLaw63


@dataclass
class MatLaw48:
    """``/MAT/LAW48`` or ``/MAT/ZHAO``: Zhao strain-rate hardening plasticity."""
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    chard: float = 0.0
    sig_max: float = 0.0
    c: float = 0.0
    d: float = 0.0
    m: float = 0.0
    e1: float = 0.0
    k: float = 0.0
    eps_rate_0: float = 0.0
    fcut: float = 0.0
    eps_max: float = 0.0
    eps_t1: float = 0.0
    eps_t2: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @property
    def sigy(self) -> float:
        return self.a

    @property
    def hard(self) -> float:
        return self.n


MatZhao = MatLaw48
MatPlasZhao = MatLaw48


@dataclass
class MatLaw26:
    """``/MAT/LAW26`` or ``/MAT/SESAM``: SESAME equation of state & hydrodynamic constitutive model."""
    id: int = 0
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    eps_max: float = 0.0
    sig_max: float = 0.0
    e0: float = 0.0
    sesam301: str = ""
    c: float = 0.0
    eps0: float = 0.0
    m: float = 0.0
    tmelt: float = 0.0
    tmax: float = 0.0
    title: str = ""

    @property
    def rho0(self) -> float:
        return self.refer_rho if self.refer_rho != 0.0 else self.rho

    @property
    def sigy(self) -> float:
        return self.a

    @property
    def hard(self) -> float:
        return self.n


MatSesam = MatLaw26
MatSesame = MatLaw26


@dataclass
class PropType12:
    """``/PROP/TYPE12`` or ``/PROP/SPR_PUL``: Pulley spring / sliding cable property."""
    id: int = 0
    mass: float = 0.0
    isensor: int = 0
    isflag: int = 0
    ileng: int = 0
    fric: float = 0.0
    stiff1: float = 0.0
    damp1: float = 0.0
    acoeft1: float = 1.0
    bcoeft1: float = 0.0
    dcoeft1: float = 1.0
    fun_a1: int = 0
    hflag1: int = 0
    fun_b1: int = 0
    min_rup1: float = -1.0e30
    max_rup1: float = 1.0e30
    prop_x_f: float = 1.0
    prop_x_e: float = 0.0
    scale1: float = 1.0
    title: str = ""

    @property
    def k(self) -> float:
        return self.stiff1

    @property
    def c(self) -> float:
        return self.damp1

    @property
    def sensor_id(self) -> int:
        return self.isensor


PropSprPul = PropType12
PropPulley = PropType12


@dataclass
class PropType15:
    """``/PROP/TYPE15`` or ``/PROP/POROUS``: Porous solid property."""
    id: int = 0
    qa: float = 0.0
    qb: float = 0.0
    h: float = 0.1
    poros: float = 1.0
    r1: float = 0.0
    r2: float = 0.0
    r3: float = 0.0
    skew_csid: int = 0
    ihon: int = 0
    itu: int = 0
    alpha: float = 0.1
    l_mix: float = 0.0
    irby: int = 0
    title: str = ""

    @property
    def porosity(self) -> float:
        return self.poros

    @property
    def skew_id(self) -> int:
        return self.skew_csid


PropPorous = PropType15
PropSolidPorous = PropType15


@dataclass
class PropStrandLayer:
    """Layer definition for /PROP/TYPE28 (NSTRAND)."""
    type_name: str = ""
    k_id: int = 0
    mu: float = 0.0


@dataclass
class PropType28:
    """``/PROP/TYPE28`` or ``/PROP/NSTRAND``: Multi-strand cable / wire rope property."""
    id: int = 0
    mass: float = 0.0
    k: float = 0.0
    c: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    strain1: float = -1.0e30
    strain2: float = 1.0e30
    mu1: float = 0.0
    mu2: float = 0.0
    layers: list[PropStrandLayer] = field(default_factory=list)
    title: str = ""


PropNstrand = PropType28
PropStrand = PropType28


# --- M186: Hydrodynamic Fluid, Boundary Layer, Foam-Air, Multi-Material, Barlat 3D, KJoint, Muscle, Stitch ---

@dataclass
class MatLaw6:
    """``/MAT/LAW6`` or ``/MAT/VISC_FLUID`` / ``/MAT/HYDRO`` / ``/MAT/K-EPS``: Hydrodynamic fluid with EOS & turbulence."""
    id: int = 0
    rho: float = 0.0
    rho_ref: float = 0.0
    nu: float = 0.0
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    pmin: float = 0.0
    psh: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    e0: float = 0.0
    r0k0: float = 0.0
    ssl: float = 0.0
    c_mu: float = 0.0
    sig_k: float = 0.0
    sig_eps: float = 0.0
    bulk_ratio: float = 0.0
    c1_e: float = 0.0
    c2_e: float = 0.0
    c3_e: float = 0.0
    kappa: float = 0.0
    e_wall: float = 0.0
    alpha: float = 0.0
    gsi_t: float = 0.0
    title: str = ""


MatViscFluid = MatLaw6
MatHydro = MatLaw6
MatHydroVisc = MatLaw6


@dataclass
class MatLaw11:
    """``/MAT/LAW11`` or ``/MAT/BOUND`` / ``/MAT/B-K-EPS``: Boundary fluid & k-epsilon turbulence model."""
    id: int = 0
    rho: float = 0.0
    rho_ref: float = 0.0
    itype: int = 1
    psh: float = 0.0
    scale: float = 1.0
    node1: int = 0
    gamma: float = 1.4
    k_cdi: float = 0.0
    h: float = 0.0
    c1: float = 0.0
    fun_a1: int = 0
    fun_a2: int = 0
    pscale: float = 1.0
    fun_a6: int = 0
    e0: float = 0.0
    xt_fun: int = 0
    yt_fun: int = 0
    title: str = ""


MatBound = MatLaw11
MatBkEps = MatLaw11


@dataclass
class MatLaw77Curve:
    """Loading/unloading curve entry for /MAT/LAW77 (FOAM_AIR)."""
    fct_id: int = 0
    strain_rate: float = 0.0
    scale: float = 1.0


@dataclass
class MatLaw77:
    """``/MAT/LAW77`` or ``/MAT/FOAM_AIR`` / ``/MAT/FOAM_HYST``: Foam with gas cavity and hysteresis unloading."""
    id: int = 0
    rho: float = 0.0
    rho_ref: float = 0.0
    e0: float = 0.0
    nu: float = 0.0
    emax: float = 0.0
    epsmax: float = 0.0
    fcut: float = 0.0
    fsmooth: int = 0
    nload: int = 0
    nunload: int = 0
    iflag: int = 0
    shape: float = 0.0
    hyst: float = 0.0
    load_curves: list[MatLaw77Curve] = field(default_factory=list)
    unload_curves: list[MatLaw77Curve] = field(default_factory=list)
    rho_gas: float = 0.0
    p0: float = 0.0
    gamma: float = 1.4
    poros: float = 1.0
    rho_ext: float = 0.0
    pext: float = 0.0
    iclos: int = 0
    inc_gas: int = 0
    title: str = ""


MatFoamAir = MatLaw77
MatFoamHyst = MatLaw77


@dataclass
class MatMultiFluidFraction:
    """Sub-material fraction entry for /MAT/LAW151 (MULTIFLUID)."""
    mat_id: int = 0
    vol_frac: float = 0.0


@dataclass
class MatLaw151:
    """``/MAT/LAW151`` or ``/MAT/MULTIFLUID`` / ``/MAT/MULTI_MAT``: Multi-material mixture law."""
    id: int = 0
    fractions: list[MatMultiFluidFraction] = field(default_factory=list)
    title: str = ""


MatMultiMat = MatLaw151
MatMultifluidMat = MatLaw151


@dataclass
class MatLaw187Rate:
    """Rate-dependent yield function entry for /MAT/LAW187 (BARLAT20003D)."""
    fct_id: int = 0
    scale: float = 1.0
    strain_rate: float = 0.0


@dataclass
class MatLaw187:
    """``/MAT/LAW187`` or ``/MAT/BARLAT20003D`` / ``/MAT/BARLAT_3D``: Barlat 2000 3D anisotropic plasticity."""
    id: int = 0
    rho: float = 0.0
    rho_ref: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iflag: int = 0
    vp: int = 0
    c: float = 0.0
    p_exp: float = 0.0
    alpha1: float = 1.0
    alpha2: float = 1.0
    alpha3: float = 1.0
    alpha4: float = 1.0
    alpha5: float = 1.0
    alpha6: float = 1.0
    alpha7: float = 1.0
    alpha8: float = 1.0
    alpha9: float = 1.0
    alpha10: float = 1.0
    alpha11: float = 1.0
    alpha12: float = 1.0
    a_exp: int = 8
    alpha_xy: float = 1.0
    n_exp: float = 0.0
    fcut: float = 0.0
    fsmooth: int = 0
    nrate: int = 0
    a_hard: float = 0.0
    eps0: float = 0.0
    q: float = 0.0
    b_hard: float = 0.0
    k0: float = 0.0
    rates: list[MatLaw187Rate] = field(default_factory=list)
    title: str = ""


MatBarlat20003D = MatLaw187
MatBarlat3D = MatLaw187
MatPlasBarlat3D = MatLaw187


@dataclass
class PropType33:
    """``/PROP/TYPE33`` or ``/PROP/KJOINT`` / ``/PROP/KINEMATIC_JOINT``: 6-DOF kinematic joint property."""
    id: int = 0
    joint_type: int = 1
    skew_flag: int = 0
    id_sk1: int = 0
    id_sk2: int = 0
    xk: float = 0.0
    cr: float = 0.0
    kn: float = 0.0
    krx: float = 0.0
    kry: float = 0.0
    krz: float = 0.0
    ktx: float = 0.0
    kty: float = 0.0
    ktz: float = 0.0
    xr_fun: int = 0
    yr_fun: int = 0
    zr_fun: int = 0
    xt_fun: int = 0
    yt_fun: int = 0
    zt_fun: int = 0
    crx: float = 0.0
    cry: float = 0.0
    crz: float = 0.0
    ctx: float = 0.0
    cty: float = 0.0
    ctz: float = 0.0
    crx_fun: int = 0
    cry_fun: int = 0
    crz_fun: int = 0
    ctx_fun: int = 0
    cty_fun: int = 0
    ctz_fun: int = 0
    title: str = ""


PropKjoint = PropType33
PropKinematicJoint = PropType33


@dataclass
class PropType46:
    """``/PROP/TYPE46`` or ``/PROP/SPR_MUSCLE`` / ``/PROP/MUSCLE``: Hill-type muscle spring property."""
    id: int = 0
    mass: float = 0.0
    stiff0: float = 0.0
    vel_max: float = 0.0
    nforce: float = 0.0
    stiff1: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    fun_c1: int = 0
    fun_d1: int = 0
    mat_imass: int = 0
    damp1: float = 0.0
    epsi: int = 0
    fscale11: float = 1.0
    fscale22: float = 1.0
    fscale21: float = 1.0
    fscale12: float = 1.0
    title: str = ""


PropSprMuscle = PropType46
PropMuscle = PropType46


@dataclass
class PropType35:
    """``/PROP/TYPE35`` or ``/PROP/STITCH`` / ``/PROP/SEW``: Stitch seam fastener property."""
    id: int = 0
    amas: float = 0.0
    elastif: float = 0.0
    xlim1: float = 0.0
    xk: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    fun_c1: int = 0
    fun_d1: int = 0
    damg: float = 0.0
    fdelay: float = 0.0
    title: str = ""


PropStitch = PropType35
PropSew = PropType35


# M187: Geotechnical, Hydrodynamic, Tabulated Plasticity & Advanced Joint/Interface Suite

@dataclass
class MatLaw3:
    """``/MAT/LAW3`` or ``/MAT/PLAS_BOST``: Elastoplastic material with Cowper-Symonds rate hardening."""
    id: int = 0
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sig_y: float = 0.0
    e_t: float = 0.0
    c: float = 0.0
    p: float = 0.0
    title: str = ""


MatPlasBost = MatLaw3


@dataclass
class MatLaw4:
    """``/MAT/LAW4`` or ``/MAT/HYD_JCOOK``: Hydrodynamic Johnson-Cook elastoplasticity with EOS."""
    id: int = 0
    rho_i: float = 0.0
    c0_eos: float = 0.0
    s_eos: float = 0.0
    gamma0: float = 0.0
    a_eos: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    c: float = 0.0
    eps_max: float = 0.0
    sig_max: float = 0.0
    t0: float = 0.0
    tm: float = 0.0
    m: float = 0.0
    cp: float = 0.0
    pmin: float = 0.0
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    title: str = ""


MatHydJcook = MatLaw4


@dataclass
class MatLaw5:
    """``/MAT/LAW5`` or ``/MAT/JCOOK_TAB``: Tabulated Johnson-Cook plasticity with scale functions."""
    id: int = 0
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    c: float = 0.0
    sig_max: float = 0.0
    fct_id1: int = 0
    fct_id2: int = 0
    fct_id3: int = 0
    fct_id4: int = 0
    fct_id5: int = 0
    title: str = ""


MatJcookTab = MatLaw5


@dataclass
class MatLaw10:
    """``/MAT/LAW10`` or ``/MAT/SOIL`` / ``/MAT/SOIL_CONC``: Soil and crushable concrete model."""
    id: int = 0
    rho0: float = 0.0
    g: float = 0.0
    k: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    p_cut: float = 0.0
    p_min: float = 0.0
    fct_id_p: int = 0
    title: str = ""


MatSoil = MatLaw10
MatSoilConc = MatLaw10


@dataclass
class MatLaw14:
    """``/MAT/LAW14``, ``/MAT/CAM_CLAY`` (M187), or ``/MAT/COMPSO`` (M189)."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    # Cam-Clay fields (M187)
    g: float = 0.0
    nu: float = 0.0
    m: float = 0.0
    lamda: float = 0.0
    kappa: float = 0.0
    e0: float = 0.0
    pc0: float = 0.0
    # Composite Solid fields (M189)
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    prab: float = 0.0
    prbc: float = 0.0
    prca: float = 0.0
    gab: float = 0.0
    gbc: float = 0.0
    gca: float = 0.0
    sigt1: float = 0.0
    sigt2: float = 0.0
    sigt3: float = 0.0
    damage: float = 0.0
    beta: float = 0.0
    hard: float = 0.0
    sig_max: float = 0.0
    sigyt1: float = 0.0
    sigyt2: float = 0.0
    sigyc1: float = 0.0
    sigyc2: float = 0.0
    sigt12: float = 0.0
    sigt23: float = 0.0
    sigc12: float = 0.0
    sigc23: float = 0.0
    alpha_fib: float = 0.0
    e_fib: float = 0.0
    src: float = 0.0
    srp: float = 0.0
    strflag: int = 0
    title: str = ""


MatCamClay = MatLaw14
MatCamclay = MatLaw14
MatCompso = MatLaw14
MatCompSol = MatLaw14


@dataclass
class MatLaw21:
    """``/MAT/LAW21`` or ``/MAT/DUCKHUB``: Drucker-Prager / Cap geological plasticity model."""
    id: int = 0
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    w: float = 0.0
    d: float = 0.0
    x0: float = 0.0
    title: str = ""


MatDuckhub = MatLaw21


@dataclass
class MatLaw32:
    """``/MAT/LAW32`` or ``/MAT/HILL_TAB``: Tabulated Hill orthotropic plasticity model."""
    id: int = 0
    rho0: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    e3: float = 0.0
    nu12: float = 0.0
    nu23: float = 0.0
    nu31: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    fct_id11: int = 0
    fct_id22: int = 0
    fct_id33: int = 0
    fct_id12: int = 0
    fct_id23: int = 0
    fct_id31: int = 0
    title: str = ""


MatHillTab = MatLaw32


@dataclass
class MatLaw37:
    """``/MAT/LAW37`` or ``/MAT/BIQUAD``: Biquadratic anisotropic yield criterion."""
    id: int = 0
    rho0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    c7: float = 0.0
    c8: float = 0.0
    p: float = 0.0
    q: float = 0.0
    title: str = ""


MatBiquad = MatLaw37


@dataclass
class PropType45:
    """``/PROP/TYPE45`` or ``/PROP/KJOINT2`` / ``/PROP/KINEMATIC_JOINT2``: 6-DOF Kinematic Joint Type 2."""
    id: int = 0
    joint_type: int = 1
    kn: float = 0.0
    scale: float = 1.0
    cr: float = 0.0
    isensor: int = 0
    skew1: int = 0
    skew2: int = 0
    ktx: float = 0.0
    kty: float = 0.0
    ktz: float = 0.0
    xt_fun: int = 0
    yt_fun: int = 0
    zt_fun: int = 0
    xn: float = 0.0
    yn: float = 0.0
    zn: float = 0.0
    xc: float = 0.0
    yc: float = 0.0
    zc: float = 0.0
    ctx: float = 0.0
    cty: float = 0.0
    ctz: float = 0.0
    ctx_fun: int = 0
    cty_fun: int = 0
    ctz_fun: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    title: str = ""


PropKjoint2 = PropType45
PropKinematicJoint2 = PropType45


@dataclass
class PropType36:
    """``/PROP/TYPE36`` or ``/PROP/PREDIT``: Progressive delamination interface property."""
    id: int = 0
    lutype: int = 0
    skew_csid: int = 0
    prop_id1: int = 0
    prop_id2: int = 0
    xk: float = 0.0
    mat_id: int = 0
    area: float = 0.0
    ixx: float = 0.0
    iyy: float = 0.0
    izz: float = 0.0
    ray: float = 0.0
    title: str = ""


PropPredit = PropType36


# ----------------------------------------------------------------------------
# M188: Composite, Honeycomb, Concrete Damage & Advanced Shell/Solid Props
# ----------------------------------------------------------------------------

@dataclass
class MatLaw12:
    """``/MAT/LAW12`` or ``/MAT/3PARBI`` / ``/MAT/3D_COMP`` / ``/MAT/RAGAB``: 3-parameter Drucker-Prager / 3D composite."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    e33: float = 0.0
    nu12: float = 0.0
    nu23: float = 0.0
    nu31: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    sig_t1: float = 0.0
    sig_t2: float = 0.0
    sig_t3: float = 0.0
    delta: float = 0.0
    b: float = 0.0
    n: float = 0.0
    fmax: float = 0.0
    sig_1yt: float = 0.0
    sig_2yt: float = 0.0
    sig_1yc: float = 0.0
    sig_2yc: float = 0.0
    sig_12yt: float = 0.0
    sig_12yc: float = 0.0
    sig_23yt: float = 0.0
    sig_23yc: float = 0.0
    sig_3yt: float = 0.0
    sig_3yc: float = 0.0
    sig_13yt: float = 0.0
    sig_13yc: float = 0.0
    alpha: float = 0.0
    efib: float = 0.0
    c: float = 0.0
    eps0: float = 0.0
    icc: int = 0
    title: str = ""


Mat3parbi = MatLaw12
Mat3dComp = MatLaw12
MatRagab = MatLaw12


@dataclass
class MatLaw13:
    """``/MAT/LAW13`` or ``/MAT/HONEYCOMB`` / ``/MAT/RIGID``: Honeycomb crush / rigid material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    title: str = ""


MatHoneycomb = MatLaw13


@dataclass
class MatLaw15:
    """``/MAT/LAW15`` or ``/MAT/CHANG`` / ``/MAT/CHANG_CHANG``: Chang-Chang composite failure model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    nu12: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    b: float = 0.0
    n: float = 0.0
    fmax: float = 0.0
    wpmax: float = 0.0
    wpref: float = 0.0
    ioff: int = 0
    sig_1yt: float = 0.0
    sig_2yt: float = 0.0
    sig_1yc: float = 0.0
    sig_2yc: float = 0.0
    alpha: float = 0.0
    sig_12yc: float = 0.0
    sig_12yt: float = 0.0
    c: float = 0.0
    eps_dot_0: float = 0.0
    icc: int = 0
    beta: float = 0.0
    tmax: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s12: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    title: str = ""


MatChang = MatLaw15
MatChangChang = MatLaw15


@dataclass
class MatLaw18:
    """``/MAT/LAW18`` or ``/MAT/CONCR_DRA`` / ``/MAT/DRAGON`` / ``/MAT/THERM``: Concrete damage / thermal model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    spheat: float = 0.0
    a: float = 0.0
    b: float = 0.0
    fct_idt: int = 0
    t0: float = 0.0
    scale: float = 0.0
    fct_idsph: int = 0
    fct_idas: int = 0
    fscalesph: float = 0.0
    fscalee: float = 0.0
    fscalek: float = 0.0
    title: str = ""


MatConcrDra = MatLaw18
MatDragon = MatLaw18
MatTherm = MatLaw18


@dataclass
class MatLaw22:
    """``/MAT/LAW22`` or ``/MAT/TSAI_WU`` / ``/MAT/DAMA``: Tsai-Wu anisotropic composite damage model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    beta: float = 0.0
    n: float = 0.0
    eps_max: float = 0.0
    sig_max: float = 0.0
    c: float = 0.0
    eps_dot_0: float = 0.0
    icc: int = 0
    eps_dam: float = 0.0
    e_t: float = 0.0
    title: str = ""


MatTsaiWu = MatLaw22
MatDama = MatLaw22


@dataclass
class MatLaw25:
    """``/MAT/LAW25`` or ``/MAT/COMP_PLAS`` / ``/MAT/COMPSH``: Composite anisotropic plasticity model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    nu12: float = 0.0
    iform: int = 0
    e33: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    eps_f1: float = 0.0
    eps_f2: float = 0.0
    eps_t1: float = 0.0
    eps_m1: float = 0.0
    eps_t2: float = 0.0
    eps_m2: float = 0.0
    dmax: float = 0.0
    wpmax: float = 0.0
    wpref: float = 0.0
    ioff: int = 0
    b: float = 0.0
    n: float = 0.0
    fmax: float = 0.0
    sig_1yt: float = 0.0
    sig_2yt: float = 0.0
    sig_1yc: float = 0.0
    sig_2yc: float = 0.0
    alpha: float = 0.0
    sig_12yc: float = 0.0
    sig_12yt: float = 0.0
    c: float = 0.0
    eps_rate_0: float = 0.0
    icc: int = 0
    title: str = ""


MatCompPlas = MatLaw25
MatCompsh = MatLaw25


@dataclass
class MatLaw28:
    """``/MAT/LAW28`` or ``/MAT/HONEYCOMB_SOL``: Solid honeycomb crush material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    e33: float = 0.0
    g12: float = 0.0
    g23: float = 0.0
    g31: float = 0.0
    fun_id11: int = 0
    fun_id22: int = 0
    fun_id33: int = 0
    iflag1: int = 0
    fscale11: float = 0.0
    fscale22: float = 0.0
    fscale33: float = 0.0
    eps_max11: float = 0.0
    eps_max22: float = 0.0
    eps_max33: float = 0.0
    fun_id12: int = 0
    fun_id23: int = 0
    fun_id31: int = 0
    iflag2: int = 0
    fscale12: float = 0.0
    fscale23: float = 0.0
    fscale31: float = 0.0
    eps_max12: float = 0.0
    eps_max23: float = 0.0
    eps_max31: float = 0.0
    title: str = ""

    @property
    def fun_a1(self) -> int:
        return self.fun_id11

    @property
    def fun_b1(self) -> int:
        return self.fun_id22

    @property
    def fun_a2(self) -> int:
        return self.fun_id33

    @property
    def gflag(self) -> int:
        return self.iflag1

    @property
    def epsr1(self) -> float:
        return self.eps_max11

    @property
    def epsr2(self) -> float:
        return self.eps_max22

    @property
    def epsr3(self) -> float:
        return self.eps_max33

    @property
    def fun_a3(self) -> int:
        return self.fun_id12

    @property
    def fun_b3(self) -> int:
        return self.fun_id23

    @property
    def fun_a4(self) -> int:
        return self.fun_id31

    @property
    def vflag(self) -> int:
        return self.iflag2

    @property
    def fscale13(self) -> float:
        return self.fscale31

    @property
    def epsr4(self) -> float:
        return self.eps_max12

    @property
    def epsr5(self) -> float:
        return self.eps_max23

    @property
    def epsr6(self) -> float:
        return self.eps_max31


MatHoneycombSol = MatLaw28


@dataclass
class PropType9:
    """``/PROP/TYPE9`` or ``/PROP/SH_ORTH``: Orthotropic shell property."""
    id: int = 0
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    nip: int = 1
    istrain: int = 0
    thick: float = 0.0
    ashear: float = 0.833333
    ithick: int = 0
    iplas: int = 0
    vx: float = 1.0
    vy: float = 0.0
    vz: float = 0.0
    phi: float = 0.0
    title: str = ""


PropShOrth = PropType9
PropShellOrth = PropType9


@dataclass
class PropType10:
    """``/PROP/TYPE10`` or ``/PROP/SH_COMP``: Multi-layer composite shell property."""
    id: int = 0
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    nip: int = 1
    istrain: int = 0
    thick: float = 0.0
    ashear: float = 0.833333
    ithick: int = 0
    iplas: int = 0
    vx: float = 1.0
    vy: float = 0.0
    vz: float = 0.0
    phis: list[float] = field(default_factory=list)
    title: str = ""


PropShComp = PropType10
PropShellComp = PropType10


@dataclass
class PropType51:
    """``/PROP/TYPE51``, ``/PROP/P51`` or ``/PROP/SH_COH``: Cohesive shell / composite stack property."""
    id: int = 0
    ishell: int = 0
    ismstr: int = 0
    ish3n: int = 0
    idrill: int = 0
    pthk: float = 0.0
    zshift: float = 0.0
    p_thick_fail: float = 0.0
    z0: float = 0.0
    hm: float = 0.0
    hf: float = 0.0
    hr: float = 0.0
    dm: float = 0.0
    dn: float = 0.0
    istrain: int = 0
    ashear: float = 0.833333
    iint: int = 0
    ithick: int = 0
    failexp: float = 0.0
    fexp: float = 0.0
    vx: float = 1.0
    vy: float = 0.0
    vz: float = 0.0
    idsk: int = 0
    skew_id: int = 0
    iorth: int = 0
    ipos: int = 0
    irp: int = 0
    refplane: int = 0
    layers: List[Any] = field(default_factory=list)
    title: str = ""


PropShCoh = PropType51
PropShellCoh = PropType51


@dataclass
class PropType5:
    """``/PROP/TYPE5`` or ``/PROP/RIVET``: Rivet connection property."""
    id: int = 0
    mass: float = 0.0
    stiffness: float = 0.0
    fn_fail: float = 0.0
    ft_fail: float = 0.0
    nforce: float = 0.0
    tforce: float = 0.0
    length: float = 0.0
    wflag: int = 0
    imod: int = 1
    title: str = ""

    @property
    def fn(self) -> float:
        return self.nforce or self.fn_fail

    @property
    def ft(self) -> float:
        return self.tforce or self.ft_fail


PropRivet = PropType5


# -------------------------------------------------------------------------
# M189: Gurson, Gray Cast Iron, Composite Solid, Connector, Martensite Materials,
# Advanced Failure Criteria & Generalized Spring/Solid Properties
# -------------------------------------------------------------------------


@dataclass
class MatLaw52:
    """``/MAT/LAW52`` or ``/MAT/GURSON``: Gurson-Tvergaard-Needleman porous metal plasticity."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    iflag: int = 0
    fsmooth: int = 0
    fcut: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    c: float = 0.0
    pc: float = 0.0
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    s_n: float = 0.0
    eps_n: float = 0.0
    f_i: float = 0.0
    f_n: float = 0.0
    f_c: float = 0.0
    f_f: float = 0.0
    title: str = ""


MatGurson = MatLaw52
MatPlasGurs = MatLaw52


@dataclass
class MatLaw16:
    """``/MAT/LAW16`` or ``/MAT/GRAY``: Gray cast iron EOS and asymmetric plasticity model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    p0: float = 0.0
    c: float = 0.0
    s: float = 0.0
    gamma0: float = 0.0
    a: float = 0.0
    e0: float = 0.0
    v0: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sig_y: float = 0.0
    beta: float = 0.0
    hard: float = 0.0
    sig_max: float = 0.0
    eps_max: float = 0.0
    title: str = ""


MatGray = MatLaw16
MatCastIron = MatLaw16


@dataclass
class MatLaw59:
    """``/MAT/LAW59`` or ``/MAT/CONNECT``: Connector / fastener material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    g0: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    iflag: int = 0
    functions: List[Dict[str, Any]] = field(default_factory=list)
    title: str = ""


MatConnect = MatLaw59
MatConnector = MatLaw59


@dataclass
class MatLaw64:
    """``/MAT/LAW64``: Martensitic transformation plasticity model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    cp: float = 0.0
    d: float = 0.0
    n: float = 0.0
    md: float = 0.0
    v0: float = 0.0
    vmc: float = 0.0
    funct_id_0: int = 0
    funct_id_1: int = 0
    scale_0: float = 1.0
    scale_1: float = 1.0
    t_ini: float = 0.0
    title: str = ""


MatTransfoMart = MatLaw64
MatMartensite = MatLaw64


@dataclass
class FailWierzbicki:
    """``/FAIL/WIERZBICKI`` or ``/FAIL/MMC``: Modified Mohr-Coulomb ductile fracture model."""
    id: int = 0
    mat_id: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    m: float = 0.0
    n: float = 0.0
    ifail_sh: int = 0
    ifail_so: int = 0
    imoy: int = 0
    title: str = ""


FailMmc = FailWierzbicki


@dataclass
class FailWilkins:
    """``/FAIL/WILKINS``: Wilkins cumulative damage fracture model."""
    id: int = 0
    mat_id: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    plim: float = 0.0
    df: float = 0.0
    ifail_sh: int = 0
    ifail_so: int = 0
    title: str = ""


@dataclass
class PropType14:
    """``/PROP/TYPE14`` or ``/PROP/SOLID``: Generalized 3D solid property."""
    id: int = 0
    isolid: int = 14
    ismstr: int = 0
    icpre: int = 0
    inpts_r: int = 1
    inpts_s: int = 1
    inpts_t: int = 1
    i_rot: int = 0
    iframe: int = 0
    dn: float = 0.0
    qa: float = 1.1
    qb: float = 0.05
    h: float = 0.1
    deltat_min: float = 0.0
    istrain: int = 0
    qa_l: float = 0.0
    qb_l: float = 0.0
    h_l: float = 0.0
    iplas: int = 0
    icstr: int = 0
    title: str = ""


PropSolGene = PropType14
PropSolid = PropType14


@dataclass
class PropType8:
    """``/PROP/TYPE8`` or ``/PROP/SPR_GENE``: Generalized 6-DOF nonlinear spring property."""
    id: int = 0
    mass: float = 0.0
    inertia: float = 0.0
    skew_id: int = 0
    sensor_id: int = 0
    isflag: int = 0
    ifail: int = 0
    iequil: int = 0
    dofs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    title: str = ""


PropSprGene = PropType8
PropSpringGene = PropType8


@dataclass
class PropType25:
    """``/PROP/TYPE25`` or ``/PROP/SPR_AXI``: Axisymmetric nonlinear spring property."""
    id: int = 0
    mass: float = 0.0
    inertia: float = 0.0
    skew_id: int = 0
    sensor_id: int = 0
    isflag: int = 0
    ifail: int = 0
    ileng: int = 0
    ifail2: int = 0
    tension: Dict[str, Any] = field(default_factory=dict)
    shear: Dict[str, Any] = field(default_factory=dict)
    title: str = ""


PropSprAxi = PropType25
PropSpringAxi = PropType25


@dataclass
class PropType32:
    """``/PROP/TYPE32`` or ``/PROP/SPR_PRE``: Preloaded spring property."""
    id: int = 0
    mass: float = 0.0
    sensor_id: int = 0
    ilock: int = 0
    stiff0: float = 0.0
    f1: float = 0.0
    d1: float = 0.0
    e1: float = 0.0
    stiff1: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    scale_t: float = 1.0
    scale_d: float = 1.0
    scale_f: float = 1.0
    title: str = ""


PropSprPre = PropType32
PropSpringPre = PropType32


@dataclass
class PropType43:
    """``/PROP/TYPE43`` or ``/PROP/CONNECT``: Connector element property."""
    id: int = 0
    ismstr: int = 1
    thick: float = 0.0
    title: str = ""


PropConnect = PropType43
PropPropConnect = PropType43


@dataclass
class MatLaw68:
    """``/MAT/LAW68`` or ``/MAT/COSSER``: 3D Cosserat continuum material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e_11: float = 0.0
    e_22: float = 0.0
    e_33: float = 0.0
    g_12: float = 0.0
    g_23: float = 0.0
    g_31: float = 0.0
    fun_id11i: int = 0
    fun_id22i: int = 0
    fun_id33i: int = 0
    iflag1: int = 0
    fscale11i: float = 1.0
    fscale22i: float = 1.0
    fscale33i: float = 1.0
    eps_max11i: float = 0.0
    eps_max22i: float = 0.0
    eps_max33i: float = 0.0
    fun_id12i: int = 0
    fun_id23i: int = 0
    fun_id31i: int = 0
    iflag2: int = 0
    fscale12i: float = 1.0
    fscale23i: float = 1.0
    fscale31i: float = 1.0
    eps_max12i: float = 0.0
    eps_max23i: float = 0.0
    eps_max31i: float = 0.0
    fun_id21i: int = 0
    fun_id32i: int = 0
    fun_id13i: int = 0
    fscale21i: float = 1.0
    fscale32i: float = 1.0
    fscale13i: float = 1.0
    fun_id11r: int = 0
    fun_id22r: int = 0
    fun_id33r: int = 0
    fscale11r: float = 1.0
    fscale22r: float = 1.0
    fscale33r: float = 1.0
    eps_trans11r: float = 0.0
    eps_trans22r: float = 0.0
    eps_trans33r: float = 0.0
    fun_id12r: int = 0
    fun_id23r: int = 0
    fun_id31r: int = 0
    fscale12r: float = 1.0
    fscale23r: float = 1.0
    fscale31r: float = 1.0
    eps_trans12r: float = 0.0
    eps_trans23r: float = 0.0
    eps_trans31r: float = 0.0
    fun_id21r: int = 0
    fun_id32r: int = 0
    fun_id13r: int = 0
    fscale21r: float = 1.0
    fscale32r: float = 1.0
    fscale13r: float = 1.0
    title: str = ""


MatCosser = MatLaw68
MatCosserat = MatLaw68


@dataclass
class MatLaw72:
    """``/MAT/LAW72`` or ``/MAT/HILL_MMC``: Hill orthotropic plasticity with MMC ductile fracture."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sig0: float = 0.0
    eps0: float = 0.0
    n: float = 0.0
    f: float = 0.0
    g: float = 0.0
    h: float = 0.0
    big_n: float = 0.0
    l: float = 0.0
    m: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    mmc_m: float = 0.0
    dc: float = 0.0
    title: str = ""


MatHillMmc = MatLaw72


@dataclass
class MatLaw65:
    """``/MAT/LAW65`` or ``/MAT/ELASTOMER``: 3D Elastomer hyperelastic material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e0: float = 0.0
    nu: float = 0.0
    eps_max: float = 0.0
    nrate: int = 0
    fsmooth: int = 0
    fcut: float = 0.0
    rates: List[Dict[str, Any]] = field(default_factory=list)
    title: str = ""


MatElastomer = MatLaw65


@dataclass
class MatLaw58:
    """``/MAT/LAW58`` or ``/MAT/FABR_A``: Anisotropic fabric material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e1: float = 0.0
    b1: float = 0.0
    e2: float = 0.0
    b2: float = 0.0
    flex: float = 0.0
    g0: float = 0.0
    gt: float = 0.0
    alphat: float = 0.0
    sensor_id: int = 0
    df: float = 0.0
    ds: float = 0.0
    gfrot: float = 0.0
    zero_stress: float = 0.0
    n1: int = 0
    n2: int = 0
    s1: float = 0.0
    s2: float = 0.0
    title: str = ""


MatFabrA = MatLaw58


@dataclass
class MatLaw20:
    """``/MAT/LAW20`` or ``/MAT/BIMAT``: Bi-material mixture / layered material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    mat_id1: int = 0
    mat_id2: int = 0
    alpha1: float = 0.0
    alpha2: float = 0.0
    title: str = ""


MatBimat = MatLaw20


@dataclass
class MatLaw38:
    """``/MAT/LAW38`` or ``/MAT/VISC_TAB``: Tabulated viscoelastic polymer/foam model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu_t: float = 0.0
    nu_c: float = 0.0
    rv: float = 0.0
    iflag: int = 0
    itotal: int = 0
    beta: float = 0.0
    h: float = 0.0
    r_d: float = 0.0
    k_r: int = 0
    k_d: int = 0
    instant_mod_upd: float = 0.0
    kair: int = 0
    np: int = 0
    pscale: float = 0.0
    p0: float = 0.0
    rp: float = 0.0
    pmax: float = 0.0
    phi: float = 0.0
    ful: int = 0
    alpha_unload: float = 0.0
    eps_unload: float = 0.0
    a: float = 0.0
    b: float = 0.0
    m_func: int = 0
    cutoff: float = 0.0
    iinsta: int = 0
    e_final: float = 0.0
    epsi_final: float = 0.0
    lamb: float = 0.0
    visc: float = 0.0
    tol: float = 0.0
    title: str = ""


MatViscTab = MatLaw38


@dataclass
class MatLaw29:
    """``/MAT/LAW29`` or ``/MAT/FEM``: User/FEM material model interface."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    version: str = ""
    frelim: float = 0.0
    dtmin: float = 0.0
    nf: int = 0
    velsc: float = 0.0
    rstrat: float = 0.0
    rtemp: float = 0.0
    encrypt: int = 0
    param_1: int = 0
    el_young: float = 0.0
    el_poiss: float = 0.0
    el_bulkm: float = 0.0
    el_shear: float = 0.0
    el_ortho: int = 0
    el_shrco: float = 0.0
    param_2: int = 0
    param_3: int = 0
    pl_harde: int = 0
    pl_ortho: int = 0
    pl_iskin: int = 0
    pl_asymm: int = 0
    pl_waist: int = 0
    pl_biaxf: int = 0
    pl_compr: int = 0
    pl_damag: int = 0
    nf_curve: int = 0
    nf_ortho: int = 0
    nf_depen: int = 0
    sf_curve: int = 0
    sf_param: int = 0
    sf_postc: int = 0
    param_4: int = 0
    param_5: int = 0
    cr_harde: int = 0
    cr_ortho: int = 0
    cr_iskin: int = 0
    cr_postc: int = 0
    cr_param: int = 0
    cr_check: int = 0
    param_6: int = 0
    mf_init: int = 0
    title: str = ""


MatFem = MatLaw29
Mat29Fem = MatLaw29


@dataclass
class MatLaw34:
    """``/MAT/LAW34`` or ``/MAT/BOLTZMAN``: Boltzmann linear viscoelastic relaxation model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    k: float = 0.0
    g0: float = 0.0
    gl: float = 0.0
    beta: float = 0.0
    p0: float = 0.0
    phi: float = 0.0
    gamma0: float = 0.0
    title: str = ""


MatBoltzman = MatLaw34
MatBoltzmann = MatLaw34


@dataclass
class MatLaw23:
    """``/MAT/LAW23`` or ``/MAT/PLAS_DAMA``: Lemaitre ductile damage elastoplastic model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    eps_max: float = 0.0
    sig_max: float = 0.0
    c: float = 0.0
    eps_0: float = 0.0
    icc: int = 0
    eps_dam: float = 0.0
    e_t: float = 0.0
    title: str = ""


MatPlasDama = MatLaw23


@dataclass
class MatLaw78:
    """``/MAT/LAW78``: Rate-dependent elastoplastic constitutive law 78."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    eps_max: float = 0.0
    sig_max: float = 0.0
    y: float = 0.0
    b: float = 0.0
    c: float = 0.0
    h: float = 0.0
    b0: float = 0.0
    m: float = 0.0
    rsat: float = 0.0
    ea: float = 0.0
    ce: float = 0.0
    title: str = ""


@dataclass
class FailHashin:
    """``/FAIL/HASHIN``: Hashin 3D composite failure model."""
    mat_id: int = 0
    iform: int = 0
    ifail_sh: int = 0
    ifail_so: int = 0
    sigma_1t: float = 0.0
    sigma_2t: float = 0.0
    sigma_3t: float = 0.0
    sigma_1c: float = 0.0
    sigma_2c: float = 0.0
    sigma_c: float = 0.0
    sigma_12f: float = 0.0
    sigma_12m: float = 0.0
    sigma_23m: float = 0.0
    sigma_13m: float = 0.0
    phi: float = 0.0
    sdel: float = 0.0
    tau_max: float = 0.0
    title: str = ""


@dataclass
class FailTensstrain:
    """``/FAIL/TENSSTRAIN`` or ``/FAIL/TENSTRAIN``: Tensile strain failure model."""
    mat_id: int = 0
    eps_t1: float = 0.0
    eps_t2: float = 0.0
    eps_m1: float = 0.0
    fct_id: int = 0
    fscale: float = 1.0
    ifail_sh: int = 1
    ifail_so: int = 1
    d_adv: float = 0.0
    p_thickfail: float = 0.0
    title: str = ""


FailTenstrain = FailTensstrain


@dataclass
class FailEnergy:
    """``/FAIL/ENERGY``: Specific internal energy failure criterion."""
    mat_id: int = 0
    e1: float = 0.0
    e2: float = 0.0
    fct_id: int = 0
    title: str = ""


@dataclass
class FailUser:
    """``/FAIL/USER``: User-defined material failure model."""
    mat_id: int = 0
    user_type: str = ""
    cards: List[str] = field(default_factory=list)
    title: str = ""


@dataclass
class PropType34:
    """``/PROP/TYPE34`` or ``/PROP/SPH``: SPH particle property."""
    id: int = 0
    mass: float = 0.0
    h0: float = 0.0
    d0: float = 0.0
    qa: float = 2.0
    qb: float = 1.0
    alpha1: float = 0.0
    order: int = 0
    h: float = 0.0
    title: str = ""


PropSph = PropType34
PropPropSph = PropType34


@dataclass
class PropType29:
    """``/PROP/TYPE29``: User-defined property Type 29."""
    id: int = 0
    cards: List[str] = field(default_factory=list)
    title: str = ""


@dataclass
class PropType30:
    """``/PROP/TYPE30``: User-defined property Type 30."""
    id: int = 0
    cards: List[str] = field(default_factory=list)
    title: str = ""


@dataclass
class PropType31:
    """``/PROP/TYPE31``: User-defined property Type 31."""
    id: int = 0
    cards: List[str] = field(default_factory=list)
    title: str = ""


# ==============================================================================
# Milestone M191: Advanced Materials, Multi-axial & Visual Failures, Preloads & BCs
# ==============================================================================

@dataclass
class MatLaw100:
    """``/MAT/LAW100`` or ``/MAT/SPOTWELD``: Spotweld / structural adhesive material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    flag_he: int = 0
    flag_cr: int = 0
    c10: float = 0.0
    c01: float = 0.0
    c20: float = 0.0
    c11: float = 0.0
    c02: float = 0.0
    c30: float = 0.0
    c21: float = 0.0
    c12: float = 0.0
    c03: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    mue1: float = 0.0
    d: float = 0.0
    lambda_m: float = 0.0
    itype: int = 0
    fct_id_ab: int = 0
    nu: float = 0.0
    fct_id_sm: int = 0
    fct_id_bm: int = 0
    fscale_sm: float = 1.0
    fscale_bm: float = 1.0
    a_pl: float = 0.0
    sigma_pl: float = 0.0
    f_pl: float = 0.0
    epsilon_f: float = 0.0
    n_pl: int = 0
    title: str = ""


MatSpotweld = MatLaw100
MatStructuralAdhesive = MatLaw100


@dataclass
class MatLaw97:
    """``/MAT/LAW97`` or ``/MAT/EXPLOSIVE_JWLS``: High-explosive detonation equation of state with JWLS parameters."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    p0: float = 0.0
    psh: float = 0.0
    ibfrac: int = 0
    d: float = 0.0
    pcj: float = 0.0
    e0: float = 0.0
    omega: float = 0.0
    c: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0
    a5: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    r3: float = 0.0
    r4: float = 0.0
    r5: float = 0.0
    title: str = ""


MatExplosiveJwls = MatLaw97
MatJwls = MatLaw97


@dataclass
class MatLaw71:
    """``/MAT/LAW71`` or ``/MAT/SUPER_ELAS``: Nitinol / Superelastic shape memory alloy model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    e_mart: float = 0.0
    sig_sas: float = 0.0
    sig_fas: float = 0.0
    sig_ssa: float = 0.0
    sig_fsa: float = 0.0
    alpha: float = 0.0
    epsl: float = 0.0
    cas: float = 0.0
    csa: float = 0.0
    tsas: float = 0.0
    tfas: float = 0.0
    tssa: float = 0.0
    tfsa: float = 0.0
    cp: float = 0.0
    tini: float = 0.0
    title: str = ""


MatSuperElas = MatLaw71
MatNitinol = MatLaw71


@dataclass
class MatLaw73:
    """``/MAT/LAW73`` or ``/MAT/THERM_HILL``: Thermal Hill orthotropic material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    chard: float = 0.0
    eps_max: float = 0.0
    epst1: float = 0.0
    epst2: float = 0.0
    fun_a1: int = 0
    fscale: float = 1.0
    pscale: float = 1.0
    t_initial: float = 0.0
    spheat: float = 0.0
    iyield: int = 0
    yr_fun: int = 0
    efib: float = 0.0
    c: float = 0.0
    title: str = ""


MatThermHill = MatLaw73
MatHillTherm = MatLaw73


@dataclass
class MatLaw84:
    """``/MAT/LAW84`` or ``/MAT/SWIFT_VOCE``: Swift-Voce hardening plastic material with thermal coupling."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    fcut: float = 0.0
    cap_end: float = 0.0
    pc: float = 0.0
    pr: float = 0.0
    t0: float = 0.0
    c2_t: float = 0.0
    a2: float = 0.0
    c1_c: float = 0.0
    vol: float = 0.0
    nut: float = 0.0
    fscale11: float = 1.0
    fscale22: float = 1.0
    fscale33: float = 1.0
    fscale12: float = 1.0
    fscale23: float = 1.0
    scale1: float = 0.0
    scale2: float = 0.0
    scale3: float = 0.0
    scale4: float = 0.0
    scale5: float = 0.0
    title: str = ""


MatSwiftVoce = MatLaw84
MatPlasSwiftVoce = MatLaw84


@dataclass
class MatLaw93:
    """``/MAT/LAW93`` or ``/MAT/ORTH_HILL``: 3D Orthotropic Hill plasticity model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    e33: float = 0.0
    g12: float = 0.0
    nu12: float = 0.0
    g13: float = 0.0
    g23: float = 0.0
    nu13: float = 0.0
    nu23: float = 0.0
    nl: int = 0
    sigma_y: float = 0.0
    qr1: float = 0.0
    cr1: float = 0.0
    qr2: float = 0.0
    cr2: float = 0.0
    r11: float = 1.0
    r22: float = 1.0
    r12: float = 1.0
    r33: float = 1.0
    r13: float = 1.0
    r23: float = 1.0
    fcut: float = 0.0
    vp: int = 0
    curves: List[Dict[str, Any]] = field(default_factory=list)
    title: str = ""


MatOrthHill = MatLaw93


@dataclass
class MatLaw133:
    """``/MAT/LAW133`` or ``/MAT/GRANULAR``: Granular material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    nu: float = 0.0
    pmin: float = 0.0
    fct_id_g: int = 0
    fscale_g: float = 1.0
    fct_id_y: int = 0
    fscale_y: float = 1.0
    title: str = ""


MatGranular = MatLaw133


@dataclass
class MatLaw101:
    """``/MAT/LAW101`` or ``/MAT/PLAS_POLY``: Polymer viscoplasticity model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    alpha1: float = 0.0
    nu: float = 0.0
    ve1: float = 0.0
    ve2: float = 0.0
    epsilonref: float = 0.0
    gamma0: float = 0.0
    alpha_p: float = 0.0
    deltah: float = 0.0
    vol: float = 0.0
    m: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    alphak1: float = 0.0
    alphak2: float = 0.0
    hard: float = 0.0
    zeta1i: float = 0.0
    c5: float = 0.0
    c6: float = 0.0
    c7: float = 0.0
    c8: float = 0.0
    c9: float = 0.0
    c10: float = 0.0
    title: str = ""


MatPlasPoly = MatLaw101


@dataclass
class MatLaw43:
    """``/MAT/LAW43`` or ``/MAT/HILL_TAB``: Tabulated Hill orthotropic material model."""
    id: int = 0
    rho0: float = 0.0
    rhor: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    yr_fun: int = 0
    efib: float = 0.0
    c: float = 0.0
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0
    chard: float = 0.0
    iyield: int = 0
    eps: float = 0.0
    epst1: float = 0.0
    epst2: float = 0.0
    num_curves: int = 0
    fsmooth: int = 0
    fcut: float = 0.0
    curves: List[Dict[str, Any]] = field(default_factory=list)
    title: str = ""


MatHillTab = MatLaw43


@dataclass
class FailLemaitre:
    """``/FAIL/LEMAITRE``: Lemaitre continuum ductile damage failure model."""
    mat_id: int = 0
    eps_d: float = 0.0
    s_d: float = 0.0
    dc: float = 0.0
    failip: int = 0
    p_thickfail: float = 0.0
    fail_id: int = 0
    title: str = ""


FailTabulated2 = FailTab2


@dataclass
class FailAlter:
    """``/FAIL/ALTER``: Alter glass/laminate crack propagation failure model."""
    mat_id: int = 0
    exp_n: float = 0.0
    v0: float = 0.0
    vc: float = 0.0
    ema: int = 0
    irate: int = 0
    iside: int = 0
    mode: int = 0
    cr_foil: float = 0.0
    cr_air: float = 0.0
    cr_core: float = 0.0
    cr_edge: float = 0.0
    kic: float = 0.0
    kth: float = 0.0
    rlen: float = 0.0
    tdel: float = 0.0
    kres1: float = 0.0
    kres2: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailVisual:
    """``/FAIL/VISUAL``: Visual failure indicator model."""
    mat_id: int = 0
    vtype: int = 0
    c_min: float = 0.0
    c_max: float = 0.0
    alpha_exp: float = 0.0
    f_cutoff: float = 0.0
    f_flag: int = 0
    strdef: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailOrthstrain:
    """``/FAIL/ORTHSTRAIN``: Directional orthotropic strain failure criterion."""
    mat_id: int = 0
    pthk: float = 0.0
    eps_dot_ref: float = 0.0
    fcut: float = 0.0
    fct_idel: int = 0
    fscale_el: float = 1.0
    ei_ref: float = 0.0
    strdef: int = 0
    eps_11tf: float = 0.0
    eps_11tm: float = 0.0
    fct_id_11t: int = 0
    eps_11cf: float = 0.0
    eps_11cm: float = 0.0
    fct_id_11c: int = 0
    eps_22tf: float = 0.0
    eps_22tm: float = 0.0
    fct_id_22t: int = 0
    eps_22cf: float = 0.0
    eps_22cm: float = 0.0
    fct_id_22c: int = 0
    eps_33tf: float = 0.0
    eps_33tm: float = 0.0
    fct_id_33t: int = 0
    eps_33cf: float = 0.0
    eps_33cm: float = 0.0
    fct_id_33c: int = 0
    eps_12tf: float = 0.0
    eps_12tm: float = 0.0
    fct_id_12t: int = 0
    fail_id: int = 0
    title: str = ""


# ============================================================================
# M193 Dataclasses: Extended Failure, Materials, Properties & State Directives
# ============================================================================

@dataclass
class FailEMC:
    """``/FAIL/EMC``: Extended Mohr-Coulomb ductile fracture model."""
    mat_id: int = 0
    a_emc: float = 0.0
    n_emc: float = 0.0
    b0: float = 0.0
    c: float = 0.0
    gamma: float = 0.0
    epsilon_dot_0: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailNXT:
    """``/FAIL/NXT``: NXT ductile fracture model."""
    mat_id: int = 0
    fct_id1: int = 0
    fct_id2: int = 0
    ifail_sh: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailTButcher:
    """``/FAIL/TBUTCHER``: Tuler-Butcher cumulative damage dynamic fracture."""
    mat_id: int = 0
    lam: float = 0.0
    k: float = 0.0
    sigma_r: float = 0.0
    ifail_sh: int = 0
    ifail_so: int = 0
    iduct: int = 0
    ixfem: int = 0
    a: float = 0.0
    b: float = 0.0
    dadv: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailMullins:
    """``/FAIL/MULLINS`` & ``/FAIL/MULLINS_OR``: Mullins effect elastomer damage."""
    mat_id: int = 0
    coefr: float = 1.0
    beta: float = 0.0
    coefm: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailCockcroft:
    """``/FAIL/COCKCROFT``: Cockcroft-Latham ductile failure model."""
    mat_id: int = 0
    c0: float = 0.0
    alpha: float = 1.0
    failip: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class MatLaw53:
    """``/MAT/LAW53`` & ``/MAT/TSAI_TAB``: Tsai-Wu tabulated orthotropic plasticity."""
    id: int = 0
    rho: float = 0.0
    ref_rho: float = 0.0
    e1: float = 0.0
    e2: float = 0.0
    gab: float = 0.0
    gbc: float = 0.0
    fun_a1: int = 0
    fun_b1: int = 0
    fun_a3: int = 0
    fun_a5: int = 0
    fun_a6: int = 0
    sfac11: float = 1.0
    sfac22: float = 1.0
    sfac12: float = 1.0
    sfac23: float = 1.0
    sfac45: float = 1.0
    title: str = ""


@dataclass
class MatLaw54:
    """``/MAT/LAW54`` & ``/MAT/PREDIT``: Specialized progressive damage plasticity."""
    id: int = 0
    rho: float = 0.0
    ref_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    ifunc: int = 0
    a: float = 0.0
    b: float = 0.0
    n: float = 0.0
    sfac: float = 1.0
    ay: float = 0.0
    az: float = 0.0
    by: float = 0.0
    bz: float = 0.0
    cx: float = 0.0
    dc: float = 0.0
    rc: float = 0.0
    eps_max: float = 0.0
    title: str = ""


@dataclass
class MatLaw74:
    """``/MAT/LAW74`` & ``/MAT/HILL_THERM``: Thermal Hill orthotropic plasticity."""
    id: int = 0
    rho: float = 0.0
    ref_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    eps_p_max: float = 0.0
    eps_t: float = 0.0
    eps_m: float = 0.0
    fsmooth: int = 0
    c_hard: float = 0.0
    fcut: float = 0.0
    sig11y: float = 0.0
    sig22y: float = 0.0
    sig33y: float = 0.0
    sig12y: float = 0.0
    sig23y: float = 0.0
    sig31y: float = 0.0
    tab_id: int = 0
    sigma_scale: float = 1.0
    epspt_scale: float = 1.0
    ti: float = 0.0
    rho0_cp: float = 0.0
    title: str = ""


@dataclass
class MatLaw82:
    """``/MAT/LAW82`` & ``/MAT/OGDEN``: Ogden hyperelastic material."""
    id: int = 0
    rho: float = 0.0
    ref_rho: float = 0.0
    order: int = 0
    nu: float = 0.475
    mu_arr: List[float] = field(default_factory=list)
    alpha_arr: List[float] = field(default_factory=list)
    gamma_arr: List[float] = field(default_factory=list)
    title: str = ""


@dataclass
class PropIntBeamIP:
    y: float = 0.0
    z: float = 0.0
    area: float = 0.0


@dataclass
class PropType18:
    """``/PROP/TYPE18`` & ``/PROP/INT_BEAM``: Integrated beam property."""
    id: int = 0
    isflag: int = 0
    ismstr: int = 0
    dm: float = 0.0
    df: float = 0.0
    nip: int = 0
    iref: int = 0
    y0: float = 0.0
    z0: float = 0.0
    ips: List[PropIntBeamIP] = field(default_factory=list)
    nitrs: int = 0
    l1: float = 0.0
    l2: float = 0.0
    l3: float = 0.0
    l4: float = 0.0
    l5: float = 0.0
    l6: float = 0.0
    wx1: int = 0
    wy1: int = 0
    wz1: int = 0
    wx2: int = 0
    wy2: int = 0
    wz2: int = 0
    title: str = ""


@dataclass
class DefInterType11:
    """``/DEF_INTER/TYPE11``: Default parameters for interface TYPE11."""
    istf: int = 5
    igap: int = 1000
    ikrem: int = 1
    noddel11: int = 1000
    iform: int = 1
    inactiv: int = 1000


@dataclass
class DefInterType19:
    """``/DEF_INTER/TYPE19``: Default parameters for interface TYPE19."""
    istf: int = 1000
    igap: int = 1000
    iedge: int = 2
    ibag: int = 2
    idel7: int = 1000
    icurv: int = 0
    inactiv: int = 1000
    iform: int = 1


@dataclass
class DefInterType25:
    """``/DEF_INTER/TYPE25``: Default parameters for interface TYPE25."""
    istf: int = 0
    igap: int = 0
    irem_i2: int = 0
    idel: int = 0
    itied: int = 0
    ishape: int = 0
    irs: int = 1000


@dataclass
class StateDirective:
    """``/STATE/...``: Element state initialization / restart directives."""
    kind: str = ""
    subtype: str = ""
    option: int = 0
    val: float = 0.0


@dataclass
class FailXFEM:
    """``/FAIL/XFEM/...``: Extended FEM fracture criteria."""
    mat_id: int = 0
    model_name: str = ""
    fct_id: int = 0
    scale: float = 1.0
    eps: float = 0.0
    sigma: float = 0.0
    sig0: float = 0.0
    lam: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    d4: float = 0.0
    d5: float = 0.0
    eps_dot_0: float = 1.0
    k: float = 0.0
    sigma_r: float = 0.0
    ifail_sh: int = 1
    iduct: int = 0
    a: float = 0.0
    b: float = 0.0
    fail_id: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class SphFlow:
    """``/SPH_FLOW/id`` or ``/SPH/FLOW/id`` (M194): SPH flow boundary condition."""
    id: int
    title: str = ""
    surf_id: int = 0
    part_id: int = 0
    fct_id: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class MidDirective:
    """``/MID/id`` (M194): Material ID assignment / mapping card."""
    id: int
    mat_id: int
    title: str = ""


@dataclass
class PidDirective:
    """``/PID/id`` (M194): Part/Property ID assignment / mapping card."""
    id: int
    prop_id: int
    title: str = ""


@dataclass
class SphParticle:
    """SPH particle element (M194)."""
    id: int
    part_id: int = 0
    node_id: int = 0


@dataclass
class FailTab1:
    """``/FAIL/TAB1`` (M194): Tabulated failure model Version 1."""
    mat_id: int = 0
    ifail_sh: int = 1
    ifail_so: int = 1
    p_thickfail: float = 0.0
    p_thinfail: float = 0.0
    ixfem: int = 0
    dcrit: float = 1.0
    d: float = 0.0
    n: float = 1.0
    dadv: float = 0.0
    fct_idd: int = 0
    table1_id: int = 0
    xscale1: float = 1.0
    xscale2: float = 1.0
    table2_id: int = 0
    xscale3: float = 1.0
    xscale4: float = 1.0
    fct_id_el: int = 0
    fscale_el: float = 1.0
    el_ref: float = 1.0
    inst_start: float = 0.0
    fad_exp: float = 1.0
    ch_i_f: float = 0.0
    fct_id_t: int = 0
    fscale_t: float = 1.0
    ifunc: int = 0
    eps_max: float = 0.0
    scale: float = 1.0
    fail_id: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class MatLaw40:
    """``/MAT/LAW40`` or ``/MAT/CONCR_SUB`` (M194): Concrete subgrade model."""
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class MatLaw80:
    """``/MAT/LAW80`` or ``/MAT/BARLAT3`` (M194): Barlat 3-parameter plasticity."""
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class MatLaw102:
    """``/MAT/LAW102``, ``/MAT/HILL_48`` (M194), or ``/MAT/DPRAG2`` (M195)."""
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    b0: float = 0.0
    b1: float = 0.0
    icrit: int = 1
    params: dict = field(default_factory=dict)


@dataclass
class MatNLocal:
    """``/MAT/NLOCAL`` (M194): Nonlocal plastic strain regularisation."""
    id: int
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class PropType13:
    """``/PROP/TYPE13`` or ``/PROP/SPR_PULL`` (M194): Pulling spring property."""
    id: int
    title: str = ""
    stiff: float = 0.0
    f_max: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class DefInterType2:
    """``/DEF_INTER/TYPE2`` (M194): Default Type 2 interface parameters."""
    istf: int = 0
    igap: int = 0
    iref: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class EngineTHRecord:
    """Engine /TH time history request record (M194)."""
    th_type: str
    id: int = 0
    title: str = ""
    vars: list = field(default_factory=list)
    ids: list = field(default_factory=list)


MatConcrSub = MatLaw40
MatHill48 = MatLaw102
PropSprPull = PropType13


@dataclass
class MatLaw103:
    """``/MAT/LAW103`` or ``/MAT/HENSEL_SPITTEL`` (M195): Hensel-Spittel hot-forming material law."""
    id: int
    title: str = ""
    rho: float = 0.0
    refer_rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    a0: float = 0.0
    m1: float = 0.0
    m2: float = 0.0
    m3: float = 0.0
    m4: float = 0.0
    m5: float = 0.0
    m7: float = 0.0
    fsmooth: int = 0
    fcut: float = 0.0
    eps_0: float = 0.0
    pmin: float = -1.0e30
    rhocp: float = 0.0
    t0: float = 0.0
    eta: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class MatLaw108:
    """``/MAT/LAW108`` or ``/MAT/SPR_GENE`` (M195): Generalized 6-DOF nonlinear spring material."""
    id: int
    title: str = ""
    rho: float = 0.0
    ifail: int = 0
    iequil: int = 0
    ifail2: int = 0
    k: list[float] = field(default_factory=lambda: [0.0]*6)
    c: list[float] = field(default_factory=lambda: [0.0]*6)
    a: list[float] = field(default_factory=lambda: [0.0]*6)
    b: list[float] = field(default_factory=lambda: [0.0]*6)
    d: list[float] = field(default_factory=lambda: [0.0]*6)
    fct_id1: list[int] = field(default_factory=lambda: [0]*6)
    h: list[float] = field(default_factory=lambda: [0.0]*6)
    fct_id2: list[int] = field(default_factory=lambda: [0]*6)
    fct_id3: list[int] = field(default_factory=lambda: [0]*6)
    fct_id4: list[int] = field(default_factory=lambda: [0]*6)
    delta_min: list[float] = field(default_factory=lambda: [0.0]*6)
    delta_max: list[float] = field(default_factory=lambda: [0.0]*6)
    f_val: list[float] = field(default_factory=lambda: [0.0]*6)
    e_val: list[float] = field(default_factory=lambda: [0.0]*6)
    ascale: list[float] = field(default_factory=lambda: [1.0]*6)
    hscale: list[float] = field(default_factory=lambda: [1.0]*6)
    fsmooth: int = 0
    fcut: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class MatPlasPredef:
    """``/MAT/PLAS_PREDEF`` (M195): Predefined plasticity material model."""
    id: int
    title: str = ""
    mat_name: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    sigy: float = 0.0
    uts: float = 0.0
    e_uts: float = 0.0
    epsp_f: float = 0.0
    vp: float = 0.0
    c: float = 0.0
    p: float = 0.0
    n: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class MatDPrag2:
    """``/MAT/DPRAG2`` (M195): Drucker-Prager 2nd formulation material."""
    id: int
    title: str = ""
    rho: float = 0.0
    iform: int = 1
    e: float = 0.0
    nu: float = 0.0
    c: float = 0.0
    phi: float = 0.0
    amax: float = 1.0e30
    pmin: float = -1.0e30
    params: dict = field(default_factory=dict)


@dataclass
class PropType23:
    """``/PROP/TYPE23`` or ``/PROP/SPR_MAT`` (M195): Spring material property."""
    id: int
    title: str = ""
    mass: float = 0.0
    skew_id: int = 0
    isens: int = 0
    iflag: int = 0
    imass: int = 2
    area_or_volume: float = 0.0
    inertia: float = 0.0
    sensor_id: int = 0
    isflag: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class FunctSmooth:
    """``/FUNCT_SMOOTH`` (M195): Smoothed curve function."""
    id: int
    title: str = ""
    smooth_type: int = 0
    order: int = 3
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def evaluate(self, x_val: float) -> float:
        """Evaluate function at x_val using linear/interpolated curve."""
        if not self.x:
            return 0.0
        if len(self.x) == 1:
            return float(self.y[0])
        return float(np.interp(x_val, self.x, self.y))


@dataclass
class DampFreqRange:
    """``/DAMP/FREQUENCY_RANGE`` or ``/DAMP/FREQ_RANGE`` (M195): Frequency-range damping."""
    id: int = 0
    title: str = ""
    fmin: float = 0.0
    fmax: float = 0.0
    damp: float = 0.0
    itype: int = 0
    cdamp: float = 0.0
    grpart_id: int = 0
    tstart: float = 0.0
    tstop: float = 1.0e30
    freq_low: float = 0.0
    freq_high: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class DampFunct:
    """``/DAMP/FUNCT`` (M195): Function-dependent mass damping."""
    id: int = 0
    title: str = ""
    fct_id: int = 0
    damp_scale: float = 1.0
    itype: int = 0
    func_id: int = 0
    grnod_id: int = 0
    alpha: float = 0.0
    alpha_x: float = 0.0
    alpha_y: float = 0.0
    alpha_z: float = 0.0
    alpha_xx: float = 0.0
    alpha_yy: float = 0.0
    alpha_zz: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class IniStateTable:
    """Generic initial state table record for /INI* keywords (M195)."""
    keyword: str = ""
    id: int = 0
    title: str = ""
    rows: List[dict] = field(default_factory=list)


MatHenselSpittel = MatLaw103
MatSprGene = MatLaw108
PropSprMat = PropType23
DampFrequencyRange = DampFreqRange


@dataclass
class FrameNod:
    """``/FRAME/NOD`` or ``/FRAME/NODE`` (M196): Nodal coordinate reference frame."""
    id: int = 0
    title: str = ""
    originnodeid: int = 0
    axisnodeid: int = 0
    planenodeid: int = 0
    globalyaxis: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    globalzaxis: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    displayaxis: int = 0
    displayplane: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class InterType18:
    """``/INTER/TYPE18``: Fluid-structure Lagrangian-Eulerian/ALE coupling contact interface."""
    id: int = 0
    title: str = ""
    grnod_id: int = 0
    surf_id: int = 0
    grbric_id: int = 0
    istf: int = 0
    igap: int = 0
    multimp: int = 4
    ibag: int = 0
    idel18: int = 0
    iauto: int = 0
    stfac: float = 1.0
    vref: float = 0.0
    gap: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    stiff_dc: float = 0.0
    sort_fact: float = 0.2
    params: dict = field(default_factory=dict)


@dataclass
class TableBlock:
    """``/TABLE``, ``/TABLE/0``, ``/TABLE/1``: Multi-dimensional lookup tables."""
    id: int = 0
    title: str = ""
    dim: int = 1
    ref_id: int = 0
    x_values: List[float] = field(default_factory=list)
    y_values: List[float] = field(default_factory=list)
    curves: List[int] = field(default_factory=list)
    params: dict = field(default_factory=dict)


@dataclass
class MatLaw62:
    """``/MAT/LAW62`` or ``/MAT/VISC_ELAS`` (M197): Viscoelastic material."""
    id: int = 0
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class MatLaw88:
    """``/MAT/LAW88`` or ``/MAT/HONEYCOMB`` (M197): Honeycomb material."""
    id: int = 0
    title: str = ""
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class MatLaw93:
    """``/MAT/LAW93`` or ``/MAT/ORTH_HILL`` (M197): Orthotropic Hill material."""
    id: int = 0
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    e11: float = 0.0
    e22: float = 0.0
    e33: float = 0.0
    g12: float = 0.0
    nu12: float = 0.0
    g13: float = 0.0
    g23: float = 0.0
    nu13: float = 0.0
    nu23: float = 0.0
    nl: int = 0
    sigma_y: float = 0.0
    qr1: float = 0.0
    cr1: float = 0.0
    qr2: float = 0.0
    cr2: float = 0.0
    r11: float = 1.0
    r22: float = 1.0
    r12: float = 1.0
    r33: float = 1.0
    r13: float = 1.0
    r23: float = 1.0
    fcut: float = 0.0
    vp: int = 0
    curves: list = field(default_factory=list)
    rho: float = 0.0
    e: float = 0.0
    nu: float = 0.0
    params: dict = field(default_factory=dict)


MatViscElas = MatLaw62
MatHoneycomb = MatLaw88
MatOrthHill = MatLaw93


@dataclass
class CNode:
    """``/CNODE`` (M198): Commented coordinate node definition.

    Fortran origin: ``starter/source/elements/reader/hm_read_node.F`` / CFG ``cnode.cfg``.
    """
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    comments: List[str] = field(default_factory=list)


@dataclass
class Upbeam:
    """``/UPBEAM`` or ``/UPBEAM/INT_BEAM`` (M198): Integrated beam cross-section update."""
    id: int
    title: str = ""
    grnd_id: int = 0
    i_updt: int = 0
    eps_max: float = 0.0
    npt_int: int = 0


@dataclass
class RelaxSystem:
    """``/RELAX`` or ``/RELAX/SYSTEM`` or ``/RELAX/DYNA`` (M198): Quasi-static dynamic relaxation."""
    id: int
    title: str = ""
    t_start: float = 0.0
    t_stop: float = 0.0
    damp_coeff: float = 0.0
    i_damp: int = 0
    v_lim: float = 0.0
    eps_tol: float = 0.0




@dataclass
class MonvolComm:
    """``/MONVOL/COMM`` or ``/MONVOL/COMMUNICATION`` (M198): Direct inter-chamber communication between monitored volumes."""
    id: int
    title: str = ""
    monvol1_id: int = 0
    monvol2_id: int = 0
    surface_id: int = 0
    cd: float = 0.0
    a_vent: float = 0.0
    fct_id: int = 0
    sens_id: int = 0


@dataclass
class TransformMatrix:
    """``/TRANSFORM/MATRIX`` or ``/MATRIX`` (M199): Affine 3D matrix transformation."""
    id: int
    title: str = ""
    grnod_id: int = 0
    matrix: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    translation: tuple = (0.0, 0.0, 0.0)
    sub_id: int = 0
    submodel: int = 0

    @property
    def m11(self) -> float:
        return self.matrix[0][0]

    @property
    def m12(self) -> float:
        return self.matrix[0][1]

    @property
    def m13(self) -> float:
        return self.matrix[0][2]

    @property
    def m21(self) -> float:
        return self.matrix[1][0]

    @property
    def m22(self) -> float:
        return self.matrix[1][1]

    @property
    def m23(self) -> float:
        return self.matrix[1][2]

    @property
    def m31(self) -> float:
        return self.matrix[2][0]

    @property
    def m32(self) -> float:
        return self.matrix[2][1]

    @property
    def m33(self) -> float:
        return self.matrix[2][2]

    @property
    def tx(self) -> float:
        return self.translation[0]

    @property
    def ty(self) -> float:
        return self.translation[1]

    @property
    def tz(self) -> float:
        return self.translation[2]


@dataclass
class InterType10:
    """``/INTER/TYPE10`` (M199): Secondary node group to main surface contact interface."""
    id: int
    title: str = ""
    grnod_id: int = 0
    surf_id: int = 0
    multimp: int = 0
    idel: int = 0
    stfac: float = 1.0
    gap: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    itied: int = 0
    inactiv: int = 0
    stiff_dc: float = 0.0
    sort_fact: float = 0.2
    params: dict = field(default_factory=dict)


@dataclass
class InterType12:
    """``/INTER/TYPE12`` (M199): General sliding/tied surface-to-surface interface with interpolation."""
    id: int
    title: str = ""
    surf_ids: int = 0
    surf_idm: int = 0
    interpol: int = 0
    tol: float = 0.02
    tstart: float = 0.0
    tstop: float = 1.0e30
    itied: int = 0
    bcopt: int = 0
    skew_id: int = 0
    node_c: int = 0
    xc: float = 0.0
    yc: float = 0.0
    zc: float = 0.0
    theta: float = 0.0
    xn: float = 0.0
    yn: float = 0.0
    zn: float = 0.0
    xt: float = 0.0
    yt: float = 0.0
    zt: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float, float]:
        return (self.xc, self.yc, self.zc)

    @property
    def normal(self) -> tuple[float, float, float]:
        return (self.xn, self.yn, self.zn)

    @property
    def tangent(self) -> tuple[float, float, float]:
        return (self.xt, self.yt, self.zt)



@dataclass
class MatLaw51:
    """``/MAT/LAW51`` or ``/MAT/DRUCKER_PRAGER`` or ``/MAT/MULTIFLUID`` (M199): Multi-material / Drucker-Prager brittle model."""
    id: int = 0
    title: str = ""
    rho0: float = 0.0
    rhor: float = 0.0
    iform: int = 0
    pext: float = 0.0
    nu: float = 0.0
    lamda: float = 0.0
    scale: float = 1.0
    rho: float = 0.0
    e: float = 0.0
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    fc: float = 0.0
    ft: float = 0.0
    fmax: float = 0.0
    fres: float = 0.0
    eps_c: float = 0.0
    eps_t: float = 0.0
    eps_res: float = 0.0
    b: float = 0.0
    iflag: int = 0
    icomp: int = 0
    itot: int = 0
    pc: float = 0.0
    gamma: float = 0.0
    pt: float = 0.0
    psi: float = 0.0
    p0: float = 0.0
    beta: float = 0.0
    epsp_max: float = 0.0
    fac_e: float = 1.0
    params: dict = field(default_factory=dict)


MatMultiFluid = MatLaw51
MatDruckerPrager = MatLaw51
MatBrittle = MatLaw51
MatMultimat = MatLaw51


# ============================================================================
# M200 Entities: DETPOINT, DTIX
# ============================================================================

@dataclass
class DetPointNode:
    """``/DFS/DETPOINT/NODE`` or ``/DETPOINT/NODE`` (M200): Detonation point at node."""
    id: int = 0
    ishadow: int = 0
    iframe1: int = 0
    iframe2: int = 0
    r0_shadow: float = 0.0
    radius: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    node_id1: int = 0
    node_id: int = 0
    title: str = ""

    def __post_init__(self):
        if not self.r0_shadow and self.radius:
            self.r0_shadow = self.radius
        elif not self.radius and self.r0_shadow:
            self.radius = self.r0_shadow
        if not self.node_id1 and self.node_id:
            self.node_id1 = self.node_id
        elif not self.node_id and self.node_id1:
            self.node_id = self.node_id1


@dataclass
class DetPointSet:
    """``/DFS/DETPOINT/SET`` or ``/DETPOINT/SET`` / ``/DETPOINT/GRNOD`` (M200): Detonation point on node group."""
    id: int = 0
    ishadow: int = 0
    iframe1: int = 0
    iframe2: int = 0
    r0_shadow: float = 0.0
    radius: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    grnod_id1: int = 0
    grnod_id: int = 0
    title: str = ""

    def __post_init__(self):
        if not self.r0_shadow and self.radius:
            self.r0_shadow = self.radius
        elif not self.radius and self.r0_shadow:
            self.radius = self.r0_shadow
        if not self.grnod_id1 and self.grnod_id:
            self.grnod_id1 = self.grnod_id
        elif not self.grnod_id and self.grnod_id1:
            self.grnod_id = self.grnod_id1


@dataclass
class DtixControl:
    """``/DTIX`` or ``/ENG/DTIX`` (M200): Initial and maximum explicit time step control."""
    id: int = 1
    t_ini: float = 0.0
    t_max: float = 0.0
    tini: float = 0.0
    tmax: float = 0.0

    def __post_init__(self):
        if not self.t_ini and self.tini:
            self.t_ini = self.tini
        elif not self.tini and self.t_ini:
            self.tini = self.t_ini
        if not self.t_max and self.tmax:
            self.t_max = self.tmax
        elif not self.tmax and self.t_max:
            self.tmax = self.t_max


# ============================================================================
# M201 Entities: TH_SUBS, THPART, WAV_SHA, SENSORS, PCOMPP, TYPE51
# ============================================================================

@dataclass
class ThSubs:
    """``/TH/SUBS/id`` (M201): Substructure Time History output block."""
    id: int = 0
    title: str = ""
    prefix: str = "TH"
    vars: List[str] = field(default_factory=list)
    subs_ids: List[int] = field(default_factory=list)


@dataclass
class ThPartGroup:
    """``/THPART/GR.../id`` (M201): Group-based Time History part output block."""
    id: int = 0
    title: str = ""
    elem_type: str = "SHEL"  # 'BEAM', 'BRIC', 'QUAD', 'SH3N', 'SHEL', 'SPRI', 'TRUS'
    grelem_id: int = 0


@dataclass
class DfsWavSha:
    """``/DFS/WAV_SHA/id`` or ``/WAVE/id`` (M201): Wave shaper / spherical ignition modifier."""
    id: int = 0
    title: str = ""
    xdet: float = 0.0
    ydet: float = 0.0
    zdet: float = 0.0
    tdet: float = 0.0
    mat_id: int = 0
    grnod_id: int = 0


@dataclass
class SensorDistSurf:
    """``/SENSOR/DIST_SURF/id`` (M201): Surface distance threshold sensor."""
    id: int = 0
    title: str = ""
    surf_id: int = 0
    node_id: int = 0
    surf_target_id: int = 0
    dist_min: float = 0.0
    dist_max: float = 0.0
    t_delay: float = 0.0
    tdelay: float = 0.0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    tmin: float = 0.0
    dmin: float = 0.0
    dmax: float = 0.0


@dataclass
class SensorSensAndOr:
    """``/SENSOR/SENS_AND_OR/id`` (M201): Compound Boolean logical combination sensor."""
    id: int = 0
    title: str = ""
    logic_type: str = "AND"  # 'AND' | 'OR' | 'NAND' | 'NOR'
    sensor_id1: int = 0
    sensor_id2: int = 0
    sens_id1: int = 0
    sens_id2: int = 0
    t_delay: float = 0.0
    tdelay: float = 0.0


@dataclass
class PropPcompp:
    """``/PROP/PCOMPP/id`` (M201): Ply-based composite shell property."""
    id: int = 0
    title: str = ""
    laminate_id: int = 0


PropP51 = PropType51
PropTshP51 = PropType51
WaveShaperDfs = DfsWavSha


@dataclass
class HeatConvec:
    """``/HEAT/CONVEC/id``, ``/HEAT/CONVECTION/id`` or ``/CONVEC/id`` (M202): Thermal surface convection boundary condition."""
    id: int = 0
    title: str = ""
    surf_id: int = 0
    funct_id: int = 0
    sensor_id: int = 0
    ascale: float = 1.0
    fscale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    h: float = 0.0


@dataclass
class HeatRadiation:
    """``/HEAT/RADIATION/id``, ``/HEAT/RAD/id`` or ``/RADIATION/id`` (M202): Thermal surface radiation boundary condition."""
    id: int = 0
    title: str = ""
    surf_id: int = 0
    funct_id: int = 0
    sensor_id: int = 0
    ascale: float = 1.0
    fscale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    emissivity: float = 0.0
    emiss: float = 0.0

    def __post_init__(self):
        if self.emissivity != 0.0 and self.emiss == 0.0:
            self.emiss = self.emissivity
        elif self.emiss != 0.0 and self.emissivity == 0.0:
            self.emissivity = self.emiss


@dataclass
class SurfSurf:
    """``/SURF/SURF/id`` or ``/SURFSURF/id`` (M202): Surface composed of other surfaces or surface-to-surface interaction."""
    id: int = 0
    title: str = ""
    surf_ids: list[int] = field(default_factory=list)
    surf1_id: int = 0
    surf2_id: int = 0
    iflag: int = 0
    gap: float = 0.0
    fric: float = 0.0



@dataclass
class BcsLagmul:
    """``/BCS/LAGMUL/id`` (M202): Lagrange multiplier constraint on node group."""
    id: int = 0
    title: str = ""
    tra: str = "111"
    rot: str = "111"
    skew_id: int = 0
    grnod_id: int = 0


@dataclass
class Spcnd:
    """``/SPCND/id`` (M202): Single point constraint on node."""
    id: int = 0
    title: str = ""
    node_id: int = 0
    dof: str = "111111"
    f_sens: float = 0.0
    tstart: float = 0.0
    tstop: float = 1.0e30


@dataclass
class Ddw:
    """``/DDW/id`` (M202): Deep draw wall stamping tool."""
    id: int = 0
    title: str = ""
    tool_type: int = 1
    surf_id: int = 0
    grnod_id: int = 0
    fct_id: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0


@dataclass
class DdwPoint:
    """``/DDW/POINT/id`` (M202): Point-based deep draw wall."""
    id: int = 0
    title: str = ""
    node_id: int = 0
    fct_id: int = 0
    dir: str = "Z"
    scale: float = 1.0


@dataclass
class Stamping:
    """``/STAMPING`` or ``/STAMP`` (M202): Stamping simulation controls."""
    hf_timescale: float = 1.0
    datalines: list[str] = field(default_factory=list)


@dataclass
class WindowUser:
    """``/WINDOW/USER/id`` or ``/USERWI/id`` (M202): User-defined analysis window."""
    id: int = 0
    title: str = ""
    lines: list[str] = field(default_factory=list)


@dataclass
class HeatFlux:
    """``/HEAT/FLUX/id`` or ``/FLUX/id`` (M202): Thermal heat flux boundary condition."""
    id: int = 0
    title: str = ""
    surf_id: int = 0
    funct_id: int = 0
    sensor_id: int = 0
    ascale: float = 1.0
    fscale: float = 1.0
    tstart: float = 0.0
    tstop: float = 1.0e30
    q: float = 0.0



@dataclass
class SensorWork:
    """``/SENSOR/WORK/id`` (M202): Internal/plastic work threshold sensor."""
    id: int = 0
    title: str = ""
    node_id1: int = 0
    node_id2: int = 0
    object_id: int = 0
    sens_type: int = 1
    t_delay: float = 0.0
    tdelay: float = 0.0
    w_max: float = 0.0
    work_max: float = 0.0
    tmin: float = 0.0
    sect_id: int = 0
    int_id: int = 0
    rbody_id: int = 0
    rwall_id: int = 0

    def __post_init__(self):
        if self.t_delay != 0.0 and self.tdelay == 0.0:
            self.tdelay = self.t_delay
        elif self.tdelay != 0.0 and self.t_delay == 0.0:
            self.t_delay = self.tdelay
        if self.w_max != 0.0 and self.work_max == 0.0:
            self.work_max = self.w_max
        elif self.work_max != 0.0 and self.w_max == 0.0:
            self.w_max = self.work_max
        if self.object_id != 0 and self.node_id1 == 0:
            self.node_id1 = self.object_id
        elif self.node_id1 != 0 and self.object_id == 0:
            self.object_id = self.node_id1



LagmulGear = GearConstraint
LagmulRack = RackConstraint
LagmulDiff = DiffConstraint
InterType26 = GuidedCable
InterGuidedCable = GuidedCable


@dataclass
class SensorPython:
    """``/SENSOR/PYTHON/id`` (M203): Python-scripted sensor function."""
    id: int = 0
    title: str = ""
    script_name: str = ""
    func_name: str = ""
    code: str = ""
    t_delay: float = 0.0
    tdelay: float = 0.0
    t_act: float = 0.0
    sensor_type: str = "PYTHON"

    def __post_init__(self):
        if self.t_delay != 0.0 and self.tdelay == 0.0:
            self.tdelay = self.t_delay
        elif self.tdelay != 0.0 and self.t_delay == 0.0:
            self.t_delay = self.tdelay


@dataclass
class TransformPos:
    """``/TRANSFORM/POS/id``, ``/TRANSFORM/POSITION/id``, ``/POS/id`` (M203): 6-point 3D alignment transformation."""
    id: int = 0
    title: str = ""
    grnod_id: int = 0
    node1: int = 0
    node2: int = 0
    node3: int = 0
    node4: int = 0
    node5: int = 0
    node6: int = 0
    node_ids: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    submodel_id: int = 0
    submodel: int = 0
    points: List[Tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not any(self.node_ids) and any((self.node1, self.node2, self.node3, self.node4, self.node5, self.node6)):
            self.node_ids = (self.node1, self.node2, self.node3, self.node4, self.node5, self.node6)
        elif any(self.node_ids) and not any((self.node1, self.node2, self.node3, self.node4, self.node5, self.node6)):
            self.node1 = self.node_ids[0] if len(self.node_ids) > 0 else 0
            self.node2 = self.node_ids[1] if len(self.node_ids) > 1 else 0
            self.node3 = self.node_ids[2] if len(self.node_ids) > 2 else 0
            self.node4 = self.node_ids[3] if len(self.node_ids) > 3 else 0
            self.node5 = self.node_ids[4] if len(self.node_ids) > 4 else 0
            self.node6 = self.node_ids[5] if len(self.node_ids) > 5 else 0
        if self.submodel_id != 0 and self.submodel == 0:
            self.submodel = self.submodel_id
        elif self.submodel != 0 and self.submodel_id == 0:
            self.submodel_id = self.submodel


PosTransform = TransformPos
TransformPosition = TransformPos


@dataclass
class ChecksumDirective:
    """``/CHECKSUM/START``, ``/CHECKSUM/END`` (M203): Checksum calculation block directive."""
    id: int = 0
    title: str = ""
    action: str = "START"  # "START" or "END"
    val1: int = 0
    val2: int = 0


BoxRect = Box
BoxCyl = Box
BoxSphere = Box
BoxCylin = Box
BoxSpher = Box


@dataclass
class FailJohnson:
    """``/FAIL/JOHNSON/mat_id`` or ``/FAIL/JOHN_COOK/mat_id`` (M203): Johnson-Cook failure model."""
    id: int = 0
    mat_id: int = 0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    d4: float = 0.0
    d5: float = 0.0
    eps_dot_0: float = 1.0
    ifail_sh: int = 1
    ifail_so: int = 1
    epsf_min: float = 0.0
    dadv: float = 0.0
    ixfem: int = 0
    failip: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailBiquad:
    """``/FAIL/BIQUAD/mat_id`` (M203): Biquadratic failure criterion."""
    id: int = 0
    mat_id: int = 0
    c1: float = 0.0
    c2: float = 0.0
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0
    p_thickfail: float = 0.0
    m_flag: int = 0
    s_flag: int = 2
    inst_start: float = 0.0
    ireg: int = 0
    fct_idel: int = 0
    ei_ref: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    r4: float = 0.0
    r5: float = 0.0
    icoup: int = 0
    dcrit: float = 0.0
    exp: float = 0.0
    failip: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailFld:
    """``/FAIL/FLD/mat_id`` (M203): Forming Limit Diagram failure criterion."""
    id: int = 0
    mat_id: int = 0
    fct_id: int = 0
    ifail_sh: int = 1
    i_marg: int = 0
    fct_idadv: int = 0
    rani: float = 0.0
    dadv: float = 0.0
    istrain: int = 0
    ixfem: int = 0
    factor_marginal: float = 0.0
    factor_loosemetal: float = 0.0
    fcut: float = 0.0
    alpha: float = 0.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailConnect:
    """``/FAIL/CONNECT/mat_id`` (M203): Connector failure criterion."""
    id: int = 0
    mat_id: int = 0
    epsilon_maxn: float = 0.0
    exponent_n: float = 1.0
    alpha_n: float = 1.0
    r_fct_id_n: int = 0
    ifail: int = 0
    ifail_so: int = 0
    isym: int = 0
    epsilon_maxt: float = 0.0
    exponent_t: float = 1.0
    alpha_t: float = 1.0
    r_fct_id_t: int = 0
    ei_max: float = 0.0
    en_max: float = 0.0
    et_max: float = 0.0
    n_n: float = 0.0
    n_t: float = 0.0
    t_max: float = 0.0
    n_soft: float = 0.0
    area_scale: float = 1.0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailFractalDmg:
    """``/FAIL/FRACTAL_DMG/mat_id`` (M203): Fractal damage percolation failure criterion."""
    id: int = 0
    mat_id: int = 0
    grsh4n_1: int = 0
    grsh3n_1: int = 0
    grsh4n_2: int = 0
    grsh3n_2: int = 0
    damage: float = 0.0
    probability: float = 0.0
    seed: int = 0
    num_walk: int = 0
    printout: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class FailOrthenerg:
    """``/FAIL/ORTHENERG/mat_id`` (M203): Orthotropic energy failure criterion."""
    id: int = 0
    mat_id: int = 0
    pthickfail: float = 0.0
    nmod: int = 0
    failip: int = 0
    sigma_11t: float = 0.0
    g_11t: float = 0.0
    ishap11t: int = 0
    sigma_11c: float = 0.0
    g_11c: float = 0.0
    ishap11c: int = 0
    sigma_22t: float = 0.0
    g_22t: float = 0.0
    ishap22t: int = 0
    sigma_22c: float = 0.0
    g_22c: float = 0.0
    ishap22c: int = 0
    sigma_33t: float = 0.0
    g_33t: float = 0.0
    ishap33t: int = 0
    sigma_33c: float = 0.0
    g_33c: float = 0.0
    ishap33c: int = 0
    sigma_12t: float = 0.0
    g_12t: float = 0.0
    ishap12t: int = 0
    sigma_12c: float = 0.0
    g_12c: float = 0.0
    ishap12c: int = 0
    sigma_23t: float = 0.0
    g_23t: float = 0.0
    ishap23t: int = 0
    sigma_23c: float = 0.0
    g_23c: float = 0.0
    ishap23c: int = 0
    sigma_31t: float = 0.0
    g_31t: float = 0.0
    ishap31t: int = 0
    sigma_31c: float = 0.0
    g_31c: float = 0.0
    ishap31c: int = 0
    fail_id: int = 0
    title: str = ""


@dataclass
class AdmeshSet:
    """``/ADMESH/SET/id`` (M203): Adaptive meshing on element/node set."""
    id: int = 0
    title: str = ""
    angle_criteria: float = 0.0
    inilev: int = 0
    thkerr: float = 0.0
    part_ids: List[int] = field(default_factory=list)
    grnd_id: int = 0
    level: int = 0
    tdelay: float = 0.0


@dataclass
class GaugeSph:
    """``/GAUGE/SPH/id`` (M203): SPH gauge point measurement."""
    id: int = 0
    title: str = ""
    node_id: int = 0
    fcut: float = 0.0
    shell_id: int = 0
    dist: float = 0.0


@dataclass
class HeatSolver:
    """``/HEAT/SOLVER/id``, ``/HEAT/GLOBAL/id`` (M204): Thermal transient solver controls."""
    id: int = 1
    title: str = ""
    isolv: int = 1
    itype: int = 1
    ttol: float = 1e-3
    dttmax: float = 1.0
    dttmin: float = 1e-6


@dataclass
class XfemControl:
    """``/XFEM[/<subtype>]/id`` (M204): X-FEM extended finite element enrichment control."""
    id: int = 0
    title: str = ""
    subtype: str = "SHELL"
    grpart_id: int = 0
    crack_id: int = 0
    ifail: int = 0
    i_enrich: int = 1


@dataclass
class SensorSubsystem:
    """``/SENSOR/{AIRBAG|MONVOL|SHELL|SOLID|SPH}/sens_ID`` (M204): Subsystem threshold sensor."""
    id: int = 0
    title: str = ""
    kind: str = "SHELL"
    target_id: int = 0
    v1: float = 0.0
    v2: float = 0.0
    tmin: float = 0.0
    tdelay: float = 0.0


SensorAirbag = SensorSubsystem
SensorShell = SensorSubsystem
SensorSolid = SensorSubsystem
SensorSph = SensorSubsystem
HeatGlobal = HeatSolver


@dataclass
class FlowBoundary:
    """``/FLOW[/<subtype>]/id`` or ``/ALE/FLOW/id`` (M205): ALE flow boundary condition."""
    id: int = 1
    title: str = ""
    subtype: str = "INFLOW"
    surf_id: int = 0
    flow_type: int = 0
    rho: float = 0.0
    pres: float = 0.0
    temp: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    sens_id: int = 0


@dataclass
class HeatRadCav:
    """``/HEAT/RAD_CAV/id`` (M205): Cavity radiation surface-to-surface coupling."""
    id: int = 1
    title: str = ""
    surf_id1: int = 0
    surf_id2: int = 0
    emissivity1: float = 1.0
    emissivity2: float = 1.0
    view_factor: float = 1.0


@dataclass
class PropSpringTors:
    """``/PROP/TYPE19`` or ``/PROP/SPR_TORS/id`` (M205): Torsional spring property."""
    id: int = 1
    title: str = ""
    mass: float = 0.0
    stiffness_k: float = 0.0
    damping_c: float = 0.0
    fcut: float = 0.0

    @property
    def k(self) -> float:
        return self.stiffness_k

    @property
    def c(self) -> float:
        return self.damping_c


@dataclass
class PropSpringBend:
    """``/PROP/TYPE20`` or ``/PROP/SPR_BEND/id`` (M205): Bending spring property."""
    id: int = 1
    title: str = ""
    mass: float = 0.0
    stiffness_k: float = 0.0
    damping_c: float = 0.0
    fcut: float = 0.0

    @property
    def k(self) -> float:
        return self.stiffness_k

    @property
    def c(self) -> float:
        return self.damping_c


PropType19 = PropSpringTors
PropType20 = PropSpringBend




