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
      the listed parts becomes an edge (elem_1D_line_buffer.F).
    """

    id: int
    title: str = ""
    surf_ids: List[int] = field(default_factory=list)         # /LINE/SURF
    seg_nodes: List[List[int]] = field(default_factory=list)  # /LINE/SEG (user ids)
    edge_surf_ids: List[int] = field(default_factory=list)    # /LINE/EDGE (M37)
    line_ids: List[int] = field(default_factory=list)         # /LINE/LINE (M37)
    part_ids: List[int] = field(default_factory=list)         # /LINE/PART (M37)
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
    """/LOAD/CENTRI (M93): centrifugal rotational load on a node group.

    Fortran origin: ``starter/source/loads/general/load_centri/hm_read_load_centri.F``.
    """
    id: int
    funct_id: int
    dir: str = "XX"         # rotation axis: X, Y, Z, XX, YY, ZZ
    frame_id: int = 0       # reference frame
    sens_id: int = 0        # /SENSOR gating
    grnod_id: int = 0       # node group
    ivar: int = 1           # 1 = ignore d_omega/dt, 2 = account for d_omega/dt
    scale_x: float = 1.0    # Ascalex (time scale)
    scale_y: float = 1.0    # Fscaley (rotational velocity scale)
    title: str = ""


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
    """/DAMP (M6): Rayleigh MASS damping — force f = -alpha m v on every
    node of the group, active in the [tstart, tstop] window.

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
    # Duration limit (M97)
    tmin: float = 0.0
    title: str = ""


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
    """/ADMAS (M5): concentrated non-structural mass added to every node
    of a group (Radioss type-0 semantics: the value is PER NODE).

    Fortran origin: ``starter/source/tools/admas/hm_read_admas.F``. Beyond
    its normal use (payload, joints), this is how a *moving rigid wall
    with a mass* is built in this port: the wall is tied to a node whose
    inertia comes from /ADMAS (see RigidWall.node_id).
    """

    id: int
    grnod_id: int
    mass: float
    title: str = ""


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
    igap: int = 0
    stfac: float = 1.0
    fric: float = 0.0
    gap: float = 0.0
    gap_max: float = 0.0  # igap=1 cap, 0 = no cap
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
    # ---- TYPE18 (M60) / TYPE10 -------------------------------------------
    ibag: int = 0
    multimp: int = 4
    idel18: int = 0
    idel10: int = 0       # type 10: segment deletion flag
    tstart: float = 0.0   # type 10: activation time
    tstop: float = 1e30   # type 10: deactivation time
    inactiv: int = 0      # type 10: initial penetration treatment
    stiff_dc: float = 0.0
    sort_fact: float = 0.2


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
    """``/TABLE/dim/table_ID`` (1D, 2D, ... tabular functions)."""
    id: int
    dim: int
    x: np.ndarray
    y: np.ndarray


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
    """Container surface entry for /INIVOL (M94)."""
    surf_id: int
    ale_phase: int = 1
    fill_opt: int = 0       # 0 = along normal, 1 = against normal (reversed)
    icumu: int = 0          # 0 = erase, 1 = additive, -1 = subtractive
    fill_ratio: float = 1.0 # filling volume fraction in [0, 1]


@dataclass
class InitialVolume:
    """/INIVOL (M94): initial volume fraction for multi-material fluid / ALE.

    Fortran origin: ``starter/source/initial_conditions/inivol/hm_read_inivol.F90``.
    """
    id: int
    part_id: int = 0
    title: str = ""
    containers: List[InivolContainer] = field(default_factory=list)


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
    """/IMPFLUX (M95): imposed surface or volumetric heat flux.

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
    """/INIBRI (M96): initial state for solid/brick elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F``.
    """
    elem_id: int
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [sxx, syy, szz, sxy, syz, sxz]
    epsp: float = 0.0      # plastic strain
    rho: float = 0.0       # initial density
    ener: float = 0.0      # internal energy


@dataclass
class InitialShellState:
    """/INISHE and /INISH3 (M96): initial state for shell elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F``.
    """
    elem_id: int
    thick: float = 0.0     # initial thickness override
    epsp: float = 0.0      # plastic strain
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(6))  # membrane stress
    sigma_b: np.ndarray = field(default_factory=lambda: np.zeros(6)) # bending stress
    em: float = 0.0        # membrane energy
    eb: float = 0.0        # bending energy
    h_energy: np.ndarray = field(default_factory=lambda: np.zeros(3)) # H1, H2, H3


@dataclass
class InitialTrussState:
    """/INITRU (M97): initial state for truss elements.

    Fortran origin: ``starter/source/elements/initia/hm_read_inistate_d00.F`` and
    ``starter/source/elements/truss/tsigini.F``.
    """
    elem_id: int
    prop_type: int = 2
    eint: float = 0.0      # initial internal energy
    force: float = 0.0     # initial axial force / tension
    area: float = 0.0      # initial area override
    epsp: float = 0.0      # plastic strain


@dataclass
class InitialBeamState:
    """/INIBEA (M97): initial state for beam elements.

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


@dataclass
class InitialSpringState:
    """/INISPR (M97): initial state for spring elements.

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
    """/LOAD/PBLAST (M99): air/ground blast pressure load.

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


@dataclass
class Ply:
    """/PLY/ply_id (M100): Composite ply definition.

    Fortran origin: ``starter/source/model/laminate/leclamply.F``.
    """
    id: int
    mat_id: int
    thick: float
    title: str = ""
    skew_id: int = 0


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
    """/BCS/NRF (M102): Non-reflecting boundary condition.

    Fortran origin: ``starter/source/boundary_conditions/hm_read_bcs_nrf.F90``.
    """
    id: int
    title: str = ""
    grnod_id: int = 0


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
    """/RLINK (M102): Standard rigid link definition between node group and main/skew frame.

    Fortran origin: ``starter/source/constraints/rigidlink/hm_read_rlink.F``.
    """
    id: int
    title: str = ""
    dofs: Tuple[int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1)
    skew_id: int = 0
    grnod_id: int = 0
    ipol: int = 0


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
    """/MERGE/RBODY (M102): Merge rigid bodies.

    Fortran origin: ``starter/source/constraints/general/merge/hm_read_merge.F``.
    """
    id: int
    title: str = ""
    items: List[Tuple[int, int, int, int, int]] = field(default_factory=list)  # (main_id, m_type, secon_id, s_type, iflag)


@dataclass
class IniCrackSegment:
    """Segment definition for /INICRACK."""
    node_id1: int
    node_id2: int
    ratio: float = 0.0


@dataclass
class IniCrack:
    """/INICRACK (M102): Initial crack definition for XFEM.

    Fortran origin: ``starter/source/initial_conditions/inicrack/hm_read_inicrack.F``.
    """
    id: int
    title: str = ""
    segments: List[IniCrackSegment] = field(default_factory=list)


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
    """/DAMP/INTER (M103): Interface / relative velocity damping.

    Fortran origin: ``starter/source/general_controls/damping/hm_read_damp.F``.
    """
    id: int
    title: str = ""
    nb_time_step: int = 0
    damp_range: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    grnod_id: int = 0
    skew_id: int = 0
    tstart: float = 0.0
    tstop: float = 0.0


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
    """/INIGRAV (M104): Initial gravity equilibrium state.

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


@dataclass
class IniMap1D:
    """/INIMAP1D (M104): 1D mapped field initial condition.

    Fortran origin: ``starter/source/initial_conditions/inimap/hm_read_inimap1d.F``.
    """
    id: int
    title: str = ""
    map_type: int = 0
    node_id1: int = 0
    node_id2: int = 0
    grbric_id: int = 0
    grquad_id: int = 0
    grsh3n_id: int = 0
    fscale_v: float = 1.0
    filename: str = ""


@dataclass
class IniMap2D:
    """/INIMAP2D (M104): 2D mapped field initial condition.

    Fortran origin: ``starter/source/initial_conditions/inimap/hm_read_inimap2d.F``.
    """
    id: int
    title: str = ""
    map_type: int = 0
    node_id1: int = 0
    node_id2: int = 0
    node_id3: int = 0
    grbric_id: int = 0
    fscale_v: float = 1.0
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
class LeakMat:
    """/LEAK/MAT or /LEAK (M105): Airbag fabric leakage model.

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
class UserWindow:
    """/USERWI (M106): User window / data card lines.

    Fortran origin: ``starter/source/starter/userwi.F`` / CFG ``userwi.cfg``.
    """
    lines: List[str] = field(default_factory=list)








