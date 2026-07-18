"""
M39 small-bug pack — the M38 quick-fix backlog (VALIDATION.md §7 item 2 /
``coverage_results_m38.json`` ``delta_vs_m37.new_bugs``), plus the M37
degenerate-brick leftover.

Every fixture is a VERBATIM card from the deck that surfaced the bug, so
these are regressions against the real corpus, not invented input:

* **M38-NEW-2** — the null-density MAT CHECK false-fires on /MAT/VOID
  (LAW0), which is massless BY DESIGN (RD-E-2700 BAT_CIR / BAT_SQR); and
  the ``_ALLOWED_LAWS`` family map rejected LAW0 on /BEAM and /TRUSS even
  though ``hm_read_mat00.F`` declares the void law BEAM_ALL/TRUSS
  compatible (the M38 prop-pack's OPEN item).
* **M38-NEW-1** — the /SPRING mass check misapplied TYPE4's requirement to
  /PROP/SPR_PRE (TYPE32), whose real mass field the port never read
  (RD-V-0031).
* **M38-NEW-4** — the RBODY shared-slave-node check hard-errored where
  ``checkrby.F`` only errors when EVERY sharing body is ACTIVE; the
  RD-E-1200 BIKERC bicycle shares nodes between /SENSOR-gated bodies.
* **M37 leftover** — degenerated /BRICK cards with 5/6/7 distinct nodes
  (collapsed penta/pyramid) were refused; the reference runs them through
  the ordinary 8-node kernel with the repeated connectivity as written
  (RD-V-0700 HEXA_DEGE, RD_V_0240 Modele_HEXA_P14_Isolid24).

The checks that MUST still fire (a spring that genuinely has no mass, two
genuinely ACTIVE bodies sharing a node, a brick collapsed past any 3-D
shape) get their own tests here — the fixes narrow the checks to what the
reference checks, they do not remove them.
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8, spring
from pyradioss.input import prop_reader
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_prop, read_rbody
from pyradioss.model.entities import Material, Part, Property, RigidBody
from pyradioss.model.model import Model
from pyradioss.starter import checks
from pyradioss.starter.initialization import (
    _convert_degenerated_bricks, build_element_groups,
    initialize_rigid_bodies)

# fixed 10/20-char dialect header (Invers >= 90 -> column-cut reading)
FIXED_BEGIN = (
    "/BEGIN\nm39-unit-test\n      2021         0\n"
    "                  kg                  mm                  ms\n"
    "                  kg                  mm                  ms\n")


def _parse_blocks(tmp_path, body, kw, fn):
    """Parse a deck body and feed every ``kw`` block to ``fn``."""
    deck = tmp_path / "d.rad"
    deck.write_text(FIXED_BEGIN + body + "/END\n")
    log = MessageLog()
    model = Model()
    for b in read_deck(str(deck)):
        if b.parts and b.parts[0] == kw:
            fn(b, model, log)
    return model, log


# ============================================================================
# M38-NEW-2 — /MAT/VOID (LAW0) is massless by design
# ============================================================================
# Evidence: rd_e/RD-E-2700_Football/27_Football_shoot/Bathenay_circular/
# BAT_CIR_0000.rad declares /MAT/VOID/12 with RHO0 = 0 and uses it on skin
# shells + sh3n.  Upstream hm_read_mat00.F applies NO positivity check to
# RHO0 (it guards the one division: SDSP = SQRT(YOUNG/MAX(RHOR,EM20))), and
# the cfg is decisive: matl_void0.cfg's CHECK block demands MAT_RHO >= 0,
# where every load-bearing law's cfg demands MAT_RHO > 0
# (matl2_plas_johns.cfg).


def _void_shell_model(law=0, rho0=0.0, family="SHELL"):
    """A one-element model whose part carries a null-density material."""
    m = Model()
    m.add_nodes(np.array([1, 2, 3, 4]),
                np.array([[0.0, 0, 0], [1.0, 0, 0],
                          [1.0, 1.0, 0], [0.0, 1.0, 0]]))
    m.materials[12] = Material(id=12, law=law, rho0=rho0,
                               title="void", params={"E": 0.0, "nu": 0.0})
    m.properties[1] = Property(id=1, type=0,
                               params=prop_reader._universal_geo_params())
    m.parts[1] = Part(id=1, prop_id=1, mat_id=12)
    nodes = {"SHELL": [1, 2, 3, 4], "SH3N": [1, 2, 3],
             "TRUSS": [1, 2], "BEAM": [1, 2, 3]}[family]
    m.raw_elems[family].append((1, 1, nodes))
    return m


def test_void_null_density_accepted_on_shells():
    """/MAT/VOID/12 with RHO0=0 must NOT raise the null-density MAT CHECK
    (BAT_CIR / BAT_SQR — M38-NEW-2)."""
    m = _void_shell_model()
    log = MessageLog()
    build_element_groups(m, log)
    checks.check_model(m, log)
    assert not any("initial density" in e for e in log.errors), log.errors


def test_void_null_density_accepted_on_sh3n():
    m = _void_shell_model(family="SH3N")
    log = MessageLog()
    build_element_groups(m, log)
    checks.check_model(m, log)
    assert not any("initial density" in e for e in log.errors), log.errors


@pytest.mark.parametrize("law", [1, 2, 36])
def test_null_density_still_fatal_for_load_bearing_laws(law):
    """The exemption is LAW0-only: a real law with RHO0 = 0 still errors
    (the check is narrowed, not removed)."""
    m = _void_shell_model(law=law, rho0=0.0)
    log = MessageLog()
    build_element_groups(m, log)
    checks.check_model(m, log)
    assert any("initial density" in e for e in log.errors), log.errors


def test_void_law0_allowed_on_beam_and_truss_families():
    """hm_read_mat00.F tags LAW0 SOLID_ISOTROPIC / SHELL_ISOTROPIC /
    SPRING_MATERIAL / BEAM_ALL / TRUSS / SPH — every family.  The port's
    _ALLOWED_LAWS map used to stop at shells/solids, so a void beam passed
    the PROPERTY check (prop_type_ok is already universal) and failed the
    MATERIAL one (the M38 prop-pack OPEN item)."""
    for family in ("trusses", "beams", "shells", "sh3n", "bricks", "tetras"):
        assert 0 in checks._ALLOWED_LAWS[family], family


def test_void_law0_exempt_set():
    assert 0 in checks._NULL_RHO0_OK_LAWS          # /MAT/VOID
    assert 51 in checks._NULL_RHO0_OK_LAWS         # multimaterial ALE (M38)
    assert 151 in checks._NULL_RHO0_OK_LAWS
    assert 1 not in checks._NULL_RHO0_OK_LAWS      # LAW1 still needs RHO0


def test_void_shell_dt_factor_survives_null_density():
    """The shell time-step factor must not divide by the null density —
    the guard solid_hexa8._exact_dt_factor already had, and the exact
    reason BAT_CIR crashed with ZeroDivisionError once the MAT CHECK let
    the deck through (upstream: the element claims no time step, SSP=0)."""
    from pyradioss.elements import shell_bt4, shell_tri3
    m = _void_shell_model()
    log = MessageLog()
    build_element_groups(m, log)
    for name, group in m.element_groups():
        node_idx, mass_c, _in = shell_bt4.init_group(group, m, log) \
            if name == "shells" else (None, None, None)
    assert not log.errors, log.errors


def test_void_truss_sound_speed_guarded():
    """A /MAT/VOID truss (E = 0, RHO0 = 0) must give c = 0 -> no time-step
    claim, never a 0/0 NaN.  Mirrors hm_read_mat00.F's own guard
    SDSP = SQRT(YOUNG/MAX(RHOR,EM20))."""
    from pyradioss.elements import truss
    m = _void_shell_model(family="TRUSS")
    m.properties[1].params["area"] = 1.0
    log = MessageLog()
    build_element_groups(m, log)
    (name, group), = list(m.element_groups())
    truss.init_group(group, m, log)
    m.x = m.x0.copy()
    fint = np.zeros((m.numnod, 3))
    dtc = truss.forces(group, m.x, np.zeros((m.numnod, 3)),
                       np.zeros((m.numnod, 3)), 1e-3, fint, None)
    assert np.all(np.isfinite(dtc)), dtc          # never NaN
    assert np.all(dtc > 1e20)                     # claims no dt of its own
    assert np.allclose(fint, 0.0)                 # void carries no force


# ============================================================================
# M38-NEW-1 — /PROP/SPR_PRE (TYPE32) carries a real mass
# ============================================================================
# /PROP/SPR_PRE/1 VERBATIM from rd_v_elements/RD-V-0031_Pretensioner_
# Spring_Type_32/0031_spring_type_32/Spring_TYPE32_element_0000.rad.
# cfg prop_p32_spr_pre.cfg (radioss100):
#   CARD("%20lg                              %10d%10d", MASS, ISENSOR, ILock)
#   CARD("%20lg%20lg%20lg%20lg%20lg", STIFF0, F1, D1, E1, STIFF1)
#   CARD("%10d%10d                    %20lg%20lg%20lg", FUN_A1, FUN_B1, ...)

SPR_PRE_1 = (
    "/PROP/SPR_PRE/1\n"
    "Spring pretentioner 1\n"
    "#                  M                               sensor_ID     Ilock\n"
    "                1E-5                                       0         1\n"
    "#              Stif0                  F1                  D1"
    "                  E1               Stif1\n"
    "                2000                1000                  50"
    "                   0                   0\n"
    "#funct_ID1 funct_ID2\n"
    "         0         0\n")


def test_spr_pre_parses_the_real_mass(tmp_path):
    """The card's M = 1E-5 must reach the property.  Before M39 the
    TYPE32 spelling fell through to the generic InactiveProperty branch and
    got _universal_geo_params()'s placeholder mass = 0.0 — which the
    /SPRING kernel then reported as '/PROP/SPRING mass must be > 0'."""
    model, log = _parse_blocks(tmp_path, SPR_PRE_1, "PROP", read_prop)
    p = model.properties[1]
    assert p.type == 32
    assert p.params["mass"] == pytest.approx(1e-5)
    assert p.params["stif0"] == pytest.approx(2000.0)
    assert p.params["f1"] == pytest.approx(1000.0)
    assert p.params["d1"] == pytest.approx(50.0)
    assert p.params["ilock"] == 1
    assert p.params["sens_id"] == 0
    # hm_read_prop32.F RINI32: STIFM(I) = STIF0 + STIF1
    assert p.params["k"] == pytest.approx(2000.0)
    # physics still not ported -> the Engine must still refuse it
    assert getattr(p, "inactive", False)
    assert p.prop_name == "SPR_PRE"


def _spring_model(prop):
    m = Model()
    m.add_nodes(np.array([1, 2]), np.array([[0.0, 0, 0], [1.0, 0, 0]]))
    m.properties[prop.id] = prop
    m.parts[1] = Part(id=1, prop_id=prop.id, mat_id=0)
    m.raw_elems["SPRING"].append((1, 1, [1, 2]))
    return m


def test_spr_pre_spring_no_false_mass_error(tmp_path):
    """RD-V-0031: the SPRING INIT mass check must not fire on a TYPE32
    whose card carries M = 1E-5 (M38-NEW-1)."""
    model, log = _parse_blocks(tmp_path, SPR_PRE_1, "PROP", read_prop)
    m = _spring_model(model.properties[1])
    log = MessageLog()
    build_element_groups(m, log)
    (name, group), = list(m.element_groups())
    spring.init_group(group, m, log)
    assert not any("mass must be" in e for e in log.errors), log.errors


def test_type4_spring_zero_mass_still_errors():
    """The mass requirement is NARROWED, not removed: a genuine /PROP/
    SPRING (TYPE4) with mass 0 has no stable time step and must still be
    a fatal Starter error."""
    p = Property(id=7, type=4, params={"mass": 0.0, "k": 100.0, "c": 0.0})
    m = _spring_model(p)
    log = MessageLog()
    build_element_groups(m, log)
    (name, group), = list(m.element_groups())
    spring.init_group(group, m, log)
    assert any("mass must be" in e for e in log.errors), log.errors
    assert any("/PROP/SPRING/7" in e for e in log.errors), log.errors


def test_spr_pre_zero_mass_accepted_like_fortran(tmp_path):
    """A blank/zero TYPE32 pretensioner mass is LEGAL (M40, M39-BUG-SPRPRE).
    The real Fortran Starter does NOT enforce MASS > 0 for /PROP/SPR_PRE:
    hm_read_prop32.F reads MASS with HM_GET_FLOATV (a blank field gives 0)
    and never checks it — its only errors are the F1/D1/E1/STIF1 over-
    specification (MSGID 408) and the zero spring LENGTH (MSGID 406).  The
    ``MASS > 0`` in prop_p32_spr_pre.cfg's CHECK block is a HyperMesh-GUI
    validation, not a Starter one.  Verified by running the real
    starter_win64.exe on RD-HWX-T-1010 cantilever_completed (whose SPR_PRE
    card has a BLANK mass): 0 errors, listing ``MASS = 0.000000000000``.
    So the port must NOT error — it reads the mass (0) and lets the Engine
    refuse the group for the honest reason (unported pretensioner physics),
    which is the ERROR->SKIPS the M39 mass check wrongly pre-empted."""
    body = SPR_PRE_1.replace("                1E-5", "                   0")
    model, log = _parse_blocks(tmp_path, body, "PROP", read_prop)
    assert model.properties[1].params["mass"] == 0.0     # blank -> 0, read
    m = _spring_model(model.properties[1])
    log = MessageLog()
    build_element_groups(m, log)
    (name, group), = list(m.element_groups())
    spring.init_group(group, m, log)
    assert not any("mass must be" in e for e in log.errors), log.errors
    # ...but the unported pretensioner physics still refuses the group,
    # so the deck is SKIPS (engine-refused), not a Starter ERROR.
    with pytest.raises(prop_reader.InactivePropertyError):
        prop_reader.refuse_inactive_properties(m)


def test_inactive_spring_prop_gets_no_placeholder_mass_error():
    """A spring spelling whose mass the port does NOT read (e.g. SPR_PUL
    TYPE12) carries _universal_geo_params()'s placeholder 0.0.  Mass-
    checking a placeholder invents a deck error that does not exist; the
    InactiveProperty refusal is the honest message instead."""
    p = prop_reader.InactiveProperty(
        id=9, type=12, params=prop_reader._universal_geo_params(),
        prop_name="SPR_PUL")
    m = _spring_model(p)
    log = MessageLog()
    build_element_groups(m, log)
    (name, group), = list(m.element_groups())
    spring.init_group(group, m, log)
    assert not any("mass must be" in e for e in log.errors), log.errors
    # ... and the Engine still refuses the model
    with pytest.raises(prop_reader.InactivePropertyError):
        prop_reader.refuse_inactive_properties(m)


def test_mass_required_spring_types_scope():
    # M40 (M39-BUG-SPRPRE): TYPE32 removed — the Fortran Starter does not
    # enforce MASS > 0 for /PROP/SPR_PRE (hm_read_prop32.F has no mass check;
    # the cfg CHECK is HyperMesh-GUI-only, confirmed by running starter_win64
    # on cantilever_completed).  Only TYPE4 /SPRING genuinely requires it.
    assert spring._MASS_REQUIRED_SPRING_TYPES == frozenset({4})


# ============================================================================
# M38-NEW-4 — RBODY shared slave nodes are legal between SENSOR-gated bodies
# ============================================================================
# /RBODY/23 'all' and /RBODY/24 'all less wheels' VERBATIM from
# rd_e/RD-E-1200_Bicycle/12_Bicycle/Bike/BIKERC_1506_1104_0000.rad — the
# bicycle defines deliberately OVERLAPPING bodies, each gated by its own
# /SENSOR (ISENS 10 and 7), so only one owns the shared nodes at a time.
# checkrby.F errors (MSGID 3121) only when IACTI > 0, i.e. every sharing
# body is ACTIVE; if any is sensor-gated (NPBY(7)==0 <- sens_ID != 0) it
# warns instead (MSGID 1026, "sensor effect will be treated later").
# hm_read_rbody.F: IF(ISENS == 0) NPBY(7,NRB)=1 ELSE NPBY(7,NRB)=0.

RBODY_23 = (
    "/RBODY/23\n"
    "all\n"
    "#     RBID     ISENS     NSKEW    ISPHER                MASS   Gnod_id"
    "     IKREM      ICOG   Surf_id\n"
    "    322917        10         0         0                   0       264"
    "         0         2         0\n"
    "#                Jxx                 Jyy                 Jzz\n"
    "                   0                   0                   0\n")


def test_rbody_reads_sens_id_from_the_real_card(tmp_path):
    """The port warns sens_ID as ignored (it does not gate the kinematics)
    but must CARRY the value: the shared-node check needs the reference's
    ACTIVE/INACTIVE distinction."""
    model, log = _parse_blocks(tmp_path, RBODY_23, "RBODY", read_rbody)
    rb, = model.rbodies
    assert rb.id == 23
    assert rb.master_id == 322917
    assert rb.grnod_id == 264
    assert rb.sens_id == 10                   # <- the fix
    assert rb.icog == 2 or rb.icog in (0, 1)  # ICoG=2 approximated (warned)


def _two_body_model(sens_a, sens_b):
    """Two rigid bodies whose slave groups OVERLAP on node 3."""
    from pyradioss.model.model import NodeGroup
    m = Model()
    ids = np.array([1, 2, 3, 4, 90, 91])
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0],
                  [3.0, 0, 0], [0.5, 1.0, 0], [2.5, 1.0, 0]])
    m.add_nodes(ids, x)
    m.mass = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    m.inertia = np.zeros(6)
    m.x = m.x0.copy()
    m.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0, 1, 2]))
    m.node_groups[2] = NodeGroup(id=2, node_idx=np.array([2, 3]))
    m.rbodies.append(RigidBody(id=1, kind="RBODY", master_id=90, grnod_id=1,
                               jadd=np.zeros(3), icog=1, sens_id=sens_a))
    m.rbodies.append(RigidBody(id=2, kind="RBODY", master_id=91, grnod_id=2,
                               jadd=np.zeros(3), icog=1, sens_id=sens_b))
    return m


def test_rbody_shared_node_between_sensor_gated_bodies_is_a_warning():
    """BIKERC: both bodies /SENSOR-gated -> checkrby.F's IACTI == 0 branch
    -> MSGID 1026 WARNING, not an error (M38-NEW-4)."""
    m = _two_body_model(sens_a=10, sens_b=7)
    log = MessageLog()
    initialize_rigid_bodies(m, log)
    assert not any("already belong" in e for e in log.errors), log.errors
    assert any("shares slave node" in w for w in log.warnings), log.warnings
    # both bodies actually initialize
    assert m.rbodies[0].mass_total > 0
    assert m.rbodies[1].mass_total > 0


def test_rbody_shared_node_between_active_bodies_still_errors():
    """The check is NARROWED, not removed: two bodies with no sensor are
    both ACTIVE from t=0, so sharing a node is a genuine over-constraint —
    checkrby.F's IACTI > 0 branch, MSGID 3121."""
    m = _two_body_model(sens_a=0, sens_b=0)
    log = MessageLog()
    initialize_rigid_bodies(m, log)
    assert any("already belong" in e for e in log.errors), log.errors


def test_rbody_shared_node_one_active_one_gated_is_a_warning():
    """IACTI = 0 as soon as ANY sharing body is inactive (checkrby.F sets
    IACTI=0 inside the loop over the sharing bodies)."""
    m = _two_body_model(sens_a=0, sens_b=7)
    log = MessageLog()
    initialize_rigid_bodies(m, log)
    assert not any("already belong" in e for e in log.errors), log.errors


# ============================================================================
# M37 leftover — degenerated /BRICK = collapsed hexa, run as written
# ============================================================================
# /BRICK 1 VERBATIM from rd_v_failure/RD-V-0700/.../HEXA_DEGE/
# HEXA_DEGE_0000.rad:
#     brick_ID  n1  n2  n3  n4  n5  n6  n7  n8
#            1   1   1   3   4   5   5   7   8
# = the classic collapsed-hexa WEDGE (bottom face 1-1-3-4 -> triangle
# 1-3-4, top face 5-5-7-8 -> triangle 5-7-8): 6 distinct nodes.
# The reference keeps the 8-node connectivity and runs it through the
# ordinary brick kernel (degenes8.F only COUNTS the collapse); its Starter
# reports NUMELS = 40 for this deck's 40 wedges.

_HEXA_DEGE_COORDS = {
    1: (0.0, 0.0, 0.0), 2: (10.0, 0.0, 0.0), 3: (10.0, 10.0, 0.0),
    4: (0.0, 10.0, 0.0), 5: (0.0, 0.0, 10.0), 6: (10.0, 0.0, 10.0),
    7: (10.0, 10.0, 10.0), 8: (0.0, 10.0, 10.0),
}
WEDGE_CONN = [1, 1, 3, 4, 5, 5, 7, 8]        # /BRICK 1 verbatim


def _degen_model(conns):
    m = Model()
    ids = np.array(sorted(_HEXA_DEGE_COORDS))
    x = np.array([_HEXA_DEGE_COORDS[i] for i in ids])
    m.add_nodes(ids, x)
    m.materials[1] = Material(id=1, law=1, rho0=7.8e-6,
                              params={"E": 210.0, "nu": 0.3})
    m.properties[1] = Property(id=1, type=14,
                               params=prop_reader._universal_geo_params())
    m.parts[1] = Part(id=1, prop_id=1, mat_id=1)
    for k, c in enumerate(conns, start=1):
        m.raw_elems["BRICK"].append((k, 1, c))
    return m


def test_degenerate_wedge_is_kept_as_a_collapsed_hexa():
    """The 6-distinct-node wedge must survive as a /BRICK with its
    repeated connectivity AS WRITTEN — not be refused, not be rewritten."""
    m = _degen_model([WEDGE_CONN])
    log = MessageLog()
    _convert_degenerated_bricks(m, log)
    assert not log.errors, log.errors
    assert len(m.raw_elems["BRICK"]) == 1
    assert not m.raw_elems["TETRA4"]
    eid, pid, nodes = m.raw_elems["BRICK"][0]
    assert nodes == WEDGE_CONN               # repeats preserved


def test_degenerate_wedge_volume_is_exact():
    """The 1-point centroid rule on the collapsed hexa integrates the
    wedge volume EXACTLY: half of the 10x10x10 cube = 500.  (This is why
    the port's TOTAL MASS / COG on HEXA_DEGE match the reference Starter's
    0.156 and (33.75, -36.25, 5.0) to every printed digit.)"""
    xe = np.array([[_HEXA_DEGE_COORDS[n] for n in WEDGE_CONN]])
    dndx, vol = solid_hexa8._geometry(xe)
    assert vol[0] == pytest.approx(500.0)


def test_degenerate_wedge_pair_tiles_the_cube():
    """/BRICK 1 and /BRICK 21 of HEXA_DEGE's part 1 are the two wedges the
    cube is split into — together they must give the cube's volume."""
    other = [1, 1, 2, 3, 5, 5, 6, 7]          # /BRICK 21 verbatim
    tot = 0.0
    for conn in (WEDGE_CONN, other):
        xe = np.array([[_HEXA_DEGE_COORDS[n] for n in conn]])
        tot += float(solid_hexa8._geometry(xe)[1][0])
    assert tot == pytest.approx(1000.0)


def test_degenerate_wedge_mass_lumping_doubles_the_collapsed_node():
    """A repeated node collects its coincident corners' shares: local 0+1
    are both node 1, so it takes 2/8 of the element mass (the reference's
    collapsed lumping — verified against its COG, a mass-weighted
    quantity, on HEXA_DEGE)."""
    m = _degen_model([WEDGE_CONN])
    log = MessageLog()
    build_element_groups(m, log)
    (name, group), = list(m.element_groups())
    node_idx, mass_c, _ = solid_hexa8.init_group(group, m, log)
    assert not log.errors, log.errors
    nodal = np.zeros(m.numnod)
    np.add.at(nodal, node_idx, mass_c)
    emass = 7.8e-6 * 500.0
    assert nodal.sum() == pytest.approx(emass)          # nothing lost
    assert nodal[m.node_index(1)] == pytest.approx(emass * 2 / 8)   # collapsed
    assert nodal[m.node_index(5)] == pytest.approx(emass * 2 / 8)   # collapsed
    assert nodal[m.node_index(3)] == pytest.approx(emass * 1 / 8)   # ordinary


def test_degenerate_wedge_hourglass_is_orthogonal_to_real_motion():
    """The Flanagan-Belytschko gammas must annihilate rigid-body motion and
    uniform strain on the COLLAPSED element too — otherwise the hourglass
    damper would fight genuine deformation."""
    xe = np.array([[_HEXA_DEGE_COORDS[n] for n in WEDGE_CONN]])
    dndx, vol = solid_hexa8._geometry(xe)
    hx = solid_hexa8._H @ xe
    gamma = solid_hexa8._H[None, :, :] - hx @ dndx.transpose(0, 2, 1)
    X = xe[0]
    fields = {
        "translate": np.tile([1.0, 0.0, 0.0], (8, 1)),
        "rotate_z": np.stack([-X[:, 1], X[:, 0], np.zeros(8)], axis=1),
        "stretch_x": np.stack([X[:, 0], np.zeros(8), np.zeros(8)], axis=1),
        "shear_xy": np.stack([X[:, 1], np.zeros(8), np.zeros(8)], axis=1),
    }
    for nm, v in fields.items():
        assert np.abs(gamma[0] @ v).max() < 1e-9, nm


def test_fully_collapsed_brick_still_becomes_a_tetra():
    """The 4-distinct pattern keeps its /TETRA4 promotion (the port's
    constant-strain tetra is the better element there)."""
    m = _degen_model([[1, 2, 3, 3, 5, 5, 5, 5]])
    log = MessageLog()
    _convert_degenerated_bricks(m, log)
    assert not log.errors, log.errors
    assert not m.raw_elems["BRICK"]
    assert m.raw_elems["TETRA4"] == [(1, 1, [1, 2, 3, 5])]


def test_brick_collapsed_past_a_solid_still_errors():
    """Fewer than 4 distinct nodes has no volume to integrate — still a
    hard error (the refusal is narrowed to the genuinely broken case)."""
    m = _degen_model([[1, 1, 1, 1, 5, 5, 5, 5]])
    log = MessageLog()
    _convert_degenerated_bricks(m, log)
    assert any("distinct node" in e for e in log.errors), log.errors


def test_full_hexa_unchanged():
    """Regression guard: an ordinary 8-distinct brick is untouched."""
    m = _degen_model([[1, 2, 3, 4, 5, 6, 7, 8]])
    log = MessageLog()
    _convert_degenerated_bricks(m, log)
    assert not log.errors, log.errors
    assert m.raw_elems["BRICK"] == [(1, 1, [1, 2, 3, 4, 5, 6, 7, 8])]
    assert not m.raw_elems["TETRA4"]
