"""M40 residuals pack — the four M39 deferred small items (VALIDATION §7,
PORTING_GUIDE M39 "Deferred out of M39" list), each with a regression test.

Each item, its ROOT CAUSE, and the evidence-deck outcome:

1. DEGENERATE-BRICK ENGINE STALL (VALIDATION §7 item 3).  c12/c18 = the
   RD-V-0700 HEXA_DEGE decks: the M39 starter accepts a collapsed penta/
   pyramid /BRICK (`initialization._convert_degenerated_bricks`, 6-distinct
   nodes -> run-as-collapsed-hexa) but the M39 §3.3 measurement had the
   engine terminate at ~12 cycles / t=0.0045 with "NUMERICAL ENERGY
   INJECTION -31.9% EXCEEDS LIMIT 30.0%".  ROOT CAUSE: NOT the degenerate
   geometry — it is the M39 §3.4 /PROP/SOLID reader bug (the flag card ends
   in Dn=0.0, a FLOAT, defeating the "skip all-integer cards" heuristic, so
   `h` was read as Itetra4 = -1: a NEGATIVE Flanagan-Belytschko hourglass
   viscosity = an AMPLIFIER, HE runs negative and the injection guard fires
   in ~12 cycles).  That reader fix (a DIFFERENT builder's §3.4 work,
   committed on the M39 core) cures the root cause for EVERY V0700 brick,
   degenerate included.  RE-MEASURED on the M40 tree: c12 (LAW2) runs to
   t=10.46ms (35% of /RUN) and c18 (LAW36) to t=8.84ms (29%) — stable,
   ERROR% ~= 0, HE bounded/round-off, dt matching the Fortran node-controlled
   step (3.74e-4) — vs M39's 0.01-0.02% coverage.  The tests below are the
   degenerate-GEOMETRY anti-regression canary (a collapsed-hexa wedge must
   run STABLY with HE >= ~0); the fixed-format /PROP/SOLID reader itself is
   separately guarded by test_m39_solid_impvel.

2. M39-BUG-SPRPRE (VALIDATION §7 item 2 / §4.8).  RD-HWX-T-1010
   cantilever_completed (+DYREL) have a BLANK /PROP/SPR_PRE mass; the M39
   `mass > 0` check fired as their sole ERROR.  ROOT CAUSE: the port was
   STRICTER than Fortran.  hm_read_prop32.F reads MASS with HM_GET_FLOATV
   (blank -> 0) and NEVER checks it (only MSGID 408 F1/D1/E1/STIF1 over-
   specification and MSGID 406 zero LENGTH); the `MASS > 0` in
   prop_p32_spr_pre.cfg's CHECK block is a HyperMesh-GUI validation.
   VERIFIED by running the real starter_win64.exe on cantilever_completed:
   0 ERRORS, listing "MASS. . . = 0.000000000000".  FIX: TYPE32 removed from
   `spring._MASS_REQUIRED_SPRING_TYPES` — the port accepts a blank/zero
   pretensioner mass, the Engine refuses the (unported) group instead:
   ERROR -> SKIPS on both decks.

3. THE ROTATIONAL + SKEW EDGE (VALIDATION §7, flagged M39).  A rotational
   /IMPVEL/IMPDISP (dir XX/YY/ZZ -> dof 3..5) that ALSO names a /SKEW indexed
   `skews.axes[row][dof]` (dof 3..5) out of the (3,3) axes -> IndexError, and
   even guarded would have driven `v` not `vr`.  FIX (kinematics.apply_
   kinematic): dof 3..5 select axis `dof-3` and drive the ANGULAR velocity
   `vr` about it against the rotational inertia — the exact analogue fixvel.F
   runs (its SKEW-projection VV/A0/AA acts on the VR/AR arrays for a
   rotational DOF).  Tested both ways: dof<3-with-skew (translational,
   unchanged) and dof>=3-with-skew (the fixed rotational path).

4. c52 /PROP/SPR_PRE (TYPE32) ELEMENT PHYSICS (VALIDATION §7 item 3 tail).
   ASSESSED tractable but DEFERRED: the pretensioner kernel is ruser32.F
   (~260 lines: a 1-DOF axial spring, sensor-gated, four ITYP pretension
   laws, an Ilock retractor lock, per-element UVAR state) — a full ACTIVE
   spring in the shared kernel needing per-ITYP channel validation, not a
   residual.  Left InactiveProperty; the reader carries all the data a future
   port needs.  See prop_reader.parse_spr_pre for the full assessment.
"""

import contextlib
import io
import os

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import spring
from pyradioss.engine.engine import run_engine
from pyradioss.engine.kinematics import LoadsAndConstraints
from pyradioss.input import prop_reader
from pyradioss.input.card_layouts import fmt_float, fmt_int, fmt_str
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_prop
from pyradioss.model.entities import Part
from pyradioss.model.model import Model
from pyradioss.starter.initialization import build_element_groups
from pyradioss.starter.starter import run_starter

_LAW1 = "/MAT/LAW1/1\nsteel\n7.8e-6\n210. 0.3\n"
# const-1.0 curve: the driven face jumps 0 -> 1 at cycle 1 (the impulsive
# start that seeds the hourglass modes a negative viscosity would detonate)
_FUNCT = "/FUNCT/1\nconst\n0.0 1.0\n10.0 1.0\n"
_ENGINE = "/RUN/{n}/1\n0.02\n/DT\n0.9 0\n/PRINT/-1\n/STOP\n1000.0\n"


def _run(make_deck, name, starter, engine):
    """Run starter+engine; return (model, [ERROR% per cycle], [HE per cycle])."""
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    errs, he = [], []
    out_path = os.path.splitext(e)[0] + ".out"
    for line in open(out_path):
        tok = line.split()
        if len(tok) == 10 and tok[0].isdigit():
            try:
                errs.append(float(tok[9]))    # ERROR% column
                he.append(float(tok[5]))      # ENERGY-HE column
            except ValueError:
                pass
    return model, errs, he


# ============================================================================
# ITEM 1 — the collapsed-hexa (degenerate) /BRICK must run STABLY
# ============================================================================
# WEDGE_CONN is the classic collapse pattern '1 1 3 4 5 5 7 8' (bottom and
# top faces each collapsed along one edge -> a triangular prism), VERBATIM
# from /BRICK 1 of rd_v_failure/RD-V-0700/.../HEXA_DEGE.  It uses 6 distinct
# node ids (1,3,4,5,7,8), the coincident pairs being (n1=n2) and (n5=n6).


def _wedge_starter():
    # bottom triangle (1,3,4) fully fixed, top triangle (5,7,8) driven in +z:
    # the fixed-bottom / free-top asymmetry seeds the hourglass modes, so a
    # negative (misread h=-1) viscosity detonates and a correct h stays quiet
    return (
        "/BEGIN\nm40 wedge\n"
        "/NODE\n1 0 0 0\n3 1 1 0\n4 0 1 0\n5 0 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 1 3 4 5 5 7 8\n"           # brick_ID 1, conn = wedge
        "/PART/1\nwedge\n1 1\n" + _LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"    # free short form: h=0.1 > 0
        "/GRNOD/NODE/1\nbottom\n1 3 4\n"
        "/GRNOD/NODE/2\ntop\n5 7 8\n"
        "/BCS/1\nfix bottom\n111 111 0 1\n"
        "/IMPVEL/1\npush z\n1 Z 2 1.0\n" + _FUNCT + "/END\n")


def test_degenerate_wedge_kept_as_collapsed_hexa(make_deck):
    """The 6-distinct-node wedge survives as ONE /BRICK (collapsed hexa) —
    not rejected, not converted to a /TETRA4 (which is the 4-distinct path).
    This is the M39 starter unlock the engine stall sat downstream of."""
    s, _ = make_deck("WDGS", _wedge_starter(), "/RUN/X/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s)
    assert model.bricks is not None and model.bricks.n == 1
    assert model.tetras is None or model.tetras.n == 0


def test_degenerate_wedge_runs_stably_no_energy_injection(make_deck):
    """ITEM 1 canary.  Under an impulsive /IMPVEL start the collapsed-hexa
    wedge must run to the END, energy-balanced, with HOURGLASS ENERGY that is
    never significantly NEGATIVE (the amplifier signature that terminated
    c12/c18 at ~12 cycles / t=0.0045 with -31.9% injection in M39).  A
    healthy run prints hundreds of cycles here."""
    model, errs, he = _run(make_deck, "WDGR", _wedge_starter(),
                           _ENGINE.format(n="WDGR"))
    assert errs, "no ledger lines parsed"
    # ran to completion — no numerical-injection abort, no dt collapse
    assert not model.engine_state.stop_reason, model.engine_state.stop_reason
    assert model.engine_state.t == pytest.approx(0.02, rel=1e-6)
    # progressed FAR past the M39 ~12-cycle stall
    assert len(errs) > 50, f"only {len(errs)} cycles printed — stalled?"
    # every printed cycle balanced (no energy injection)
    assert max(abs(x) for x in errs) < 0.5, max(abs(x) for x in errs)
    # HE stays >= ~0: a NEGATIVE HE is the misread-h=-1 amplifier signature
    he_scale = max(1.0, max(abs(h) for h in he))
    assert min(he) > -1e-6 * he_scale, min(he)


def test_degenerate_wedge_timestep_healthy(make_deck):
    """The collapsed element must claim a POSITIVE, non-collapsing time step
    (the 'dt collapse from a degenerate direction' hypothesis of §7): the
    engine advanced a full t=0.02 in a bounded cycle count, so the mean step
    is far above zero and never underflowed the /DT floor."""
    model, errs, _ = _run(make_deck, "WDGT", _wedge_starter(),
                          _ENGINE.format(n="WDGT"))
    mean_dt = 0.02 / len(errs)
    assert mean_dt > 1e-6, mean_dt        # not collapsed to the noise floor
    assert model.engine_state.t == pytest.approx(0.02, rel=1e-6)


# ============================================================================
# ITEM 2 — a BLANK /PROP/SPR_PRE mass is legal Starter data (M39-BUG-SPRPRE)
# ============================================================================
# The card VERBATIM from RD-HWX-T-1010 cantilever_completed line 3028-3035:
# a BLANK mass field, Stif0 = 13744.468, fct_ID1 = 1 (all other fields
# blank).  The real starter_win64.exe accepts it (0 errors, MASS = 0.000).
_SPR_PRE_BLANK_MASS = (
    "/PROP/SPR_PRE/2\n"
    "prop_spring\n"
    "#                  M                                 sens_ID     llock\n"
    "                                                                      \n"
    "#              Stif0                  F1                  D1"
    "                  E1               Stif1\n"
    "           13744.468                                            "
    "                    \n"
    "#  fct_ID1   fct_ID2                                 Scale_t"
    "             Scale_d             Scale_f\n"
    "                   1                                                   \n")

FIXED_BEGIN = (
    "/BEGIN\nm40-sprpre\n      2021         0\n"
    "                  kg                  mm                  ms\n"
    "                  kg                  mm                  ms\n")


def _spr_pre_property(tmp_path, body):
    deck = tmp_path / "p_0000.rad"
    deck.write_text(FIXED_BEGIN + body + "/END\n")
    log, model = MessageLog(), Model()
    for b in read_deck(str(deck)):
        if b.parts and b.parts[0] == "PROP":
            read_prop(b, model, log)
    return model, log


def _spring_model(prop):
    m = Model()
    m.add_nodes(np.array([1, 2]), np.array([[0.0, 0, 0], [1.0, 0, 0]]))
    m.properties[prop.id] = prop
    m.parts[1] = Part(id=1, prop_id=prop.id, mat_id=0)
    m.raw_elems["SPRING"].append((1, 1, [1, 2]))
    return m


def test_spr_pre_blank_mass_reads_zero(tmp_path):
    """The blank M field reads as 0.0 (HM_GET_FLOATV's blank default), the
    property still carrying the real Stif0 and the time-function fct_ID2 (the
    '1' sits in the fct_ID2 column — the Fortran listing prints 'TIME
    DEPENDING F=f(t-t0)', i.e. the ITYP3 time-pretension law)."""
    model, _ = _spr_pre_property(tmp_path, _SPR_PRE_BLANK_MASS)
    p = model.properties[2]
    assert p.type == 32
    assert p.params["mass"] == 0.0
    assert p.params["stif0"] == pytest.approx(13744.468)
    assert p.params["fct_id1"] == 0
    assert p.params["fct_id2"] == 1


def test_spr_pre_blank_mass_no_starter_error(tmp_path):
    """ITEM 2: the SPRING INIT mass check must NOT fire on the blank-mass
    pretensioner — the Fortran Starter accepts it (verified against
    starter_win64.exe on cantilever_completed: 0 errors, MASS = 0.000)."""
    model, _ = _spr_pre_property(tmp_path, _SPR_PRE_BLANK_MASS)
    m = _spring_model(model.properties[2])
    log = MessageLog()
    build_element_groups(m, log)
    (_, group), = list(m.element_groups())
    spring.init_group(group, m, log)
    assert not any("mass must be" in e for e in log.errors), log.errors


def test_spr_pre_blank_mass_engine_refuses_group(tmp_path):
    """...and the deck is SKIPS, not ERROR: the pretensioner physics is
    unported, so the Engine refuses the TYPE32 group (the honest message the
    over-strict mass check was pre-empting)."""
    model, _ = _spr_pre_property(tmp_path, _SPR_PRE_BLANK_MASS)
    m = _spring_model(model.properties[2])
    build_element_groups(m, MessageLog())
    with pytest.raises(prop_reader.InactivePropertyError):
        prop_reader.refuse_inactive_properties(m)


# ============================================================================
# ITEM 3 — a rotational /IMPVEL naming a /SKEW drives vr about the skew axis
# ============================================================================
# A /SKEW/FIX rotated 45 deg about Z: its X' axis is (0.707, 0.707, 0) — a
# GENUINELY rotated axis, so a rotational /IMPVEL/XX in this skew must drive
# vr along that rotated axis (not global X), proving the skew is consumed.
_STEEL_TRUSS = (_LAW1 + "/PROP/TRUSS/1\nbar\n1.0\n")


def _v3(a, b, c):
    return fmt_float(a) + fmt_float(b) + fmt_float(c)


def _skew45z(sid=1):
    # yax=(-1,1,0), zax=(0,0,1) -> X' = Y' x Z' = (0.707, 0.707, 0)
    return (f"/SKEW/FIX/{sid}\nskew45z\n"
            + _v3(0.0, 0.0, 0.0) + "\n" + _v3(-1.0, 1.0, 0.0) + "\n"
            + _v3(0.0, 0.0, 1.0) + "\n")


def _imp_official(kind, iid, fct, direction, skew, grnod, scale=1.0):
    """The OFFICIAL fixed-column /IMPVEL card (the free short form has no
    skew field): fct Dir skew sens grnod frame Icoor / Ax Fy Tstart Tstop."""
    return (f"/{kind}/{iid}\nimp\n"
            + fmt_int(fct) + fmt_str(direction) + fmt_int(skew)
            + fmt_int(0) + fmt_int(grnod) + fmt_int(0) + fmt_int(0) + "\n"
            + fmt_float(1.0) + fmt_float(scale) + fmt_float(0.0)
            + fmt_float(1e30) + "\n")


def _skew_imp_model(direction, scale=2.0):
    """A 2-node truss with a rotated /SKEW and one /IMP* on node 2 naming it;
    returns the started model (skew_row resolved)."""
    import textwrap
    import tempfile
    d = tempfile.mkdtemp()
    starter = (
        "/BEGIN\nrotskew\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n/TRUSS/1\n1 1 2\n"
        "/PART/1\nbar\n1 1\n" + _STEEL_TRUSS +
        "/GRNOD/NODE/1\ndriven\n2\n" + _FUNCT +
        _skew45z(1) + _imp_official("IMPVEL", 1, 1, direction, 1, 1, scale)
        + "/END\n")
    p = os.path.join(d, "SK_0000.rad")
    open(p, "w").write(textwrap.dedent(starter))
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(p)


def test_skew_rotational_impvel_drives_vr_about_skew_axis():
    """ITEM 3 (dof>=3): a rotational /IMPVEL/XX naming a skew must NOT raise
    IndexError; it drives the ANGULAR velocity vr about the skew's X' axis
    (dof 3 -> axis 0), books work against the rotational inertia at the
    leapfrog midstep, and leaves the translation untouched."""
    m = _skew_imp_model("XX", scale=2.0)
    imp = m.impvel[0]
    assert imp.dof == 3 and getattr(imp, "skew_row", 0) == 1
    loads = LoadsAndConstraints(m, MessageLog())
    assert len(loads.skew_impvel) == 1 and loads.skew_impvel[0][2] == 3
    i2 = m.node_index(2)
    J, wimp = 2.5, 2.0
    inertia = np.full(m.numnod, J)
    v = np.zeros_like(m.x)
    vr = np.zeros_like(m.x)
    # a successful call IS the no-IndexError assertion
    w = loads.apply_kinematic(0.5, v, vr, m.mass, m.x, 1e-4,
                              np.zeros_like(v), inertia, np.zeros_like(vr))
    eX = m.skews.axes[1][0]                       # the skew X' axis (rotated)
    assert abs(eX[0] - 0.5 ** 0.5) < 1e-9 and abs(eX[2]) < 1e-12  # genuinely rotated
    assert np.allclose(vr[i2], wimp * eX)         # vr driven ALONG X'
    assert np.abs(v[i2]).max() == 0.0             # translation untouched
    # impulsive-start midstep work = 1/2 J w_imp^2 (angular analogue)
    assert w == pytest.approx(0.5 * J * wimp ** 2, rel=1e-12)


def test_skew_translational_impvel_preserved():
    """ITEM 3 (dof<3, existing behavior preserved): a TRANSLATIONAL /IMPVEL/X
    naming the SAME skew still drives the translational velocity v along X'
    and leaves the angular velocity vr untouched — byte-for-byte the pre-M40
    skew path."""
    m = _skew_imp_model("X", scale=2.0)
    imp = m.impvel[0]
    assert imp.dof == 0 and getattr(imp, "skew_row", 0) == 1
    loads = LoadsAndConstraints(m, MessageLog())
    i2 = m.node_index(2)
    wimp = 2.0
    v = np.zeros_like(m.x)
    vr = np.zeros_like(m.x)
    loads.apply_kinematic(0.5, v, vr, m.mass, m.x, 1e-4,
                          np.zeros_like(v), None, None)
    eX = m.skews.axes[1][0]
    assert np.allclose(v[i2], wimp * eX)          # v driven ALONG X'
    assert np.abs(vr[i2]).max() == 0.0            # rotation untouched


def test_skew_rotational_impdisp_finite_difference():
    """ITEM 3, the /IMPDISP sibling: a rotational /IMPDISP/YY in a skew must
    also stay on the rotational path — finite-difference the imposed angle
    onto vr about the skew's Y' axis (dof 4 -> axis 1), no IndexError."""
    import tempfile
    import textwrap
    d = tempfile.mkdtemp()
    starter = (
        "/BEGIN\nrotskewd\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n/TRUSS/1\n1 1 2\n"
        "/PART/1\nbar\n1 1\n" + _STEEL_TRUSS +
        "/GRNOD/NODE/1\ndriven\n2\n"
        "/FUNCT/1\nramp\n0.0 0.0\n10.0 10.0\n"        # d(t) = t
        + _skew45z(1) + _imp_official("IMPDISP", 1, 1, "YY", 1, 1, 1.0)
        + "/END\n")
    p = os.path.join(d, "SKD_0000.rad")
    open(p, "w").write(textwrap.dedent(starter))
    with contextlib.redirect_stdout(io.StringIO()):
        m = run_starter(p)
    assert m.impvel == [] and m.impdisp[0].dof == 4
    loads = LoadsAndConstraints(m, MessageLog())
    i2 = m.node_index(2)
    v = np.zeros_like(m.x)
    vr = np.zeros_like(m.x)
    loads.apply_kinematic(0.4, v, vr, m.mass, m.x, 0.05,
                          np.zeros_like(v), np.ones(m.numnod),
                          np.zeros_like(vr))
    eY = m.skews.axes[1][1]                        # the skew Y' axis
    assert np.allclose(vr[i2], 1.0 * eY)           # ramp slope 1.0 about Y'
    assert np.abs(v[i2]).max() == 0.0


# ============================================================================
# ITEM 4 — /PROP/SPR_PRE (TYPE32) stays InactiveProperty (physics deferred)
# ============================================================================

def test_spr_pre_type32_stays_inactive_property(tmp_path):
    """ITEM 4: the pretensioner physics (ruser32.F) is a full active-spring
    port, not a residual — TYPE32 stays an InactiveProperty and the Engine
    refuses element groups that use it.  The reader STILL carries every datum
    a future kernel needs (mass, Stif0/Stif1, F1/D1/E1, fct_ID1/2, ilock,
    sens_id) so the port lands cleanly when the kernel is written."""
    # RD-V-0031 pretensioner #1 (ITYP1: F1/D1 set): the c52 verification card
    body = (
        "/PROP/SPR_PRE/1\nSpring pretentioner 1\n"
        "#                  M                               sensor_ID"
        "     Ilock\n"
        "                1E-5                                       0"
        "         1\n"
        "#              Stif0                  F1                  D1"
        "                  E1               Stif1\n"
        "                2000                1000                  50"
        "                   0                   0\n"
        "#funct_ID1 funct_ID2\n         0         0\n")
    model, _ = _spr_pre_property(tmp_path, body)
    p = model.properties[1]
    assert getattr(p, "inactive", False)          # Engine will refuse it
    assert p.prop_name == "SPR_PRE" and p.type == 32
    # the data a future ruser32.F port consumes is all present
    assert p.params["mass"] == pytest.approx(1e-5)
    assert p.params["stif0"] == pytest.approx(2000.0)
    assert p.params["f1"] == pytest.approx(1000.0)
    assert p.params["d1"] == pytest.approx(50.0)
    assert p.params["ilock"] == 1
    assert p.params["k"] == pytest.approx(2000.0)  # STIFM = Stif0 + Stif1


def test_spr_pre_type32_not_in_mass_required_set():
    """The corollary of the item-2 fix: TYPE32 is no longer mass-checked (the
    Fortran Starter does not require it), TYPE4 still is."""
    assert 32 not in spring._MASS_REQUIRED_SPRING_TYPES
    assert 4 in spring._MASS_REQUIRED_SPRING_TYPES
