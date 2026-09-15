"""
M38 /PROP pack — parsing + physics for the corpus-blocker properties.

Covers, per the milestone brief:

* parse fixtures from REAL fixed-format corpus cards (SH_ORTH, SPR_GENE,
  SPR_BEAM, VOID, INJECT1, the E0500 /PROP/BEAM section card);
* the SH_ORTH TYPE9 orthotropy orientation math vs a hand calc, and the
  strain/stress frame rotation's energy invariance;
* the TYPE8 SPR_GENE 6-DOF spring force/moment vs the closed form;
* VOID (TYPE0) no-stiffness semantics + universal family compatibility;
* the INJECT1 InactiveProperty flow (Starter accepts, Engine refuses only
  when elements use it);
* the mat_ID = 0 /PART rule (legal on a spring part, still rejected on a
  solid part);
* LAW19 fabric orientation end-to-end: the fiber angle changes the shell
  stress (kernel level, cfg-free) and the RD-V-0230 corpus deck runs to a
  NORMAL Starter termination (guarded on the cfg tree).
"""

import contextlib
import io
import math
import os

import numpy as np
import pytest

from pyradioss import materials
from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_bt4, shell_ortho, spring, spring_general
from pyradioss.input import mat_reader, prop_reader
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_prop
from pyradioss.model.entities import Material, Part, Property
from pyradioss.model.model import ElementGroup, Model
from pyradioss.starter.initialization import build_element_groups

# fixed 10/20-char dialect header (Invers >= 90 -> column-cut reading)
FIXED_BEGIN = (
    "/BEGIN\nm38-unit-test\n      2021         0\n"
    "                  kg                  mm                  ms\n"
    "                  kg                  mm                  ms\n")

_HAS_CFG = mat_reader.catalogue().schema("FABRI") is not None


def _card(*fields) -> str:
    """One fixed card from (value, width) fields, each right-justified in
    its column (the %10d / %20lg columns of the cfg FORMAT)."""
    return "".join(f"{v:>{w}}" for v, w in fields)


def _parse_props(tmp_path, body):
    """Parse a /PROP body (fixed dialect) through read_prop and return
    (model, log)."""
    deck = tmp_path / "p.rad"
    deck.write_text(FIXED_BEGIN + body + "/END\n")
    log = MessageLog()
    model = Model()
    for b in read_deck(str(deck)):
        if b.parts and b.parts[0] == "PROP":
            read_prop(b, model, log)
    return model, log


# ============================================================================
# Parse fixtures — REAL corpus cards
# ============================================================================

# /PROP/SH_ORTH/1 verbatim from RD-V-0230 (0230_shell_mat_019_01)
SH_ORTH_00 = (
    "/PROP/SH_ORTH/1\n"
    "Shell 00 degre\n"
    "        24        11         0         0                                "
    "       0\n"
    "                   0                   0                   0             "
    "      0                   0\n"
    "         1         0                   1                   0             "
    "      0         0\n"
    "                   1                   0                   0             "
    "      0\n")

SH_ORTH_45 = SH_ORTH_00.replace("/PROP/SH_ORTH/1", "/PROP/SH_ORTH/3").replace(
    "Shell 00 degre", "Shell 45 degre").replace(
    "                   1                   0                   0             "
    "      0\n",
    "                   1                   1                   0             "
    "      0\n")


def test_sh_orth_parse(tmp_path):
    model, log = _parse_props(tmp_path, SH_ORTH_00)
    assert not log.errors, log.errors
    p = model.properties[1]
    assert p.type == 9
    assert p.title == "Shell 00 degre"
    assert p.params["thick"] == pytest.approx(1.0)
    assert p.params["nip"] == 1
    assert (p.params["vx"], p.params["vy"], p.params["vz"]) == (1.0, 0.0, 0.0)
    assert p.params["phi"] == pytest.approx(0.0)


def test_sh_orth_45_vector(tmp_path):
    model, log = _parse_props(tmp_path, SH_ORTH_45)
    p = model.properties[3]
    assert (p.params["vx"], p.params["vy"]) == (1.0, 1.0)


def test_spr_gene_parse(tmp_path):
    # header: Mass Inertia skew sens Isflag Ifail Ifail2 Iequil
    head = _card((2.0, 20), (3.0, 20), (0, 10), (0, 10), (0, 10), (0, 10),
                 (0, 10), (0, 10))
    blocks = []
    for i in range(1, 7):
        kc = _card((10.0 * i, 20), (0.5 * i, 20), (0, 20), (0, 20), (0, 20))
        func = _card((0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10),
                     (0, 20), (0, 20))
        fe = _card((0, 20), (0, 20), (0, 20), (0, 20))
        blocks += [kc, func, fe]
    rate = _card((0, 10), (0, 20))
    body = ("/PROP/SPR_GENE/7\nspr gene test\n"
            + head + "\n" + "\n".join(blocks) + "\n" + rate + "\n")
    model, log = _parse_props(tmp_path, body)
    assert not log.errors, log.errors
    p = model.properties[7]
    assert p.type == 8
    assert p.params["mass"] == pytest.approx(2.0)
    assert p.params["inertia"] == pytest.approx(3.0)
    for i in range(1, 7):
        assert p.params[f"k{i}"] == pytest.approx(10.0 * i)
        assert p.params[f"c{i}"] == pytest.approx(0.5 * i)


def test_spr_beam_parse(tmp_path):
    head = _card((0.015, 20), (0.15, 20), (0, 10), (0, 10), (0, 10), (0, 10),
                 (0, 10), (0, 10))
    blocks = []
    for i in range(1, 7):
        kc = _card((1000.0, 20), (0, 20), (0, 20), (0, 20), (0, 20))
        func = _card((0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10),
                     (0, 20), (0, 20))
        fe = _card((0, 20), (0, 20), (0, 20), (0, 20))
        blocks += [kc, func, fe]
    body = ("/PROP/SPR_BEAM/22\nSPRI_13_Bolt\n"
            + head + "\n" + "\n".join(blocks) + "\n")
    model, log = _parse_props(tmp_path, body)
    assert not log.errors, log.errors
    p = model.properties[22]
    assert p.type == 13
    assert p.params["mass"] == pytest.approx(0.015)
    assert p.params["inertia"] == pytest.approx(0.15)
    assert p.params["k1"] == pytest.approx(1000.0)
    assert p.params["k6"] == pytest.approx(1000.0)


def test_void_parse(tmp_path):
    # /PROP/VOID/51 verbatim from SEAT_ADYREL
    body = "/PROP/VOID/51\ndummy\n                   1\n"
    model, log = _parse_props(tmp_path, body)
    assert not log.errors, log.errors
    p = model.properties[51]
    assert p.type == 0
    assert p.params["thick"] == pytest.approx(1.0)
    # universal geometry defaults so it runs on any family
    for key in ("nip", "hm", "qa", "qb", "h", "area"):
        assert key in p.params


def test_beam_section_fixed_parse(tmp_path):
    # RD-E-0500 Beam frame: Area/Iyy/Izz/Ixx are whole numbers (36/108/
    # 108/216) — the pre-M38 free-format 'skip pure-integer cards'
    # heuristic wrongly reported the section card missing.
    body = ("/PROP/BEAM/1\nBEAM\n"
            "                   0\n"
            "                   0                   0\n"
            "                  36                 108                 108     "
            "            216\n"
            "   000 000         0\n")
    model, log = _parse_props(tmp_path, body)
    assert not log.errors, log.errors
    p = model.properties[1]
    assert p.type == 3
    assert p.params["area"] == pytest.approx(36.0)
    assert p.params["iyy"] == pytest.approx(108.0)
    assert p.params["izz"] == pytest.approx(108.0)
    assert p.params["ixx"] == pytest.approx(216.0)


def test_inject1_inactive_parse(tmp_path):
    # /PROP/INJECT1/1001 verbatim from the airbag injector corpus deck
    body = ("/PROP/INJECT1/1001\nInjector\n"
            "         1         0                   0\n"
            "      1002         1         2                             0     "
            "              0\n")
    model, log = _parse_props(tmp_path, body)
    p = model.properties[1001]
    assert isinstance(p, prop_reader.InactiveProperty)
    assert getattr(p, "inactive", False) is True
    assert p.prop_name == "INJECT1"
    # safe geometry so a Starter that reaches init never divides by zero
    assert p.params["thick"] > 0.0


# ============================================================================
# SH_ORTH orientation math (hand calc) + rotation invariance
# ============================================================================

def test_sh_orth_orientation_hand_calc():
    # element local frame == global (identity E)
    E = np.zeros((3, 3, 3))
    for i in range(3):
        E[i] = np.eye(3)

    def prop(vx, vy, vz, phi=0.0):
        return Property(id=1, type=9, params={"vx": vx, "vy": vy, "vz": vz,
                                              "phi": phi})
    slices = [(slice(0, 1), None, prop(1, 0, 0)),      # 0 deg -> e1
              (slice(1, 2), None, prop(0, 1, 0)),      # 90 deg -> e2
              (slice(2, 3), None, prop(1, 1, 0))]      # 45 deg
    cs = shell_ortho.build_group_ortho(slices, E, 3)
    assert cs[0] == pytest.approx([1.0, 0.0])
    assert cs[1] == pytest.approx([0.0, 1.0])
    assert cs[2] == pytest.approx([math.cos(math.pi / 4)] * 2)
    # Phi adds to the projected angle: V=e1, Phi=30 deg -> (cos30, sin30)
    cs2 = shell_ortho.build_group_ortho(
        [(slice(0, 1), None, prop(1, 0, 0, 30.0))], E, 1)
    assert cs2[0] == pytest.approx([math.cos(math.radians(30)),
                                    math.sin(math.radians(30))])


def test_orientation_uses_element_frame_not_global():
    # a shell whose local e1 is the GLOBAL y axis: a fiber along global x
    # must land at 90 deg in the element frame (self-consistency)
    E = np.zeros((1, 3, 3))
    E[0, :, 0] = [0.0, 1.0, 0.0]     # e1 = +y
    E[0, :, 1] = [-1.0, 0.0, 0.0]    # e2 = -x
    E[0, :, 2] = [0.0, 0.0, 1.0]
    cs = shell_ortho.build_group_ortho(
        [(slice(0, 1), None, Property(id=1, type=9,
                                      params={"vx": 1.0, "vy": 0.0,
                                              "vz": 0.0, "phi": 0.0}))], E, 1)
    # V=e_x projected: VR = V.e1 = 0, VS = V.e2 = -1 -> (0, -1)
    assert cs[0] == pytest.approx([0.0, -1.0])


def test_strain_stress_rotation_energy_invariance():
    rng = np.random.default_rng(0)
    th = rng.uniform(-1.5, 1.5, size=5)
    cs = np.stack([np.cos(th), np.sin(th)], axis=1)
    deps = rng.normal(size=(5, 3)) * 1e-3
    sig_m = rng.normal(size=(5, 3)) * 100.0     # material-frame stress
    deps_m = shell_ortho.rot_strain_e2m(deps, cs)
    sig_e = shell_ortho.rot_stress_m2e(sig_m, cs)
    # sig:deps is frame invariant (Voigt with engineering shear)
    e_mat = np.einsum("ij,ij->i", sig_m, deps_m)
    e_elem = np.einsum("ij,ij->i", sig_e, deps)
    assert np.allclose(e_mat, e_elem)


# ============================================================================
# TYPE8 SPR_GENE 6-DOF spring force/moment vs closed form
# ============================================================================

def _type8_group(k, c=None, mass=2.0, inertia=3.0):
    c = c or [0.0] * 6
    m = Model()
    m.add_nodes(np.array([1, 2]), np.array([[0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]]))
    params = {"mass": mass, "inertia": inertia, "skew_id": 0}
    for i in range(6):
        params[f"k{i + 1}"] = k[i]
        params[f"c{i + 1}"] = c[i]
    prop = Property(id=1, type=8, params=params)
    g = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1]]),
                     part=np.array([0]))
    g.state["slices"] = [(slice(0, 1), None, prop)]
    return m, g


def test_type8_spring_force_moment_closed_form():
    k = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    m, g = _type8_group(k, mass=2.0, inertia=3.0)
    log = MessageLog()
    node_idx, massn, inertn = spring.init_group(g, m, log)
    assert not log.errors
    # lumped mass/inertia half/half to the two nodes
    assert massn == pytest.approx([1.0, 1.0])
    assert inertn == pytest.approx([1.5, 1.5])

    x = m.x0.copy()
    x[1] = [0.1, 0.2, 0.3]                       # relative translation
    v = np.zeros((2, 3))
    vr = np.zeros((2, 3))
    vr[1] = [1.0, 2.0, 3.0]                      # relative angular velocity
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    dt = 1.0e-3
    spring.forces(g, x, v, vr, dt, fint, mint)
    # global (skew 0) frame: F_i = k_i * delta_i, restoring on node 2
    assert fint[0] == pytest.approx([k[0] * 0.1, k[1] * 0.2, k[2] * 0.3])
    assert fint[1] == pytest.approx(-fint[0])
    # theta_i = omega_i * dt (rate integrated); M_i = k_{3+i} theta_i
    assert mint[0] == pytest.approx([k[3] * 1.0 * dt, k[4] * 2.0 * dt,
                                     k[5] * 3.0 * dt])
    assert mint[1] == pytest.approx(-mint[0])


def test_type8_spring_damping_force():
    k = [0.0] * 6
    c = [7.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    m, g = _type8_group(k, c=c, mass=1.0, inertia=1.0)
    log = MessageLog()
    spring.init_group(g, m, log)
    x = m.x0.copy()
    v = np.zeros((2, 3))
    v[1] = [0.5, 0.0, 0.0]                       # relative velocity along e1
    fint = np.zeros((2, 3))
    mint = np.zeros((2, 3))
    spring.forces(g, x, v, np.zeros((2, 3)), 1e-3, fint, mint)
    assert fint[0] == pytest.approx([7.0 * 0.5, 0.0, 0.0])


def test_mixed_type4_type8_group():
    # a /SPRING group with one axial TYPE4 element and one 6-DOF TYPE8
    # element must resolve each through its own path
    m = Model()
    m.add_nodes(np.array([1, 2, 3, 4]),
                np.array([[0.0, 0, 0], [1.0, 0, 0],
                          [0.0, 0, 0], [0.0, 0, 0]]))
    p4 = Property(id=1, type=4, params={"mass": 1.0, "k": 100.0, "c": 0.0})
    p8 = Property(id=2, type=8, params={"mass": 1.0, "inertia": 1.0,
                                        "skew_id": 0,
                                        **{f"k{i}": 0.0 for i in range(1, 7)},
                                        **{f"c{i}": 0.0 for i in range(1, 7)}})
    p8.params["k1"] = 200.0
    g = ElementGroup(ids=np.array([10, 20]), conn=np.array([[0, 1], [2, 3]]),
                     part=np.array([0, 1]))
    g.state["slices"] = [(slice(0, 1), None, p4), (slice(1, 2), None, p8)]
    log = MessageLog()
    spring.init_group(g, m, log)
    assert not log.errors
    x = m.x0.copy()
    x[1] = [1.1, 0, 0]        # axial stretch 0.1
    x[3] = [0.05, 0, 0]       # TYPE8 relative translation 0.05 along e1
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    spring.forces(g, x, np.zeros((4, 3)), np.zeros((4, 3)), 1e-3, fint, mint)
    assert fint[0][0] == pytest.approx(100.0 * 0.1)     # TYPE4 axial
    assert fint[2][0] == pytest.approx(200.0 * 0.05)    # TYPE8 DOF1


# ============================================================================
# VOID no-stiffness + universal family compatibility
# ============================================================================

def test_void_no_stiffness():
    # a void material (law 0) zeros the shell stress unconditionally
    void = Material(id=1, law=0, rho0=1.0e-6, params={"E": 0.0, "nu": 0.0})
    sig = np.array([[100.0, 50.0, 20.0]])
    deps = np.array([[0.01, -0.02, 0.005]])
    out, _epsp = materials.shell_update(void, sig, deps,
                                        np.zeros(1), 1e-3, {})
    assert np.all(out == 0.0)


def test_void_prop_type_universal():
    void = Property(id=51, type=0, params={"thick": 1.0})
    for req in (1, 2, 3, 14):     # shell, truss, beam, solid families
        assert prop_reader.prop_type_ok(req, void)


def test_ortho_prop_ok_on_shells_but_not_solids():
    orth = Property(id=1, type=9, params={})
    assert prop_reader.prop_type_ok(1, orth)      # shells
    assert not prop_reader.prop_type_ok(14, orth)  # solids


# ============================================================================
# INJECT1 InactiveProperty flow
# ============================================================================

def _shell_model_with_prop(prop):
    m = Model()
    m.add_nodes(np.array([1, 2, 3, 4]),
                np.array([[0.0, 0, 0], [1.0, 0, 0],
                          [1.0, 1.0, 0], [0.0, 1.0, 0]]))
    m.properties[prop.id] = prop
    m.materials[1] = Material(id=1, law=1, rho0=1.0e-6,
                              params={"E": 200.0, "nu": 0.3})
    m.parts[1] = Part(id=1, prop_id=prop.id, mat_id=1)
    m.raw_elems["SHELL"].append((1, 1, [1, 2, 3, 4]))
    return m


def test_inject1_refused_when_used_by_elements():
    inj = prop_reader.InactiveProperty(
        id=1001, type=-1, params=prop_reader._universal_geo_params(),
        prop_name="INJECT1")
    m = _shell_model_with_prop(inj)
    log = MessageLog()
    build_element_groups(m, log)
    # Starter builds the group (accepts); Engine refuses
    assert not any("property" in e.lower() and "not defined" in e.lower()
                   for e in log.errors)
    with pytest.raises(prop_reader.InactivePropertyError):
        prop_reader.refuse_inactive_properties(m, MessageLog())


def test_inactive_prop_not_refused_when_unused():
    m = Model()
    m.properties[1001] = prop_reader.InactiveProperty(
        id=1001, type=-1, params={}, prop_name="INJECT1")
    # no element group references it -> no refusal (used only by a /MONVOL)
    prop_reader.refuse_inactive_properties(m, MessageLog())   # must not raise


# ============================================================================
# mat_ID = 0 /PART rule (hm_read_part.F)
# ============================================================================

def test_mat0_spring_part_accepted():
    m = Model()
    m.add_nodes(np.array([1, 2]), np.array([[0.0, 0, 0], [1.0, 0, 0]]))
    m.properties[1] = Property(id=1, type=4,
                               params={"mass": 1.0, "k": 10.0, "c": 0.0})
    m.parts[1] = Part(id=1, prop_id=1, mat_id=0)     # mat 0 -> fictitious
    m.raw_elems["SPRING"].append((1, 1, [1, 2]))
    log = MessageLog()
    build_element_groups(m, log)
    assert not log.errors, log.errors
    assert m.springs is not None
    assert len(m.springs.state["slices"]) == 1
    _sl, mat, _prop = m.springs.state["slices"][0]
    assert mat is not None and not getattr(mat, "inactive", False)


def test_mat0_spring_type8_part_accepted():
    m = Model()
    m.add_nodes(np.array([1, 2]), np.array([[0.0, 0, 0], [0.0, 0, 0]]))
    params = {"mass": 1.0, "inertia": 1.0, "skew_id": 0}
    params.update({f"k{i}": 1.0 for i in range(1, 7)})
    params.update({f"c{i}": 0.0 for i in range(1, 7)})
    m.properties[8] = Property(id=8, type=8, params=params)
    m.parts[1] = Part(id=1, prop_id=8, mat_id=0)
    m.raw_elems["SPRING"].append((1, 1, [1, 2]))
    log = MessageLog()
    build_element_groups(m, log)
    assert not log.errors, log.errors


def test_mat0_solid_part_rejected():
    m = Model()
    m.add_nodes(np.arange(1, 9),
                np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                          [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                         dtype=float))
    m.properties[1] = Property(id=1, type=14,
                               params={"qa": 1.1, "qb": 0.05, "h": 0.1})
    m.parts[1] = Part(id=1, prop_id=1, mat_id=0)     # mat 0 ILLEGAL on solid
    m.raw_elems["BRICK"].append((1, 1, [1, 2, 3, 4, 5, 6, 7, 8]))
    log = MessageLog()
    build_element_groups(m, log)
    assert any("material 0 not defined" in e for e in log.errors), log.errors


# ============================================================================
# LAW19 fabric orientation — kernel level (cfg-free) and corpus (guarded)
# ============================================================================

def _fabric_material():
    """Build a LAW19 FabricMaterial straight from a hand record (no cfg):
    strongly anisotropic E11 != E22 so the fiber angle changes the stress."""
    from pyradioss.materials import law19_fabric
    rec = mat_reader.GenericMaterialRecord(
        law_name="FABRI", law_number=19, id=1, title="fab",
        params={"MAT_EA": 3.0, "MAT_EB": 0.5, "MAT_PRAB": 0.0,
                "MAT_GAB": 0.05, "MAT_GBC": 0.05, "MAT_GCA": 0.05,
                "MAT_REDFACT": 1.0, "M58_Zerostress": 0.0},
        density=8.0e-7)
    return law19_fabric.build_fabric(rec)


def _unit_shell_group(prop, mat):
    """A single flat unit shell in the z=0 plane (local e1 = +x)."""
    m = Model()
    m.add_nodes(np.array([1, 2, 3, 4]),
                np.array([[0.0, 0, 0], [1.0, 0, 0],
                          [1.0, 1.0, 0], [0.0, 1.0, 0]]))
    g = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1, 2, 3]]),
                     part=np.array([0]))
    g.state["slices"] = [(slice(0, 1), mat, prop)]
    log = MessageLog()
    shell_bt4.init_group(g, m, log)
    return m, g


def test_law19_orientation_changes_shell_stress():
    mat = _fabric_material()

    def prop(vx, vy):
        return Property(id=1, type=9,
                        params={"thick": 1.0, "nip": 1, "hm": 0.0, "hf": 0.0,
                                "hr": 0.0, "vx": vx, "vy": vy, "vz": 0.0,
                                "phi": 0.0})
    # 0-deg fiber (along x) and 90-deg fiber (along y)
    m0, g0 = _unit_shell_group(prop(1.0, 0.0), mat)
    m9, g9 = _unit_shell_group(prop(0.0, 1.0), mat)
    assert g0.state["ortho"] is not None
    # stretch the same +x edge on both: node 2,3 move +x by a small amount
    for m, g in ((m0, g0), (m9, g9)):
        v = np.zeros((4, 3))
        v[1] = v[2] = [1.0, 0.0, 0.0]           # uniaxial-ish x stretch rate
        fint = np.zeros((4, 3))
        mint = np.zeros((4, 3))
        for _ in range(3):
            shell_bt4.forces(g, m.x0, v, np.zeros((4, 3)), 1e-4, fint, mint)
        g.state["_fint"] = fint.copy()
    fx0 = abs(g0.state["_fint"][1, 0])
    fx9 = abs(g9.state["_fint"][1, 0])
    # E11 (3.0) is 6x E22 (0.5): the x-fiber shell resists the x stretch
    # much more stiffly than the y-fiber shell -> larger reaction force
    assert fx0 > 2.0 * fx9 > 0.0


@pytest.mark.skipif(not _HAS_CFG,
                    reason="hm_cfg_files CFG tree not found "
                           "(set PYRADIOSS_HM_CFG; see ci.yml / M37)")
def test_law19_prop9_corpus_end_to_end(tmp_path):
    import shutil
    from pyradioss.starter.starter import run_starter
    # vendored corpus deck (tests/data/rd_decks); PYRADIOSS_RD_DECKS
    # overrides with a full corpus extract — see test_m37_materials_pack1
    rd_decks = os.environ.get(
        "PYRADIOSS_RD_DECKS",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data", "rd_decks"))
    src = os.path.join(rd_decks, "rd_v_material", "RD-V-0230_Fabric_LAW19",
                       "0230_fabric_LAW19", "0230_shell_mat_019_01",
                       "SHELL_LAW19_PROP9_0000.rad")
    if not os.path.exists(src):
        pytest.skip("RD-V-0230 corpus deck not extracted")
    # copy out of the shared corpus dir so the run's .out/.rst stay local
    deck = str(tmp_path / "SHELL_LAW19_PROP9_0000.rad")
    shutil.copy(src, deck)
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(deck, log)
    assert not log.errors
    # every part resolved to a TYPE9 orthotropic shell property
    shells = model.shells if model.shells is not None else model.shells_qeph
    assert shells is not None
    assert shells.state.get("ortho") is not None
