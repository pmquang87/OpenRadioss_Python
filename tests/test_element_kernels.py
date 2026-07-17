"""Single-element kernel tests: prescribe nodal velocities and check the
stress / force response against hand calculations (the classic 'patch
test at one element' used to validate explicit kernels)."""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import (beam_type3, shell_bt4, shell_tri3,
                                solid_hexa8, solid_tetra4)
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_node_groups,
                                              resolve_surfaces)

STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""


def _build(deck_text, tmp_path):
    f = tmp_path / "K_0000.rad"
    f.write_text(deck_text)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    resolve_surfaces(model, log)
    initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    return model, log


def test_hexa8_uniaxial_strain_stress_and_forces(tmp_path):
    deck = (
        "/BEGIN\nunit cube\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n"
        "1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
    )
    model, log = _build(deck, tmp_path)
    g = model.bricks
    assert g.n == 1
    # element mass = rho * V = 7.8e-6
    assert g.state["mass"][0] == pytest.approx(7.8e-6)

    # prescribe uniaxial strain rate: vx = x * rate (all other v = 0)
    rate = 1e-3
    v = np.zeros_like(model.x)
    v[:, 0] = model.x[:, 0] * rate
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    dt = 1e-3
    dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)

    mat = g.state["slices"][0][1]
    expected = (mat.K + 4 * mat.G / 3) * rate * dt
    assert g.state["sig"][0, 0] == pytest.approx(expected, rel=1e-10)
    # forces must balance (free body): sum = 0
    assert np.abs(fint.sum(axis=0)).max() < 1e-12
    # tension: nodes on x=1 pulled back (-x), on x=0 pulled +x
    assert fint[1, 0] < 0 and fint[0, 0] > 0
    # critical dt: lc/c (= 1mm / 6021 mm/ms for the cube) times the exact
    # eigenvalue correction, ~0.73 for a cube at nu = 0.3 (see the kernel
    # docstring — lc/c alone overestimates the true stability limit)
    c = mat.sound_speed_solid()
    assert dtc[0] == pytest.approx(g.state["dtfac"][0] / c, rel=1e-6)
    assert 0.5 / c < dtc[0] < 1.0 / c


def test_hexa8_rigid_rotation_gives_no_stress(tmp_path):
    """Objectivity check: a rigid rotation velocity field must produce
    (to first order) no stress — this exercises the spin/Jaumann terms."""
    deck = (
        "/BEGIN\nunit cube\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    g = model.bricks
    omega = np.array([0.0, 0.0, 1.0])           # spin about z
    v = np.cross(np.broadcast_to(omega, model.x.shape), model.x)
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    solid_hexa8.forces(g, model.x, v, model.vr, 1e-4, fint, mint)
    # pure rotation: D = 0 exactly, so stress stays 0
    assert np.abs(g.state["sig"]).max() < 1e-12


def test_shell_membrane_tension(tmp_path):
    deck = (
        "/BEGIN\nunit shell\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nplate\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nplate prop\n1 0 0 0\n0.01 0.01 0.01 0 0\n"
        "3 0 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    g = model.shells
    t = g.state["thick"][0]
    assert t == pytest.approx(0.1)
    assert g.state["mass"][0] == pytest.approx(7.8e-6 * 0.1)

    # uniaxial membrane stretch: vx = x * rate
    rate = 1e-3
    dt = 1e-3
    v = np.zeros_like(model.x)
    v[:, 0] = model.x[:, 0] * rate
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)

    mat = g.state["slices"][0][1]
    # plane stress, eps_yy = 0 (edges held): sig_xx = E/(1-nu^2)*eps
    exp = mat.E / (1 - mat.nu ** 2) * rate * dt
    for k in range(3):  # all three layers identical in pure membrane
        assert g.state["sig"][0, k, 0] == pytest.approx(exp, rel=1e-10)
    assert np.abs(fint.sum(axis=0)).max() < 1e-12   # equilibrium
    assert np.abs(mint).max() < 1e-12               # no bending


# ============================================================================
# M2 elements
# ============================================================================

TET_DECK = (
    "/BEGIN\nunit tet\n"
    "/NODE\n"
    "1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
    "/TETRA4/1\n1 1 2 3 4\n"
    "/PART/1\ntet\n1 1\n" + STEEL_LAW1 +
    "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
)


def test_tetra4_uniaxial_strain_stress_and_forces(tmp_path):
    model, _ = _build(TET_DECK, tmp_path)
    g = model.tetras
    assert g.n == 1
    # element mass = rho * V = rho / 6
    assert g.state["mass"][0] == pytest.approx(7.8e-6 / 6.0)

    rate = 1e-3
    v = np.zeros_like(model.x)
    v[:, 0] = model.x[:, 0] * rate           # uniaxial strain rate
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    dt = 1e-3
    dtc = solid_tetra4.forces(g, model.x, v, model.vr, dt, fint, mint)

    mat = g.state["slices"][0][1]
    expected = (mat.K + 4 * mat.G / 3) * rate * dt
    assert g.state["sig"][0, 0] == pytest.approx(expected, rel=1e-10)
    assert np.abs(fint.sum(axis=0)).max() < 1e-12       # free-body balance
    # tension along x: node 2 (x=1) pulled back, node 1 pulled forward
    assert fint[1, 0] < 0 and fint[0, 0] > 0
    # dt = dtfac * lc/c with lc = 3V/Amax (min altitude), and the exact
    # eigenvalue correction must genuinely bite (below 1)
    c = mat.sound_speed_solid()
    lc = 3.0 * (1 / 6.0) / (np.sqrt(3) / 2)  # largest face is the oblique one
    assert dtc[0] == pytest.approx(g.state["dtfac"][0] * lc / c, rel=1e-6)
    assert 0.3 < g.state["dtfac"][0] < 1.0


def test_tetra4_rigid_rotation_gives_no_stress(tmp_path):
    model, _ = _build(TET_DECK, tmp_path)
    g = model.tetras
    omega = np.array([0.3, -0.2, 1.0])
    v = np.cross(np.broadcast_to(omega, model.x.shape), model.x)
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    solid_tetra4.forces(g, model.x, v, model.vr, 1e-4, fint, mint)
    assert np.abs(g.state["sig"]).max() < 1e-12


# The SAME four points as TET_DECK, listed in the REAL Radioss winding
# (VOLDP>0, i.e. V_std<0 in the port's isoparametric det): nodes 2 and 4 are
# swapped on the card relative to TET_DECK. This is how every official mesh
# is written (RD-V-0020 / RD-V-0240), which the port used to flag as
# "zero or negative volume" on 100% of elements (M38-BUG-3).
TET_DECK_OFFICIAL = (
    "/BEGIN\nunit tet official\n"
    "/NODE\n"
    "1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
    "/TETRA4/1\n1 1 4 3 2\n"
    "/PART/1\ntet\n1 1\n" + STEEL_LAW1 +
    "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
)


def test_tetra4_official_winding_initializes(tmp_path):
    """A /TETRA4 in the real Radioss winding (VOLDP>0) must initialise with a
    POSITIVE volume via the s4coor3/hm_read_solid 2<->4 node-swap
    canonicalisation — _build() asserts the starter logs NO error — and then
    give the physically identical response to the port-winding twin
    (test_tetra4_uniaxial_strain_stress_and_forces). Regression for M38-BUG-3."""
    model, _ = _build(TET_DECK_OFFICIAL, tmp_path)
    g = model.tetras
    assert g.n == 1
    # positive physical volume (|V| = 1/6) and mass, exactly the port twin
    assert g.state["vol0"][0] == pytest.approx(1.0 / 6.0)
    assert g.state["mass"][0] == pytest.approx(7.8e-6 / 6.0)

    rate, dt = 1e-3, 1e-3
    v = np.zeros_like(model.x)
    v[:, 0] = model.x[:, 0] * rate                    # uniaxial strain rate
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    solid_tetra4.forces(g, model.x, v, model.vr, dt, fint, mint)

    mat = g.state["slices"][0][1]
    expected = (mat.K + 4 * mat.G / 3) * rate * dt
    assert g.state["sig"][0, 0] == pytest.approx(expected, rel=1e-10)
    assert np.abs(fint.sum(axis=0)).max() < 1e-12       # free-body balance
    # winding-independent physics: node id 2 at x=1 pulled back, node id 1
    # at the origin pulled forward — same as the port-winding fixture
    assert fint[1, 0] < 0 and fint[0, 0] > 0


def test_tetra4_degenerate_volume_is_caught(tmp_path):
    """A genuinely degenerate (coplanar, zero-volume) /TETRA4 must STILL be
    caught: the winding-canonicalisation swap cannot rescue |V|~0, mirroring
    s4deri3.F's DET<=0 guard (MSGID 245 for a solid property). This is the
    guard the M38 fix must not silently disable with an unconditional |V|."""
    deck = (
        "/BEGIN\ndegen tet\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 0 1 0\n4 1 1 0\n"        # all in the z=0 plane
        "/TETRA4/1\n1 1 2 3 4\n"
        "/PART/1\ntet\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
    )
    f = tmp_path / "K_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    resolve_surfaces(model, log)
    initialize_elements_and_mass(model, log)
    assert any("volume" in e.lower() for e in log.errors)


SH3N_DECK = (
    "/BEGIN\nunit triangle\n"
    "/NODE\n1 0 0 0\n2 1 0 0\n3 0 1 0\n"
    "/SH3N/1\n1 1 2 3\n"
    "/PART/1\nplate\n1 1\n" + STEEL_LAW1 +
    "/PROP/SHELL/1\nplate prop\n1 0 0 0\n0.01 0.01 0.01 0 0\n"
    "3 0 0.1\n/END\n"
)


def test_sh3n_membrane_tension(tmp_path):
    model, _ = _build(SH3N_DECK, tmp_path)
    g = model.sh3n
    assert g.n == 1
    assert g.state["mass"][0] == pytest.approx(7.8e-6 * 0.1 * 0.5)

    rate = 1e-3
    dt = 1e-3
    v = np.zeros_like(model.x)
    v[:, 0] = model.x[:, 0] * rate
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    shell_tri3.forces(g, model.x, v, model.vr, dt, fint, mint)

    mat = g.state["slices"][0][1]
    exp = mat.E / (1 - mat.nu ** 2) * rate * dt   # plane stress, eps_yy = 0
    for k in range(3):
        assert g.state["sig"][0, k, 0] == pytest.approx(exp, rel=1e-10)
    assert np.abs(fint.sum(axis=0)).max() < 1e-12
    assert np.abs(mint).max() < 1e-12


def test_sh3n_rigid_rotation_no_stress_no_shear(tmp_path):
    """Out-of-plane rigid rotation: the C0 shear field must vanish exactly
    (B1.x = 1 cancellation, see the kernel docstring) — the element
    produces neither membrane stress nor transverse shear."""
    model, _ = _build(SH3N_DECK, tmp_path)
    g = model.sh3n
    omega = np.array([1.0, 0.0, 0.0])            # spin about the x axis
    v = np.cross(np.broadcast_to(omega, model.x.shape), model.x)
    vr = np.broadcast_to(omega, model.x.shape).copy()
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    shell_tri3.forces(g, model.x, v, vr, 1e-4, fint, mint)
    assert np.abs(g.state["sig"]).max() < 1e-12
    assert np.abs(g.state["qshear"]).max() < 1e-12
    assert np.abs(fint).max() < 1e-12 and np.abs(mint).max() < 1e-12


BEAM_DECK = (
    "/BEGIN\none beam\n"
    "/NODE\n1 0 0 0\n2 10 0 0\n3 0 1 0\n"
    "/BEAM/1\n1 1 2 3\n"
    "/PART/1\nbeam\n1 1\n" + STEEL_LAW1 +
    "/PROP/BEAM/1\nsquare 5x5\n25.0 52.0833333 52.0833333 88.0\n/END\n"
)


def _beam_setup(tmp_path):
    model, _ = _build(BEAM_DECK, tmp_path)
    g = model.beams
    assert g.n == 1
    mat = g.state["slices"][0][1]
    prop = g.state["slices"][0][2]
    v = np.zeros_like(model.x)
    vr = np.zeros_like(model.x)
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    return model, g, mat, prop.params, v, vr, fint, mint


def test_beam_axial_force(tmp_path):
    model, g, mat, p, v, vr, fint, mint = _beam_setup(tmp_path)
    # element mass = rho * A * L, split on the two END nodes only
    assert g.state["mass"][0] == pytest.approx(7.8e-6 * 25.0 * 10.0)
    assert model.mass[2] >= 1e29           # orientation node frozen

    rate, dt, L = 1e-3, 1e-3, 10.0
    v[1, 0] = L * rate                      # stretch: eps_dot = rate
    beam_type3.forces(g, model.x, v, vr, dt, fint, mint)
    N = mat.E * p["area"] * rate * dt
    assert g.state["fres"][0, 0] == pytest.approx(N, rel=1e-12)
    # tension pulls node 1 toward +x, node 2 toward -x (fint = -internal)
    assert fint[0, 0] == pytest.approx(N, rel=1e-12)
    assert fint[1, 0] == pytest.approx(-N, rel=1e-12)
    assert np.abs(mint).max() < 1e-15


def test_beam_pure_bending_moment(tmp_path):
    model, g, mat, p, v, vr, fint, mint = _beam_setup(tmp_path)
    w, dt, L = 1e-3, 1e-3, 10.0
    vr[0, 1] = -w                           # antisymmetric rotation rates:
    vr[1, 1] = +w                           # pure curvature, no shear
    beam_type3.forces(g, model.x, v, vr, dt, fint, mint)
    My = mat.E * p["iyy"] * (2 * w / L) * dt
    assert g.state["mres"][0, 1] == pytest.approx(My, rel=1e-12)
    assert np.abs(g.state["fres"]).max() < 1e-15      # no N, no shear
    assert np.abs(fint).max() < 1e-15
    assert mint[0, 1] == pytest.approx(My, rel=1e-12)
    assert mint[1, 1] == pytest.approx(-My, rel=1e-12)


def test_beam_torsion(tmp_path):
    model, g, mat, p, v, vr, fint, mint = _beam_setup(tmp_path)
    w, dt, L = 1e-3, 1e-3, 10.0
    vr[1, 0] = w                            # twist rate w/L
    beam_type3.forces(g, model.x, v, vr, dt, fint, mint)
    Mx = mat.G * p["ixx"] * (w / L) * dt
    assert g.state["mres"][0, 0] == pytest.approx(Mx, rel=1e-12)
    assert mint[0, 0] == pytest.approx(Mx, rel=1e-12)
    assert mint[1, 0] == pytest.approx(-Mx, rel=1e-12)
    assert np.abs(fint).max() < 1e-15


def test_beam_shear_force_and_moment_equilibrium(tmp_path):
    """A shear-producing rotation must leave the element in exact global
    equilibrium: sum F = 0 and sum (m_i + x_i x f_i) = 0 — this pins down
    the Qy*L/2 transfer terms of the internal moments."""
    model, g, mat, p, v, vr, fint, mint = _beam_setup(tmp_path)
    w, dt = 1e-3, 1e-3
    vr[1, 2] = w                            # rotation about z at node 2 only
    beam_type3.forces(g, model.x, v, vr, dt, fint, mint)
    assert np.abs(g.state["fres"][0, 1]) > 0          # shear force present
    assert np.abs(fint.sum(axis=0)).max() < 1e-15
    torque = mint.sum(axis=0) + np.cross(model.x, fint).sum(axis=0)
    assert np.abs(torque).max() < 1e-13


def test_shell_bt4_stiffness_hourglass(tmp_path):
    """BLT84 stiffness hourglass: a pure w-hourglass velocity pattern
    builds a restoring force that GROWS linearly with time (Q += k q_dot
    dt), matching the closed-form modal stiffness — and a linear velocity
    field must leave the hourglass state exactly untouched."""
    deck = (
        "/BEGIN\nunit shell\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nplate\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nplate prop\n1 0 0 0\n0.01 0.01 0.01 0 0\n"
        "3 0 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    g = model.shells
    mat = g.state["slices"][0][1]
    t, A, hf = 0.1, 1.0, 0.01
    bb = 2.0                                 # B1.B1 + B2.B2 of the unit square
    k_w = hf * (5.0 / 6.0) * mat.G * t * A * bb / 8.0

    w0, dt = 1e-3, 1e-3
    v = np.zeros_like(model.x)
    v[:4, 2] = np.array([1.0, -1.0, 1.0, -1.0]) * w0   # pure hourglass mode
    f1 = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    shell_bt4.forces(g, model.x, v, model.vr, dt, f1, mint)
    # gamma = h on the square, so q_dot = 4 w0 and f_i = -Q h_i
    Q1 = k_w * 4.0 * w0 * dt
    assert f1[0, 2] == pytest.approx(-Q1, rel=1e-10)
    # no REAL stress from the hourglass pattern (B1.h = B2.h = 0)
    assert np.abs(g.state["sig"]).max() < 1e-15
    assert np.abs(g.state["qshear"]).max() < 1e-15

    f2 = np.zeros_like(model.x)
    shell_bt4.forces(g, model.x, v, model.vr, dt, f2, mint)
    # stiffness control: same velocity again -> force has DOUBLED
    assert f2[0, 2] == pytest.approx(-2.0 * Q1, rel=1e-10)
    # stored (not dissipated) hourglass energy is tracked
    assert g.state["ehour"][0] > 0

    # orthogonality: a linear velocity field leaves hgq untouched
    g.state["hgq"][:] = 0.0
    v = np.zeros_like(model.x)
    v[:, 2] = 0.3 + 0.7 * model.x[:, 0] - 0.2 * model.x[:, 1]
    shell_bt4.forces(g, model.x, v, model.vr, dt, f2, mint)
    assert np.abs(g.state["hgq"]).max() < 1e-15


def test_degenerated_brick_converts_to_tetra(tmp_path):
    """A /BRICK card with the classic 4-distinct-node collapse pattern
    must become a /TETRA4 element with positive volume."""
    deck = (
        "/BEGIN\ndegen\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
        "/BRICK/1\n1 1 2 3 3 4 4 4 4\n"
        "/PART/1\ntet\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
    )
    model, log = _build(deck, tmp_path)
    assert model.bricks is None
    assert model.tetras.n == 1
    assert model.tetras.state["mass"][0] == pytest.approx(7.8e-6 / 6.0)


def test_degenerated_brick_penta_rejected(tmp_path):
    """A 6-distinct-node (penta) collapse is refused with a clear error."""
    deck = (
        "/BEGIN\ndegen penta\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 6 5\n"
        "/PART/1\npenta\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
    )
    f = tmp_path / "K_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    build_element_groups(model, log)
    assert any("penta" in e.lower() or "degenerated" in e.lower()
               for e in log.errors)


def test_surfaces_from_tetra_and_sh3n_parts(tmp_path):
    """/SURF/PART on tetra and 3-node-shell parts: the free tetra faces
    and every triangle become degenerate 4-node segments (n4 = n3), the
    convention the TYPE7 contact narrow phase already handles."""
    deck = (
        "/BEGIN\nsurfaces\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
        "11 5 0 0\n12 6 0 0\n13 5 1 0\n"
        "/TETRA4/1\n1 1 2 3 4\n"
        "/SH3N/2\n2 11 12 13\n"
        "/PART/1\ntet\n1 1\n"
        "/PART/2\ntri\n2 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/PROP/SHELL/2\nshell\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n"
        "/SURF/PART/1\nboth parts\n1 2\n"
        "/END\n"
    )
    model, _ = _build(deck, tmp_path)
    segs = model.surfaces[1].segments
    # 4 free tetra faces + 1 triangle, all with the 3rd node repeated
    assert segs.shape == (5, 4)
    assert np.all(segs[:, 3] == segs[:, 2])
