"""Single-element kernel tests: prescribe nodal velocities and check the
stress / force response against hand calculations (the classic 'patch
test at one element' used to validate explicit kernels)."""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8, shell_bt4
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
