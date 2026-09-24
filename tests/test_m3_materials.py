"""M3 unit tests: LAW36 / LAW27 / LAW42 material kernels against
closed-form results, the /FAIL criteria, the element-deletion plumbing
and the global beam plasticity — all at single-element / single-point
level (the analytic full-run validations are in test_m3_integration.py)."""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.elements import beam_type3, shell_bt4, solid_hexa8
from pyradioss.failure import biquad, johnson
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials import law27_brittle, law36_tabulated, law42_ogden
from pyradioss.model.entities import FailureModel, Material
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_materials,
                                              resolve_node_groups,
                                              resolve_surfaces)


def _build(deck_text, tmp_path):
    f = tmp_path / "K_0000.rad"
    f.write_text(deck_text)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    resolve_materials(model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    resolve_surfaces(model, log)
    initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    return model, log


# ============================================================================
# LAW36 — tabulated plasticity
# ============================================================================

def _mat36(curves, rates):
    """Material with resolved curve arrays (what resolve_materials builds)."""
    params = {"E": 210.0, "nu": 0.3, "eps_p_max": 1e30,
              "funct_ids": list(range(len(curves)))}
    cxs, cys, css = [], [], []
    for (x, y) in curves:
        f = FunctTable(1, x, y)
        cxs.append(f.x)
        cys.append(f.y)
        css.append(f.slope)
    params.update(curve_x=cxs, curve_y=cys, curve_s=css,
                  rates=np.asarray(rates, dtype=float))
    return Material(id=1, law=36, rho0=7.8e-6, params=params)


def test_law36_solid_stress_sits_exactly_on_the_curve():
    """Sustained deviatoric flow: the von Mises stress must land EXACTLY
    on the tabulated curve at the reached plastic strain — piecewise
    linear hardening makes the return-map consistency condition linear,
    so there is no iteration error to hide behind."""
    # bilinear curve: sy = 0.4 + 1.0*ep up to 0.1, then flat 0.5
    mat = _mat36([([0.0, 0.1, 10.0], [0.4, 0.5, 0.5])], [0.0])
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    deps = np.zeros((1, 6))
    deps[0] = [1e-4, -5e-5, -5e-5, 0, 0, 0]      # pure deviatoric push
    for _ in range(600):
        law36_tabulated.solid_update(mat, sig, deps, epsp, 1e-3)
    assert 0.01 < epsp[0] < 0.1                  # inside the first segment
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm == pytest.approx(0.4 + 1.0 * epsp[0], rel=1e-9)
    # keep pushing well past the knee: flat segment -> exactly 0.5
    for _ in range(3000):
        law36_tabulated.solid_update(mat, sig, deps, epsp, 1e-3)
    assert epsp[0] > 0.1
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm == pytest.approx(0.5, rel=1e-9)


def test_law36_end_slope_extrapolation():
    """Beyond the last table point the curve keeps its final slope (the
    FINTER convention) — a rising last segment keeps hardening."""
    mat = _mat36([([0.0, 0.05], [0.4, 0.45])], [0.0])   # slope 1.0, open end
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    deps = np.zeros((1, 6))
    deps[0] = [1e-4, -5e-5, -5e-5, 0, 0, 0]
    for _ in range(2000):
        law36_tabulated.solid_update(mat, sig, deps, epsp, 1e-3)
    assert epsp[0] > 0.05                        # beyond the table
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm == pytest.approx(0.4 + 1.0 * epsp[0], rel=1e-9)


def test_law36_strain_rate_interpolation():
    """Two flat curves at rates 0 and 10, driven at exactly rate 5: the
    flow stress must be the linear blend (0.4 + 0.8)/2 = 0.6."""
    mat = _mat36([([0.0, 1.0], [0.4, 0.4]), ([0.0, 1.0], [0.8, 0.8])],
                 [0.0, 10.0])
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    d, dt = 5e-4, 1e-4                           # dev eq rate = d/dt = 5.0
    deps = np.zeros((1, 6))
    deps[0] = [d, -d / 2, -d / 2, 0, 0, 0]
    for _ in range(200):
        law36_tabulated.solid_update(mat, sig, deps, epsp, dt)
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm == pytest.approx(0.6, rel=1e-9)
    # above the last tabulated rate: clamps to the last curve (BUG-MAT-03 alignment)
    for _ in range(50):
        law36_tabulated.solid_update(mat, sig, deps * 40, epsp, dt)
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm == pytest.approx(0.8, rel=1e-9)


def test_law36_shell_plane_stress_on_curve():
    """Equibiaxial plane-stress flow: von Mises = |sigma| must track the
    tabulated curve (the LAW2 shell test, tabulated flavour)."""
    mat = _mat36([([0.0, 1.0], [0.4, 0.9])], [0.0])     # sy = 0.4 + 0.5 ep
    sig = np.zeros((1, 3))
    epsp = np.zeros(1)
    deps = np.tile([1e-4, 1e-4, 0.0], (1, 1))
    for _ in range(2000):
        law36_tabulated.shell_update(mat, sig, deps, epsp, 1e-3)
    seq = np.sqrt(sig[0, 0] ** 2 - sig[0, 0] * sig[0, 1] + sig[0, 1] ** 2
                  + 3 * sig[0, 2] ** 2)
    assert epsp[0] > 0.05
    assert seq == pytest.approx(0.4 + 0.5 * epsp[0], rel=1e-9)


def test_law36_deck_roundtrip_single_element(tmp_path):
    """Full parser + starter + hexa kernel: the /MAT/LAW36 cards resolve
    their /FUNCT references and the element stress follows the curve."""
    deck = (
        "/BEGIN\nlaw36 cube\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n"
        "/MAT/LAW36/1\ntabulated steel\n7.8e-6\n210. 0.3\n1 0\n11\n"
        "/FUNCT/11\nhardening\n0.0 0.4\n1.0 1.4\n"
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    g = model.bricks
    mat = g.state["slices"][0][1]
    assert mat.law == 36 and len(mat.params["curve_x"]) == 1

    # drive uniaxial strain far past yield (single big loop of kernel calls
    # with a frozen geometry — a pure constitutive exercise)
    rate, dt = 1e-2, 1e-3
    v = np.zeros_like(model.x)
    v[:, 0] = model.x[:, 0] * rate
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    for _ in range(400):
        solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
    ep = g.state["epsp"][0]
    assert ep > 0.001
    s = g.state["sig"][0]
    vm = np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2
                        + (s[2] - s[0]) ** 2))
    assert vm == pytest.approx(0.4 + 1.0 * ep, rel=1e-6)


# ============================================================================
# LAW27 — brittle cracking
# ============================================================================

def _mat27():
    return Material(id=1, law=27, rho0=2.5e-6, params={
        "E": 70.0, "nu": 0.2,
        "eps_t1": 1e-3, "eps_m1": 2e-3, "dmax1": 0.8, "eps_f1": 3e-3,
        "eps_t2": 1e-3, "eps_m2": 2e-3, "dmax2": 0.8, "eps_f2": 3e-3})


def _extra27():
    return {
        "eps27": np.zeros((1, 3)),
        "crk27": np.zeros(1),
        "ang27": np.zeros(1),
        "dmg27": np.zeros((1, 2)),
        "layfail": np.ones(1)
    }


def test_law27_damage_curve_and_rupture():
    """Incremental formulation matches Fortran exactly (one cycle delay
    in damage application and crack state tracking)."""
    mat = _mat27()
    ex = _extra27()
    sig = np.zeros((1, 3))
    cps = 70.0 / (1 - 0.2 ** 2)
    step = np.array([[5e-4, 0.0, 0.0]])

    # elastic below eps_t (eps = 5e-4)
    law27_brittle.shell_update(mat, sig, step, None, 1e-3, ex)
    assert sig[0, 0] == pytest.approx(cps * 5e-4, rel=1e-12)
    assert ex["dmg27"][0, 0] == 0.0

    # eps = 1.5e-3: crack initiates but stiffness for this cycle is
    # from old damage (0.0). Damage updates at end to 0.4.
    law27_brittle.shell_update(mat, sig, step, None, 1e-3, ex)
    law27_brittle.shell_update(mat, sig, step, None, 1e-3, ex)
    assert ex["dmg27"][0, 0] == pytest.approx(0.4, rel=1e-12)
    assert sig[0, 0] == pytest.approx(cps * 1.5e-3 * 0.6, rel=1e-12)

def test_law27_unilateral_damage_and_memory():
    """Secant unloading uses degraded stiffness until crack closes."""
    mat = _mat27()
    ex = _extra27()
    sig = np.zeros((1, 3))
    cps = 70.0 / (1 - 0.2 ** 2)

    # open the crack to eps = 1.5e-3 (d = 0.4 at end)
    law27_brittle.shell_update(
        mat, sig, np.array([[1.5e-3, 0.0, 0.0]]), None, 1e-3, ex)
    assert ex["dmg27"][0, 0] == pytest.approx(0.4)

    # push into compression by deps = -2.5e-3.
    # total strain eps = -1.0e-3. Since eps < 0, crack closes (d = 0 active).
    # sig = cps * -1.0e-3 = -0.07291666667
    law27_brittle.shell_update(
        mat, sig, np.array([[-2.5e-3, 0.0, 0.0]]), None, 1e-3, ex)
    assert sig[0, 0] == pytest.approx(cps * -1.0e-3, rel=1e-12)

def test_law27_crack_direction_memory():
    """Directional damage."""
    mat = _mat27()
    ex = _extra27()
    sig = np.zeros((1, 3))
    cps = 70.0 / (1 - 0.2 ** 2)
    law27_brittle.shell_update(
        mat, sig, np.array([[1.5e-3, 0.0, 0.0]]), None, 1e-3, ex)
    assert ex["ang27"][0] == pytest.approx(0.0)          # crack normal = x
    
    # pull y to 1.5e-3. Initial damage in Y is 0.0.
    law27_brittle.shell_update(
        mat, sig, np.array([[0.0, 1.5e-3, 0.0]]), None, 1e-3, ex)
    assert ex["dmg27"][0, 1] == pytest.approx(0.4, rel=1e-12)
    # The total strain is exx=1.5e-3, eyy=1.5e-3.
    # s2 = cps * (eyy + nu*exx) = cps * 1.8e-3
    # Crack in Y has damage 0.4.
    # sig_y = s2 * 0.6 = cps * 1.8e-3 * 0.6 = 0.07875
    assert sig[0, 1] == pytest.approx(cps * 1.8e-3 * 0.6, rel=1e-12)


# ============================================================================
# LAW42 — Ogden hyperelasticity
# ============================================================================

def _mat42(mu, alpha, nu=0.495, rho=1e-6):
    G0 = sum(m * a for m, a in zip(mu, alpha)) / 2.0
    return Material(id=1, law=42, rho0=rho, params={
        "E": 2.0 * G0 * (1.0 + nu), "nu": nu, "mu": mu, "alpha": alpha})


def _ogden_uniaxial(mu, alpha, lam):
    """Closed-form incompressible uniaxial Cauchy stress
    sigma = sum mu_p (lam^a_p - lam^(-a_p/2))."""
    return sum(m * (lam ** a - lam ** (-a / 2.0))
               for m, a in zip(mu, alpha))


def test_law42_uniaxial_closed_form():
    """Incompressible uniaxial F: sigma_xx - sigma_lat must equal the
    Ogden closed form exactly (J = 1 kills the penalty term)."""
    mu, al = [0.002, -0.0004], [2.0, -2.0]               # Mooney-Rivlin
    mat = _mat42(mu, al)
    for lam in (1.2, 2.0, 0.6):
        F = np.diag([lam, lam ** -0.5, lam ** -0.5])[None, :, :]
        sig = np.zeros((1, 6))
        _, _, c = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)),
                                           None, 1e-3, {"F": F})
        assert sig[0, 0] - sig[0, 1] == pytest.approx(
            _ogden_uniaxial(mu, al, lam), rel=1e-10)
        assert sig[0, 1] == pytest.approx(sig[0, 2], rel=1e-10)
        assert np.abs(sig[0, 3:]).max() < 1e-15


def test_law42_volumetric_penalty_and_zero_at_identity():
    """Pure dilatation: deviatoric part vanishes, sigma = K (J - 1) I;
    and F = I gives exactly zero stress (hyperelastic consistency)."""
    mat = _mat42([0.002], [2.0])
    K = mat.K
    s = 1.02
    F = (s * np.eye(3))[None, :, :]
    sig = np.zeros((1, 6))
    law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-3,
                             {"F": F})
    assert sig[0, 0] == pytest.approx(K * (s ** 3 - 1.0), rel=1e-10)
    assert sig[0, 0] == pytest.approx(sig[0, 1], rel=1e-12)

    sig = np.ones((1, 6))          # garbage in: total law overwrites
    law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1e-3,
                             {"F": np.eye(3)[None, :, :]})
    assert np.abs(sig).max() < 1e-12


def test_law42_sound_speed_grows_with_stretch():
    """The tangent stiffness grows like lambda^alpha: the returned sound
    speed at lambda = 2 must clearly exceed the ground-state one — this
    is the hook that keeps the Courant dt a bound (M2 lesson). nu = 0.3
    keeps the (constant) bulk term from masking the shear stiffening."""
    mat = _mat42([0.002], [3.0], nu=0.3)
    sig = np.zeros((1, 6))
    _, _, c1 = law42_ogden.solid_update(
        mat, sig, np.zeros((1, 6)), None, 1e-3,
        {"F": np.eye(3)[None, :, :]})
    lam = 2.0
    F = np.diag([lam, lam ** -0.5, lam ** -0.5])[None, :, :]
    _, _, c2 = law42_ogden.solid_update(mat, sig, np.zeros((1, 6)),
                                        None, 1e-3, {"F": F})
    # ground state: c1 matches the material's linear-elastic estimate
    assert c1[0] == pytest.approx(mat.sound_speed_solid(), rel=1e-6)
    assert c2[0] > 1.2 * c1[0]


def test_law42_hexa_kernel_deformation_gradient(tmp_path):
    """The solid kernel computes F exactly from the initial gradients:
    stretch the cube's coordinates and the stress must match the
    closed form for that F (uniaxial-incompressible here)."""
    deck = (
        "/BEGIN\nrubber cube\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\nrubber\n1 1\n"
        "/MAT/LAW42/1\nrubber\n1.0e-6\n0.002 -0.0004\n2.0 -2.0\n0.495\n"
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    g = model.bricks
    mat = g.state["slices"][0][1]
    assert mat.law == 42
    assert "dndx0" in g.state                    # F plumbing switched on

    lam = 1.5
    model.x[:, 0] *= lam
    model.x[:, 1] *= lam ** -0.5
    model.x[:, 2] *= lam ** -0.5
    v = np.zeros_like(model.x)
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    dtc = solid_hexa8.forces(g, model.x, v, model.vr, 1e-4, fint, mint)
    s = g.state["sig"][0]
    assert s[0] - s[1] == pytest.approx(
        _ogden_uniaxial([0.002, -0.0004], [2.0, -2.0], lam), rel=1e-9)
    assert np.abs(fint.sum(axis=0)).max() < 1e-12     # free-body balance
    # the time step must have shrunk relative to the undeformed element:
    # c grew (tangent stiffening) while lc shrank (lateral contraction)
    c0 = mat.sound_speed_solid()
    assert dtc[0] < g.state["dtfac"][0] * (lam ** -0.5) / c0


# ============================================================================
# /FAIL criteria
# ============================================================================

def test_fail_johnson_triaxiality_and_accumulation():
    """Uniaxial tension has sigma* = 1/3 exactly: the damage increment
    must be d_eps_p / [(D1 + D2 exp(D3/3))]."""
    fail = FailureModel(type="JOHNSON", params={
        "D1": 0.05, "D2": 3.44, "D3": -2.12, "D4": 0.0, "eps_dot_0": 1.0})
    sig = np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])   # uniaxial tension
    dama = np.zeros(1)
    eps_f = 0.05 + 3.44 * np.exp(-2.12 / 3.0)
    broken = johnson.solid_step(fail, sig, np.array([0.01]),
                                np.zeros((1, 6)), 1e-3, dama)
    assert dama[0] == pytest.approx(0.01 / eps_f, rel=1e-12)
    assert not broken[0]
    # drive to rupture
    n_more = int(np.ceil(eps_f / 0.01)) + 1
    for _ in range(n_more):
        broken = johnson.solid_step(fail, sig, np.array([0.01]),
                                    np.zeros((1, 6)), 1e-3, dama)
    assert broken[0]


def test_fail_biquad_hits_the_five_anchor_points():
    """The two parabolas must pass exactly through the five calibration
    strains at the canonical triaxialities."""
    params = {"c1": 0.9, "c2": 0.55, "c3": 0.35, "c4": 0.25, "c5": 0.30, "s_flag": 0}
    biquad.fit(params)
    fail = FailureModel(type="BIQUAD", params=params)
    tri = np.array([-1 / 3, 0.0, 1 / 3, 1 / np.sqrt(3), 2 / 3])
    ef = biquad.eps_f(fail, tri)
    for k, key in enumerate(("c1", "c2", "c3", "c4", "c5")):
        assert ef[k] == pytest.approx(params[key], rel=1e-12)


def test_hexa_deletion_by_fail_johnson(tmp_path):
    """Kernel-level deletion: a hexa with /FAIL/JOHNSON (constant
    eps_f = D1) must switch OFF when eps_p reaches D1, zero its stress
    and stop producing forces and time-step constraints."""
    deck = (
        "/BEGIN\nfailing cube\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n"
        "/MAT/LAW2/1\nsteel\n7.8e-6\n210. 0.3\n0.4 0.5 0.5\n"
        "/FAIL/JOHNSON/1\n0.05 0 0 0 0\n"
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    g = model.bricks
    assert g.state["chk_fail"] and "dama" in g.state

    rate, dt = 1e-2, 1e-2                       # 1e-4 strain per call
    v = np.zeros_like(model.x)
    v[:, 0] = model.x[:, 0] * rate
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    for _ in range(2000):
        dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
        if g.state["off"][0] == 0.0:
            break
    assert g.state["off"][0] == 0.0
    # plastic strain at rupture ~ eps_f = D1 (one-step overshoot allowed)
    assert g.state["epsp"][0] == pytest.approx(0.05, rel=0.05)
    assert np.abs(g.state["sig"]).max() == 0.0
    # a dead element: no force, no dt claim, frozen plastic strain
    fint[:] = 0.0
    ep_before = g.state["epsp"][0]
    dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
    assert np.abs(fint).max() < 1e-14
    assert dtc[0] > 1e20
    assert g.state["epsp"][0] == ep_before


def test_shell_layer_failure_deletes_element(tmp_path):
    """Shell /FAIL: membrane tension breaks every layer at once, and with
    the default Ifail_sh = 1 the first broken layer kills the element —
    forces, moments and stresses all drop to zero."""
    deck = (
        "/BEGIN\nfailing shell\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nplate\n1 1\n"
        "/MAT/LAW2/1\nsteel\n7.8e-6\n210. 0.3\n0.4 0.5 0.5\n"
        "/FAIL/BIQUAD/1\n0.9 0.55 0.35 0.25 0.30\n"
        "/PROP/SHELL/1\nplate\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    g = model.shells
    assert g.state["chk_fail"]

    rate, dt = 1e-1, 1e-2                       # strong biaxial pull
    v = np.zeros_like(model.x)
    v[:, 0] = model.x[:, 0] * rate
    v[:, 1] = model.x[:, 1] * rate
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    for _ in range(500):
        shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
        if g.state["off"][0] == 0.0:
            break
    assert g.state["off"][0] == 0.0
    assert np.abs(g.state["sig"]).max() == 0.0
    assert np.abs(g.state["qshear"]).max() == 0.0
    fint[:] = 0.0
    mint[:] = 0.0
    dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
    assert np.abs(fint).max() < 1e-14 and np.abs(mint).max() < 1e-14
    assert dtc[0] > 1e20


def test_eps_p_max_deletes_solid(tmp_path):
    """The /MAT/LAW2 eps_p_max threshold (an M1 documented gap) now
    deletes the element through the same plumbing."""
    deck = (
        "/BEGIN\nepsmax cube\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n"
        "/MAT/LAW2/1\nsteel\n7.8e-6\n210. 0.3\n0.4 0.5 0.5 0.02\n"
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    g = model.bricks
    assert g.state["chk_fail"]
    rate, dt = 1e-2, 1e-2                       # 1e-4 strain per call
    v = np.zeros_like(model.x)
    v[:, 0] = model.x[:, 0] * rate
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    for _ in range(1000):
        solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
        if g.state["off"][0] == 0.0:
            break
    assert g.state["off"][0] == 0.0
    assert g.state["epsp"][0] == pytest.approx(0.02, rel=0.1)


# ============================================================================
# Plastic beams (LAW2 global model)
# ============================================================================

BEAM_PLASTIC_DECK = (
    "/BEGIN\nplastic beam\n"
    "/NODE\n1 0 0 0\n2 10 0 0\n3 0 1 0\n"
    "/BEAM/1\n1 1 2 3\n"
    "/PART/1\nbeam\n1 1\n"
    "/MAT/LAW2/1\nelastic-perfectly-plastic\n7.8e-6\n210. 0.3\n"
    "0.4 0.0 1.0\n"                       # A = 0.4, B = 0 -> sy = 0.4
    "/PROP/BEAM/1\nsquare 5x5\n25.0 52.0833333 52.0833333 88.0\n/END\n"
)


def test_beam_axial_plastic_saturation(tmp_path):
    """Elastic-perfectly-plastic beam pulled axially: N saturates at
    exactly A * sigma_y (the 1-D closed form the global model must hit)."""
    model, _ = _build(BEAM_PLASTIC_DECK, tmp_path)
    g = model.beams
    assert "epsp" in g.state
    v = np.zeros_like(model.x)
    vr = np.zeros_like(model.x)
    v[1, 0] = 10.0 * 1e-2                        # eps_dot = 1e-2
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    for _ in range(200):
        beam_type3.forces(g, model.x, v, vr, 1e-3, fint, mint)
    assert g.state["epsp"][0] > 0.0
    assert g.state["fres"][0, 0] == pytest.approx(25.0 * 0.4, rel=1e-9)


def test_beam_bending_plastic_saturation(tmp_path):
    """Pure bending: the global model yields at exactly M = W * sigma_y
    with the rectangle-exact elastic modulus W = sqrt(I*A/3) = b h^2/6
    (documented: no elastic-core spread toward the 1.5 W sy hinge)."""
    model, _ = _build(BEAM_PLASTIC_DECK, tmp_path)
    g = model.beams
    v = np.zeros_like(model.x)
    vr = np.zeros_like(model.x)
    vr[0, 1] = -1e-2                             # antisymmetric: pure ky
    vr[1, 1] = +1e-2
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    for _ in range(600):                         # elastic ~380 steps to yield
        beam_type3.forces(g, model.x, v, vr, 1e-3, fint, mint)
    Wy = np.sqrt(52.0833333 * 25.0 / 3.0)        # = 20.833 = 5*5^2/6
    assert Wy == pytest.approx(5.0 * 25.0 / 6.0, rel=1e-6)
    assert g.state["mres"][0, 1] == pytest.approx(Wy * 0.4, rel=1e-9)


def test_beam_elastic_below_yield_unchanged(tmp_path):
    """Below yield the plastic beam must behave exactly like the M2
    elastic one (regression guard on the elastic path)."""
    model, _ = _build(BEAM_PLASTIC_DECK, tmp_path)
    g = model.beams
    mat = g.state["slices"][0][1]
    v = np.zeros_like(model.x)
    vr = np.zeros_like(model.x)
    v[1, 0] = 10.0 * 1e-4
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    beam_type3.forces(g, model.x, v, vr, 1e-3, fint, mint)
    N = mat.E * 25.0 * 1e-4 * 1e-3
    assert g.state["fres"][0, 0] == pytest.approx(N, rel=1e-12)
    assert g.state["epsp"][0] == 0.0
