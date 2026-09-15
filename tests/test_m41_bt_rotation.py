"""M41 — BT large-rotation rate kinematics: the cdefo3.F second-order
rigid-rotation membrane correction and the spinning-strip canary.

The finding (VALIDATION M40 §3 + the M41 forensics)
---------------------------------------------------
On RD-E-1000 c41 (BT_type1 both engines) the port tracked the Fortran IE
to 4 digits until t≈900 ms of the roll, then a transverse-w hourglass
mode blew up ~100 ms AHEAD of the Fortran engine's own blow-up of the
SAME mode (the flutter is real physics of the rolled BT strip in BOTH
codes — Fortran's HE reaches 3.1e5 = 76 % of IE by its own end of run;
it only survives to TSTOP because with no /STOP card its energy-error
stop threshold is infinite, ecrit.F l.563 + freform.F DEMXS=EP30).  The
port's element defect was the MISSING cdefo3.F IHBE<=1 membrane-rate
correction; with it the port's flutter onset now sits at/below the
Fortran engine's on the differential mini-roll rig.

The upstream contract (cdefo3.F lines 103-130, the IHBE<=1 branch)
------------------------------------------------------------------
The corotational frame is evaluated on the end-of-step geometry while
the nodal velocities sit at mid-step, so an element rolling rigidly at
rate w about an IN-PLANE axis measures a spurious membrane stretching
rate: with v^{n-1/2} = (x^n - x^{n-1})/dt and the frame at x^n, the
in-plane leak of the out-of-plane rotation velocity is EXACTLY

    d_xx(raw) = + w^2 dt / 2          (frame leads velocities by dt/2)

Upstream compensates with the quadratic diagonal-vz term

    TMP1A = (dt/4) (VZ13-VZ24)^2 / (PY1+PY2)   ->  d_xx += -w^2 dt

i.e. cdefo3's type-1 branch OVERCORRECTS the bias by exactly 2x,
leaving d_xx(corrected) = -w^2 dt / 2 — the port mirrors the upstream
formula bit-for-bit, not the "ideal" zero.  (The IHBE==2/3 branch
subtracts GZX^2*(dt/2)*x_i from the velocities instead, which IS the
exact cancellation — a genuine upstream family asymmetry, kept.)

These tests pin
---------------
* the closed-form raw bias +w^2 dt/2 and the corrected -w^2 dt/2 on a
  stencil-exact rigid roll, at several accumulated roll angles (the
  correction is frame-covariant — angle-independent);
* the family gate: the correction runs for the BT type-1 cards (Ishell
  0/1/2 -> engine IHBE <= 1, hm_read_prop01.F lines 302-315) and is OFF
  for type 3 / type 4 / BATOZ decks;
* the spinning-strip canary: a free BT element spun through N = 3 full
  revolutions about an in-plane axis with a seeded w-hourglass velocity
  keeps its hourglass energy BOUNDED (no negative-damping flutter) and
  its internal energy at the centrifugal-elastic scale — the regression
  canary for the M40/M41 forensic headline.
"""

import contextlib
import io

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_bt4
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_node_groups,
                                              resolve_surfaces)

#: the c41 ROLLING element: 16 x 15.2 rectangle, t = 20, E = 1000, nu = 0,
#: rho = 0.01 (g/mm/ms units — the RD-E-1000 recipe).
A_, B_, T_ = 16.0, 15.2, 20.0


def _one_element(tmp_path, ishell=1):
    """One free c41-geometry BT shell, centered at the origin."""
    deck = (
        "/BEGIN\nm41 roll\n/NODE\n"
        f"1 {-A_ / 2} {-B_ / 2} 0\n2 {A_ / 2} {-B_ / 2} 0\n"
        f"3 {A_ / 2} {B_ / 2} 0\n4 {-A_ / 2} {B_ / 2} 0\n"
        "/SHELL/1\n1 1 2 3 4\n/PART/1\np\n1 1\n"
        "/MAT/LAW1/1\nroll\n0.01\n1000. 0.\n"
        f"/PROP/SHELL/1\nsh\n{ishell} 0 0 0\n0 0 0 0 0\n3 0 {T_}\n/END\n")
    f = tmp_path / "M41ROLL_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_starter_deck(read_deck(str(f)), model, log)
        build_element_groups(model, log)
        resolve_node_groups(model, log)
        resolve_surfaces(model, log)
        initialize_elements_and_mass(model, log)
    assert not log.errors, log.errors
    return model


def _roty(th):
    """Rotation matrix about the global y axis (the roll axis of the
    strip: an IN-PLANE axis of the x-y planar element)."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _stencil_roll(model, theta, w, dt):
    """Set the engine stencil for a rigid roll: x = R(theta) x0 (the
    end-of-step geometry the frame is built from) and v = the leapfrog
    mid-step velocity that carried x there, (x^n - x^{n-1})/dt."""
    x0 = model.x0
    model.x = x0 @ _roty(theta).T
    model.v = (model.x - x0 @ _roty(theta - w * dt).T) / dt
    model.vr = np.zeros_like(model.v)
    model.vr[:, 1] = w


# ---------------------------------------------------------------------------
# 1. the cdefo3.F closed form: raw bias +w^2 dt/2, corrected -w^2 dt/2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0.3, 1.7, 3.1, 5.9])
def test_rigid_roll_membrane_rate_correction(tmp_path, theta):
    """One production forces() call on a stencil-exact rigid roll: the
    membrane strain increment must be the upstream cdefo3 net
    -w^2 dt/2 * dt (the 2x overcorrection of the +w^2 dt/2 frame-lag
    bias), at every accumulated roll angle; bending and transverse
    shear stay at rigid-rotation zero."""
    w, dt = 0.05, 0.02
    n = 4
    fint, mint = np.zeros((n, 3)), np.zeros((n, 3))

    # corrected (production BT type-1 path)
    model = _one_element(tmp_path, ishell=1)
    g = model.shells
    _stencil_roll(model, theta, w, dt)
    shell_bt4.forces(g, model.x, model.v, model.vr, dt, fint, mint)
    deps_xx = g.state["sig"][0, :, 0].mean() / 1000.0   # E=1000, nu=0
    expect = -0.5 * w * w * dt * dt                     # rate * dt
    assert deps_xx == pytest.approx(expect, rel=1e-4)
    # rigid roll: no curvature, no accumulated transverse-shear stress
    # beyond the O(w^2 dt) stencil leak of the raw rates
    assert np.abs(g.state["qshear"][0]).max() < 1e-6
    assert np.abs(g.state["sig"][0, :, 1]).max() < 1e-3 * abs(
        g.state["sig"][0, 0, 0])

    # the raw bias (the pre-M41 kernel = the non-type-1 family path):
    # rot2_mask off -> +w^2 dt/2, the equal-and-opposite reading
    model2 = _one_element(tmp_path, ishell=1)
    g2 = model2.shells
    g2.state["ihbe_mask"][:] = 99
    _stencil_roll(model2, theta, w, dt)
    fint[:] = 0.0
    mint[:] = 0.0
    shell_bt4.forces(g2, model2.x, model2.v, model2.vr, dt, fint, mint)
    deps_raw = g2.state["sig"][0, :, 0].mean() / 1000.0
    assert deps_raw == pytest.approx(+0.5 * w * w * dt * dt, rel=1e-4)
    # correction magnitude = exactly -w^2 dt (2x the bias), upstream form
    assert (deps_xx - deps_raw) == pytest.approx(-w * w * dt * dt,
                                                 rel=1e-4)


# ---------------------------------------------------------------------------
# 2. the family gate (hm_read_prop01.F card -> engine IHBE map)
# ---------------------------------------------------------------------------

def test_correction_gated_on_bt_type1_family(tmp_path):
    """ihbe_mask maps the Ishell cards to engine IHBE formulations."""
    for card, expect in ((0, 0), (1, 1), (2, 0), (3, 2), (4, 4)):
        model = _one_element(tmp_path, ishell=card)
        assert model.shells.state["ihbe_mask"][0] == expect, card


# ---------------------------------------------------------------------------
# 3. the spinning-strip canary: N revolutions, hourglass energy bounded
# ---------------------------------------------------------------------------

def test_spinning_strip_hourglass_bounded(tmp_path):
    """A free c41 element spun through 3 FULL revolutions about its
    in-plane y axis (leapfrog on the production kernel, the engine
    stencil), with a seeded transverse-w hourglass velocity: the
    hourglass energy must stay BOUNDED (no exponential pumping — the
    M40 c41 flutter grew e^{2.7e-3/cycle}, which over these ~8700
    cycles would be 10 decades) and the internal energy must stay at
    the centrifugal-elastic scale.  Measured on the fixed kernel:
    HE_max 7.8e-8, IE_max 2.3e-4 — the bounds carry 60x / 20x margin
    without tolerating any exponential growth."""
    omega, revs, seed = 0.05, 3.0, 1e-5
    model = _one_element(tmp_path, ishell=1)
    g = model.shells
    n = model.numnod
    x = model.x0.copy()
    v = np.zeros((n, 3))
    v[:, 0] = omega * x[:, 2]
    v[:, 2] = -omega * x[:, 0]                  # rigid roll about y
    v[:, 2] += seed * np.array([1.0, -1.0, 1.0, -1.0])   # w-hourglass seed
    vr = np.zeros((n, 3))
    vr[:, 1] = omega
    m = model.mass[:, None]
    inert = model.inertia[:, None]
    fint, mint = np.zeros((n, 3)), np.zeros((n, 3))
    # the element's own stable-dt claim (dt=0 primes nothing: deps = 0)
    dt_e = shell_bt4.forces(g, x, v, vr, 0.0, fint, mint)
    dt = 0.9 * float(dt_e.min())
    ncyc = int(revs * 2.0 * np.pi / omega / dt)
    he = np.empty(ncyc)
    for k in range(ncyc):
        fint[:] = 0.0
        mint[:] = 0.0
        shell_bt4.forces(g, x, v, vr, dt, fint, mint)
        v += fint / m * dt
        vr += mint / inert * dt
        x += v * dt
        he[k] = g.state["ehour"].sum()
    ie_max = float(g.state["eint"].max())
    he_max = float(np.abs(he).max())
    assert he_max < 5e-6, f"hourglass energy pumped: HE_max {he_max:.3e}"
    assert ie_max < 5e-3, f"spurious membrane strain: IE_max {ie_max:.3e}"
    # no exponential growth across the roll: the late-window envelope is
    # not orders beyond the early one (flutter would be 10+ decades)
    early = np.abs(he[: ncyc // 4]).max()
    late = np.abs(he[-ncyc // 4:]).max()
    assert late < 100.0 * max(early, 1e-12), (early, late)


def test_bt_ishell_3_and_4_forces(tmp_path):
    """Ensure ishell=3 and ishell=4 (IHBE 2 and 4) execute forces without NameError (AUD-006)."""
    for ishell in (3, 4):
        m = _one_element(tmp_path, ishell=ishell)
        fint = np.zeros_like(m.x0)
        mint = np.zeros_like(m.x0)
        dt = shell_bt4.forces(m.shells, m.x0, np.zeros_like(m.x0), np.zeros_like(m.x0), 1e-6, fint, mint)
        assert np.all(dt > 0.0)

