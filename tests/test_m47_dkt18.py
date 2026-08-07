"""M39 — shell fidelity: the Belytschko-Tsay hourglass against upstream.

Fortran reference: ``engine/source/elements/shell/coque/chvis3.F``, reached
from ``cforc3.F`` (lines 593-638) for every Ishell except 2 — i.e. the
/PROP/SHELL default — with the engine constants pinned in
``engine/source/engine/radioss2.F`` lines 638-640::

    HELAS = HALF      HVISC = HALF      HVLIN = ZERO

HVLIN = 0 kills chvis3's LINEAR (sound-speed) viscous branch identically,
leaving each mode with an ELASTIC stiffness plus a QUADRATIC viscous
damper::

    SHFPR3 = SHF / (3 (1 + nu))                            (chvis3 l.143)
    (B1+B2) = PX1^2+PY1^2+PX2^2+PY2^2                      (chvis3 l.174)

    elastic   HH1 = hm E t / 8                       modes 0,1 (membrane)
              HH2 = hf E SHFPR3 t^3 / (8 (B1+B2))    mode 2    (bending)
    viscous   H1Q = (25/2) rho hm t sqrt(A)          modes 0,1
              H2Q = (25/2) rho hf sqrt(SHFPR3) t^2   mode 2
              H3Q = (25/2) 0.072169 rho hr t^2 A     modes 3,4

Upstream keeps its gradient operators AREA-scaled (``cderi3.F`` l.172:
``PX1 = HALF*(Y2-Y4)``, so PX = A * B with B the operator this port
builds), hence ``(B1+B2)_upstream = A^2 * bb / 2`` and the elastic
coefficients carry NO area factor.

Before M39 ``shell_bt4`` ran the LS-DYNA **BLT84** hourglass instead
(``k_m = hm E t A bb / 8``, ``k_w = hf kappa G t A bb / 8``,
``k_r = hr E t^3 A bb / 192``, no viscous branch at all). Measured on
examples/box_beam_impact against the Fortran engine:

    HE (hourglass energy) rel-RMS   0.584  ->  0.094
    HE as % of peak(IE+KE)          0.069  ->  5.32   (Fortran 4.46)

The old form was not "2 orders low in stiffness" as M36 read it — it was
too STIFF (membrane by A*bb ~ 2, transverse by ~3 (B1+B2)^2/(A t^2), about
300x at box_beam's L/t = 10) but purely ELASTIC, so it STORED the
hourglass energy and dissipated none. The quadratic dampers are where the
Fortran's 4.4 % lives.
"""

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

STEEL = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""

T_PLATE = 0.1                       # plate thickness of the unit-square deck


def _plate(tmp_path, hm=0.01, hf=0.01, hr=0.01, t=T_PLATE, ishell=1):
    """One unit-square BT4 shell (node 1 at the origin, area 1, bb = 2)."""
    deck = (
        "/BEGIN\nm39 unit shell\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nplate\n1 1\n" + STEEL +
        f"/PROP/SHELL/1\nplate prop\n{ishell} 0 0 0\n{hm} {hf} {hr} 0 0\n"
        f"3 0 {t}\n/END\n"
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
    assert not log.errors, log.errors
    g = model.shells
    return g, model, g.state["slices"][0][1]


@pytest.fixture
def shell_plate(tmp_path):
    """The default unit plate: hm = hf = hr = 0.01, t = 0.1."""
    return _plate(tmp_path)


@pytest.fixture
def make_shell_plate(tmp_path):
    """Factory for a unit plate with chosen hourglass coefficients."""
    def _make(**kw):
        return _plate(tmp_path, **kw)
    return _make


# ---------------------------------------------------------------------------
# coefficient-level parity with chvis3.F
# ---------------------------------------------------------------------------

def _expect(mat, hm, hf, hr, t, A, bb):
    """The chvis3.F coefficients, written out independently of the port."""
    b12 = A ** 2 * bb / 2.0                    # (B1+B2) upstream PX scaling
    shf = 5.0 / 6.0                            # ccoef3.F: Ashear=0 -> SHF=FSH
    shfpr3 = shf / (3.0 * (1.0 + mat.nu))
    return dict(
        hh1=hm * mat.E * t / 8.0,
        hh2=hf * mat.E * shfpr3 * t ** 3 / (8.0 * b12),
        h1q=12.5 * mat.rho0 * hm * t * np.sqrt(A),
        h2q=12.5 * mat.rho0 * hf * np.sqrt(shfpr3) * t ** 2,
        h3q=12.5 * 0.072169 * mat.rho0 * hr * t ** 2 * A,
    )


def test_membrane_hourglass_matches_chvis3(shell_plate):
    """Membrane mode: elastic HH1 = hm E t / 8 (NO area/bb factor — the
    A*bb the port carried before M39 came from LS-DYNA's BLT84 form) plus
    the quadratic damper H1Q."""
    g, model, mat = shell_plate
    t, A, hm = 0.1, 1.0, 0.01
    e = _expect(mat, hm, 0.01, 0.01, t, A, bb=2.0)

    v0, dt = 1e-3, 1e-3
    v = np.zeros_like(model.x)
    v[:4, 0] = np.array([1.0, -1.0, 1.0, -1.0]) * v0     # x-hourglass
    f = np.zeros_like(model.x)
    m = np.zeros_like(model.x)
    shell_bt4.forces(g, model.x, v, model.vr, dt, f, m)

    qd = 4.0 * v0                              # gamma = h on the square
    want = e["hh1"] * qd * dt + qd * e["h1q"] * abs(qd)
    assert f[0, 0] == pytest.approx(-want, rel=1e-10)
    # the hourglass pattern makes no real membrane stress
    assert np.abs(g.state["sig"]).max() < 1e-15


def test_bending_hourglass_matches_chvis3(shell_plate):
    """Transverse 'w' mode: HH2 = hf E SHFPR3 t^3 / (8 (B1+B2)) — note the
    DIVISION by (B1+B2), where BLT84 multiplied by A*bb."""
    g, model, mat = shell_plate
    t, A, hf = 0.1, 1.0, 0.01
    e = _expect(mat, 0.01, hf, 0.01, t, A, bb=2.0)

    w0, dt = 1e-3, 1e-3
    v = np.zeros_like(model.x)
    v[:4, 2] = np.array([1.0, -1.0, 1.0, -1.0]) * w0
    f = np.zeros_like(model.x)
    m = np.zeros_like(model.x)
    shell_bt4.forces(g, model.x, v, model.vr, dt, f, m)

    qd = 4.0 * w0
    want = e["hh2"] * qd * dt + qd * e["h2q"] * abs(qd)
    assert f[0, 2] == pytest.approx(-want, rel=1e-10)


def test_rotation_hourglass_is_purely_viscous(shell_plate):
    """chvis3 lines 332-334 ASSIGN HOUR(4..5) = HG*(H3L+H3Q|HG|) rather
    than accumulating them: the rotation modes have NO elastic branch, and
    with HVLIN = 0 only the quadratic damper survives. Their modal rate
    also rides the RAW h = (1,-1,1,-1), not gamma (lines 327-330)."""
    g, model, mat = shell_plate
    t, A, hr = 0.1, 1.0, 0.01
    e = _expect(mat, 0.01, 0.01, hr, t, A, bb=2.0)

    r0, dt = 1e-3, 1e-3
    vr = np.zeros_like(model.vr)
    vr[:4, 0] = np.array([1.0, -1.0, 1.0, -1.0]) * r0     # thx-hourglass
    f = np.zeros_like(model.x)
    m = np.zeros_like(model.x)
    shell_bt4.forces(g, model.x, np.zeros_like(model.x), vr, dt, f, m)

    qd = 4.0 * r0
    want = qd * e["h3q"] * abs(qd)             # damper alone, no HH term
    assert m[0, 0] == pytest.approx(-want, rel=1e-10)
    # purely viscous -> no persistent elastic state is stored for 3,4
    assert np.abs(g.state["hgq"][:, 3:]).max() == 0.0

    # ... and because it is a damper, a CONSTANT rate gives a CONSTANT
    # force (an elastic branch would keep accumulating)
    f2 = np.zeros_like(model.x)
    m2 = np.zeros_like(model.x)
    shell_bt4.forces(g, model.x, np.zeros_like(model.x), vr, dt, f2, m2)
    assert m2[0, 0] == pytest.approx(-want, rel=1e-10)


def test_viscous_hourglass_dissipates_and_never_returns(shell_plate):
    """The M36 box_beam finding in miniature, on the rotation modes — the
    ones chvis3 makes PURELY viscous, so they isolate dissipation from
    storage exactly.

    A damper cannot give energy back: driving the mode out and then back
    at the mirrored rate books dt*H3Q*|qd|^3 EVERY cycle, both legs, and
    HE rises monotonically to 2N dt H3Q |qd|^3. The pre-M39 BLT84 control
    made these modes elastic, so the same round trip returned everything
    and HE ended at ~0 — which is why box_beam_impact booked 0.069 % of
    its energy in hourglass control against the Fortran's 4.46 %.
    """
    g, model, mat = shell_plate
    r0, dt, N = 1e-2, 1e-3, 20
    e = _expect(mat, 0.01, 0.01, 0.01, T_PLATE, 1.0, bb=2.0)
    f = np.zeros_like(model.x)
    m = np.zeros_like(model.x)
    h = np.array([1.0, -1.0, 1.0, -1.0])
    zero_v = np.zeros_like(model.x)
    qd = 4.0 * r0
    per_cycle = dt * e["h3q"] * abs(qd) ** 3      # dt * F * qd, F = qd H3Q|qd|

    seen = []
    for leg in (+1.0, -1.0):
        for _ in range(N):
            vr = np.zeros_like(model.vr)
            vr[:4, 0] = h * r0 * leg
            shell_bt4.forces(g, model.x, zero_v, vr, dt, f, m)
            seen.append(g.state["ehour"][0])

    # monotone: every cycle adds energy, the mirrored leg included
    assert all(b > a for a, b in zip(seen, seen[1:]))
    # and it adds exactly the analytic damper work each cycle
    assert seen[-1] == pytest.approx(2 * N * per_cycle, rel=1e-9)
    # nothing was stored: the rotation modes hold no elastic state
    assert np.abs(g.state["hgq"][:, 3:]).max() == 0.0


def test_elastic_hourglass_energy_follows_chvis3_integration(shell_plate):
    """The membrane/bending modes DO store: their Q returns to zero over a
    mirrored round trip. chvis3 books EHOU with the ALREADY-UPDATED HOUR
    (lines 286/298: HOUR += HG*HH1 then EHOU = HOUR1A*HG1), a backward
    -Euler quadrature whose O(dt) residual over the trip is exactly
    2N k qd^2 dt^2 / 2 -> this pins the port to upstream's quadrature
    rather than to the (more accurate, but wrong-reference) midpoint rule
    the port used before M39."""
    g, model, mat = shell_plate
    w0, dt, N = 1e-2, 1e-3, 20
    e = _expect(mat, 0.01, 0.01, 0.01, T_PLATE, 1.0, bb=2.0)
    f = np.zeros_like(model.x)
    m = np.zeros_like(model.x)
    h = np.array([1.0, -1.0, 1.0, -1.0])
    qd = 4.0 * w0

    for leg in (+1.0, -1.0):
        for _ in range(N):
            v = np.zeros_like(model.x)
            v[:4, 2] = h * w0 * leg
            shell_bt4.forces(g, model.x, v, model.vr, dt, f, m)

    # the elastic amplitude is fully recovered
    assert np.abs(g.state["hgq"][0, 2]) < 1e-18
    # residual = backward-Euler bias (N k qd^2 dt^2) + the damper's 2N legs
    bias = N * e["hh2"] * qd ** 2 * dt ** 2
    visc = 2 * N * dt * e["h2q"] * abs(qd) ** 3
    assert g.state["ehour"][0] == pytest.approx(bias + visc, rel=1e-9)


def test_linear_velocity_field_makes_no_hourglass(shell_plate):
    """Flanagan-Belytschko orthogonality (chvis3 GAMA1..4, lines 122-140):
    gamma is orthogonal to rigid motion AND to any linear field, so neither
    the elastic state nor the viscous force may respond to one."""
    g, model, _ = shell_plate
    dt = 1e-3
    v = np.zeros_like(model.x)
    v[:, 0] = 0.4 + 0.9 * model.x[:, 0] - 0.3 * model.x[:, 1]
    v[:, 1] = -0.2 + 0.1 * model.x[:, 0] + 0.5 * model.x[:, 1]
    v[:, 2] = 0.3 + 0.7 * model.x[:, 0] - 0.2 * model.x[:, 1]
    f = np.zeros_like(model.x)
    m = np.zeros_like(model.x)
    shell_bt4.forces(g, model.x, v, model.vr, dt, f, m)
    assert np.abs(g.state["hgq"]).max() < 1e-15
    assert g.state["ehour"][0] == pytest.approx(0.0, abs=1e-18)


def test_hourglass_forces_are_self_equilibrated(shell_plate):
    """Whatever the coefficients, the hourglass must inject no net force or
    torque: sum_i gamma_i = 0 and sum_i h_i = 0 on any quad."""
    g, model, _ = shell_plate
    dt = 1e-3
    rng = np.random.default_rng(39)
    v = rng.normal(scale=1e-3, size=model.x.shape)
    vr = rng.normal(scale=1e-3, size=model.vr.shape)
    f = np.zeros_like(model.x)
    m = np.zeros_like(model.x)
    shell_bt4.forces(g, model.x, v, vr, dt, f, m)
    assert np.abs(f.sum(axis=0)).max() < 1e-12
    total = m.sum(axis=0) + np.cross(model.x, f).sum(axis=0)
    assert np.abs(total).max() < 1e-12


# ---------------------------------------------------------------------------
# /PROP/SHELL hourglass defaults (hm_read_prop01.F lines 204-212)
# ---------------------------------------------------------------------------

def test_zero_hourglass_coefficients_take_the_upstream_default(
        make_shell_plate):
    """A blank/zero Hm Hf Hr does NOT mean "no hourglass": hm_read_prop01.F
    lines 208-212 substitute EM02 = 0.01 for each zero (for any Ishell but
    3). The deck cannot switch the control off this way — a 1-point element
    with no hourglass control is rank deficient — so the port's 0 -> 0.01
    fallback is upstream behaviour, not a shortcut."""
    g0, model, _ = make_shell_plate(hm=0.0, hf=0.0, hr=0.0)
    p0 = g0.state["slices"][0][2].params
    assert (p0["hm"], p0["hf"], p0["hr"]) == (0.01, 0.01, 0.01)


def test_ishell3_hourglass_defaults_are_ten_times_higher(make_shell_plate):
    """hm_read_prop01.F lines 204-207: Ishell = 3 (the BT type-3 / Hallquist
    -Liu member of the family) defaults Hm and Hf to **EM01 = 0.1**, ten
    times the EM02 every other Ishell gets — Hr stays EM02. The port read
    every Ishell the same before M39, running type-3 decks (official cases
    c42/c43, E1000_Bending_BT_BT_type3) at a tenth of their membrane and
    flexural hourglass."""
    g3, _, _ = make_shell_plate(hm=0.0, hf=0.0, hr=0.0, ishell=3)
    p3 = g3.state["slices"][0][2].params
    assert (p3["hm"], p3["hf"], p3["hr"]) == (0.1, 0.1, 0.01)

    # an EXPLICIT value is always kept — the default only fills a zero
    g3e, _, _ = make_shell_plate(hm=0.02, hf=0.03, hr=0.04, ishell=3)
    p3e = g3e.state["slices"][0][2].params
    assert (p3e["hm"], p3e["hf"], p3e["hr"]) == (0.02, 0.03, 0.04)
