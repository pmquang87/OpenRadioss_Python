"""
M10 validations: implicit DYNAMICS — Newmark-beta time integration with
HHT-alpha numerical dissipation (/IMPL/DYNA).

Every new capability gets at least one ANALYTIC check (the port's
philosophy):

* the lumped mass matrix in equation space (model.mass on translations,
  model.inertia on shell rotations, /BCS condensation);
* an SDOF free vibration whose measured period elongation MATCHES Newmark's
  known O(dt^2) dispersion, (omega dt)^2 / 12 for the trapezoidal rule —
  quantitatively, not just "small" — with the exact amplitude and the
  discrete energy conserved to round-off;
* UNCONDITIONAL stability demonstrated far above the explicit critical time
  step (the leapfrog is unstable for omega dt > 2 — asserted on the same
  discrete system — while the implicit run stays bounded and energy-exact);
* HHT alpha < 0 draining the unresolvable high modes of a bar while the
  trapezoidal rule conserves the total (energy balance asserted both ways);
* the quasi-static limit reproducing the M8 implicit-STATIC answer on the
  shell cantilever;
* a step-force transient matching BOTH the closed form and the EXPLICIT
  solver on the same deck (cross-solver consistency);
* with /IMPL/NONLIN: a large-rotation swinging pendulum whose quarter
  period matches the elliptic-integral closed form, staying stable with a
  quadratically converging Newton;
* the /IMPL/DYNA card mirror of the original reader (freimpl.F: /1 = HHT
  alpha itself, /2 = gamma beta, /DAMP deferred) and the explicit deferral
  of the LAW2 strain-rate term (never du/1 — see dynamics.py).

See PORTING_GUIDE.md roadmap M10.
"""

import contextlib
import io

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

pytest.importorskip("scipy")   # implicit requires scipy (optional otherwise)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model


def _starter_model(make_deck, name, starter):
    s, _ = make_deck(name, starter, "#\n/RUN/X/1\n1.0\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


# ---- an SDOF: a single axial truss, tip node free along x only -------------
# k = EA/L, m = rho*A*L/2 lumped at the tip -> omega = sqrt(k/m) exactly
# (the kernels see the frozen frame under linear geometry, so the discrete
# system IS this SDOF, bit for bit).
L_SD, E_SD, A_SD, RHO_SD = 100.0, 210.0, 1.0, 7.8e-6
M_SD = RHO_SD * A_SD * L_SD / 2.0
K_SD = E_SD * A_SD / L_SD
OM_SD = np.sqrt(K_SD / M_SD)
T_SD = 2.0 * np.pi / OM_SD


def _sdof_deck(v0=0.0, force=0.0):
    """Two-node truss along x; node 1 pinned, node 2 free along x only.
    ``v0``: /INIVEL initial axial velocity; ``force``: constant axial
    /CLOAD (a step load switched on at t = 0)."""
    load = ""
    if force:
        load = f"""/FUNCT/1
step
       0.0       1.0
    1000.0       1.0
/CLOAD/1
axial step
         1         X         2       {force}
"""
    inivel = ""
    if v0:
        inivel = f"""/INIVEL/TRA/1
kick
      {v0}       0.0       0.0         2
"""
    return f"""\
#RADIOSS STARTER
/BEGIN
SDOF
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{L_SD:20.10f}{0.0:20.10f}{0.0:20.10f}
/TRUSS/1
         1         1         2
/PART/1
bar
         1         1
/MAT/LAW1/1
steel
   {RHO_SD}
     {E_SD}       0.0
/PROP/TRUSS/1
bar
       {A_SD}
/GRNOD/NODE/1
pivot
1
/GRNOD/NODE/2
tip
2
/BCS/1
pin
       111       000         0         1
/BCS/2
axial only
       011       000         0         2
{inivel}{load}/END
"""


def _dyna_engine(name, t_end, dt, card="/IMPL/DYNA/2\n0.5  0.25\n",
                 extra=""):
    return (f"#\n/RUN/{name}/1\n{t_end}\n{card}"
            f"/IMPL/DTINI\n{dt}\n{extra}/END\n")


def _tip_u(model, node_id=2, comp=0):
    """Tip-displacement time series from the dynamics history."""
    idx = model.node_index(node_id)
    h = model.implicit_result.history
    t = np.array(h["t"])
    u = np.array([uu[idx, comp] for uu in h["u"]])
    return t, u


def _period_from_upcrossings(t, u):
    """Mean period between successive zero UP-crossings (linear interp)."""
    s = np.sign(u)
    ups = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    assert len(ups) >= 2, "need at least two up-crossings"

    def cross(i):
        return t[i] + (t[i + 1] - t[i]) * (-u[i]) / (u[i + 1] - u[i])

    return (cross(ups[-1]) - cross(ups[0])) / (len(ups) - 1)


# ----------------------------------------------------------------------------
# The lumped mass matrix in equation space
# ----------------------------------------------------------------------------

def test_lumped_mass_matrix(make_deck):
    """M is the DIAGONAL of model.mass on translational equations and
    model.inertia on shell rotational equations, condensed exactly like K
    (/BCS rows dropped) — no consistent-mass matrix, like the original's
    MS/IN use in IMP_DYNAM."""
    from pyradioss.implicit.dofmap import DofMap
    from pyradioss.implicit.dynamics import _lumped_mass_eq

    # a shell strip: nodes carry mass AND rotational inertia
    nx, ny, Lx, b, t = 4, 2, 40.0, 10.0, 1.0

    def nid(i, j):
        return 1 + i + j * (nx + 1)
    nodes = [f"{nid(i,j):10d}{i*Lx/nx:20.10f}{j*b/ny:20.10f}{0.0:20.10f}"
             for j in range(ny + 1) for i in range(nx + 1)]
    shells = []
    eid = 0
    for j in range(ny):
        for i in range(nx):
            eid += 1
            c = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            shells.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in c))
    fixed = [nid(0, j) for j in range(ny + 1)]
    starter = f"""\
#RADIOSS STARTER
/BEGIN
MASS
/NODE
{chr(10).join(nodes)}
/SHELL/1
{chr(10).join(shells)}
/PART/1
plate
         1         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.3
/PROP/SHELL/1
shell
       {t}         3      0.01
/GRNOD/NODE/1
fixed
{" ".join(map(str, fixed))}
/BCS/1
clamp
       111       111         0         1
/END
"""
    model = _starter_model(make_deck, "MASSM", starter)
    dof = DofMap(model)
    M = _lumped_mass_eq(model, dof)
    assert M.shape == (dof.ndof,)
    assert (M > 0.0).all()          # every shell equation carries mass/inertia
    # translational equations carry the nodal mass, rotational the inertia
    n = model.numnod
    for i in range(n):
        for c in range(3):
            e = dof.eq[i * 6 + c]
            if e >= 0:
                assert M[e] == model.mass[i]
            e = dof.eq[i * 6 + 3 + c]
            if e >= 0:
                assert M[e] == model.inertia[i]
    # the clamped row contributes no equations at all (condensed, like K)
    for nn in fixed:
        i = model.node_index(nn)
        assert (dof.eq[i * 6:(i + 1) * 6] == -1).all()
    # total translational mass in M = 3 * (sum of free nodal masses)
    free = np.ones(n, dtype=bool)
    free[[model.node_index(nn) for nn in fixed]] = False
    assert np.isclose(M.sum(),
                      3.0 * model.mass[free].sum()
                      + 3.0 * model.inertia[free].sum())


# ----------------------------------------------------------------------------
# Free vibration: exact amplitude, Newmark's O(dt^2) period dispersion,
# discrete energy conservation (trapezoidal rule)
# ----------------------------------------------------------------------------

def test_sdof_period_dispersion_and_energy(make_deck):
    """v0-kick free vibration of the SDOF truss: amplitude = v0/omega to
    round-off, the measured period elongation MATCHES the closed-form
    Newmark (trapezoidal) dispersion (omega dt)^2 / 12, the error drops
    4x when dt halves (O(dt^2)), and the discrete energy is conserved to
    round-off (the trapezoidal rule's exact conservation on linear
    systems)."""
    v0 = 0.01
    amp = v0 / OM_SD
    errs = []
    for div in (40, 80):                      # dt = T/40, then T/80
        dt = T_SD / div
        starter = _sdof_deck(v0=v0)
        model = _run(make_deck, f"SDF{div}", starter,
                     _dyna_engine(f"SDF{div}", 3.2 * T_SD, dt))
        res = model.implicit_result
        assert res.converged
        # linear problem: one Newton solve per step (2 residual evals)
        assert max(i.iterations for i in res.increments) == 2

        t, u = _tip_u(model)
        assert np.abs(u).max() == pytest.approx(amp, rel=1e-3)
        pe = _period_from_upcrossings(t, u) / T_SD - 1.0
        pe_exact = (OM_SD * dt) ** 2 / 12.0
        assert pe == pytest.approx(pe_exact, rel=0.05)
        errs.append(pe)

        # trapezoidal rule: the energy balance closes to round-off
        h = res.history
        e0 = 0.5 * M_SD * v0 ** 2
        assert np.abs(h["bal"]).max() < 1e-10 * e0
    # O(dt^2): halving dt quarters the period error
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.1)


def test_unconditional_stability_beyond_explicit_dt(make_deck):
    """The explicit leapfrog is unstable beyond omega dt = 2 (= the truss
    critical step L/c for this SDOF) — asserted on the same discrete
    system by direct recursion. The implicit trapezoidal run at
    dt = 20x that limit stays BOUNDED at the exact amplitude and conserves
    the discrete energy over many steps: unconditional stability."""
    v0 = 0.01
    dt_crit = 2.0 / OM_SD               # omega dt = 2 (= L/c here)
    dt = 20.0 * dt_crit

    # the leapfrog on the same SDOF at this dt explodes within a few steps
    u_e, v_e = 0.0, v0
    growth = []
    for _ in range(50):
        a_e = -(K_SD / M_SD) * u_e
        v_e = v_e + a_e * dt
        u_e = u_e + v_e * dt
        growth.append(abs(u_e))
    assert growth[-1] > 1e6 * (v0 / OM_SD), "leapfrog should be unstable here"

    starter = _sdof_deck(v0=v0)
    model = _run(make_deck, "STAB", starter,
                 _dyna_engine("STAB", 100.0 * dt, dt))
    res = model.implicit_result
    assert res.converged
    t, u = _tip_u(model)
    # bounded by the exact amplitude (the trapezoidal rule cannot grow)
    assert np.abs(u).max() <= (v0 / OM_SD) * (1.0 + 1e-9)
    h = res.history
    e0 = 0.5 * M_SD * v0 ** 2
    assert abs(h["bal"][-1]) < 1e-10 * e0


# ----------------------------------------------------------------------------
# HHT-alpha: high-mode dissipation vs trapezoidal conservation (a bar)
# ----------------------------------------------------------------------------

def _bar_chain_deck(nel, v0):
    """A fixed-free bar as a chain of ``nel`` equal trusses along x, every
    node restrained to x-motion, uniform initial velocity v0 — the jump
    excitation spreads energy across ALL nel longitudinal modes."""
    h = L_SD / nel
    nodes = [f"{i+1:10d}{i*h:20.10f}{0.0:20.10f}{0.0:20.10f}"
             for i in range(nel + 1)]
    bars = [f"{i+1:10d}{i+1:10d}{i+2:10d}" for i in range(nel)]
    moving = " ".join(str(i + 2) for i in range(nel))
    return f"""\
#RADIOSS STARTER
/BEGIN
BAR
/NODE
{chr(10).join(nodes)}
/TRUSS/1
{chr(10).join(bars)}
/PART/1
bar
         1         1
/MAT/LAW1/1
steel
   {RHO_SD}
     {E_SD}       0.0
/PROP/TRUSS/1
bar
       {A_SD}
/GRNOD/NODE/1
root
1
/GRNOD/NODE/2
moving
{moving}
/BCS/1
clamp root
       111       000         0         1
/BCS/2
axial only
       011       000         0         2
/INIVEL/TRA/1
uniform kick
      {v0}       0.0       0.0         2
/END
"""


def test_hht_damps_high_modes_trapezoidal_conserves(make_deck):
    """An 8-element bar with a uniform velocity jump (energy in every
    longitudinal mode), stepped at dt that leaves the high modes
    unresolvable: the TRAPEZOIDAL rule conserves the total energy to
    round-off (no algorithmic dissipation at ANY frequency — its spectral
    radius is 1), while HHT alpha = -0.3 drains the high-mode content
    monotonically toward the resolved low modes (rho_inf = 7/13) — the
    documented role of the alpha-method."""
    nel, v0 = 8, 0.01
    c = np.sqrt(E_SD / RHO_SD)
    T1 = 4.0 * L_SD / c              # fundamental period of the fixed-free bar
    # dt resolves mode 1 (omega1 dt ~ 0.4) but leaves modes 2+ unresolvable
    # (omega2 dt ~ 1.2 ... omega8 dt ~ 4): ~19 % of the jump's energy sits
    # there (participation 8/((2j-1) pi)^2) and HHT must drain it
    dt = T1 / 16.0
    e0 = 0.5 * (RHO_SD * A_SD * L_SD - RHO_SD * A_SD * L_SD / nel / 2.0) \
        * v0 ** 2                    # KE of the moving nodes

    starter = _bar_chain_deck(nel, v0)
    m_tr = _run(make_deck, "BTRAP", starter,
                _dyna_engine("BTRAP", 4.0 * T1, dt))
    m_ht = _run(make_deck, "BHHT", starter,
                _dyna_engine("BHHT", 4.0 * T1, dt, card="/IMPL/DYNA/1\n"
                             "-0.3\n"))
    assert m_tr.implicit_result.converged
    assert m_ht.implicit_result.converged
    assert m_ht.implicit_result.alpha == -0.3
    assert m_ht.implicit_result.gamma == pytest.approx(0.8)
    assert m_ht.implicit_result.beta == pytest.approx(0.4225)

    def energy(m):
        h = m.implicit_result.history
        return np.array(h["ke"]) + np.array(h["ie"]), np.array(h["bal"])

    e_tr, bal_tr = energy(m_tr)
    e_ht, _ = energy(m_ht)
    assert np.abs(bal_tr).max() < 1e-9 * e0          # trapezoidal: conserved
    assert np.isclose(e_tr[-1], e0, rtol=1e-9)
    # HHT: dissipation visible, and never energy CREATION. (The strictly
    # monotone HHT quantity is the ALGORITHMIC energy including velocity-
    # displacement cross terms — the physical KE+IE may wiggle by a
    # fraction of a percent step to step while decaying.)
    assert e_ht[-1] < 0.92 * e0
    assert (e_ht <= e0 * (1.0 + 1e-9)).all()
    assert (np.diff(e_ht) <= 2e-3 * e0).all()
    # ...but the resolved fundamental survives: the bar still oscillates
    _, u = _tip_u(m_ht, node_id=nel + 1)
    assert np.abs(u[-len(u) // 4:]).max() > 0.2 * np.abs(u).max()


# ----------------------------------------------------------------------------
# Quasi-static limit: dynamics reproduces the M8 implicit-static answer
# ----------------------------------------------------------------------------

def _cantilever_deck(F):
    """The M8 shell cantilever (20x2 BT4 strip), transverse tip load F."""
    Lx, b, t = 100.0, 10.0, 1.0
    nx, ny = 20, 2

    def nid(i, j):
        return 1 + i + j * (nx + 1)
    nodes = [f"{nid(i,j):10d}{i*Lx/nx:20.10f}{j*b/ny:20.10f}{0.0:20.10f}"
             for j in range(ny + 1) for i in range(nx + 1)]
    shells = []
    eid = 0
    for j in range(ny):
        for i in range(nx):
            eid += 1
            c = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            shells.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in c))
    fixed = [nid(0, j) for j in range(ny + 1)]
    tip = [nid(nx, j) for j in range(ny + 1)]
    starter = f"""\
#RADIOSS STARTER
/BEGIN
CANT
/NODE
{chr(10).join(nodes)}
/SHELL/1
{chr(10).join(shells)}
/PART/1
plate
         1         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.3
/PROP/SHELL/1
shell
       1.0         5      0.01
/GRNOD/NODE/1
fixed
{" ".join(map(str, fixed))}
/GRNOD/NODE/2
tip
{" ".join(map(str, tip))}
/BCS/1
clamp
       111       111         0         1
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
tip
         1         Z         2       {F / len(tip)}
/END
"""
    return starter, tip


def test_quasi_static_limit_matches_static(make_deck):
    """The shell cantilever loaded by a ramp much slower than its first
    period, with a touch of HHT damping to settle the transient: the final
    tip deflection reproduces the M8 implicit-STATIC answer (and hence
    Euler-Bernoulli) — dynamics degenerates to statics when inertia
    becomes irrelevant."""
    F = 0.001
    starter, tip = _cantilever_deck(F)

    # the static reference (the M8 driver on the same deck)
    ms = _run(make_deck, "QSS", starter, "#\n/RUN/QSS/1\n1.0\n/IMPL\n/END\n")
    tipidx = [ms.node_index(nn) for nn in tip]
    w_static = (ms.x - ms.x0)[tipidx, 2].mean()

    # dynamics: /FUNCT ramps over 0..100 (>> T1 ~ 12 ms), then holds; HHT
    # alpha = -0.3 settles the residual ringing over the long hold
    starter_d = starter.replace(
        """ramp
       0.0       0.0
     100.0     100.0""",
        """ramp
       0.0       0.0
     100.0       1.0
    1000.0       1.0""")
    md = _run(make_deck, "QSD", starter_d,
              _dyna_engine("QSD", 400.0, 2.0, card="/IMPL/DYNA/1\n-0.3\n"))
    assert md.implicit_result.converged
    w_dyn = (md.x - md.x0)[tipidx, 2].mean()
    assert w_dyn == pytest.approx(w_static, rel=0.01)
    # the settled mean is tighter than any single sample of the ring-down
    h = md.implicit_result.history
    w_hist = np.array(h["u"])[:, tipidx, 2].mean(axis=1)
    w_mean = w_hist[-len(w_hist) // 4:].mean()
    assert w_mean == pytest.approx(w_static, rel=2e-3)


# ----------------------------------------------------------------------------
# Step-force transient: closed form + explicit cross-check on the same deck
# ----------------------------------------------------------------------------

def test_step_response_vs_closed_form(make_deck):
    """A step axial force on the SDOF truss: the implicit transient matches
    u(t) = (F/k)(1 - cos omega t) POINTWISE and lands on the dynamic-factor-
    2 peak (the sanity closed form every time integrator must pass)."""
    F = 0.05
    t_end = np.pi / OM_SD                     # first peak of the response
    dt = T_SD / 200.0
    starter = _sdof_deck(force=F)

    mi = _run(make_deck, "STPI", starter, _dyna_engine("STPI", t_end, dt))
    assert mi.implicit_result.converged
    t, u = _tip_u(mi)
    u_exact = (F / K_SD) * (1.0 - np.cos(OM_SD * t))
    assert np.abs(u - u_exact).max() < 2e-3 * (2.0 * F / K_SD)
    assert u[-1] == pytest.approx(2.0 * F / K_SD, rel=1e-3)


def test_transient_matches_explicit(make_deck):
    """Cross-solver consistency on a real transient: the shell cantilever
    under a suddenly applied tip load, implicit (trapezoidal, dt = T1/240)
    vs the EXPLICIT leapfrog on the SAME deck. The comparison samples the
    first response peak — where the velocity vanishes, so the explicit
    run's end-time granularity and the O(dt^2) phase errors cannot leak
    into the compared value. (The peak also sits at the dynamic-factor-2
    level of the dominant mode, checked as a sanity bound.)"""
    F = 0.001
    starter, tip = _cantilever_deck(F)
    starter = starter.replace(
        """ramp
       0.0       0.0
     100.0     100.0""",
        """step
       0.0       1.0
    1000.0       1.0""")

    # implicit transient through the first peak (T1 ~ 12 ms)
    mi = _run(make_deck, "XCI", starter, _dyna_engine("XCI", 8.0, 0.05))
    assert mi.implicit_result.converged
    h = mi.implicit_result.history
    tipidx = [mi.node_index(nn) for nn in tip]
    t = np.array(h["t"])
    w = np.array(h["u"])[:, tipidx, 2].mean(axis=1)
    j = int(np.argmax(np.abs(w)))
    assert 0 < j < len(w) - 1, "peak must be interior to the window"
    t_star, w_peak = t[j], w[j]

    # the explicit leapfrog on the very same starter deck, run to t*
    s, e = make_deck("XCE", starter,
                     f"#\n/RUN/XCE/1\n{t_star}\n/DT\n0.9  0.0\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        me = run_engine(e)
    w_exp = (me.x - me.x0)[tipidx, 2].mean()
    assert w_peak == pytest.approx(w_exp, rel=5e-3)

    # sanity: the peak is the dynamic-amplification-2 level of the beam
    I = 10.0 * 1.0 ** 3 / 12.0
    w_st = F * 100.0 ** 3 / (3.0 * 210.0 * I)
    assert w_peak == pytest.approx(2.0 * w_st, rel=0.05)


# ----------------------------------------------------------------------------
# /IMPL/NONLIN dynamics: the large-rotation pendulum vs the elliptic period
# ----------------------------------------------------------------------------

def test_nlgeom_pendulum_elliptic_period(make_deck):
    """A stiff truss pendulum released from 60 degrees under gravity
    (/IMPL/DYNA + /IMPL/NONLIN): the tip swings through the vertical at
    EXACTLY a quarter of the large-amplitude period
    T = 4 sqrt(L/g) K(sin^2(theta0/2)) (the elliptic-integral pendulum —
    7.3% longer than the linear 2 pi sqrt(L/g) at this amplitude, so
    small-rotation theory CANNOT pass this), the bar length is preserved
    through the large rotation, the energy balance stays closed, and
    Newton keeps its quadratic convergence throughout."""
    from scipy.special import ellipk
    L, G = 100.0, 9.81e-3
    E, AREA, RHO = 210.0, 1.0, 7.8e-6
    th0 = np.pi / 3.0
    om0 = np.sqrt(G / L)
    T_lin = 2.0 * np.pi / om0
    T_nl = 4.0 / om0 * ellipk(np.sin(th0 / 2.0) ** 2)
    assert T_nl / T_lin > 1.07                    # genuinely large amplitude

    x2, y2 = L * np.sin(th0), -L * np.cos(th0)
    starter = f"""\
#RADIOSS STARTER
/BEGIN
PEND
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{x2:20.10f}{y2:20.10f}{0.0:20.10f}
/TRUSS/1
         1         1         2
/PART/1
bar
         1         1
/MAT/LAW1/1
steel
   {RHO}
     {E}       0.0
/PROP/TRUSS/1
bar
       {AREA}
/GRNOD/NODE/1
pivot
1
/GRNOD/NODE/2
tip
2
/BCS/1
pin
       111       000         0         1
/BCS/2
in plane
       001       000         0         2
/FUNCT/1
const gravity
       0.0       1.0
   10000.0       1.0
/GRAV/1
g down
         1        Y         0      {-G}
/END
"""
    dt = T_lin / 400.0
    model = _run(make_deck, "PEND", starter,
                 _dyna_engine("PEND", 0.35 * T_nl, dt,
                              extra="/IMPL/NONLIN\n/IMPL/NEWTON\n1e-8  25\n"))
    res = model.implicit_result
    assert res.converged, res.stop_reason

    h = res.history
    t = np.array(h["t"])
    idx = model.node_index(2)
    xt = model.x0[idx] + np.array([uu[idx] for uu in h["u"]])
    theta = np.arctan2(xt[:, 0], -xt[:, 1])
    # the history starts AFTER the first step, so theta[0] = th0 - O(dt^2)
    assert theta[0] == pytest.approx(th0, rel=1e-3)

    # quarter period = first crossing of the vertical (theta = 0)
    j = np.where(np.sign(theta[:-1]) != np.sign(theta[1:]))[0]
    assert len(j), "pendulum never crossed the vertical"
    j = j[0]
    tq = t[j] + (t[j + 1] - t[j]) * (-theta[j]) / (theta[j + 1] - theta[j])
    assert tq == pytest.approx(T_nl / 4.0, rel=0.005)
    # ...and clearly NOT the small-angle quarter period
    assert abs(tq - T_lin / 4.0) > 10.0 * abs(tq - T_nl / 4.0)

    # the bar is inextensible on the gravity scale: radius preserved
    r = np.linalg.norm(xt[:, :2], axis=1)
    assert np.abs(r - L).max() < 1e-4 * L

    # energy: |IE + KE - Wext| stays small vs the potential-energy scale
    m_tip = RHO * AREA * L / 2.0
    scale = m_tip * G * L
    assert np.abs(h["bal"]).max() < 1e-3 * scale

    # quadratic Newton: the worst step still collapses its residual
    worst = max(res.increments, key=lambda i2: i2.iterations)
    assert worst.iterations <= 5
    r_tail = worst.residuals
    assert r_tail[-1] < 1e-8 * max(r_tail[0], 1e-30)


# ----------------------------------------------------------------------------
# /IMPL/DYNA card parsing mirrors the original reader (freimpl.F)
# ----------------------------------------------------------------------------

def _parse(tmp_path, text):
    from pyradioss.common.messages import MessageLog
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck
    p = tmp_path / "E_0001.rad"
    p.write_text(text)
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        ec = parse_engine_deck(read_deck(str(p)), log)
    return ec, log


def test_impl_dyna_card_parsing(tmp_path):
    """/IMPL/DYNA/1 reads the HHT alpha ITSELF (freimpl.F HHT_A — not a
    spectral radius); /IMPL/DYNA/2 reads gamma then beta (DY_G = NM_A,
    DY_B = NM_B in imp_dyna.F); bare /IMPL stays STATIC; /IMPL/DYNA/DAMP
    reads a then b and IMPLIES dynamics (M11 removed the M10 deferral —
    freimpl.F: IF (IDYNA==0) IDYNA=1)."""
    ec, _ = _parse(tmp_path, "#\n/RUN/A/1\n1.0\n/IMPL/DYNA/1\n-0.05\n/END\n")
    assert ec.implicit and ec.impl_dyna == 1
    assert ec.impl_dyna_alpha == pytest.approx(-0.05)

    ec, _ = _parse(tmp_path, "#\n/RUN/A/1\n1.0\n/IMPL/DYNA/2\n0.6  0.3025\n"
                             "/END\n")
    assert ec.impl_dyna == 2
    assert ec.impl_dyna_gamma == pytest.approx(0.6)
    assert ec.impl_dyna_beta == pytest.approx(0.3025)

    # bare /IMPL/DYNA = trapezoidal defaults
    ec, _ = _parse(tmp_path, "#\n/RUN/A/1\n1.0\n/IMPL/DYNA\n/END\n")
    assert ec.impl_dyna == 2
    assert ec.impl_dyna_gamma == 0.5 and ec.impl_dyna_beta == 0.25

    # a bare /IMPL stays STATIC (the M8/M9 default is untouched)
    ec, _ = _parse(tmp_path, "#\n/RUN/A/1\n1.0\n/IMPL\n/END\n")
    assert ec.implicit and ec.impl_dyna == 0

    # /IMPL/DYNA/DAMP (M11): DAMPA_IMP then DAMPB_IMP, and the card alone
    # switches the run dynamic exactly like the original reader
    ec, log = _parse(tmp_path, "#\n/RUN/A/1\n1.0\n/IMPL/DYNA/DAMP\n"
                               "0.1  0.01\n/END\n")
    assert ec.impl_dyna_damp
    assert ec.impl_dyna == 1 and ec.impl_dyna_alpha == 0.0
    assert ec.impl_dyna_dampa == pytest.approx(0.1)
    assert ec.impl_dyna_dampb == pytest.approx(0.01)


def test_impl_dyna_stability_warnings(tmp_path):
    """Out-of-range HHT alpha and conditionally-stable Newmark pairs warn
    (2 beta >= gamma >= 1/2 is the unconditional-stability region)."""
    _, log = _parse(tmp_path, "#\n/RUN/A/1\n1.0\n/IMPL/DYNA/1\n-0.6\n/END\n")
    _, log2 = _parse(tmp_path, "#\n/RUN/A/1\n1.0\n/IMPL/DYNA/2\n"
                               "0.5  0.1\n/END\n")
    assert any("alpha" in w for w in log.warnings)
    assert any("stable" in w for w in log2.warnings)


# ----------------------------------------------------------------------------
# Rate devices: explicitly deferred, never du/1
# ----------------------------------------------------------------------------

def test_law2_rate_term_deferred(make_deck):
    """A LAW2 material WITH a strain-rate coefficient under /IMPL/DYNA: the
    driver zeroes c with a warning (rate-dependent plasticity under
    implicit dynamics is DEFERRED — the pseudo-velocity kernel drive would
    otherwise feed the rate term du/1), and the solid bulk viscosity is
    zeroed exactly like statics."""
    L = 10.0
    starter = f"""\
#RADIOSS STARTER
/BEGIN
RATE
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{L:20.10f}{0.0:20.10f}{0.0:20.10f}
         3{L:20.10f}{L:20.10f}{0.0:20.10f}
         4{0.0:20.10f}{L:20.10f}{0.0:20.10f}
         5{0.0:20.10f}{0.0:20.10f}{L:20.10f}
         6{L:20.10f}{0.0:20.10f}{L:20.10f}
         7{L:20.10f}{L:20.10f}{L:20.10f}
         8{0.0:20.10f}{L:20.10f}{L:20.10f}
/BRICK/1
         1         1         2         3         4         5         6         7         8
/PART/1
cube
         1         1
/MAT/LAW2/1
steel jc with rate term
   7.8e-6
     210.0       0.3
       0.4       0.5       0.5
      0.05       1.0
/PROP/SOLID/1
solid
       1.1      0.05       0.1
/GRNOD/NODE/1
base
1 2 3 4
/BCS/1
clamp base
       111       000         0         1
/FUNCT/1
pull
       0.0       0.0
     100.0     100.0
/GRNOD/NODE/2
top
5 6 7 8
/CLOAD/1
pull top
         1         Z         2       0.01
/END
"""
    model = _run(make_deck, "RATE", starter,
                 _dyna_engine("RATE", 1.0, 0.1))
    assert model.implicit_result.converged
    for mid, mat in model.materials.items():
        if mat.law == 2:
            assert mat.params["c"] == 0.0        # deferred: zeroed + warned
    for sl, mat, prop in model.bricks.state["slices"]:
        assert prop.params["qa"] == 0.0
        assert prop.params["qb"] == 0.0


def test_impvel_refused_under_dynamics(make_deck):
    """/IMPVEL under /IMPL/DYNA is a REAL dynamic boundary condition the
    port does not integrate — it must refuse loudly (use /IMPDISP), never
    silently ignore it."""
    starter = _sdof_deck().replace(
        "/END\n",
        """/FUNCT/9
vel
       0.0       0.1
     100.0       0.1
/IMPVEL/1
drive
         9         X         2       1.0
/END
""")
    s, e = make_deck("IVREF", starter, _dyna_engine("IVREF", 1.0, 0.1))
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(NotImplementedError, match="IMPVEL"):
            run_engine(e)


def test_arclength_dyna_contradiction(make_deck):
    """/IMPL/ARCL + /IMPL/DYNA is a contradiction (a static continuation
    vs a time march): refused with a clear error."""
    starter = _sdof_deck(force=0.05)
    s, e = make_deck("ARCD", starter,
                     "#\n/RUN/ARCD/1\n1.0\n/IMPL/DYNA/2\n0.5 0.25\n"
                     "/IMPL/ARCL\n/IMPL/DTINI\n0.1\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(ValueError, match="ARCL"):
            run_engine(e)


# ----------------------------------------------------------------------------
# /IMPDISP under dynamics: the displacement drive at physical time
# ----------------------------------------------------------------------------

def test_impdisp_dynamic_drive(make_deck):
    """A slow /IMPDISP ramp on the SDOF truss under dynamics lands the tip
    exactly on the prescribed displacement at every recorded time (the
    drive is condensed and seeded exactly), and the reaction work booked
    in the ledger matches the stored strain energy in the quasi-static
    regime."""
    d_end = 0.01
    t_end = 50.0 * T_SD                     # ~static ramp
    starter = _sdof_deck().replace(
        "/END\n",
        f"""/FUNCT/9
ramp
       0.0       0.0
     {t_end}       {d_end}
/IMPDISP/1
pull tip
         9         X         2       1.0
/END
""")
    model = _run(make_deck, "IDD", starter,
                 _dyna_engine("IDD", t_end, t_end / 100.0,
                              card="/IMPL/DYNA/1\n-0.05\n"))
    res = model.implicit_result
    assert res.converged
    t, u = _tip_u(model)
    assert np.allclose(u, d_end * t / t_end, atol=1e-9 * d_end)
    # quasi-static: external (constraint) work ~ stored strain energy
    h = res.history
    w = h["wext"][-1]
    ie = h["ie"][-1]
    assert w == pytest.approx(0.5 * K_SD * d_end ** 2, rel=0.02)
    assert ie == pytest.approx(w, rel=0.02)
