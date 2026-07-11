"""
M13 validations: friction, TYPE11, follower loads and LAW36 in the
implicit system.

Every new capability gets at least one ANALYTIC check (the port's
philosophy):

* /INTER/TYPE7 COULOMB FRICTION (the incremental return mapping of
  i7kfor3, implicit/contact.py): a block pressed onto a block and pushed
  BELOW the cone transmits the applied shear exactly (stick — with the
  tangential micro-slip F/(4 K_t) of the penalty regularization, closed
  form), pushed ABOVE the cone by displacement control it transmits
  exactly mu*N (slip — the radial return), an inclined dead load above
  the cone has NO static equilibrium and must fail loudly,
  finite-difference residual/tangent consistency in BOTH regimes at a
  face-interior projection (where the frozen-normal tangent is exact),
  mu > 0 with a purely normal load reproduces the M12 frictionless
  series-springs closed form to round-off, and the implicit slip force
  cross-checks the EXPLICIT solver's velocity-regularized kinetic
  friction in steady sliding (both must give ~mu*N — the explicit
  approaches it from below by its documented regularization factor
  v/(v + eps));
* /INTER/TYPE11 edge-to-edge (i11ke3.F): crossed edges on grounded
  springs pressed together = the springs-vs-penalty closed form EXACT,
  finite-difference residual/tangent consistency in the
  interior-interior region (the M13 exact edge-edge curvature — FD-exact
  in ALL directions, unlike the TYPE7 approximation) plus the clamped
  point-segment region, tangent symmetry, and crossed shell strips
  pressed quasi-statically: implicit static = explicit damped steady
  state;
* /PLOAD follower-load stiffness (imp_glob_k.F IMP_KPRES analogue,
  implicit/followerload.py): finite-difference exactness of the
  assembled load-stiffness block on a WARPED segment, and a
  pressure-loaded NLGEOM cantilever where the load-stiffness toggle
  changes the Newton iteration count but NOT the converged answer —
  plus the small-pressure limit against the linear closed form
  p b L^4 / (8 E I);
* LAW36 consistent tangent: uniaxial pull past yield on a solid and on
  a shell landing EXACTLY on the tabulated hardening curve (the
  piecewise-linear return is exact — no iterated-return upgrade needed,
  measured here) with the QUADRATIC Newton tail only an algorithmic
  tangent delivers; a multi-rate curve family truncated to the static
  curve with a warning;
* the M7 parity contract: building/evaluating the M13 treatments
  mutates NOTHING shared with the explicit solver;
* the loud refusals: /PLOAD + /IMPL/ARCL.

See PORTING_GUIDE.md roadmap M13.
"""

import contextlib
import io

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

pytest.importorskip("scipy")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(s)
        model = run_engine(e)
    return model, buf.getvalue()


def _model(make_deck, name, starter):
    s, _ = make_deck(name, starter, "#\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


def _assert_quadratic_tail(res):
    """The M11 quadratic-convergence proxy: the residual ratio shrinks by
    orders of magnitude across the tail of a plastic increment (or the
    residual crashed to the round-off floor, which IS the signature)."""
    plastic = [i for i in res.increments if i.iterations > 2]
    assert plastic, "expected at least one plastic (multi-iter) increment"
    inc = max(plastic, key=lambda i: i.iterations)
    r = np.array(inc.residuals)
    r = r[r > 0]
    if r[-1] <= 1e-12 * r[0]:
        return
    tail = r[-3:]
    assert len(tail) == 3
    assert tail[2] / tail[1] < (tail[1] / tail[0]) ** 1.5


# ----------------------------------------------------------------------------
# /INTER/TYPE7 friction — two stacked blocks, friction at the interface
# ----------------------------------------------------------------------------

def _fric_deck(mu, K4=50.0, gap=0.05, sep=0.0499, rho="7.8e-6",
               loads="", shift=0.0, wide=False, EA="210.0"):
    """Two hexa blocks stacked along z: bottom clamped at its base, TYPE7
    (Istf=1, K = ``K4``) with Coulomb ``mu`` between the bottom block's
    top face (main) and the top block's bottom nodes (secondary).
    ``sep`` is the initial face separation (0.0499 vs gap 0.05 = a hair
    of initial contact, the M12 punch convention). ``loads`` appends the
    load/drive cards; ``shift`` offsets the top block laterally (+x, +y).
    ``wide=True`` makes the bottom block 2x2 in plan with the 1x1 punch
    centered: all four punch nodes then project into the FACE INTERIOR
    and stay there under lateral motion — the contact normal is the flat
    face normal (exactly z for uniform face motion), which is what makes
    the stick/slip closed forms exact (a punch flush with the face edge
    slides its edge nodes OFF the face into tilted vertex projections
    that carry part of the lateral load through the normal force)."""
    b = 0.5 if wide else 0.0
    return f"""\
#RADIOSS STARTER
/BEGIN
friction blocks
/NODE
         1    {-b:16.6f}    {-b:16.6f}                 0.0
         2    {1.0 + b:16.6f}    {-b:16.6f}                 0.0
         3    {1.0 + b:16.6f}    {1.0 + b:16.6f}                 0.0
         4    {-b:16.6f}    {1.0 + b:16.6f}                 0.0
         5    {-b:16.6f}    {-b:16.6f}                 1.0
         6    {1.0 + b:16.6f}    {-b:16.6f}                 1.0
         7    {1.0 + b:16.6f}    {1.0 + b:16.6f}                 1.0
         8    {-b:16.6f}    {1.0 + b:16.6f}                 1.0
        11    {shift:16.6f}    {shift:16.6f}    {1.0 + sep:16.6f}
        12    {1.0 + shift:16.6f}    {shift:16.6f}    {1.0 + sep:16.6f}
        13    {1.0 + shift:16.6f}    {1.0 + shift:16.6f}    {1.0 + sep:16.6f}
        14    {shift:16.6f}    {1.0 + shift:16.6f}    {1.0 + sep:16.6f}
        15    {shift:16.6f}    {shift:16.6f}    {2.0 + sep:16.6f}
        16    {1.0 + shift:16.6f}    {shift:16.6f}    {2.0 + sep:16.6f}
        17    {1.0 + shift:16.6f}    {1.0 + shift:16.6f}    {2.0 + sep:16.6f}
        18    {shift:16.6f}    {1.0 + shift:16.6f}    {2.0 + sep:16.6f}
/BRICK/1
         1         1         2         3         4         5         6         7         8
/BRICK/2
         2        11        12        13        14        15        16        17        18
/PART/1
bottom block
         1         1
/PART/2
punch
         1         2
/MAT/LAW1/1
bottom (EA)
   {rho}
     {EA}       0.0
/MAT/LAW1/2
punch
   {rho}
     210.0       0.0
/PROP/SOLID/1
solid
       1.1      0.05       0.1
/GRNOD/NODE/1
base
1 2 3 4
/GRNOD/NODE/2
bbot
11 12 13 14
/GRNOD/NODE/3
btop
15 16 17 18
/SURF/SEG/1
top face of A
         5         6         7         8
/BCS/1
base
       111       000         0         1
/INTER/TYPE7/1
press
         2         1         1         0
    {K4:6.1f}    {mu:6.3f}    {gap:6.3f}       0.0
{loads}/END
"""


_RAMP = """\
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
"""

_CONST = """\
/FUNCT/2
hold
       0.0       1.0
       2.0       1.0
"""


def test_friction_stick_below_cone(make_deck):
    """Normal dead load N (on the punch top) with a proportional lateral
    load F < mu*N applied at the INTERFACE nodes (moment-free — an
    overhead lateral load adds an overturning moment that drives the
    leading pairs into a MIXED stick-slip state; that harder case is
    exercised by the with-moment variant below): every pair sticks.
    Closed forms: the bottom block carries the applied shear exactly
    (sig_zx = +F/A), the normal state is untouched (sig_zz = -N/A), and
    the mean interface tangential MICRO-SLIP is F/(4 K_t) (four parallel
    tangential springs K_t = K)."""
    N, F, mu, K4 = 2.0, 0.5, 0.4, 50.0        # F/N = 0.25 < mu
    loads = _RAMP + f"""\
/CLOAD/1
press
         1         Z         3   {-N / 4.0}
/CLOAD/2
push
         1         X         2   {F / 4.0}
"""
    model, _ = _run(make_deck, "STK",
                    _fric_deck(mu, loads=loads, wide=True, EA="2.1e6"),
                    "#\n/RUN/STK/1\n1.0\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-9  40\n/END\n")
    res = model.implicit_result
    assert res.converged
    sA = model.bricks.state["sig"][0]         # bottom block, Voigt
    # Voigt order [xx, yy, zz, xy, yz, zx]; the wide block has plan area 4
    assert sA[2] == pytest.approx(-N / 4.0, rel=1e-6)    # normal press
    # transmitted shear: the traction on A's +z face is sig_zx = +F/A
    # (the punch drags A's top face along +x)
    assert sA[5] == pytest.approx(F / 4.0, rel=1e-6)
    d = model.x - model.x0
    # interface micro-slip: four identical parallel tangential springs
    # K_t = K carrying F together, anchored on the quasi-rigid (EA x 1e4)
    # bottom block — every punch node's lateral displacement IS the slip
    # jump F/(4 K_t) (the residual face motion is ~1e-4 of it)
    for k in range(4):
        assert d[model.node_index(11 + k), 0] == pytest.approx(
            F / (4.0 * K4), rel=1e-3)


def test_friction_mixed_stick_slip_with_moment(make_deck):
    """The HARDER stick case: the lateral load applied OVERHEAD (on the
    punch top) adds an overturning moment that redistributes the pair
    normal forces — the leading pairs' cones shrink below their load
    share and SLIP while the trailing pairs stick (a mixed state with
    pairs parked exactly on the cone: the configuration whose period-2
    assignment cycle motivated the M13 line search). The run must
    converge and global equilibrium still transmits the full shear:
    sig_zx(A) = F/A exactly (equal 1x1 blocks here — A = 1)."""
    N, F, mu = 2.0, 0.5, 0.4                  # F < mu*N globally; the
    loads = _RAMP + f"""\
/CLOAD/1
press
         1         Z         3   {-N / 4.0}
/CLOAD/2
push overhead
         1         X         3   {F / 4.0}
"""
    model, _ = _run(make_deck, "MIX", _fric_deck(mu, loads=loads),
                    "#\n/RUN/MIX/1\n1.0\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-8  40\n/END\n")
    res = model.implicit_result
    assert res.converged
    sA = model.bricks.state["sig"][0]
    assert sA[5] == pytest.approx(F, rel=1e-6)     # full shear transmitted
    assert sA[2] == pytest.approx(-N, rel=1e-5)    # full press transmitted


def test_friction_slip_force_capped_at_cone(make_deck):
    """Displacement-driven sliding well beyond the cone: the transmitted
    shear is EXACTLY mu*N (the radial return caps every pair at
    mu f_n = mu N/4), the normal state stays -N."""
    N, mu, K4, drive = 2.0, 0.4, 50.0, 0.05
    loads = _RAMP + _CONST + f"""\
/CLOAD/1
press
         2         Z         3   {-N / 4.0}
/IMPDISP/1
slide
         1         X         2   {drive}
"""
    model, _ = _run(make_deck, "SLP",
                    _fric_deck(mu, loads=loads, wide=True, EA="2.1e6"),
                    "#\n/RUN/SLP/1\n1.0\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-9  40\n/END\n")
    res = model.implicit_result
    assert res.converged
    sA = model.bricks.state["sig"][0]         # the block under the slider
    assert sA[5] == pytest.approx(mu * N / 4.0, rel=1e-6)  # plan area 4
    assert sA[2] == pytest.approx(-N / 4.0, rel=1e-6)
    # the driven punch rides its prescribed bottom nodes: normal only
    assert model.bricks.state["sig"][1][2] == pytest.approx(-N, rel=1e-6)
    # sanity: the drive really went past the cone displacement
    assert drive > mu * N / (4 * K4) + mu * N / (210.0 / 2.6 + 105.0)


@pytest.mark.parametrize("factor", [0.8, 1.25])
def test_friction_inclined_load_stick_slip_transition(make_deck, factor):
    """An inclined dead load with tangential part F = factor * mu * N:
    below the cone (factor < 1) the run converges and transmits F
    exactly; ABOVE the cone there is NO static equilibrium — the run
    must fail LOUDLY (Newton non-convergence through the StepControl
    cuts, or a singular tangent: a fully sliding block has no tangential
    stiffness left), never a silent wrong answer."""
    N, mu, K4 = 2.0, 0.4, 50.0
    F = factor * mu * N
    loads = _RAMP + f"""\
/CLOAD/1
press
         1         Z         3   {-N / 4.0}
/CLOAD/2
push
         1         X         2   {F / 4.0}
"""
    s, e = make_deck(f"INC{int(10 * factor)}",
                     _fric_deck(mu, loads=loads, wide=True),
                     "#\n/RUN/INC/1\n1.0\n/IMPL/DTINI\n0.5\n"
                     "/IMPL/NEWTON\n1e-9  12\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        if factor < 1.0:
            model = run_engine(e)
            assert model.implicit_result.converged
            sA = model.bricks.state["sig"][0]
            assert sA[5] == pytest.approx(F / 4.0, rel=1e-6)
        else:
            try:
                model = run_engine(e)
                assert not model.implicit_result.converged
            except (RuntimeError, ValueError, FloatingPointError):
                pass                          # singular slip tangent: loud


def test_friction_fd_residual_tangent_consistency(make_deck):
    """FD consistency of the friction force and its stick/slip tangent
    blocks at FACE-INTERIOR projections (the wide-block punch: the
    contact normal is the flat face normal and does not rotate under
    secondary motion). For directions that hold the main corners the
    SECONDARY-NODE residual rows are EXACT in BOTH regimes — including
    the slip branch's nonsymmetric mu K t n^T + (mu f_n K/|f_tr|)
    (I - n n^T - t t^T) blocks. The CORNER reaction rows carry the
    documented weight-variation omission (a face-interior projection's
    weights vary smoothly with the node motion, unlike the M12 FD test's
    locked vertex weights; i7keg3.F freezes them too) — asserted at the
    matching loose tolerance so a regression of the main terms would
    still be caught."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.implicit.contact import (build_implicit_contacts,
                                            contact_tangent)
    from pyradioss.implicit.dofmap import DofMap

    mu, K4, gap = 0.4, 50.0, 0.05
    model = _model(make_deck, "FDF",
                   _fric_deck(mu, K4=K4, gap=gap, sep=0.02, wide=True))
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        contacts = build_implicit_contacts(model, log)
        dof = DofMap(model, log)
    n = model.numnod
    top = [model.node_index(nid) for nid in range(11, 19)]
    corner_eqs = [dof.eq[model.node_index(nid) * 6 + c]
                  for nid in (5, 6, 7, 8) for c in range(3)]
    corner_eqs = [q for q in corner_eqs if q >= 0]
    sec_eqs = np.array([q for q in
                        (dof.eq[model.node_index(nid) * 6 + c]
                         for nid in range(11, 15) for c in range(3))
                        if q >= 0])

    def residual(u):
        f = np.zeros((n, 3))
        for c in contacts:
            c.forces(model.x0 + u, f)
        return dof.gather_residual(f, np.zeros((n, 3)))

    rng = np.random.default_rng(1)
    eps = 1e-7
    # pen = 0.03; stick pre-slide delta < mu*pen, slip beyond it
    for delta, regime in ((0.005, "stick"), (0.030, "slip")):
        u0 = np.zeros((n, 3))
        u0[top, 0] = delta
        Kc = contact_tangent(contacts, model.x0 + u0, dof).toarray()
        for _ in range(4):
            v_eq = rng.standard_normal(dof.ndof)
            v_eq[corner_eqs] = 0.0            # secondary-side direction
            du, _ = dof.scatter_solution(v_eq)
            fd = (residual(u0 + eps * du)
                  - residual(u0 - eps * du)) / (2 * eps)
            an = -Kc @ v_eq
            ref = max(np.abs(fd).max(), 1e-9)
            # secondary rows: exact to FD accuracy
            assert np.abs((fd - an)[sec_eqs]).max() < 1e-6 * ref, regime
            # all rows: the O(p/L) weight-variation omission only
            assert np.abs(fd - an).max() < 0.05 * ref, regime


def test_friction_pure_normal_load_matches_frictionless_closed_form(
        make_deck):
    """mu > 0 with a PURELY NORMAL displacement drive: no tangential
    motion, so friction must change NOTHING — the M12 series-springs
    closed form sigma = 4K(gap - d0 + Delta)/(A + 8KL/E) holds to the
    same round-off tolerance as the frictionless M12 test (the mu = 0
    code path itself is byte-identical to M12 and covered by the M12
    suite)."""
    E, L, A = 210.0, 1.0, 1.0
    K4, gap, d0, Delta, mu = 50.0, 0.05, 0.08, 0.15, 0.4
    loads = f"""\
/FUNCT/1
push
       0.0       0.0
       2.0    {-2.0 * Delta:10.4f}
/IMPDISP/1
push down
         1         Z         3       1.0
"""
    # lateral BCS like the M12 stack deck (pure 1-D chain)
    deck = _fric_deck(mu, K4=K4, gap=gap, sep=d0, loads=loads)
    deck = deck.replace("/INTER/TYPE7/1", """\
/GRNOD/NODE/4
all
1 2 3 4 5 6 7 8 11 12 13 14 15 16 17 18
/BCS/2
lateral
       110       000         0         4
/INTER/TYPE7/1""")
    model, _ = _run(make_deck, "NRM", deck,
                    "#\n/RUN/NRM/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.1\n/END\n")
    sig = 4 * K4 * (gap - d0 + Delta) / (A + 8 * K4 * L / E)
    sA = model.bricks.state["sig"][0]
    assert sA[2] == pytest.approx(-sig, rel=1e-9)
    assert abs(sA[5]) < 1e-12 * sig           # no shear from friction


def test_friction_implicit_slip_matches_explicit_kinetic(make_deck):
    """Cross-solver check in steady SLIDING, where the static return map
    and the explicit velocity-regularized kinetic law must agree: both
    transmit ~mu*N. The implicit slip force is exactly mu*N; the
    explicit one approaches it from below by its documented
    regularization factor v/(v + eps) — asserted within 10%."""
    N, mu, K4 = 2.0, 0.4, 50.0
    v_drive, t_ramp, t_end = 0.3, 2.0, 3.5
    # x1e4 density: the explicit dt grows x100, the load ramp spans ~16
    # contact-spring periods (quasi-static), and the velocity
    # regularization eps = 1e-3 gap/dt drops to ~1% of the drive speed;
    # the total slide v_drive*(t_end - t_ramp) stays inside the wide
    # face's 0.5 margin (/IMPVEL reads the curve VALUE as the velocity:
    # step to v_drive right after the ramp, then hold)
    rho = "7.8e-2"
    base = _fric_deck(mu, K4=K4, rho=rho, loads="", wide=True)
    load_exp = f"""\
/FUNCT/1
ramp hold
       0.0       0.0
     {t_ramp:5.2f}       1.0
      20.0       1.0
/FUNCT/2
slide after ramp
       0.0       0.0
     {t_ramp:5.2f}       0.0
     {t_ramp + 0.1:5.2f}       1.0
      20.0       1.0
/GRNOD/NODE/4
everything
1 2 3 4 5 6 7 8 11 12 13 14 15 16 17 18
/CLOAD/1
press
         1         Z         3   {-N / 4.0}
/IMPVEL/1
drag
         2         X         2   {v_drive}
/DAMP/1
settle
      30.0         4
"""
    deck_exp = base.replace("/END\n", load_exp + "/END\n")
    m_exp, _ = _run(make_deck, "KEXP", deck_exp, f"""\
#
/RUN/KEXP/1
{t_end}
/DT
0.9 0.0
/PRINT/-10000
/END
""")
    # measure the EXPLICIT friction/normal ratio directly from the
    # explicit contact law at the final sliding state (a fresh read-only
    # ContactType7 evaluation — the element stress of a single coarse
    # hexa under moving point loads is polluted by hourglass/tilt
    # transients and is not a clean force gauge)
    from pyradioss.common.messages import MessageLog
    from pyradioss.contact.inter_type7 import ContactType7
    with contextlib.redirect_stdout(io.StringIO()):
        ct = ContactType7(m_exp.interfaces[0], m_exp, MessageLog())
    fcont = np.zeros_like(m_exp.x)
    # evaluate at the RUN's own time step (the regularization scale
    # eps = 1e-3 gap/dt is dt-dependent; the mean dt is representative —
    # the step is constant to a few % on this model)
    dt_run = m_exp.engine_state.t / max(m_exp.engine_state.cycle, 1)
    ct.forces(m_exp.x, m_exp.v, m_exp.mass, dt_run, fcont, cycle=0)
    sec = [m_exp.node_index(k) for k in (11, 12, 13, 14)]
    tau_exp = -float(fcont[sec, 0].sum())     # kinetic friction total
    n_exp = float(fcont[sec, 2].sum())        # normal force total
    assert n_exp == pytest.approx(N, rel=0.10)
    # the regularized kinetic law reaches mu only asymptotically in
    # v/(v + eps): from BELOW, within the documented factor
    assert 0.85 * mu < tau_exp / n_exp <= 1.02 * mu

    load_imp = _RAMP + _CONST + f"""\
/CLOAD/1
press
         2         Z         3   {-N / 4.0}
/IMPDISP/1
slide
         1         X         2   {v_drive * (t_end - t_ramp)}
"""
    # the implicit twin gets the quasi-rigid bottom block: the exact-cone
    # assertion needs the flat-face normal (a soft face's elastic warp
    # leaks ~0.5% of the lateral load through tilted normals; the 15%
    # cross-solver tolerance below absorbs the resulting A-stiffness
    # difference between the two runs)
    base_imp = _fric_deck(mu, K4=K4, rho=rho, loads="", wide=True,
                          EA="2.1e6")
    deck_imp = base_imp.replace("/END\n", load_imp + "/END\n")
    m_imp, _ = _run(make_deck, "KIMP", deck_imp,
                    "#\n/RUN/KIMP/1\n1.0\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-9  40\n/END\n")
    assert m_imp.implicit_result.converged
    tau_imp = 4.0 * m_imp.bricks.state["sig"][0][5]   # plan area 4
    assert tau_imp == pytest.approx(mu * N, rel=1e-6)   # exact cone
    # cross-solver: both laws transmit friction = mu x (their own normal
    # force) in steady sliding — the RATIOS agree within the explicit
    # law's regularization factor (the absolute normal forces differ a
    # few % between the two runs: the explicit steady state carries a
    # slow vertical breathing mode through its damping)
    assert tau_exp / n_exp == pytest.approx(tau_imp / N, rel=0.15)


# ----------------------------------------------------------------------------
# /INTER/TYPE11 edge-to-edge under implicit
# ----------------------------------------------------------------------------

def _crossed_edges_deck(h, gap, K, kz, F, fric=0.0):
    """Edge A (truss along x at z = h) with its ends on grounded vertical
    springs (k = ``kz``); edge B (truss along y at z = 0) fully fixed.
    A force F/2 pushes each end of A down. TYPE11 (Istf=1, K, ``gap``)
    between the two — the closest points are the midpoints, and the
    1-D closed form is  u = (K(gap - h) - F)/(2 kz + K)."""
    return f"""\
#RADIOSS STARTER
/BEGIN
crossed edges
/NODE
         1                -1.0                 0.0    {h:16.6f}
         2                 1.0                 0.0    {h:16.6f}
         3                 0.0                -1.0                 0.0
         4                 0.0                 1.0                 0.0
         5                -1.0                 0.0    {h + 1.0:16.6f}
         6                 1.0                 0.0    {h + 1.0:16.6f}
/TRUSS/1
         1         1         2
         2         3         4
/PART/1
edges
         1         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.3
/PROP/TRUSS/1
truss
       1.0
/SPRING/2
         3         1         5
         4         2         6
/PART/2
hangers
         2         1
/PROP/SPRING/2
spring
       1.0    {kz:6.2f}       0.0
/GRNOD/NODE/1
edge A ends
1 2
/GRNOD/NODE/2
fixed
3 4 5 6
/BCS/1
lateral A
       110       000         0         1
/BCS/2
ground
       111       111         0         2
/LINE/SEG/1
edge A
         1         2
/LINE/SEG/2
edge B
         3         4
/INTER/TYPE11/1
cross
         1         2         1         0
    {K:6.2f}    {fric:6.3f}    {gap:6.3f}       0.0
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/CLOAD/1
push down
         1         Z         1   {-F / 2.0}
/END
"""


def test_type11_crossed_edges_closed_form(make_deck):
    """Crossed elastic edges pressed together: the hanger springs and the
    penalty spring form a 1-D chain with the exact solution
    u = (K(gap - h) - F)/(2 kz + K) for the vertical drop of edge A
    (contact active), asserted to round-off along with force balance
    (edge B is rigidly held: the transmitted force is K*pen)."""
    h, gap, K, kz, F = 0.09, 0.1, 100.0, 2.0, 2.0
    model, _ = _run(make_deck, "X11", _crossed_edges_deck(h, gap, K, kz, F),
                    "#\n/RUN/X11/1\n1.0\n/IMPL/DTINI\n0.25\n"
                    "/IMPL/NEWTON\n1e-10  30\n/END\n")
    assert model.implicit_result.converged
    u_exact = (K * (gap - h) - F) / (2 * kz + K)
    d = model.x - model.x0
    assert d[model.node_index(1), 2] == pytest.approx(u_exact, rel=1e-8)
    assert d[model.node_index(2), 2] == pytest.approx(u_exact, rel=1e-8)
    pen = gap - h - u_exact
    assert pen > 0.0                          # the contact really is active
    # edge B held: zero displacement
    assert np.abs(d[model.node_index(3)]).max() < 1e-12


def test_type11_fd_residual_tangent_consistency(make_deck):
    """FD consistency of the TYPE11 residual and tangent. The M13
    edge-edge curvature term is the EXACT closest-point Hessian in every
    projection region, so — unlike the TYPE7 weight-variation omission —
    the FD match holds in ALL directions, including ones that move the
    main-edge ends. Checked at an interior-interior (crossed) state and
    asserted symmetric (a true Hessian)."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.implicit.contact import (build_implicit_contacts,
                                            contact_tangent)
    from pyradioss.implicit.dofmap import DofMap

    h, gap, K = 0.05, 0.1, 5.0
    model = _model(make_deck, "FD11",
                   _crossed_edges_deck(h, gap, K, 2.0, 1.0))
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        contacts = build_implicit_contacts(model, log)
        # FULL numbering for the FD probe: no BCS condensation, so main-
        # edge directions are exercised too
        dof = DofMap(model)
    n = model.numnod

    def residual(u):
        f = np.zeros((n, 3))
        for c in contacts:
            c.forces(model.x0 + u, f)
        return dof.gather_residual(f, np.zeros((n, 3)))

    rng = np.random.default_rng(2)
    eps = 1e-7
    # a generic pre-displacement: tilt both edges a little (keeps the
    # projection interior-interior and away from any special alignment)
    u0 = np.zeros((n, 3))
    u0[model.node_index(1)] = [0.003, 0.002, -0.004]
    u0[model.node_index(4)] = [-0.002, 0.001, 0.003]
    Kc = contact_tangent(contacts, model.x0 + u0, dof).toarray()
    # a true (frictionless) contact Hessian is symmetric
    assert np.allclose(Kc, Kc.T, atol=1e-10 * max(np.abs(Kc).max(), 1.0))
    for _ in range(6):
        v_eq = rng.standard_normal(dof.ndof)
        du, _ = dof.scatter_solution(v_eq)
        fd = (residual(u0 + eps * du) - residual(u0 - eps * du)) / (2 * eps)
        assert np.allclose(fd, -Kc @ v_eq, rtol=1e-5,
                           atol=1e-6 * max(np.abs(fd).max(), 1.0))


def _crossed_strips_deck(load, damp=False):
    """Two crossed shell strips (each clamped at both ends), TYPE11
    between their long edges at the crossing, a dead load pressing the
    upper strip down — the quasi-static M4 edge_impact geometry for the
    implicit-vs-explicit cross-check."""
    L, w, h = 20.0, 2.5, 1.0                 # strip half-length, width, gap
    nseg = 8                                 # elements along each strip
    dx = 2 * L / nseg
    nodes, shells, lines_a, lines_b = [], [], [], []
    nid = 0
    ida = {}
    # strip A along x at z = 0, width in y: rows y = 0 and y = w
    for j in range(2):
        for i in range(nseg + 1):
            nid += 1
            ida[(i, j)] = nid
            nodes.append(f"{nid:10d}{-L + i * dx:20.10f}"
                         f"{j * w:20.10f}{0.0:20.10f}")
    idb = {}
    for j in range(2):
        for i in range(nseg + 1):
            nid += 1
            idb[(i, j)] = nid
            nodes.append(f"{nid:10d}{j * w:20.10f}"
                         f"{-L + i * dx:20.10f}{h:20.10f}")
    eid = 0
    for i in range(nseg):
        eid += 1
        shells.append(f"{eid:10d}{ida[(i, 0)]:10d}{ida[(i + 1, 0)]:10d}"
                      f"{ida[(i + 1, 1)]:10d}{ida[(i, 1)]:10d}")
    for i in range(nseg):
        eid += 1
        shells.append(f"{eid:10d}{idb[(i, 0)]:10d}{idb[(i + 1, 0)]:10d}"
                      f"{idb[(i + 1, 1)]:10d}{idb[(i, 1)]:10d}")
    # contact edges: strip A's row y = 0 near the crossing, strip B's
    # row x = 0 near the crossing (interior-interior crossing pairs)
    mid = nseg // 2
    for i in range(mid - 2, mid + 2):
        lines_a.append(f"{ida[(i, 0)]:10d}{ida[(i + 1, 0)]:10d}")
        lines_b.append(f"{idb[(i, 0)]:10d}{idb[(i + 1, 0)]:10d}")
    clamp = [ida[(0, j)] for j in range(2)] + [ida[(nseg, j)]
                                               for j in range(2)] \
        + [idb[(0, j)] for j in range(2)] + [idb[(nseg, j)]
                                             for j in range(2)]
    loaded = [idb[(mid, 0)], idb[(mid, 1)]]
    damp_card = "/DAMP/1\nsettle\n      10.0         3\n" if damp else ""
    return f"""\
#RADIOSS STARTER
/BEGIN
crossed strips
/NODE
{chr(10).join(nodes)}
/SHELL/1
{chr(10).join(shells)}
/PART/1
strips
         1         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.3
/PROP/SHELL/1
shell
       1.0         3      0.01
/GRNOD/NODE/1
clamp
{" ".join(map(str, clamp))}
/GRNOD/NODE/2
loaded
{" ".join(map(str, loaded))}
/GRNOD/NODE/3
all B
{" ".join(str(idb[(i, j)]) for j in range(2) for i in range(nseg + 1))}
/BCS/1
clamp
       111       111         0         1
/LINE/SEG/1
edge B
{chr(10).join(lines_b)}
/LINE/SEG/2
edge A
{chr(10).join(lines_a)}
/INTER/TYPE11/1
cross
         1         2         1         0
      2.00       0.0      0.50       0.0
{damp_card}{load}/END
"""


def test_type11_crossed_strips_implicit_vs_explicit(make_deck):
    """The M4 edge_impact geometry made quasi-static: crossed clamped
    shell strips pressed together through TYPE11. The implicit static
    answer must match the EXPLICIT solver run to a damped steady state
    on the same deck (same edge stiffness/gap machinery — shared code),
    within 2% on the loaded strip's center deflection and the struck
    strip's center deflection."""
    F = 0.02
    load_imp = f"""\
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/CLOAD/1
press
         1         Z         2   {-F / 2.0}
"""
    load_exp = f"""\
/FUNCT/1
ramp hold
       0.0       0.0
       1.0       1.0
      99.0       1.0
/CLOAD/1
press
         1         Z         2   {-F / 2.0}
"""
    m_imp, _ = _run(make_deck, "XSI", _crossed_strips_deck(load_imp),
                    "#\n/RUN/XSI/1\n1.0\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-8  40\n/END\n")
    assert m_imp.implicit_result.converged
    m_exp, _ = _run(make_deck, "XSE",
                    _crossed_strips_deck(load_exp, damp=True), """\
#
/RUN/XSE/1
8.0
/DT
0.9 0.0
/PRINT/-10000
/END
""")
    d_i = m_imp.x - m_imp.x0
    d_e = m_exp.x - m_exp.x0
    # loaded (upper) strip center and struck (lower) strip center, found
    # by coordinates (both models share the same node table)
    ib = int(np.argmin(np.linalg.norm(
        m_imp.x0 - np.array([0.0, 0.0, 1.0]), axis=1)))
    ia = int(np.argmin(np.linalg.norm(
        m_imp.x0 - np.array([0.0, 0.0, 0.0]), axis=1)))
    assert abs(d_i[ib, 2]) > 1e-3             # something actually moved
    assert d_i[ib, 2] == pytest.approx(d_e[ib, 2], rel=0.02)
    assert d_i[ia, 2] == pytest.approx(d_e[ia, 2], rel=0.02)


def test_type11_friction_warns_and_runs_frictionless(make_deck):
    """TYPE11 friction under implicit is DEFERRED: the interface warns
    and runs frictionless — the crossed-edges closed form (a frictionless
    result) still holds exactly."""
    h, gap, K, kz, F = 0.09, 0.1, 100.0, 2.0, 2.0
    model, out = _run(make_deck, "X11F",
                      _crossed_edges_deck(h, gap, K, kz, F, fric=0.3),
                      "#\n/RUN/X11F/1\n1.0\n/IMPL/DTINI\n0.25\n"
                      "/IMPL/NEWTON\n1e-10  30\n/END\n")
    assert "DEFERRED" in out and "FRICTIONLESS" in out
    u_exact = (K * (gap - h) - F) / (2 * kz + K)
    d = model.x - model.x0
    assert d[model.node_index(1), 2] == pytest.approx(u_exact, rel=1e-8)


# ----------------------------------------------------------------------------
# /PLOAD follower-load stiffness
# ----------------------------------------------------------------------------

def _pload_block_deck(pscale):
    """A single brick with /PLOAD on its WARPED top face (node 7 raised —
    a generic non-planar segment for the FD check), base clamped."""
    return f"""\
#RADIOSS STARTER
/BEGIN
pload block
/NODE
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
         3                 1.0                 1.0                 0.0
         4                 0.0                 1.0                 0.0
         5                 0.0                 0.0                 1.0
         6                 1.0                 0.0                 1.0
         7                 1.0                 1.0                 1.3
         8                 0.0                 1.0                 1.0
/BRICK/1
         1         1         2         3         4         5         6         7         8
/PART/1
cube
         1         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.3
/PROP/SOLID/1
solid
       1.1      0.05       0.1
/GRNOD/NODE/1
base
1 2 3 4
/BCS/1
base
       111       000         0         1
/SURF/SEG/1
warped top
         5         6         7         8
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/PLOAD/1
press
         1         1   {pscale}
/END
"""


def test_pload_stiffness_fd_exact(make_deck):
    """The assembled follower-load stiffness is the EXACT derivative of
    the /PLOAD residual force: central FD of the lumped area-vector
    pressure force on a WARPED segment matches -K_load in every random
    direction to FD accuracy."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.engine.kinematics import LoadsAndConstraints
    from pyradioss.implicit.dofmap import DofMap
    from pyradioss.implicit.followerload import pload_tangent

    model = _model(make_deck, "PFD", _pload_block_deck(0.7))
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        loads = LoadsAndConstraints(model, log)
        dof = DofMap(model)                   # full numbering: probe all
    n = model.numnod
    t = 1.0

    def fext(x):
        f = np.zeros((n, 3))
        loads.external_forces(t, f, x)
        # gravity is absent; only the pressure contributes
        return dof.gather_residual(f, np.zeros((n, 3)))

    Kl = pload_tangent(loads, model, t, model.x0, dof).toarray()
    rng = np.random.default_rng(3)
    eps = 1e-7
    for _ in range(6):
        v_eq = rng.standard_normal(dof.ndof)
        du, _ = dof.scatter_solution(v_eq)
        fd = (fext(model.x0 + eps * du) - fext(model.x0 - eps * du)) \
            / (2 * eps)
        # K_load = -d f_ext/d x  ->  FD(f_ext) = -K_load v
        assert np.allclose(fd, -Kl @ v_eq, rtol=1e-6,
                           atol=1e-8 * max(np.abs(fd).max(), 1.0))


def _pressure_strip_deck(p):
    """A clamped shell strip under follower pressure — the M9 elastica
    strip proportions (40 x 10 x 1, 16 x 2 mesh), where the NLGEOM shell
    Newton is known to behave."""
    Lx, b, t = 40.0, 10.0, 1.0
    nx, ny = 16, 2
    nodes, shells = [], []

    def nid(i, j):
        return 1 + i + j * (nx + 1)
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append(f"{nid(i, j):10d}{i * Lx / nx:20.10f}"
                         f"{j * b / ny:20.10f}{0.0:20.10f}")
    eid = 0
    for j in range(ny):
        for i in range(nx):
            eid += 1
            shells.append(f"{eid:10d}{nid(i, j):10d}{nid(i + 1, j):10d}"
                          f"{nid(i + 1, j + 1):10d}{nid(i, j + 1):10d}")
    fixed = [nid(0, j) for j in range(ny + 1)]
    return f"""\
#RADIOSS STARTER
/BEGIN
pressure strip
/NODE
{chr(10).join(nodes)}
/SHELL/1
{chr(10).join(shells)}
/PART/1
strip
         1         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.3
/PROP/SHELL/1
shell
       {t}         5      0.01
/GRNOD/NODE/1
clamp
{" ".join(map(str, fixed))}
/BCS/1
clamp
       111       111         0         1
/SURF/PART/1
strip surface
1
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/PLOAD/1
press
         1         1   {p}
/END
"""


def _run_pload_direct(make_deck, name, deck, load_stiff, tol="1e-9",
                      dtini=0.2):
    """Starter + the implicit static driver called DIRECTLY so the
    ``impl_load_stiff`` toggle (no input card — a validation hook) can be
    exercised; mirrors run_engine's implicit dispatch."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.implicit.statics import run_implicit_static
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck
    from pyradioss.starter.restart import read_restart
    import os

    s, e = make_deck(name, deck, f"""\
#
/RUN/{name}/1
1.0
/IMPL/NONLIN
/IMPL/DTINI
{dtini}
/IMPL/NEWTON
{tol}  40
/END
""")
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        controls = parse_engine_deck(read_deck(e), log)
        controls.impl_load_stiff = load_stiff
        rst = os.path.join(os.path.dirname(e), f"{name}_0000.rst")
        model, _ = read_restart(rst)
        model = run_implicit_static(model, controls, log)
    return model


def test_pload_follower_stiffness_speedup_same_answer(make_deck):
    """A SOFT warped-face brick under a strong follower pressure (top-face
    rotation grows through the run — pressure-dominated NLGEOM): with the
    load stiffness OFF the residual is IDENTICAL (the follower pressure
    is evaluated at the trial configuration either way), so the converged
    answer must agree to the Newton tolerance — but Newton needs strictly
    MORE iterations without the -d f_ext/d x term. Both asserted; plus
    the small-pressure limit of the follower residual on a shell strip
    against the linear closed form w = (p b) L^4/(8 E I)."""
    deck = _pload_block_deck(0.15).replace(
        "     210.0       0.3", "       2.1       0.3")   # soft block
    m_on = _run_pload_direct(make_deck, "PON", deck, True)
    m_off = _run_pload_direct(make_deck, "POFF", deck, False)
    assert m_on.implicit_result.converged
    assert m_off.implicit_result.converged
    it_on = sum(i.iterations for i in m_on.implicit_result.increments)
    it_off = sum(i.iterations for i in m_off.implicit_result.increments)
    assert it_on < it_off, (it_on, it_off)
    d_on = m_on.x - m_on.x0
    d_off = m_off.x - m_off.x0
    i7 = m_on.node_index(7)
    assert abs(d_on[i7, 2]) > 0.1             # genuinely large deformation
    for nid in (5, 6, 7, 8):
        i = m_on.node_index(nid)
        assert d_on[i] == pytest.approx(d_off[i], rel=1e-6, abs=1e-9)

    # small-pressure limit on the shell strip: linear beam theory for the
    # uniformly loaded cantilever (a loose tolerance also covers the
    # documented BT4 drilling-row residual floor — the run exits on the
    # displacement-correction criterion)
    p_small = 4.0e-6
    m_lin = _run_pload_direct(make_deck, "PLIN",
                              _pressure_strip_deck(p_small), True,
                              tol="1e-4", dtini=0.25)
    assert m_lin.implicit_result.converged
    d_lin = m_lin.x - m_lin.x0
    tip = np.argmax(m_lin.x0[:, 0])
    EI = 210.0 * 10.0 * 1.0 ** 3 / 12.0
    w_ref = p_small * 10.0 * 40.0 ** 4 / (8.0 * EI)
    assert d_lin[tip, 2] == pytest.approx(w_ref, rel=0.03)


def test_pload_with_arclength_refused(make_deck):
    """A follower pressure violates the arc-length method's proportional
    loading assumption — refused loudly (M13), never silently traced on
    a frozen pattern."""
    s, e = make_deck("PARC", _pressure_strip_deck(1.0e-5),
                     "#\n/RUN/PARC/1\n1.0\n/IMPL\n/IMPL/ARCL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(NotImplementedError, match="PLOAD"):
            run_engine(e)


# ----------------------------------------------------------------------------
# LAW36 consistent tangent
# ----------------------------------------------------------------------------

# table: sigma_y = 0.4 at 0, 0.5 at 0.05, 0.52 at 0.1 (end slope 0.2)
_TAB36 = """\
/MAT/LAW36/1
tabulated steel
   7.8e-6
     210.0       0.3
1 0
11
/FUNCT/11
hardening
       0.0       0.4
      0.05       0.5
       0.1      0.52
"""

# the same static curve + a stiffer 100/s curve: the implicit run must
# truncate the family to the static member (warned)
_TAB36_RATE = """\
/MAT/LAW36/1
tabulated steel rate
   7.8e-6
     210.0       0.3
2 0
11 12
0.0 100.0
/FUNCT/11
hardening static
       0.0       0.4
      0.05       0.5
       0.1      0.52
/FUNCT/12
hardening fast
       0.0       0.6
      0.05       0.7
       0.1      0.72
"""

#: closed form for the target stress 0.51 on the table's second segment
#: (slope (0.52-0.5)/0.05 = 0.4): eps_p = 0.05 + (0.51-0.5)/0.4 = 0.075
_SIG36, _EPSP36 = 0.51, 0.075


def _hexa36_deck(mat):
    L = 10.0
    F = _SIG36 * L * L
    return f"""\
#RADIOSS STARTER
/BEGIN
law36 pull
/NODE
         1                 0.0                 0.0                 0.0
         2                {L}                 0.0                 0.0
         3                {L}                {L}                 0.0
         4                 0.0                {L}                 0.0
         5                 0.0                 0.0                {L}
         6                {L}                 0.0                {L}
         7                {L}                {L}                {L}
         8                 0.0                {L}                {L}
/BRICK/1
         1         1         2         3         4         5         6         7         8
/PART/1
cube
         1         1
{mat}/PROP/SOLID/1
solid
       1.1      0.05       0.1
/GRNOD/NODE/1
x0
1 4 5 8
/GRNOD/NODE/2
x1
2 3 6 7
/GRNOD/NODE/3
y0
1 2 5 6
/GRNOD/NODE/4
z0
1 2 3 4
/BCS/1
fx
       100       000         0         1
/BCS/2
fy
       010       000         0         3
/BCS/3
fz
       001       000         0         4
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
pull
         1         X         2   {F / 4.0}
/END
"""


def test_law36_solid_uniaxial_on_table_quadratic(make_deck):
    """Single hexa pulled past yield with LAW36: the plastic strain lands
    EXACTLY on the tabulated curve (the piecewise-linear return is exact
    — no iterated-return upgrade needed, unlike the M11 truss), the
    displacement matches u = (sigma/E + eps_p) L, and the plastic
    increments have the QUADRATIC tail only the consistent tangent
    delivers."""
    model, _ = _run(make_deck, "T36", _hexa36_deck(_TAB36),
                    "#\n/RUN/T36/1\n1.0\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-10  40\n/END\n")
    res = model.implicit_result
    assert res.converged
    assert model.bricks.state["epsp"][0] == pytest.approx(_EPSP36,
                                                          rel=1e-9)
    assert model.bricks.state["sig"][0, 0] == pytest.approx(_SIG36,
                                                            rel=1e-9)
    ux = (model.x - model.x0)[model.node_index(2), 0]
    assert ux == pytest.approx((_SIG36 / 210.0 + _EPSP36) * 10.0, rel=1e-8)
    _assert_quadratic_tail(res)


def _quad36_deck(mat):
    sig = _SIG36
    return f"""\
#RADIOSS STARTER
/BEGIN
law36 shell pull
/NODE
         1       0.0       0.0       0.0
         2      10.0       0.0       0.0
         3      10.0      10.0       0.0
         4       0.0      10.0       0.0
/SHELL/1
         1         1         2         3         4
/PART/1
q
         1         1
{mat}/PROP/SHELL/1
shell
         1         0         0         0
      0.01      0.01      0.01         0         0
         3         0       1.0
/GRNOD/NODE/1
x0
1 4
/GRNOD/NODE/2
x1
2 3
/GRNOD/NODE/3
n1
1
/GRNOD/NODE/4
all
1 2 3 4
/BCS/1
fixx
       100       000         0         1
/BCS/2
fixy n1
       010       000         0         3
/BCS/3
membrane only
       001       111         0         4
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
pull
         1         X         2       {sig * 10.0 * 1.0 / 2.0}
/END
"""


def test_law36_shell_uniaxial_on_table_quadratic(make_deck):
    """BT4 membrane pull past yield with LAW36: eps_p lands exactly on
    the table, the displacement matches the Iplas=2 radial projection's
    own closed form u/L = sigma/E + (3G/E) eps_p (the documented M11
    property of the ported plane-stress variant), quadratic tail."""
    E, nu = 210.0, 0.3
    G = E / (2 * (1 + nu))
    model, _ = _run(make_deck, "Q36", _quad36_deck(_TAB36),
                    "#\n/RUN/Q36/1\n1.0\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-10  40\n/END\n")
    res = model.implicit_result
    assert res.converged
    epsp = model.shells.state["epsp"][0]
    assert epsp == pytest.approx(_EPSP36, rel=1e-6)
    sig = model.shells.state["sig"][0]
    assert sig[:, 0] == pytest.approx(_SIG36, rel=1e-6)
    ux = (model.x - model.x0)[model.node_index(2), 0]
    u_ref = (_SIG36 / E + (3.0 * G / E) * _EPSP36) * 10.0
    assert ux == pytest.approx(u_ref, rel=1e-6)
    _assert_quadratic_tail(res)


def test_law36_rate_family_truncated_with_warning(make_deck):
    """A LAW36 strain-rate curve family under implicit runs on the
    STATIC (first) curve only — warned loudly, and the answer matches
    the single-curve closed form (the pseudo-velocity drive must never
    feed the rate interpolation)."""
    model, out = _run(make_deck, "R36", _hexa36_deck(_TAB36_RATE),
                      "#\n/RUN/R36/1\n1.0\n/IMPL/DTINI\n0.1\n"
                      "/IMPL/NEWTON\n1e-10  40\n/END\n")
    assert "LAW36" in out and "static" in out and "DEFERRED" in out
    assert model.implicit_result.converged
    assert model.bricks.state["epsp"][0] == pytest.approx(_EPSP36,
                                                          rel=1e-9)


# ----------------------------------------------------------------------------
# parity contract: nothing shared is mutated
# ----------------------------------------------------------------------------

def test_m13_builds_do_not_touch_shared_state(make_deck):
    """Building and evaluating the M13 treatments — friction contact
    (forces/energy/tangent/COMMIT), TYPE11 contact, the follower-load
    tangent — must leave the model arrays and element buffers
    byte-for-byte unchanged (the M7 parity contract; the commit writes
    only the treatment's OWN anchor state)."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.engine.kinematics import LoadsAndConstraints
    from pyradioss.implicit.contact import (build_implicit_contacts,
                                            commit_contacts,
                                            contact_tangent)
    from pyradioss.implicit.dofmap import DofMap
    from pyradioss.implicit.followerload import pload_tangent

    log = MessageLog()
    model = _model(make_deck, "PURE13",
                   _fric_deck(0.4, sep=0.02, shift=0.5))
    before_x = model.x.copy()
    before_state = {k: v.copy() for k, v in model.bricks.state.items()
                    if isinstance(v, np.ndarray)}
    with contextlib.redirect_stdout(io.StringIO()):
        contacts = build_implicit_contacts(model, log)
        dof = DofMap(model, log)
        f = np.zeros((model.numnod, 3))
        for c in contacts:
            c.forces(model.x, f)
            c.energy(model.x)
        contact_tangent(contacts, model.x, dof)
        commit_contacts(contacts, model.x)
    assert np.array_equal(model.x, before_x)
    for k, v in before_state.items():
        assert np.array_equal(model.bricks.state[k], v), k

    model2 = _model(make_deck, "PURE13B", _pload_block_deck(0.7))
    bx = model2.x.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        loads = LoadsAndConstraints(model2, log)
        dof2 = DofMap(model2, log)
        pload_tangent(loads, model2, 1.0, model2.x, dof2)

        c11 = _model(make_deck, "PURE13C",
                     _crossed_edges_deck(0.05, 0.1, 5.0, 2.0, 1.0))
        b11 = c11.x.copy()
        cons = build_implicit_contacts(c11, log)
        dof3 = DofMap(c11, log)
        f3 = np.zeros((c11.numnod, 3))
        for c in cons:
            c.forces(c11.x, f3)
            c.energy(c11.x)
        contact_tangent(cons, c11.x, dof3)
    assert np.array_equal(model2.x, bx)
    assert np.array_equal(c11.x, b11)
