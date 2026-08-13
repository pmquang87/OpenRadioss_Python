"""
M15 validations: friction MODELS (Ifric > 0: MFROT mu(p, v) + IFQ
filtering) in BOTH solvers, the LAW27 implicit shell tangent and the
LAW2 beam resultant-plasticity tangent.

Every new capability gets at least one ANALYTIC check (the port's
philosophy):

* the MFROT laws themselves (contact/friction.py, the i7for3.F blocks):
  exact formula checks for MFROT 1..4 including every Renard branch and
  the branch-joint continuity, the EM30 floor, and the static limits
  mu(p, v=0) with their pressure derivatives FD-verified;
* EXPLICIT friction models (inter_type7/11): the transmitted tangential
  force at a PRESCRIBED pressure matches mu(p, v) * Fn * v/(v + eps)
  exactly (kernel-level, the closed form recomputed independently), the
  mu(v) curve traced through all three Renard branches in steady
  sliding, the IFQ filter reproducing the exact discrete first-order
  response F_k = (1 - (1-alpha)^k) F_target and the reader's XFILTR
  mapping for Ifiltr 1/2/3; the TYPE11 port-extension pressure
  definition p = fn/(L*gap) verified the same way; Ifric = 0 decks
  BIT-IDENTICAL (the mu(p,v) evaluator provably never called — it is
  monkeypatched to raise); numba backend parity on an MFROT deck (the
  friction path is downstream of the mirrored narrow phase);
* IMPLICIT friction models (implicit/contact.py): the slip closed form
  with the pressure-dependent cone (transmitted shear = sum mu(p_i)
  fn_i, exact), stick unchanged below the cone, FD residual/tangent
  consistency in BOTH regimes with the mu'(p) coupling block active
  (secondary-side directions: the frozen-area rows are exact), mfrot=0
  provably never evaluating the model, the velocity terms reduced to
  their static limit (warned, and the v-coefficients provably absent
  from the converged force), IFQ ignored with a warning, and the
  implicit-vs-explicit steady-sliding cross-check at matched pressure;
* LAW27 implicit tangent (materials/law27_brittle.py): pre-crack runs
  reproduce LAW1 exactly, law-level FD consistency in EVERY branch
  (uncracked / open-frozen / open-growing / closed / broken), the
  cracked-then-compressed layer recovering the FULL closed-crack
  stiffness (closed form), and the implicit-vs-explicit crack-pattern
  cross-check on a notched shell strip;
* LAW2 BEAM tangent (elements/beam_type3.py): the cantilever forming its
  hinge at EXACTLY the resultant-surface limit load (perfectly plastic:
  the converged shear matches the closed-form root of
  sqrt((F*arm/W)^2 + 3(F/A)^2) = sy to 1e-6, the root element parked ON
  the yield surface to round-off), quadratic Newton tails with
  hardening, the iterated-return decision MEASURED (the explicit
  5-iteration budget visibly off the curve at a virgin-yield
  implicit-size increment, the implicit 60-budget at round-off), LAW1
  beams routed through the new hook BIT-IDENTICALLY, and the
  implicit-vs-explicit quasi-static cross-check;
* the M7 parity/no-shared-mutation contract and the loud refusals
  (Ifiltr >= 10, Ifric = 5).

See PORTING_GUIDE.md roadmap M15.
"""

import contextlib
import io

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.contact import friction
from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            run_starter(s)
            model = run_engine(e)
    except Exception as exc:
        print("STARTER ERROR OUTPUT:\n", buf.getvalue())
        raise
    return model, buf.getvalue()


def _model(make_deck, name, starter):
    s, _ = make_deck(name, starter, "#\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


def _assert_quadratic_tail(res):
    """The M11 quadratic-convergence proxy (see test_m13) — on CONVERGED
    increments (a step-control-cut trial's plateau says nothing about
    the tangent)."""
    plastic = [i for i in res.increments
               if i.iterations > 2 and i.converged]
    assert plastic, "expected at least one plastic (multi-iter) increment"
    inc = max(plastic, key=lambda i: i.iterations)
    r = np.array(inc.residuals)
    r = r[r > 0]
    if r[-1] <= 1e-12 * r[0]:
        return
    tail = r[-3:]
    assert len(tail) == 3
    assert tail[2] / tail[1] < (tail[1] / tail[0]) ** 1.5


def _fric_deck(mu, K4=50.0, gap=0.05, sep=0.0499, loads="", wide=True,
               EA="2.1e6", ifric=0, ifq=0, xfreq=0.0, coefs=None):
    """The M13 two-block friction deck extended with the M15 friction-
    model fields (Ifric/Ifiltr on card 2, Xfreq on card 3, C1..C6 on the
    optional card 4). ``wide=True`` keeps all four punch nodes in the
    FACE INTERIOR of the single 2x2-plan bottom segment (area 4) — the
    closed forms below rely on it."""
    b = 0.5 if wide else 0.0
    ccard = ""
    if ifric > 0:
        c = coefs or (0.0,) * 6
        ccard = ("".join(f"{v:10.4f}" for v in c)) + "\n"
    return f"""\
#RADIOSS STARTER
/BEGIN
friction model blocks
/NODE
         1    {-b:16.6f}    {-b:16.6f}                 0.0
         2    {1.0 + b:16.6f}    {-b:16.6f}                 0.0
         3    {1.0 + b:16.6f}    {1.0 + b:16.6f}                 0.0
         4    {-b:16.6f}    {1.0 + b:16.6f}                 0.0
         5    {-b:16.6f}    {-b:16.6f}                 1.0
         6    {1.0 + b:16.6f}    {-b:16.6f}                 1.0
         7    {1.0 + b:16.6f}    {1.0 + b:16.6f}                 1.0
         8    {-b:16.6f}    {1.0 + b:16.6f}                 1.0
        11                 0.0                 0.0    {1.0 + sep:16.6f}
        12                 1.0                 0.0    {1.0 + sep:16.6f}
        13                 1.0                 1.0    {1.0 + sep:16.6f}
        14                 0.0                 1.0    {1.0 + sep:16.6f}
        15                 0.0                 0.0    {2.0 + sep:16.6f}
        16                 1.0                 0.0    {2.0 + sep:16.6f}
        17                 1.0                 1.0    {2.0 + sep:16.6f}
        18                 0.0                 1.0    {2.0 + sep:16.6f}
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
bottom
   7.8e-6
     {EA}       0.0
/MAT/LAW1/2
punch
   7.8e-6
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
         2         1         1         0         0   {ifric:7d}   {ifq:7d}
    {K4:6.1f}    {mu:6.3f}    {gap:6.3f}       0.0    {xfreq:8.4f}
{ccard}{loads}/END
"""


_RAMP = """\
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
"""


# ----------------------------------------------------------------------------
# the MFROT laws (contact/friction.py) — formula-level
# ----------------------------------------------------------------------------

def test_mfrot_law_formulas():
    """Exact values of every MFROT law against its i7for3.F expression,
    the Renard branch JOINTS continuous, and the EM30 floor."""
    c1 = np.array([0.1, 0.05, 0.02, 0.01, 0.03, 0.0])
    p, v = np.array([2.0]), np.array([1.5])
    mu = friction.mu_kinetic(1, 0.3, c1, p, v)
    assert mu[0] == pytest.approx(
        0.3 + (0.1 + 0.01 * 2) * 2 + (0.05 + 0.02 * 2) * 1.5
        + 0.03 * 1.5 ** 2, rel=1e-14)
    c2 = np.array([0.02, -0.5, 0.05, -0.3, 0.2, -0.1])
    mu = friction.mu_kinetic(2, 0.1, c2, p, v)
    assert mu[0] == pytest.approx(
        0.1 + 0.02 * np.exp(-0.5 * 1.5) * 4 + 0.05 * np.exp(-0.3 * 1.5) * 2
        + 0.2 * np.exp(-0.1 * 1.5), rel=1e-14)
    # Renard: C1 static, C3 at Vcr1, C4 at Vcr2, ->C2 asymptote
    c3 = np.array([0.2, 0.35, 0.3, 0.25, 1.0, 2.0])
    z = np.array([0.0])
    assert friction.mu_kinetic(3, 0.0, c3, z, np.array([0.0]))[0] \
        == pytest.approx(0.2)
    assert friction.mu_kinetic(3, 0.0, c3, z, np.array([1.0]))[0] \
        == pytest.approx(0.3)
    # continuity across both joints
    for vj in (1.0, 2.0):
        lo = friction.mu_kinetic(3, 0.0, c3, z, np.array([vj - 1e-10]))[0]
        hi = friction.mu_kinetic(3, 0.0, c3, z, np.array([vj + 1e-10]))[0]
        assert lo == pytest.approx(hi, abs=1e-8)
    # large-v asymptote -> C2
    assert friction.mu_kinetic(3, 0.0, c3, z, np.array([1e4]))[0] \
        == pytest.approx(0.35, abs=1e-6)
    # exponential decay: mu0 at v=0, C1 at v=inf
    c4 = np.array([0.15, 2.0, 0.0, 0.0, 0.0, 0.0])
    assert friction.mu_kinetic(4, 0.4, c4, z, np.array([0.0]))[0] \
        == pytest.approx(0.4)
    assert friction.mu_kinetic(4, 0.4, c4, z, np.array([50.0]))[0] \
        == pytest.approx(0.15, abs=1e-6)
    # the EM30 floor of i7for3.F
    cneg = np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert friction.mu_kinetic(1, 0.1, cneg, p, np.array([0.0]))[0] == 1e-30


def test_mfrot_static_limits_and_pressure_derivative():
    """mu_static == mu_kinetic at v = 0 for every law, and the returned
    dmu/dp FD-consistent (the mu'(p) the implicit coupling block uses).
    MFROT 3 lands on its static coefficient C1; MFROT 4 on the card
    Fric — both with mu' = 0."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.1, 3.0, 5)
    cases = [(1, 0.3, np.array([0.1, 0.05, 0.02, 0.01, 0.03, 0.0])),
             (2, 0.1, np.array([0.02, -0.5, 0.05, -0.3, 0.2, -0.1])),
             (3, 0.0, np.array([0.2, 0.35, 0.3, 0.25, 1.0, 2.0])),
             (4, 0.4, np.array([0.15, 2.0, 0.0, 0.0, 0.0, 0.0]))]
    for mfrot, mu0, c in cases:
        mu, dmu = friction.mu_static(mfrot, mu0, c, p)
        muk = friction.mu_kinetic(mfrot, mu0, c, p, np.zeros_like(p))
        assert np.allclose(mu, muk, rtol=1e-14)
        h = 1e-7
        fd = (friction.mu_static(mfrot, mu0, c, p + h)[0]
              - friction.mu_static(mfrot, mu0, c, p - h)[0]) / (2 * h)
        assert np.allclose(dmu, fd, rtol=1e-6, atol=1e-9), mfrot
    assert np.all(friction.mu_static(3, 0.0, cases[2][2], p)[1] == 0.0)
    assert np.all(friction.mu_static(4, 0.4, cases[3][2], p)[1] == 0.0)


# ----------------------------------------------------------------------------
# reader (starter_keywords) — the hm_read_inter_type07.F mapping
# ----------------------------------------------------------------------------

def test_reader_friction_fields_and_xfiltr_mapping(make_deck):
    """Ifric/Ifiltr/Xfreq/C1..C6 read exactly as the original reader:
    XFILTR = Xfreq (IFQ 1), 2*pi/Xfreq (IFQ 2), 2*pi*Xfreq (IFQ 3);
    C6 only read for Ifric > 1."""
    m = _model(make_deck, "RD1", _fric_deck(
        0.2, ifric=1, ifq=1, xfreq=0.25,
        coefs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    itf = m.interfaces[0]
    assert itf.mfrot == 1 and itf.ifq == 1
    assert itf.xfiltr == pytest.approx(0.25)
    # C6 NOT read for Ifric = 1 (hm_read_inter_type07.F)
    assert itf.fric_c == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.0))
    m = _model(make_deck, "RD2", _fric_deck(
        0.2, ifric=2, ifq=2, xfreq=100.0,
        coefs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    itf = m.interfaces[0]
    assert itf.xfiltr == pytest.approx(2.0 * np.pi / 100.0)
    assert itf.fric_c[5] == pytest.approx(0.6)     # C6 read for Ifric > 1
    m = _model(make_deck, "RD3", _fric_deck(
        0.2, ifric=3, ifq=3, xfreq=2.0,
        coefs=(0.2, 0.35, 0.3, 0.25, 1.0, 2.0)))
    assert m.interfaces[0].xfiltr == pytest.approx(4.0 * np.pi)


def test_reader_refusals(make_deck):
    """Ifiltr >= 14 (out of range incremental formulation) and an
    out-of-range Ifric are refused loudly."""
    with pytest.raises(StarterError):
        _model(make_deck, "RF1", _fric_deck(0.2, ifric=1, ifq=14,
                                            coefs=(0.1,) * 6))
    with pytest.raises(StarterError):
        _model(make_deck, "RF2", _fric_deck(0.2, ifric=5, ifq=0))
    # XFILTR out of range for IFQ 1 (the MSGID 554 mirror)
    with pytest.raises(StarterError):
        _model(make_deck, "RF3", _fric_deck(0.2, ifric=1, ifq=1,
                                            xfreq=1.5, coefs=(0.1,) * 6))


# ----------------------------------------------------------------------------
# EXPLICIT friction models — kernel-level closed forms
# ----------------------------------------------------------------------------

def _explicit_contact(model):
    from pyradioss.contact.inter_type7 import ContactType7
    with contextlib.redirect_stdout(io.StringIO()):
        return ContactType7(model.interfaces[0], model, MessageLog())


def _slide_state(model, vt):
    """Punch nodes sliding laterally at speed vt, everything else still."""
    n = model.numnod
    v = np.zeros((n, 3))
    for nid in (11, 12, 13, 14, 15, 16, 17, 18):
        v[model.node_index(nid), 0] = vt
    return model.x.copy(), v


def test_explicit_mfrot1_transmitted_force_exact(make_deck):
    """MFROT 1 at a prescribed penetration: the transmitted tangential
    force equals mu(p, v) * Fn * v/(v + eps) with every factor recomputed
    INDEPENDENTLY here — Fn = K*pen per pair (4 identical face-interior
    pairs), p = Fn/AREA with AREA the 2x2 main face, eps the documented
    regularization 1e-3*gap/dt."""
    mu0, K4, gap, sep = 0.3, 50.0, 0.05, 0.03
    coefs = (0.08, 0.02, 0.005, 0.01, 0.003, 0.0)
    model = _model(make_deck, "XF1", _fric_deck(
        mu0, K4=K4, gap=gap, sep=sep, ifric=1, coefs=coefs))
    c7 = _explicit_contact(model)
    n = model.numnod
    vt, dt = 2.0, 1e-3
    x, v = _slide_state(model, vt)
    fcont = np.zeros((n, 3))
    c7.forces(x, v, model.mass, dt, fcont, cycle=0)
    # independent closed form
    pen = gap - sep
    Fn = K4 * pen                       # vn = 0: no damper term
    area = 4.0                          # the 2x2 bottom face (one segment)
    p = Fn / area
    mu = mu0 + (coefs[0] + coefs[3] * p) * p \
        + (coefs[1] + coefs[2] * p) * vt + coefs[4] * vt ** 2
    eps_reg = 1e-3 * gap / dt
    ft_pair = mu * Fn * vt / (vt + eps_reg)
    fx = np.array([fcont[model.node_index(nid), 0]
                   for nid in (11, 12, 13, 14)])
    assert fx == pytest.approx(-ft_pair, rel=1e-12)
    # normal force untouched by the model
    fz = np.array([fcont[model.node_index(nid), 2]
                   for nid in (11, 12, 13, 14)])
    assert fz == pytest.approx(Fn, rel=1e-12)


def test_explicit_renard_mu_v_curve(make_deck):
    """Steady sliding at several speeds traces the MFROT 3 (Renard)
    mu(v) curve through ALL THREE branches (kernel-level: the recovered
    mu = |Ft|/(Fn * v/(v+eps)) lands on the formula)."""
    K4, gap, sep = 50.0, 0.05, 0.03
    coefs = (0.2, 0.35, 0.3, 0.25, 1.0, 2.0)
    model = _model(make_deck, "XF3", _fric_deck(
        0.0, K4=K4, gap=gap, sep=sep, ifric=3, coefs=coefs))
    c7 = _explicit_contact(model)
    n = model.numnod
    dt = 1e-3
    Fn = K4 * (gap - sep)
    eps_reg = 1e-3 * gap / dt
    c = np.array(coefs)
    for vt in (0.4, 1.0, 1.5, 2.0, 5.0):
        x, v = _slide_state(model, vt)
        fcont = np.zeros((n, 3))
        c7.forces(x, v, model.mass, dt, fcont, cycle=0)
        fx = fcont[model.node_index(11), 0]
        mu_rec = -fx / (Fn * vt / (vt + eps_reg))
        mu_exp = friction.mu_kinetic(3, 0.0, c, np.array([Fn / 4.0]),
                                     np.array([vt]))[0]
        assert mu_rec == pytest.approx(mu_exp, rel=1e-12), vt


def test_ifq_filter_exact_first_order_response(make_deck):
    """IFQ = 1 (alpha = Xfreq): under CONSTANT sliding conditions the
    filtered tangential force follows the exact discrete first-order
    step response F_k = (1 - (1-alpha)^k) * F_target; IFQ = 3 derives
    the per-cycle alpha from the cutoff frequency, alpha = 2*pi*Xfreq*dt
    (the documented MIN(1, .) deviation from the source's MAX)."""
    mu0, K4, gap, sep = 0.3, 50.0, 0.05, 0.03
    for ifq, xfreq, dt in ((1, 0.25, 1e-3), (3, 20.0, 1e-3)):
        model = _model(make_deck, f"IFQ{ifq}", _fric_deck(
            mu0, K4=K4, gap=gap, sep=sep, ifric=1, ifq=ifq, xfreq=xfreq,
            coefs=(0.0,) * 6))
        c7 = _explicit_contact(model)
        n = model.numnod
        vt = 2.0
        x, v = _slide_state(model, vt)
        Fn = K4 * (gap - sep)
        eps_reg = 1e-3 * gap / dt
        ftarget = -mu0 * Fn * vt / (vt + eps_reg)
        alpha = xfreq if ifq == 1 else min(1.0, 2.0 * np.pi * xfreq * dt)
        for k in range(1, 6):
            fcont = np.zeros((n, 3))
            c7.forces(x, v, model.mass, dt, fcont, cycle=0)
            fx = fcont[model.node_index(11), 0]
            expect = (1.0 - (1.0 - alpha) ** k) * ftarget
            assert fx == pytest.approx(expect, rel=1e-12), (ifq, k)


def test_explicit_ifric0_never_evaluates_the_model(make_deck,
                                                   monkeypatch):
    """The Ifric = 0 bit-identity contract, asserted the strong way: the
    mu(p, v) evaluator is monkeypatched to raise, and a CONSTANT-mu
    explicit contact cycle still runs — the M4 code path provably never
    touches the M15 module. (The IFQ filter path is guarded the same
    way.)"""
    def boom(*a, **k):
        raise AssertionError("mu_kinetic called on an Ifric = 0 deck")

    monkeypatch.setattr(friction, "mu_kinetic", boom)
    monkeypatch.setattr(friction, "apply_filter", boom)
    model = _model(make_deck, "BIT0", _fric_deck(0.3, sep=0.03))
    c7 = _explicit_contact(model)
    x, v = _slide_state(model, 1.0)
    fcont = np.zeros((model.numnod, 3))
    c7.forces(x, v, model.mass, 1e-3, fcont, cycle=0)
    assert np.abs(fcont).max() > 0.0        # friction force did flow


def test_explicit_type11_pressure_definition(make_deck):
    """The TYPE11 port extension: mu(p) with p = fn/(L_main * gap_pair)
    — verified kernel-level on crossed edges (the closed form recomputed
    independently)."""
    deck = """\
#RADIOSS STARTER
/BEGIN
crossed edges
/NODE
         1              -1.0               0.0               0.0
         2               1.0               0.0               0.0
         3               0.0              -1.0              0.04
         4               0.0               1.0              0.04
/TRUSS/1
         1         1         2
/TRUSS/2
         2         3         4
/PART/1
a
         1         1
/PART/2
b
         1         1
/MAT/LAW1/1
m
   7.8e-6
     210.0       0.0
/PROP/TRUSS/1
t
       1.0
/LINE/SEG/1
line a
         1         2
/LINE/SEG/2
line b
         3         4
/INTER/TYPE11/1
xf
         1         2         1         0         0         1         0
      25.0       0.1      0.10       0.0       0.0
    0.0500    0.0000    0.0000    0.0000    0.0000    0.0000
/END
"""
    model = _model(make_deck, "T11P", deck)
    from pyradioss.contact.inter_type11 import ContactType11
    with contextlib.redirect_stdout(io.StringIO()):
        c11 = ContactType11(model.interfaces[0], model, MessageLog())
    n = model.numnod
    v = np.zeros((n, 3))
    v[2] = v[3] = [1.5, 0.0, 0.0]       # secondary edge slides in x
    dt = 1e-3
    fcont = np.zeros((n, 3))
    c11.forces(model.x, v, model.mass, dt, fcont, cycle=0)
    K, gap, mu0, C1 = 25.0, 0.10, 0.1, 0.05
    pen = gap - 0.04
    Fn = K * pen
    Lm = 2.0                            # main edge length
    p = Fn / (Lm * gap)
    vt = 1.5
    mu = mu0 + C1 * p
    eps_reg = 1e-3 * gap / dt
    ft = mu * Fn * vt / (vt + eps_reg)
    # secondary ends carry (1-s), s of the force; s = 0.5 at the crossing
    assert fcont[2, 0] == pytest.approx(-0.5 * ft, rel=1e-9)
    assert fcont[3, 0] == pytest.approx(-0.5 * ft, rel=1e-9)


def test_numba_parity_on_mfrot_forces(make_deck):
    """The M7 parity contract with the friction models ON, asserted at
    the level where it is DEFINED — the kernel call: one full contact
    force evaluation (narrow phase + penalty + MFROT mu(p, v) + IFQ
    filter) under each backend agrees to the documented per-kernel
    reduction round-off. The mu(p, v)/IFQ code is the SAME NumPy path in
    both backends — the only backend-dependent block is the mirrored
    t7_narrow, UNCHANGED by M15 (a full violent transient would amplify
    its 1e-15-class reduction differences chaotically, proving nothing
    about the friction path; test_m7's kernel and smooth-run parity
    cover the rest of the contract)."""
    pytest.importorskip("numba")
    import pyradioss.accel as accel

    model = _model(make_deck, "NBP",
                   _fric_deck(0.2, sep=0.03, ifric=1, ifq=1, xfreq=0.5,
                              coefs=(0.05, 0.01, 0.0, 0.0, 0.0, 0.0)))
    x, v = _slide_state(model, 1.3)
    x = x + 1e-4 * np.sin(np.arange(x.size)).reshape(x.shape)  # generic
    out = {}
    for backend in ("numpy", "numba"):
        accel.select_backend(backend)
        try:
            c7 = _explicit_contact(model)
            f = np.zeros((model.numnod, 3))
            c7.forces(x, v, model.mass, 1e-3, f, cycle=0)
            # a second call exercises the IFQ anchor path under the jit
            # narrow phase too
            c7.forces(x, v, model.mass, 1e-3, f, cycle=1)
            out[backend] = f
        finally:
            accel.select_backend("numpy")
    scale = np.abs(out["numpy"]).max()
    assert np.allclose(out["numpy"], out["numba"],
                       rtol=1e-12, atol=1e-13 * scale)


# ----------------------------------------------------------------------------
# IMPLICIT friction models
# ----------------------------------------------------------------------------

def test_implicit_mfrot_slip_closed_form(make_deck):
    """Pressure-dependent cone in SLIP: dead normal load N + a lateral
    displacement drive far beyond the cone. Each of the four
    face-interior pairs carries fn = N/4 at pressure p = fn/AREA
    (AREA = 4), so the transmitted shear is exactly
    4 * mu(p) * N/4 = (mu0 + C1 N/16) * N — measured on the bottom
    block's sig_zx."""
    N, mu0, C1, K4 = 2.0, 0.25, 0.4, 50.0
    loads = _RAMP + f"""\
/CLOAD/1
press
         1         Z         3   {-N / 4.0}
/IMPDISP/1
drag
         1         X         2      0.40
"""
    model, _ = _run(make_deck, "IMS",
                    _fric_deck(mu0, K4=K4, loads=loads, ifric=1,
                               coefs=(C1, 0.0, 0.0, 0.0, 0.0, 0.0)),
                    "#\n/RUN/IMS/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-9  40\n/END\n")
    res = model.implicit_result
    assert res.converged
    sA = model.bricks.state["sig"][0]
    fn_pair = N / 4.0
    p = fn_pair / 4.0
    mu = mu0 + C1 * p
    # bottom block plan area 4: sig_zx = total shear / 4, sig_zz = -N/4
    assert sA[2] == pytest.approx(-N / 4.0, rel=1e-6)
    assert sA[5] == pytest.approx(mu * N / 4.0, rel=1e-6)


def test_implicit_mfrot_stick_below_cone(make_deck):
    """STICK with the pressure-dependent cone: below mu(p)*N the model
    only moves the cone — the transmitted shear equals the applied
    lateral load exactly, as in the constant-mu M13 closed form."""
    N, F, mu0, C1 = 2.0, 0.5, 0.25, 0.4     # cone = (0.25+0.05)*2 = 0.6 > F
    loads = _RAMP + f"""\
/CLOAD/1
press
         1         Z         3   {-N / 4.0}
/CLOAD/2
push
         1         X         2   {F / 4.0}
"""
    model, _ = _run(make_deck, "IMK",
                    _fric_deck(mu0, loads=loads, ifric=1,
                               coefs=(C1, 0.0, 0.0, 0.0, 0.0, 0.0)),
                    "#\n/RUN/IMK/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-9  40\n/END\n")
    assert model.implicit_result.converged
    sA = model.bricks.state["sig"][0]
    assert sA[5] == pytest.approx(F / 4.0, rel=1e-6)


def test_implicit_fd_consistency_with_mu_p_coupling(make_deck):
    """FD residual/tangent consistency in BOTH regimes with the mu'(p)
    coupling block ACTIVE (MFROT 1, large C1). Directions hold the main
    corners, so the segment AREA is exactly frozen and the secondary-
    node rows — including the new mu_t = mu + fn mu'/A slip block — are
    FD-exact; all rows stay within the documented O(p/L) weight-
    variation tolerance."""
    from pyradioss.implicit.contact import (build_implicit_contacts,
                                            contact_tangent)
    from pyradioss.implicit.dofmap import DofMap

    mu0, C1, K4, gap = 0.2, 2.0, 50.0, 0.05
    model = _model(make_deck, "FDM",
                   _fric_deck(mu0, K4=K4, gap=gap, sep=0.02, ifric=1,
                              coefs=(C1, 0.0, 0.0, 0.0, 0.0, 0.0)))
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

    rng = np.random.default_rng(7)
    eps = 1e-7
    for delta, regime in ((0.002, "stick"), (0.050, "slip")):
        u0 = np.zeros((n, 3))
        u0[top, 0] = delta
        Kc = contact_tangent(contacts, model.x0 + u0, dof).toarray()
        for _ in range(4):
            v_eq = rng.standard_normal(dof.ndof)
            v_eq[corner_eqs] = 0.0
            du, _ = dof.scatter_solution(v_eq)
            fd = (residual(u0 + eps * du)
                  - residual(u0 - eps * du)) / (2 * eps)
            an = -Kc @ v_eq
            ref = max(np.abs(fd).max(), 1e-9)
            assert np.abs((fd - an)[sec_eqs]).max() < 1e-6 * ref, regime
            assert np.abs(fd - an).max() < 0.05 * ref, regime


def test_implicit_mfrot0_never_evaluates_the_model(make_deck,
                                                   monkeypatch):
    """mfrot = 0 bit-identity, the strong way: mu_static monkeypatched
    to raise and a constant-mu implicit friction run (the M13 slip
    geometry) still converges — the M13/M14 scalar path provably never
    touches the M15 model."""
    def boom(*a, **k):
        raise AssertionError("mu_static called on an Ifric = 0 deck")

    monkeypatch.setattr(friction, "mu_static", boom)
    N = 2.0
    loads = _RAMP + f"""\
/CLOAD/1
press
         1         Z         3   {-N / 4.0}
/IMPDISP/1
drag
         1         X         2      0.40
"""
    model, _ = _run(make_deck, "IM0",
                    _fric_deck(0.3, loads=loads),
                    "#\n/RUN/IM0/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-9  40\n/END\n")
    assert model.implicit_result.converged
    sA = model.bricks.state["sig"][0]
    assert sA[5] == pytest.approx(0.3 * N / 4.0, rel=1e-6)


def test_implicit_velocity_terms_reduce_to_static_limit(make_deck):
    """A deck whose MFROT 1 coefficients carry VELOCITY terms (C2, C5)
    warns once and converges on the STATIC limit mu(p, v=0) — the C2/C5
    contribution provably absent from the transmitted shear."""
    N, mu0, C1 = 2.0, 0.25, 0.4
    loads = _RAMP + f"""\
/CLOAD/1
press
         1         Z         3   {-N / 4.0}
/IMPDISP/1
drag
         1         X         2      0.40
"""
    model, out = _run(make_deck, "IMV",
                      _fric_deck(mu0, loads=loads, ifric=1,
                                 coefs=(C1, 5.0, 0.0, 0.0, 3.0, 0.0)),
                      "#\n/RUN/IMV/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.1\n"
                      "/IMPL/NEWTON\n1e-9  40\n/END\n")
    assert model.implicit_result.converged
    assert "STATIC LIMIT" in out
    sA = model.bricks.state["sig"][0]
    mu = mu0 + C1 * (N / 16.0)          # v = 0: no C2/C5 contribution
    assert sA[5] == pytest.approx(mu * N / 4.0, rel=1e-6)


def test_implicit_ifq_ignored_with_warning(make_deck):
    """Ifiltr > 0 under the implicit solver: warned, and the converged
    state equals the unfiltered run EXACTLY (the filter's DC limit)."""
    N = 2.0
    loads = _RAMP + f"""\
/CLOAD/1
press
         1         Z         3   {-N / 4.0}
/IMPDISP/1
drag
         1         X         2      0.40
"""
    eng = ("#\n/RUN/{}/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.1\n"
           "/IMPL/NEWTON\n1e-9  40\n/END\n")
    m1, out = _run(make_deck, "IQ1",
                   _fric_deck(0.3, loads=loads, ifric=1, ifq=1, xfreq=0.5,
                              coefs=(0.4, 0.0, 0.0, 0.0, 0.0, 0.0)),
                   eng.format("IQ1"))
    assert "IGNORED under the implicit solver" in out
    m2, _ = _run(make_deck, "IQ2",
                 _fric_deck(0.3, loads=loads, ifric=1,
                            coefs=(0.4, 0.0, 0.0, 0.0, 0.0, 0.0)),
                 eng.format("IQ2"))
    assert np.array_equal(m1.x, m2.x)


def test_implicit_vs_explicit_steady_sliding_matched_pressure(make_deck):
    """The M13 cross-solver identity extended to mu(p): the EXPLICIT
    velocity-regularized kinetic law, evaluated ON the implicit
    converged slip state (same positions, the slide velocity imposed),
    transmits mu(p) * fn * v/(v + eps) per pair with the SAME mu(p) the
    implicit cone used — the two solvers' friction laws agree in steady
    sliding at matched pressure by construction of the static limit."""
    N, mu0, C1, K4, gap = 2.0, 0.25, 0.4, 50.0, 0.05
    loads = _RAMP + f"""\
/CLOAD/1
press
         1         Z         3   {-N / 4.0}
/IMPDISP/1
drag
         1         X         2      0.40
"""
    model, _ = _run(make_deck, "IXE",
                    _fric_deck(mu0, K4=K4, gap=gap, loads=loads, ifric=1,
                               coefs=(C1, 0.0, 0.0, 0.0, 0.0, 0.0)),
                    "#\n/RUN/IXE/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-9  40\n/END\n")
    assert model.implicit_result.converged
    c7 = _explicit_contact(model)
    n = model.numnod
    vt, dt = 2.0, 1e-3
    v = np.zeros((n, 3))
    for nid in (11, 12, 13, 14, 15, 16, 17, 18):
        v[model.node_index(nid), 0] = vt
    fcont = np.zeros((n, 3))
    c7.forces(model.x, v, model.mass, dt, fcont, cycle=0)
    ft = -sum(fcont[model.node_index(nid), 0] for nid in (11, 12, 13, 14))
    fn_i = np.array([fcont[model.node_index(nid), 2]
                     for nid in (11, 12, 13, 14)])
    # the implicit state carries sum fn_i = N; the explicit kinetic law
    # on that state approaches sum mu(p_i) fn_i from below by v/(v+eps).
    # p_i uses the CURRENT (slightly deformed) main-face area, exactly
    # like both solvers do.
    xs = [model.x[model.node_index(nid)] for nid in (5, 6, 7, 8)]
    area = 0.5 * np.linalg.norm(np.cross(xs[2] - xs[0], xs[3] - xs[1]))
    eps_reg = 1e-3 * gap / dt
    mu_i = mu0 + C1 * fn_i / area
    # rel 1e-5: the nodal x/z force components are exact projections
    # only up to the pressed block's O(1e-6) shear tilt of the contact
    # normal — the identity itself is what is being checked
    assert ft == pytest.approx(
        float((mu_i * fn_i).sum()) * vt / (vt + eps_reg), rel=1e-5)
    assert fn_i.sum() == pytest.approx(N, rel=1e-5)


# ----------------------------------------------------------------------------
# LAW27 implicit shell tangent
# ----------------------------------------------------------------------------

def _law27_shell_deck(eps_t1="0.001", extra_cards="", E="70000.0",
                      drive=0.05, mat27=True):
    """One 10x10 shell element clamped at x=0, pulled at x=10 by
    /IMPDISP (drive), LAW27 (or LAW1 for the pre-crack twin)."""
    mat = (f"""/MAT/LAW27/1
glass
   2.5e-6
     {E}      0.22
       0.0       0.0       0.0       0.0       0.0
       0.0       0.0       0.0
     {eps_t1}     0.005       0.9      0.10
    0.0012     0.006      0.85      0.10
""" if mat27 else f"""/MAT/LAW1/1
glass el
   2.5e-6
     {E}      0.22
""")
    return f"""\
#RADIOSS STARTER
/BEGIN
law27 strip
/NODE
         1               0.0               0.0               0.0
         2              10.0               0.0               0.0
         3              10.0              10.0               0.0
         4               0.0              10.0               0.0
/SHELL/1
         1         1         2         3         4
/PART/1
sheet
         1         1
{mat}/PROP/SHELL/1
sheet
         1.0         3
/GRNOD/NODE/1
root
1 4
/GRNOD/NODE/2
tip
2 3
/BCS/1
root
       111       111         0         1
/BCS/2
tip
       011       111         0         2
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/IMPDISP/1
pull
         1         X         2      {drive}
{extra_cards}/END
"""


def test_law27_precrack_equals_law1_exactly(make_deck):
    """Below the initiation strain the LAW27 implicit tangent is the
    plane-stress elastic C: the run reproduces the LAW1 twin EXACTLY
    and converges like an elastic problem."""
    eng = ("#\n/RUN/{}/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.25\n"
           "/IMPL/NEWTON\n1e-10  20\n/END\n")
    m27, _ = _run(make_deck, "L27E",
                  _law27_shell_deck(drive=0.005), eng.format("L27E"))
    m01, _ = _run(make_deck, "L01E",
                  _law27_shell_deck(drive=0.005, mat27=False),
                  eng.format("L01E"))
    assert m27.implicit_result.converged
    assert np.allclose(m27.x, m01.x, rtol=0, atol=1e-12)
    assert np.all(m27.shells.state["mat_extra"]["crk27"] == 0.0)
    # elastic-grade convergence: every increment in <= 2 iterations
    assert max(i.iterations for i in m27.implicit_result.increments) <= 2


def test_law27_tangent_fd_every_branch():
    """Law-level FD of consistent_shell_tangent against shell_update in
    EVERY branch: uncracked, cracked-open GROWING damage, cracked-open
    FROZEN damage (unloaded), cracked CLOSED (compression across the
    crack), rotated mixed state, and broken."""
    from types import SimpleNamespace
    from pyradioss.materials import law27_brittle as l27

    mat = SimpleNamespace(
        E=70000.0, nu=0.22, G=70000.0 / (2 * 1.22),
        params=dict(eps_t1=0.001, eps_m1=0.005, eps_t2=0.0012,
                    eps_m2=0.006, dmax1=0.9, dmax2=0.85,
                    eps_f1=0.1, eps_f2=0.1))

    def make_extra(m):
        return dict(eps27=np.zeros((m, 3)), crk27=np.zeros(m),
                    ang27=np.zeros(m), dmg27=np.zeros((m, 2)),
                    layfail=np.ones(m))

    def stress_at(eps_total, extra0):
        ex = {k: v.copy() for k, v in extra0.items()}
        deps = eps_total - ex["eps27"]
        sig = np.zeros((len(eps_total), 3))
        l27.shell_update(mat, sig, deps, None, 1.0, ex)
        return sig, ex

    m = 6
    extra = make_extra(m)
    eps = np.zeros((m, 3))
    eps[1] = [0.003, 0.0005, 0.001]     # -> cracked, growing
    eps[2] = [0.003, 0.0005, 0.001]     # -> cracked, then unloaded
    eps[3] = [0.004, 0.0, 0.0]          # -> cracked, then compressed
    eps[4] = [0.002, 0.0025, 0.003]     # rotated mixed state
    eps[5] = [0.2, 0.0, 0.0]            # -> broken
    _, extra = stress_at(eps, extra)
    eps2 = eps.copy()
    eps2[2] = [0.002, 0.0004, 0.0008]   # freeze element 2's damage
    _, extra = stress_at(eps2, extra)
    eps3 = eps2.copy()
    eps3[3] = [-0.002, 0.0, 0.0]        # close element 3's crack
    _, extra = stress_at(eps3, extra)

    C = l27.consistent_shell_tangent(mat, extra)
    assert np.all(C[5] == 0.0)          # broken: zero tangent
    h = 1e-9
    s0, _ = stress_at(eps3, extra)
    for el in range(m):
        Cfd = np.zeros((3, 3))
        for j in range(3):
            ep = eps3.copy()
            ep[el, j] += h
            sp, _ = stress_at(ep, extra)
            Cfd[:, j] = (sp[el] - s0[el]) / h
        ref = max(np.abs(C[el]).max(), 1.0)
        assert np.abs(Cfd - C[el]).max() < 3e-6 * ref, el


def test_law27_crack_then_compression_recovers_closed_stiffness(
        make_deck):
    """Load reversal across the unilateral switch: pull past initiation
    (crack opens, damage grows), then drive into COMPRESSION. The closed
    crack transmits full stiffness: the final state matches the LAW1
    elastic closed form u_x -> sig_xx = E/(1-nu^2)*(ex + nu*ey) with the
    lateral edges free (plane stress in a uniaxial-STRESS state along x:
    sigma = E*ex... measured directly against the LAW1 twin driven to
    the same final displacement), and Newton survives the non-smooth
    switch (the M13 line-search backstop)."""
    eng = ("#\n/RUN/{}/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.05\n"
           "/IMPL/NEWTON\n1e-9  40\n/END\n")
    # drive: +0.03 (cracks at eps_t1 = 1e-3 -> eps = 3e-3) then reverse
    # to -0.02 (compression) via the table
    # the implicit drive follows d(lam) with lam in (0, 1]: up to +0.03
    # at half load (cracks at eps = 3e-3 > eps_t1), down to -0.02 at the
    # end (compression across the OPEN crack -> unilateral closure)
    rev = """\
/FUNCT/9
updown
       0.0       0.0
       0.5       3.0
       1.0      -2.0
"""
    deck = _law27_shell_deck(drive=0.01, extra_cards=rev)
    deck = deck.replace("""/IMPDISP/1
pull
         1         X         2      0.01""", """/IMPDISP/1
pull
         9         X         2      0.01""")
    m27, _ = _run(make_deck, "L27R", deck, eng.format("L27R"))
    assert m27.implicit_result.converged
    st = m27.shells.state
    # the layer cracked on the way up...
    assert np.all(st["mat_extra"]["crk27"][0] == 1.0)
    assert st["mat_extra"]["dmg27"][0, :, 0].max() > 0.1
    # ...and the final compressed state transmits FULL stiffness: same
    # answer as the elastic LAW1 twin driven monotonically to -0.02
    m01, _ = _run(make_deck, "L01R",
                  _law27_shell_deck(drive=-0.02, mat27=False),
                  eng.format("L01R"))
    assert np.allclose(m27.x, m01.x, atol=1e-8)
    s27 = st["sig"][0, 0]
    s01 = m01.shells.state["sig"][0, 0]
    assert s27 == pytest.approx(s01, rel=1e-6)


@pytest.mark.slow          # measured 1120.5 s serial (2026-07 full-suite run)
def test_law27_implicit_vs_explicit_notched_strip(make_deck):
    """Crack-PATTERN cross-check on a notched strip: a 3x1 row of shell
    elements whose middle element has a LOWER initiation strain (the
    'notch'), pulled to a displacement that cracks ONLY the notch. The
    implicit run and the explicit quasi-static run must agree on WHICH
    layers cracked (the crk27 pattern), the crack ANGLE, and the damage
    level (5%)."""
    def deck(impl):
        # both solvers land on the SAME final drive 0.03: the implicit
        # d(lam=1) = 0.03, the explicit d(t=20) = 0.03 (f(20) = 1)
        drv = "/FUNCT/1\nramp\n       0.0       0.0\n" \
              "       2.0       2.0\n" if impl else \
              "/FUNCT/1\nramp\n       0.0       0.0\n" \
              "      40.0       2.0\n"
        return f"""\
#RADIOSS STARTER
/BEGIN
notched strip
/NODE
         1               0.0               0.0               0.0
         2              10.0               0.0               0.0
         3              20.0               0.0               0.0
         4              30.0               0.0               0.0
         5               0.0              10.0               0.0
         6              10.0              10.0               0.0
         7              20.0              10.0               0.0
         8              30.0              10.0               0.0
/SHELL/1
         1         1         2         6         5
/SHELL/2
         2         2         3         7         6
/SHELL/3
         3         3         4         8         7
/PART/1
strip
         1         1
/PART/2
notch
         1         2
    /MAT/LAW27/1
    glass
       2.5e-6
         70000.0      0.22
           0.0       0.0       0.0       0.0       0.0
           0.0       0.0       0.0
         0.004     0.008       0.9      0.10
         0.004     0.008      0.85      0.10
    /MAT/LAW27/2
    weak glass
       2.5e-6
         70000.0      0.22
           0.0       0.0       0.0       0.0       0.0
           0.0       0.0       0.0
         0.001     0.005       0.9      0.10
        0.0012     0.006      0.85      0.10
/PROP/SHELL/1
sheet
         1.0         3
/GRNOD/NODE/1
root
1 5
/GRNOD/NODE/2
tip
4 8
/BCS/1
root
       111       111         0         1
/BCS/2
tip
       011       111         0         2
{drv}/IMPDISP/1
pull
         1         X         2      0.03
/END
"""
    # the /SHELL cards put element 2 in part 2? No - parts are by the
    # /SHELL block id in this reader; re-read: /SHELL/1 holds ALL three
    # elements of part 1 unless split. The deck above puts each element
    # in its own /SHELL block -> part ids 1, 2, 3; part 3 undefined.
    # Simpler: move the middle element to part 2 explicitly.
    imp = deck(True).replace("""/SHELL/1
         1         1         2         6         5
/SHELL/2
         2         2         3         7         6
/SHELL/3
         3         3         4         8         7""", """/SHELL/1
         1         1         2         6         5
         3         3         4         8         7
/SHELL/2
         2         2         3         7         6""")
    exp = deck(False).replace("""/SHELL/1
         1         1         2         6         5
/SHELL/2
         2         2         3         7         6
/SHELL/3
         3         3         4         8         7""", """/SHELL/1
         1         1         2         6         5
         3         3         4         8         7
/SHELL/2
         2         2         3         7         6""")
    m_imp, _ = _run(make_deck, "NSI", imp,
                    "#\n/RUN/NSI/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.05\n"
                    "/IMPL/NEWTON\n1e-9  40\n/END\n")
    assert m_imp.implicit_result.converged
    m_exp, _ = _run(make_deck, "NSE", exp,
                    "#\n/RUN/NSE/1\n20.0\n/DT\n0.9 0.0\n/END\n")
    sti, ste = m_imp.shells.state, m_exp.shells.state
    crki = sti["mat_extra"]["crk27"]
    crke = ste["mat_extra"]["crk27"]
    # identical crack pattern: the notch cracked, the outer elements not
    assert np.array_equal(crki > 0, crke > 0)
    notch = np.where(crki.any(axis=1))[0]
    assert len(notch) == 1
    el = notch[0]
    # same crack angle (a uniaxial pull: theta = 0) and damage within 5%
    assert sti["mat_extra"]["ang27"][el] == pytest.approx(
        ste["mat_extra"]["ang27"][el], abs=1e-6)
    di = sti["mat_extra"]["dmg27"][el, :, 0]
    de = ste["mat_extra"]["dmg27"][el, :, 0]
    assert di == pytest.approx(de, rel=0.05)


# ----------------------------------------------------------------------------
# LAW2 beam resultant-plasticity tangent
# ----------------------------------------------------------------------------

def _beam_deck(B, n_h, smax, drive, L=10.0, ne=4):
    nodes = ""
    for i in range(ne + 1):
        nodes += f"{i+1:10d}{L*i/ne:20.10f}{0.0:20.10f}{0.0:20.10f}\n"
    nodes += f"{99:10d}{0.0:20.10f}{1.0:20.10f}{0.0:20.10f}\n"
    beams = ""
    for i in range(ne):
        beams += f"{i+1:10d}{i+1:10d}{i+2:10d}{99:10d}\n"
    return f"""\
#RADIOSS STARTER
/BEGIN
beam hinge
/NODE
{nodes}/BEAM/1
{beams}/PART/1
beam
         1         1
/MAT/LAW2/1
steel
   7.8e-6
     210000.0       0.3
     200.0       {B}       {n_h}     {smax}       0.0
/PROP/BEAM/1
beam
       1.0   0.0833333333   0.0833333333   0.1666666667
/GRNOD/NODE/1
root
1
/GRNOD/NODE/2
tip
{ne+1}
/BCS/1
root
       111       111         0         1
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/IMPDISP/1
tip
         1         Z         2      {drive}
/END
"""


_BEAM_ENG = ("#\n/RUN/{}/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.05\n"
             "/IMPL/NEWTON\n1e-9  40\n/END\n")


def test_beam_hinge_at_resultant_limit_load(make_deck):
    """The port's global-plasticity convention (M3 notes): the hinge
    forms at EXACTLY the resultant yield surface — perfectly plastic
    (B = 0, sig_max = A), tip displacement-driven far past yield. The
    converged shear (constant along the beam) matches the closed-form
    root of sqrt((F*arm/W)^2 + 3 (F/A)^2) = sy with arm = L - dx/2 (the
    one-point element integrates at mid-span) and W = sqrt(I A/3), and
    the root element is parked ON the yield surface to round-off. This
    run also exercises the two M15 DRIVER guards it exposed (the
    non-finite-residual cut and the singular-factor cut — a too-large
    increment spuriously yields enough H = 0 elements to form a
    transient mechanism; the StepControl cuts through it)."""
    np.seterr(over="ignore", invalid="ignore")
    try:
        model, _ = _run(make_deck, "BHP", _beam_deck(0.0, 1.0, 200.0, 0.30),
                        _BEAM_ENG.format("BHP"))
    finally:
        np.seterr(over="warn", invalid="warn")
    res = model.implicit_result
    assert res.converged
    st = model.beams.state
    A_area, Iyy, sy, L, ne = 1.0, 0.0833333333, 200.0, 10.0, 4
    W = np.sqrt(Iyy * A_area / 3.0)
    arm = L - (L / ne) / 2.0
    from scipy.optimize import brentq
    F = brentq(lambda F: np.hypot(F * arm / W, np.sqrt(3.0) * F / A_area)
               - sy, 1e-6, 1e3)
    assert st["fres"][:, 2] == pytest.approx(F, rel=1e-6)
    # the yielded root element sits exactly on the surface
    Nn, Qy, Qz = st["fres"][0]
    Mx, My, Mz = st["mres"][0]
    sn = abs(Nn) / A_area + abs(My) / st["wy"][0] + abs(Mz) / st["wz"][0]
    tau = abs(Mx) / st["wx"][0] + np.hypot(Qy, Qz) / A_area
    assert np.hypot(sn, np.sqrt(3.0) * tau) == pytest.approx(sy, rel=1e-9)
    assert st["epsp"][0] > 1e-3 and np.all(st["epsp"][1:] == 0.0)


def test_beam_hardening_quadratic_tail_and_on_curve(make_deck):
    """Hardening run (B > 0): quadratic Newton tails (the consistent-
    tangent signature) and the yielded element landing EXACTLY on the
    Johnson-Cook curve sy = A + B*ep^n (the ITERATED implicit return)."""
    model, _ = _run(make_deck, "BHH", _beam_deck(400.0, 0.5, 1e30, 0.40),
                    _BEAM_ENG.format("BHH"))
    res = model.implicit_result
    assert res.converged
    _assert_quadratic_tail(res)
    st = model.beams.state
    A_area, sy0 = 1.0, 200.0
    ep = st["epsp"][0]
    assert ep > 1e-4
    Nn, Qy, Qz = st["fres"][0]
    Mx, My, Mz = st["mres"][0]
    sn = abs(Nn) / A_area + abs(My) / st["wy"][0] + abs(Mz) / st["wz"][0]
    tau = abs(Mx) / st["wx"][0] + np.hypot(Qy, Qz) / A_area
    seq = np.hypot(sn, np.sqrt(3.0) * tau)
    assert seq == pytest.approx(sy0 + 400.0 * ep ** 0.5, rel=1e-9)


def test_beam_iterated_return_measured(make_deck):
    """The MEASUREMENT behind the implicit iterated-return decision (the
    M11 truss lesson replayed in resultant space). The escape from the
    near-virgin JC slope B*n*e^(n-1) needs O(1/n) Newton iterations
    before the quadratic tail (dl_{k+1} ~ dl_k^(1-n) while H >> E):
    at n = 0.5 the historical 5-iteration explicit budget happens to
    suffice, but at n = 0.2 — a perfectly ordinary hardening exponent —
    it leaves the consistency residual |seq - E*dl - sy(ep0+dl)| at an
    O(1) fraction of sy while the implicit 60-budget is at round-off.
    That measured gap is why implicit_internal_forces exists (and why
    the explicit kernel, whose tiny steps re-enter every cycle, keeps
    its bit-identical 5)."""
    from types import SimpleNamespace
    from pyradioss.elements.beam_type3 import _global_plastic_return

    mat = SimpleNamespace(E=210000.0, law=2,
                          params=dict(A=200.0, B=400.0, n=0.2,
                                      sig_max=1e30))
    p = dict(area=1.0)

    def state(my):
        return dict(fres=np.array([[0.0, 0.0, 0.0]]),
                    mres=np.array([[0.0, my, 0.0]]),
                    epsp=np.zeros(1),
                    wy=np.array([np.sqrt(0.0833333333 / 3.0)]),
                    wz=np.array([np.sqrt(0.0833333333 / 3.0)]),
                    wx=np.array([np.sqrt(0.1666666667 / 3.0)]))

    def residual_after(iters):
        st = state(60.0)                # seq ~ 360 >> sy0: virgin yield
        _global_plastic_return(st, slice(0, 1), mat, p, iters)
        ep = st["epsp"][0]
        sy = 200.0 + 400.0 * max(ep, 1e-20) ** 0.2
        seq_tr = 60.0 / np.sqrt(0.0833333333 / 3.0)
        return abs(seq_tr - mat.E * ep - sy), ep

    r5, ep5 = residual_after(5)
    r60, ep60 = residual_after(60)
    assert r5 > 1.0                     # visibly off the curve
    assert r60 < 1e-8                   # converged to round-off
    assert ep60 > ep5                   # the 5-iter return under-flows


def test_beam_elastic_hook_route_bit_identical(make_deck):
    """LAW1 beams route through the NEW implicit hook — and it must be
    the OLD route bit for bit: implicit_internal_forces(nlgeom=False)
    equals forces(x_ref, u, ur, 1.0) exactly, and the nlgeom branch
    equals the historical forces(x_mid) + static_internal_forces(x_end)
    pairing exactly, state arrays included."""
    import copy
    from pyradioss.elements import beam_type3 as bt

    deck = _beam_deck(0.0, 1.0, 200.0, 0.1).replace(
        "/MAT/LAW2/1\nsteel\n   7.8e-6\n     210000.0       0.3\n"
        "     200.0       0.0       1.0     200.0       0.0\n",
        "/MAT/LAW1/1\nsteel\n   7.8e-6\n     210000.0       0.3\n")
    model = _model(make_deck, "BEL", deck)
    g = model.beams
    n = model.numnod
    rng = np.random.default_rng(11)
    u = 0.01 * rng.standard_normal((n, 3))
    ur = 0.01 * rng.standard_normal((n, 3))

    for nlg in (False, True):
        snap = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in g.state.items()}
        fa, ma = np.zeros((n, 3)), np.zeros((n, 3))
        bt.implicit_internal_forces(g, model.x0, u, ur, fa, ma, nlg)
        state_a = {k: v.copy() for k, v in g.state.items()
                   if isinstance(v, np.ndarray)}
        # restore and replay the OLD route
        for k, v in snap.items():
            if isinstance(v, np.ndarray):
                g.state[k][...] = v
        fb, mb = np.zeros((n, 3)), np.zeros((n, 3))
        if not nlg:
            bt.forces(g, model.x0, u, ur, 1.0, fb, mb)
        else:
            jf, jm = np.zeros((n, 3)), np.zeros((n, 3))
            bt.forces(g, model.x0 + 0.5 * u, u, ur, 1.0, jf, jm)
            bt.static_internal_forces(g, model.x0 + u, u, ur, fb, mb)
        assert np.array_equal(fa, fb) and np.array_equal(ma, mb), nlg
        for k, v in state_a.items():
            assert np.array_equal(v, g.state[k]), (nlg, k)


@pytest.mark.slow          # measured 206.5 s serial (2026-07 full-suite run)
def test_beam_implicit_vs_explicit_quasistatic(make_deck):
    """Cross-solver check: the hardening cantilever driven quasi-
    statically by the EXPLICIT leapfrog (a slow smooth /IMPDISP ramp —
    ~500 fundamental periods; no damping needed, the kinematic drive
    injects negligible kinetic energy) lands within 2% of the implicit
    converged state — tip force (the constant shear resultant) and
    plastic strain. (Measured at 2 ms: Qz 4.4497 vs 4.4538, 0.09%.)"""
    imp, _ = _run(make_deck, "BXI", _beam_deck(400.0, 0.5, 1e30, 0.40),
                  _BEAM_ENG.format("BXI"))
    assert imp.implicit_result.converged
    # the explicit ramp lands on the SAME final drive: f(t=2) = 1.0
    exp_deck = _beam_deck(400.0, 0.5, 1e30, 0.40).replace(
        "/FUNCT/1\nramp\n       0.0       0.0\n       2.0       2.0\n",
        "/FUNCT/1\nramp\n       0.0       0.0\n       4.0       2.0\n")
    exp, _ = _run(make_deck, "BXE", exp_deck,
                  "#\n/RUN/BXE/1\n2.0\n/DT\n0.9 0.0\n/END\n")
    qi = imp.beams.state["fres"][:, 2]
    qe = exp.beams.state["fres"][:, 2]
    assert qe.mean() == pytest.approx(qi.mean(), rel=0.02)
    assert exp.beams.state["epsp"][0] == pytest.approx(
        imp.beams.state["epsp"][0], rel=0.05)


def test_beam_nlgeom_plastic_converges(make_deck):
    """/IMPL/NONLIN with the plastic beam: the midpoint/end-assembly
    route converges with quadratic tails and matches the linear-geometry
    answer at this moderate drive (sanity on the nlgeom hook branch)."""
    lin, _ = _run(make_deck, "BNL1", _beam_deck(400.0, 0.5, 1e30, 0.30),
                  _BEAM_ENG.format("BNL1"))
    eng = ("#\n/RUN/BNL2/1\n1.0\n/IMPL\n/IMPL/NONLIN\n/IMPL/DTINI\n0.05\n"
           "/IMPL/NEWTON\n1e-9  40\n/END\n")
    nlg, _ = _run(make_deck, "BNL2", _beam_deck(400.0, 0.5, 1e30, 0.30),
                  eng)
    assert nlg.implicit_result.converged
    # tip shear within 2% of the small-displacement answer (drive/L = 3%)
    assert nlg.beams.state["fres"][0, 2] == pytest.approx(
        lin.beams.state["fres"][0, 2], rel=0.02)


# ----------------------------------------------------------------------------
# contracts
# ----------------------------------------------------------------------------

def test_parity_no_shared_mutation(make_deck):
    """The M14-style contract extended to M15: building and evaluating
    the friction-model implicit contacts mutates NOTHING shared with the
    explicit solver — model arrays bit-identical before/after."""
    from pyradioss.implicit.contact import build_implicit_contacts

    model = _model(make_deck, "PAR",
                   _fric_deck(0.2, sep=0.02, ifric=1,
                              coefs=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)))
    x0 = model.x.copy()
    mass0 = model.mass.copy()
    itf_fields = (model.interfaces[0].fric, model.interfaces[0].mfrot,
                  tuple(model.interfaces[0].fric_c))
    with contextlib.redirect_stdout(io.StringIO()):
        contacts = build_implicit_contacts(model, MessageLog())
    n = model.numnod
    u = np.zeros((n, 3))
    top = [model.node_index(nid) for nid in range(11, 19)]
    u[top, 0] = 0.05
    f = np.zeros((n, 3))
    for c in contacts:
        c.forces(model.x + u, f)
        c.commit(model.x + u)
        c.forces(model.x + u, f)
    assert np.array_equal(model.x, x0)
    assert np.array_equal(model.mass, mass0)
    assert (model.interfaces[0].fric, model.interfaces[0].mfrot,
            tuple(model.interfaces[0].fric_c)) == itf_fields
