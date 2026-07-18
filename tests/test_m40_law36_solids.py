"""M40: LAW36 solids forensics — the RD-V-0700 material-fidelity fix.

M39 measured (VALIDATION.md §3.4 item 1): on the V0700 family the LAW2
solids matched Fortran at IE rel_rms 0.023 while the LAW36 solids on the
IDENTICAL geometry deviated ~0.19.  The branch-by-branch diff of
pyradioss/materials/law36_tabulated.py against
engine/source/materials/mat/mat036/sigeps36.F found two root causes,
both OUTSIDE the return-mapping algebra (which matches upstream exactly
on a table segment):

1. the per-curve Fscale_i (sigeps36.F YFAC, applied to the curve VALUE
   and SLOPE at every evaluation) was parsed but dropped with a
   "curves used unscaled" warning.  The V0700 decks tabulate the curve
   in MPa with Fscale = 1e-3 into GPa work units — the port's yield
   was 1000x too high, the material NEVER yielded, and /FAIL/JOHNSON
   (damage driven by the plastic-strain increment) never engaged;

2. /FAIL/JOHNSON with a non-positive failure strain: the reference
   (fail_johnson.F) floors eps_f at EPSF_MIN (default 0) and
   accumulates damage ONLY where eps_f > 0 — a negative eps_f FREEZES
   the damage.  The port divided by max(eps_f, 1e-20), instant-deleting
   such points.  The V0700 calibration (D1=-0.1, D2=0.6, D3=-1.2) goes
   negative beyond triaxiality ln(6)/1.2 = 1.4931, exactly where the
   deck's confined/hydrostatic load states live: the reference keeps
   those elements alive forever, the old port deleted them on their
   first plastic increment.

The fixtures below are byte-faithful copies of the c19 deck blocks
(rd_v_failure/RD-V-0700/.../HEXA_ELEM_SOLID_18/MAT_PROP.inc), so these
tests pin the fixed branches against hand-computed values from the
deck's ACTUAL curve.
"""

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8
from pyradioss.failure import johnson
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials import law36_tabulated
from pyradioss.model.entities import FailureModel
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_materials,
                                              resolve_node_groups,
                                              resolve_surfaces)

# ---------------------------------------------------------------------------
# Byte-faithful c19 blocks (HEXAP14_18 / MAT_PROP.inc).  The full 66-point
# hardening curve of /FUNCT/5 — yield [MPa] vs plastic strain — with
# Fscale_1 = 1e-3 converting it to the deck's GPa work units.
# ---------------------------------------------------------------------------

C19_CURVE = """\
                 0.0             389.674
          0.00326906              400.23
          0.00693011             409.589
           0.0104447             418.254
           0.0118464             422.462
           0.0129266             426.827
           0.0136484             433.649
           0.0152255              443.84
           0.0185893             452.254
           0.0218744             469.641
           0.0253395             483.106
           0.0287915             494.684
           0.0322314             505.958
           0.0356586             515.583
           0.0390739             524.886
           0.0424768             533.055
           0.0458683             541.267
           0.0492477             548.545
           0.0526155             555.479
           0.0559716             561.572
           0.0593166             568.023
           0.0626503             574.174
           0.0659722             578.918
           0.0692833              584.18
           0.0725831             588.852
           0.0758722             593.598
             0.07915              597.58
           0.0824175             602.192
           0.0856737             605.639
           0.0889195             609.378
           0.0921548             613.073
           0.0953795              616.55
           0.0985937             619.696
            0.101798             623.306
            0.104991             626.295
            0.108175             629.521
            0.111348             632.813
            0.114511              635.36
            0.117664             637.852
            0.120807             640.172
            0.123941             642.727
            0.127064              645.58
            0.130178             647.791
            0.133282             650.416
            0.136376             652.452
            0.139461             655.142
            0.142537              657.36
            0.145603               659.4
            0.148659             661.259
            0.151706              663.78
            0.154744              665.64
            0.157773             667.196
            0.160793             670.145
            0.163803             671.639
            0.166804              673.68
            0.169797              676.21
             0.17278             677.208
            0.175754             678.631
             0.17872             680.051
            0.181677             681.901
            0.184625             683.688
            0.187564             684.912
            0.190495             686.069
              0.1936             686.607
             0.19622             687.802
            0.199418             689.023
"""

# the /MAT + /FAIL + /PROP blocks exactly as MAT_PROP.inc writes them
C19_MAT = """\
/MAT/PLAS_TAB/1
Steel
#        Init. dens.
7.80000000000000E-06
#                  E                  Nu           Eps_p_max               Eps_t               Eps_m
               221.0                 0.3                 0.0                 0.0
#  N_funct  F_smooth              C_hard               F_cut               Eps_f                  VP
         1         0
#  fct_IDp              Fscale   Fct_IDE                EInf                  CE

# func_ID1  func_ID2  func_ID3  func_ID4  func_ID5
         5
#           Fscale_1            Fscale_2            Fscale_3            Fscale_4            Fscale_5
1.00000000000000E-03
#          Eps_dot_1           Eps_dot_2           Eps_dot_3           Eps_dot_4           Eps_dot_5
                 0.0
/FAIL/JOHNSON/1
#                 D1                  D2                  D3                  D4                  D5
                -0.1                 0.6                -1.2
#      EPSILON_DOT_0  IFAIL_SH  IFAIL_SO                                    DADV               IXFEM

/PROP/SOLID/100001
Foam
#   Isolid    Ismstr      Iale     Icpre  Itetra10     Inpts   Itetra4    Iframe                  Dn
        18         0                  -1                   0         0        -1                 0.0
#                 qa                  qb                   h              Lambda                  Mu
                 1.1                0.05                 0.1
#         deltaT_min            Vdef_min            Vdef_max             ASP_max             COL_min
                 0.0                 0.0
"""

C19_DECK = (
    "#RADIOSS STARTER\n"
    "/BEGIN\n"
    "c19mat\n"
    "      2024         0\n"
    "                  kg                  mm                  ms\n"
    "                  kg                  mm                  ms\n"
    "/NODE\n"
    "         1                 0.0                 0.0                 0.0\n"
    "         2                10.0                 0.0                 0.0\n"
    "         3                10.0                10.0                 0.0\n"
    "         4                 0.0                10.0                 0.0\n"
    "         5                 0.0                 0.0                10.0\n"
    "         6                10.0                 0.0                10.0\n"
    "         7                10.0                10.0                10.0\n"
    "         8                 0.0                10.0                10.0\n"
    "/BRICK/1\n"
    "         1         1         2         3         4         5         6"
    "         7         8\n"
    "/PART/1\n"
    "part\n"
    "    100001         1\n"
    + C19_MAT
    + "/FUNCT/5\n"
    "curve2 MATLAW36\n"
    "#                  X                   Y\n"
    + C19_CURVE
    + "/END\n"
)

FS = 1.0e-3                      # the deck's Fscale_1
E, NU = 221.0, 0.3               # GPa
G = E / (2.0 * (1.0 + NU))       # = 85.0 exactly
K = E / (3.0 * (1.0 - 2.0 * NU))
X0, Y0 = 0.0, 389.674            # first curve point (MPa)
X1, Y1 = 0.00326906, 400.23      # second curve point
H1 = (Y1 - Y0) / (X1 - X0) * FS  # scaled first-segment hardening slope


def _table():
    rows = [ln.split() for ln in C19_CURVE.strip().splitlines()]
    x = np.array([float(a) for a, _ in rows])
    y = np.array([float(b) for _, b in rows])
    return x, y


def _build(tmp_path):
    f = tmp_path / "C19_0000.rad"
    f.write_text(C19_DECK)
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


# ===========================================================================
# 1. Reader + resolve: Fscale_i lands in params and scales the curve arrays
# ===========================================================================

def test_c19_fscale_parsed_and_applied(tmp_path):
    """The real-dialect Fscale_1 = 1e-3 must be stored (params['yfac'])
    and baked into the resolved ordinates AND slopes — value and
    derivative, exactly what sigeps36.F's YFAC multiplies."""
    model, log = _build(tmp_path)
    mat = model.materials[1]
    assert mat.law == 36
    assert mat.params["yfac"] == [FS]
    tx, ty = _table()
    np.testing.assert_allclose(mat.params["curve_x"][0], tx, rtol=1e-13)
    np.testing.assert_allclose(mat.params["curve_y"][0], ty * FS,
                               rtol=1e-13)
    np.testing.assert_allclose(mat.params["curve_s"][0],
                               np.diff(ty * FS) / np.diff(tx), rtol=1e-13)
    # the fix must not re-introduce the old warning
    assert not any("unscaled" in w for w in log.warnings)


# ===========================================================================
# 2. Kernel vs hand computation on the deck's actual first segment
# ===========================================================================

def test_c19_single_step_return_hand_computed(tmp_path):
    """One uniaxial-strain increment d = 2.5e-3 from virgin state.

    Hand algebra (all from the deck's numbers):
        q_tr    = 2 G d                     (trial von Mises)
        p       = K d                       (trial pressure)
        sy0     = 389.674e-3                (scaled first ordinate)
        dl      = (q_tr - sy0) / (3G + H1)  (consistency on segment 1 —
                                             linear, so exact)
        sy_new  = sy0 + H1 dl
        scale   = sy_new / q_tr
    landing INSIDE segment 1 (dl < 0.00326906), so the whole return is
    one closed-form expression — the strongest possible pin."""
    model, _ = _build(tmp_path)
    mat = model.materials[1]

    d = 2.5e-3
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    deps = np.zeros((1, 6))
    deps[0, 0] = d
    law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-3)

    q_tr = 2.0 * G * d
    sy0 = Y0 * FS
    assert q_tr > sy0                      # the step must actually yield
    dl = (q_tr - sy0) / (3.0 * G + H1)
    assert dl < X1                         # stays inside segment 1
    sy_new = sy0 + H1 * dl
    scale = sy_new / q_tr

    assert epsp[0] == pytest.approx(dl, rel=1e-10)
    p = K * d
    assert sig[0, 0] == pytest.approx(scale * 4.0 * G * d / 3.0 + p,
                                      rel=1e-10)
    assert sig[0, 1] == pytest.approx(-scale * 2.0 * G * d / 3.0 + p,
                                      rel=1e-10)
    assert sig[0, 2] == pytest.approx(sig[0, 1], rel=1e-12)
    # von Mises sits exactly on the scaled curve at the reached strain
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm == pytest.approx(sy_new, rel=1e-10)
    # trace preserved by the deviatoric return (pressure untouched)
    assert (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0 == pytest.approx(
        p, rel=1e-10)


def test_c19_multi_segment_walk_tracks_scaled_curve(tmp_path):
    """600 deviatoric pushes walk the return across ~38 table segments;
    the von Mises stress must sit on the SCALED curve at the reached
    plastic strain to 1e-9 — every segment's scaled slope checked."""
    model, _ = _build(tmp_path)
    mat = model.materials[1]
    tx, ty = _table()

    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    a = 2e-4
    deps = np.zeros((1, 6))
    deps[0, :3] = [a, -a / 2.0, -a / 2.0]
    for _ in range(600):
        law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-3)
    ep = epsp[0]
    assert ep > 0.1                        # deep into the table
    k = np.searchsorted(tx, ep) - 1
    assert 30 < k < len(tx) - 1            # inside, tens of segments in
    sy_exact = (ty[k] + (ty[k + 1] - ty[k]) / (tx[k + 1] - tx[k])
                * (ep - tx[k])) * FS
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm == pytest.approx(sy_exact, rel=1e-9)


# ===========================================================================
# 3. Per-curve Fscale in a strain-rate family
# ===========================================================================

def test_fscale_per_curve_in_rate_family(tmp_path):
    """Two flat 0.4 curves with Fscale 0.5 / 2.0 at rates 0 / 10: the
    yield at rate 5 must blend the SCALED values, (0.2 + 0.8)/2 = 0.5 —
    sigeps36.F applies YFAC per curve BEFORE the rate interpolation."""
    deck = (
        "/MAT/LAW36/7\n"
        "two curves\n"
        "        7.800000E-06\n"
        "               210.0                 0.3                   0\n"
        "         2         0\n"
        "         0                 1.0\n"
        "       201       202\n"
        "                 0.5                 2.0\n"
        "                   0                10.0\n"
        "/FUNCT/201\nflat A\n0.0 0.4\n1.0 0.4\n"
        "/FUNCT/202\nflat B\n0.0 0.4\n1.0 0.4\n"
    )
    f = tmp_path / "RATE_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    resolve_materials(model, log)
    assert not log.errors, log.errors
    mat = model.materials[7]
    assert mat.params["yfac"] == [0.5, 2.0]
    assert mat.params["curve_y"][0][0] == pytest.approx(0.2, rel=1e-13)
    assert mat.params["curve_y"][1][0] == pytest.approx(0.8, rel=1e-13)
    sy, H = law36_tabulated._yield_stress(
        mat, np.zeros(1), np.array([5.0]))
    assert sy[0] == pytest.approx(0.5, rel=1e-12)
    assert H[0] == pytest.approx(0.0, abs=1e-15)


# ===========================================================================
# 4. /FAIL/JOHNSON sequencing: non-positive eps_f freezes the damage
# ===========================================================================

def _jc(params=None):
    p = {"D1": -0.1, "D2": 0.6, "D3": -1.2, "D4": 0.0, "D5": 0.0,
         "eps_dot_0": 1.0}
    if params:
        p.update(params)
    return FailureModel(type="JOHNSON", ifail_sh=1, params=p)


def test_johnson_negative_epsf_freezes_solid_damage():
    """fail_johnson.F: EPSF = MAX(EPSF, EPSF_MIN=0); damage accumulates
    ONLY where EPSF > 0.  With the c19 constants eps_f < 0 beyond
    sigma* = ln(6)/1.2 = 1.4931 — a hydrostatic-tension point must
    accumulate NOTHING (the old port instant-deleted it)."""
    fail = _jc()
    dama = np.zeros(1)
    deps = np.zeros((1, 6))
    sig_hydro = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])  # triax -> +inf
    for _ in range(5):
        broken = johnson.solid_step(fail, sig_hydro, np.array([0.05]),
                                    deps, 1e-3, dama)
    assert dama[0] == 0.0
    assert not broken[0]

    # uniaxial tension (triax = 1/3) accumulates exactly dpla/eps_f
    dama = np.zeros(1)
    sig_uni = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    eps_f = -0.1 + 0.6 * np.exp(-1.2 / 3.0)
    broken = johnson.solid_step(fail, sig_uni, np.array([0.05]),
                                deps, 1e-3, dama)
    assert dama[0] == pytest.approx(0.05 / eps_f, rel=1e-12)
    assert not broken[0]
    # keep pushing: damage caps at 1 (DFMAX = MIN(ONE, DFMAX)) and breaks
    for _ in range(20):
        broken = johnson.solid_step(fail, sig_uni, np.array([0.05]),
                                    deps, 1e-3, dama)
    assert dama[0] == 1.0
    assert broken[0]


def test_johnson_negative_epsf_freezes_shell_damage():
    """Same contract in the shell step (fail_johnson_c.F is identical):
    constants that go negative at equibiaxial tension freeze the layer
    instead of failing it."""
    fail = _jc({"D1": -0.5, "D2": 0.3, "D3": -1.0})
    dama = np.zeros(1)
    deps = np.zeros((1, 3))
    sig_biax = np.array([[1.0, 1.0, 0.0]])     # triax 2/3, eps_f = -0.346
    broken = johnson.shell_step(fail, sig_biax, np.array([0.1]),
                                deps, 1e-3, dama)
    assert dama[0] == 0.0 and not broken[0]
    # uniaxial (triax 1/3): eps_f = -0.5 + 0.3 e^{-1/3} = 0.2149 > 0 wait
    # -0.5 + 0.3*0.7165 = -0.285 < 0 — still frozen; use tension-free
    # shear (triax 0): eps_f = -0.5 + 0.3 = -0.2 < 0 too.  These
    # constants freeze everywhere reachable in plane stress — assert so.
    for s in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]):
        d = np.zeros(1)
        broken = johnson.shell_step(fail, np.array([s]), np.array([0.1]),
                                    np.zeros((1, 3)), 1e-3, d)
        assert d[0] == 0.0 and not broken[0]


def test_johnson_epsf_min_floor():
    """A positive EPSF_MIN (params['eps_f_min'], read where decks carry
    it) floors the failure strain: damage then accumulates at the floor
    rate instead of freezing."""
    fail = _jc({"eps_f_min": 0.25})
    dama = np.zeros(1)
    sig_hydro = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    johnson.solid_step(fail, sig_hydro, np.array([0.05]),
                       np.zeros((1, 6)), 1e-3, dama)
    assert dama[0] == pytest.approx(0.05 / 0.25, rel=1e-12)


# ===========================================================================
# 5. Wiring: the hexa kernel on the c19 material yields at the scaled
#    curve and the JC damage plateaus (does NOT delete) past the root
# ===========================================================================

def test_c19_hexa_kernel_yields_and_damage_plateaus(tmp_path):
    """Uniaxial-strain tension on the c19 cube — the geometry is MOVED
    each step (x += v dt) so the kernel's density really drops and the
    M40 total pressure P = K (1 - 1/J) develops: (a) the element yields
    at the SCALED curve level (0.39 GPa, not 390); (b) the JC damage
    grows only inside the eps_f > 0 triaxiality window and FREEZES once
    the pressure drives sigma* past 1.4931 — the element must survive
    with 0 < D < 1 (the old port deleted it there; Fortran keeps it —
    same mechanism that kept c19's confined/hydrostatic elements
    alive)."""
    model, _ = _build(tmp_path)
    g = model.bricks
    mat = g.state["slices"][0][1]
    assert mat.fail is not None and mat.fail.type == "JOHNSON"

    rate, dt = 5e-2, 1e-3        # dexx = rate*dt = 5e-5 per step
    x = model.x.copy()
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)
    v = np.zeros_like(x)
    dama_hist = []
    for _ in range(400):
        v[:, 0] = x[:, 0] * rate        # uniform stretching in x
        solid_hexa8.forces(g, x, v, model.vr, dt, fint, mint)
        x[:, 0] += v[:, 0] * dt         # advance the geometry
        dama_hist.append(g.state["dama"][0])
    ep = g.state["epsp"][0]
    assert ep > 5e-3                        # yielded far past onset
    s = g.state["sig"][0]
    vm = np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2
                        + (s[2] - s[0]) ** 2))
    tx, ty = _table()
    k = np.searchsorted(tx, ep) - 1
    sy_exact = (ty[k] + (ty[k + 1] - ty[k]) / (tx[k + 1] - tx[k])
                * (ep - tx[k])) * FS
    assert vm == pytest.approx(sy_exact, rel=1e-6)
    # the M40 total pressure has developed (P = K (1 - 1/J) > 0) and
    # the triaxiality is far beyond the eps_f root by now
    p = (s[0] + s[1] + s[2]) / 3.0
    assert p > 0.0
    assert p / vm > 1.4931
    # damage engaged in the early window, then froze; element alive
    assert 0.0 < g.state["dama"][0] < 1.0
    assert g.state["off"][0] == 1.0
    assert dama_hist[-1] == dama_hist[200]  # frozen, not creeping


def test_slashless_fail_header_recovered(tmp_path):
    """The c23 TETRA deck's MAT_PROP.inc writes ``FAIL/JOHNSON/1``
    WITHOUT the leading slash; the real Starter still reads the /FAIL
    block (the bundled reference listing prints the Johnson-Cook damage
    parameters).  The port lexer must recover it the same way — before
    M40 the whole block was swallowed as stray /MAT data cards and c23
    ran with NO failure model (zero deletions vs Fortran's eleven)."""
    deck = C19_DECK.replace("/FAIL/JOHNSON/1", "FAIL/JOHNSON/1")
    assert "\nFAIL/JOHNSON/1\n" in deck          # the mutation took
    f = tmp_path / "SL_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    from pyradioss.starter.initialization import resolve_materials as _rm
    _rm(model, log)
    assert not log.errors, log.errors
    mat = model.materials[1]
    assert mat.fail is not None and mat.fail.type == "JOHNSON"
    assert mat.fail.params["D1"] == pytest.approx(-0.1)
    # a title containing a slash must NOT be mistaken for a header:
    # the recovery demands the strict KEYWORD/KEY2/id shape at column 1
    from pyradioss.input.deck_reader import _SLASHLESS_HEADER
    assert not _SLASHLESS_HEADER.match("FAIL SAFE/2 bracket assembly")
    assert not _SLASHLESS_HEADER.match("  FAIL/JOHNSON/1")   # indented
    assert not _SLASHLESS_HEADER.match("FAIL/JOHNSON/abc")   # no int id
    assert _SLASHLESS_HEADER.match("FAIL/JOHNSON/1")


def test_total_pressure_matches_reference_form():
    """M40 pressure split (sigeps36.F IEOS==0): with the kernel density
    in ``extra`` the trace must be 3 K (1 - rho/rho0) — saturating at K
    in expansion — NOT the hypoelastic 3 K ln J; without a density the
    documented fallback reproduces the old trace increment.  Pure
    volumetric stretch: no deviator, no yield — pressure only."""
    from pyradioss.model.entities import Material
    tx, ty = _table()
    mat = Material(id=1, law=36, rho0=7.8e-6, params={
        "E": E, "nu": NU, "eps_p_max": 1e30, "funct_ids": [5],
        "curve_x": [tx], "curve_y": [ty * FS],
        "curve_s": [np.diff(ty * FS) / np.diff(tx)],
        "rates": np.array([0.0])})

    d = 0.4                       # one huge volumetric step, J = e^1.2
    deps = np.tile([d, d, d, 0.0, 0.0, 0.0], (1, 1))
    J = np.exp(3.0 * d)
    rho = np.array([mat.rho0 / J])

    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-3,
                                 extra={"rho": rho})
    p_total = K * (1.0 - 1.0 / J)            # bounded by K
    assert sig[0, 0] == pytest.approx(p_total, rel=1e-12)
    assert sig[0, 1] == pytest.approx(p_total, rel=1e-12)
    assert epsp[0] == 0.0                    # no deviatoric flow

    sig2 = np.zeros((1, 6))
    law36_tabulated.solid_update(mat, sig2, deps, np.zeros(1), dt=1e-3)
    assert sig2[0, 0] == pytest.approx(K * 3.0 * d, rel=1e-12)  # fallback
    # the two forms differ by design at large J: ln J vs 1 - 1/J
    assert sig2[0, 0] / sig[0, 0] == pytest.approx(
        3.0 * d / (1.0 - 1.0 / J), rel=1e-12)
