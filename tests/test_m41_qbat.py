"""M41 — the QBAT shell (/PROP/SHELL Ishell=12, Batoz-Dhatt fully
integrated 4-node quadrilateral, engine IHBE=11).

Fortran origin of the kernel under test:
``engine/source/elements/shell/coqueba/`` — cbaforc3.F (driver),
cbacoor.F (frame/geometry/condensed dt length), cbadef.F (assumed-strain
[B] operators), cbastra3.F (strain increments), cbavisc.F (dn numerical
damping), cbaener.F (assumed-shear energy corrections), cbafori.F (force
assembly), cbaproj.F (local->global + warped rigid projection).

Verification layers (the M41 plan):
  A. starter dispatch: Ishell=12 parts split out of the BT group into
     ``model.shells_qbat`` (elements.SHELL_ISHELL_GROUPS), everything
     else untouched;
  B. single-element PATCH TESTS against closed forms at rel 1e-8 —
     membrane stretch N = C t eps, pure bending
     M = E t^3 kappa / (12 (1 - nu^2)), twist
     Mxy = E t^3 c / (12 (1 + nu)) — run INVISCID (amu = 0): the closed
     forms carry no cbavisc.F dn damping (its work is checked separately
     in the HE-contract and ringdown tests);
  C. rigid-rotation objectivity: residual strain < 1e-12 at the
     realistic per-step rotation w*dt = 1e-6, plus the quadratic
     (w*dt)^2 convergence that proves the residual is the documented
     second-order corotational term (cbacoor.F's explicit spin
     corrections are first-order exact, like upstream), flat AND warped;
  D. the HE == 0 CONTRACT: full integration books NO stiffness-hourglass
     energy — ``ehour`` carries only cbavisc.F's dn work (upstream books
     it to the same PARTSAV(8) slot), identically 0.0 at dn = 0;
  E. dt claim: the cbacoor.F lines 1080-1099 condensed characteristic
     length (FACDT = 4/3) times cndt3.F's (sqrt(1+dn^2)-dn)/ssp,
     compared against an independent transliteration;
  F. material-plumbing reuse: a LAW2 elastoplastic membrane step gives
     layer-for-layer the SAME stress/epsp as the established BT4 kernel
     (identical deps -> identical materials.shell_update path);
  G. the analytic cantilever: explicit step-load ringdown of a 10x1
     strip vs Euler-Bernoulli tip deflection delta = F L^3 / (3 E I) and
     first-mode period, through the FULL starter+engine pipeline, with
     the energy balance closed and HE ~ 0 (where the pre-M41 BT-fallback
     stored bending energy as hourglass).
"""

import contextlib
import io
import re

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import KERNELS, SHELL_ISHELL_GROUPS, shell_qbat
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_materials,
                                              resolve_node_groups,
                                              resolve_surfaces)
from pyradioss.starter.starter import run_starter

# material constants of the standard steel fragment (Mg / mm / ms units)
E_STEEL, NU_STEEL, RHO_STEEL = 210.0, 0.3, 7.8e-6


def _build(deck: str, tmp_path):
    f = tmp_path / "K_0000.rad"
    f.write_text(deck)
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


def _plate_deck(nu=NU_STEEL, t=0.1, nip=3, warp=0.0, ishell=12):
    """One unit-square shell, node 1 at the origin; ``warp`` lifts node 3
    out of plane (the cbacoor.F warped branch)."""
    return (
        "/BEGIN\nm41 qbat unit shell\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n"
        f"3 1 1 {warp}\n4 0 1 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nplate\n1 1\n"
        f"/MAT/LAW1/1\nsteel\n{RHO_STEEL}\n{E_STEEL} {nu}\n"
        "/PROP/SHELL/1\nplate prop\n12 0 0 0\n0 0 0 0 0\n"
        f"{nip} 0 {t}\n/END\n"
    )


def _qbat_plate(tmp_path, inviscid=True, **kw):
    """Built + initialized single-element QBAT group. ``inviscid`` zeroes
    the cbavisc.F dn damping so closed-form (inviscid) patch answers
    apply exactly; the dn work itself is verified by the HE-contract and
    ringdown tests."""
    model, _ = _build(_plate_deck(**kw), tmp_path)
    assert model.shells is None          # everything routed to QBAT
    g = model.shells_qbat
    assert g is not None and g.n == 1
    if inviscid:
        g.state["amu"][:] = 0.0
    return g, model


def _one_call(g, model, vfield, dt=1.0):
    """A single forces() call on velocity field ``vfield(x) -> (v, vr)``.
    dt = 1 makes strain = rate numerically (patch convention)."""
    fint = np.zeros((model.numnod, 3))
    mint = np.zeros((model.numnod, 3))
    v = np.zeros((model.numnod, 3))
    vr = np.zeros((model.numnod, 3))
    for i in range(model.numnod):
        vi, ri = vfield(model.x[i])
        v[i], vr[i] = vi, ri
    dte = shell_qbat.forces(g, model.x, v, vr, dt, fint, mint)
    return fint, mint, dte


def _moment_resultant(model, fint, mint, about):
    """Global moment resultant sum(r x f) + sum(m) about ``about``."""
    r = model.x - about[None, :]
    return np.cross(r, fint).sum(axis=0) + mint.sum(axis=0)


# ============================================================================
# A. starter dispatch
# ============================================================================

def test_ishell12_routes_to_qbat_kernel(tmp_path):
    """Ishell=12 is in SHELL_ISHELL_GROUPS -> shells_qbat, whose KERNELS
    entry is the shell_qbat module (starter/engine wiring)."""
    assert SHELL_ISHELL_GROUPS[12] == "shells_qbat"
    assert KERNELS["shells_qbat"] is shell_qbat
    model, _ = _build(_plate_deck(), tmp_path)
    assert model.shells is None
    assert model.shells_qbat is not None and model.shells_qbat.n == 1
    # the group iterates (engine force loop sees it)
    names = [n for n, _ in model.element_groups()]
    assert names == ["shells_qbat"]


def test_mixed_deck_splits_bt_and_qbat(tmp_path):
    """A deck mixing Ishell=1 (BT) and Ishell=12 (QBAT) parts: each part
    routes WHOLLY to its kernel; the BT part is byte-identically the
    generic shells group (untouched dispatch contract)."""
    deck = (
        "/BEGIN\nmixed shells\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 2 0 0\n6 2 1 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/SHELL/2\n2 2 5 6 3\n"
        "/PART/1\nbt plate\n1 1\n"
        "/PART/2\nqbat plate\n2 1\n"
        f"/MAT/LAW1/1\nsteel\n{RHO_STEEL}\n{E_STEEL} {NU_STEEL}\n"
        "/PROP/SHELL/1\nbt prop\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n"
        "/PROP/SHELL/2\nqbat prop\n12 0 0 0\n0 0 0 0 0\n3 0 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    assert model.shells is not None and model.shells.n == 1
    assert model.shells_qbat is not None and model.shells_qbat.n == 1
    assert model.shells.ids.tolist() == [1]
    assert model.shells_qbat.ids.tolist() == [2]
    # slices carry the right properties
    assert model.shells.state["slices"][0][2].params["ishell"] == 1
    assert model.shells_qbat.state["slices"][0][2].params["ishell"] == 12


def test_qbat_mass_and_inertia_lumping(tmp_path):
    """m = rho t A, per-node m/4; I = m/4 (t^2+A)/12 — the cinmas.F
    FAC=TWELVE lumping of the IHBE>=11 family (validated against the c04
    /RBODY gather in M40)."""
    g, model = _qbat_plate(tmp_path)
    t, A = 0.1, 1.0
    m = RHO_STEEL * t * A
    assert g.state["mass"][0] == pytest.approx(m, rel=1e-12)
    assert model.mass[:4] == pytest.approx(m / 4.0, rel=1e-12)
    assert model.inertia[:4] == pytest.approx(
        m / 4.0 * (t * t + A) / 12.0, rel=1e-12)


# ============================================================================
# B. single-element patch tests (closed forms, rel 1e-8)
# ============================================================================

def test_patch_membrane_stretch(tmp_path):
    """vx = eps * x: N = C t eps with C = E/(1-nu^2); edge nodes carry
    -+N/2, sig uniform across GPs and layers with the plane-stress nu
    coupling, eint = N eps A / 2 — all at rel 1e-8."""
    g, model = _qbat_plate(tmp_path)
    t, eps = 0.1, 1.0e-3
    C = E_STEEL / (1.0 - NU_STEEL ** 2)
    fint, mint, _ = _one_call(g, model, lambda x: ((eps * x[0], 0, 0),
                                                   (0, 0, 0)))
    N1 = C * eps * t
    assert fint[:, 0] == pytest.approx([N1 / 2, -N1 / 2, -N1 / 2, N1 / 2],
                                       rel=1e-8)
    # every GP x layer identical: sxx = C eps, syy = nu C eps, sxy = 0
    sig = g.state["sig"][0]
    assert sig[:, 0] == pytest.approx(C * eps, rel=1e-8)
    assert sig[:, 1] == pytest.approx(NU_STEEL * C * eps, rel=1e-8)
    assert np.abs(sig[:, 2]).max() < 1e-15
    assert g.state["eint"][0] == pytest.approx(0.5 * N1 * eps, rel=1e-8)
    # free body: force AND moment resultants vanish
    assert np.abs(fint.sum(axis=0)).max() < 1e-15
    assert np.abs(_moment_resultant(model, fint, mint,
                                    model.x.mean(axis=0))).max() < 1e-15


def test_patch_pure_bending_moment(tmp_path):
    """THE closed form of the task: M = E t^3 kappa / (12 (1 - nu^2)).

    Kirchhoff-consistent field (theta_y = kappa x, w = -kappa x^2 / 2) so
    the 4 mid-edge assumed-shear samples (cbadef.F BC block) vanish
    DISCRETELY: edge nodes carry mint_y = -+M/2 with ZERO transverse
    force. kappa = 1e-5 keeps the O(kappa^2/8) membrane strain of the
    upstream explicit spin correction (cbacoor.F lines 415-437) at
    relative 2e-9 in eint — inside the 1e-8 budget (it is upstream
    physics, not porting error)."""
    g, model = _qbat_plate(tmp_path)
    t, kap = 0.1, 1.0e-5
    M = E_STEEL * t ** 3 * kap / (12.0 * (1.0 - NU_STEEL ** 2))
    fint, mint, _ = _one_call(
        g, model, lambda x: ((0, 0, -kap * x[0] ** 2 / 2.0),
                             (0, kap * x[0], 0)))
    assert mint[:, 1] == pytest.approx([M / 2, -M / 2, -M / 2, M / 2],
                                       rel=1e-8)
    # pure moment: no transverse shear force at all (assumed field exact)
    assert np.abs(fint[:, 2]).max() < 1e-16
    assert np.abs(g.state["qshear"]).max() < 1e-16
    assert g.state["eint"][0] == pytest.approx(0.5 * M * kap, rel=1e-8)
    # layer stress: the z-ANTISYMMETRIC part is the bending closed form
    # C kappa z at rel 1e-8; the z-SYMMETRIC part is EXACTLY the
    # documented spin-correction membrane strain -kappa^2/8 (cbacoor.F
    # lines 415-437 — upstream physics, quantitatively pinned here)
    C = E_STEEL / (1.0 - NU_STEEL ** 2)
    zrel, _ = g.state["zw"][0]
    nipm = g.state["nip_max"]
    for ng in range(4):
        s_lo = g.state["sig"][0, ng * nipm + 0, 0]       # z = -z_out
        s_hi = g.state["sig"][0, ng * nipm + len(zrel) - 1, 0]
        assert 0.5 * (s_hi - s_lo) == pytest.approx(
            C * kap * abs(zrel[-1]) * t, rel=1e-8)
        assert 0.5 * (s_hi + s_lo) == pytest.approx(
            -C * kap ** 2 / 8.0, rel=1e-6)
    assert np.abs(_moment_resultant(model, fint, mint,
                                    model.x.mean(axis=0))).max() < 1e-16


def test_patch_twist(tmp_path):
    """w = -c x y (theta_x = c x, theta_y = -c y): pure torsion,
    kxy rate = 2c, Mxy = E t^3 c / (12 (1 + nu)) = G t^3 (2c) / 12; the
    corner moment pattern of the smoke closed form, eint = Mxy (2c) A/2."""
    g, model = _qbat_plate(tmp_path)
    t, c = 0.1, 1.0e-5
    Mxy = E_STEEL * t ** 3 * c / (12.0 * (1.0 + NU_STEEL))
    fint, mint, _ = _one_call(
        g, model, lambda x: ((0, 0, -c * x[0] * x[1]),
                             (-c * x[0], c * x[1], 0)))
    expect = Mxy / 2.0
    assert mint[:, 0] == pytest.approx([-expect, expect, expect, -expect],
                                       rel=1e-8)
    assert mint[:, 1] == pytest.approx([expect, expect, -expect, -expect],
                                       rel=1e-8)
    assert g.state["eint"][0] == pytest.approx(Mxy * c, rel=1e-8)
    assert np.abs(_moment_resultant(model, fint, mint,
                                    model.x.mean(axis=0))).max() < 1e-16


def test_patch_warped_self_equilibrium(tmp_path):
    """Warped element (node 3 lifted 0.05): the cbaproj.F free-rigid-mode
    projection makes the global force system EXACTLY self-equilibrated —
    zero force and zero moment resultant to round-off — and the membrane
    answer stays within the flat closed form to the warp order."""
    g, model = _qbat_plate(tmp_path, warp=0.05)
    assert len(g.state) and g.n == 1
    t, eps = 0.1, 1.0e-3
    C = E_STEEL / (1.0 - NU_STEEL ** 2)
    fint, mint, _ = _one_call(g, model, lambda x: ((eps * x[0], 0, 0),
                                                   (0, 0, 0)))
    N1 = C * eps * t
    assert np.abs(fint.sum(axis=0)).max() < 1e-15
    assert np.abs(_moment_resultant(model, fint, mint,
                                    model.x.mean(axis=0))).max() < 1e-15
    # warp = 5e-2 out of plane: membrane force within ~warp^2 of flat
    assert fint[:, 0] == pytest.approx([N1 / 2, -N1 / 2, -N1 / 2, N1 / 2],
                                       rel=5e-3)


# ============================================================================
# C. rigid-rotation objectivity
# ============================================================================

def _rigid_residual(tmp_path_factory_dir, warp, wdt, dt=1.0e-3):
    g, model = _qbat_plate(tmp_path_factory_dir, warp=warp)
    w = np.array([0.3, -0.2, 0.4])
    w = w / np.linalg.norm(w) * (wdt / dt)
    x0 = model.x.mean(axis=0)
    fint, mint, _ = _one_call(
        g, model, lambda x: (np.cross(w, x - x0), w), dt=dt)
    return (np.abs(g.state["sig"]).max(), np.abs(g.state["qshear"]).max(),
            g.state["eint"][0], np.abs(fint).max())


@pytest.mark.parametrize("warp", [0.0, 0.05])
def test_rigid_rotation_objectivity(tmp_path, warp):
    """A rigid rotation must produce no stress. The corotational frame +
    the cbacoor.F explicit spin corrections are first-order exact, so the
    per-call residual strain is the upstream O((w dt)^2) term:

      * at the realistic per-step rotation w dt = 1e-6 the residual
        strain |sig|/E is BELOW 1e-12 (the M41 objectivity target);
      * halving-order check: residual(1e-3) / residual(1e-5) = 1e4
        (exactly quadratic — proves it is the documented second-order
        term, not a first-order porting error), flat AND warped."""
    s6, q6, e6, _ = _rigid_residual(tmp_path, warp, 1.0e-6)
    assert s6 / E_STEEL < 1.0e-12
    assert q6 / (0.5 * E_STEEL / (1 + NU_STEEL)) < 1.0e-12
    assert e6 < 1.0e-20
    s3, _, _, _ = _rigid_residual(tmp_path, warp, 1.0e-3)
    s5, _, _, _ = _rigid_residual(tmp_path, warp, 1.0e-5)
    assert s3 / s5 == pytest.approx(1.0e4, rel=0.02)


def test_rigid_translation_exact(tmp_path):
    """Uniform translation: identically zero everything (round-off)."""
    for warp in (0.0, 0.05):
        g, model = _qbat_plate(tmp_path, warp=warp)
        fint, mint, _ = _one_call(
            g, model, lambda x: (np.array([0.3, -0.1, 0.2]), np.zeros(3)))
        assert np.abs(g.state["sig"]).max() < 1e-16
        assert np.abs(fint).max() < 1e-16
        assert np.abs(mint).max() < 1e-16
        assert g.state["eint"][0] == 0.0


# ============================================================================
# D. the HE == 0 contract (full integration: no hourglass)
# ============================================================================

def test_hourglass_energy_identically_zero_inviscid(tmp_path):
    """The M41 point: QBAT has NO hourglass control — with the cbavisc.F
    dn damping off, ``ehour`` is IDENTICALLY 0.0 through a bending step
    that the pre-M41 BT-fallback booked (up to 40% of EW) as hourglass.
    An hourglass-PATTERN field (+w at nodes 1,3 / -w at 2,4) is a REAL
    deformation for the 2x2-integrated element: it must produce INTERNAL
    energy and forces, never HE."""
    g, model = _qbat_plate(tmp_path)
    kap = 1.0e-4
    _one_call(g, model, lambda x: ((0, 0, -kap * x[0] ** 2 / 2),
                                   (0, kap * x[0], 0)))
    assert g.state["ehour"][0] == 0.0
    assert g.state["eint"][0] > 0.0
    # the w-hourglass pattern (the mode BT 1-point cannot see)
    g2, model2 = _qbat_plate(tmp_path)
    sgn = {0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0}
    idx = {tuple(model2.x[i, :2]): i for i in range(4)}
    assert len(idx) == 4

    def hg_field(x):
        # +- w by corner parity
        i = idx[tuple(x[:2])]
        return (0, 0, 1.0e-3 * sgn[i]), (0, 0, 0)

    fint, mint, _ = _one_call(g2, model2, hg_field)
    assert g2.state["ehour"][0] == 0.0
    assert g2.state["eint"][0] > 0.0          # fully integrated: real energy
    assert np.abs(fint[:, 2]).max() > 0.0     # resisted by real forces


def test_viscous_dn_work_books_to_ehour_slot(tmp_path):
    """With the default dn = 1e-3 (cncoef3.F lines 426-431, IHBE=11) the
    cbavisc.F damping work books to ``ehour`` — upstream's PARTSAV(8)
    slot — at EXACTLY the closed form of cbavisc.F's membrane branch:
    per GP  FOR1 += 1.414 dn rho ssp sqrt(CDET) * (exx + nu eyy), work =
    FOR1 * exx * CDET * t * dt (the run-level 'invisible next to IE'
    property is asserted on the cantilever ringdown)."""
    g, model = _qbat_plate(tmp_path, inviscid=False)
    dn = 1.0e-3
    assert g.state["amu"][0] == pytest.approx(dn)
    eps, dt = 1.0e-3, 1.0e-4
    _one_call(g, model, lambda x: ((eps * x[0], 0, 0), (0, 0, 0)), dt=dt)
    t, A = 0.1, 1.0
    ssp = np.sqrt(E_STEEL / (RHO_STEEL * (1.0 - NU_STEEL ** 2)))
    visc = 1.414 * dn * RHO_STEEL * ssp * np.sqrt(A / 4.0)
    assert g.state["ehour"][0] == pytest.approx(
        visc * eps ** 2 * t * dt * A, rel=1e-10)
    assert g.state["ehour"][0] > 0.0


# ============================================================================
# E. dt claim (cbacoor.F condensed length, FACDT = 4/3 + cndt3.F)
# ============================================================================

def _lc_ref(xy):
    """Independent transliteration of cbacoor.F lines 1080-1099 for a
    FLAT element given its local corner coordinates (4, 2)."""
    c = xy - xy.mean(axis=0)
    rx = c[1, 0] + c[2, 0] - c[3, 0] - c[0, 0]
    ry = c[1, 1] + c[2, 1] - c[3, 1] - c[0, 1]
    sx = -c[1, 0] + c[2, 0] + c[3, 0] - c[0, 0]
    sy = -c[1, 1] + c[2, 1] + c[3, 1] - c[0, 1]
    c1, c2 = np.hypot(rx, ry), np.hypot(sx, sy)
    cmax, cmin = max(c1, c2), min(c1, c2)
    fac1 = min(0.5, 0.25 * (cmax / cmin - 1.0)) + 1.0
    area = 0.5 * abs((xy[2, 0] - xy[0, 0]) * (xy[3, 1] - xy[1, 1])
                     - (xy[2, 1] - xy[0, 1]) * (xy[3, 0] - xy[1, 0]))
    fac2 = 4.0 * area / (c1 * c2)
    fac2 = 3.413 * max(0.0, fac2 - 0.7071)
    fac2 = 0.78 + 0.22 * fac2 ** 3
    faci = 2.0 * fac1 * fac2
    x13, y13 = 0.5 * (c[0] - c[2])
    x24, y24 = 0.5 * (c[1] - c[3])
    ll = max(x13 ** 2 + y13 ** 2, x24 ** 2 + y24 ** 2)
    lm = max(abs(c[1, 0] * c[3, 1] - c[1, 1] * c[3, 0]),
             abs(c[0, 0] * c[2, 1] - c[0, 1] * c[2, 0]))
    s1 = np.sqrt(faci * (4.0 / 3.0 + lm / area) * ll)
    return area / s1


@pytest.mark.parametrize("shape", ["square", "rect", "trapez"])
def test_dt_claim_condensed_length(tmp_path, shape):
    """dt_e = LC * (sqrt(1+dn^2) - dn) / ssp with LC the FACDT=4/3
    condensed length recomputed from current geometry (the M40 claim,
    now computed by the owning kernel) — vs an independent
    transliteration, three planform shapes."""
    corners = {
        "square": [(0, 0), (1, 0), (1, 1), (0, 1)],
        "rect": [(0, 0), (2, 0), (2, 1), (0, 1)],
        "trapez": [(0, 0), (1.5, 0), (1.1, 0.9), (0.2, 1.0)],
    }[shape]
    nodes = "".join(f"{i+1} {x} {y} 0\n" for i, (x, y) in enumerate(corners))
    deck = (
        "/BEGIN\ndt shape\n/NODE\n" + nodes +
        "/SHELL/1\n1 1 2 3 4\n/PART/1\nplate\n1 1\n"
        f"/MAT/LAW1/1\nsteel\n{RHO_STEEL}\n{E_STEEL} {NU_STEEL}\n"
        "/PROP/SHELL/1\nprop\n12 0 0 0\n0 0 0 0 0\n3 0 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    g = model.shells_qbat
    fint = np.zeros((model.numnod, 3))
    mint = np.zeros((model.numnod, 3))
    dte = shell_qbat.forces(g, model.x, model.v, model.vr, 0.0, fint, mint)
    dn = 1.0e-3                            # cncoef3.F default at IHBE=11
    ssp = np.sqrt(E_STEEL / (RHO_STEEL * (1.0 - NU_STEEL ** 2)))
    lc = _lc_ref(np.array(corners, dtype=float))
    assert dte[0] == pytest.approx(
        lc * (np.sqrt(1.0 + dn * dn) - dn) / ssp, rel=1e-10)


# ============================================================================
# F. material-plumbing reuse (LAW2 layer parity with the BT4 kernel)
# ============================================================================

def test_law2_layers_match_bt4_membrane(tmp_path):
    """An elastoplastic membrane step: QBAT's per-GP layers see the SAME
    deps as BT4's 1-point layers (uniform stretch), so the reused
    materials.shell_update must give layer-for-layer identical stress and
    plastic strain — the plumbing-reuse contract."""
    from pyradioss.elements import shell_bt4
    law2 = (f"/MAT/LAW2/1\nsteel\n{RHO_STEEL}\n{E_STEEL} {NU_STEEL}\n"
            "0.4 0.5 0.5\n")

    def deck(ishell, hcard):
        return ("/BEGIN\ntwin\n"
                "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
                "/SHELL/1\n1 1 2 3 4\n/PART/1\nplate\n1 1\n" + law2 +
                f"/PROP/SHELL/1\nprop\n{ishell} 0 0 0\n{hcard}\n"
                "5 0 0.1\n/END\n")

    mq, _ = _build(deck(12, "0 0 0 0 0"), tmp_path)
    (tmp_path / "K_0000.rad").unlink()
    mb, _ = _build(deck(1, "0.01 0.01 0.01 0 0"), tmp_path)
    gq, gb = mq.shells_qbat, mb.shells
    gq.state["amu"][:] = 0.0                 # inviscid for exact parity
    eps = 5.0e-3                             # well past yield 0.4/210
    for m, g, kern in ((mq, gq, shell_qbat), (mb, gb, shell_bt4)):
        v = np.zeros((m.numnod, 3))
        v[:, 0] = eps * m.x[:, 0]
        kern.forces(g, m.x, v, m.vr, 1.0, np.zeros((m.numnod, 3)),
                    np.zeros((m.numnod, 3)))
    nipm = gq.state["nip_max"]
    assert nipm == 5
    assert gb.state["epsp"][0, 0] > 0.0      # genuinely plastic step
    for ng in range(4):                      # every GP == the BT layer stack
        sl = slice(ng * nipm, ng * nipm + 5)
        np.testing.assert_allclose(gq.state["sig"][0, sl], gb.state["sig"][0],
                                   rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(gq.state["epsp"][0, sl],
                                   gb.state["epsp"][0], rtol=1e-12)


def test_mixed_nip_all_layer_deletion(tmp_path):
    """GP-major deletion regression: a group mixing nip=5 and nip=3 parts
    under /FAIL/JOHNSON Ifail_sh=2 (ALL layers must break). The layer
    axis is k = GP*nip_max + layer, so the nip=3 slice carries PAD slots
    that never break — counting 'the first 4*nip_max columns' (the BT
    contiguous-layout rule) would leave those elements IMMORTAL. The
    QBAT-local rule counts each slice's REAL slots only: both elements
    must delete on the same uniform stretch history, forces drop to
    zero, dt claim goes inert (EP30)."""
    law = (f"/MAT/LAW2/1\nsteel\n{RHO_STEEL}\n{E_STEEL} {NU_STEEL}\n"
           "0.4 0.0 0.0\n"
           "/FAIL/JOHNSON/1\n0.05 0 0 0 0\n1.0 2\n")
    deck = (
        "/BEGIN\nmixed nip fail\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 3 0 0\n6 4 0 0\n7 4 1 0\n8 3 1 0\n"
        "/SHELL/1\n1 1 2 3 4\n/SHELL/2\n2 5 6 7 8\n"
        "/PART/1\nnip5\n1 1\n/PART/2\nnip3\n2 1\n" + law +
        "/PROP/SHELL/1\np5\n12 0 0 0\n0 0 0 0 0\n5 0 0.1\n"
        "/PROP/SHELL/2\np3\n12 0 0 0\n0 0 0 0 0\n3 0 0.1\n/END\n"
    )
    model, _ = _build(deck, tmp_path)
    g = model.shells_qbat
    assert g.n == 2 and g.state["nip_max"] == 5
    assert g.state["chk_fail"]
    assert g.state["slices"][0][1].fail.ifail_sh == 2
    v = np.zeros((model.numnod, 3))
    v[:, 0] = 0.01 * (model.x[:, 0] % 3.0)       # eps = 1% per call, both
    fint = np.zeros((model.numnod, 3))
    mint = np.zeros((model.numnod, 3))
    dte = None
    for _ in range(10):                          # cum eps_p >> eps_f = 0.05
        fint[:] = 0.0
        mint[:] = 0.0
        dte = shell_qbat.forces(g, model.x, v, model.vr, 1.0, fint, mint)
    assert g.state["off"].tolist() == [0.0, 0.0]
    assert np.abs(fint).max() == 0.0
    assert np.all(dte >= 1.0e29)                 # dead elements claim EP30
    assert np.abs(g.state["sig"]).max() == 0.0


# ============================================================================
# F2. implicit: the documented refusal + the mass side of the contract
# ============================================================================

def test_implicit_refusal_and_consistent_mass(tmp_path):
    """QBAT deliberately ships NO tangent()/kgeo() (module docstring): an
    /IMPL deck with a shells_qbat group must be refused LOUDLY by the
    implicit assembly's supported-kernel gate, never silently skipped.
    consistent_mass() IS provided for kernel-contract completeness:
    symmetric, translation entries totalling the element mass, rotary
    entries totalling m t^2/12."""
    from pyradioss.implicit import assembly
    assert "shells_qbat" not in assembly._TANGENT_KERNELS
    assert not hasattr(shell_qbat, "tangent")
    assert not hasattr(shell_qbat, "kgeo")
    g, model = _qbat_plate(tmp_path)
    with pytest.raises(NotImplementedError, match="shells_qbat"):
        assembly.element_triplets("shells_qbat", g, model.x, None)
    me, edofs = shell_qbat.consistent_mass(g)
    assert me.shape == (1, 24, 24) and edofs.shape == (1, 24)
    np.testing.assert_allclose(me[0], me[0].T, rtol=0, atol=0)
    m = g.state["mass"][0]
    tx = [6 * a for a in range(4)]               # x-translation dofs
    rx = [6 * a + 3 for a in range(4)]           # x-rotation dofs
    assert me[0][np.ix_(tx, tx)].sum() == pytest.approx(m, rel=1e-12)
    assert me[0][np.ix_(rx, rx)].sum() == pytest.approx(
        m * 0.1 ** 2 / 12.0, rel=1e-12)


# ============================================================================
# G. the analytic cantilever: explicit step-load ringdown (full pipeline)
# ============================================================================

def _read_th(path):
    with open(path) as fh:
        fh.readline()
        cols = fh.readline().strip().split(",")
        data = np.loadtxt(fh, delimiter=",", ndmin=2)
    return {c: data[:, k] for k, c in enumerate(cols)}


def _final_summary(out_path):
    text = open(out_path).read()

    def grab(label):
        return float(re.search(label + r"\s*\.[ .]*:\s*([-\d.Ee+]+)",
                               text).group(1))

    return {"ERR": grab("ENERGY ERROR"), "IE": grab("INTERNAL ENERGY"),
            "HE": grab("HOURGLASS ENERGY"), "EW": grab("EXTERNAL WORK"),
            "NORMAL": "ENGINE TERMINATION : NORMAL" in text}


def test_cantilever_euler_bernoulli_ringdown(make_deck):
    """10x1 QBAT strip cantilever (L=10, b=1, t=0.1, nu=0 so the plate
    strip IS the Euler-Bernoulli beam: C = E), tip step load F:

      * static deflection: mean tip DZ over the last full period ==
        delta = F L^3 / (3 E I),   I = b t^3/12  (rel 3%: h^2 mesh
        discretization + the ~1.5% multi-mode ripple of a step response);
      * first-mode period from the zero crossings of DZ - delta vs the
        analytic (1.875104)^2 sqrt(EI/(rho b t)) / (2 pi L^2) (rel 2%);
      * energy balance closed (|ERR| < 1%) and the M41 HE contract:
        HE/IE < 1e-3 where the pre-M41 BT-fallback stored the bending
        energy partly (up to 40%) as hourglass;
      * runs the FULL starter -> dispatch -> engine -> /TH pipeline."""
    L, b, t = 10.0, 1.0, 0.1
    ne = 10
    nodes, shells = [], []
    for i in range(ne + 1):
        nodes.append(f"{2*i+1} {float(i)} 0 0")
        nodes.append(f"{2*i+2} {float(i)} {b} 0")
    for i in range(ne):
        shells.append(f"{i+1} {2*i+1} {2*i+3} {2*i+4} {2*i+2}")
    F = 1.05e-6                       # delta = 0.02 = L/500: linear range
    I = b * t ** 3 / 12.0
    delta = F * L ** 3 / (3.0 * E_STEEL * I)
    f1 = (1.875104068712 ** 2 / (2.0 * np.pi * L ** 2)) \
        * np.sqrt(E_STEEL * I / (RHO_STEEL * b * t))
    T1 = 1.0 / f1
    starter = (
        "/BEGIN\nqbat cantilever\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/SHELL/1\n" + "\n".join(shells) + "\n"
        "/PART/1\nstrip\n1 1\n"
        f"/MAT/LAW1/1\nsteel\n{RHO_STEEL}\n{E_STEEL} 0.0\n"
        "/PROP/SHELL/1\nqbat\n12 0 0 0\n0 0 0 0 0\n3 0 0.1\n"
        "/GRNOD/NODE/1\nclamp\n1 2\n"
        "/GRNOD/NODE/2\ntip\n21 22\n"
        "/BCS/1\nclamp\n111 111 0 1\n"
        f"/FUNCT/1\nstep\n0.0 1.0\n100.0 1.0\n"
        f"/CLOAD/1\ntip step\n1 Z 2 {F / 2.0}\n"
        f"/TH/NODE/1\ntip\nDZ\n21\n/END\n"
    )
    engine = (f"/RUN/CANT/1\n{3.0}\n/DT\n0.9 0\n/TFILE\n0.002\n"
              "/PRINT/-10000\n")
    s, e = make_deck("CANT", starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    assert model.shells_qbat is not None and model.shells_qbat.n == ne
    summ = _final_summary(e.replace("_0001.rad", "_0001.out"))
    assert summ["NORMAL"]
    assert abs(summ["ERR"]) < 1.0
    # THE M41 contract: hourglass ~ 0 (only cbavisc dn work), where the
    # BT fallback stored up to 40% of EW as HE on BATOZ decks
    assert summ["HE"] < 1.0e-3 * summ["IE"]
    th = _read_th(e.replace("_0001.rad", "T01.csv"))
    tt, dz = th["TIME"], th["N21_DZ"]
    # static deflection: mean over the last full analytic period
    win = tt >= tt[-1] - T1
    mean_dz = dz[win].mean()
    assert mean_dz == pytest.approx(delta, rel=0.03)
    # period from the zero crossings of (dz - mean): average the last
    # two full periods (2 crossings per period)
    sgn = np.sign(dz - mean_dz)
    cross = np.where(np.diff(sgn) != 0)[0]
    tc = []
    for k in cross:
        # linear interpolation of the crossing time
        f = (mean_dz - dz[k]) / (dz[k + 1] - dz[k])
        tc.append(tt[k] + f * (tt[k + 1] - tt[k]))
    tc = np.array(tc)
    assert len(tc) >= 5
    periods = np.diff(tc)[-4:] * 2.0
    assert periods.mean() == pytest.approx(T1, rel=0.02)
    # ringdown amplitude persists (elastic, dn ~ 1e-3: barely damped)
    last = dz[tt >= tt[-1] - T1]
    assert last.max() - last.min() > 1.0 * delta
