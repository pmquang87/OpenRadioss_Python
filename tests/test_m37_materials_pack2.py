"""
M37 material physics pack 2 — LAW19 fabric, LAW24 concrete, LAW81
Drucker-Prager with cap.

Every test drives the law kernels directly (the sigeps contract: stress
arrays mutated in place, ``extra`` views like the element kernels build
them) with parameter sets whose expected responses are hand-derived from
the upstream Fortran formulas cited in each law module.
"""

import copy

import numpy as np
import pytest

from pyradioss import materials
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input import mat_reader

# Deck-level tests here parse /MAT cards through the cfg-driven generic
# reader, which needs OpenRadioss's hm_cfg_files (not vendored — license;
# CI fetches a sparse checkout, ci.yml). Skip loudly without them.
pytestmark = pytest.mark.skipif(
    mat_reader.catalogue().schema("FABRI") is None,
    reason="hm_cfg_files CFG tree not found — set PYRADIOSS_HM_CFG "
           "(see ci.yml / PORTING_GUIDE M37)")


# ============================================================================
# helpers
# ============================================================================

def _parse_mat(tmp_path, text):
    """Parse one embedded /MAT block through the real cfg-driven reader
    and return the built (physics) material."""
    deck = tmp_path / "m.rad"
    deck.write_text("/BEGIN\ntest\n      2022         0\n"
                    "                  kg                   m"
                    "                   s\n"
                    "                  kg                   m"
                    "                   s\n" + text + "/END\n")
    log = MessageLog()

    class _M:
        materials = {}

    model = _M()
    model.materials = {}
    for b in read_deck(str(deck)):
        if b.key0 == "MAT":
            mat_reader.read_generic_mat(b, model, log)
    assert model.materials, "material did not parse"
    (mat,) = model.materials.values()
    assert not getattr(mat, "inactive", False), \
        "builder did not activate the law"
    return mat


def _f20(*vals):
    """One fixed-format card of 20-char fields."""
    return "".join(f"{v:>20}" for v in vals) + "\n"


def _shell_extra(mat, n, nip=1, k=0):
    """Allocate per-layer extra arrays exactly like the shell kernels
    (extra_shapes -> zeros((n,)+shape)), return the layer-k view dict."""
    full = {name: np.zeros((n,) + shape)
            for name, shape in materials.extra_shapes(mat, nip).items()}
    extra = {name: arr[:, k] for name, arr in full.items()}
    extra["layfail"] = np.ones(n)
    return extra


def _solid_extra(mat, n):
    """Allocate solid extra arrays exactly like the solid kernels."""
    return {name: np.zeros((n,) + shape)
            for name, shape in materials.extra_shapes(mat).items()}


def _rot6_stress(sig6, R):
    """Rotate Voigt stress [xx,yy,zz,xy,yz,zx] by R (sig' = R S R^T)."""
    S = np.array([[sig6[0], sig6[3], sig6[5]],
                  [sig6[3], sig6[1], sig6[4]],
                  [sig6[5], sig6[4], sig6[2]]])
    Sp = R @ S @ R.T
    return np.array([Sp[0, 0], Sp[1, 1], Sp[2, 2],
                     Sp[0, 1], Sp[1, 2], Sp[0, 2]])


def _rot6_strain(eps6, R):
    """Rotate an engineering-shear Voigt strain by R."""
    E = np.array([[eps6[0], eps6[3] / 2, eps6[5] / 2],
                  [eps6[3] / 2, eps6[1], eps6[4] / 2],
                  [eps6[5] / 2, eps6[4] / 2, eps6[2]]])
    Ep = R @ E @ R.T
    return np.array([Ep[0, 0], Ep[1, 1], Ep[2, 2],
                     2 * Ep[0, 1], 2 * Ep[1, 2], 2 * Ep[0, 2]])


# ============================================================================
# LAW19 — fabric
# ============================================================================

FABRI_CARD = (
    "/MAT/FABRI/7\n"
    "unit fabric\n" + _f20("8E-7", 0) +
    _f20(".3", ".5", ".2") +          # E11, E22, NU12
    _f20(".05", ".04", ".06") +       # G12, G23, G31
    _f20(".1", "", "0") + "\n")       # RE, blank, ZEROSTRESS=0


@pytest.fixture()
def fabric(tmp_path):
    return _parse_mat(tmp_path, FABRI_CARD)


class TestLaw19Fabric:
    def test_builder_constants(self, fabric):
        # hm_read_mat19.F derivations
        e11, e22, n12 = 0.3, 0.5, 0.2
        n21 = n12 * e22 / e11
        detc = 1.0 - n12 * n21
        p = fabric.params
        assert p["A11"] == pytest.approx(e11 / detc)
        assert p["A22"] == pytest.approx(e22 / detc)
        assert p["A12"] == pytest.approx(e11 / detc * n21)
        # kernel-facing constants: PM(20)/PM(21)/PM(22)/PM(27)
        assert fabric.E == pytest.approx(max(e11, e22) / detc)
        assert fabric.nu == pytest.approx(np.sqrt(n12 * n21))
        assert fabric.G == pytest.approx(0.06)      # max(G12,G23,G31)
        assert fabric.sound_speed_shell() == pytest.approx(
            np.sqrt(max(e11, e22) / detc / 8e-7))

    def test_uniaxial_warp_weft(self, fabric):
        """Pure warp / weft strain -> the orthotropic A matrix exactly."""
        p = fabric.params
        extra = _shell_extra(fabric, 2)
        sig = np.zeros((2, 3))
        deps = np.array([[1e-3, 0.0, 0.0],       # warp
                         [0.0, 1e-3, 0.0]])      # weft
        materials.shell_update(fabric, sig, deps, None, 1e-5, extra)
        assert sig[0, 0] == pytest.approx(p["A11"] * 1e-3)
        assert sig[0, 1] == pytest.approx(p["A12"] * 1e-3)
        assert sig[0, 2] == 0.0
        assert sig[1, 0] == pytest.approx(p["A12"] * 1e-3)
        assert sig[1, 1] == pytest.approx(p["A22"] * 1e-3)
        # total-strain law: a second identical increment doubles it
        materials.shell_update(fabric, sig, deps, None, 1e-5, extra)
        assert sig[0, 0] == pytest.approx(p["A11"] * 2e-3)

    def test_shear_reduces_compressive_principal(self, fabric):
        """Pure shear = principal tension + principal compression: the
        BETA blend keeps the tensile principal stress and scales the
        compressive one by RCOMP exactly (the algebraic meaning of
        sigeps19c.F line 115)."""
        extra = _shell_extra(fabric, 1)
        sig = np.zeros((1, 3))
        deps = np.array([[0.0, 0.0, 2e-3]])
        materials.shell_update(fabric, sig, deps, None, 1e-5, extra)
        tau = 0.05 * 2e-3                     # unreduced G12 * gamma
        s = 0.5 * (sig[0, 0] + sig[0, 1])
        r = np.hypot(0.5 * (sig[0, 0] - sig[0, 1]), sig[0, 2])
        assert s + r == pytest.approx(tau)          # P2 kept
        assert s - r == pytest.approx(-0.1 * tau)   # P1 scaled by RCOMP

    def test_zero_compression_bicompression(self, fabric):
        """Both principal stresses <= 0: whole tensor scaled by RCOMP
        (sigeps19c.F lines 119-124)."""
        extra = _shell_extra(fabric, 1)
        sig = np.zeros((1, 3))
        deps = np.array([[-1e-3, -1e-3, 0.0]])
        materials.shell_update(fabric, sig, deps, None, 1e-5, extra)
        p = fabric.params
        full = np.array([(p["A11"] + p["A12"]) * -1e-3,
                         (p["A12"] + p["A22"]) * -1e-3])
        assert sig[0, 0] == pytest.approx(0.1 * full[0])
        assert sig[0, 1] == pytest.approx(0.1 * full[1])

    def test_zero_compression_mixed(self, fabric):
        """P1 < 0 < P2: BETA blend around the tensile principal value
        (sigeps19c.F lines 114-118)."""
        p = fabric.params
        extra = _shell_extra(fabric, 1)
        sig = np.zeros((1, 3))
        # strain chosen so sig_xx < 0 < sig_yy, no shear
        exx = -2e-3
        eyy = 1e-3
        deps = np.array([[exx, eyy, 0.0]])
        materials.shell_update(fabric, sig, deps, None, 1e-5, extra)
        sxx = p["A11"] * exx + p["A12"] * eyy
        syy = p["A12"] * exx + p["A22"] * eyy
        assert sxx < 0.0 < syy          # the intended quadrant
        s = 0.5 * (sxx + syy)
        r = abs(0.5 * (sxx - syy))
        p2 = s + r
        beta = 0.5 * ((1.0 - 0.1) * s / r + 1.0 + 0.1)
        assert sig[0, 0] == pytest.approx(beta * (sxx - p2) + p2)
        assert sig[0, 1] == pytest.approx(beta * (syy - p2) + p2)

    def test_pure_tension_untouched(self, fabric):
        """No compression -> no reduction at all."""
        p = fabric.params
        extra = _shell_extra(fabric, 1)
        sig = np.zeros((1, 3))
        deps = np.array([[1e-3, 1e-3, 0.0]])
        materials.shell_update(fabric, sig, deps, None, 1e-5, extra)
        assert sig[0, 0] == pytest.approx((p["A11"] + p["A12"]) * 1e-3)

    def test_zerostress_reference_state(self, tmp_path):
        """ZEROSTRESS = 1: the t = 0 stress is stored and zeroed, then
        relaxed by DSIG whenever the increment unloads it (sigeps19c.F
        lines 131-168)."""
        card = FABRI_CARD.replace(_f20(".1", "", "0"),
                                  _f20(".1", "", "1"))
        mat = _parse_mat(tmp_path, card)
        p = mat.params
        extra = _shell_extra(mat, 1)
        sig = np.zeros((1, 3))
        e0 = 1e-3
        deps = np.array([[e0, 0.0, 0.0]])
        dt = 1e-5
        # cycle 1 (TT = 0 <= TSTART): store SIGI, output zero
        materials.shell_update(mat, sig, deps, None, dt, extra)
        s1 = p["A11"] * e0
        assert sig[0, 0] == 0.0
        assert extra["sigi19"][0, 0] == pytest.approx(s1)
        # cycle 2: further LOADING (DSIG > 0 while SIGI > 0) -> SIGI
        # keeps its value, output = full - SIGI
        materials.shell_update(mat, sig, deps, None, dt, extra)
        s2 = p["A11"] * 2 * e0
        assert extra["sigi19"][0, 0] == pytest.approx(s1)
        assert sig[0, 0] == pytest.approx(s2 - s1)
        # cycle 3: UNLOADING -> SIGI relaxes by ZEROSTRESS*DSIG
        # DSIG = SIGN - SIGO - SIGI with SIGO = the cycle-2 output
        materials.shell_update(mat, sig, np.array([[-e0, 0.0, 0.0]]),
                               None, dt, extra)
        dsig = s1 - (s2 - s1) - s1          # = s1 - s2 < 0
        si = max(0.0, s1 + 1.0 * dsig)
        assert extra["sigi19"][0, 0] == pytest.approx(si)
        assert sig[0, 0] == pytest.approx(s1 - si)

    def test_rcomp_default_is_one(self, tmp_path):
        """RE = 0 on the card -> RCOMP = 1 (no reduction)."""
        card = FABRI_CARD.replace(_f20(".1", "", "0"),
                                  _f20("0", "", "0"))
        mat = _parse_mat(tmp_path, card)
        assert mat.params["RCOMP"] == 1.0


# ============================================================================
# LAW24 — concrete (Kupfer RD-E-4701 parameters)
# ============================================================================

CONC_CARD = (
    "/MAT/CONC/1\n"
    "Concrete\n" + _f20(".0022", 0) +
    _f20("31700", ".22", 0) +                       # E, nu, Icap
    _f20("32.22", "0.1", "1.15") +                  # fc, ft/fc, fb/fc
    _f20(0, 0, 0) +                                 # Ht, Dsup, epsmax
    _f20(".35", 0, 0, 0) +                          # ky, rt, rc, Hbp
    _f20("-0.6", "0.2", 0) +                        # alpha_y, alpha_f, vmax
    _f20(0, 0, 0) +                                 # fk, f0, Hv0
    _f20(0, 0, 0) +                                 # steel E, sy, Et
    _f20(0, 0, 0))                                  # arm1..3


FC = 32.22
FT = 0.1 * FC


@pytest.fixture()
def conc(tmp_path):
    return _parse_mat(tmp_path, CONC_CARD)


def _uniaxial_drive(mat, n_steps, d_axial, lat_tol=1e-4, axis=0):
    """Strain-driven uniaxial-STRESS loop: each step finds the lateral
    strain increments that zero the two lateral normal stresses (trial
    evaluations on state copies, Newton with the elastic compliance),
    then commits.  Returns the (n_steps,) axial stress history."""
    extra = _solid_extra(mat, 1)
    sig = np.zeros((1, 6))
    lat = [i for i in range(3) if i != axis]
    hist = np.zeros(n_steps)
    dlat = np.zeros(2)
    p = mat.params
    a11, a12 = p["A11c"], p["A12c"]
    # 2x2 elastic sensitivity of the lateral stresses to the lateral
    # strain increments
    J = np.array([[a11, a12], [a12, a11]])
    Jinv = np.linalg.inv(J)
    for it_step in range(n_steps):
        for _ in range(40):
            s_try = sig.copy()
            e_try = copy.deepcopy(extra)
            deps = np.zeros((1, 6))
            deps[0, axis] = d_axial
            deps[0, lat[0]] = dlat[0]
            deps[0, lat[1]] = dlat[1]
            materials.solid_update(mat, s_try, deps, None, 1e-5, e_try)
            res = np.array([s_try[0, lat[0]], s_try[0, lat[1]]])
            if np.abs(res).max() < lat_tol * FC:
                break
            dlat -= Jinv @ res
        sig = s_try
        extra = e_try
        hist[it_step] = sig[0, axis]
    return hist, sig, extra


class TestLaw24Concrete:
    def test_builder_pm_table(self, conc):
        """hm_read_mat24.F defaults + Ottosen fit: the surface must pass
        EXACTLY through the uniaxial compressive strength fc and the
        tensile strength ft (that is what the AA/BC/BT/AC algebra
        encodes)."""
        p = conc.params
        assert p["RC"] == pytest.approx(-FC / 3.0)
        assert p["RO0"] == pytest.approx(-0.8 * FC)
        assert p["QQ"] == pytest.approx(2.0)           # Ht default = -E
        assert p["EPST"] == pytest.approx(FT / 31700.0)
        aa, ac = p["AA"], p["AC"]
        bc, bt = p["BC"], p["BT"]

        def rf(sm, cs3t):
            bb = 0.5 * ((1 - cs3t) * bc + (1 + cs3t) * bt)
            return (-bb + np.sqrt(bb * bb - aa * sm + ac)) / aa

        # uniaxial compression: sm = -fc/3, cos3t = -1, r = sqrt(2/3) fc
        assert rf(-FC / 3.0, -1.0) == pytest.approx(np.sqrt(2.0 / 3.0) * FC,
                                                    rel=1e-10)
        # uniaxial tension: sm = ft/3, cos3t = +1, r = sqrt(2/3) ft
        assert rf(FT / 3.0, 1.0) == pytest.approx(np.sqrt(2.0 / 3.0) * FT,
                                                  rel=1e-10)

    def test_sound_speed_constant(self, conc):
        """m24law.F: SSP = sqrt(PM(24)/PM(1)) — constant, returned by the
        law so the kernel's dt uses rho0 not the current density."""
        extra = _solid_extra(conc, 1)
        sig = np.zeros((1, 6))
        _, _, c = materials.solid_update(conc, sig, np.zeros((1, 6)),
                                         None, 1e-5, extra)
        assert c[0] == pytest.approx(
            np.sqrt(conc.params["A11c"] / 0.0022))

    def test_elastic_slope_is_E(self, conc):
        """Small uniaxial-stress steps: the initial tangent is Young's
        modulus."""
        hist, _, _ = _uniaxial_drive(conc, 3, -1e-5)
        eps = -1e-5 * np.arange(1, 4)
        slopes = hist / eps
        assert slopes == pytest.approx(31700.0, rel=1e-3)

    def test_uniaxial_compression_peak_fc(self, conc):
        """Kupfer C000: strain-driven uniaxial compression peaks at the
        uniaxial compressive strength fc' (the VK -> 1 failure-surface
        cap of plas24.F)."""
        hist, sig, extra = _uniaxial_drive(conc, 260, -2e-5)
        peak = -hist.min()
        assert peak == pytest.approx(FC, rel=0.05)
        # hardening happened: the peak is beyond the proportional yield
        assert extra["vk24"][0] == pytest.approx(1.0, abs=1e-6)
        assert extra["vk024"][0] == pytest.approx(1.0, abs=1e-6)
        # no tensile damage in compression
        assert np.all(extra["dam24"] == 0.0)

    def test_uniaxial_tension_cutoff_ft(self, conc):
        """Kupfer T000: the tensile peak is the uniaxial strength
        ft = 0.1 fc; beyond it the fixed crack softens the response and
        the damage state records the crack."""
        hist, sig, extra = _uniaxial_drive(conc, 60, 4e-6)
        peak = hist.max()
        assert peak == pytest.approx(FT, rel=0.08)
        # softening after the peak
        assert hist[-1] < 0.75 * peak
        # a crack opened in direction 1 with its rupture strain recorded
        assert extra["dam24"][0, 0] > 0.0
        assert extra["epsf24"][0, 0] > 0.0
        # crack normal aligned with the load axis
        assert abs(extra["ang24"][0, 0]) == pytest.approx(1.0, abs=1e-6)

    def test_tension_objectivity(self, conc):
        """The same uniaxial-strain tension path rotated 45 deg about z
        gives the rotated stress tensor (crack frame machinery is
        covariant)."""
        theta = np.pi / 4
        cz, sz = np.cos(theta), np.sin(theta)
        R = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])

        deps = np.zeros(6)
        deps[0] = 5e-6                       # uniaxial STRAIN tension
        depsr = _rot6_strain(deps, R)

        sig_a = np.zeros((1, 6))
        ext_a = _solid_extra(conc, 1)
        sig_b = np.zeros((1, 6))
        ext_b = _solid_extra(conc, 1)
        for _ in range(50):
            materials.solid_update(conc, sig_a, deps[None, :], None,
                                   1e-5, ext_a)
            materials.solid_update(conc, sig_b, depsr[None, :], None,
                                   1e-5, ext_b)
        expect = _rot6_stress(sig_a[0], R)
        assert sig_b[0] == pytest.approx(expect, rel=1e-6, abs=1e-8 * FC)
        # damage grew identically
        assert ext_b["dam24"][0].max() == pytest.approx(
            ext_a["dam24"][0].max(), rel=1e-6)

    def test_reinforcement_refused(self, tmp_path):
        """Documented cut: ARM > 0 must refuse cleanly, not silently
        drop the steel."""
        card = CONC_CARD.replace(_f20(0, 0, 0) + _f20(0, 0, 0),
                                 _f20(0, 0, 0) + _f20(".1", 0, 0), 1)
        # replace only the LAST card (arm percentages)
        card = CONC_CARD[:-len(_f20(0, 0, 0))] + _f20(".05", 0, 0)
        deck = tmp_path / "arm.rad"
        deck.write_text("/BEGIN\nt\n      2022         0\n"
                        "                  kg                   m"
                        "                   s\n"
                        "                  kg                   m"
                        "                   s\n" + card + "/END\n")
        log = MessageLog()

        class _M:
            pass

        model = _M()
        model.materials = {}
        for b in read_deck(str(deck)):
            if b.key0 == "MAT":
                mat_reader.read_generic_mat(b, model, log)
        # builder raised -> error logged, material NOT registered active
        assert not model.materials or \
            getattr(list(model.materials.values())[0], "inactive", False) \
            or any("not ported" in str(m) for m in log.errors)


# ============================================================================
# LAW81 — Drucker-Prager with cap
# ============================================================================

def _law81(k0=20000.0, g0=12000.0, phi=30.0, psi=0.0, c0=5.0,
           pb0=1e6, alpha=0.5, rho0=2e-3):
    """Direct-built LAW81 material (bypasses the deck: the builder is
    exercised separately on the corpus card)."""
    from pyradioss.materials.law81_druckerprager import build_law81

    class _R:
        pass

    r = _R()
    r.id = 1
    r.title = "dp"
    r.density = rho0
    r.params = {"K0": k0, "MAT_G0": g0, "MAT_COH0": c0, "MAT_PB0": pb0,
                "MAT_Beta": phi, "Psi": psi, "MAT_ALPHA": alpha,
                "MAT_EPS": 0.0, "MAT_SRP": 0.0, "Iflag": 0,
                "FUN_A1": 0, "FUN_A2": 0, "FUN_A3": 0, "FUN_A4": 0}
    return build_law81(r)


class TestLaw81DruckerPrager:
    def test_builder_corpus_card(self, tmp_path):
        """The Kupfer LAW81 card parses and builds with tan(beta) and
        the c/Pb scale-factor defaults of hm_read_mat81.F90."""
        card = ("/MAT/LAW81/1\n Concret\n" +
                _f20("0.0022") +
                _f20("18869.0476190476", "12991.8032786885") +
                _f20("68.7903728190", "") +
                _f20("0.0736137946", "", "") +
                "         0         0        23        24         0\n")
        mat = _parse_mat(tmp_path, card)
        p = mat.params
        assert p["TGPHI"] == pytest.approx(np.tan(np.radians(68.7903728190)))
        assert p["TGPSI"] == 0.0
        assert p["C0"] == 1.0 and p["PB0"] == 1.0     # scale factors
        assert p["funct81_ids"] == [0, 0, 23, 24]
        assert mat.K == pytest.approx(18869.0476190476)
        assert mat.G == pytest.approx(12991.8032786885)

    def test_cone_yield_exact_pq(self):
        """p-q plane: with psi = 0 a pure-shear drive keeps p constant,
        and the cutting plane lands EXACTLY on q = p tan(phi) + c for
        every pre-pressure below the cap (the consistency condition is
        linear there)."""
        mat = _law81()
        tgphi = mat.params["TGPHI"]
        for p_target in (2.0, 10.0, 50.0):
            extra = _solid_extra(mat, 1)
            sig = np.zeros((1, 6))
            # hydrostatic pre-compression (elastic, exact)
            ev = -p_target / (3.0 * 20000.0)
            deps = np.zeros((1, 6))
            deps[0, :3] = ev
            materials.solid_update(mat, sig, deps, None, 1e-5, extra)
            pr = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
            assert pr == pytest.approx(p_target, rel=1e-12)
            # pure shear to (and beyond) yield at constant p
            deps = np.zeros((1, 6))
            deps[0, 3] = 2e-4                        # engineering gamma
            for _ in range(40):
                materials.solid_update(mat, sig, deps, None, 1e-5, extra)
            pr = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
            q = np.sqrt(3.0) * abs(sig[0, 3])
            assert pr == pytest.approx(p_target, rel=1e-9)   # psi = 0
            assert q == pytest.approx(pr * tgphi + 5.0, rel=1e-9)
            # plastic strain is purely deviatoric
            assert extra["epspd81"][0] > 0.0
            assert extra["epspv81"][0] == pytest.approx(0.0, abs=1e-14)

    def test_apex_return_exact(self):
        """Hydrostatic tension beyond the apex: p clamps to -c/tan(phi)
        exactly (sigeps81 tri-traction block)."""
        mat = _law81()
        tgphi = mat.params["TGPHI"]
        extra = _solid_extra(mat, 1)
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        deps[0, :3] = 5.0 / (3.0 * 20000.0) * 10.0    # way past the apex
        materials.solid_update(mat, sig, deps, None, 1e-5, extra)
        pr = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
        assert pr == pytest.approx(-5.0 / tgphi, rel=1e-12)
        assert extra["epspv81"][0] < 0.0              # tensile vol. p.s.

    def test_cap_return_exact(self):
        """Hydrostatic compression beyond the (flat) cap: the Newton on
        ftrc = p - Pb converges to p = Pb exactly in one iteration."""
        mat = _law81(pb0=10.0)
        extra = _solid_extra(mat, 1)
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        deps[0, :3] = -30.0 / (3.0 * 20000.0)         # trial p = 30 > Pb
        materials.solid_update(mat, sig, deps, None, 1e-5, extra)
        pr = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
        assert pr == pytest.approx(10.0, rel=1e-12)
        assert extra["epspv81"][0] == pytest.approx(20.0 / 20000.0,
                                                    rel=1e-12)

    def test_cap_hardening_function(self):
        """A tabulated Pb(epspv) moves the cap out: after a compaction
        step the pressure sits on the HARDENED cap, consistently with
        the volumetric plastic strain (Pb(epspv) = p)."""
        mat = _law81(pb0=1.0)
        # Pb = 10 + 2000 * epspv (scale factor Pb0 = 1)
        mat.params["curve81_pb"] = (np.array([0.0, 1.0]),
                                    np.array([10.0, 2010.0]))
        extra = _solid_extra(mat, 1)
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        deps[0, :3] = -30.0 / (3.0 * 20000.0)
        materials.solid_update(mat, sig, deps, None, 1e-5, extra)
        pr = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
        ev = extra["epspv81"][0]
        assert ev > 0.0
        assert pr == pytest.approx(10.0 + 2000.0 * ev, rel=1e-6)

    def test_sound_speed(self):
        mat = _law81()
        extra = _solid_extra(mat, 1)
        sig = np.zeros((1, 6))
        _, _, c = materials.solid_update(mat, sig, np.zeros((1, 6)),
                                         None, 1e-5, extra)
        assert c[0] == pytest.approx(
            np.sqrt((20000.0 + 4.0 / 3.0 * 12000.0) / 2e-3))

    def test_objectivity(self):
        """A rotated strain path yields the rotated stress (isotropic
        return map)."""
        mat = _law81()
        rng = np.random.default_rng(3)
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]

        path = [np.array([-2e-4, -2e-4, -2e-4, 0, 0, 0]),
                np.array([0, 0, 0, 3e-4, 0, 0]),
                np.array([1e-4, -2e-4, 0, 1e-4, 2e-4, 0])]
        sig_a = np.zeros((1, 6))
        ext_a = _solid_extra(mat, 1)
        sig_b = np.zeros((1, 6))
        ext_b = _solid_extra(mat, 1)
        for d in path:
            for _ in range(10):
                materials.solid_update(mat, sig_a, d[None, :], None,
                                       1e-5, ext_a)
                materials.solid_update(mat, sig_b,
                                       _rot6_strain(d, Q)[None, :], None,
                                       1e-5, ext_b)
        expect = _rot6_stress(sig_a[0], Q)
        assert sig_b[0] == pytest.approx(expect, rel=1e-9, abs=1e-12)
        assert ext_b["epspd81"][0] == pytest.approx(ext_a["epspd81"][0],
                                                    rel=1e-9)

    def test_epsp_reported(self):
        """The kernel-facing epsp array receives the deviatoric
        equivalent plastic strain (in-place, like every ported law)."""
        mat = _law81()
        extra = _solid_extra(mat, 1)
        sig = np.zeros((1, 6))
        epsp = np.zeros(1)
        deps = np.zeros((1, 6))
        deps[0, 3] = 5e-3                     # yields immediately
        materials.solid_update(mat, sig, deps, epsp, 1e-5, extra)
        assert epsp[0] == extra["epspd81"][0] > 0.0


# ============================================================================
# dispatch / registry / checks integration
# ============================================================================

class TestIntegration:
    def test_registry_entries(self):
        from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in ("FABRI", "LAW19", "CONC", "LAW24", "LAW81"):
            assert key in MAT_PHYSICS_REGISTRY

    def test_extra_shapes(self, tmp_path):
        fab = _parse_mat(tmp_path, FABRI_CARD)
        assert materials.extra_shapes(fab, 3) == {
            "eps19": (3, 3), "sigi19": (3, 3), "t19": (3,)}
        conc = _parse_mat(tmp_path, CONC_CARD)
        shapes = materials.extra_shapes(conc)
        assert shapes["strain24"] == (6,) and shapes["vk024"] == ()
        law81 = _law81()
        assert set(materials.extra_shapes(law81)) == {"epspd81", "epspv81"}
        assert materials.needs_env(conc) and materials.needs_env(law81)
        assert not materials.needs_env(fab)

    def test_checks_allow_new_laws(self):
        from pyradioss.starter.checks import _ALLOWED_LAWS
        assert {24, 81} <= _ALLOWED_LAWS["bricks"]
        assert 19 in _ALLOWED_LAWS["shells"]
