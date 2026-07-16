#!/usr/bin/env python3
"""
Generate the JOINT_NONGAUSSIAN_FATIGUE example deck: the NON-GAUSSIAN JOINT-TENSOR
DISTRIBUTION fatigue life of a base-clamped SOLID-brick cantilever whose full 6x6
stress-TENSOR COMPONENTS carry DIFFERENT kurtoses — a VECTOR (component-wise)
Winterstein-Hermite / translation-process transform of the CORRELATED stress tensor,
so the target kurtosis is imposed JOINTLY on the tensor COMPONENTS (preserving the
6x6 covariance / cross-PSD) and the resolved critical plane INHERITS the INDUCED
non-Gaussianity (M33, /IMPL/FATIG/NGAUSS/JOINT/WVILLE) — the FIRST item M32 deferred.

This is deliberately the SAME solid-brick cantilever the M27 joint_evolutionary_fatigue
and the M32 (scalar) nongaussian_wigner_ville_fatigue examples build on, RE-RUN under a
JOINT non-Gaussian TENSOR. Where M24/M32 imposed the target kurtosis on the ALREADY-
RESOLVED equivalent scalar (the von-Mises / critical-plane scalar — a SCALAR Hermite
transform), M33 imposes a PER-COMPONENT kurtosis on the six Voigt stress components
(sigma_xx sigma_yy sigma_zz sigma_xy sigma_yz sigma_zx) via a VECTOR Hermite transform
of the correlated tensor, and lets the critical-plane reduction INHERIT the induced
kurtosis — a genuinely multiaxial non-Gaussian distribution. Because the resolved-plane
scalar s = p^T sigma is a LINEAR projection of a component-wise-transformed correlated
Gaussian, its INDUCED kurtosis is a closed-form function of the per-component kurtoses
AND the full 6x6 correlation (Grigoriu translation-process theory; the vector
Winterstein-Hermite model; Lutes & Sarkani "Random Vibrations"), cross-validated by a
MULTIVARIATE non-Gaussian Monte-Carlo.

THE DEMONSTRATOR: a spiky, non-proportional multiaxial transient. The excitation window
sweeps the modal band (fc 40 -> 600 Hz, a narrow bw — the M31/M32 continuous instant
spectrum) while DIFFERENT tensor components carry DIFFERENT kurtoses: the AXIAL bending
stress sigma_xx is strongly leptokurtic (g4_xx ~ 9 — a spiky, intermittent bending
transient), the TORSION / TRANSVERSE-SHEAR components sigma_xy / sigma_zx moderately
leptokurtic (g4 ~ 7 / 6), and the minor components near-Gaussian. Because the critical
plane mixes these components THROUGH the tensor correlation, the resolved-plane INDUCED
kurtosis DIFFERS from imposing any single kurtosis directly on the scalar — the M33
JOINT life DIFFERS from the M32 EQUIVALENT-SCALAR life. Both are printed SIDE BY SIDE
with the M31/M27 Gaussian life.

Model (units mm / ms / kg -> forces kN, stresses GPa, frequencies kHz): IDENTICAL to
the M27 joint_evolutionary_fatigue example — a rectangular (2y x 1z) solid-brick
cantilever with an OFFSET skew tip force so the root elements see bending + transverse
shear + torsion (a genuinely multiaxial, non-proportional stress tensor):
  * a straight cantilever of NX /BRICK hexa8 solids along +x, clamped at x = 0;
  * steel LAW1 (E = 210 GPa, nu = 0.3, rho = 7.8e-6 kg/mm^3);
  * a random tip FORCE PSD, skew (y + z) AND offset (unequal across the width);
  * /FUNCT/10 — a FLAT band-limited input force PSD S_ff(f);
  * /FUNCT/20 — the RMS MISSION PROFILE (run-up / dwell / run-down), so the joint
    non-Gaussian tensor composes with the M25 RMS non-stationarity + the M31 continuous
    instantaneous spectrum.

S-N curve: N = C*S^-m with m = 5 and C = 1e4 (identical to the M21/M27 examples).

What /IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/NSTAT does (a PORT sub-flag — the
open-source engine has NO frequency-domain / spectral / multiaxial / non-Gaussian /
time-frequency fatigue solver of any kind; freimpl.F reads only DYNA / BUCKL / DT /
NONLIN / ARCL and its sole "PSD" token is IMUMPSD, a MUMPS flag):
  1..8. everything the M21 STATIONARY multiaxial, the M25 NON-STATIONARY, the M26/M27
     WINDOWED / JOINT-tensor evolutionary, the M31 CONTINUOUS Wigner-Ville and the M32
     TIME-VARYING equivalent-scalar NON-GAUSSIAN paths do — reported UNCHANGED
     (byte-identical whether or not the M33 joint path runs);
  9. THEN, alongside, the M33 JOINT-TENSOR NON-GAUSSIAN analysis: impose the
     PER-COMPONENT kurtoses on the six Voigt tensor components (a vector Hermite /
     translation transform preserving the marginal variances / covariance), compute
     the INDUCED critical-plane kurtosis (closed form) at each instant, scale the
     Gaussian rate by the induced-kurtosis lambda_ng and Miner-INTEGRATE — reporting
     the per-component kurtoses, the induced critical-plane kurtosis, the covariance
     preservation error, and the JOINT-tensor life NEXT TO the M32 equivalent-scalar
     and the M31/M27 Gaussian lives;
  10. a MULTIVARIATE non-Gaussian Monte-Carlo cross-check (the M21 multivariate
     synthesiser pushed through the VECTOR Hermite transform per instant, projected
     onto the per-instant critical plane, ASTM E1049 rainflow over the WHOLE record —
     its resolved-projection sample kurtosis tracking the induced value).

What to look for in the listing (JOINT_NONGAUSSIAN_FATIGUE_0001.out):
  * the M21 / M25 / M26 / M27 / M31 / M32 blocks are UNCHANGED;
  * the "** JOINT-TENSOR NON-GAUSSIAN DISTRIBUTION FATIGUE **
    (/IMPL/FATIG/NGAUSS+JOINT+WVILLE)" block reports the PER-COMPONENT kurtoses g4_c
    (Voigt), the INDUCED critical-plane kurtosis g4^s (the resolved plane inheriting
    the joint tensor non-Gaussianity — DIFFERENT from any single per-component value),
    the covariance PRESERVATION error (the marginals exact, the off-diagonal leading-
    order), and for each reduction the JOINT / SCALAR-EQUIVALENT / GAUSSIAN damage
    rate + life SIDE BY SIDE — the JOINT life DIFFERING from the M32 equivalent-scalar
    one (the M33 <-> M32 boundary), plus the multivariate non-Gaussian Monte-Carlo.

The full result (the M21 reductions + the M25 ``nonstationary``, M26 ``evolutionary``,
M27 ``joint_evolutionary``, M31 ``wigner_ville`` and M32
``wigner_ville['nongaussian']`` sub-entries, with the M33 joint-tensor correction under
``wigner_ville['joint_nongaussian']``) is on ``model.implicit_result.fatigue``.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i JOINT_NONGAUSSIAN_FATIGUE_0000.rad
    pyradioss-engine  -i JOINT_NONGAUSSIAN_FATIGUE_0001.rad
"""

import os
# M36: decks are emitted through the package's fixed-format 2022 writer
# (pyradioss/input/deck_writer.py) — model definition above is unchanged,
# only the emission format moved to the real Radioss dialect.
import sys as _sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from pyradioss.input import deck_writer  # noqa: E402


NX = 8               # bricks along the cantilever axis (x)
WY = 2.0             # cross-section WIDTH in y (mm) — rectangular 2(y) x 1(z)
HZ = 1.0             # cross-section height in z (mm)
E = 210.0            # Young's modulus (GPa)
NU = 0.3             # Poisson ratio (a genuine 3D solid: nu couples the axes)
RHO = 7.8e-6         # density (kg/mm^3)
ZETA = 0.03          # modal damping (3 %) for the FRF
NMODE = 12           # modes to extract (span the bending AND torsion modes)
FMIN = 0.0           # analysis band start (kHz)
FMAX = 800.0         # analysis band end (kHz) — spans bending + torsion modes
NF = 2000            # PSD sweep points
S0 = 1.0             # flat input tip-force PSD (kN^2/kHz)
FSCALE = 1.0e-4      # tip-force pattern amplitude (kN)
SN_M = 5.0           # S-N slope m (N = C*S^-m)
SN_C = 1.0e4         # S-N coefficient C (stress units GPa)
MCDUR = 400.0        # Monte-Carlo cross-check record length (ms)
SEED = 33000         # Monte-Carlo random seed (reproducible)
NPLANE = 24          # candidate-plane azimuth divisions (15 deg)
NSEG = 12            # blocks the RMS mission profile is sampled into (M25)
# M27/M31 joint-tensor drifting-shape schedule: a resonance SWEEP through the modes
FC0 = 40.0           # window centre freq at the first window (Hz, ~1st bending)
FC1 = 600.0          # window centre freq at the last window (Hz, ~torsion/high)
BW0 = 20.0           # window bandwidth start (Hz)
BW1 = 60.0           # window bandwidth end (broadens as the sweep climbs)
NWIN = 6             # time-windows the joint-tensor spectrogram is sampled into
REFINE = 6           # M31 fine instants per window (nt = NWIN * REFINE)
SMOOTH = 0.0         # M31 Cohen-class cross-term smoothing width (0 = raw WVD)
# M32 equivalent-scalar reference kurtosis (imposed on the resolved scalar): the
# per-component MAX, so the M32 reference is the "if we imposed the worst-case kurtosis
# directly on the scalar" answer, printed alongside the M33 JOINT one.
KURT0 = 9.0
# M33 PER-COMPONENT target kurtoses g4_c on the six Voigt components
#   sigma_xx  sigma_yy  sigma_zz  sigma_xy  sigma_yz  sigma_zx
# the AXIAL BENDING stress sigma_xx is strongly leptokurtic (spiky bending transient),
# the TORSION / TRANSVERSE-SHEAR components moderately leptokurtic, the minor
# components near-Gaussian — so the resolved critical plane inherits a JOINT induced
# kurtosis distinct from any single value.
JOINT_KURT = [9.0, 4.0, 3.0, 7.0, 5.0, 6.0]

here = os.path.dirname(os.path.abspath(__file__))


def node_id(ix, iy, iz):
    """Structured-grid node id: (NX+1) along x, 2 along y, 2 along z."""
    return ix * 4 + iy * 2 + iz + 1


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "JOINT_NONGAUSSIAN_FATIGUE - solid-brick cantilever, joint non-Gaussian "
             "tensor (pyradioss /IMPL/FATIG/NGAUSS/JOINT/WVILLE)"]

    # ---- nodes: (NX+1) x 2 x 2 structured grid (rectangular 2y x 1z) --------
    lines.append("/NODE")
    for ix in range(NX + 1):
        for iy in range(2):
            for iz in range(2):
                nid = node_id(ix, iy, iz)
                lines.append(f"{nid:10d}{ix * HZ:20.10f}{iy * WY:20.10f}"
                             f"{iz * HZ:20.10f}")

    # ---- bricks: one hexa8 per axial slot ----------------------------------
    lines.append("/BRICK/1")
    for ix in range(NX):
        n = [node_id(ix, 0, 0), node_id(ix + 1, 0, 0),
             node_id(ix + 1, 1, 0), node_id(ix, 1, 0),
             node_id(ix, 0, 1), node_id(ix + 1, 0, 1),
             node_id(ix + 1, 1, 1), node_id(ix, 1, 1)]
        lines.append(f"{ix + 1:10d}" + "".join(f"{v:10d}" for v in n))

    lines += ["/PART/1", "cantilever", "         1         1",
              "/MAT/LAW1/1", "steel elastic", f"   {RHO}",
              f"     {E}       {NU}",
              "/PROP/SOLID/1", "solid", "         0"]

    # ---- constraints: clamp the x = 0 root face (4 nodes) ------------------
    root = [node_id(0, iy, iz) for iy in range(2) for iz in range(2)]
    tip = [node_id(NX, iy, iz) for iy in range(2) for iz in range(2)]
    root_ids = "\n".join(str(i) for i in root)
    lines += ["/GRNOD/NODE/1", "root", root_ids,
              "/BCS/1", "clamped root",
              "       111       111         0         1"]

    # ---- functions: a UNIT curve for the force PATTERN, and the input PSD ---
    lines += ["/FUNCT/1", "unit pattern curve",
              "       0.0           1.0", "  100000.0           1.0"]
    lines += ["/FUNCT/10", "flat input PSD S_ff(f)",
              f"       0.0{S0:14g}", f"  100000.0{S0:14g}"]
    # ---- /FUNCT/20: the RMS MISSION PROFILE (level vs time) ----------------
    lines += ["/FUNCT/20", "RMS mission profile (run-up / dwell / run-down)",
              "       0.0       0.4",
              "     150.0       1.5",
              "     350.0       1.5",
              "     500.0       0.3"]

    # ---- tip force pattern: OFFSET skew (y + z, unequal across the width) ---
    per = FSCALE / len(tip)
    fz_of = {0: 1.4, 1: 0.6}       # more z-force on the y=0 side (a torsion couple)
    fy_frac = 0.4
    for k, nid in enumerate(tip):
        iy = (nid - 1) // 2 % 2     # the node's y index (0 or 1)
        lines += [f"/GRNOD/NODE/{10 + k}", f"tipn{k}", str(nid),
                  f"/CLOAD/{10 + 2 * k}", "fy",
                  f"         1         Y{10 + k:10d}{per * fy_frac:12g}",
                  f"/CLOAD/{11 + 2 * k}", "fz",
                  f"         1         Z{10 + k:10d}{per * fz_of[iy]:12g}"]
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "JOINT_NONGAUSSIAN_FATIGUE_0000.rad"), runname="JOINT_NONGAUSSIAN_FATIGUE")

    # ---- engine: a bare implicit static step + the M33 joint non-Gaussian ---
    jk = "  ".join(f"{v}" for v in JOINT_KURT)
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/JOINT_NONGAUSSIAN_FATIGUE/1",
        "1.0",                              # one (zero-load) static increment
        "/IMPL",                            # implicit static driver hosts it
        # M33 = M27 joint-tensor (MULT+EVOL+JOINT) + M31 continuous (WVILLE) + M24/M32
        # non-Gaussian (NGAUSS) + M25 non-stationary (NSTAT)
        "/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/NSTAT",
        # line 1 (PSD sweep): fmin fmax nf funct nmode  (funct 10 = S_ff)
        f"{FMIN}  {FMAX}  {NF}  10  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (M24/M32 kurtosis + M33 per-component): kurt skew kfunct kurt1
        #         k_xx k_yy k_zz k_xy k_yz k_zx  (the M32 equivalent-scalar reference
        #         is the constant kurt = KURT0 on col 0; cols 4..9 are the M33 JOINT
        #         per-component targets)
        f"{KURT0}  0.0  0  0  {jk}",
        # line 4 (M25 non-stationary): RMS modulation /FUNCT id + block count
        f"20  {NSEG}",
        # line 5 (M26/M27/M31 drifting-shape + refine/smooth): fc0 fc1 bw0 bw1 nwin
        #         refine smooth  (resonance sweep, continuous instant grid)
        f"{FC0}  {FC1}  {BW0}  {BW1}  {NWIN}  {REFINE}  {SMOOTH}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "JOINT_NONGAUSSIAN_FATIGUE_0001.rad"))
    print("wrote JOINT_NONGAUSSIAN_FATIGUE_0000.rad / "
          "JOINT_NONGAUSSIAN_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
