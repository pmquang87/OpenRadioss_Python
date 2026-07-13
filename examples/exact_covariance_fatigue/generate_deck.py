#!/usr/bin/env python3
"""
Generate the EXACT_COVARIANCE_FATIGUE example deck: the EXACT translation-process
CORRELATION-DISTORTION INVERSION fatigue life of a base-clamped SOLID-brick cantilever
whose STRONGLY CROSS-CORRELATED, strongly leptokurtic 6x6 stress-TENSOR components make
the M33 leading-order covariance distortion VISIBLE — the Grigoriu / Nataf / Cario-Nelson
NORTA correlation matching that solves the underlying-Gaussian correlation rho^U so the
component-wise Winterstein-Hermite (translation) transform reproduces the TARGET 6x6
covariance EXACTLY (not merely to M33's leading order), driving the covariance
preservation error to ~0 (M34, /IMPL/FATIG/NGAUSS/JOINT/WVILLE/EXACT) — the FIRST item
M33 deferred.

This is deliberately the SAME solid-brick cantilever the M33 joint_nongaussian_fatigue
example builds on, RE-RUN with a STRONGER cross-correlation (a nearly-proportional skew
tip load so the axial-bending, in-plane-shear and transverse-shear components all TRACK
together) and STRONG per-component kurtoses on those CORRELATED components — the regime
where the M33 memoryless (translation) transform distorts the cross-covariance the MOST.
Where M33 imposed the per-component kurtoses on a tensor whose underlying-Gaussian
correlation was TAKEN as the target correlation (the marginals exact, the cross-covariance
LEADING-ORDER, the residual reported as a preservation-error diagnostic), M34 INVERTS the
translation distortion PER COMPONENT PAIR: for each (c, c') it solves the cubic
correlation-matching equation

    kappa_c kappa_c' [ rho^U + 2 h3_c h3_c' rho^U^2 + 6 h4_c h4_c' rho^U^3 ] = R_cc'

for the underlying rho^U (a monotone root-find), assembles the 6x6 rho^U and REPAIRS it
to the nearest valid positive-definite correlation matrix (Higham 2002), so the joint
non-Gaussian tensor has EXACTLY the target covariance AND the per-component kurtoses. The
induced critical-plane kurtosis / damage is then computed on a COVARIANCE-EXACT joint
tensor, cross-validated by the M33 multivariate non-Gaussian Monte-Carlo synthesised with
the CORRECTED underlying correlation (so the record's SAMPLE covariance matches the target
where M33's drifted). Theory: Grigoriu translation-process correlation distortion
(Grigoriu 1995/1998); the Nataf transformation (Nataf 1962; Der Kiureghian & Liu 1986);
Cario & Nelson NORTA (1997); Vale & Maurelli 1983; Higham 2002 (nearest correlation
matrix); the M33 leading-order base.

THE DEMONSTRATOR: a strongly-correlated, spiky multiaxial transient. The excitation window
sweeps the modal band (fc 40 -> 600 Hz, the M31/M32 continuous instant spectrum) while a
nearly-proportional skew tip load makes the root stress tensor STRONGLY CROSS-CORRELATED,
and the dominant CORRELATED components carry STRONG kurtoses (the axial bending sigma_xx
g4 ~ 9, the in-plane / transverse shear sigma_xy / sigma_zx g4 ~ 8). In this regime the
M33 leading-order translation distorts the transformed cross-covariance by a VISIBLE
amount (a few percent off-diagonal); the M34 NORTA inversion drives that preservation
error to ~0. The block prints the EXACT covariance preservation error NEXT TO the M33
leading-order one, and the EXACT-covariance induced kurtosis / damage / life ALONGSIDE the
M33 leading-order joint, the M32 equivalent-scalar and the M31/M27 Gaussian lives.

Model (units mm / ms / kg -> forces kN, stresses GPa, frequencies kHz): IDENTICAL to the
M33 joint_nongaussian_fatigue example — a rectangular (2y x 1z) solid-brick cantilever
with a skew tip force so the root elements see bending + transverse shear + torsion, but
with the tip pattern tuned NEARLY PROPORTIONAL (fy and fz strongly coupled on the same
node) so the resulting stress-tensor components are strongly cross-correlated:
  * a straight cantilever of NX /BRICK hexa8 solids along +x, clamped at x = 0;
  * steel LAW1 (E = 210 GPa, nu = 0.3, rho = 7.8e-6 kg/mm^3);
  * a random tip FORCE PSD, skew (y + z) but nearly proportional (strong correlation);
  * /FUNCT/10 — a FLAT band-limited input force PSD S_ff(f);
  * /FUNCT/20 — the RMS MISSION PROFILE (run-up / dwell / run-down).

S-N curve: N = C*S^-m with m = 5 and C = 1e4 (identical to the M21/M27/M33 examples).

What /IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/EXACT/NSTAT does (a PORT sub-flag — the
open-source engine has NO frequency-domain / spectral / multiaxial / non-Gaussian /
translation / copula fatigue solver of any kind; freimpl.F reads only DYNA / BUCKL / DT /
NONLIN / ARCL and its sole "PSD" token is IMUMPSD, a MUMPS flag):
  1..9. everything the M21 STATIONARY multiaxial, the M25 NON-STATIONARY, the M26/M27
     WINDOWED / JOINT-tensor evolutionary, the M31 CONTINUOUS Wigner-Ville, the M32
     TIME-VARYING equivalent-scalar and the M33 JOINT-tensor leading-order NON-GAUSSIAN
     paths do — reported UNCHANGED (byte-identical whether or not the M34 exact path runs);
  10. THEN, alongside, the M34 EXACT-covariance analysis: solve the NORTA underlying
     correlation rho^U per component pair (+ Higham PD repair), recompute the induced
     critical-plane kurtosis / lambda_ng on the COVARIANCE-EXACT joint tensor, Miner-
     INTEGRATE, and report the (now ~0) covariance preservation error and the exact-vs-
     leading-order induced kurtosis / damage / life;
  11. the CORRECTED multivariate non-Gaussian Monte-Carlo (synthesised with the underlying
     correlation rho^U so the transformed record's SAMPLE covariance matches the target).

What to look for in the listing (EXACT_COVARIANCE_FATIGUE_0001.out):
  * the M21 / M25 / M26 / M27 / M31 / M32 / M33 blocks are UNCHANGED;
  * the "** JOINT-TENSOR NON-GAUSSIAN DISTRIBUTION FATIGUE **" block (M33) reports a
    VISIBLE covariance PRESERVATION error (a few percent off-diagonal — the strong-
    correlation regime);
  * the "** EXACT TRANSLATION-PROCESS CORRELATION-DISTORTION INVERSION **
    (/IMPL/FATIG/NGAUSS+JOINT+WVILLE+EXACT)" block (M34) reports the NORTA correlation
    matching + Higham repair, the COVARIANCE PRESERVATION EXACT / M33 error (the exact
    driven to ~1e-15 next to the M33 leading-order few-percent), the EXACT induced
    critical-plane kurtosis, and for each reduction the EXACT / M33-LEADING damage rate +
    life SIDE BY SIDE, plus the covariance-exact Monte-Carlo (its sample-covariance error
    ~0 where the M33 record's drifted).

The full result (the M21 reductions + the M25 ``nonstationary``, M26 ``evolutionary``,
M27 ``joint_evolutionary``, M31 ``wigner_ville``, M32 ``wigner_ville['nongaussian']`` and
M33 ``wigner_ville['joint_nongaussian']`` sub-entries, with the M34 covariance-exact
correction under ``wigner_ville['joint_nongaussian']['exact_covariance']``) is on
``model.implicit_result.fatigue``.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i EXACT_COVARIANCE_FATIGUE_0000.rad
    pyradioss-engine  -i EXACT_COVARIANCE_FATIGUE_0001.rad
"""

import os

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
SEED = 34000         # Monte-Carlo random seed (reproducible)
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
# directly on the scalar" answer, printed alongside the M33/M34 JOINT ones.
KURT0 = 9.0
# M34 PER-COMPONENT target kurtoses g4_c on the six Voigt components
#   sigma_xx  sigma_yy  sigma_zz  sigma_xy  sigma_yz  sigma_zx
# STRONG kurtoses on the strongly-CORRELATED dominant components (axial bending sigma_xx
# and the two shear components sigma_xy / sigma_zx) so the M33 leading-order translation
# distorts the cross-covariance the MOST — the regime the M34 NORTA inversion corrects.
JOINT_KURT = [9.0, 5.0, 4.0, 8.0, 5.0, 8.0]

here = os.path.dirname(os.path.abspath(__file__))


def node_id(ix, iy, iz):
    """Structured-grid node id: (NX+1) along x, 2 along y, 2 along z."""
    return ix * 4 + iy * 2 + iz + 1


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "EXACT_COVARIANCE_FATIGUE - solid-brick cantilever, NORTA covariance-exact "
             "joint non-Gaussian tensor (pyradioss /IMPL/FATIG/NGAUSS/JOINT/WVILLE/EXACT)"]

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

    # ---- tip force pattern: NEARLY PROPORTIONAL skew (y + z coupled on the
    #      same node) so the root stress-tensor components are STRONGLY CROSS-
    #      CORRELATED (the regime where the M33 translation distortion is largest) --
    per = FSCALE / len(tip)
    fy_frac = 0.7                  # a strong, common y-component on every tip node
    fz_frac = 1.0                  # a common z-component too (proportional -> correlated)
    for k, nid in enumerate(tip):
        lines += [f"/GRNOD/NODE/{10 + k}", f"tipn{k}", str(nid),
                  f"/CLOAD/{10 + 2 * k}", "fy",
                  f"         1         Y{10 + k:10d}{per * fy_frac:12g}",
                  f"/CLOAD/{11 + 2 * k}", "fz",
                  f"         1         Z{10 + k:10d}{per * fz_frac:12g}"]
    lines.append("/END")

    with open(os.path.join(here, "EXACT_COVARIANCE_FATIGUE_0000.rad"),
              "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- engine: a bare implicit static step + the M34 exact-covariance joint --
    jk = "  ".join(f"{v}" for v in JOINT_KURT)
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/EXACT_COVARIANCE_FATIGUE/1",
        "1.0",                              # one (zero-load) static increment
        "/IMPL",                            # implicit static driver hosts it
        # M34 = M33 joint-tensor (MULT+EVOL+JOINT) + M31 continuous (WVILLE) + M24/M32
        # non-Gaussian (NGAUSS) + the EXACT NORTA correlation inversion (EXACT) + M25
        # non-stationary (NSTAT)
        "/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/EXACT/NSTAT",
        # line 1 (PSD sweep): fmin fmax nf funct nmode  (funct 10 = S_ff)
        f"{FMIN}  {FMAX}  {NF}  10  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (M24/M32 kurtosis + M33/M34 per-component): kurt skew kfunct kurt1
        #         k_xx k_yy k_zz k_xy k_yz k_zx  (the M32 equivalent-scalar reference is
        #         the constant kurt = KURT0 on col 0; cols 4..9 are the JOINT per-
        #         component targets the M34 exact inversion operates on)
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
    with open(os.path.join(here, "EXACT_COVARIANCE_FATIGUE_0001.rad"),
              "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print("wrote EXACT_COVARIANCE_FATIGUE_0000.rad / "
          "EXACT_COVARIANCE_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
