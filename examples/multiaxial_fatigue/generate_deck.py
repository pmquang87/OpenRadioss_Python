#!/usr/bin/env python3
"""
Generate the MULTIAXIAL_FATIGUE example deck: the MULTIAXIAL / CRITICAL-PLANE
spectral fatigue life of a base-clamped SOLID-brick cantilever driven by a
random tip force PSD — the first pyradioss MULTIAXIAL (stress-tensor) fatigue
analysis (M21, /IMPL/FATIG/MULT), built on the M20 vector stress modes (the
full 6-component Voigt stress FRF per element) and the multiaxial reductions of
implicit/multiaxial_fatigue.py (equivalent von Mises + max-normal / max-shear
critical plane).

Where M20's spectral_fatigue example asks the fatigue question of a SCALAR
stress-resultant channel (a spring force), M21 asks it of a genuine MULTIAXIAL
stress STATE. A cantilever loaded transversely at the tip develops, at its
CLAMPED ROOT, a combined stress state — bending sigma_xx PLUS a transverse
shear sigma_zx (and, off the neutral axis, some sigma_yy / sigma_xy) — so the
von Mises equivalent stress differs from any single component and the critical
plane is NOT a coordinate plane. This is the classic multiaxial high-cycle
fatigue setting (Preumont & Piefort 1994; Pitoiset & Preumont 2000; Socie &
Marquis, "Multiaxial Fatigue").

Model (units mm / ms / kg -> forces kN, stresses GPa, frequencies kHz):
  * a straight cantilever of NX /BRICK hexa8 solids along +x, a 1x1 mm square
    cross-section, clamped at the x = 0 face (all 4 root nodes fully fixed);
  * steel LAW1 (E = 210 GPa, nu = 0.3, rho = 7.8e-6 kg/mm^3);
  * a random tip FORCE PSD applied at the free-end nodes in a SKEW transverse
    direction (z with a little y) so the root elements see a bending + shear
    (genuinely multiaxial) stress; light modal damping (3 %) at analysis time;
  * /FUNCT/10 — a FLAT band-limited input force PSD S_ff(f).

What /IMPL/FATIG/MULT does (a PORT sub-card — the open-source engine has NO
frequency-domain / spectral-fatigue solver of any kind, scalar OR multiaxial;
freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL and its sole "PSD" token
is IMUMPSD, a MUMPS flag):
  1. extract the modes and build the force-excitation FRF H(Omega) (the M17
     real-mode machinery);
  2. recover the per-mode STRESS modes by running the force kernels on each
     mode shape (read-only, the M20 recovery), keeping the FULL 6-component
     Voigt stress FRF H_sigma(Omega) = sum_i sigma_i q_i(Omega) per element;
  3. form the stress-tensor cross-PSD S_sigmasigma(Omega) = H_sigma S_ff
     H_sigma^H (a 6x6 Hermitian matrix per frequency per element);
  4. reduce it to a scalar EQUIVALENT-stress PSD three ways — equivalent VON
     MISES (Preumont-Piefort, the trace(Q S) projection), MAX-NORMAL-stress and
     MAX-SHEAR-stress CRITICAL PLANE (searched over candidate planes) — and run
     the S-N curve N = C*S^-m under a Miner sum by the narrow-band (Bendat),
     Dirlik, Wirsching-Light and Tovo-Benasciutti estimators on each;
  5. pick the CRITICAL ELEMENT (highest von Mises Dirlik damage — the clamped
     root) and report the damage rate / life for every reduction, plus the
     critical-plane orientations;
  6. run a seeded Monte-Carlo cross-check: synthesise the 6 CORRELATED Gaussian
     stress-component histories from the cross-PSD (multivariate spectral
     representation via a per-bin eigendecomposition / Cholesky of S_sigmasigma),
     project onto the max-shear critical plane, ASTM E1049 rainflow, Miner.

S-N curve: N = C*S^-m with m = 5 (a typical steel slope) and C = 1e4 (in the
stress units GPa over the ms time base) — chosen so the critical root element
(RMS von Mises ~ 0.02 GPa = 20 MPa) shows a finite, physically-legible life
under this PSD.

What to look for in the listing (MULTIAXIAL_FATIGUE_0001.out), the
"** MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE ** (/IMPL/FATIG/MULT)" block:
  * the CRITICAL ELEMENT is a clamped-root brick (the maximum bending + shear);
  * three reduction sub-blocks — EQUIVALENT VON MISES, MAX-NORMAL-STRESS
    CRITICAL PLANE and MAX-SHEAR-STRESS CRITICAL PLANE — each with its RMS
    equivalent stress, the crossing / peak rates, the irregularity factor and
    the four-estimator damage / life table;
  * the critical-plane NORMALS: the MAX-NORMAL plane is TILTED (~[0.68 0.18
    0.71], NOT a coordinate axis — the multiaxial signature of a combined
    bending + shear state, where a purely uniaxial stress would sit on the
    loading axis); the MAX-SHEAR plane is the cross-section plane (normal ~ the
    beam axis x), where the transverse shear resolves;
  * the von-Mises life is the SHORTEST (it aggregates all deviatoric
    components); the two critical-plane lives are LONGER (each counts only one
    resolved scalar) and comparable to each other here;
  * the MONTE-CARLO row (the correlated-history rainflow on the shear plane)
    agrees with that plane's Dirlik within the documented scatter.

The full multiaxial result (per-element von Mises damage, the three reductions
with their moments / damage / life / critical-plane orientation, the 6x6
cross-PSD moment matrices and the Monte-Carlo cross-check) is on
``model.implicit_result.fatigue`` (the M20 dict, extended with ``multiaxial``).

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i MULTIAXIAL_FATIGUE_0000.rad
    pyradioss-engine  -i MULTIAXIAL_FATIGUE_0001.rad
"""

import os

NX = 6               # bricks along the cantilever axis (x)
H = 1.0              # cross-section side (mm) — 1x1 square
E = 210.0            # Young's modulus (GPa)
NU = 0.3             # Poisson ratio (a genuine 3D solid: nu couples the axes)
RHO = 7.8e-6         # density (kg/mm^3)
ZETA = 0.03          # modal damping (3 %) for the FRF
NMODE = 8            # modes to extract
FMIN = 0.0           # analysis band start (kHz)
FMAX = 400.0         # analysis band end (kHz) — spans the low bending modes
NF = 1500            # PSD sweep points
S0 = 1.0             # flat input tip-force PSD (kN^2/kHz)
FSCALE = 1.0e-4      # tip-force pattern amplitude (kN) — 1 kN on 1 mm^2 = 1 GPa,
#                      so a small pattern keeps the root stress in the ~20 MPa
#                      high-cycle-fatigue range
FY = 0.3             # tip-force y fraction (skew load -> shear + biaxial root)
FZ = 1.0             # tip-force z fraction (transverse bending)
SN_M = 5.0           # S-N slope m (N = C*S^-m)
SN_C = 1.0e4         # S-N coefficient C (stress units GPa)
MCDUR = 400.0        # Monte-Carlo cross-check record length (ms)
SEED = 21000         # Monte-Carlo random seed (reproducible)
NPLANE = 24          # candidate-plane azimuth divisions (15 deg)

here = os.path.dirname(os.path.abspath(__file__))


def node_id(ix, iy, iz):
    """Structured-grid node id: (NX+1) along x, 2 along y, 2 along z."""
    return ix * 4 + iy * 2 + iz + 1


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "MULTIAXIAL_FATIGUE - solid-brick cantilever "
             "(pyradioss /IMPL/FATIG/MULT)"]

    # ---- nodes: (NX+1) x 2 x 2 structured grid -----------------------------
    lines.append("/NODE")
    for ix in range(NX + 1):
        for iy in range(2):
            for iz in range(2):
                nid = node_id(ix, iy, iz)
                lines.append(f"{nid:10d}{ix * H:20.10f}{iy * H:20.10f}"
                             f"{iz * H:20.10f}")

    # ---- bricks: one hexa8 per axial slot ----------------------------------
    # hexa8 node order: bottom face (z=0) CCW then top face (z=1) CCW
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
    tip_ids = "\n".join(str(i) for i in tip)
    lines += ["/GRNOD/NODE/1", "root", root_ids,
              "/GRNOD/NODE/2", "tip", tip_ids,
              "/BCS/1", "clamped root",
              "       111       111         0         1"]

    # ---- functions: a UNIT curve for the force PATTERN, and the input PSD ---
    # IMPORTANT: the /CLOAD spatial pattern references the UNIT curve /FUNCT/1
    # (amplitude 1), NOT the PSD — otherwise the input magnitude is applied
    # TWICE (once as the FRF force amplitude, once as S_ff). The PSD /FUNCT/10
    # feeds the fatigue card only.
    lines += ["/FUNCT/1", "unit pattern curve",
              "       0.0           1.0", "  100000.0           1.0"]
    lines += ["/FUNCT/10", "flat input PSD S_ff(f)",
              f"       0.0{S0:14g}", f"  100000.0{S0:14g}"]

    # ---- tip force pattern: skew transverse (z + a little y) ---------------
    # split the FSCALE pattern across the 4 tip nodes (the spatial force pattern
    # of the random process); /CLOAD card order is: fct_ID  Dir  grnod_ID Fscale
    per = FSCALE / len(tip)
    for k, nid in enumerate(tip):
        lines += [f"/GRNOD/NODE/{10 + k}", f"tipn{k}", str(nid),
                  f"/CLOAD/{10 + 2 * k}", "fy",
                  f"         1         Y{10 + k:10d}{per * FY:12g}",
                  f"/CLOAD/{11 + 2 * k}", "fz",
                  f"         1         Z{10 + k:10d}{per * FZ:12g}"]
    lines.append("/END")

    with open(os.path.join(here, "MULTIAXIAL_FATIGUE_0000.rad"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- engine: a bare implicit static step + the M21 multiaxial analysis -
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/MULTIAXIAL_FATIGUE/1",
        "1.0",                       # one (zero-load) static increment
        "/IMPL",                     # implicit static driver hosts the analysis
        "/IMPL/FATIG/MULT",          # M21: multiaxial / critical-plane fatigue
        # line 1 (PSD sweep): fmin fmax nf funct nmode  (funct 10 = S_ff)
        f"{FMIN}  {FMAX}  {NF}  10  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    with open(os.path.join(here, "MULTIAXIAL_FATIGUE_0001.rad"), "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print("wrote MULTIAXIAL_FATIGUE_0000.rad / MULTIAXIAL_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
