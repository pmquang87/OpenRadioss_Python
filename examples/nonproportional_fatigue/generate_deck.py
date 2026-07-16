#!/usr/bin/env python3
"""
Generate the NONPROPORTIONAL_FATIGUE example deck: the NON-PROPORTIONAL,
CRITICAL-PLANE, TIME-DOMAIN PATH-COUNTING fatigue life of a base-clamped
RECTANGULAR-section solid-brick cantilever driven by a random tip force applied
in TWO transverse directions — the first pyradioss NON-PROPORTIONAL multiaxial
fatigue analysis (M22, /IMPL/FATIG/MULT/NPROP), built on the M21 multivariate
synthesiser + candidate-plane machinery and the shear-path amplitude / Findley /
Fatemi-Socie machinery of implicit/nonproportional_fatigue.py.

Where the M21 multiaxial_fatigue example drives the SAME kind of cantilever with
a single skew tip force — a PROPORTIONAL multiaxial state (all stress components
rise and fall together, so the M21 spectral projection is exact and the critical
plane is fixed) — M22 makes the load path genuinely NON-PROPORTIONAL: the
rotating-principal-axes case the M21 spectral method deliberately deferred.

HOW A SINGLE RANDOM INPUT PRODUCES A NON-PROPORTIONAL STRESS STATE. The port's
random process is a SINGLE scalar input (one PSD), so the stress-tensor cross-PSD
is rank-1 (M19-M21) — but that does NOT force a proportional TIME history. Two
DIFFERENT structural modes with DIFFERENT stress-component signatures, resonating
at DIFFERENT frequencies, decorrelate the components in time. This deck arranges
exactly that:
  * a RECTANGULAR cross-section (WY = 1 mm wide in y, WZ = 2 mm tall in z) makes
    the two transverse bending stiffnesses UNEQUAL (I_y != I_z), so the "bending
    about y" and "bending about z" modes sit at DISTINCT frequencies (~33 and
    ~60 kHz here);
  * the tip force is applied in BOTH transverse directions (+y AND +z, equal), so
    BOTH bending modes are excited;
  * at the clamped root the transverse shear sigma_xy (from y-bending, ~33 kHz)
    and sigma_zx (from z-bending, ~60 kHz) therefore oscillate at DIFFERENT
    frequencies and drift out of phase — the resolved SHEAR on the cross-section
    plane traces a rotating 2-D PATH (a near-circle), not a line. That is a
    non-proportional stress state: F_np ~ 0.9 (0 = a proportional line, 1 = a
    circle).

Model (units mm / ms / kg -> forces kN, stresses GPa, frequencies kHz):
  * a straight cantilever of NX /BRICK hexa8 solids along +x, a WY x WZ = 1 x 2 mm
    RECTANGULAR cross-section, clamped at the x = 0 face (all 4 root nodes fixed);
  * steel LAW1 (E = 210 GPa, nu = 0.3, rho = 7.8e-6 kg/mm^3);
  * a random tip FORCE PSD applied at the free-end nodes in BOTH +y and +z (equal
    amplitudes) so the two split bending modes are both excited; 3 % modal damping
    at analysis time;
  * /FUNCT/10 — a FLAT band-limited input force PSD S_ff(f) spanning both modes.

What /IMPL/FATIG/MULT/NPROP does (a PORT sub-flag — the open-source engine has NO
frequency-domain / spectral / critical-plane / path-counting fatigue solver of
any kind; freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL and its sole
"PSD" token is IMUMPSD, a MUMPS flag):
  1.-4. everything the M21 /IMPL/FATIG/MULT does — build the FRF, recover the
     6-Voigt stress modes, form the stress-tensor cross-PSD, and report the
     equivalent VON MISES / MAX-NORMAL / MAX-SHEAR spectral reductions (the
     PROPORTIONAL-spectral answer);
  5. THEN, ALONGSIDE, synthesise the correlated Gaussian stress-component
     histories from the critical element's cross-PSD (the M21 multivariate
     spectral-representation synthesiser, seeded), and on each candidate plane
     resolve the NORMAL history sigma_n(t) and the 2-D SHEAR PATH (tau_a, tau_b);
  6. measure the shear-path amplitude three ways — MINIMUM CIRCUMSCRIBED CIRCLE
     (Papadopoulos), LONGEST CHORD, MAXIMUM RECTANGULAR HULL (Mamiya-Araujo) —
     and the non-proportionality factor F_np, then search the critical plane and
     count the TIME-DOMAIN damage by FINDLEY (tau_a + k sigma_n,max), FATEMI-SOCIE
     (gamma_a(1 + k sigma_n,max/sigma_y)) and the pure shear-path model, with the
     per-plane max normal stress folded in.

S-N curve: N = C*S^-m with m = 5 (a typical steel slope) and C = 1e4 (stress
units GPa over the ms time base) — the same as the M21 example so the
NON-PROPORTIONAL lives are directly comparable to the M21 PROPORTIONAL-spectral
lives.

What to look for in the listing (NONPROPORTIONAL_FATIGUE_0001.out):
  * the M21 "** MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE **" block (the
    proportional-spectral answer) — UNCHANGED, printed first;
  * then the M22 "** NON-PROPORTIONAL MULTIAXIAL FATIGUE **" block:
    - the SHEAR-PATH AMPLITUDE line: MCC ~ CHORD/2 < MRH (the MRH is ~sqrt(2)
      larger on a rotating path — the extra amplitude the scalar projection and
      the MCC miss);
    - the NON-PROPORTIONALITY F_np ~ 0.9 (the rotating shear path is near a
      circle — the multiaxial signature the SPLIT bending modes create);
    - the FINDLEY / FATEMI-SOCIE / SHEAR-PATH critical planes: the critical plane
      is the cross-section plane (normal ~ the beam axis x), where the two split
      transverse shears rotate; the path factor g ~ 1.4 (the MRH / scalar ratio),
      so the NON-PROPORTIONAL damage is ~g^m higher — and the LIFE shorter — than
      a scalar projection of the same state would report (the whole point).

The full non-proportional result (the three critical-plane models with their
critical plane / sigma_n,max / shear amplitude / F_np / damage / life, the
amplitude comparison and the seed) is on
``model.implicit_result.fatigue['nprop_result']`` (alongside the M21 spectral
dict). The M21 spectral reductions are byte-identical whether or not NPROP runs.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i NONPROPORTIONAL_FATIGUE_0000.rad
    pyradioss-engine  -i NONPROPORTIONAL_FATIGUE_0001.rad
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


NX = 6               # bricks along the cantilever axis (x)
WY = 1.0             # cross-section width in y (mm) — the RECTANGULAR section...
WZ = 2.0             # cross-section height in z (mm) — ...WY != WZ splits the two
#                      transverse bending frequencies (the non-proportional key)
E = 210.0            # Young's modulus (GPa)
NU = 0.3             # Poisson ratio (a genuine 3D solid: nu couples the axes)
RHO = 7.8e-6         # density (kg/mm^3)
ZETA = 0.03          # modal damping (3 %) for the FRF
NMODE = 8            # modes to extract (span both split bending modes)
FMIN = 0.0           # analysis band start (kHz)
FMAX = 500.0         # analysis band end (kHz) — spans both bending modes
NF = 800             # PSD sweep points
S0 = 1.0             # flat input tip-force PSD (kN^2/kHz)
FSCALE = 1.0e-4      # tip-force pattern amplitude (kN) — keeps the root stress
#                      in the ~20 MPa high-cycle-fatigue range
FY = 1.0             # tip-force y fraction (excites bending about z, ~60 kHz)
FZ = 1.0             # tip-force z fraction (excites bending about y, ~33 kHz)
#                      BOTH nonzero -> both split modes -> non-proportional root
SN_M = 5.0           # S-N slope m (N = C*S^-m) — matches the M21 example
SN_C = 1.0e4         # S-N coefficient C (stress units GPa) — matches M21
K_NP = 0.3           # Findley / Fatemi-Socie normal-sensitivity constant k
SIGY = 0.5           # yield stress sigma_y (GPa) for Fatemi-Socie
MCDUR = 60.0         # synthesised-history / path-count record length (ms)
SEED = 22000         # random seed (reproducible)

here = os.path.dirname(os.path.abspath(__file__))


def node_id(ix, iy, iz):
    """Structured-grid node id: (NX+1) along x, 2 along y, 2 along z."""
    return ix * 4 + iy * 2 + iz + 1


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "NONPROPORTIONAL_FATIGUE - rectangular-section solid-brick "
             "cantilever (pyradioss /IMPL/FATIG/MULT/NPROP)"]

    # ---- nodes: (NX+1) x 2 x 2 structured grid, RECTANGULAR WY x WZ section ---
    lines.append("/NODE")
    for ix in range(NX + 1):
        for iy in range(2):
            for iz in range(2):
                nid = node_id(ix, iy, iz)
                lines.append(f"{nid:10d}{ix * 1.0:20.10f}{iy * WY:20.10f}"
                             f"{iz * WZ:20.10f}")

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

    # ---- tip force pattern: BOTH transverse directions (+y AND +z) ---------
    # a single random process (one /FUNCT PSD) drives BOTH directions; the two
    # SPLIT bending modes it excites at different frequencies are what make the
    # root stress non-proportional (module / example docstring)
    per = FSCALE / len(tip)
    for k, nid in enumerate(tip):
        lines += [f"/GRNOD/NODE/{10 + k}", f"tipn{k}", str(nid),
                  f"/CLOAD/{10 + 2 * k}", "fy",
                  f"         1         Y{10 + k:10d}{per * FY:12g}",
                  f"/CLOAD/{11 + 2 * k}", "fz",
                  f"         1         Z{10 + k:10d}{per * FZ:12g}"]
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "NONPROPORTIONAL_FATIGUE_0000.rad"), runname="NONPROPORTIONAL_FATIGUE")

    # ---- engine: a bare implicit static step + the M22 analysis ------------
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/NONPROPORTIONAL_FATIGUE/1",
        "1.0",                       # one (zero-load) static increment
        "/IMPL",                     # implicit static driver hosts the analysis
        "/IMPL/FATIG/MULT/NPROP",    # M22: non-proportional critical-plane path
        # line 1 (PSD sweep): fmin fmax nf funct nmode  (funct 10 = S_ff)
        f"{FMIN}  {FMAX}  {NF}  10  {NMODE}",
        # line 2 (S-N + path count): m C zeta mean ult mcdur seed k sigma_y
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}  {K_NP}  {SIGY}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "NONPROPORTIONAL_FATIGUE_0001.rad"))
    print("wrote NONPROPORTIONAL_FATIGUE_0000.rad / "
          "NONPROPORTIONAL_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
