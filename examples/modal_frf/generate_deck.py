#!/usr/bin/env python3
"""
Generate the MODALFRF example deck: the harmonic FREQUENCY RESPONSE (FRF) of a
shaker-driven cantilever mast — the first pyradioss frequency-domain analysis
(M17, /IMPL/FREQ), built on the M16 modal eigenpairs and the complex modal
transfer function of implicit/modal_response.py.

Model (units mm / ms / kg -> forces kN, stresses GPa, frequencies kHz):
  * the SAME vertical steel tube mast as ``modal_mast`` / ``antenna_mast`` —
    height 2000 mm, D = 60.3 x t = 4 mm (A = 707.5 mm^2, I = 2.818e5 mm^4),
    20 /BEAM elements, base clamped;
  * a unit lateral harmonic force at the tip (a shaker): the /CLOAD pattern is
    the harmonic FORCE AMPLITUDE, and /IMPL/FREQ sweeps the forcing frequency
    Omega across the first two bending resonances.

What /IMPL/FREQ does (a PORT card — the open-source engine has NO
frequency-domain path; freimpl.F reads only DYNA / BUCKL / DT / ARCL):
  extract the lowest modes (K - omega^2 M) phi = 0 on the CONSISTENT mass,
  then evaluate the steady-state complex FRF

      q_i(Omega) = (phi_i^T F) / (omega_i^2 - Omega^2 + 2 i zeta_i omega_i
                                  Omega),
      u(Omega)   = sum_i phi_i q_i(Omega)

  over the swept band, reporting amplitude and phase. At each resonance
  Omega -> omega_i the tip amplitude spikes to 1/(2 zeta_i) of its static
  deflection (the half-power bandwidth Delta_Omega/omega = 2 zeta_i);
  test_m17_modalresp.py asserts these.

What to look for in the listing (MODALFRF_0001.out):
  * the "** FREQUENCY RESPONSE (harmonic) ** (/IMPL/FREQ)" block prints the
    sweep band and the RESONANCES (the natural frequencies the FRF peaks
    coincide with) — the degenerate bending pair f1 ~ 0.0145 kHz and f2 ~
    0.0915 kHz. The complex FRF (u(Omega), amplitude/phase) is stored on
    model.implicit_result.freq_response for post-processing.

The response is a steady-state amplitude PER FREQUENCY, not a time history:
between the two resonances the amplitude dips (the anti-resonance region) and
the phase rolls through 180 deg across each peak — the signature of a damped
MDOF FRF.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i MODALFRF_0000.rad
    pyradioss-engine  -i MODALFRF_0001.rad
"""

import os

NEL = 20             # beam elements along the mast
H = 2000.0           # mast height (mm)
AREA = 707.5         # tube section area (mm^2)
IYY = 2.818e5        # bending inertia (mm^4), both axes
IXX = 2.0 * IYY      # torsion constant of a thin tube = polar = 2I
NMODE = 6            # modes retained in the superposition
FMIN = 0.001         # sweep start (kHz)
FMAX = 0.12          # sweep end (kHz) — just past the 2nd bending resonance
NF = 4000            # sweep points
ZETA = 0.01          # uniform modal damping (1 %)

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "MODALFRF - frequency response of a shaker-driven cantilever "
             "mast (pyradioss /IMPL/FREQ)"]

    # ---- nodes: mast along +z, plus one orientation node ------------------
    lines.append("/NODE")
    for i in range(NEL + 1):
        z = i * H / NEL
        lines.append(f"{i + 1:10d}{0.0:20.10f}{0.0:20.10f}{z:20.10f}")
    lines.append(f"{99:10d}{0.0:20.10f}{1000.0:20.10f}{0.0:20.10f}")

    # ---- beams: N1 N2 + shared orientation node 99 ------------------------
    lines.append("/BEAM/1")
    for i in range(NEL):
        lines.append(f"{i + 1:10d}{i + 1:10d}{i + 2:10d}{99:10d}")

    lines += ["/PART/1", "mast tube", "         1         1",
              "/MAT/LAW1/1", "steel elastic", "   7.8e-6", "     210.0       0.3",
              "/PROP/BEAM/1", "tube D60.3x4",
              f"{AREA:10.4g}{IYY:10.4g}{IYY:10.4g}{IXX:10.4g}"]

    # ---- clamp the base ---------------------------------------------------
    lines += ["/GRNOD/NODE/1", "base", "1",
              "/BCS/1", "clamp base (translations + rotations)",
              "       111       111         0         1"]

    # ---- a unit lateral harmonic force at the tip (the shaker pattern) ----
    lines += ["/GRNOD/NODE/2", "tip", f"{NEL + 1}",
              "/FUNCT/1", "unit amplitude", "       0.0       1.0",
              "  100000.0       1.0",
              "/CLOAD/1", "tip lateral shaker",
              f"         1         Y         2       1.0"]
    lines.append("/END")

    with open(os.path.join(here, "MODALFRF_0000.rad"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- engine: a bare implicit static step + the FRF sweep --------------
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/MODALFRF/1",
        "1.0",                       # one (zero-load) static increment
        "/IMPL",                     # implicit static driver hosts the sweep
        "/IMPL/FREQ",                # M17: harmonic frequency response
        f"{FMIN}  {FMAX}  {NF}  {ZETA}  {NMODE}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    with open(os.path.join(here, "MODALFRF_0001.rad"), "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print("wrote MODALFRF_0000.rad / MODALFRF_0001.rad")


if __name__ == "__main__":
    main()
