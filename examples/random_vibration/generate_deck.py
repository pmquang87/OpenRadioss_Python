#!/usr/bin/env python3
"""
Generate the RANDOM_VIBRATION example deck: the RANDOM / SPECTRAL (PSD)
response and the RESPONSE SPECTRUM (SRSS/CQC) of a base-excited multi-mass
"instrument stack" — the first pyradioss stochastic / envelope analysis (M19,
/IMPL/PSD + /IMPL/RSPEC), built on the M16/M17 modal FRF (random_response.py)
and the M16 participation (response_spectrum.py).

Why a base-excited spring-mass stack? It is the canonical shaker-table /
seismic demonstrator (Newland, "An Introduction to Random Vibrations...";
Chopra, "Dynamics of Structures", ch. 13). A fixed-free chain of masses shaken
at its BASE is exactly the multi-DOF oscillator a random-vibration qualification
test (MIL-STD-810 / launch acceptance) or a response-spectrum seismic design
addresses: the same modes, driven by a random support acceleration (a PSD) or
enveloped by a design response spectrum.

Model (units mm / ms / kg -> forces kN, frequencies kHz):
  * N = 5 masses: node 1 is the shaken BASE (pinned — the support), nodes 2..6
    the stacked masses along +x, one /SPRING per storey (its own /PART +
    /PROP/SPRING 'M K C');
  * each storey: mass M = 2e-3 kg (half lumps onto each node), stiffness
    K = 800 kN/mm, a light modal damping added at analysis time (zeta = 3 %);
  * /FUNCT/10 — a FLAT band-limited input PSD S_aa(f) (a white base-
    acceleration over the analysis band), for /IMPL/PSD/BASE;
  * /FUNCT/20 — a plateau DESIGN RESPONSE SPECTRUM Sa(f) (a smoothed seismic /
    shock envelope), for /IMPL/RSPEC.

What /IMPL/PSD/BASE does (a PORT card — the open-source engine has NO random-
vibration path; freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL):
  extract the modes, build the base-excitation FRF H(Omega) (the M16
  participation feed), form the stationary response PSD
  S_uu(Omega) = |H(Omega)|^2 S_aa(Omega), and integrate the SPECTRAL MOMENTS
  m_0, m_1, m_2 to report the RMS relative displacement sigma_u = sqrt(m_0)
  per DOF and the mean zero-crossing rate nu_0 = (1/2pi) sqrt(m_2/m_0).

What /IMPL/RSPEC does (a PORT card too): combine the per-mode peaks
  r_i = Gamma_i Sa(omega_i)/omega_i^2 phi_i (the participation-scaled spectral
  ordinate) by SRSS and by CQC (Der Kiureghian 1981, the closed-form modal
  correlation rho_ij), reporting the peak modal-combination response envelope.

What to look for in the listing (RANDOM_VIBRATION_0001.out):
  * the "** RANDOM / SPECTRAL (PSD) RESPONSE ** (/IMPL/PSD)" block prints the
    peak RMS response, the spectral moments m0/m1/m2 at that DOF, and the mean
    zero-crossing rate (the "apparent frequency" of the random response — it
    sits near the fundamental for this narrow-band-dominated stack);
  * the "** RESPONSE SPECTRUM (modal combination) ** (/IMPL/RSPEC)" block prints
    the participation table (Gamma_i, Sa(omega_i), the modal peak Gamma Sa/w^2)
    and the SRSS and CQC peak envelopes. For this stack the modes are
    well separated, so CQC ~ SRSS (the "MAX |CQC - SRSS|" line is small) —
    CQC's cross-terms only matter for CLOSELY-SPACED modes (a symmetric /
    tuned structure; the M19 tests assert the gap on a near-degenerate pair).

The response PSD, moments and RMS are on
``model.implicit_result.random_response``; the SRSS/CQC peak envelopes on
``model.implicit_result.response_spectrum``.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i RANDOM_VIBRATION_0000.rad
    pyradioss-engine  -i RANDOM_VIBRATION_0001.rad
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


NMASS = 5            # stacked masses
DX = 10.0            # node spacing (mm)
MASS = 2.0e-3        # per-storey mass (kg) — half lumps onto each node
K = 800.0            # per-storey stiffness (kN/mm)
ZETA = 0.03          # modal damping (3 %) for the FRF / spectrum
NMODE = 5            # modes to extract
FMIN = 0.0           # analysis band start (1/ms)
FMAX = 250.0         # analysis band end (1/ms) — past the highest mode (~199)
NF = 3000            # PSD sweep points
S0 = 1.0e-4          # flat input base-acceleration PSD level ((mm/ms^2)^2/kHz)
SA = 50.0            # design-spectrum plateau (mm/ms^2)

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "RANDOM_VIBRATION - base-excited instrument stack "
             "(pyradioss /IMPL/PSD + /IMPL/RSPEC)"]

    # ---- nodes: stack along +x (node 1 = shaken base) ---------------------
    lines.append("/NODE")
    for i in range(NMASS + 1):
        lines.append(f"{i + 1:10d}{i * DX:20.10f}{0.0:20.10f}{0.0:20.10f}")

    # ---- one /SPRING (+ /PART + /PROP) per storey -------------------------
    for i in range(NMASS):
        lines.append(f"/SPRING/{i + 1}")
        lines.append(f"{i + 1:10d}{i + 1:10d}{i + 2:10d}")
    for i in range(NMASS):
        lines += [f"/PART/{i + 1}", f"storey{i + 1}", f"{i + 1:10d}         1"]

    lines += ["/MAT/LAW1/1", "steel elastic", "   7.8e-6",
              "     210.0       0.0"]
    for i in range(NMASS):
        lines += [f"/PROP/SPRING/{i + 1}", f"sp{i + 1}",
                  f"{MASS:12g}{K:12g}         0.0"]

    # ---- constraints: base (node 1) pinned, stack axial (x) only ----------
    stack = "\n".join(str(i + 2) for i in range(NMASS))
    lines += ["/GRNOD/NODE/1", "base", "1",
              "/GRNOD/NODE/2", "stack", stack,
              "/BCS/1", "pinned base (the support)",
              "       111       111         0         1",
              "/BCS/2", "axial only (stacked masses)",
              "       011       111         0         2"]

    # ---- /FUNCT/10: flat band-limited input base-acceleration PSD ---------
    lines += ["/FUNCT/10", "flat input PSD S_aa(f)",
              f"       0.0{S0:12g}", f"  100000.0{S0:12g}"]
    # ---- /FUNCT/20: plateau design response spectrum Sa(f) ----------------
    lines += ["/FUNCT/20", "design response spectrum Sa(f)",
              f"       0.0{SA:12g}", f"  100000.0{SA:12g}"]
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "RANDOM_VIBRATION_0000.rad"), runname="RANDOM_VIBRATION")

    # ---- engine: a bare implicit static step + the two M19 analyses -------
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/RANDOM_VIBRATION/1",
        "1.0",                       # one (zero-load) static increment
        "/IMPL",                     # implicit static driver hosts the analyses
        "/IMPL/PSD/BASE",            # M19: random base-acceleration PSD response
        # card: fmin fmax nf funct dir nmode  (dir 0 = x, funct 10 = S_aa)
        f"{FMIN}  {FMAX}  {NF}  10  0  {NMODE}",
        "/IMPL/RSPEC",               # M19: design-response-spectrum combination
        # card: funct dir zeta nmode  (funct 20 = Sa, dir 0 = x)
        f"20  0  {ZETA}  {NMODE}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "RANDOM_VIBRATION_0001.rad"))
    print("wrote RANDOM_VIBRATION_0000.rad / RANDOM_VIBRATION_0001.rad")


if __name__ == "__main__":
    main()
