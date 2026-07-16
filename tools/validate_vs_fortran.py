#!/usr/bin/env python
"""
Differential validation of the pyradioss port against the Fortran
OpenRadioss binaries (the RESEARCH_AUDIT's #1 recommendation).

Two modes
=========

parity (default)
    For each example under ``examples/`` (or ``--only name1,name2``):

    1. run the real Fortran Starter + Engine on the example's deck pair
       in a scratch directory, convert the binary T01 time-history with
       ``th_to_csv_win64.exe``;
    2. run the pyradioss Starter + Engine on the *original* deck pair in
       a sibling scratch directory (the port writes its T01 as CSV
       directly);
    3. compare every overlapping time-history channel (global internal /
       kinetic / hourglass / contact energies, external work, mass,
       momentum components, and /TH/PART energies where unambiguous) and
       report the relative RMS deviation and the final-time deviation —
       the actual numbers are always printed, the tolerance only sets
       the MATCH/DEVIATION label.

    Classification per example:

    - ``MATCH``          all compared channels within ``--tol`` (default 5 %)
    - ``DEVIATION``      Fortran and pyradioss both ran; some channel above tol
    - ``PORT-ONLY``      the Fortran chain cannot run this deck. Subtags:
                         ``(implicit)`` engine deck uses the port's /IMPL
                         cards; ``(dialect)`` the real Starter rejects the
                         port's deck dialect (see the DIALECT NOTE below);
                         ``(starter-reject)`` translated deck still refused.
    - ``FORTRAN-FAIL``   Fortran starter passed but engine/converter failed
    - ``PYRADIOSS-FAIL`` the port itself failed on its own example
    - ``SKIPPED-SLOW``   pyradioss side exceeded ``--timeout`` (default 240 s)
                         or the global ``--budget`` (default 2700 s)

    DIALECT NOTE (measured, see VALIDATION.md): the port's example decks
    are written in pyradioss' own *free-format* dialect (whitespace
    tokens, 10-char real fields, trailing default cards omitted, port
    field order on load cards).  The real Starter reads **fixed-format
    Radioss 2022** (20-char real fields, exact card layouts).  Feeding
    the decks to the Fortran Starter unmodified fails at /BEGIN
    ("INPUT FORMAT 0 NOT SUPPORTED"), and even where a deck can be
    nursed past the reader the model is *silently different* (e.g.
    Poisson ratio 0.3 lands in the E field's columns and reads as 0).
    Therefore this harness carries a small, per-keyword *translator*
    (``--shim translate``, the default) that re-emits a deck copy in the
    real fixed format.  Every translated layout was taken from the
    authoritative ``hm_cfg_files`` CARD definitions shipped with the
    Fortran build, and the translation is *formatting only* — same
    values, same meaning, sources under examples/ are never touched.
    Decks using keywords without a verified translator are honestly
    classified PORT-ONLY(dialect) and the raw Fortran error is recorded
    (``--shim begin`` inserts only the /BEGIN version card so that the
    per-deck real error is visible; ``--shim none`` runs the deck raw).

coverage
    Run ONLY the pyradioss Starter on the given native ``.rad`` deck(s)
    (e.g. the Ryan Lee k2rad conversions) and tabulate every keyword the
    port skipped, warned about, or errored on — a keyword-coverage
    census of the port against real-world decks.

Examples
========
    python tools/validate_vs_fortran.py parity
    python tools/validate_vs_fortran.py parity --only tensile_bar,box_beam_impact
    python tools/validate_vs_fortran.py coverage E:/openradioss_run/Ryan_Lee_Examples/ton-mm-s/runs/W12_k2rad/W12_0000.rad

Results land in ``<workdir>/parity_results.json`` (parity) /
``<workdir>/coverage_<deck>.json`` (coverage) plus a console table.

Environment: the Fortran binaries and their runtime come from
``C:/OpenRadioss`` and Intel oneAPI, exactly like the proven
E:/openradioss_run/Ryan_Lee_Examples/ton-mm-s/runs/run_batch.ps1 (non-MPI
single-process path: starter_win64.exe -np 1 -nt 1, engine_win64.exe -nt 1).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pyradioss.input.deck_reader import read_deck, KeywordBlock  # noqa: E402

# ----------------------------------------------------------------------------
# Fortran toolchain (mirrors run_batch.ps1, non-MPI path)
# ----------------------------------------------------------------------------

OR_ROOT = r"C:\OpenRadioss"
ONEAPI = r"C:\Program Files (x86)\Intel\oneAPI"
STARTER_EXE = os.path.join(OR_ROOT, "exec", "starter_win64.exe")
ENGINE_EXE = os.path.join(OR_ROOT, "exec", "engine_win64.exe")
TH2CSV_EXE = os.path.join(OR_ROOT, "exec", "th_to_csv_win64.exe")

DEFAULT_WORKDIR = os.environ.get(
    "VALRUNS_DIR",
    os.path.join(os.environ.get("TEMP", REPO), "valruns"))


def fortran_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["RAD_CFG_PATH"] = os.path.join(OR_ROOT, "hm_cfg_files")
    env["RAD_H3D_PATH"] = os.path.join(OR_ROOT, "extlib", "h3d", "lib", "win64")
    env["OMP_NUM_THREADS"] = "1"
    env["KMP_AFFINITY"] = "disabled"
    env["KMP_STACKSIZE"] = "400m"
    env["I_MPI_ROOT"] = os.path.join(ONEAPI, "mpi", "latest")
    env["I_MPI_OFI_LIBRARY_INTERNAL"] = "1"
    env["PATH"] = ";".join([
        os.path.join(OR_ROOT, "extlib", "hm_reader", "win64"),
        os.path.join(OR_ROOT, "extlib", "intelOneAPI_runtime", "win64"),
        os.path.join(ONEAPI, "mpi", "latest", "bin"),
        os.path.join(ONEAPI, "mpi", "latest", "libfabric", "bin"),
        env.get("PATH", ""),
    ])
    return env


def run_cmd(cmd: List[str], cwd: str, timeout: float,
            env: Optional[Dict[str, str]] = None) -> Tuple[int, str, float]:
    """Run a command, return (rc, tail-of-output, elapsed). rc=-9 on timeout."""
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="replace")
        out = p.stdout or ""
        return p.returncode, out[-4000:], time.time() - t0
    except subprocess.TimeoutExpired:
        return -9, "TIMEOUT", time.time() - t0


# ----------------------------------------------------------------------------
# Deck dialect translation  (port free-format  ->  real Radioss 2022 fixed)
#
# Every layout below is copied from the CARD("...") definitions in
# C:/OpenRadioss/hm_cfg_files/config/CFG/radioss*/ — the exact format
# strings this Starter build parses with.  Only formatting is changed;
# values and their meaning are preserved (the port's card semantics were
# verified against pyradioss/input/starter_keywords.py docstrings).
# ----------------------------------------------------------------------------

def _i10(v) -> str:
    return f"{int(float(v)):>10d}"


def _f20(v) -> str:
    return f"{float(v):>20.10G}"


def _b(width: int = 10) -> str:
    return " " * width


def _title_cards(block: KeywordBlock) -> Tuple[str, list]:
    """First card = title (Radioss convention for titled blocks)."""
    if not block.cards:
        return "", []
    return block.cards[0].raw.rstrip(), block.cards[1:]


def _header(block: KeywordBlock) -> str:
    return "/" + "/".join(block.parts)


class Untranslatable(Exception):
    pass


def tr_begin(block, runname):
    # /BEGIN: runname, input-version card, input/work unit cards
    # (identical units in and out -> no conversion; labels arbitrary).
    return ["/BEGIN", runname, "      2022         0",
            "                  Mg                  mm                   s",
            "                  Mg                  mm                   s"]


def tr_passthrough(block, runname):
    return [_header(block)] + [c.raw.rstrip() for c in block.cards]


def tr_node(block, runname):
    # cfg: %10d%20lg%20lg%20lg
    out = [_header(block)]
    for c in block.cards:
        t = c.tokens()
        out.append(_i10(t[0]) + "".join(_f20(x) for x in t[1:4]))
    return out


def tr_int_elements(block, runname):
    # BRICK/SHELL/SH3N/TRUSS/BEAM/SPRING element cards: all-int I10 fields
    out = [_header(block)]
    for c in block.cards:
        out.append("".join(_i10(x) for x in c.tokens()))
    return out


def tr_part(block, runname):
    title, cards = _title_cards(block)
    t = cards[0].tokens()
    return [_header(block), title, _i10(t[0]) + _i10(t[1])]


def tr_mat_law1(block, runname):
    # cfg matl1 'elast': rho / E nu   (all %20lg)
    title, cards = _title_cards(block)
    out = [_header(block), title, _f20(cards[0].tokens()[0])]
    e, nu = cards[1].tokens()[:2]
    out.append(_f20(e) + _f20(nu))
    return out


def tr_mat_law2(block, runname):
    # cfg matl2_plas_johns (radioss2023):
    #   rho / E nu Iflag VP / a b n epsmax sigmax / c eps0 ICC ... / m ...
    title, cards = _title_cards(block)
    out = [_header(block), title, _f20(cards[0].tokens()[0])]
    e, nu = cards[1].tokens()[:2]
    out.append(_f20(e) + _f20(nu))          # Iflag/VP blank -> defaults
    t = cards[2].tokens()
    out.append("".join(_f20(x) for x in t[:5]))
    if len(cards) >= 4:                     # c eps0  (ICC.. defaults)
        t = cards[3].tokens()
        out.append("".join(_f20(x) for x in t[:2]))
    if len(cards) >= 5:                     # m Tmelt rhoCp Tr
        t = cards[4].tokens()
        out.append("".join(_f20(x) for x in t[:4]))
    return out


def tr_mat_law42(block, runname):
    # cfg matl42_Ogden: rho / nu sig_cut fBulk fscale M Iform /
    #                   mu1-5 / mu6-10 / alpha1-5 / alpha6-10
    # port: rho / mu1-5 / alpha1-5 / [nu]
    title, cards = _title_cards(block)
    rho = cards[0].tokens()[0]
    mu = [float(x) for x in cards[1].tokens()[:5]]
    al = [float(x) for x in cards[2].tokens()[:5]] if len(cards) >= 3 else []
    nu = float(cards[3].tokens()[0]) if len(cards) >= 4 else 0.495
    z5 = "".join(_f20(0.0) for _ in range(5))
    return [_header(block), title, _f20(rho), _f20(nu),
            "".join(_f20(x) for x in mu), z5,
            "".join(_f20(x) for x in al), z5]


def tr_eos(block, runname):
    # cfg mat_EOS radioss2022 — IDEAL-GAS: title / Gamma P0 PSH T0 RHO0
    #                           POLYNOMIAL: title / C0..C3 / C4 C5 E0 PSH RHO0
    # NOTE the real /EOS has a TITLE card the port dialect omits.
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    cards = block.cards
    if kind == "IDEAL-GAS":
        t = cards[0].tokens()
        gamma = t[0]
        p0 = t[1] if len(t) > 1 else 0.0
        return [_header(block), "ideal gas (translated)",
                _f20(gamma) + _f20(p0)]
    if kind == "POLYNOMIAL":
        t = cards[0].tokens()
        e0 = cards[1].tokens()[0] if len(cards) > 1 else 0.0
        return [_header(block), "polynomial EOS (translated)",
                "".join(_f20(x) for x in t[:4]),
                _f20(t[4]) + _f20(t[5]) + _f20(e0)]
    raise Untranslatable(f"/EOS/{kind}")


def tr_prop_solid(block, runname):
    # cfg prop_p14_solid: title / Isolid Ismstr Iale Icpre Itet10 Inpts
    #   Itet4 Iframe dn / qa qb h Lambda Mu / dtmin ...
    # port: single float card 'qa qb h' (int flag cards ignored).
    # Isolid=1 emitted explicitly = 8-node 1-point + viscous hourglass,
    # the only formulation the port implements.
    title, cards = _title_cards(block)
    qa, qb, h = 1.1, 0.05, 0.1
    for c in cards:
        toks = c.tokens()
        if toks and not all(t.lstrip("+-").isdigit() for t in toks):
            vals = [float(x) for x in toks[:3]]
            vals += [qa, qb, h][len(vals):]
            qa, qb, h = vals
            break
    return [_header(block), title, _i10(1),
            _f20(qa) + _f20(qb) + _f20(h)]


def tr_prop_shell(block, runname):
    # cfg prop_p1_shell: title / Ishell Ismstr Ish3n Idrill /
    #   Hm Hf Hr Dm Dn (F20) / N Istrain Thick AShear <10sp> Ithick Iplas
    title, cards = _title_cards(block)
    hm = hf = hr = 0.01
    nip, thick = 3, 1.0
    ishell = 1        # Belytschko-Tsay — the port's only shell formulation
    if cards and any("." in t or "e" in t.lower()
                     for t in cards[0].tokens()):
        # short form: Thick [N] [hm]
        t = [float(x) for x in cards[0].tokens()[:3]]
        thick = t[0]
        if len(t) > 1 and t[1]:
            nip = int(t[1])
        if len(t) > 2 and t[2]:
            hm = hf = hr = t[2]
    else:
        if cards:
            f = cards[0].tokens()
            if f and int(f[0]) > 0:
                ishell = int(f[0])
        if len(cards) >= 2:
            t = [float(x) for x in cards[1].tokens()[:3]]
            hm, hf, hr = (t + [0.01] * 3)[:3]
            hm, hf, hr = hm or 0.01, hf or 0.01, hr or 0.01
        if len(cards) >= 3:
            t = cards[2].tokens()
            nip = int(float(t[0])) or 3
            thick = float(t[2]) if len(t) > 2 else 1.0
    return [_header(block), title,
            _i10(ishell),
            _f20(hm) + _f20(hf) + _f20(hr),
            _i10(nip) + _b(10) + _f20(thick)]


def tr_prop_beam(block, runname):
    # cfg prop_p3_beam: title / <10sp>Ismstr / Dm Df / Area Iyy Izz Ixx /
    #                   Wdof+Ishear
    # port: short form single card 'Area Iyy Izz Ixx'
    title, cards = _title_cards(block)
    data = [c for c in cards if not all(t.lstrip("+-").isdigit()
                                        for t in c.tokens())]
    t = data[0].tokens() if data else cards[-1].tokens()
    return [_header(block), title,
            _b(20),                                   # Ismstr default
            _f20(0.0) + _f20(0.0),                    # Dm Df
            "".join(_f20(x) for x in t[:4]),
            _b(20)]                                   # Wdof/Ishear default


def tr_bcs(block, runname):
    # cfg bcs: title / '   TTT RRR' skew(I10) grnod(I10)
    title, cards = _title_cards(block)
    t = cards[0].tokens()
    tra, rot = t[0].zfill(3), t[1].zfill(3)
    return [_header(block), title,
            f"   {tra} {rot}" + _i10(t[2]) + _i10(t[3])]


def tr_grnod(block, runname):
    # GRNOD/NODE|PART: title + ids, 10 per card, I10 fields
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind not in ("NODE", "PART"):
        raise Untranslatable(f"/GRNOD/{kind}")
    title, cards = _title_cards(block)
    ids: List[int] = []
    for c in cards:
        ids.extend(int(x) for x in c.tokens())
    out = [_header(block), title]
    for i in range(0, len(ids), 10):
        out.append("".join(_i10(x) for x in ids[i:i + 10]))
    return out


def tr_surf_part(block, runname):
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind != "PART":
        raise Untranslatable(f"/SURF/{kind}")
    title, cards = _title_cards(block)
    ids = []
    for c in cards:
        ids.extend(int(x) for x in c.tokens())
    out = [_header(block), title]
    for i in range(0, len(ids), 10):
        out.append("".join(_i10(x) for x in ids[i:i + 10]))
    return out


def tr_funct(block, runname):
    # cfg funct: title / one (X,Y) pair per card, %20lg%20lg
    title, cards = _title_cards(block)
    out = [_header(block), title]
    for c in cards:
        t = c.tokens()
        out.append(_f20(t[0]) + _f20(t[1]))
    return out


def tr_impvel_impdisp(block, runname):
    # cfg impvel/impdisp: title / fct(I10) Dir(A10) skew sens grnod frame
    #   icoor / Scale_x(F20) Scale_y(F20) Tstart Tstop
    # port card: fct Dir grnod [scale]
    title, cards = _title_cards(block)
    t = cards[0].tokens()
    fct, direc, grnod = t[0], t[1].upper(), t[2]
    scale = float(t[3]) if len(t) > 3 else 1.0
    return [_header(block), title,
            _i10(fct) + f"{direc:>10}" + _b(10) + _b(10) + _i10(grnod),
            _f20(1.0) + _f20(scale)]


def tr_cload(block, runname):
    # cfg cload radioss2023 (single card):
    #   fct(I10) Dir(A10) skew sens grnod Itypfun Ascalex(F20) Fscaley(F20)
    # port card: fct Dir grnod [scale] [sens]
    title, cards = _title_cards(block)
    t = cards[0].tokens()
    fct, direc, grnod = t[0], t[1].upper(), t[2]
    scale = float(t[3]) if len(t) > 3 else 1.0
    sens = int(float(t[4])) if len(t) > 4 else 0
    return [_header(block), title,
            _i10(fct) + f"{direc:>10}" + _b(10) + _i10(sens) + _i10(grnod)
            + _b(10) + _f20(1.0) + _f20(scale)]


def tr_inivel(block, runname):
    # cfg inivel: title / Vx Vy Vz (3xF20) grnod(I10) skew(I10)
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind != "TRA":
        raise Untranslatable(f"/INIVEL/{kind}")
    title, cards = _title_cards(block)
    t = cards[0].tokens()
    return [_header(block), title,
            _f20(t[0]) + _f20(t[1]) + _f20(t[2]) + _i10(t[3])]


def tr_rwall(block, runname):
    # cfg RWALL/plane.cfg: title / node slide grnd1 grnd2 (I10) /
    #   d fric ... (F20) / XM YM ZM / XM1 YM1 ZM1
    # port: grnod slide fric dist [node] / XM YM ZM / XM1 YM1 ZM1
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind != "PLANE":
        raise Untranslatable(f"/RWALL/{kind}")
    title, cards = _title_cards(block)
    t = cards[0].tokens()
    grnod = int(t[0]) if t else 0
    slide = int(t[1]) if len(t) > 1 else 0
    fric = float(t[2]) if len(t) > 2 else 0.0
    dist = float(t[3]) if len(t) > 3 else 0.0
    node = int(float(t[4])) if len(t) > 4 else 0
    m = cards[1].tokens()[:3]
    m1 = cards[2].tokens()[:3]
    # semantics mapping (measured): the port's dist=0 means "track ALL
    # nodes every cycle"; the real d is the secondary-node search distance
    # and d=0 selects NOTHING (the wall never acts).  Emit d=1e30 for 0.
    d = dist if dist > 0 else 1e30
    return [_header(block), title,
            _i10(node) + _i10(slide) + _i10(grnod) + _i10(0),
            _f20(d) + _f20(fric),
            "".join(_f20(x) for x in m),
            "".join(_f20(x) for x in m1)]


def tr_th(block, runname):
    # cfg th_node / th_part: title / var names %-10s /
    #   NODE: one card per id  '%10d%10d%-80s' (id, skew, name)
    #   PART: ids packed %10d
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind not in ("NODE", "PART"):
        raise Untranslatable(f"/TH/{kind}")
    title, cards = _title_cards(block)
    vars_ = cards[0].tokens()
    ids: List[int] = []
    for c in cards[1:]:
        ids.extend(int(x) for x in c.tokens())
    out = [_header(block), title, "".join(f"{v:<10}" for v in vars_)]
    if kind == "NODE":
        out += [_i10(i) for i in ids]
    else:
        for i in range(0, len(ids), 10):
            out.append("".join(_i10(x) for x in ids[i:i + 10]))
    return out


def tr_end(block, runname):
    return ["/END"]


TRANSLATORS = {
    "BEGIN": tr_begin,
    "TITLE": tr_passthrough,
    "END": tr_end,
    "NODE": tr_node,
    "BRICK": tr_int_elements,
    "TETRA4": tr_int_elements,
    "SHELL": tr_int_elements,
    "SH3N": tr_int_elements,
    "BEAM": tr_int_elements,
    "PART": tr_part,
    "MAT/LAW1": tr_mat_law1,
    "MAT/ELAST": tr_mat_law1,
    "MAT/LAW2": tr_mat_law2,
    "MAT/PLAS_JOHNS": tr_mat_law2,
    "MAT/LAW42": tr_mat_law42,
    "MAT/OGDEN": tr_mat_law42,
    "EOS/IDEAL-GAS": tr_eos,
    "EOS/POLYNOMIAL": tr_eos,
    "PROP/SOLID": tr_prop_solid,
    "PROP/TYPE14": tr_prop_solid,
    "PROP/SHELL": tr_prop_shell,
    "PROP/TYPE1": tr_prop_shell,
    "PROP/BEAM": tr_prop_beam,
    "PROP/TYPE3": tr_prop_beam,
    "BCS": tr_bcs,
    "GRNOD/NODE": tr_grnod,
    "GRNOD/PART": tr_grnod,
    "SURF/PART": tr_surf_part,
    "FUNCT": tr_funct,
    "IMPVEL": tr_impvel_impdisp,
    "IMPDISP": tr_impvel_impdisp,
    "CLOAD": tr_cload,
    "INIVEL/TRA": tr_inivel,
    "RWALL/PLANE": tr_rwall,
    "TH/NODE": tr_th,
    "TH/PART": tr_th,
}


def block_key(block: KeywordBlock) -> str:
    """Lookup key: 'MAT/LAW2' style (keyword includes subtype, no id)."""
    return block.keyword


def translate_starter_deck(path: str, runname: str) -> Tuple[Optional[str],
                                                             List[str]]:
    """Return (translated text, untranslatable keys). None text if any."""
    blocks = read_deck(path)
    missing = sorted({block_key(b) for b in blocks
                      if block_key(b) not in TRANSLATORS})
    if missing:
        return None, missing
    out = ["#RADIOSS STARTER",
           "# translated to fixed 2022 format by tools/validate_vs_fortran.py"
           " (formatting only)"]
    try:
        for b in blocks:
            out.extend(TRANSLATORS[block_key(b)](b, runname))
    except (Untranslatable, ValueError, IndexError, KeyError) as exc:
        return None, [f"{type(exc).__name__}: {exc}"]
    return "\n".join(out) + "\n", []


def shim_begin_only(path: str) -> str:
    """Insert the 2022 version + unit cards after the /BEGIN title card
    (the minimum to get past the real reader's front door)."""
    lines = open(path, errors="replace").read().splitlines()
    out, i = [], 0
    if not (lines and lines[0].startswith("#RADIOSS STARTER")):
        # the real reader demands '#RADIOSS STARTER' as the first card
        # (ERROR 100201) — two example decks omit it
        out.append("#RADIOSS STARTER")
    while i < len(lines):
        out.append(lines[i])
        if lines[i].strip().upper().startswith("/BEGIN"):
            if i + 1 < len(lines) and not lines[i + 1].lstrip().startswith("/"):
                i += 1
                out.append(lines[i])          # keep the run-name card
            out += ["      2022         0",
                    "                  Mg                  mm                   s",
                    "                  Mg                  mm                   s"]
        i += 1
    return "\n".join(out) + "\n"


def strip_engine_stop(path: str) -> str:
    """Drop the port's /STOP block from the engine deck copy: the real
    Engine's reader hits EOF on it (measured: forrtl severe(24) on unit 30);
    everything else passes through unchanged."""
    lines = open(path, errors="replace").read().splitlines()
    out, skip = [], False
    for ln in lines:
        if ln.strip().upper().startswith("/STOP"):
            skip = True
            continue
        if skip and not ln.lstrip().startswith("/"):
            continue                     # /STOP data card(s)
        skip = False
        out.append(ln)
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Time-history CSV parsing + comparison
# ----------------------------------------------------------------------------

# Fortran th_to_csv global column -> port column (pyradioss T01 CSV)
GLOBAL_MAP = [
    ("INTERNAL ENERGY", "IE"),
    ("KINETIC ENERGY", "KE"),
    ("HOURGLASS ENERGY", "HE"),
    ("CONTACT ENERGY", "CE"),
    ("EXTERNAL WORK", "EW"),
    ("MASS", "MASS"),
    ("X-MOMENTUM", "MOMX"),
    ("Y-MOMENTUM", "MOMY"),
    ("Z-MOMENTUM", "MOMZ"),
]


def read_csv_columns(path: str) -> Tuple[List[str], "np.ndarray"]:
    import numpy as np
    rows = []
    header: List[str] = []
    with open(path, newline="", errors="replace") as fh:
        for rec in csv.reader(fh):
            rec = [c.strip().strip('"') for c in rec if c.strip() != ""]
            if not rec:
                continue
            if rec[0].startswith("#"):
                continue
            if not header:
                header = rec
                continue
            try:
                rows.append([float(x) for x in rec])
            except ValueError:
                continue
    n = min(len(r) for r in rows) if rows else 0
    data = np.array([r[:n] for r in rows]) if rows else np.zeros((0, 0))
    return header[:n] if n else header, data


def compare_channels(f_hdr, f_dat, p_hdr, p_dat, part_titles=None):
    """Return list of dicts: channel, rel_rms, final_dev (fractions)."""
    import numpy as np
    res = []
    if f_dat.size == 0 or p_dat.size == 0:
        return res
    tf, tp = f_dat[:, 0], p_dat[:, 0]
    tend = min(tf[-1], tp[-1])
    grid = tp[tp <= tend + 1e-12]
    if len(grid) < 3:
        return res

    def col(hdr, dat, name, exact=True):
        for j, h in enumerate(hdr):
            if (h == name) if exact else (name in h):
                return dat[:, j]
        return None

    pairs = []
    for fname, pname in GLOBAL_MAP:
        a, b = col(f_hdr, f_dat, fname), col(p_hdr, p_dat, pname)
        if a is not None and b is not None:
            pairs.append((pname, a, b))
    # derived total energy
    fie, fke = col(f_hdr, f_dat, "INTERNAL ENERGY"), col(f_hdr, f_dat,
                                                         "KINETIC ENERGY")
    pie, pke = col(p_hdr, p_dat, "IE"), col(p_hdr, p_dat, "KE")
    if all(x is not None for x in (fie, fke, pie, pke)):
        pairs.append(("IE+KE", fie + fke, pie + pke))
    # /TH/PART energies: fortran header 'title ... IE'; port 'P<id>_IE'
    for suffix in ("IE", "KE"):
        fcols = [j for j, h in enumerate(f_hdr)
                 if re.search(rf"\s{suffix}\s*$", h)]
        pcols = [j for j, h in enumerate(p_hdr)
                 if re.fullmatch(rf"P\d+_{suffix}", h)]
        if len(fcols) == 1 and len(pcols) == 1:
            pairs.append((f"part_{suffix}", f_dat[:, fcols[0]],
                          p_dat[:, pcols[0]]))

    # group reference scales: a channel that is numerically ~zero on BOTH
    # sides relative to its group's dominant channel (e.g. the transverse
    # momentum of a uniaxial test) is noise, not physics — it is still
    # reported, but marked insignificant and excluded from MATCH/DEVIATION.
    def group_of(nm):
        if nm.startswith("MOM"):
            return "momentum"
        if nm == "MASS":
            return "mass"
        return "energy"

    interp = {}
    for nm, a, b in pairs:
        interp[nm] = (np.interp(grid, tf, a), np.interp(grid, tp, b))
    gref: Dict[str, float] = {}
    for nm, (ai, bi) in interp.items():
        g = group_of(nm)
        gref[g] = max(gref.get(g, 0.0), float(np.max(np.abs(ai))),
                      float(np.max(np.abs(bi))))

    for nm, a, b in pairs:
        ai, bi = interp[nm]
        denom = max(float(np.max(np.abs(ai))), float(np.max(np.abs(bi))))
        ref = gref[group_of(nm)]
        if denom <= 0 or (ref > 0 and denom < 1e-9 * ref):
            continue    # empty on both sides — nothing to compare
        rel_rms = float(np.sqrt(np.mean((ai - bi) ** 2)) / denom)
        final = float(abs(ai[-1] - bi[-1]) / denom)
        # significance rule (documented in VALIDATION.md): a channel drives
        # the MATCH/DEVIATION class only when it carries at least 1 % of its
        # group's dominant scale — e.g. the transverse momentum of an
        # axially loaded mast (0.1 % of the axial momentum) is numerical
        # noise on both sides; it is still printed, prefixed '~'.
        res.append({"channel": nm, "rel_rms": rel_rms, "final_dev": final,
                    "scale": denom,
                    "significant": bool(denom >= 1e-2 * ref)})
    return res


# ----------------------------------------------------------------------------
# Fortran + pyradioss single-example drivers
# ----------------------------------------------------------------------------

def first_starter_error(out_file: str, log_tail: str) -> str:
    if os.path.exists(out_file):
        txt = open(out_file, errors="replace").read()
        m = re.search(r"ERROR ID\s*:\s*\S+\n\*\*[^\n]*\nDESCRIPTION[^\n]*\n"
                      r"((?:--[^\n]*\n|[^\n]*\n){0,3})", txt)
        if m:
            desc = " | ".join(l.strip() for l in m.group(0).splitlines()[:6]
                              if l.strip())
            return desc[:300]
    tail = " | ".join(l.strip() for l in log_tail.splitlines()[-4:]
                      if l.strip())
    return tail[:300]


def run_fortran(name: str, runname: str, deck0: str, deck1: str,
                workdir: str, shim: str) -> Dict:
    """Run starter+engine+th_to_csv. Returns dict with status/csv/error."""
    rd = os.path.join(workdir, "fortran", name)
    shutil.rmtree(rd, ignore_errors=True)
    os.makedirs(rd)
    d0 = os.path.join(rd, os.path.basename(deck0))
    d1 = os.path.join(rd, os.path.basename(deck1))
    info: Dict = {"mode": shim, "dir": rd}

    if shim == "translate":
        text, missing = translate_starter_deck(deck0, runname)
        if text is None:
            info.update(status="untranslatable", missing=missing)
            # still probe with the begin shim to capture the real error
            open(d0, "w").write(shim_begin_only(deck0))
            env = fortran_env()
            rc, tail, dt = run_cmd(
                [STARTER_EXE, "-i", os.path.basename(d0), "-np", "1",
                 "-nt", "1"], rd, 180, env)
            info["probe_error"] = first_starter_error(
                os.path.join(rd, f"{runname}_0000.out"), tail)
            return info
        open(d0, "w").write(text)
    elif shim == "begin":
        open(d0, "w").write(shim_begin_only(deck0))
    else:
        shutil.copy(deck0, d0)
    open(d1, "w").write(strip_engine_stop(deck1)
                        if shim != "none" else open(deck1).read())

    env = fortran_env()
    rc, tail, dt = run_cmd([STARTER_EXE, "-i", os.path.basename(d0),
                            "-np", "1", "-nt", "1"], rd, 300, env)
    info["starter_rc"] = rc
    info["starter_time"] = round(dt, 1)
    out0 = os.path.join(rd, f"{runname}_0000.out")
    nerr = 0
    if os.path.exists(out0):
        nerr = len(re.findall(r"^ERROR ID", open(out0, errors="replace")
                              .read(), re.M))
    rst = os.path.join(rd, f"{runname}_0000_0001.rst")
    if rc != 0 or nerr or not os.path.exists(rst):
        info.update(status="starter-reject", nerr=nerr,
                    error=first_starter_error(out0, tail))
        return info

    rc, tail, dt = run_cmd([ENGINE_EXE, "-i", os.path.basename(d1),
                            "-nt", "1"], rd, 900, env)
    info["engine_rc"] = rc
    info["engine_time"] = round(dt, 1)
    t01 = os.path.join(rd, f"{runname}T01")
    normal = "NORMAL TERMINATION" in tail or (
        os.path.exists(os.path.join(rd, f"{runname}_0001.out")) and
        "NORMAL TERMINATION" in open(os.path.join(rd, f"{runname}_0001.out"),
                                     errors="replace").read())
    if not os.path.exists(t01):
        info.update(status="engine-fail",
                    error=tail.splitlines()[-1] if tail else f"rc={rc}")
        return info
    rc2, tail2, _ = run_cmd([TH2CSV_EXE, os.path.basename(t01)], rd, 120, env)
    csvp = t01 + ".csv"
    if not os.path.exists(csvp):
        info.update(status="th2csv-fail", error=tail2[-200:])
        return info
    info.update(status="ok" if normal else "engine-partial", csv=csvp)
    return info


def run_pyradioss(name: str, runname: str, deck0: str, deck1: str,
                  workdir: str, timeout: float) -> Dict:
    rd = os.path.join(workdir, "pyradioss", name)
    shutil.rmtree(rd, ignore_errors=True)
    os.makedirs(rd)
    shutil.copy(deck0, rd)
    shutil.copy(deck1, rd)
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    info: Dict = {"dir": rd}
    rc, tail, dt = run_cmd([sys.executable, "-m", "pyradioss.starter", "-i",
                            os.path.basename(deck0)], rd, timeout, env)
    info["starter_rc"] = rc
    info["starter_time"] = round(dt, 1)
    if rc == -9:
        info["status"] = "timeout"
        return info
    if rc != 0:
        info.update(status="starter-fail", error=tail[-300:])
        return info
    remain = max(20.0, timeout - dt)
    rc, tail, dt = run_cmd([sys.executable, "-m", "pyradioss.engine", "-i",
                            os.path.basename(deck1)], rd, remain, env)
    info["engine_rc"] = rc
    info["engine_time"] = round(dt, 1)
    if rc == -9:
        info["status"] = "timeout"
        return info
    csvp = os.path.join(rd, f"{runname}T01.csv")
    if rc != 0:
        info.update(status="engine-fail", error=tail[-300:])
        return info
    if not os.path.exists(csvp):
        # normal termination without a T-file (e.g. /IMPL runs write no
        # explicit time history) — success, but nothing to compare.
        info.update(status="ok-no-th")
        return info
    info.update(status="ok", csv=csvp)
    return info


# ----------------------------------------------------------------------------
# parity mode
# ----------------------------------------------------------------------------

def find_examples(only: Optional[List[str]]) -> List[Tuple[str, str, str, str]]:
    """[(name, runname, deck0, deck1)] sorted explicit-first."""
    exdir = os.path.join(REPO, "examples")
    items = []
    for name in sorted(os.listdir(exdir)):
        d = os.path.join(exdir, name)
        if not os.path.isdir(d):
            continue
        if only and name not in only:
            continue
        d0 = [f for f in os.listdir(d) if f.endswith("_0000.rad")]
        if not d0:
            continue
        runname = d0[0][:-len("_0000.rad")]
        deck0 = os.path.join(d, d0[0])
        deck1 = os.path.join(d, f"{runname}_0001.rad")
        if not os.path.exists(deck1):
            continue
        implicit = "/IMPL" in open(deck1, errors="replace").read()
        items.append((implicit, name, runname, deck0, deck1))
    items.sort(key=lambda it: (it[0], it[1]))     # explicit first
    return [(n, r, a, b) for _, n, r, a, b in items]


def parity(args) -> int:
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    only = args.only.split(",") if args.only else None
    examples = find_examples(only)
    if not examples:
        print("no examples found", file=sys.stderr)
        return 2
    budget_left = args.budget
    results = []
    retry_queue = []

    def one(name, runname, deck0, deck1, budget_left):
        implicit = "/IMPL" in open(deck1, errors="replace").read()
        row = {"example": name, "runname": runname, "implicit": implicit}

        # ---- Fortran side ------------------------------------------------
        if implicit:
            # the engine controls (/IMPL, /IMPL/FATIG/...) are port
            # extensions; the Fortran chain cannot run them.  Probe the
            # starter anyway so the table carries the real error message.
            f = run_fortran(name, runname, deck0, deck1, workdir,
                            "begin" if args.shim != "none" else "none")
            f["status"] = "implicit-port-card"
            row["class"] = "PORT-ONLY(implicit)"
        else:
            f = run_fortran(name, runname, deck0, deck1, workdir, args.shim)
            if f["status"] == "untranslatable":
                row["class"] = "PORT-ONLY(dialect)"
            elif f["status"] == "starter-reject":
                row["class"] = ("PORT-ONLY(dialect)" if args.shim != "translate"
                                else "PORT-ONLY(starter-reject)")
            elif f["status"] in ("engine-fail", "th2csv-fail"):
                row["class"] = "FORTRAN-FAIL"
        row["fortran"] = {k: v for k, v in f.items() if k != "dir"}

        # ---- pyradioss side ----------------------------------------------
        if budget_left <= 0:
            row["pyradioss"] = {"status": "skipped-budget"}
            row.setdefault("class", "SKIPPED-SLOW")
            return row, budget_left
        t0 = time.time()
        p = run_pyradioss(name, runname, deck0, deck1, workdir,
                          min(args.timeout, max(30, budget_left)))
        budget_left -= time.time() - t0
        row["pyradioss"] = {k: v for k, v in p.items() if k != "dir"}
        if p["status"] == "timeout":
            row["class"] = "SKIPPED-SLOW"
            return row, budget_left
        if p["status"] not in ("ok", "ok-no-th"):
            row["class"] = "PYRADIOSS-FAIL"

        # ---- comparison ----------------------------------------------------
        if f.get("csv") and p.get("csv"):
            fh, fd = read_csv_columns(f["csv"])
            ph, pd = read_csv_columns(p["csv"])
            ch = compare_channels(fh, fd, ph, pd)
            row["channels"] = ch
            sig = [c for c in ch if c.get("significant", True)]
            if sig:
                worst = max(c["rel_rms"] for c in sig)
                row["max_rel_rms"] = worst
                row["class"] = "MATCH" if worst <= args.tol else "DEVIATION"
            else:
                row["class"] = "FORTRAN-FAIL"
                row["fortran"]["error"] = "no overlapping channels"
        elif "class" not in row:
            row["class"] = "FORTRAN-FAIL"
        return row, budget_left

    for name, runname, deck0, deck1 in examples:
        print(f"=== {name} ...", flush=True)
        row, budget_left = one(name, runname, deck0, deck1, budget_left)
        if row.get("pyradioss", {}).get("status") in ("starter-fail",
                                                      "engine-fail"):
            retry_queue.append((name, runname, deck0, deck1))
        results.append(row)
        print(f"    -> {row.get('class')}", flush=True)

    # one retry for pyradioss failures (concurrent-edit protection)
    for name, runname, deck0, deck1 in retry_queue:
        if budget_left <= 0:
            break
        print(f"=== retry {name} ...", flush=True)
        time.sleep(30)
        row, budget_left = one(name, runname, deck0, deck1, budget_left)
        for i, r in enumerate(results):
            if r["example"] == name:
                results[i] = row
        print(f"    -> {row.get('class')} (retry)", flush=True)

    out = os.path.join(workdir, "parity_results.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nresults JSON: {out}\n")

    # ---- console table ------------------------------------------------------
    hdr = f"{'example':<34} {'class':<26} {'maxRMS':>8}  channels"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        chs = r.get("channels", [])
        chtxt = " ".join(
            ("" if c.get("significant", True) else "~") +
            f"{c['channel']}={c['rel_rms']:.3G}" for c in chs)
        rms = f"{r.get('max_rel_rms', float('nan')):.3G}" \
            if "max_rel_rms" in r else "-"
        print(f"{r['example']:<34} {r.get('class', '?'):<26} {rms:>8}  "
              f"{chtxt}")
    return 0


# ----------------------------------------------------------------------------
# coverage mode
# ----------------------------------------------------------------------------

def coverage(args) -> int:
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    from pyradioss.input.starter_keywords import KEYWORD_PARSERS
    rc_all = 0
    for deck in args.decks:
        deck = os.path.abspath(deck)
        name = os.path.splitext(os.path.basename(deck))[0]
        rd = os.path.join(workdir, "coverage", name)
        shutil.rmtree(rd, ignore_errors=True)
        os.makedirs(rd)
        shutil.copy(deck, rd)

        # 1. keyword census straight from the lexer
        census: Dict[str, int] = {}
        for b in read_deck(deck):
            census[b.keyword] = census.get(b.keyword, 0) + 1

        # 2. run the port starter, harvest its messages
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO
        rc, tail, dt = run_cmd([sys.executable, "-m", "pyradioss.starter",
                                "-i", os.path.basename(deck)], rd,
                               args.timeout, env)
        listing = os.path.join(
            rd, re.sub(r"_0000$", "", name) + "_0000.out")
        text = tail
        if os.path.exists(listing):
            text = open(listing, errors="replace").read() + "\n" + tail
        msgs: Dict[str, List[str]] = {}
        for m in re.finditer(r"\*\*\s*(WARNING|ERROR)[^\n]*\n?([^\n]*)", text):
            line = (m.group(0).replace("\n", " ").strip())[:200]
            kw = "?"
            km = re.search(r"/([A-Z0-9_/\-]+)", line)
            if km:
                kw = km.group(1).split("/")[0]
            msgs.setdefault(kw, []).append(line)

        rows = []
        for kw in sorted(census):
            k0 = kw.split("/")[0]
            ported = k0 in KEYWORD_PARSERS
            note = ""
            for mk, ml in msgs.items():
                for line in ml:
                    if f"/{kw}" in line or (not ported and f"/{k0}" in line):
                        note = line
                        break
                if note:
                    break
            rows.append({"keyword": kw, "blocks": census[kw],
                         "dispatch": "ported" if ported else "UNKNOWN",
                         "message": note})
        result = {"deck": deck, "starter_rc": rc, "elapsed": round(dt, 1),
                  "rows": rows,
                  "other_messages": {k: v[:3] for k, v in msgs.items()},
                  "tail": tail[-1500:]}
        outp = os.path.join(workdir, f"coverage_{name}.json")
        json.dump(result, open(outp, "w"), indent=1)
        print(f"\n### coverage {name}  (starter rc={rc}, {dt:.0f}s)"
              f"  -> {outp}")
        print(f"{'keyword':<28} {'blocks':>6} {'dispatch':<9} message")
        for r in rows:
            print(f"{r['keyword']:<28} {r['blocks']:>6} {r['dispatch']:<9} "
                  f"{r['message'][:90]}")
        if rc not in (0,):
            rc_all = 1
    return rc_all


# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")

    pp = sub.add_parser("parity", help="Fortran vs pyradioss on examples/")
    pp.add_argument("--only", help="comma-separated example names")
    pp.add_argument("--tol", type=float, default=0.05,
                    help="rel RMS tolerance for MATCH (default 0.05)")
    pp.add_argument("--timeout", type=float, default=240,
                    help="per-example pyradioss wall clock cap (s)")
    pp.add_argument("--budget", type=float, default=2700,
                    help="total pyradioss wall clock budget (s)")
    pp.add_argument("--shim", choices=["none", "begin", "translate"],
                    default="translate",
                    help="Fortran-side deck handling (see module docstring)")
    pp.add_argument("--workdir", default=DEFAULT_WORKDIR)

    cp = sub.add_parser("coverage", help="pyradioss starter keyword census")
    cp.add_argument("decks", nargs="+", help=".rad starter decks")
    cp.add_argument("--timeout", type=float, default=900)
    cp.add_argument("--workdir", default=DEFAULT_WORKDIR)

    args = ap.parse_args(argv)
    if args.mode == "coverage":
        return coverage(args)
    if args.mode is None:
        args = ap.parse_args(["parity"] + (argv or sys.argv[1:]))
    return parity(args)


if __name__ == "__main__":
    sys.exit(main())
