"""
Starter keyword parsers: /KEYWORD blocks → Model.

Fortran origin: the ``hm_read_*.F`` routines under
``starter/source/elements/reader``, ``starter/source/materials``,
``starter/source/properties``, ``starter/source/loads``,
``starter/source/constraints`` ... — one routine per keyword, all driven
from the reading loop in ``starter/source/starter/lectur.F``. This module
has exactly that shape: a dispatch table ``KEYWORD_PARSERS`` mapping the
keyword to a small function ``read_<keyword>(block, model, log)``.

Card layouts
------------
Each parser documents the exact card layout it accepts. The layouts follow
the official Radioss input reference; where this port simplifies (fewer
optional fields, no skew/frame/sensor support yet) the docstring says so.
Fields are read as whitespace-separated tokens (see deck_reader.py for the
free-format/fixed-format discussion); **omitted trailing fields take their
default; blank fields in the middle of a card must be written as 0**.

Unknown keywords are *skipped with a warning*, so real decks containing
not-yet-ported options degrade gracefully instead of crashing — the same
philosophy as the original Starter, which flags unsupported options in the
listing.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..common.messages import MessageLog
from ..common.tables import FunctTable
from ..model.entities import (
    AddedMass, BoundaryCondition, Box, ConcentratedLoad, Damping, Gravity,
    ImposedDisplacement, ImposedVelocity, InitialVelocity, Interface, Line,
    Material, Mpc, NodeGroup, Part, PressureLoad, Property, Rbe3, RigidBody,
    RigidWall, Section, Sensor, Surface, THRequest,
)
from ..model.model import Model
from .deck_reader import Card, KeywordBlock, _to_float


# ----------------------------------------------------------------------------
# Small helpers shared by the parsers
# ----------------------------------------------------------------------------

def _fixed_vals(card: Card, widths: List[int]) -> List[str]:
    """Cut ``card.raw`` at the given column ``widths`` (the Fortran fixed
    format, e.g. ``[10, 10, 20, 20]``), returning stripped strings — ''
    where the line is blank or too short. Used for the REAL fixed-format
    card layouts whose fields are NOT all 10 characters wide (mixed
    ``%10d``/``%20lg`` cards), where a whitespace-token view would shift
    on blank fields."""
    line, pos, out = card.raw, 0, []
    for w in widths:
        out.append(line[pos:pos + w].strip())
        pos += w
    return out


def _ival(s: str, default: int = 0) -> int:
    """Fixed field -> int; blank -> default (Fortran blank-reads-as-zero)."""
    return int(s) if s else default


def _fval(s: str, default: float = 0.0) -> float:
    """Fixed field -> float; blank -> default."""
    return _to_float(s) if s else default


def _is_numeric_card(card: Card) -> bool:
    """True if every token of the card parses as a number — used to decide
    whether the first card of a block is a title or already data."""
    toks = card.tokens()
    if not toks:
        return False
    for t in toks:
        try:
            float(t.replace("D", "E").replace("d", "e"))
        except ValueError:
            return False
    return True


def _title_and_data(block: KeywordBlock):
    """Split block cards into (title, data_cards). Most option blocks start
    with a free-text title card; we accept its absence for convenience."""
    if block.cards and not _is_numeric_card(block.cards[0]):
        return block.cards[0].raw.strip(), block.cards[1:]
    return "", block.cards


def _floats(card: Card, n: int, defaults: Optional[List[float]] = None) -> List[float]:
    """First n tokens as floats; missing trailing tokens take defaults (or 0)."""
    toks = card.floats()
    out = list(toks[:n])
    while len(out) < n:
        d = defaults[len(out)] if defaults and len(out) < len(defaults) else 0.0
        out.append(d)
    return out


def _direction(tok: str) -> np.ndarray:
    """Parse a direction token: the Radioss axis letters X/Y/Z (also
    accepts lowercase). Returns a unit vector."""
    axis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
    t = tok.upper()
    if t not in axis:
        raise ValueError(f"unsupported direction '{tok}' (expected X, Y or Z)")
    return np.array(axis[t], dtype=float)


# ============================================================================
# Control blocks
# ============================================================================

def read_begin(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BEGIN``: first card is the run name (also used as model title).
    The original also reads input version and unit cards here; units are the
    user's responsibility in this port (consistent-units philosophy), so any
    further cards are ignored."""
    if block.cards:
        model.title = block.cards[0].raw.strip()


def read_title(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    if block.cards:
        model.title = block.cards[0].raw.strip()


def read_end(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/END``: nothing to do — the reading loop stops naturally."""


# ============================================================================
# Mesh
# ============================================================================

def read_node(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/NODE`` — one card per node::

        node_ID   Xc   Yc   Zc

    Fortran: starter/source/elements/reader/hm_read_node.F.
    """
    ids, xyz = [], []
    for card in block.cards:
        t = card.tokens()
        if len(t) < 4:
            log.error(f"/NODE card needs 4 fields, got {len(t)}", card.source)
            continue
        ids.append(int(t[0]))
        xyz.append([float(v) for v in card.floats()[1:4]])
    if ids:
        model.add_nodes(np.array(ids), np.array(xyz))


def _read_elems(block: KeywordBlock, model: Model, log: MessageLog,
                etype: str, nnode: int) -> None:
    """Common reader for element blocks ``/BRICK/part_ID`` etc. — one card
    per element::

        elem_ID   node_ID1 ... node_IDk   [ignored extras]

    The block user-ID is the **part ID** the elements belong to (Radioss
    convention since the 44 format: elements are grouped under their part).
    Extra trailing fields (e.g. per-element shell thickness) are ignored
    with a warning, once.
    """
    part_id = block.user_id
    if part_id is None:
        log.error(f"/{etype} block without part id", block.source)
        return
    warned_extra = False
    for card in block.cards:
        t = card.ints()
        if len(t) < 1 + nnode:
            log.error(f"/{etype} card needs {1 + nnode} ids, got {len(t)}",
                      card.source)
            continue
        if len(t) > 1 + nnode and not warned_extra:
            log.warning(f"/{etype}: extra fields on element cards ignored",
                        card.source)
            warned_extra = True
        model.raw_elems[etype].append((t[0], part_id, t[1:1 + nnode]))


def read_brick(block, model, log):
    """``/BRICK/part_ID``: 8-node solids (elem_ID + 8 node IDs).
    Degenerated bricks with 4 distinct nodes (the classic tetra-in-brick
    convention, e.g. n1 n2 n3 n3 n5 n5 n5 n5) are converted to /TETRA4
    elements by the Starter; other repeated-node patterns (penta/pyramid)
    are rejected with a clear error (see initialization.py)."""
    _read_elems(block, model, log, "BRICK", 8)


def read_tetra4(block, model, log):
    """``/TETRA4/part_ID``: 4-node solids (elem_ID + 4 node IDs, base
    triangle 1-2-3 counter-clockwise seen from node 4). Uses the same
    /PROP/TYPE14 (SOLID) property as bricks."""
    _read_elems(block, model, log, "TETRA4", 4)


def read_shell(block, model, log):
    """``/SHELL/part_ID``: 4-node shells (elem_ID + 4 node IDs)."""
    _read_elems(block, model, log, "SHELL", 4)


def read_sh3n(block, model, log):
    """``/SH3N/part_ID``: 3-node shells (elem_ID + 3 node IDs). Uses the
    same /PROP/TYPE1 (SHELL) property as 4-node shells."""
    _read_elems(block, model, log, "SH3N", 3)


def read_truss(block, model, log):
    """``/TRUSS/part_ID``: 2-node trusses (elem_ID + 2 node IDs)."""
    _read_elems(block, model, log, "TRUSS", 2)


def read_spring(block, model, log):
    """``/SPRING/part_ID``: 2-node springs (elem_ID + 2 node IDs)."""
    _read_elems(block, model, log, "SPRING", 2)


def read_beam(block, model, log):
    """``/BEAM/part_ID``: 2-node beams + orientation node (elem_ID + N1 N2
    N3). N3 orients the local y axis (in the N1-N2-N3 plane) and carries
    neither mass nor force — it may be any node, including a standalone
    one (which the mass check then freezes, harmlessly)."""
    _read_elems(block, model, log, "BEAM", 3)


# ============================================================================
# Part / material / property
# ============================================================================

def read_part(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PART/part_ID``::

        card 1:  part_title
        card 2:  prop_ID   mat_ID   [subset_ID  ignored]

    Fortran: starter/source/model/assembling/hm_read_part.F.
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/PART/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].ints()
    model.parts[block.user_id] = Part(
        id=block.user_id, prop_id=t[0], mat_id=t[1], title=title)


def read_mat(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MAT/LAW<n>/mat_ID`` (aliases /MAT/ELAST, /MAT/PLAS_JOHNS,
    /MAT/PLAS_TAB, /MAT/PLAS_BRIT, /MAT/OGDEN).

    LAW1 (linear elastic) — Fortran starter/source/materials/mat/mat001::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  E   nu

    LAW2 (Johnson–Cook) — Fortran .../mat002 (Iflag=0 classic input)::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  E   nu
        card 4:  A   B   n   eps_p_max   sig_max
        card 5:  c   eps_dot_0   [ICC  Fsmooth  F_cut  — ignored]
        card 6:  m   T_melt   rho_Cp   [T_i]          (optional — M6)

      yield stress
      sigma_y = (A + B*eps_p^n) (1 + c*ln(eps_dot/eps_dot_0)) (1 - T*^m)
      capped at sig_max; the element is DELETED when the plastic strain
      reaches eps_p_max (since M3). Card 5 is optional (no rate effect if
      absent). Card 6 (M6) turns on the ADIABATIC thermal terms: the
      plastic work heats the material, dT = sigma_y d(eps_p) / rho_Cp
      (rho_Cp = specific heat per unit volume), and the homologous
      temperature T* = (T - T_i)/(T_melt - T_i) softens the yield stress
      (and feeds /FAIL/JOHNSON's D5 term). T_i defaults to 298 K; there
      is no heat conduction (adiabatic — the crash/impact regime).

    LAW27 (brittle, shells only) — Fortran .../mat027::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  E   nu
        card 4:  eps_t1   eps_m1   dmax1   eps_f1     (crack direction 1)
        card 5:  eps_t2   eps_m2   dmax2   eps_f2     (optional, = card 4)

      tensile cracking: damage starts at strain eps_t, reaches dmax at
      eps_m, layer breaks at eps_f (see law27_brittle.py). The plastic
      block of the original PLAS_BRIT is not ported (elastic to crack).

    LAW36 (tabulated plasticity) — Fortran .../mat036. TWO dialects are
    accepted (dispatched on the card count — the real layout always has
    at least 6 data cards, the compact one at most 5):

      * the port's compact layout::

            card 1:  mat_title
            card 2:  rho_0
            card 3:  E   nu
            card 4:  N_funct   [eps_p_max]
            card 5:  fct_ID1 ... fct_ID_N     (hardening curves eps_p->sig_y)
            card 6:  rate_1 ... rate_N        (required when N_funct > 1,
                     strictly increasing strain rates, one per curve)

      * the REAL fixed-format layout (cfg ``matl36_plas_tab.cfg``
        radioss2017+ / ``hm_read_mat36.F``), as written by real decks::

            card 1:  mat_title
            card 2:  rho_0                                        (%20lg)
            card 3:  E   Nu   Eps_p_max   Eps_t   Eps_m           (5 %20lg)
            card 4:  N_funct  F_smooth  C_hard  F_cut  Eps_f  VP
                     (%10d %10d %20lg %20lg %20lg 10x %10d)
            card 5:  fct_IDp  Fscale  fct_IDE  EInf  CE
                     (%10d %20lg %10d %20lg %20lg)
            card 6+: fct_ID1...  (%10d, 5 per card), then
                     Fscale_1... (%20lg, 5 per card, 0 -> 1.0), then
                     Eps_dot_1... (%20lg, 5 per card)

        Fields the port does not implement (F_smooth/C_hard/F_cut/Eps_f/
        VP/Eps_t/Eps_m, the fct_IDp pressure function, the fct_IDE
        modulus evolution, per-curve Fscale != 1) are accepted and
        reported in ONE warning, mirroring the 'accepted, ignored'
        contract of the original Starter listing.

      the yield stress follows the /FUNCT curves, linearly interpolated
      in strain rate; the element is deleted at eps_p_max (0 = no limit).

    LAW42 (Ogden hyperelastic, solids only) — Fortran .../mat042::

        card 1:  mat_title
        card 2:  rho_0
        card 3:  mu_1  mu_2  mu_3  mu_4  mu_5
        card 4:  alpha_1 ... alpha_5
        card 5:  nu                          (default 0.495, near-incompr.)

      W = sum mu_p/alpha_p (lb1^a + lb2^a + lb3^a - 3) + K/2 (J-1)^2;
      every used pair must satisfy mu_p*alpha_p > 0; the ground-state
      shear modulus is G0 = sum(mu_p*alpha_p)/2 and the derived E, K
      follow from nu (stored in params so the generic elastic machinery
      — time step, contact stiffness — works unchanged).
    """
    lawname = block.parts[1].upper() if len(block.parts) > 1 else ""
    law_aliases = {"LAW1": 1, "ELAST": 1, "LAW2": 2, "PLAS_JOHNS": 2,
                   "LAW27": 27, "PLAS_BRIT": 27,
                   "LAW36": 36, "PLAS_TAB": 36,
                   "LAW42": 42, "OGDEN": 42}
    if lawname not in law_aliases:
        log.warning(f"/MAT/{lawname} not ported — material skipped "
                    f"(supported: LAW1/ELAST, LAW2/PLAS_JOHNS, "
                    f"LAW27/PLAS_BRIT, LAW36/PLAS_TAB, LAW42/OGDEN)",
                    block.source)
        return
    law = law_aliases[lawname]
    title, cards = _title_and_data(block)
    if len(cards) < 2:
        log.error(f"/MAT/{lawname}/{block.user_id}: needs at least rho and "
                  f"elasticity cards", block.source)
        return
    rho0 = cards[0].floats()[0]

    if law == 42:
        # cards: mu / alpha / nu — the elastic constants are DERIVED
        mu = _floats(cards[1], 5)
        al = _floats(cards[2], 5) if len(cards) >= 3 else [0.0] * 5
        nu = cards[3].floats()[0] if len(cards) >= 4 else 0.495
        nu = nu if nu > 0 else 0.495
        used = [(m, a) for m, a in zip(mu, al) if m != 0.0]
        if not used:
            log.error(f"/MAT/LAW42/{block.user_id}: all mu_p are zero",
                      block.source)
            return
        if any(m * a <= 0.0 for m, a in used):
            log.error(f"/MAT/LAW42/{block.user_id}: every Ogden pair must "
                      f"satisfy mu_p * alpha_p > 0 (material stability)",
                      block.source)
            return
        G0 = sum(m * a for m, a in used) / 2.0
        params = {"E": 2.0 * G0 * (1.0 + nu), "nu": nu,
                  "mu": [m for m, _ in used], "alpha": [a for _, a in used]}
        model.materials[block.user_id] = Material(
            id=block.user_id, law=law, rho0=rho0, title=title, params=params)
        return

    E, nu = _floats(cards[1], 2)
    params = {"E": E, "nu": nu}
    if law == 2:
        if len(cards) < 3:
            log.error(f"/MAT/LAW2/{block.user_id}: missing A,B,n card",
                      block.source)
            return
        A, B, n, epsmax, sigmax = _floats(
            cards[2], 5, defaults=[0, 0, 1.0, 1e30, 1e30])
        # Radioss conventions: eps_p_max=0 means "no limit", sig_max=0 too.
        params.update(A=A, B=B, n=n if n > 0 else 1.0,
                      eps_p_max=epsmax if epsmax > 0 else 1e30,
                      sig_max=sigmax if sigmax > 0 else 1e30)
        if len(cards) >= 4:
            c, eps0 = _floats(cards[3], 2, defaults=[0.0, 1.0])
            params.update(c=c, eps_dot_0=eps0 if eps0 > 0 else 1.0)
        else:
            params.update(c=0.0, eps_dot_0=1.0)
        if len(cards) >= 5:                    # thermal card (M6)
            mT, tmelt, rho_cp, ti = _floats(
                cards[4], 4, defaults=[0.0, 0.0, 0.0, 298.0])
            ti = ti if ti > 0 else 298.0
            if mT > 0 and (tmelt <= ti or rho_cp <= 0):
                log.error(f"/MAT/LAW2/{block.user_id}: thermal card needs "
                          f"T_melt > T_i and rho_Cp > 0", block.source)
            elif mT > 0:
                params.update(mT=mT, T_melt=tmelt, rho_cp=rho_cp, T_i=ti)
    elif law == 27:
        if len(cards) < 3:
            log.error(f"/MAT/LAW27/{block.user_id}: missing damage card "
                      f"'eps_t1 eps_m1 dmax1 eps_f1'", block.source)
            return
        t1, m1, d1, f1 = _floats(cards[2], 4,
                                 defaults=[0.0, 0.0, 0.999, 1e30])
        if not (0.0 < t1 < m1):
            log.error(f"/MAT/LAW27/{block.user_id}: need 0 < eps_t1 < "
                      f"eps_m1", block.source)
            return
        d1 = min(d1 if d1 > 0 else 0.999, 1.0)
        f1 = f1 if f1 > 0 else 1e30
        if len(cards) >= 4:
            t2, m2, d2, f2 = _floats(cards[3], 4, defaults=[t1, m1, d1, f1])
            t2, m2 = (t2 if t2 > 0 else t1), (m2 if m2 > 0 else m1)
            d2 = min(d2 if d2 > 0 else d1, 1.0)
            f2 = f2 if f2 > 0 else f1
        else:
            t2, m2, d2, f2 = t1, m1, d1, f1
        params.update(eps_t1=t1, eps_m1=m1, dmax1=d1, eps_f1=f1,
                      eps_t2=t2, eps_m2=m2, dmax2=d2, eps_f2=f2)
    elif law == 36:
        if len(cards) < 4:
            log.error(f"/MAT/LAW36/{block.user_id}: needs N_funct and "
                      f"function-ID cards", block.source)
            return
        if len(cards) >= 6:
            # ---- the REAL fixed-format layout (see docstring) -------------
            # rho / E-card already read; Eps_p_max sits ON the E-card here.
            v1 = _floats(cards[1], 5)
            params["eps_p_max"] = v1[2] if v1[2] > 0 else 1e30
            ign: List[str] = []          # accepted-but-not-ported fields
            if v1[3] != 0.0 or v1[4] != 0.0:
                ign.append(f"Eps_t={v1[3]:g} Eps_m={v1[4]:g}")
            # card 4: N_funct F_smooth C_hard F_cut Eps_f (10 blank) VP
            f2 = _fixed_vals(cards[2], [10, 10, 20, 20, 20, 10, 10])
            nfun = _ival(f2[0])
            if nfun <= 0:
                log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} "
                          f"(needs at least one hardening curve)",
                          block.source)
                return
            for name, s in (("F_smooth", f2[1]), ("C_hard", f2[2]),
                            ("F_cut", f2[3]), ("Eps_f", f2[4]),
                            ("VP", f2[6])):
                if s and _to_float(s) != 0.0:
                    ign.append(f"{name}={s}")
            # card 5: fct_IDp Fscale fct_IDE EInf CE
            f3 = _fixed_vals(cards[3], [10, 20, 10, 20, 20])
            if _ival(f3[0]) != 0:
                ign.append(f"fct_IDp={f3[0]} (pressure-dependent yield)")
            if _ival(f3[2]) != 0:
                ign.append(f"fct_IDE={f3[2]} (modulus evolution EInf/CE)")
            # function-id cards (5 per card), then Fscale_i, then Eps_dot_i
            idx, fids = 4, []
            while idx < len(cards) and len(fids) < nfun:
                fids.extend(cards[idx].ints())
                idx += 1
            if len(fids) < nfun:
                log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} but "
                          f"only {len(fids)} function ids given",
                          block.source)
                return
            params["funct_ids"] = fids[:nfun]
            nlist = (nfun + 4) // 5
            yfac: List[float] = []
            for _ in range(nlist):
                if idx < len(cards):
                    yfac.extend(cards[idx].floats())
                    idx += 1
            # hm_read_mat36.F: YFAC == 0 -> 1.0 (default scale)
            yfac = [y if y != 0.0 else 1.0 for y in yfac[:nfun]]
            if any(y != 1.0 for y in yfac):
                ign.append(f"Fscale_i={yfac} (curves used unscaled)")
            rates: List[float] = []
            for _ in range(nlist):
                if idx < len(cards):
                    rates.extend(cards[idx].floats())
                    idx += 1
            if nfun > 1:
                if len(rates) < nfun:
                    log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} "
                              f"needs {nfun} strain rates (Eps_dot_i cards)",
                              block.source)
                    return
                rates = rates[:nfun]
                if any(b <= a for a, b in zip(rates, rates[1:])):
                    log.error(f"/MAT/LAW36/{block.user_id}: strain rates "
                              f"must be strictly increasing", block.source)
                    return
                params["rates"] = rates
            else:
                params["rates"] = [0.0]
            if ign:
                log.warning(f"/MAT/LAW36/{block.user_id}: real-format "
                            f"fields not ported — ignored: "
                            f"{'; '.join(ign)}", block.source)
        else:
            # ---- the port's compact layout ---------------------------------
            v = _floats(cards[2], 2, defaults=[1, 0.0])
            nfun = int(v[0]) if v[0] > 0 else 1
            params["eps_p_max"] = v[1] if v[1] > 0 else 1e30
            fids = cards[3].ints()
            if len(fids) < nfun:
                log.error(f"/MAT/LAW36/{block.user_id}: N_funct={nfun} but "
                          f"only {len(fids)} function ids given",
                          block.source)
                return
            params["funct_ids"] = fids[:nfun]
            if nfun > 1:
                if len(cards) < 5:
                    log.error(f"/MAT/LAW36/{block.user_id}: N_funct>1 needs "
                              f"a strain-rate card", block.source)
                    return
                rates = _floats(cards[4], nfun)
                if any(b <= a for a, b in zip(rates, rates[1:])):
                    log.error(f"/MAT/LAW36/{block.user_id}: strain rates "
                              f"must be strictly increasing", block.source)
                    return
                params["rates"] = rates
            else:
                params["rates"] = [0.0]
    model.materials[block.user_id] = Material(
        id=block.user_id, law=law, rho0=rho0, title=title, params=params)


def read_fail(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FAIL/JOHNSON/mat_ID`` and ``/FAIL/BIQUAD/mat_ID``: attach a
    failure criterion to a material (the trailing id IS the material id —
    Radioss convention; there is no title card).

    JOHNSON — Fortran starter/source/materials/fail/johnson_cook::

        card 1:  D1   D2   D3   D4   [D5]
        card 2:  eps_dot_0   Ifail_sh        (optional; defaults 1.0, 1)

      eps_f = (D1 + D2*exp(D3*sigma*)) * (1 + D4*ln(rate/eps_dot_0))
              * (1 + D5*T*),
      damage D += d_eps_p/eps_f, break at D >= 1. Ifail_sh: 1 = delete
      the shell when ONE layer breaks (default), 2 = when ALL layers do.
      D5 (M6) needs the material's adiabatic temperature (the LAW2
      thermal card) — the Starter warns and drops it otherwise.

    BIQUAD — Fortran starter/source/materials/fail/biquad::

        card 1:  c1   c2   c3   c4   c5
        card 2:  Ifail_sh                    (optional; default 1)

      failure plastic strains at triaxialities -1/3, 0, 1/3, 2/3, 1 —
      two parabolas through them (see pyradioss/failure/biquad.py). The
      M-flag material presets and S-flag of the original are not ported:
      give the five coefficients explicitly.

    Solids break when their single integration point does (the ported
    hexa/tetra are one-point elements, so the original's Ifail_so
    variants are moot here).
    """
    from ..failure import biquad as fail_biquad
    from ..model.entities import FailureModel
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind not in ("JOHNSON", "BIQUAD"):
        log.warning(f"/FAIL/{kind} not ported — skipped "
                    f"(supported: JOHNSON, BIQUAD)", block.source)
        return
    mat_id = block.user_id
    cards = block.cards
    if not cards:
        log.error(f"/FAIL/{kind}/{mat_id}: missing data card", block.source)
        return
    if kind == "JOHNSON":
        D1, D2, D3, D4, D5 = _floats(cards[0], 5)
        eps0, ifail_sh = 1.0, 1
        if len(cards) > 1:
            v = _floats(cards[1], 2, defaults=[1.0, 1])
            eps0 = v[0] if v[0] > 0 else 1.0
            ifail_sh = int(v[1]) if v[1] in (1, 2) else 1
        fm = FailureModel(type="JOHNSON", ifail_sh=ifail_sh,
                          params={"D1": D1, "D2": D2, "D3": D3, "D4": D4,
                                  "D5": D5, "eps_dot_0": eps0})
        if D1 <= 0.0 and D2 <= 0.0:
            log.error(f"/FAIL/JOHNSON/{mat_id}: D1 and D2 both <= 0 gives "
                      f"a zero failure strain", block.source)
            return
    else:  # BIQUAD
        c1, c2, c3, c4, c5 = _floats(cards[0], 5)
        if min(c1, c2, c3, c4, c5) <= 0.0:
            log.error(f"/FAIL/BIQUAD/{mat_id}: all five failure strains "
                      f"c1..c5 must be > 0 (presets not ported)",
                      block.source)
            return
        ifail_sh = 1
        if len(cards) > 1:
            v = cards[1].ints()
            if v and v[0] in (1, 2):
                ifail_sh = v[0]
        params = {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}
        fail_biquad.fit(params)   # pre-compute the two parabolas
        fm = FailureModel(type="BIQUAD", ifail_sh=ifail_sh, params=params)
    # attachment to the material happens in the Starter resolve step
    # (initialization.resolve_materials) so deck order does not matter
    model.raw_fails.append((mat_id, fm, block.source))


def read_eos(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/EOS/POLYNOMIAL/mat_ID`` and ``/EOS/IDEAL-GAS/mat_ID`` (M6):
    attach an equation of state to a material (the trailing id IS the
    material id, like /FAIL; the EOS pressure then replaces the law's
    own pressure for solid elements — laws 1, 2 and 36).

    POLYNOMIAL — Fortran starter/source/materials/eos (polynomial)::

        card 1:  C0   C1   C2   C3   C4   C5
        card 2:  E0                       (initial energy per unit
                                           initial volume; optional, 0)

      p = C0 + C1*mu + C2*max(mu,0)^2 + C3*mu^3 + (C4 + C5*mu)*E with
      mu = rho/rho0 - 1 (C2 dropped in tension, Radioss convention).

    IDEAL-GAS::

        card 1:  gamma   P0

      the perfect gas p = (gamma-1)*(1+mu)*E, stored as the equivalent
      polynomial C4 = C5 = gamma-1 with E0 = P0/(gamma-1). P0 > 0 makes
      a pre-pressurized gas (it pushes from cycle 1 — confine it).
    """
    from ..model.entities import EquationOfState
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    kind = {"IDEAL_GAS": "IDEAL-GAS"}.get(kind, kind)
    if kind not in ("POLYNOMIAL", "IDEAL-GAS"):
        log.warning(f"/EOS/{kind} not ported — skipped (supported: "
                    f"POLYNOMIAL, IDEAL-GAS)", block.source)
        return
    mat_id = block.user_id
    cards = block.cards
    if not cards:
        log.error(f"/EOS/{kind}/{mat_id}: missing data card", block.source)
        return
    if kind == "POLYNOMIAL":
        c0, c1, c2, c3, c4, c5 = _floats(cards[0], 6)
        e0 = cards[1].floats()[0] if len(cards) > 1 else 0.0
        params = {"c0": c0, "c1": c1, "c2": c2, "c3": c3, "c4": c4,
                  "c5": c5, "e0": e0}
    else:
        gamma, p0 = _floats(cards[0], 2, defaults=[1.4, 0.0])
        if gamma <= 1.0:
            log.error(f"/EOS/IDEAL-GAS/{mat_id}: gamma must be > 1",
                      block.source)
            return
        params = {"c0": 0.0, "c1": 0.0, "c2": 0.0, "c3": 0.0,
                  "c4": gamma - 1.0, "c5": gamma - 1.0,
                  "e0": p0 / (gamma - 1.0), "gamma": gamma}
    model.raw_eos.append((mat_id, EquationOfState(kind=kind, params=params),
                          block.source))


def read_prop(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PROP/TYPE<n>/prop_ID`` (aliases SHELL, TRUSS, SPRING, SOLID).

    TYPE1 / SHELL  (Fortran starter/source/properties/p01_shell)::

        card 1:  prop_title
        card 2:  Ishell  Ismstr  Ish3n  Idrill        (ints — only read,
                 the port always uses the Belytschko–Tsay formulation)
        card 3:  hm   hf   hr   dm   dn               (hourglass coefficients:
                 membrane, flexural, rotational; d* damping — dm/dn ignored)
        card 4:  N   Istrain   Thick                  (N = through-thickness
                 integration points, default 3; Thick = shell thickness)

      Cards 2 and 3 may be omitted **only together with everything after
      them**, so in practice give all 4 cards. To keep tiny decks easy, a
      block whose first data card holds a float that is not an int is
      interpreted as the short form:   card 2: Thick  [N]  [hm]

    TYPE2 / TRUSS::   card 1: title,  card 2: Area

    TYPE3 / BEAM  (Fortran starter/source/properties/p03_beam)::

        card 1:  prop_title
        card 2:  Ishear  dm  df       (flags/damping — read and ignored:
                 the port always includes Timoshenko shear, no damping)
        card 3:  Area   Iyy   Izz   Ixx

      Iyy/Izz = bending inertias about the local y/z axes, Ixx = torsion
      constant. Ixx = 0 defaults to Iyy + Izz (polar, exact for circular
      sections only — give the real torsion constant for others).
      Short form: a single data card 'Area Iyy Izz Ixx'.

    TYPE4 / SPRING:: card 1: title,  card 2: Mass  K  C
      (linear spring: F = K*dl + C*dl_dot; Mass is lumped half/half)

    TYPE14 / SOLID:: card 1: title,
        card 2:  Isolid  Ismstr  ...  (ints — read and ignored: 1-point +
                 Flanagan–Belytschko hourglass is the only ported option)
        card 3:  qa   qb   h          (bulk viscosity quadratic/linear,
                 hourglass coefficient; defaults 1.1 / 0.05 / 0.1)
      Short form: a single data card with 'qa qb h' floats, or none at all
      (all defaults).
    """
    typename = block.parts[1].upper() if len(block.parts) > 1 else ""
    aliases = {"TYPE1": 1, "SHELL": 1, "TYPE2": 2, "TRUSS": 2,
               "TYPE3": 3, "BEAM": 3,
               "TYPE4": 4, "SPRING": 4, "TYPE14": 14, "SOLID": 14}
    if typename not in aliases:
        log.warning(f"/PROP/{typename} not ported — property skipped "
                    f"(supported: TYPE1/SHELL, TYPE2/TRUSS, TYPE3/BEAM, "
                    f"TYPE4/SPRING, TYPE14/SOLID)", block.source)
        return
    ptype = aliases[typename]
    title, cards = _title_and_data(block)
    params: Dict[str, float] = {}

    if ptype == 1:  # SHELL
        params = {"thick": 1.0, "nip": 3, "hm": 0.01, "hf": 0.01, "hr": 0.01}
        # Detect short form: first card contains a non-integer float.
        if cards and any("." in tok or "e" in tok.lower()
                         for tok in cards[0].tokens()):
            vals = _floats(cards[0], 3, defaults=[1.0, 3, 0.01])
            params["thick"], params["nip"], params["hm"] = \
                vals[0], int(vals[1]) if vals[1] else 3, vals[2] or 0.01
            params["hf"] = params["hr"] = params["hm"]
        else:
            # full form: skip flags card, read hourglass + N/Thick cards
            if len(cards) >= 2:
                hm, hf, hr = _floats(cards[1], 3,
                                     defaults=[0.01, 0.01, 0.01])[:3]
                params["hm"], params["hf"], params["hr"] = \
                    hm or 0.01, hf or 0.01, hr or 0.01
            if len(cards) >= 3:
                v = _floats(cards[2], 3, defaults=[3, 0, 1.0])
                params["nip"] = int(v[0]) if v[0] else 3
                params["thick"] = v[2]
            else:
                log.error(f"/PROP/SHELL/{block.user_id}: thickness card "
                          f"missing", block.source)
    elif ptype == 2:  # TRUSS
        if not cards:
            log.error(f"/PROP/TRUSS/{block.user_id}: area card missing",
                      block.source)
            return
        params = {"area": cards[0].floats()[0]}
    elif ptype == 3:  # BEAM
        # skip pure-integer flag cards (Ishear...), read the section card
        data = [c for c in cards if not all(tok.lstrip("+-").isdigit()
                                            for tok in c.tokens())]
        if not data:
            log.error(f"/PROP/BEAM/{block.user_id}: section card "
                      f"'Area Iyy Izz Ixx' missing", block.source)
            return
        a, iyy, izz, ixx = _floats(data[0], 4)
        if a <= 0 or iyy <= 0 or izz <= 0:
            log.error(f"/PROP/BEAM/{block.user_id}: Area, Iyy and Izz "
                      f"must be > 0", block.source)
            return
        params = {"area": a, "iyy": iyy, "izz": izz,
                  "ixx": ixx if ixx > 0 else iyy + izz}
    elif ptype == 4:  # SPRING
        if not cards:
            log.error(f"/PROP/SPRING/{block.user_id}: data card missing",
                      block.source)
            return
        m, k, c = _floats(cards[0], 3)
        params = {"mass": m, "k": k, "c": c}
    elif ptype == 14:  # SOLID
        from ..common.constants import DEFAULT_HOURGLASS, DEFAULT_QA, DEFAULT_QB
        params = {"qa": DEFAULT_QA, "qb": DEFAULT_QB, "h": DEFAULT_HOURGLASS}
        data = [c for c in cards if not all(tok.lstrip("+-").isdigit()
                                            for tok in c.tokens())]
        if data:
            qa, qb, h = _floats(data[0], 3,
                                defaults=[DEFAULT_QA, DEFAULT_QB,
                                          DEFAULT_HOURGLASS])
            params = {"qa": qa or DEFAULT_QA, "qb": qb or DEFAULT_QB,
                      "h": h or DEFAULT_HOURGLASS}

    model.properties[block.user_id] = Property(
        id=block.user_id, type=ptype, title=title, params=params)


# ============================================================================
# Functions, groups, boxes, surfaces
# ============================================================================

def read_funct(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/FUNCT/fct_ID``: title card then one (X, Y) pair per card."""
    title, cards = _title_and_data(block)
    pts = [(_floats(c, 2)[0], _floats(c, 2)[1]) for c in cards if c.tokens()]
    if len(pts) < 2:
        log.error(f"/FUNCT/{block.user_id}: needs at least 2 points",
                  block.source)
        return
    x, y = zip(*pts)
    model.functions[block.user_id] = FunctTable(block.user_id, x, y, title)


def read_grnod(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GRNOD/NODE|PART|BOX/grnod_ID``: title card, then entity IDs
    (any number per card). NODE lists nodes, PART takes all nodes of the
    parts, BOX takes all nodes inside /BOX volumes."""
    kind = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    title, cards = _title_and_data(block)
    ids: List[int] = []
    for c in cards:
        ids.extend(c.ints())
    g = model.node_groups.setdefault(
        block.user_id, NodeGroup(id=block.user_id, title=title))
    if kind == "NODE":
        g.node_ids.extend(ids)
    elif kind == "PART":
        g.part_ids.extend(ids)
    elif kind == "BOX":
        g.box_ids.extend(ids)
    else:
        log.warning(f"/GRNOD/{kind} not ported (NODE, PART, BOX supported)",
                    block.source)


def read_box(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BOX/RECTA/box_ID``: title card, then the two diagonal corner
    points — either one card with 6 floats or two cards with 3 floats."""
    title, cards = _title_and_data(block)
    vals: List[float] = []
    for c in cards:
        vals.extend(c.floats())
    if len(vals) < 6:
        log.error(f"/BOX/RECTA/{block.user_id}: needs 6 coordinates",
                  block.source)
        return
    p1, p2 = np.array(vals[:3]), np.array(vals[3:6])
    model.boxes[block.user_id] = Box(
        id=block.user_id, corner_min=np.minimum(p1, p2),
        corner_max=np.maximum(p1, p2), title=title)


def read_surf(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SURF/PART|SEG/surf_ID``: title card, then

    * PART: part IDs (the Starter extracts the free outer faces / shell
      faces of those parts into segments). The real Radioss ``EXT``
      qualifier (``/SURF/PART/EXT`` — external faces only) is accepted
      but IGNORED with a warning: the port's /SURF/PART extraction
      already returns the free outer faces, but the two treatments are
      not guaranteed identical on every mesh.
    * SEG:  one segment per card, in either dialect —

        - port compact: ``n1 n2 n3 [n4]``  (3 ids = triangle),
        - REAL fixed format (``hm_read_surf.F`` 'SEG'):
          ``seg_ID n1 n2 n3 n4`` — 5 fields; the leading segment id is
          dropped, ``n4 = 0`` means a triangle (upstream: N4=0 -> N3).

      Cards with 4 ids are read as the compact quad ``n1..n4`` — a REAL
      triangle card that leaves N4 blank instead of writing 0 is
      ambiguous with it and would be misread (real writers, e.g. k2rad,
      write all 5 fields).
    """
    kparts = block.keyword.split("/")      # ids already stripped
    kind = kparts[1] if len(kparts) > 1 else "SEG"
    title, cards = _title_and_data(block)
    s = model.surfaces.setdefault(
        block.user_id, Surface(id=block.user_id, title=title))
    if kind == "PART":
        quals = kparts[2:]
        if quals:
            log.warning(f"/SURF/PART/{'/'.join(quals)}/{block.user_id}: the "
                        f"{'/'.join(quals)} qualifier is ignored — treated "
                        f"as plain /SURF/PART (the port extracts the free "
                        f"outer faces of the parts)", block.source)
        for c in cards:
            s.part_ids.extend(c.ints())
    elif kind == "SEG":
        for c in cards:
            t = c.ints()
            if len(t) == 5:
                t = t[1:]                  # real dialect: drop seg_ID
            if len(t) == 3:
                t = t + [t[2]]
            if len(t) != 4:
                log.error(f"/SURF/SEG card needs 3 or 4 node ids, or "
                          f"seg_ID + 4 node ids (real format)", c.source)
                continue
            if t[3] == 0:
                t[3] = t[2]                # upstream: N4 = 0 -> triangle
            s.seg_nodes.append(t)
    else:
        log.warning(f"/SURF/{kind} not ported (PART, SEG supported)",
                    block.source)


# ============================================================================
# Boundary conditions, initial conditions, loads
# ============================================================================

def read_bcs(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/BCS/bcs_ID``::

        card 1:  title
        card 2:  Trarot   skew_ID   grnod_ID

    ``Trarot`` is the classic pair of 3-digit binary flags
    ``XYZ XYZ`` — first triple = translations, second = rotations,
    1 = fixed. Example: ``111 000`` clamps translations only.
    skew_ID must be 0 (skew frames not ported).
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/BCS/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    if len(t) < 4:
        log.error(f"/BCS/{block.user_id}: card 2 needs 'tra rot skew grnod'",
                  block.source)
        return
    tra, rot, skew, grnod = t[0], t[1], int(t[2]), int(t[3])
    if skew != 0:
        log.warning(f"/BCS/{block.user_id}: skew frames not ported, "
                    f"skew_ID ignored", block.source)
    fix_tra = np.array([ch == "1" for ch in tra.zfill(3)])
    fix_rot = np.array([ch == "1" for ch in rot.zfill(3)])
    model.bcs.append(BoundaryCondition(
        id=block.user_id, grnod_id=grnod, fix_tra=fix_tra, fix_rot=fix_rot,
        title=title))


def read_inivel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INIVEL/TRA/inivel_ID``::

        card 1:  title
        card 2:  Vx   Vy   Vz   grnod_ID

    ``/INIVEL/AXIS/inivel_ID`` (M5) — initial rotation about an axis::

        card 1:  title
        card 2:  omega   Dir(X|Y|Z)   grnod_ID   Xp   Yp   Zp

      every node of the group receives v += omega * d x (x0 - P), the
      velocity field of a rigid rotation at rate omega about the axis
      through P = (Xp,Yp,Zp) along Dir. This is how a spinning /RBODY is
      initialized. (The translational Vt fields of the full Radioss AXIS
      card are covered by adding a /INIVEL/TRA on the same group.)
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "TRA"
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/INIVEL/{block.user_id}: missing data card", block.source)
        return
    if kind == "TRA":
        v = _floats(cards[0], 3)
        toks = cards[0].tokens()
        grnod = int(float(toks[3])) if len(toks) > 3 else 0
        model.inivel.append(InitialVelocity(
            id=block.user_id, grnod_id=grnod, v=np.array(v), title=title))
    elif kind == "AXIS":
        t = cards[0].tokens()
        omega = float(t[0])
        axis = _direction(t[1])
        grnod = int(t[2]) if len(t) > 2 else 0
        origin = np.array([float(x) for x in t[3:6]]) if len(t) >= 6 \
            else np.zeros(3)
        model.inivel.append(InitialVelocity(
            id=block.user_id, grnod_id=grnod, v=np.zeros(3), title=title,
            kind="AXIS", omega=omega, axis=axis, origin=origin))
    else:
        log.warning(f"/INIVEL/{kind} not ported (TRA, AXIS supported)",
                    block.source)


def read_grav(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/GRAV/grav_ID``::

        card 1:  title
        card 2:  fct_ID   Dir(X|Y|Z)   grnod_ID   Fscale

      acceleration a(t) = Fscale * f(t) applied along Dir to the group
      (grnod_ID = 0 → all nodes). Fscale defaults to 1.
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/GRAV/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    fct = int(t[0])
    direction = _direction(t[1])
    grnod = int(t[2]) if len(t) > 2 else 0
    scale = float(t[3]) if len(t) > 3 else 1.0
    model.gravity.append(Gravity(
        id=block.user_id, grnod_id=grnod or None, funct_id=fct,
        direction=direction, scale=scale, title=title))


def read_cload(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/CLOAD/cload_ID``::

        card 1:  title
        card 2:  fct_ID   Dir(X|Y|Z)   grnod_ID   Fscale   [sens_ID]

      force F(t) = Fscale * f(t) applied along Dir to EVERY node of the
      group (Radioss semantics: per node, not divided among them).
      sens_ID (M6): the load waits for /SENSOR sens_ID and then follows
      f(t - t_fire) — the curve is the load's own history from activation.
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/CLOAD/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    model.cloads.append(ConcentratedLoad(
        id=block.user_id, funct_id=int(t[0]), direction=_direction(t[1]),
        grnod_id=int(t[2]), scale=float(t[3]) if len(t) > 3 else 1.0,
        sens_id=int(float(t[4])) if len(t) > 4 else 0, title=title))


def read_impvel(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPVEL/impvel_ID``::

        card 1:  title
        card 2:  fct_ID   Dir(X|Y|Z)   grnod_ID   Fscale

      kinematic condition v(t) = Fscale * f(t) imposed on that DOF of the
      group's nodes (overrides equations of motion for that DOF).
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/IMPVEL/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    dof = {"X": 0, "Y": 1, "Z": 2}[t[1].upper()]
    model.impvel.append(ImposedVelocity(
        id=block.user_id, funct_id=int(t[0]), dof=dof, grnod_id=int(t[2]),
        scale=float(t[3]) if len(t) > 3 else 1.0, title=title))


def read_impdisp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/IMPDISP/impdisp_ID`` (M5)::

        card 1:  title
        card 2:  fct_ID   Dir(X|Y|Z)   grnod_ID   Fscale

      kinematic condition: the DOF's displacement follows
      d(t) = Fscale * f(t) exactly (the Engine sets the velocity each
      cycle so the node lands at x0 + d(t+dt)). The curve should start at
      f(0) = 0 — a nonzero start makes the node JUMP in the first cycle.
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/IMPDISP/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    dof = {"X": 0, "Y": 1, "Z": 2}[t[1].upper()]
    model.impdisp.append(ImposedDisplacement(
        id=block.user_id, funct_id=int(t[0]), dof=dof, grnod_id=int(t[2]),
        scale=float(t[3]) if len(t) > 3 else 1.0, title=title))


def read_pload(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/PLOAD/pload_ID`` (M5)::

        card 1:  title
        card 2:  surf_ID   fct_ID   Fscale   [sens_ID]

      follower pressure p(t) = Fscale * f(t) on every segment of the
      surface, acting along the current segment normal (node ordering
      n1-n2-n3-n4, right-hand rule: positive p pushes along +n). The
      resultant p*A of each segment is lumped to its corners.
      sens_ID (M6): waits for /SENSOR sens_ID, then follows p(t - t_fire).
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/PLOAD/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    model.ploads.append(PressureLoad(
        id=block.user_id, surf_id=int(t[0]), funct_id=int(t[1]),
        scale=float(t[2]) if len(t) > 2 else 1.0,
        sens_id=int(float(t[3])) if len(t) > 3 else 0, title=title))


def read_admas(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/ADMAS/admas_ID`` (M5)::

        card 1:  title
        card 2:  Mass   grnod_ID

      adds Mass to EVERY node of the group (the Radioss type-0 per-node
      semantics; the distributed-total variants are not ported). Applied
      before the massless-node check, so a standalone node + /ADMAS is a
      legitimate free point mass — e.g. the carrier node of a moving
      /RWALL.
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/ADMAS/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    mass = float(t[0])
    if mass <= 0.0:
        log.error(f"/ADMAS/{block.user_id}: Mass must be > 0", block.source)
        return
    model.admas.append(AddedMass(
        id=block.user_id, grnod_id=int(t[1]), mass=mass, title=title))


def read_damp(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/DAMP/damp_ID`` (M6)::

        card 1:  title
        card 2:  Alpha   grnod_ID   [Tstart   Tstop]

      Rayleigh MASS damping: force f = -Alpha * m * v on every node of
      the group while Tstart <= t <= Tstop (defaults: the whole run).
      Applied as the exact per-cycle integrating factor and its
      dissipation booked into the DE ledger — see engine/damping.py.
      The stiffness-proportional Beta branch of the Radioss card is not
      ported (it needs K*v products this explicit port never assembles).
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/DAMP/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    alpha = float(t[0])
    if alpha < 0.0:
        log.error(f"/DAMP/{block.user_id}: Alpha must be >= 0", block.source)
        return
    model.damps.append(Damping(
        id=block.user_id, alpha=alpha, grnod_id=int(t[1]),
        tstart=float(t[2]) if len(t) > 2 else 0.0,
        tstop=float(t[3]) if len(t) > 3 else 1e30, title=title))


def read_sensor(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SENSOR/TIME/sens_ID`` and ``/SENSOR/DISP/sens_ID`` (M6)::

        /SENSOR/TIME:  card 1: title,  card 2: Tdelay
        /SENSOR/DISP:  card 1: title,  card 2: node_ID   Dmin

      TIME fires at t = Tdelay; DISP fires when the node's displacement
      magnitude first exceeds Dmin. Sensors LATCH (once fired, active
      forever) and gate /CLOAD, /PLOAD and /INTER/TYPE7|11 through their
      sens_ID field — see engine/sensors.py for the exact semantics.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind not in ("TIME", "DISP"):
        log.warning(f"/SENSOR/{kind} not ported (TIME, DISP supported)",
                    block.source)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/SENSOR/{block.user_id}: missing data card",
                  block.source)
        return
    t = cards[0].tokens()
    if kind == "TIME":
        model.sensors.append(Sensor(
            id=block.user_id, kind="TIME", tdelay=float(t[0]), title=title))
    else:
        if len(t) < 2:
            log.error(f"/SENSOR/DISP/{block.user_id}: card 2 needs "
                      f"'node_ID Dmin'", block.source)
            return
        dmin = float(t[1])
        if dmin <= 0.0:
            log.error(f"/SENSOR/DISP/{block.user_id}: Dmin must be > 0",
                      block.source)
            return
        model.sensors.append(Sensor(
            id=block.user_id, kind="DISP", node_id=int(t[0]), dmin=dmin,
            title=title))


def read_mpc(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/MPC/mpc_ID`` (M6) — one linear multi-point constraint row::

        card 1:  title
        card 2+: node_ID   dof   coef        (one term per card)

      imposing  sum_k coef_k * u(node_k, dof_k) = 0  with dof 1/2/3 the
      X/Y/Z translations and 4/5/6 the rotations (rotational terms need
      the node to carry rotational inertia — shell/beam nodes). At least
      two terms are required. See engine/mpc.py for the Lagrange
      treatment and its zero-work property.
    """
    title, cards = _title_and_data(block)
    nodes, dofs, coefs = [], [], []
    for c in cards:
        t = c.tokens()
        if len(t) < 3:
            log.error(f"/MPC/{block.user_id}: term card needs "
                      f"'node_ID dof coef'", c.source)
            continue
        dof = int(t[1])
        if dof not in (1, 2, 3, 4, 5, 6):
            log.error(f"/MPC/{block.user_id}: dof must be 1..6, got {dof}",
                      c.source)
            continue
        nodes.append(int(t[0]))
        dofs.append(dof)
        coefs.append(float(t[2]))
    if len(nodes) < 2:
        log.error(f"/MPC/{block.user_id}: a constraint needs at least two "
                  f"terms", block.source)
        return
    model.mpcs.append(Mpc(id=block.user_id, node_ids=nodes, dofs=dofs,
                          coefs=coefs, title=title))


# ============================================================================
# Rigid bodies, rigid links, interpolation constraints, sections (M5)
# ============================================================================

def read_rbody(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RBODY/rbody_ID`` (M5)::

        card 1:  title
        card 2:  node_ID   grnod_ID   Mass   Icog
        card 3:  Jxx   Jyy   Jzz          (optional added inertia)

      node_ID = master node (usually a standalone node); grnod_ID = the
      slave nodes. The Starter computes the body mass, center of gravity
      and inertia tensor from the slave nodal masses; Mass and Jxx/Jyy/Jzz
      are extra mass/inertia lumped at the COG. Icog = 1 (default) moves
      the master node to the COG (the Radioss ICoG behaviour); 0 keeps it
      where it is (it is then simply carried rigidly).

      Not ported from the full card (documented M5 simplifications):
      sensors, skew/spherical inertia frames, IKREM and the surface
      envelope.
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/RBODY/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].tokens()
    if len(t) < 2:
        log.error(f"/RBODY/{block.user_id}: card 2 needs 'node_ID grnod_ID'",
                  block.source)
        return
    mass = float(t[2]) if len(t) > 2 else 0.0
    icog = int(float(t[3])) if len(t) > 3 else 1
    jadd = np.array(_floats(cards[1], 3)) if len(cards) > 1 else np.zeros(3)
    if np.any(jadd < 0.0):
        log.error(f"/RBODY/{block.user_id}: added inertia must be >= 0",
                  block.source)
        return
    model.rbodies.append(RigidBody(
        id=block.user_id, kind="RBODY", master_id=int(t[0]),
        grnod_id=int(t[1]), added_mass=mass, jadd=jadd, icog=icog,
        title=title))


def read_rbe2(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RBE2/rbe2_ID`` (M5)::

        card 1:  title
        card 2:  node_ID   grnod_ID

      rigid link: the slave nodes (grnod_ID) move rigidly with the master
      node node_ID. Unlike /RBODY the master is a structural node: it
      keeps its position, its own mass and the forces of the elements
      attached to it feed the link's rigid equation of motion. Only the
      full 6-DOF tie is ported (no per-DOF flags).
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/RBE2/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].ints()
    if len(t) < 2:
        log.error(f"/RBE2/{block.user_id}: card 2 needs 'node_ID grnod_ID'",
                  block.source)
        return
    model.rbodies.append(RigidBody(
        id=block.user_id, kind="RBE2", master_id=t[0], grnod_id=t[1],
        added_mass=0.0, jadd=np.zeros(3), icog=0, title=title))


def read_rbe3(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RBE3/rbe3_ID`` (M5)::

        card 1:  title
        card 2:  node_ID   grnod_ID

      interpolation constraint: the dependent node node_ID follows the
      weighted-average (least-squares rigid fit) motion of the master
      nodes in grnod_ID, and forces applied at the dependent node are
      distributed to the masters without stiffening the model. Uniform
      unit weights (the per-set weights/DOF flags of the full card are
      not ported).
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/RBE3/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].ints()
    if len(t) < 2:
        log.error(f"/RBE3/{block.user_id}: card 2 needs 'node_ID grnod_ID'",
                  block.source)
        return
    model.rbe3.append(Rbe3(
        id=block.user_id, ref_id=t[0], grnod_id=t[1], title=title))


def read_sect(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/SECT/sect_ID`` (M5)::

        card 1:  title
        card 2:  grnod_ID   [node_ID_ref]

      section-force output: grnod_ID must contain ALL nodes of one side
      of the cut (a /GRNOD/PART of the side's parts is the natural
      input); the reported force/moment is what the other side transmits
      through the cut (see the Section entity docstring for the side-sum
      identity). node_ID_ref: moment reference node (its current
      position); 0/absent = the fixed initial centroid of the side set.
      Output via /TH/SECT.
    """
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/SECT/{block.user_id}: missing data card", block.source)
        return
    t = cards[0].ints()
    model.sections.append(Section(
        id=block.user_id, grnod_id=t[0],
        node_id_ref=t[1] if len(t) > 1 else 0, title=title))


# ============================================================================
# Rigid wall, contact
# ============================================================================

def read_rwall(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/RWALL/PLANE|SPHER|CYL/rwall_ID`` (geometries + motion since M5)::

        card 1:  title
        card 2:  grnod_ID   Slide   fric   Dist   node_ID
        PLANE:
        card 3:  XM    YM    ZM        (point M on the plane)
        card 4:  XM1   YM1   ZM1       (point M1: normal = M->M1)
        SPHER:
        card 3:  XM    YM    ZM        (center)
        card 4:  R                     (radius; nodes live OUTSIDE)
        CYL:
        card 3:  XM    YM    ZM        (point on the axis)
        card 4:  XM1   YM1   ZM1       (point M1: axis = M->M1)
        card 5:  R                     (radius; nodes live outside)

      Slide: 0 = frictionless sliding, 1 = tied, 2 = sliding + friction.
      grnod_ID = 0 → all (real-mass) nodes are wall candidates.
      Dist = search distance (0 → all candidates tracked every cycle).
      node_ID > 0 → MOVING wall tied to that node (M5): the wall
      translates with the node and the contact impulses react on it —
      give the node its inertia with /ADMAS + /INIVEL for a free wall,
      or drive it with /IMPVEL for an imposed-motion wall.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "PLANE"
    if kind not in ("PLANE", "SPHER", "CYL"):
        log.warning(f"/RWALL/{kind} not ported (PLANE, SPHER, CYL "
                    f"supported)", block.source)
        return
    title, cards = _title_and_data(block)
    ncards = {"PLANE": 3, "SPHER": 3, "CYL": 4}[kind]
    if len(cards) < ncards:
        log.error(f"/RWALL/{kind}/{block.user_id}: needs {ncards} data "
                  f"cards", block.source)
        return
    t = cards[0].tokens()
    grnod = int(t[0]) if t else 0
    slide = int(t[1]) if len(t) > 1 else 0
    fric = float(t[2]) if len(t) > 2 else 0.0
    dist = float(t[3]) if len(t) > 3 else 0.0
    node_id = int(float(t[4])) if len(t) > 4 else 0
    m = np.array(_floats(cards[1], 3))
    normal = np.array([0.0, 0.0, 1.0])
    radius = 0.0
    if kind in ("PLANE", "CYL"):
        m1 = np.array(_floats(cards[2], 3))
        n = m1 - m
        nn = np.linalg.norm(n)
        if nn < 1e-20:
            log.error(f"/RWALL/{block.user_id}: M and M1 coincide "
                      f"(zero normal/axis)", block.source)
            return
        normal = n / nn
    if kind in ("SPHER", "CYL"):
        rcard = cards[2] if kind == "SPHER" else cards[3]
        radius = rcard.floats()[0]
        if radius <= 0.0:
            log.error(f"/RWALL/{kind}/{block.user_id}: radius must be > 0",
                      block.source)
            return
    model.rwalls.append(RigidWall(
        id=block.user_id, point=m, normal=normal, slide=slide, fric=fric,
        grnod_id=grnod or None, dist=dist, title=title, geom=kind,
        radius=radius, node_id=node_id))


def read_inter(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/INTER/TYPE7|TYPE2|TYPE11/inter_ID`` — the three contact types.

    Fortran: ``starter/source/interfaces/int07|02|11/hm_read_inter*.F``.
    The port keeps a compact card layout (a strict subset of the Radioss
    fields, in the Radioss order where they exist):

    ``/INTER/TYPE7`` (penalty node-to-surface)::

        card 1:  title
        card 2:  grnod_ID  surf_ID  Istf  Igap  [sens_ID]  [Ifric]  [Ifiltr]
        card 3:  Stfac     Fric     Gapmin  Gapmax  [Xfreq]   (all optional)
        card 4:  C1  C2  C3  C4  C5  C6         (only read when Ifric > 0)

      sens_ID (M6): the interface stays inactive (no forces, no time-step
      claim) until /SENSOR sens_ID fires.

      grnod_ID = 0 → *self-impact*: the secondary nodes default to the
      nodes of the main surface itself (Radioss single-surface input).
      Istf 0..5 and Igap 0/1 as documented on
      :class:`pyradioss.model.entities.Interface`. For Istf=1, Stfac is
      the constant penalty stiffness itself (force/length); otherwise it
      scales the element-based stiffness (default 1.0).

      Ifric/Ifiltr/Xfreq/C1..C6 (M15 — the friction MODELS of
      ``hm_read_inter_type07.F``, mirrored field for field against the
      SOURCE): Ifric = MFROT 1..4 selects the mu(p, v) law (C1..C5 read
      for Ifric > 0, C6 for Ifric > 1 — the original's optional card 8);
      Ifiltr = IFQ 1/2/3 turns the tangential-force first-order filter
      on, with the coefficient derived from Xfreq exactly as the reader
      does (1: Xfreq itself, must be in [0, 1]; 2: 2*pi/Xfreq, Xfreq a
      period in cycles; 3: 2*pi*Xfreq, Xfreq a cutoff frequency —
      per-cycle alpha = XFILTR*dt). Xfreq = 0 switches the filter OFF
      whatever Ifiltr says — exactly the reference (``IF (ALPHA==0.)
      IFQ = 0`` in hm_read_inter_type07.F). IFQ >= 10 (the MODFR = 2
      incremental stiffness formulation) is refused loudly — deferred
      (see contact/friction.py; the implicit solver's return mapping IS
      that formulation). Laws and filter live in contact/friction.py.

      TYPE7 also accepts the REAL fixed-format layout (cfg
      ``inter_type7.cfg`` radioss2020+ / ``hm_read_inter_type07.F``),
      detected by its card count — 6+ data cards where the compact
      layout above has at most 3::

        card 2:  grnod_ID surf_ID Istf Ithe Igap __ Ibag Idel Icurv Iadm
        card 3:  Fscale_gap  Gap_max  Fpenmax  __  Itied
        card 4:  Stmin  Stmax  %mesh_size  dtmin  Irem_gap  Irem_i2
        [Icurv 1/2 only: node_ID1 node_ID2]
        card 5:  Stfac  Fric  Gapmin  Tstart  Tstop
        card 6:  IBC  __  Inacti  VisS  VisF  Bumult
        card 7:  Ifric Ifiltr Xfreq Iform sens_ID fct_IDF AscaleF fric_ID
        [Ifric > 0 only: C1..C5]  [Ifric > 1 only: C6]

      Real fields with no port equivalent (Ithe/Ibag/Idel/Icurv/Iadm,
      Fscale_gap/Fpenmax/Itied, Stmin/Stmax/%mesh_size/dtmin/Irem_*,
      Tstart/Tstop, IBC/Inacti/VisS/VisF/Bumult, fct_IDF/AscaleF/
      fric_ID) are accepted and, when set to a non-default value,
      reported in ONE warning. Iform = 2 (the incremental tangential
      formulation) is an ERROR when friction is actually active
      (Fric != 0 or Ifric > 0 — same machinery as the IFQ >= 10
      refusal); with no friction it is inert and only warned about.

    ``/INTER/TYPE2`` (tied, kinematic)::

        card 1:  title
        card 2:  grnod_ID  surf_ID  dsearch

      Every secondary node within ``dsearch`` of the main surface
      (0 → auto: twice the main segment size) is glued to its closest
      segment for the whole run. Not-found nodes are left free (warning).

    ``/INTER/TYPE11`` (penalty edge-to-edge)::

        card 1:  title
        card 2:  line_ID1  line_ID2  Istf  Igap  [sens_ID]  [Ifric]  [Ifiltr]
        card 3:  Stfac     Fric      Gapmin  Gapmax  [Xfreq]
        card 4:  C1  C2  C3  C4  C5  C6         (only read when Ifric > 0)

      The TYPE11 friction-model fields are a documented port EXTENSION:
      the ORIGINAL TYPE11 card has none and its engine never evaluates
      MFROT (i11mainf.F forces MFROT = 0 — checked; see
      contact/friction.py for the edge-pair pressure definition).

    Options NOT ported (accepted Radioss fields ignored elsewhere in the
    line): Inacti, Tstart/Tstop, thermal contact, the IFQ >= 10 / MODFR=2
    incremental tangential formulation (refused), Igap 2/3 mesh-size gap
    scaling.
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else ""
    if kind not in ("TYPE7", "TYPE2", "TYPE11"):
        log.warning(f"/INTER/{kind} not ported (TYPE2, TYPE7, TYPE11 "
                    f"supported)", block.source)
        return
    title, cards = _title_and_data(block)
    if not cards:
        log.error(f"/INTER/{kind}/{block.user_id}: missing data card",
                  block.source)
        return
    toks = cards[0].tokens()

    if kind == "TYPE2":
        f = _floats(cards[0], 3)
        model.interfaces.append(Interface(
            id=block.user_id, type=2, grnod_id=int(toks[0]),
            surf_id=int(toks[1]), dsearch=f[2], title=title))
        return

    if kind == "TYPE7" and len(cards) >= 6:
        # ==== the REAL fixed-format TYPE7 layout (see docstring) ===========
        ign: List[str] = []            # non-default fields the port ignores
        f0 = _fixed_vals(cards[0], [10] * 10)
        id1, id2 = _ival(f0[0]), _ival(f0[1])
        istf, igap = _ival(f0[2]), _ival(f0[4])
        for name, s in (("Ithe", f0[3]), ("Ibag", f0[6]), ("Idel", f0[7]),
                        ("Iadm", f0[9])):
            if _ival(s) != 0:
                ign.append(f"{name}={s}")
        icurv = _ival(f0[8])
        f1 = _fixed_vals(cards[1], [20, 20, 20, 20, 10])
        gap_max = _fval(f1[1])
        if f1[0] and _to_float(f1[0]) not in (0.0, 1.0):  # 0/1 = default
            ign.append(f"Fscale_gap={f1[0]}")
        for name, s in (("Fpenmax", f1[2]), ("Itied", f1[4])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        f2 = _fixed_vals(cards[2], [20, 20, 20, 20, 10, 10])
        for name, s in (("Stmin", f2[0]), ("Stmax", f2[1]),
                        ("%mesh_size", f2[2]), ("dtmin", f2[3]),
                        ("Irem_gap", f2[4]), ("Irem_i2", f2[5])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        icard = 3
        if icurv in (1, 2):            # the optional curvature node card
            ign.append(f"Icurv={icurv} (node card skipped)")
            icard += 1
        elif icurv != 0:
            ign.append(f"Icurv={icurv}")
        if len(cards) < icard + 3:
            log.error(f"/INTER/TYPE7/{block.user_id}: real-format block "
                      f"needs 6 data cards (got {len(cards)})", block.source)
            return
        f3 = _fixed_vals(cards[icard], [20] * 5)
        stfac, fric, gap = _fval(f3[0]), _fval(f3[1]), _fval(f3[2])
        for name, s in (("Tstart", f3[3]), ("Tstop", f3[4])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        f4 = _fixed_vals(cards[icard + 1], [7, 1, 1, 1, 20, 10, 20, 20, 20])
        if _ival(f4[1]) or _ival(f4[2]) or _ival(f4[3]):
            ign.append(f"IBC={f4[1] or '0'}{f4[2] or '0'}{f4[3] or '0'}")
        for name, s in (("Inacti", f4[5]), ("VisS", f4[6]),
                        ("VisF", f4[7]), ("Bumult", f4[8])):
            if s and _to_float(s) != 0.0:
                ign.append(f"{name}={s}")
        f5 = _fixed_vals(cards[icard + 2],
                         [10, 10, 20, 10, 10, 10, 20, 10])
        mfrot, ifq = _ival(f5[0]), _ival(f5[1])
        xfreq, iform, sens = _fval(f5[2]), _ival(f5[3]), _ival(f5[4])
        for name, s in (("fct_IDF", f5[5]), ("fric_ID", f5[7])):
            if _ival(s) != 0:
                ign.append(f"{name}={s}")
        if f5[6] and _to_float(f5[6]) not in (0.0, 1.0):
            ign.append(f"AscaleF={f5[6]}")
        # Iform = MODFR: 2 selects the incremental (stiffness) tangential
        # formulation (upstream turns it into IFQ >= 10) — the same
        # not-ported path the compact dialect refuses. Without friction
        # it changes nothing and is only reported.
        if iform == 2:
            if fric != 0.0 or mfrot > 0:
                log.error(f"/INTER/TYPE7/{block.user_id}: Iform=2 (the "
                          f"incremental stiffness tangential formulation) "
                          f"is not ported — use Iform 0/1", block.source)
            else:
                ign.append("Iform=2 (no friction defined — inert)")
        # C1..C5 (Ifric > 0) and C6 (Ifric > 1) cards
        fric_c = (0.0,) * 6
        icard += 3
        if mfrot > 0:
            cc = [0.0] * 6
            if len(cards) > icard:
                cc[:5] = _floats(cards[icard], 5)
                icard += 1
                if mfrot > 1 and len(cards) > icard:
                    cc[5] = _floats(cards[icard], 1)[0]
                    icard += 1
            else:
                log.warning(f"/INTER/TYPE7/{block.user_id}: Ifric={mfrot} "
                            f"without a C1..C5 card — all coefficients 0",
                            block.source)
            fric_c = tuple(cc)
        if ign:
            log.warning(f"/INTER/TYPE7/{block.user_id}: real-format fields "
                        f"not ported — ignored: {'; '.join(ign)}",
                        block.source)
    else:
        # ==== the port's compact layout =====================================
        t = cards[0].ints()
        id1 = t[0]
        id2 = t[1]
        istf = t[2] if len(t) > 2 else 0
        igap = t[3] if len(t) > 3 else 0
        sens = t[4] if len(t) > 4 else 0
        mfrot = t[5] if len(t) > 5 else 0        # Ifric (M15)
        ifq = t[6] if len(t) > 6 else 0          # Ifiltr (M15)
        stfac, fric, gap, gap_max, xfreq = (1.0, 0.0, 0.0, 0.0, 0.0)
        if len(cards) > 1:
            stfac, fric, gap, gap_max, xfreq = _floats(
                cards[1], 5, defaults=[1.0, 0.0, 0.0, 0.0, 0.0])
        # ---- optional C1..C6 card (the original's card 8, Ifric > 0) ------
        fric_c = (0.0,) * 6
        if mfrot in (1, 2, 3, 4):
            if len(cards) > 2:
                cc = _floats(cards[2], 6, defaults=[0.0] * 6)
                # C6 is only read for Ifric > 1 (hm_read_inter_type07.F)
                fric_c = tuple(cc[:5]) + ((cc[5],) if mfrot > 1 else (0.0,))
            else:
                log.warning(f"/INTER/{kind}/{block.user_id}: Ifric={mfrot} "
                            f"without a C1..C6 card — all coefficients 0",
                            block.source)

    # ==== shared validation (both dialects) ================================
    if istf not in (0, 1, 2, 3, 4, 5):
        log.error(f"/INTER/{kind}/{block.user_id}: Istf={istf} (0..5)",
                  block.source)
    if igap not in (0, 1):
        log.error(f"/INTER/{kind}/{block.user_id}: Igap={igap} not ported "
                  f"(0 constant, 1 variable)", block.source)
    if mfrot not in (0, 1, 2, 3, 4):
        log.error(f"/INTER/{kind}/{block.user_id}: Ifric={mfrot} (0..4: "
                  f"Coulomb / generalized viscous / Darmstadt / Renard / "
                  f"exponential decay)", block.source)
        mfrot = 0
    if ifq >= 10:
        # MODFR = 2 / the incremental (stiffness) tangential formulation —
        # a different explicit force path, deferred loudly (M15; the
        # implicit solver's return mapping IS that formulation)
        log.error(f"/INTER/{kind}/{block.user_id}: Ifiltr={ifq} (the "
                  f"IFQ >= 10 incremental stiffness formulation) is not "
                  f"ported — use Ifiltr 0..3", block.source)
        ifq = 0
    if ifq not in (0, 1, 2, 3):
        log.error(f"/INTER/{kind}/{block.user_id}: Ifiltr={ifq} (0..3)",
                  block.source)
        ifq = 0
    if stfac == 0.0 and istf != 1:
        stfac = 1.0                # Radioss: Stfac = 0 -> default scale 1.0
    if istf == 1 and stfac <= 0.0:
        log.error(f"/INTER/{kind}/{block.user_id}: Istf=1 needs a "
                  f"positive Stfac (it IS the stiffness)", block.source)
    # ---- the XFILTR mapping of hm_read_inter_type07.F (M15, checked) ------
    # Xfreq (ALPHA) = 0 switches the filter OFF whatever Ifiltr says —
    # exactly the reference: IF (ALPHA==0.) IFQ = 0. Then IFQ=1: Xfreq IS
    # the coefficient; IFQ=2: 2*pi/Xfreq (a period in cycles); IFQ=3:
    # 2*pi*Xfreq (a cutoff frequency — alpha = XFILTR*dt per cycle). The
    # original's MSGID 554 errors are mirrored.
    if xfreq == 0.0:
        ifq = 0
    xfiltr = 0.0
    if ifq > 0:
        if ifq == 1:
            xfiltr = xfreq
        elif ifq == 2:
            xfiltr = (2.0 * np.pi / xfreq) if xfreq > 0.0 else -1.0
        elif ifq == 3:
            xfiltr = 2.0 * np.pi * xfreq
        if xfiltr < 0.0 or (xfiltr > 1.0 and ifq <= 2):
            log.error(f"/INTER/{kind}/{block.user_id}: friction filtering "
                      f"factor out of range (Xfreq={xfreq:g} -> "
                      f"XFILTR={xfiltr:g}, must be in [0,1] for "
                      f"Ifiltr 1/2)", block.source)
            ifq, xfiltr = 0, 0.0
    if kind == "TYPE7":
        model.interfaces.append(Interface(
            id=block.user_id, type=7, grnod_id=id1, surf_id=id2,
            istf=istf, igap=igap, stfac=stfac, fric=fric, gap=gap,
            gap_max=gap_max, sens_id=sens, mfrot=mfrot, ifq=ifq,
            xfiltr=xfiltr, fric_c=fric_c, title=title))
    else:                          # TYPE11
        if mfrot > 0 or ifq > 0:
            # the original TYPE11 has no friction models at all (checked:
            # i11mainf.F forces MFROT = 0) — the port extension is
            # announced so nobody mistakes it for Radioss behaviour
            log.info(f"     /INTER/TYPE11/{block.user_id}: FRICTION "
                     f"MODEL Ifric={mfrot} Ifiltr={ifq} — A PORT "
                     f"EXTENSION (the original TYPE11 never evaluates "
                     f"MFROT; see contact/friction.py)")
        model.interfaces.append(Interface(
            id=block.user_id, type=11, line_id1=id1, line_id2=id2,
            istf=istf, igap=igap, stfac=stfac, fric=fric, gap=gap,
            gap_max=gap_max, sens_id=sens, mfrot=mfrot, ifq=ifq,
            xfiltr=xfiltr, fric_c=fric_c, title=title))


def read_line(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/LINE/SURF/line_ID`` or ``/LINE/SEG/line_ID`` — edge sets for
    /INTER/TYPE11 (Fortran: hm_read_lines.F → IGRSLIN)::

        /LINE/SURF: card 1 = title, card 2+ = surf_IDs (any number/card)
                    → every unique edge of those surfaces' segments
        /LINE/SEG:  card 1 = title, card 2+ = node_ID1 node_ID2 per card
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "SURF"
    if kind not in ("SURF", "SEG"):
        log.warning(f"/LINE/{kind} not ported (SURF, SEG supported)",
                    block.source)
        return
    title, cards = _title_and_data(block)
    line = model.lines.setdefault(block.user_id,
                                  Line(id=block.user_id, title=title))
    if kind == "SURF":
        for card in cards:
            line.surf_ids.extend(card.ints())
    else:
        for card in cards:
            t = card.ints()
            if len(t) < 2:
                log.error(f"/LINE/SEG/{block.user_id}: a segment needs 2 "
                          f"node ids", card.source)
                continue
            line.seg_nodes.append(t[:2])


# ============================================================================
# Time-history requests
# ============================================================================

def read_th(block: KeywordBlock, model: Model, log: MessageLog) -> None:
    """``/TH/NODE|PART|SECT/th_ID``::

        card 1:  title
        card 2:  variable names (e.g. ``DX DY DZ VX VY VZ``) or ``DEF``
        card 3+: object IDs (any number per card)

      DEF expands to the Radioss default set for the object type.
      SECT (M5) variables: FX FY FZ MX MY MZ (section force/moment).
    """
    kind = block.parts[1].upper() if len(block.parts) > 1 else "NODE"
    if kind not in ("NODE", "PART", "SECT"):
        log.warning(f"/TH/{kind} not ported (NODE, PART, SECT supported)",
                    block.source)
        return
    title, cards = _title_and_data(block)
    if len(cards) < 2:
        log.error(f"/TH/{kind}/{block.user_id}: needs variables + ids cards",
                  block.source)
        return
    variables = [v.upper() for v in cards[0].tokens()]
    if variables == ["DEF"]:
        variables = {"NODE": ["DX", "DY", "DZ", "VX", "VY", "VZ"],
                     "PART": ["IE", "KE"],
                     "SECT": ["FX", "FY", "FZ", "MX", "MY", "MZ"]}[kind]
    ids: List[int] = []
    for c in cards[1:]:
        ids.extend(c.ints())
    model.th_requests.append(THRequest(
        id=block.user_id, kind=kind, ids=ids, variables=variables,
        title=title))


# ============================================================================
# Dispatch table (the Fortran 'select case' of lectur.F)
# ============================================================================

KEYWORD_PARSERS: Dict[str, Callable] = {
    "BEGIN": read_begin,
    "TITLE": read_title,
    "END": read_end,
    "NODE": read_node,
    "BRICK": read_brick,
    "TETRA4": read_tetra4,
    "SHELL": read_shell,
    "SH3N": read_sh3n,
    "TRUSS": read_truss,
    "SPRING": read_spring,
    "BEAM": read_beam,
    "PART": read_part,
    "MAT": read_mat,
    "FAIL": read_fail,
    "EOS": read_eos,
    "PROP": read_prop,
    "FUNCT": read_funct,
    "GRNOD": read_grnod,
    "BOX": read_box,
    "SURF": read_surf,
    "BCS": read_bcs,
    "INIVEL": read_inivel,
    "GRAV": read_grav,
    "CLOAD": read_cload,
    "IMPVEL": read_impvel,
    "IMPDISP": read_impdisp,
    "PLOAD": read_pload,
    "ADMAS": read_admas,
    "DAMP": read_damp,
    "SENSOR": read_sensor,
    "MPC": read_mpc,
    "RBODY": read_rbody,
    "RBE2": read_rbe2,
    "RBE3": read_rbe3,
    "SECT": read_sect,
    "RWALL": read_rwall,
    "INTER": read_inter,
    "LINE": read_line,
    "TH": read_th,
}


def parse_starter_deck(blocks: List[KeywordBlock], model: Model,
                       log: MessageLog) -> None:
    """Dispatch every block to its parser (unknown → warning + skip)."""
    for block in blocks:
        parser = KEYWORD_PARSERS.get(block.key0)
        if parser is None:
            log.warning(f"keyword /{'/'.join(block.parts)} not ported — "
                        f"block skipped", block.source)
            continue
        try:
            parser(block, model, log)
        except (ValueError, IndexError, KeyError) as exc:
            log.error(f"while reading /{'/'.join(block.parts)}: {exc}",
                      block.source)
