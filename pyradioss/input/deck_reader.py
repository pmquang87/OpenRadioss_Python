"""
Deck lexer: split a Radioss ``.rad`` file into keyword blocks.

Fortran origin: the *hm_reader* layer driven from ``starter/source/reader``
(and, historically, ``starter/source/starter/freform.F`` — the "free format"
reader). The physical file format is:

* A line beginning with ``/`` opens a **keyword block**, e.g.::

      /MAT/LAW2/17
      /BRICK/3
      /INTER/TYPE7/1

  The keyword line is a ``/``-separated path. The *last* component is the
  user ID when it parses as an integer (``/MAT/LAW2/17`` → keyword
  ``MAT/LAW2``, id 17). Some keywords carry no ID (``/BEGIN``, ``/END``).

* Every following line until the next ``/`` line (or EOF) is a **data card**
  of that block.

* Lines starting with ``#`` or ``$`` are comments. Radioss decks make heavy
  use of ``#---1----|----2----|…`` ruler comments to mark the 10-column
  fields; they are skipped here like any comment.

* ``#include FILE`` (a *directive*, despite starting with '#') splices
  another file — used by every real-world deck to separate mesh and setup.
  We resolve the path relative to the including file, recursively.

* Data cards are historically **fixed format**: 100-character lines cut in
  ten fields of 10 characters (20-character fields for the "double
  precision" input variant). Since Radioss 2020 the input is keyword-based
  and tolerant; in practice almost all decks are also readable as
  whitespace-separated tokens *except* when text fields (titles) or empty
  fixed columns matter.

  Port decision (documented deviation): each :class:`Card` keeps the **raw
  line**, and offers BOTH access modes:

    - :meth:`Card.tokens`  — whitespace split (used for numeric cards),
    - :meth:`Card.fields`  — 10-column fixed slicing (used when a card mixes
      blanks-as-defaults, e.g. /BCS DOF flags),
    - :meth:`Card.cut`     — arbitrary-width fixed slicing against the
      shared :mod:`card_layouts` table (M37: real packed decks),

  so each keyword parser chooses the appropriate view, exactly like the
  Fortran ``KFORMAT``/free-format dual reading.

* **Fixed-dialect detection (M37).** Real decks declare their input
  version on the second /BEGIN card (``2022  0``); the port's own
  M36-regenerated decks do too (one dialect since M36).  When a version
  >= 90 is declared (the 10/20-character era of the cfg card layouts),
  every block is flagged ``fixed=True``.  The real Starter counts
  whitespace-only lines as **blank cards** (all fields default), and the
  card *indices* of the fixed layouts only line up when they are kept:
  the lexer records where blank lines sat (:attr:`KeywordBlock.blank_slots`)
  and :meth:`KeywordBlock.fixed_cards` reconstructs the true card
  stream.  :attr:`KeywordBlock.cards` stays the historical blank-skipped
  list, so every parser keeps its legacy behaviour until it explicitly
  opts into the fixed view (see starter_keywords).

* ``/PARAMETER`` substitution (M37): real decks reference parameters as
  ``&NAME`` inside any field (``/PARAMETER/GLOBAL/REAL`` blocks define
  them).  The reader substitutes the value into the raw line **in
  place**, preserving the column layout (the value replaces the
  ``&NAME`` token and only consumes following spaces), exactly like the
  reference reader's textual substitution.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

from .card_layouts import LAYOUTS, split_fixed


# ----------------------------------------------------------------------------
# Data holders
# ----------------------------------------------------------------------------

@dataclass
class Card:
    """One data card (line) of a keyword block."""

    raw: str          # the line as read, without trailing newline
    source: str = ""  # "file:lineno" for error messages
    #: cached whitespace tokens
    _tokens: Optional[List[str]] = field(default=None, repr=False)

    def tokens(self) -> List[str]:
        """Whitespace-separated tokens (free-format view)."""
        if self._tokens is None:
            self._tokens = self.raw.split()
        return self._tokens

    def fields(self, width: int = 10, n: int = 10) -> List[str]:
        """Fixed-column view: ``n`` fields of ``width`` chars, stripped.

        This mirrors the classic Radioss fixed format (10 x 10 characters
        per 100-column card). Missing columns come back as ''.
        """
        line = self.raw.rstrip("\n")
        return [line[i * width:(i + 1) * width].strip() for i in range(n)]

    def cut(self, layout_key: Union[str, Sequence[int]]) -> List[str]:
        """Cut the raw line at the column widths of the shared
        :data:`card_layouts.LAYOUTS` entry ``layout_key`` (M37) — the way
        packed real cards with NO whitespace between fields are split."""
        if isinstance(layout_key, (list, tuple)):
            return split_fixed(self.raw, layout_key)
        return split_fixed(self.raw, LAYOUTS[layout_key])

    @property
    def is_blank(self) -> bool:
        """True for a blank card (whitespace-only line, kept for fixed
        decks): every field at its default."""
        return not self.raw.strip()

    # -- typed helpers used by the keyword parsers ---------------------------
    def ints(self) -> List[int]:
        """All tokens parsed as ints (for connectivity cards)."""
        return [_to_int(t) for t in self.tokens()]

    def floats(self) -> List[float]:
        """All tokens parsed as floats (Fortran-style '1.0D3' accepted)."""
        return [_to_float(t) for t in self.tokens()]


def parse_fortran_float(s: str) -> float:
    """Centralized Fortran scientific notation normalizer: handles D/d exponents
    and omitted-E sign notation (e.g., '1.5-3' -> '1.5E-3')."""
    cleaned = s.strip().rstrip(",")
    return float(re.sub(r'(?<=[0-9.])([+-])(?=[0-9])', r'E\1', cleaned.replace('D', 'E').replace('d', 'e').strip()))


def _to_float(tok: str) -> float:
    """Parse a Fortran-flavoured real: allows D exponents ('1.5D-3') and omitted-E notation."""
    return parse_fortran_float(tok)


def _to_int(tok: str) -> int:
    """Parse an integer: allows float strings by truncating them (e.g., '500.0' -> 500)."""
    s = tok.strip().rstrip(",")
    try:
        return int(s)
    except ValueError:
        return int(parse_fortran_float(s))


@dataclass
class KeywordBlock:
    """One ``/KEYWORD`` block: header path + user id + data cards."""

    keyword: str            # normalized path without ids, e.g. "MAT/LAW2"
    parts: List[str]        # all header components, e.g. ["MAT","LAW2","17"]
    user_id: Optional[int]  # trailing integer component, if any
    cards: List[Card]
    source: str = ""        # "file:lineno" of the header line
    #: positions (indices into ``cards``) where the physical file had a
    #: whitespace-only line — a *blank card* to the real fixed-format
    #: reader (every field at its default), invisible to the token
    #: parsers above.  Recorded so format-exact consumers (the M37
    #: cfg-driven /MAT reader) can reconstruct the true card stream;
    #: one entry per blank line, duplicates = consecutive blanks.
    blank_slots: List[int] = field(default_factory=list)
    #: True when the deck's /BEGIN declared a real input version >= 90
    #: (set by read_deck's post-pass): the block is in the REAL fixed
    #: 10/20-character dialect and parsers may use :meth:`fixed_cards` +
    #: the card_layouts column widths.
    fixed: bool = False
    #: the optional LOCAL UNIT id of the block (M37): the general Radioss
    #: header syntax is ``/KEY/.../user_ID/unit_ID`` — when the header
    #: ends in TWO integers the first is the user id and the second
    #: references a /UNIT block whose unit system the block's quantities
    #: are written in (converted to the /BEGIN work units at resolve
    #: time, see input/units.py). None = no local unit (work units).
    unit_id: Optional[int] = None

    @property
    def key0(self) -> str:
        """First component ('MAT' for /MAT/LAW2/17) — used for dispatch."""
        return self.parts[0].upper()

    def fixed_cards(self) -> List[Card]:
        """The TRUE fixed-format card stream: ``cards`` with the recorded
        blank cards re-inserted at their :attr:`blank_slots` positions —
        what the real fixed reader sees (a blank card = every field at
        its default), so per-layout card indices line up exactly."""
        if not self.blank_slots:
            return list(self.cards)
        out: List[Card] = []
        slots, si = self.blank_slots, 0          # ascending by construction
        for i, c in enumerate(self.cards):
            while si < len(slots) and slots[si] == i:
                out.append(Card(raw="", source=self.source))
                si += 1
            out.append(c)
        while si < len(slots):
            out.append(Card(raw="", source=self.source))
            si += 1
        return out


# ----------------------------------------------------------------------------
# Lexer
# ----------------------------------------------------------------------------

#: Header line missing its leading '/' (M40, see read_deck): matched
#: against the RAW line so column 1 is enforced like the real reader's
#: option scan.  Deliberately NARROW — only heads with corpus evidence
#: of real-Starter recovery (/FAIL), and only the KEYWORD/KEY2/id shape,
#: so titles or free-format data can never match.
_SLASHLESS_HEADER = re.compile(r"^(?:FAIL)/[A-Z0-9_]+(?:/\d+)+\s*$")


def _is_comment(line: str) -> bool:
    s = line.lstrip()
    # '#include' is a directive, not a comment — tested before calling this.
    return s.startswith("#") or s.startswith("$")


def read_deck(path: str, _depth: int = 0) -> List[KeywordBlock]:
    """Read a deck file (recursively following ``#include``) into blocks.

    Parameters
    ----------
    path : str
        The ``*.rad`` file to read.

    Returns
    -------
    list of KeywordBlock, in file order (include files are spliced in place,
    exactly as the Fortran reader concatenates them).
    """
    if _depth > 20:
        raise RecursionError(f"#include nesting deeper than 20 at {path}")

    blocks: List[KeywordBlock] = []
    current: Optional[KeywordBlock] = None
    base_dir = os.path.dirname(os.path.abspath(path))

    with open(path, "r", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            stripped = line.strip()
            here = f"{os.path.basename(path)}:{lineno}"

            if not stripped:
                # blank line: not a card for the token parsers, but the
                # real fixed reader counts it as a BLANK CARD — remember
                # where it sat (see KeywordBlock.blank_slots)
                if current is not None:
                    current.blank_slots.append(len(current.cards))
                continue

            # -- #include directive (before generic comment handling) -------
            low = stripped.lower()
            if low.startswith("#include"):
                inc = stripped[len("#include"):].strip().strip('"').strip("'")
                inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
                # Close the current block: an include splices *blocks*, it
                # never continues the cards of an open block.
                if current is not None:
                    blocks.append(current)
                    current = None
                try:
                    blocks.extend(read_deck(inc_path, _depth + 1))
                except FileNotFoundError:
                    # Record a synthetic error block so the caller sees the
                    # failure (the Fortran starter also flags missing includes
                    # and continues).  We append a block with keyword
                    # "__INCLUDE_ERROR__" so read_all_blocks can log it.
                    err_block = KeywordBlock(
                        keyword="__INCLUDE_ERROR__",
                        parts=["__INCLUDE_ERROR__"],
                        user_id=0,
                        cards=[],
                        source=f"{path}:{lineno}",
                        fixed=False,
                    )
                    err_block._include_path = inc_path
                    blocks.append(err_block)
                continue

            if _is_comment(stripped):
                continue

            # -- keyword header MISSING its leading slash (M40) --------------
            # Observed in the official corpus: the RD-V-0700 TETRA deck's
            # MAT_PROP.inc writes ``FAIL/JOHNSON/1`` at column 1 (no '/'),
            # and the REAL Starter still reads the /FAIL block — its
            # hm_reader recovers the option (the bundled reference listing
            # TETRA_0000.out.fortran_ref prints the JOHNSON COOK DAMAGE
            # PARAMETERS).  The port lexer used to swallow the whole block
            # as stray data cards of the previous /MAT — silently running
            # WITHOUT the failure model (no c23 element ever deleted).
            # Recover the same narrow way: only for the option heads seen
            # in the wild (FAIL), only at column 1, and only when the line
            # has the strict KEYWORD/KEY2/id shape of a header.
            if _SLASHLESS_HEADER.match(line):
                stripped = "/" + stripped

            # -- keyword header line ----------------------------------------
            if stripped.startswith("/"):
                if current is not None:
                    blocks.append(current)
                parts = [p for p in stripped[1:].split("/") if p != ""]
                # Trailing integer component = user ID (e.g. /BRICK/3).
                # TWO trailing integers = user ID + local UNIT id
                # (``/MAT/PLAS_JOHNS/1/1``, the general Radioss
                # ``/KEY/.../user_ID/unit_ID`` header syntax — M37).
                user_id: Optional[int] = None
                unit_id: Optional[int] = None
                kw_parts = parts
                if len(parts) > 1:
                    try:
                        user_id = _to_int(parts[-1])
                        kw_parts = parts[:-1]
                    except ValueError:
                        user_id = None
                if user_id is not None and len(parts) > 2:
                    try:
                        first = _to_int(parts[-2])
                    except ValueError:
                        first = None
                    if first is not None:
                        user_id, unit_id = first, user_id
                        kw_parts = parts[:-2]
                current = KeywordBlock(
                    keyword="/".join(p.upper() for p in kw_parts),
                    parts=parts,
                    user_id=user_id,
                    unit_id=unit_id,
                    cards=[],
                    source=here,
                )
                continue

            # -- data card ----------------------------------------------------
            if current is None:
                # Data before any keyword: the Fortran reader ignores the
                # deck banner lines; we do the same silently.
                continue
            current.cards.append(Card(raw=line, source=here))

    if current is not None:
        blocks.append(current)

    if _depth == 0:
        _finalize_deck(blocks)
    return blocks


# ----------------------------------------------------------------------------
# Depth-0 post-passes: fixed-dialect detection + /PARAMETER substitution
# ----------------------------------------------------------------------------

def input_version(blocks: List[KeywordBlock]) -> int:
    """The input version declared on the /BEGIN block's second card
    (``Invers Irun``), 0 when absent — legacy port decks carry only the
    run-name card."""
    for b in blocks:
        if b.key0 != "BEGIN":
            continue
        if len(b.cards) >= 2:
            toks = b.cards[1].tokens()
            if toks:
                try:
                    return int(float(toks[0]))
                except ValueError:
                    return 0
        return 0
    return 0


_PARAM_REF = re.compile(r"&[A-Za-z0-9_]+")


def _finalize_deck(blocks: List[KeywordBlock]) -> None:
    """Post-passes over the complete (include-spliced) block list."""
    # 1. fixed-dialect flag: /BEGIN declares an input version >= 90 (the
    #    10/20-character column era of the cfg layouts; older 8-column
    #    dialects keep the token fallback).
    if input_version(blocks) >= 90:
        for b in blocks:
            b.fixed = True

    # 2. /PARAMETER substitution: replace &NAME references in card text
    #    (the reference reader substitutes textually before parsing).
    params: Dict[str, str] = {}
    for b in blocks:
        if b.key0 != "PARAMETER" or not b.cards:
            continue
        # /PARAMETER/<scope>/<REAL|INTEGER|TEXT>/<id>:
        # card 1 = title, card 2 = "%-10s" name + value field
        # (parameter_float.cfg: CARD("%-10s%20lg", NAME, VALUE)).
        # TEXT parameters put the value on a FOLLOW-UP card — not
        # substitutable numerically; skipped (the parser then reports
        # any &NAME reference to one, the honest failure).
        subtype = b.parts[2].upper() if len(b.parts) > 2 else ""
        if subtype not in ("REAL", "INTEGER", "INT"):
            continue
        defcard = b.cards[1] if len(b.cards) >= 2 else b.cards[0]
        name = defcard.raw[:10].strip()
        rest = defcard.raw[10:].split()
        if name and rest:
            params[name.upper()] = rest[0]
    if not params:
        return
    for b in blocks:
        if b.key0 == "PARAMETER":
            continue
        for c in b.cards:
            if "&" not in c.raw:
                continue
            c.raw = _substitute_params(c.raw, params)
            c._tokens = None                       # invalidate token cache


def _substitute_params(raw: str, params: Dict[str, str]) -> str:
    """Replace each ``&NAME`` with its value IN PLACE, preserving the
    column layout: the value starts where the reference started and may
    only consume the spaces that follow it (never a neighbouring field's
    text). Unknown names are left untouched — the parser then reports
    the unresolved token, which is the honest failure."""
    out = raw
    # right-to-left so earlier match positions stay valid after surgery
    for m in reversed(list(_PARAM_REF.finditer(raw))):
        val = params.get(m.group(0)[1:].upper())
        if val is None:
            continue
        a, b = m.span()
        end = b
        while end < len(out) and out[end] == " ":
            end += 1
        avail = end - a
        if len(val) <= b - a:                  # fits in the token itself
            out = out[:a] + val + " " * (b - a - len(val)) + out[b:]
        elif len(val) <= avail:                # consume following spaces
            out = out[:a] + val + out[a + len(val):]
        # else: cannot fit without shifting columns — leave unresolved
    return out
