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

  so each keyword parser chooses the appropriate view, exactly like the
  Fortran ``KFORMAT``/free-format dual reading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


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

    # -- typed helpers used by the keyword parsers ---------------------------
    def ints(self) -> List[int]:
        """All tokens parsed as ints (for connectivity cards)."""
        return [int(t) for t in self.tokens()]

    def floats(self) -> List[float]:
        """All tokens parsed as floats (Fortran-style '1.0D3' accepted)."""
        return [_to_float(t) for t in self.tokens()]


def _to_float(tok: str) -> float:
    """Parse a Fortran-flavoured real: allows D exponents ('1.5D-3')."""
    try:
        return float(tok)
    except ValueError:
        return float(tok.replace("D", "E").replace("d", "e"))


@dataclass
class KeywordBlock:
    """One ``/KEYWORD`` block: header path + user id + data cards."""

    keyword: str            # normalized path without ids, e.g. "MAT/LAW2"
    parts: List[str]        # all header components, e.g. ["MAT","LAW2","17"]
    user_id: Optional[int]  # trailing integer component, if any
    cards: List[Card]
    source: str = ""        # "file:lineno" of the header line

    @property
    def key0(self) -> str:
        """First component ('MAT' for /MAT/LAW2/17) — used for dispatch."""
        return self.parts[0].upper()


# ----------------------------------------------------------------------------
# Lexer
# ----------------------------------------------------------------------------

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
                continue  # blank line

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
                blocks.extend(read_deck(inc_path, _depth + 1))
                continue

            if _is_comment(stripped):
                continue

            # -- keyword header line ----------------------------------------
            if stripped.startswith("/"):
                if current is not None:
                    blocks.append(current)
                parts = [p for p in stripped[1:].split("/") if p != ""]
                # Trailing integer component = user ID (e.g. /BRICK/3).
                user_id: Optional[int] = None
                kw_parts = parts
                if len(parts) > 1:
                    try:
                        user_id = int(parts[-1])
                        kw_parts = parts[:-1]
                    except ValueError:
                        user_id = None
                current = KeywordBlock(
                    keyword="/".join(p.upper() for p in kw_parts),
                    parts=parts,
                    user_id=user_id,
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
    return blocks
