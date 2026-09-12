"""
Message / error reporting in the style of the Radioss listings.

Fortran origin: the ANMESSAGE/MESSAGE machinery
(``starter/source/output/message/`` and ``engine/source/output/message/``)
which prints numbered ``** ERROR`` / ``** WARNING`` blocks both to the
terminal and to the ``*.out`` listing, and counts them so the Starter can
refuse to write a restart file when errors were found.

We reproduce the observable behaviour (numbered, counted, dual-destination
messages) with a small collector class instead of the Fortran global state.
"""

from __future__ import annotations

from typing import List, Optional, TextIO


class StarterError(Exception):
    """Fatal model error (deck cannot be run). The Starter raises this after
    printing all collected errors, mirroring the original's behaviour of
    scanning the whole deck before aborting so the user sees *all* problems
    at once, not just the first one."""


class MessageLog:
    """Collects warnings/errors and mirrors them to a listing file."""

    def __init__(self):
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self._listing: Optional[TextIO] = None

    @property
    def messages(self) -> List[str]:
        return self.errors + self.warnings

    def attach_listing(self, fh: TextIO) -> None:
        """Duplicate every future message into an open ``*.out`` file."""
        self._listing = fh

    def _emit(self, text: str) -> None:
        print(text)
        if self._listing is not None:
            self._listing.write(text + "\n")

    def info(self, text: str) -> None:
        self._emit(text)

    def warning(self, text: str, where: str = "") -> None:
        """A non-fatal problem (the run continues), e.g. an unknown keyword
        that is skipped. Format mimics the Radioss '** WARNING' blocks."""
        n = len(self.warnings) + 1
        msg = f" ** WARNING {n:5d} {('IN ' + where) if where else ''}\n    {text}"
        self.warnings.append(msg)
        self._emit(msg)

    def error(self, text: str, where: str = "") -> None:
        """A fatal model error. Collected (not raised) so that deck reading
        can continue and report every problem; call :meth:`check` at the end
        of the Starter to abort if any error was recorded."""
        n = len(self.errors) + 1
        msg = f" ** ERROR {n:5d} {('IN ' + where) if where else ''}\n    {text}"
        self.errors.append(msg)
        self._emit(msg)

    def check(self) -> None:
        """Raise StarterError if any error message was collected."""
        if self.errors:
            raise StarterError(
                f"{len(self.errors)} error(s) found while processing the deck "
                f"(see messages above); restart file not written."
            )

    @property
    def has_errors(self) -> bool:
        """True if any error was logged."""
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        """True if any warning was logged."""
        return bool(self.warnings)

    def summary(self) -> str:
        """The classic end-of-listing tally."""
        return (
            f"\n     {len(self.errors):10d} ERROR(S)\n"
            f"     {len(self.warnings):10d} WARNING(S)\n"
        )
