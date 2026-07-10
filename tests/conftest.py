"""Shared pytest fixtures: tiny deck builders used by several tests."""

import os
import textwrap

import pytest


@pytest.fixture
def make_deck(tmp_path):
    """Write a starter+engine deck pair into a temp dir; returns paths."""

    def _make(run_name: str, starter_text: str, engine_text: str):
        s = tmp_path / f"{run_name}_0000.rad"
        e = tmp_path / f"{run_name}_0001.rad"
        s.write_text(textwrap.dedent(starter_text))
        e.write_text(textwrap.dedent(engine_text))
        return str(s), str(e)

    return _make
