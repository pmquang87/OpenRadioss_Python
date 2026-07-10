"""
The Starter main program.

Fortran origin: ``starter/source/starter/starter0.F`` (program shell) and
``lectur.F`` (the giant reading/initialization driver). Sequence and the
produced files match the original:

    RunName_0000.rad  --Starter-->  RunName_0000.out  (listing)
                                    RunName_0000.rst  (restart for Engine)
"""

from __future__ import annotations

import os
import time

import numpy as np

from .. import banner
from ..common.messages import MessageLog, StarterError
from ..input.deck_reader import read_deck
from ..input.starter_keywords import parse_starter_deck
from ..model.model import Model
from .checks import check_model
from .initialization import (build_element_groups,
                             initialize_elements_and_mass,
                             resolve_node_groups, resolve_surfaces)
from .restart import write_restart


def run_name_from_input(path: str) -> str:
    """'MYRUN_0000.rad' -> 'MYRUN' (the Radioss run-name convention)."""
    base = os.path.basename(path)
    for suffix in ("_0000.rad", "_0000.RAD"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


def _listing_summary(model: Model, log: MessageLog) -> None:
    """Model summary tables of the *_0000.out listing (mass, counts...)."""
    log.info("\n                       MODEL SUMMARY")
    log.info("                       -------------")
    log.info(f"     TITLE . . . . . . . . . . . . . . : {model.title}")
    log.info(f"     NUMBER OF NODES . . . . . . . . . : {model.numnod}")
    counts = {"bricks": "BRICK", "tetras": "TETRA4", "shells": "SHELL",
              "sh3n": "SH3N", "trusses": "TRUSS", "springs": "SPRING",
              "beams": "BEAM"}
    for attr, kw in counts.items():
        g = getattr(model, attr)
        if g is not None:
            log.info(f"     NUMBER OF /{kw:<6} ELEMENTS  . . . . : {g.n}")
    log.info(f"     NUMBER OF PARTS . . . . . . . . . : {len(model.parts)}")
    log.info(f"     NUMBER OF MATERIALS . . . . . . . : "
             f"{len(model.materials)}")

    # total mass & center of gravity (frozen nodes' 1e30 masses excluded)
    real = model.mass < 1e29
    m = model.mass[real]
    xm = model.x0[real]
    total = m.sum()
    if total > 0:
        cg = (m[:, None] * xm).sum(axis=0) / total
        log.info(f"     TOTAL MASS  . . . . . . . . . . . : {total:14.7E}")
        log.info(f"     CENTER OF GRAVITY . . . . . . . . : "
                 f"{cg[0]:14.7E} {cg[1]:14.7E} {cg[2]:14.7E}")

    log.info("\n     MATERIALS")
    for mat in model.materials.values():
        log.info(f"       MAT {mat.id:8d}  LAW{mat.law:<3d} RHO="
                 f"{mat.rho0:12.5E}  {mat.title}")
    log.info("\n     PARTS")
    for part in model.parts.values():
        log.info(f"       PART {part.id:8d}  PROP={part.prop_id:<8d} "
                 f"MAT={part.mat_id:<8d} {part.title}")


def run_starter(input_file: str, log: MessageLog | None = None) -> Model:
    """Run the full Starter on ``input_file``; returns the initialized
    model (and writes the .out listing and .rst restart next to it)."""
    log = log or MessageLog()
    run_name = run_name_from_input(input_file)
    out_dir = os.path.dirname(os.path.abspath(input_file))
    listing_path = os.path.join(out_dir, f"{run_name}_0000.out")

    with open(listing_path, "w") as listing:
        log.attach_listing(listing)
        log.info(banner())
        log.info(f" STARTER INPUT FILE . . . . . . . . . : {input_file}")
        log.info(f" RUN NAME . . . . . . . . . . . . . . : {run_name}")
        t0 = time.time()

        # 1. read + parse the deck (lectur.F)
        blocks = read_deck(input_file)
        model = Model()
        parse_starter_deck(blocks, model, log)

        # 2. finalize: ids->indices, element groups, node groups, surfaces
        build_element_groups(model, log)
        resolve_node_groups(model, log)
        resolve_surfaces(model, log)

        # 3. checks before any heavy work (fail early with ALL messages)
        check_model(model, log)

        # 4. element buffers + lumped mass + initial conditions
        if not log.errors:
            initialize_elements_and_mass(model, log)
            _listing_summary(model, log)

        log.info(log.summary())
        log.info(f" STARTER ELAPSED TIME . . . . . . . . : "
                 f"{time.time() - t0:10.3f} s")

        # 5. restart file — only for a clean model, like the original
        log.check()  # raises StarterError if errors were collected
        rst_path = os.path.join(out_dir, f"{run_name}_0000.rst")
        write_restart(model, rst_path)
        log.info(f" RESTART FILE WRITTEN . . . . . . . . : {rst_path}")
        log.info("\n     ------------------------------------------------")
        log.info("     STARTER TERMINATION : NORMAL")
        log.info("     ------------------------------------------------")

    return model
