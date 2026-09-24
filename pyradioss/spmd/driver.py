"""
The ``-np N`` launcher of the Engine: runs the NSPMD domains of a
decomposed model.

Fortran origin
--------------
* ``engine/source/engine/radioss2.F`` — the engine main: every MPI
  process opens ITS domain restart ``RunName_{run-1}_{ISPMD+1:04d}.rst``;
* ``engine/source/mpi/init/inipar.F`` — process bootstrap and the
  ``NSPMD /= NNODES`` coherence test: the number of domain restart files
  written by the Starter (``NSPMD``) must equal the number of processes
  ``mpirun`` started (``NNODES``), else::

      THE REQUIRED NUMBER OF MPI PROCESSES DOES NOT MATCH MPIRUN
      PLEASE, RUN WITH THE PROPER NUMBER OF MPI PROCESSES
      REQUIRED (number of .rst files) NSPMD = ...
      AVAILABLE (-np argument of mpirun)    = ...

* ``engine/source/mpi/init/spmd_chkw.F`` — only domain 0 prints.

Two launch modes
----------------
* under ``mpirun -np N python -m pyradioss.engine -np N -i ...``
  (:func:`~pyradioss.spmd.comm.mpi_world_size` > 1): one domain per MPI
  process over :class:`~pyradioss.spmd.comm.Mpi4pyComm`;
* otherwise: the N domains run as N threads of THIS process over
  :class:`~pyradioss.spmd.comm.ThreadComm` (no MPI library needed — a
  functional, not a performance, path; see comm.py).  A domain that
  raises aborts the whole group (the role of ``MPI_ABORT`` in
  ``spmd_kill.F``): the collective barrier is broken and the pending
  point-to-point receives are poisoned so no domain waits forever, and the
  FIRST exception is re-raised to the caller.
"""

from __future__ import annotations

import glob
import os
import threading
from typing import List, Optional

import numpy as np

from ..common.messages import MessageLog
from .comm import Mpi4pyComm, ThreadComm, mpi_world_size
from .exchange import MSGOFF_EXCH_A


def count_domain_restarts(input_file: str) -> int:
    """Number of domain restart files ``RunName_{run-1:04d}_pppp.rst``
    next to the engine input (the ``NSPMD`` the Starter decomposed into)."""
    from ..engine.engine import run_name_from_input
    run_name, run_num = run_name_from_input(input_file)
    out_dir = os.path.dirname(os.path.abspath(input_file))
    pat = os.path.join(out_dir, f"{glob.escape(run_name)}_{run_num - 1:04d}_"
                                "[0-9][0-9][0-9][0-9].rst")
    return len(glob.glob(pat))


def _mismatch(nspmd_files: int, available: int, mpi: bool) -> RuntimeError:
    """The inipar.F coherence-test message."""
    what = "(-np argument of mpirun)   " if mpi else "(-np argument)             "
    return RuntimeError(
        "THE REQUIRED NUMBER OF MPI PROCESSES DOES NOT MATCH MPIRUN\n"
        "PLEASE, RUN WITH THE PROPER NUMBER OF MPI PROCESSES\n"
        f"REQUIRED (number of .rst files) NSPMD = {nspmd_files}\n"
        f"AVAILABLE {what} = {available}\n"
        " E R R O R     T E R M I N A T I O N")


def _rank_log(rank: int, log: Optional[MessageLog]) -> Optional[MessageLog]:
    """Domain 0 uses the caller's log; the others get ``None`` so
    run_engine gives them its silent per-domain log (spmd_chkw.F — only P0
    prints; PYRADIOSS_SPMD_LOG_ALL=1 adds per-domain listings)."""
    return log if rank == 0 else None


def run_engine_spmd(input_file: str, nspmd: int,
                    log: Optional[MessageLog] = None):
    """Run the Engine on the ``nspmd`` domains of a decomposed model and
    return domain 0's result (the GLOBAL output view with the reduced
    ledgers — see engine.run_engine).

    Under mpirun every process calls this and gets its own domain's return
    value (the global view on rank 0, its local model elsewhere)."""
    from ..engine.engine import run_engine

    nfiles = count_domain_restarts(input_file)
    world = mpi_world_size()
    if world > 1:
        comm = Mpi4pyComm()
        if nfiles != comm.size:
            err = _mismatch(nfiles, comm.size, mpi=True)
            if comm.is_root:
                print(str(err))
            raise err
        try:
            return run_engine(input_file, _rank_log(comm.rank, log),
                              comm=comm)
        except BaseException:
            comm.abort(2)
            raise

    nspmd = int(nspmd)
    if nfiles != nspmd:
        raise _mismatch(nfiles, nspmd, mpi=False)
    if nspmd <= 1:
        return run_engine(input_file, log)

    comms = ThreadComm.group(nspmd)
    results: List = [None] * nspmd
    errors: List[Optional[BaseException]] = [None] * nspmd
    order: List[int] = []            # ranks in the order they failed
    lock = threading.Lock()
    aborted = threading.Event()

    def _abort_group(dead: int) -> None:
        """MPI_ABORT for the thread group: poison every receive another
        domain may post from the dead one (a message of the wrong shape
        makes the waiting irecv raise) and break the collective barrier.
        Every dying domain poisons ITS outgoing queues, so a domain
        blocked on a receive from a domain that died of the abort itself
        is released too."""
        for r in range(nspmd):
            if r != dead:
                try:
                    comms[dead].isend(np.empty((0,)), r, MSGOFF_EXCH_A)
                except Exception:    # pragma: no cover - defensive
                    pass
        if not aborted.is_set():
            aborted.set()
            try:
                comms[dead].abort(2)
            except SystemExit:
                pass

    def _worker(rank: int) -> None:
        try:
            results[rank] = run_engine(input_file, _rank_log(rank, log),
                                       comm=comms[rank])
        except BaseException as exc:     # noqa: BLE001 - re-raised below
            with lock:
                errors[rank] = exc
                order.append(rank)
            _abort_group(rank)

    threads = [threading.Thread(target=_worker, args=(r,), daemon=True,
                                name=f"pyradioss-spmd-{r}")
               for r in range(nspmd)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if order:
        first = errors[order[0]]
        raise first
    return results[0]
