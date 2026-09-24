"""
``pyradioss.spmd`` — the port of the OpenRadioss SPMD (Single Program,
Multiple Data) domain-decomposition and message-passing layer.

Fortran origin
--------------
* ``engine/source/mpi/`` (289 files) — the ``SPMD_*`` wrappers and every
  per-subsystem exchange routine (``spmd_exch_a.F``, ``spmd_glob_min5.F``,
  ``spmd_exch_v.F``, ``spmd_collect.F`` ...);
* ``starter/source/spmd/`` (11 files) — the domain decomposition
  (``domdec1.F``, ``domdec2.F``, ``initwg.F``, ``frontplus.F``,
  ``ddtools.F``) and the per-domain restart split (``ddsplit.F``).

Layout of the port
------------------
* :mod:`pyradioss.spmd.comm`     — communicator backends (serial, threads,
  mpi4py) and the ``glob_min`` custom reduction;
* :mod:`pyradioss.spmd.domdec`   — Starter side: element weights, weighted
  RCB partition, frontier nodes (``IFRONT``/``IFRONTPLUS``), ``NODGLOB``,
  ``WEIGHT``, entity ownership rules and the per-domain model slice;
* :mod:`pyradioss.spmd.exchange` — Engine side: the per-cycle frontier
  force/stiffness sum, the global time-step packet, the weighted global
  sums and the rank-0 gathers for the listing / TH / ANIM outputs;
* :mod:`pyradioss.spmd.driver`   — the ``-np N`` launcher (mpi4py under
  ``mpirun``, otherwise one thread per domain in this process).
"""

from .comm import (  # noqa: F401
    SPMD_MAX, SPMD_MIN, SPMD_SUM, SPMD_PROD,
    Comm, Request, SerialComm, SERIAL, ThreadComm, Mpi4pyComm,
    mpi_world_size,
)

__all__ = [
    "SPMD_MAX", "SPMD_MIN", "SPMD_SUM", "SPMD_PROD",
    "Comm", "Request", "SerialComm", "SERIAL", "ThreadComm", "Mpi4pyComm",
    "mpi_world_size",
]
