# Memory & Development State

**Current State**: Working on M43 (Exclude QBAT/QEPH from numba auto rule).
- Created implementation plan for M43.
- Approved by user.
- Fixed the logic in `pyradioss/accel/__init__.py`.
- Wrote and passed tests for M43.
- Fast tier tests for M43 are running.
- Profiling timing check on `rigid_impactor` (QBAT deck) is running to get numpy vs numba timings.

**Important Info**:
- The project aims to implement features one milestone at a time (currently M42 to M46).
- M42 is complete.
- M43 excludes QBAT/QEPH elements from the auto `numba` rule because they lack JIT kernels, meaning numba falls back to pure Python loops for them which are slower than pure NumPy.


## M44 (QBAT) Status Update
Completed translation of QBAT kernel hotspots into explicit Numba loops in pyradioss/accel/jit_kernels/shells_qbat.py.
Validation vs Fortran for QBAT examples yielded perfect MATCH.
Spawned subagent for M45 (QEPH).



## M45 (QEPH) Status Update
Completed translation of QEPH kernel hotspots into explicit Numba loops in pyradioss/accel/jit_kernels/shells_qeph.py using 5 subagents in parallel.
Validation vs tests/test_m41_qeph.py with Numba yielded perfect MATCH.



## M46 Status Update
Completed shell_bt4 dt-branch reconciliation. Removed dead code and unused _condensed_length from pyradioss/elements/shell_bt4.py.

