
# QEPH Numba Translation Instructions
Translate the Python/NumPy logic from pyradioss/elements/shell_qeph.py into Numba 
opython kernels in pyradioss/accel/jit_kernels/shells_qeph.py.

1. **Loop Unrolling**:
   You must manually unroll ALL vector and matrix operations into explicit scalar loops over 
 elements. Do NOT use any NumPy array operations, slicing that returns arrays, or 
p.sum/
p.dot/
p.einsum inside the loops.
   
2. **Pre and Post Wrappers**:
   Modify pyradioss/elements/shell_qeph.py's orces function.
   Split it such that everything BEFORE the st['slices'] integration loop is passed into a qeph_pre JIT kernel.
   Everything AFTER the integration loop is passed into a qeph_post JIT kernel.
   The integration loop itself MUST REMAIN IN PYTHON, just like for QBAT.
   
3. **No Legacy Deletions**:
   Do NOT delete any legacy _kinematics, _rates, _fint_const, _fint_stab, _project functions from shell_qeph.py. Keep them so the pure numpy backend works exactly as before. Add jit_pre = accel_get('qeph_pre') and if not None, call it. Else call a python _pre wrapper. Same for post.

4. **Parity**:
   The 
umba output must remain BYTE-IDENTICAL. Be extremely careful with signs and accumulation variables.
