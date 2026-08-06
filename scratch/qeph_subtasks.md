
# QEPH Numba Unrolling Task
You are tasked with unrolling specific functions from pyradioss/elements/shell_qeph.py into Numba 
opython compatible scalar loops.
DO NOT use 
p.einsum, 
p.dot, 
p.cross, or array-returning slicing inside the loops. You must write explicit scalar loops over 
 elements. 

Example:
`python
def unrolled_func(xe, ve):
    n = len(xe)
    out = np.empty(n)
    for e in range(n):
        # DO NOT DO: r = xe[e, 1] - xe[e, 0]
        rx = xe[e, 1, 0] - xe[e, 0, 0]
        ry = xe[e, 1, 1] - xe[e, 0, 1]
        rz = xe[e, 1, 2] - xe[e, 0, 2]
        # ...
`
Save your result in scratch/qeph_unrolled_{TASK_NAME}.py. 
You ONLY need to write the unrolled functions. Do NOT integrate it into shell_qeph.py. We will do that later.
