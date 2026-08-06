import ast
import sys
import os

from codegen import CodeGen

cg = CodeGen()

with open('pyradioss/elements/shell_qeph.py', 'r') as fp:
    src = fp.read()

funcs = ['_geometry', '_sym3_inv', '_kinematics', '_rates', '_fint_const', '_fint_stab', '_project']
for func in funcs:
    print('Unrolling', func)
    cg.unroll_function(src, func, 'n')

with open('C:/Users/pmqua/.gemini/antigravity/brain/ae4b6e32-96b3-40b2-8749-a7e52e5aa382/scratch/qeph_unrolled.py', 'w') as fp:
    fp.write('\n\n'.join(cg.results))
