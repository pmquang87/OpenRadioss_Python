import re

with open('pyradioss/accel/jit_kernels/shells_qbat.py', 'r') as f:
    text = f.read()

# Fix r13x, r24x, rhix, rtix calculation!
rep = '''          r13x = cx0 - cx2; r13y = cy0 - cy2
          r24x = cx1 - cx3; r24y = cy1 - cy3
          rhix = cx0 - cx1 + cx2 - cx3; rhiy = cy0 - cy1 + cy2 - cy3
          rtix = cx0 + cx1 + cx2 + cx3; rtiy = cy0 + cy1 + cy2 + cy3'''
text = re.sub(r'          rr0x = cx0\*e1x.*\n.*\n.*\n.*\n          rtix = rr0x \+ rr1x \+ rr2x \+ rr3x; rtiy = rr0y \+ rr1y \+ rr2y \+ rr3y', rep, text)

with open('pyradioss/accel/jit_kernels/shells_qbat.py', 'w') as f:
    f.write(text)
print('Fixed r13, r24, rhi, rti!')
