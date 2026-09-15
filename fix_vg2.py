import re

with open('pyradioss/accel/jit_kernels/shells_qbat.py', 'r') as f:
    text = f.read()

rep = 'vg2x = ve[e, 0, 0] - ve[e, 1, 0] + ve[e, 2, 0] - ve[e, 3, 0]; vg2y = ve[e, 0, 1] - ve[e, 1, 1] + ve[e, 2, 1] - ve[e, 3, 1]; vg2z = ve[e, 0, 2] - ve[e, 1, 2] + ve[e, 2, 2] - ve[e, 3, 2]'
text = re.sub(r'vg2x = vg0x - vg1x; vg2y = vg0y - vg1y; vg2z = vg0z - vg1z', rep, text)

with open('pyradioss/accel/jit_kernels/shells_qbat.py', 'w') as f:
    f.write(text)
print('Fixed vg2!')
