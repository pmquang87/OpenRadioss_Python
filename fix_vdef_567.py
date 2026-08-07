import re

with open('pyradioss/accel/jit_kernels/shells_qbat.py', 'r') as f:
    text = f.read()

rep = '''              vdef[e, ng, 5] = bm[e, ng, 0] * r13y + bm[e, ng, 1] * r24y + bm[e, ng, 2] * rhiy
              vdef[e, ng, 6] = -(bm[e, ng, 4] * r13x + bm[e, ng, 5] * r24x + bm[e, ng, 6] * rhix)
              vdef[e, ng, 7] = -(bm[e, ng, 0] * r13x + bm[e, ng, 1] * r24x + bm[e, ng, 2] * rhix) + bm[e, ng, 4] * r13y + bm[e, ng, 5] * r24y + bm[e, ng, 6] * rhiy'''

text = re.sub(r'              kxx = 0\.0; kyy = 0\.0; kxy = 0\.0.*?vdef\[e, ng, 7\] = kxy', rep, text, flags=re.DOTALL)

with open('pyradioss/accel/jit_kernels/shells_qbat.py', 'w') as f:
    f.write(text)
print('Fixed vdef 5, 6, 7 with DOTALL!')
