import re

with open('pyradioss/accel/jit_kernels/shells_qbat.py', 'r') as f:
    text = f.read()

rep = '''              ksi = _VPG[ng, 0]; eta = _VPG[ng, 1]
              a_1 = 0.25 / cdet[e, ng] if cdet[e, ng] > 1e-20 else 0.25 / 1e-20
              c11 = (my34 + my13 * ksi) * a_1
              c12 = (-my23 + my13 * eta) * a_1
              c21 = (-mx34 - mx13 * ksi) * a_1
              c22 = (mx23 - mx13 * eta) * a_1
              
              beta1_1 = my34 + my23 * eta
              beta2 = mx34 + mx23 * eta
              ksi1_1 = -my23 + my13 * ksi
              ksi2 = mx23 - mx13 * ksi
              b1 = c11 * beta1_1 + c12 * ksi1_1
              b2 = c21 * beta1_1 + c22 * ksi1_1
              
              beta1_2 = my13 + my23 * eta
              ksi1_2 = my13 + my34 * ksi
              
              bc[e, ng, 0] = 0.25 * b1
              bc[e, ng, 1] = 0.25 * b2
              bc[e, ng, 2] = beta1_2 * c11 + ksi1_2 * c12
              bc[e, ng, 3] = beta1_2 * c21 + ksi1_2 * c22
              bc[e, ng, 4] = c11 - c12
              bc[e, ng, 5] = c21 - c22
              bc[e, ng, 6] = -beta1_2 * c11 + ksi1_2 * c12
              bc[e, ng, 7] = -beta1_2 * c21 + ksi1_2 * c22
              bc[e, ng, 8] = c11 + c12
              bc[e, ng, 9] = c21 + c22
              bc[e, ng, 10] = -beta1_2 * c11 - ksi1_2 * c12
              bc[e, ng, 11] = -beta1_2 * c21 - ksi1_2 * c22
              bc[e, ng, 12] = -c11 + c12
              bc[e, ng, 13] = -c21 + c22
              bc[e, ng, 14] = beta1_2 * c11 - ksi1_2 * c12
              bc[e, ng, 15] = beta1_2 * c21 - ksi1_2 * c22
              bc[e, ng, 16] = c11 * beta2 + c12 * ksi2
              bc[e, ng, 17] = c21 * beta2 + c22 * ksi2'''

text = re.sub(r'              ksi = _VPG\[ng, 0\]; eta = _VPG\[ng, 1\]\n              \n              c0 =.*\n.*\n.*\n.*\n              \n              for m in range\(4\):\n                  bc\[e, ng, m\].*\n                  bc\[e, ng, 4 \+ m\].*\n                  bc\[e, ng, 8 \+ m\].*\n                  bc\[e, ng, 12 \+ m\].*\n                  bc\[e, ng, 16 \+ m\].*\n                  bc\[e, ng, 20 \+ m\].*', rep, text)

with open('pyradioss/accel/jit_kernels/shells_qbat.py', 'w') as f:
    f.write(text)
print('Fixed bc!')
