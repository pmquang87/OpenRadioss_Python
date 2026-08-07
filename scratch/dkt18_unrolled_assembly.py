import numpy as np

def cdkfint3(vol, thk0, force, mom, px2, py2, px3, py3, bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3, f11, f12, f13, f21, f22, f23, f32, f33, m11, m12, m13, m21, m22, m23):
    n = len(vol)
    for i in range(n):
        c2 = vol[i]
        fx2 = c2 * (px2[i] * force[i, 0] + py2[i] * force[i, 2])
        fy2 = c2 * (py2[i] * force[i, 1] + px2[i] * force[i, 2])
        fx3 = c2 * (px3[i] * force[i, 0] + py3[i] * force[i, 2])
        fy3 = c2 * (py3[i] * force[i, 1] + px3[i] * force[i, 2])
        f12[i] += fx2
        f22[i] += fy2
        f13[i] += fx3
        f23[i] += fy3

    for i in range(n):
        c2 = vol[i] * thk0[i]
        f32[i] += c2 * (bz1[i, 0] * mom[i, 0] + bz2[i, 0] * mom[i, 1] + bz3[i, 0] * mom[i, 2])
        f33[i] += c2 * (bz1[i, 1] * mom[i, 0] + bz2[i, 1] * mom[i, 1] + bz3[i, 1] * mom[i, 2])
        m11[i] += c2 * (brx1[i, 0] * mom[i, 0] + brx2[i, 0] * mom[i, 1] + brx3[i, 0] * mom[i, 2])
        m21[i] += c2 * (bry1[i, 0] * mom[i, 0] + bry2[i, 0] * mom[i, 1] + bry3[i, 0] * mom[i, 2])
        m12[i] += c2 * (brx1[i, 1] * mom[i, 0] + brx2[i, 1] * mom[i, 1] + brx3[i, 1] * mom[i, 2])
        m22[i] += c2 * (bry1[i, 1] * mom[i, 0] + bry2[i, 1] * mom[i, 1] + bry3[i, 1] * mom[i, 2])
        m13[i] += c2 * (brx1[i, 2] * mom[i, 0] + brx2[i, 2] * mom[i, 1] + brx3[i, 2] * mom[i, 2])
        m23[i] += c2 * (bry1[i, 2] * mom[i, 0] + bry2[i, 2] * mom[i, 1] + bry3[i, 2] * mom[i, 2])

def cdkfcum3(px2, py2, px3, py3, r11, r12, r13, r21, r22, r23, r31, r32, r33, f11, f12, f13, f21, f22, f23, f31, f32, f33, m11, m12, m13, m21, m22, m23, m31, m32, m33):
    n = len(f12)
    for i in range(n):
        lx = r11[i] * f12[i] + r12[i] * f22[i] + r13[i] * f32[i]
        ly = r21[i] * f12[i] + r22[i] * f22[i] + r23[i] * f32[i]
        lz = r31[i] * f12[i] + r32[i] * f22[i] + r33[i] * f32[i]
        f12[i] = lx
        f22[i] = ly
        f32[i] = lz

        lx = r11[i] * f13[i] + r12[i] * f23[i] + r13[i] * f33[i]
        ly = r21[i] * f13[i] + r22[i] * f23[i] + r23[i] * f33[i]
        lz = r31[i] * f13[i] + r32[i] * f23[i] + r33[i] * f33[i]
        f13[i] = lx
        f23[i] = ly
        f33[i] = lz

        f11[i] = -f12[i] - f13[i]
        f21[i] = -f22[i] - f23[i]
        f31[i] = -f32[i] - f33[i]

    for i in range(n):
        lx = r11[i] * m11[i] + r12[i] * m21[i]
        ly = r21[i] * m11[i] + r22[i] * m21[i]
        m31[i] = r31[i] * m11[i] + r32[i] * m21[i]
        m11[i] = lx
        m21[i] = ly

        lx = r11[i] * m12[i] + r12[i] * m22[i]
        ly = r21[i] * m12[i] + r22[i] * m22[i]
        m32[i] = r31[i] * m12[i] + r32[i] * m22[i]
        m12[i] = lx
        m22[i] = ly

        lx = r11[i] * m13[i] + r12[i] * m23[i]
        ly = r21[i] * m13[i] + r22[i] * m23[i]
        m33[i] = r31[i] * m13[i] + r32[i] * m23[i]
        m13[i] = lx
        m23[i] = ly
