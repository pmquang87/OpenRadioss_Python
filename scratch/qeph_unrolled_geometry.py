import numpy as np

def _geometry(xe):
    n = len(xe)
    
    E = np.empty((n, 3, 3))
    area = np.empty(n)
    a_i = np.empty(n)
    z1 = np.empty(n)
    corx = np.empty((n, 4))
    cory = np.empty((n, 4))
    x13 = np.empty(n)
    x24 = np.empty(n)
    y13 = np.empty(n)
    y24 = np.empty(n)
    mx13 = np.empty(n)
    mx23 = np.empty(n)
    mx34 = np.empty(n)
    my13 = np.empty(n)
    my23 = np.empty(n)
    my34 = np.empty(n)
    l13 = np.empty(n)
    l24 = np.empty(n)
    ll = np.empty(n)
    lm = np.empty(n)

    EM20 = 1e-20
    FACDT = 1.25

    for e in range(n):
        # covariant vectors R = x2+x3-x1-x4, S = x3+x4-x1-x2
        rx = xe[e, 1, 0] + xe[e, 2, 0] - xe[e, 0, 0] - xe[e, 3, 0]
        ry = xe[e, 1, 1] + xe[e, 2, 1] - xe[e, 0, 1] - xe[e, 3, 1]
        rz = xe[e, 1, 2] + xe[e, 2, 2] - xe[e, 0, 2] - xe[e, 3, 2]

        sx = xe[e, 2, 0] + xe[e, 3, 0] - xe[e, 0, 0] - xe[e, 1, 0]
        sy = xe[e, 2, 1] + xe[e, 3, 1] - xe[e, 0, 1] - xe[e, 1, 1]
        sz = xe[e, 2, 2] + xe[e, 3, 2] - xe[e, 0, 2] - xe[e, 1, 2]

        # cross3(r, s)
        e3x = ry * sz - rz * sy
        e3y = rz * sx - rx * sz
        e3z = rx * sy - ry * sx

        det = np.sqrt(e3x * e3x + e3y * e3y + e3z * e3z)
        area[e] = 0.25 * det
        
        det_max = det if det > EM20 else EM20
        e3x /= det_max
        e3y /= det_max
        e3z /= det_max

        c1c1 = rx * rx + ry * ry + rz * rz
        c2c2 = sx * sx + sy * sy + sz * sz

        c1c1_max = c1c1 if c1c1 > EM20 else EM20
        c2c2_max = c2c2 if c2c2 > EM20 else EM20

        if c1c1 > 0.0:
            c2_1 = np.sqrt(c2c2 / c1c1_max)
            c1_1 = 1.0
        else:
            c2_1 = 1.0
            c1_1 = np.sqrt(c1c1 / c2c2_max)

        # cross3(s, e3)
        se3x = sy * e3z - sz * e3y
        se3y = sz * e3x - sx * e3z
        se3z = sx * e3y - sy * e3x

        e1x = rx * c2_1 + se3x * c1_1
        e1y = ry * c2_1 + se3y * c1_1
        e1z = rz * c2_1 + se3z * c1_1

        norm_e1 = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        norm_e1_max = norm_e1 if norm_e1 > EM20 else EM20
        
        e1x /= norm_e1_max
        e1y /= norm_e1_max
        e1z /= norm_e1_max

        # cross3(e3, e1)
        e2x = e3y * e1z - e3z * e1y
        e2y = e3z * e1x - e3x * e1z
        e2z = e3x * e1y - e3y * e1x

        E[e, 0, 0] = e1x; E[e, 0, 1] = e2x; E[e, 0, 2] = e3x
        E[e, 1, 0] = e1y; E[e, 1, 1] = e2y; E[e, 1, 2] = e3y
        E[e, 2, 0] = e1z; E[e, 2, 1] = e2z; E[e, 2, 2] = e3z

        # local corner coordinates relative to NODE 1
        d2x = xe[e, 1, 0] - xe[e, 0, 0]
        d2y = xe[e, 1, 1] - xe[e, 0, 1]
        d2z = xe[e, 1, 2] - xe[e, 0, 2]

        d3x = xe[e, 2, 0] - xe[e, 0, 0]
        d3y = xe[e, 2, 1] - xe[e, 0, 1]
        d3z = xe[e, 2, 2] - xe[e, 0, 2]

        d4x = xe[e, 3, 0] - xe[e, 0, 0]
        d4y = xe[e, 3, 1] - xe[e, 0, 1]
        d4z = xe[e, 3, 2] - xe[e, 0, 2]

        xl2 = d2x * e1x + d2y * e1y + d2z * e1z
        yl2 = d2x * e2x + d2y * e2y + d2z * e2z
        
        xl3 = d3x * e1x + d3y * e1y + d3z * e1z
        yl3 = d3x * e2x + d3y * e2y + d3z * e2z
        
        xl4 = d4x * e1x + d4y * e1y + d4z * e1z
        yl4 = d4x * e2x + d4y * e2y + d4z * e2z

        centerx = 0.25 * (xe[e, 0, 0] + xe[e, 1, 0] + xe[e, 2, 0] + xe[e, 3, 0])
        centery = 0.25 * (xe[e, 0, 1] + xe[e, 1, 1] + xe[e, 2, 1] + xe[e, 3, 1])
        centerz = 0.25 * (xe[e, 0, 2] + xe[e, 1, 2] + xe[e, 2, 2] + xe[e, 3, 2])

        z1_val = (xe[e, 0, 0] - centerx) * e3x + (xe[e, 0, 1] - centery) * e3y + (xe[e, 0, 2] - centerz) * e3z
        z1[e] = z1_val

        # centered corner coordinates
        cx0 = 0.25 * (xl2 + xl3 + xl4)
        cy0 = 0.25 * (yl2 + yl3 + yl4)

        corx[e, 0] = -cx0
        corx[e, 1] = xl2 - cx0
        corx[e, 2] = xl3 - cx0
        corx[e, 3] = xl4 - cx0

        cory[e, 0] = -cy0
        cory[e, 1] = yl2 - cy0
        cory[e, 2] = yl3 - cy0
        cory[e, 3] = yl4 - cy0

        x13_val = 0.5 * (corx[e, 0] - corx[e, 2])
        x24_val = 0.5 * (corx[e, 1] - corx[e, 3])
        y13_val = 0.5 * (cory[e, 0] - cory[e, 2])
        y24_val = 0.5 * (cory[e, 1] - cory[e, 3])

        x13[e] = x13_val
        x24[e] = x24_val
        y13[e] = y13_val
        y24[e] = y24_val

        mx13[e] = 0.5 * (corx[e, 0] + corx[e, 2])
        mx23[e] = 0.5 * (corx[e, 1] + corx[e, 2])
        mx34[e] = 0.5 * (corx[e, 2] + corx[e, 3])
        
        my13[e] = 0.5 * (cory[e, 0] + cory[e, 2])
        my23[e] = 0.5 * (cory[e, 1] + cory[e, 2])
        my34[e] = 0.5 * (cory[e, 2] + cory[e, 3])

        l13_val = x13_val * x13_val + y13_val * y13_val
        l24_val = x24_val * x24_val + y24_val * y24_val

        l13[e] = l13_val
        l24[e] = l24_val

        lm[e] = 0.5 * (l13_val + l24_val)

        a_i_val = 1.0 / (area[e] if area[e] > EM20 else EM20)
        a_i[e] = a_i_val

        # taper term HS
        v1 = abs(corx[e, 1] * cory[e, 3] - cory[e, 1] * corx[e, 3])
        v2 = abs(corx[e, 0] * cory[e, 2] - cory[e, 0] * corx[e, 2])
        hs = (v1 if v1 > v2 else v2) * a_i_val

        # condensed characteristic length LL
        rx_ll = xl2 + xl3 - xl4
        ry_ll = yl2 + yl3 - yl4
        sx_ll = -xl2 + xl3 + xl4
        sy_ll = -yl2 + yl3 + yl4

        c1 = np.sqrt(rx_ll * rx_ll + ry_ll * ry_ll)
        c2 = np.sqrt(sx_ll * sx_ll + sy_ll * sy_ll)

        min_c = c1 if c1 < c2 else c2
        min_c = min_c if min_c > EM20 else EM20

        max_c = c1 if c1 > c2 else c2

        v_fac1 = 0.25 * (max_c / min_c - 1.0)
        fac1 = (v_fac1 if v_fac1 < 0.5 else 0.5) + 1.0

        c1c2 = c1 * c2
        c1c2_max = c1c2 if c1c2 > EM20 else EM20
        fac2 = 4.0 * area[e] / c1c2_max
        fac2 = fac2 - 0.7071
        fac2 = fac2 if fac2 > 0.0 else 0.0
        fac2 = 3.413 * fac2
        fac2 = 0.78 + 0.22 * (fac2 * fac2 * fac2)

        faci = 2.0 * fac1 * fac2

        lldiag = l13_val if l13_val > l24_val else l24_val

        s1 = np.sqrt(faci * (FACDT + hs) * lldiag)
        s1 = s1 if s1 > 1.0e-10 else 1.0e-10

        ll[e] = area[e] / s1

    return dict(E=E, area=area, a_i=a_i, z1=z1, corx=corx, cory=cory,
                x13=x13, x24=x24, y13=y13, y24=y24,
                mx13=mx13, mx23=mx23, mx34=mx34,
                my13=my13, my23=my23, my34=my34,
                l13=l13, l24=l24, ll=ll, lm=lm)
