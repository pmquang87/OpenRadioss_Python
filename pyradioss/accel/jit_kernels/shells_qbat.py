import numpy as np
from numba import njit
from pyradioss.common.constants import EM20, EP30



EM20 = 1e-20
_PG = 0.577350269189626
_VPG = np.array([[-_PG, -_PG], [_PG, -_PG], [_PG, _PG], [-_PG, _PG]])
_KSI_N = np.array([-1.0, 1.0, 1.0, -1.0])
_ETA_N = np.array([-1.0, -1.0, 1.0, 1.0])
_VKSI = np.array([
    [-0.10566243, -0.10566243, -0.39433757, -0.39433757],
    [ 0.10566243,  0.10566243,  0.39433757,  0.39433757],
    [ 0.39433757,  0.39433757,  0.10566243,  0.10566243],
    [-0.39433757, -0.39433757, -0.10566243, -0.10566243]
])
_VETA = np.array([
    [-0.10566243, -0.39433757, -0.39433757, -0.10566243],
    [-0.39433757, -0.10566243, -0.10566243, -0.39433757],
    [ 0.39433757,  0.10566243,  0.10566243,  0.39433757],
    [ 0.10566243,  0.39433757,  0.39433757,  0.10566243]
])

@njit(cache=True)
def qbat_pre_flat(xe, ve, vre, off, dt):
    n = xe.shape[0]
    E = np.empty((n, 3, 3))
    area = np.empty(n)
    lc = np.empty(n)
    vdef3 = np.empty(n)
    cdet = np.empty((n, 4))
    vdef = np.zeros((n, 4, 8))
    bm = np.zeros((n, 4, 8))
    bc = np.zeros((n, 4, 24))
    x13n = np.empty(n); x24n = np.empty(n); y13n = np.empty(n); y24n = np.empty(n)
    
    for e in range(n):
        if off[e] <= 0.0:
            continue
            
        # Frame
        rx = xe[e, 1, 0] + xe[e, 2, 0] - xe[e, 0, 0] - xe[e, 3, 0]
        ry = xe[e, 1, 1] + xe[e, 2, 1] - xe[e, 0, 1] - xe[e, 3, 1]
        rz = xe[e, 1, 2] + xe[e, 2, 2] - xe[e, 0, 2] - xe[e, 3, 2]
        sx = xe[e, 2, 0] + xe[e, 3, 0] - xe[e, 0, 0] - xe[e, 1, 0]
        sy = xe[e, 2, 1] + xe[e, 3, 1] - xe[e, 0, 1] - xe[e, 1, 1]
        sz = xe[e, 2, 2] + xe[e, 3, 2] - xe[e, 0, 2] - xe[e, 1, 2]
        
        e3x = ry * sz - rz * sy
        e3y = rz * sx - rx * sz
        e3z = rx * sy - ry * sx
        det = np.sqrt(e3x**2 + e3y**2 + e3z**2)
        det_max = det if det > EM20 else EM20
        e3x /= det_max
        e3y /= det_max
        e3z /= det_max
        
        c1c1 = rx**2 + ry**2 + rz**2
        c2c2 = sx**2 + sy**2 + sz**2
        c21 = np.sqrt(c2c2 / (c1c1 if c1c1 > EM20 else EM20))
        
        s_x_e3_x = sy * e3z - sz * e3y
        s_x_e3_y = sz * e3x - sx * e3z
        s_x_e3_z = sx * e3y - sy * e3x
        
        e1x = rx * c21 + s_x_e3_x
        e1y = ry * c21 + s_x_e3_y
        e1z = rz * c21 + s_x_e3_z
        e1_norm = np.sqrt(e1x**2 + e1y**2 + e1z**2)
        e1_norm_max = e1_norm if e1_norm > EM20 else EM20
        e1x /= e1_norm_max; e1y /= e1_norm_max; e1z /= e1_norm_max
        
        e2x = e3y * e1z - e3z * e1y
        e2y = e3z * e1x - e3x * e1z
        e2z = e3x * e1y - e3y * e1x
        
        E[e, 0, 0] = e1x; E[e, 0, 1] = e2x; E[e, 0, 2] = e3x
        E[e, 1, 0] = e1y; E[e, 1, 1] = e2y; E[e, 1, 2] = e3y
        E[e, 2, 0] = e1z; E[e, 2, 1] = e2z; E[e, 2, 2] = e3z
        
        a = 0.25 * det
        area[e] = a
        area_i = 1.0 / (a if a > EM20 else EM20)
        
        # d @ E
        x0=0.0; y0=0.0; z0=0.0
        d1x = xe[e, 1, 0] - xe[e, 0, 0]; d1y = xe[e, 1, 1] - xe[e, 0, 1]; d1z = xe[e, 1, 2] - xe[e, 0, 2]
        x1 = d1x * e1x + d1y * e1y + d1z * e1z; y1 = d1x * e2x + d1y * e2y + d1z * e2z; z1 = d1x * e3x + d1y * e3y + d1z * e3z
        d2x = xe[e, 2, 0] - xe[e, 0, 0]; d2y = xe[e, 2, 1] - xe[e, 0, 1]; d2z = xe[e, 2, 2] - xe[e, 0, 2]
        x2 = d2x * e1x + d2y * e1y + d2z * e1z; y2 = d2x * e2x + d2y * e2y + d2z * e2z; z2 = d2x * e3x + d2y * e3y + d2z * e3z
        d3x = xe[e, 3, 0] - xe[e, 0, 0]; d3y = xe[e, 3, 1] - xe[e, 0, 1]; d3z = xe[e, 3, 2] - xe[e, 0, 2]
        x3 = d3x * e1x + d3y * e1y + d3z * e1z; y3 = d3x * e2x + d3y * e2y + d3z * e2z; z3 = d3x * e3x + d3y * e3y + d3z * e3z
        
        cx_m = 0.25 * (x0 + x1 + x2 + x3); cy_m = 0.25 * (y0 + y1 + y2 + y3)
        cx0 = x0 - cx_m; cy0 = y0 - cy_m
        cx1 = x1 - cx_m; cy1 = y1 - cy_m
        cx2 = x2 - cx_m; cy2 = y2 - cy_m
        cx3 = x3 - cx_m; cy3 = y3 - cy_m
        
        x13 = 0.5 * (cx0 - cx2); y13 = 0.5 * (cy0 - cy2)
        x24 = 0.5 * (cx1 - cx3); y24 = 0.5 * (cy1 - cy3)
        mx13 = 0.5 * (cx0 + cx2); my13 = 0.5 * (cy0 + cy2)
        mx23 = 0.5 * (cx1 + cx2); my23 = 0.5 * (cy1 + cy2)
        mx34 = 0.5 * (cx2 + cx3); my34 = 0.5 * (cy2 + cy3)
        
        x13n[e] = x13 * area_i; x24n[e] = x24 * area_i; y13n[e] = y13 * area_i; y24n[e] = y24 * area_i
        
        # velocities
        vg0x = ve[e, 0, 0] - ve[e, 2, 0]; vg0y = ve[e, 0, 1] - ve[e, 2, 1]; vg0z = ve[e, 0, 2] - ve[e, 2, 2]
        vg1x = ve[e, 1, 0] - ve[e, 3, 0]; vg1y = ve[e, 1, 1] - ve[e, 3, 1]; vg1z = ve[e, 1, 2] - ve[e, 3, 2]
        vg2x = vg0x - vg1x; vg2y = vg0y - vg1y; vg2z = vg0z - vg1z
        
        v13x = vg0x * e1x + vg0y * e1y + vg0z * e1z
        v13y = vg0x * e2x + vg0y * e2y + vg0z * e2z
        v13z = vg0x * e3x + vg0y * e3y + vg0z * e3z
        v24x = vg1x * e1x + vg1y * e1y + vg1z * e1z
        v24y = vg1x * e2x + vg1y * e2y + vg1z * e2z
        v24z = vg1x * e3x + vg1y * e3y + vg1z * e3z
        vhix = vg2x * e1x + vg2y * e1y + vg2z * e1z
        vhiy = vg2x * e2x + vg2y * e2y + vg2z * e2z
        vhiz = vg2x * e3x + vg2y * e3y + vg2z * e3z
        
        dt05 = 0.5 * dt
        dt025 = 0.25 * dt
        exz = y24 * v13z - y13 * v24z
        eyz = -x24 * v13z + x13 * v24z
        ddry = dt05 * exz * area_i
        ddrx = dt05 * eyz * area_i
        v13x_c = v13x; v24x_c = v24x; vhix_c = vhix
        den1 = x13 - x24
        ddrz1 = 0.0 if abs(den1) < 1.0e-10 else dt025 * (v13y - v24y) / den1
        v13x -= ddry * v13z + ddrz1 * v13y
        v24x -= ddry * v24z + ddrz1 * v24y
        vhix -= ddry * vhiz + ddrz1 * vhiy
        den2 = y13 + y24
        ddrz2 = 0.0 if abs(den2) < 1.0e-10 else dt025 * (v13x_c + v24x_c) / den2
        v13y -= ddrx * v13z + ddrz2 * v13x_c
        v24y -= ddrx * v24z + ddrz2 * v24x_c
        vhiy -= ddrx * vhiz + ddrz2 * vhix_c
        
        # RR
        rr0x = vre[e, 0, 0] * e1x + vre[e, 0, 1] * e1y + vre[e, 0, 2] * e1z
        rr0y = vre[e, 0, 0] * e2x + vre[e, 0, 1] * e2y + vre[e, 0, 2] * e2z
        rr1x = vre[e, 1, 0] * e1x + vre[e, 1, 1] * e1y + vre[e, 1, 2] * e1z
        rr1y = vre[e, 1, 0] * e2x + vre[e, 1, 1] * e2y + vre[e, 1, 2] * e2z
        rr2x = vre[e, 2, 0] * e1x + vre[e, 2, 1] * e1y + vre[e, 2, 2] * e1z
        rr2y = vre[e, 2, 0] * e2x + vre[e, 2, 1] * e2y + vre[e, 2, 2] * e2z
        rr3x = vre[e, 3, 0] * e1x + vre[e, 3, 1] * e1y + vre[e, 3, 2] * e1z
        rr3y = vre[e, 3, 0] * e2x + vre[e, 3, 1] * e2y + vre[e, 3, 2] * e2z
        
        r13x = rr0x - rr2x; r13y = rr0y - rr2y
        r24x = rr1x - rr3x; r24y = rr1y - rr3y
        rhix = rr0x - rr1x + rr2x - rr3x; rhiy = rr0y - rr1y + rr2y - rr3y
        rtix = rr0x + rr1x + rr2x + rr3x; rtiy = rr0y + rr1y + rr2y + rr3y
        
        vdef3[e] = (y24n[e] * v13y - y13n[e] * v24y - x24n[e] * v13x + x13n[e] * v24x)
        
        j1 = (mx23 * my13 - mx13 * my23) * _PG
        j2 = -(mx13 * my34 - mx34 * my13) * _PG
        j0 = 0.25 * a
        j00 = abs(j0 + j2 - j1)
        j01 = abs(j0 + j2 + j1)
        j02 = abs(j0 - j2 + j1)
        j03 = abs(j0 - j2 - j1)
        cdet[e, 0] = j00; cdet[e, 1] = j01; cdet[e, 2] = j02; cdet[e, 3] = j03
        
        j1h = (my23 - my34) * _PG; j2h = -(my23 + my34) * _PG
        hx0 = j1h / j00; hx1 = j2h / j01; hx2 = -j1h / j02; hx3 = -j2h / j03
        j1y = (mx34 - mx23) * _PG; j2y = (mx34 + mx23) * _PG
        hy0 = j1y / j00; hy1 = j2y / j01; hy2 = -j1y / j02; hy3 = -j2y / j03
        
        gama1 = -mx13 * y24 + my13 * x24
        gama2 = mx13 * y13 - my13 * x13
        gama1n = gama1 * area_i
        gama2n = gama2 * area_i
        
        lm1 = abs(cx1*cy3 - cy1*cx3)
        lm2 = abs(cx0*cy2 - cy0*cx2)
        lm = lm1 if lm1 > lm2 else lm2
        lc[e] = area[e] / (lm if lm > EM20 else EM20)
        
        # Gauss loop
        hx_arr = (hx0, hx1, hx2, hx3)
        hy_arr = (hy0, hy1, hy2, hy3)
        for ng in range(4):
            bm[e, ng, 0] = y24n[e] + hx_arr[ng] * gama2n
            bm[e, ng, 1] = -y13n[e] + hx_arr[ng] * gama1n
            bm[e, ng, 2] = -x24n[e] + hy_arr[ng] * gama2n
            bm[e, ng, 3] = x13n[e] + hy_arr[ng] * gama1n
            bm[e, ng, 4] = hx_arr[ng]
            bm[e, ng, 5] = hy_arr[ng]
            bm[e, ng, 6] = y24n[e]
            bm[e, ng, 7] = -y13n[e]
            
            ksi = _VPG[ng, 0]; eta = _VPG[ng, 1]
            
            c0 =  0.25 * (y24 + ksi * y13);       c1 = -0.25 * (y13 + eta * y24)
            c2 =  0.25 * (x24 + ksi * x13);       c3 = -0.25 * (x13 + eta * x24)
            c4 = -0.25 * (x24 * y13 + x13 * y24); c5 =  0.25 * (y24**2 + y13**2)
            c6 =  0.25 * (x24**2 + x13**2);       c7 = -c4
            
            for m in range(4):
                bc[e, ng, m] = c0 * _VKSI[ng, m] + c1 * _VETA[ng, m]
                bc[e, ng, 4 + m] = c2 * _VKSI[ng, m] + c3 * _VETA[ng, m]
                bc[e, ng, 8 + m] = c4 * _VKSI[ng, m] + c5 * _VETA[ng, m]
                bc[e, ng, 12 + m] = c6 * _VKSI[ng, m] + c7 * _VETA[ng, m]
                bc[e, ng, 16 + m] = 0.5 * _VKSI[ng, m]
                bc[e, ng, 20 + m] = 0.5 * _VETA[ng, m]
                
            vdef[e, ng, 0] = bm[e, ng, 0]*v13x + bm[e, ng, 1]*v24x + bm[e, ng, 4]*vhix
            vdef[e, ng, 1] = bm[e, ng, 2]*v13y + bm[e, ng, 3]*v24y + bm[e, ng, 5]*vhiy
            vdef[e, ng, 3] = bm[e, ng, 0]*v13z + bm[e, ng, 1]*v24z + bm[e, ng, 4]*vhiz + bm[e, ng, 6]*r24y + bm[e, ng, 7]*r13y
            vdef[e, ng, 4] = bm[e, ng, 2]*v13z + bm[e, ng, 3]*v24z + bm[e, ng, 5]*vhiz - bm[e, ng, 6]*r24x - bm[e, ng, 7]*r13x
            vdef[e, ng, 2] = vdef3[e]
            
            kxx = 0.0; kyy = 0.0; kxy = 0.0
            for m in range(4):
                kxx += bc[e, ng, m] * rr2y if m == 2 else 0.0 # simplified indexing of R, wait, we must unroll bc * rxyz
            
            # actually we can explicitly write bc @ rxyz:
            # bc is shape 24, R is [4 nodes x 2 comps x,y]. 
            # In flat_gp: bc @ rxyz
            # R is passed as [rx1, rx2, rx3, rx4, ry1, ry2, ry3, ry4] in cdefo.F?
            # No, in python `vd[5] = np.einsum("k,k->", bc[:4], rxyz[:, 1])`
            # Wait, `rxyz` is shape (4, 2).
            # vd[5] = bc[:4] @ rxyz[:, 1]
            # vd[6] = bc[4:8] @ (-rxyz[:, 0])
            # vd[7] = bc[8:12] @ rxyz[:, 1] + bc[12:16] @ (-rxyz[:, 0]) + bc[16:20] @ rxyz[:, 0] + bc[20:24] @ rxyz[:, 1]
            kxx = bc[e, ng, 0]*r13y + bc[e, ng, 1]*r24y + bc[e, ng, 2]*rhiy + bc[e, ng, 3]*rtiy
            kyy = bc[e, ng, 4]*(-r13x) + bc[e, ng, 5]*(-r24x) + bc[e, ng, 6]*(-rhix) + bc[e, ng, 7]*(-rtix)
            kxy = (bc[e, ng, 8]*r13y + bc[e, ng, 9]*r24y + bc[e, ng, 10]*rhiy + bc[e, ng, 11]*rtiy +
                   bc[e, ng, 12]*(-r13x) + bc[e, ng, 13]*(-r24x) + bc[e, ng, 14]*(-rhix) + bc[e, ng, 15]*(-rtix) +
                   bc[e, ng, 16]*r13x + bc[e, ng, 17]*r24x + bc[e, ng, 18]*rhix + bc[e, ng, 19]*rtix +
                   bc[e, ng, 20]*r13y + bc[e, ng, 21]*r24y + bc[e, ng, 22]*rhiy + bc[e, ng, 23]*rtiy)
            
            vdef[e, ng, 5] = kxx
            vdef[e, ng, 6] = kyy
            vdef[e, ng, 7] = kxy
            
    return E, area, lc, vdef3, cdet, vdef, bm, bc, x13n, x24n, y13n, y24n

@njit(cache=True)
def qbat_post_flat(n, E, off, thick, volg, forpg, mompg, for_mean, cdet, bm, bc, x13n, x24n, y13n, y24n):
    fg = np.zeros((n, 4, 3))
    mg = np.zeros((n, 4, 3))
    for e in range(n):
        if off[e] <= 0.0: continue
        
        vf_00 = 0.0; vf_01 = 0.0; vf_02 = 0.0
        vf_10 = 0.0; vf_11 = 0.0; vf_12 = 0.0
        vf_20 = 0.0; vf_21 = 0.0; vf_22 = 0.0
        vf_30 = 0.0; vf_31 = 0.0; vf_32 = 0.0
        
        vm_00 = 0.0; vm_01 = 0.0
        vm_10 = 0.0; vm_11 = 0.0
        vm_20 = 0.0; vm_21 = 0.0
        vm_30 = 0.0; vm_31 = 0.0
        
        for ng in range(4):
            c = cdet[e, ng]
            npg0 = forpg[e, ng, 0] * thick[e]; npg1 = forpg[e, ng, 1] * thick[e]; npg2 = forpg[e, ng, 2] * thick[e]
            mpg0 = mompg[e, ng, 0] * thick[e]**2; mpg1 = mompg[e, ng, 1] * thick[e]**2; mpg2 = mompg[e, ng, 2] * thick[e]**2
            q0 = forpg[e, ng, 4] * thick[e]; q1 = forpg[e, ng, 3] * thick[e]
            
            f13x = c * (bm[e, ng, 0] * npg0 + bm[e, ng, 2] * npg2)
            f24x = c * (bm[e, ng, 1] * npg0 + bm[e, ng, 3] * npg2)
            fhix = c * bm[e, ng, 4] * npg0
            f13y = c * (bm[e, ng, 2] * npg1 + bm[e, ng, 0] * npg2)
            f24y = c * (bm[e, ng, 3] * npg1 + bm[e, ng, 1] * npg2)
            fhiy = c * bm[e, ng, 5] * npg1
            
            vf_00 += f13x + fhix; vf_10 -= f13x - fhix
            vf_20 += f24x - fhix; vf_30 -= f24x + fhix
            vf_01 += f13y + fhiy; vf_11 -= f13y - fhiy
            vf_21 += f24y - fhiy; vf_31 -= f24y + fhiy
            
            f13z = c * (bm[e, ng, 0] * q0 + bm[e, ng, 2] * q1)
            f24z = c * (bm[e, ng, 1] * q0 + bm[e, ng, 3] * q1)
            fhiz = c * (bm[e, ng, 4] * q0 + bm[e, ng, 5] * q1)
            
            vf_02 += f13z + fhiz; vf_12 -= f13z - fhiz
            vf_22 += f24z - fhiz; vf_32 -= f24z + fhiz
            
            # M
            m00 = c * (bc[e, ng, 0]*mpg0 - bc[e, ng, 4]*mpg2 + bc[e, ng, 8]*mpg2 - bc[e, ng, 12]*mpg1 + bc[e, ng, 16]*mpg2 + bc[e, ng, 20]*mpg1)
            m01 = c * (bc[e, ng, 0]*mpg2 - bc[e, ng, 4]*mpg1 + bc[e, ng, 8]*mpg0 - bc[e, ng, 12]*mpg2 + bc[e, ng, 16]*mpg0 + bc[e, ng, 20]*mpg2)
            m10 = c * (bc[e, ng, 1]*mpg0 - bc[e, ng, 5]*mpg2 + bc[e, ng, 9]*mpg2 - bc[e, ng, 13]*mpg1 + bc[e, ng, 17]*mpg2 + bc[e, ng, 21]*mpg1)
            m11 = c * (bc[e, ng, 1]*mpg2 - bc[e, ng, 5]*mpg1 + bc[e, ng, 9]*mpg0 - bc[e, ng, 13]*mpg2 + bc[e, ng, 17]*mpg0 + bc[e, ng, 21]*mpg2)
            m20 = c * (bc[e, ng, 2]*mpg0 - bc[e, ng, 6]*mpg2 + bc[e, ng, 10]*mpg2 - bc[e, ng, 14]*mpg1 + bc[e, ng, 18]*mpg2 + bc[e, ng, 22]*mpg1)
            m21 = c * (bc[e, ng, 2]*mpg2 - bc[e, ng, 6]*mpg1 + bc[e, ng, 10]*mpg0 - bc[e, ng, 14]*mpg2 + bc[e, ng, 18]*mpg0 + bc[e, ng, 22]*mpg2)
            m30 = c * (bc[e, ng, 3]*mpg0 - bc[e, ng, 7]*mpg2 + bc[e, ng, 11]*mpg2 - bc[e, ng, 15]*mpg1 + bc[e, ng, 19]*mpg2 + bc[e, ng, 23]*mpg1)
            m31 = c * (bc[e, ng, 3]*mpg2 - bc[e, ng, 7]*mpg1 + bc[e, ng, 11]*mpg0 - bc[e, ng, 15]*mpg2 + bc[e, ng, 19]*mpg0 + bc[e, ng, 23]*mpg2)
            
            vm_01 += m00 + m10 + m20 + m30
            vm_21 -= m00 - m10 + m20 - m30
            vm_11 += m00 - m10 - m20 + m30
            vm_31 -= m00 + m10 - m20 - m30
            
            vm_00 -= m01 + m11 + m21 + m31
            vm_20 += m01 - m11 + m21 - m31
            vm_10 -= m01 - m11 - m21 + m31
            vm_30 += m01 + m11 - m21 - m31
            
            # + qsh terms
            vm_00 -= c * bm[e, ng, 7] * q0
            vm_10 -= c * bm[e, ng, 6] * q0
            vm_20 += c * bm[e, ng, 7] * q0
            vm_30 += c * bm[e, ng, 6] * q0
            
            vm_01 += c * bm[e, ng, 7] * q1
            vm_11 += c * bm[e, ng, 6] * q1
            vm_21 -= c * bm[e, ng, 7] * q1
            vm_31 -= c * bm[e, ng, 6] * q1
            
        thoff = volg[e] * for_mean[e, 2] * off[e]
        vf_00 += -thoff * x24n[e]
        vf_10 += thoff * y24n[e]
        vf_20 += thoff * x13n[e]
        vf_30 += -thoff * y13n[e]
        
        # cbaproj
        e1x = E[e, 0, 0]; e1y = E[e, 1, 0]; e1z = E[e, 2, 0]
        e2x = E[e, 0, 1]; e2y = E[e, 1, 1]; e2z = E[e, 2, 1]
        e3x = E[e, 0, 2]; e3y = E[e, 1, 2]; e3z = E[e, 2, 2]
        
        fg[e, 0, 0] = vf_00*e1x + vf_01*e2x + vf_02*e3x
        fg[e, 0, 1] = vf_00*e1y + vf_01*e2y + vf_02*e3y
        fg[e, 0, 2] = vf_00*e1z + vf_01*e2z + vf_02*e3z
        
        fg[e, 1, 0] = vf_10*e1x + vf_11*e2x + vf_12*e3x
        fg[e, 1, 1] = vf_10*e1y + vf_11*e2y + vf_12*e3y
        fg[e, 1, 2] = vf_10*e1z + vf_11*e2z + vf_12*e3z
        
        fg[e, 2, 0] = vf_20*e1x + vf_21*e2x + vf_22*e3x
        fg[e, 2, 1] = vf_20*e1y + vf_21*e2y + vf_22*e3y
        fg[e, 2, 2] = vf_20*e1z + vf_21*e2z + vf_22*e3z
        
        fg[e, 3, 0] = vf_30*e1x + vf_31*e2x + vf_32*e3x
        fg[e, 3, 1] = vf_30*e1y + vf_31*e2y + vf_32*e3y
        fg[e, 3, 2] = vf_30*e1z + vf_31*e2z + vf_32*e3z
        
        mg[e, 0, 0] = vm_00*e1x + vm_01*e2x
        mg[e, 0, 1] = vm_00*e1y + vm_01*e2y
        mg[e, 0, 2] = vm_00*e1z + vm_01*e2z
        
        mg[e, 1, 0] = vm_10*e1x + vm_11*e2x
        mg[e, 1, 1] = vm_10*e1y + vm_11*e2y
        mg[e, 1, 2] = vm_10*e1z + vm_11*e2z
        
        mg[e, 2, 0] = vm_20*e1x + vm_21*e2x
        mg[e, 2, 1] = vm_20*e1y + vm_21*e2y
        mg[e, 2, 2] = vm_20*e1z + vm_21*e2z
        
        mg[e, 3, 0] = vm_30*e1x + vm_31*e2x
        mg[e, 3, 1] = vm_30*e1y + vm_31*e2y
        mg[e, 3, 2] = vm_30*e1z + vm_31*e2z
        
    return fg, mg




@njit(cache=True)
def qbat_pre(xe, ve, vre, off, dt, force_flat):
    n = xe.shape[0]
    
    # 1. Determine flat/warped (same logic as shell_qbat.py lines 356-376)
    d = xe - xe[:, 0:1, :]
    xl = np.empty((n, 4, 3))
    E_0 = np.empty((n, 3, 3))
    for e in range(n):
        c21x = d[e, 1, 0]
        c21y = d[e, 1, 1]
        c21z = d[e, 1, 2]
        r1 = 1.0 / max(np.sqrt(c21x**2 + c21y**2 + c21z**2), EM20)
        c21x *= r1
        c21y *= r1
        c21z *= r1
        c34x = d[e, 2, 0] - d[e, 3, 0]
        c34y = d[e, 2, 1] - d[e, 3, 1]
        c34z = d[e, 2, 2] - d[e, 3, 2]
        r2 = 1.0 / max(np.sqrt(c34x**2 + c34y**2 + c34z**2), EM20)
        c34x *= r2
        c34y *= r2
        c34z *= r2
        x = c21x + c34x
        y = c21y + c34y
        z = c21z + c34z
        n1 = 1.0 / max(np.sqrt(x**2 + y**2 + z**2), EM20)
        e1x = x * n1; e1y = y * n1; e1z = z * n1
        
        c41x = d[e, 3, 0]
        c41y = d[e, 3, 1]
        c41z = d[e, 3, 2]
        r3 = 1.0 / max(np.sqrt(c41x**2 + c41y**2 + c41z**2), EM20)
        c41x *= r3
        c41y *= r3
        c41z *= r3
        c23x = d[e, 0, 0] - d[e, 2, 0]  # note d[0]=0 so -d[2]
        c23y = d[e, 0, 1] - d[e, 2, 1]
        c23z = d[e, 0, 2] - d[e, 2, 2]
        r4 = 1.0 / max(np.sqrt(c23x**2 + c23y**2 + c23z**2), EM20)
        c23x *= r4
        c23y *= r4
        c23z *= r4
        x2 = c41x + c23x
        y2 = c41y + c23y
        z2 = c41z + c23z
        n2 = 1.0 / max(np.sqrt(x2**2 + y2**2 + z2**2), EM20)
        a2x = x2 * n2; a2y = y2 * n2; a2z = z2 * n2
        
        e3x = e1y * a2z - e1z * a2y
        e3y = e1z * a2x - e1x * a2z
        e3z = e1x * a2y - e1y * a2x
        n3 = 1.0 / max(np.sqrt(e3x**2 + e3y**2 + e3z**2), EM20)
        e3x *= n3; e3y *= n3; e3z *= n3
        
        e2x = e3y * e1z - e3z * e1y
        e2y = e3z * e1x - e3x * e1z
        e2z = e3x * e1y - e3y * e1x
        
        E_0[e, 0, 0] = e1x; E_0[e, 0, 1] = e2x; E_0[e, 0, 2] = e3x
        E_0[e, 1, 0] = e1y; E_0[e, 1, 1] = e2y; E_0[e, 1, 2] = e3y
        E_0[e, 2, 0] = e1z; E_0[e, 2, 1] = e2z; E_0[e, 2, 2] = e3z
        
        for i in range(4):
            xl[e, i, 0] = d[e, i, 0] * e1x + d[e, i, 1] * e1y + d[e, i, 2] * e1z
            xl[e, i, 1] = d[e, i, 0] * e2x + d[e, i, 1] * e2y + d[e, i, 2] * e2z
            xl[e, i, 2] = d[e, i, 0] * e3x + d[e, i, 1] * e3y + d[e, i, 2] * e3z
            
    zl1 = np.empty(n)
    l24 = np.empty(n)
    l13 = np.empty(n)
    for e in range(n):
        zl1[e] = -0.25 * (xl[e, 0, 2] + xl[e, 1, 2] + xl[e, 2, 2] + xl[e, 3, 2])
        l24[e] = np.sqrt((xl[e, 3, 0] - xl[e, 1, 0])**2 + (xl[e, 3, 1] - xl[e, 1, 1])**2)
        l13[e] = np.sqrt((xl[e, 2, 0] - xl[e, 0, 0])**2 + (xl[e, 2, 1] - xl[e, 0, 1])**2)
        
    flat_mask = np.empty(n, dtype=np.bool_)
    for e in range(n):
        v1 = 2.0 * zl1[e] / max(l24[e], EM20)
        v2 = 2.0 * zl1[e] / max(l13[e], EM20)
        if v1 > 1.0: v1 = 1.0
        if v1 < -1.0: v1 = -1.0
        if v2 > 1.0: v2 = 1.0
        if v2 < -1.0: v2 = -1.0
        warp_angle = np.arcsin(v1) - np.arcsin(v2)
        flat_mask[e] = np.abs(warp_angle) < 0.0872664626
        if force_flat[e]:
            flat_mask[e] = True
            
    # count
    num_f = 0
    for e in range(n):
        if flat_mask[e]: num_f += 1
    num_w = n - num_f
    
    i_f = np.empty(num_f, dtype=np.int64)
    i_w = np.empty(num_w, dtype=np.int64)
    idx_f = 0
    idx_w = 0
    for e in range(n):
        if flat_mask[e]:
            i_f[idx_f] = e
            idx_f += 1
        else:
            i_w[idx_w] = e
            idx_w += 1
            
    E = np.empty((n, 3, 3))
    area = np.empty(n)
    lc = np.empty(n)
    vdef3 = np.empty(n)
    cdet = np.empty(n)
    vdef = np.zeros((n, 8))
    
    # Flat
    if num_f > 0:
        E_f, area_f, lc_f, vdef3_f, cdet_f, vdef_f, bm_f, bc_f, x13n_f, x24n_f, y13n_f, y24n_f = qbat_pre_flat(xe[i_f], ve[i_f], vre[i_f], dt)
        for i in range(num_f):
            E[i_f[i]] = E_f[i]
            area[i_f[i]] = area_f[i]
            lc[i_f[i]] = lc_f[i]
            vdef3[i_f[i]] = vdef3_f[i]
            cdet[i_f[i]] = cdet_f[i]
            vdef[i_f[i]] = vdef_f[i]
    else:
        bm_f = np.empty((0, 4, 8))
        bc_f = np.empty((0, 4, 5, 2))
        x13n_f = np.empty((0, 4))
        x24n_f = np.empty((0, 4))
        y13n_f = np.empty((0, 4))
        y24n_f = np.empty((0, 4))

    # Warp
    if num_w > 0:
        E_w, area_w, lc_w, vdef3_w, cdet_w, vdef_w2, bmw_w, bmfw_w, bfw_w, bcq_w, tc_w, vqn_w, corel_w, di_w, x13n_w, x24n_w, y13n_w, y24n_w = qbat_pre_warp(xe[i_w], ve[i_w], vre[i_w], dt)
        for i in range(num_w):
            E[i_w[i]] = E_w[i]
            area[i_w[i]] = area_w[i]
            lc[i_w[i]] = lc_w[i]
            vdef3[i_w[i]] = vdef3_w[i]
            cdet[i_w[i]] = cdet_w[i]
            vdef[i_w[i]] = vdef_w2[i]
    else:
        bmw_w = np.empty((0, 4, 3, 2))
        bmfw_w = np.empty((0, 4, 3, 3))
        bfw_w = np.empty((0, 4, 2, 3))
        bcq_w = np.empty((0, 4, 5, 2))
        tc_w = np.empty((0, 2, 2))
        vqn_w = np.empty((0, 4, 9))
        corel_w = np.empty((0, 3, 4))
        di_w = np.empty((0, 6))
        x13n_w = np.empty((0, 4))
        x24n_w = np.empty((0, 4))
        y13n_w = np.empty((0, 4))
        y24n_w = np.empty((0, 4))
        
    return (E, area, lc, vdef3, cdet, vdef, i_f, i_w, bm_f, bc_f, bmw_w, bmfw_w, bfw_w, bcq_w, tc_w, vqn_w, corel_w, di_w, x13n_f, x24n_f, y13n_f, y24n_f, x13n_w, x24n_w, y13n_w, y24n_w)

@njit(cache=True)
def qbat_post(n, E, off, thick, volg, forpg, mompg, for_mean, cdet,
              i_f, i_w, bm_f, bc_f, bmw_w, bmfw_w, bfw_w, bcq_w, tc_w, vqn_w, corel_w, di_w,
              x13n_f, x24n_f, y13n_f, y24n_f,
              x13n_w, x24n_w, y13n_w, y24n_w):
    
    fg = np.zeros((n, 4, 3))
    mg = np.zeros((n, 4, 3))
    
    if len(i_f) > 0:
        fg_f, mg_f = qbat_post_flat(len(i_f), E[i_f], off[i_f], thick[i_f], volg[i_f], forpg[i_f], mompg[i_f], for_mean[i_f], cdet[i_f], bm_f, bc_f, x13n_f, x24n_f, y13n_f, y24n_f)
        for i in range(len(i_f)):
            fg[i_f[i]] = fg_f[i]
            mg[i_f[i]] = mg_f[i]
        
    if len(i_w) > 0:
        fg_w, mg_w = qbat_post_warp(len(i_w), E[i_w], off[i_w], thick[i_w], volg[i_w], forpg[i_w], mompg[i_w], for_mean[i_w], cdet[i_w], bmw_w, bmfw_w, bfw_w, bcq_w, tc_w, vqn_w, corel_w, di_w, x13n_w, x24n_w, y13n_w, y24n_w)
        for i in range(len(i_w)):
            fg[i_w[i]] = fg_w[i]
            mg[i_w[i]] = mg_w[i]
        
    return fg, mg
