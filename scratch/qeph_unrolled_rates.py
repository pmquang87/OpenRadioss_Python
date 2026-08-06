import numpy as np
from pyradioss.common.constants import EM20

def _sym3_inv(d):
    """Inverse of the packed symmetric 3x3 [d1,d2,d3,d4,d5,d6] =
    [[1,4,5],[4,2,6],[5,6,3]] — czcorp5.F lines 269-284 (A3INVDP)."""
    n = len(d)
    di = np.empty((n, 6))
    for e in range(n):
        d0 = d[e, 0]
        d1 = d[e, 1]
        d2 = d[e, 2]
        d3 = d[e, 3]
        d4 = d[e, 4]
        d5 = d[e, 5]
        
        abc = d0 * d1 * d2
        xxyz2 = d0 * d5 * d5
        yyxz2 = d1 * d4 * d4
        zzxy2 = d2 * d3 * d3
        
        deta = abs(abc + 2.0 * d3 * d4 * d5 - xxyz2 - yyxz2 - zzxy2)
        deta_max = deta if deta > EM20 else EM20
        deta_inv = 1.0 / deta_max
        
        d0_max = d0 if d0 > EM20 else EM20
        di[e, 0] = (abc - xxyz2) * deta_inv / d0_max
        
        d1_max = d1 if d1 > EM20 else EM20
        di[e, 1] = (abc - yyxz2) * deta_inv / d1_max
        
        d2_max = d2 if d2 > EM20 else EM20
        di[e, 2] = (abc - zzxy2) * deta_inv / d2_max
        
        di[e, 3] = (d4 * d5 - d3 * d2) * deta_inv
        di[e, 4] = (d5 * d3 - d4 * d1) * deta_inv
        di[e, 5] = (d3 * d4 - d5 * d0) * deta_inv
        
    return di


def _rates(G, v13, v24, vhi, rl, alive):
    """VDEF (n,8) = [exx eyy exy gxz gyz kxx kyy kxy] and VHG (n,6) —
    czdef.F lines 114-173 verbatim (with the warp additions)."""
    area = G["area"]
    a_i = G["a_i"]
    z1 = G["z1"]
    x13 = G["x13"]
    x24 = G["x24"]
    y13 = G["y13"]
    y24 = G["y24"]
    mx13 = G["mx13"]
    mx23 = G["mx23"]
    mx34 = G["mx34"]
    my13 = G["my13"]
    my23 = G["my23"]
    my34 = G["my34"]
    
    n = len(area)
    vdef = np.empty((n, 8))
    vhg = np.empty((n, 6))
    
    for e in range(n):
        if not alive[e]:
            for i in range(8):
                vdef[e, i] = 0.0
            for i in range(6):
                vhg[e, i] = 0.0
            continue
            
        a_i_e = a_i[e]
        area_e = area[e]
        z1_e = z1[e]
        x13_e = x13[e]
        x24_e = x24[e]
        y13_e = y13[e]
        y24_e = y24[e]
        mx13_e = mx13[e]
        mx23_e = mx23[e]
        mx34_e = mx34[e]
        my13_e = my13[e]
        my23_e = my23[e]
        my34_e = my34[e]
        
        rl00 = rl[e, 0, 0]
        rl01 = rl[e, 0, 1]
        rl10 = rl[e, 1, 0]
        rl11 = rl[e, 1, 1]
        rl20 = rl[e, 2, 0]
        rl21 = rl[e, 2, 1]
        rl30 = rl[e, 3, 0]
        rl31 = rl[e, 3, 1]
        
        r13_0 = (rl00 - rl20) * a_i_e
        r13_1 = (rl01 - rl21) * a_i_e
        r24_0 = (rl10 - rl30) * a_i_e
        r24_1 = (rl11 - rl31) * a_i_e
        
        rsom_0 = (rl00 + rl10 + rl20 + rl30) * a_i_e
        rsom_1 = (rl01 + rl11 + rl21 + rl31) * a_i_e
        
        rhi_0 = 0.25 * (rl00 - rl10 + rl20 - rl30)
        rhi_1 = 0.25 * (rl01 - rl11 + rl21 - rl31)
        
        v13_0 = v13[e, 0]
        v13_1 = v13[e, 1]
        v13_2 = v13[e, 2]
        
        v24_0 = v24[e, 0]
        v24_1 = v24[e, 1]
        v24_2 = v24[e, 2]
        
        vhi_0 = vhi[e, 0]
        vhi_1 = vhi[e, 1]
        vhi_2 = vhi[e, 2]
        
        # membrane
        vdef0 = y24_e * v13_0 - y13_e * v24_0
        vdef1 = -x24_e * v13_1 + x13_e * v24_1
        bxv2 = y24_e * v13_1 - y13_e * v24_1
        byv1 = -x24_e * v13_0 + x13_e * v24_0
        vdef2 = bxv2 + byv1
        
        # flexion
        vdef5 = y24_e * r13_1 - y13_e * r24_1
        vdef6 = x24_e * r13_0 - x13_e * r24_0
        bxr1 = y13_e * r24_0 - y24_e * r13_0
        byr2 = -x24_e * r13_1 + x13_e * r24_1
        vdef7 = bxr1 + byr2
        
        # transverse shear
        bcxy = 0.25 * area_e
        bcx = v13_2 - my13_e * r13_0 + mx13_e * r13_1
        bcy = v24_2 + my13_e * r24_0 - mx13_e * r24_1
        vdef3 = y24_e * bcx - y13_e * bcy + bcxy * rsom_1
        vdef4 = x13_e * bcy - x24_e * bcx - bcxy * rsom_0
        
        # hourglass rates
        vhg0 = vhi_0 - mx13_e * vdef0 - my13_e * byv1
        vhg1 = vhi_1 - mx13_e * bxv2 - my13_e * vdef1
        vhg2 = rhi_1 - mx13_e * vdef5 - my13_e * byr2
        vhg3 = -rhi_0 - mx13_e * bxr1 - my13_e * vdef6
        
        term4_1 = my13_e * rsom_0 - my23_e * (r13_0 + r24_0)
        term4_2 = mx23_e * (r13_1 + r24_1) - mx13_e * rsom_1
        vhg4 = (vhi_2 * 4.0 - (term4_1 + term4_2) * area_e) * 4.0
        
        term5_1 = my13_e * rsom_0 - my34_e * (r13_0 - r24_0)
        term5_2 = mx34_e * (r13_1 - r24_1) - mx13_e * rsom_1
        vhg5 = (vhi_2 * 4.0 - (term5_1 + term5_2) * area_e) * 4.0
        
        vhg0 += (y24_e * v13_2 - y13_e * v24_2) * z1_e
        vhg1 += (-x24_e * v13_2 + x13_e * v24_2) * z1_e
        
        # warp additions to the curvatures
        deta1 = z1_e * 4.0 * a_i_e
        vdef5 += (x13_e * v13_0 - x24_e * v24_0) * deta1
        vdef6 += (y13_e * v13_1 - y24_e * v24_1) * deta1
        vdef7 += (x13_e * v13_1 - x24_e * v24_1 + y13_e * v13_0 - y24_e * v24_0) * deta1
        
        vdef[e, 0] = vdef0
        vdef[e, 1] = vdef1
        vdef[e, 2] = vdef2
        vdef[e, 3] = vdef3
        vdef[e, 4] = vdef4
        vdef[e, 5] = vdef5
        vdef[e, 6] = vdef6
        vdef[e, 7] = vdef7
        
        vhg[e, 0] = vhg0
        vhg[e, 1] = vhg1
        vhg[e, 2] = vhg2
        vhg[e, 3] = vhg3
        vhg[e, 4] = vhg4
        vhg[e, 5] = vhg5
        
    return vdef, vhg
