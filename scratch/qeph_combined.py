
import numpy as np
from numba import njit

@njit(cache=True)
def qeph_pre(x_conn, v_conn, vr_conn, dt, st_npt1, alive):
    n = len(x_conn)
    # xe = x_conn
    xe = x_conn
    ve = v_conn
    vre = vr_conn
    npt1 = st_npt1
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


    n = len(area)
    
    # outputs
    v13 = np.zeros((n, 3))
    v24 = np.zeros((n, 3))
    vhi = np.zeros((n, 3))
    rl = np.zeros((n, 4, 2))
    plat = np.zeros(n, dtype=np.bool_)
    
    # Warped element projection operators
    vqn = np.zeros((n, 4, 3))
    di = np.zeros((n, 6))
    db = np.zeros((n, 4, 3))
    
    TOL_PLAT = 1.0e-8
    EM20 = 1e-20
    
    dt05 = 0.5 * dt
    dt025 = 0.25 * dt
    
    for e in range(n):
        c_area = area[e]
        c_a_i = a_i[e]
        c_z1 = z1[e]
        c_x13 = x13[e]
        c_x24 = x24[e]
        c_y13 = y13[e]
        c_y24 = y24[e]
        c_mx13 = mx13[e]
        c_my13 = my13[e]
        c_lm = lm[e]
        
        vloc = np.zeros((4, 3))
        rloc = np.zeros((4, 3))
        for node in range(4):
            for basis in range(3):
                val_v = 0.0
                val_r = 0.0
                for coord in range(3):
                    val_v += ve[e, node, coord] * E[e, coord, basis]
                    val_r += vre[e, node, coord] * E[e, coord, basis]
                vloc[node, basis] = val_v
                rloc[node, basis] = val_r
                if basis < 2:
                    rl[e, node, basis] = val_r
        
        c_v13_0 = vloc[0, 0] - vloc[2, 0]
        c_v13_1 = vloc[0, 1] - vloc[2, 1]
        c_v13_2 = vloc[0, 2] - vloc[2, 2]
        
        c_v24_0 = vloc[1, 0] - vloc[3, 0]
        c_v24_1 = vloc[1, 1] - vloc[3, 1]
        c_v24_2 = vloc[1, 2] - vloc[3, 2]
        
        c_vhi_0 = vloc[0, 0] - vloc[1, 0] + vloc[2, 0] - vloc[3, 0]
        c_vhi_1 = vloc[0, 1] - vloc[1, 1] + vloc[2, 1] - vloc[3, 1]
        c_vhi_2 = vloc[0, 2] - vloc[1, 2] + vloc[2, 2] - vloc[3, 2]
        
        if dt != 0.0:
            exz = c_y24 * c_v13_2 - c_y13 * c_v24_2
            eyz = -c_x24 * c_v13_2 + c_x13 * c_v24_2
            ddry = dt05 * exz * c_a_i
            ddrx = dt05 * eyz * c_a_i
            
            v13x = c_v13_0
            v24x = c_v24_0
            vhix = c_vhi_0
            
            den1 = c_x13 - c_x24
            if abs(den1) < 1.0e-10:
                ddrz1 = 0.0
            else:
                ddrz1 = dt025 * (c_v13_1 - c_v24_1) / den1
                
            c_v13_0 -= ddry * c_v13_2 + ddrz1 * c_v13_1
            c_v24_0 -= ddry * c_v24_2 + ddrz1 * c_v24_1
            c_vhi_0 -= ddry * c_vhi_2 + ddrz1 * c_vhi_1
            
            den2 = c_y13 + c_y24
            if abs(den2) < 1.0e-10:
                ddrz2 = 0.0
            else:
                ddrz2 = dt025 * (v13x + v24x) / den2
                
            c_v13_1 -= ddrx * c_v13_2 + ddrz2 * v13x
            c_v24_1 -= ddrx * c_v24_2 + ddrz2 * v24x
            c_vhi_1 -= ddrx * c_vhi_2 + ddrz2 * vhix
            
        is_plat = (c_z1 * c_z1 < c_lm * TOL_PLAT) or npt1[e]
        plat[e] = is_plat
        
        if is_plat:
            c_z1 = 0.0
        else:
            z2 = c_z1 * c_z1
            a_4 = 0.25 * c_area
            
            sz1 = c_mx13 * c_y24 - c_my13 * c_x24
            sz = z2 * l24[e]
            sl = 1.0 / np.sqrt(sz + (a_4 + sz1)**2)
            vqn_0_0 = -c_z1 * c_y24 * sl
            vqn_0_1 = c_z1 * c_x24 * sl
            vqn_0_2 = (a_4 + sz1) * sl
            
            sl = 1.0 / np.sqrt(sz + (a_4 - sz1)**2)
            vqn_2_0 = c_z1 * c_y24 * sl
            vqn_2_1 = -c_z1 * c_x24 * sl
            vqn_2_2 = (a_4 - sz1) * sl
            
            sz1 = c_mx13 * c_y13 - c_my13 * c_x13
            sz = z2 * l13[e]
            sl = 1.0 / np.sqrt(sz + (a_4 + sz1)**2)
            vqn_1_0 = -c_z1 * c_y13 * sl
            vqn_1_1 = c_z1 * c_x13 * sl
            vqn_1_2 = (a_4 + sz1) * sl
            
            sl = 1.0 / np.sqrt(sz + (a_4 - sz1)**2)
            vqn_3_0 = c_z1 * c_y13 * sl
            vqn_3_1 = -c_z1 * c_x13 * sl
            vqn_3_2 = (a_4 - sz1) * sl
            
            vqn[e, 0, 0] = vqn_0_0
            vqn[e, 0, 1] = vqn_0_1
            vqn[e, 0, 2] = vqn_0_2
            vqn[e, 1, 0] = vqn_1_0
            vqn[e, 1, 1] = vqn_1_1
            vqn[e, 1, 2] = vqn_1_2
            vqn[e, 2, 0] = vqn_2_0
            vqn[e, 2, 1] = vqn_2_1
            vqn[e, 2, 2] = vqn_2_2
            vqn[e, 3, 0] = vqn_3_0
            vqn[e, 3, 1] = vqn_3_1
            vqn[e, 3, 2] = vqn_3_2
            
            rr_0_0 = rl[e, 0, 0]; rr_0_1 = rl[e, 0, 1]; rr_0_2 = rloc[0, 2]
            rr_1_0 = rl[e, 1, 0]; rr_1_1 = rl[e, 1, 1]; rr_1_2 = rloc[1, 2]
            rr_2_0 = rl[e, 2, 0]; rr_2_1 = rl[e, 2, 1]; rr_2_2 = rloc[2, 2]
            rr_3_0 = rl[e, 3, 0]; rr_3_1 = rl[e, 3, 1]; rr_3_2 = rloc[3, 2]
            
            sum_rr_0 = rr_0_0 + rr_1_0 + rr_2_0 + rr_3_0
            sum_rr_1 = rr_0_1 + rr_1_1 + rr_2_1 + rr_3_1
            sum_rr_2 = rr_0_2 + rr_1_2 + rr_2_2 + rr_3_2
            
            ar_0 = -c_z1 * c_vhi_1 + c_y13 * c_v13_2 + c_y24 * c_v24_2 + c_my13 * c_vhi_2 + sum_rr_0
            ar_1 = c_z1 * c_vhi_0 - c_x13 * c_v13_2 - c_x24 * c_v24_2 - c_mx13 * c_vhi_2 + sum_rr_1
            ar_2 = c_x13 * c_v13_1 + c_x24 * c_v24_1 + c_mx13 * c_vhi_1 - c_y13 * c_v13_0 - c_y24 * c_v24_0 - c_my13 * c_vhi_0 + sum_rr_2
                    
            ad_0 = vqn_0_0 * rr_0_0 + vqn_0_1 * rr_0_1 + vqn_0_2 * rr_0_2
            ad_1 = vqn_1_0 * rr_1_0 + vqn_1_1 * rr_1_1 + vqn_1_2 * rr_1_2
            ad_2 = vqn_2_0 * rr_2_0 + vqn_2_1 * rr_2_1 + vqn_2_2 * rr_2_2
            ad_3 = vqn_3_0 * rr_3_0 + vqn_3_1 * rr_3_1 + vqn_3_2 * rr_3_2
            
            cx0 = corx[e, 0]; cx1 = corx[e, 1]; cx2 = corx[e, 2]; cx3 = corx[e, 3]
            cy0 = cory[e, 0]; cy1 = cory[e, 1]; cy2 = cory[e, 2]; cy3 = cory[e, 3]
            
            xx = cx0**2 + cx1**2 + cx2**2 + cx3**2
            yy = cy0**2 + cy1**2 + cy2**2 + cy3**2
            xy = cx0*cy0 + cx1*cy1 + cx2*cy2 + cx3*cy3
            
            xz = (cx0 - cx1 + cx2 - cx3) * c_z1
            yz = (cy0 - cy1 + cy2 - cy3) * c_z1
            zz = 4.0 * z2
            
            btb_0_0 = vqn_0_0*vqn_0_0 + vqn_1_0*vqn_1_0 + vqn_2_0*vqn_2_0 + vqn_3_0*vqn_3_0
            btb_1_1 = vqn_0_1*vqn_0_1 + vqn_1_1*vqn_1_1 + vqn_2_1*vqn_2_1 + vqn_3_1*vqn_3_1
            btb_2_2 = vqn_0_2*vqn_0_2 + vqn_1_2*vqn_1_2 + vqn_2_2*vqn_2_2 + vqn_3_2*vqn_3_2
            btb_0_1 = vqn_0_0*vqn_0_1 + vqn_1_0*vqn_1_1 + vqn_2_0*vqn_2_1 + vqn_3_0*vqn_3_1
            btb_0_2 = vqn_0_0*vqn_0_2 + vqn_1_0*vqn_1_2 + vqn_2_0*vqn_2_2 + vqn_3_0*vqn_3_2
            btb_1_2 = vqn_0_1*vqn_0_2 + vqn_1_1*vqn_1_2 + vqn_2_1*vqn_2_2 + vqn_3_1*vqn_3_2
            
            d_0 = yy + zz + 4.0 - btb_0_0
            d_1 = xx + zz + 4.0 - btb_1_1
            d_2 = xx + yy + 4.0 - btb_2_2
            d_3 = -xy - btb_0_1
            d_4 = -xz - btb_0_2
            d_5 = -yz - btb_1_2
            
            abc = d_0 * d_1 * d_2
            xxyz2 = d_0 * d_5**2
            yyxz2 = d_1 * d_4**2
            zzxy2 = d_2 * d_3**2
            deta = abs(abc + 2.0 * d_3 * d_4 * d_5 - xxyz2 - yyxz2 - zzxy2)
            deta = 1.0 / max(deta, EM20)
            
            di_w_0 = (abc - xxyz2) * deta / max(d_0, EM20)
            di_w_1 = (abc - yyxz2) * deta / max(d_1, EM20)
            di_w_2 = (abc - zzxy2) * deta / max(d_2, EM20)
            di_w_3 = (d_4 * d_5 - d_3 * d_2) * deta
            di_w_4 = (d_5 * d_3 - d_4 * d_1) * deta
            di_w_5 = (d_3 * d_4 - d_5 * d_0) * deta
            
            di[e, 0] = di_w_0
            di[e, 1] = di_w_1
            di[e, 2] = di_w_2
            di[e, 3] = di_w_3
            di[e, 4] = di_w_4
            di[e, 5] = di_w_5
            
            dimat_0_0 = di_w_0; dimat_1_1 = di_w_1; dimat_2_2 = di_w_2
            dimat_0_1 = di_w_3; dimat_1_0 = di_w_3
            dimat_0_2 = di_w_4; dimat_2_0 = di_w_4
            dimat_1_2 = di_w_5; dimat_2_1 = di_w_5
            
            db_0_0 = dimat_0_0*vqn_0_0 + dimat_0_1*vqn_0_1 + dimat_0_2*vqn_0_2
            db_0_1 = dimat_1_0*vqn_0_0 + dimat_1_1*vqn_0_1 + dimat_1_2*vqn_0_2
            db_0_2 = dimat_2_0*vqn_0_0 + dimat_2_1*vqn_0_1 + dimat_2_2*vqn_0_2
            
            db_1_0 = dimat_0_0*vqn_1_0 + dimat_0_1*vqn_1_1 + dimat_0_2*vqn_1_2
            db_1_1 = dimat_1_0*vqn_1_0 + dimat_1_1*vqn_1_1 + dimat_1_2*vqn_1_2
            db_1_2 = dimat_2_0*vqn_1_0 + dimat_2_1*vqn_1_1 + dimat_2_2*vqn_1_2
            
            db_2_0 = dimat_0_0*vqn_2_0 + dimat_0_1*vqn_2_1 + dimat_0_2*vqn_2_2
            db_2_1 = dimat_1_0*vqn_2_0 + dimat_1_1*vqn_2_1 + dimat_1_2*vqn_2_2
            db_2_2 = dimat_2_0*vqn_2_0 + dimat_2_1*vqn_2_1 + dimat_2_2*vqn_2_2
            
            db_3_0 = dimat_0_0*vqn_3_0 + dimat_0_1*vqn_3_1 + dimat_0_2*vqn_3_2
            db_3_1 = dimat_1_0*vqn_3_0 + dimat_1_1*vqn_3_1 + dimat_1_2*vqn_3_2
            db_3_2 = dimat_2_0*vqn_3_0 + dimat_2_1*vqn_3_1 + dimat_2_2*vqn_3_2
            
            db[e, 0, 0] = db_0_0; db[e, 0, 1] = db_0_1; db[e, 0, 2] = db_0_2
            db[e, 1, 0] = db_1_0; db[e, 1, 1] = db_1_1; db[e, 1, 2] = db_1_2
            db[e, 2, 0] = db_2_0; db[e, 2, 1] = db_2_1; db[e, 2, 2] = db_2_2
            db[e, 3, 0] = db_3_0; db[e, 3, 1] = db_3_1; db[e, 3, 2] = db_3_2
            
            dbad_0 = db_0_0*ad_0 + db_1_0*ad_1 + db_2_0*ad_2 + db_3_0*ad_3
            dbad_1 = db_0_1*ad_0 + db_1_1*ad_1 + db_2_1*ad_2 + db_3_1*ad_3
            dbad_2 = db_0_2*ad_0 + db_1_2*ad_1 + db_2_2*ad_2 + db_3_2*ad_3
            
            alr_0 = dimat_0_0*ar_0 + dimat_0_1*ar_1 + dimat_0_2*ar_2 - dbad_0
            alr_1 = dimat_1_0*ar_0 + dimat_1_1*ar_1 + dimat_1_2*ar_2 - dbad_1
            alr_2 = dimat_2_0*ar_0 + dimat_2_1*ar_1 + dimat_2_2*ar_2 - dbad_2
            
            vqn_dbad_0 = vqn_0_0*dbad_0 + vqn_0_1*dbad_1 + vqn_0_2*dbad_2
            vqn_dbad_1 = vqn_1_0*dbad_0 + vqn_1_1*dbad_1 + vqn_1_2*dbad_2
            vqn_dbad_2 = vqn_2_0*dbad_0 + vqn_2_1*dbad_1 + vqn_2_2*dbad_2
            vqn_dbad_3 = vqn_3_0*dbad_0 + vqn_3_1*dbad_1 + vqn_3_2*dbad_2
            
            db_ar_0 = db_0_0*ar_0 + db_0_1*ar_1 + db_0_2*ar_2
            db_ar_1 = db_1_0*ar_0 + db_1_1*ar_1 + db_1_2*ar_2
            db_ar_2 = db_2_0*ar_0 + db_2_1*ar_1 + db_2_2*ar_2
            db_ar_3 = db_3_0*ar_0 + db_3_1*ar_1 + db_3_2*ar_2
            
            ald_0 = ad_0 + vqn_dbad_0 - db_ar_0
            ald_1 = ad_1 + vqn_dbad_1 - db_ar_1
            ald_2 = ad_2 + vqn_dbad_2 - db_ar_2
            ald_3 = ad_3 + vqn_dbad_3 - db_ar_3
            
            c1 = 2.0 * alr_2
            dv13_0 = c1 * c_y13
            dv24_0 = c1 * c_y24
            dvhi_0 = 4.0 * (alr_2 * c_my13 - c_z1 * alr_1)
            
            dv13_1 = -c1 * c_x13
            dv24_1 = -c1 * c_x24
            dvhi_1 = -4.0 * (alr_2 * c_mx13 - c_z1 * alr_0)
            
            dv13_2 = -2.0 * (c_y13 * alr_0 - c_x13 * alr_1)
            dv24_2 = -2.0 * (c_y24 * alr_0 - c_x24 * alr_1)
            dvhi_2 = 4.0 * (c_mx13 * alr_1 - c_my13 * alr_0)
            
            c_v13_0 += dv13_0
            c_v24_0 += dv24_0
            c_vhi_0 += dvhi_0
            c_v13_1 += dv13_1
            c_v24_1 += dv24_1
            c_vhi_1 += dvhi_1
            c_v13_2 += dv13_2
            c_v24_2 += dv24_2
            c_vhi_2 += dvhi_2
            
            rl[e, 0, 0] = rr_0_0 - alr_0 - vqn_0_0 * ald_0
            rl[e, 0, 1] = rr_0_1 - alr_1 - vqn_0_1 * ald_0
            rl[e, 1, 0] = rr_1_0 - alr_0 - vqn_1_0 * ald_1
            rl[e, 1, 1] = rr_1_1 - alr_1 - vqn_1_1 * ald_1
            rl[e, 2, 0] = rr_2_0 - alr_0 - vqn_2_0 * ald_2
            rl[e, 2, 1] = rr_2_1 - alr_1 - vqn_2_1 * ald_2
            rl[e, 3, 0] = rr_3_0 - alr_0 - vqn_3_0 * ald_3
            rl[e, 3, 1] = rr_3_1 - alr_1 - vqn_3_1 * ald_3
            
        v13[e, 0] = c_v13_0 * c_a_i
        v13[e, 1] = c_v13_1 * c_a_i
        v13[e, 2] = c_v13_2 * c_a_i
        
        v24[e, 0] = c_v24_0 * c_a_i
        v24[e, 1] = c_v24_1 * c_a_i
        v24[e, 2] = c_v24_2 * c_a_i
        
        vhi[e, 0] = c_vhi_0 * 0.25
        vhi[e, 1] = c_vhi_1 * 0.25
        vhi[e, 2] = c_vhi_2 * 0.25


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
        

    return vdef, vhg, plat, vqn, di, db, E, area, a_i, z1, corx, cory, x13, x24, y13, y24, mx13, mx23, mx34, my13, my23, my34, l13, l24, ll, lm

@njit(cache=True)
def qeph_post(thick, Nres, Mres, qres, st_amu, st_cspd, st_yld, st_fmat, vhg, dt, alive, plat, vqn, di, db, E, area, a_i, z1, corx, cory, x13, x24, y13, y24, mx13, mx23, mx34, my13, my23, my34, l13, l24, ll, lm):
    n = len(thick)
    n = len(x13)
    VF = np.zeros((n, 3, 4))
    VM = np.zeros((n, 2, 4))
    
    for e in range(n):
        s1geo_e = my34[e] * mx23[e] - my23[e] * mx34[e]
        
        qx_e = qres[e, 0]
        qy_e = qres[e, 1]
        
        N1_e = Nres[e, 0]
        N2_e = Nres[e, 1]
        N3_e = Nres[e, 2]
        
        M1_e = Mres[e, 0]
        M2_e = Mres[e, 1]
        M3_e = Mres[e, 2]
        
        vf_0_0 = y24[e] * N1_e - x24[e] * N3_e
        vf_1_0 = -x24[e] * N2_e + y24[e] * N3_e
        vf_2_0 = -x24[e] * qy_e + y24[e] * qx_e
        
        VF[e, 0, 0] = vf_0_0
        VF[e, 1, 0] = vf_1_0
        VF[e, 2, 0] = vf_2_0
        
        VM[e, 0, 0] = x24[e] * M2_e - y24[e] * M3_e - my13[e] * vf_2_0
        VM[e, 1, 0] = y24[e] * M1_e - x24[e] * M3_e + mx13[e] * vf_2_0
        
        VM[e, 0, 2] = -s1geo_e * qy_e
        VM[e, 1, 2] = s1geo_e * qx_e
        
        vf_0_1 = -y13[e] * N1_e + x13[e] * N3_e
        vf_1_1 = x13[e] * N2_e - y13[e] * N3_e
        vf_2_1 = x13[e] * qy_e - y13[e] * qx_e
        
        VF[e, 0, 1] = vf_0_1
        VF[e, 1, 1] = vf_1_1
        VF[e, 2, 1] = vf_2_1
        
        VM[e, 0, 1] = -x13[e] * M2_e + y13[e] * M3_e + my13[e] * vf_2_1
        VM[e, 1, 1] = -y13[e] * M1_e + x13[e] * M3_e - mx13[e] * vf_2_1
        
        VM[e, 0, 3] = VM[e, 0, 2]
        VM[e, 1, 3] = VM[e, 1, 2]
        
        c2_e = z1[e] * 4.0 * a_i[e]
        
        VF[e, 0, 0] += c2_e * (x13[e] * M1_e + y13[e] * M3_e)
        VF[e, 1, 0] += c2_e * (y13[e] * M2_e + x13[e] * M3_e)
        
        VF[e, 0, 1] -= c2_e * (x24[e] * M1_e + y24[e] * M3_e)
        VF[e, 1, 1] -= c2_e * (y24[e] * M2_e + x24[e] * M3_e)


@njit(cache=True)
    n = len(z1)
    FL = np.empty((n, 3, 4))
    ML3 = np.zeros((n, 3, 4))
    
    fg = np.zeros((n, 4, 3))
    mg = np.zeros((n, 4, 3))
    
    hpat0, hpat1, hpat2, hpat3 = 1.0, -1.0, 1.0, -1.0
    
    for e in range(n):
        for c in range(3):
            FL[e, c, 0] = VF[e, c, 0] + VF[e, c, 2]
            FL[e, c, 1] = VF[e, c, 1] + VF[e, c, 3]
            FL[e, c, 2] = -VF[e, c, 0] + VF[e, c, 2]
            FL[e, c, 3] = -VF[e, c, 1] + VF[e, c, 3]
            
        for c in range(2):
            ML3[e, c, 0] = VM[e, c, 0] + VM[e, c, 2]
            ML3[e, c, 1] = VM[e, c, 1] + VM[e, c, 3]
            ML3[e, c, 2] = -VM[e, c, 0] + VM[e, c, 2]
            ML3[e, c, 3] = -VM[e, c, 1] + VM[e, c, 3]
            
        if not plat[e]:
            ar0 = (-z1[e] * (FL[e, 1, 0] * hpat0 + FL[e, 1, 1] * hpat1 + FL[e, 1, 2] * hpat2 + FL[e, 1, 3] * hpat3)
                   + (cory[e, 0] * FL[e, 2, 0] + cory[e, 1] * FL[e, 2, 1] + cory[e, 2] * FL[e, 2, 2] + cory[e, 3] * FL[e, 2, 3])
                   + (ML3[e, 0, 0] + ML3[e, 0, 1] + ML3[e, 0, 2] + ML3[e, 0, 3]))
                   
            ar1 = (z1[e] * (FL[e, 0, 0] * hpat0 + FL[e, 0, 1] * hpat1 + FL[e, 0, 2] * hpat2 + FL[e, 0, 3] * hpat3)
                   - (corx[e, 0] * FL[e, 2, 0] + corx[e, 1] * FL[e, 2, 1] + corx[e, 2] * FL[e, 2, 2] + corx[e, 3] * FL[e, 2, 3])
                   + (ML3[e, 1, 0] + ML3[e, 1, 1] + ML3[e, 1, 2] + ML3[e, 1, 3]))
                   
            ar2 = ((corx[e, 0] * FL[e, 1, 0] + corx[e, 1] * FL[e, 1, 1] + corx[e, 2] * FL[e, 1, 2] + corx[e, 3] * FL[e, 1, 3])
                   - (cory[e, 0] * FL[e, 0, 0] + cory[e, 1] * FL[e, 0, 1] + cory[e, 2] * FL[e, 0, 2] + cory[e, 3] * FL[e, 0, 3]))
                   
            ad0 = vqn[e, 0, 0] * ML3[e, 0, 0] + vqn[e, 0, 1] * ML3[e, 1, 0] + vqn[e, 0, 2] * ML3[e, 2, 0]
            ad1 = vqn[e, 1, 0] * ML3[e, 0, 1] + vqn[e, 1, 1] * ML3[e, 1, 1] + vqn[e, 1, 2] * ML3[e, 2, 1]
            ad2 = vqn[e, 2, 0] * ML3[e, 0, 2] + vqn[e, 2, 1] * ML3[e, 1, 2] + vqn[e, 2, 2] * ML3[e, 2, 2]
            ad3 = vqn[e, 3, 0] * ML3[e, 0, 3] + vqn[e, 3, 1] * ML3[e, 1, 3] + vqn[e, 3, 2] * ML3[e, 2, 3]
            
            dbad0 = db[e, 0, 0] * ad0 + db[e, 1, 0] * ad1 + db[e, 2, 0] * ad2 + db[e, 3, 0] * ad3
            dbad1 = db[e, 0, 1] * ad0 + db[e, 1, 1] * ad1 + db[e, 2, 1] * ad2 + db[e, 3, 1] * ad3
            dbad2 = db[e, 0, 2] * ad0 + db[e, 1, 2] * ad1 + db[e, 2, 2] * ad2 + db[e, 3, 2] * ad3
            
            dimat_00 = di[e, 0]; dimat_11 = di[e, 1]; dimat_22 = di[e, 2]
            dimat_01 = di[e, 3]; dimat_10 = di[e, 3]
            dimat_02 = di[e, 4]; dimat_20 = di[e, 4]
            dimat_12 = di[e, 5]; dimat_21 = di[e, 5]
            
            alr0 = dimat_00 * ar0 + dimat_01 * ar1 + dimat_02 * ar2 - dbad0
            alr1 = dimat_10 * ar0 + dimat_11 * ar1 + dimat_12 * ar2 - dbad1
            alr2 = dimat_20 * ar0 + dimat_21 * ar1 + dimat_22 * ar2 - dbad2
            
            vqn_dbad0 = vqn[e, 0, 0] * dbad0 + vqn[e, 0, 1] * dbad1 + vqn[e, 0, 2] * dbad2
            vqn_dbad1 = vqn[e, 1, 0] * dbad0 + vqn[e, 1, 1] * dbad1 + vqn[e, 1, 2] * dbad2
            vqn_dbad2 = vqn[e, 2, 0] * dbad0 + vqn[e, 2, 1] * dbad1 + vqn[e, 2, 2] * dbad2
            vqn_dbad3 = vqn[e, 3, 0] * dbad0 + vqn[e, 3, 1] * dbad1 + vqn[e, 3, 2] * dbad2
            
            db_ar0 = db[e, 0, 0] * ar0 + db[e, 0, 1] * ar1 + db[e, 0, 2] * ar2
            db_ar1 = db[e, 1, 0] * ar0 + db[e, 1, 1] * ar1 + db[e, 1, 2] * ar2
            db_ar2 = db[e, 2, 0] * ar0 + db[e, 2, 1] * ar1 + db[e, 2, 2] * ar2
            db_ar3 = db[e, 3, 0] * ar0 + db[e, 3, 1] * ar1 + db[e, 3, 2] * ar2
            
            ald0 = ad0 + vqn_dbad0 - db_ar0
            ald1 = ad1 + vqn_dbad1 - db_ar1
            ald2 = ad2 + vqn_dbad2 - db_ar2
            ald3 = ad3 + vqn_dbad3 - db_ar3
            
            c1_y = z1[e] * alr1
            c1_x = z1[e] * alr0
            
            dF0_0 = -c1_y * hpat0 + cory[e, 0] * alr2
            dF0_1 = -c1_y * hpat1 + cory[e, 1] * alr2
            dF0_2 = -c1_y * hpat2 + cory[e, 2] * alr2
            dF0_3 = -c1_y * hpat3 + cory[e, 3] * alr2
            
            dF1_0 = c1_x * hpat0 - corx[e, 0] * alr2
            dF1_1 = c1_x * hpat1 - corx[e, 1] * alr2
            dF1_2 = c1_x * hpat2 - corx[e, 2] * alr2
            dF1_3 = c1_x * hpat3 - corx[e, 3] * alr2
            
            dF2_0 = -cory[e, 0] * alr0 + corx[e, 0] * alr1
            dF2_1 = -cory[e, 1] * alr0 + corx[e, 1] * alr1
            dF2_2 = -cory[e, 2] * alr0 + corx[e, 2] * alr1
            dF2_3 = -cory[e, 3] * alr0 + corx[e, 3] * alr1
            
            FL[e, 0, 0] += dF0_0; FL[e, 0, 1] += dF0_1; FL[e, 0, 2] += dF0_2; FL[e, 0, 3] += dF0_3
            FL[e, 1, 0] += dF1_0; FL[e, 1, 1] += dF1_1; FL[e, 1, 2] += dF1_2; FL[e, 1, 3] += dF1_3
            FL[e, 2, 0] += dF2_0; FL[e, 2, 1] += dF2_1; FL[e, 2, 2] += dF2_2; FL[e, 2, 3] += dF2_3
            
            ML3[e, 0, 0] = ML3[e, 0, 0] - alr0 - vqn[e, 0, 0] * ald0
            ML3[e, 0, 1] = ML3[e, 0, 1] - alr0 - vqn[e, 1, 0] * ald1
            ML3[e, 0, 2] = ML3[e, 0, 2] - alr0 - vqn[e, 2, 0] * ald2
            ML3[e, 0, 3] = ML3[e, 0, 3] - alr0 - vqn[e, 3, 0] * ald3
            
            ML3[e, 1, 0] = ML3[e, 1, 0] - alr1 - vqn[e, 0, 1] * ald0
            ML3[e, 1, 1] = ML3[e, 1, 1] - alr1 - vqn[e, 1, 1] * ald1
            ML3[e, 1, 2] = ML3[e, 1, 2] - alr1 - vqn[e, 2, 1] * ald2
            ML3[e, 1, 3] = ML3[e, 1, 3] - alr1 - vqn[e, 3, 1] * ald3
            
            ML3[e, 2, 0] = -alr2 - vqn[e, 0, 2] * ald0
            ML3[e, 2, 1] = -alr2 - vqn[e, 1, 2] * ald1
            ML3[e, 2, 2] = -alr2 - vqn[e, 2, 2] * ald2
            ML3[e, 2, 3] = -alr2 - vqn[e, 3, 2] * ald3
            
        for a in range(4):
            for b in range(3):
                fg_val = 0.0
                mg_val = 0.0
                for c in range(3):
                    fg_val += E[e, b, c] * FL[e, c, a]
                    mg_val += E[e, b, c] * ML3[e, c, a]
                fg[e, a, b] = fg_val
                mg[e, a, b] = mg_val
                

               vg, a11, a12, npt1, gs, amu, rho0, gsr, shfsr, a11sr, a12sr, gmod,
               vhg, dt, alive, Nres, Mres, VF, VM, thick, eint, ehour, sigy2_arr, has_yield):
    
    for i in range(n):
        if not alive[i]:
            continue
            
        area_i = area[i]
        a_i_i = a_i[i]
        z1_i = z1[i]
        x13_i, x24_i = x13[i], x24[i]
        y13_i, y24_i = y13[i], y24[i]
        mx13_i, mx23_i, mx34_i = mx13[i], mx23[i], mx34[i]
        my13_i, my23_i, my34_i = my13[i], my23[i], my34[i]
        
        a11_i = a11[i]
        a12_i = a12[i]
        npt1_i = npt1[i]
        t_i = thick[i]
        
        fbend_i = 0.0 if npt1_i else 1.0 / 12.0
        fbend_v_i = 0.0 if npt1_i else _FBEND_V
        c6_i = t_i * t_i * fbend_i
        
        dhg_0 = vhg[i, 0] * dt
        dhg_1 = vhg[i, 1] * dt
        dhg_2 = vhg[i, 2] * dt
        dhg_3 = vhg[i, 3] * dt
        dhg_4 = vhg[i, 4] * dt
        dhg_5 = vhg[i, 5] * dt
        
        c3g_i = 4.0 * a_i_i
        hxx_i = c3g_i * my34_i
        hyy_i = c3g_i * mx34_i
        hxx_k_i = c3g_i * my23_i
        hyy_k_i = c3g_i * mx23_i
        
        c1m_i = a11_i * _CVIS
        c2m_i = a12_i * _CVIS
        
        cxx_i = hxx_i * dhg_0
        cyy_i = hyy_i * dhg_1
        cxx_k_i = hxx_k_i * dhg_0
        cyy_k_i = hyy_k_i * dhg_1
        bxx_i = hxx_i * dhg_2
        byy_i = hyy_i * dhg_3
        bxx_k_i = hxx_k_i * dhg_2
        byy_k_i = hyy_k_i * dhg_3
        
        dg_0 = c1m_i * cxx_i - c2m_i * cyy_i
        dg_1 = c1m_i * cyy_i - c2m_i * cxx_i
        dg_2 = c1m_i * bxx_i - c2m_i * byy_i
        dg_3 = c1m_i * byy_i - c2m_i * bxx_i
        dg_6 = c1m_i * cxx_k_i - c2m_i * cyy_k_i
        dg_7 = c1m_i * cyy_k_i - c2m_i * cxx_k_i
        dg_8 = c1m_i * bxx_k_i - c2m_i * byy_k_i
        dg_9 = c1m_i * byy_k_i - c2m_i * bxx_k_i
        
        c2s_i = _CVIS * gs[i] / 64.0
        dg_4 = c2s_i * hxx_i * dhg_4
        dg_5 = c2s_i * hyy_i * dhg_4
        dg_10 = c2s_i * hxx_k_i * dhg_5
        dg_11 = c2s_i * hyy_k_i * dhg_5
        
        vg_0 = vg[i, 0]
        vg_1 = vg[i, 1]
        vg_2 = vg[i, 2]
        vg_3 = vg[i, 3]
        vg_4 = vg[i, 4]
        vg_5 = vg[i, 5]
        vg_6 = vg[i, 6]
        vg_7 = vg[i, 7]
        vg_8 = vg[i, 8]
        vg_9 = vg[i, 9]
        vg_10 = vg[i, 10]
        vg_11 = vg[i, 11]
        
        ss1o = my34_i * vg_0 + my23_i * vg_6
        ss2o = mx23_i * vg_7 + mx34_i * vg_1
        sf1o = my34_i * vg_2 + my23_i * vg_8
        sf2o = -mx23_i * vg_9 - mx34_i * vg_3
        sc5o = my34_i * vg_4 + mx34_i * vg_5
        sc6o = my23_i * vg_10 + mx23_i * vg_11
        
        c5_i = 0.5 * 1.0 * t_i * _C7
        esx = ss1o * dhg_0 + ss2o * dhg_1
        etmp1 = c5_i * (esx + 0.25 * (sc5o * dhg_4 + sc6o * dhg_5))
        emx = (sf1o * dhg_2 - sf2o * dhg_3) * c6_i
        etmp2 = c5_i * emx
        
        vg_0 += dg_0
        vg_1 += dg_1
        vg_2 += dg_2
        vg_3 += dg_3
        vg_4 += dg_4
        vg_5 += dg_5
        vg_6 += dg_6
        vg_7 += dg_7
        vg_8 += dg_8
        vg_9 += dg_9
        vg_10 += dg_10
        vg_11 += dg_11
        
        if has_yield[i]:
            sigy2_i = sigy2_arr[i]
            n_0 = Nres[i, 0] / max(t_i, EM20)
            n_1 = Nres[i, 1] / max(t_i, EM20)
            n_2 = Nres[i, 2] / max(t_i, EM20)
            m_0 = Mres[i, 0] / max(t_i * t_i, EM20)
            m_1 = Mres[i, 1] / max(t_i * t_i, EM20)
            m_2 = Mres[i, 2] / max(t_i * t_i, EM20)
            
            sxy0 = (n_0 * n_0 + n_1 * n_1 - n_0 * n_1 + 3.0 * n_2 * n_2)
            mxy0 = (m_0 * m_0 + m_1 * m_1 - m_0 * m_1 + 3.0 * m_2 * m_2)
            
            cnn_i = _COEF
            cmm_i = _COEF * t_i / 16.0
            
            cnnx = cnn_i * vg_0
            cnny = cnn_i * vg_1
            cnnx_k = cnn_i * vg_6
            cnny_k = cnn_i * vg_7
            
            cmmx = cmm_i * vg_2
            cmmy = cmm_i * vg_3
            cmmx_k = cmm_i * vg_8
            cmmy_k = cmm_i * vg_9
            
            sxy0 += cnnx * cnnx + cnny * cnny - cnnx * cnny
            mxy0 += cmmx * cmmx + cmmy * cmmy - cmmx * cmmy
            sxy0 += cnnx_k * cnnx_k + cnny_k * cnny_k - cnnx_k * cnny_k
            mxy0 += cmmx_k * cmmx_k + cmmy_k * cmmy_k - cmmx_k * cmmy_k
            sxy0 += abs(cnnx * (2.0 * cnnx_k - cnny_k) + cnny * (2.0 * cnny_k - cnnx_k))
            mxy0 += abs(cmmx * (2.0 * cmmx_k - cmmy_k) + cmmy * (2.0 * cmmy_k - cmmx_k))
            
            svm = sxy0 + 25.0 * mxy0
            
            if svm > sigy2_i:
                eh1 = min(sxy0 / sigy2_i, 1.0) * _COEFH
                eh2 = _COEFH
                if esx < 0.0:
                    eh1 = 0.0
                if emx < 0.0:
                    eh2 = 0.0
                
                vg_0 -= eh1 * dg_0
                vg_1 -= eh1 * dg_1
                vg_6 -= eh1 * dg_6
                vg_7 -= eh1 * dg_7
                
                vg_2 -= eh2 * dg_2
                vg_3 -= eh2 * dg_3
                vg_8 -= eh2 * dg_8
                vg_9 -= eh2 * dg_9

        vg[i, 0] = vg_0
        vg[i, 1] = vg_1
        vg[i, 2] = vg_2
        vg[i, 3] = vg_3
        vg[i, 4] = vg_4
        vg[i, 5] = vg_5
        vg[i, 6] = vg_6
        vg[i, 7] = vg_7
        vg[i, 8] = vg_8
        vg[i, 9] = vg_9
        vg[i, 10] = vg_10
        vg[i, 11] = vg_11
        
        c8_i = _C7 * 1.0
        ss1 = (my34_i * vg_0 + my23_i * vg_6) * c8_i
        ss2 = (mx23_i * vg_7 + mx34_i * vg_1) * c8_i
        sf1 = (my34_i * vg_2 + my23_i * vg_8) * c8_i
        sf2 = -(mx23_i * vg_9 + mx34_i * vg_3) * c8_i
        hsura_i = t_i * a_i_i
        c2t_i = c8_i * t_i
        sc5 = (my34_i * vg_4 + mx34_i * vg_5) * c2t_i
        sc6 = (my23_i * vg_10 + mx23_i * vg_11) * c2t_i
        ss3 = sc5 + sc6
        
        hvl_i = amu[i] * sqrt(rho0[i] * area_i * _CVIS) * 1.0
        ssv0 = my23_i * my23_i
        ssv1 = my34_i * my34_i
        ssv2 = mx23_i * mx23_i
        ssv3 = mx34_i * mx34_i
        
        hxx_v_i = _STIER * (ssv1 + ssv0)
        hxy_v_i = -_STIER * (my34_i * mx34_i + my23_i * mx23_i)
        hyy_v_i = _STIER * (ssv2 + ssv3)
        c2v_i = hvl_i * gsr[i] * shfsr[i] * _UNDOUZSR
        cxz_v_i = (ssv1 + ssv3) * c2v_i
        cyz_v_i = (ssv2 + ssv0) * c2v_i
        
        aux_i = a_i_i * hvl_i
        c1mv_i = a11sr[i] * aux_i
        c2mv_i = a12sr[i] * aux_i
        cxx_v_i = c1mv_i * hxx_v_i
        cyy_v_i = c1mv_i * hyy_v_i
        cxy_v_i = c2mv_i * hxy_v_i
        
        vhg_0 = vhg[i, 0]
        vhg_1 = vhg[i, 1]
        vhg_2 = vhg[i, 2]
        vhg_3 = vhg[i, 3]
        vhg_4 = vhg[i, 4]
        vhg_5 = vhg[i, 5]
        
        ss1_v = cxx_v_i * vhg_0 + cxy_v_i * vhg_1
        ss2_v = cyy_v_i * vhg_1 + cxy_v_i * vhg_0
        sf1_v = (cxx_v_i * vhg_2 + cxy_v_i * vhg_3) * fbend_v_i
        sf2_v = (-cyy_v_i * vhg_3 - cxy_v_i * vhg_2) * fbend_v_i
        sc5_v = cxz_v_i * vhg_4 * hsura_i
        sc6_v = cyz_v_i * vhg_5 * hsura_i
        
        ss1t = ss1 + ss1_v
        ss2t = ss2 + ss2_v
        sc5t = sc5 + sc5_v
        sc6t = sc6 + sc6_v
        ss3t = ss3 + sc5_v + sc6_v
        sf1t = sf1 + sf1_v
        sf2t = sf2 + sf2_v
        
        y13s = my13_i * ss3t
        x13s = mx13_i * ss3t
        y34s6 = my34_i * sc6t
        y23s5 = my23_i * sc5t
        x23s5 = mx23_i * sc5t
        x34s6 = mx34_i * sc6t
        
        c2n_i = 0.25 * t_i
        b13_i = (my13_i * x24_i - mx13_i * y24_i) * hsura_i
        b24_i = (mx13_i * y13_i - my13_i * x13_i) * hsura_i
        
        VF[i, 0, 0] += b13_i * ss1t
        VF[i, 0, 2] = c2n_i * ss1t
        VF[i, 1, 0] += b13_i * ss2t
        VF[i, 1, 2] = c2n_i * ss2t
        VF[i, 2, 2] = ss3t
        
        VF[i, 0, 1] += b24_i * ss1t
        VF[i, 0, 3] = -VF[i, 0, 2]
        VF[i, 1, 1] += b24_i * ss2t
        VF[i, 1, 3] = -VF[i, 1, 2]
        VF[i, 2, 3] = -VF[i, 2, 2]
        
        c3a_i = c6_i * b13_i
        c4a_i = c6_i * c2n_i
        
        VM[i, 0, 0] += c3a_i * sf2t + y23s5 + y34s6
        VM[i, 0, 2] += c4a_i * sf2t - y13s
        VM[i, 1, 0] += c3a_i * sf1t - x23s5 - x34s6
        VM[i, 1, 2] += c4a_i * sf1t + x13s
        
        c3b_i = c6_i * b24_i
        VM[i, 0, 1] += c3b_i * sf2t + y23s5 - y34s6
        VM[i, 0, 3] += -c4a_i * sf2t - y13s
        VM[i, 1, 1] += c3b_i * sf1t - x23s5 + x34s6
        VM[i, 1, 3] += -c4a_i * sf1t + x13s
        
        c2z_i = z1_i * hsura_i
        VF[i, 2, 0] += c2z_i * (ss1t * y24_i - ss2t * x24_i)
        VF[i, 2, 1] += c2z_i * (-ss1t * y13_i + ss2t * x13_i)
        
        if npt1_i:
            c2nm_i = (1.0 / 12.0) * gmod[i] * rho0[i] * area_i
            hvl_nm_i = 25.0 * amu[i] * sqrt(max(c2nm_i, 0.0)) * 1.0
            cxz_nm_i = (my34_i * my34_i + mx34_i * mx34_i) * hvl_nm_i
            cyz_nm_i = (my23_i * my23_i + mx23_i * mx23_i) * hvl_nm_i
            sc5_nm_i = cxz_nm_i * vhg_4 * hsura_i
            sc6_nm_i = cyz_nm_i * vhg_5 * hsura_i
            ss3_nm_i = sc5_nm_i + sc6_nm_i
            VF[i, 2, 2] += ss3_nm_i
            VF[i, 2, 3] -= ss3_nm_i
            ehour[i] += (sc5_nm_i * vhg_4 + sc6_nm_i * vhg_5) * dt
            
        esy = ((ss1 * dhg_0 + ss2 * dhg_1) * t_i + 0.25 * (sc5 * dhg_4 + sc6 * dhg_5))
        etmp1 = etmp1 + 0.5 * esy
        emy = sf1 * dhg_2 - sf2 * dhg_3
        etmp2 = etmp2 + 0.5 * c6_i * emy * t_i
        eint[i] += etmp1 + etmp2
        
        tesy = ((ss1_v * dhg_0 + ss2_v * dhg_1) * t_i
                + (sf1_v * dhg_2 - sf2_v * dhg_3) * t_i * c6_i
                + 0.25 * (sc5_v * dhg_4 + sc6_v * dhg_5))
        ehour[i] += tesy

    
    fg = np.empty((n, 4, 3))
    mg = np.empty((n, 4, 3))
    # fg, mg from project... wait, project returns fg, mg?
    # we will fix up the return.
    dt_e = np.empty(n)
    for e in range(n):
        visc = np.sqrt(1.0 + st_amu[e] ** 2) - st_amu[e]
        if st_cspd[e] > 0.0:
            dt_e[e] = visc * ll[e] / max(st_cspd[e], 1e-20)
        else:
            dt_e[e] = 1e30
    return fg, mg, dt_e
