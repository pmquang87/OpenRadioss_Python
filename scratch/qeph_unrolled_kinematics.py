import numpy as np

def _kinematics(E, area, a_i, z1, x13, x24, y13, y24, mx13, my13, lm, corx, cory, l24, l13, ve, vre, dt, npt1):
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

    return v13, v24, vhi, rl, plat, vqn, di, db
