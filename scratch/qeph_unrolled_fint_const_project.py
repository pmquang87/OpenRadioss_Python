import numpy as np
from numba import njit

@njit(cache=True)
def unrolled_fint_const(x13, x24, y13, y24, mx13, my13, my34, mx23, my23, mx34, z1, a_i, Nres, Mres, qres):
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

    return VF, VM

@njit(cache=True)
def unrolled_project(E, z1, corx, cory, VF, VM, plat, vqn, di, db):
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
                
    return fg, mg
