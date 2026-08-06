import numpy as np
from math import sqrt
import numba as nb

_CVIS = 1.0             # GEO(17) after hm_read_prop01.F 197-230 (Ishell=24)
_DN_DEFAULT = 0.015     # ZEP015
_COEF = 0.85            # ZEP85
_COEFH = 0.999          # ZEP999
_STIER = 16.0 / 3.0     # FIVEP333
_FBEND_V = 3.464        # THREEP464
_UNDOUZSR = np.sqrt(1.0 / 12.0)
_TOL_PLAS = 1.0e-18     # TOL
_C7 = 4.0 / 3.0         # C7 = FOUR_OVER_3
_FACDT = 1.25           # FIVE_OVER_4
EM20 = 1.0e-20

@nb.njit(cache=True)
def _fint_stab(n, area, a_i, z1, x13, x24, y13, y24, mx13, mx23, mx34, my13, my23, my34,
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
