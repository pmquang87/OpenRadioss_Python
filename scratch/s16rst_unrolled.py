def s16rst(r, s, t):
    # constants
    HALF = 0.5
    
    # helper variables
    u_m_r = HALF * (1.0 - r)
    u_p_r = HALF * (1.0 + r)
    
    u_m_s = HALF * (1.0 - s)
    u_p_s = HALF * (1.0 + s)
    
    u_m_t = HALF * (1.0 - t)
    u_p_t = HALF * (1.0 + t)
    
    ums_umt = u_m_s * u_m_t
    ums_upt = u_m_s * u_p_t
    ups_umt = u_p_s * u_m_t
    ups_upt = u_p_s * u_p_t
    
    umr_ums = u_m_r * u_m_s
    umr_ups = u_m_r * u_p_s
    upr_ums = u_p_r * u_m_s
    upr_ups = u_p_r * u_p_s
    
    umt_umr = u_m_t * u_m_r
    umt_upr = u_m_t * u_p_r
    upt_umr = u_p_t * u_m_r
    upt_upr = u_p_t * u_p_r

    # NI values (indices 0 to 15, matching Fortran 1 to 16)
    ni_0 = u_m_r * ums_umt * (-r - t - 1.0)
    ni_1 = u_m_r * ums_upt * (-r + t - 1.0)
    ni_2 = u_p_r * ums_upt * ( r + t - 1.0)
    ni_3 = u_p_r * ums_umt * ( r - t - 1.0)
    ni_4 = u_m_r * ups_umt * (-r - t - 1.0)
    ni_5 = u_m_r * ups_upt * (-r + t - 1.0)
    ni_6 = u_p_r * ups_upt * ( r + t - 1.0)
    ni_7 = u_p_r * ups_umt * ( r - t - 1.0)
    
    a_r = 1.0 - r * r
    ni_9  = a_r * ums_upt
    ni_11 = a_r * ums_umt
    ni_13 = a_r * ups_upt
    ni_15 = a_r * ups_umt
    
    a_t = 1.0 - t * t
    ni_8  = a_t * umr_ums
    ni_10 = a_t * upr_ums
    ni_12 = a_t * umr_ups
    ni_14 = a_t * upr_ups

    # DNIDR values
    dnidr_0 = -ums_umt * (-HALF * t - r)
    dnidr_1 = -ums_upt * ( HALF * t - r)
    dnidr_2 =  ums_upt * ( HALF * t + r)
    dnidr_3 =  ums_umt * (-HALF * t + r)
    dnidr_4 = -ups_umt * (-HALF * t - r)
    dnidr_5 = -ups_upt * ( HALF * t - r)
    dnidr_6 =  ups_upt * ( HALF * t + r)
    dnidr_7 =  ups_umt * (-HALF * t + r)

    a_t_half = HALF * a_t
    dnidr_8  = -a_t_half * u_m_s
    dnidr_10 =  a_t_half * u_m_s
    dnidr_12 = -a_t_half * u_p_s
    dnidr_14 =  a_t_half * u_p_s

    a_r_r2 = -2.0 * r
    dnidr_9  = a_r_r2 * ums_upt
    dnidr_11 = a_r_r2 * ums_umt
    dnidr_13 = a_r_r2 * ups_upt
    dnidr_15 = a_r_r2 * ups_umt
    
    # DNIDS values
    dnids_0 = -umt_umr * (-r - t - 1.0) * HALF
    dnids_1 = -upt_umr * (-r + t - 1.0) * HALF
    dnids_2 = -upt_upr * ( r + t - 1.0) * HALF
    dnids_3 = -umt_upr * ( r - t - 1.0) * HALF
    dnids_4 =  umt_umr * (-r - t - 1.0) * HALF
    dnids_5 =  upt_umr * (-r + t - 1.0) * HALF
    dnids_6 =  upt_upr * ( r + t - 1.0) * HALF
    dnids_7 =  umt_upr * ( r - t - 1.0) * HALF
    
    a_r_half = HALF * a_r
    dnids_9  = -a_r_half * u_p_t
    dnids_11 = -a_r_half * u_m_t
    dnids_13 =  a_r_half * u_p_t
    dnids_15 =  a_r_half * u_m_t
    
    dnids_8  = -a_t_half * u_m_r
    dnids_10 = -a_t_half * u_p_r
    dnids_12 =  a_t_half * u_m_r
    dnids_14 =  a_t_half * u_p_r

    # DNIDT values
    dnidt_0 = -umr_ums * (-HALF * r - t)
    dnidt_1 =  umr_ums * (-HALF * r + t)
    dnidt_2 =  upr_ums * ( HALF * r + t)
    dnidt_3 = -upr_ums * ( HALF * r - t)
    dnidt_4 = -umr_ups * (-HALF * r - t)
    dnidt_5 =  umr_ups * (-HALF * r + t)
    dnidt_6 =  upr_ups * ( HALF * r + t)
    dnidt_7 = -upr_ups * ( HALF * r - t)

    dnidt_9  =  a_r_half * u_m_s
    dnidt_11 = -a_r_half * u_m_s
    dnidt_13 =  a_r_half * u_p_s
    dnidt_15 = -a_r_half * u_p_s

    a_t_t2 = -2.0 * t
    dnidt_8  = a_t_t2 * umr_ums
    dnidt_10 = a_t_t2 * upr_ums
    dnidt_12 = a_t_t2 * umr_ups
    dnidt_14 = a_t_t2 * upr_ups

    ni = (ni_0, ni_1, ni_2, ni_3, ni_4, ni_5, ni_6, ni_7, ni_8, ni_9, ni_10, ni_11, ni_12, ni_13, ni_14, ni_15)
    dnidr = (dnidr_0, dnidr_1, dnidr_2, dnidr_3, dnidr_4, dnidr_5, dnidr_6, dnidr_7, dnidr_8, dnidr_9, dnidr_10, dnidr_11, dnidr_12, dnidr_13, dnidr_14, dnidr_15)
    dnids = (dnids_0, dnids_1, dnids_2, dnids_3, dnids_4, dnids_5, dnids_6, dnids_7, dnids_8, dnids_9, dnids_10, dnids_11, dnids_12, dnids_13, dnids_14, dnids_15)
    dnidt = (dnidt_0, dnidt_1, dnidt_2, dnidt_3, dnidt_4, dnidt_5, dnidt_6, dnidt_7, dnidt_8, dnidt_9, dnidt_10, dnidt_11, dnidt_12, dnidt_13, dnidt_14, dnidt_15)
    
    return ni, dnidr, dnids, dnidt
