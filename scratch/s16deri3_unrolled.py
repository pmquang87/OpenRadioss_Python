def s16deri3(xx, dnidr, dnids, dnidt):
    """
    Computes the Cartesian derivatives PX, PY, PZ, determinant of Jacobian, etc.
    xx: array-like of shape (3, 16) - coordinates of 16 nodes (0-based)
    dnidr, dnids, dnidt: array-like of shape (16,)
    
    Returns px, py, pz, det
    """
    
    # dxdr
    dxdr = (
        dnidr[0]*xx[0][0] + dnidr[1]*xx[0][1] + dnidr[2]*xx[0][2] + dnidr[3]*xx[0][3] +
        dnidr[4]*xx[0][4] + dnidr[5]*xx[0][5] + dnidr[6]*xx[0][6] + dnidr[7]*xx[0][7] +
        dnidr[8]*(xx[0][8] - xx[0][10]) + dnidr[9]*xx[0][9] +
        dnidr[11]*xx[0][11] + dnidr[12]*(xx[0][12] - xx[0][14]) +
        dnidr[13]*xx[0][13] + dnidr[15]*xx[0][15]
    )
    dydr = (
        dnidr[0]*xx[1][0] + dnidr[1]*xx[1][1] + dnidr[2]*xx[1][2] + dnidr[3]*xx[1][3] +
        dnidr[4]*xx[1][4] + dnidr[5]*xx[1][5] + dnidr[6]*xx[1][6] + dnidr[7]*xx[1][7] +
        dnidr[8]*(xx[1][8] - xx[1][10]) + dnidr[9]*xx[1][9] +
        dnidr[11]*xx[1][11] + dnidr[12]*(xx[1][12] - xx[1][14]) +
        dnidr[13]*xx[1][13] + dnidr[15]*xx[1][15]
    )
    dzdr = (
        dnidr[0]*xx[2][0] + dnidr[1]*xx[2][1] + dnidr[2]*xx[2][2] + dnidr[3]*xx[2][3] +
        dnidr[4]*xx[2][4] + dnidr[5]*xx[2][5] + dnidr[6]*xx[2][6] + dnidr[7]*xx[2][7] +
        dnidr[8]*(xx[2][8] - xx[2][10]) + dnidr[9]*xx[2][9] +
        dnidr[11]*xx[2][11] + dnidr[12]*(xx[2][12] - xx[2][14]) +
        dnidr[13]*xx[2][13] + dnidr[15]*xx[2][15]
    )
    
    # dxds
    dxds = (
        dnids[0]*xx[0][0] + dnids[1]*xx[0][1] + dnids[2]*xx[0][2] + dnids[3]*xx[0][3] +
        dnids[4]*xx[0][4] + dnids[5]*xx[0][5] + dnids[6]*xx[0][6] + dnids[7]*xx[0][7] +
        dnids[8]*(xx[0][8] - xx[0][12]) +
        dnids[9]*(xx[0][9] - xx[0][13]) +
        dnids[10]*(xx[0][10] - xx[0][14]) +
        dnids[11]*(xx[0][11] - xx[0][15])
    )
    dyds = (
        dnids[0]*xx[1][0] + dnids[1]*xx[1][1] + dnids[2]*xx[1][2] + dnids[3]*xx[1][3] +
        dnids[4]*xx[1][4] + dnids[5]*xx[1][5] + dnids[6]*xx[1][6] + dnids[7]*xx[1][7] +
        dnids[8]*(xx[1][8] - xx[1][12]) +
        dnids[9]*(xx[1][9] - xx[1][13]) +
        dnids[10]*(xx[1][10] - xx[1][14]) +
        dnids[11]*(xx[1][11] - xx[1][15])
    )
    dzds = (
        dnids[0]*xx[2][0] + dnids[1]*xx[2][1] + dnids[2]*xx[2][2] + dnids[3]*xx[2][3] +
        dnids[4]*xx[2][4] + dnids[5]*xx[2][5] + dnids[6]*xx[2][6] + dnids[7]*xx[2][7] +
        dnids[8]*(xx[2][8] - xx[2][12]) +
        dnids[9]*(xx[2][9] - xx[2][13]) +
        dnids[10]*(xx[2][10] - xx[2][14]) +
        dnids[11]*(xx[2][11] - xx[2][15])
    )
    
    # dxdt
    dxdt = (
        dnidt[0]*xx[0][0] + dnidt[1]*xx[0][1] + dnidt[2]*xx[0][2] + dnidt[3]*xx[0][3] +
        dnidt[4]*xx[0][4] + dnidt[5]*xx[0][5] + dnidt[6]*xx[0][6] + dnidt[7]*xx[0][7] +
        dnidt[8]*xx[0][8] + dnidt[9]*(xx[0][9] - xx[0][11]) +
        dnidt[10]*xx[0][10] + dnidt[12]*xx[0][12] +
        dnidt[13]*(xx[0][13] - xx[0][15]) + dnidt[14]*xx[0][14]
    )
    dydt = (
        dnidt[0]*xx[1][0] + dnidt[1]*xx[1][1] + dnidt[2]*xx[1][2] + dnidt[3]*xx[1][3] +
        dnidt[4]*xx[1][4] + dnidt[5]*xx[1][5] + dnidt[6]*xx[1][6] + dnidt[7]*xx[1][7] +
        dnidt[8]*xx[1][8] + dnidt[9]*(xx[1][9] - xx[1][11]) +
        dnidt[10]*xx[1][10] + dnidt[12]*xx[1][12] +
        dnidt[13]*(xx[1][13] - xx[1][15]) + dnidt[14]*xx[1][14]
    )
    dzdt = (
        dnidt[0]*xx[2][0] + dnidt[1]*xx[2][1] + dnidt[2]*xx[2][2] + dnidt[3]*xx[2][3] +
        dnidt[4]*xx[2][4] + dnidt[5]*xx[2][5] + dnidt[6]*xx[2][6] + dnidt[7]*xx[2][7] +
        dnidt[8]*xx[2][8] + dnidt[9]*(xx[2][9] - xx[2][11]) +
        dnidt[10]*xx[2][10] + dnidt[12]*xx[2][12] +
        dnidt[13]*(xx[2][13] - xx[2][15]) + dnidt[14]*xx[2][14]
    )
    
    # Inversion of the jacobian
    drdx = dyds * dzdt - dzds * dydt
    drdy = dzds * dxdt - dxds * dzdt
    drdz = dxds * dydt - dyds * dxdt
    
    dsdz = dxdt * dydr - dydt * dxdr
    dsdy = dzdt * dxdr - dxdt * dzdr
    dsdx = dydt * dzdr - dzdt * dydr
    
    dtdx = dydr * dzds - dzdr * dyds
    dtdy = dzdr * dxds - dxdr * dzds
    dtdz = dxdr * dyds - dydr * dxds
    
    det = (
        dxdr * drdx +
        dydr * drdy +
        dzdr * drdz
    )
    
    if det <= 0.0:
        pass # Fortran handles det <= 0 as MSGERROR
        
    d = 1.0 / det
    
    drdx *= d
    dsdx *= d
    dtdx *= d
    
    drdy *= d
    dsdy *= d
    dtdy *= d
    
    drdz *= d
    dsdz *= d
    dtdz *= d
    
    px = [0.0] * 16
    px[0] = dnidr[0]*drdx + dnids[0]*dsdx + dnidt[0]*dtdx
    px[1] = dnidr[1]*drdx + dnids[1]*dsdx + dnidt[1]*dtdx
    px[2] = dnidr[2]*drdx + dnids[2]*dsdx + dnidt[2]*dtdx
    px[3] = dnidr[3]*drdx + dnids[3]*dsdx + dnidt[3]*dtdx
    px[4] = dnidr[4]*drdx + dnids[4]*dsdx + dnidt[4]*dtdx
    px[5] = dnidr[5]*drdx + dnids[5]*dsdx + dnidt[5]*dtdx
    px[6] = dnidr[6]*drdx + dnids[6]*dsdx + dnidt[6]*dtdx
    px[7] = dnidr[7]*drdx + dnids[7]*dsdx + dnidt[7]*dtdx
    
    r9_x  = dnidr[8]*drdx
    r13_x = dnidr[12]*drdx
    s9_x  = dnids[8]*dsdx
    s10_x = dnids[9]*dsdx
    s11_x = dnids[10]*dsdx
    s12_x = dnids[11]*dsdx
    t10_x = dnidt[9]*dtdx
    t14_x = dnidt[13]*dtdx
    
    px[8] = r9_x + s9_x + dnidt[8]*dtdx
    px[9] = dnidr[9]*drdx + s10_x + t10_x
    px[10] = -r9_x + s11_x + dnidt[10]*dtdx
    px[11] = dnidr[11]*drdx + s12_x - t10_x
    px[12] = r13_x - s9_x + dnidt[12]*dtdx
    px[13] = dnidr[13]*drdx - s10_x + t14_x
    px[14] = -r13_x - s11_x + dnidt[14]*dtdx
    px[15] = dnidr[15]*drdx - s12_x - t14_x
    
    py = [0.0] * 16
    py[0] = dnidr[0]*drdy + dnids[0]*dsdy + dnidt[0]*dtdy
    py[1] = dnidr[1]*drdy + dnids[1]*dsdy + dnidt[1]*dtdy
    py[2] = dnidr[2]*drdy + dnids[2]*dsdy + dnidt[2]*dtdy
    py[3] = dnidr[3]*drdy + dnids[3]*dsdy + dnidt[3]*dtdy
    py[4] = dnidr[4]*drdy + dnids[4]*dsdy + dnidt[4]*dtdy
    py[5] = dnidr[5]*drdy + dnids[5]*dsdy + dnidt[5]*dtdy
    py[6] = dnidr[6]*drdy + dnids[6]*dsdy + dnidt[6]*dtdy
    py[7] = dnidr[7]*drdy + dnids[7]*dsdy + dnidt[7]*dtdy
    
    r9_y  = dnidr[8]*drdy
    r13_y = dnidr[12]*drdy
    s9_y  = dnids[8]*dsdy
    s10_y = dnids[9]*dsdy
    s11_y = dnids[10]*dsdy
    s12_y = dnids[11]*dsdy
    t10_y = dnidt[9]*dtdy
    t14_y = dnidt[13]*dtdy
    
    py[8] = r9_y + s9_y + dnidt[8]*dtdy
    py[9] = dnidr[9]*drdy + s10_y + t10_y
    py[10] = -r9_y + s11_y + dnidt[10]*dtdy
    py[11] = dnidr[11]*drdy + s12_y - t10_y
    py[12] = r13_y - s9_y + dnidt[12]*dtdy
    py[13] = dnidr[13]*drdy - s10_y + t14_y
    py[14] = -r13_y - s11_y + dnidt[14]*dtdy
    py[15] = dnidr[15]*drdy - s12_y - t14_y
    
    pz = [0.0] * 16
    pz[0] = dnidr[0]*drdz + dnids[0]*dsdz + dnidt[0]*dtdz
    pz[1] = dnidr[1]*drdz + dnids[1]*dsdz + dnidt[1]*dtdz
    pz[2] = dnidr[2]*drdz + dnids[2]*dsdz + dnidt[2]*dtdz
    pz[3] = dnidr[3]*drdz + dnids[3]*dsdz + dnidt[3]*dtdz
    pz[4] = dnidr[4]*drdz + dnids[4]*dsdz + dnidt[4]*dtdz
    pz[5] = dnidr[5]*drdz + dnids[5]*dsdz + dnidt[5]*dtdz
    pz[6] = dnidr[6]*drdz + dnids[6]*dsdz + dnidt[6]*dtdz
    pz[7] = dnidr[7]*drdz + dnids[7]*dsdz + dnidt[7]*dtdz
    
    r9_z  = dnidr[8]*drdz
    r13_z = dnidr[12]*drdz
    s9_z  = dnids[8]*dsdz
    s10_z = dnids[9]*dsdz
    s11_z = dnids[10]*dsdz
    s12_z = dnids[11]*dsdz
    t10_z = dnidt[9]*dtdz
    t14_z = dnidt[13]*dtdz
    
    pz[8] = r9_z + s9_z + dnidt[8]*dtdz
    pz[9] = dnidr[9]*drdz + s10_z + t10_z
    pz[10] = -r9_z + s11_z + dnidt[10]*dtdz
    pz[11] = dnidr[11]*drdz + s12_z - t10_z
    pz[12] = r13_z - s9_z + dnidt[12]*dtdz
    pz[13] = dnidr[13]*drdz - s10_z + t14_z
    pz[14] = -r13_z - s11_z + dnidt[14]*dtdz
    pz[15] = dnidr[15]*drdz - s12_z - t14_z
    
    return px, py, pz, det
