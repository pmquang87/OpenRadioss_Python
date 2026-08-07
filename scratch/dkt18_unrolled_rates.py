import numpy as np

def cdkdefo3(vlx, vly, px2, py2, px3, py3, exx, eyy, exy, exz, eyz, dt1, epsdot, nft, istrain, gstr, vdef, iepsdot):
    """
    Python translation of CDKDEFO3 (membrane rates).
    """
    n = len(px2)
    
    for i in range(n):
        vdef[i, 0] = px2[i] * vlx[i, 0] + px3[i] * vlx[i, 1]
        vdef[i, 1] = py2[i] * vly[i, 0] + py3[i] * vly[i, 1]
        vdef[i, 2] = px2[i] * vly[i, 0] + px3[i] * vly[i, 1] + py2[i] * vlx[i, 0] + py3[i] * vlx[i, 1]
        exz[i] = 0.0
        eyz[i] = 0.0

    if iepsdot:
        for i in range(n):
            j = i + nft
            epsdot[0, j] = vdef[i, 0]
            epsdot[1, j] = vdef[i, 1]
            epsdot[2, j] = vdef[i, 2]

    for i in range(n):
        exx[i] = vdef[i, 0] * dt1
        eyy[i] = vdef[i, 1] * dt1
        exy[i] = vdef[i, 2] * dt1

    if istrain:
        for i in range(n):
            gstr[i, 0] += exx[i]
            gstr[i, 1] += eyy[i]
            gstr[i, 2] += exy[i]

def cdkcurv3(bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3, vlz, rlx, rly, kxx, kyy, kxy):
    """
    Python translation of CDKCURV3 (bending rates).
    """
    n = len(kxx)
    
    for i in range(n):
        kxx[i] = (bz1[i, 0] * vlz[i, 0] + bz1[i, 1] * vlz[i, 1] +
                  brx1[i, 0] * rlx[i, 0] + brx1[i, 1] * rlx[i, 1] + brx1[i, 2] * rlx[i, 2] +
                  bry1[i, 0] * rly[i, 0] + bry1[i, 1] * rly[i, 1] + bry1[i, 2] * rly[i, 2])
        
        kyy[i] = (bz2[i, 0] * vlz[i, 0] + bz2[i, 1] * vlz[i, 1] +
                  brx2[i, 0] * rlx[i, 0] + brx2[i, 1] * rlx[i, 1] + brx2[i, 2] * rlx[i, 2] +
                  bry2[i, 0] * rly[i, 0] + bry2[i, 1] * rly[i, 1] + bry2[i, 2] * rly[i, 2])
                  
        kxy[i] = (bz3[i, 0] * vlz[i, 0] + bz3[i, 1] * vlz[i, 1] +
                  brx3[i, 0] * rlx[i, 0] + brx3[i, 1] * rlx[i, 1] + brx3[i, 2] * rlx[i, 2] +
                  bry3[i, 0] * rly[i, 0] + bry3[i, 1] * rly[i, 1] + bry3[i, 2] * rly[i, 2])
