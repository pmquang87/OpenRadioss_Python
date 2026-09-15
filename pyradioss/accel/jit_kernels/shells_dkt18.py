"""DKT18 shell unrolled scalar loops for Numba JIT."""

import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def dec(fn):
            return fn
        return dec

@njit(cache=True)
def cdkcoor3(xe, ve, re, dt1):
    """
    Unrolled geometry calculations for DKT18 (CDKCOOR3 + CLSKEW3)
    xe: (n, 3, 3) - node coordinates (element, node, x/y/z)
    ve: (n, 3, 3) - node translational velocities
    re: (n, 3, 3) - node rotational velocities
    dt1: float - time step
    
    Returns:
    area2: (n,)
    xl2, yl2, xl3, yl3: (n,)
    vlx, vly, vlz: (n, 2)
    rlx, rly: (n, 3)
    e1, e2, e3: (n, 3) - local frame
    """
    n = xe.shape[0]
    
    area2 = np.empty(n)
    xl2 = np.empty(n)
    yl2 = np.empty(n)
    xl3 = np.empty(n)
    yl3 = np.empty(n)
    
    vlx = np.empty((n, 2))
    vly = np.empty((n, 2))
    vlz = np.empty((n, 2))
    
    rlx = np.empty((n, 3))
    rly = np.empty((n, 3))
    
    e1x_out = np.empty(n)
    e1y_out = np.empty(n)
    e1z_out = np.empty(n)
    e2x_out = np.empty(n)
    e2y_out = np.empty(n)
    e2z_out = np.empty(n)
    e3x_out = np.empty(n)
    e3y_out = np.empty(n)
    e3z_out = np.empty(n)
    
    dt05 = 0.5 * dt1
    EM20 = 1e-20
    
    for e in range(n):
        # Coordinates
        x1 = xe[e, 0, 0]
        y1 = xe[e, 0, 1]
        z1 = xe[e, 0, 2]
        x2 = xe[e, 1, 0]
        y2 = xe[e, 1, 1]
        z2 = xe[e, 1, 2]
        x3 = xe[e, 2, 0]
        y3 = xe[e, 2, 1]
        z3 = xe[e, 2, 2]
        
        # Velocities
        vx1 = ve[e, 0, 0]
        vy1 = ve[e, 0, 1]
        vz1 = ve[e, 0, 2]
        
        vx2 = ve[e, 1, 0] - vx1
        vy2 = ve[e, 1, 1] - vy1
        vz2 = ve[e, 1, 2] - vz1
        
        vx3 = ve[e, 2, 0] - vx1
        vy3 = ve[e, 2, 1] - vy1
        vz3 = ve[e, 2, 2] - vz1
        
        # Rotational velocities
        rx1 = re[e, 0, 0]
        ry1 = re[e, 0, 1]
        rz1 = re[e, 0, 2]
        rx2 = re[e, 1, 0]
        ry2 = re[e, 1, 1]
        rz2 = re[e, 1, 2]
        rx3 = re[e, 2, 0]
        ry3 = re[e, 2, 1]
        rz3 = re[e, 2, 2]
        
        # Relative coordinates
        rx = x2 - x1
        ry = y2 - y1
        rz = z2 - z1
        sx = x3 - x1
        sy = y3 - y1
        sz = z3 - z1
        
        # CLSKEW3 logic (IREP=0)
        e3x = ry * sz - rz * sy
        e3y = rz * sx - rx * sz
        e3z = rx * sy - ry * sx
        
        det = np.sqrt(e3x*e3x + e3y*e3y + e3z*e3z)
        det_max = max(EM20, det)
        if det <= EM20:
            e3x = 0.0
            e3y = 0.0
            e3z = 1.0
            e1x = 1.0
            e1y = 0.0
            e1z = 0.0
            e2x = 0.0
            e2y = 1.0
            e2z = 0.0
        else:
            cc = 1.0 / det_max
            
            e3x *= cc
            e3y *= cc
            e3z *= cc
            
            c1c1 = rx*rx + ry*ry + rz*rz
            c2c2 = sx*sx + sy*sy + sz*sz
            
            if c1c1 != 0.0:
                c2_1 = np.sqrt(c2c2 / max(EM20, c1c1))
                c1_1 = 1.0
            elif c2c2 != 0.0:
                c2_1 = 1.0
                c1_1 = np.sqrt(c1c1 / max(EM20, c2c2))
            else:
                c2_1 = 1.0
                c1_1 = 1.0
                
            e1x = rx * c2_1 + (sy * e3z - sz * e3y) * c1_1
            e1y = ry * c2_1 + (sz * e3x - sx * e3z) * c1_1
            e1z = rz * c2_1 + (sx * e3y - sy * e3x) * c1_1
            
            c1 = np.sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
            if c1 != 0.0:
                c1 = 1.0 / max(EM20, c1)
                
            e1x *= c1
            e1y *= c1
            e1z *= c1
            
            e2x = e3y * e1z - e3z * e1y
            e2y = e3z * e1x - e3x * e1z
            e2z = e3x * e1y - e3y * e1x
        
        e1x_out[e] = e1x
        e1y_out[e] = e1y
        e1z_out[e] = e1z
        e2x_out[e] = e2x
        e2y_out[e] = e2y
        e2z_out[e] = e2z
        e3x_out[e] = e3x
        e3y_out[e] = e3y
        e3z_out[e] = e3z
        
        area2[e] = det
        
        # Local coordinates
        xl2_val = e1x * rx + e1y * ry + e1z * rz
        yl2_val = e2x * rx + e2y * ry + e2z * rz
        xl3_val = e1x * sx + e1y * sy + e1z * sz
        yl3_val = e2x * sx + e2y * sy + e2z * sz
        
        xl2[e] = xl2_val
        yl2[e] = yl2_val
        xl3[e] = xl3_val
        yl3[e] = yl3_val
        
        # Local velocities
        vlx1 = e1x * vx2 + e1y * vy2 + e1z * vz2
        vlx2 = e1x * vx3 + e1y * vy3 + e1z * vz3
        vly1 = e2x * vx2 + e2y * vy2 + e2z * vz2
        vly2 = e2x * vx3 + e2y * vy3 + e2z * vz3
        vlz1 = e3x * vx2 + e3y * vy2 + e3z * vz2
        vlz2 = e3x * vx3 + e3y * vy3 + e3z * vz3
        
        # Local rotational velocities
        rlx[e, 0] = e1x * rx1 + e1y * ry1 + e1z * rz1
        rlx[e, 1] = e1x * rx2 + e1y * ry2 + e1z * rz2
        rlx[e, 2] = e1x * rx3 + e1y * ry3 + e1z * rz3
        rly[e, 0] = e2x * rx1 + e2y * ry1 + e2z * rz1
        rly[e, 1] = e2x * rx2 + e2y * ry2 + e2z * rz2
        rly[e, 2] = e2x * rx3 + e2y * ry3 + e2z * rz3
        
        # Correction 2nd order rigid rotation
        exz = yl3_val * vlz1 - yl2_val * vlz2
        eyz = -xl3_val * vlz1 + xl2_val * vlz2
        
        if det <= EM20:
            ddry = 0.0
            ddrx = 0.0
        else:
            ddry = dt05 * exz / det_max
            ddrx = dt05 * eyz / det_max
        
        v21x = vlx1
        v31x = vlx2
        
        ddrz1 = dt05 * vly1 / xl2_val if abs(xl2_val) > 1e-10 else 0.0
        ddrz2 = dt05 * v31x / yl3_val if abs(yl3_val) > 1e-10 else 0.0
        
        vlx[e, 0] = vlx1 - ddry * vlz1 - ddrz1 * vly1
        vlx[e, 1] = vlx2 - ddry * vlz2 - ddrz1 * vly2
        vly[e, 0] = vly1 - ddrx * vlz1 - ddrz2 * v21x
        vly[e, 1] = vly2 - ddrx * vlz2 - ddrz2 * v31x
        vlz[e, 0] = vlz1
        vlz[e, 1] = vlz2
        
    return area2, xl2, yl2, xl3, yl3, vlx, vly, vlz, rlx, rly, (e1x_out, e1y_out, e1z_out, e2x_out, e2y_out, e2z_out, e3x_out, e3y_out, e3z_out)


@njit(cache=True)
def cdkderi3(px2, py2, px3, py3, px, py, pxy, pyy, ksi, eta):
    """
    Python translation of OpenRadioss CDKDERI3
    """
    n = len(px2)
    
    bz1 = np.empty((n, 2))
    bz2 = np.empty((n, 2))
    bz3 = np.empty((n, 2))
    brx1 = np.empty((n, 3))
    brx2 = np.empty((n, 3))
    brx3 = np.empty((n, 3))
    bry1 = np.empty((n, 3))
    bry2 = np.empty((n, 3))
    bry3 = np.empty((n, 3))
    
    x2y = 1.0 - 2.0 * ksi - eta
    y2x = 1.0 - 2.0 * eta - ksi
    x6 = 6.0 * ksi - 1.0
    y6 = 6.0 * eta - 1.0
    xy6 = x6 + y6 - 3.0
    
    for i in range(n):
        xr1 = px[i, 2] * x2y
        xr2 = px[i, 1] * eta
        xr3 = px[i, 0] * eta
        xs1 = -px[i, 1] * y2x
        xs2 = -px[i, 2] * ksi
        xs3 = -px[i, 0] * ksi
        
        bz1[i, 0] = px2[i] * (-xr1 + xr3) - px3[i] * (xs3 + xs2)
        bz1[i, 1] = -px2[i] * (xr2 + xr3) + px3[i] * (-xs1 + xs3)
        
        yr1 = py[i, 2] * x2y
        yr2 = py[i, 1] * eta
        yr3 = py[i, 0] * eta
        ys1 = -py[i, 1] * y2x
        ys2 = -py[i, 2] * ksi
        ys3 = -py[i, 0] * ksi
        
        bz2[i, 0] = py2[i] * (-yr1 + yr3) - py3[i] * (ys3 + ys2)
        bz2[i, 1] = -py2[i] * (yr2 + yr3) + py3[i] * (-ys1 + ys3)
        
        bz3[i, 0] = px2[i] * (-yr1 + yr3) - px3[i] * (ys3 + ys2) + py2[i] * (-xr1 + xr3) - py3[i] * (xs3 + xs2)
        bz3[i, 1] = -px2[i] * (yr2 + yr3) + px3[i] * (-ys1 + ys3) - py2[i] * (xr2 + xr3) + py3[i] * (-xs1 + xs3)
        
        xr1 = pxy[i, 2] * x2y
        xr2 = -pxy[i, 1] * eta
        xr3 = pxy[i, 0] * eta
        xs1 = pxy[i, 1] * y2x
        xs2 = -pxy[i, 2] * ksi
        xs3 = pxy[i, 0] * ksi
        
        brx1[i, 0] = px2[i] * (xr1 + xr2) + px3[i] * (xs1 + xs2)
        brx1[i, 1] = px2[i] * (xr1 + xr3) + px3[i] * (xs3 + xs2)
        brx1[i, 2] = px2[i] * (xr2 + xr3) + px3[i] * (xs1 + xs3)
        
        yr1 = pyy[i, 2] * x2y
        yr2 = 1.0 - pyy[i, 1] * eta
        yr3 = pyy[i, 0] * eta - 1.0
        ys1 = pyy[i, 1] * y2x
        ys2 = 1.0 - pyy[i, 2] * ksi
        ys3 = pyy[i, 0] * ksi - 1.0
        
        brx2[i, 0] = py2[i] * (yr1 + yr2) + py3[i] * (ys1 + ys2)
        brx2[i, 1] = py2[i] * (yr1 + yr3) + py3[i] * (ys3 + ys2)
        brx2[i, 2] = py2[i] * (yr2 + yr3) + py3[i] * (ys1 + ys3)
        
        rxy1_0 = px2[i] * (yr1 + yr2) + px3[i] * (ys1 + ys2)
        rxy2_0 = py2[i] * (xr1 + xr2) + py3[i] * (xs1 + xs2)
        rxy1_1 = px2[i] * (yr1 + yr3) + px3[i] * (ys3 + ys2)
        rxy2_1 = py2[i] * (xr1 + xr3) + py3[i] * (xs3 + xs2)
        rxy1_2 = px2[i] * (yr2 + yr3) + px3[i] * (ys1 + ys3)
        rxy2_2 = py2[i] * (xr2 + xr3) + py3[i] * (xs1 + xs3)
        
        brx3[i, 0] = rxy1_0 + rxy2_0
        brx3[i, 1] = rxy1_1 + rxy2_1
        brx3[i, 2] = rxy1_2 + rxy2_2
        
        bry1[i, 0] = rxy1_0 + (px2[i] + px3[i]) * xy6
        bry1[i, 1] = rxy1_1 + px2[i] * x6
        bry1[i, 2] = rxy1_2 + px3[i] * y6
        
        bry2[i, 0] = -rxy2_0
        bry2[i, 1] = -rxy2_1
        bry2[i, 2] = -rxy2_2
        
        bry3[i, 0] = (py2[i] + py3[i]) * xy6 + brx2[i, 0] - brx1[i, 0]
        bry3[i, 1] = py2[i] * x6 + brx2[i, 1] - brx1[i, 1]
        bry3[i, 2] = py3[i] * y6 + brx2[i, 2] - brx1[i, 2]

    return bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3



@njit(cache=True)
def cdkderic3(x2, y2, x3, y3, area2, volg, nu, thk2):
    """
    Python translation of OpenRadioss CDKDERIC3
    """
    n = len(x2)
    
    alpe = np.empty(n)
    aldt = np.empty(n)
    px2 = np.empty(n)
    py2 = np.empty(n)
    px3 = np.empty(n)
    py3 = np.empty(n)
    px = np.empty((n, 3))
    py = np.empty((n, 3))
    pxy = np.empty((n, 3))
    pyy = np.empty((n, 3))
    vol = np.empty(n)
    
    one = 1.0
    two = 2.0
    three = 3.0
    third = 1.0 / 3.0
    em20 = 1e-20
    
    for i in range(n):
        areai = one / max(area2[i], em20)
        px2[i] = y3[i] * areai
        py2[i] = -x3[i] * areai
        px3[i] = -y2[i] * areai
        py3[i] = x2[i] * areai
        
        x32 = x3[i] - x2[i]
        y32 = y3[i] - y2[i]
        
        al1 = x32 * x32 + y32 * y32
        al2 = x3[i] * x3[i] + y3[i] * y3[i]
        al3 = x2[i] * x2[i] + y2[i] * y2[i]
        
        almax = max(al1, al2, al3)
        almin = min(al1, al2, al3)
        
        fac = 1.0 + 0.6 * (1.0 + nu[i]) * thk2[i] / max(almin, em20)
        almax = almax * fac
        
        aldt[i] = area2[i] / np.sqrt(max(almax, em20))
        alpe[i] = one
        
        al4 = three / max(al1, em20)
        al5 = three / max(al2, em20)
        al6 = three / max(al3, em20)
        
        pxy[i, 0] = x32 * y32 * al4
        pxy[i, 1] = x3[i] * y3[i] * al5
        pxy[i, 2] = x2[i] * y2[i] * al6
        
        pyy[i, 0] = y32 * y32 * al4
        pyy[i, 1] = y3[i] * y3[i] * al5
        pyy[i, 2] = y2[i] * y2[i] * al6
        
        al4 = two * al4
        al5 = two * al5
        al6 = two * al6
        
        px[i, 0] = x32 * al4
        px[i, 1] = -x3[i] * al5
        px[i, 2] = x2[i] * al6
        
        py[i, 0] = y32 * al4
        py[i, 1] = -y3[i] * al5
        py[i, 2] = y2[i] * al6
        
        vol[i] = third * volg[i]

    return alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, vol


@njit(cache=True)
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
        
        exx[i] = vdef[i, 0] * dt1
        eyy[i] = vdef[i, 1] * dt1
        exy[i] = vdef[i, 2] * dt1

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


@njit(cache=True)
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


@njit(cache=True)
def cdkfint3(vol, thk0, force, mom, px2, py2, px3, py3, bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3, f11, f12, f13, f21, f22, f23, f32, f33, m11, m12, m13, m21, m22, m23):
    n = len(vol)
    for i in range(n):
        c2 = vol[i]
        fx2 = c2 * (px2[i] * force[i, 0] + py2[i] * force[i, 2])
        fy2 = c2 * (py2[i] * force[i, 1] + px2[i] * force[i, 2])
        fx3 = c2 * (px3[i] * force[i, 0] + py3[i] * force[i, 2])
        fy3 = c2 * (py3[i] * force[i, 1] + px3[i] * force[i, 2])
        f12[i] += fx2
        f22[i] += fy2
        f13[i] += fx3
        f23[i] += fy3

    for i in range(n):
        c2 = vol[i] * thk0[i]
        f32[i] += c2 * (bz1[i, 0] * mom[i, 0] + bz2[i, 0] * mom[i, 1] + bz3[i, 0] * mom[i, 2])
        f33[i] += c2 * (bz1[i, 1] * mom[i, 0] + bz2[i, 1] * mom[i, 1] + bz3[i, 1] * mom[i, 2])
        m11[i] += c2 * (brx1[i, 0] * mom[i, 0] + brx2[i, 0] * mom[i, 1] + brx3[i, 0] * mom[i, 2])
        m21[i] += c2 * (bry1[i, 0] * mom[i, 0] + bry2[i, 0] * mom[i, 1] + bry3[i, 0] * mom[i, 2])
        m12[i] += c2 * (brx1[i, 1] * mom[i, 0] + brx2[i, 1] * mom[i, 1] + brx3[i, 1] * mom[i, 2])
        m22[i] += c2 * (bry1[i, 1] * mom[i, 0] + bry2[i, 1] * mom[i, 1] + bry3[i, 1] * mom[i, 2])
        m13[i] += c2 * (brx1[i, 2] * mom[i, 0] + brx2[i, 2] * mom[i, 1] + brx3[i, 2] * mom[i, 2])
        m23[i] += c2 * (bry1[i, 2] * mom[i, 0] + bry2[i, 2] * mom[i, 1] + bry3[i, 2] * mom[i, 2])


@njit(cache=True)
def cdkfcum3(px2, py2, px3, py3, r11, r12, r13, r21, r22, r23, r31, r32, r33, f11, f12, f13, f21, f22, f23, f31, f32, f33, m11, m12, m13, m21, m22, m23, m31, m32, m33):
    n = len(f12)
    for i in range(n):
        lx = r11[i] * f12[i] + r12[i] * f22[i] + r13[i] * f32[i]
        ly = r21[i] * f12[i] + r22[i] * f22[i] + r23[i] * f32[i]
        lz = r31[i] * f12[i] + r32[i] * f22[i] + r33[i] * f32[i]
        f12[i] = lx
        f22[i] = ly
        f32[i] = lz

        lx = r11[i] * f13[i] + r12[i] * f23[i] + r13[i] * f33[i]
        ly = r21[i] * f13[i] + r22[i] * f23[i] + r23[i] * f33[i]
        lz = r31[i] * f13[i] + r32[i] * f23[i] + r33[i] * f33[i]
        f13[i] = lx
        f23[i] = ly
        f33[i] = lz

        f11[i] = -f12[i] - f13[i]
        f21[i] = -f22[i] - f23[i]
        f31[i] = -f32[i] - f33[i]

    for i in range(n):
        lx = r11[i] * m11[i] + r12[i] * m21[i]
        ly = r21[i] * m11[i] + r22[i] * m21[i]
        m31[i] = r31[i] * m11[i] + r32[i] * m21[i]
        m11[i] = lx
        m21[i] = ly

        lx = r11[i] * m12[i] + r12[i] * m22[i]
        ly = r21[i] * m12[i] + r22[i] * m22[i]
        m32[i] = r31[i] * m12[i] + r32[i] * m22[i]
        m12[i] = lx
        m22[i] = ly

        lx = r11[i] * m13[i] + r12[i] * m23[i]
        ly = r21[i] * m13[i] + r22[i] * m23[i]
        m33[i] = r31[i] * m13[i] + r32[i] * m23[i]
        m13[i] = lx
        m23[i] = ly
