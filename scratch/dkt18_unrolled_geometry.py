import numpy as np

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
        
        ddry = dt05 * exz / det_max
        ddrx = dt05 * eyz / det_max
        
        v21x = vlx1
        v31x = vlx2
        
        ddrz1 = dt05 * vly1 / xl2_val if xl2_val != 0.0 else 0.0
        ddrz2 = dt05 * v31x / yl3_val if yl3_val != 0.0 else 0.0
        
        vlx[e, 0] = vlx1 - ddry * vlz1 - ddrz1 * vly1
        vlx[e, 1] = vlx2 - ddry * vlz2 - ddrz1 * vly2
        vly[e, 0] = vly1 - ddrx * vlz1 - ddrz2 * v21x
        vly[e, 1] = vly2 - ddrx * vlz2 - ddrz2 * v31x
        vlz[e, 0] = vlz1
        vlz[e, 1] = vlz2
        
    return area2, xl2, yl2, xl3, yl3, vlx, vly, vlz, rlx, rly, (e1x_out, e1y_out, e1z_out, e2x_out, e2y_out, e2z_out, e3x_out, e3y_out, e3z_out)
