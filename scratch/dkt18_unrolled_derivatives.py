import numpy as np

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
        areai = one / area2[i]
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
        
        fac = 1.0 + 0.6 * (1.0 + nu[i]) * thk2[i] / almin
        almax = almax * fac
        
        aldt[i] = area2[i] / np.sqrt(almax)
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
