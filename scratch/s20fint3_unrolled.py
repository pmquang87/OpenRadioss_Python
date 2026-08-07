import numpy as np

def s20fint3(px, py, pz, sig, voln):
    """
    Computes internal forces for the 16-node thick shell (solide16) element.
    Uses 0-based indexing.
    
    px, py, pz : arrays of shape (16,) containing Cartesian derivatives
    sig        : array of shape (6,) containing stress components (xx, yy, zz, xy, yz, zx)
    voln       : volume integration weight for this integration point
    
    Returns:
    fint       : array of shape (48,) containing the interleaved nodal internal forces (fx, fy, fz)
    """
    fint = np.zeros(48, dtype=np.float64)
    
    s1 = sig[0] * voln
    s2 = sig[1] * voln
    s3 = sig[2] * voln
    s4 = sig[3] * voln
    s5 = sig[4] * voln
    s6 = sig[5] * voln
    
    for n in range(16):
        fint[3 * n + 0] = s1 * px[n] + s4 * py[n] + s6 * pz[n]
        fint[3 * n + 1] = s4 * px[n] + s2 * py[n] + s5 * pz[n]
        fint[3 * n + 2] = s6 * px[n] + s5 * py[n] + s3 * pz[n]
        
    return fint
