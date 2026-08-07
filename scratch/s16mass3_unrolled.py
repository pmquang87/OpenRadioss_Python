def init_mass(n, fill, rho, vol, dtx, dtelem, mass, mss, mssx, nc, stifn, deltax2):
    TWO = 2.0
    SIXTY4 = 64.0
    EM20 = 1e-20
    THIRTY2 = 32.0
    THREE = 3.0
    HALF = 0.5
    
    for i in range(n):
        mass[i] = fill[i] * rho[i] * vol[i]
        
        if dtelem[i] > dtx[i]:
            dtelem[i] = dtx[i]
            
        dtx_sq = dtx[i] * dtx[i]
        max_val = dtx_sq if dtx_sq > EM20 else EM20
        sti = fill[i] * rho[i] * vol[i] * TWO / SIXTY4 / max_val
        
        am = mass[i] / THIRTY2
        bm = mass[i] * THREE / THIRTY2
        
        mss[i, 0] = am
        mss[i, 1] = am
        mss[i, 2] = am
        mss[i, 3] = am
        mss[i, 4] = am
        mss[i, 5] = am
        mss[i, 6] = am
        mss[i, 7] = am
        
        stifn[nc[i, 0]] += sti * deltax2[i]
        stifn[nc[i, 1]] += sti * deltax2[i]
        stifn[nc[i, 2]] += sti * deltax2[i]
        stifn[nc[i, 3]] += sti * deltax2[i]
        stifn[nc[i, 4]] += sti * deltax2[i]
        stifn[nc[i, 5]] += sti * deltax2[i]
        stifn[nc[i, 6]] += sti * deltax2[i]
        stifn[nc[i, 7]] += sti * deltax2[i]
        
        # N=9 (IPERM1=1, IPERM2=2)
        if nc[i, 8] >= 0:
            mssx[i, 0] = bm
            stifn[nc[i, 8]] += sti
        else:
            mss[i, 0] += HALF * bm
            mss[i, 1] += HALF * bm
            stifn[nc[i, 0]] += HALF * sti
            stifn[nc[i, 1]] += HALF * sti

        # N=10 (IPERM1=2, IPERM2=3)
        if nc[i, 9] >= 0:
            mssx[i, 1] = bm
            stifn[nc[i, 9]] += sti
        else:
            mss[i, 1] += HALF * bm
            mss[i, 2] += HALF * bm
            stifn[nc[i, 1]] += HALF * sti
            stifn[nc[i, 2]] += HALF * sti

        # N=11 (IPERM1=3, IPERM2=4)
        if nc[i, 10] >= 0:
            mssx[i, 2] = bm
            stifn[nc[i, 10]] += sti
        else:
            mss[i, 2] += HALF * bm
            mss[i, 3] += HALF * bm
            stifn[nc[i, 2]] += HALF * sti
            stifn[nc[i, 3]] += HALF * sti

        # N=12 (IPERM1=4, IPERM2=1)
        if nc[i, 11] >= 0:
            mssx[i, 3] = bm
            stifn[nc[i, 11]] += sti
        else:
            mss[i, 3] += HALF * bm
            mss[i, 0] += HALF * bm
            stifn[nc[i, 3]] += HALF * sti
            stifn[nc[i, 0]] += HALF * sti

        # N=13 (IPERM1=5, IPERM2=6)
        if nc[i, 12] >= 0:
            mssx[i, 4] = bm
            stifn[nc[i, 12]] += sti
        else:
            mss[i, 4] += HALF * bm
            mss[i, 5] += HALF * bm
            stifn[nc[i, 4]] += HALF * sti
            stifn[nc[i, 5]] += HALF * sti

        # N=14 (IPERM1=6, IPERM2=7)
        if nc[i, 13] >= 0:
            mssx[i, 5] = bm
            stifn[nc[i, 13]] += sti
        else:
            mss[i, 5] += HALF * bm
            mss[i, 6] += HALF * bm
            stifn[nc[i, 5]] += HALF * sti
            stifn[nc[i, 6]] += HALF * sti

        # N=15 (IPERM1=7, IPERM2=8)
        if nc[i, 14] >= 0:
            mssx[i, 6] = bm
            stifn[nc[i, 14]] += sti
        else:
            mss[i, 6] += HALF * bm
            mss[i, 7] += HALF * bm
            stifn[nc[i, 6]] += HALF * sti
            stifn[nc[i, 7]] += HALF * sti

        # N=16 (IPERM1=8, IPERM2=5)
        if nc[i, 15] >= 0:
            mssx[i, 7] = bm
            stifn[nc[i, 15]] += sti
        else:
            mss[i, 7] += HALF * bm
            mss[i, 4] += HALF * bm
            stifn[nc[i, 7]] += HALF * sti
            stifn[nc[i, 4]] += HALF * sti
