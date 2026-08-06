def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    n = group.n
    thick = st["thick"]
    off = st["off"]
    alive = off > 0.0
    nip_max = st["nip_max"]

    # a group whose every slice runs one integration point is forced FLAT
    # (cbacoor.F line 444 'OR NPT==1')
    force_flat = np.zeros(n, dtype=bool)
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if len(st["zw"][isl][0]) == 1:
            force_flat[sl] = True

    g = _cbacoor(x[conn], v[conn], vr[conn], off, dt, force_flat)
    i_f, i_w = g["i_f"], g["i_w"]
    area = g["area"]

    # ---- CBADEFSH: constant assumed membrane shear ------------------------
    vdef3 = np.zeros(n)
    if len(i_f):
        vf_ = g["vxyz_f"][i_f]
        vdef3[i_f] = (g["y24n"][i_f] * vf_[:, 0, 1]
                      - g["y13n"][i_f] * vf_[:, 1, 1]
                      - g["x24n"][i_f] * vf_[:, 0, 0]
                      + g["x13n"][i_f] * vf_[:, 1, 0])
    if len(i_w):
        vw_ = g["vxyz_w"]
        vdef3[i_w] = (g["y24n"][i_w] * (vw_[:, 0, 1] - vw_[:, 2, 1])
                      + g["y13n"][i_w] * (-vw_[:, 1, 1] + vw_[:, 3, 1])
                      - g["x24n"][i_w] * (vw_[:, 0, 0] - vw_[:, 2, 0])
                      + g["x13n"][i_w] * (vw_[:, 1, 0] - vw_[:, 3, 0]))
    vdef3[~alive] = 0.0
    g["vdef3"] = vdef3
    volg = area * thick

    # CBAENERS (pre): + FOR3_mean_old * vdef3 * A*t*dt/2  (cbaforc3 l.566)
    st["eint"] += off * volg * dt * 0.5 * st["for_mean"][:, 2] * vdef3

    # per-slice constants
    gs_mod = np.zeros(n)                          # GS = G*SHF (0 if nip==1)
    bend_visc = np.ones(n)                        # cbavisc.F 'NPT /= 1' gate
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        nip = len(st["zw"][isl][0])
        gs_mod[sl] = 0.0 if nip == 1 else SHEAR_FACTOR * mat.G
        if nip == 1:
            bend_visc[sl] = 0.0

    sig = st["sig"]
    qsh = st["qshear"]
    forpg = st["forpg"]
    mompg = st["mompg"]
    epsp_old = st["epsp"].copy() if st["chk_fail"] else None

    vf = np.zeros((n, 3, 4))                     # local generalized forces
    vm = np.zeros((n, 2, 4))
    de = np.zeros(n)                             # eint increment
    dehg = np.zeros(n)                           # ehour increment (cbavisc)

    ops_f = []
    ops_w = []
    for ng in range(4):
        cdet = g["jac"][:, ng]
        vdef = np.zeros((n, 8))
        if len(i_f):
            bm, bc, vd = _flat_gp(g, ng)
            vd[~alive[i_f]] = 0.0
            vdef[i_f] = vd
            ops_f.append((bm, bc))
        else:
            ops_f.append(None)
        if len(i_w):
            bmw, bmfw, bfw, bcq, tc, vd = _warp_gp(g, ng)
            vd[~alive[i_w]] = 0.0
            vdef[i_w] = vd
            ops_w.append((bmw, bmfw, bfw, bcq, tc))
        else:
            ops_w.append(None)
        vdef[:, 2] = vdef3

        # strains (cbastra3.F): EXZ=VDEF4, EYZ=VDEF5
        exx = vdef[:, 0] * dt
        eyy = vdef[:, 1] * dt
        exy = vdef[:, 2] * dt
        exz = vdef[:, 3] * dt
        eyz = vdef[:, 4] * dt
        kxx = vdef[:, 5] * dt
        kyy = vdef[:, 6] * dt
        kxy = vdef[:, 7] * dt

        # CBAENER (pre): remove the per-GP old-stress shear work
        de -= 0.5 * off * thick * cdet * forpg[:, ng, 2] * exy

        # ---- layer stress updates + resultants ---------------------------
        npg_ = np.zeros((n, 3))                  # membrane N (force/length)
        mpg_ = np.zeros((n, 3))                  # moment M (moment/length)
        for isl, (sl, mat, prop) in enumerate(st["slices"]):
            zrel, wrel = st["zw"][isl]
            t_sl = thick[sl]
            dm = np.stack([exx[sl], eyy[sl], exy[sl]], axis=1)
            kap = np.stack([kxx[sl], kyy[sl], kxy[sl]], axis=1)
            for il in range(len(zrel)):
                k = ng * nip_max + il
                zk = zrel[il] * t_sl
                wk = wrel[il] * t_sl
                deps = dm + zk[:, None] * kap
                s_old = sig[sl, k, :].copy()
                s_new, _ = materials.shell_update(
                    mat, sig[sl, k, :], deps, st["epsp"][sl, k], dt,
                    _layer_extra(st, sl, k))
                if st["chk_fail"]:
                    _layer_failure(st, sl, mat, k, s_new, epsp_old,
                                   deps, dt)
                sig[sl, k, :] = s_new
                s_mid = 0.5 * (s_old + s_new)
                de[sl] += cdet[sl] * wk * np.einsum("nk,nk->n", s_mid, deps)
                npg_[sl] += wk[:, None] * s_new
                mpg_[sl] += (wk * zk)[:, None] * s_new
            # elastic transverse shear (per GP)
            qold = qsh[sl, ng].copy()
            dq = np.stack([exz[sl], eyz[sl]], axis=1)
            qsh[sl, ng] += gs_mod[sl][:, None] * dq
            de[sl] += cdet[sl] * t_sl * np.einsum(
                "nk,nk->n", 0.5 * (qold + qsh[sl, ng]), dq)

        # CBAENER (post): remove the per-GP NEW-stress (pre-viscous) work
        de -= 0.5 * off * thick * cdet * (npg_[:, 2] / np.maximum(
            thick, EM20)) * exy

        # ---- CBAVISC: dn numerical damping ------------------------------
        visc = _ONEP414 * off * st["amu"] * st["rho0"] * st["ssp0"] \
            * np.sqrt(np.maximum(cdet, 0.0))
        nu = st["nu0"]
        gg = 0.5 / (1.0 + nu)
        fx = visc * (vdef[:, 0] + nu * vdef[:, 1])
        fy = visc * (vdef[:, 1] + nu * vdef[:, 0])
        fxy = visc * vdef[:, 2] * gg
        npg_v = npg_.copy()
        npg_v[:, 0] += fx * thick
        npg_v[:, 1] += fy * thick
        npg_v[:, 2] += fxy * thick
        dv = cdet * thick * dt
        dehg += (fx * vdef[:, 0] + fy * vdef[:, 1]) * dv
        viscb = _ZEP3 * thick * visc * bend_visc
        mvx = viscb * (vdef[:, 5] + nu * vdef[:, 6])
        mvy = viscb * (vdef[:, 6] + nu * vdef[:, 5])
        mvxy = viscb * vdef[:, 7] * gg
        mpg_v = mpg_.copy()
        t2 = thick ** 2
        mpg_v[:, 0] += mvx * t2
        mpg_v[:, 1] += mvy * t2
        mpg_v[:, 2] += mvxy * t2
        dehg += (mvx * vdef[:, 5] + mvy * vdef[:, 6]
                 + mvxy * vdef[:, 7]) * dv * thick

        # persist the GBUF%FORPG / MOMPG state (stress / M/t^2 units)
        t_i = 1.0 / np.maximum(thick, EM20)
        forpg[:, ng, 0:3] = npg_v * t_i[:, None]
        forpg[:, ng, 3] = qsh[:, ng, 1]          # FOR(4) = sig_yz
        forpg[:, ng, 4] = qsh[:, ng, 0]          # FOR(5) = sig_zx
        mompg[:, ng] = mpg_v * (t_i ** 2)[:, None]

        # ---- CBAFORI: internal force assembly ----------------------------
        q_pg = qsh[:, ng] * thick[:, None]       # physical [q_xz, q_yz]
        if len(i_f):
            _fori_flat(vf, vm, g, ops_f[ng][0], ops_f[ng][1],
                       cdet, npg_v, mpg_v, q_pg)
        if len(i_w):
            _fori_warp(vf, vm, g, ops_w[ng], cdet, npg_v, mpg_v, q_pg)

    # ---- after the Gauss loop --------------------------------------------
    for_mean = forpg.mean(axis=1)                # GBUF%FOR (cbaforc3 962)
    st["for_mean"] = for_mean

    # CBAFORCT: constant membrane shear force from the MEAN resultant
    thoff = volg * for_mean[:, 2] * off
    if len(i_f):
        th_f = thoff[i_f]
        vf[i_f, 0, 0] += -th_f * g["x24n"][i_f]
        vf[i_f, 1, 0] += th_f * g["y24n"][i_f]
        vf[i_f, 0, 1] += th_f * g["x13n"][i_f]
        vf[i_f, 1, 1] += -th_f * g["y13n"][i_f]
    if len(i_w):
        th_w = thoff[i_w]
        sx1 = -th_w * g["x24n"][i_w]
        sy1 = th_w * g["y24n"][i_w]
        sx2 = th_w * g["x13n"][i_w]
        sy2 = -th_w * g["y13n"][i_w]
        vf[i_w, 0, 0] += sx1
        vf[i_w, 1, 0] += sy1
        vf[i_w, 0, 1] += sx2
        vf[i_w, 1, 1] += sy2
        vf[i_w, 0, 2] -= sx1
        vf[i_w, 1, 2] -= sy1
        vf[i_w, 0, 3] -= sx2
        vf[i_w, 1, 3] -= sy2

    # CBAENERS (post): + FOR3_mean_new * vdef3 * A*t*dt/2
    de += off * volg * dt * 0.5 * for_mean[:, 2] * vdef3

    # ---- element deletion from the layer flags ---------------------------
    if st["chk_fail"]:
        alive = _element_deletion_gpmajor(st)
        off = st["off"]
        if not alive.all():
            dead = ~alive
            sig[dead] = 0.0
            qsh[dead] = 0.0
            forpg[dead] = 0.0
            mompg[dead] = 0.0
            st["for_mean"][dead] = 0.0

    st["eint"] += de
    st["ehour"] += dehg

    # ---- CBAPROJ: local -> global, rigid projection, OFF -----------------
    fg, mg = _cbaproj(g, vf, vm, off)

    # accumulate NEGATED (cupdtn3.F: F -= F11)
    flat_idx = conn.reshape(-1)
    scatter_add3(fint, flat_idx, -fg.reshape(-1, 3))
    scatter_add3(mint, flat_idx, -mg.reshape(-1, 3))

    # ---- dt claim (cndt3.F): condensed LC * (sqrt(1+dn^2)-dn) / ssp ------
    viscdt = np.sqrt(1.0 + st["amu"] ** 2) - st["amu"]
    dt_e = g["lc"] * viscdt / np.maximum(st["ssp0"], EM20)
    return np.where(alive, dt_e, EP30)


def _fori_flat(vf, vm, g, bm, bc, cdet, npg, mpg, q_pg):
    """CBAFORI flat branch (cbafori.F lines 73-119): physical resultants,
    factors C2 -> CDET (forces) and C1 -> CDET (moments) after the
    FF = sigma / MM = M/t^2 unit reductions cancel THK0/TH12."""
    i = g["i_f"]
    c = cdet[i]
    n1, n2 = npg[i, 0], npg[i, 1]
    m1, m2, m3 = mpg[i, 0], mpg[i, 1], mpg[i, 2]
    qx, qy = q_pg[i, 0], q_pg[i, 1]
    cm1 = c * (bm[:, 6] * m2 + bm[:, 2] * m3)
    cm2 = c * (bm[:, 2] * m1 + bm[:, 6] * m3)
    cc1 = c * (bc[:, 14] * qx + bc[:, 15] * qy)
    cc2 = c * (bc[:, 16] * qx + bc[:, 17] * qy)
    vf[i, 0, 0] += c * bm[:, 0] * n1
    vf[i, 1, 0] += c * bm[:, 4] * n2
    vf[i, 2, 0] += c * (bc[:, 0] * qx + bc[:, 1] * qy)
    vm[i, 0, 0] += c * (bc[:, 2] * qx + bc[:, 3] * qy) \
        - c * (bm[:, 4] * m2 + bm[:, 0] * m3)
    vm[i, 1, 0] += c * (bc[:, 4] * qx + bc[:, 5] * qy) \
        + c * (bm[:, 0] * m1 + bm[:, 4] * m3)
    vf[i, 0, 2] += c * bm[:, 2] * n1
    vf[i, 1, 2] += c * bm[:, 6] * n2
    vf[i, 2, 2] += c * (bc[:, 12] * qx + bc[:, 13] * qy)
    vm[i, 0, 2] += cc1 - cm1
    vm[i, 1, 2] += cc2 + cm2
    vf[i, 0, 1] += c * bm[:, 1] * n1
    vf[i, 1, 1] += c * bm[:, 5] * n2
    vf[i, 2, 1] += c * (bc[:, 6] * qx + bc[:, 7] * qy)
    vm[i, 0, 1] += c * (bc[:, 8] * qx + bc[:, 9] * qy) \
        - c * (bm[:, 5] * m2 + bm[:, 1] * m3)
    vm[i, 1, 1] += c * (bc[:, 10] * qx + bc[:, 11] * qy) \
        + c * (bm[:, 1] * m1 + bm[:, 5] * m3)
    # slot 4 = -slot 3 for the forces (assigned per cycle in the Fortran,
    # equivalent to mirroring the accumulation)
    vf[i, 0, 3] = -vf[i, 0, 2]
    vf[i, 1, 3] = -vf[i, 1, 2]
    vf[i, 2, 3] = -vf[i, 2, 2]
    vm[i, 0, 3] += cc1 + cm1
    vm[i, 1, 3] += cc2 - cm2


def _fori_warp(vf, vm, g, ops, cdet, npg, mpg, q_pg):
    """CBAFORI warped branch (cbafori.F lines 120-272)."""
    iw = g["i_w"]
    bmw, bmfw, bfw, bcq, tc = ops
    c = cdet[iw]
    nn = npg[iw]
    mm = mpg[iw]
    qx, qy = q_pg[iw, 0], q_pg[iw, 1]
    bcx = tc[:, 0, 0] * qx + tc[:, 0, 1] * qy
    bcy = tc[:, 1, 0] * qx + tc[:, 1, 1] * qy
    for j in range(4):
        for comp in range(3):
            vf[iw, comp, j] += c * (
                bmw[:, j, comp, 0] * nn[:, 0] + bmw[:, j, comp, 1] * nn[:, 1]
                + bcq[:, j, comp, 0] * bcx + bcq[:, j, comp, 1] * bcy
                + bmfw[:, j, comp, 0] * mm[:, 0]
                + bmfw[:, j, comp, 1] * mm[:, 1]
                + bmfw[:, j, comp, 2] * mm[:, 2])
        for a in range(2):
            vm[iw, a, j] += c * (
                bcq[:, j, 3 + a, 0] * bcx + bcq[:, j, 3 + a, 1] * bcy
                + bfw[:, j, a, 0] * mm[:, 0] + bfw[:, j, a, 1] * mm[:, 1]
                + bfw[:, j, a, 2] * mm[:, 2])


def _cbaproj