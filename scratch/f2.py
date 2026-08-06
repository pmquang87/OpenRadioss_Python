def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    n = group.n
    xe = x[conn]
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

    from pyradioss.accel import get as accel_get
    jit_pre = accel_get("qbat_pre")
    if jit_pre is not None:
        (E, area, lc, vdef3, cdet, vdef, i_f, i_w, bm_f, bc_f,
         bmw_w, bmfw_w, bfw_w, bcq_w, tc_w, vqn_w, corel_w, di_w,
         x13n_f, x24n_f, y13n_f, y24n_f,
         x13n_w, x24n_w, y13n_w, y24n_w) = jit_pre(xe, v[conn], vr[conn], off, dt, force_flat)
    else:
        (E, area, lc, vdef3, cdet, vdef, i_f, i_w, bm_f, bc_f,
         bmw_w, bmfw_w, bfw_w, bcq_w, tc_w, vqn_w, corel_w, di_w,
         x13n_f, x24n_f, y13n_f, y24n_f,
         x13n_w, x24n_w, y13n_w, y24n_w) = _pre(xe, v[conn], vr[conn], off, dt, force_flat)

    volg = area * thick
    st["eint"] += off * volg * dt * 0.5 * st["for_mean"][:, 2] * vdef3

    # per-slice constants
    gs_mod = np.zeros(n)
    bend_visc = np.ones(n)
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

    de = np.zeros(n)
    dehg = np.zeros(n)

    for ng in range(4):
        exx = vdef[:, ng, 0] * dt
        eyy = vdef[:, ng, 1] * dt
        exy = vdef[:, ng, 2] * dt
        exz = vdef[:, ng, 3] * dt
        eyz = vdef[:, ng, 4] * dt
        kxx = vdef[:, ng, 5] * dt
        kyy = vdef[:, ng, 6] * dt
        kxy = vdef[:, ng, 7] * dt

        de -= 0.5 * off * thick * cdet[:, ng] * forpg[:, ng, 2] * exy

        npg_ = np.zeros((n, 3))
        mpg_ = np.zeros((n, 3))
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
                de[sl] += cdet[sl, ng] * wk * np.einsum("nk,nk->n", s_mid, deps)
                npg_[sl] += wk[:, None] * s_new
                mpg_[sl] += (wk * zk)[:, None] * s_new
            qold = qsh[sl, ng].copy()
            dq = np.stack([exz[sl], eyz[sl]], axis=1)
            qsh[sl, ng] += gs_mod[sl][:, None] * dq
            de[sl] += cdet[sl, ng] * t_sl * np.einsum(
                "nk,nk->n", 0.5 * (qold + qsh[sl, ng]), dq)

        de -= 0.5 * off * thick * cdet[:, ng] * (npg_[:, 2] / np.maximum(
            thick, EM20)) * exy

        visc = _ONEP414 * off * st["amu"] * st["rho0"] * st["ssp0"]             * np.sqrt(np.maximum(cdet[:, ng], 0.0))
        nu = st["nu0"]
        gg = 0.5 / (1.0 + nu)
        fx = visc * (vdef[:, ng, 0] + nu * vdef[:, ng, 1])
        fy = visc * (vdef[:, ng, 1] + nu * vdef[:, ng, 0])
        fxy = visc * vdef[:, ng, 2] * gg
        npg_v = npg_.copy()
        npg_v[:, 0] += fx * thick
        npg_v[:, 1] += fy * thick
        npg_v[:, 2] += fxy * thick
        dv = cdet[:, ng] * thick * dt
        dehg += (fx * vdef[:, ng, 0] + fy * vdef[:, ng, 1]) * dv
        viscb = _ZEP3 * thick * visc * bend_visc
        mvx = viscb * (vdef[:, ng, 5] + nu * vdef[:, ng, 6])
        mvy = viscb * (vdef[:, ng, 6] + nu * vdef[:, ng, 5])
        mvxy = viscb * vdef[:, ng, 7] * gg
        mpg_v = mpg_.copy()
        t2 = thick ** 2
        mpg_v[:, 0] += mvx * t2
        mpg_v[:, 1] += mvy * t2
        mpg_v[:, 2] += mvxy * t2
        dehg += (mvx * vdef[:, ng, 5] + mvy * vdef[:, ng, 6]
                 + mvxy * vdef[:, ng, 7]) * dv * thick

        t_i = 1.0 / np.maximum(thick, EM20)
        forpg[:, ng, 0:3] = npg_v * t_i[:, None]
        forpg[:, ng, 3] = qsh[:, ng, 1]
        forpg[:, ng, 4] = qsh[:, ng, 0]
        mompg[:, ng] = mpg_v * (t_i ** 2)[:, None]

    for_mean = forpg.mean(axis=1)
    st["for_mean"] = for_mean

    jit_post = accel_get("qbat_post")
    if jit_post is not None:
        fg, mg = jit_post(
            n, E, off, thick, volg, forpg, mompg, for_mean, cdet,
            i_f, i_w, bm_f, bc_f, bmw_w, bmfw_w, bfw_w, bcq_w, tc_w, vqn_w, corel_w, di_w,
            x13n_f, x24n_f, y13n_f, y24n_f,
            x13n_w, x24n_w, y13n_w, y24n_w
        )
    else:
        fg, mg = _post(
            n, E, off, thick, volg, forpg, mompg, for_mean, cdet,
            i_f, i_w, bm_f, bc_f, bmw_w, bmfw_w, bfw_w, bcq_w, tc_w, vqn_w, corel_w, di_w,
            x13n_f, x24n_f, y13n_f, y24n_f,
            x13n_w, x24n_w, y13n_w, y24n_w
        )

    de += off * volg * dt * 0.5 * for_mean[:, 2] * vdef3

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

    flat_idx = conn.reshape(-1)
    scatter_add3(fint, flat_idx, -fg.reshape(-1, 3))
    scatter_add3(mint, flat_idx, -mg.reshape(-1, 3))

    viscdt = np.sqrt(1.0 + st["amu"] ** 2) - st["amu"]
    dt_e = lc * viscdt / np.maximum(st["ssp0"], EM20)
    return np.where(alive, dt_e, EP30)

def _fori_flat