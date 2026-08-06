import sys

content = open('pyradioss/elements/shell_qeph.py').read()

pre_post = """def _pre(x_conn, v_conn, vr_conn, dt, st_npt1, alive):
    G = _geometry(x_conn)
    v13, v24, vhi, rl, plat, vqn, di, db = _kinematics(G, v_conn, vr_conn, dt, st_npt1)
    vdef, vhg = _rates(G, v13, v24, vhi, rl, alive)
    return G, vdef, vhg, plat, vqn, di, db

def _post(G, thick, Nres, Mres, qres, st, vhg, dt, alive, plat, vqn, di, db):
    VF, VM = _fint_const(G, thick, Nres, Mres, qres)
    if not alive.all():
        VF[~alive] = 0.0
        VM[~alive] = 0.0
    _fint_stab(G, st, vhg, dt, alive, Nres, Mres, VF, VM, thick)
    fg, mg = _project(G, VF, VM, plat, vqn, di, db)
    
    import numpy as np
    visc = np.sqrt(1.0 + st["amu"] ** 2) - st["amu"]
    dt_e = np.where(st["cspd"] > 0.0,
                    visc * G["ll"] / np.maximum(st["cspd"], 1e-20), 1e30)
    return fg, mg, dt_e

def forces("""

new_content = content.replace('def forces(', pre_post)

forces_body = """def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    n = group.n
    thick = st["thick"]
    alive = st["off"] > 0.0

    from pyradioss.accel import get as accel_get
    jit_pre = accel_get("qeph_pre")
    if jit_pre is not None:
        G, vdef, vhg, plat, vqn, di, db = jit_pre(x[conn], v[conn], vr[conn], dt, st["npt1"], alive)
    else:
        G, vdef, vhg, plat, vqn, di, db = _pre(x[conn], v[conn], vr[conn], dt, st["npt1"], alive)

    dm = vdef[:, 0:3] * dt
    gsr2 = vdef[:, 3:5]
    kap = vdef[:, 5:8] * dt

    sig = st["sig"]
    epsp_old = st["epsp"].copy() if st["chk_fail"] else None
    Nres = np.zeros((n, 3))
    Mres = np.zeros((n, 3))
    de_layers = np.zeros(n)
    nip_of = []
    ortho_all = st.get("ortho")
    area = G["area"]
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        zrel, wrel = st["zw"][isl]
        nip_of.append(len(zrel))
        t_sl = thick[sl]
        cs = ortho_all[sl] if (ortho_all is not None and getattr(
            prop, "type", 0) in shell_ortho.ORTHO_PROP_TYPES) else None
        for k in range(len(zrel)):
            zk = zrel[k] * t_sl
            wk = wrel[k] * t_sl
            deps = dm[sl] + zk[:, None] * kap[sl]
            if cs is not None:
                deps = shell_ortho.rot_strain_e2m(deps, cs)
            s_old = sig[sl, k, :].copy()
            s_new, _ = materials.shell_update(
                mat, sig[sl, k, :], deps, st["epsp"][sl, k], dt,
                _layer_extra(st, sl, k))
            if st["chk_fail"]:
                _layer_failure(st, sl, mat, k, s_new, epsp_old, deps, dt)
            sig[sl, k, :] = s_new
            s_mid = 0.5 * (s_old + s_new)
            de_layers[sl] += wk * np.einsum("nk,nk->n", s_mid, deps)
            s_res = shell_ortho.rot_stress_m2e(s_new, cs) if cs is not None else s_new
            Nres[sl] += wk[:, None] * s_res
            Mres[sl] += (wk * zk)[:, None] * s_res
        qold = st["qshear"][sl].copy()
        st["qshear"][sl] += st["gs"][sl][:, None] * gsr2[sl] * dt
        de_layers[sl] += t_sl * np.einsum(
            "nk,nk->n", 0.5 * (qold + st["qshear"][sl]), gsr2[sl] * dt)

    if st["chk_fail"]:
        alive = _element_deletion(st, nip_of)
        if not alive.all():
            dead = ~alive
            Nres[dead] = 0.0
            Mres[dead] = 0.0
            sig[dead] = 0.0
            st["qshear"][dead] = 0.0
            st["hgstr"][dead] = 0.0
    qres = st["qshear"] * thick[:, None]
    st["eint"] += area * de_layers

    jit_post = accel_get("qeph_post")
    if jit_post is not None:
        fg, mg, dt_e = jit_post(G, thick, Nres, Mres, qres, st, vhg, dt, alive, plat, vqn, di, db)
    else:
        fg, mg, dt_e = _post(G, thick, Nres, Mres, qres, st, vhg, dt, alive, plat, vqn, di, db)

    flat = conn.reshape(-1)
    scatter_add3(fint, flat, -fg.reshape(-1, 3))
    scatter_add3(mint, flat, -mg.reshape(-1, 3))

    return np.where(alive, dt_e, EP30)
"""

idx1 = new_content.find('def forces(')
new_content = new_content[:idx1] + forces_body
open('pyradioss/elements/shell_qeph.py', 'w').write(new_content)
print('Patched shell_qeph.py')
