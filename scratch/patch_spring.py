import re

with open('pyradioss/elements/spring.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Update init_group
init_old = '''    is6 = np.isin(kind, list(spring_general.SPRING_PROP_TYPES))
    idx4 = np.where(~is6)[0]
    idx6 = np.where(is6)[0]'''

init_new = '''    is6 = np.isin(kind, list(spring_general.SPRING_PROP_TYPES))
    is32 = (kind == 32)
    idx4 = np.where(~is6 & ~is32)[0]
    idx6 = np.where(is6)[0]
    idx32 = np.where(is32)[0]
    
    if len(idx32):
        st["stif0"] = np.zeros(n)
        st["stif1"] = np.zeros(n)
        st["ityp"] = np.zeros(n, dtype=np.int64)
        st["f1"] = np.zeros(n)
        st["d1"] = np.zeros(n)
        st["scale_t"] = np.zeros(n)
        st["scale_d"] = np.zeros(n)
        st["scale_f"] = np.zeros(n)
        st["sens_id"] = np.zeros(n, dtype=np.int64)
        st["fct_id1"] = np.zeros(n, dtype=np.int64)
        st["fct_id2"] = np.zeros(n, dtype=np.int64)
        st["ilock"] = np.zeros(n, dtype=np.int64)
        st["uvar1"] = np.zeros(n)
        st["uvar2"] = np.zeros(n)
        st["uvar3"] = np.zeros(n)
        for sl, mat, prop in st["slices"]:
            if getattr(prop, "type", 4) == 32:
                st["stif0"][sl] = prop.params["stif0"]
                st["stif1"][sl] = prop.params["stif1"]
                st["ityp"][sl] = prop.params["ityp"]
                st["f1"][sl] = prop.params["f1"]
                st["d1"][sl] = prop.params["d1"]
                st["scale_t"][sl] = prop.params["scale_t"]
                st["scale_d"][sl] = prop.params["scale_d"]
                st["scale_f"][sl] = prop.params["scale_f"]
                st["sens_id"][sl] = prop.params["sens_id"]
                st["fct_id1"][sl] = prop.params["fct_id1"]
                st["fct_id2"][sl] = prop.params["fct_id2"]
                st["ilock"][sl] = prop.params["ilock"]'''

text = text.replace(init_old, init_new)

init_update_old = '''    st.update(L0=L0, mass=mass, k=k, cdamp=cdamp,
              force=np.zeros(n), eint=np.zeros(n), ehour=np.zeros(n),
              idx4=idx4, idx6=idx6)'''
init_update_new = '''    st.update(L0=L0, mass=mass, k=k, cdamp=cdamp,
              force=np.zeros(n), eint=np.zeros(n), ehour=np.zeros(n),
              idx4=idx4, idx6=idx6, idx32=idx32, model=model)'''
text = text.replace(init_update_old, init_update_new)

# Insert _forces_axial_type32 and update forces
forces_old = '''def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    idx6 = st.get("idx6")
    if idx6 is None or len(idx6) == 0:
        # pure axial group (every existing spring deck): the whole-group
        # vectorized path, bit-identical to the pre-M38 kernel
        return _forces_axial(group, x, v, dt, fint, slice(None))
    dtc = np.full(group.n, EP30)
    idx4 = st["idx4"]
    if len(idx4):
        dtc[idx4] = _forces_axial(group, x, v, dt, fint, idx4)
    dtc[idx6] = spring_general.forces6(group, x, v, vr, dt, fint, mint, idx6)
    return dtc'''

forces_new = '''def _forces_axial_type32(group, x, v, dt, fint, idx):
    st = group.state
    model = st["model"]
    t = getattr(model, "t", 0.0)
    sensors = getattr(model, "sensors_state", None)
    
    conn = group.conn[idx]
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    L = np.maximum(norm3(dx), EM20)
    a = dx / L[:, None]
    Ldot = np.einsum("nb,nb->n", v[conn[:, 1]] - v[conn[:, 0]], a)

    F = st["force"][idx].copy()
    stif0 = st["stif0"][idx]
    stif1 = st["stif1"][idx]
    scale_t = st["scale_t"][idx]
    scale_d = st["scale_d"][idx]
    scale_f = st["scale_f"][idx]
    ityp = st["ityp"][idx]
    f1 = st["f1"][idx]
    d1 = st["d1"][idx]
    ilock = st["ilock"][idx]
    sens_id = st["sens_id"][idx]
    
    uvar1 = st["uvar1"][idx]
    uvar2 = st["uvar2"][idx]
    uvar3 = st["uvar3"][idx]
    
    tacti = np.zeros(len(idx))
    iact = np.ones(len(idx), dtype=bool)
    
    if sensors is not None:
        for i, s_id in enumerate(sens_id):
            if s_id > 0:
                tf = sensors.fire_time.get(s_id, t)
                tacti[i] = max(0.0, t - tf)
                if tacti[i] == 0.0:
                    iact[i] = False
            else:
                tacti[i] = t
    else:
        tacti[:] = t
    
    not_act = ~iact
    if np.any(not_act):
        uvar2[not_act] = 0.0
        F[not_act] += stif0[not_act] * dt * Ldot[not_act]
        st["k"][idx[not_act]] = stif0[not_act]
        
    act = iact
    if np.any(act):
        mask_just_act = act & (uvar2 == 0.0)
        uvar1[mask_just_act] = 0.0
        uvar2[mask_just_act] = 1.0
        
        uvar1[act] += dt * Ldot[act]
        F[act] += stif0[act] * dt * Ldot[act]
        st["k"][idx[act]] = stif0[act]
        
        for it in (1, 2, 3, 4):
            mask = act & (ityp == it)
            if not np.any(mask): continue
            
            X = uvar1[mask]
            cur_F = F[mask]
            cur_ilock = ilock[mask]
            cur_d1 = d1[mask]
            cur_uvar3 = uvar3[mask]
            
            if it == 1:
                FF = f1[mask] + stif1[mask] * X
                cur_uvar3 = np.where((cur_F > FF) & (cur_ilock == 2), 1.0, cur_uvar3)
                cur_F = np.where((FF > 0) & (cur_uvar3 == 0.0), np.maximum(FF, cur_F), cur_F)
                
            elif it == 2:
                FF = np.zeros(len(X))
                for local_i, global_i in enumerate(np.where(mask)[0]):
                    func = model.functions.get(st["fct_id1"][idx[global_i]])
                    if func:
                        FF[local_i] = scale_f[global_i] * func.eval(X[local_i] * scale_d[global_i])
                cur_uvar3 = np.where(((X < cur_d1) & (cur_d1 != 0.0)) | ((cur_F > FF) & (cur_ilock == 2)), 1.0, cur_uvar3)
                cur_F = np.where((FF > 0) & (cur_uvar3 == 0.0), np.maximum(FF, cur_F), cur_F)
                
            elif it == 3:
                F0 = np.zeros(len(X))
                for local_i, global_i in enumerate(np.where(mask)[0]):
                    func = model.functions.get(st["fct_id2"][idx[global_i]])
                    if func:
                        F0[local_i] = scale_f[global_i] * func.eval(tacti[global_i] * scale_t[global_i])
                cur_uvar3 = np.where(((X < cur_d1) & (cur_d1 != 0.0)) | ((cur_F > F0) & (cur_ilock == 2)), 1.0, cur_uvar3)
                cur_F = np.where((F0 > 0) & (cur_uvar3 == 0.0), np.maximum(F0, cur_F), cur_F)
                
            elif it == 4:
                F0 = np.zeros(len(X))
                FF = np.zeros(len(X))
                for local_i, global_i in enumerate(np.where(mask)[0]):
                    f2 = model.functions.get(st["fct_id2"][idx[global_i]])
                    f1_obj = model.functions.get(st["fct_id1"][idx[global_i]])
                    if f2:
                        F0[local_i] = scale_f[global_i] * f2.eval(tacti[global_i] * scale_t[global_i])
                    if f1_obj:
                        FF[local_i] = F0[local_i] * f1_obj.eval(X[local_i] * scale_d[global_i])
                cur_uvar3 = np.where(((X < cur_d1) & (cur_d1 != 0.0)) | ((cur_F > FF) & (cur_ilock == 2)), 1.0, cur_uvar3)
                cur_F = np.where((FF > 0) & (cur_uvar3 == 0.0), np.maximum(FF, cur_F), cur_F)
                
            F[mask] = cur_F
            uvar3[mask] = cur_uvar3

    F_old = st["force"][idx].copy()
    st["eint"][idx] += 0.5 * (F_old + F) * Ldot * dt
    st["force"][idx] = F
    st["uvar1"][idx] = uvar1
    st["uvar2"][idx] = uvar2
    st["uvar3"][idx] = uvar3
    
    fvec = F[:, None] * a
    np.add.at(fint, conn[:, 0], fvec)
    np.add.at(fint, conn[:, 1], -fvec)
    
    k_dt = np.maximum(st["k"][idx], EM20)
    omega = 2.0 * np.sqrt(k_dt / st["mass"][idx])
    dt_crit = 2.0 / omega
    return np.where(st["k"][idx] > 0, dt_crit, EP30)

def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    idx6 = st.get("idx6")
    idx4 = st.get("idx4")
    idx32 = st.get("idx32")
    if (idx6 is None or len(idx6) == 0) and (idx32 is None or len(idx32) == 0):
        # pure axial TYPE4 group
        return _forces_axial(group, x, v, dt, fint, slice(None))
    dtc = np.full(group.n, EP30)
    if idx4 is not None and len(idx4):
        dtc[idx4] = _forces_axial(group, x, v, dt, fint, idx4)
    if idx32 is not None and len(idx32):
        dtc[idx32] = _forces_axial_type32(group, x, v, dt, fint, idx32)
    if idx6 is not None and len(idx6):
        dtc[idx6] = spring_general.forces6(group, x, v, vr, dt, fint, mint, idx6)
    return dtc'''

text = text.replace(forces_old, forces_new)

with open('pyradioss/elements/spring.py', 'w', encoding='utf-8') as f:
    f.write(text)
