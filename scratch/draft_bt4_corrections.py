import numpy as np

def generate_shell_bt4_corrections():
    code = """
    # M50: IHBE == 2/3 and 4 branches (from cdefo3.F)
    ihbe_mask = st.get("_ihbe_mask")
    if ihbe_mask is None:
        ihbe_mask = np.zeros(n, dtype=int)
        for sl, mat, prop in group.state["slices"]:
            card = int(prop.params.get("ishell", 0))
            ihbe = {0: 0, 1: 1, 2: 0, 4: 4}.get(card, card - 1 if card > 0 else 0)
            ihbe_mask[sl] = ihbe
        st["_ihbe_mask"] = ihbe_mask

    mask23 = ((ihbe_mask == 2) | (ihbe_mask == 3)) * alive
    if mask23.any() and not st.get("_impl_static_hg"):
        idx = mask23
        z2 = xl[idx, 1, 2] - xl[idx, 0, 2]
        gzx = np.sum(B1[idx] * V[idx, :, 2], axis=1)
        gzy = np.sum(B2[idx] * V[idx, :, 2], axis=1)
        exzz2 = gzx * z2
        eyzz2 = gzy * z2
        dt1v4 = 0.5 * dt
        exz2 = gzx * gzx * dt1v4
        eyz2 = gzy * gzy * dt1v4

        dm[idx, 0] -= 2.0 * exz2
        dm[idx, 1] -= 2.0 * eyz2

        zzz = np.zeros_like(exz2)
        ihbe2 = (ihbe_mask[idx] == 2)
        if ihbe2.any():
            zzz[ihbe2] = (exz2[ihbe2] + eyz2[ihbe2]) * z2[ihbe2]

        x_rel = xl[idx, :, 0] - xl[idx, 0:1, 0]
        y_rel = xl[idx, :, 1] - xl[idx, 0:1, 1]

        corr_x = np.zeros((np.count_nonzero(mask23), 4))
        corr_x[:, 1] = exzz2
        corr_x[:, 3] = exzz2
        corr_x -= exz2[:, None] * x_rel
        V[idx, :, 0] += corr_x

        corr_y = np.zeros((np.count_nonzero(mask23), 4))
        corr_y[:, 1] = eyzz2
        corr_y[:, 3] = eyzz2
        corr_y -= eyz2[:, None] * y_rel
        V[idx, :, 1] += corr_y

        corr_z = np.zeros((np.count_nonzero(mask23), 4))
        corr_z[:, 1] = -zzz
        corr_z[:, 3] = -zzz
        corr_z -= gzx[:, None] * x_rel + gzy[:, None] * y_rel
        V[idx, :, 2] += corr_z

    mask4 = (ihbe_mask == 4) * alive
    if mask4.any() and not st.get("_impl_static_hg"):
        idx = mask4
        z2 = xl[idx, 1, 2] - xl[idx, 0, 2]
        zz2 = 0.5 * z2
        gzx = np.sum(B1[idx] * V[idx, :, 2], axis=1)
        gzy = np.sum(B2[idx] * V[idx, :, 2], axis=1)
        exzz2 = gzx * zz2
        eyzz2 = gzy * zz2
        dt1v4 = 0.5 * dt
        exz2 = gzx * gzx * dt1v4
        eyz2 = gzy * gzy * dt1v4

        px1 = B1[idx, 0] * area[idx]
        px2 = B1[idx, 1] * area[idx]
        py1 = B2[idx, 0] * area[idx]
        py2 = B2[idx, 1] * area[idx]

        dm[idx, 0] += exz2
        dm[idx, 1] += eyz2

        corr_x = np.zeros((np.count_nonzero(mask4), 4))
        corr_x[:, 0] = -exzz2 - exz2 * py2
        corr_x[:, 2] = -exzz2 + exz2 * py2
        corr_x[:, 1] =  exzz2 + exz2 * py1
        corr_x[:, 3] =  exzz2 - exz2 * py1
        V[idx, :, 0] += corr_x

        corr_y = np.zeros((np.count_nonzero(mask4), 4))
        corr_y[:, 0] = -eyzz2 + eyz2 * px2
        corr_y[:, 2] = -eyzz2 - eyz2 * px2
        corr_y[:, 1] =  eyzz2 - eyz2 * px1
        corr_y[:, 3] =  eyzz2 + eyz2 * px1
        V[idx, :, 1] += corr_y
"""
    return code

if __name__ == "__main__":
    print(generate_shell_bt4_corrections())
