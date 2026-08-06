with open('pyradioss/elements/shell_qeph.py', 'r') as f:
    text = f.read()

text = text.replace(
    'G, vdef, vhg, plat, vqn, di, db = jit_pre(x[conn], v[conn], vr[conn], dt, st["npt1"], alive)',
    '''vdef, vhg, plat, vqn, di, db, E, area, a_i, z1, corx, cory, x13, x24, y13, y24, mx13, mx23, mx34, my13, my23, my34, l13, l24, ll, lm = jit_pre(x[conn], v[conn], vr[conn], dt, st["npt1"], alive)
        G = dict(E=E, area=area, a_i=a_i, z1=z1, corx=corx, cory=cory, x13=x13, x24=x24, y13=y13, y24=y24, mx13=mx13, mx23=mx23, mx34=mx34, my13=my13, my23=my23, my34=my34, l13=l13, l24=l24, ll=ll, lm=lm)'''
)

text = text.replace(
    'fg, mg, dt_e = jit_post(G, thick, Nres, Mres, qres, st, vhg, dt, alive, plat, vqn, di, db)',
    'fg, mg, dt_e = jit_post(thick, Nres, Mres, qres, st["amu"], st["cspd"], st["yld"], st["fmat"], vhg, dt, alive, plat, vqn, di, db, E, area, a_i, z1, corx, cory, x13, x24, y13, y24, mx13, mx23, mx34, my13, my23, my34, l13, l24, ll, lm)'
)

with open('pyradioss/elements/shell_qeph.py', 'w') as f:
    f.write(text)
