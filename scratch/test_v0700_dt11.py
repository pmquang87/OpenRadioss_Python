import sys
import numpy as np
from pathlib import Path
from pyradioss.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.deck_reader import read_deck
from tests.test_m40_residuals import _wedge_starter
from pyradioss.starter.initialization import build_element_groups, initialize_elements_and_mass, resolve_materials, resolve_node_groups, resolve_surfaces
from pyradioss.elements.solid_hexa8 import _exact_dt_factor, _geometry, _char_length

try:
    deck_str = _wedge_starter()
    tmp = Path('scratch/tmp_wedge2.rad')
    tmp.write_text(deck_str)
    
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(tmp)), model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    resolve_surfaces(model, log)
    resolve_materials(model, log)
    initialize_elements_and_mass(model, log)

    for name, group in model.element_groups():
        if name == 'bricks':
            xe = model.x0[group.conn]
            dndx0, vol = _geometry(xe)
            lc0 = _char_length(xe, vol)
            
            # Replicate _exact_dt_factor without the np.minimum
            slices = group.state["slices"]
            n = len(vol)
            b = dndx0
            S = np.einsum("nia,nib->nab", b, b)
            BBt = np.zeros((n, 6, 6))
            Sxx, Syy, Szz = S[:, 0, 0], S[:, 1, 1], S[:, 2, 2]
            Sxy, Syz, Sxz = S[:, 0, 1], S[:, 1, 2], S[:, 0, 2]
            BBt[:, 0, 0], BBt[:, 1, 1], BBt[:, 2, 2] = Sxx, Syy, Szz
            BBt[:, 3, 3] = Sxx + Syy
            BBt[:, 4, 4] = Syy + Szz
            BBt[:, 5, 5] = Sxx + Szz
            BBt[:, 0, 3] = BBt[:, 3, 0] = Sxy
            BBt[:, 1, 3] = BBt[:, 3, 1] = Sxy
            BBt[:, 0, 5] = BBt[:, 5, 0] = Sxz
            BBt[:, 2, 5] = BBt[:, 5, 2] = Sxz
            BBt[:, 1, 4] = BBt[:, 4, 1] = Syz
            BBt[:, 2, 4] = BBt[:, 4, 2] = Syz
            BBt[:, 3, 4] = BBt[:, 4, 3] = Sxz
            BBt[:, 3, 5] = BBt[:, 5, 3] = Syz
            BBt[:, 4, 5] = BBt[:, 5, 4] = Sxy

            c = np.zeros(n)
            w2max = np.zeros(n)
            for sl, mat, prop in slices:
                lam, G = mat.lam, mat.G
                C = np.array([
                    [lam + 2 * G, lam, lam, 0, 0, 0],
                    [lam, lam + 2 * G, lam, 0, 0, 0],
                    [lam, lam, lam + 2 * G, 0, 0, 0],
                    [0, 0, 0, G, 0, 0],
                    [0, 0, 0, 0, G, 0],
                    [0, 0, 0, 0, 0, G],
                ])
                eig = np.linalg.eigvals(C[None, :, :] @ BBt[sl])
                w2max[sl] = (8.0 / mat.rho0) * eig.real.max(axis=1)
                c[sl] = np.sqrt((mat.K + 4.0 * mat.G / 3.0) / mat.rho0)

            dt_exact = 2.0 / np.sqrt(np.maximum(w2max, 1e-20))
            raw_fac = dt_exact / (lc0 / c)
            print("raw_fac:", raw_fac)
            print("lc0:", lc0)
            print("dt_exact:", dt_exact)
except Exception as e:
    import traceback
    traceback.print_exc()
