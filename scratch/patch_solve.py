import re

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'r', encoding='utf8') as f:
    text = f.read()

target = '''    for c in range(n):
        for cp in range(c + 1, n):
            if not (nz[c] and nz[cp]):
                rho[c, cp] = rho[cp, c] = 0.0
                continue
            A1 = kappa[c] * kappa[cp]
            A2 = A1 * 2.0 * h3[c] * h3[cp]
            A3 = A1 * 6.0 * h4[c] * h4[cp]
            r, _res, feasible = _solve_pair_rho(A1, A2, A3, float(R[c, cp]))
            rho[c, cp] = rho[cp, c] = r
            if not feasible:
                n_infeasible += 1'''

replacement = '''    for c in range(n):
        for cp in range(c + 1, n):
            if not (nz[c] and nz[cp]):
                rho[c, cp] = rho[cp, c] = 0.0
                continue
            if copula == "t":
                nu = copula_params if copula_params is not None else 4.0
                r = _solve_pair_rho_t_copula(float(R[c, cp]), h3[c], h4[c], kappa[c], h3[cp], h4[cp], kappa[cp], nu)
                # t-copula MC solve doesn't return feasible status, assume feasible unless r is exactly bounded
                feasible = abs(r) < 0.99
            else:
                A1 = kappa[c] * kappa[cp]
                A2 = A1 * 2.0 * h3[c] * h3[cp]
                A3 = A1 * 6.0 * h4[c] * h4[cp]
                r, _res, feasible = _solve_pair_rho(A1, A2, A3, float(R[c, cp]))
            rho[c, cp] = rho[cp, c] = r
            if not feasible:
                n_infeasible += 1'''

text = text.replace(target, replacement)

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'w', encoding='utf8') as f:
    f.write(text)
