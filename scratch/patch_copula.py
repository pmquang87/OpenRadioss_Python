import re

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'r', encoding='utf8') as f:
    text = f.read()

# 1. induced_projection_moments
target = 'def induced_projection_moments(M0, proj, gamma3, gamma4, model="winterstein",\n                               return_components=False, underlying_R=None):'
replacement = 'def induced_projection_moments(M0, proj, gamma3, gamma4, model="winterstein",\n                               return_components=False, underlying_R=None,\n                               copula="gaussian", copula_params=None):'
text = text.replace(target, replacement)

# 2. solve_underlying_correlation
target2 = 'def solve_underlying_correlation(M0, gamma3, gamma4, model="winterstein",\n                                 repair=True, var_floor=1e-9):'
replacement2 = 'def solve_underlying_correlation(M0, gamma3, gamma4, model="winterstein",\n                                 repair=True, var_floor=1e-9,\n                                 copula="gaussian", copula_params=None):'
text = text.replace(target2, replacement2)

# 3. synthesize_joint_nongaussian_history
target3 = 'def synthesize_joint_nongaussian_history(omega, Scross, durations, fc, bw, seed,\n                                         kurt, skew=0.0, scales=None, refine=8,\n                                         smooth=0.0, kurt_grid=None, skew_grid=None,\n                                         fs=None, model="winterstein", exact=False):'
replacement3 = 'def synthesize_joint_nongaussian_history(omega, Scross, durations, fc, bw, seed,\n                                         kurt, skew=0.0, scales=None, refine=8,\n                                         smooth=0.0, kurt_grid=None, skew_grid=None,\n                                         fs=None, model="winterstein", exact=False,\n                                         copula="gaussian", copula_params=None):'
text = text.replace(target3, replacement3)

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'w', encoding='utf8') as f:
    f.write(text)
