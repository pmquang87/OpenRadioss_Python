import re

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'r', encoding='utf8') as f:
    text = f.read()

target1 = '''def _rescale_block_to_underlying(Sw, freqs, gamma3, gamma4, model):'''
replacement1 = '''def _rescale_block_to_underlying(Sw, freqs, gamma3, gamma4, model, copula="gaussian", copula_params=None):'''
text = text.replace(target1, replacement1)

target2 = '''    sol = solve_underlying_correlation(M0, gamma3, gamma4, model=model,
                                       repair=True)'''
replacement2 = '''    sol = solve_underlying_correlation(M0, gamma3, gamma4, model=model,
                                       repair=True, copula=copula, copula_params=copula_params)'''
text = text.replace(target2, replacement2)

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'w', encoding='utf8') as f:
    f.write(text)
