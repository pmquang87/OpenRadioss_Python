import re

with open('pyradioss/model/model.py', 'r', encoding='utf8') as f:
    text = f.read()

target = '    impl_fatig_skew: float = 0.0     # target skewness gamma_3 (0 = symmetric)'
replacement = '    impl_fatig_skew: float = 0.0     # target skewness gamma_3 (0 = symmetric)\n    impl_fatig_copula: str = "gaussian"  # target copula ("gaussian", "t")\n    impl_fatig_copula_params: float = 4.0 # copula degrees of freedom (for t-copula)'
text = text.replace(target, replacement)

with open('pyradioss/model/model.py', 'w', encoding='utf8') as f:
    f.write(text)
