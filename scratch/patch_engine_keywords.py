import re

with open('pyradioss/input/engine_keywords.py', 'r', encoding='utf8') as f:
    text = f.read()

target1 = '''                    is_exact = bool(subs & {"EXACT", "NORTA", "NATAF",
                                            "GRIGORIU", "COVEXACT"})'''
replacement1 = '''                    is_exact = bool(subs & {"EXACT", "NORTA", "NATAF",
                                            "GRIGORIU", "COVEXACT"})
                    # M36: NON-GAUSSIAN COPULA / NON-TRANSLATION JOINT DISTRIBUTION
                    # Replaces the Gaussian copula with a t-copula.
                    is_copula = bool(subs & {"COPULA", "TCOPULA"})'''
text = text.replace(target1, replacement1)

target2 = '''                        if len(vE) > 6 and vE[6] >= 0:
                            ec.impl_fatig_wv_smooth = float(vE[6])'''
replacement2 = '''                        if len(vE) > 6 and vE[6] >= 0:
                            ec.impl_fatig_wv_smooth = float(vE[6])
                            
                    # M36 COPULA: the copula type and params live on a DEDICATED
                    # card line AFTER the M26 drifting-shape line (and before MINPUT)
                    if is_copula:
                        cline = 2 + (1 if is_ngauss else 0) + (1 if is_nstat else 0) + (1 if is_evol else 0)
                        vC = (block.cards[cline].floats() if len(block.cards) > cline else [])
                        ec.impl_fatig_copula = "t"
                        if len(vC) > 0 and vC[0] > 0.0:
                            ec.impl_fatig_copula_params = vC[0]
                        else:
                            ec.impl_fatig_copula_params = 4.0'''
text = text.replace(target2, replacement2)

target3 = '''                    if is_minput:
                        miline = (2 + (1 if is_ngauss else 0)
                                  + (1 if is_nstat else 0)
                                  + (1 if is_evol else 0))'''
replacement3 = '''                    if is_minput:
                        miline = (2 + (1 if is_ngauss else 0)
                                  + (1 if is_nstat else 0)
                                  + (1 if is_evol else 0)
                                  + (1 if is_copula else 0))'''
text = text.replace(target3, replacement3)

with open('pyradioss/input/engine_keywords.py', 'w', encoding='utf8') as f:
    f.write(text)
