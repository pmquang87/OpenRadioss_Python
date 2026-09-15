import re

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'r', encoding='utf8') as f:
    text = f.read()

target1 = '''def joint_lambda_ng(M0, proj, gamma3, gamma4, m, alpha2=1.0,
                    bandwidth_correction=True, model="winterstein"):'''
replacement1 = '''def joint_lambda_ng(M0, proj, gamma3, gamma4, m, alpha2=1.0,
                    bandwidth_correction=True, model="winterstein",
                    copula="gaussian", copula_params=None):'''
text = text.replace(target1, replacement1)

target2 = '''    _, g3s, g4s = induced_projection_moments(M0, proj, gamma3, gamma4, model=model)'''
replacement2 = '''    _, g3s, g4s = induced_projection_moments(M0, proj, gamma3, gamma4, model=model,
                                             copula=copula, copula_params=copula_params)'''
text = text.replace(target2, replacement2)

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'w', encoding='utf8') as f:
    f.write(text)

with open('pyradioss/implicit/random_response.py', 'r', encoding='utf8') as f:
    text = f.read()

target3 = '''                lam, g3s, g4s = jng.joint_lambda_ng(
                    M0, p, g3, g4, m, alpha2=a2, bandwidth_correction=bw, model=model)'''
replacement3 = '''                lam, g3s, g4s = jng.joint_lambda_ng(
                    M0, p, g3, g4, m, alpha2=a2, bandwidth_correction=bw, model=model,
                    copula=getattr(ip, "impl_fatig_copula", "gaussian"),
                    copula_params=getattr(ip, "impl_fatig_copula_params", None))'''
text = text.replace(target3, replacement3)

with open('pyradioss/implicit/random_response.py', 'w', encoding='utf8') as f:
    f.write(text)
