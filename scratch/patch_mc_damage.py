import re

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'r', encoding='utf8') as f:
    text = f.read()

target1 = '''def joint_nongaussian_monte_carlo_damage(omega, Scross, durations, fc, bw, m, C,
                                         seed, kurt, skew=0.0, scales=None, refine=8,
                                         smooth=0.0, kurt_grid=None, skew_grid=None,
                                         fs=None, mean_stress=0.0, ultimate=0.0,
                                         naz=24, npol=13, model="winterstein",
                                         reduction="shear_plane", summary=None,
                                         exact=False):'''
replacement1 = '''def joint_nongaussian_monte_carlo_damage(omega, Scross, durations, fc, bw, m, C,
                                         seed, kurt, skew=0.0, scales=None, refine=8,
                                         smooth=0.0, kurt_grid=None, skew_grid=None,
                                         fs=None, mean_stress=0.0, ultimate=0.0,
                                         naz=24, npol=13, model="winterstein",
                                         reduction="shear_plane", summary=None,
                                         exact=False, copula="gaussian", copula_params=None):'''
text = text.replace(target1, replacement1)

target2 = '''    t, X, info = synthesize_joint_nongaussian_history(
        omega, Scross, durations, fc, bw, seed, kurt, skew=skew, scales=scales,
        refine=refine, smooth=smooth, kurt_grid=kurt_grid, skew_grid=skew_grid,
        fs=fs, model=model, exact=exact)'''
replacement2 = '''    t, X, info = synthesize_joint_nongaussian_history(
        omega, Scross, durations, fc, bw, seed, kurt, skew=skew, scales=scales,
        refine=refine, smooth=smooth, kurt_grid=kurt_grid, skew_grid=skew_grid,
        fs=fs, model=model, exact=exact, copula=copula, copula_params=copula_params)'''
text = text.replace(target2, replacement2)

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'w', encoding='utf8') as f:
    f.write(text)
