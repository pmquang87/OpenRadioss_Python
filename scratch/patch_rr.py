import re

with open('pyradioss/implicit/random_response.py', 'r', encoding='utf8') as f:
    text = f.read()

target1 = '''        mc = jng.joint_nongaussian_monte_carlo_damage(
            w, Sy, durations, fc, bw, m, C, seed, kurt, skew=skew, scales=scales,
            refine=rfn, smooth=sm, kurt_grid=kgrid, skew_grid=sgrid, fs=fs,
            mean_stress=mstress, ultimate=ult, naz=naz, npol=npol, model=model,
            reduction=reduction, summary=j_sum)'''
replacement1 = '''        mc = jng.joint_nongaussian_monte_carlo_damage(
            w, Sy, durations, fc, bw, m, C, seed, kurt, skew=skew, scales=scales,
            refine=rfn, smooth=sm, kurt_grid=kgrid, skew_grid=sgrid, fs=fs,
            mean_stress=mstress, ultimate=ult, naz=naz, npol=npol, model=model,
            reduction=reduction, summary=j_sum,
            copula=getattr(ip, "impl_fatig_copula", "gaussian"),
            copula_params=getattr(ip, "impl_fatig_copula_params", None))'''
text = text.replace(target1, replacement1)

target2 = '''            ex_mc = jng.joint_nongaussian_monte_carlo_damage(
                w, Sy, durations, fc, bw, m, C, seed, kurt, skew=skew, scales=scales,
                refine=rfn, smooth=sm, kurt_grid=kgrid, skew_grid=sgrid, fs=fs,
                mean_stress=mstress, ultimate=ult, naz=naz, npol=npol, model=model,
                reduction=reduction, summary=ex_sum, exact=True)'''
replacement2 = '''            ex_mc = jng.joint_nongaussian_monte_carlo_damage(
                w, Sy, durations, fc, bw, m, C, seed, kurt, skew=skew, scales=scales,
                refine=rfn, smooth=sm, kurt_grid=kgrid, skew_grid=sgrid, fs=fs,
                mean_stress=mstress, ultimate=ult, naz=naz, npol=npol, model=model,
                reduction=reduction, summary=ex_sum, exact=True,
                copula=getattr(ip, "impl_fatig_copula", "gaussian"),
                copula_params=getattr(ip, "impl_fatig_copula_params", None))'''
text = text.replace(target2, replacement2)

with open('pyradioss/implicit/random_response.py', 'w', encoding='utf8') as f:
    f.write(text)
