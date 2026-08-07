import numpy as np
from pyradioss.model.model import Model
from pyradioss.model.entities import MonitoredVolume
from pyradioss.materials.mat_gas import GasMaterial
from pyradioss.engine.airbag import update_airbag_thermodynamics

def test_adiabatic_compression():
    model = Model()
    
    cp = 1004.0
    cv = 717.0
    r_spec = cp - cv
    gamma = cp / cv
    
    mat = GasMaterial(id=1, law=999, rho0=1.0, params={'CPA': cp, 'R_igc': r_spec, 'MW': 1.0}, title="Gas")
    model.materials[1] = mat
    
    mv = MonitoredVolume(
        id=1,
        matid=1,
        pext=101325.0, # 1 atm
        t_initial=300.0,
        iequil=1,
    )
    
    mv.volume = 1.0 # 1 m^3
    model.monitored_volumes[1] = mv
    
    update_airbag_thermodynamics(mv, model, 0.0, 0.0)
    
    print(f"Init: P={mv.pressure}, T={mv.temperature}, M={mv.mass}")
    
    n_steps = 100
    v_end = 0.5
    vols = np.linspace(1.0, v_end, n_steps + 1)
    
    for i in range(1, n_steps + 1):
        mv.volume = vols[i]
        update_airbag_thermodynamics(mv, model, 0.001, 0.001 * i)
        
    print(f"End: P={mv.pressure}, T={mv.temperature}")
    
    p_analytical = 101325.0 * (1.0 / 0.5)**gamma
    t_analytical = 300.0 * (1.0 / 0.5)**(gamma - 1.0)
    
    print(f"Analytical: P={p_analytical}, T={t_analytical}")
    
    # Radioss explicit integration of thermodynamics has ~1% error for 100 steps
    # We just need to check if the logic matches analytical within reason.
    assert np.isclose(mv.pressure, p_analytical, rtol=1e-2)
    assert np.isclose(mv.temperature, t_analytical, rtol=1e-2)

if __name__ == '__main__':
    test_adiabatic_compression()
