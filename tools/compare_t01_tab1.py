import numpy as np

def compare_t01(f1, f2, rtol=0.05, atol=1e-3):
    d1 = np.genfromtxt(f1, delimiter=',', skip_header=1)
    d2 = np.genfromtxt(f2, delimiter=',', skip_header=1)
    
    if d1.shape != d2.shape:
        print(f"Shapes differ: {d1.shape} vs {d2.shape}")
        
    diff = np.abs(d1 - d2)
    max_diff = np.max(diff)
    print(f"Max absolute difference: {max_diff}")
    
    # Calculate rel_rms
    # Ignore time column (index 0) for RMS comparison if desired
    # For now, just print the max diff
    match = np.allclose(d1[:, 1:], d2[:, 1:], rtol=rtol, atol=atol)
    print(f"Match (rtol={rtol}, atol={atol}): {match}")
    
    if not match:
        # Find which columns differ
        for i in range(1, d1.shape[1]):
            if not np.allclose(d1[:, i], d2[:, i], rtol=rtol, atol=atol):
                print(f"Column {i} differs. Max diff: {np.max(np.abs(d1[:, i] - d2[:, i]))}")

if __name__ == '__main__':
    base_dir = "C:/Users/pmqua/.gemini/antigravity/brain/93ac0308-1509-4fd5-93df-987958311651/scratch/RD-E-2602/ductile_failure_model/plate_model/TAB1_model/Ishell=1_without_epsmax"
    f1 = f"{base_dir}/FAILURE_TAB1T01_fortran.csv"
    f2 = f"{base_dir}/FAILURE_TAB1T01.csv"
    compare_t01(f1, f2)
