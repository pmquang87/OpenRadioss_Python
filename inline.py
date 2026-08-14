import os
import sys

def inline(file_path):
    dir_name = os.path.dirname(file_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    out = []
    for ln in lines:
        if ln.startswith('#include'):
            inc_file = ln.split()[1].strip()
            inc_path = os.path.join(dir_name, inc_file)
            print(f'Inlining {inc_path} into {file_path}')
            with open(inc_path, 'r') as inc_f:
                out.extend(inc_f.readlines())
        else:
            out.append(ln)
        
    with open(file_path, 'w') as f:
        f.writelines(out)

inline('examples/ks2_test/KS2_model_v01_0000.rad')
