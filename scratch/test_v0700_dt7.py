import sys
import numpy as np
from tests.test_m40_law36_solids import _build
from pathlib import Path
tmp_path = Path('scratch/tmp_v0700')
tmp_path.mkdir(exist_ok=True)
model, log = _build(tmp_path)
for name, group in model.element_groups():
    print(name)
    if hasattr(group, 'state'):
        print('state keys:', group.state.keys())
        if 'dtfac' in group.state:
            print('dtfac:', group.state['dtfac'])
