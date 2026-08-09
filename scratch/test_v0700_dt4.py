import sys
import numpy as np
from tests.test_m40_law36_solids import _build
from pathlib import Path
tmp_path = Path('scratch/tmp_v0700')
tmp_path.mkdir(exist_ok=True)
model, log = _build(tmp_path)
print(model.__dict__.keys())
for group in model.element_groups:
    print('group:', group.type, 'dtfac:', group.state.get('dtfac'))
