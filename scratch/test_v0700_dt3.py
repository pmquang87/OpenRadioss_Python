import sys
import numpy as np
from tests.test_m40_law36_solids import _build
from pathlib import Path
tmp_path = Path('scratch/tmp_v0700')
tmp_path.mkdir(exist_ok=True)
model, log = _build(tmp_path)
for group in model.starter_state.solid_hexa8_groups:
    print('dtfac:', group.state['dtfac'])
