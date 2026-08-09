import sys
import numpy as np
from tests.test_m40_law36_solids import _build
from pyradioss.model import Model
from pyradioss.messages import MessageLog
from pathlib import Path
tmp_path = Path('scratch/tmp_v0700')
tmp_path.mkdir(exist_ok=True)
model, log = _build(tmp_path)
for group in model.engine_state.solid_hexa8_groups:
    print('dtfac:', group.state['dtfac'])
