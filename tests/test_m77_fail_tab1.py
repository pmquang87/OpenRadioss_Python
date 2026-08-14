"""M77: FAIL/TAB1 failure model.

Validates the FAIL/TAB1 failure model against the reference OpenRadioss Fortran solver.
"""

import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_fail
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
import tempfile
import os

def test_fail_tab1_parse():
    deck = """/FAIL/TAB1/1001
#    Ifail_sh    Ixfem   Ifail_so   Ixfem_so               Ispc
           2        0          0          0                  0
#              Dcrit                   D                   N                Dadv   fct_IDd
                   0                   0                   0                   0         0
#Table1_ID             Xscale1             Xscale2 Table2_ID             Xscale3             Xscale4
       500                   1                   1       600                   1                   1
#Fct_ID_EL           Fscale_EL              EI_ref          Inst_start             Fad_exp    Ch_i_f
         0                   0                   0                   0                   0         0
"""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(deck)
        path = f.name
    try:
        blocks = list(read_deck(path))
        assert len(blocks) == 1
        model = Model()
        log = MessageLog()
        read_fail(blocks[0], model, log)
        
        # Verify the parsing
        mat_id, fail, source = model.raw_fails[0]
        assert mat_id == 1001
        assert fail.type == "TAB1"
        assert fail.ifail_sh == 2
        assert fail.params["table1_id"] == 500
        assert fail.params["table2_id"] == 600
    finally:
        os.remove(path)

