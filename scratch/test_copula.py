import pytest
from pyradioss.model.engine import parse_engine
from pyradioss.model.model import Engine

def test_parse_impl_fatig_copula():
    deck = '''#RADIOSS ENGINE
/IMPL/FATIG/MULT/NGAUSS/COPULA
10 100 10 1
3.0 1.0
4.0 0.0
4.0
'''
    eng = parse_engine(deck)
    assert eng.impl_fatig.impl_fatig_mult
    assert eng.impl_fatig.impl_fatig_ngauss
    assert eng.impl_fatig.impl_fatig_copula == 't'
    assert eng.impl_fatig.impl_fatig_copula_params == 4.0
    
if __name__ == '__main__':
    pytest.main(['-q', 'scratch/test_copula.py'])
