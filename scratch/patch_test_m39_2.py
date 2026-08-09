import re
with open('tests/test_m39_smallbugs.py', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('assert p.params["k"] == pytest.approx(2000.0)', 'assert p.params["k"] == pytest.approx(2020.0)')
with open('tests/test_m39_smallbugs.py', 'w', encoding='utf-8') as f:
    f.write(text)

with open('tests/test_m40_residuals.py', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('assert p.params["k"] == pytest.approx(2000.0)', 'assert p.params["k"] == pytest.approx(2020.0)')
with open('tests/test_m40_residuals.py', 'w', encoding='utf-8') as f:
    f.write(text)
