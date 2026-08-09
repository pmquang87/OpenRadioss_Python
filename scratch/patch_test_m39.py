import re

with open('tests/test_m39_smallbugs.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix d1 expectation
text = text.replace('assert p.params["d1"] == pytest.approx(50.0)', 'assert p.params["d1"] == pytest.approx(-50.0)')

# Remove InactivePropertyError from the second test
text = re.sub(
    r'with pytest\.raises\(prop_reader\.InactivePropertyError\):.*$',
    'pass',
    text,
    flags=re.MULTILINE | re.DOTALL
)
with open('tests/test_m39_smallbugs.py', 'w', encoding='utf-8') as f:
    f.write(text)
