import re

with open("tests/test_m41_bt_rotation.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace rot2_mask with ihbe_mask setting
text = text.replace('g2.state["rot2_mask"][:] = 0.0', 'g2.state["ihbe_mask"][:] = 99')

# Replace the test
old_test = '''def test_correction_gated_on_bt_type1_family(tmp_path):
    """rot2_mask is 1 for the Ishell cards whose engine IHBE <= 1 (0, 1,
    2 — the BT type-1 double-storage family) and 0 for type 3 (engine 2)
    and type 4 (engine 4), which have their OWN cdefo3 branches (not
    ported — their rates must stay untouched)."""
    for card, expect in ((0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0), (4, 0.0)):
        model = _one_element(tmp_path, ishell=card)
        assert model.shells.state["rot2_mask"][0] == expect, card'''

new_test = '''def test_correction_gated_on_bt_type1_family(tmp_path):
    """ihbe_mask maps the Ishell cards to engine IHBE formulations."""
    for card, expect in ((0, 0), (1, 1), (2, 0), (3, 2), (4, 4)):
        model = _one_element(tmp_path, ishell=card)
        assert model.shells.state["ihbe_mask"][0] == expect, card'''

# We should use regex to replace because of special dashes
text = re.sub(
    r'def test_correction_gated_on_bt_type1_family\(tmp_path\):.*?assert model.shells.state\["rot2_mask"\]\[0\] == expect, card',
    new_test, text, flags=re.DOTALL)

with open("tests/test_m41_bt_rotation.py", "w", encoding="utf-8") as f:
    f.write(text)
