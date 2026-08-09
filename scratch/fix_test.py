import re

with open('tests/test_m41_guard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove everything from # M67: NAN/INF onwards
index = content.find('# ---------------------------------------------------------------------------\\n# M67: NAN/INF')
if index != -1:
    content = content[:index]

code = '''# ---------------------------------------------------------------------------
# M67: NAN/INF divergence backstop tests KE, IE, and HE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nan_field", ["KE", "IE", "HE"])
def test_nan_inf_divergence_backstop_catches_all_energy_channels(make_deck, monkeypatch, nan_field):
    \"\"\"The unconditional NAN/INF backstop must catch NaN in any of the primary
    energy channels (KE, IE, HE), not just KE.\"\"\"
    
    orig_energies = eng._energies
    
    def mocked_energies(model, state):
        e = orig_energies(model, state)
        if state.cycle >= 1:
            e[nan_field] = float("nan")
        return e
        
    monkeypatch.setattr(eng, "_energies", mocked_energies)
    
    # Run the same stable starter deck, but print every cycle so the guard is
    # evaluated immediately when the NaN is injected at cycle 1.
    model, rows = _run(make_deck, f"MNAN_{nan_field}", _teeth_brick_starter("0.1"),
                       f"/RUN/MNAN_{nan_field}/1\\n0.02\\n/DT\\n0.9 0\\n/PRINT/-1\\n")
    
    stop = model.engine_state.stop_reason
    assert stop and "NAN/INF DETECTED" in stop, f"Did not catch NaN in {nan_field}"
    assert model.engine_state.cycle <= 2
'''
with open('tests/test_m41_guard.py', 'w', encoding='utf-8') as f:
    f.write(content + code)
