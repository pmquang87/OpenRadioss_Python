import sys

code = '''
# ---------------------------------------------------------------------------
# M67: NAN/INF divergence backstop tests KE, IE, and HE
# ---------------------------------------------------------------------------

def test_nan_inf_divergence_backstop_catches_all_energy_channels(make_deck, monkeypatch):
    \"\"\"The unconditional NAN/INF backstop must catch NaN in any of the primary
    energy channels (KE, IE, HE), not just KE.\"\"\"
    
    orig_energies = eng._energies
    
    for nan_field in ["KE", "IE", "HE"]:
        def mocked_energies(model, state):
            e = orig_energies(model, state)
            if state.cycle >= 1:
                e[nan_field] = float("nan")
            return e
            
        monkeypatch.setattr(eng, "_energies", mocked_energies)
        
        # Run the same stable starter deck as test_teeth_positive_control_stays_balanced
        model, rows = _run(make_deck, f"MNAN_{nan_field}", _teeth_brick_starter("0.1"),
                           f"/RUN/MNAN_{nan_field}/1\\n0.02\\n/DT\\n0.9 0\\n/PRINT/-100\\n")
        
        stop = model.engine_state.stop_reason
        assert stop == "NAN/INF DETECTED — RUN DIVERGED"
        assert model.engine_state.cycle <= 2
'''

with open('tests/test_m41_guard.py', 'a', encoding='utf-8') as f:
    f.write(code)
