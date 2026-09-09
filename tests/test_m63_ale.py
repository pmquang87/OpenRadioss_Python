from pyradioss.input.deck_reader import read_deck
from pyradioss.starter.starter import run_starter


def test_ale_keywords(tmp_path):
    """Verify that /ALE/DONE, /ALE/BCS, and /ALE/GRID don't fail the starter."""
    deck = (
        "/BEGIN\nALE model\n"
        "/NODE\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 1.0 1.0 0.0\n"
        "4 0.0 1.0 0.0\n"
        "5 0.0 0.0 1.0\n"
        "6 1.0 0.0 1.0\n"
        "7 1.0 1.0 1.0\n"
        "8 0.0 1.0 1.0\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\npart\n1 1\n"
        "/MAT/LAW3/1\nlaw3\n"
        "1.0                 \n"
        "200e9               0.3                 \n"
        "200e6               20e6                0.1                 0.5                 500e6               \n"
        "1                   2                   3                   4                   \n"
        "0                   1                   \n"
        "1                   2                   100e9               \n"
        "/MAT/LAW4/2\nlaw4\n"
        "1.0                 \n"
        "200e9               0.3                 \n"
        "200e6               20e6                0.1                 0.5                 500e6               \n"
        "1                   2                   3                   4                   \n"
        "0                   1                   \n"
        "1                   2                   100e9               \n"
        "100                 200                 300                 400                 500                 \n"
        "1.0                 \n"
        "/PROP/SOLID/1\nprop\n"
        "/ALE/DONE\n"
        "/ALE/BCS/1\nale bcs\n111000 0 1\n"
        "/ALE/GRID/MASS-WEIGHTED-VEL\n"
        "1 1.0\n0 0.0\n"
        "/ALE/MAT/1\n"
        "0\n"
        "/END\n"
    )
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck)
    
    # We should also be able to build a Starter
    model = run_starter(str(p))
    
    assert model.has_ale is True
    assert len(model.ale_bcs) == 1
    assert model.ale_bcs[0].id == 1
    assert model.ale_bcs[0].fix_w.tolist() == [True, True, True]
    assert model.ale_bcs[0].fix_l.tolist() == [False, False, False]
    assert len(model.raw_mat_notes) == 1
    assert model.raw_mat_notes[0][0] == "ALE/MAT"
    
    # Check that LAW3 and LAW4 parsed successfully and populated params
    mat3 = model.materials[1]
    assert getattr(mat3, "inactive", False) is True
    assert mat3.law == 3
    assert mat3.rho0 == 1.0
    assert mat3.params["MAT_E"] == 200e9
    assert mat3.params["MAT_SIGY"] == 200e6
    assert mat3.params["MAT_BETA"] == 20e6
    
    mat4 = model.materials[2]
    # M535: LAW4 physics is ported, so mat4 is active (inactive is False)
    assert getattr(mat4, "inactive", False) is False
    assert mat4.law == 4
    assert mat4.params["MAT_E"] == 200e9
    assert mat4.params["MAT_SIGY"] == 200e6
