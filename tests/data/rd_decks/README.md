# Vendored official-deck corpus extract

Small input decks from the **OpenRadioss example & verification library**
(© Altair Engineering Inc., same upstream project this port is derived from;
see the repo LICENSE). They are test data for the corpus-guarded tests in
`tests/test_m37_materials_pack1.py` and `tests/test_m38_props.py`.

Vendored here (starter + engine deck each, plain text):

| Path | Source zip | Used by |
|---|---|---|
| `rd_e/RD-E-5200_Creep/52_cylinder_creep/cylinder_creep_beta_001/` | RD-E-5200_Creep.zip | `test_law40_corpus_kelvinmax_deck_builds` |
| `rd_v_material/RD-V-0220_Foam_LAW70/0220_foam_LAW70/0220_foam_LAW70_{0..3}/` | RD-V-0220_Foam_LAW70.zip | `test_law70_oracle_deck_material_resolves` |
| `rd_v_material/RD-V-0230_Fabric_LAW19/0230_fabric_LAW19/0230_shell_mat_019_01/` | RD-V-0230_Fabric_LAW19.zip | `test_law19_prop9_corpus_end_to_end` |

## Getting more decks (the full corpus)

Tests whose decks are too large to vendor (e.g. `test_law151_...` needs the
15 MB `RD-E-1300_Shock_tube` blast deck) skip unless they find the file.
To run them, point the environment variable **`PYRADIOSS_RD_DECKS`** at a
directory with the same layout (`rd_e/<RD-E-name>/...`,
`rd_v_<category>/<RD-V-name>/...`) holding a fuller extract.

Sources for the zips:

- On this machine: `E:\openradioss_run\demos_example\` — `example\RD-E-*.zip`
  and `verification\<category>\RD-V-*.zip` (READ-ONLY: copy out, never write
  or extract in place).
- Public: the OpenRadioss example library — model files linked from
  <https://openradioss.atlassian.net/wiki/spaces/OPENRADIOSS/> (the RD-E
  example manual and RD-V verification manual pages), or the
  [OpenRadioss docs](https://github.com/OpenRadioss/OpenRadioss) project.

Extract each `RD-E-XXXX_Name.zip` to `rd_e/RD-E-XXXX_Name/` and each
`RD-V-XXXX_Name.zip` to `rd_v_<category>/RD-V-XXXX_Name/` (the zips contain
the inner deck directories, e.g. `52_cylinder_creep/...`).
