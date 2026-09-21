from types import SimpleNamespace
from research.ga_ssw.run_recovered_direction_smoke import deposited_gaussians,decoded_ls,encoded_ls

def test_gaussian_count_excludes_failed_height_preparation():
    result=SimpleNamespace(records=[SimpleNamespace(climb=[
        {'status':'nonpositive_height','weight':-1.}, {'status':'height_prepared','weight':1.},
        {'status':'evaluation_failed','weight':2.,'center':[[0,0,0]],'direction':[[1,0,0]]},
        {'status':'rotation_failed'}, {'status':'height_prepared','weight':1.,'center':[[0,0,0]],'direction':[[1,0,0]]}])])
    assert deposited_gaussians(result)==2

def test_native_ls_settings_json_roundtrip_keeps_tables_and_prequench():
    from pamssw.standalone.ls_native_reference import NativeLSSettings
    from pamssw.standalone.ls_prequench import LSPrequenchSettings
    source=NativeLSSettings({(29,29):3.6298000812530518},{(29,29):1.875},
        target_mev_per_atom=20.,prequench=LSPrequenchSettings(.1,50,'force_or_step_limit'))
    got=decoded_ls(encoded_ls(source))
    assert got.bond_energies==source.bond_energies and got.bond_lengths==source.bond_lengths
    assert got.prequench==source.prequench
