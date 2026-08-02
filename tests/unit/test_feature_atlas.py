from pathlib import Path

from pamssw.feature_atlas import AtlasConfig, AtlasRecord, load_manifest, records_from_archive


def _xyz(path: Path) -> None:
    path.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")


def test_load_jsonl_manifest_resolves_paths(tmp_path: Path):
    structure = tmp_path / "one.xyz"
    _xyz(structure)
    manifest = tmp_path / "records.jsonl"
    manifest.write_text('{"record_id":"a","phase":"p","campaign_trial":2,"structure":"one.xyz"}\n')

    records = load_manifest(manifest)

    assert records == [AtlasRecord("a", str(structure.resolve()), "p", "accepted_minimum", None, 2, None, {})]


def test_records_from_archive_keeps_start_and_archive_provenance(tmp_path: Path):
    start = tmp_path / "starting.xyz"
    _xyz(start)
    accepted = tmp_path / "accepted_minima"
    accepted.mkdir()
    _xyz(accepted / "trial0001_entry0001_accepted.xyz")
    (tmp_path / "accepted_structures.jsonl").write_text(
        '{"trial_index":1,"discovered_entry_id":1,"energy":-2.5,"seed_entry_id":0}\n'
    )

    records = records_from_archive(tmp_path, phase="restart", campaign_offset=10, start_structure=start, start_energy_eV=-1.0)

    assert [record.record_id for record in records] == ["restart:start", "restart:t1:e1"]
    assert records[1].campaign_trial == 11
    assert records[1].archive_energy_eV == -2.5


def test_config_defaults_are_production_compatible():
    config = AtlasConfig(model="model", output_dir="out")
    assert config.batch_size == 128
    assert config.dtype == "float32"
    assert config.enable_cueq is True
    assert config.umap_neighbors == 30
