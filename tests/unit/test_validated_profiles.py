from pamssw.profiles import (
    available_validated_profiles,
    validated_ls_ssw_config,
    validated_profile_metadata,
)


PROFILE = "c60_direction_efficient_validated_20260729"


def test_c60_validated_profile_exposes_exact_reviewed_search_configuration(
    tmp_path,
):
    output_dir = tmp_path / "c60-search"

    config = validated_ls_ssw_config(
        PROFILE,
        output_dir=output_dir,
        max_trials=200,
        rng_seed=43,
        max_force_evals=100_000,
    )

    assert config.max_trials == 200
    assert config.rng_seed == 43
    assert config.max_force_evals == 100_000
    assert config.max_steps_per_walk == 8
    assert config.oracle_candidates == 4
    assert config.enable_momentum_candidate is True
    assert config.proposal_relax_steps == 80
    assert config.proposal_optimizer == "safe-lbfgs-total"
    assert config.proposal_fmax == 0.05
    assert config.quench_optimizer == "ase-lbfgs"
    assert config.quench_fallback_optimizer == "ase-fire"
    assert config.quench_fmax == 0.01
    assert config.quench_maxiter == 400
    assert config.direction_selection_mode == "discrete"
    assert config.direction_synthesis_mode == "none"
    assert config.direction_type_ucb_enabled is False
    assert config.archive_escape_momentum_enabled is False
    assert config.direction_probe_enabled is False
    assert config.plateau_evolution_enabled is False
    assert config.step_length_mode == "per_atom_rms"
    assert config.target_step_rms == 0.08
    assert config.max_step_rms == 0.15
    assert config.local_softening_mode == "active_neighbors"
    assert config.local_softening_active_count == 3
    assert config.accepted_structures_log == str(
        output_dir / "accepted_structures.jsonl"
    )
    assert config.accepted_structures_dir == str(
        output_dir / "accepted_minima"
    )
    assert config.direction_diagnostics_path == str(
        output_dir / "direction_trace.jsonl"
    )


def test_validated_profile_is_named_and_reports_its_evidence_ceiling():
    assert available_validated_profiles() == (PROFILE,)

    metadata = validated_profile_metadata(PROFILE)

    assert metadata["system"] == "C60"
    assert metadata["source_commit"] == (
        "04c7bd011fd3b9beaa7dff775ffa5e203ef00a92"
    )
    assert metadata["model_sha256"] == (
        "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
    )
    assert metadata["production_default_changed"] is False
    assert "fixed-starter" in metadata["claim_ceiling"]
    assert "universal" in metadata["claim_ceiling"]


def test_validated_profile_builder_is_part_of_the_public_api():
    from pamssw import (
        available_validated_profiles as public_available,
        validated_ls_ssw_config as public_builder,
        validated_profile_metadata as public_metadata,
    )

    assert public_available is available_validated_profiles
    assert public_builder is validated_ls_ssw_config
    assert public_metadata is validated_profile_metadata
