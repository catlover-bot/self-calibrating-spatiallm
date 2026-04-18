from pathlib import Path

from self_calibrating_spatiallm.robustness.config import RobustnessBoundaryConfig


def test_load_default_config_and_resolve_paths() -> None:
    config_path = Path("configs/robustness_boundary.default.json").resolve()
    config = RobustnessBoundaryConfig.load_json(config_path)

    assert config.name == "robustness_boundary_v1"
    assert config.seed == 23
    assert config.splits
    assert config.perturbations

    small_split = next(split for split in config.splits if split.split_name == "small")
    small_manifest = config.resolve_manifest_path(small_split)
    assert small_manifest.is_absolute()
    assert small_manifest.name == "small_eval_pack.json"

    output_dir = config.resolve_output_dir()
    assert output_dir.is_absolute()
    assert str(output_dir).endswith("outputs/eval_pack/robustness_boundary/latest")
