"""Robustness-boundary experiment helpers."""

from self_calibrating_spatiallm.robustness.analysis import (
    build_boundary_rows,
    build_boundary_summary,
    export_language_boundary_artifacts,
    render_boundary_summary_markdown,
)
from self_calibrating_spatiallm.robustness.config import (
    PerturbationFamilyConfig,
    RobustnessBoundaryConfig,
    RobustnessSplitConfig,
)
from self_calibrating_spatiallm.robustness.perturbations import (
    SUPPORTED_PERTURBATIONS,
    PerturbationResult,
    apply_perturbation,
    derive_variant_seed,
    severity_bucket,
    write_ascii_ply,
)
from self_calibrating_spatiallm.robustness.runner import run_robustness_boundary_experiment

__all__ = [
    "SUPPORTED_PERTURBATIONS",
    "PerturbationFamilyConfig",
    "PerturbationResult",
    "RobustnessBoundaryConfig",
    "RobustnessSplitConfig",
    "apply_perturbation",
    "build_boundary_rows",
    "build_boundary_summary",
    "derive_variant_seed",
    "export_language_boundary_artifacts",
    "render_boundary_summary_markdown",
    "run_robustness_boundary_experiment",
    "severity_bucket",
    "write_ascii_ply",
]

