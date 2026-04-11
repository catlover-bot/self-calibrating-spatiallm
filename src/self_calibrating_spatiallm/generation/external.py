"""External SpatialLM generator adapter."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from self_calibrating_spatiallm.artifacts import (
    CalibratedPointCloud,
    Point3D,
    SceneObject,
    ScenePrediction,
    SceneRelation,
)
from self_calibrating_spatiallm.generation.base import SceneGenerator
from self_calibrating_spatiallm.generation.spatiallm_io import export_spatiallm_input


class ExternalGeneratorError(RuntimeError):
    """Raised when external scene generation cannot complete successfully."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class SpatialLMExternalGenerator(SceneGenerator):
    """Adapter for invoking external SpatialLM wrappers from the pipeline."""

    name = "spatiallm_external"

    def __init__(
        self,
        command: str | Sequence[str] | None = None,
        timeout_sec: int = 300,
        output_json_path: Path | None = None,
        command_env_var: str = "SCSLM_SPATIALLM_COMMAND",
        export_format: str = "json",
    ) -> None:
        self.timeout_sec = timeout_sec
        self.output_json_path = output_json_path
        self.command_env_var = command_env_var
        self.export_format = export_format

        if isinstance(command, str):
            self.command = shlex.split(command)
        elif command is None:
            self.command = None
        else:
            self.command = [str(item) for item in command]

        self._last_raw_output: str | None = None
        self._last_execution_info: dict[str, Any] | None = None

    def get_last_raw_output(self) -> str | None:
        return self._last_raw_output

    def get_last_execution_info(self) -> dict[str, Any] | None:
        return self._last_execution_info

    def generate(self, calibrated: CalibratedPointCloud) -> ScenePrediction:
        self._last_raw_output = None
        self._last_execution_info = None

        command_template, command_source = self._resolve_command_template()
        if command_template is None and self.output_json_path is None:
            details = {
                "command_source": command_source,
                "output_json_path": None,
                "command_env_var": self.command_env_var,
            }
            self._last_execution_info = {
                "generator_mode": "external",
                "generator_name": self.name,
                "success": False,
                "error": "external generator not configured",
                **details,
            }
            raise ExternalGeneratorError(
                "SpatialLM external generator is not configured. "
                "Provide spatiallm_command in config, set the command env var, "
                "or provide spatiallm_output_json.",
                details=details,
            )

        with tempfile.TemporaryDirectory(prefix="scslm_external_gen_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            input_json = tmp_dir / "calibrated_point_cloud.json"
            output_json = self.output_json_path or (tmp_dir / "scene_prediction.json")
            spatiallm_input = tmp_dir / f"spatiallm_input.{self.export_format}"

            calibrated.save_json(input_json)
            export_info = export_spatiallm_input(
                calibrated=calibrated,
                output_path=spatiallm_input,
                export_format=self.export_format,
            )
            exported_text = spatiallm_input.read_text(encoding="utf-8")
            export_payload_summary = _summarize_spatiallm_export_payload(export_info.get("payload"))
            export_payload_hash = sha256(exported_text.encode("utf-8")).hexdigest()

            invoked_command: list[str] | None = None
            return_code = 0
            stdout = ""
            stderr = ""

            if command_template is not None:
                invoked_command = [
                    token.format(
                        input_json=str(input_json),
                        output_json=str(output_json),
                        spatiallm_input=str(spatiallm_input),
                        scene_id=calibrated.sample_id,
                    )
                    for token in command_template
                ]
                completed = self._run_command(invoked_command)
                return_code = int(completed.returncode)
                stdout = completed.stdout
                stderr = completed.stderr

            payload, payload_source, payload_warnings = self._load_prediction_payload(
                output_json=output_json,
                stdout=stdout,
            )
            output_json_text = output_json.read_text(encoding="utf-8") if output_json.exists() else ""

            prediction, parse_info = self._parse_prediction_with_hooks(payload, sample_id=calibrated.sample_id)
            prediction_summary = _summarize_prediction(prediction)
            propagation_diagnostics = {
                "generator_mode": "external",
                "command_source": command_source,
                "payload_source": payload_source,
                "parse_mode": parse_info.get("parse_mode"),
                "partial_parse": bool(parse_info.get("partial_parse")),
                "parse_warning_count": len(parse_info.get("warnings", [])),
                "spatiallm_export_summary": export_payload_summary,
                "prediction_summary": prediction_summary,
            }

            self._last_raw_output = self._compose_raw_output(stdout, stderr)
            self._last_execution_info = {
                "generator_mode": "external",
                "generator_name": self.name,
                "command_source": command_source,
                "command_template": command_template,
                "invoked_command": invoked_command,
                "timeout_sec": self.timeout_sec,
                "return_code": return_code,
                "stdout": stdout,
                "stderr": stderr,
                "success": True,
                "spatiallm_export": {
                    "format": export_info["format"],
                    "num_points": export_info["num_points"],
                    "payload_hash_sha256": export_payload_hash,
                    "payload_summary": export_payload_summary,
                },
                "payload_source": payload_source,
                "payload_warnings": payload_warnings,
                "parse_mode": parse_info.get("parse_mode"),
                "partial_parse": bool(parse_info.get("partial_parse")),
                "parse_warnings": parse_info["warnings"],
                "prediction_summary": prediction_summary,
                "raw_files": {
                    "exported_spatiallm_input_text": exported_text,
                    "output_json_text": output_json_text,
                },
            }

        prediction.metadata = {
            **prediction.metadata,
            "generator_adapter": self.name,
            "raw_output_preview": (self._last_raw_output or "")[:4000],
            "payload_source": self._last_execution_info.get("payload_source"),
            "parse_warnings": self._last_execution_info.get("parse_warnings", []),
            "propagation_diagnostics": propagation_diagnostics,
        }
        return prediction

    def _resolve_command_template(self) -> tuple[list[str] | None, str]:
        if self.command is not None:
            return list(self.command), "config"

        from_env = os.environ.get(self.command_env_var)
        if from_env:
            return shlex.split(from_env), f"env:{self.command_env_var}"

        return None, "none"

    def _run_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except FileNotFoundError as error:
            binary = command[0] if command else "<empty>"
            details = {"invoked_command": command, "timeout_sec": self.timeout_sec}
            self._last_execution_info = {
                "generator_mode": "external",
                "generator_name": self.name,
                "success": False,
                "error": f"command not found: {binary}",
                **details,
            }
            raise ExternalGeneratorError(
                f"SpatialLM command not found: {binary}. "
                "Install/configure the external generator and retry.",
                details=details,
            ) from error
        except subprocess.TimeoutExpired as error:
            details = {
                "invoked_command": command,
                "timeout_sec": self.timeout_sec,
            }
            self._last_execution_info = {
                "generator_mode": "external",
                "generator_name": self.name,
                "success": False,
                "error": "command timeout",
                **details,
            }
            raise ExternalGeneratorError(
                f"SpatialLM external command timed out after {self.timeout_sec}s.",
                details=details,
            ) from error

        if completed.returncode != 0:
            details = {
                "invoked_command": command,
                "return_code": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            self._last_execution_info = {
                "generator_mode": "external",
                "generator_name": self.name,
                "success": False,
                "error": "non-zero return code",
                **details,
            }
            raise ExternalGeneratorError(
                "SpatialLM external command failed "
                f"(exit code {completed.returncode}).\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}",
                details=details,
            )

        return completed

    def _load_prediction_payload(
        self,
        output_json: Path,
        stdout: str,
    ) -> tuple[dict[str, Any], str, list[str]]:
        warnings: list[str] = []

        if output_json.exists():
            try:
                payload = json.loads(output_json.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload, "output_json", warnings
                warnings.append("output_json payload is not an object")
            except json.JSONDecodeError:
                warnings.append("output_json is not valid JSON")

        if stdout.strip():
            try:
                payload = json.loads(stdout)
                if isinstance(payload, dict):
                    return payload, "stdout_json", warnings
                warnings.append("stdout JSON payload is not an object")
            except json.JSONDecodeError:
                warnings.append("stdout is not valid JSON")

        raise ExternalGeneratorError(
            "External generator produced no parseable JSON output. "
            "Expected JSON at output_json path or JSON object in stdout.",
            details={
                "output_json_path": str(output_json),
                "stdout_preview": stdout[:500],
                "warnings": warnings,
            },
        )

    def _parse_prediction_with_hooks(
        self,
        payload: dict[str, Any],
        sample_id: str,
    ) -> tuple[ScenePrediction, dict[str, Any]]:
        warnings: list[str] = []
        candidate = payload.get("scene_prediction", payload)

        if isinstance(candidate, dict):
            as_full_schema = dict(candidate)
            as_full_schema.setdefault("sample_id", sample_id)
            as_full_schema.setdefault("generator_name", self.name)
            try:
                return ScenePrediction.from_dict(as_full_schema), {
                    "warnings": warnings,
                    "parse_mode": "full_schema",
                    "partial_parse": False,
                }
            except Exception as error:
                warnings.append(f"full_schema_parse_failed: {error}")
        else:
            warnings.append("scene_prediction payload is not an object")
            candidate = {}

        objects = self._parse_objects(candidate.get("objects") or candidate.get("instances"), warnings)
        relations = self._parse_relations(candidate.get("relations"), warnings)

        metadata = candidate.get("metadata", {})
        if not isinstance(metadata, dict):
            warnings.append("metadata field is not an object")
            metadata = {}

        prediction = ScenePrediction(
            sample_id=sample_id,
            generator_name=str(candidate.get("generator_name", self.name)),
            objects=objects,
            relations=relations,
            metadata={
                **metadata,
                "partial_parse": True,
                "partial_payload_keys": sorted(candidate.keys()),
            },
        )
        return prediction, {
            "warnings": warnings,
            "parse_mode": "partial_schema",
            "partial_parse": True,
        }

    def _parse_objects(self, raw_objects: Any, warnings: list[str]) -> list[SceneObject]:
        if raw_objects is None:
            warnings.append("no objects field found in payload")
            return []
        if not isinstance(raw_objects, list):
            warnings.append("objects field is not a list")
            return []

        parsed: list[SceneObject] = []
        for index, item in enumerate(raw_objects):
            if not isinstance(item, dict):
                warnings.append(f"objects[{index}] is not an object")
                continue

            object_id = str(item.get("object_id") or item.get("id") or f"ext_obj_{index}")
            label = str(item.get("label") or item.get("category") or item.get("class") or "unknown")
            confidence = float(item.get("confidence", item.get("score", 0.5)))

            position = _parse_point(item.get("position") or item.get("center") or item.get("centroid"))
            size = _parse_point(item.get("size") or item.get("dimensions") or item.get("extent"), default=0.5)

            parsed.append(
                SceneObject(
                    object_id=object_id,
                    label=label,
                    position=position,
                    size=size,
                    confidence=confidence,
                    attributes={"external_raw": item},
                )
            )
        return parsed

    def _parse_relations(self, raw_relations: Any, warnings: list[str]) -> list[SceneRelation]:
        if raw_relations is None:
            return []
        if not isinstance(raw_relations, list):
            warnings.append("relations field is not a list")
            return []

        parsed: list[SceneRelation] = []
        for index, item in enumerate(raw_relations):
            if not isinstance(item, dict):
                warnings.append(f"relations[{index}] is not an object")
                continue

            subject = item.get("subject_id") or item.get("subject")
            predicate = item.get("predicate") or item.get("relation")
            obj = item.get("object_id") or item.get("object")
            if not subject or not predicate or not obj:
                warnings.append(f"relations[{index}] missing subject/predicate/object")
                continue

            parsed.append(
                SceneRelation(
                    subject_id=str(subject),
                    predicate=str(predicate),
                    object_id=str(obj),
                    score=float(item.get("score", item.get("confidence", 0.5))),
                    metadata={"external_raw": item},
                )
            )
        return parsed

    @staticmethod
    def _compose_raw_output(stdout: str, stderr: str) -> str:
        stdout = stdout.strip()
        stderr = stderr.strip()
        chunks: list[str] = []
        if stdout:
            chunks.append("[stdout]\n" + stdout)
        if stderr:
            chunks.append("[stderr]\n" + stderr)
        return "\n\n".join(chunks)


def _parse_point(raw: Any, default: float = 0.0) -> Point3D:
    if isinstance(raw, dict):
        try:
            return Point3D(
                x=float(raw.get("x", default)),
                y=float(raw.get("y", default)),
                z=float(raw.get("z", default)),
            )
        except Exception:
            return Point3D(x=default, y=default, z=default)

    if isinstance(raw, list) and len(raw) >= 3:
        try:
            return Point3D(x=float(raw[0]), y=float(raw[1]), z=float(raw[2]))
        except Exception:
            return Point3D(x=default, y=default, z=default)

    return Point3D(x=default, y=default, z=default)


def _summarize_prediction(prediction: ScenePrediction) -> dict[str, Any]:
    object_labels = sorted({str(obj.label) for obj in prediction.objects})
    relation_predicates = sorted({str(rel.predicate) for rel in prediction.relations})
    return {
        "object_count": len(prediction.objects),
        "relation_count": len(prediction.relations),
        "object_labels": object_labels,
        "relation_predicates": relation_predicates,
    }


def _summarize_spatiallm_export_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    axis = payload.get("axis_convention", {})
    normalization = payload.get("normalization", {})
    points = payload.get("points_xyz", [])

    ranges_xyz = None
    if isinstance(points, list) and points:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            try:
                xs.append(float(point.get("x", 0.0)))
                ys.append(float(point.get("y", 0.0)))
                zs.append(float(point.get("z", 0.0)))
            except (TypeError, ValueError):
                continue
        if xs and ys and zs:
            ranges_xyz = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]

    up_vector = axis.get("up_vector") if isinstance(axis, dict) else None
    horizontal_axis = axis.get("horizontal_axis") if isinstance(axis, dict) else None

    return {
        "scene_id": payload.get("scene_id"),
        "num_points": int(payload.get("num_points", 0)),
        "coordinate_frame": axis.get("coordinate_frame") if isinstance(axis, dict) else None,
        "up_vector": up_vector if isinstance(up_vector, dict) else None,
        "horizontal_axis": horizontal_axis if isinstance(horizontal_axis, dict) else None,
        "normalization_applied": bool(normalization.get("applied")) if isinstance(normalization, dict) else False,
        "ranges_xyz": ranges_xyz,
    }
