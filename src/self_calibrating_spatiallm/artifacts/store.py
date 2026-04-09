"""Artifact persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Protocol


class SaveableJsonArtifact(Protocol):
    def save_json(self, path: Path) -> Path:
        ...


class ArtifactStore:
    """Writes pipeline artifacts into a run directory."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_artifact(self, filename: str, artifact: SaveableJsonArtifact) -> Path:
        return artifact.save_json(self.root_dir / filename)

    def save_text(self, filename: str, content: str) -> Path:
        path = self.root_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def save_json(self, filename: str, payload: object) -> Path:
        path = self.root_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def save_manifest(self, entries: Mapping[str, Path], filename: str = "manifest.json") -> Path:
        payload = {key: self._relative(value) for key, value in entries.items()}
        path = self.root_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root_dir))
        except ValueError:
            return str(path)
