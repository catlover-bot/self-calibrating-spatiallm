import json
import subprocess
import sys
from pathlib import Path


def test_backend_parses_and_compacts_spatiallm_layout_text(tmp_path: Path) -> None:
    backend_script = Path(__file__).resolve().parents[2] / "tools" / "spatiallm_backend.py"
    fake_real_backend = tmp_path / "fake_real_backend.py"
    input_manifest = tmp_path / "input.json"
    output_json = tmp_path / "output.json"

    input_manifest.write_text(
        json.dumps(
            {
                "scene_id": "scene-test",
                "points_xyz": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            }
        ),
        encoding="utf-8",
    )

    layout_lines = []
    for index in range(30):
        offset = 0.01 * index
        layout_lines.append(f"wall_{index}=Wall(0,{offset},0,5,{offset},0,2.7,0.12)")
    for index in range(30):
        offset = 0.01 * index
        wall_id = 100 + index
        layout_lines.append(f"wall_{wall_id}=Wall(5,{offset},0,5,{4+offset},0,2.7,0.12)")
    layout_lines.append("door_0=Door(wall_0,2.5,0.0,1.0,0.9,2.0)")
    layout_lines.append("window_0=Window(wall_100,5.0,2.0,1.4,1.2,1.0)")
    layout_text = "\n".join(layout_lines)

    fake_real_backend.write_text(
        "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--input')",
                "parser.add_argument('--output', required=True)",
                "parser.add_argument('--scene')",
                "args = parser.parse_args()",
                f"Path(args.output).write_text({layout_text!r}, encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )

    real_command = (
        f"{sys.executable} {fake_real_backend} "
        "--input {spatiallm_input} --output {output_json} --scene {scene_id}"
    )
    cmd = [
        sys.executable,
        str(backend_script),
        "--input",
        str(input_manifest),
        "--output",
        str(output_json),
        "--scene",
        "scene-test",
        "--real-command",
        real_command,
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    prediction = payload["scene_prediction"]
    objects = prediction["objects"]
    relations = prediction["relations"]
    labels = [obj["label"] for obj in objects]
    wall_count = sum(1 for label in labels if label == "wall")

    assert "door" in labels
    assert "window" in labels
    assert "floor" in labels
    assert wall_count < 20  # raw text contains 60 wall lines; should be compacted substantially.
    assert any(rel["predicate"] == "attached-to" for rel in relations)
    assert prediction["metadata"]["relation_reconstruction_count"] >= 1
