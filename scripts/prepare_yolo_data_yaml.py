from __future__ import annotations

import argparse
import ast
from pathlib import Path


def parse_names(data_yaml: Path) -> list[str]:
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            names = ast.literal_eval(stripped.split(":", 1)[1].strip())
            return [str(name) for name in names]
    raise ValueError(f"Could not find names in {data_yaml}")


def _path_for_yaml(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def build_yolo_data_yaml(dataset: Path) -> str:
    dataset = dataset.resolve()
    names = parse_names(dataset / "data.yaml")
    lines = [
        f"train: {_path_for_yaml(dataset / 'train' / 'images')}",
        f"val: {_path_for_yaml(dataset / 'valid' / 'images')}",
        f"test: {_path_for_yaml(dataset / 'test' / 'images')}",
        "",
        f"nc: {len(names)}",
        f"names: {names!r}",
        "",
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a local absolute-path YOLO data.yaml.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    content = build_yolo_data_yaml(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"YOLO data config created: {args.output}")


if __name__ == "__main__":
    main()
