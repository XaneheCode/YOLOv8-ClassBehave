from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import requests


DEFAULT_WORKSPACE = "studentactivity"
DEFAULT_PROJECT = "new-student-classroom-activity-2"
DEFAULT_VERSION = 2
DEFAULT_FORMAT = "yolov8"
DEFAULT_OUTPUT = Path("datasets/student-classroom-activity-v2")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a Roboflow dataset export.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY") or os.getenv("RF_API_KEY"))
    parser.add_argument("--force", action="store_true", help="Replace an existing dataset directory.")
    return parser


def require_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    raise SystemExit(
        "Roboflow requires an API key for dataset downloads. "
        "Set it for this terminal with: $env:ROBOFLOW_API_KEY='your_key'"
    )


def api_url(workspace: str, project: str, version: int, export_format: str) -> str:
    return f"https://api.roboflow.com/{workspace}/{project}/{version}/{export_format}"


def fetch_download_url(url: str, api_key: str) -> str:
    response = requests.get(url, params={"api_key": api_key}, timeout=60)
    if response.status_code == 401:
        raise SystemExit("Roboflow rejected the API key. Check ROBOFLOW_API_KEY.")
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = response.json()
        for key in ("link", "download", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        raise SystemExit(f"Roboflow response did not contain a download URL: {payload}")

    if "zip" in content_type or response.content[:2] == b"PK":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as f:
            f.write(response.content)
            return Path(f.name).as_uri()

    raise SystemExit(f"Unexpected Roboflow response content type: {content_type}")


def download_zip(download_url: str, zip_path: Path) -> None:
    parsed = urlparse(download_url)
    if parsed.scheme == "file":
        shutil.copyfile(Path(url2pathname(parsed.path)), zip_path)
        return

    with requests.get(download_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def normalize_extracted_dataset(output: Path) -> None:
    data_yaml = list(output.rglob("data.yaml"))
    if not data_yaml:
        raise SystemExit(f"Download extracted, but no data.yaml was found under {output}")

    root = data_yaml[0].parent
    if root == output:
        return

    temp_root = output.with_name(output.name + "_normalized")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)

    for child in root.iterdir():
        shutil.move(str(child), temp_root / child.name)

    shutil.rmtree(output)
    temp_root.rename(output)


def verify_yolov8_dataset(output: Path) -> None:
    required = [output / "data.yaml", output / "test" / "images"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit("Dataset is missing expected YOLOv8 paths: " + ", ".join(str(p) for p in missing))

    image_count = sum(1 for path in (output / "test" / "images").iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if image_count == 0:
        raise SystemExit(f"No test images found under {output / 'test' / 'images'}")
    print(f"Dataset ready: {output}")
    print(f"Test images: {image_count}")


def main() -> None:
    args = build_arg_parser().parse_args()
    api_key = require_api_key(args.api_key)
    output = args.output

    if output.exists():
        if not args.force:
            print(f"Dataset already exists: {output}")
            verify_yolov8_dataset(output)
            return
        shutil.rmtree(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "roboflow_dataset.zip"
        url = api_url(args.workspace, args.project, args.version, args.format)
        download_url = fetch_download_url(url, api_key)
        print(f"Downloading Roboflow export: {args.workspace}/{args.project}/{args.version}/{args.format}")
        download_zip(download_url, zip_path)
        output.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output)

    normalize_extracted_dataset(output)
    verify_yolov8_dataset(output)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Network error while downloading dataset: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
