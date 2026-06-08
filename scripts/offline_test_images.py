from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline YOLO inference on dataset test/images.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/student-classroom-activity-v2"))
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("output/offline_test/student-classroom-activity-v2"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser


def iter_images(images_dir: Path, limit: int) -> list[Path]:
    if not images_dir.exists():
        raise SystemExit(f"Image directory does not exist: {images_dir}")
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found under {images_dir}")
    if limit > 0:
        return images[:limit]
    return images


def write_prediction_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "label", "confidence", "x1", "y1", "x2", "y2", "sleep_candidate"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()
    images_dir = args.images or args.dataset / "test" / "images"
    image_paths = iter_images(images_dir, args.limit)

    model = YOLO(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for image_path in image_paths:
        results = model(str(image_path), conf=args.conf, verbose=False)
        result = results[0]
        annotated = result.plot()
        cv2.imwrite(str(args.output_dir / image_path.name), annotated)

        if result.boxes is None or len(result.boxes) == 0:
            rows.append(
                {
                    "image": image_path.name,
                    "label": "",
                    "confidence": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "sleep_candidate": "false",
                }
            )
            continue

        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = str(result.names.get(cls_id, cls_id))
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            rows.append(
                {
                    "image": image_path.name,
                    "label": label,
                    "confidence": f"{confidence:.4f}",
                    "x1": f"{x1:.1f}",
                    "y1": f"{y1:.1f}",
                    "x2": f"{x2:.1f}",
                    "y2": f"{y2:.1f}",
                    "sleep_candidate": str(label.lower() in {"sleep", "desk_sleep", "head_down", "lying"}).lower(),
                }
            )

    csv_path = args.output_dir / "predictions.csv"
    write_prediction_rows(csv_path, rows)
    print(f"Images tested: {len(image_paths)}")
    print(f"Prediction rows: {len(rows)}")
    print(f"Annotated images: {args.output_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

