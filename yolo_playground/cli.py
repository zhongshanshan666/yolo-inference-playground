from __future__ import annotations

import argparse
from pathlib import Path

from yolo_playground.pipeline.runner import create_pipeline
from yolo_playground.sources import create_source
from yolo_playground.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO inference playground")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/yolov8n.onnx.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Image / folder / video path, webcam:N, or RTSP URL",
    )
    parser.add_argument(
        "--sink",
        type=str,
        default=None,
        help="Comma-separated sinks: visualize,json,video_writer",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live window (visualize sink)",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.sink:
        overrides["inference"] = {"sinks": [s.strip() for s in args.sink.split(",")]}
    if args.save_dir:
        overrides.setdefault("inference", {})["save_dir"] = args.save_dir
    if args.show:
        overrides.setdefault("inference", {})["show"] = True

    cfg = load_config(args.config, overrides=overrides or None)
    source = create_source(args.source)
    pipeline = create_pipeline(cfg, source)
    pipeline.run()


if __name__ == "__main__":
    main()
