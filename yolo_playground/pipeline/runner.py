from __future__ import annotations

from pathlib import Path
from typing import Any

from yolo_playground.core.model import BaseYoloModel
from yolo_playground.core.registry import BACKENDS, SINKS
from yolo_playground.core.sink import BaseResultSink
from yolo_playground.core.source import BaseFrameSource
from yolo_playground.core.types import InferResult
from yolo_playground.utils.logger import setup_logger, timer

logger = setup_logger()


class InferencePipeline:
    """Connect Source -> Model -> Sinks."""

    def __init__(
        self,
        model: BaseYoloModel,
        source: BaseFrameSource,
        sinks: list[BaseResultSink],
        save_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.source = source
        self.sinks = sinks
        self.save_dir = save_dir

    def run(self) -> list[InferResult]:
        results: list[InferResult] = []
        for sink in self.sinks:
            sink.open(self.save_dir)

        try:
            with self.source as src:
                for frame, meta in src:
                    with timer() as elapsed:
                        detections, _ = self.model.infer(frame)
                    result = InferResult(
                        frame_meta=meta,
                        detections=detections,
                        latency_ms=elapsed[0] if elapsed else 0.0,
                    )
                    results.append(result)
                    for sink in self.sinks:
                        sink.write(frame, result)
                    logger.info(
                        "frame=%d dets=%d latency=%.1fms src=%s",
                        meta.frame_id,
                        len(detections),
                        result.latency_ms,
                        meta.source_path,
                    )
        finally:
            for sink in self.sinks:
                sink.close()

        return results


def build_backend(cfg: dict[str, Any]):
    backend_type = cfg.get("type", "onnxruntime")
    if backend_type not in BACKENDS:
        raise KeyError(f"Unknown backend '{backend_type}'. Available: {list(BACKENDS)}")
    return BACKENDS[backend_type]()


def build_sinks(names: list[str], extra: dict[str, Any] | None = None) -> list[BaseResultSink]:
    extra = extra or {}
    sinks: list[BaseResultSink] = []
    for name in names:
        if name not in SINKS:
            raise KeyError(f"Unknown sink '{name}'. Available: {list(SINKS)}")
        sink_cls = SINKS[name]
        if name == "visualize":
            sinks.append(sink_cls(show=extra.get("show", False)))
        elif name == "video_writer":
            sinks.append(sink_cls(fps=extra.get("fps", 30.0)))
        else:
            sinks.append(sink_cls())
    return sinks


def create_pipeline(cfg: dict[str, Any], source: BaseFrameSource) -> InferencePipeline:
    # Ensure registries are populated
    import yolo_playground.backends  # noqa: F401
    import yolo_playground.models  # noqa: F401
    import yolo_playground.sinks  # noqa: F401

    model_cfg = cfg["model"]
    backend_cfg = cfg.get("backend", {})
    infer_cfg = cfg.get("inference", {})

    backend = build_backend(backend_cfg)
    backend.load(model_cfg["weights"], **backend_cfg)

    h, w = model_cfg.get("input_size", [640, 640])
    backend.warmup((1, 3, h, w))

    model = BaseYoloModel.from_config(model_cfg, backend)
    sinks = build_sinks(infer_cfg.get("sinks", ["visualize"]), extra=infer_cfg)
    save_dir = Path(infer_cfg.get("save_dir", "outputs"))

    return InferencePipeline(model=model, source=source, sinks=sinks, save_dir=save_dir)
