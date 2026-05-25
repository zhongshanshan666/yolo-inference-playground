# YOLO Inference Playground

多 YOLO 版本（v5 / v8 / v11 / v26）推理实验项目，首期支持 **Ultralytics 官方导出的 ONNX**，预处理采用 **numpy + OpenCV** 混合实现；架构预留 TensorRT / RKNN 扩展。

## 快速开始

```bash
# 安装（开发模式）
pip install -e .

# 导出 ONNX（Ultralytics CLI 示例）
yolo export model=yolov8n.pt format=onnx imgsz=640
yolo export model=yolov5n.pt format=onnx imgsz=640    # 经典 v5，输出 (1, 25200, 85)
yolo export model=yolov5nu.pt format=onnx imgsz=640   # YOLOv5u，输出 (1, 84, 8400)

# 将 *.onnx 放到 weights/ 目录
mkdir weights

# YOLOv8 单张图片
python run.py -c configs/yolov8n.onnx.yaml -s path/to/image.jpg

# YOLOv5 单张图片
python run.py -c configs/yolov5n.onnx.yaml -s assets/bus.jpg
python run.py -c configs/yolov5nu.onnx.yaml -s assets/bus.jpg

# 文件夹
python run.py -c configs/yolov8n.onnx.yaml -s path/to/images/

# 视频
python run.py -c configs/yolov8n.onnx.yaml -s path/to/video.mp4 --sink visualize,video_writer

# 摄像头（预留）
python run.py -c configs/yolov8n.onnx.yaml -s webcam:0 --show
```

也可使用入口命令：`yolo-infer -c configs/yolov8n.onnx.yaml -s image.jpg`

## 项目结构

```
yolo_playground/
├── core/           # 抽象接口、类型、注册表
├── backends/       # ONNX Runtime（TensorRT / RKNN 占位）
├── models/         # YOLO 各版本前后处理
├── preprocess/     # letterbox + BGR→NCHW tensor
├── postprocess/    # NMS、坐标还原
├── sources/        # 图片 / 文件夹 / 视频 / 流
├── sinks/          # 可视化 / JSON / 写视频
├── pipeline/       # 推理编排
└── cli.py          # 命令行入口
```

## 配置说明

编辑 `configs/*.yaml`：

| 字段 | 说明 |
|------|------|
| `model.type` | `yolov5` / `yolov8` / `yolo11` / `yolo26` |
| `model.weights` | ONNX 路径 |
| `model.input_size` | `[H, W]`，需与导出时 `imgsz` 一致 |
| `backend.type` | 目前仅 `onnxruntime` |
| `inference.sinks` | `visualize` / `json` / `video_writer` |

## Ultralytics ONNX 约定

- **输入**：`images`，形状 `[1, 3, H, W]`，float32，RGB，归一化到 `[0, 1]`
- **v8 / v11 / v26 输出**：`(1, 4+nc, N)`，前 4 列为 `cx, cy, w, h`（letterbox 像素坐标）
- **v5 经典输出**（`yolov5n.pt` 等）：`(1, N, 4+1+nc)`，含 objectness，如 `(1, 25200, 85)`
- **v5u 输出**（`yolov5nu.pt` 等）：`(1, 4+nc, N)`，与 v8 相同布局，如 `(1, 84, 8400)`

`model.type: yolov5` 会根据输出通道数自动选择 decode 路径。

## 测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

See LICENSE.
