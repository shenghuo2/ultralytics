---
comments: true
description: Learn how to evaluate your YOLO11 model's performance in real-world scenarios using benchmark mode. Optimize speed, accuracy, and resource allocation across export formats.
keywords: model benchmarking, YOLO11, Ultralytics, performance evaluation, export formats, ONNX, TensorRT, OpenVINO, CoreML, TensorFlow, optimization, mAP50-95, inference time
---

# 使用 Ultralytics YOLO 进行模型基准测试

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO生态系统和集成">

## 基准可视化

!!! tip "刷新浏览器"

由于可能存在 cookie 问题，您可能需要刷新页面才能正确查看图表。

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400"></canvas>

## 介绍

一旦您的模型经过训练和验证,下一个合乎逻辑的步骤就是评估其在各种真实场景中的表现。Ultralytics YOLO11中的基准测试模式正是为此目的而设计,它提供了一个强大的框架来评估速度和[accuracy](https://www.ultralytics.com/glossary/accuracy)您的模型可以导出为多种格式。

<p align="center">
<br>
<iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rEQlAaevEFc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
</iframe>
<br>
<strong>观看：</strong>基准测试Ultralytics YOLO11模型 | 如何在不同硬件上比较模型性能？
</p>

## 为什么基准测试至关重要？

- **明智决策：** 深入了解速度与准确性之间的权衡。
- **资源分配：** 了解不同导出格式在不同硬件上的表现。
- **优化：** 了解哪种导出格式能为您的特定用例提供最佳性能。
- **成本效益：**根据基准测试结果更高效地利用硬件资源。

### 关键指标（基准模式）

- **mAP50-95:** 对于[object detection](https://www.ultralytics.com/glossary/object-detection)、分割和姿态估计。
- **准确率前5名：** 对于[image classification](https://www.ultralytics.com/glossary/image-classification).
- **推理时间：** 每张图像所需的毫秒数。

### 支持的导出格式

- **ONNX:** 为获得最佳CPU性能
- **TensorRT:** 为实现最大的 GPU 效率
- OpenVINO: 用于英特尔硬件优化
- CoreML、TensorFlow SavedModel 等：满足多样化的部署需求。

!!! 提示

    * 导出为ONNX或OpenVINO格式，可实现高达3倍的CPU加速。
    * 导出至TensorRT以获得高达5倍的GPU加速。

## 使用示例

在所有支持的导出格式上运行YOLO11n基准测试,包括ONNX、TensorRT等。有关导出参数的完整列表,请参阅下面的参数部分。

!!! 示例

=== "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark on GPU
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)

        # Benchmark specific export format
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, format="onnx")
        ```

=== "命令行界面"

        ```bash
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 half=False device=0

        # Benchmark specific export format
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 format=onnx
        ```

## 论点

类似以下这`model`, `data`, `imgsz`, `half`, `device`, `verbose`和`format`为用户提供灵活性,以便根据其特定需求微调基准测试,并轻松比较不同导出格式的性能。

| 键名       | 默认值 | 描述                                                                                                                                                                                             |
| --------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`   | `None`| 指定模型文件的路径。接受`.pt`和`.yaml`格式，例如，`"yolo11n.pt"`对于预训练模型或配置文件。
| `data`    | `None`| YAML 文件路径，用于定义基准测试的数据集，通常包括路径和设置[validation data](https://www.ultralytics.com/glossary/validation-data)示例：`"coco8.yaml"`. |
| `imgsz`   | `640`| 模型的输入图像尺寸。可以是单个整数用于正方形图像，或者是一个元组`(width, height)`对于非正方形，例如，`(640, 480)`.                                                          |
| `half`    | `False`| 启用 FP16（半精度）推理，减少内存使用，并可能在兼容的硬件上提高速度。使用`half=True`以启用。
| `int8`    | `False`| 在支持的设备上激活 INT8 量化以进一步优化性能,特别适用于边缘设备。设置`int8=True`使用。
| `device`  | `None`| 定义用于基准测试的计算设备，例如`"cpu"`我理解了。我会以专业翻译的身份,将您提供的内容翻译或改写成简体中文,同时尊重原意并保持原有格式。我只会提供翻译或改写后的内容,不会添加任何解释、评论或额外文字。请提供您希望我翻译或改写的内容。`"cuda:0"`.                                                                                                                      |
| `verbose` | `False`| 控制日志输出的详细程度。布尔值；设置`verbose=True`对于详细日志或用于错误阈值的浮点数。
| `format`  | `''`| 对模型进行单一导出格式的基准测试。即`format=onnx`                                                                                                                                        |

## 导出格式

基准测试将尝试自动运行以下列出的所有可能导出格式。或者，您可以通过使用`format`参数，可接受以下任何一种格式。

{% include "macros/export-table.md" %}

查看全文`export`细节在[Export](../modes/export.md)页面。

## 常见问题

### 如何使用Ultralytics对我的YOLO11模型性能进行基准测试？

Ultralytics YOLO11提供了一种基准测试模式，用于评估您的模型在不同导出格式下的性能。这种模式提供了以下关键指标的洞察：[mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map)(mAP50-95)、准确率和推理时间（以毫秒为单位）。您可以使用Python或CLI命令来运行基准测试。例如，要在GPU上进行基准测试：

!!! 示例

=== "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark on GPU
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

=== "命令行界面"

        ```bash
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

有关基准参数的更多详细信息，请访问[Arguments](#arguments)章节。

### 将YOLO11模型导出为不同格式有什么好处？

将YOLO11模型导出为不同格式，例如[ONNX](https://docs.ultralytics.com/integrations/onnx/), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/)和[OpenVINO](https://docs.ultralytics.com/integrations/openvino/)允许您根据部署环境优化性能。例如:

- ONNX:提供高达3倍的CPU加速。
- TensorRT：提供高达5倍的GPU加速。
- **OpenVINO:** 专门针对英特尔硬件进行优化。

这些格式既提高了模型的速度，又提升了其准确性，使其在各种实际应用中更加高效。访问[Export](../modes/export.md)请参阅完整页面以获取详细信息。

### 为什么基准测试在评估YOLO11模型中至关重要？

对YOLO11模型进行基准测试对于以下几个原因至关重要：

- **明智决策：** 理解速度与准确性之间的权衡。
- **资源分配：** 评估不同硬件选项的性能表现。
- **优化：** 确定哪种导出格式能为特定用例提供最佳性能。
- **成本效率：** 根据基准测试结果优化硬件使用。

关键指标如mAP50-95、Top-5准确率和推理时间有助于进行这些评估。参考[Key Metrics](#key-metrics-in-benchmark-mode)有关更多信息,请参阅 section 部分。

### YOLO11支持哪些导出格式,它们各有什么优势?

YOLO11支持多种导出格式，每种格式都针对特定的硬件和使用场景进行了优化：

- ONNX：最适合 CPU 性能。
- TensorRT：GPU 效率的理想选择。
- **OpenVINO：**针对英特尔硬件优化。
- **CoreML与[TensorFlow](https://www.ultralytics.com/glossary/tensorflow)：**对iOS和通用机器学习应用很有用。

要获取支持的格式完整列表及其各自的优势,请查看[Supported Export Formats](#supported-export-formats)章节。

### 我可以使用哪些参数来微调我的YOLO11基准测试?

运行基准测试时，可以根据具体需求自定义以下几个参数：

- **model:** 模型文件的路径（例如，"yolo11n.pt"）。
- **data:** YAML 文件路径，用于定义数据集（例如，"coco8.yaml"）。
- **imgsz:** 输入图像的尺寸，可以是单个整数或元组。
- **半精度：** 启用 FP16 推理以获得更好的性能。
- **int8:** 为边缘设备激活INT8量化。
- **设备:** 指定计算设备（例如，"cpu"、"cuda:0"）。
- **详细程度：** 控制日志记录的详细级别。

有关完整的参数列表，请参阅[Arguments](#arguments)章节。
