---
comments: true
description: 了解如何将您的YOLO11模型导出为各种格式，如ONNX、TensorRT和CoreML。实现最大兼容性和性能。
keywords: YOLO11, 模型导出, ONNX, TensorRT, CoreML, Ultralytics, 人工智能, 机器学习, 推理, 部署
---

# 使用Ultralytics YOLO进行模型导出

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO生态系统和集成">

## 简介

训练模型的最终目标是将其部署到实际应用中。Ultralytics YOLO11的导出模式提供了多种选项，可以将您训练好的模型导出为不同格式，使其可以在各种平台和设备上部署。本综合指南旨在引导您了解模型导出的细节，展示如何实现最大兼容性和性能。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何导出自定义训练的Ultralytics YOLO模型并在网络摄像头上运行实时推理。
</p>

## 为什么选择YOLO11的导出模式？

- **多功能性：**可导出为多种格式，包括[ONNX](../integrations/onnx.md)、[TensorRT](../integrations/tensorrt.md)、[CoreML](../integrations/coreml.md)等。
- **性能：**使用TensorRT可获得高达5倍的GPU加速，使用ONNX或[OpenVINO](../integrations/openvino.md)可获得3倍的CPU加速。
- **兼容性：**使您的模型可以在众多硬件和软件环境中通用部署。
- **易用性：**简单的CLI和Python API，可快速直观地导出模型。

### 导出模式的主要特点

以下是一些突出的功能：

- **一键导出：**简单的命令即可导出为不同格式。
- **批量导出：**导出支持批量推理的模型。
- **优化推理：**导出的模型经过优化，可实现更快的推理时间。
- **教程视频：**深入的指南和教程，确保顺畅的导出体验。

!!! tip

    * 导出为[ONNX](../integrations/onnx.md)或[OpenVINO](../integrations/openvino.md)可获得高达3倍的CPU加速。
    * 导出为[TensorRT](../integrations/tensorrt.md)可获得高达5倍的GPU加速。

## 使用示例

将YOLO11n模型导出为不同格式，如ONNX或TensorRT。有关导出参数的完整列表，请参阅下面的参数部分。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx      # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

## 参数

此表详细说明了将YOLO模型导出为不同格式的配置和选项。这些设置对于优化导出模型的性能、大小和在各种平台和环境中的兼容性至关重要。正确的配置可确保模型准备好以最佳效率部署到预期的应用中。

{% include "macros/export-args.md" %}

调整这些参数可以根据特定要求自定义导出过程，例如部署环境、硬件限制和性能目标。选择适当的格式和设置对于在模型大小、速度和[准确性](https://www.ultralytics.com/glossary/accuracy)之间达到最佳平衡至关重要。

## 导出格式

下表列出了可用的YOLO11导出格式。您可以使用`format`参数导出为任何格式，例如`format='onnx'`或`format='engine'`。您可以直接在导出的模型上进行预测或验证，例如`yolo predict model=yolo11n.onnx`。导出完成后会显示您的模型的使用示例。

{% include "macros/export-table.md" %}

## 常见问题

### 如何将YOLO11模型导出为ONNX格式？

使用Ultralytics将YOLO11模型导出为ONNX格式非常简单。它提供了Python和CLI两种方法来导出模型。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx      # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

有关该过程的更多详细信息，包括处理不同输入大小等高级选项，请参阅[ONNX集成指南](../integrations/onnx.md)。

### 使用TensorRT进行模型导出有哪些好处？

使用TensorRT进行模型导出可以显著提高性能。导出为TensorRT的YOLO11模型可以实现高达5倍的GPU加速，非常适合实时推理应用。

- **多功能性：**针对特定硬件设置优化模型。
- **速度：**通过高级优化实现更快的推理。
- **兼容性：**与NVIDIA硬件无缝集成。

要了解更多关于集成TensorRT的信息，请参阅[TensorRT集成指南](../integrations/tensorrt.md)。

### 如何在导出YOLO11模型时启用INT8量化？

INT8量化是压缩模型并加速推理的绝佳方法，特别是在边缘设备上。以下是如何启用INT8量化：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # 加载模型
        model.export(format="engine", int8=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=engine int8=True # 导出带有INT8量化的TensorRT模型
        ```

INT8量化可以应用于各种格式，如[TensorRT](../integrations/tensorrt.md)、[OpenVINO](../integrations/openvino.md)和[CoreML](../integrations/coreml.md)。为获得最佳量化结果，请使用`data`参数提供具有代表性的[数据集](https://docs.ultralytics.com/datasets/)。

### 为什么在导出模型时动态输入大小很重要？

动态输入大小允许导出的模型处理不同的图像尺寸，为不同的使用场景提供灵活性并优化处理效率。在导出为[ONNX](../integrations/onnx.md)或[TensorRT](../integrations/tensorrt.md)等格式时，启用动态输入大小可确保模型能够无缝适应不同的输入形状。

要启用此功能，在导出时使用`dynamic=True`标志：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx dynamic=True
        ```

动态输入大小对于输入尺寸可能变化的应用特别有用，例如视频处理或处理来自不同来源的图像。

### 优化模型性能需要考虑哪些关键导出参数？

理解和配置导出参数对于优化模型性能至关重要：

- **`format:`** 导出模型的目标格式（例如，`onnx`、`torchscript`、`tensorflow`）。
- **`imgsz:`** 模型输入所需的图像大小（例如，`640`或`(height, width)`）。
- **`half:`** 启用FP16量化，减小模型大小并可能加速推理。
- **`optimize:`** 应用特定优化，适用于移动或受限环境。
- **`int8:`** 启用INT8量化，对[边缘AI](https://www.ultralytics.com/blog/deploying-computer-vision-applications-on-edge-ai-devices)部署非常有益。

对于在特定硬件平台上的部署，请考虑使用专门的导出格式，如NVIDIA GPU的[TensorRT](../integrations/tensorrt.md)、Apple设备的[CoreML](../integrations/coreml.md)或Google Coral设备的[Edge TPU](../integrations/edge-tpu.md)。
