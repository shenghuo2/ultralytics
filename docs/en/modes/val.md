---
comments: true
description: 学习如何使用精确的指标、易用的工具和自定义设置来验证您的YOLO11模型,以获得最佳性能。
keywords: Ultralytics, YOLO11, 模型验证, 机器学习, 目标检测, mAP指标, Python API, CLI
---

# 使用Ultralytics YOLO进行模型验证

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO生态系统和集成">

## 简介

验证是机器学习流程中的关键步骤,它允许您评估训练模型的质量。Ultralytics YOLO11中的Val模式提供了一套强大的工具和指标,用于评估目标检测模型的性能。本指南作为一个完整的资源,帮助您了解如何有效地使用Val模式,以确保您的模型既准确又可靠。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?start=47"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看:</strong> Ultralytics模式教程:验证
</p>

## 为什么使用Ultralytics YOLO进行验证?

以下是使用YOLO11的Val模式的优势:

- **精确度:** 获得准确的指标,如mAP50、mAP75和mAP50-95,以全面评估您的模型。
- **便利性:** 利用内置功能记住训练设置,简化验证过程。
- **灵活性:** 使用相同或不同的数据集和图像大小验证您的模型。
- **超参数调整:** 使用验证指标微调您的模型以获得更好的性能。

### Val模式的主要特点

以下是YOLO11的Val模式提供的显著功能:

- **自动设置:** 模型记住其训练配置,便于直接验证。
- **多指标支持:** 基于一系列准确性指标评估您的模型。
- **CLI和Python API:** 根据您的偏好选择命令行界面或Python API进行验证。
- **数据兼容性:** 与训练阶段使用的数据集以及自定义数据集无缝配合。

!!! tip

    * YOLO11模型会自动记住其训练设置,因此您可以轻松地以相同的图像大小和原始数据集验证模型,只需使用`yolo val model=yolo11n.pt`或`model('yolo11n.pt').val()`

## 使用示例

在COCO8数据集上验证训练好的YOLO11n模型的准确性。不需要任何参数,因为`model`将其训练`data`和参数保留为模型属性。有关验证参数的完整列表,请参见下面的参数部分。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 不需要参数,数据集和设置已记住
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # 包含每个类别map50-95的列表
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolo11n.pt      # 验证官方模型
        yolo detect val model=path/to/best.pt # 验证自定义模型
        ```

## YOLO模型验证的参数

在验证YOLO模型时,可以微调几个参数以优化评估过程。这些参数控制输入图像大小、批处理和性能阈值等方面。以下是每个参数的详细分解,以帮助您有效地自定义验证设置。

{% include "macros/validation-args.md" %}

这些设置中的每一个都在验证过程中发挥着重要作用,允许对YOLO模型进行可定制和高效的评估。根据您的具体需求和资源调整这些参数可以帮助在准确性和性能之间取得最佳平衡。

### 带参数的验证示例

以下示例展示了在Python和CLI中使用自定义参数进行YOLO模型验证。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")

        # 自定义验证设置
        validation_results = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
        ```

    === "CLI"

        ```bash
        yolo val model=yolo11n.pt data=coco8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=0
        ```

## 常见问题

### 如何使用Ultralytics验证我的YOLO11模型?

要验证您的YOLO11模型,您可以使用Ultralytics提供的Val模式。例如,使用Python API,您可以加载模型并运行验证:

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 验证模型
metrics = model.val()
print(metrics.box.map)  # map50-95
```

或者,您可以使用命令行界面(CLI):

```bash
yolo val model=yolo11n.pt
```

要进一步自定义,您可以在Python和CLI模式下调整各种参数,如`imgsz`、`batch`和`conf`。查看[YOLO模型验证的参数](#yolo模型验证的参数)部分以获取完整的参数列表。

### 从YOLO11模型验证中可以获得哪些指标?

YOLO11模型验证提供了几个关键指标来评估模型性能。这些包括:

- mAP50(IoU阈值为0.5的平均精度均值)
- mAP75(IoU阈值为0.75的平均精度均值)
- mAP50-95(多个IoU阈值从0.5到0.95的平均精度均值)

使用Python API,您可以按如下方式访问这些指标:

```python
metrics = model.val()  # 假设已加载`model`
print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.maps)  # 每个类别的mAP50-95列表
```

为了全面评估性能,审查所有这些指标至关重要。有关更多详细信息,请参阅[Val模式的主要特点](#val模式的主要特点)。

### 使用Ultralytics YOLO进行验证有哪些优势?

使用Ultralytics YOLO进行验证提供了几个优势:

- **精确度:** YOLO11提供准确的性能指标,包括mAP50、mAP75和mAP50-95。
- **便利性:** 模型记住其训练设置,使验证变得简单直接。
- **灵活性:** 您可以针对相同或不同的数据集和图像大小进行验证。
- **超参数调整:** 验证指标有助于微调模型以获得更好的性能。

这些优势确保您的模型得到彻底评估,并可以优化以获得更好的结果。在[为什么使用Ultralytics YOLO进行验证](#为什么使用ultralytics-yolo进行验证)部分了解更多关于这些优势的信息。

### 我可以使用自定义数据集验证我的YOLO11模型吗?

是的,您可以使用自定义数据集验证您的YOLO11模型。指定`data`参数为您的数据集配置文件的路径。该文件应包括验证数据的路径、类名和其他相关详细信息。

Python示例:

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 使用自定义数据集验证
metrics = model.val(data="path/to/your/custom_dataset.yaml")
print(metrics.box.map)  # map50-95
```

使用CLI的示例:

```bash
yolo val model=yolo11n.pt data=path/to/your/custom_dataset.yaml
```

有关验证期间更多可自定义选项,请参见[带参数的验证示例](#带参数的验证示例)部分。

### 如何在YOLO11中将验证结果保存到JSON文件?

要将验证结果保存到JSON文件,您可以在运行验证时将`save_json`参数设置为`True`。这可以在Python API和CLI中完成。

Python示例:

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 将验证结果保存到JSON
metrics = model.val(save_json=True)
```

使用CLI的示例:

```bash
yolo val model=yolo11n.pt save_json=True
```

这个功能对于进一步分析或与其他工具集成特别有用。查看[YOLO模型验证的参数](#yolo模型验证的参数)以获取更多详细信息。
