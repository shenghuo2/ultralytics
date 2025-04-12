---
comments: true
description: 学习如何使用YOLO11高效训练目标检测模型,包括设置、数据增强和硬件利用的全面说明。
keywords: Ultralytics, YOLO11, 模型训练, 深度学习, 目标检测, GPU训练, 数据集增强, 超参数调优, 模型性能, Apple Silicon训练
---

# 使用Ultralytics YOLO进行模型训练

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO生态系统和集成">

## 简介

训练深度学习模型涉及向其输入数据并调整其参数,以便它能够做出准确的预测。Ultralytics YOLO11的训练模式旨在有效且高效地训练目标检测模型,充分利用现代硬件功能。本指南旨在涵盖您开始使用YOLO11强大功能集训练自己的模型所需的所有细节。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看:</strong> 如何在Google Colab中对自定义数据集训练YOLO模型。
</p>

## 为什么选择Ultralytics YOLO进行训练?

以下是选择YOLO11训练模式的一些令人信服的理由:

- **效率:** 无论您是使用单GPU设置还是跨多个GPU扩展,都能充分利用您的硬件。
- **多功能性:** 除了现成的数据集(如COCO、VOC和ImageNet)外,还可以在自定义数据集上进行训练。
- **用户友好:** 简单而强大的CLI和Python接口,提供直观的训练体验。
- **超参数灵活性:** 广泛的可自定义超参数,用于微调模型性能。

### 训练模式的主要特点

以下是YOLO11训练模式的一些显著特点:

- **自动数据集下载:** 首次使用时自动下载COCO、VOC和ImageNet等标准数据集。
- **多GPU支持:** 无缝地跨多个GPU扩展您的训练工作,以加快处理速度。
- **超参数配置:** 可以通过YAML配置文件或CLI参数修改超参数。
- **可视化和监控:** 实时跟踪训练指标并可视化学习过程,以获得更好的洞察力。

!!! tip

    * YOLO11数据集如COCO、VOC、ImageNet等在首次使用时会自动下载,例如`yolo train data=coco.yaml`

## 使用示例

在COCO8数据集上训练YOLO11n,训练100个epoch,图像大小为640。可以使用`device`参数指定训练设备。如果未传递参数,将使用可用的GPU `device=0`,否则将使用`device='cpu'`。有关完整的训练参数列表,请参阅下面的参数部分。

!!! warning "Windows多进程错误"

    在Windows上,当以脚本形式启动训练时,您可能会收到`RuntimeError`。在训练代码之前添加`if __name__ == "__main__":`块可以解决这个问题。

!!! example "单GPU和CPU训练示例"

    设备会自动确定。如果有GPU可用,则会使用GPU,否则训练将在CPU上开始。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.yaml")  # 从YAML构建新模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型(推荐用于训练)
        model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # 从YAML构建并传输权重

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从YAML构建新模型并从头开始训练
        yolo detect train data=coco8.yaml model=yolo11n.yaml epochs=100 imgsz=640

        # 从预训练的*.pt模型开始训练
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640

        # 从YAML构建新模型,传输预训练权重并开始训练
        yolo detect train data=coco8.yaml model=yolo11n.yaml pretrained=yolo11n.pt epochs=100 imgsz=640
        ```

### 多GPU训练

多GPU训练允许通过在多个GPU之间分配训练负载来更有效地利用可用的硬件资源。此功能可通过Python API和命令行界面使用。要启用多GPU训练,请指定您希望使用的GPU设备ID。

!!! example "多GPU训练示例"

    要使用2个GPU(CUDA设备0和1)进行训练,请使用以下命令。根据需要扩展到更多GPU。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型(推荐用于训练)

        # 使用2个GPU训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # 从预训练的*.pt模型开始训练,使用GPU 0和1
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=0,1
        ```

### Apple Silicon MPS训练

随着对Apple silicon芯片的支持集成到Ultralytics YOLO模型中,现在可以在利用强大的Metal Performance Shaders (MPS)框架的设备上训练模型。MPS提供了一种高性能的方式来执行Apple自定义silicon上的计算和图像处理任务。

要在Apple silicon芯片上启用训练,您应该在启动训练过程时将'mps'指定为设备。以下是如何在Python和命令行中执行此操作的示例:

!!! example "MPS训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型(推荐用于训练)

        # 使用MPS训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")
        ```

    === "CLI"

        ```bash
        # 从预训练的*.pt模型开始使用MPS进行训练
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=mps
        ```

利用Apple silicon芯片的计算能力,这使得训练任务的处理更加高效。有关更详细的指导和高级配置选项,请参阅[PyTorch MPS文档](https://pytorch.org/docs/stable/notes/mps.html)。

### 恢复中断的训练

从先前保存的状态恢复训练是使用深度学习模型时的一个关键功能。这在各种情况下都很有用,比如训练过程意外中断,或者当您希望继续用新数据或更多epoch训练模型时。

恢复训练时,Ultralytics YOLO会加载最后保存模型的权重,并恢复优化器状态、学习率调度器和epoch数。这允许您从中断的地方无缝地继续训练过程。

在Ultralytics YOLO中,您可以通过在调用`train`方法时将`resume`参数设置为`True`,并指定包含部分训练模型权重的`.pt`文件的路径来轻松恢复训练。

以下是如何使用Python和命令行恢复中断训练的示例:

!!! example "恢复训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/last.pt")  # 加载部分训练的模型

        # 恢复训练
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        # 恢复中断的训练
        yolo train resume model=path/to/last.pt
        ```

通过设置`resume=True`,`train`函数将使用存储在'path/to/last.pt'文件中的状态继续训练。如果省略`resume`参数或将其设置为`False`,`train`函数将开始新的训练会话。

请记住,默认情况下,检查点会在每个epoch结束时保存,或使用`save_period`参数以固定间隔保存,因此您必须至少完成1个epoch才能恢复训练运行。

## 训练设置

YOLO模型的训练设置包括训练过程中使用的各种超参数和配置。这些设置影响模型的性能、速度和准确性。关键的训练设置包括批量大小、学习率、动量和权重衰减。此外,优化器的选择、损失函数和训练数据集的组成也会影响训练过程。仔细调整和实验这些设置对于优化性能至关重要。

{% include "macros/train-args.md" %}

!!! info "批量大小设置说明"

    `batch`参数可以通过三种方式配置:

    - **固定批量大小**: 设置一个整数值(例如,`batch=16`),直接指定每批图像的数量。
    - **自动模式(60% GPU内存)**: 使用`batch=-1`自动调整批量大小,以利用约60%的CUDA内存。
    - **带利用率分数的自动模式**: 设置一个分数值(例如,`batch=0.70`)根据指定的GPU内存使用率分数调整批量大小。

## 增强设置和超参数

增强技术对于通过在训练数据中引入可变性来提高YOLO模型的鲁棒性和性能至关重要,有助于模型更好地泛化到未见过的数据。下表概述了每个增强参数的目的和效果:

{% include "macros/augmentation-args.md" %}

这些设置可以根据数据集和任务的具体要求进行调整。尝试不同的值可以帮助找到导致最佳模型性能的最佳增强策略。

!!! info

    有关训练增强操作的更多信息,请参阅[参考部分](../reference/data/augment.md)。

## 日志记录

在训练YOLO11模型时,您可能会发现跟踪模型随时间的性能很有价值。这就是日志记录发挥作用的地方。Ultralytics YOLO提供对三种类型的记录器的支持 - [Comet](../integrations/comet.md)、[ClearML](../integrations/clearml.md)和[TensorBoard](../integrations/tensorboard.md)。

要使用记录器,请从上面的代码片段中的下拉菜单中选择一个并运行它。所选的记录器将被安装和初始化。

### Comet

[Comet](../integrations/comet.md)是一个平台,允许数据科学家和开发人员跟踪、比较、解释和优化实验和模型。它提供了实时指标、代码差异和超参数跟踪等功能。

要使用Comet:

!!! example

    === "Python"

        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

记得在他们的网站上登录您的Comet帐户并获取您的API密钥。您需要将其添加到环境变量或脚本中以记录您的实验。

### ClearML

[ClearML](https://clear.ml/)是一个开源平台,可自动跟踪实验并帮助高效共享资源。它旨在帮助团队更有效地管理、执行和重现他们的ML工作。

要使用ClearML:

!!! example

    === "Python"

        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

运行此脚本后,您需要在浏览器上登录您的ClearML帐户并验证您的会话。

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard)是[TensorFlow](https://www.ultralytics.com/glossary/tensorflow)的可视化工具包。它允许你可视化你的TensorFlow图，绘制关于图执行的定量指标，并显示通过它的其他数据，如图像。

在[Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)中使用TensorBoard：

!!! example

    === "CLI"

        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs # 替换为'runs'目录
        ```

要在本地使用TensorBoard，运行以下命令并在`http://localhost:6006/`查看结果。

!!! example

    === "CLI"

        ```bash
        tensorboard --logdir ultralytics/runs # 替换为'runs'目录
        ```

这将加载TensorBoard并将其指向保存训练日志的目录。

设置好日志记录器后，你就可以继续进行模型训练。所有训练指标将自动记录在你选择的平台上，你可以访问这些日志来监控模型的性能随时间的变化，比较不同的模型，并识别需要改进的领域。

## 常见问题

### 如何使用Ultralytics YOLO11训练[目标检测](https://www.ultralytics.com/glossary/object-detection)模型？

要使用Ultralytics YOLO11训练目标检测模型，你可以使用Python API或CLI。以下是两种方式的示例：

!!! example "单GPU和CPU训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

更多详情，请参阅[训练设置](#train-settings)部分。

### Ultralytics YOLO11的训练模式有哪些主要特点？

Ultralytics YOLO11的训练模式主要特点包括：

- **自动数据集下载：** 自动下载标准数据集，如COCO、VOC和ImageNet。
- **多GPU支持：** 跨多个GPU扩展训练，以加快处理速度。
- **超参数配置：** 通过YAML文件或CLI参数自定义超参数。
- **可视化和监控：** 实时跟踪训练指标，以获得更好的洞察。

这些特点使训练变得高效且可根据你的需求进行定制。更多详情，请参阅[训练模式的主要特点](#key-features-of-train-mode)部分。

### 如何在Ultralytics YOLO11中从中断的会话恢复训练？

要从中断的会话恢复训练，将`resume`参数设置为`True`，并指定最后保存的检查点的路径。

!!! example "恢复训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载部分训练的模型
        model = YOLO("path/to/last.pt")

        # 恢复训练
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        yolo train resume model=path/to/last.pt
        ```

查看[恢复中断的训练](#resuming-interrupted-trainings)部分获取更多信息。

### 我可以在Apple silicon芯片上训练YOLO11模型吗？

是的，Ultralytics YOLO11支持在Apple silicon芯片上使用Metal Performance Shaders (MPS)框架进行训练。将'mps'指定为你的训练设备。

!!! example "MPS训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo11n.pt")

        # 在Apple silicon芯片（M1/M2/M3/M4）上训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=mps
        ```

更多详情，请参阅[Apple Silicon MPS训练](#apple-silicon-mps-training)部分。

### 常见的训练设置有哪些，我如何配置它们？

Ultralytics YOLO11允许你通过参数配置各种训练设置，如批量大小、学习率、训练周期等。以下是一个简要概述：

| 参数     | 默认值 | 描述                                                   |
| -------- | ------ | ------------------------------------------------------ |
| `model`  | `None` | 用于训练的模型文件路径。                               |
| `data`   | `None` | 数据集配置文件的路径（例如，`coco8.yaml`）。           |
| `epochs` | `100`  | 总训练周期数。                                         |
| `batch`  | `16`   | 批量大小，可调整为整数或自动模式。                     |
| `imgsz`  | `640`  | 训练的目标图像大小。                                   |
| `device` | `None` | 用于训练的计算设备，如`cpu`、`0`、`0,1`或`mps`。       |
| `save`   | `True` | 启用保存训练检查点和最终模型权重。                     |

有关训练设置的深入指南，请查看[训练设置](#train-settings)部分。
