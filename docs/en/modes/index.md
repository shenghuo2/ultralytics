---
评论: true
描述: 探索Ultralytics YOLO11的多种模式,包括训练、验证、预测、导出、跟踪和基准测试。最大化模型性能和效率。
关键词: Ultralytics, YOLO11, 机器学习, 模型训练, 验证, 预测, 导出, 跟踪, 基准测试, 物体检测
---

# Ultralytics YOLO11 模式

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO生态系统和集成">

## 简介

Ultralytics YOLO11不仅仅是另一个物体检测模型;它是一个versatile框架,旨在覆盖机器学习模型的整个生命周期—从数据摄入和模型训练到验证、部署和实际跟踪。每种模式都服务于特定目的,并设计为为不同任务和用例提供所需的灵活性和效率。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看:</strong> Ultralytics模式教程:训练、验证、预测、导出和基准测试。
</p>

### 模式概览

了解Ultralytics YOLO11支持的不同**模式**对于充分利用您的模型至关重要:

- **训练**模式:在自定义或预加载数据集上微调您的模型。
- **验证**模式:训练后的检查点,用于验证模型性能。
- **预测**模式:释放您的模型在真实世界数据上的预测能力。
- **导出**模式:使您的模型准备好以各种格式部署。
- **跟踪**模式:将您的物体检测模型扩展到实时跟踪应用。
- **基准测试**模式:分析您的模型在不同部署环境中的速度和准确性。

这份全面指南旨在为您提供每种模式的概述和实际洞察,帮助您充分发挥YOLO11的潜力。

## [训练](train.md)

训练模式用于在自定义数据集上训练YOLO11模型。在此模式下,使用指定的数据集和超参数训练模型。训练过程涉及优化模型的参数,以便它可以准确预测图像中物体的类别和位置。训练对于创建可以识别与您的应用相关的特定物体的模型至关重要。

[训练示例](train.md){ .md-button }

[Val](val.md)

Val模式用于在训练完成后验证YOLO11模型。在这种模式下,模型会在验证集上进行评估,以衡量其准确性和泛化性能。验证有助于识别潜在问题,如[过拟合](https://www.ultralytics.com/glossary/overfitting),并提供诸如[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)(mAP)等指标来量化模型性能。这种模式对于调整超参数和提高整体模型效果至关重要。

[Val示例](val.md){ .md-button }

## [Predict](predict.md)

Predict模式用于使用训练好的YOLO11模型对新的图像或视频进行预测。在这种模式下,模型从检查点文件加载,用户可以提供图像或视频来执行推理。模型识别并定位输入媒体中的对象,使其可以应用于实际问题。Predict模式是将训练好的模型应用于解决实际问题的入口。

[Predict示例](predict.md){ .md-button }

## [Export](export.md)

Export模式用于将YOLO11模型转换为适合在不同平台和设备上部署的格式。这种模式将PyTorch模型转换为优化的格式,如ONNX、TensorRT或CoreML,使其能够在生产环境中部署。导出对于将模型与各种软件应用程序或硬件设备集成至关重要,通常会带来显著的性能提升。

[Export示例](export.md){ .md-button }

## [Track](track.md)

Track模式扩展了YOLO11的对象检测能力,可以跟踪视频帧或实时流中的对象。这种模式对于需要持续对象识别的应用特别有价值,例如[监控系统](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai)或[自动驾驶汽车](https://www.ultralytics.com/solutions/ai-in-automotive)。Track模式实现了复杂的算法,如ByteTrack,以在帧之间保持对象身份,即使对象暂时从视图中消失。

[Track示例](track.md){ .md-button }

## [Benchmark](benchmark.md)

Benchmark模式对YOLO11的各种导出格式的速度和准确性进行分析。这种模式提供了全面的指标,包括模型大小、准确性(检测任务的mAP50-95或分类任务的accuracy_top5)以及不同格式(如ONNX、[OpenVINO](https://docs.ultralytics.com/integrations/openvino/)和TensorRT)的推理时间。基准测试有助于您根据部署环境中对速度和准确性的具体要求选择最佳的导出格式。

[Benchmark示例](benchmark.md){ .md-button }

## 常见问题

### 如何使用Ultralytics YOLO11训练自定义[对象检测](https://www.ultralytics.com/glossary/object-detection)模型?

使用Ultralytics YOLO11训练自定义对象检测模型涉及使用train模式。您需要一个按YOLO格式组织的数据集,包含图像和相应的注释文件。使用以下命令开始训练过程:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的YOLO模型(您可以选择n、s、m、l或x版本)
        model = YOLO("yolo11n.pt")

        # 在自定义数据集上开始训练
        model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从命令行训练YOLO模型
        yolo train data=path/to/dataset.yaml epochs=100 imgsz=640
        ```

有关更详细的说明,您可以参考[Ultralytics训练指南](../modes/train.md)。

### Ultralytics YOLO11使用哪些指标来验证模型的性能?

Ultralytics YOLO11在验证过程中使用各种指标来评估模型性能。这些包括:

- **mAP(平均精度均值)**: 这评估对象检测的准确性。
- **IOU(交并比)**: 测量预测边界框和真实边界框之间的重叠。
- **[精确度](https://www.ultralytics.com/glossary/precision)和[召回率](https://www.ultralytics.com/glossary/recall)**: 精确度测量真阳性检测与总检测阳性的比率,而召回率测量真阳性检测与总实际阳性的比率。

您可以运行以下命令开始验证:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练或自定义YOLO模型
        model = YOLO("yolo11n.pt")

        # 在您的数据集上运行验证
        model.val(data="path/to/validation.yaml")
        ```

    === "CLI"

        ```bash
        # 从命令行验证YOLO模型
        yolo val data=path/to/validation.yaml
        ```

有关更多详细信息,请参阅[验证指南](../modes/val.md)。

### 如何导出YOLO11模型以进行部署?

Ultralytics YOLO11提供导出功能,可将训练好的模型转换为各种部署格式,如ONNX、TensorRT、CoreML等。使用以下示例导出您的模型:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载训练好的YOLO模型
        model = YOLO("yolo11n.pt")

        # 将模型导出为ONNX格式(您可以根据需要指定其他格式)
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        # 从命令行将YOLO模型导出为ONNX格式
        yolo export model=yolo11n.pt format=onnx
        ```

每种导出格式的详细步骤可以在[导出指南](../modes/export.md)中找到。

### Ultralytics YOLO11中benchmark模式的目的是什么?

Ultralytics YOLO11中的Benchmark模式用于分析各种导出格式(如ONNX、TensorRT和OpenVINO)的速度和[准确性](https://www.ultralytics.com/glossary/accuracy)。它提供了诸如模型大小、对象检测的`mAP50-95`以及在不同硬件设置下的推理时间等指标,帮助您为部署需求选择最合适的格式。

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 在GPU(设备0)上运行基准测试
        # 您可以根据需要调整模型、数据集、图像大小和精度等参数
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        # 从命令行对YOLO模型进行基准测试
        # 根据您的具体用例调整参数
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

有关更多详细信息,请参阅[基准测试指南](../modes/benchmark.md)。

### 如何使用Ultralytics YOLO11进行实时对象跟踪?

使用Ultralytics YOLO11的track模式可以实现实时对象跟踪。这种模式扩展了对象检测能力,可以跟踪视频帧或实时流中的对象。使用以下示例启用跟踪:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的YOLO模型
        model = YOLO("yolo11n.pt")

        # 开始跟踪视频中的对象
        # 您也可以使用实时视频流或网络摄像头输入
        model.track(source="path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # 从命令行对视频进行对象跟踪
        # 您可以指定不同的源,如网络摄像头(0)或RTSP流
        yolo track source=path/to/video.mp4
        ```

有关详细说明,请访问[跟踪指南](../modes/track.md)。
