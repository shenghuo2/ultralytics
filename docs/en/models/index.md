---
comments: true
description: 探索Ultralytics支持的各种模型，包括YOLOv3到YOLO11、NAS、SAM和RT-DETR，适用于检测、分割等多种任务。
keywords: Ultralytics, 支持模型, YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, SAM, SAM2, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, YOLO-World, 目标检测, 图像分割, 分类, 姿态估计, 多目标跟踪
---

# Ultralytics 支持的模型

欢迎来到Ultralytics的模型文档！我们支持多种模型，每种模型都针对特定任务进行了优化，如[目标检测](../tasks/detect.md)、[实例分割](../tasks/segment.md)、[图像分类](../tasks/classify.md)、[姿态估计](../tasks/pose.md)和[多目标跟踪](../modes/track.md)。如果您有兴趣将您的模型架构贡献给Ultralytics，请查看我们的[贡献指南](../help/contributing.md)。

![Ultralytics YOLO11 对比图](https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png)

## 特色模型

以下是一些支持的主要模型：

1. **[YOLOv3](yolov3.md)**: YOLO模型家族的第三代，由Joseph Redmon提出，以其高效的实时目标检测能力闻名。
2. **[YOLOv4](yolov4.md)**: 由Alexey Bochkovskiy在2020年发布的YOLOv3的darknet原生更新版本。
3. **[YOLOv5](yolov5.md)**: Ultralytics改进的YOLO架构版本，相比之前版本提供了更好的性能和速度折衷。
4. **[YOLOv6](yolov6.md)**: 由[美团](https://www.meituan.com/)在2022年发布，并广泛应用于该公司的自主配送机器人。
5. **[YOLOv7](yolov7.md)**: 由YOLOv4作者在2022年发布的更新版YOLO模型。
6. **[YOLOv8](yolov8.md)**: 一个多功能模型，具备增强功能如[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)、姿态/关键点估计和分类。
7. **[YOLOv9](yolov9.md)**: 一个实验性模型，基于Ultralytics [YOLOv5](yolov5.md)代码库训练，实现了可编程梯度信息（PGI）。
8. **[YOLOv10](yolov10.md)**: 由清华大学提出，采用无NMS训练和效率-精度驱动的架构，提供最先进的性能和延迟。
9. **[YOLO11](yolo11.md) 🚀 新**: Ultralytics最新的YOLO模型，在检测、分割、姿态估计、跟踪和分类等多个任务中提供最先进的性能。
10. **[Segment Anything Model (SAM)](sam.md)**: Meta的原始Segment Anything Model (SAM)。
11. **[Segment Anything Model 2 (SAM2)](sam-2.md)**: Meta的下一代Segment Anything Model (SAM)，适用于视频和图像。
12. **[Mobile Segment Anything Model (MobileSAM)](mobile-sam.md)**: 由庆熙大学开发的适用于移动应用的MobileSAM。
13. **[Fast Segment Anything Model (FastSAM)](fast-sam.md)**: 由中国科学院自动化研究所图像与视频分析组开发的FastSAM。
14. **[YOLO-NAS](yolo-nas.md)**: YOLO [神经架构搜索](https://www.ultralytics.com/glossary/neural-architecture-search-nas) (NAS) 模型。
15. **[Realtime Detection Transformers (RT-DETR)](rtdetr.md)**: 百度PaddlePaddle的实时检测[Transformer](https://www.ultralytics.com/glossary/transformer) (RT-DETR) 模型。
16. **[YOLO-World](yolo-world.md)**: 腾讯AI Lab的实时开放词汇目标检测模型。
17. **[YOLOE](yoloe.md)**: 一种改进的开放词汇目标检测器，在保持YOLO实时性能的同时，能够检测超出其训练数据的任意类别。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 仅用几行代码运行Ultralytics YOLO模型。
</p>

## 入门：使用示例

本示例提供简单的YOLO训练和推理示例。有关这些及其他[模式](../modes/index.md)的完整文档，请参阅[Predict](../modes/predict.md)、[Train](../modes/train.md)、[Val](../modes/val.md)和[Export](../modes/export.md)文档页面。

注意以下示例适用于YOLOv8 [检测](../tasks/detect.md)模型的[目标检测](https://www.ultralytics.com/glossary/object-detection)。有关其他支持的任务，请参阅[分割](../tasks/segment.md)、[分类](../tasks/classify.md)和[姿态](../tasks/pose.md)文档。

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch)预训练的`*.pt`模型以及配置文件`*.yaml`可以传递给`YOLO()`、`SAM()`、`NAS()`和`RTDETR()`类以在Python中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 加载一个COCO预训练的YOLOv8n模型
        model = YOLO("yolov8n.pt")

        # 显示模型信息（可选）
        model.info()

        # 在COCO8示例数据集上训练模型100个周期
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用YOLOv8n模型在'bus.jpg'图像上运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        可直接使用CLI命令运行模型：

        ```bash
        # 加载一个COCO预训练的YOLOv8n模型并在COCO8示例数据集上训练100个周期
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载一个COCO预训练的YOLOv8n模型并在'bus.jpg'图像上运行推理
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 贡献新模型

有兴趣将您的模型贡献给Ultralytics吗？太好了！我们始终欢迎扩展我们的模型组合。

1. **Fork仓库**：首先Fork [Ultralytics GitHub仓库](https://github.com/ultralytics/ultralytics)。

2. **克隆您的Fork**：将您的Fork克隆到本地机器并创建一个新分支进行开发。

3. **实现您的模型**：按照我们的[贡献指南](../help/contributing.md)提供的编码标准和指南添加您的模型。

4. **全面测试**：确保对您的模型进行严格测试，包括单独测试和作为管道的一部分测试。

5. **创建拉取请求**：一旦您对模型满意，创建一个拉取请求到主仓库进行审核。

6. **代码审核与合并**：审核后，如果您的模型符合我们的标准，它将被合并到主仓库。

详细步骤请参阅我们的[贡献指南](../help/contributing.md)。

## 常见问题

### 使用Ultralytics YOLO11进行目标检测的主要优势是什么？

Ultralytics YOLO11提供增强的功能，如实时目标检测、实例分割、姿态估计和分类。其优化的架构确保在不牺牲[精度](https://www.ultralytics.com/glossary/accuracy)的同时实现高速性能，非常适合跨多种AI领域的应用。YOLO11在先前版本的基础上进行了改进，提升了性能并增加了更多功能，详见[YOLO11文档页](../models/yolo11.md)。

### 如何在自定义数据上训练YOLO模型？

使用Ultralytics的库可以轻松在自定义数据上训练YOLO模型。以下是快速示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载一个YOLO模型
        model = YOLO("yolo11n.pt")  # 或其他YOLO模型

        # 在自定义数据集上训练模型
        results = model.train(data="custom_data.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"
    

        ```bash
        yolo train model=yolo11n.pt data='custom_data.yaml' epochs=100 imgsz=640
        ```

更详细的说明请访问[训练](../modes/train.md)文档页。

### Ultralytics支持哪些YOLO版本？

Ultralytics支持从YOLOv3到YOLO11的全面YOLO（You Only Look Once）版本，以及YOLO-NAS、SAM和RT-DETR等模型。每个版本都针对检测、分割和分类等任务进行了优化。有关每个模型的详细信息，请参阅[Ultralytics支持的模型](../models/index.md)文档。

### 为什么我应该使用Ultralytics HUB进行[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)项目？

[Ultralytics HUB](../hub/index.md)提供了一个无代码的端到端平台，用于训练、部署和管理YOLO模型。它简化了复杂的工作流程，使用户能够专注于模型性能和应用。HUB还提供[云训练能力](../hub/cloud-training.md)、全面的数据集管理以及适合初学者和经验丰富的开发者的用户友好界面。

### YOLO11可以执行哪些类型的任务，与其他YOLO版本相比如何？

YOLO11是一个多功能模型，能够执行包括目标检测、实例分割、分类和姿态估计在内的任务。与早期版本相比，YOLO11通过其优化的架构和无锚设计，在速度和精度上有显著提升。更深入的比较请参阅[YOLO11文档](../models/yolo11.md)和[任务页](../tasks/index.md)以获取具体任务的详细信息。
