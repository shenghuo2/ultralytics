---
comments: true
description: 探索基于Ultralytics YOLOv8技术构建的YOLO-World模型，实现高效实时的开放词汇目标检测。以最小计算成本达成卓越性能。
keywords: YOLO-World, Ultralytics, 开放词汇检测, YOLOv8, 实时目标检测, 机器学习, 计算机视觉, 人工智能, 深度学习, 模型训练
---

# YOLO-World模型

YOLO-World模型提出了一种基于[Ultralytics](https://www.ultralytics.com/) [YOLOv8](yolov8.md)的先进实时开放词汇检测方案。这项创新技术能够根据描述性文本检测图像中的任意对象。在保持竞争力的性能同时大幅降低计算需求，使其成为众多视觉应用的通用工具。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/cfTKj96TjSE"
    title="YouTube视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看:</strong> 在自定义数据集上的YOLO-World训练流程
</p>

![YOLO-World模型架构概览](https://github.com/ultralytics/docs/releases/download/0/yolo-world-model-architecture-overview.avif)

## 概述

传统开放词汇检测模型常依赖计算密集型[Transformer](https://www.ultralytics.com/glossary/transformer)架构，且受限于预定义类别体系。YOLO-World通过融合视觉-[语言建模](https://www.ultralytics.com/glossary/language-modeling)技术，在大规模数据集上预训练，赋予YOLOv8框架开放词汇检测能力，在零样本场景中以惊人效率识别海量对象。

## 核心优势

1. **实时性能：** 利用CNN的计算速度优势，为需要即时结果的行业提供高速开放词汇检测方案

2. **高效能表现：** 在保持性能的同时大幅降低计算资源需求，相比SAM等模型仅需极小计算成本即可实现实时应用

3. **离线词汇推理：** 创新的"先提示后检测"策略，通过预计算的自定义提示词（如描述文本或类别）生成离线词汇嵌入，显著优化检测流程

4. **YOLOv8赋能：** 基于[Ultralytics YOLOv8](yolov8.md)构建，集合实时检测领域最新突破，提供无与伦比的准确率和速度

5. **基准测试领先：** 在标准测试中，速度与效率全面超越MDETR和GLIP系列等现有开放词汇检测器，单块NVIDIA V100GPU即展现卓越性能

6. **多场景应用：** 创新方法为各类视觉任务开辟新可能，速度较现有方法提升数个数量级

## 可用模型、支持任务与运行模式

下表详细列出各预训练模型参数、支持任务类型及其与[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)、[导出](../modes/export.md)等运行模式的兼容性（✅表示支持，❌表示不支持）。

!!! note

    所有YOLOv8-World权重均迁移自官方[YOLO-World](https://github.com/AILab-CVC/YOLO-World)仓库，我们对其卓越贡献表示认可。

| 模型类型       | 预训练权重                                                                                             | 支持任务                           | 推理 | 验证 | 训练 | 导出 |
| -------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------- | ---- | ---- | ---- | ---- |
| YOLOv8s-world   | [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt)   | [目标检测](../tasks/detect.md)    | ✅   | ✅   | ✅   | ❌   |
| YOLOv8s-worldv2 | [yolov8s-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt) | [目标检测](../tasks/detect.md)    | ✅   | ✅   | ✅   | ✅   |
| YOLOv8m-world   | [yolov8m-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-world.pt)   | [目标检测](../tasks/detect.md)    | ✅   | ✅   | ✅   | ❌   |
| YOLOv8m-worldv2 | [yolov8m-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-worldv2.pt) | [目标检测](../tasks/detect.md)    | ✅   | ✅   | ✅   | ✅   |
| YOLOv8l-world   | [yolov8l-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-world.pt)   | [目标检测](../tasks/detect.md)    | ✅   | ✅   | ✅   | ❌   |
| YOLOv8l-worldv2 | [yolov8l-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-worldv2.pt) | [目标检测](../tasks/detect.md)    | ✅   | ✅   | ✅   | ✅   |
| YOLOv8x-world   | [yolov8x-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-world.pt)   | [目标检测](../tasks/detect.md)    | ✅   | ✅   | ✅   | ❌   |
| YOLOv8x-worldv2 | [yolov8x-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt) | [目标检测](../tasks/detect.md)    | ✅   | ✅   | ✅   | ✅   |

## COCO数据集零样本迁移表现

!!! tip "性能指标"

    === "检测(COCO)"

        | 模型类型       | mAP  | mAP50 | mAP75 |
        | -------------- | ---- | ----- | ----- |
        | yolov8s-world   | 37.4 | 52.0  | 40.6  |
        | yolov8s-worldv2 | 37.7 | 52.2  | 41.0  |
        | yolov8m-world   | 42.0 | 57.0  | 45.6  |
        | yolov8m-worldv2 | 43.0 | 58.4  | 46.8  |
        | yolov8l-world   | 45.7 | 61.3  | 49.8  |
        | yolov8l-worldv2 | 45.8 | 61.3  | 49.8  |
        | yolov8x-world   | 47.0 | 63.0  | 51.2  |
        | yolov8x-worldv2 | 47.1 | 62.8  | 51.4  |

## 使用示例

YOLO-World模型可轻松集成到Python应用中。Ultralytics提供友好的[Python API](../usage/python.md)和[CLI命令](../usage/cli.md)简化开发流程。

### 训练使用

!!! tip

    强烈推荐使用`yolov8-worldv2`模型进行自定义训练，因其支持确定性训练并可轻松导出onnx/tensorrt等格式。

通过`train`方法进行[目标检测](https://www.ultralytics.com/glossary/object-detection)训练示例如下：

!!! example

    === "Python"

        可将[PyTorch](https://www.ultralytics.com/glossary/pytorch)预训练的`*.pt`模型及配置文件`*.yaml`传递给`YOLOWorld()`类创建实例:

        ```python
        from ultralytics import YOLOWorld

        # 加载预训练YOLOv8s-worldv2模型
        model = YOLOWorld("yolov8s-worldv2.pt")

        # 在COCO8示例数据集上训练100轮次
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用训练好的模型对'bus.jpg'进行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        ```bash
        # 加载预训练YOLOv8s-worldv2模型并在COCO8数据集上训练100轮次
        yolo train model=yolov8s-worldv2.yaml data=coco8.yaml epochs=100 imgsz=640
        ```

### 预测使用

通过`predict`方法进行目标检测示例如下：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOWorld

        # 初始化YOLO-World模型
        model = YOLOWorld("yolov8s-world.pt")  # 可选yolov8m/l-world.pt不同规格

        # 对指定图片执行推理
        results = model.predict("path/to/image.jpg")

        # 显示结果
        results[0].show()
        ```

    === "CLI"

        ```bash
        # 使用YOLO-World模型进行目标检测
        yolo predict model=yolov8s-world.pt source=path/to/image.jpg imgsz=640
        ```

这段代码展示了加载预训练模型并对图像进行预测的简易过程。

### 验证使用

模型验证流程如下：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 创建YOLO-World模型
        model = YOLO("yolov8s-world.pt")  # 可选yolov8m/l-world.pt不同规格

        # 在COCO8数据集上验证模型
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # 使用指定图像尺寸验证YOLO-World模型
        yolo val model=yolov8s-world.pt data=coco8.yaml imgsz=640
        ```

### 追踪使用

对视频/图像进行[目标追踪](https://www.ultralytics.com/glossary/object-tracking)示例如下：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 创建YOLO-World模型
        model = YOLO("yolov8s-world.pt")  # 可选yolov8m/l-world.pt不同规格

        # 对视频进行目标追踪
        results = model.track(source="path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # 使用指定图像尺寸追踪视频中的目标
        yolo track model=yolov8s-world.pt imgsz=640 source="path/to/video.mp4"
        ```

!!! note

    Ultralytics提供的YOLO-World模型预置了[COCO数据集](../datasets/detect/coco.md)80个标准类别作为离线词汇，无需额外配置即可直接识别这些常见对象。

### 设置提示词

![YOLO-World提示词类别名称概览](https://github.com/ultralytics/docs/releases/download/0/yolo-world-prompt-class-names-overview.avif)

YOLO-World框架支持通过自定义提示词动态指定类别，使用户能够在不重新训练的情况下，将模型定制至特定需求。这一特性尤其适用于将模型调整至原本不在[训练数据](https://www.ultralytics.com/glossary/training-data)范围内的新领域或特定任务。通过设置自定义提示词，用户能有效引导模型关注目标对象，从而提升检测结果的针对性和[准确度](https://www.ultralytics.com/glossary/accuracy)。

例如，若您的应用只需检测"人"和"公交车"两类对象，可直接指定这些类别：

!!! example

    === "自定义推理提示"

        ```python
        from ultralytics import YOLO

        # 初始化YOLO-World模型
        model = YOLO("yolov8s-world.pt")  # 可选yolov8m/l-world.pt

        # 定义自定义类别
        model.set_classes(["人物", "公交车"])

        # 对指定类别执行图像推理
        results = model.predict("path/to/image.jpg")

        # 显示结果
        results[0].show()
        ```

设置自定义类别后可保存模型，创建专用于特定场景的版本：

!!! example

    === "持久化自定义词汇模型"

        先加载模型并设置自定义类别后保存：

        ```python
        from ultralytics import YOLO

        # 初始化YOLO-World模型
        model = YOLO("yolov8s-world.pt")  # 可选yolov8m/l-world.pt

        # 定义自定义类别
        model.set_classes(["人物", "公交车"])

        # 保存含自定义词汇的模型
        model.save("custom_yolov8s.pt")
        ```

        保存后的模型即成为专注检测特定类别的专用模型：

        ```python
        from ultralytics import YOLO

        # 加载自定义模型
        model = YOLO("custom_yolov8s.pt")

        # 执行自定义类别检测
        results = model.predict("path/to/image.jpg")

        # 显示结果
        results[0].show()
        ```

### 保存自定义词汇模型的优势

- **高效性**：专注相关对象检测，减少计算开销
- **灵活性**：快速适配新领域检测任务，无需大量数据收集
- **简便性**：消除运行时重复指定类别的需求
- **精准度**：通过聚焦特定对象提升检测准确性

这种方法为定制最先进的[目标检测](../tasks/detect.md)模型以适应特定任务提供了强有力的手段，使得先进的人工智能技术更易于获取，并适用于更广泛的实际应用领域。

## 从零复现官方结果(实验性)

### 准备数据集

- 训练数据

| 数据集                                                           | 类型                                                        | 样本数 | 标注框数 | 标注文件                                                                                                                           |
| ----------------------------------------------------------------- | ----------------------------------------------------------- | ------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| [Objects365v1](https://opendatalab.com/OpenDataLab/Objects365_v1) | 检测                                                        | 609k   | 9621k    | [objects365_train.json](https://opendatalab.com/OpenDataLab/Objects365_v1)                                                         |
| [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)  | [基础检测](https://www.ultralytics.com/glossary/grounding)  | 621k   | 3681k    | [final_mixed_train_no_coco.json](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_mixed_train_no_coco.json) |
| [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)     | 基础检测                                                    | 149k   | 641k     | [final_flickr_separateGT_train.json](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_flickr_separateGT_train.json) |

- 验证数据

| 数据集                                                                                                 | 类型      | 标注文件                                                                                       |
| ------------------------------------------------------------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------- |
| [LVIS minival](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml) | 检测      | [minival.txt](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml) |

### 启动从零训练

!!! note

    `WorldTrainerFromScratch`是高度定制化的训练器，支持同时在检测数据集和基础检测数据集上训练yolo-world模型。详情请查看[ultralytics.model.yolo.world.train_world.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py)。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOWorld
        from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

        data = dict(
            train=dict(
                yolo_data=["Objects365.yaml"],
                grounding_data=[
                    dict(
                        img_path="../datasets/flickr30k/images",
                        json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
                    ),
                    dict(
                        img_path="../datasets/GQA/images",
                        json_file="../datasets/GQA/final_mixed_train_no_coco.json",
                    ),
                ],
            ),
            val=dict(yolo_data=["lvis.yaml"]),
        )
        model = YOLOWorld("yolov8s-worldv2.yaml")
        model.train(data=data, batch=128, epochs=100, trainer=WorldTrainerFromScratch)
        ```

## 引用与致谢

我们衷心感谢[腾讯AILab计算机视觉中心](https://www.tencent.com/)在实时开放词汇目标检测领域的开创性工作：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{cheng2024yolow,
        title={YOLO-World: Real-Time Open-Vocabulary Object Detection},
        author={Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying},
        journal={arXiv preprint arXiv:2401.17270},
        year={2024}
        }
        ```

更多研究细节请参阅[arXiv论文](https://arxiv.org/pdf/2401.17270v2)，项目源码及资源请访问[GitHub仓库](https://github.com/AILab-CVC/YOLO-World)。

## 常见问题

### YOLO-World模型是什么？其工作原理是什么？

YOLO-World是基于[Ultralytics YOLOv8](yolov8.md)框架的先进实时目标检测模型，专长于开放词汇检测任务。通过视觉语言建模和大规模预训练，该模型能根据文本描述识别图像中的任意对象，在显著降低计算需求的同时保持卓越性能。

### YOLO-World如何通过自定义提示进行推理？

YOLO-World采用“先提示后检测”策略，利用离线词汇库提升效率。自定义提示（如图片说明或特定对象类别）会被预先编码并存储为离线词汇库的[嵌入向量](https://www.ultralytics.com/glossary/embeddings)。这种方法无需重新训练即可简化检测流程。您可以通过以下方式动态设置这些提示，以适应特定检测任务：

```python
from ultralytics import YOLOWorld

# 初始化YOLO-World模型
model = YOLOWorld("yolov8s-world.pt")

# 定义自定义类别
model.set_classes(["人物", "公交车"])

# 对图像执行预测
results = model.predict("path/to/image.jpg")

# 显示结果
results[0].show()
```

### 为什么选择YOLO-World而非传统开放词汇检测模型？

YOLO-World相比传统开放词汇检测模型具备以下优势：

- **实时性能：** 利用CNN的计算速度，实现快速高效的检测。
- **高效低资源消耗：** 在保持高性能的同时，显著降低计算和资源需求。
- **可定制提示：** 支持动态设置提示，用户无需重新训练即可指定自定义检测类别。
- **基准测试优异：** 在标准测试中，速度和效率均优于MDETR、GLIP等其他开放词汇检测器。

### 如何在自己的数据集上训练YOLO-World模型？

通过Python API或CLI命令即可轻松训练。以下是使用Python启动训练的示例：

```python
from ultralytics import YOLOWorld

# 加载预训练的YOLOv8s-worldv2模型
model = YOLOWorld("yolov8s-worldv2.pt")

# 在COCO8数据集上训练100轮次
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

或使用CLI：

```bash
yolo train model=yolov8s-worldv2.yaml data=coco8.yaml epochs=100 imgsz=640
```

### 有哪些预训练YOLO-World模型及其支持的任务？

Ultralytics提供多种预训练模型，支持不同任务和模式：

| 模型类型       | 预训练权重                                                                                               | 支持任务                                | 推理  | 验证  | 训练  | 导出  |
| -------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------- | ----- | ----- | ----- | ----- |
| YOLOv8s-world   | [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt)     | [目标检测](../tasks/detect.md)         | ✅    | ✅    | ✅    | ❌    |
| YOLOv8s-worldv2 | [yolov8s-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt) | [目标检测](../tasks/detect.md)         | ✅    | ✅    | ✅    | ✅    |
| YOLOv8m-world   | [yolov8m-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-world.pt)     | [目标检测](../tasks/detect.md)         | ✅    | ✅    | ✅    | ❌    |
| YOLOv8m-worldv2 | [yolov8m-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-worldv2.pt) | [目标检测](../tasks/detect.md)         | ✅    | ✅    | ✅    | ✅    |
| YOLOv8l-world   | [yolov8l-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-world.pt)     | [目标检测](../tasks/detect.md)         | ✅    | ✅    | ✅    | ❌    |
| YOLOv8l-worldv2 | [yolov8l-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-worldv2.pt) | [目标检测](../tasks/detect.md)         | ✅    | ✅    | ✅    | ✅    |
| YOLOv8x-world   | [yolov8x-world.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-world.pt)     | [目标检测](../tasks/detect.md)         | ✅    | ✅    | ✅    | ❌    |
| YOLOv8x-worldv2 | [yolov8x-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt) | [目标检测](../tasks/detect.md)         | ✅    | ✅    | ✅    | ✅    |

### 如何从头复现YOLO-World的官方结果？

需准备数据集并使用定制训练器启动训练：

```python
from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

data = {
    "train": {
        "yolo_data": ["Objects365.yaml"],
        "grounding_data": [
            {
                "img_path": "../datasets/flickr30k/images",
                "json_file": "../datasets/flickr30k/final_flickr_separateGT_train.json",
            },
            {
                "img_path": "../datasets/GQA/images",
                "json_file": "../datasets/GQA/final_mixed_train_no_coco.json",
            },
        ],
    },
    "val": {"yolo_data": ["lvis.yaml"]},
}

model = YOLOWorld("yolov8s-worldv2.yaml")
model.train(data=data, batch=128, epochs=100, trainer=WorldTrainerFromScratch)
```
