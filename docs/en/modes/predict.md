---
comments: true
description: 利用Ultralytics YOLO11对各种数据源进行实时高速推理。了解预测模式、关键特性和实际应用。
keywords: Ultralytics, YOLO11, 模型预测, 推理, 预测模式, 实时推理, 计算机视觉, 机器学习, 流式处理, 高性能
---

# 使用Ultralytics YOLO进行模型预测

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO生态系统和集成">

## 简介

在机器学习和计算机视觉领域,从视觉数据中提取意义的过程被称为"推理"或"预测"。Ultralytics YOLO11提供了一个强大的功能,称为**预测模式**,专为在各种数据源上进行高性能实时推理而设计。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看:</strong> 如何从Ultralytics YOLO模型中提取输出用于自定义项目。
</p>

## 现实世界应用

|                   制造业                   |                        体育                        |                   安全                    |
| :-----------------------------------------------: | :--------------------------------------------------: | :-----------------------------------------: |
| ![车辆零部件检测][car spare parts] | ![足球运动员检测][football player detect] | ![人员跌倒检测][human fall detect] |
|           车辆零部件检测           |              足球运动员检测               |            人员跌倒检测            |

## 为什么使用Ultralytics YOLO进行推理?

以下是为什么应该考虑使用YOLO11的预测模式满足各种推理需求的原因:

- **多功能性:** 能够对图像、视频甚至实时流进行推理。
- **性能:** 专为实时高速处理而设计,同时不牺牲准确性。
- **易用性:** 直观的Python和CLI界面,便于快速部署和测试。
- **高度可定制:** 各种设置和参数可根据您的具体要求调整模型的推理行为。

### 预测模式的主要特性

YOLO11的预测模式设计得强大而多功能,具有以下特点:

- **多数据源兼容性:** 无论您的数据是单张图像、图像集合、视频文件还是实时视频流,预测模式都能应对。
- **流式模式:** 使用流式功能生成内存高效的`Results`对象生成器。通过在预测器的调用方法中设置`stream=True`来启用此功能。
- **批处理:** 能够在单个批次中处理多张图像或视频帧,进一步加快推理时间。
- **易于集成:** 由于其灵活的API,可以轻松集成到现有数据管道和其他软件组件中。

Ultralytics YOLO模型返回Python的`Results`对象列表,或者当在推理过程中传递`stream=True`给模型时,返回内存高效的Python `Results`对象生成器:

!!! example "预测"

    === "使用`stream=False`返回列表"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 预训练的YOLO11n模型

        # 对图像列表进行批量推理
        results = model(["image1.jpg", "image2.jpg"])  # 返回Results对象列表

        # 处理结果列表
        for result in results:
            boxes = result.boxes  # 用于边界框输出的Boxes对象
            masks = result.masks  # 用于分割掩码输出的Masks对象
            keypoints = result.keypoints  # 用于姿态输出的Keypoints对象
            probs = result.probs  # 用于分类输出的Probs对象
            obb = result.obb  # 用于OBB输出的Oriented boxes对象
            result.show()  # 显示在屏幕上
            result.save(filename="result.jpg")  # 保存到磁盘
        ```

    === "使用`stream=True`返回生成器"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo11n.pt")  # 预训练的YOLO11n模型

        # 对图像列表进行批量推理
        results = model(["image1.jpg", "image2.jpg"], stream=True)  # 返回Results对象生成器

        # 处理结果生成器
        for result in results:
            boxes = result.boxes  # 用于边界框输出的Boxes对象
            masks = result.masks  # 用于分割掩码输出的Masks对象
            keypoints = result.keypoints  # 用于姿态输出的Keypoints对象
            probs = result.probs  # 用于分类输出的Probs对象
            obb = result.obb  # 用于OBB输出的Oriented boxes对象
            result.show()  # 显示在屏幕上
            result.save(filename="result.jpg")  # 保存到磁盘
        ```

## 推理源

YOLO11可以处理不同类型的输入源进行推理,如下表所示。这些源包括静态图像、视频流和各种数据格式。表格还指出了每个源是否可以在流模式下使用参数`stream=True` ✅。流模式对于处理视频或实时流很有益,因为它创建了一个结果生成器,而不是将所有帧加载到内存中。

!!! tip

    对于处理长视频或大型数据集,使用`stream=True`可以有效管理内存。当`stream=False`时,所有帧或数据点的结果都存储在内存中,这可能会快速累积并导致大型输入的内存不足错误。相比之下,`stream=True`使用生成器,只在内存中保留当前帧或数据点的结果,显著减少内存消耗并防止内存不足问题。

| 源                                                    | 示例                                       | 类型            | 注释                                                                                        |
| ----------------------------------------------------- | ------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------- |
| 图像                                                  | `'image.jpg'`                              | `str` 或 `Path` | 单个图像文件。                                                                              |
| URL                                                   | `'https://ultralytics.com/images/bus.jpg'` | `str`           | 图像的URL。                                                                                 |
| 截图                                                  | `'screen'`                                 | `str`           | 捕获屏幕截图。                                                                              |
| PIL                                                   | `Image.open('image.jpg')`                  | `PIL.Image`     | HWC格式,RGB通道。                                                                           |
| [OpenCV](https://www.ultralytics.com/glossary/opencv) | `cv2.imread('image.jpg')`                  | `np.ndarray`    | HWC格式,BGR通道 `uint8 (0-255)`。                                                           |
| numpy                                                 | `np.zeros((640,1280,3))`                   | `np.ndarray`    | HWC格式,BGR通道 `uint8 (0-255)`。                                                           |
| torch                                                 | `torch.zeros(16,3,320,640)`                | `torch.Tensor`  | BCHW格式,RGB通道 `float32 (0.0-1.0)`。                                                      |
| CSV                                                   | `'sources.csv'`                            | `str` 或 `Path` | 包含图像、视频或目录路径的CSV文件。                                                         |
| 视频 ✅                                               | `'video.mp4'`                              | `str` 或 `Path` | MP4、AVI等格式的视频文件。                                                                  |
| 目录 ✅                                               | `'path/'`                                  | `str` 或 `Path` | 包含图像或视频的目录路径。                                                                  |
| glob ✅                                               | `'path/*.jpg'`                             | `str`           | 匹配多个文件的glob模式。使用`*`字符作为通配符。                                             |
| YouTube ✅                                            | `'https://youtu.be/LNwODJXcvt4'`           | `str`           | YouTube视频的URL。                                                                          |
| 流 ✅                                                 | `'rtsp://example.com/media.mp4'`           | `str`           | 流媒体协议的URL,如RTSP、RTMP、TCP或IP地址。                                                 |
| 多流 ✅                                               | `'list.streams'`                           | `str` 或 `Path` | `*.streams`文本文件,每行一个流URL,即8个流将以批量大小8运行。                                |
| 网络摄像头 ✅                                         | `0`                                        | `int`           | 连接的相机设备的索引,用于运行推理。                                                         |

以下是每种源类型的代码示例:

!!! example "预测源"

    === "图像"

        对图像文件运行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 定义图像文件的路径
        source = "path/to/image.jpg"

        # 对源运行推理
        results = model(source)  # Results对象列表
        ```

    === "截图"

        对当前屏幕内容作为截图运行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 将当前截图定义为源
        source = "screen"

        # 对源运行推理
        results = model(source)  # Results对象列表
        ```

    === "URL"

        对远程托管的图像或视频通过URL运行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 定义远程图像或视频URL
        source = "https://ultralytics.com/images/bus.jpg"

        # 对源运行推理
        results = model(source)  # Results对象列表
        ```

    === "PIL"

        对使用Python图像库(PIL)打开的图像运行推理。
        ```python
        from PIL import Image

        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 使用PIL打开图像
        source = Image.open("path/to/image.jpg")

        # 对源运行推理
        results = model(source)  # Results对象列表
        ```

    === "OpenCV"

        对使用OpenCV读取的图像运行推理。
        ```python
        import cv2

        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 使用OpenCV读取图像
        source = cv2.imread("path/to/image.jpg")

        # 对源运行推理
        results = model(source)  # Results对象列表
        ```

    === "numpy"

        对表示为numpy数组的图像运行推理。
        ```python
        import numpy as np

        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 创建一个随机numpy数组,HWC形状(640, 640, 3),值范围[0, 255],类型uint8
        source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype="uint8")

        # 对源运行推理
        results = model(source)  # Results对象列表
        ```

    === "torch"

        对表示为[PyTorch](https://www.ultralytics.com/glossary/pytorch)张量的图像运行推理。
        ```python
        import torch

        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 创建一个随机torch张量,BCHW形状(1, 3, 640, 640),值范围[0, 1],类型float32
        source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        # 对源运行推理
        results = model(source)  # Results对象列表
        ```

    === "CSV"

        对CSV文件中列出的图像、URL、视频和目录集合运行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 定义包含图像、URL、视频和目录的CSV文件的路径
        source = "path/to/file.csv"

        # 对源运行推理
        results = model(source)  # Results对象列表
        ```

    === "视频"

        对视频文件运行推理。通过使用`stream=True`,您可以创建Results对象的生成器以减少内存使用。
        ```python
        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 定义视频文件的路径
        source = "path/to/video.mp4"

        # 对源运行推理
        results = model(source, stream=True)  # Results对象的生成器
        ```

    === "目录"

        对目录中的所有图像和视频运行推理。要同时捕获子目录中的图像和视频,请使用glob模式,即`path/to/dir/**/*`。
        ```python
        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 定义包含用于推理的图像和视频的目录路径
        source = "path/to/dir"

        # 对源运行推理
        results = model(source, stream=True)  # Results对象的生成器
        ```

    === "glob"

        对匹配带有`*`字符的glob表达式的所有图像和视频运行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 定义目录中所有JPG文件的glob搜索
        source = "path/to/dir/*.jpg"

        # 或定义包括子目录在内的所有JPG文件的递归glob搜索
        source = "path/to/dir/**/*.jpg"

        # 对源运行推理
        results = model(source, stream=True)  # Results对象的生成器
        ```

    === "YouTube"

        对YouTube视频运行推理。通过使用`stream=True`,您可以为长视频创建Results对象的生成器以减少内存使用。
        ```python
        from ultralytics import YOLO

        # 加载预训练的YOLO11n模型
        model = YOLO("yolo11n.pt")

        # 将源定义为YouTube视频URL
        source = "https://youtu.be/LNwODJXcvt4"

        # 对源运行推理
        results = model(source, stream=True)  # Results对象的生成器
        ```

=== "流"

    使用流模式在使用RTSP、RTMP、TCP或IP地址协议的实时视频流上运行推理。如果提供单个流,模型将以[批量大小](https://www.ultralytics.com/glossary/batch-size)为1进行推理。对于多个流,可以使用 `.streams` 文本文件执行批量推理,其中批量大小由提供的流数量决定(例如,8个流的批量大小为8)。

    ```python
    from ultralytics import YOLO

    # 加载预训练的YOLO11n模型
    model = YOLO("yolo11n.pt")

    # 单个流,批量大小为1的推理
    source = "rtsp://example.com/media.mp4"  # RTSP、RTMP、TCP或IP流地址

    # 在源上运行推理
    results = model(source, stream=True)  # Results对象的生成器
    ```

    对于单个流使用,批量大小默认设置为1,允许对视频流进行高效的实时处理。

=== "多流"

    要同时处理多个视频流,请使用包含流源的 `.streams` 文本文件。模型将运行批量推理,批量大小等于流的数量。这种设置可以同时高效处理多个馈送。

    ```python
    from ultralytics import YOLO

    # 加载预训练的YOLO11n模型
    model = YOLO("yolo11n.pt")

    # 多个流进行批量推理(例如,8个流的批量大小为8)
    source = "path/to/list.streams"  # *.streams 文本文件,每行一个流地址

    # 在源上运行推理
    results = model(source, stream=True)  # Results对象的生成器
    ```

    `.streams` 文本文件示例:

    ```txt
    rtsp://example.com/media1.mp4
    rtsp://example.com/media2.mp4
    rtmp://example2.com/live
    tcp://192.168.1.100:554
    ...
    ```

    文件中的每一行代表一个流源,允许您同时监控和对多个视频流进行推理。

=== "网络摄像头"

    您可以通过将特定摄像头的索引传递给 `source` 来对连接的摄像头设备运行推理。

    ```python
    from ultralytics import YOLO

    # 加载预训练的YOLO11n模型
    model = YOLO("yolo11n.pt")

    # 在源上运行推理
    results = model(source=0, stream=True)  # Results对象的生成器
    ```

## 推理参数

`model.predict()` 接受多个可在推理时传递的参数,以覆盖默认值:

!!! example

    ```python
    from ultralytics import YOLO

    # 加载预训练的YOLO11n模型
    model = YOLO("yolo11n.pt")

    # 在'bus.jpg'上运行推理,带参数
    model.predict("https://ultralytics.com/images/bus.jpg", save=True, imgsz=320, conf=0.5)
    ```

推理参数:

{% include "macros/predict-args.md" %}

可视化参数:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table() }}

## 图像和视频格式

YOLO11支持各种图像和视频格式,如[ultralytics/data/utils.py](https://github.com/ultralytics/ultralytics/blob/
## 图像和视频格式

YOLO11支持各种图像和视频格式，如[ultralytics/data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py)中所指定。请参见下表中的有效后缀和示例预测命令。

### 图像

下表包含有效的Ultralytics图像格式。

!!! 注意

    HEIC图像仅支持推理，不支持训练。

| 图像后缀 | 示例预测命令 | 参考 |
| -------- | ------------ | ---- |
| `.bmp`   | `yolo predict source=image.bmp` | [Microsoft BMP文件格式](https://en.wikipedia.org/wiki/BMP_file_format) |
| `.dng`   | `yolo predict source=image.dng` | [Adobe DNG](https://en.wikipedia.org/wiki/Digital_Negative) |
| `.jpeg`  | `yolo predict source=image.jpeg` | [JPEG](https://en.wikipedia.org/wiki/JPEG) |
| `.jpg`   | `yolo predict source=image.jpg` | [JPEG](https://en.wikipedia.org/wiki/JPEG) |
| `.mpo`   | `yolo predict source=image.mpo` | [多图像对象](https://fileinfo.com/extension/mpo) |
| `.png`   | `yolo predict source=image.png` | [便携式网络图形](https://en.wikipedia.org/wiki/PNG) |
| `.tif`   | `yolo predict source=image.tif` | [标签图像文件格式](https://en.wikipedia.org/wiki/TIFF) |
| `.tiff`  | `yolo predict source=image.tiff` | [标签图像文件格式](https://en.wikipedia.org/wiki/TIFF) |
| `.webp`  | `yolo predict source=image.webp` | [WebP](https://en.wikipedia.org/wiki/WebP) |
| `.pfm`   | `yolo predict source=image.pfm` | [便携式浮点图](https://en.wikipedia.org/wiki/Netpbm#File_formats) |
| `.HEIC`  | `yolo predict source=image.HEIC` | [高效图像格式](https://en.wikipedia.org/wiki/HEIF) |

### 视频

下表包含有效的Ultralytics视频格式。

| 视频后缀 | 示例预测命令 | 参考 |
| -------- | ------------ | ---- |
| `.asf`   | `yolo predict source=video.asf` | [高级系统格式](https://en.wikipedia.org/wiki/Advanced_Systems_Format) |
| `.avi`   | `yolo predict source=video.avi` | [音频视频交错](https://en.wikipedia.org/wiki/Audio_Video_Interleave) |
| `.gif`   | `yolo predict source=video.gif` | [图形交换格式](https://en.wikipedia.org/wiki/GIF) |
| `.m4v`   | `yolo predict source=video.m4v` | [MPEG-4第14部分](https://en.wikipedia.org/wiki/M4V) |
| `.mkv`   | `yolo predict source=video.mkv` | [Matroska](https://en.wikipedia.org/wiki/Matroska) |
| `.mov`   | `yolo predict source=video.mov` | [QuickTime文件格式](https://en.wikipedia.org/wiki/QuickTime_File_Format) |
| `.mp4`   | `yolo predict source=video.mp4` | [MPEG-4第14部分 - 维基百科](https://en.wikipedia.org/wiki/MPEG-4_Part_14) |
| `.mpeg`  | `yolo predict source=video.mpeg` | [MPEG-1第2部分](https://en.wikipedia.org/wiki/MPEG-1) |
| `.mpg`   | `yolo predict source=video.mpg` | [MPEG-1第2部分](https://en.wikipedia.org/wiki/MPEG-1) |
| `.ts`    | `yolo predict source=video.ts` | [MPEG传输流](https://en.wikipedia.org/wiki/MPEG_transport_stream) |
| `.wmv`   | `yolo predict source=video.wmv` | [Windows媒体视频](https://en.wikipedia.org/wiki/Windows_Media_Video) |
| `.webm`  | `yolo predict source=video.webm` | [WebM项目](https://en.wikipedia.org/wiki/WebM) |

## 处理结果

所有Ultralytics `predict()`调用都将返回一个`Results`对象列表：

!!! 示例 "结果"

    ```python
    from ultralytics import YOLO

    # 加载预训练的YOLO11n模型
    model = YOLO("yolo11n.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/bus.jpg")
    results = model(
        [
            "https://ultralytics.com/images/bus.jpg",
            "https://ultralytics.com/images/zidane.jpg",
        ]
    )  # 批量推理
    ```

`Results`对象具有以下属性：

| 属性 | 类型 | 描述 |
| ---- | ---- | ---- |
| `orig_img` | `numpy.ndarray` | 原始图像作为numpy数组。 |
| `orig_shape` | `tuple` | 原始图像形状，格式为(高度, 宽度)。 |
| `boxes` | `Boxes, optional` | 包含检测边界框的Boxes对象。 |
| `masks` | `Masks, optional` | 包含检测掩码的Masks对象。 |
| `probs` | `Probs, optional` | 包含分类任务每个类别概率的Probs对象。 |
| `keypoints` | `Keypoints, optional` | 包含每个对象检测到的关键点的Keypoints对象。 |
| `obb` | `OBB, optional` | 包含定向边界框的OBB对象。 |
| `speed` | `dict` | 包含预处理、推理和后处理速度（每张图像的毫秒数）的字典。 |
| `names` | `dict` | 将类别索引映射到类别名称的字典。 |
| `path` | `str` | 图像文件的路径。 |
| `save_dir` | `str, optional` | 保存结果的目录。 |

`Results`对象具有以下方法：

| 方法 | 返回类型 | 描述 |
| ---- | -------- | ---- |
| `update()` | `None` | 使用新的检测数据（边界框、掩码、概率、obb、关键点）更新Results对象。 |
| `cpu()` | `Results` | 返回所有张量移至CPU内存的Results对象副本。 |
| `numpy()` | `Results` | 返回所有张量转换为numpy数组的Results对象副本。 |
| `cuda()` | `Results` | 返回所有张量移至GPU内存的Results对象副本。 |
| `to()` | `Results` | 返回张量移至指定设备和数据类型的Results对象副本。 |
| `new()` | `Results` | 创建具有相同图像、路径、名称和速度属性的新Results对象。 |
| `plot()` | `np.ndarray` | 在输入RGB图像上绘制检测结果并返回带注释的图像。 |
| `show()` | `None` | 显示带有注释推理结果的图像。 |
| `save()` | `str` | 保存带注释的推理结果图像到文件并返回文件名。 |
| `verbose()` | `str` | 返回每个任务的日志字符串，详细说明检测和分类结果。 |
| `save_txt()` | `str` | 将检测结果保存到文本文件并返回保存文件的路径。 |
| `save_crop()` | `None` | 将裁剪的检测图像保存到指定目录。 |
| `summary()` | `List[Dict]` | 将推理结果转换为摘要字典，可选择进行归一化。 |
| `to_df()` | `DataFrame` | 将检测结果转换为Pandas DataFrame。 |
| `to_csv()` | `str` | 将检测结果转换为CSV格式。 |
| `to_xml()` | `str` | 将检测结果转换为XML格式。 |
| `to_html()` | `str` | 将检测结果转换为HTML格式。 |
| `to_json()` | `str` | 将检测结果转换为JSON格式。 |
| `to_sql()` | `None` | 将检测结果转换为SQL兼容格式并保存到数据库。 |

有关更多详细信息，请参阅[`Results`类文档](../reference/engine/results.md)。

### 边界框

`Boxes`对象可用于索引、操作和将边界框转换为不同格式。

!!! 示例 "边界框"

    ```python
    from ultralytics import YOLO

    # 加载预训练的YOLO11n模型
    model = YOLO("yolo11n.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/bus.jpg")  # 结果列表

    # 查看结果
    for r in results:
        print(r.boxes)  # 打印包含检测边界框的Boxes对象
    ```

以下是`Boxes`类方法和属性的表格，包括它们的名称、类型和描述：

| 名称 | 类型 | 描述 |
| ---- | ---- | ---- |
| `cpu()` | 方法 | 将对象移至CPU内存。 |
| `numpy()` | 方法 | 将对象转换为numpy数组。 |
| `cuda()` | 方法 | 将对象移至CUDA内存。 |
| `to()` | 方法 | 将对象移至指定设备。 |
| `xyxy` | 属性 (`torch.Tensor`) | 以xyxy格式返回边界框。 |
| `conf` | 属性 (`torch.Tensor`) | 返回边界框的置信度值。 |
| `cls` | 属性 (`torch.Tensor`) | 返回边界框的类别值。 |
| `id` | 属性 (`torch.Tensor`) | 返回边界框的跟踪ID（如果可用）。 |
| `xywh` | 属性 (`torch.Tensor`) | 以xywh格式返回边界框。 |
| `xyxyn` | 属性 (`torch.Tensor`) | 以xyxy格式返回边界框，并按原始图像大小归一化。 |
| `xywhn` | 属性 (`torch.Tensor`) | 以xywh格式返回边界框，并按原始图像大小归一化。 |

有关更多详细信息，请参阅[`Boxes`类文档](../reference/engine/results.md#ultralytics.engine.results.Boxes)。

### 掩码

`Masks`对象可用于索引、操作和将掩码转换为分段。

!!! 示例 "掩码"

    ```python
    from ultralytics import YOLO

    # 加载预训练的YOLO11n-seg分割模型
    model = YOLO("yolo11n-seg.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/bus.jpg")  # 结果列表

    # 查看结果
    for r in results:
        print(r.masks)  # 打印包含检测到的实例掩码的Masks对象
    ```

以下是`Masks`类方法和属性的表格，包括它们的名称、类型和描述：

| 名称 | 类型 | 描述 |
| ---- | ---- | ---- |
| `cpu()` | 方法 | 返回CPU内存上的掩码张量。 |
| `numpy()` | 方法 | 将掩码张量作为numpy数组返回。 |
| `cuda()` | 方法 | 返回GPU内存上的掩码张量。 |
| `to()` | 方法 | 返回具有指定设备和数据类型的掩码张量。 |
| `xyn` | 属性 (`torch.Tensor`) | 表示为张量的归一化分段列表。 |
| `xy` | 属性 (`torch.Tensor`) | 表示为张量的像素坐标分段列表。 |

有关更多详细信息，请参阅[`Masks`类文档](../reference/engine/results.md#ultralytics.engine.results.Masks)。
### 关键点

`Keypoints` 对象可用于索引、操作和标准化坐标。

!!! 示例 "关键点"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLO11n-pose 姿势模型
    model = YOLO("yolo11n-pose.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/bus.jpg")  # 结果列表

    # 查看结果
    for r in results:
        print(r.keypoints)  # 打印包含检测到的关键点的 Keypoints 对象
    ```

以下是 `Keypoints` 类方法和属性的表格，包括它们的名称、类型和描述：

| 名称      | 类型                      | 描述                                           |
| --------- | ------------------------- | ---------------------------------------------- |
| `cpu()`   | 方法                      | 返回 CPU 内存上的关键点张量。                  |
| `numpy()` | 方法                      | 将关键点张量作为 numpy 数组返回。              |
| `cuda()`  | 方法                      | 返回 GPU 内存上的关键点张量。                  |
| `to()`    | 方法                      | 返回指定设备和数据类型的关键点张量。           |
| `xyn`     | 属性 (`torch.Tensor`)     | 以张量表示的标准化关键点列表。                 |
| `xy`      | 属性 (`torch.Tensor`)     | 以张量表示的像素坐标关键点列表。               |
| `conf`    | 属性 (`torch.Tensor`)     | 返回关键点的置信度值（如果可用），否则为None。 |

更多详情请参阅 [`Keypoints` 类文档](../reference/engine/results.md#ultralytics.engine.results.Keypoints)。

### 概率

`Probs` 对象可用于索引、获取分类的 `top1` 和 `top5` 索引和分数。

!!! 示例 "概率"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLO11n-cls 分类模型
    model = YOLO("yolo11n-cls.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/bus.jpg")  # 结果列表

    # 查看结果
    for r in results:
        print(r.probs)  # 打印包含检测到的类别概率的 Probs 对象
    ```

以下是 `Probs` 类的方法和属性汇总表：

| 名称       | 类型                      | 描述                                                   |
| ---------- | ------------------------- | ------------------------------------------------------ |
| `cpu()`    | 方法                      | 返回 CPU 内存上的 probs 张量副本。                     |
| `numpy()`  | 方法                      | 将 probs 张量作为 numpy 数组返回。                     |
| `cuda()`   | 方法                      | 返回 GPU 内存上的 probs 张量副本。                     |
| `to()`     | 方法                      | 返回指定设备和数据类型的 probs 张量副本。              |
| `top1`     | 属性 (`int`)              | 排名第一的类别索引。                                   |
| `top5`     | 属性 (`list[int]`)        | 排名前五的类别索引。                                   |
| `top1conf` | 属性 (`torch.Tensor`)     | 排名第一的类别置信度。                                 |
| `top5conf` | 属性 (`torch.Tensor`)     | 排名前五的类别置信度。                                 |

更多详细信息请参阅 [`Probs` 类文档](../reference/engine/results.md#ultralytics.engine.results.Probs)。

### OBB

`OBB` 对象可用于索引、操作和将定向边界框转换为不同格式。

!!! 示例 "OBB"

    ```python
    from ultralytics import YOLO

    # 加载预训练的 YOLO11n 模型
    model = YOLO("yolo11n-obb.pt")

    # 对图像进行推理
    results = model("https://ultralytics.com/images/boats.jpg")  # 结果列表

    # 查看结果
    for r in results:
        print(r.obb)  # 打印包含定向检测边界框的 OBB 对象
    ```

以下是 `OBB` 类方法和属性的表格，包括它们的名称、类型和描述：

| 名称        | 类型                      | 描述                                       |
| ----------- | ------------------------- | ------------------------------------------ |
| `cpu()`     | 方法                      | 将对象移至 CPU 内存。                      |
| `numpy()`   | 方法                      | 将对象转换为 numpy 数组。                  |
| `cuda()`    | 方法                      | 将对象移至 CUDA 内存。                     |
| `to()`      | 方法                      | 将对象移至指定设备。                       |
| `conf`      | 属性 (`torch.Tensor`)     | 返回边界框的置信度值。                     |
| `cls`       | 属性 (`torch.Tensor`)     | 返回边界框的类别值。                       |
| `id`        | 属性 (`torch.Tensor`)     | 返回边界框的跟踪 ID（如果可用）。          |
| `xyxy`      | 属性 (`torch.Tensor`)     | 以 xyxy 格式返回水平边界框。               |
| `xywhr`     | 属性 (`torch.Tensor`)     | 以 xywhr 格式返回旋转边界框。              |
| `xyxyxyxy`  | 属性 (`torch.Tensor`)     | 以 xyxyxyxy 格式返回旋转边界框。           |
| `xyxyxyxyn` | 属性 (`torch.Tensor`)     | 以 xyxyxyxy 格式返回归一化的旋转边界框。   |

更多详细信息请参阅 [`OBB` 类文档](../reference/engine/results.md#ultralytics.engine.results.OBB)。

## 绘制结果

`Results` 对象中的 `plot()` 方法通过在原始图像上叠加检测到的对象（如边界框、掩码、关键点和概率）来实现预测结果的可视化。此方法返回一个 NumPy 数组形式的带注释图像，便于显示或保存。

!!! 示例 "绘图"

    ```python
    from PIL import Image

    from ultralytics import YOLO

    # 加载预训练的 YOLO11n 模型
    model = YOLO("yolo11n.pt")

    # 对 'bus.jpg' 进行推理
    results = model(["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/zidane.jpg"])  # 结果列表

    # 可视化结果
    for i, r in enumerate(results):
        # 绘制结果图像
        im_bgr = r.plot()  # BGR 顺序的 numpy 数组
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB 顺序的 PIL 图像

        # 在支持的环境中显示结果
        r.show()

        # 将结果保存到磁盘
        r.save(filename=f"results{i}.jpg")
    ```

### `plot()` 方法参数

`plot()` 方法支持多种参数来自定义输出：

| 参数         | 类型            | 描述                                                           | 默认值        |
| ------------ | --------------- | -------------------------------------------------------------- | ------------- |
| `conf`       | `bool`          | 包含检测置信度分数。                                           | `True`        |
| `line_width` | `float`         | 边界框线宽。如果为 `None`，则根据图像大小缩放。                | `None`        |
| `font_size`  | `float`         | 文本字体大小。如果为 `None`，则根据图像大小缩放。              | `None`        |
| `font`       | `str`           | 文本注释的字体名称。                                           | `'Arial.ttf'` |
| `pil`        | `bool`          | 将图像作为 PIL Image 对象返回。                                | `False`       |
| `img`        | `numpy.ndarray` | 用于绘图的替代图像。如果为 `None`，则使用原始图像。            | `None`        |
| `im_gpu`     | `torch.Tensor`  | GPU 加速图像，用于更快的掩码绘制。形状：(1, 3, 640, 640)。     | `None`        |
| `kpt_radius` | `int`           | 绘制关键点的半径。                                             | `5`           |
| `kpt_line`   | `bool`          | 用线连接关键点。                                               | `True`        |
| `labels`     | `bool`          | 在注释中包含类别标签。                                         | `True`        |
| `boxes`      | `bool`          | 在图像上叠加边界框。                                           | `True`        |
| `masks`      | `bool`          | 在图像上叠加掩码。                                             | `True`        |
| `probs`      | `bool`          | 包含分类概率。                                                 | `True`        |
| `show`       | `bool`          | 使用默认图像查看器直接显示带注释的图像。                       | `False`       |
| `save`       | `bool`          | 如果为 `True`，将带注释的图像保存到 `filename` 指定的文件。    | `False`       |
| `filename`   | `str`           | 如果 `save` 为 `True`，保存带注释图像的文件路径和名称。        | `None`        |
| `color_mode` | `str`           | 指定颜色模式，例如 'instance' 或 'class'。                     | `'class'`     |

## 线程安全推理

当您在不同线程上并行运行多个 YOLO 模型时，确保推理过程的线程安全至关重要。线程安全推理保证每个线程的预测是隔离的，不会相互干扰，避免竞态条件并确保输出的一致性和可靠性。

在多线程应用程序中使用 YOLO 模型时，重要的是为每个线程实例化单独的模型对象或使用线程本地存储以防止冲突：

!!! 示例 "线程安全推理"

    在每个线程内实例化单个模型以进行线程安全推理：
    ```python
    from threading import Thread

    from ultralytics import YOLO


    def thread_safe_predict(model, image_path):
        """使用本地实例化的 YOLO 模型对图像执行线程安全预测。"""
        model = YOLO(model)
        results = model.predict(image_path)
        # 处理结果


    # 启动线程，每个线程都有自己的模型实例
    Thread(target=thread_safe_predict, args=("yolo11n.pt", "image1.jpg")).start()
    Thread(target=thread_safe_predict, args=("yolo11n.pt", "image2.jpg")).start()
    ```
有关使用 YOLO 模型进行线程安全推理的深入探讨和分步说明,请参阅我们的 [YOLO 线程安全推理指南](../guides/yolo-thread-safe-inference.md)。该指南将为您提供所有必要的信息,以避免常见陷阱并确保多线程推理顺利运行。

## 流式源 `for` 循环

这是一个使用 OpenCV (`cv2`) 和 YOLO 对视频帧进行推理的 Python 脚本。此脚本假设您已经安装了必要的包 (`opencv-python` 和 `ultralytics`)。

!!! 示例 "流式 for 循环"

    ```python
    import cv2

    from ultralytics import YOLO

    # 加载 YOLO 模型
    model = YOLO("yolo11n.pt")

    # 打开视频文件
    video_path = "path/to/your/video/file.mp4"
    cap = cv2.VideoCapture(video_path)

    # 遍历视频帧
    while cap.isOpened():
        # 从视频中读取一帧
        success, frame = cap.read()

        if success:
            # 对帧运行 YOLO 推理
            results = model(frame)

            # 在帧上可视化结果
            annotated_frame = results[0].plot()

            # 显示带注释的帧
            cv2.imshow("YOLO Inference", annotated_frame)

            # 如果按下 'q' 键,则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 如果到达视频末尾,则退出循环
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()
    ```

这个脚本将对视频的每一帧进行预测,可视化结果,并在窗口中显示它们。通过按 'q' 键可以退出循环。

[汽车备件]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1
[足球运动员检测]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442
[人体跌倒检测]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43

## 常见问题

### 什么是 Ultralytics YOLO 及其用于实时推理的预测模式?

Ultralytics YOLO 是一个最先进的模型,用于实时[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和分类。它的**预测模式**允许用户对各种数据源(如图像、视频和实时流)进行高速推理。它专为性能和多功能性而设计,还提供批处理和流式处理模式。有关其功能的更多详细信息,请查看 [Ultralytics YOLO 预测模式](#key-features-of-predict-mode)。

### 如何使用 Ultralytics YOLO 在不同数据源上运行推理?

Ultralytics YOLO 可以处理各种数据源,包括单个图像、视频、目录、URL 和流。您可以在 `model.predict()` 调用中指定数据源。例如,使用 `'image.jpg'` 表示本地图像,或使用 `'https://ultralytics.com/images/bus.jpg'` 表示 URL。查看文档中各种[推理源](#inference-sources)的详细示例。

### 如何优化 YOLO 推理速度和内存使用?

要优化推理速度并有效管理内存,您可以在预测器的调用方法中设置 `stream=True` 来使用流式模式。流式模式生成内存高效的 `Results` 对象生成器,而不是将所有帧加载到内存中。对于处理长视频或大型数据集,流式模式特别有用。了解更多关于[流式模式](#key-features-of-predict-mode)的信息。

### Ultralytics YOLO 支持哪些推理参数?

YOLO 中的 `model.predict()` 方法支持各种参数,如 `conf`、`iou`、`imgsz`、`device` 等。这些参数允许您自定义推理过程,设置置信度阈值、图像大小和用于计算的设备等参数。这些参数的详细描述可以在[推理参数](#inference-arguments)部分找到。

### 如何可视化和保存 YOLO 预测的结果?

使用 YOLO 运行推理后,`Results` 对象包含用于显示和保存带注释图像的方法。您可以使用 `result.show()` 和 `result.save(filename="result.jpg")` 等方法来可视化和保存结果。有关这些方法的完整列表,请参阅[处理结果](#working-with-results)部分。
