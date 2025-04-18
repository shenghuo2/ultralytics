| 参数              | 类型    | 默认值           | 范围          | 描述                                                                                                                                                                   |
| ----------------- | ------- | --------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hsv_h`           | `float` | `0.015`         | `0.0 - 1.0`| 通过色轮的一部分调整图像的色调，引入色彩变化。帮助模型在不同光照条件下实现泛化。 |
| `hsv_s`           | `float` | `0.7`           | `0.0 - 1.0`| 通过一定比例改变图像的饱和度，影响颜色的强度。适用于模拟不同的环境条件。 |
| `hsv_v`           | `float` | `0.4`           | `0.0 - 1.0`| 通过调整图像的亮度值，帮助模型在各种光照条件下都能表现良好。 |
| `degrees`         | `float` | `0.0`           | `-180 - +180`| 在指定的角度范围内随机旋转图像，提高模型识别各种方向上的物体的能力。 |
| `translate`       | `float` | `0.1`           | `0.0 - 1.0`| 水平和垂直方向上按图像尺寸的一部分平移图像，有助于学习检测部分可见的物体。 |
| `scale`           | `float` | `0.5`           | `>=0.0`| 通过增益因子缩放图像，模拟不同距离下的物体与相机的关系。 |
| `shear`           | `float` | `0.0`           | `-180 - +180`| 通过指定角度剪切图像，模拟从不同角度观看物体的效果。 |
| `perspective`     | `float` | `0.0`           | `0.0 - 0.001`| 对图像应用随机透视变换，增强模型理解三维空间中物体的能力。 |
| `flipud`          | `float` | `0.0`           | `0.0 - 1.0`| 以指定的概率将图像上下翻转，在不影响对象特征的情况下增加数据的多样性。 |
| `fliplr`          | `float` | `0.5`           | `0.0 - 1.0`| 以指定的概率将图像从左到右翻转，有助于学习对称物体和增加数据集多样性。 |
| `bgr`             | `float` | `0.0`           | `0.0 - 1.0`| 以指定的概率将图像通道从RGB翻转为BGR，有助于增强对不正确通道排序的鲁棒性。 |
| `mosaic`          | `float` | `1.0`           | `0.0 - 1.0`| 将四张训练图像合并为一张，模拟不同的场景组合和物体交互。对于复杂场景理解非常有效。 |
| `mixup`           | `float` | `0.0`           | `0.0 - 1.0`| 混合两张图像及其标签，创建一个合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。 |
| `copy_paste`      | `float` | `0.0`           | `0.0 - 1.0`| 跨图像复制和粘贴对象，有助于增加对象实例数量并学习对象遮挡。需要分割标签。 |
| `copy_paste_mode` | `str`   | `'flip'`我将把这段内容翻译成简体中文，保持原始格式和含义：

        | -             | 在选项中选择复制粘贴增强方法 (`"flip"`, `"mixup"`).                                                                                      |
| `auto_augment`    | `str`   | `'randaugment'`| -             | 自动应用预定义的增强策略 (`randaugment`, `autoaugment`, `augmix`), 通过多样化视觉特征来优化分类任务。|
| `erasing`         | `float` | `0.4`           | `0.0 - 0.9`| 在分类训练过程中随机擦除图像的一部分，鼓励模型关注不太明显的特征来进行识别。 |
| `crop_fraction`   | `float` | `1.0`           | `0.1 - 1.0`| 将分类图像裁剪至其原始尺寸的一部分，以突出中心特征并适应物体比例，减少背景干扰。 |
