# 项目架构与模块说明

## 1. 项目定位
本项目为一个基于 PyTorch 的目标检测系统，支持 YOLO 系列模型，具备训练、推理、评估、蒸馏等完整流程，适用于自定义数据集。

## 2. 目录结构与功能
- **mars.py**：主程序入口，根据命令行参数选择训练、评估或流水线模式，调用 `MarsEngine`。
- **engine/**：核心引擎模块，包含训练、评估调度逻辑（`engine.py`）、评估器（`evaluator.py`）、Trainer 基类（`trainer/base.py`）。
- **train/**：训练相关模块，包括损失函数（`loss.py`）、任务分配器（`tal.py`）、优化器（`opt.py`）、学习率调度器（`sched.py`）、蒸馏损失（`distilloss.py`）。
- **model/**：模型结构定义，分为 `base`（YOLO 主干、颈部、头部等）和 `distillation`（教师/学生模型）两大类。
- **dl/**：数据集与数据增强，主要为 VOC 格式数据集加载（`vocdataset.py`）和增强（`aug.py`）。
- **inference/**：推理相关，含预测器（`predictor.py`）和可视化（`painter.py`）。
- **eval/**：评估相关，含 mAP 计算（`map.py`）。
- **factory/**：工厂模式，负责 Trainer、Model、Config 的动态加载与实例化。
- **config/**：配置对象定义（`mconfig.py`），描述训练、模型、数据等参数。
- **cfgops/**：配置文件与数据集划分，含具体实验配置（如 `c1.py`）和数据集 split 文件。
- **misc/**：杂项工具，如日志（`log.py`）、图像处理、bbox 操作等。
- **marsdata/**：数据集目录，含图片、标注、划分文件等。
- **manual/**：数据预处理与转换脚本。

## 3. 主要流程说明
- **配置加载**：通过 `factory/configfactory.py` 和 `cfgops/` 下的配置脚本，动态生成 `ModelConfig` 对象，统一管理所有参数。
- **模型构建**：`factory/modelfactory.py` 根据配置动态加载 `model/` 下的模型结构（支持 YOLO 及蒸馏结构）。
- **训练流程**：`engine/engine.py` 调度，`engine/trainer/base.py` 实现标准训练循环，支持断点续训、验证集评估、模型保存。
- **损失与优化**：支持 BCE、IoU、DFL 等多种损失，优化器和调度器可配置。
- **推理与评估**：推理由 `inference/predictor.py` 实现，评估由 `engine/evaluator.py` 和 `eval/map.py` 实现，支持 mAP 等指标。
- **蒸馏支持**：`model/distillation/` 和 `train/distilloss.py` 实现教师-学生模型结构与损失。

## 4. 数据与配置
- 支持 VOC 格式数据集，数据集划分文件位于 `cfgops/mar20/splits/v5/`。
- 配置文件（如 `cfgops/c1.py`）可灵活切换不同实验设置（如 full、teacher、distillation 等）。

## 5. 扩展性与可维护性
- 工厂模式便于扩展新模型、Trainer、配置。
- 配置与代码解耦，便于实验管理和复现。

---

为了方便调试 暂时设置self.epochValidation = True为False
mcon 48

将保存设置为每5个轮次保存一次 base 159

## Mosaic 数据增强实现说明

Mosaic 数据增强是一种强大的数据增强技术，它通过将多张图片拼接在一起，增加目标尺度的多样性，提高模型的泛化能力。以下是具体实现：

### 1. 核心参数
```python
class DataAugmentationProcessor:
    def __init__(self, inputShape, jitter=0.3, rescalef=(0.4, 1.8), flipProb=0.5, huef=0.015, satf=0.7, valf=0.4):
        # ... 其他参数 ...
        self.mosaicProb = 0.3  # Mosaic 增强的概率
```

### 2. 增强流程
1. **随机选择是否使用 Mosaic**：
```python
def processEnhancement(self, image, boxList):
    if random.random() < self.mosaicProb:
        return self.processMosaic(image, boxList)
    else:
        return self.processSimple(image, boxList)
```

2. **创建拼接画布**：
```python
# 创建新的画布
canvas = np.zeros((h, w, 3), dtype=np.uint8)

# 随机选择拼接点（限制在更小的范围内以减少计算量）
cut_x = random.randint(int(w * 0.4), int(w * 0.6))
cut_y = random.randint(int(h * 0.4), int(h * 0.6))
```

3. **定义四个区域**：
```python
regions = [
    (0, 0, cut_x, cut_y),           # 左上
    (cut_x, 0, w, cut_y),           # 右上
    (0, cut_y, cut_x, h),           # 左下
    (cut_x, cut_y, w, h)            # 右下
]
```

4. **处理每个区域**：
```python
for i, (x1, y1, x2, y2) in enumerate(regions):
    # 提取区域图像和边界框
    img_region = image[y1:y2, x1:x2]
    box_region = [box for box in boxList if self._is_box_in_region(box, x1, y1, x2, y2)]
    
    # 对区域进行数据增强
    img_region, box_region = self._augment_region(img_region, box_region)
    
    # 确保区域大小匹配
    region_h, region_w = y2 - y1, x2 - x1
    img_region = cv2.resize(img_region, (region_w, region_h))
    
    # 将增强后的区域放入画布
    canvas[y1:y2, x1:x2] = img_region
```

### 3. 辅助函数
1. **判断边界框是否在区域内**：
```python
def _is_box_in_region(self, box, x1, y1, x2, y2):
    box_x1, box_y1, box_x2, box_y2 = box[:4]
    return (box_x1 >= x1 and box_x2 <= x2 and 
            box_y1 >= y1 and box_y2 <= y2)
```

2. **区域数据增强**：
```python
def _augment_region(self, image, boxes):
    # 随机缩放
    scale = random.uniform(self.rescalef[0], self.rescalef[1])
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h))
    
    # 随机水平翻转
    if random.random() < self.flipProb:
        image = cv2.flip(image, 1)
    
    # 颜色抖动
    image = self._color_jitter(image)
    
    return image, boxes
```

### 4. 性能优化
1. 限制拼接点范围（0.4-0.6）以减少计算量
2. 使用 NumPy 数组操作提高效率
3. 减少不必要的图像格式转换
4. 优化内存使用，减少临时变量

### 5. 注意事项
1. Mosaic 增强会增加 CPU 占用
2. 需要确保边界框坐标的正确性
3. 建议使用较小的增强概率（0.3）
4. 可能需要更多的训练轮次才能收敛