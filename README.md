水果成熟度数据集，5种水果，每种三个成熟度，还有一个腐烂水果，16类别，3198张图片，已按8:2划分好，标签yolo格式
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/82a05808a9644e13a5d75b60922436d5.png)

# Classes
names:
0: apple-H
1: apple-M
2: apple-L
3: banana-H
4: banana-M
5: banana-L
6: grape-H
7: grape-M
8: grape-L
9: strawberry-H
10: strawberry-M
11: strawberry-L
12: persimmon-H
13: persimmon-M
14: persimmon-L
15: bad friut

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ba970011c34d46d58aabdbd40c66a266.png)
、
### 📊 数据集总体统计

| 项目 | 内容 |
|------|------|
| 数据集名称 | 水果成熟度与腐烂状态检测数据集 |
| 图像总数 | 3,198 张 |
| 类别数量 | 16 类 |
| 包含水果种类 | 苹果（apple）、香蕉（banana）、葡萄（grape）、草莓（strawberry）、柿子（persimmon） |
| 成熟度分级 | 每种水果分为：高成熟度（H）、中等成熟度（M）、低成熟度（L） |
| 特殊类别 | `bad fruit`（腐烂水果，泛指各类变质果实） |
| 数据划分 | 训练集 : 测试集 = 8 : 2 |
| 标注格式 | YOLO `.txt` 格式（归一化坐标） |
| 是否已划分 | ✅ 已预先划分好 `train` 和 `test` 集 |
| 适用任务 | 水果分类、成熟度识别、品质检测 |
| 可直接用于 | YOLOv5 / YOLOv8 / YOLOv11 等目标检测模型训练 |

---

### 🔢 数据集划分详情

| 数据集类型 | 图像数量 | 占比 | 图片数 |
|-----------|----------|------|--------|
| 训练集（train） | 80% | 2,558 张 |
| 测试集（test）  | 20% | 640 张  |

> ✅ 所有标签已转换为 YOLO 格式 `.txt` 文件，与图像同名存放。

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9ebaa4a808914c4095eb41776f20af23.png)


### 🍎🍓🍇 水果类别明细表（共 16 类）

| ID | 类别英文名 | 中文含义 | 水果种类 | 成熟度/状态 |
|----|------------|----------|----------|-------------|
| 0  | `apple-H`         | 苹果 - 高成熟度 | 苹果 | 成熟（High） |
| 1  | `apple-M`         | 苹果 - 中成熟度 | 苹果 | 中等（Medium） |
| 2  | `apple-L`         | 苹果 - 低成熟度 | 苹果 | 未熟（Low） |
| 3  | `banana-H`        | 香蕉 - 高成熟度 | 香蕉 | 成熟（High） |
| 4  | `banana-M`        | 香蕉 - 中成熟度 | 香蕉 | 中等（Medium） |
| 5  | `banana-L`        | 香蕉 - 低成熟度 | 香蕉 | 未熟（Low） |
| 6  | `grape-H`         | 葡萄 - 高成熟度 | 葡萄 | 成熟（High） |
| 7  | `grape-M`         | 葡萄 - 中成熟度 | 葡萄 | 中等（Medium） |
| 8  | `grape-L`         | 葡萄 - 低成熟度 | 葡萄 | 未熟（Low） |
| 9  | `strawberry-H`    | 草莓 - 高成熟度 | 草莓 | 成熟（High） |
| 10 | `strawberry-M`    | 草莓 - 中成熟度 | 草莓 | 中等（Medium） |
| 11 | `strawberry-L`    | 草莓 - 低成熟度 | 草莓 | 未熟（Low） |
| 12 | `persimmon-H`     | 柿子 - 高成熟度 | 柿子 | 成熟（High） |
| 13 | `persimmon-M`     | 柿子 - 中成熟度 | 柿子 | 中等（Medium） |
| 14 | `persimmon-L`     | 柿子 - 低成熟度 | 柿子 | 未熟（Low） |
| 15 | `bad fruit`       | 腐烂水果 | 泛指所有种类 | 变质、发霉、腐烂 |

> ⚠️ 注意：`bad fruit` 是一个通用异常类别，可能包含任意水果的腐败状态。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3b3c97bb7eff41be8b0b1021982afaef.png)

### ✅ `data.yaml` 示例配置文件

```yaml
# data.yaml
train: ./images/train
test: ./images/test

nc: 16

names: ['apple-H', 'apple-M', 'apple-L',
        'banana-H', 'banana-M', 'banana-L',
        'grape-H', 'grape-M', 'grape-L',
        'strawberry-H', 'strawberry-M', 'strawberry-L',
        'persimmon-H', 'persimmon-M', 'persimmon-L',
        'bad fruit']
```

---

基于YOLOv8的水果成熟度检测系统涉及到几个步骤，包括环境配置、数据准备、模型训练、评估和部署。以下是一个简化的代码示例，帮助您开始使用YOLOv8进行训练。仅供参考学习使用。

### 步骤 1: 环境配置

环境中安装了必要的库。YOLOv8是Ultralytics发布的模型之一，因此我们可以通过pip来安装`ultralytics`包：

```bash
pip install ultralytics
```

### 步骤 2: 数据准备

搞一下符合YOLO格式的数据集，且已经准备好了一个`data.yaml`文件（如上述表格中所示），该文件包含了路径到您的训练和测试图像以及类别信息。

### 步骤 3: 模型训练

使用YOLOv8进行训练。下面是一段Python代码示例，演示如何加载数据集并启动训练过程：

```python
from ultralytics import YOLO

# 加载预训练的YOLOv8n模型（或选择其他规模如s, m, l, x）
model = YOLO('yolov8n.yaml')  # 或者直接从权重开始 'yolov8n.pt'

# 开始训练
results = model.train(data='path/to/your/data.yaml',  # 数据集配置文件路径
                      epochs=100,  # 训练轮数
                      imgsz=640,  # 输入图像尺寸
                      batch=16,  # 批次大小
                      name='fruit_ripeness_detection',  # 训练结果保存目录名称
                      device='0'  # 使用的GPU ID，'cpu'表示使用CPU
                      )

# 训练完成后，模型会自动保存在runs/detect/fruit_ripeness_detection目录下
```

### 步骤 4: 模型评估

训练结束后，可以使用如下代码对模型进行评估：

```python
# 在验证集上评估模型性能
metrics = model.val()  # 这将在验证集上运行评估，并返回指标
print(metrics.box.map)  # 输出mAP值，用于衡量目标检测模型的整体性能
```

### 步骤 5: 部署

一旦模型训练完成并且评估满意后，您可以将模型导出为不同的格式以适应不同的部署场景，例如ONNX、TensorFlow SavedModel等：

```python
# 导出模型为ONNX格式
success = model.export(format='onnx')
print(f"Export success: {success}")
```

请根据您的实际需求调整参数和路径。
