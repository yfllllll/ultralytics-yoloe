# YOLOE 多数据集内部版本说明

本仓库是在 Ultralytics 官方代码基础上维护的内部版本，主要解决 YOLOE 同时训练多个类别体系不同的
YOLO 检测数据集时，文本特征无法正确组成 batch 的问题。

## 修改内容

- 每个子数据集使用自己的类别词表和正负类文本池，不在不同数据集之间共享负类。
- 所有训练数据集采用统一的文本槽位数量，取训练数据集中最大的类别数，上限为 80。
- 类别较少的数据集用零补齐剩余文本槽位，并通过 `text_mask` 排除 padding 槽位的分类 BCE loss。
- 删除会跨图片错误切分文本特征的 flatten + reshape 处理，按图片分别构造固定尺寸文本特征。
- 支持单类训练数据集。
- 支持配置多个验证数据集；每个验证集使用自己的类别词表独立验证，指标分别记录。

## 已验证场景

真实数据测试使用以下组合：

- `coco8.yaml`：80 类
- `construction-ppe.yaml`：11 类
- `african-wildlife.yaml`：4 类
- `batch=12`、`imgsz=32`、训练 1 个 epoch

训练完成 183 个混合 batch，三个验证集均完成训练中验证和最终 `best.pt` 验证，没有再出现
`txt_feats.reshape(...)` 错误。

## 多数据集配置

```yaml
train:
  yolo_data:
    - coco8.yaml
    - construction-ppe.yaml
    - african-wildlife.yaml

val:
  yolo_data:
    - coco8.yaml
    - construction-ppe.yaml
    - african-wildlife.yaml
```

## 训练示例

```python
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train import YOLOETrainerFromScratch

model = YOLOE("yoloe-11n.yaml")
model.train(
    data="path/to/multi-dataset.yaml",
    trainer=YOLOETrainerFromScratch,
    epochs=100,
    imgsz=640,
    batch=12,
)
```

## 冻结 Chinese-CLIP 训练

中文实验使用 `OFA-Sys/chinese-clip-vit-base-patch16`。编码器固定为评估模式，所有参数均关闭梯度，类别
特征在训练开始前按子数据集分别缓存，不会进入 YOLOE 优化器。

```python
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train import YOLOETrainerFromScratch

model = YOLOE("yoloe-11n.yaml")
model.set_text_model("chineseclip:OFA-Sys/chinese-clip-vit-base-patch16")
model.train(
    data="path/to/yoloe-multidataset.yaml",
    trainer=YOLOETrainerFromScratch,
    epochs=100,
    imgsz=640,
    batch=12,
)
```

对照实验保持数据、初始化权重、随机种子和训练参数不变，只切换文本编码器：

```python
# MobileCLIP baseline
model.set_text_model("mobileclip:blt")

# Chinese-CLIP experiment
model.set_text_model("chineseclip:OFA-Sys/chinese-clip-vit-base-patch16")
```

Chinese-CLIP 实验应在各子数据集 YAML 中使用真实中文类别名，例如 `电瓶车`、`摩托车`、`自行车`。
拼音 `dianpingche` 不等同于中文提示词，不适合用于评估中文编码能力。

首次使用 Chinese-CLIP 需要安装 `transformers` 并下载 Hugging Face 权重。文本缓存文件名包含编码器名称，
不会与 MobileCLIP 缓存混用。

## 自动生成多数据集配置

假设数据目录中包含任意层级的子目录，每个实际数据集目录都有以下结构：

```text
dataset-name/
├── images/                 # 图片直接放在这里，不要求 train/val 子目录
├── labels/                 # 与图片同名的 YOLO txt 标签
└── data.yaml 或 dataset.yaml
```

原始 YAML 只需要类别映射，不需要 `train`、`val` 或 `path`：

```yaml
names:
  0: 电瓶车
  1: 摩托车
  2: 自行车
```

递归扫描并生成配置：

```bash
python -m ultralytics.data.generate_yoloe_multidataset \
    /path/to/datasets-root \
    --output /path/to/yoloe-multidataset.yaml \
    --val-ratio 0.2 \
    --seed 0
```

工具对每个子数据集独立随机拆分，默认 80% 训练、20% 验证。同一随机种子会产生相同拆分。图片和标签
不会被移动或复制；工具使用绝对图片路径生成 `train.txt` 和 `val.txt`。

严格要求所有类别名包含中文字符：

```bash
python -m ultralytics.data.generate_yoloe_multidataset \
    /path/to/datasets-root \
    --output /path/to/yoloe-multidataset-cn.yaml \
    --require-chinese-names
```

假设输出为 `/path/to/yoloe-multidataset.yaml`，工具还会创建：

```text
/path/to/yoloe-multidataset_datasets/
└── 原始相对目录/数据集名称/
    ├── data.yaml           # 完整的子数据集配置
    ├── train.txt           # 训练图片绝对路径
    └── val.txt             # 验证图片绝对路径
```

总配置引用自动生成的子数据集配置：

```yaml
train:
  yolo_data:
    - /path/to/yoloe-multidataset_datasets/dataset-a/data.yaml
    - /path/to/yoloe-multidataset_datasets/group/dataset-b/data.yaml
val:
  yolo_data:
    - /path/to/yoloe-multidataset_datasets/dataset-a/data.yaml
    - /path/to/yoloe-multidataset_datasets/group/dataset-b/data.yaml
```

每个子数据集至少需要 2 张图片，才能保证训练集和验证集都不为空。

## 分支和版本

- 内部稳定分支：`internal/yoloe-multidataset`
- 中文文本编码实验分支：`codex/chinese-clip-support`
- 首个已验证版本：`yoloe-multidataset-v1`
- 内部代码远程：`origin`，指向 `yfllllll/ultralytics-yoloe`
- 官方代码远程：`upstream`，指向 `ultralytics/ultralytics`

## 同步官方更新

进入内部代码目录：

```bash
cd /Users/yfliu/codes/yoloe/ultralytics-internal
git fetch upstream
git merge upstream/main
```

这会把 Ultralytics 官方更新合并到内部稳定分支，同时保留多数据集修改。如果发生冲突，Git 会停止
合并并列出冲突文件，不会直接覆盖内部修改。

## 查看或恢复稳定版本

查看首个稳定版本：

```bash
git switch --detach yoloe-multidataset-v1
```

回到内部稳定分支：

```bash
git switch internal/yoloe-multidataset
```

## 目录用途

- `/Users/yfliu/codes/yoloe/ultralytics`：官方 `main` 基线。
- `/Users/yfliu/codes/yoloe/ultralytics-internal`：包含内部修改，用于训练和继续开发。
