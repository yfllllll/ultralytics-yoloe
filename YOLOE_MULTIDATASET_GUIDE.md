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

## 分支和版本

- 内部稳定分支：`internal/yoloe-multidataset`
- 首个已验证版本：`yoloe-multidataset-v1`
- 官方代码远程：`origin`，指向 `ultralytics/ultralytics`

## 同步官方更新

进入内部代码目录：

```bash
cd /Users/yfliu/codes/yoloe/ultralytics-internal
git fetch origin
git merge origin/main
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
