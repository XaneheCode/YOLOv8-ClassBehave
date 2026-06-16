# 现场照片标注说明

日期：2026-06-12

## 文件位置

现场照片已复制到：

```text
datasets/field-photos-2026-06-12/images
```

文件名：

```text
field_001.jpg
field_002.jpg
field_003.jpg
field_004.jpg
field_005.jpg
field_006.jpg
```

## 初步检测结果

使用当前展示基线模型：

```text
models/classroom_behaviour_6cls.pt
```

在 6 张现场图上只检出 12 个行为框，漏检严重。因此不建议直接使用基线模型输出作为训练标签。

基线模型输出：

```text
output/field_photos_baseline_autolabel
```

使用通用 `yolov8n.pt` 的 person 类进行人体预标注，共检出 78 个人体框：

| 图片 | person 预标框 |
| --- | ---: |
| `field_001.jpg` | 18 |
| `field_002.jpg` | 21 |
| `field_003.jpg` | 11 |
| `field_004.jpg` | 11 |
| `field_005.jpg` | 12 |
| `field_006.jpg` | 5 |

person 预标注文件：

```text
datasets/field-photos-2026-06-12/person_prelabels
datasets/field-photos-2026-06-12/person-prelabel.yaml
```

person 预标注可视化：

```text
output/field_photos_person_autolabel/contact_sheet.jpg
```

## 是否能直接用于 6 类训练

暂时不建议直接用于 6 类训练。

原因：

- `field_001.jpg` 和 `field_002.jpg` 是大教室远景，学生很多，通用 person 检测仍然漏掉大量小目标。
- 若训练时图片里存在大量未标注的目标，YOLO 会把这些目标当作背景，可能损害模型。
- person 预标注只有“人”的框，没有行为类别。
- 行为类别需要人工按 6 类规则确认，不能由 person 框自动转换。

## 推荐使用方式

### 推荐 1：作为真实验证集

这 6 张照片非常适合作为现场验证集，用来比较模型泛化能力。

建议保留：

```text
datasets/field-photos-2026-06-12/images
```

每次训练新模型后，在这 6 张图上离线推理，观察是否比旧模型检出更多真实学生、是否减少误检。

### 推荐 2：人工复核后做训练集

如果要加入训练，需要先人工复核：

- 对每一个目标框确认是否准确。
- 给每个框改成 6 类之一。
- 删除误检框。
- 补上漏检学生框。
- 对无法归入 6 类的站立、走动、背对镜头、只露局部人物，可以不标或单独保留为后续扩展类。

## AI 辅助标注版本

已基于通用 person 检测框和人工视觉判断，生成一版保守的 6 类 YOLOv8 标注。

标准 YOLOv8 数据集目录：

```text
datasets/field-photos-2026-06-12-yolo6cls
```

结构：

```text
datasets/field-photos-2026-06-12-yolo6cls/
  images/
  labels/
  data.yaml
```

标注框数量：

| 类别 | 数量 |
| --- | ---: |
| `Hand-raise` | 0 |
| `Reading` | 10 |
| `Writing` | 17 |
| `Useing-Phone` | 12 |
| `Head-down` | 3 |
| `Sleeping` | 0 |
| 合计 | 42 |

复核图：

```text
output/field_photos_ai_assisted_labels/contact_sheet.jpg
```

注意：

- 这版标注偏保守，只标清晰可判断的目标。
- 大教室远景中仍有许多小目标未标，暂不建议单独大量训练。
- 更适合作为“现场增强样本”的一小部分，混入更大的训练集中使用。
- 使用前建议人工看一遍复核图，删除明显错标，再加入训练。

6 类顺序：

```text
0 Hand-raise
1 Reading
2 Writing
3 Useing-Phone
4 Head-down
5 Sleeping
```

## 标注规则

| 视觉情况 | 建议类别 |
| --- | --- |
| 明确举手 | `Hand-raise` |
| 看书、看资料、视线明显落在书本或纸质材料上 | `Reading` |
| 手持笔并在纸上书写、做题 | `Writing` |
| 明确看手机或手持手机操作 | `Useing-Phone` |
| 低头但看不清是在写字、看书或手机 | `Head-down` |
| 趴桌、伏案、明显睡觉姿势 | `Sleeping` |

注意：

- 不要把普通低头写字全部标成 `Sleeping`。
- 不要把所有低头都标成 `Useing-Phone`，必须能看到手机或明显手持手机。
- 只露出半个头或被显示器严重遮挡的人，若无法判断行为，可以不标。

## 关于 REBALANCE 数据集

`REBALANCE-instance-new-student-classroom-activity-5.v1i.yolov8` 可用于增强异常类，但不适合作为完整 6 类主训练集。

类别：

```text
phone, sleep, study
```

统计：

| split | 图片 | 标注框 |
| --- | ---: | ---: |
| train | 2277 | 4747 |
| valid | 223 | 437 |
| test | 112 | 234 |

建议映射：

| 原类别 | 建议用途 |
| --- | --- |
| `phone` | 映射到 `Useing-Phone` |
| `sleep` | 映射到 `Sleeping` |
| `study` | 不建议直接映射到 `Reading` 或 `Writing`，除非重新人工细分 |

训练建议：

- 可以用 `phone` 和 `sleep` 样本补强异常类。
- `study` 类不要直接混入 6 类训练，否则会模糊 `Reading`、`Writing`、`Head-down` 的边界。
