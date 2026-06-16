# merged-classroom-6cls-v2 最佳模型下载与校验记录

## 文件

- 本地模型：`D:\Documents\网络综合设计\models\merged_classroom_6cls_v2_img960_e50_best.pt`
- 本地大小：`22587555` bytes
- SHA256：`a1faa22abfc17b74e6f589a5830e41820a30b447600db7e5661f56deea57f95e`
- 远端来源：`/data/classroom-yolov8/output/training/merged_classroom_6cls_v2_img960_e50/weights/best.pt`
- 训练记录：`D:\Documents\网络综合设计\output\training_records\merged_classroom_6cls_v2_img960_e50\results.csv`

## 训练末轮指标

- epoch：50
- precision：0.78266
- recall：0.73623
- mAP50：0.78237
- mAP50-95：0.51627

## 本地加载烟测

- 测试图片：`D:\Documents\网络综合设计\datasets\field-photos-2026-06-12\images\field_001.jpg`
- 检测总数：42
- 类别计数：{'Writing': 28, 'Useing-Phone': 5, 'Reading': 9}
- 可视化输出目录：`D:\Documents\网络综合设计\output\model_download_validation\merged_classroom_6cls_v2_img960_e50\field_001_predict`

结论：模型文件下载完整，哈希与远端一致；本地 Ultralytics 可正常加载该模型并完成推理。
