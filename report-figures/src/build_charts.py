#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成项目报告所需的专业图表（matplotlib，出版级风格）。
柱状/雷达/热力图均使用真实实验数据；训练收敛曲线为示意图，端点与基线为实测值。
"""
import os
from math import pi
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体（文泉驿正黑）----
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_manager.fontManager.addfont(FONT_PATH)
_zh = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams.update({
    "font.sans-serif": [_zh, "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
})

OUT = os.path.join(os.path.dirname(__file__), "..")

# 出版级配色（seaborn muted 风格）
BLUE = "#4C72B0"; GREEN = "#55A868"; RED = "#C44E52"; PURPLE = "#8172B3"
GOLD = "#CCB974"; CYAN = "#64B5CD"; GRAY = "#AEB4BF"
PAL6 = [BLUE, GREEN, CYAN, RED, GOLD, PURPLE]


def despine(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa0a6")
    ax.tick_params(colors="#3c4043", length=4)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#e9ecef", lw=1.0)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def _labels(ax, bars, fmt="{:.3f}", dy=0.004, fs=10):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs, color="#222")


# 1) 人体检测：CrowdHuman vs 现场微调（纵轴拉伸以突出差异）
def fig_person_finetune_metrics():
    metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]
    base = [0.833, 0.866, 0.931, 0.855]
    new = [0.905, 0.872, 0.954, 0.795]
    x = np.arange(len(metrics)); w = 0.40
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar(x - w/2, base, w, label="CrowdHuman 原始权重", color=GRAY, edgecolor="white", zorder=3)
    b2 = ax.bar(x + w/2, new, w, label="现场微调 v1", color=BLUE, edgecolor="white", zorder=3)
    _labels(ax, b1, dy=0.003); _labels(ax, b2, dy=0.003)
    ax.set_ylim(0.74, 1.0); ax.set_ylabel("指标值")
    ax.set_title("人体检测微调前后验证指标对比（现场验证集）")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    ax.text(0.5, -0.24, "注：纵轴自 0.74 起，以突出指标差异；具体数值见柱顶标注。",
            transform=ax.transAxes, fontsize=9, color="#999", ha="center")
    despine(ax)
    save(fig, "fig_person_finetune_metrics.png")


# 2) 人体检测：独立现场照片逐图检出数（真实数据）
def fig_person_detect_counts():
    imgs = ["field_001", "field_002", "field_003", "field_004", "field_005", "field_006"]
    base = [85, 76, 10, 7, 15, 4]
    new = [131, 100, 13, 11, 20, 10]
    x = np.arange(len(imgs)); w = 0.40
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    b1 = ax.bar(x - w/2, base, w, label=f"CrowdHuman 原始权重（合计 {sum(base)}）", color=GRAY, edgecolor="white", zorder=3)
    b2 = ax.bar(x + w/2, new, w, label=f"现场微调 v1（合计 {sum(new)}）", color=BLUE, edgecolor="white", zorder=3)
    _labels(ax, b1, fmt="{:.0f}", dy=1); _labels(ax, b2, fmt="{:.0f}", dy=1)
    ax.set_ylabel("人体检出数"); ax.set_title("独立现场照片人体检出数对比（conf=0.25, imgsz=960）")
    ax.set_xticks(x); ax.set_xticklabels(imgs, rotation=12)
    ax.legend(loc="upper right", frameon=False)
    despine(ax)
    save(fig, "fig_person_detect_counts.png")


# 3) 人体检测微调 mAP 收敛示意图（逐轮示意，端点/基线实测）
def fig_person_finetune_curve():
    e = np.arange(1, 51)
    conv = lambda s, f, t: f - (f - s) * np.exp(-(e - 1) / t)
    map50 = conv(0.882, 0.954, 9.0)
    map5095 = conv(0.700, 0.795, 9.0)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(e, map50, color=BLUE, lw=2.4, label="微调 mAP50（→0.954）")
    ax.plot(e, map5095, color=RED, lw=2.4, label="微调 mAP50-95（→0.795）")
    ax.axhline(0.931, color=BLUE, ls="--", lw=1.4, alpha=0.7, label="CrowdHuman 基线 mAP50=0.931")
    ax.axhline(0.855, color=RED, ls="--", lw=1.4, alpha=0.7, label="CrowdHuman 基线 mAP50-95=0.855")
    ax.scatter([50, 50], [0.954, 0.795], color=[BLUE, RED], zorder=5, s=30)
    ax.annotate("0.954", (50, 0.954), textcoords="offset points", xytext=(-28, 6), color=BLUE, fontsize=10)
    ax.annotate("0.795", (50, 0.795), textcoords="offset points", xytext=(-28, -14), color=RED, fontsize=10)
    ax.set_xlabel("训练轮次 epoch"); ax.set_ylabel("mAP")
    ax.set_title("人体检测现场微调收敛示意图（CrowdHuman 预训练 → 现场微调 50 轮）")
    ax.set_ylim(0.6, 1.0); ax.set_xlim(1, 52)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.text(2, 0.625, "示意：逐轮曲线为示意，端点(0.954/0.795)与基线(0.931/0.855)为实测值",
            fontsize=9, color="#888", style="italic")
    despine(ax, grid_axis="both")
    save(fig, "fig_person_finetune_curve.png")


# 4) 六类行为 YOLO 训练指标对比（纵轴拉伸）
def fig_sixcls_metrics():
    metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]
    m1 = [0.739, 0.685, 0.709, 0.466]
    m2 = [0.783, 0.736, 0.782, 0.516]
    x = np.arange(len(metrics)); w = 0.40
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar(x - w/2, m1, w, label="Student Behaviour v6（e20/640）", color=GRAY, edgecolor="white", zorder=3)
    b2 = ax.bar(x + w/2, m2, w, label="merged-classroom-6cls-v2（e50/960）", color=GREEN, edgecolor="white", zorder=3)
    _labels(ax, b1, dy=0.004); _labels(ax, b2, dy=0.004)
    ax.set_ylim(0.40, 0.85); ax.set_ylabel("指标值")
    ax.set_title("六类行为 YOLO 自训练模型验证指标对比")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=9)
    ax.text(0.5, -0.24, "注：纵轴自 0.40 起，以突出指标差异；具体数值见柱顶标注。",
            transform=ax.transAxes, fontsize=9, color="#999", ha="center")
    despine(ax)
    save(fig, "fig_sixcls_metrics.png")


# 5) 大模型行为分类：总体加权得分率（仅加权指标）
def fig_vlm_weighted():
    methods = ["融合", "GPT", "千问"]
    rate = [93.10, 89.29, 74.05]
    colors = [BLUE, GREEN, GOLD]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(methods, rate, width=0.52, color=colors, edgecolor="white", zorder=3)
    for b, r in zip(bars, rate):
        ax.text(b.get_x() + b.get_width()/2, r + 0.8, f"{r:.2f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#222")
    ax.set_ylim(0, 105); ax.set_ylabel("加权得分率")
    ax.set_title("大模型课堂行为分类总体加权得分率（n=84）")
    despine(ax)
    save(fig, "fig_vlm_weighted.png")


# 6) 大模型行为分类：三方法分类别加权得分率雷达图（真实数据）
def fig_vlm_radar():
    cats = ["举手", "看书", "写字", "使用手机", "低头", "睡觉"]
    data = {
        "融合": [140.00, 113.75, 111.43, 63.64, 70.00, 0.00],
        "GPT":  [140.00, 113.75, 113.14, 33.64, 116.67, 0.00],
        "千问": [70.00, 52.50, 71.43, 99.09, 70.00, 46.67],
    }
    colors = {"融合": BLUE, "GPT": GREEN, "千问": GOLD}
    N = len(cats)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    fig = plt.figure(figsize=(7.0, 6.4))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylim(0, 140)
    ax.set_yticks([50, 100, 140])
    ax.set_yticklabels(["50%", "100%", "140%"], color="#888", fontsize=9)
    ax.set_rlabel_position(18)
    for name, vals in data.items():
        v = vals + vals[:1]
        ax.plot(angles, v, color=colors[name], lw=2.2, label=name)
        ax.fill(angles, v, color=colors[name], alpha=0.10)
    ax.set_title("三种方法分类别加权得分率对比（雷达图）", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10), frameon=False)
    save(fig, "fig_vlm_radar.png")


# 7) 大模型行为分类：分类别加权得分率热力图（真实数据，ML 论文风）
def fig_vlm_heatmap():
    classes = ["举手 (n=2)", "看书 (n=16)", "写字 (n=35)", "使用手机 (n=22)", "低头 (n=6)", "睡觉 (n=3)"]
    methods = ["融合", "GPT", "千问"]
    M = np.array([
        [140.00, 140.00, 70.00],
        [113.75, 113.75, 52.50],
        [111.43, 113.14, 71.43],
        [63.64, 33.64, 99.09],
        [70.00, 116.67, 70.00],
        [0.00, 0.00, 46.67],
    ])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=140, aspect="auto")
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    color="white" if v > 72 else "#222", fontsize=11, fontweight="bold")
    ax.set_title("分类别加权得分率热力图（行=行为类别，列=方法）", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("加权得分率 (%)", fontsize=10)
    ax.set_xticks(np.arange(-.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(classes), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    save(fig, "fig_vlm_heatmap.png")


# 7) 大模型行为分类：融合方法混淆矩阵（真实逐目标数据 CSV）
def fig_vlm_confusion():
    csv_path = os.path.join(OUT, "data_vlm_predictions_comparison.csv")
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    order = ["Hand-raise", "Reading", "Writing", "Useing-Phone", "Head-down", "Sleeping"]
    zh = ["举手", "看书", "写字", "使用手机", "低头", "睡觉"]
    idx = {l: i for i, l in enumerate(order)}
    M = np.zeros((6, 6), int)
    for r in rows:
        if r["provider"] != "fusion":
            continue
        M[idx[r["truth_label"]], idx[r["pred_label"]]] += 1

    row_sum = M.sum(axis=1, keepdims=True)
    rown = np.divide(M, np.maximum(row_sum, 1))
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(M[i, j]), ha="center", va="center",
                    color="white" if rown[i, j] > 0.5 else "#333", fontsize=12,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(6)); ax.set_xticklabels(zh, rotation=20, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels([f"{z}\n(n={int(row_sum[i,0])})" for i, z in enumerate(zh)])
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("融合方法行为分类混淆矩阵（n=84，单元格为计数）")
    # 高亮 Reading/Writing「学习」块（展示口径合并为正确）
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.5, 0.5), 2, 2, fill=False, edgecolor="#16a34a", lw=2.6))
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("行归一化比例（按真实类别）", fontsize=10)
    save(fig, "fig_vlm_confusion.png")


if __name__ == "__main__":
    fig_person_finetune_metrics()
    fig_person_detect_counts()
    fig_person_finetune_curve()
    fig_sixcls_metrics()
    fig_vlm_weighted()
    fig_vlm_radar()
    fig_vlm_confusion()
    print("ALL DONE")
