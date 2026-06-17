#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成项目报告所需的专业图表（matplotlib）。
所有柱状/对比图均使用真实实验数据；训练收敛曲线为示意图，端点与基线为实测值。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体注册（文泉驿正黑）----
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_manager.fontManager.addfont(FONT_PATH)
_zh = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.sans-serif"] = [_zh, "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.dpi"] = 160
plt.rcParams["font.size"] = 12

OUT = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(OUT, exist_ok=True)

# 统一配色
C_BASE = "#9aa7b4"      # 基线/原始
C_NEW = "#2f6fed"       # 我方/微调
C_FUSE = "#2f6fed"
C_GPT = "#16a34a"
C_QWEN = "#f59e0b"
PALETTE6 = ["#2f6fed", "#16a34a", "#22c3a6", "#ef4444", "#f59e0b", "#8b5cf6"]


def _bar_labels(ax, bars, fmt="{:.3f}", dy=0.005):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=10)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# 1) 人体检测：CrowdHuman 原始权重 vs 现场微调 v1（真实数据）
def fig_person_finetune_metrics():
    metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]
    base = [0.833, 0.866, 0.931, 0.855]
    new = [0.905, 0.872, 0.954, 0.795]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    b1 = ax.bar(x - w/2, base, w, label="CrowdHuman 原始权重", color=C_BASE, edgecolor="white")
    b2 = ax.bar(x + w/2, new, w, label="现场微调 v1", color=C_NEW, edgecolor="white")
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_ylim(0, 1.05); ax.set_ylabel("指标值")
    ax.set_title("人体检测微调前后验证指标对比（现场验证集）")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_person_finetune_metrics.png")


# 2) 人体检测：独立现场照片逐图检出数（真实数据）
def fig_person_detect_counts():
    imgs = ["field_001", "field_002", "field_003", "field_004", "field_005", "field_006"]
    base = [85, 76, 10, 7, 15, 4]
    new = [131, 100, 13, 11, 20, 10]
    x = np.arange(len(imgs)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    b1 = ax.bar(x - w/2, base, w, label=f"CrowdHuman 原始权重（合计 {sum(base)}）", color=C_BASE, edgecolor="white")
    b2 = ax.bar(x + w/2, new, w, label=f"现场微调 v1（合计 {sum(new)}）", color=C_NEW, edgecolor="white")
    _bar_labels(ax, b1, fmt="{:.0f}", dy=1); _bar_labels(ax, b2, fmt="{:.0f}", dy=1)
    ax.set_ylabel("人体检出数"); ax.set_title("独立现场照片人体检出数对比（conf=0.25, imgsz=960）")
    ax.set_xticks(x); ax.set_xticklabels(imgs, rotation=15)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_person_detect_counts.png")


# 3) 人体检测微调 mAP 收敛示意图（逐轮为示意，端点与基线为实测）
def fig_person_finetune_curve():
    e = np.arange(1, 51)
    def conv(start, final, tau):
        return final - (final - start) * np.exp(-(e - 1) / tau)
    map50 = conv(0.882, 0.954, 9.0)
    map5095 = conv(0.700, 0.795, 9.0)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(e, map50, color=C_NEW, lw=2.2, label="微调 mAP50（→0.954）")
    ax.plot(e, map5095, color="#ef4444", lw=2.2, label="微调 mAP50-95（→0.795）")
    ax.axhline(0.931, color=C_NEW, ls="--", lw=1.4, alpha=0.8, label="CrowdHuman 基线 mAP50=0.931")
    ax.axhline(0.855, color="#ef4444", ls="--", lw=1.4, alpha=0.8, label="CrowdHuman 基线 mAP50-95=0.855")
    ax.scatter([50, 50], [0.954, 0.795], color=["#2f6fed", "#ef4444"], zorder=5)
    ax.annotate("0.954", (50, 0.954), textcoords="offset points", xytext=(-26, 6), color=C_NEW, fontsize=10)
    ax.annotate("0.795", (50, 0.795), textcoords="offset points", xytext=(-26, -14), color="#ef4444", fontsize=10)
    ax.set_xlabel("训练轮次 epoch"); ax.set_ylabel("mAP")
    ax.set_title("人体检测现场微调收敛示意图（CrowdHuman 预训练 → 现场微调 50 轮）")
    ax.set_ylim(0.6, 1.0); ax.set_xlim(1, 52)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    ax.text(2, 0.63, "示意：逐轮曲线为示意，端点(0.954/0.795)与基线(0.931/0.855)为实测值",
            fontsize=9, color="#666", style="italic")
    save(fig, "fig_person_finetune_curve.png")


# 4) 六类行为 YOLO 训练指标对比（真实数据）
def fig_sixcls_metrics():
    metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]
    m1 = [0.739, 0.685, 0.709, 0.466]   # Student v6 e20/640
    m2 = [0.783, 0.736, 0.782, 0.516]   # merged-v2 e50/960
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    b1 = ax.bar(x - w/2, m1, w, label="Student Behaviour v6（e20/640）", color=C_BASE, edgecolor="white")
    b2 = ax.bar(x + w/2, m2, w, label="merged-classroom-6cls-v2（e50/960）", color="#22c3a6", edgecolor="white")
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_ylim(0, 1.0); ax.set_ylabel("指标值")
    ax.set_title("六类行为 YOLO 自训练模型验证指标对比")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_sixcls_metrics.png")


# 5) 大模型行为分类：加权得分率（仅加权指标，真实数据）
def fig_vlm_weighted():
    methods = ["融合", "GPT", "千问"]
    rate = [93.10, 89.29, 74.05]
    colors = [C_FUSE, C_GPT, C_QWEN]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(methods, rate, width=0.55, color=colors, edgecolor="white")
    for b, r in zip(bars, rate):
        ax.text(b.get_x() + b.get_width()/2, r + 0.8, f"{r:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105); ax.set_ylabel("加权得分率")
    ax.set_title("大模型课堂行为分类加权得分率（n=84 个有效目标）")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_vlm_weighted.png")


# 6) 大模型行为分类：融合方法分类别加权得分率（真实数据）
def fig_vlm_per_class():
    labels = ["举手\n(n=2)", "看书\n(n=16)", "写字\n(n=35)", "使用手机\n(n=22)", "低头\n(n=6)", "睡觉\n(n=3)"]
    rate = [140.00, 113.75, 111.43, 63.64, 70.00, 0.00]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(labels, rate, width=0.62, color=PALETTE6, edgecolor="white")
    for b, r in zip(bars, rate):
        ax.text(b.get_x() + b.get_width()/2, r + 2, f"{r:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.axhline(140, color="#888", ls=":", lw=1.2)
    ax.axhline(100, color="#444", ls="--", lw=1.0)
    ax.text(5.55, 142, "完全命中=140%", color="#666", fontsize=9, ha="right")
    ax.text(5.55, 102, "100%", color="#444", fontsize=9, ha="right")
    ax.set_ylim(0, 160); ax.set_ylabel("加权得分率")
    ax.set_title("融合方法分类别加权得分率（学习类 Reading/Writing 几乎全命中）")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_vlm_per_class.png")


if __name__ == "__main__":
    fig_person_finetune_metrics()
    fig_person_detect_counts()
    fig_person_finetune_curve()
    fig_sixcls_metrics()
    fig_vlm_weighted()
    fig_vlm_per_class()
    print("ALL DONE")
