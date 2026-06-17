#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 Word 转换准备：① 生成带中文字体的 pandoc 参考样式文档；② 预处理 Markdown
（为图片设置合适显示宽度、把居中图题转换为 pandoc 可识别的样式块）。
用法：python build_docx.py <default_ref.docx> <input.md> <out_ref.docx> <out.md>
"""
import re
import struct
import sys

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

DEFAULT_REF, INPUT_MD, OUT_REF, OUT_MD = sys.argv[1:5]
REPO = "/home/user/YOLOv8-ClassBehave"

BODY_LATIN = "Times New Roman"
BODY_CJK = "SimSun"      # 宋体
HEAD_LATIN = "Arial"
HEAD_CJK = "SimHei"      # 黑体


def set_fonts(style, latin, cjk):
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    rfonts.set(qn("w:cs"), latin)
    rfonts.set(qn("w:eastAsia"), cjk)


def build_reference():
    doc = Document(DEFAULT_REF)
    styles = doc.styles

    existing = []
    for s in styles:
        try:
            existing.append(s.name)
        except Exception:
            pass

    normal = styles["Normal"]
    normal.font.size = Pt(12)            # 小四
    set_fonts(normal, BODY_LATIN, BODY_CJK)
    normal.paragraph_format.line_spacing = 1.25

    for s in styles:
        try:
            nm = s.name or ""
        except Exception:
            continue
        if nm.startswith("Heading") or nm == "Title":
            try:
                set_fonts(s, HEAD_LATIN, HEAD_CJK)
            except Exception as exc:
                print("skip style", nm, exc)

    # 图题样式（居中、灰色、小五）
    if "ImageCaption" not in existing:
        cap = styles.add_style("ImageCaption", WD_STYLE_TYPE.PARAGRAPH)
        cap.base_style = styles["Normal"]
        cap.font.size = Pt(10.5)
        cap.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_before = Pt(2)
        cap.paragraph_format.space_after = Pt(10)
        set_fonts(cap, BODY_LATIN, BODY_CJK)

    if "FigureImage" not in existing:
        fig = styles.add_style("FigureImage", WD_STYLE_TYPE.PARAGRAPH)
        fig.base_style = styles["Normal"]
        fig.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fig.paragraph_format.space_before = Pt(8)
        fig.paragraph_format.space_after = Pt(2)
        set_fonts(fig, BODY_LATIN, BODY_CJK)

    doc.save(OUT_REF)
    print("reference saved:", OUT_REF)


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(26)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def img_width_cm(path):
    w, h = png_size(path)
    # 页面可用区约 16cm 宽、22cm 高；按比例约束，避免溢出
    max_w, max_h = 15.0, 20.0
    width = min(max_w, max_h * (w / h))
    return round(width, 1)


CAP_RE = re.compile(r'^<p align="center">(.*)</p>\s*$')
IMG_RE = re.compile(r'^!\[([^\]]*)\]\((report-figures/[^)]+)\)\s*$')


def html_inline_to_md(s):
    s = s.replace("<b>", "**").replace("</b>", "**")
    s = s.replace("<code>", "`").replace("</code>", "`")
    return s.strip()


def preprocess():
    out = []
    with open(INPUT_MD, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = IMG_RE.match(line)
            if m:
                alt, path = m.group(1), m.group(2)
                wcm = img_width_cm(f"{REPO}/{path}")
                out.append("")
                out.append('::: {custom-style="FigureImage"}')
                out.append(f"![{alt}]({path}){{width={wcm}cm}}")
                out.append(":::")
                out.append("")
                continue
            c = CAP_RE.match(line)
            if c:
                inner = html_inline_to_md(c.group(1))
                out.append("")
                out.append('::: {custom-style="ImageCaption"}')
                out.append(inner)
                out.append(":::")
                out.append("")
                continue
            out.append(line)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("preprocessed md saved:", OUT_MD)


if __name__ == "__main__":
    build_reference()
    preprocess()
    print("DONE")
