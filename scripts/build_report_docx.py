"""High-Quality Word Document (.docx) Builder for the Project Report.

Features:
- Proper 1-inch margins
- Elegant Cover Page with Page Break before Section 1
- Formatted tables with custom column widths, alternating cell shading, and styled headers
- Code blocks rendered in Consolas monospace inside shaded callout boxes
- Clean typography with proper heading styles
- Converts raw LaTeX formulas into clean, readable Unicode math
- Directly embeds all 16 multi-disciplinary figures with high-resolution scaling and centered captions
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "PROJECT_REPORT.md"
OUT = ROOT / "docs" / "PROJECT_REPORT.docx"
RESULTS = ROOT / "results"


def add_runs(paragraph, text: str):
    """Add text to a paragraph, rendering **bold** and `code` segments cleanly."""
    parts = re.split(r"(`[^`]+`)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        else:
            subparts = re.split(r"(\*\*[^*]+\*\*)", part)
            for sub in subparts:
                if not sub:
                    continue
                if sub.startswith("**") and sub.endswith("**"):
                    run = paragraph.add_run(sub[2:-2])
                    run.bold = True
                else:
                    paragraph.add_run(sub)


def shade_cell(cell, hex_color: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tc_pr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f"</w:tcMar>"
    )
    tc_pr.append(tcMar)


def add_table(doc, rows: list[list[str]]):
    if not rows:
        return
    header, body = rows[0], rows[1:]
    table = doc.add_table(rows=1 + len(body), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Style Header
    for c, text in enumerate(header):
        cell = table.rows[0].cells[c]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(clean_math(text))
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        shade_cell(cell, "D9E2F3")
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)

    # Style Body
    for r, row in enumerate(body, start=1):
        bg_color = "F8FAFC" if r % 2 == 1 else "FFFFFF"
        for c in range(len(header)):
            cell = table.rows[r].cells[c]
            cell.text = ""
            p = cell.paragraphs[0]
            val = row[c] if c < len(row) else ""
            add_runs(p, clean_math(val))
            for run in p.runs:
                run.font.name = "Calibri"
                run.font.size = Pt(9.0)
            shade_cell(cell, bg_color)
            set_cell_margins(cell, top=90, bottom=90, left=120, right=120)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_after = Pt(6)


def add_code_block(doc, code_lines: list[str]):
    """Render code or terminal outputs inside a clean callout box."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    shade_cell(cell, "F1F5F9")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05

    for line in code_lines:
        run = p.add_run(line + "\n")
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    doc.add_paragraph()  # Spacer


def parse_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def clean_math(text: str) -> str:
    """Convert raw LaTeX math expressions into clean readable Unicode representations."""
    t = text.replace("$$", "").replace("$", "")
    t = t.replace(r"\mathbf{W}_e", "W_e")
    t = t.replace(r"\mathbf{W}_p", "W_p")
    t = t.replace(r"\mathbf{W}_u", "W_u")
    t = t.replace(r"\mathbf{W}_Q", "W_Q")
    t = t.replace(r"\mathbf{W}_K", "W_K")
    t = t.replace(r"\mathbf{W}_V", "W_V")
    t = t.replace(r"\mathbf{W}_1", "W_1")
    t = t.replace(r"\mathbf{W}_2", "W_2")
    t = t.replace(r"\mathbf{x}'_l", "x'_l")
    t = t.replace(r"\mathbf{x}_{l-1}", "x_{l-1}")
    t = t.replace(r"\mathbf{x}_l", "x_l")
    t = t.replace(r"\mathbf{x}_0", "x_0")
    t = t.replace(r"\mathbf{x}_L", "x_L")
    t = t.replace(r"\hat{\mathbf{y}}", "y_pred")
    t = t.replace(r"\mathbf{Q}", "Q")
    t = t.replace(r"\mathbf{K}", "K")
    t = t.replace(r"\mathbf{V}", "V")
    t = t.replace(r"\mathbf{X}", "X")
    t = t.replace(r"\mathbf{t}", "t")
    t = t.replace(r"\mathbf{M}", "M")
    t = t.replace(r"\mathbf{h}", "h")
    t = t.replace(r"\mathbf{b}_1", "b_1")
    t = t.replace(r"\mathbf{b}_2", "b_2")
    t = t.replace(r"\text{MHA}", "MHA")
    t = t.replace(r"\text{MLP}", "MLP")
    t = t.replace(r"\text{LayerNorm}", "LayerNorm")
    t = t.replace(r"\text{Softmax}", "Softmax")
    t = t.replace(r"\text{GELU}", "GeLU")
    t = t.replace(r"\text{Attention}", "Attention")
    t = t.replace(r"\text{Mask}", "Mask")
    t = t.replace(r"\mathbb{R}^{V \times D}", "R^(V × D)")
    t = t.replace(r"\mathbb{R}^{T \times D}", "R^(T × D)")
    t = t.replace(r"\mathbb{R}^{D \times V}", "R^(D × V)")
    t = t.replace(r"\in", "in")
    t = t.replace(r"\cdot", "·")
    t = t.replace(r"\times", "×")
    t = t.replace(r"\sqrt{d_k}", "sqrt(d_k)")
    t = t.replace(r"\sqrt{2 N_{\text{layer}}}", "sqrt(2·N_layer)")
    t = t.replace(r"\sigma = \frac{0.02}{\sqrt{2 N_{\text{layer}}}}", "σ = 0.02 / sqrt(2·N_layer)")
    t = t.replace(r"\sigma_{\text{init}} = \frac{0.02}{\sqrt{2 \cdot N_{\text{layer}}}}", "σ_init = 0.02 / sqrt(2·N_layer)")
    t = t.replace(r"\|g\|_2 \le 1.0", "||g||_2 <= 1.0")
    t = t.replace(r"\beta_1 = 0.9", "β_1 = 0.9")
    t = t.replace(r"\beta_2 = 0.95", "β_2 = 0.95")
    t = t.replace(r"\eta_{\max} = 6 \times 10^{-4}", "η_max = 6×10^-4")
    t = t.replace(r"\eta_{\min} = 6 \times 10^{-5}", "η_min = 6×10^-5")
    t = t.replace(r"\le", "<=")
    t = t.replace(r"\ge", ">=")
    t = t.replace(r"-\infty", "-inf")
    t = t.replace(r"\pi", "π")
    t = t.replace(r"\prod", "∏")
    t = t.replace(r"\dots", "...")
    t = t.replace(r"\quad", "   ")
    t = t.replace(r"\longrightarrow", " ---> ")
    return t


def main():
    doc = Document()

    # Set 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Style defaults
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    style.paragraph_format.line_spacing = 1.15
    style.paragraph_format.space_after = Pt(5)

    doc.styles["Heading 1"].font.name = "Calibri"
    doc.styles["Heading 1"].font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)
    doc.styles["Heading 1"].font.size = Pt(16)
    doc.styles["Heading 1"].font.bold = True
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(14)
    doc.styles["Heading 1"].paragraph_format.space_after = Pt(6)

    doc.styles["Heading 2"].font.name = "Calibri"
    doc.styles["Heading 2"].font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
    doc.styles["Heading 2"].font.size = Pt(13)
    doc.styles["Heading 2"].font.bold = True
    doc.styles["Heading 2"].paragraph_format.space_before = Pt(10)
    doc.styles["Heading 2"].paragraph_format.space_after = Pt(4)

    doc.styles["Heading 3"].font.name = "Calibri"
    doc.styles["Heading 3"].font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    doc.styles["Heading 3"].font.size = Pt(11.5)
    doc.styles["Heading 3"].font.bold = True
    doc.styles["Heading 3"].paragraph_format.space_before = Pt(8)
    doc.styles["Heading 3"].paragraph_format.space_after = Pt(3)

    lines = SRC.read_text(encoding="utf-8").splitlines()
    i = 0
    in_code_block = False
    code_lines = []
    is_cover_page = True

    while i < len(lines):
        line = lines[i].rstrip()

        # Handle Code Blocks
        if line.startswith("```"):
            if in_code_block:
                add_code_block(doc, code_lines)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Skip horizontal rules
        if line.strip() == "---":
            i += 1
            continue

        # Handle Markdown Images: ![Caption](../results/file.png)
        img_match = re.match(r"^!\[(.*?)\]\((.*?)\)", line.strip())
        if img_match:
            img_rel_path = img_match.group(2).strip()
            fname = Path(img_rel_path).name
            full_path = RESULTS / fname
            if full_path.exists():
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(8)
                p_img.paragraph_format.space_after = Pt(2)
                run = p_img.add_run()
                run.add_picture(str(full_path), width=Inches(5.8))
            i += 1
            # Check if next line is an italicized caption: *Figure X...*
            if i < len(lines) and lines[i].strip().startswith("*Figure"):
                cap_line = lines[i].strip().strip("*")
                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_cap.paragraph_format.space_after = Pt(12)
                run_cap = p_cap.add_run(cap_line)
                run_cap.font.name = "Calibri"
                run_cap.font.size = Pt(9.0)
                run_cap.italic = True
                run_cap.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
                i += 1
            continue

        # Tables
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not re.match(r"^\|\s*:?-+", lines[i].strip()):
                    rows.append(parse_row(lines[i]))
                i += 1
            add_table(doc, rows)
            continue

        if not line.strip():
            i += 1
            continue

        # Headings
        if line.startswith("# "):
            title = doc.add_paragraph()
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = title.add_run(line[2:].strip())
            run.bold = True
            run.font.size = Pt(20)
            run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x5F)
            title.paragraph_format.space_after = Pt(10)
        elif line.startswith("## "):
            heading_text = line[3:].strip()
            
            # If leaving cover page to Section 1, add page break!
            if is_cover_page and heading_text.startswith("1."):
                doc.add_page_break()
                is_cover_page = False
            elif heading_text.startswith("References"):
                doc.add_page_break()

            doc.add_heading(heading_text, level=1)
        elif line.startswith("### "):
            heading_text = line[4:].strip()
            doc.add_heading(heading_text, level=3)
        elif line.startswith("#### "):
            heading_text = line[5:].strip()
            p = doc.add_paragraph()
            run = p.add_run(heading_text)
            run.bold = True
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
        elif re.match(r"^\s*-\s+", line):
            p = doc.add_paragraph(style="List Bullet")
            cleaned = clean_math(re.sub(r"^\s*-\s+", "", line))
            add_runs(p, cleaned)
            p.paragraph_format.space_after = Pt(2)
        elif re.match(r"^\s*\d+\.\s+", line):
            p = doc.add_paragraph(style="List Number")
            cleaned = clean_math(re.sub(r"^\s*\d+\.\s+", "", line))
            add_runs(p, cleaned)
            p.paragraph_format.space_after = Pt(2)
        else:
            p = doc.add_paragraph()
            cleaned = clean_math(line.strip())
            add_runs(p, cleaned)
            # Center cover page elements
            if is_cover_page:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(4)

        i += 1

    doc.save(OUT)
    print(f"Successfully generated clean Word Report: {OUT}")


if __name__ == "__main__":
    main()
