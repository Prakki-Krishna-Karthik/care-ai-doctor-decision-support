from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PNG_PATH = ROOT / "CARE_AI_architecture_diagram.png"
SVG_PATH = ROOT / "CARE_AI_architecture_diagram.svg"
W, H = 2400, 1500
DRAWIO_STYLE = True


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


TITLE = font(32, True)
SUBTITLE = font(16)
SECTION = font(18, True)
BOX_TITLE = font(17, True)
BODY = font(14)
SMALL = font(12)
SMALL_BOLD = font(12, True)

BG = "#FFFFFF"
INK = "#1F1F1F"
MUTED = "#333333"
LINE = "#4D6885"
COLORS = {
    "data": "#FFFFFF",
    "prep": "#FFFFFF",
    "model": "#FFFFFF",
    "reliability": "#FFFFFF",
    "output": "#FFFFFF",
    "eval": "#FFFFFF",
    "deploy": "#FFFFFF",
}


def rounded(draw, xy, fill, outline=LINE, radius=6, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw, text, x, y, w, fnt, fill=INK):
    box = draw.textbbox((0, 0), text, font=fnt)
    tw = box[2] - box[0]
    draw.text((x + (w - tw) / 2, y), text, font=fnt, fill=fill)


def draw_box(draw, x, y, w, h, title, lines, fill, accent=None):
    rounded(draw, (x, y, x + w, y + h), fill)
    if accent and not DRAWIO_STYLE:
        draw.rounded_rectangle((x, y, x + 9, y + h), radius=8, fill=accent)
    draw.text((x + 24, y + 18), title, font=BOX_TITLE, fill=INK)
    ty = y + 58
    for line in lines:
        wrapped = wrap(line, width=max(26, int(w / 12)))
        for part in wrapped:
            draw.text((x + 28, ty), "• " + part, font=BODY, fill=MUTED)
            ty += 25
        ty += 3


def arrow(draw, start, end, label=None, dashed=False):
    x1, y1 = start
    x2, y2 = end
    if dashed:
        steps = max(1, int(abs(x2 - x1) / 16))
        for i in range(0, steps, 2):
            xa = x1 + (x2 - x1) * i / steps
            xb = x1 + (x2 - x1) * min(i + 1, steps) / steps
            draw.line((xa, y1, xb, y2), fill=LINE, width=4)
    else:
        draw.line((x1, y1, x2, y2), fill=LINE, width=4)
    # Arrow head, oriented for the common left-to-right or vertical flows.
    if abs(x2 - x1) >= abs(y2 - y1):
        pts = [(x2, y2), (x2 - 14, y2 - 8), (x2 - 14, y2 + 8)]
    elif y2 >= y1:
        pts = [(x2, y2), (x2 - 8, y2 - 14), (x2 + 8, y2 - 14)]
    else:
        pts = [(x2, y2), (x2 - 8, y2 + 14), (x2 + 8, y2 + 14)]
    draw.polygon(pts, fill=LINE)
    if label:
        lx = (x1 + x2) // 2
        ly = (y1 + y2) // 2 - 26
        draw.rounded_rectangle((lx - 72, ly - 2, lx + 72, ly + 25), radius=7, fill=BG)
        centered(draw, label, lx - 72, ly + 1, 144, SMALL_BOLD, MUTED)


def ortho_arrow(draw, start, end, label=None, dashed=False, via_x=None, via_y=None):
    """Draw a clean Manhattan-style connector that never cuts through a box."""
    x1, y1 = start
    x2, y2 = end
    if x1 == x2 or y1 == y2:
        points = [(x1, y1), (x2, y2)]
    elif via_x is not None:
        points = [(x1, y1), (via_x, y1), (via_x, y2), (x2, y2)]
    else:
        mid_y = via_y if via_y is not None else (y1 + y2) // 2
        points = [(x1, y1), (x1, mid_y), (x2, mid_y), (x2, y2)]
    for a, b in zip(points, points[1:]):
        if dashed:
            length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
            step = 18
            for pos in range(0, max(1, length), step * 2):
                end_pos = min(pos + step, length)
                if a[0] != b[0]:
                    xa = a[0] + (b[0] - a[0]) * pos / length
                    xb = a[0] + (b[0] - a[0]) * end_pos / length
                    draw.line((xa, a[1], xb, b[1]), fill=LINE, width=4)
                else:
                    ya = a[1] + (b[1] - a[1]) * pos / length
                    yb = a[1] + (b[1] - a[1]) * end_pos / length
                    draw.line((a[0], ya, b[0], yb), fill=LINE, width=4)
        else:
            draw.line((a[0], a[1], b[0], b[1]), fill=LINE, width=4)
    # Orient the arrow head along the final segment.
    px, py = points[-2]
    if x2 != px:
        direction = 1 if x2 > px else -1
        pts = [(x2, y2), (x2 - direction * 14, y2 - 8), (x2 - direction * 14, y2 + 8)]
    else:
        direction = 1 if y2 > py else -1
        pts = [(x2, y2), (x2 - 8, y2 - direction * 14), (x2 + 8, y2 - direction * 14)]
    draw.polygon(pts, fill=LINE)
    if label:
        # Place the label above the first segment, away from box edges.
        if x1 != points[1][0]:
            lx, ly = (x1 + points[1][0]) // 2, y1 - 28
        else:
            lx, ly = x1 + 18, (y1 + points[1][1]) // 2 - 12
        draw.rounded_rectangle((lx - 58, ly - 2, lx + 58, ly + 24), radius=7, fill=BG)
        centered(draw, label, lx - 58, ly + 1, 116, SMALL_BOLD, MUTED)


def build_png():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "CARE-AI: Reliability-Aware Multimodal Chest X-ray Decision Support", font=TITLE, fill=INK)
    draw.text((73, 90), "Implemented local pipeline: data integrity → trained image/clinical branches → reliability and uncertainty → RAG evidence → clinician-facing output", font=SUBTITLE, fill=MUTED)

    # Section bands.
    sections = [
        (40, 155, 350, "1  DATA SOURCES", COLORS["data"]),
        (410, 155, 350, "2  PREPARATION", COLORS["prep"]),
        (780, 155, 520, "3  MULTIMODAL MODEL", COLORS["model"]),
        (1320, 155, 400, "4  TRUST LAYER", COLORS["reliability"]),
        (1740, 155, 600, "5  DECISION SUPPORT", COLORS["output"]),
    ]
    for x, y, w, label, fill in sections:
        rounded(draw, (x, y, x + w, y + 48), "#D9EAF7", outline="#6C8EBF", radius=3, width=1)
        centered(draw, label, x, y + 11, w, SECTION)

    # Main pipeline boxes.
    draw_box(draw, 55, 235, 320, 270, "Local data", [
        "MIMIC-CXR-JPG chest radiographs",
        "Clinical CSVs: metadata, CheXpert, splits",
        "Radiology reports ZIP for local evidence",
        "No patient data leaves the local machine",
    ], COLORS["data"], "#3D7BE8")
    draw_box(draw, 55, 555, 320, 270, "Data inventory", [
        "Recursive image discovery",
        "Filename and content duplicate checks",
        "Study/patient identity retained",
        "Missingness and quality audit",
    ], COLORS["data"], "#3D7BE8")
    draw_box(draw, 425, 235, 320, 270, "Preprocessing", [
        "Resize and normalize images to 224 × 224",
        "Join clinical records by subject/study ID",
        "Handle uncertain and missing labels",
        "Train / validation / test split by patient",
    ], COLORS["prep"], "#2C9B59")
    draw_box(draw, 425, 555, 320, 270, "Small-data safeguards", [
        "Transfer learning from ImageNet",
        "Conservative augmentation",
        "Early stopping and weight decay",
        "Patient-wise evaluation and calibration",
    ], COLORS["prep"], "#2C9B59")

    draw_box(draw, 795, 235, 235, 300, "Image branch", [
        "DenseNet-121",
        "Fine-tuned chest X-ray encoder",
        "14 disease label probabilities",
        "Grad-CAM heatmap",
    ], COLORS["model"], "#7A4DD8")
    draw_box(draw, 1050, 235, 235, 300, "Clinical branch", [
        "Structured metadata",
        "Random Forest candidate",
        "Completeness and feature quality",
        "Retained for comparison/audit",
    ], COLORS["model"], "#7A4DD8")
    draw_box(draw, 920, 595, 235, 230, "Representation", [
        "Image embedding Hᵢ",
        "Clinical embedding H꜀",
        "Shared decision-support space",
    ], COLORS["model"], "#7A4DD8")

    draw_box(draw, 1340, 235, 360, 260, "Reliability estimation", [
        "Image: sharpness, contrast, size",
        "Clinical: completeness and usefulness",
        "Scores Rᵢ and R꜀ in [0, 1]",
        "Reliability is reported, not hidden",
    ], COLORS["reliability"], "#D78B18")
    draw_box(draw, 1340, 555, 360, 270, "Reliability-aware fusion", [
        "w꜀ = R꜀ / (R꜀ + Rᵢ + ε)",
        "wᵢ = Rᵢ / (R꜀ + Rᵢ + ε)",
        "F = w꜀H꜀ + wᵢHᵢ",
        "Weak evidence contributes less",
    ], COLORS["reliability"], "#D78B18")

    draw_box(draw, 1760, 235, 260, 260, "Prediction", [
        "14-label multilabel output",
        "Calibrated probabilities",
        "Low / moderate / high review risk",
        "Threshold-based abstention",
    ], COLORS["output"], "#D64B46")
    draw_box(draw, 2045, 235, 260, 260, "Uncertainty", [
        "Original image view",
        "Contrast-adjusted view",
        "Brightness-adjusted view",
        "Ensemble spread and review priority",
    ], COLORS["output"], "#D64B46")
    draw_box(draw, 1760, 555, 545, 270, "Local RAG and report comparison", [
        "Retrieve relevant reports from local MIMIC-CXR archive",
        "TF-IDF similarity with study-aware matching",
        "Negation-aware comparison of predicted findings",
        "Evidence note and local citation returned to the reviewer",
    ], COLORS["output"], "#D64B46")
    draw_box(draw, 1760, 885, 545, 235, "Clinician-facing decision support", [
        "Image + predicted findings + Grad-CAM",
        "Confidence, uncertainty, and review priority",
        "Retrieved evidence and report comparison",
        "Feedback stored locally for future audit",
    ], COLORS["deploy"], "#596A80")

    # Arrows.
    ortho_arrow(draw, (375, 370), (425, 370), "ingest")
    ortho_arrow(draw, (215, 505), (215, 555), "audit")
    ortho_arrow(draw, (745, 370), (795, 370), "image")
    ortho_arrow(draw, (745, 690), (920, 710), "features", via_x=835)
    ortho_arrow(draw, (912, 535), (960, 595), "Hᵢ", via_y=565)
    ortho_arrow(draw, (1168, 535), (1115, 595), "H꜀", via_y=565)
    ortho_arrow(draw, (1285, 370), (1340, 370), "quality")
    ortho_arrow(draw, (1155, 710), (1340, 690), "fuse", via_x=1240)
    ortho_arrow(draw, (1520, 495), (1520, 555), "weights")
    ortho_arrow(draw, (1700, 690), (1760, 365), "F", via_x=1730)
    ortho_arrow(draw, (2020, 365), (2045, 365), "spread")
    ortho_arrow(draw, (1890, 495), (1890, 555), "evidence")
    ortho_arrow(draw, (2040, 825), (2040, 885), "support")

    # Evaluation / governance footer.
    rounded(draw, (55, 930, 1695, 1375), COLORS["eval"], outline="#6C8EBF", radius=6, width=2)
    draw.text((85, 960), "6  EVALUATION, GOVERNANCE, AND RESEARCH ANALYSIS", font=SECTION, fill=INK)
    draw_box(draw, 85, 1020, 350, 275, "Predictive performance", [
        "AUROC and AUPRC",
        "Precision, recall, F1-score",
        "Confusion matrices by label",
        "Per-label and macro averages",
    ], "#FFFFFF", "#6C8EBF")
    draw_box(draw, 470, 1020, 350, 275, "Calibration and uncertainty", [
        "Brier score and ECE",
        "Reliability diagrams",
        "Confidence versus error",
        "Selective prediction / abstention",
    ], "#FFFFFF", "#6C8EBF")
    draw_box(draw, 855, 1020, 350, 275, "Robustness and fairness", [
        "Missing clinical data scenarios",
        "Degraded image quality scenarios",
        "View-position cohort audit",
        "Age/sex unavailable in supplied metadata",
    ], "#FFFFFF", "#6C8EBF")
    draw_box(draw, 1240, 1020, 415, 275, "Model-selection logic", [
        "Compare image, clinical, and fusion candidates",
        "Choose using calibration, runtime, and stability",
        "Current deployment: trained DenseNet-121",
        "Clinical branch remains an auditable comparator",
    ], "#FFFFFF", "#6C8EBF")

    # Feedback loop from UI back to artifacts.
    ortho_arrow(draw, (1760, 1000), (1695, 1250), "feedback", dashed=True, via_x=1725)

    # Deployment note.
    rounded(draw, (55, 1400, 2345, 1460), COLORS["deploy"], outline="#6C8EBF", radius=4, width=1)
    draw.text((80, 1418), "Current implementation: FastAPI backend + browser dashboard; trained artifacts are loaded locally; RAG uses the local report archive; outputs are decision support, not an autonomous diagnosis.", font=SMALL_BOLD, fill=INK)

    image.save(PNG_PATH, "PNG", optimize=True)


def svg_text(x, y, text, size=17, weight="400", fill=INK, anchor="start"):
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" font-size="{size}px" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{safe}</text>'


def svg_ortho_arrow(x1, y1, x2, y2, via_x=None, via_y=None, dashed=False):
    if x1 == x2 or y1 == y2:
        points = [(x1, y1), (x2, y2)]
    elif via_x is not None:
        points = [(x1, y1), (via_x, y1), (via_x, y2), (x2, y2)]
    else:
        mid_y = via_y if via_y is not None else (y1 + y2) // 2
        points = [(x1, y1), (x1, mid_y), (x2, mid_y), (x2, y2)]
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    items = []
    for (xa, ya), (xb, yb) in zip(points, points[1:]):
        items.append(f'<line x1="{xa}" y1="{ya}" x2="{xb}" y2="{yb}"{dash}/>')
    px, py = points[-2]
    if x2 != px:
        direction = 1 if x2 > px else -1
        poly = f"{x2},{y2} {x2-direction*14},{y2-8} {x2-direction*14},{y2+8}"
    else:
        direction = 1 if y2 > py else -1
        poly = f"{x2},{y2} {x2-8},{y2-direction*14} {x2+8},{y2-direction*14}"
    items.append(f'<polygon points="{poly}"/>')
    return "".join(items)


def build_svg():
    # A lightweight, editable SVG companion with the main architecture at a readable scale.
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        svg_text(70, 70, "CARE-AI: Reliability-Aware Multimodal Chest X-ray Decision Support", 32, "700"),
        svg_text(73, 105, "Data integrity → trained branches → reliability and uncertainty → local RAG evidence → clinician-facing output", 16, "400", MUTED),
    ]
    # SVG uses the same visual grouping as the PNG and is intentionally concise/editable.
    boxes = [
        (55,235,320,270,"Local data",["MIMIC-CXR-JPG images","Clinical CSVs and splits","Local radiology report archive"],COLORS["data"]),
        (425,235,320,270,"Preprocessing",["Recursive discovery and dedup","Clinical joins and missingness","Patient-wise split"],COLORS["prep"]),
        (795,235,235,300,"Image branch",["DenseNet-121","14 label probabilities","Grad-CAM"],COLORS["model"]),
        (1050,235,235,300,"Clinical branch",["Structured metadata","Random Forest candidate","Audit comparator"],COLORS["model"]),
        (1340,235,360,260,"Reliability estimation",["Image quality score Rᵢ","Clinical quality score R꜀","Scores in [0,1]"],COLORS["reliability"]),
        (1340,555,360,270,"Reliability-aware fusion",["w꜀ and wᵢ from reliability","F = w꜀H꜀ + wᵢHᵢ","Weak evidence contributes less"],COLORS["reliability"]),
        (1760,235,260,260,"Prediction",["14-label output","Calibrated probabilities","Abstention gate"],COLORS["output"]),
        (2045,235,260,260,"Uncertainty",["Three image views","Ensemble spread","Review priority"],COLORS["output"]),
        (1760,555,545,270,"Local RAG and report comparison",["TF-IDF retrieval from local reports","Study-aware matching","Negation-aware comparison"],COLORS["output"]),
        (1760,885,545,235,"Clinician-facing decision support",["Predictions + Grad-CAM","Confidence and uncertainty","Evidence and feedback"],COLORS["deploy"]),
    ]
    for x,y,w,h,title,lines,fill in boxes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{LINE}" stroke-width="2"/>')
        parts.append(svg_text(x+24,y+40,title,17,"700"))
        yy=y+82
        for line in lines:
            parts.append(svg_text(x+28,yy,"• "+line,14,"400",MUTED))
            yy += 29
    # Main arrows.
    arrows = [
        (375,370,425,370,None,None,False),
        (745,370,795,370,None,None,False),
        (1285,370,1340,370,None,None,False),
        (1700,690,1760,365,1730,None,False),
        (2020,365,2045,365,None,None,False),
        (1890,495,1890,555,None,None,False),
        (2040,825,2040,885,None,None,False),
        (1760,1000,1695,1250,1725,None,True),
    ]
    parts.append(f'<g stroke="{LINE}" stroke-width="4" fill="{LINE}">')
    for x1,y1,x2,y2,via_x,via_y,dashed in arrows:
        parts.append(svg_ortho_arrow(x1,y1,x2,y2,via_x,via_y,dashed))
    parts.append('</g>')
    parts.append(f'<rect x="55" y="930" width="1640" height="445" rx="6" fill="{COLORS["eval"]}" stroke="#6C8EBF" stroke-width="2"/>')
    parts.append(svg_text(85, 975, "6  EVALUATION, GOVERNANCE, AND RESEARCH ANALYSIS", 18, "700"))
    footer = ["AUROC / AUPRC / F1 / confusion matrices", "Brier score / ECE / reliability diagrams", "Robustness to missing clinical data and degraded images", "View-position cohort audit; unavailable demographics reported explicitly", "Model selection uses performance, calibration, runtime, and stability"]
    for i, line in enumerate(footer):
        parts.append(svg_text(100, 1040 + i * 52, "• " + line, 14, "400", MUTED))
    parts.append(f'<rect x="55" y="1400" width="2290" height="60" rx="4" fill="{COLORS["deploy"]}" stroke="#6C8EBF" stroke-width="1"/>')
    parts.append(svg_text(80, 1438, "Current implementation: FastAPI backend + browser dashboard; local trained artifacts; local report retrieval; decision support rather than autonomous diagnosis.", 13, "700"))
    parts.append('</svg>')
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    build_png()
    build_svg()
    print(PNG_PATH)
    print(SVG_PATH)
