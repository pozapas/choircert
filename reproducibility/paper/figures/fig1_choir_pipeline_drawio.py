"""Generate an editable draw.io version of Fig. 1.

The source of truth for geometry is the reviewed matplotlib Fig. 1 layout in
fig1_choir_pipeline.py. Labels are kept notation-first so the draw.io file stays
faithful to the zero-prose figure grammar.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import xml.etree.ElementTree as ET


OUT = Path(__file__).resolve().parent

INK = "#1a1a1a"
HDR_BLUE = "#1f4e79"
CORAL = "#c1443c"
GOLD = "#a87c2a"
SLATE = "#697784"
MIST = "#e6eeec"
WHITE = "#ffffff"

SCALE = 92.0
MARGIN_X = 40.0
MARGIN_Y = 34.0
HEIGHT_UNITS = 7.65
PAGE_W = 1400
PAGE_H = 760


def sx(x: float) -> float:
    return MARGIN_X + x * SCALE


def sy(y: float) -> float:
    return MARGIN_Y + (HEIGHT_UNITS - y) * SCALE


class Drawio:
    def __init__(self) -> None:
        self.root_cells: list[ET.Element] = [
            ET.Element("mxCell", {"id": "0"}),
            ET.Element("mxCell", {"id": "1", "parent": "0"}),
        ]
        self.next_id = 2
        self.nodes: dict[str, tuple[float, float, float]] = {}

    def _id(self, prefix: str) -> str:
        cid = f"{prefix}{self.next_id}"
        self.next_id += 1
        return cid

    def cell(self, **attrs: str) -> ET.Element:
        c = ET.Element("mxCell", attrs)
        self.root_cells.append(c)
        return c

    def geom(self, cell: ET.Element, **attrs: str) -> None:
        ET.SubElement(cell, "mxGeometry", {**attrs, "as": "geometry"})

    def add_text(self, x: float, y: float, w: float, h: float, value: str,
                 color: str = INK, size: int = 14, align: str = "center",
                 valign: str = "middle", italic: bool = False,
                 bold: bool = False) -> str:
        font_style = (2 if italic else 0) + (1 if bold else 0)
        cid = self._id("t")
        style = (
            "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
            f"align={align};verticalAlign={valign};fontFamily=Times New Roman;"
            f"fontSize={size};fontColor={color};fontStyle={font_style};"
            "spacing=0;spacingTop=0;spacingBottom=0;"
        )
        c = self.cell(id=cid, value=value, style=style, vertex="1", parent="1")
        self.geom(c, x=f"{sx(x):.2f}", y=f"{sy(y) - h / 2:.2f}",
                  width=f"{w:.2f}", height=f"{h:.2f}")
        return cid

    def add_node(self, name: str, x: float, y: float, label: str, r: float = 0.42,
                 shaded: bool = False, size: int = 18) -> str:
        cid = f"n_{name}"
        d = 2 * r * SCALE
        style = (
            "ellipse;whiteSpace=wrap;html=1;aspect=fixed;"
            f"strokeColor={INK};fillColor={MIST if shaded else WHITE};"
            "strokeWidth=1.4;fontFamily=Times New Roman;"
            f"fontSize={size};fontColor={INK};"
        )
        c = self.cell(id=cid, value=label, style=style, vertex="1", parent="1")
        self.geom(c, x=f"{sx(x) - d / 2:.2f}", y=f"{sy(y) - d / 2:.2f}",
                  width=f"{d:.2f}", height=f"{d:.2f}")
        self.nodes[name] = (x, y, r)
        return cid

    def add_rect(self, name: str, x: float, y: float, w: float, h: float,
                 label: str = "", color: str = INK, fill: str = WHITE,
                 rounded: bool = True, size: int = 16, dashed: bool = False) -> str:
        cid = f"n_{name}"
        style = (
            f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;"
            f"strokeColor={color};fillColor={fill};strokeWidth=1.4;"
            f"fontFamily=Times New Roman;fontSize={size};fontColor={INK};"
        )
        if dashed:
            style += "dashed=1;dashPattern=4 3;"
        c = self.cell(id=cid, value=label, style=style, vertex="1", parent="1")
        self.geom(c, x=f"{sx(x):.2f}", y=f"{sy(y) - h * SCALE:.2f}",
                  width=f"{w * SCALE:.2f}", height=f"{h * SCALE:.2f}")
        self.nodes[name] = (x + w / 2, y - h / 2, max(w, h) / 2)
        return cid

    def add_plate(self, x: float, y: float, w: float, h: float, color: str) -> str:
        cid = self._id("p")
        style = (
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=none;"
            f"strokeColor={color};strokeWidth=1.8;"
        )
        c = self.cell(id=cid, value="", style=style, vertex="1", parent="1")
        self.geom(c, x=f"{sx(x):.2f}", y=f"{sy(y + h):.2f}",
                  width=f"{w * SCALE:.2f}", height=f"{h * SCALE:.2f}")
        return cid

    def add_edge(self, source: str | tuple[float, float],
                 target: str | tuple[float, float], color: str = INK,
                 dashed: bool = False, curved: bool = False,
                 waypoints: list[tuple[float, float]] | None = None) -> str:
        cid = self._id("e")
        style = (
            "endArrow=block;html=1;rounded=0;strokeWidth=1.6;"
            f"strokeColor={color};"
        )
        if dashed:
            style += "dashed=1;dashPattern=4 3;"
        if curved:
            style += "curved=1;"
        attrs = {"id": cid, "value": "", "style": style, "edge": "1", "parent": "1"}
        source_point = None
        target_point = None
        if isinstance(source, str):
            attrs["source"] = f"n_{source}"
        else:
            source_point = source
        if isinstance(target, str):
            attrs["target"] = f"n_{target}"
        else:
            target_point = target
        c = self.cell(**attrs)
        g = ET.SubElement(c, "mxGeometry", {"relative": "1", "as": "geometry"})
        if source_point is not None:
            ET.SubElement(g, "mxPoint", {"x": f"{sx(source_point[0]):.2f}",
                                         "y": f"{sy(source_point[1]):.2f}",
                                         "as": "sourcePoint"})
        if target_point is not None:
            ET.SubElement(g, "mxPoint", {"x": f"{sx(target_point[0]):.2f}",
                                         "y": f"{sy(target_point[1]):.2f}",
                                         "as": "targetPoint"})
        if waypoints:
            arr = ET.SubElement(g, "Array", {"as": "points"})
            for x, y in waypoints:
                ET.SubElement(arr, "mxPoint", {"x": f"{sx(x):.2f}", "y": f"{sy(y):.2f}"})
        return cid

    def add_hyper(self, x: float, y: float, label: str, target: str,
                  color: str = SLATE, dashed: bool = False) -> None:
        self.add_text(x - 0.35, y, 64, 20, label, color=color, size=15, italic=True)
        self.add_edge((x, y - 0.10), target, color=color, dashed=dashed)

    def write(self, path: Path) -> None:
        mxfile = ET.Element("mxfile", {
            "host": "app.diagrams.net",
            "modified": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agent": "Codex",
            "version": "24.7.17",
            "type": "device",
        })
        diagram = ET.SubElement(mxfile, "diagram", {"id": "fig1-choir", "name": "Fig 1 CHOIR pipeline"})
        model = ET.SubElement(diagram, "mxGraphModel", {
            "dx": "1400", "dy": "760", "grid": "1", "gridSize": "10",
            "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
            "fold": "1", "page": "1", "pageScale": "1",
            "pageWidth": str(PAGE_W), "pageHeight": str(PAGE_H),
            "math": "1", "shadow": "0",
        })
        root = ET.SubElement(model, "root")
        for cell in self.root_cells:
            root.append(cell)
        ET.indent(mxfile, space="  ")
        path.write_text(ET.tostring(mxfile, encoding="unicode"), encoding="utf-8")


d = Drawio()

# Plates.
d.add_plate(0.75, 1.30, 2.55, 4.10, CORAL)
d.add_plate(4.30, 1.05, 3.05, 4.55, HDR_BLUE)
d.add_plate(5.00, 1.75, 3.70, 2.70, GOLD)
d.add_plate(9.60, 0.95, 4.45, 4.65, SLATE)
d.add_text(0.80, 1.55, 150, 22, r"$$i\in\mathcal{I}_{\mathrm{cal}}$$", color=CORAL, size=15, align="left", italic=True)
d.add_text(4.35, 1.30, 100, 22, r"$$c=1{:}C$$", color=HDR_BLUE, size=15, align="left", italic=True)
d.add_text(7.85, 4.30, 105, 22, r"$$\gamma\in\mathcal{G}$$", color=GOLD, size=15, align="right", italic=True)
d.add_text(13.08, 1.22, 115, 22, r"$$t=1,2,\ldots$$", color=SLATE, size=15, align="right", italic=True)

# Fitted tier.
d.add_node("F", 1.35, 6.55, r"$$\hat F(\cdot\mid x)$$", r=0.50, size=21)
d.add_node("c", 4.00, 6.55, r"$$\hat c(x)$$", size=20)
d.add_node("g", 5.90, 6.55, r"$$g(x)$$", size=20)
d.add_node("w", 7.80, 6.55, r"$$\hat w(x)$$", size=20)
for x, target in [(1.35, "F"), (4.00, "c"), (5.90, "g"), (7.80, "w")]:
    d.add_hyper(x, 7.38, r"$$\mathcal{I}_{\mathrm{tr}}$$", target)
d.add_rect("Xstar", 8.93, 7.55, 0.42, 0.42, r"$$X^*$$", color=GOLD, rounded=False, size=18)
d.add_edge((9.02, 7.10), "w", color=GOLD, dashed=True)

# Calibration.
d.add_node("XY", 2.35, 4.35, r"$$(X_i,\tilde Y_i)$$", r=0.52, shaded=True, size=18)
d.add_node("S", 2.00, 2.75, r"$$S_i$$", r=0.38, size=20)
d.add_text(1.78, 5.78, 360, 24, r"$$s(x,y)=\max\{\hat F(y{-}1\mid x),\,1-\hat F(y\mid x)\}$$",
           size=15, italic=True, align="left")
d.add_edge("F", "S")
d.add_edge("XY", "S")

# Product cell and risk lambda.
d.add_node("q", 6.15, 3.10, r"$$\hat q_{c\gamma}$$", size=19)
d.add_text(5.30, 2.38, 170, 22, r"$$\lceil(1-\alpha)(n_{c\gamma}+1)\rceil$$", size=14)
d.add_edge("S", "q")
d.add_edge("c", "q")
d.add_edge("g", "q")
d.add_edge("w", "q", color=GOLD, dashed=True)
d.add_node("lam", 8.00, 1.18, r"$$\hat\lambda$$", r=0.36, size=20)
d.add_hyper(9.30, 1.05, r"$$\kappa,\beta$$", "lam")
d.add_text(7.05, 0.38, 305, 22, r"$$\mathbb{E}[\kappa(Y)\mathbf{1}\{Y\notin C_{\hat\lambda}\}]\leq\beta$$",
           color=SLATE, size=14, italic=True, align="left")

# Deployment.
d.add_node("x", 10.45, 4.95, r"$$x_t$$", r=0.36, shaded=True, size=19)
d.add_node("Ct", 11.90, 4.60, r"$$\tilde C_t$$", size=20)
d.add_node("Cop", 13.25, 4.60, r"$$C^{\oplus}_t$$", size=20)
d.add_node("M", 10.35, 2.05, r"$$M_t$$", r=0.38, size=20)
d.add_text(11.22, 5.38, 120, 20, r"$$[\tilde a_t,\tilde b_t]$$", color=SLATE, size=14, italic=True)
d.add_text(10.78, 3.55, 335, 22, r"$$\mathbb{P}(\tilde Y_t\in\tilde C_t)\geq 1-\alpha\;\;\forall(c,\gamma)$$",
           color=SLATE, size=14, italic=True, align="left")
d.add_text(12.05, 5.95, 360, 22, r"$$\mathbb{P}(Y_t\in C^{\oplus}_t)\geq 1-\alpha-\delta$$",
           color=SLATE, size=14, italic=True, align="left")
d.add_hyper(12.75, 5.35, r"$$(\mathcal{T},\delta)$$", "Cop")
d.add_hyper(9.75, 3.15, r"$$\varepsilon$$", "M")
d.add_text(9.82, 1.32, 240, 24, r"$$M_t=\prod_{s\leq t}\varepsilon\,p_s^{\varepsilon-1}$$",
           size=14, italic=True, align="left")

d.add_rect("Cert", 11.85, 2.73, 2.10, 0.86,
           r"$$\mathrm{Cert}(\gamma)$$<br>$$\left(1-\alpha,\hat\Delta_{\mathrm{emp}}^{\mathrm{LCB}},\cdot\right)$$",
           color=INK, rounded=True, size=16)

d.add_edge("q", "Ct")
d.add_edge("lam", "Ct")
d.add_edge("x", "Ct")
d.add_edge("Ct", "Cop")
d.add_edge("Cop", (13.25, 2.73))
d.add_edge("x", "M")
d.add_hyper(11.45, 1.42, r"$$\hat\Delta_{\mathrm{emp}}$$", "Cert", color=GOLD)

# Feedback arc and alarm label.
d.add_edge((9.95, 1.90), (3.32, 2.25), color=CORAL, dashed=True,
           curved=True, waypoints=[(7.7, 1.45), (5.4, 1.35)])
d.add_text(5.54, 1.42, 170, 22, r"$$M_t\geq 1/\alpha_{\mathrm{mon}}$$", color=CORAL, size=14, italic=True)

drawio_path = OUT / "fig1_choir_pipeline.drawio"
d.write(drawio_path)

# Basic draw.io file self-check.
tree = ET.parse(drawio_path)
root = tree.getroot()
cells = root.findall(".//mxCell")
vertices = [c for c in cells if c.get("vertex") == "1"]
edges = [c for c in cells if c.get("edge") == "1"]
labels = " ".join(c.get("value", "") for c in cells)
required = [
    r"\hat F", r"\hat c", r"g(x)", r"\hat w", r"\hat q_{c\gamma}",
    r"\tilde C_t", r"C^{\oplus}_t", r"M_t", r"\mathrm{Cert}",
]
required_ok = all(token in labels for token in required)
math_enabled = root.find(".//mxGraphModel").get("math") == "1"
valued_vertices = [c for c in vertices if c.get("value")]
latex_wrapped = all("$$" in c.get("value", "") for c in valued_vertices)
latex_blocks = sum(c.get("value", "").count("$$") // 2 for c in valued_vertices)
non_math = []
for c in valued_vertices:
    stripped = re.sub(r"\$\$.*?\$\$", " ", c.get("value", ""))
    stripped = re.sub(r"<br\s*/?>", " ", stripped)
    if re.search(r"[A-Za-z]{2,}", stripped):
        non_math.append(c.get("value", ""))

status = "PASS" if (
    len(vertices) >= 30 and len(edges) >= 22 and required_ok
    and math_enabled and latex_wrapped and latex_blocks >= 30 and not non_math
) else "FAIL"
(OUT / "floor_selfcheck_fig1_drawio.txt").write_text(
    "\n".join([
        f"DRAWIO SELF-CHECK: {status}",
        f"xml_well_formed: PASS",
        f"vertices_ge_30: {'PASS' if len(vertices) >= 30 else 'FAIL'} ({len(vertices)})",
        f"edges_ge_22: {'PASS' if len(edges) >= 22 else 'FAIL'} ({len(edges)})",
        f"required_notation_present: {'PASS' if required_ok else 'FAIL'}",
        f"drawio_math_enabled: {'PASS' if math_enabled else 'FAIL'}",
        f"valued_vertices_latex_wrapped: {'PASS' if latex_wrapped else 'FAIL'}",
        f"latex_blocks_ge_30: {'PASS' if latex_blocks >= 30 else 'FAIL'} ({latex_blocks})",
        f"no_text_outside_math_blocks: {'PASS' if not non_math else 'FAIL ' + str(non_math[:4])}",
    ]) + "\n",
    encoding="utf-8",
)
print((OUT / "floor_selfcheck_fig1_drawio.txt").read_text(encoding="utf-8"))
