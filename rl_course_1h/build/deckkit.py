"""Helpers for building the RL course deck with python-pptx (theme, text, shapes, grids, charts)."""
import re
import uuid

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (XL_AXIS_CROSSES, XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE,
                             XL_TICK_LABEL_POSITION)
from pptx.enum.dml import MSO_LINE, MSO_THEME_COLOR
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

W, H = 13.333, 7.5
COURSE = "Reinforcement Learning from Zero to Q-Learning"

# ---------------------------------------------------------------- palette
PALETTE = dict(ink="14202B", paper="FFFFFF", pine="0F2A24", mist="EEF3F1", green="0B6E4F",
               amber="F2A900", coral="D1495B", blue="2E86AB", plum="6C5B7B", slate="8D99A6")
THEME_ENUM = dict(ink=MSO_THEME_COLOR.TEXT_1, paper=MSO_THEME_COLOR.BACKGROUND_1,
                  pine=MSO_THEME_COLOR.TEXT_2, mist=MSO_THEME_COLOR.BACKGROUND_2,
                  green=MSO_THEME_COLOR.ACCENT_1, amber=MSO_THEME_COLOR.ACCENT_2,
                  coral=MSO_THEME_COLOR.ACCENT_3, blue=MSO_THEME_COLOR.ACCENT_4,
                  plum=MSO_THEME_COLOR.ACCENT_5, slate=MSO_THEME_COLOR.ACCENT_6)
# plain RGB extras (tints and a muted text colour)
EXTRA = dict(muted="55646F", mint="9ED9BC", green_t="E1F1E9", amber_t="FDF1CF", coral_t="F8E1E4",
             blue_t="DFEEF5", plum_t="E9E5EE", slate_t="ECEFF2", pine2="17403A", line="D5DCE2",
             amber_l="F7D488", code_dim="8FA8A0")


def setc(cf, key):
    """Apply a colour to a ColorFormat: theme name, extra/hex name, or 6-digit hex."""
    if key in THEME_ENUM:
        cf.theme_color = THEME_ENUM[key]
    else:
        cf.rgb = RGBColor.from_string(EXTRA.get(key, key))


def hexof(key):
    return PALETTE.get(key) or EXTRA.get(key) or key


# ---------------------------------------------------------------- presentation + theme
def new_presentation():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    _patch_theme(prs)
    _patch_master(prs)
    return prs


def _patch_theme(prs):
    part = prs.slide_master.part.part_related_by(RT.THEME)
    root = etree.fromstring(part.blob)
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    clr = root.find(".//a:clrScheme", ns)
    clr.set("name", "RL Course")
    mapping = dict(dk1="ink", lt1="paper", dk2="pine", lt2="mist", accent1="green", accent2="amber",
                   accent3="coral", accent4="blue", accent5="plum", accent6="slate",
                   hlink="blue", folHlink="plum")
    for tag, key in mapping.items():
        el = clr.find(f"a:{tag}", ns)
        for child in list(el):
            el.remove(child)
        etree.SubElement(el, qn("a:srgbClr")).set("val", PALETTE[key])
    fonts = root.find(".//a:fontScheme", ns)
    fonts.set("name", "RL Course")
    fonts.find("a:majorFont/a:latin", ns).set("typeface", "Cambria")
    fonts.find("a:minorFont/a:latin", ns).set("typeface", "Calibri")
    part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _patch_master(prs):
    ts = prs.slide_master.element.find(qn("p:txStyles")).find(qn("p:titleStyle"))
    lvl1 = ts.find(qn("a:lvl1pPr"))
    lvl1.set("algn", "l")
    d = lvl1.find(qn("a:defRPr"))
    d.set("sz", "3600")
    d.set("b", "1")
    layout = prs.slide_layouts[5]                      # "Title Only"
    t = layout.placeholders[0]
    t.left, t.top, t.width, t.height = Inches(1.4), Inches(0.4), Inches(11.35), Inches(0.8)


def add_sections(prs, sections):
    """sections: list of (name, [slide indices])  ->  PowerPoint sections (p14:sectionLst)."""
    P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
    pres = prs.part._element
    ids = [s.get("id") for s in pres.find(qn("p:sldIdLst"))]
    ext_lst = pres.find(qn("p:extLst"))
    if ext_lst is None:
        ext_lst = etree.SubElement(pres, qn("p:extLst"))
    ext = etree.SubElement(ext_lst, qn("p:ext"))
    ext.set("uri", "{521415D9-36F7-43E2-AB2F-B90AF26B5E84}")
    lst = etree.SubElement(ext, "{%s}sectionLst" % P14, nsmap={"p14": P14})
    for name, idxs in sections:
        sec = etree.SubElement(lst, "{%s}section" % P14)
        sec.set("name", name)
        sec.set("id", "{%s}" % str(uuid.uuid4()).upper())
        sl = etree.SubElement(sec, "{%s}sldIdLst" % P14)
        for i in idxs:
            etree.SubElement(sl, "{%s}sldId" % P14).set("id", ids[i])


# ---------------------------------------------------------------- text
TOKEN = re.compile(r"(\*\*|_\{[^}]*\}|\^\{[^}]*\}|\[\[[^\]]*\]\])")


def add_runs(p, text, size, color="ink", bold=False, italic=False, font=None):
    """Mini-markup:  _{sub}  ^{sup}  **bold toggle**  [[coloured bold text|colour]]"""
    bold_state = bold
    for tok in TOKEN.split(text):
        if not tok:
            continue
        if tok == "**":
            bold_state = not bold_state
            continue
        col, base, sub, sup, force_bold = color, tok, False, False, False
        if tok.startswith("_{"):
            base, sub = tok[2:-1], True
        elif tok.startswith("^{"):
            base, sup = tok[2:-1], True
        elif tok.startswith("[["):
            inner, col = tok[2:-2].rsplit("|", 1)
            add_runs(p, inner, size, col, True, italic, font)      # recurse so sub/sup work inside colour spans
            continue
        r = p.add_run()
        r.text = base
        f = r.font
        f.size = Pt(size)
        f.bold = bold_state or force_bold
        f.italic = italic
        if font:
            f.name = font
        setc(f.color, col)
        rPr = r._r.get_or_add_rPr()
        if sub:
            rPr.set("baseline", "-25000")
        if sup:
            rPr.set("baseline", "30000")


def _bullet(p, level_indent=0.22, char="•", color=None):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(level_indent))))
    pPr.set("indent", str(-int(Inches(level_indent))))
    for tag in ("a:buNone", "a:buChar", "a:buFont", "a:buClr"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    if color:
        bc = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(bc, qn("a:srgbClr")).set("val", hexof(color))
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(pPr, qn("a:buChar")).set("char", char)


def text(slide, x, y, w, h, content, size=16, color="ink", bold=False, italic=False, font=None,
         align="l", anchor="t", margin=0.0, bullets=False, space_after=0, line_spacing=None,
         name=None, parse=True, bullet_color="green"):
    """Add a text box. `content` is a string or a list of paragraphs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    paras = content if isinstance(content, list) else [content]
    for i, ptxt in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if space_after:
            p.space_after = Pt(space_after)
        if line_spacing:
            p.line_spacing = line_spacing
        if bullets:
            _bullet(p, color=bullet_color)
        if parse:
            add_runs(p, ptxt, size, color, bold, italic, font)
        else:
            r = p.add_run()
            r.text = ptxt
            r.font.size, r.font.bold, r.font.italic = Pt(size), bold, italic
            if font:
                r.font.name = font
            setc(r.font.color, color)
    return tb


def math(slide, x, y, w, h, content, size=24, color="ink", align="l", anchor="m", **kw):
    """Equation text in Cambria."""
    return text(slide, x, y, w, h, content, size=size, color=color, font="Cambria", align=align,
                anchor=anchor, **kw)


# ---------------------------------------------------------------- shapes
def box(slide, x, y, w, h, fill="mist", line=None, radius=0.06, shape=None, line_w=1.5, name=None):
    kind = shape or (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE)
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / max(min(w, h), 0.01))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        setc(s.fill.fore_color, fill)
    if line:
        setc(s.line.color, line)
        s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def label_in(shape, content, size=16, color="ink", bold=False, font=None, align="c", anchor="m"):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    paras = content if isinstance(content, list) else [content]
    for i, ptxt in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        add_runs(p, ptxt, size, color, bold, False, font)


def badge(slide, x, y, d, label, fill="green", color="paper", size=16, name=None):
    s = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=d * 0.22, name=name)
    label_in(s, label, size=size, color=color, bold=True)
    return s


def circle(slide, cx, cy, d, filled=False, color="ink", fill_color=None):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2), Inches(d), Inches(d))
    s.fill.solid()
    setc(s.fill.fore_color, (fill_color or color) if filled else "paper")
    setc(s.line.color, color)
    s.line.width = Pt(2)
    s.shadow.inherit = False
    return s


def line(slide, x1, y1, x2, y2, color="slate", w=2.0, arrow=False, dash=False, both=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    setc(c.line.color, color)
    c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    if dash:
        etree.SubElement(ln, qn("a:prstDash")).set("val", "dash")
    if both:
        he = etree.SubElement(ln, qn("a:headEnd"))
        he.set("type", "triangle"), he.set("w", "med"), he.set("len", "med")
    if arrow:
        te = etree.SubElement(ln, qn("a:tailEnd"))
        te.set("type", "triangle"), te.set("w", "med"), te.set("len", "med")
    return c


def alt(shape, description):
    shape._element.xpath(".//p:cNvPr")[0].set("descr", description)


def code_block(slide, x, y, w, h, lines, size=14, name="Code"):
    """Dark code panel; text after '#' is dimmed."""
    box(slide, x, y, w, h, fill="pine", radius=0.1, name=name + " panel")
    tb = slide.shapes.add_textbox(Inches(x + 0.22), Inches(y + 0.16), Inches(w - 0.44), Inches(h - 0.3))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        code, hash_, comment = ln.partition("#")
        for part, col in ((code, "paper"), (hash_ + comment, "code_dim")):
            if part:
                r = p.add_run()
                r.text = part
                r.font.name, r.font.size = "Courier New", Pt(size)
                setc(r.font.color, col)
    return tb


# ---------------------------------------------------------------- slide scaffolding
def content_slide(prs, title, module=None, notes="", number=None, chip_color="green"):
    s = prs.slides.add_slide(prs.slide_layouts[5])
    t = s.shapes.title
    t.text_frame.text = title
    if module:
        badge(s, 0.6, 0.5, 0.62, module, fill=chip_color, size=18, name="Module tile " + module,
              color="ink" if chip_color == "amber" else "paper")
    else:       # no module tile: title starts at the left margin (set all four dimensions)
        t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.4), Inches(12.1), Inches(0.8)
    footer(s, number)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def footer(slide, number=None, dark=False):
    col = "code_dim" if dark else "muted"
    text(slide, 0.6, 7.06, 8, 0.28, COURSE, size=10, color=col, name="Footer")
    tb = text(slide, 11.9, 7.06, 0.85, 0.28, "", size=10, color=col, align="r", name="Slide number")
    p = tb.text_frame.paragraphs[0]
    fld = etree.SubElement(p._p, qn("a:fld"))
    fld.set("id", "{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}")
    fld.set("type", "slidenum")
    rPr = etree.SubElement(fld, qn("a:rPr"))
    rPr.set("lang", "en-US"), rPr.set("sz", "1000")
    sf = etree.SubElement(rPr, qn("a:solidFill"))
    etree.SubElement(sf, qn("a:srgbClr")).set("val", hexof(col))
    etree.SubElement(fld, qn("a:t")).text = "‹#›"


def dark_background(slide, key="pine"):
    slide.background.fill.solid()
    setc(slide.background.fill.fore_color, key)


# ---------------------------------------------------------------- grid drawing
def heat(v, vmax):
    """value -> hex colour: soft coral (negative) .. white .. soft green (positive)."""
    neg, pos = (0xE8, 0x80, 0x8E), (0x4F, 0xB6, 0x8D)
    t = max(-1.0, min(1.0, v / vmax if vmax else 0.0))
    end = pos if t >= 0 else neg
    a = abs(t)
    return "".join(f"{int(255 + (e - 255) * a):02X}" for e in end)


def draw_grid(slide, env, x, y, cell, values=None, actions=None, path=None, vmax=None,
              fmt="{:.2f}", arrow_pt=None, value_pt=None, dark=False, show_start=True,
              random_policy=False, name="Gridworld", start_label=True):
    """Draw a GridWorld as native shapes. Returns nothing; every cell is its own rectangle."""
    from gridworld import ARROWS
    nonterm = [s for s in range(env.n_states) if not env.terminal[s]]
    if vmax is None and values is not None:
        vmax = max(abs(values[s]) for s in nonterm) or 1.0
    arrow_pt = arrow_pt or max(10, int(cell * 26))
    value_pt = value_pt or max(8, int(cell * 15))
    empty = "pine2" if dark else "mist"
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = env.pos2s.get((r, c))
            cx, cy = x + c * cell, y + r * cell
            if s is None:
                box(slide, cx, cy, cell, cell, fill="0A1A16" if dark else "ink", radius=0, line="pine" if dark else "paper",
                    line_w=1.25, name=f"{name} wall")
                continue
            kind = env.kind[s]
            if env.terminal[s]:
                fill = {"G": "amber", "P": "coral", "E": "amber_l"}[kind]
                sh = box(slide, cx, cy, cell, cell, fill=fill, radius=0, line="pine" if dark else "paper",
                         line_w=1.25, name=f"{name} {kind}")
                label_in(sh, kind, size=max(11, int(cell * 30)), bold=True,
                         color="paper" if kind == "P" else "ink")
                continue
            fill = heat(values[s], vmax) if values is not None else empty
            sh = box(slide, cx, cy, cell, cell, fill=fill, radius=0, line="pine" if dark else "paper",
                     line_w=1.25, name=f"{name} cell")
            paras = []                       # (text, point size)
            col = "mint" if dark else "ink"
            if random_policy:                # all four arrows: the uniform random policy
                paras += [("↑ ↓", max(10, arrow_pt // 2)), ("← →", max(10, arrow_pt // 2))]
            elif actions is not None:
                paras.append((ARROWS[int(actions[s])], arrow_pt))
            if values is not None:
                paras.append((fmt.format(values[s]), value_pt))
            if paras:
                tf = sh.text_frame
                tf.word_wrap = False
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                tf.vertical_anchor = MSO_ANCHOR.MIDDLE
                for i, (ptxt, pt_) in enumerate(paras):
                    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    p.alignment = PP_ALIGN.CENTER
                    rr = p.add_run()
                    rr.text = ptxt
                    rr.font.size = Pt(pt_)
                    setc(rr.font.color, col)
            if show_start and s == env.start and cell >= 0.45 and start_label:
                text(slide, cx + 0.03, cy + 0.01, 0.25, 0.2, "S", size=9, color="blue", bold=True, name="Start marker")
    if path:
        for s in path:
            r, c = env.cells[s]
            sh = box(slide, x + c * cell, y + r * cell, cell, cell, fill=None, radius=0, line="blue", line_w=3,
                     name=f"{name} path")


# ---------------------------------------------------------------- charts
def _style_axis(ax, title=None, size=12, grid=False, number_format=None):
    ax.tick_labels.font.size = Pt(size)
    setc(ax.tick_labels.font.color, "muted")
    setc(ax.format.line.color, "line")
    ax.has_major_gridlines = grid
    if grid:
        setc(ax.major_gridlines.format.line.color, "line")
        ax.major_gridlines.format.line.width = Pt(0.75)
    if number_format:
        ax.tick_labels.number_format = number_format
        ax.tick_labels.number_format_is_linked = False
    if title:
        ax.has_title = True
        ax.axis_title.text_frame.text = title
        r = ax.axis_title.text_frame.paragraphs[0].runs[0]
        r.font.size, r.font.bold = Pt(size), False
        setc(r.font.color, "muted")


def xy_chart(slide, x, y, w, h, series, xtitle=None, ytitle=None, xmin=None, xmax=None, ymin=None, ymax=None,
             legend=True, title=None, log_y=False, xmajor=None, ymajor=None, line_w=2.5, ynumfmt=None,
             xnumfmt=None, description=None, legend_size=12, name="Chart"):
    """series: list of (name, xs, ys, colour_key[, dash])."""
    cd = XyChartData()
    for item in series:
        s = cd.add_series(item[0])
        for xv, yv in zip(item[1], item[2]):
            s.add_data_point(float(xv), float(yv))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = name
    if description:
        alt(gf, description)
    ch = gf.chart
    ch.font.size = Pt(12)
    ch.font.name = "Calibri"
    ch.has_title = bool(title)
    if title:
        ch.chart_title.text_frame.text = title
        r = ch.chart_title.text_frame.paragraphs[0].runs[0]
        r.font.size, r.font.bold = Pt(14), True
        setc(r.font.color, "ink")
        ch.chart_title.include_in_layout = False
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(legend_size)
        setc(ch.legend.font.color, "ink")
    for ser, item in zip(ch.plots[0].series, series):
        ser.smooth = False
        ser.marker.style = XL_MARKER_STYLE.NONE
        setc(ser.format.line.color, item[3])
        ser.format.line.width = Pt(line_w)
        if len(item) > 4 and item[4]:
            ser.format.line.dash_style = MSO_LINE.DASH
    cat, val = ch.category_axis, ch.value_axis
    _style_axis(cat, xtitle, number_format=xnumfmt)
    _style_axis(val, ytitle, grid=True, number_format=ynumfmt)
    if xmin is not None:
        cat.minimum_scale = xmin
    if xmax is not None:
        cat.maximum_scale = xmax
    if ymin is not None:
        val.minimum_scale = ymin
    if ymax is not None:
        val.maximum_scale = ymax
    if xmajor:
        cat.major_unit = xmajor
    if ymajor:
        val.major_unit = ymajor
    if log_y:
        scaling = val._element.find(qn("c:scaling"))
        lb = etree.Element(qn("c:logBase"))
        lb.set("val", "10")
        scaling.insert(0, lb)
    # keep the x axis and its labels at the bottom even when y has negative values or a log scale
    cat.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    if ymin is not None:
        val.crosses = XL_AXIS_CROSSES.MINIMUM
    return gf


def bar_chart(slide, x, y, w, h, categories, values, colors=None, ytitle=None, xtitle=None, ymin=None, ymax=None,
              numfmt="0.0", description=None, name="Bar chart"):
    cd = CategoryChartData()
    cd.categories = categories
    cd.add_series("value", values)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = name
    if description:
        alt(gf, description)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.size = Pt(12)
    ch.font.name = "Calibri"
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = numfmt, False
    dl.font.size = Pt(11)
    setc(dl.font.color, "ink")
    ser = plot.series[0]
    ser.invert_if_negative = False
    for i in range(len(values)):
        pt = ser.points[i]
        pt.format.fill.solid()
        setc(pt.format.fill.fore_color, (colors[i] if colors else "slate"))
    for dpt in ser._element.findall(qn("c:dPt")):        # negative bars must keep their own colour, not invert to white
        if dpt.find(qn("c:invertIfNegative")) is None:
            inv = etree.Element(qn("c:invertIfNegative"))
            inv.set("val", "0")
            dpt.find(qn("c:idx")).addnext(inv)
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    _style_axis(ch.category_axis, xtitle)
    _style_axis(ch.value_axis, ytitle, grid=True)
    if ymin is not None:
        ch.value_axis.minimum_scale = ymin
    if ymax is not None:
        ch.value_axis.maximum_scale = ymax
    return gf
