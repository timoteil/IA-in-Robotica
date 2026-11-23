"""Rough layout check for the deck: estimates whether text fits its box, using Arial / Georgia / Courier metrics
as stand-ins for Calibri / Cambria / Courier New. Meant as a smoke test, not a renderer.

    python build/check_layout.py RL_Bridge_DQN_DDPG.pptx
"""
import sys

from PIL import ImageFont
from pptx import Presentation

SUP = "/System/Library/Fonts/Supplemental/"
FILES = {"arial": SUP + "Arial.ttf", "georgia": SUP + "Georgia.ttf", "courier": SUP + "Courier New.ttf"}
SCALE = {"arial": 0.92, "georgia": 0.93, "courier": 1.0}      # Calibri is narrower than Arial, Cambria than Georgia
cache = {}


def font(kind, size):
    k = (kind, round(size * 4))
    if k not in cache:
        cache[k] = ImageFont.truetype(FILES[kind], int(round(size * 10)))
    return cache[k]


def width_in(txt, kind, size):
    return font(kind, size).getlength(txt) / 10.0 / 72.0 * SCALE[kind]


def kind_of(name):
    if name and "Courier" in name:
        return "courier"
    if name and "Cambria" in name:
        return "georgia"
    return "arial"


def para_height(p, width, default=14, title=False):
    runs = list(p.runs)
    if not runs:
        return default * 1.2 / 72
    size = max((r.font.size.pt if r.font.size else default) for r in runs)
    kind = "georgia" if title else kind_of(runs[0].font.name)
    bold = True if title else any(r.font.bold for r in runs)
    txt = "".join(r.text for r in runs)
    indent = 0.22 if p._p.pPr is not None and p._p.pPr.get("marL") else 0
    avail = width - indent
    lines = 0
    for seg in txt.split("\n"):
        cur, n = "", 1
        for w in seg.split(" "):
            t = (cur + " " + w).strip()
            if width_in(t, kind, size) * (1.06 if bold else 1) > avail and cur:
                n, cur = n + 1, w
            else:
                cur = t
        lines += n
    sa = p.space_after.pt if p.space_after is not None else 0
    return lines * size * 1.2 / 72 + sa / 72


def main(path):
    prs = Presentation(path)
    bad = 0
    for i, sl in enumerate(prs.slides, 1):
        for sh in sl.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                tf = sh.text_frame
                w = sh.width / 914400 - (tf.margin_left + tf.margin_right) / 914400
                is_title = sh.is_placeholder
                need = sum(para_height(p, w, 36 if is_title else 14, is_title) for p in tf.paragraphs) + (tf.margin_top + tf.margin_bottom) / 914400
                have = sh.height / 914400
                if need > have + 0.08:
                    bad += 1
                    print(f"slide {i:2d} OVERFLOW {sh.name!r}: need {need:.2f} in, have {have:.2f} in  {tf.text[:50]!r}")
            if sh.has_table:
                tbl, total = sh.table, 0
                for r in tbl.rows:
                    rh, mx = r.height / 914400, 0
                    for ci, c in enumerate(r.cells):
                        w = tbl.columns[ci].width / 914400 - 0.2
                        mx = max(mx, sum(para_height(p, w) for p in c.text_frame.paragraphs) + 0.08)
                    total += max(rh, mx)
                    if mx > rh + 0.05:
                        print(f"slide {i:2d} table row grows: {mx:.2f} > {rh:.2f}  {r.cells[0].text_frame.text[:30]!r}")
                print(f"slide {i:2d} table {sh.name!r}: bottom at {sh.top / 914400 + total:.2f} in")
        for sh in sl.shapes:
            if (sh.top + sh.height) / 914400 > 7.0 and sh.name not in ("Footer", "Slide number") and not sh.has_table:
                print(f"slide {i:2d} low shape {sh.name!r}: bottom {(sh.top + sh.height) / 914400:.2f} in")
    print("text overflows:", bad)


if __name__ == "__main__":
    main(sys.argv[1])
