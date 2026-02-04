# ui_tags.py
from __future__ import annotations
from typing import Dict, Iterable, List, Optional
import colorsys

# Color‑blind friendly, high‑contrast palette (10)
PALETTE_10 = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
]

def _distinct_palette(n: int, s: float = 0.65, v: float = 0.75) -> List[str]:
    """HSV palette retained for legacy use."""
    if n <= 0:
        return []
    hues = [(i / n) % 1.0 for i in range(n)]
    colors = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append("#{0:02x}{1:02x}{2:02x}".format(int(r*255), int(g*255), int(b*255)))
    return colors

def _css_escape(s: str) -> str:
    return s.replace('"', '\\"')

def build_color_map(
    labels: Iterable[str],
    preferred: Optional[Dict[str, str]] = None,
    *,
    max_colors: int = 10,
    palette: Optional[List[str]] = None,
    cycle: bool = True,
) -> Dict[str, str]:
    """
    Build label->color map.
      - 'preferred' entries win.
      - remaining labels map to a fixed small palette.
      - if there are more labels than 'max_colors':
          * cycle through the palette if 'cycle' is True, or
          * assign a neutral gray.
    """
    labels = [str(x) for x in labels if x is not None]
    labels = list(dict.fromkeys(labels))  # stable de-duplication

    mapping: Dict[str, str] = {}
    if preferred:
        for k, v in preferred.items():
            mapping[str(k)] = str(v)

    # choose palette
    base = list(PALETTE_10 if palette is None else palette)
    if max_colors and len(base) > max_colors:
        base = base[:max_colors]
    if max_colors and len(base) < max_colors:
        # if user passed a shorter palette, repeat it to reach max_colors
        reps = (max_colors + len(base) - 1) // len(base)
        base = (base * reps)[:max_colors]

    # assign colors to remaining labels
    NEUTRAL = "#A0AEC0"  # gray 400
    missing = [lab for lab in labels if lab not in mapping]
    for i, lab in enumerate(missing):
        if len(base) == 0:
            mapping[lab] = NEUTRAL
        elif i < len(base):
            mapping[lab] = base[i]
        else:
            mapping[lab] = base[i % len(base)] if cycle else NEUTRAL

    return mapping

def build_multiselect_tag_css(mapping: Dict[str, str]) -> str:
    base = """
/* Base tag styling */
.stMultiSelect div[data-baseweb="tag"] {
    background-color: #4a5568 !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    padding: 2px 8px !important;
    margin: 2px !important;
    font-size: 13px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
}
.stMultiSelect div[data-baseweb="tag"] svg {
    fill: #ffffff !important;
}
"""
    rules = []
    for lab, color in mapping.items():
        safe = _css_escape(lab)
        rules.append(f"""
/* Tag for: {safe} */
.stMultiSelect div[data-baseweb="tag"]:has(div:contains("{safe}")) {{
    background-color: {color} !important;
}}
""")
    return "<style>\n" + base + "".join(rules) + "\n</style>"

def inject_multiselect_tag_css(st, mapping: Dict[str, str]) -> None:
    css = build_multiselect_tag_css(mapping)
    st.markdown(css, unsafe_allow_html=True)