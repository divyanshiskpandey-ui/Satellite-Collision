"""Design system for the dashboard: palette, typography, page CSS, chart theme and small HTML helpers.

A dark mission-operations look: Inter for interface text, JetBrains Mono for telemetry readouts,
hairline-bordered panels and uppercase section labels. app.py and orbit_view.py both use it, so every
panel, chart and control matches.
"""
import html

import altair as alt

# ---------------------------------------------------------------- palette
PAGE = "#070b13"          # page background
SURFACE = "#0d1320"       # panels and chart backgrounds
SURFACE_2 = "#121a2a"     # raised elements inside panels (tiles, buttons)
HAIRLINE = "#1c2638"
HAIRLINE_2 = "#2a3954"
INK = "#e6ebf3"           # primary text
INK_2 = "#a4afc2"         # secondary text
MUTED = "#6d7891"         # captions, axis titles
ACCENT = "#4da3ff"        # interactive elements
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"
MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, monospace"

# status colours carry the tier (validated set, same in both themes); every use is paired with a label
TIER_COL = {"critical": "#d03b3b", "elevated": "#fab219", "safe": "#0ca30c"}
ICON = {"critical": "▲", "elevated": "●", "safe": "○"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ---------- page */
/* text elements only: a wildcard would also override Streamlit's icon font and show icon names as text */
html, body, .stApp, .stMarkdown, .stText, p, label, li, input, textarea, select, button {{ font-family: {FONT}; }}
[data-testid="stIconMaterial"], span[class*="material-symbols"] {{ font-family: "Material Symbols Rounded" !important; }}
.stApp {{ background: radial-gradient(1200px 600px at 70% -10%, #0f1a2e 0%, {PAGE} 55%) fixed; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.stAppDeployButton, [data-testid="stStatusWidget"] {{ display: none !important; }}
footer {{ visibility: hidden; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 1.4rem; max-width: 1560px; }}
h1, h2, h3 {{ font-family: {FONT}; letter-spacing: -.01em; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {MUTED} !important; font-size: .78rem; line-height: 1.5; }}
hr {{ border-color: {HAIRLINE} !important; }}
.block-container [data-testid="stMarkdownContainer"] > p,
.block-container [data-testid="stMarkdownContainer"] li {{ font-size: .9rem; line-height: 1.65; color: {INK_2}; }}
.block-container [data-testid="stMarkdownContainer"] strong {{ color: {INK}; }}

/* ---------- sidebar */
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {HAIRLINE}; }}
section[data-testid="stSidebar"] label p {{ font-size: .78rem; color: {INK_2}; }}

/* ---------- tabs */
[data-testid="stTab"] p {{ font-size: .76rem !important; font-weight: 600 !important; letter-spacing: .14em;
                           text-transform: uppercase; color: {MUTED}; }}
[data-testid="stTab"][data-selected="true"] p, [data-testid="stTab"][aria-selected="true"] p {{ color: {INK}; }}

/* ---------- panels: containers created with key="panel_..." */
[class*="st-key-panel"] {{ background: {SURFACE}; border: 1px solid {HAIRLINE} !important; border-radius: 12px !important; }}

/* ---------- buttons */
.stButton > button, .stDownloadButton > button {{
    background: {SURFACE_2}; color: {INK}; border: 1px solid {HAIRLINE_2}; border-radius: 8px;
    font-size: .74rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; padding: .5rem 1rem;
    transition: border-color .15s, box-shadow .15s, background .15s; }}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {ACCENT}; color: #fff; background: #16223a; box-shadow: 0 0 0 3px rgba(77,163,255,.15); }}
.stButton > button[kind="primary"] {{ background: {ACCENT}; border-color: {ACCENT}; color: #04101f; }}

/* ---------- inputs */
[data-baseweb="select"] > div {{ background: {SURFACE_2}; border-color: {HAIRLINE_2}; border-radius: 8px; }}
[data-testid="stWidgetLabel"] p {{ font-size: .72rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
                                    color: {MUTED}; }}
[data-testid="stCheckbox"] p, [data-testid="stRadio"] [role="radiogroup"] p {{
    text-transform: none !important; letter-spacing: 0 !important; font-size: .84rem !important;
    font-weight: 500 !important; color: {INK_2} !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid {HAIRLINE}; border-radius: 10px; overflow: hidden; }}
div[role="dialog"] {{ background: {SURFACE}; border: 1px solid {HAIRLINE_2}; border-radius: 14px; }}

/* ---------- Plotly controls: Play / Pause buttons, sliders, mode bar */
.js-plotly-plot .updatemenu-item-rect {{ fill: {SURFACE_2} !important; stroke: {HAIRLINE_2} !important;
                                         stroke-width: 1px !important; rx: 7px; ry: 7px; }}
.js-plotly-plot .updatemenu-button:hover .updatemenu-item-rect {{ fill: #16223a !important; stroke: {ACCENT} !important;
                                                                  filter: drop-shadow(0 0 4px rgba(77,163,255,.45)); }}
.js-plotly-plot .updatemenu-item-text {{ font-family: {FONT} !important; font-weight: 600 !important;
                                         letter-spacing: .12em; fill: {INK} !important; }}
.js-plotly-plot .slider-rail-rect {{ fill: {HAIRLINE_2} !important; stroke: none !important; rx: 2px; ry: 2px; }}
.js-plotly-plot .slider-grip-rect {{ fill: {ACCENT} !important; stroke: #dbeaff !important; stroke-width: 1px !important;
                                     rx: 5px; ry: 5px; filter: drop-shadow(0 0 5px rgba(77,163,255,.6)); }}
.js-plotly-plot .slider-group text {{ font-family: {MONO} !important; }}
.js-plotly-plot .modebar {{ background: transparent !important; }}
.js-plotly-plot .modebar-btn path {{ fill: {MUTED} !important; }}
.js-plotly-plot .modebar-btn:hover path {{ fill: {INK} !important; }}

/* ---------- header bar */
.mx-header {{ display: flex; justify-content: space-between; align-items: center; gap: 24px; flex-wrap: wrap;
              padding: 18px 22px; margin-bottom: 14px; background: linear-gradient(180deg, #0f1726, {SURFACE});
              border: 1px solid {HAIRLINE}; border-radius: 14px; position: relative; overflow: hidden; }}
.mx-header::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
                      background: linear-gradient(180deg, {ACCENT}, #2a5fa8); }}
.mx-brand {{ display: flex; align-items: center; gap: 16px; }}
.mx-eyebrow {{ font-size: .68rem; font-weight: 600; letter-spacing: .2em; text-transform: uppercase; color: {ACCENT}; }}
.mx-title {{ font-size: 1.65rem; font-weight: 700; color: {INK}; letter-spacing: -.015em; line-height: 1.2; }}
.mx-desc {{ font-size: .8rem; color: {INK_2}; margin-top: 3px; }}
.mx-meta {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.mx-chip {{ background: {SURFACE_2}; border: 1px solid {HAIRLINE}; border-radius: 9px; padding: 7px 12px; min-width: 120px; }}
.mx-chip span {{ display: block; font-size: .62rem; font-weight: 600; letter-spacing: .16em; color: {MUTED}; }}
.mx-chip b {{ font-family: {MONO}; font-size: .82rem; font-weight: 500; color: {INK}; }}
.mx-legend {{ display: flex; gap: 22px; flex-wrap: wrap; font-size: .76rem; color: {INK_2}; margin: 0 2px 18px; }}
.mx-legend b {{ font-weight: 600; letter-spacing: .08em; text-transform: uppercase; font-size: .7rem; }}

/* ---------- section headers */
.mx-section {{ display: flex; align-items: baseline; gap: 12px; margin: 30px 0 12px; padding-bottom: 8px;
               border-bottom: 1px solid {HAIRLINE}; }}
.mx-num {{ font-family: {MONO}; font-size: .74rem; color: {ACCENT}; font-weight: 600; }}
.mx-stitle {{ font-size: .86rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: {INK}; }}
.mx-ssub {{ font-size: .78rem; color: {MUTED}; margin-left: auto; }}
.mx-label {{ font-size: .7rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: {MUTED};
             margin: 4px 0 6px; }}

/* ---------- KPI tiles */
.mx-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 4px 0 16px; }}
.mx-kpi {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 12px; padding: 14px 16px 15px;
           position: relative; overflow: hidden; }}
.mx-kpi::before {{ content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: var(--k, {HAIRLINE_2}); }}
.mx-kpi span {{ display: block; font-size: .64rem; font-weight: 600; letter-spacing: .15em; text-transform: uppercase;
                color: {MUTED}; }}
.mx-kpi b {{ display: block; font-family: {MONO}; font-size: clamp(1.15rem, 1.45vw, 1.7rem); font-weight: 600;
             color: {INK}; margin-top: 6px; letter-spacing: -.02em; white-space: nowrap; }}
.mx-kpi i {{ font-style: normal; font-size: .74rem; color: {INK_2}; }}
.mx-kpi .ico {{ color: var(--k); }}

/* ---------- focus-event card */
.hero {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 14px; padding: 18px 20px; margin-bottom: 12px;
         position: relative; overflow: hidden; }}
.hero::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--tier); }}
.hero .lbl {{ color: var(--tier); font-weight: 700; letter-spacing: .14em; text-transform: uppercase; font-size: .72rem; }}
.hero .clock {{ font-family: {MONO}; color: {INK}; font-weight: 600; font-size: 2.6rem; line-height: 1.1; margin-top: 6px;
                letter-spacing: -.03em; }}
.hero .sub {{ color: {INK_2}; font-size: .8rem; margin-top: 4px; }}
.hero-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px; }}
.hero-grid div {{ background: {SURFACE_2}; border: 1px solid {HAIRLINE}; border-radius: 9px; padding: 9px 11px; }}
.hero-grid b {{ display: block; font-family: {MONO}; color: {INK}; font-size: .95rem; font-weight: 500; }}
.hero-grid span {{ color: {MUTED}; font-size: .7rem; letter-spacing: .02em; }}
.ico {{ color: var(--tier); }}
.mx-probs {{ display: flex; gap: 16px; font-size: .76rem; color: {INK_2}; margin-top: -4px; }}
.mx-probs b {{ font-family: {MONO}; color: {INK}; font-weight: 600; }}

/* ---------- orbit_view panels (stat strips, status card, response plan) */
.ov, .ov * {{ font-family: {FONT}; box-sizing: border-box; }}
.ov-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin: 6px 0 14px; }}
.ov-stat {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 11px; padding: 12px 14px; }}
.ov-stat b {{ display: block; font-family: {MONO}; font-size: 1.2rem; color: {INK}; font-weight: 600; letter-spacing: -.02em; }}
.ov-stat span {{ font-size: .72rem; color: {MUTED}; line-height: 1.4; }}
.ov-status {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-left: 4px solid var(--tier);
              border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; }}
.ov-status .tier {{ color: var(--tier); font-weight: 700; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; }}
.ov-status .count {{ font-family: {MONO}; color: {INK}; font-weight: 600; font-size: 1.9rem; line-height: 1.15; }}
.ov-status .sub {{ color: {INK_2}; font-size: .8rem; margin-top: 2px; }}
.ov-step {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 11px; padding: 11px 14px; margin-bottom: 8px; }}
.ov-step .t {{ color: {INK}; font-weight: 600; font-size: .9rem; }}
.ov-step .w {{ font-family: {MONO}; color: var(--tier); font-size: .72rem; font-weight: 500; margin-left: 8px; }}
.ov-step .d {{ color: {INK_2}; font-size: .8rem; line-height: 1.5; margin-top: 3px; }}
.ov-step.done {{ opacity: .45; }}
.ov-step.done .t {{ text-decoration: line-through; }}
</style>
"""

# an original emblem (Earth, orbit ring, satellite) — deliberately not any agency's insignia
EMBLEM = f"""
<svg width="46" height="46" viewBox="0 0 46 46" fill="none" aria-hidden="true">
  <circle cx="23" cy="23" r="22" stroke="{HAIRLINE_2}" />
  <circle cx="23" cy="23" r="9" fill="#123a6b" stroke="{ACCENT}" stroke-width="1.2"/>
  <ellipse cx="23" cy="23" rx="19" ry="7.5" transform="rotate(-24 23 23)" stroke="{ACCENT}" stroke-width="1.2" opacity=".85"/>
  <circle cx="38.6" cy="15.6" r="2.6" fill="{INK}"/>
  <circle cx="9.5" cy="30.5" r="1.8" fill="{TIER_COL['critical']}"/>
</svg>
"""


def header(title, eyebrow, desc, chips):
    """Top bar: emblem, eyebrow, title, one-line description and monospace status chips [(label, value)]."""
    chip_html = "".join(f'<div class="mx-chip"><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>'
                        for k, v in chips)
    return (f'<div class="mx-header"><div class="mx-brand">{EMBLEM}<div>'
            f'<div class="mx-eyebrow">{eyebrow}</div><div class="mx-title">{title}</div>'
            f'<div class="mx-desc">{desc}</div></div></div><div class="mx-meta">{chip_html}</div></div>')


def tier_legend():
    items = [("critical", "Pc ≥ 1e-4 or miss < 1 km · manoeuvre planning"),
             ("elevated", "Pc ≥ 1e-6 or miss 1–5 km · monitor"),
             ("safe", "Pc < 1e-6 and miss > 5 km · routine")]
    return '<div class="mx-legend">' + "".join(
        f'<div><b style="color:{TIER_COL[t]}">{ICON[t]} {t}</b>&nbsp; {d}</div>' for t, d in items) + "</div>"


def section(num, title, sub=""):
    return (f'<div class="mx-section"><span class="mx-num">{num}</span><span class="mx-stitle">{title}</span>'
            f'<span class="mx-ssub">{sub}</span></div>')


def label(text):
    return f'<div class="mx-label">{text}</div>'


def kpis(items):
    """Status tiles [(label, value, note, tier-or-None)]; a tier colours the top edge and adds its icon."""
    out = []
    for lab, val, note, tier in items:
        k = f' style="--k:{TIER_COL[tier]}"' if tier else ""
        ico = f'<span class="ico" style="display:inline;color:{TIER_COL[tier]}">{ICON[tier]}</span> ' if tier else ""
        out.append(f'<div class="mx-kpi"{k}><span>{ico}{lab}</span><b>{val}</b><i>{note}</i></div>')
    return '<div class="mx-kpis">' + "".join(out) + "</div>"


# ---------------------------------------------------------------- Altair theme
@alt.theme.register("mission", enable=True)
def _mission_theme():
    axis = dict(labelColor=INK_2, titleColor=MUTED, gridColor=HAIRLINE, domainColor=HAIRLINE_2, tickColor=HAIRLINE_2,
                labelFont=FONT, titleFont=FONT, labelFontSize=11, titleFontSize=11, titleFontWeight=500,
                titlePadding=10, labelPadding=6)
    return alt.theme.ThemeConfig({"config": {
        "background": "transparent", "font": FONT,
        "view": {"stroke": None},
        "axis": axis,
        "legend": dict(labelColor=INK_2, titleColor=MUTED, labelFont=FONT, titleFont=FONT, labelFontSize=11,
                       titleFontSize=11, titleFontWeight=500, symbolStrokeWidth=0),
        "text": dict(font=FONT, color=INK_2),
        "title": dict(font=FONT, color=INK, fontSize=13, fontWeight=600, anchor="start"),
    }})
