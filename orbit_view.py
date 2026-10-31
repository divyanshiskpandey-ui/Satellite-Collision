"""Interactive 3D orbit view and response plan, driven by the columns already in the data.

Drop this file next to app.py. Needs: pip install plotly

Columns used (ESA collision-avoidance / CDM naming, taken from the latest update of the event):
  required: relative_position_r/t/n [m], relative_velocity_r/t/n [m/s], t_j2k_sma [km]
  optional: t_sigma_r/t/n and c_sigma_r/t/n [m]  (uncertainty ellipsoid),
            t_j2k_ecc, c_j2k_sma, relative_speed  (consistency checks)
  always:   time_to_tca, miss_distance, pred_risk, shown_tier, rule_flip

How the geometry is built (no made-up orbit):
  * The frame is the target's radial-transverse-normal (RTN) frame at closest approach: Earth's centre is
    the origin, the target sits on +x at radius = its semi-major axis, and its orbit plane is x-y.
  * The debris starts from target state + relative position / velocity, then is propagated both ways with
    two-body dynamics. Its orbit, the angle between the two orbital planes and the approach path all come
    from the data. The target is treated as circular at its catalogued semi-major axis.
  * Earth's rotation is ignored and its surface texture is decorative.
"""
import functools
import math
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import RISK_FLOOR

EARTH_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "blue_marble_2048.jpg")
MU = 398600.4418          # km^3 / s^2
R_E = 6371.0              # km
BG = "#05080f"
INK, MUTED, PANEL, LINE = "#eaf1ff", "#8393b3", "#0e1626", "#1d2a44"
COL_A, COL_B = "#5ad1ff", "#ff8fd0"
TIER_COL = {"critical": "#d03b3b", "elevated": "#fab219", "safe": "#0ca30c"}   # same status colours as app.py
# Earth surface palette: deep ocean -> shallow water -> beach -> grass -> forest -> rock -> ice
EARTH_SCALE = [[0.0, "#031026"], [0.2, "#07295c"], [0.38, "#0d4f9c"], [0.485, "#38a6dd"],
               [0.505, "#d8c58f"], [0.545, "#7fae5a"], [0.65, "#3f8a45"], [0.78, "#2c6a37"],
               [0.88, "#8b7a55"], [0.94, "#cfcabb"], [1.0, "#ffffff"]]

REQUIRED = ["relative_position_r", "relative_position_t", "relative_position_n",
            "relative_velocity_r", "relative_velocity_t", "relative_velocity_n", "t_j2k_sma"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
.ov, .ov * { font-family: 'Space Grotesk', system-ui, sans-serif; box-sizing: border-box; }
.ov-strip { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:6px 0 14px; }
.ov-stat { background:#0e1626; border:1px solid #1d2a44; border-radius:10px; padding:12px 14px; }
.ov-stat b { display:block; font-size:1.3rem; color:#eaf1ff; font-weight:700; font-variant-numeric:tabular-nums; }
.ov-stat span { font-size:.8rem; color:#8393b3; }
.ov-status { background:#0e1626; border:1px solid #1d2a44; border-left:5px solid var(--tier);
             border-radius:10px; padding:14px 16px; margin-bottom:12px; }
.ov-status .tier { color:var(--tier); font-weight:700; font-size:.95rem; }
.ov-status .count { color:#eaf1ff; font-weight:700; font-size:2rem; line-height:1.15; font-variant-numeric:tabular-nums; }
.ov-status .sub { color:#8393b3; font-size:.85rem; margin-top:2px; }
.ov-step { background:#0e1626; border:1px solid #1d2a44; border-radius:10px; padding:10px 14px; margin-bottom:8px; }
.ov-step .t { color:#eaf1ff; font-weight:700; font-size:.98rem; }
.ov-step .w { color:var(--tier); font-size:.8rem; font-weight:500; margin-left:6px; }
.ov-step .d { color:#aab8d6; font-size:.86rem; line-height:1.45; margin-top:3px; }
.ov-step.done { opacity:.45; }
.ov-step.done .t { text-decoration:line-through; }
</style>
"""


# ---------------------------------------------------------------- helpers
def _num(row, name):
    try:
        v = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _fmt_dur(hours):
    m = max(0, int(round(hours * 60)))
    d, rem = divmod(m, 1440)
    h, mi = divmod(rem, 60)
    return f"{d}d {h}h {mi:02d}m" if d else f"{h}h {mi:02d}m"


def _fmt_t(seconds_before):
    m = int(round(seconds_before / 60))
    d, rem = divmod(m, 1440)
    h, mi = divmod(rem, 60)
    return f"T−{d}d {h:02d}h {mi:02d}m" if d else f"T−{h:02d}h {mi:02d}m"


def pc_text(log10_pc):
    """Collision probability as a percentage, from log10 Pc. For example -4 gives '0.0100%'.
    Below Pc 1e-10 (the floor used for modelling; the data goes down to 1e-30) it reads 'below 1e-8%'
    (words, not '<', because the text also goes into HTML cards and Plotly hover labels)."""
    if float(log10_pc) < RISK_FLOOR:
        return f"below {100 * 10.0 ** RISK_FLOOR:.0e}%".replace("e-0", "e-")
    pct = 100.0 * 10.0 ** min(float(log10_pc), 0.0)
    if pct >= 10:
        return f"{pct:.1f}%"
    if pct >= 0.1:
        return f"{pct:.2f}%"
    if pct >= 0.001:
        return f"{pct:.4f}%"
    return f"{pct:.1e}%"


def pc_odds(log10_pc):
    """The same probability as odds, for example 'about 1 in 10,000' or 'about 1 in 3.2 billion'."""
    if float(log10_pc) < RISK_FLOOR:
        return "less than 1 in 10 billion"
    n = 10.0 ** (-min(float(log10_pc), 0.0))
    if n >= 1e9:
        return f"about 1 in {n / 1e9:.1f} billion"
    if n >= 1e6:
        return f"about 1 in {n / 1e6:.1f} million"
    return f"about 1 in {n:,.0f}" if n >= 2 else "about 1 in 1"


def _acc(r):
    return -MU * r / np.linalg.norm(r) ** 3


def _rk4(r, v, h):
    k1r, k1v = v, _acc(r)
    k2r, k2v = v + 0.5 * h * k1v, _acc(r + 0.5 * h * k1r)
    k3r, k3v = v + 0.5 * h * k2v, _acc(r + 0.5 * h * k2r)
    k4r, k4v = v + h * k3v, _acc(r + h * k3r)
    return (r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r),
            v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))


@st.cache_data(show_spinner=False)
def _propagate(r0, v0, t_min, t_max, dt=30.0):
    """Two-body propagation from the closest-approach state, backward to t_min and forward to t_max."""
    r0, v0 = np.array(r0), np.array(v0)

    def run(sign, span):
        out, r, v = [r0], r0, v0
        for _ in range(int(math.ceil(max(0.0, span) / dt))):
            r, v = _rk4(r, v, sign * dt)
            out.append(r)
        return np.array(out)

    back, fwd = run(-1, -t_min), run(1, t_max)
    taus = np.concatenate([-dt * np.arange(len(back))[::-1], dt * np.arange(1, len(fwd))])
    return taus, np.vstack([back[::-1], fwd[1:]])


def _scene(rel_pos_m, rel_vel_ms, a_t):
    """Target on +x on a circular orbit in the x-y plane; debris from the relative state."""
    n_t = math.sqrt(MU / a_t ** 3)
    rt, vt = np.array([a_t, 0.0, 0.0]), np.array([0.0, a_t * n_t, 0.0])
    rc, vc = rt + np.array(rel_pos_m) / 1000.0, vt + np.array(rel_vel_ms) / 1000.0
    r, v = np.linalg.norm(rc), np.linalg.norm(vc)
    inv_a = 2 / r - v ** 2 / MU
    if inv_a <= 0:
        return None
    a_c = 1 / inv_a
    e_vec = ((v ** 2 - MU / r) * rc - np.dot(rc, vc) * vc) / MU
    ecc_c = float(np.linalg.norm(e_vec))
    h_t, h_c = np.cross(rt, vt), np.cross(rc, vc)
    plane = math.degrees(math.acos(np.clip(np.dot(h_t, h_c) / np.linalg.norm(h_t) / np.linalg.norm(h_c), -1, 1)))
    b = a_c * math.sqrt(1 - min(ecc_c, 0.99) ** 2)
    circ_c = math.pi * (3 * (a_c + b) - math.sqrt((3 * a_c + b) * (a_c + 3 * b)))   # Ramanujan
    return dict(a_t=a_t, n_t=n_t, rt=rt, vt=vt, rc=rc, vc=vc, a_c=a_c, ecc_c=ecc_c, plane=plane,
                T_t=2 * math.pi / n_t, T_c=2 * math.pi * math.sqrt(a_c ** 3 / MU), circ_c=circ_c,
                circ_t=2 * math.pi * a_t, rel_speed=float(np.linalg.norm(rel_vel_ms)) / 1000.0)


def _earth_surface(nu=180, nv=90):
    """Decorative Earth: continent-like landmasses, shallow and deep ocean, ragged polar ice.

    Returns x, y, z and a colour value in [-1, 1] (ocean < 0 < land, ice near 1) for EARTH_SCALE.
    Not a real map. The longitude frequencies are whole numbers so there is no seam."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, nu), np.linspace(0, np.pi, nv))
    x, y, z = R_E * np.cos(u) * np.sin(v), R_E * np.sin(u) * np.sin(v), R_E * np.cos(v)
    lat = np.pi / 2 - v
    pts = np.stack([np.cos(lat) * np.cos(u), np.cos(lat) * np.sin(u), np.sin(lat)], axis=-1)

    # (longitude deg, latitude deg, strength, angular radius deg): rough continent positions
    blobs = [(-100, 48, 1.0, 24), (-105, 62, 0.8, 20), (-60, -12, 1.0, 20), (-68, -38, 0.7, 11),
             (18, 8, 1.0, 26), (20, -18, 0.8, 16), (15, 50, 0.9, 16), (70, 55, 1.0, 34),
             (100, 30, 0.9, 24), (78, 22, 0.8, 11), (135, -25, 1.0, 14), (-42, 72, 0.8, 11),
             (110, -2, 0.6, 7), (140, 38, 0.6, 6)]
    elev = np.full(u.shape, -0.6)
    for lon, la, k, r in blobs:
        lon, la = np.radians(lon), np.radians(la)
        centre = np.array([np.cos(la) * np.cos(lon), np.cos(la) * np.sin(lon), np.sin(la)])
        ang = np.degrees(np.arccos(np.clip(pts @ centre, -1, 1)))
        elev += 1.2 * k * np.exp(-(ang / r) ** 2)

    # ragged coastlines and ice edge; sin(v) keeps the poles smooth
    noise = (0.10 * np.sin(6 * u + 2.0 * v) + 0.07 * np.sin(11 * u - 3.0 * v + 1.0)
             + 0.05 * np.sin(17 * u + 5.0 * v)) * np.sin(v)
    elev += noise

    c = np.where(elev > 0, 0.06 + 0.84 * np.tanh(1.6 * elev), -np.tanh(2.2 * -elev))
    ice = np.clip((np.abs(np.degrees(lat)) + 30 * noise - 68) / 6, 0, 1)
    c = np.where(ice > 0, np.maximum(c, 0.88 + 0.12 * ice), c)
    return x, y, z, c


def _sphere(r, n=48):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2))
    return r * np.cos(u) * np.sin(v), r * np.sin(u) * np.sin(v), r * np.cos(v)


def _line(p, name, color, width, opacity=1.0, dash=None, legend=False, gradient=None, group=None):
    line = dict(width=width, color=color)
    if dash:
        line["dash"] = dash
    if gradient is not None:
        line = dict(width=width, color=np.linspace(0, 1, len(p)), colorscale=[[0, gradient], [1, "#ffffff"]])
    return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", name=name, line=line, opacity=opacity,
                        hoverinfo="skip", showlegend=legend, legendgroup=group or name)


@functools.lru_cache(maxsize=1)
def _earth_texture(nlon=256, nlat=128):
    """NASA Blue Marble (public domain) on a sphere. Plotly surfaces take one value per vertex and a
    colour scale, so the image is reduced to a 128-colour palette and drawn as a stepped colour scale."""
    from PIL import Image
    img = Image.open(EARTH_IMAGE).convert("RGB").resize((nlon, nlat), Image.Resampling.LANCZOS)
    q = img.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    idx = np.asarray(q, dtype=float)
    n = int(idx.max()) + 1
    pal = np.array(q.getpalette()[:3 * n]).reshape(-1, 3)
    idx = np.hstack([idx, idx[:, :1]])                           # close the seam at 180°
    lon = np.linspace(-np.pi, np.pi, nlon + 1)
    lat = np.linspace(np.pi / 2, -np.pi / 2, nlat)
    LON, LAT = np.meshgrid(lon, lat)
    xyz = (R_E * np.cos(LAT) * np.cos(LON), R_E * np.cos(LAT) * np.sin(LON), R_E * np.sin(LAT))
    scale = []
    for i, (r, g, b) in enumerate(pal):
        scale += [[i / n, f"rgb({r},{g},{b})"], [(i + 1) / n, f"rgb({r},{g},{b})"]]
    return xyz, idx + 0.5, scale, n


def _stars(radius, n=700, seed=7):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * radius
    return go.Scatter3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], mode="markers", hoverinfo="skip", showlegend=False,
                        marker=dict(size=rng.uniform(0.6, 2.2, n), color="#dfe8ff", opacity=0.55))


def _earth_traces():
    if os.path.exists(EARTH_IMAGE):
        (ex, ey, ez), ec, scale, n = _earth_texture()
        earth = go.Surface(x=ex, y=ey, z=ez, surfacecolor=ec, cmin=0, cmax=n, colorscale=scale, showscale=False,
                           lighting=dict(ambient=0.62, diffuse=0.85, specular=0.06, roughness=0.9, fresnel=0.15),
                           lightposition=dict(x=6000, y=3000, z=2500), hoverinfo="skip", name="Earth")
    else:                                   # decorative fallback if the image is missing
        ex, ey, ez, ec = _earth_surface()
        earth = go.Surface(x=ex, y=ey, z=ez, surfacecolor=ec, cmin=-1, cmax=1, showscale=False,
                           colorscale=EARTH_SCALE,
                           lighting=dict(ambient=0.55, diffuse=0.9, specular=0.12, roughness=0.75, fresnel=0.25),
                           lightposition=dict(x=4000, y=2500, z=3000), hoverinfo="skip", name="Earth")
    traces = [earth]
    # two thin translucent shells give the atmosphere glow
    for scale, op, glow in ((1.025, 0.16, "#7fd0ff"), (1.075, 0.06, "#3f84ff")):
        gx, gy, gz = _sphere(R_E * scale, 64)
        traces.append(go.Surface(x=gx, y=gy, z=gz, surfacecolor=np.zeros_like(gx), showscale=False, opacity=op,
                                 colorscale=[[0, glow], [1, glow]], hoverinfo="skip", name="Atmosphere",
                                 lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0)))
    return traces


def event_scene(row):
    """Geometry from one CDM row: (S, rel_pos, rel_vel, sig, missing columns). S is None if it cannot be drawn."""
    vals = {c: _num(row, c) for c in REQUIRED}
    missing = [c for c in REQUIRED if vals[c] is None]
    if missing:
        return None, None, None, None, missing
    rel_pos = [vals["relative_position_r"], vals["relative_position_t"], vals["relative_position_n"]]
    rel_vel = [vals["relative_velocity_r"], vals["relative_velocity_t"], vals["relative_velocity_n"]]
    s_vals = [(_num(row, f"t_sigma_{x}"), _num(row, f"c_sigma_{x}")) for x in "rtn"]
    sig = ([math.hypot(a, b) for a, b in s_vals]
           if all(a is not None and b is not None for a, b in s_vals) else None)
    return _scene(rel_pos, rel_vel, vals["t_j2k_sma"]), rel_pos, rel_vel, sig, []


# ---------------------------------------------------------------- one event, animated towards TCA
def sim_times(ttc_s, n_days=160, n_hours=140, n_final=60):
    """Seconds before closest approach for the simulator frames, denser as TCA nears: even steps from the
    CDM to T-3 h, ~1-minute steps from T-3 h to T-20 min, then 20-second steps for the final 20 minutes."""
    edges = [ttc_s, min(3 * 3600.0, ttc_s), min(1200.0, ttc_s), 0.0]
    parts = [np.linspace(a, b, n, endpoint=False) for a, b, n in zip(edges[:-1], edges[1:-1] + [0.0],
                                                                   (n_days, n_hours, n_final)) if a > b]
    return np.concatenate(parts + [[0.0]])


def build_event_orbit(ev, tier, miss_m, pred_risk, S, ttc_s, view):
    """Earth, both orbits, and the target and debris moving from the CDM time to closest approach.

    All frames are computed here and sent to the browser once, so dragging the figure's own slider (or
    pressing Play) updates the positions, trails and separation continuously without a Streamlit rerun.
    The debris is propagated with two-body dynamics from its closest-approach state, so the two objects
    meet at the conjunction point at T-0.
    """
    tier_col = TIER_COL.get(tier, "#ffffff")
    a, n_t, T_t, T_c = S["a_t"], S["n_t"], S["T_t"], S["T_c"]
    P = S["rt"]
    dt = 30.0 if ttc_s <= 86400 else 60.0
    taus, pos = _propagate(tuple(S["rc"]), tuple(S["vc"]), float(-ttc_s - 0.3 * T_c), float(T_c), dt)

    def target(tau):
        th = n_t * np.atleast_1d(np.asarray(tau, float))
        return a * np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])

    def debris(tau):
        tau = np.atleast_1d(np.asarray(tau, float))
        return np.column_stack([np.interp(tau, taus, pos[:, i]) for i in range(3)])

    traces = _earth_traces()
    traces.append(_line(target(np.linspace(0, T_t, 240)), "Target orbit", COL_A, 2, 0.35, legend=True))
    traces.append(_line(debris(np.linspace(0, T_c, 240)), "Debris orbit", COL_B, 2, 0.35, legend=True))
    first = True                                  # the last part of each path into the conjunction point
    for fn, period, col, dim in ((target, T_t, COL_A, "#1c5a73"), (debris, T_c, COL_B, "#7a3a63")):
        p = fn(np.linspace(-0.3 * period, 0, 120))
        traces.append(_line(p, "glow", col, 14, 0.12, group="path"))
        traces.append(_line(p, "Final approach", col, 5, 0.9, legend=first, gradient=dim, group="path"))
        first = False
    hover = (f"<b>Closest approach (T−0)</b><br>Miss distance: {miss_m:,.0f} m<br>"
             f"Relative speed: {S['rel_speed']:.2f} km/s<br>Forecast collision risk: {pc_text(pred_risk)}<br>"
             f"Tier: {tier}<extra></extra>")
    traces.append(go.Scatter3d(x=[P[0]], y=[P[1]], z=[P[2]], mode="markers", hoverinfo="skip",
                               marker=dict(size=30, color=tier_col, opacity=0.18), showlegend=False))
    traces.append(go.Scatter3d(x=[P[0]], y=[P[1]], z=[P[2]], mode="markers", hovertemplate=hover,
                               name="Conjunction point",
                               marker=dict(size=9, color=tier_col, symbol="diamond", line=dict(color="white", width=1.5))))
    lab = P * 1.07
    traces.append(go.Scatter3d(x=[lab[0]], y=[lab[1]], z=[lab[2]], mode="text", hoverinfo="skip",
                               text=[f"Closest approach<br>{miss_m:,.0f} m miss"], showlegend=False,
                               textfont=dict(color=tier_col, size=12)))

    def moving(t_before):
        """The six traces that change per frame: two trails, two objects, the separation line and its label."""
        t = -float(t_before)
        pa, pb = target(t)[0], debris(t)[0]
        sep = float(np.linalg.norm(pa - pb))
        sep_txt = f"{miss_m:,.0f} m apart" if t_before == 0 else (
            f"{sep:,.0f} km apart" if sep >= 1 else f"{sep * 1000:,.0f} m apart")
        ta, tb = target(t - np.linspace(0, 0.12 * T_t, 24)), debris(t - np.linspace(0, 0.12 * T_c, 24))
        mid = (pa + pb) / 2 * 1.04
        return sep_txt, [
            go.Scatter3d(x=ta[:, 0], y=ta[:, 1], z=ta[:, 2]),
            go.Scatter3d(x=tb[:, 0], y=tb[:, 1], z=tb[:, 2]),
            go.Scatter3d(x=[pa[0]], y=[pa[1]], z=[pa[2]]),
            go.Scatter3d(x=[pb[0]], y=[pb[1]], z=[pb[2]]),
            go.Scatter3d(x=[pa[0], pb[0]], y=[pa[1], pb[1]], z=[pa[2], pb[2]]),
            go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], text=[sep_txt]),
        ]

    times = sim_times(ttc_s)
    sep0, init = moving(times[0])
    base = len(traces)
    styles = [dict(mode="lines", line=dict(color=COL_A, width=5), hoverinfo="skip", showlegend=False),
              dict(mode="lines", line=dict(color=COL_B, width=5), hoverinfo="skip", showlegend=False),
              dict(mode="markers+text", text=["T"], textposition="top center", textfont=dict(color=COL_A, size=13),
                   name="Target (the satellite being protected)", hoverinfo="name",
                   marker=dict(size=8, color=COL_A, symbol="diamond", line=dict(color="white", width=1))),
              dict(mode="markers+text", text=["D"], textposition="top center", textfont=dict(color=COL_B, size=13),
                   name="Debris (the other object)", hoverinfo="name",
                   marker=dict(size=8, color=COL_B, symbol="diamond", line=dict(color="white", width=1))),
              dict(mode="lines", line=dict(color=INK, width=2, dash="dot"), hoverinfo="skip", showlegend=False),
              dict(mode="text", textfont=dict(color=INK, size=12), hoverinfo="skip", showlegend=False)]
    for tr, style in zip(init, styles):
        tr.update(**style)
        traces.append(tr)
    idx = list(range(base, base + 6))

    frames, steps = [], []
    for i, tb in enumerate(times):
        sep_txt, data = moving(tb)
        frames.append(go.Frame(name=str(i), traces=idx, data=data))
        steps.append(dict(method="animate", label=f"{_fmt_t(tb)}  ·  {sep_txt}",
                          args=[[str(i)], dict(mode="immediate", transition=dict(duration=0),
                                               frame=dict(duration=0, redraw=True))]))

    L = max(a, float(np.max(np.linalg.norm(pos, axis=1)))) * 1.45
    ph = P / np.linalg.norm(P)
    eyes = {"Free orbit": dict(x=1.5, y=1.2, z=0.8),
            "Face the conjunction": dict(x=ph[0] * 1.7, y=ph[1] * 1.7 - 0.5, z=0.5),
            "Top down": dict(x=0.01, y=0.01, z=2.3)}
    axis = dict(visible=False, range=[-L, L], showbackground=False)
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=0, b=80), paper_bgcolor=BG, font=dict(color=INK),
        uirevision=f"{ev}-{view}",
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube", bgcolor=BG, dragmode="orbit",
                   camera=dict(eye=eyes[view], up=dict(x=0, y=0, z=1))),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(14,22,38,0.75)", bordercolor=LINE, borderwidth=1,
                    font=dict(size=12, color=INK)),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=0.0, xanchor="left", yanchor="top",
                          pad=dict(t=10, l=4), bgcolor=PANEL, bordercolor=LINE, font=dict(color=INK), buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(fromcurrent=True, mode="immediate", transition=dict(duration=0),
                                                    frame=dict(duration=45, redraw=True))]),
                              dict(label="❚❚ Pause", method="animate",
                                   args=[[None], dict(mode="immediate", transition=dict(duration=0),
                                                      frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, steps=steps, x=0.16, y=0.0, len=0.83, pad=dict(t=10, b=0),
                      currentvalue=dict(prefix="Time to collision: ", font=dict(color=INK, size=15)),
                      font=dict(color="rgba(0,0,0,0)"), ticklen=0, tickcolor="rgba(0,0,0,0)",
                      bgcolor=LINE, bordercolor=LINE, activebgcolor=COL_B)])
    return fig


# ---------------------------------------------------------------- global orbit figure
def build_orbit_figure(ev, tier, ttc_days, miss_m, pred_risk, S, frame_k, view):
    tier_col = TIER_COL.get(tier, "#ffffff")
    a, n_t, T_t, T_c = S["a_t"], S["n_t"], S["T_t"], S["T_c"]
    P = S["rt"]                                           # conjunction point (target at TCA)

    ttc_s = ttc_days * 86400.0
    t_end = -24 * (frame_k - 1) * 3600.0
    t_start = max(-24 * frame_k * 3600.0, -ttc_s)
    times = np.arange(t_start, t_end + 1, 300.0)          # 5-minute frames
    if len(times) < 2:
        times = np.array([t_start, t_start + 300.0])

    taus, pos = _propagate(tuple(S["rc"]), tuple(S["vc"]),
                           float(min(t_start - 0.15 * T_c, -0.3 * T_c)), float(T_c))

    def target(tau):
        th = n_t * np.atleast_1d(np.asarray(tau, float))
        return a * np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])

    def debris(tau):
        tau = np.atleast_1d(np.asarray(tau, float))
        return np.column_stack([np.interp(tau, taus, pos[:, i]) for i in range(3)])

    L = max(a, float(np.max(np.linalg.norm(pos, axis=1)))) * 1.45
    traces = _earth_traces()

    # full orbits (faint): target is a circle, debris is its propagated orbit over one period
    traces.append(_line(target(np.linspace(0, T_t, 240)), "Target orbit", COL_A, 2, 0.35, group="Target"))
    ring_tau = np.linspace(0, T_c, 240)
    traces.append(_line(debris(ring_tau), "Debris orbit", COL_B, 2, 0.35, group="Debris"))

    # collision trajectory: final approach to TCA, then the leg after it (dashed)
    first = True
    for fn, period, col, dim in ((target, T_t, COL_A, "#1c5a73"), (debris, T_c, COL_B, "#7a3a63")):
        app_t, post_t = np.linspace(-0.30 * period, 0, 160), np.linspace(0, 0.10 * period, 40)
        p = fn(app_t)
        traces.append(_line(p, "glow", col, 16, 0.14, group="path"))
        traces.append(_line(p, "Collision path (final approach)", col, 6, 1.0, legend=first, gradient=dim, group="path"))
        traces.append(_line(fn(post_t), "after TCA", col, 3, 0.6, dash="dash", group="path"))
        first = False

    hover = (f"<b>Closest approach</b><br>Miss distance: {miss_m:,.0f} m<br>"
             f"Relative speed: {S['rel_speed']:.2f} km/s<br>Collision risk: {pc_text(pred_risk)}<br>"
             f"log10 Pc: {pred_risk:.2f}<br>"
             f"Tier: {tier}<extra></extra>")
    traces.append(go.Scatter3d(x=[P[0]], y=[P[1]], z=[P[2]], mode="markers", hoverinfo="skip",
                               marker=dict(size=30, color=tier_col, opacity=0.18), showlegend=False))
    traces.append(go.Scatter3d(x=[P[0]], y=[P[1]], z=[P[2]], mode="markers", hovertemplate=hover,
                               name="Conjunction point",
                               marker=dict(size=9, color=tier_col, symbol="diamond",
                                           line=dict(color="white", width=1.5))))
    lab = P * 1.07
    traces.append(go.Scatter3d(x=[lab[0]], y=[lab[1]], z=[lab[2]], mode="text", hoverinfo="skip",
                               text=[f"Closest approach<br>{miss_m:,.0f} m miss"], showlegend=False,
                               textfont=dict(color=tier_col, size=12)))

    def at(t):
        offs_t, offs_c = np.linspace(0, 0.12 * T_t, 14), np.linspace(0, 0.12 * T_c, 14)
        return target(t)[0], debris(t)[0], target(t - offs_t), debris(t - offs_c)

    pa, pb, ta, tb = at(times[0])
    base = len(traces)
    moving = [base, base + 1, base + 2, base + 3]
    for p, col, nm, lbl in ((pa, COL_A, "Target (the satellite being protected)", "T"),
                            (pb, COL_B, "Debris (the other object)", "D")):
        traces.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", text=[lbl],
                                   textposition="top center", textfont=dict(color=col, size=13),
                                   name=nm, hoverinfo="name",
                                   marker=dict(size=7, color=col, symbol="diamond",
                                               line=dict(color="white", width=1))))
    for t_, col in ((ta, COL_A), (tb, COL_B)):
        traces.append(go.Scatter3d(x=t_[:, 0], y=t_[:, 1], z=t_[:, 2], mode="lines",
                                   line=dict(color=col, width=5), hoverinfo="skip", showlegend=False))

    frames, steps = [], []
    for i, t in enumerate(times):
        pa, pb, ta, tb = at(t)
        frames.append(go.Frame(name=str(i), traces=moving, data=[
            go.Scatter3d(x=[pa[0]], y=[pa[1]], z=[pa[2]]), go.Scatter3d(x=[pb[0]], y=[pb[1]], z=[pb[2]]),
            go.Scatter3d(x=ta[:, 0], y=ta[:, 1], z=ta[:, 2]), go.Scatter3d(x=tb[:, 0], y=tb[:, 1], z=tb[:, 2])]))
        steps.append(dict(method="animate", label=_fmt_t(-t),
                          args=[[str(i)], dict(mode="immediate", transition=dict(duration=0),
                                               frame=dict(duration=0, redraw=True))]))

    ph = P / np.linalg.norm(P)
    eyes = {"Free orbit": dict(x=1.5, y=1.2, z=0.8),
            "Face the conjunction": dict(x=ph[0] * 1.7, y=ph[1] * 1.7 - 0.5, z=0.5),
            "Top down": dict(x=0.01, y=0.01, z=2.3)}
    axis = dict(visible=False, range=[-L, L], showbackground=False)
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        height=700, margin=dict(l=0, r=0, t=0, b=70), paper_bgcolor=BG, font=dict(color=INK),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube", bgcolor=BG, dragmode="orbit",
                   camera=dict(eye=eyes[view], up=dict(x=0, y=0, z=1))),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(14,22,38,0.75)", bordercolor=LINE, borderwidth=1,
                    font=dict(size=12, color=INK)),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=0.0, xanchor="left", yanchor="top",
                          pad=dict(t=8, l=4), bgcolor=PANEL, bordercolor=LINE, font=dict(color=INK),
                          buttons=[
                              dict(label="Play", method="animate",
                                   args=[None, dict(fromcurrent=True, mode="immediate", transition=dict(duration=0),
                                                    frame=dict(duration=35, redraw=True))]),
                              dict(label="Pause", method="animate",
                                   args=[[None], dict(mode="immediate", transition=dict(duration=0),
                                                      frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, steps=steps, x=0.17, y=0.0, len=0.81, pad=dict(t=8, b=0),
                      currentvalue=dict(prefix="Time to closest approach: ", font=dict(color=INK, size=13)),
                      font=dict(color="rgba(0,0,0,0)"), ticklen=0, tickcolor="rgba(0,0,0,0)",
                      bgcolor=LINE, bordercolor=LINE, activebgcolor=COL_A)],
    )
    return fig


# ---------------------------------------------------------------- fleet globe (all events)
def build_fleet_globe(ev, focus_id=None, height=600):
    """Every event around the Earth; click a marker to open its details (customdata = event_id).

    ev columns: event_id, lat_deg, alt_km, inc_deg, tier, p_critical, hover.
    Latitude (geocentric_latitude) and altitude (target apogee/perigee) come from the data. The data has
    no longitude, so events are spread around the Earth by a fixed golden-angle step on event_id. The
    focus event's orbit is drawn with its real inclination through its marker.
    """
    traces = _earth_traces()
    ids = ev["event_id"].to_numpy()
    lon = np.radians((ids * 137.508) % 360.0 - 180.0)
    lat = np.radians(ev["lat_deg"].to_numpy(dtype=float))
    r = R_E + ev["alt_km"].to_numpy(dtype=float)
    pos = np.column_stack([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)])
    tiers, p_crit, hover = ev["tier"].to_numpy(), ev["p_critical"].to_numpy(dtype=float), ev["hover"].to_numpy()
    for tier, size, op in (("safe", 3.5, 0.5), ("elevated", 6.0, 0.9), ("critical", 8.0, 1.0)):
        m = tiers == tier
        if not m.any():
            continue
        traces.append(go.Scatter3d(
            x=pos[m, 0], y=pos[m, 1], z=pos[m, 2], mode="markers", name=f"{tier} ({int(m.sum())})",
            text=hover[m], customdata=ids[m].reshape(-1, 1), hovertemplate="%{text}<extra></extra>",
            marker=dict(size=size + (0 if tier == "safe" else 10 * p_crit[m]), color=TIER_COL[tier], opacity=op,
                        line=dict(color=BG, width=0.5))))

    eye = np.array([1.25, 0.85, 0.55])
    hit = np.flatnonzero(ids == focus_id) if focus_id is not None else []
    if len(hit):
        i = hit[0]
        p, col = pos[i], TIER_COL.get(tiers[i], INK)
        inc = math.radians(float(ev["inc_deg"].to_numpy()[i]))
        u0 = math.asin(float(np.clip(math.sin(lat[i]) / max(math.sin(inc), 1e-6), -1, 1)))
        raan = lon[i] - math.atan2(math.cos(inc) * math.sin(u0), math.cos(u0))
        u = np.linspace(0, 2 * np.pi, 240)
        orb = r[i] * np.column_stack([math.cos(raan) * np.cos(u) - math.sin(raan) * np.sin(u) * math.cos(inc),
                                      math.sin(raan) * np.cos(u) + math.cos(raan) * np.sin(u) * math.cos(inc),
                                      np.sin(u) * math.sin(inc)])
        traces.append(_line(orb, f"Event {focus_id} orbit", col, 3, 0.85, legend=True))
        traces.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", hoverinfo="skip", showlegend=False,
                                   marker=dict(size=28, color=col, opacity=0.25)))
        lab = p * 1.08
        traces.append(go.Scatter3d(x=[lab[0]], y=[lab[1]], z=[lab[2]], mode="text", text=[f"Event {focus_id}"],
                                   hoverinfo="skip", showlegend=False, textfont=dict(color=INK, size=13)))
        eye = p / np.linalg.norm(p) * 1.3 + np.array([0.0, 0.0, 0.2])

    L = R_E * 1.9
    traces.append(_stars(L * 0.97))
    rho, th0 = math.hypot(eye[0], eye[1]), math.atan2(eye[1], eye[0])
    frames = [go.Frame(name=str(k), layout=dict(scene_camera=dict(eye=dict(
        x=rho * math.cos(th0 + 2 * math.pi * k / 90), y=rho * math.sin(th0 + 2 * math.pi * k / 90), z=float(eye[2])))))
        for k in range(90)]
    axis = dict(visible=False, range=[-L, L], showbackground=False)
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=BG, font=dict(color=INK),
        uirevision=f"globe-{focus_id}", clickmode="event+select",
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(color=INK, size=13)),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube", bgcolor=BG, dragmode="orbit",
                   camera=dict(eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])), up=dict(x=0, y=0, z=1))),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(14,22,38,0.75)", bordercolor=LINE, borderwidth=1,
                    font=dict(size=12, color=INK)),
        updatemenus=[dict(type="buttons", direction="left", x=0.01, y=0.02, xanchor="left", yanchor="bottom",
                          bgcolor=PANEL, bordercolor=LINE, font=dict(color=INK), buttons=[
                              dict(label="Spin", method="animate",
                                   args=[None, dict(fromcurrent=True, mode="immediate", transition=dict(duration=0),
                                                    frame=dict(duration=60, redraw=True))]),
                              dict(label="Stop", method="animate",
                                   args=[[None], dict(mode="immediate", transition=dict(duration=0),
                                                      frame=dict(duration=0, redraw=False))])])])
    return fig


# ---------------------------------------------------------------- close approach (true scale, metres)
def combined_cov(row):
    """Combined target + debris position covariance (m^2) in RTN from the CDM sigmas and correlations.
    Both objects are treated in the target's RTN frame, a usual approximation for objects this close."""
    total = np.zeros((3, 3))
    for o in "tc":
        s = [_num(row, f"{o}_sigma_{a}") for a in "rtn"]
        if any(v is None for v in s):
            return None
        rho = [_num(row, f"{o}_{c}") for c in ("ct_r", "cn_r", "cn_t")]
        rho = [0.0 if v is None else float(np.clip(v, -0.999, 0.999)) for v in rho]
        R = np.eye(3)
        R[0, 1] = R[1, 0] = rho[0]
        R[0, 2] = R[2, 0] = rho[1]
        R[1, 2] = R[2, 1] = rho[2]
        total += R * np.outer(s, s)
    w, V = np.linalg.eigh(total)                       # clip to a valid (positive definite) covariance
    return V @ np.diag(np.clip(w, 1e-6, None)) @ V.T


def encounter_stats(rel_pos, rel_vel, C):
    """Numbers for the encounter plane (the plane through the target perpendicular to the relative velocity)."""
    p, v = np.asarray(rel_pos, float), np.asarray(rel_vel, float)
    vh = v / np.linalg.norm(v)
    pp = p - (p @ vh) * vh
    xh = pp / np.linalg.norm(pp) if np.linalg.norm(pp) > 1e-9 else np.cross(vh, [0, 0, 1.0]) / np.linalg.norm(np.cross(vh, [0, 0, 1.0]))
    yh = np.cross(vh, xh)
    B = np.vstack([xh, yh])
    m, Cp = B @ p, B @ C @ B.T
    w, V = np.linalg.eigh(Cp)
    return dict(B=B, m=m, Cp=Cp, w=w, V=V, sd=np.sqrt(w), sd_along_v=float(np.sqrt(vh @ C @ vh)),
                md_plane=float(np.sqrt(m @ np.linalg.solve(Cp, m))),
                md_3d=float(np.sqrt(p @ np.linalg.solve(C, p))))


def build_close_figure(tier, miss_m, rel_pos, rel_vel, sig, C=None):
    """Animated fly-by in the target's RTN frame (metres): the debris moves along its relative path,
    passes closest approach, and the 1σ / 3σ uncertainty ellipsoids (full covariance if given) are drawn
    around its predicted position at closest approach."""
    col = TIER_COL.get(tier, "#ffffff")
    rp, rv = np.array(rel_pos, float), np.array(rel_vel, float)
    speed = float(np.linalg.norm(rv))
    if C is not None:
        w, V = np.linalg.eigh(C)
        axes_len, rot = np.sqrt(w), V
    elif sig:
        axes_len, rot = np.array(sig, float), np.eye(3)
    else:
        axes_len, rot = None, None
    scale = max(miss_m, 3 * float(axes_len.max()) if axes_len is not None else 0.0, 1.0) * 1.5
    span = 2 * scale / speed
    taus = np.linspace(-span, span, 61)
    path = rp[None, :] + taus[:, None] * rv[None, :]

    data = []
    if axes_len is not None:
        for k, op, nm in ((3, 0.10, "3σ position uncertainty"), (1, 0.28, "1σ position uncertainty")):
            x, y, z = _sphere(1.0, 40)
            pts = np.stack([x, y, z], -1) * (k * axes_len) @ rot.T + rp
            data.append(go.Surface(x=pts[..., 0], y=pts[..., 1], z=pts[..., 2], surfacecolor=np.zeros_like(x),
                                   colorscale=[[0, col], [1, col]], showscale=False, opacity=op, name=nm,
                                   hoverinfo="name", showlegend=True))
    L = scale * 0.55                                  # RTN direction guides at the target
    for vec, lbl in (([L, 0, 0], "Radial"), ([0, L, 0], "Along-track"), ([0, 0, L], "Cross-track")):
        data.append(go.Scatter3d(x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode="lines+text", text=["", lbl],
                                 textfont=dict(color=MUTED, size=11), line=dict(color=MUTED, width=2),
                                 hoverinfo="skip", showlegend=False))
    data += [
        go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines", name="Debris path (relative)",
                     line=dict(color=COL_B, width=4), opacity=0.6, hoverinfo="skip"),
        go.Scatter3d(x=[0, rp[0]], y=[0, rp[1]], z=[0, rp[2]], mode="lines+text", name="Miss vector",
                     text=["", f"{miss_m:,.0f} m"], textposition="top center", textfont=dict(color=INK, size=13),
                     line=dict(color=col, width=5, dash="dash"), hoverinfo="skip"),
        go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["Satellite"], textposition="bottom center",
                     textfont=dict(color=COL_A), name="Satellite (target)",
                     marker=dict(size=10, color=COL_A, symbol="diamond", line=dict(color="white", width=1)),
                     hovertemplate="Satellite (origin)<extra></extra>"),
        go.Scatter3d(x=[rp[0]], y=[rp[1]], z=[rp[2]], mode="markers", name="Debris at closest approach",
                     marker=dict(size=6, color=col, symbol="circle-open", line=dict(color=col, width=2)),
                     hovertemplate=f"Closest approach<br>Miss distance {miss_m:,.0f} m<br>"
                                   f"Relative speed {speed / 1000:.2f} km/s<extra></extra>"),
    ]
    moving = len(data)
    data += [go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]], mode="markers", name="Debris",
                          marker=dict(size=9, color=COL_B, symbol="diamond", line=dict(color="white", width=1)),
                          hoverinfo="name"),
             go.Scatter3d(x=path[:1, 0], y=path[:1, 1], z=path[:1, 2], mode="lines", hoverinfo="skip",
                          line=dict(color=COL_B, width=8), showlegend=False)]

    frames, steps = [], []
    for i, tau in enumerate(taus):
        lo = max(0, i - 6)
        frames.append(go.Frame(name=str(i), traces=[moving, moving + 1], data=[
            go.Scatter3d(x=[path[i, 0]], y=[path[i, 1]], z=[path[i, 2]]),
            go.Scatter3d(x=path[lo:i + 1, 0], y=path[lo:i + 1, 1], z=path[lo:i + 1, 2])]))
        steps.append(dict(method="animate", label=f"TCA {tau * 1000:+.0f} ms",
                          args=[[str(i)], dict(mode="immediate", transition=dict(duration=0),
                                               frame=dict(duration=0, redraw=True))]))
    ax = dict(backgroundcolor=BG, gridcolor=LINE, zerolinecolor=MUTED, color=MUTED, showspikes=False)
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=0, b=60), paper_bgcolor=BG, font=dict(color=INK),
        scene=dict(xaxis=dict(title="Radial (m)", **ax), yaxis=dict(title="Along-track (m)", **ax),
                   zaxis=dict(title="Cross-track (m)", **ax), aspectmode="data", bgcolor=BG, dragmode="orbit",
                   camera=dict(eye=dict(x=1.5, y=1.3, z=0.9))),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(14,22,38,0.75)", bordercolor=LINE, borderwidth=1,
                    font=dict(size=12, color=INK)),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=0.0, xanchor="left", yanchor="top",
                          pad=dict(t=8, l=4), bgcolor=PANEL, bordercolor=LINE, font=dict(color=INK), buttons=[
                              dict(label="Play fly-by", method="animate",
                                   args=[None, dict(fromcurrent=True, mode="immediate", transition=dict(duration=0),
                                                    frame=dict(duration=70, redraw=True))]),
                              dict(label="Pause", method="animate",
                                   args=[[None], dict(mode="immediate", transition=dict(duration=0),
                                                      frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, steps=steps, x=0.2, y=0.0, len=0.78, pad=dict(t=8, b=0),
                      currentvalue=dict(prefix="", font=dict(color=INK, size=13)),
                      font=dict(color="rgba(0,0,0,0)"), ticklen=0, tickcolor="rgba(0,0,0,0)",
                      bgcolor=LINE, bordercolor=LINE, activebgcolor=COL_B)])
    return fig


def build_encounter_plane(tier, miss_m, E):
    """The encounter plane: centred on the satellite, perpendicular to the relative velocity. Shows the debris's
    predicted position, the density of the combined position uncertainty and its 1σ / 2σ / 3σ ellipses."""
    col = TIER_COL.get(tier, "#ffffff")
    m, Cp, w, V, sd = E["m"], E["Cp"], E["w"], E["V"], E["sd"]
    half = max(abs(m[0]) / 2 + 3.6 * sd.max(), abs(m[0]) * 0.75, 1.0)
    c = m / 2
    g = np.linspace(-half, half, 181)
    X, Y = np.meshgrid(c[0] + g, c[1] + g)
    dx = np.stack([X - m[0], Y - m[1]], -1)
    d2 = np.einsum("...i,ij,...j", dx, np.linalg.inv(Cp), dx)
    dens = np.exp(-0.5 * d2)
    data = [go.Heatmap(x=c[0] + g, y=c[1] + g, z=dens, zmin=0, zmax=1, showscale=False, hoverinfo="skip",
                       colorscale=[[0, BG], [0.08, "#0d366b"], [0.35, "#1c5cab"], [0.7, "#3987e5"], [1, "#b7d3f6"]])]
    t = np.linspace(0, 2 * np.pi, 200)
    for k, dash in ((1, "solid"), (2, "dash"), (3, "dot")):
        e = m[:, None] + V @ np.diag(k * sd) @ np.vstack([np.cos(t), np.sin(t)])
        data.append(go.Scatter(x=e[0], y=e[1], mode="lines", name=f"{k}σ", hoverinfo="name",
                               line=dict(color="#dfe8ff", width=1.5, dash=dash)))
    data += [
        go.Scatter(x=[0, m[0]], y=[0, m[1]], mode="lines", name="Miss distance", hoverinfo="skip",
                   line=dict(color=col, width=2.5, dash="dash")),
        go.Scatter(x=[0], y=[0], mode="markers+text", text=["Satellite"], textposition="bottom center",
                   textfont=dict(color=COL_A, size=13), name="Satellite",
                   marker=dict(size=14, color=COL_A, symbol="diamond", line=dict(color="white", width=1)),
                   hovertemplate="Satellite<extra></extra>"),
        go.Scatter(x=[m[0]], y=[m[1]], mode="markers+text", text=["Debris (predicted)"], textposition="top center",
                   textfont=dict(color=INK, size=13), name="Debris, predicted position",
                   marker=dict(size=12, color=col, symbol="x", line=dict(color="white", width=1)),
                   hovertemplate=f"Debris predicted position<br>{miss_m:,.0f} m from the satellite<extra></extra>"),
    ]
    fig = go.Figure(data=data)
    fig.add_annotation(x=m[0] / 2, y=m[1] / 2, text=f"{miss_m:,.0f} m · {E['md_plane']:.1f}σ", showarrow=False,
                       yshift=14, font=dict(color=INK, size=13), bgcolor="rgba(14,22,38,0.8)")
    ax = dict(gridcolor=LINE, zerolinecolor=LINE, color=MUTED)
    fig.update_layout(
        height=600, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=INK),
        xaxis=dict(title="Along the miss direction (m)", range=[c[0] - half, c[0] + half], **ax),
        yaxis=dict(title="Perpendicular, in the encounter plane (m)", range=[c[1] - half, c[1] + half],
                   scaleanchor="x", scaleratio=1, **ax),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(14,22,38,0.75)", bordercolor=LINE, borderwidth=1,
                    font=dict(size=12, color=INK)))
    return fig


# ---------------------------------------------------------------- response plan
def _plan(tier, rule_flip, ttc_h, dec_h, burn_h, dv_ms, target_km, crit_cut):
    until_dec = ttc_h - dec_h

    def deadline():
        return (f"by T−{dec_h:g}h, in {_fmt_dur(until_dec)}" if until_dec > 0
                else f"by T−{dec_h:g}h, already passed")

    steps = []
    if rule_flip:
        steps.append(("Verify the tier change", "now",
                      "The predicted tier changed since the previous update. Check the newest tracking "
                      "data by hand before trusting either tier."))
    if tier == "critical":
        steps += [
            ("Confirm the data", "now",
             "Pull the latest tracking update for both objects. Check that the covariance is realistic "
             "and the miss distance matches the previous update."),
            ("Refine the collision probability", "now",
             f"Recompute Pc with operator ephemeris and a higher-fidelity covariance. The critical "
             f"boundary is {crit_cut:.1f} on log10 Pc or a miss distance under 1 km. "
             "Contact the other object's operator if it can manoeuvre."),
            ("Go or no-go on a manoeuvre", deadline(),
             "The analyst and flight director compare the refined Pc with the mission threshold "
             "and decide. The model only supports this decision."),
            ("Plan the avoidance burn", deadline(),
             f"An along-track burn about {burn_h:g} h before closest approach needs roughly "
             f"{dv_ms:.3f} m/s to reach a {target_km:g} km miss distance (first-order estimate). "
             "Screen the post-burn orbit for new conjunctions."),
            ("Upload and execute", f"T−{burn_h:g}h",
             "Upload the command sequence before the burn window and confirm the burn in telemetry."),
            ("Verify the result", "after the burn",
             "Track the new orbit, recompute miss distance and Pc, then stand down or repeat."),
        ]
    elif tier == "elevated":
        steps += [
            ("Raise the tracking cadence", "now",
             "Request more frequent tracking updates for both objects until the event clears."),
            ("Re-screen on every update", "ongoing",
             f"Escalate to the critical plan if log10 Pc rises past {crit_cut:.1f}, "
             "the tier changes, or the miss distance drops below 1 km."),
            ("Prepare a contingency burn", deadline(),
             f"Pre-compute a burn plan (about {dv_ms:.3f} m/s for {target_km:g} km) so a go decision "
             "can be executed without delay."),
            ("Checkpoint before the deadline", deadline(),
             "Review the latest update. Escalate if the tier is critical, otherwise keep monitoring."),
            ("Close out at closest approach", "at TCA", "Record the final miss distance for the event log."),
        ]
    else:
        steps += [
            ("Keep routine monitoring", "now", "No action needed. The event stays in the normal screening queue."),
            ("Re-check on the next update", "next update",
             "If the tier or the miss distance changes, this plan changes with it."),
        ]
    return steps


# ---------------------------------------------------------------- Streamlit entry
def render_orbit_section(row, cuts, show_orbit=True):
    """row: one CDM of the selected event, with all its CSV columns.
    show_orbit=False skips the global orbit figure (when the page already shows one) and keeps the
    to-scale close approach and the response plan."""
    st.markdown(CSS, unsafe_allow_html=True)
    ev = row["event_id"]
    tier = str(row.get("shown_tier", row.get("pred_tier", "safe")))
    ttc_days = float(row["time_to_tca"])
    miss_m = float(row["miss_distance"])
    risk = float(row["pred_risk"])
    flip = bool(row.get("rule_flip", False))
    ttc_h = ttc_days * 24
    n_frames = max(1, math.ceil(ttc_h / 24))
    col = TIER_COL.get(tier, "#fff")

    st.subheader("Orbit view and response plan" if show_orbit else "Close approach and response plan")

    # ---- real geometry from the event's own columns
    S, rel_pos, rel_vel, sig, missing = event_scene(row)
    if missing:
        st.warning("The orbit view needs these columns, but they are missing or empty for this event: "
                   + ", ".join(missing) + ". The response plan below still works.")
    elif S is None:
        st.warning("The debris state in this row gives a non-elliptical orbit, so the orbit cannot be drawn.")

    # ---- controls + stat strip
    frame_k = view = None
    if show_orbit and S:
        ks = list(range(n_frames, 0, -1))
        c1, c2 = st.columns(2)
        frame_k = c1.selectbox(
            "24-hour frame", ks, index=len(ks) - 1, key=f"frame_{ev}",
            format_func=lambda k: f"Day {n_frames - k + 1} of {n_frames}: T−{24 * k}h to T−{24 * (k - 1)}h"
            + (" (final approach)" if k == 1 else ""))
        view = c2.radio("Camera", ["Free orbit", "Face the conjunction", "Top down"], horizontal=True, key=f"view_{ev}")

    stats = [(pc_text(risk), f"estimated collision risk ({pc_odds(risk)})"),
             (_fmt_dur(ttc_h), "to closest approach")]
    if show_orbit:
        stats.append((f"{n_frames} × 24 h", f"{ttc_days:.2f} days = {n_frames} frame(s)"))
    if S:
        stats += [(f"{S['T_t'] / 60:.1f} min", f"target orbit period, {86400 / S['T_t']:.1f} orbits per 24 h"),
                  (f"{S['rel_speed']:.2f} km/s", "relative speed at closest approach"),
                  (f"{S['plane']:.1f}°", "angle between the two orbital planes"),
                  (f"{S['circ_t']:,.0f} km", f"length of the target orbit (debris: {S['circ_c']:,.0f} km)")]
    st.markdown("""<div class="ov"><div class="ov-strip">""" + "".join(
        f"""<div class="ov-stat"><b>{b}</b><span>{s}</span></div>""" for b, s in stats) + "</div></div>",
        unsafe_allow_html=True)

    left, right = st.columns([5, 3], gap="large")
    with left:
        if S:
            if show_orbit:
                tab1, tab2 = st.tabs(["Orbits", "Close approach (to scale)"])
                with tab1:
                    st.plotly_chart(build_orbit_figure(ev, tier, ttc_days, miss_m, risk, S, frame_k, view),
                                    width="stretch",
                                    config=dict(displaylogo=False, scrollZoom=True,
                                                modeBarButtonsToRemove=["resetCameraLastSave3d", "toImage"]))
                    st.caption("Drag to rotate, scroll to zoom, right-drag to pan. Play or drag the slider to move "
                               "through the frame. Bright lines are the final approach to the conjunction point; "
                               "dashed lines continue past it. The miss distance is too small to draw at this scale, "
                               "so open the next tab.")
            else:
                tab2 = st.container()
            with tab2:
                C = combined_cov(row)
                E = encounter_stats(rel_pos, rel_vel, C) if C is not None else None
                cdm_risk = _num(row, "risk")
                tiles = [(f"{miss_m:,.0f} m", f"miss: R {rel_pos[0]:,.0f} · T {rel_pos[1]:,.0f} · N {rel_pos[2]:,.0f} m"),
                         (f"{np.linalg.norm(rel_vel) / 1000:,.2f} km/s", "relative speed at closest approach")]
                if E:
                    tiles += [(f"{E['md_plane']:.1f}σ", "satellite's distance from the debris's predicted position, "
                                                        "in the encounter plane (smaller is more dangerous)"),
                              (f"{E['sd'].max():,.0f} × {E['sd'].min():,.0f} m", "1σ uncertainty ellipse in that plane"),
                              (f"{6 * E['sd_along_v'] / np.linalg.norm(rel_vel) * 1000:,.0f} ms",
                               "time the debris takes to cross the 3σ region along its path")]
                if cdm_risk is not None:
                    tiles.append((pc_text(cdm_risk), f"Pc reported in this CDM ({pc_odds(cdm_risk)})"))
                st.markdown("""<div class="ov"><div class="ov-strip">""" + "".join(
                    f"""<div class="ov-stat"><b>{b}</b><span>{s}</span></div>""" for b, s in tiles) + "</div></div>",
                    unsafe_allow_html=True)
                t_fly, t_plane = st.tabs(["3D fly-by (to scale)", "Encounter plane"])
                with t_fly:
                    st.plotly_chart(build_close_figure(tier, miss_m, rel_pos, rel_vel, sig, C), width="stretch",
                                    key=f"flyby_{ev}", config=dict(displaylogo=False, scrollZoom=True))
                    st.caption("Press Play fly-by to watch the debris pass the satellite at the real relative speed "
                               "(the slider is in milliseconds from closest approach). Axes are metres in the "
                               "satellite's radial / along-track / cross-track frame. The shells are the combined "
                               "1σ and 3σ position uncertainty"
                               + (" from the CDM's sigmas and correlations." if C is not None else " (diagonal only)."))
                with t_plane:
                    if E:
                        st.plotly_chart(build_encounter_plane(tier, miss_m, E), width="stretch",
                                        key=f"bplane_{ev}", config=dict(displaylogo=False))
                        st.caption("The view conjunction analysts use: looking along the relative velocity, so the "
                                   "debris passes straight through the page. Brighter blue is where the debris is more "
                                   "likely to be. The collision probability is the share of that cloud that falls on "
                                   "the satellite, so the closer the satellite sits to the bright centre (fewer σ), "
                                   "the higher the risk. Combined covariance from the CDM, both objects in the "
                                   "satellite's RTN frame (an approximation).")
                    else:
                        st.info("This CDM lacks the sigma columns needed for the encounter plane.")
            with st.expander("Data used and checks"):
                lines = [f"- Relative position at closest approach: R {rel_pos[0]:,.0f}, T {rel_pos[1]:,.0f}, "
                         f"N {rel_pos[2]:,.0f} m; length {np.linalg.norm(rel_pos):,.0f} m vs miss_distance {miss_m:,.0f} m.",
                         f"- Target semi-major axis {S['a_t']:,.1f} km (period {S['T_t'] / 60:.2f} min), treated as circular"
                         + (f"; catalogued eccentricity {_num(row, 't_j2k_ecc'):.4f}." if _num(row, 't_j2k_ecc') is not None else "."),
                         f"- Debris from the relative state: semi-major axis {S['a_c']:,.1f} km, eccentricity {S['ecc_c']:.4f}, "
                         f"period {S['T_c'] / 60:.2f} min."]
                if _num(row, "c_j2k_sma") is not None:
                    lines.append(f"- Debris semi-major axis in the data: {_num(row, 'c_j2k_sma'):,.1f} km.")
                if _num(row, "relative_speed") is not None:
                    lines.append(f"- relative_speed column {_num(row, 'relative_speed'):,.1f} m/s vs "
                                 f"{S['rel_speed'] * 1000:,.1f} m/s from the velocity components.")
                lines.append("- Debris is propagated with two-body dynamics from the closest-approach state, so "
                             "positions far from TCA drift from reality (no J2 or drag). Earth's rotation is ignored.")
                st.markdown("\n".join(lines))
        else:
            st.info("Add the missing columns to see the orbit.")

    with right:
        with st.expander("Plan settings", expanded=False):
            dec_h = st.number_input("Decision lead time (h before TCA)", 1.0, 240.0, 24.0, 1.0, key=f"dec_{ev}")
            burn_h = st.number_input("Burn lead time (h before TCA)", 1.0, 240.0, 12.0, 1.0, key=f"burn_{ev}")
            default_tgt = float(np.clip(round(3 * max(sig) / 1000 * 2) / 2, 1.0, 50.0)) if sig else 5.0
            target_km = st.number_input(
                "Target miss distance (km)", 1.0, 50.0, default_tgt, 0.5, key=f"tgt_{ev}",
                help="Default is 3σ of this event's combined position uncertainty (at least 1 km)." if sig else None)
        dv_ms = target_km * 1000.0 / (3.0 * burn_h * 3600.0)       # along-track drift ~ 3 * dv * t
        until_dec = ttc_h - dec_h
        dec_txt = (f"decision deadline in {_fmt_dur(until_dec)}" if until_dec > 0 else "decision deadline already passed")
        st.markdown(
            f"""<div class="ov"><div class="ov-status" style="--tier:{col}">
            <div class="tier">{tier.capitalize()} · miss {miss_m:,.0f} m</div>
            <div class="count">{pc_text(risk)}</div>
            <div class="sub">estimated collision risk, {pc_odds(risk)} (log10 Pc {risk:.1f})</div>
            <div class="sub" style="margin-top:8px">{_fmt_t(ttc_h * 3600)} to closest approach · {dec_txt}</div>
            </div></div>""", unsafe_allow_html=True)
        if tier != "safe" and until_dec <= 0:
            st.warning("The decision deadline has passed for these lead times. Use a compressed timeline "
                       "or shorten the lead times in Plan settings.")

        steps = _plan(tier, flip, ttc_h, dec_h, burn_h, dv_ms, target_km, float(cuts["critical_cut"]))
        bar = st.empty()
        done_n = 0
        for i, (title, when, detail) in enumerate(steps):
            k1, k2 = st.columns([1, 11])
            done = k1.checkbox("done", key=f"step_{ev}_{tier}_{i}", label_visibility="collapsed")
            done_n += int(done)
            k2.markdown(
                f"""<div class="ov"><div class="ov-step{' done' if done else ''}" style="--tier:{col}">
                <span class="t">{title}</span><span class="w">{when}</span>
                <div class="d">{detail}</div></div></div>""", unsafe_allow_html=True)
        bar.progress(done_n / len(steps), text=f"{done_n} of {len(steps)} steps done")
        st.caption("Generic decision support. Follow your operator's conjunction assessment procedure; "
                   "a qualified analyst makes the manoeuvre decision. The Δv figure is a first-order "
                   "along-track estimate.")