import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import CRIT_LOG_PC, CRIT_MISS_M, ELEV_LOG_PC, ELEV_MISS_M
from orbit_view import (BG, CSS, TIER_COL, _fmt_dur, _fmt_t, build_event_orbit, build_fleet_globe, event_scene,
                        pc_odds, pc_text, render_orbit_section)

OUT = "outputs/v2"
st.set_page_config(page_title="Conjunction triage", page_icon="🛰️", layout="wide")
alt.data_transformers.disable_max_rows()

# status colours carry the tier (same in both themes); every use is paired with the tier name
DARK = getattr(getattr(st.context, "theme", None), "type", "dark") == "dark"
TIERS = ["critical", "elevated", "safe"]
TIER_SCALE = alt.Scale(domain=TIERS, range=[TIER_COL[t] for t in TIERS])
BLUE, ORANGE = ("#3987e5", "#d95926") if DARK else ("#2a78d6", "#eb6834")
RULE_INK, TRUTH_INK = ("#c3c2b7", "#f0efec") if DARK else ("#52514e", "#0b0b0b")
SERIES_SCALE = alt.Scale(domain=["observed (CDM)", "forecast of final CDM"], range=[BLUE, ORANGE])
ICON = {"critical": "▲", "elevated": "●", "safe": "○"}
ORDER = {"critical": 0, "elevated": 1, "safe": 2}
DECISION_H = 24.0       # decision deadline shown on the front page, hours before closest approach

HERO_CSS = """
<style>
.hero { background:#0e1626; border:1px solid #1d2a44; border-left:6px solid var(--tier); border-radius:12px;
        padding:16px 18px; margin-bottom:10px; }
.hero .lbl { color:var(--tier); font-weight:700; letter-spacing:.05em; text-transform:uppercase; font-size:.85rem; }
.hero .clock { color:#eaf1ff; font-weight:700; font-size:2.7rem; line-height:1.1; font-variant-numeric:tabular-nums; }
.hero .sub { color:#8393b3; font-size:.88rem; margin-top:2px; }
.hero-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
.hero-grid div { background:#0b1220; border:1px solid #1d2a44; border-radius:8px; padding:8px 10px; }
.hero-grid b { display:block; color:#eaf1ff; font-size:1.05rem; font-variant-numeric:tabular-nums; }
.hero-grid span { color:#8393b3; font-size:.78rem; }
.ico { color:var(--tier); }
</style>
"""


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    upd = pd.read_csv(f"{OUT}/test_update_predictions.csv")
    hist = pd.read_csv(f"{OUT}/test_cdm_history.csv")
    meta = json.load(open(f"{OUT}/meta.json"))
    metrics = json.load(open(f"{OUT}/test_metrics.json"))
    cv = pd.read_csv(f"{OUT}/cv_feature_sets.csv")
    imp = pd.read_csv(f"{OUT}/feature_importance.csv")
    cal = pd.read_csv(f"{OUT}/test_calibration.csv")
    upd["forecast_pc"] = upd["pred_final_risk"].map(pc_text)
    upd["latest_pc"] = upd["risk"].map(pc_text)
    upd["p_crit_pct"] = 100 * upd["p_critical"]
    upd["tier"] = upd["pred_tier"].map(lambda t: f"{ICON[t]} {t}")
    upd["countdown"] = (upd["time_to_tca"] * 86400).map(_fmt_t)
    upd["alt_km"] = (upd["t_h_apo"] + upd["t_h_per"]) / 2
    return upd, hist, meta, metrics, cv, imp, cal


needed = ["test_update_predictions.csv", "test_cdm_history.csv", "meta.json", "test_metrics.json"]
if not all(os.path.exists(f"{OUT}/{f}") for f in needed):
    st.error(f"Model outputs not found in {OUT}/. Run `python train.py` and then `python evaluate.py`.")
    st.stop()
upd, hist, meta, metrics, cv, imp, cal = load()
latest = upd[upd["is_latest_input"]]           # one row per event: the forecast after its most recent CDM


def why(r):
    out = []
    if r.pred_final_risk >= CRIT_LOG_PC:
        out.append("forecast Pc ≥ 1e-4")
    if r.pred_final_miss < CRIT_MISS_M:
        out.append("forecast miss < 1 km")
    if r.risk >= CRIT_LOG_PC or r.miss_distance < CRIT_MISS_M:
        out.append("latest CDM already critical")
    if not out and r.pred_tier != "safe":
        out.append("trend and probability")
    return ", ".join(out) or "no warning"


# ---------------------------------------------------------------- sidebar
# slider starts at the first quarter-day that contains an event, so the page is never empty
lo = float(np.ceil(latest["time_to_tca"].min() * 4) / 4)
hi = float(np.ceil(latest["time_to_tca"].max() * 4) / 4)
with st.sidebar:
    st.header("Controls")
    window = st.slider("Window: days to closest approach", lo, hi, min(lo + 1.0, hi), 0.25,
                       help="Uses each event's most recent CDM. The test data stops 2 days before closest approach.")
    show_safe = st.checkbox("Show safe events on the globe", value=True)
    show_truth = st.checkbox("Show true final values (test set, evaluation only)", value=False)
    st.divider()
    st.markdown(f"**Model:** XGBoost, feature set *{meta['feature set']}*  \n"
                f"**Manoeuvre flag:** P(critical) ≥ {meta['p_critical_cut']:.0%}  \n"
                f"**Elevated:** P(elevated or critical) ≥ {meta['p_elevated_cut']:.0%}")
    st.caption("Click a marker on the globe or a point on the countdown board for an event's details, pick an event "
               "by number, or select a row in the warnings queue. The approach simulator further down animates the "
               "approach as you drag its slider.")

win = latest[latest["time_to_tca"] <= window].copy()
win["o"] = win["pred_tier"].map(ORDER)
win = win.sort_values(["o", "p_critical", "time_to_tca"], ascending=[True, False, True])
win["why"] = [why(r) for r in win.itertuples()]

# ---------------------------------------------------------------- focus event (shared by picker, board, queue)
# any event can be picked by number; most urgent in the window first, then the rest
ranked = latest.assign(o=latest["pred_tier"].map(ORDER), inwin=latest["time_to_tca"] <= window)
ranked = ranked.sort_values(["inwin", "o", "p_critical"], ascending=[False, True, False])
all_ids = ranked["event_id"].tolist()
ids = win["event_id"].tolist()
if ids and st.session_state.get("focus") not in all_ids:
    st.session_state["focus"] = ids[0]
QUEUE = win[win["pred_tier"] != "safe"].sort_values(["o", "time_to_tca"]).reset_index(drop=True)


def pick_from_board():
    pts = st.session_state["board"]["selection"].get("pick") or []
    if pts:
        st.session_state["focus"] = int(pts[0]["event_id"])
        st.session_state["popup"] = int(pts[0]["event_id"])


def pick_from_queue():
    rows = st.session_state["queue"]["selection"]["rows"]
    if rows:
        st.session_state["focus"] = int(QUEUE.iloc[rows[0]]["event_id"])


def event_card(r, eid, clock=None, clock_sub=None):
    """The countdown card for one event (r: its latest-CDM row)."""
    col = TIER_COL[r["pred_tier"]]
    until_dec = r["time_to_tca"] * 24 - DECISION_H
    dec = (f"decision deadline (T−{DECISION_H:g}h) in {_fmt_dur(until_dec)}" if until_dec > 0
           else f"past the decision deadline (T−{DECISION_H:g}h)")
    grid = [(r["forecast_pc"], f"forecast final Pc, {pc_odds(r['pred_final_risk'])}"),
            (f"{r['pred_final_miss']:,.0f} m", "forecast final miss distance"),
            (r["latest_pc"], "Pc in the latest CDM"),
            (f"{r['miss_distance']:,.0f} m", "miss in the latest CDM"),
            (f"{r['relative_speed'] / 1000:,.2f} km/s", "closing speed"),
            (str(r["c_object_type"]).title(), "other object"),
            (f"{r['geocentric_latitude']:+.1f}° · {r['alt_km']:,.0f} km", "conjunction latitude · altitude"),
            (f"{r['t_j2k_inc']:.1f}°", "satellite orbit inclination")]
    if show_truth:
        grid.append((r["final_tier_name"], "true final tier"))
    return (f'<div class="ov"><div class="hero" style="--tier:{col}">'
            f'<div class="lbl">Event {eid} · {ICON[r["pred_tier"]]} {r["pred_tier"]}'
            f'{" · manoeuvre planning" if r["maneuver_flag"] else ""}</div>'
            f'<div class="clock">{clock or r["countdown"]}</div>'
            f'<div class="sub">{clock_sub or f"time to collision (closest approach), as of CDM {int(r["n_cdms"])}"} · {dec}</div>'
            f'<div class="sub">Why: {why(r)}</div>'
            '<div class="hero-grid">' + "".join(f"<div><b>{b}</b><span>{s}</span></div>" for b, s in grid)
            + "</div></div></div>")


def prob_bar(r):
    probs = pd.DataFrame({"tier": TIERS[::-1], "p": [r["p_safe"], r["p_elevated"], r["p_critical"]], "o": [0, 1, 2]})
    return alt.Chart(probs).mark_bar(height=26, stroke=BG, strokeWidth=2).encode(
        x=alt.X("p:Q", stack="zero", scale=alt.Scale(domain=[0, 1]), axis=None),
        color=alt.Color("tier:N", scale=TIER_SCALE, legend=None), order=alt.Order("o:Q"),
        tooltip=[alt.Tooltip("tier:N"), alt.Tooltip("p:Q", title="probability", format=".1%")]).properties(height=40)


def spark(eid, obs_col, fc_col, title, log):
    """Small observed-vs-forecast history chart for the pop-up."""
    h = hist[(hist["event_id"] == eid) & (~hist["is_final"])]
    u = upd[upd["event_id"] == eid]
    rows = pd.concat([h[["time_to_tca", obs_col]].rename(columns={obs_col: "v"}).assign(series="observed (CDM)"),
                      u[["time_to_tca", fc_col]].rename(columns={fc_col: "v"}).assign(series="forecast of final CDM")])
    rows["v"] = rows["v"].clip(lower=1.0) if log else rows["v"].clip(lower=-10)
    return alt.Chart(rows).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        x=alt.X("time_to_tca:Q", scale=alt.Scale(reverse=True), title="days to closest approach"),
        y=alt.Y("v:Q", title=title, scale=alt.Scale(type="log") if log else alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=SERIES_SCALE, title="", legend=alt.Legend(orient="bottom")),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=list(SERIES_SCALE.domain), range=[[1, 0], [6, 4]]),
                                  legend=None),
        tooltip=[alt.Tooltip("series:N", title=""), alt.Tooltip("time_to_tca:Q", title="days", format=".2f"),
                 alt.Tooltip("v:Q", title=title, format=",.2f")]).properties(height=180)


@st.dialog("Event details", width="large")
def event_dialog(eid):
    r = ranked.set_index("event_id").loc[eid]
    st.markdown(CSS + HERO_CSS + event_card(r, eid), unsafe_allow_html=True)
    st.altair_chart(prob_bar(r), width="stretch")
    st.markdown(f"○ safe **{r['p_safe']:.0%}** · ● elevated **{r['p_elevated']:.0%}** · "
                f"▲ critical **{r['p_critical']:.0%}**")
    c1, c2 = st.columns(2)
    c1.altair_chart(spark(eid, "risk", "pred_final_risk", "log10 Pc", False), width="stretch")
    c2.altair_chart(spark(eid, "miss_distance", "pred_final_miss", "miss (m)", True), width="stretch")
    st.caption("Latitude and altitude are from the CDM; the globe's longitude is illustrative because the data has none.")
    if st.button("Open the full analysis below", type="primary", width="stretch"):
        st.rerun()


def pick_from_globe():
    for p in st.session_state["globe"]["selection"]["points"]:
        cd = p.get("customdata")
        if cd is None:
            continue
        eid = int(cd[0] if isinstance(cd, (list, tuple)) else cd)
        st.session_state["focus"] = eid
        st.session_state["popup"] = eid
        break


# ---------------------------------------------------------------- header
st.markdown(CSS + HERO_CSS, unsafe_allow_html=True)
st.title("🛰️ Satellite conjunction triage")
st.caption("Forecasts each event's final risk tier from the CDMs received so far. Critical: Pc ≥ 1e-4 or miss < 1 km. "
           "Elevated: Pc ≥ 1e-6 or miss 1–5 km. Safe: otherwise. Decision support only: a qualified analyst decides "
           "on any manoeuvre.")

tab_main, tab_all, tab_eval = st.tabs(["Mission control", "All events", "Evaluation & method"])

with tab_main:
    n_crit = int(win["maneuver_flag"].sum())
    flagged = win[win["pred_tier"] != "safe"]
    nxt = _fmt_t(flagged["time_to_tca"].min() * 86400) if len(flagged) else "none"
    tiles = [(f"{len(win):,}", f"events within {window:g} days", None),
             (f"{n_crit}", "manoeuvre planning (forecast critical)", "critical"),
             (f"{int((win['pred_tier'] == 'elevated').sum())}", "forecast elevated", "elevated"),
             (f"{int(win['rule_flip'].sum())}", "tier changed since previous CDM", None),
             (nxt, "next flagged closest approach", None)]
    st.markdown('<div class="ov"><div class="ov-strip">' + "".join(
        f'<div class="ov-stat"><b>{v}</b><span>'
        + (f'<span class="ico" style="--tier:{TIER_COL[t]}">{ICON[t]}</span> ' if t else "") + f"{lbl}</span></div>"
        for v, lbl, t in tiles) + "</div></div>", unsafe_allow_html=True)

    if not ids:
        st.info("No events in this window. Widen it in the sidebar.")
        st.stop()

    # ---- globe with every event (click a marker for its details) + countdown card for the focus event
    label = ranked.set_index("event_id")
    g_col, f_col = st.columns([3, 2], gap="large")
    with f_col:
        focus = st.selectbox("Event number", all_ids, key="focus",
                             format_func=lambda e: f"{e} · {label.at[e, 'tier']} · {label.at[e, 'countdown']}",
                             help="Type a number to search, or click a marker on the globe. Events in the window "
                                  "come first, most urgent first.")
        r = label.loc[focus]
        st.markdown(event_card(r, focus), unsafe_allow_html=True)
        st.markdown("**Forecast final tier probabilities**")
        st.altair_chart(prob_bar(r), width="stretch")
        st.markdown(f"○ safe **{r['p_safe']:.0%}** · ● elevated **{r['p_elevated']:.0%}** · "
                    f"▲ critical **{r['p_critical']:.0%}**")

    with g_col:
        g = win if show_safe else win[win["pred_tier"] != "safe"]
        if focus not in set(g["event_id"]):
            g = pd.concat([g, label.loc[[focus]].reset_index()], ignore_index=True)
        g = g.assign(lat_deg=g["geocentric_latitude"], inc_deg=g["t_j2k_inc"], tier=g["pred_tier"],
                     hover=[f"<b>Event {e}</b> · {ICON[t]} {t}<br>P(critical) {p:.0%} · {c}<br>"
                            f"forecast Pc {fp} · miss {m:,.0f} m<br><i>click for details</i>"
                            for e, t, p, c, fp, m in zip(g["event_id"], g["pred_tier"], g["p_critical"],
                                                         g["countdown"], g["forecast_pc"], g["pred_final_miss"])])
        st.plotly_chart(build_fleet_globe(g, focus), width="stretch", key="globe", on_select=pick_from_globe,
                        selection_mode="points", config=dict(displaylogo=False, scrollZoom=True))
        st.caption("Each marker is one event at its conjunction latitude and altitude (from the data); size grows with "
                   "P(critical). Click a marker for its details. The data has no longitude, so events are spread "
                   "around the Earth evenly. The ring is the focus event's orbit at its real inclination. "
                   "Earth image: NASA Blue Marble.")

    popup = st.session_state.pop("popup", None)
    if popup is not None and popup in label.index:
        event_dialog(popup)

    # ---- countdown board
    st.subheader("Countdown board")
    board = win.assign(is_focus=win["event_id"] == focus)
    pick = alt.selection_point(name="pick", fields=["event_id"], on="click")
    pts = alt.Chart(board).mark_circle(stroke=BG, strokeWidth=1, opacity=0.85).encode(
        x=alt.X("time_to_tca:Q", scale=alt.Scale(reverse=True, domain=[lo, window], clamp=True),
                title="Days to closest approach (latest CDM) → closer"),
        y=alt.Y("p_critical:Q", title="P(final tier critical)", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("pred_tier:N", scale=TIER_SCALE, title="Forecast tier"),
        size=alt.condition("datum.is_focus", alt.value(420), alt.value(70)),
        tooltip=[alt.Tooltip("event_id:Q", title="event"), alt.Tooltip("pred_tier:N", title="forecast tier"),
                 alt.Tooltip("countdown:N", title="time to closest approach"),
                 alt.Tooltip("p_critical:Q", title="P(critical)", format=".0%"),
                 alt.Tooltip("forecast_pc:N", title="forecast Pc"),
                 alt.Tooltip("pred_final_miss:Q", title="forecast miss (m)", format=",.0f")]).add_params(pick)
    cut = pd.DataFrame({"y": [meta["p_critical_cut"]], "label": [f"manoeuvre flag at {meta['p_critical_cut']:.0%}"]})
    rules = (alt.Chart(cut).mark_rule(strokeDash=[4, 4], color=RULE_INK).encode(y="y:Q")
             + alt.Chart(cut).mark_text(align="left", dy=-7, x=6, color=RULE_INK).encode(y="y:Q", text="label:N"))
    st.altair_chart((pts + rules).properties(height=300), width="stretch", key="board",
                    on_select=pick_from_board, selection_mode="pick")
    st.caption("Every event in the window. Right is closer to closest approach; higher is more likely to end critical. "
               "Click a point to focus it. The large point is the focus event.")

    # ---- warnings queue
    st.subheader(f"Warnings queue · {len(QUEUE)} event(s)")
    qcols = ["tier", "event_id", "countdown", "p_crit_pct", "forecast_pc", "pred_final_miss", "miss_distance",
             "why", "rule_flip"] + (["final_tier_name"] if show_truth else [])
    st.dataframe(QUEUE[qcols], key="queue", on_select=pick_from_queue, selection_mode="single-row",
                 hide_index=True, width="stretch", height=320, column_config={
                     "tier": st.column_config.TextColumn("forecast tier"),
                     "event_id": st.column_config.NumberColumn("event", format="%d"),
                     "countdown": st.column_config.TextColumn("time to closest approach"),
                     "p_crit_pct": st.column_config.ProgressColumn("P(critical)", min_value=0, max_value=100,
                                                                   format="%.0f%%"),
                     "forecast_pc": st.column_config.TextColumn("forecast final Pc"),
                     "pred_final_miss": st.column_config.NumberColumn("forecast final miss (m)", format="%,.0f"),
                     "miss_distance": st.column_config.NumberColumn("latest miss (m)", format="%,.0f"),
                     "why": st.column_config.TextColumn("why flagged"),
                     "rule_flip": st.column_config.CheckboxColumn("tier changed"),
                     "final_tier_name": st.column_config.TextColumn("true final tier")})
    st.download_button("Download warnings (CSV)", QUEUE[qcols + ["pred_final_risk", "time_to_tca"]].to_csv(index=False),
                       "warnings.csv", "text/csv")
    st.caption("Critical first, then elevated, soonest first. Select a row to focus the event.")

    # ---- deep dive: replay, history, orbit
    st.divider()
    st.header(f"Event {focus}")
    u = upd[upd["event_id"] == focus].sort_values("time_to_tca", ascending=False).reset_index(drop=True)
    labels = [f"CDM {int(n)} · {c}" for n, c in zip(u["n_cdms"], u["countdown"])]
    k = len(u) - 1
    if len(u) > 1:
        k = st.select_slider("Replay the event: the forecast after each CDM", options=list(range(len(u))),
                             value=len(u) - 1, format_func=lambda i: labels[i], key=f"replay_{focus}")
    row = u.iloc[k]
    h = hist[hist["event_id"] == focus]
    seen = h[(~h["is_final"]) & (h["time_to_tca"] >= row["time_to_tca"] - 1e-9)]
    u_seen = u.iloc[:k + 1]
    x = alt.X("time_to_tca:Q", scale=alt.Scale(reverse=True), title="Days to closest approach")

    def history_chart(obs_col, fc_col, title, log, rules_df):
        o = seen[["time_to_tca", obs_col]].rename(columns={obs_col: "value"}).assign(series="observed (CDM)")
        f = u_seen[["time_to_tca", fc_col]].rename(columns={fc_col: "value"}).assign(series="forecast of final CDM")
        rows = pd.concat([o, f])
        clip = (lambda s: s.clip(lower=1.0)) if log else (lambda s: s.clip(lower=-10))
        rows["value"] = clip(rows["value"])
        y = alt.Y("value:Q", title=title, scale=alt.Scale(type="log") if log else alt.Scale(zero=False))
        lines = alt.Chart(rows).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
            x=x, y=y, color=alt.Color("series:N", scale=SERIES_SCALE, title=""),
            strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=list(SERIES_SCALE.domain),
                                                                   range=[[1, 0], [6, 4]]), legend=None),
            tooltip=[alt.Tooltip("series:N", title=""), alt.Tooltip("time_to_tca:Q", title="days to TCA", format=".2f"),
                     alt.Tooltip("value:Q", title=title, format=",.2f")])
        layers = (lines
                  + alt.Chart(rules_df).mark_rule(strokeDash=[2, 3], color=RULE_INK).encode(y="y:Q")
                  + alt.Chart(rules_df).mark_text(align="left", dy=-6, x=4, color=RULE_INK).encode(y="y:Q", text="label:N"))
        if show_truth:
            fin = h[h["is_final"]].rename(columns={obs_col: "value"})
            fin = fin.assign(value=clip(fin["value"]))
            layers += alt.Chart(fin).mark_point(shape="diamond", size=160, filled=True, color=TRUTH_INK).encode(
                x=x, y="value:Q", tooltip=[alt.Tooltip("value:Q", title="true final", format=",.2f")])
        return layers.properties(height=250)

    left, right = st.columns(2)
    with left:
        st.markdown("**Collision probability (log10 Pc)**")
        st.altair_chart(history_chart("risk", "pred_final_risk", "log10 Pc", False, pd.DataFrame(
            {"y": [CRIT_LOG_PC, ELEV_LOG_PC], "label": ["critical: Pc 1e-4", "elevated: Pc 1e-6"]})), width="stretch")
    with right:
        st.markdown("**Miss distance (m, log scale)**")
        st.altair_chart(history_chart("miss_distance", "pred_final_miss", "miss distance (m)", True, pd.DataFrame(
            {"y": [CRIT_MISS_M, ELEV_MISS_M], "label": ["critical: 1 km", "elevated: 5 km"]})), width="stretch")
    pline = alt.Chart(u_seen).mark_line(strokeWidth=2, color=BLUE, point=alt.OverlayMarkDef(size=64, color=BLUE)).encode(
        x=x, y=alt.Y("p_critical:Q", title="P(critical)", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("countdown:N", title="time to closest approach"),
                 alt.Tooltip("p_critical:Q", title="P(critical)", format=".0%"),
                 alt.Tooltip("pred_tier:N", title="forecast tier")])
    st.markdown("**P(final tier is critical), after each CDM**")
    st.altair_chart((pline + rules).properties(height=170), width="stretch")
    st.caption("Solid blue: values reported in each CDM. Dashed orange: the model's forecast of the final CDM after each "
               "CDM. Move the replay slider to step through the event. "
               + ("Diamond: the true final CDM." if show_truth else "Tick 'Show true final values' to add the true final CDM."))

    # ---- approach simulator: the selected event only; the figure's own slider animates it in the browser
    st.subheader("Approach simulator")
    S, rel_pos, rel_vel, sig, missing = event_scene(row)
    ttc_s = float(row["time_to_tca"]) * 86400
    view = st.radio("Camera", ["Free orbit", "Face the conjunction", "Top down"], key=f"cam_{focus}", horizontal=True)
    if S is None:
        st.warning("This CDM is missing geometry columns or gives a non-elliptical debris orbit, so the approach "
                   "cannot be simulated." + (f" Missing: {', '.join(missing)}." if missing else ""))
    else:
        tiles = [(_fmt_t(ttc_s), f"to closest approach at CDM {int(row['n_cdms'])}"),
                 (f"{row['miss_distance']:,.0f} m", "separation at T−0 (the miss distance)"),
                 (f"{S['rel_speed']:.2f} km/s", "closing speed at closest approach"),
                 (f"{S['plane']:.1f}°", "angle between the two orbit planes")]
        st.markdown('<div class="ov"><div class="ov-strip">' + "".join(
            f'<div class="ov-stat"><b>{b}</b><span>{s}</span></div>' for b, s in tiles) + "</div></div>",
            unsafe_allow_html=True)
        st.plotly_chart(build_event_orbit(focus, row["pred_tier"], float(row["miss_distance"]),
                                          float(row["pred_final_risk"]), S, ttc_s, view),
                        width="stretch", key=f"orbit_{focus}_{k}",
                        config=dict(displaylogo=False, scrollZoom=True,
                                    modeBarButtonsToRemove=["resetCameraLastSave3d", "toImage"]))
        st.caption("Drag the slider under the globe (or press Play) and the satellite (T) and debris (D) move "
                   "continuously towards closest approach; the label shows the time to collision and their "
                   "separation. Steps are even until T−20 min, then every 20 seconds for the final approach. "
                   "Positions come from the selected CDM's relative state with two-body dynamics, so they are "
                   "approximate far from closest approach.")

    orbit_row = row.copy()
    orbit_row["shown_tier"] = orbit_row["pred_tier"]
    orbit_row["pred_risk"] = orbit_row["pred_final_risk"]
    render_orbit_section(orbit_row, {"critical_cut": CRIT_LOG_PC}, show_orbit=False)
    st.caption("Orbit geometry and miss distance are from the selected CDM; the collision risk is the forecast for the "
               "final CDM.")

# ---------------------------------------------------------------- all events
with tab_all:
    st.subheader("Forecast final Pc vs forecast final miss distance")
    pts = win.assign(miss_plot=win["pred_final_miss"].clip(lower=1.0), risk_plot=win["pred_final_risk"].clip(lower=-10))
    sc = alt.Chart(pts).mark_circle(size=70, opacity=0.8, stroke=BG, strokeWidth=1).encode(
        x=alt.X("miss_plot:Q", scale=alt.Scale(type="log"), title="Forecast final miss distance (m, log scale)"),
        y=alt.Y("risk_plot:Q", title="Forecast final log10 Pc"),
        color=alt.Color("pred_tier:N", scale=TIER_SCALE, title="Forecast tier"),
        tooltip=[alt.Tooltip("event_id:Q", title="event"), alt.Tooltip("pred_tier:N", title="forecast tier"),
                 alt.Tooltip("p_critical:Q", title="P(critical)", format=".0%"),
                 alt.Tooltip("forecast_pc:N", title="forecast Pc"),
                 alt.Tooltip("pred_final_miss:Q", title="forecast miss (m)", format=",.0f"),
                 alt.Tooltip("countdown:N", title="time to closest approach")])
    vr = pd.DataFrame({"x": [CRIT_MISS_M, ELEV_MISS_M], "label": ["1 km", "5 km"]})
    hr = pd.DataFrame({"y": [CRIT_LOG_PC, ELEV_LOG_PC], "label": ["Pc 1e-4", "Pc 1e-6"]})
    bounds = (alt.Chart(vr).mark_rule(strokeDash=[4, 4], color=RULE_INK).encode(x="x:Q")
              + alt.Chart(vr).mark_text(align="left", dx=4, y=8, color=RULE_INK).encode(x="x:Q", text="label:N")
              + alt.Chart(hr).mark_rule(strokeDash=[4, 4], color=RULE_INK).encode(y="y:Q")
              + alt.Chart(hr).mark_text(align="left", dy=-6, x=4, color=RULE_INK).encode(y="y:Q", text="label:N"))
    st.altair_chart((sc + bounds).properties(height=420), width="stretch")
    st.caption("Dashed lines are the tier boundaries. Colour is the classifier's forecast tier, from calibrated "
               "probabilities, so an event on the safe side of both lines can still be flagged when its trend points "
               "towards a close approach. Pc below 1e-10 is drawn at -10.")
    st.subheader(f"All {len(win):,} events in the window")
    cols = ["tier", "event_id", "countdown", "p_crit_pct", "forecast_pc", "pred_final_miss", "latest_pc",
            "miss_distance", "why", "rule_flip"] + (["final_tier_name"] if show_truth else [])
    st.dataframe(win[cols], hide_index=True, width="stretch", column_config={
        "tier": st.column_config.TextColumn("forecast tier"),
        "event_id": st.column_config.NumberColumn("event", format="%d"),
        "countdown": st.column_config.TextColumn("time to closest approach"),
        "p_crit_pct": st.column_config.ProgressColumn("P(critical)", min_value=0, max_value=100, format="%.0f%%"),
        "forecast_pc": st.column_config.TextColumn("forecast final Pc"),
        "pred_final_miss": st.column_config.NumberColumn("forecast final miss (m)", format="%,.0f"),
        "latest_pc": st.column_config.TextColumn("latest Pc"),
        "miss_distance": st.column_config.NumberColumn("latest miss (m)", format="%,.0f"),
        "why": st.column_config.TextColumn("why flagged"),
        "rule_flip": st.column_config.CheckboxColumn("tier changed"),
        "final_tier_name": st.column_config.TextColumn("true final tier")})

# ---------------------------------------------------------------- evaluation & method
with tab_eval:
    m, b = metrics["model"], metrics["persistence baseline"]
    st.markdown(f"Test set: **{metrics['test events']:,} events**, {metrics['events scored (2+ CDMs)']:,} scored "
                f"per event on the forecast made after the second-to-last CDM; "
                f"{metrics['events not forecastable (only 1 CDM)']} events have a single CDM and cannot be forecast. "
                f"Baseline: apply the tier rule to the latest CDM received (assume nothing changes).")
    keys = ["true critical", "critical recall", "critical precision", "critical F2", "criticals called safe",
            "safe events flagged", "accuracy", "macro F1"]
    tbl = pd.DataFrame({"model": [m[k] for k in keys], "persistence baseline": [b[k] for k in keys]}, index=keys)
    left, right = st.columns(2)
    with left:
        st.markdown("**Final-tier forecast, per event**")
        st.dataframe(tbl.style.format(lambda v: f"{v:,.0f}" if float(v).is_integer() else f"{v:.3f}"), width="stretch")
        st.markdown("**Regression RMSE (log10 units)**")
        st.dataframe(pd.Series(metrics["regression RMSE"], name="RMSE").round(3), width="stretch")
        st.markdown(f"**Probabilities:** log loss {metrics['probabilities']['log loss']:.3f}, "
                    f"multiclass Brier {metrics['probabilities']['Brier (multiclass)']:.3f}")
    with right:
        names = ["safe", "elevated", "critical"]
        key = "confusion (rows true, cols pred: safe, elevated, critical)"
        for title, mm in [("Model", m), ("Persistence baseline", b)]:
            st.markdown(f"**Confusion matrix, {title}** (rows true, columns predicted)")
            st.dataframe(pd.DataFrame(mm[key], index=[f"true {n}" for n in names], columns=names), width="stretch")

    st.subheader("Calibration of P(critical)")
    cal_line = alt.Chart(cal).mark_line(strokeWidth=2, color=BLUE, point=alt.OverlayMarkDef(size=80, color=BLUE)).encode(
        x=alt.X("mean_predicted:Q", title="Mean predicted P(critical)", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        y=alt.Y("observed_rate:Q", title="Observed share critical", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        tooltip=["bin", "events", alt.Tooltip("mean_predicted:Q", format=".1%"),
                 alt.Tooltip("observed_rate:Q", format=".1%")])
    diag = alt.Chart(pd.DataFrame({"a": [0, 1]})).mark_line(strokeDash=[4, 4], color=RULE_INK).encode(x="a:Q", y="a:Q")
    c1, c2 = st.columns([3, 2])
    c1.altair_chart((diag + cal_line).properties(height=300), width="stretch")
    c2.dataframe(cal, hide_index=True, width="stretch")
    c2.caption("Points on the dashed diagonal mean the probability can be read at face value. "
               "Bins with few events are noisy.")

    st.subheader("Feature choice")
    st.markdown("The brief's suggested columns describe one CDM. Two kinds of extra information were tested by "
                "grouped 5-fold cross-validation on the training data only: (B) the latest CDM's own Pc and tier "
                "plus how risk and miss distance have moved across the event's CDMs, and (C) the full position "
                "and velocity uncertainty and orbit geometry, since Pc depends on miss distance relative to the "
                "combined covariance. The set with the lowest calibrated log loss was chosen: "
                f"**{meta['feature set']}**.")
    st.dataframe(cv.drop(columns=["temperature"]).round(3), hide_index=True, width="stretch")
    bars = alt.Chart(imp.head(15)).mark_bar(color=BLUE, cornerRadiusEnd=4, height=14).encode(
        x=alt.X("gain share:Q", title="Share of total gain", axis=alt.Axis(format="%")),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", alt.Tooltip("gain share:Q", format=".1%")])
    st.markdown("**Top 15 features in the final classifier**")
    st.altair_chart(bars.properties(height=360), width="stretch")

    st.subheader("Known limits")
    st.markdown(f"""
- The test file stops 2 days before closest approach, so "final" means each event's last CDM at ≥ 2 days, not the
  CDM at closest approach. Training uses the same cut so both sets are built the same way.
- Features, temperature and thresholds were chosen on the training data only; the test set is scored by `evaluate.py`.
- The forecast Pc does not beat persistence (see the RMSE table); the tier classifier and the miss-distance forecast do.
- The globe has no real longitude: the data does not include it, so events are spread around the Earth evenly.
- Pc in the data is floored at 1e-30; below 1e-10 it is treated as 1e-10.
- Manoeuvre flag: P(critical) ≥ {meta['p_critical_cut']:.0%}, tuned for F2 (recall weighted over precision).
""")
