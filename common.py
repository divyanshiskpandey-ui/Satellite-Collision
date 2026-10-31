"""Shared pieces: the tier rule, example building, feature lists and metrics.

train.py, evaluate.py, app.py and the tests all import from here, so the tier rule and the
features cannot drift apart between training and the dashboard.

The task: for each event, forecast the tier of its FINAL CDM from the CDMs received before it.
The test file stops 2 days before closest approach, so "final" means the event's last CDM at
>= 2 days. Train is cut at the same point so train and test examples are built the same way.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, log_loss

# ---------------------------------------------------------------- tier rule (requirements)
CRIT_LOG_PC, ELEV_LOG_PC = -4.0, -6.0          # Pc >= 1e-4 critical, Pc >= 1e-6 elevated
CRIT_MISS_M, ELEV_MISS_M = 1000.0, 5000.0      # miss < 1 km critical, 1-5 km elevated
SAFE, ELEVATED, CRITICAL = 0, 1, 2
TIER_NAMES = np.array(["safe", "elevated", "critical"])

MIN_TCA = 2.0          # days; the test data has no CDMs closer to TCA than this
RISK_FLOOR = -10.0     # log10 Pc below this is treated as -10 (the data uses -30 for "Pc ~ 0")


def tier_code(risk, miss):
    """Critical: risk >= -4 OR miss < 1000 m. Elevated: risk >= -6 OR 1000 <= miss <= 5000 m.
    Safe: risk < -6 AND miss > 5000 m. Returns 0 safe, 1 elevated, 2 critical."""
    risk, miss = np.asarray(risk, dtype=float), np.asarray(miss, dtype=float)
    return np.where((risk >= CRIT_LOG_PC) | (miss < CRIT_MISS_M), CRITICAL,
                    np.where((risk >= ELEV_LOG_PC) | (miss <= ELEV_MISS_M), ELEVATED, SAFE))


def decide(p, t_crit, t_elev):
    """Tier from class probabilities p (n, 3). Critical takes precedence, then elevated-or-worse."""
    return np.where(p[:, CRITICAL] >= t_crit, CRITICAL,
                    np.where(p[:, ELEVATED] + p[:, CRITICAL] >= t_elev, ELEVATED, SAFE))


def softmax(margin):
    e = np.exp(margin - margin.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- features
# suggested in the brief, taken from the latest CDM received
SUGGESTED = ["time_to_tca", "miss_distance", "relative_speed", "relative_position_r",
             "relative_position_t", "relative_position_n", "c_object_type",
             "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP"]
# the latest CDM's own Pc and tier, plus how risk and miss distance have moved so far in the event
TREND = ["risk_c", "tier_now", "tier_max", "n_cdms", "obs_span",
         "risk_delta", "risk_slope", "risk_max", "risk_first",
         "log_miss", "miss_delta", "miss_slope", "miss_min", "miss_first",
         "mre_delta", "mre_slope", "mre_max"]
# full position/velocity uncertainty and orbit geometry: Pc depends on miss distance relative
# to the combined covariance, which the two radial sigmas alone cannot describe
EXTENDED = ["t_sigma_t", "t_sigma_n", "c_sigma_t", "c_sigma_n",
            "t_sigma_rdot", "c_sigma_rdot", "t_sigma_tdot", "c_sigma_tdot",
            "t_sigma_ndot", "c_sigma_ndot", "mahalanobis_distance", "log_t_cov_det", "log_c_cov_det",
            "sigma_comb_max", "miss_over_sigma",
            "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
            "t_j2k_sma", "t_j2k_ecc", "t_j2k_inc", "c_j2k_sma", "c_j2k_ecc", "c_j2k_inc",
            "t_h_apo", "t_h_per", "c_h_apo", "c_h_per"]
FEATURE_SETS = {
    "A suggested columns": SUGGESTED,
    "B suggested + latest Pc and trends": SUGGESTED + TREND,
    "C B + covariance and geometry": SUGGESTED + TREND + EXTENDED,
}


def build(raw, categories):
    """One row per CDM at >= MIN_TCA days, sorted oldest first within each event.

    Each row's features use only that CDM and earlier CDMs of the same event. The final CDM's
    values appear only in the final_* target columns. n_total is future information and is used
    for sample weights and evaluation flags only, never as a feature.
    """
    d = raw[raw["time_to_tca"] >= MIN_TCA].copy()
    d = d.sort_values(["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort")
    d = d.reset_index(drop=True)

    d["risk_c"] = d["risk"].clip(lower=RISK_FLOOR)
    d["log_miss"] = np.log10(d["miss_distance"] + 1.0)
    d["log_t_cov_det"] = np.log10(d["t_position_covariance_det"].clip(lower=1e-30))
    d["log_c_cov_det"] = np.log10(d["c_position_covariance_det"].clip(lower=1e-30))
    comb = np.hypot(d[["t_sigma_r", "t_sigma_t", "t_sigma_n"]].values,
                    d[["c_sigma_r", "c_sigma_t", "c_sigma_n"]].values)
    d["sigma_comb_max"] = comb.max(axis=1)
    d["miss_over_sigma"] = d["miss_distance"] / (d["sigma_comb_max"] + 1e-9)
    d["tier_now"] = tier_code(d["risk"], d["miss_distance"])

    g = d.groupby("event_id", sort=False)
    d["n_cdms"] = g.cumcount() + 1
    d["n_total"] = g["event_id"].transform("size")
    d["obs_span"] = g["time_to_tca"].transform("first") - d["time_to_tca"]
    dt = g["time_to_tca"].shift(1) - d["time_to_tca"]
    dt = dt.where(dt > 1e-6)
    for col, name in [("risk_c", "risk"), ("log_miss", "miss"), ("max_risk_estimate", "mre")]:
        d[f"{name}_delta"] = d[col] - g[col].shift(1)
        d[f"{name}_slope"] = d[f"{name}_delta"] / dt
    d["risk_max"] = g["risk_c"].cummax()
    d["risk_first"] = g["risk_c"].transform("first")
    d["miss_min"] = g["log_miss"].cummin()
    d["miss_first"] = g["log_miss"].transform("first")
    d["mre_max"] = g["max_risk_estimate"].cummax()
    d["tier_max"] = g["tier_now"].cummax()

    final = d.loc[d["n_cdms"] == d["n_total"]].set_index("event_id")[["time_to_tca", "risk", "miss_distance"]]
    final.columns = ["final_tca", "final_risk", "final_miss"]
    d = d.join(final, on="event_id")
    d["final_tier"] = tier_code(d["final_risk"], d["final_miss"])
    d["final_risk_c"] = d["final_risk"].clip(lower=RISK_FLOOR)
    d["final_log_miss"] = np.log10(d["final_miss"] + 1.0)
    # regression targets are the change from the latest CDM: trees predict "about the same as now"
    # far better as a residual than as a level spread over 10 orders of magnitude
    d["final_risk_change"] = d["final_risk_c"] - d["risk_c"]
    d["final_miss_change"] = d["final_log_miss"] - d["log_miss"]
    d["is_final"] = d["n_cdms"] == d["n_total"]
    d["is_latest_input"] = d["n_cdms"] == d["n_total"] - 1
    d["weight"] = 1.0 / (d["n_total"] - 1).clip(lower=1)      # every event counts the same

    # XGBoost needs finite values that fit in float32
    feats = sorted(set(sum(FEATURE_SETS.values(), [])) - {"c_object_type"})
    d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).mask(d[feats].abs() > 1e30)
    d["c_object_type"] = pd.Categorical(d["c_object_type"], categories=categories)
    return d


def examples(d):
    """Forecasting examples: every CDM except each event's final one (the target)."""
    return d[~d["is_final"]].reset_index(drop=True)


# ---------------------------------------------------------------- metrics
def tier_metrics(true, pred):
    true, pred = np.asarray(true), np.asarray(pred)
    crit, safe = true == CRITICAL, true == SAFE
    pred_crit = pred == CRITICAL
    return {
        "events": int(len(true)),
        "true critical": int(crit.sum()),
        "critical recall": float((pred_crit & crit).sum() / max(crit.sum(), 1)),
        "critical precision": float((pred_crit & crit).sum() / max(pred_crit.sum(), 1)),
        "critical F2": float(fbeta_score(crit, pred_crit, beta=2, zero_division=0)),
        "criticals called safe": int((crit & (pred == SAFE)).sum()),
        "safe events flagged": float((pred[safe] != SAFE).mean()) if safe.any() else 0.0,
        "accuracy": float((true == pred).mean()),
        "macro F1": float(f1_score(true, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "confusion (rows true, cols pred: safe, elevated, critical)":
            confusion_matrix(true, pred, labels=[0, 1, 2]).tolist(),
    }


def prob_metrics(true, p, weight=None):
    onehot = np.eye(3)[np.asarray(true)]
    brier = ((p - onehot) ** 2).sum(axis=1)
    return {"log loss": float(log_loss(true, p, labels=[0, 1, 2], sample_weight=weight)),
            "Brier (multiclass)": float(np.average(brier, weights=weight))}


def calibration_table(true_crit, p_crit, edges=(0, .05, .1, .2, .4, .6, .8, 1.0001)):
    b = pd.cut(p_crit, edges, right=False)
    t = pd.DataFrame({"bin": b, "p": p_crit, "y": np.asarray(true_crit, dtype=float)})
    t = t.groupby("bin", observed=True).agg(events=("y", "size"), mean_predicted=("p", "mean"),
                                            observed_rate=("y", "mean")).reset_index()
    t["bin"] = t["bin"].astype(str)
    return t
