"""Score the TEST set with the models from train.py and write the files the dashboard reads.

Nothing here is tuned: features, temperature and thresholds all come from outputs/v2/meta.json.
Headline metrics are per event, on each event's latest-input example (all CDMs before the final one).

Run from the project root:  python evaluate.py
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from common import (CRITICAL, TIER_NAMES, build, calibration_table, decide, examples, prob_metrics,
                    softmax, tier_metrics)

OUT = "outputs/v2"
ORBIT_COLS = ["relative_position_r", "relative_position_t", "relative_position_n",
              "relative_velocity_r", "relative_velocity_t", "relative_velocity_n", "relative_speed",
              "t_j2k_sma", "t_j2k_ecc", "c_j2k_sma", "t_sigma_r", "t_sigma_t", "t_sigma_n",
              "c_sigma_r", "c_sigma_t", "c_sigma_n",
              "geocentric_latitude", "t_h_apo", "t_h_per", "t_j2k_inc", "c_object_type",
              "t_ct_r", "t_cn_r", "t_cn_t", "c_ct_r", "c_cn_r", "c_cn_t"]

meta = json.load(open(f"{OUT}/meta.json"))
feats = meta["features"]
clf, reg_risk, reg_miss = xgb.XGBClassifier(), xgb.XGBRegressor(), xgb.XGBRegressor()
clf.load_model(f"{OUT}/classifier.json")
reg_risk.load_model(f"{OUT}/reg_final_risk.json")
reg_miss.load_model(f"{OUT}/reg_final_miss.json")

raw = pd.read_csv("data/test_data.csv")
d = build(raw, meta["categories"])
ex = examples(d)

p = softmax(clf.predict(ex[feats], output_margin=True) / meta["temperature"])
pred = decide(p, meta["p_critical_cut"], meta["p_elevated_cut"])
ex["p_safe"], ex["p_elevated"], ex["p_critical"] = p[:, 0], p[:, 1], p[:, 2]
ex["pred_code"] = pred
ex["pred_tier"] = TIER_NAMES[pred]
ex["maneuver_flag"] = pred == CRITICAL
ex["now_tier"] = TIER_NAMES[ex["tier_now"].values]
# regressors predict the change from the latest CDM (see common.build)
ex["pred_final_risk"] = ex["risk_c"] + reg_risk.predict(ex[feats])
ex["pred_final_miss"] = np.clip(10 ** (ex["log_miss"] + reg_miss.predict(ex[feats])) - 1.0, 0.0, None)
prev = ex.groupby("event_id")["pred_tier"].shift(1)
ex["rule_flip"] = prev.notna() & (prev != ex["pred_tier"])
ex["final_tier_name"] = TIER_NAMES[ex["final_tier"].values]

# ---- per-event scoring on the latest-input example
ev = ex[ex["is_latest_input"]]
y = ev["final_tier"].values
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
single = int((d.groupby("event_id")["n_total"].first() == 1).sum())
metrics = {
    "test events": int(raw["event_id"].nunique()),
    "events scored (2+ CDMs)": int(len(ev)),
    "events not forecastable (only 1 CDM)": single,
    "feature set": meta["feature set"],
    "thresholds": {"P(critical)": meta["p_critical_cut"], "P(elevated or worse)": meta["p_elevated_cut"]},
    "model": tier_metrics(y, ev["pred_code"]),
    "persistence baseline": tier_metrics(y, ev["tier_now"]),
    "probabilities": prob_metrics(y, ev[["p_safe", "p_elevated", "p_critical"]].values),
    "regression RMSE": {
        "final log10 Pc, model": rmse(ev["pred_final_risk"], ev["final_risk_c"]),
        "final log10 Pc, persistence": rmse(ev["risk_c"], ev["final_risk_c"]),
        "final log10(miss+1 m), model": rmse(np.log10(ev["pred_final_miss"] + 1), ev["final_log_miss"]),
        "final log10(miss+1 m), persistence": rmse(ev["log_miss"], ev["final_log_miss"]),
    },
    # every update, not just the latest: how early the final tier is forecast correctly
    "all updates": {"model": tier_metrics(ex["final_tier"], ex["pred_code"]),
                    "persistence baseline": tier_metrics(ex["final_tier"], ex["tier_now"])},
}
json.dump(metrics, open(f"{OUT}/test_metrics.json", "w"), indent=2)
calibration_table(y == CRITICAL, ev["p_critical"].values).to_csv(f"{OUT}/test_calibration.csv", index=False)

keep = (["event_id", "time_to_tca", "n_cdms", "is_latest_input", "risk", "risk_c", "miss_distance", "now_tier",
         "p_safe", "p_elevated", "p_critical", "pred_tier", "maneuver_flag", "rule_flip",
         "pred_final_risk", "pred_final_miss", "final_tca", "final_risk", "final_miss", "final_tier_name"]
        + ORBIT_COLS)
ex[keep].to_csv(f"{OUT}/test_update_predictions.csv", index=False)
d[["event_id", "time_to_tca", "risk", "miss_distance", "is_final"]].to_csv(f"{OUT}/test_cdm_history.csv", index=False)


def fmt(m):
    return "\n".join(f"  {k}: {round(v, 3) if isinstance(v, float) else v}" for k, v in m.items())


text = "\n".join([
    f"Test: {metrics['test events']:,} events, {metrics['events scored (2+ CDMs)']:,} scored, "
    f"{single} with a single CDM (no forecast possible).",
    f"Feature set: {meta['feature set']}. Thresholds: {metrics['thresholds']}",
    "\nMODEL, per event (forecast of the final CDM's tier from earlier CDMs):", fmt(metrics["model"]),
    "\nPERSISTENCE BASELINE (tier rule on the latest CDM received):", fmt(metrics["persistence baseline"]),
    "\nProbabilities:", fmt(metrics["probabilities"]),
    "\nRegression RMSE:", fmt(metrics["regression RMSE"]),
])
open(f"{OUT}/test_metrics.txt", "w").write(text)
print(text)
print(f"\nsaved {OUT}/test_metrics.json, test_metrics.txt, test_update_predictions.csv, test_cdm_history.csv")
