"""Train the final-tier forecaster on TRAIN ONLY. The test file is never opened here.

Steps
  1. Build forecasting examples: every CDM at >= 2 days except each event's final one, labelled
     with the tier (requirement rule: Pc and miss distance) of the event's final CDM.
  2. Grouped 5-fold CV by event compares the feature sets in common.FEATURE_SETS. The set with
     the lowest calibrated log loss on each event's latest-input example is chosen.
  3. On the chosen set's out-of-fold predictions: fit a temperature for calibration, then tune
     the probability thresholds for critical and elevated-or-worse by F2 (a miss costs more
     than a false alarm).
  4. Refit the classifier and two regressors (final log10 Pc, final miss distance) on all of
     train and save everything to outputs/v2/.

Run from the project root:  python train.py   then   python evaluate.py
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize_scalar
from sklearn.metrics import fbeta_score, log_loss
from sklearn.model_selection import GroupKFold

from common import (CRITICAL, FEATURE_SETS, MIN_TCA, RISK_FLOOR, build, calibration_table, decide,
                    examples, prob_metrics, softmax, tier_metrics)

OUT = "outputs/v2"
PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
              tree_method="hist", enable_categorical=True, n_jobs=-1, random_state=0)
GRID = np.round(np.arange(0.02, 0.91, 0.01), 2)

raw = pd.read_csv("data/train_data.csv")
categories = sorted(raw["c_object_type"].dropna().unique().tolist())
d = build(raw, categories)
ex = examples(d)
y, w, groups = ex["final_tier"].values, ex["weight"].values, ex["event_id"].values
last = ex["is_latest_input"].values
print(f"train: {raw['event_id'].nunique():,} events | {d['event_id'].nunique():,} with a CDM at >= {MIN_TCA} days | "
      f"{ex['event_id'].nunique():,} with 2+ such CDMs -> {len(ex):,} examples")
print("final-tier balance per event (safe, elevated, critical):", np.bincount(y[last], minlength=3).tolist())

folds = list(GroupKFold(n_splits=5).split(ex, groups=groups))


def fit_temperature(margin, true, weight):
    return float(minimize_scalar(lambda t: log_loss(true, softmax(margin / t), labels=[0, 1, 2], sample_weight=weight),
                                 bounds=(0.2, 5.0), method="bounded").x)


def oof_margin(feats):
    m = np.zeros((len(ex), 3))
    for tr, va in folds:
        clf = xgb.XGBClassifier(objective="multi:softprob", **PARAMS)
        clf.fit(ex.iloc[tr][feats], y[tr], sample_weight=w[tr])
        m[va] = clf.predict(ex.iloc[va][feats], output_margin=True)
    return m


def oof_regression(feats, target):
    pred = np.zeros(len(ex))
    for tr, va in folds:
        reg = xgb.XGBRegressor(**PARAMS)
        reg.fit(ex.iloc[tr][feats], ex[target].values[tr], sample_weight=w[tr])
        pred[va] = reg.predict(ex.iloc[va][feats])
    return pred


def tune(p, true):
    t_crit = max(GRID, key=lambda t: fbeta_score(true == CRITICAL, p[:, CRITICAL] >= t, beta=2))
    t_elev = max(GRID, key=lambda t: fbeta_score(true >= 1, p[:, 1] + p[:, 2] >= t, beta=2))
    return float(t_crit), float(t_elev)


# ---- 2. compare feature sets (all numbers are out-of-fold, train only)
rows, margins = [], {}
for name, feats in FEATURE_SETS.items():
    print("CV:", name, flush=True)
    m = oof_margin(feats)
    temp = fit_temperature(m, y, w)
    p = softmax(m / temp)
    t_crit, t_elev = tune(p[last], y[last])
    tm = tier_metrics(y[last], decide(p[last], t_crit, t_elev))
    margins[name] = (m, temp)
    rows.append({"feature set": name, "n features": len(feats), "temperature": temp,
                 **prob_metrics(y[last], p[last]),
                 **{k: v for k, v in tm.items() if not k.startswith("confusion")}})
cv = pd.DataFrame(rows)
best = cv.sort_values("log loss").iloc[0]["feature set"]
feats = FEATURE_SETS[best]
m, temp = margins[best]
p = softmax(m / temp)
t_crit, t_elev = tune(p[last], y[last])
print(f"\nchosen: {best} | temperature {temp:.3f} | P(critical) cut {t_crit} | P(elevated or worse) cut {t_elev}")

# persistence baseline: the tier rule applied to the latest CDM received
oof = {"model": tier_metrics(y[last], decide(p[last], t_crit, t_elev)),
       "persistence baseline": tier_metrics(y[last], ex["tier_now"].values[last])}

# ---- regressors for final log10 Pc and final miss distance (out-of-fold RMSE vs persistence)
print("CV: regressors", flush=True)
risk_oof = ex["risk_c"].values + oof_regression(feats, "final_risk_change")
miss_oof = ex["log_miss"].values + oof_regression(feats, "final_miss_change")
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
yr, ym = ex["final_risk_c"].values[last], ex["final_log_miss"].values[last]
oof["regression RMSE"] = {
    f"final log10 Pc (floored at {RISK_FLOOR:g}), model": rmse(risk_oof[last], yr),
    "final log10 Pc, persistence": rmse(ex["risk_c"].values[last], yr),
    "final log10(miss+1 m), model": rmse(miss_oof[last], ym),
    "final log10(miss+1 m), persistence": rmse(ex["log_miss"].values[last], ym),
}

# ---- 4. refit on all of train
print("fitting final models on all of train", flush=True)
os.makedirs(OUT, exist_ok=True)
clf = xgb.XGBClassifier(objective="multi:softprob", **PARAMS).fit(ex[feats], y, sample_weight=w)
reg_risk = xgb.XGBRegressor(**PARAMS).fit(ex[feats], ex["final_risk_change"], sample_weight=w)
reg_miss = xgb.XGBRegressor(**PARAMS).fit(ex[feats], ex["final_miss_change"], sample_weight=w)
clf.save_model(f"{OUT}/classifier.json")
reg_risk.save_model(f"{OUT}/reg_final_risk.json")
reg_miss.save_model(f"{OUT}/reg_final_miss.json")

imp = pd.Series(clf.get_booster().get_score(importance_type="gain")).reindex(feats).fillna(0.0)
imp = (imp / imp.sum()).sort_values(ascending=False)
imp.rename("gain share").to_csv(f"{OUT}/feature_importance.csv", index_label="feature")
cv.to_csv(f"{OUT}/cv_feature_sets.csv", index=False)
calibration_table(y[last] == CRITICAL, p[last, CRITICAL]).to_csv(f"{OUT}/cv_calibration.csv", index=False)

meta = {"feature set": best, "features": feats, "categories": categories, "temperature": temp,
        "p_critical_cut": t_crit, "p_elevated_cut": t_elev, "min_tca_days": MIN_TCA,
        "risk_floor": RISK_FLOOR, "xgb_params": PARAMS, "cv_out_of_fold": oof}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)

pd.set_option("display.width", 220)
print("\nGrouped 5-fold CV on train, per event (latest-input example):\n")
print(cv.drop(columns=["temperature"]).round(3).to_string(index=False))
for k in ["model", "persistence baseline"]:
    print(f"\n{k}:", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in oof[k].items()})
print("\n", {k: round(v, 3) for k, v in oof["regression RMSE"].items()})
print("\ntop features:\n", imp.head(10).round(3).to_string())
print(f"\nsaved models and meta.json in {OUT}/. Now run: python evaluate.py")
