"""Checks for the tier rule, leakage and the saved outputs.

Run from the project root:  python tests/test_pipeline.py   (or  python -m pytest tests  if pytest is installed)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import CRITICAL, ELEVATED, SAFE, build, examples, tier_code, tier_metrics  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_tier_rule_boundaries():
    cases = [  # (risk, miss, expected)
        (-4.0, 10_000, CRITICAL), (-4.01, 10_000, ELEVATED),       # Pc 1e-4 is critical
        (-6.0, 10_000, ELEVATED), (-6.01, 10_000, SAFE),           # Pc 1e-6 is elevated
        (-30, 999.9, CRITICAL), (-30, 1000, ELEVATED),             # miss < 1 km is critical
        (-30, 5000, ELEVATED), (-30, 5000.1, SAFE),                # 5 km is still elevated
        (-3.0, 20_000, CRITICAL), (-5.0, 500, CRITICAL),           # either condition is enough
    ]
    got = tier_code([c[0] for c in cases], [c[1] for c in cases])
    assert got.tolist() == [c[2] for c in cases], got


def _toy_events():
    rng = np.random.default_rng(0)
    rows = []
    for e in range(6):
        for i, t in enumerate([6.0, 5.0, 4.0, 3.0, 2.2, 1.0]):           # the 1.0-day CDM must be dropped
            rows.append({"event_id": e, "time_to_tca": t, "risk": -8 + 0.5 * i + rng.normal(0, .1),
                         "miss_distance": 9000 - 1000 * i, "max_risk_estimate": -6.0, "relative_speed": 1e4,
                         "relative_position_r": 1.0, "relative_position_t": 1.0, "relative_position_n": 1.0,
                         "c_object_type": "DEBRIS", "F10": 100, "AP": 5,
                         "t_position_covariance_det": 1.0, "c_position_covariance_det": 1.0,
                         **{f"{o}_sigma_{a}": 10.0 for o in "tc" for a in ["r", "t", "n", "rdot", "tdot", "ndot"]},
                         **{c: 0.0 for c in ["mahalanobis_distance", "relative_velocity_r", "relative_velocity_t",
                                             "relative_velocity_n", "t_j2k_sma", "t_j2k_ecc", "t_j2k_inc", "c_j2k_sma",
                                             "c_j2k_ecc", "c_j2k_inc", "t_h_apo", "t_h_per", "c_h_apo", "c_h_per"]}})
    return pd.DataFrame(rows)


def test_features_do_not_see_the_final_cdm():
    raw = _toy_events()
    d = build(raw, ["DEBRIS"])
    assert d["time_to_tca"].min() >= 2.0
    assert (d.groupby("event_id")["is_final"].sum() == 1).all()
    ex = examples(d)
    assert len(ex) == 6 * 4 and not ex["is_final"].any()

    # change only the final CDM (2.2 days): features must not move, targets must
    raw2 = raw.copy()
    final = raw2["time_to_tca"] == 2.2
    raw2.loc[final, "risk"] = -2.0
    raw2.loc[final, "miss_distance"] = 50.0
    ex2 = examples(build(raw2, ["DEBRIS"]))
    target_cols = {c for c in ex.columns if c.startswith("final_") or c in {"n_total", "weight", "is_latest_input"}}
    feat_cols = [c for c in ex.columns if c not in target_cols]
    pd.testing.assert_frame_equal(ex[feat_cols], ex2[feat_cols])
    assert (ex2["final_tier"] == CRITICAL).all()


def test_grouped_folds_keep_events_apart():
    ex = examples(build(_toy_events(), ["DEBRIS"]))
    for tr, va in GroupKFold(n_splits=3).split(ex, groups=ex["event_id"]):
        assert not set(ex["event_id"].iloc[tr]) & set(ex["event_id"].iloc[va])


def test_saved_metrics_match_saved_predictions():
    path = os.path.join(ROOT, "outputs", "v2")
    if not os.path.exists(os.path.join(path, "test_metrics.json")):
        print("  (skipped: run train.py and evaluate.py first)")
        return
    m = json.load(open(os.path.join(path, "test_metrics.json")))
    meta = json.load(open(os.path.join(path, "meta.json")))
    u = pd.read_csv(os.path.join(path, "test_update_predictions.csv"))
    ev = u[u["is_latest_input"]]
    code = {"safe": SAFE, "elevated": ELEVATED, "critical": CRITICAL}
    true, pred = ev["final_tier_name"].map(code), ev["pred_tier"].map(code)
    recomputed = tier_metrics(true, pred)
    for k in ["critical recall", "critical precision", "criticals called safe", "safe events flagged"]:
        assert np.isclose(recomputed[k], m["model"][k]), k
    # the dashboard tier must follow from the stored probabilities and thresholds
    expect = np.where(u["p_critical"] >= meta["p_critical_cut"], "critical",
                      np.where(u["p_elevated"] + u["p_critical"] >= meta["p_elevated_cut"], "elevated", "safe"))
    assert (expect == u["pred_tier"]).all()
    # stored truth must agree with the tier rule
    assert (tier_code(ev["final_risk"], ev["final_miss"]) == true).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
