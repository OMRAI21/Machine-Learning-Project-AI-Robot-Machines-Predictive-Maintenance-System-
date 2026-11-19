
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, recall_score, precision_score, mean_absolute_error, accuracy_score

RNG = np.random.default_rng(42)
HORIZON = 14          # days: "will fail soon" window
RUL_CAP = 120         # days: RUL above this is reported as "120+"
MODES = ["bearing_wear", "gearbox_wear", "motor_overheat", "lubrication_loss"]

FEATURES = [
    "motor_temp_c", "gearbox_oil_temp_c", "vibration_mm_s", "motor_current_a",
    "torque_ripple_pct", "position_error_mm", "oil_particles_ppm",
    "load_pct", "ambient_temp_c", "operating_hours", "days_since_service",
]


def simulate_robot(robot_id, mode, life_days, n_days=None):
    """Daily readings of one robot from new until failure at `life_days`."""
    n = life_days if n_days is None else n_days
    t = np.arange(n)
    load = np.clip(RNG.normal(65, 10), 35, 95)
    amb = RNG.normal(28, 4)
    prog = (t / life_days)
    d = prog ** 2.3                       # slow early, fast late (typical wear curve)
    daily_load = np.clip(load + RNG.normal(0, 6, n), 20, 100)
    daily_amb = amb + 3 * np.sin(t / 58) + RNG.normal(0, 1.2, n)

    motor_temp = 48 + 0.22 * daily_load + 0.6 * (daily_amb - 25)
    oil_temp = 42 + 0.12 * daily_load + 0.5 * (daily_amb - 25)
    vib = 1.2 + 0.012 * daily_load
    cur = 6.0 + 0.055 * daily_load
    ripple = 1.5 + 0.01 * daily_load
    pos = 0.020 + 0.0002 * daily_load
    part = 8 + 0.05 * daily_load

    if mode == "bearing_wear":
        vib += 6.5 * d; motor_temp += 6 * d; ripple += 1.5 * d; part += 20 * d
    elif mode == "gearbox_wear":
        pos += 0.28 * d; vib += 3.2 * d; part += 140 * d; oil_temp += 9 * d; ripple += 3 * d
    elif mode == "motor_overheat":
        motor_temp += 34 * d; cur += 5.2 * d; ripple += 2 * d; vib += 0.8 * d
    elif mode == "lubrication_loss":
        oil_temp += 24 * d; vib += 2.2 * d; part += 60 * d; ripple += 2.5 * d; pos += 0.05 * d

    motor_temp += RNG.normal(0, 1.0, n); oil_temp += RNG.normal(0, 1.0, n)
    vib *= RNG.lognormal(0, 0.06, n); cur += RNG.normal(0, 0.18, n)
    ripple += RNG.normal(0, 0.12, n); pos += RNG.normal(0, 0.004, n)
    part *= RNG.lognormal(0, 0.10, n)

    service_every = RNG.integers(60, 110)
    return pd.DataFrame({
        "robot_id": robot_id, "day": t, "mode": mode,
        "motor_temp_c": motor_temp, "gearbox_oil_temp_c": oil_temp,
        "vibration_mm_s": vib, "motor_current_a": cur, "torque_ripple_pct": ripple,
        "position_error_mm": np.clip(pos, 0.005, None), "oil_particles_ppm": np.clip(part, 1, None),
        "load_pct": daily_load, "ambient_temp_c": daily_amb,
        "operating_hours": t * 16 + RNG.integers(0, 3000),
        "days_since_service": t % service_every,
        "rul_days": life_days - t,
    })


def simulate_fleet(n_robots=420):
    frames = []
    for i in range(n_robots):
        mode = MODES[i % len(MODES)]
        life = int(RNG.integers(130, 420))
        frames.append(simulate_robot(i, mode, life))
    return pd.concat(frames, ignore_index=True)


def export_forest(model, value_fn, digits=4):
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "f": t.feature.tolist(),
            "t": [round(float(x), digits) for x in t.threshold],
            "v": [value_fn(v) for v in t.value],
        })
    return trees


def main():
    df = simulate_fleet()
    df["fail_soon"] = (df.rul_days <= HORIZON).astype(int)
    df["rul_target"] = df.rul_days.clip(upper=RUL_CAP)

    X, groups = df[FEATURES], df.robot_id
    tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=1).split(X, groups=groups))
    Xtr, Xte = X.iloc[tr], X.iloc[te]

    clf = RandomForestClassifier(n_estimators=40, max_depth=8, min_samples_leaf=60,
                                 random_state=0, n_jobs=-1)
    clf.fit(Xtr, df.fail_soon.iloc[tr])
    p = clf.predict_proba(Xte)[:, 1]
    yb = df.fail_soon.iloc[te]

    reg = RandomForestRegressor(n_estimators=40, max_depth=9, min_samples_leaf=60,
                                random_state=0, n_jobs=-1)
    reg.fit(Xtr, df.rul_target.iloc[tr])
    r = reg.predict(Xte)
    near = df.rul_days.iloc[te] <= 60

    deg = df.rul_days <= 90
    tr_m = [i for i in tr if deg.iloc[i]]; te_m = [i for i in te if deg.iloc[i]]
    mode_clf = RandomForestClassifier(n_estimators=40, max_depth=8, min_samples_leaf=50,
                                      random_state=0, n_jobs=-1)
    mode_clf.fit(X.iloc[tr_m], df["mode"].iloc[tr_m])
    mpred = mode_clf.predict(X.iloc[te_m])

    metrics = {
        "horizon_days": HORIZON, "rul_cap_days": RUL_CAP,
        "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
        "n_robots": int(df.robot_id.nunique()),
        "risk_auc": round(float(roc_auc_score(yb, p)), 4),
        "risk_recall": round(float(recall_score(yb, p >= 0.5)), 4),
        "risk_precision": round(float(precision_score(yb, p >= 0.5)), 4),
        "rul_mae_days_all": round(float(mean_absolute_error(df.rul_target.iloc[te], r)), 2),
        "rul_mae_days_last60": round(float(mean_absolute_error(df.rul_days.iloc[te][near], r[near.values])), 2),
        "mode_accuracy": round(float(accuracy_score(df["mode"].iloc[te_m], mpred)), 4),
    }
    print(json.dumps(metrics, indent=2))


    clf.fit(X, df.fail_soon); reg.fit(X, df.rul_target); mode_clf.fit(X[deg], df["mode"][deg])

    healthy = df[df.rul_days > 150]
    baseline = {f: {"mean": round(float(healthy[f].mean()), 4), "std": round(float(healthy[f].std()), 4)}
                for f in FEATURES}
    ranges = {f: [round(float(df[f].quantile(0.001)), 3), round(float(df[f].quantile(0.999)) * 1.1, 3)]
              for f in FEATURES}

    model = {
        "features": FEATURES, "modes": list(mode_clf.classes_),
        "horizon_days": HORIZON, "rul_cap_days": RUL_CAP,
        "baseline": baseline, "ranges": ranges, "metrics": metrics,
        "importance": {f: round(float(v), 4) for f, v in zip(FEATURES, clf.feature_importances_)},
        "risk": export_forest(clf, lambda v: round(float(v[0][1] / v[0].sum()), 4)),
        "rul": export_forest(reg, lambda v: round(float(v[0][0]), 2)),
        "mode": export_forest(mode_clf, lambda v: [round(float(x), 3) for x in (v[0] / v[0].sum())]),
    }
    with open("model.json", "w") as f:
        json.dump(model, f, separators=(",", ":"))

    
    fleet = []
    specs = [  # name, type, site, mode, days-to-failure at "today"
        ("WLD-101", "Welding · 6-axis", "Body Shop A", "bearing_wear", 9),
        ("WLD-102", "Welding · 6-axis", "Body Shop A", "motor_overheat", 190),
        ("PNT-201", "Paint sprayer", "Paint Line 2", "lubrication_loss", 23),
        ("PNT-202", "Paint sprayer", "Paint Line 2", "gearbox_wear", 210),
        ("ASM-301", "Assembly · SCARA", "Final Assembly", "gearbox_wear", 41),
        ("ASM-302", "Assembly · SCARA", "Final Assembly", "bearing_wear", 260),
        ("MH-401", "Material handling", "Logistics", "motor_overheat", 17),
        ("MH-402", "Material handling", "Logistics", "lubrication_loss", 175),
        ("PCK-501", "Palletizing", "Packaging", "gearbox_wear", 5),
        ("PCK-502", "Palletizing", "Packaging", "bearing_wear", 120),
        ("WLD-103", "Welding · 6-axis", "Body Shop B", "lubrication_loss", 66),
        ("ASM-303", "Assembly · SCARA", "Final Assembly", "motor_overheat", 300),
    ]
    for rid, (name, rtype, site, mode, dtf) in enumerate(specs):
        life = int(dtf + RNG.integers(150, 260))
        full = simulate_robot(9000 + rid, mode, life)
        today = life - dtf
        hist = full.iloc[max(0, today - 44): today + 1].reset_index(drop=True)
        n_h = len(hist)
        fleet.append({
            "id": name, "type": rtype, "site": site,
            "history": [{"offset": i - (n_h - 1), **{f: round(float(hist.at[i, f]), 3) for f in FEATURES}}
                        for i in range(n_h)],
        })
    with open("fleet.json", "w") as f:
        json.dump(fleet, f, separators=(",", ":"))


    rows = []
    for rbt in fleet:
        last = rbt["history"][-1]
        rows.append({"robot_id": rbt["id"], **{k: last[k] for k in FEATURES}})
    pd.DataFrame(rows).to_csv("sample_readings.csv", index=False)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
