"""
Drone Design Assistant - standalone version (no training needed).
Needs: drone_design_model.joblib in the same folder.
Install:  pip install numpy pandas joblib scikit-learn==1.6.1   (same version the model was trained with)
Run:      python drone_design.py
Or import: from drone_design import design_drone, check_build
"""
import os, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLE = joblib.load(os.path.join(_HERE, "drone_design_model.joblib"))
MODELS = {k: _BUNDLE[k] for k in ("multirotor", "fixedwing")}
PROP_MODEL = _BUNDLE.get("prop_model")   # propeller CT/CP model learned from the UIUC Propeller Database

def prop_coeffs(pd_ratio, D_in, tip):
    pd_ratio, D_in, tip = map(np.asarray, (pd_ratio, D_in, tip))
    if PROP_MODEL is not None:
        X = np.column_stack([pd_ratio.ravel(), np.clip(D_in, 2, 30).ravel(), tip.ravel()])
        ct = PROP_MODEL["ct"].predict(X).reshape(pd_ratio.shape)
        cp = PROP_MODEL["cp"].predict(X).reshape(pd_ratio.shape)
    else:
        ct = 0.06 + 0.10 * pd_ratio
        cp = 0.012 + 0.08 * pd_ratio
    return np.clip(ct, 0.04, 0.25), np.clip(cp, 0.01, 0.15)


# ---------------- Physics (used to double-check the ML suggestions) ----------------
RHO, G = 1.225, 9.81
ETA_MOTOR = 0.65          # motor + ESC efficiency (realistic at hover/cruise load)
USABLE = 0.80             # usable battery fraction
LIPO_WH_PER_KG = 140.0   # LiPo pack incl. wiring/case
CELL_V = 3.7

def _component_masses(torque_nm, I_max_motor, n_mot, Wh):
    # motor mass scales with max torque (fit to typical 2306 / 2212 / 4114 / 8110 motors)
    motor_kg = n_mot * (8 + 115 * torque_nm**0.74) / 1000
    esc_kg = n_mot * (8 + 0.5 * I_max_motor) / 1000
    batt_kg = Wh / LIPO_WH_PER_KG
    return motor_kg, esc_kg, batt_kg

def sim_multirotor(d):
    """d: dict of arrays: n_motors, D_in, P_in, kv, cells, mah, payload_kg"""
    n, D_in, P_in, kv, cells, mah, pay = (np.asarray(d[k], float) for k in
        ["n_motors", "D_in", "P_in", "kv", "cells", "mah", "payload_kg"])
    D = D_in * 0.0254; V = cells * CELL_V; pdr = P_in / D_in
    rpm_max = kv * V * 0.80                      # loaded RPM at full throttle
    n_max = rpm_max / 60
    tip_max = np.pi * D * n_max
    ct_m, cp_m = prop_coeffs(pdr, D_in, tip_max)
    T_max = ct_m * RHO * n_max**2 * D**4          # N per motor
    P_mech_max = cp_m * RHO * n_max**3 * D**5
    P_el_max = P_mech_max / ETA_MOTOR
    I_max = P_el_max / V
    Wh = mah / 1000 * V
    wheelbase_mm = D_in * 25.4 / np.sin(np.pi / n) * 1.1
    frame_kg = 0.0064 * wheelbase_mm**1.79 / 1000
    avionics_kg = (50 + 0.03 * wheelbase_mm) / 1000
    m_mot, m_esc, m_bat = _component_masses(P_mech_max / (2 * np.pi * n_max), I_max, n, Wh)
    auw = frame_kg + avionics_kg + m_mot + m_esc + m_bat + pay
    T_h = auw * G / n
    # hover rpm (2 fixed-point iterations because CT depends on tip speed)
    ct_h = ct_m
    for _ in range(2):
        n_h = np.sqrt(T_h / (ct_h * RHO * D**4))
        ct_h, cp_h = prop_coeffs(pdr, D_in, np.pi * D * n_h)
    P_h = 1.2 * n * cp_h * RHO * n_h**3 * D**5 / ETA_MOTOR   # 1.2 = frame blockage, rotor interference, wind margin
    I_h = P_h / V + 0.5
    flight_min = (mah / 1000 * USABLE) / I_h * 60
    out = pd.DataFrame(dict(
        auw_kg=auw, flight_min=flight_min, tw_ratio=n * T_max / (auw * G),
        hover_throttle_pct=100 * n_h / n_max, hover_power_w=P_h,
        max_current_a=n * I_max, esc_rating_a=np.ceil(I_max * 1.25),
        motor_max_w=P_el_max, wheelbase_mm=wheelbase_mm, tip_mach=tip_max / 343,
        battery_c_needed=n * I_max / (mah / 1000)))
    out["feasible"] = ((out.tw_ratio >= 1.8) & (out.hover_throttle_pct <= 75) &
                       (out.tip_mach <= 0.7) & (out.flight_min >= 2) &
                       (out.battery_c_needed <= 120) & (out.esc_rating_a <= 120)).astype(int)
    return out

def sim_fixedwing(d):
    """d: dict of arrays: span_m, aspect, D_in, P_in, kv, cells, mah, payload_kg"""
    b, AR, D_in, P_in, kv, cells, mah, pay = (np.asarray(d[k], float) for k in
        ["span_m", "aspect", "D_in", "P_in", "kv", "cells", "mah", "payload_kg"])
    S = b**2 / AR; D = D_in * 0.0254; V = cells * CELL_V; pdr = P_in / D_in
    n_max = kv * V * 0.80 / 60
    ct_m, cp_m = prop_coeffs(pdr, D_in, np.pi * D * n_max)
    T0 = ct_m * RHO * n_max**2 * D**4
    P_mech_max = cp_m * RHO * n_max**3 * D**5
    P_el_max = P_mech_max / ETA_MOTOR
    I_max = P_el_max / V
    Wh = mah / 1000 * V
    m_mot, m_esc, m_bat = _component_masses(P_mech_max / (2 * np.pi * n_max), I_max, 1, Wh)
    airframe = 0.25 + 1.4 * S
    avionics = 0.08 + 0.08
    auw = airframe + avionics + m_mot + m_esc + m_bat + pay
    W = auw * G
    CLMAX, CD0, E = 1.3, 0.04, 0.75
    v_stall = np.sqrt(2 * W / (RHO * S * CLMAX))
    v_cruise = 1.5 * v_stall
    CL = 2 * W / (RHO * v_cruise**2 * S)
    CD = CD0 + CL**2 / (np.pi * E * AR)
    drag = 0.5 * RHO * v_cruise**2 * S * CD
    A_disk = np.pi * D**2 / 4                         # momentum-theory propeller efficiency
    eta_prop = 0.78 * 2 / (1 + np.sqrt(1 + drag / (0.5 * RHO * v_cruise**2 * A_disk)))
    P_cruise = drag * v_cruise / (eta_prop * ETA_MOTOR)
    I_c = P_cruise / V + 0.3
    flight_min = (mah / 1000 * USABLE) / I_c * 60
    pitch_speed = P_in * 0.0254 * n_max
    out = pd.DataFrame(dict(
        auw_kg=auw, flight_min=flight_min, tw_ratio=T0 / W,
        stall_speed_ms=v_stall, cruise_speed_ms=v_cruise, cruise_power_w=P_cruise,
        max_current_a=I_max, esc_rating_a=np.ceil(I_max * 1.25), motor_max_w=P_el_max,
        wing_loading_gm2=auw * 1000 / S, pitch_speed_ms=pitch_speed,
        tip_mach=np.pi * D * n_max / 343, range_km=v_cruise * flight_min * 60 / 1000,
        battery_c_needed=I_max / (mah / 1000)))
    out["feasible"] = ((out.tw_ratio >= 0.4) & (out.pitch_speed_ms >= 1.3 * v_cruise) &
                       (out.tip_mach <= 0.7) & (out.flight_min >= 3) &
                       (out.battery_c_needed <= 100) & (out.esc_rating_a <= 100)).astype(int)
    return out

def _kv_for(D_in, V, rng, n):
    tip = rng.uniform(90, 220, n)                    # realistic max tip speeds (m/s)
    rpm = tip * 60 / (np.pi * D_in * 0.0254)
    return np.round(rpm / (V * 0.80) / 10) * 10

def sample_multirotor(n, rng=RNG):
    D_in = np.round(np.exp(rng.uniform(np.log(3), np.log(30), n)), 1)
    cells = np.clip(np.round(D_in / 2.5 + rng.normal(0, 1.5, n)), 2, 14)
    d = dict(n_motors=rng.choice([4, 6, 8], n, p=[0.6, 0.25, 0.15]), D_in=D_in,
             P_in=np.round(D_in * rng.uniform(0.3, 0.9, n), 1), cells=cells,
             mah=np.round(np.exp(rng.uniform(np.log(300), np.log(32000), n)), -1),
             payload_kg=np.round(np.where(rng.random(n) < 0.3, 0, np.exp(rng.uniform(np.log(0.05), np.log(25), n))), 2))
    d["kv"] = _kv_for(D_in, cells * CELL_V, rng, n)
    return pd.DataFrame(d)

def sample_fixedwing(n, rng=RNG):
    span = rng.uniform(0.6, 4.5, n)
    D_in = np.round(rng.uniform(5, 22, n), 1)
    cells = np.clip(np.round(D_in / 3 + rng.normal(0, 1.5, n)), 2, 12)
    d = dict(span_m=np.round(span, 2), aspect=np.round(rng.uniform(5, 15, n), 1), D_in=D_in,
             P_in=np.round(D_in * rng.uniform(0.4, 1.0, n), 1), cells=cells,
             mah=np.round(np.exp(rng.uniform(np.log(500), np.log(30000), n)), -1),
             payload_kg=np.round(np.where(rng.random(n) < 0.3, 0, np.exp(rng.uniform(np.log(0.05), np.log(10), n))), 2))
    d["kv"] = _kv_for(D_in, cells * CELL_V, rng, n)
    return pd.DataFrame(d)


# ---------------- ML model helpers ----------------
def feats(X):  # log-transform positive inputs so the net learns ratios easily
    return np.log1p(X.values.astype(float))


def ml_predict(kind, X):
    m = MODELS[kind]
    P = pd.DataFrame(np.exp(m["reg"].predict(feats(X[m["feats"]]))), columns=m["targets"], index=X.index)
    P["feasible_prob"] = m["clf"].predict_proba(feats(X[m["feats"]]))[:, 1]
    return P

def _motor_class(w):
    for lim, c in [(150, "1404-1806"), (400, "2205-2306"), (800, "2806-3110"), (1500, "3510-4114"),
                   (3000, "5010-6215"), (6000, "8110-10015")]:
        if w < lim: return c
    return "heavy-lift 12xx+"

def design_drone(kind="multirotor", payload_kg=0.5, min_flight_min=15, max_size=None,
                 max_cells=None, n_motors=None, goal="flight_time", n_candidates=80_000, show=3):
    """kind: 'multirotor' or 'fixedwing'
       max_size: max wheelbase in mm (multirotor) or max wingspan in m (fixed-wing)
       goal: 'flight_time' (longest flight) or 'light' (lightest build that meets the goal)"""
    rng = np.random.default_rng()
    X = (sample_multirotor if kind == "multirotor" else sample_fixedwing)(n_candidates, rng)
    X["payload_kg"] = payload_kg
    if max_cells: X = X[X.cells <= max_cells]
    if kind == "multirotor":
        if n_motors: X["n_motors"] = n_motors
        X["wb"] = X.D_in * 25.4 / np.sin(np.pi / X.n_motors) * 1.1
        if max_size: X = X[X.wb <= max_size]
    elif max_size:
        X = X[X.span_m <= max_size]
    X = X.drop(columns=["wb"], errors="ignore")
    if X.empty:
        print("No candidates fit those limits - relax max_size / max_cells."); return None
    P = ml_predict(kind, X)                               # fast ML screening of all candidates
    ok = (P.feasible_prob > 0.6) & (P.flight_min >= min_flight_min * 1.03)
    if kind == "multirotor":
        ok &= (P.tw_ratio >= 2.0) & (P.hover_throttle_pct <= 65) & (P.max_current_a / X.n_motors <= 90)
    else:
        ok &= (P.max_current_a <= 50) & (P.tw_ratio <= 1.5) & (X.cells >= 3)
    if ok.sum() == 0:
        best = P.flight_min.idxmax()
        print(f"No build reaches {min_flight_min} min with these limits. Best found ~{P.flight_min.max():.1f} min.")
        print("Try: lower payload, allow bigger size, or more battery cells."); return None
    C = X[ok].copy(); C = C.join(P[ok])
    C = C.sort_values("flight_min" if goal == "flight_time" else "auw_kg", ascending=(goal != "flight_time"))
    top = C.head(40)
    sim = (sim_multirotor if kind == "multirotor" else sim_fixedwing)(top)   # physics check
    sim.index = top.index
    good = top[(sim.feasible == 1) & (sim.flight_min >= min_flight_min)]
    if good.empty:
        print("ML suggestions failed the physics check - try relaxing the requirements."); return None
    res = []
    for i, idx in enumerate(good.index[:show]):
        r, s = X.loc[idx], sim.loc[idx]
        print(f"\n================ OPTION {i+1} ({kind}) ================")
        if kind == "multirotor":
            print(f"Frame     : {int(r.n_motors)}-rotor, ~{s.wheelbase_mm:.0f} mm wheelbase")
            print(f"Motors    : {int(r.n_motors)} x ~{r.kv:.0f} KV, size class {_motor_class(s.motor_max_w)} "
                  f"(max ~{s.motor_max_w:.0f} W each)")
            print(f"Props     : {r.D_in:g} x {r.P_in:g} in")
            print(f"ESC       : {int(r.n_motors)} x {s.esc_rating_a:.0f} A (or higher)")
            print(f"Battery   : {int(r.cells)}S LiPo {r.mah:.0f} mAh, >= {max(25, np.ceil(s.battery_c_needed*1.1)):.0f}C")
            fc = "ArduPilot/PX4 flight controller + GPS" if s.wheelbase_mm > 350 or payload_kg > 0.3 else "Betaflight/INAV F4/F7 flight controller"
            print(f"Avionics  : {fc}")
            print(f"--> All-up weight {s.auw_kg:.2f} kg | hover flight time {s.flight_min:.1f} min | "
                  f"thrust/weight {s.tw_ratio:.1f} | hover throttle {s.hover_throttle_pct:.0f}% | "
                  f"max current {s.max_current_a:.0f} A")
        else:
            S = r.span_m**2 / r.aspect
            print(f"Airframe  : wingspan {r.span_m:.2f} m, aspect ratio {r.aspect:g}, wing area {S:.2f} m^2, "
                  f"mean chord {r.span_m/r.aspect*100:.0f} cm")
            print(f"Motor     : ~{r.kv:.0f} KV, size class {_motor_class(s.motor_max_w)} (max ~{s.motor_max_w:.0f} W)")
            print(f"Prop      : {r.D_in:g} x {r.P_in:g} in")
            print(f"ESC       : {s.esc_rating_a:.0f} A (or higher)")
            print(f"Battery   : {int(r.cells)}S LiPo {r.mah:.0f} mAh, >= {max(20, np.ceil(s.battery_c_needed*1.1)):.0f}C")
            print(f"Avionics  : ArduPilot/INAV plane flight controller + GPS, 4 servos")
            print(f"--> All-up weight {s.auw_kg:.2f} kg | flight time {s.flight_min:.0f} min | range ~{s.range_km:.0f} km | "
                  f"cruise {s.cruise_speed_ms:.0f} m/s | stall {s.stall_speed_ms:.1f} m/s | thrust/weight {s.tw_ratio:.2f}")
        res.append(pd.concat([r, s]))
    return pd.DataFrame(res)

def check_build(kind="multirotor", **parts):
    """Predict performance of a build you already have. Example:
       check_build('multirotor', n_motors=4, D_in=10, P_in=4.5, kv=920, cells=3, mah=5000, payload_kg=0)"""
    X = pd.DataFrame([parts])
    p = ml_predict(kind, X).iloc[0]; s = (sim_multirotor if kind == "multirotor" else sim_fixedwing)(X).iloc[0]
    print("Metric                 ML model   Physics")
    for t in MODELS[kind]["targets"]:
        print(f"{t:22s} {p[t]:9.2f} {s[t]:9.2f}")
    print(f"Feasible build? ML: {'yes' if p.feasible_prob>0.5 else 'NO'} ({p.feasible_prob:.0%})  Physics: {'yes' if s.feasible else 'NO'}")
    if kind == "multirotor":
        if s.tw_ratio < 2: print("Tip: thrust/weight < 2 -> bigger props, higher KV, or more cells.")
        if s.hover_throttle_pct > 65: print("Tip: hover throttle high -> drone is underpowered or too heavy.")
    if s.battery_c_needed > 60: print(f"Tip: battery must supply ~{s.battery_c_needed:.0f}C bursts - use a high-C pack.")
    return p, s


if __name__ == "__main__":
    print("#### Example 1: camera quadcopter carrying 0.5 kg for 20+ minutes, max 650 mm frame (longest flight)")
    design_drone("multirotor", payload_kg=0.5, min_flight_min=20, max_size=650, n_motors=4)
    print("\n\n#### Example 2: fixed-wing mapping drone carrying 1 kg, 60+ min, max 2.5 m span (lightest build)")
    design_drone("fixedwing", payload_kg=1.0, min_flight_min=60, max_size=2.5, goal="light")
    print("\n\n#### Example 3: check an existing build (450 quad, 10x4.5 props, 920KV, 3S 5000mAh)")
    _ = check_build("multirotor", n_motors=4, D_in=10, P_in=4.5, kv=920, cells=3, mah=5000, payload_kg=0)
