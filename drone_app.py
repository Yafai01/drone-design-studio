"""
Drone Design Studio - local web app for drone_design.py.
Designed and developed by Yafai.

Run:  python drone_app.py          (opens http://127.0.0.1:8547 in your browser)
Stop: Ctrl+C in the terminal.
Uses only the Python standard library on top of the packages drone_design.py needs.
"""
import contextlib, io, json, math, os, socket, threading, time, uuid, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

print("Loading drone design model...")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import drone_design as dd  # noqa: E402
from drone_design import (design_drone, check_build, ml_predict, sim_multirotor,  # noqa: E402
                          sim_fixedwing, MODELS, _motor_class)

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "drone_web", "index.html")
SAVED = os.path.join(HERE, "saved_builds.json")
# "file" keeps saved builds in saved_builds.json (local app); "browser" leaves them to the
# visitor's browser storage (used on serverless hosts such as Vercel, which have no lasting disk).
SAVED_MODE = os.environ.get("DDS_SAVED", "file")
LOCK = threading.Lock()          # the model is shared; run one request at a time


class BadInput(ValueError):
    pass


def num(body, key, lo, hi, default=None, integer=False, optional=False):
    v = body.get(key, default)
    if v in (None, "", "none"):
        if optional:
            return None
        raise BadInput(f"{key} is required")
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise BadInput(f"{key} must be a number")
    if not math.isfinite(v) or not lo <= v <= hi:
        raise BadInput(f"{key} must be between {lo:g} and {hi:g}")
    return int(round(v)) if integer else v


def clean(series):
    out = {}
    for k, v in series.items():
        if isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = None if not math.isfinite(float(v)) else round(float(v), 4)
        else:
            out[k] = v
    return out


def parse_env(body):
    """Operating conditions -> air density (ISA pressure at altitude) and usable battery fraction."""
    env = body.get("env") or {}
    if not isinstance(env, dict):
        raise BadInput("env must be an object")
    alt = num(env, "alt_m", -500, 6000, default=0)
    temp = num(env, "temp_c", -40, 50, optional=True)
    if temp is None:
        temp = 15 - 0.0065 * alt          # standard day (ISA) temperature at this altitude
    reserve = num(env, "reserve_pct", 0, 60, default=20)
    pressure = 101325 * (1 - 2.25577e-5 * alt) ** 5.25588
    rho = pressure / (287.05 * (temp + 273.15))
    return {"alt_m": alt, "temp_c": round(temp, 1), "standard_day": env.get("temp_c") in (None, ""),
            "reserve_pct": reserve, "rho": round(rho, 4), "density_ratio": round(rho / 1.225, 4),
            "usable": 1 - reserve / 100}


@contextlib.contextmanager
def conditions(env):
    """Run drone_design's physics at these conditions (the module reads RHO and USABLE at call time)."""
    old = dd.RHO, dd.USABLE
    dd.RHO, dd.USABLE = env["rho"], env["usable"]
    try:
        yield
    finally:
        dd.RHO, dd.USABLE = old


def public_env(env):
    return {k: v for k, v in env.items() if k != "usable"}


def kind_of(body):
    k = body.get("kind")
    if k not in ("multirotor", "fixedwing"):
        raise BadInput("kind must be multirotor or fixedwing")
    return k


def api_design(body):
    kind = kind_of(body)
    multi = kind == "multirotor"
    payload = num(body, "payload_kg", 0, 25 if multi else 10)
    flight = num(body, "min_flight_min", 1, 180 if multi else 600)
    size = num(body, "max_size", 100, 3000, optional=True) if multi else num(body, "max_size", 0.6, 4.5, optional=True)
    goal = body.get("goal", "flight_time")
    if goal not in ("flight_time", "light"):
        raise BadInput("goal must be flight_time or light")
    max_cells = num(body, "max_cells", 2, 14, integer=True, optional=True)
    n_motors = num(body, "n_motors", 4, 8, integer=True, optional=True) if multi else None
    if n_motors is not None and n_motors not in (4, 6, 8):
        raise BadInput("n_motors must be 4, 6 or 8")

    env = parse_env(body)
    buf = io.StringIO()
    with LOCK, conditions(env), contextlib.redirect_stdout(buf):
        df = design_drone(kind, payload_kg=payload, min_flight_min=flight, max_size=size,
                          max_cells=max_cells, n_motors=n_motors, goal=goal)
    if df is None or df.empty:
        msg = " ".join(l.strip() for l in buf.getvalue().splitlines() if l.strip())
        return {"ok": False, "message": msg or "No design found - try relaxing the requirements.",
                "conditions": public_env(env)}

    options = []
    for _, row in df.iterrows():
        o = clean(row)
        for k in ("n_motors", "cells", "mah", "kv"):
            if k in o:
                o[k] = int(round(o[k]))
        o["motor_class"] = _motor_class(row["motor_max_w"])
        o["battery_c"] = int(max(25 if multi else 20, math.ceil(row["battery_c_needed"] * 1.1)))
        if multi:
            o["avionics"] = ("ArduPilot/PX4 flight controller + GPS"
                             if row["wheelbase_mm"] > 350 or payload > 0.3
                             else "Betaflight/INAV F4/F7 flight controller")
        else:
            o["avionics"] = "ArduPilot/INAV plane flight controller + GPS, 4 servos"
            o["wing_area_m2"] = round(row["span_m"] ** 2 / row["aspect"], 4)
            o["chord_cm"] = round(row["span_m"] / row["aspect"] * 100, 1)
        options.append(o)
    return {"ok": True, "kind": kind, "options": options, "conditions": public_env(env)}


PART_RANGES = {   # key: (lo, hi, integer)
    "n_motors": (4, 8, True), "span_m": (0.3, 6, False), "aspect": (3, 20, False),
    "D_in": (2, 30, False), "P_in": (1, 30, False), "kv": (50, 5000, True),
    "cells": (1, 14, True), "mah": (100, 50000, True), "payload_kg": (0, 25, False),
}
SWEEPABLE = {"multirotor": ["mah", "cells", "D_in", "P_in", "kv", "payload_kg"],
             "fixedwing": ["mah", "cells", "D_in", "P_in", "kv", "payload_kg", "span_m", "aspect"]}


def api_sweep(body):
    """Vary one part across a range; return ML and physics predictions at each step."""
    kind = kind_of(body)
    parts = parse_parts(body, kind)
    param = body.get("param")
    if param not in SWEEPABLE[kind]:
        raise BadInput(f"param must be one of {', '.join(SWEEPABLE[kind])}")
    lo_lim, hi_lim, integer = PART_RANGES[param]
    lo = num(body, "lo", lo_lim, hi_lim)
    hi = num(body, "hi", lo_lim, hi_lim)
    if hi <= lo:
        raise BadInput("The end of the range must be larger than the start")
    xs = np.linspace(lo, hi, 61)
    if param == "mah":
        xs = np.unique(np.round(xs / 50) * 50).astype(int)   # battery sizes in 50 mAh steps
    elif param == "kv":
        xs = np.unique(np.round(xs / 10) * 10).astype(int)
    elif integer:
        xs = np.unique(np.round(xs)).astype(int)
    X = pd.DataFrame([parts] * len(xs))
    X[param] = xs
    env = parse_env(body)
    with LOCK, conditions(env):
        P = ml_predict(kind, X)
        Sm = (sim_multirotor if kind == "multirotor" else sim_fixedwing)(X)
    pts = []
    for i, x in enumerate(xs):
        pts.append({"x": float(x),
                    "ml_flight": round(float(P["flight_min"].iloc[i]), 3),
                    "phys_flight": round(float(Sm["flight_min"].iloc[i]), 3),
                    "ml_tw": round(float(P["tw_ratio"].iloc[i]), 4),
                    "phys_tw": round(float(Sm["tw_ratio"].iloc[i]), 4),
                    "auw": round(float(Sm["auw_kg"].iloc[i]), 4),
                    "feasible": bool(Sm["feasible"].iloc[i]),
                    "ml_prob": round(float(P["feasible_prob"].iloc[i]), 4)})
    return {"ok": True, "kind": kind, "param": param, "points": pts}


# ---------- engineering analysis: weight breakdown, key figures, sensitivity ----------
def mass_breakdown(kind, p):
    """Component masses using the same formulas as drone_design's physics simulation."""
    D_in, P_in, kv, cells, mah, pay = p["D_in"], p["P_in"], p["kv"], p["cells"], p["mah"], p["payload_kg"]
    D = D_in * 0.0254; V = cells * dd.CELL_V; pdr = P_in / D_in
    n_max = kv * V * 0.80 / 60
    ct, cp = dd.prop_coeffs(np.array([pdr]), np.array([D_in]), np.array([np.pi * D * n_max]))
    P_mech = float(cp[0]) * dd.RHO * n_max ** 3 * D ** 5
    I_max = P_mech / dd.ETA_MOTOR / V
    Wh = mah / 1000 * V
    torque = P_mech / (2 * np.pi * n_max)
    if kind == "multirotor":
        n = p["n_motors"]
        wb = D_in * 25.4 / np.sin(np.pi / n) * 1.1
        m_mot, m_esc, m_bat = dd._component_masses(torque, I_max, n, Wh)
        items = [("Battery", m_bat), ("Motors", m_mot), ("ESCs", m_esc),
                 ("Frame", 0.0064 * wb ** 1.79 / 1000), ("Avionics and wiring", (50 + 0.03 * wb) / 1000)]
    else:
        S = p["span_m"] ** 2 / p["aspect"]
        m_mot, m_esc, m_bat = dd._component_masses(torque, I_max, 1, Wh)
        items = [("Battery", m_bat), ("Motor", m_mot), ("ESC", m_esc),
                 ("Airframe", 0.25 + 1.4 * S), ("Avionics and servos", 0.16)]
    items.append(("Payload", pay))
    return [{"name": k, "kg": round(float(v), 4)} for k, v in items], Wh


def api_analysis(body):
    kind = kind_of(body)
    parts = parse_parts(body, kind)
    sim = sim_multirotor if kind == "multirotor" else sim_fixedwing
    # sensitivity: each part -10% and +10% (cells +/-1), everything else fixed
    cases = []
    for k in SWEEPABLE[kind]:
        lo_lim, hi_lim, integer = PART_RANGES[k]
        base = parts[k]
        if k == "cells":
            lo_v, hi_v = base - 1, base + 1
        elif k == "payload_kg" and base == 0:
            lo_v, hi_v = 0, 0.1
        else:
            lo_v, hi_v = base * 0.9, base * 1.1
            if k == "mah":
                lo_v, hi_v = round(lo_v / 50) * 50, round(hi_v / 50) * 50
            elif k == "kv":
                lo_v, hi_v = round(lo_v / 10) * 10, round(hi_v / 10) * 10
            elif integer:
                lo_v, hi_v = round(lo_v), round(hi_v)
        lo_v, hi_v = max(lo_lim, lo_v), min(hi_lim, hi_v)
        cases += [(k, "lo", lo_v), (k, "hi", hi_v)]
    X = pd.DataFrame([parts] * (len(cases) + 1))
    for i, (k, _, v) in enumerate(cases, start=1):
        X.loc[i, k] = v
    env = parse_env(body)
    with LOCK, conditions(env):
        return _analysis(kind, parts, sim, X, cases, env)


def _analysis(kind, parts, sim, X, cases, env):
    """Runs inside the model lock with the requested conditions applied."""
    Sm = sim(X)
    masses, Wh = mass_breakdown(kind, parts)
    s0 = Sm.iloc[0]
    sens = {}
    for i, (k, side, v) in enumerate(cases, start=1):
        r = Sm.iloc[i]
        sens.setdefault(k, {"param": k})[side] = {
            "value": float(v), "flight": round(float(r["flight_min"]), 3),
            "delta": round(float(r["flight_min"] - s0["flight_min"]), 3), "feasible": bool(r["feasible"])}
    sens = sorted(sens.values(), key=lambda e: -max(abs(e["lo"]["delta"]), abs(e["hi"]["delta"])))
    auw = float(s0["auw_kg"])
    fig = {"auw_kg": auw, "battery_wh": Wh, "battery_frac": masses[0]["kg"] / auw,
           "payload_frac": parts["payload_kg"] / auw, "flight_min": float(s0["flight_min"])}
    if kind == "multirotor":
        D = parts["D_in"] * 0.0254
        fig.update(disk_loading=auw * dd.G / (parts["n_motors"] * np.pi * (D / 2) ** 2),
                   g_per_w=auw * 1000 / float(s0["hover_power_w"]),
                   w_per_kg=float(s0["hover_power_w"]) / auw)
    else:
        S = parts["span_m"] ** 2 / parts["aspect"]
        v = float(s0["cruise_speed_ms"])
        CL = 2 * auw * dd.G / (dd.RHO * v ** 2 * S)
        CD = 0.04 + CL ** 2 / (np.pi * 0.75 * parts["aspect"])
        fig.update(l_over_d=CL / CD, cl_cruise=CL,
                   wh_per_km=float(s0["cruise_power_w"]) / (v * 3.6),
                   w_per_kg=float(s0["cruise_power_w"]) / auw)
    return {"ok": True, "kind": kind, "masses": masses, "auw_check": round(sum(m["kg"] for m in masses), 4),
            "figures": {k: round(float(v), 4) for k, v in fig.items()}, "sensitivity": sens,
            "base_flight": round(float(s0["flight_min"]), 3), "conditions": public_env(env)}


# ---------- saved builds (stored in saved_builds.json next to this script) ----------
SAVE_LOCK = threading.Lock()


def load_saved():
    try:
        with open(SAVED, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def write_saved(items):
    tmp = SAVED + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1)
    os.replace(tmp, SAVED)


def api_saved_add(body):
    kind = kind_of(body)
    parts = parse_parts(body.get("parts") or {}, kind)
    name = str(body.get("name") or "Untitled build").strip()[:80]
    metrics = {k: float(v) for k, v in (body.get("metrics") or {}).items()
               if isinstance(v, (int, float)) and math.isfinite(v) and len(k) < 40}
    env = public_env(parse_env(body))
    rec = {"id": uuid.uuid4().hex[:10], "name": name, "kind": kind, "parts": parts,
           "metrics": metrics, "env": env, "saved": time.strftime("%Y-%m-%d %H:%M")}
    with SAVE_LOCK:
        items = load_saved()
        if len(items) >= 200:
            raise BadInput("You have 200 saved builds. Delete some before saving more.")
        items.append(rec)
        write_saved(items)
    return {"ok": True, "items": items}


def api_saved_delete(build_id):
    with SAVE_LOCK:
        items = [i for i in load_saved() if i.get("id") != build_id]
        write_saved(items)
    return {"ok": True, "items": items}


def parse_parts(body, kind):
    parts = {}
    if kind == "multirotor":
        parts["n_motors"] = num(body, "n_motors", 4, 8, integer=True)
        if parts["n_motors"] not in (4, 6, 8):
            raise BadInput("n_motors must be 4, 6 or 8")
    else:
        parts["span_m"] = num(body, "span_m", 0.3, 6)
        parts["aspect"] = num(body, "aspect", 3, 20)
    parts["D_in"] = num(body, "D_in", 2, 30)
    parts["P_in"] = num(body, "P_in", 1, 30)
    parts["kv"] = num(body, "kv", 50, 5000, integer=True)
    parts["cells"] = num(body, "cells", 1, 14, integer=True)
    parts["mah"] = num(body, "mah", 100, 50000, integer=True)
    parts["payload_kg"] = num(body, "payload_kg", 0, 25)
    return parts


def api_check(body):
    kind = kind_of(body)
    parts = parse_parts(body, kind)
    env = parse_env(body)
    buf = io.StringIO()
    with LOCK, conditions(env), contextlib.redirect_stdout(buf):
        p, s = check_build(kind, **parts)
    tips = [l.strip()[4:].strip() for l in buf.getvalue().splitlines() if l.strip().startswith("Tip:")]
    metrics = [{"name": t, "ml": round(float(p[t]), 3), "physics": round(float(s[t]), 3)}
               for t in MODELS[kind]["targets"]]
    return {"ok": True, "kind": kind, "parts": parts, "metrics": metrics, "tips": tips,
            "motor_class": _motor_class(s["motor_max_w"]),
            "ml_feasible_prob": round(float(p["feasible_prob"]), 4),
            "physics_feasible": bool(s["feasible"]), "physics": clean(s), "conditions": public_env(env)}


ROUTES = {"/api/design": api_design, "/api/check": api_check, "/api/sweep": api_sweep,
          "/api/analysis": api_analysis, "/api/saved": api_saved_add}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    @property
    def route(self):
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def do_GET(self):
        if self.route == "/api/info":
            import platform, sklearn
            return self._send(200, {"ok": True, "sklearn": sklearn.__version__,
                                    "python": platform.python_version(), "saved": SAVED_MODE,
                                    "constants": {"g": dd.G, "sea_level_rho": 1.225, "eta_motor": dd.ETA_MOTOR,
                                                  "lipo_wh_per_kg": dd.LIPO_WH_PER_KG, "cell_v": dd.CELL_V}})
        if self.route == "/api/saved":
            if SAVED_MODE != "file":
                return self._send(200, {"ok": True, "items": [], "storage": SAVED_MODE})
            with SAVE_LOCK:
                return self._send(200, {"ok": True, "items": load_saved()})
        if self.route in ("/", "/index.html"):
            with open(INDEX, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        self._send(404, {"ok": False, "message": "Not found"})

    def do_POST(self):
        fn = ROUTES.get(self.route)
        if fn is api_saved_add and SAVED_MODE != "file":
            return self._send(400, {"ok": False, "message": "Saved builds are stored in your browser on this site."})
        if not fn:
            return self._send(404, {"ok": False, "message": "Not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, fn(body))
        except BadInput as e:
            self._send(400, {"ok": False, "message": str(e)})
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "message": "Invalid JSON"})
        except Exception as e:  # keep the server alive and show the error in the UI
            self._send(500, {"ok": False, "message": f"{type(e).__name__}: {e}"})

    def do_DELETE(self):
        prefix = "/api/saved/"
        if not self.route.startswith(prefix) or SAVED_MODE != "file":
            return self._send(404, {"ok": False, "message": "Not found"})
        try:
            self._send(200, api_saved_delete(self.route[len(prefix):]))
        except Exception as e:
            self._send(500, {"ok": False, "message": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        pass


def start_server(start):
    """Use the first port from `start` that nothing else is listening on."""
    for port in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                continue          # something else already answers here
        try:
            return port, ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError:
            continue
    raise RuntimeError(f"No free port found in {start}-{start + 19}")


if __name__ == "__main__":
    port, server = start_server(int(os.environ.get("DRONE_APP_PORT", "8547")))
    url = f"http://127.0.0.1:{port}"
    print(f"Drone Design Studio running at {url}  (Ctrl+C to stop)")
    if os.environ.get("DRONE_APP_NO_BROWSER") != "1":
        threading.Timer(0.8, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
