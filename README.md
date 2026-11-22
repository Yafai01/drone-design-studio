# Drone Design Studio

Design and check multirotor and fixed-wing drones with a trained scikit-learn model and a physics simulation, then get an engineering drawing, performance analysis and a PDF design report.

Designed and developed by **Yafai**.

**Live app:** <https://drone-design-studio-ten.vercel.app>

## What it does

- **Design new:** give a payload, minimum flight time and size limit. The ML model screens 80,000 random builds, the best 40 are rechecked with physics, and the top options come back with a parts list.
- **Check a build:** enter motors, props, KV, battery and payload to get flight time, weight, thrust-to-weight and every physics limit, with the ML model and physics side by side.
- **Drawing sheet:** to-scale plan view with dimensions, numbered parts and a title block.
- **3D view:** an interactive 3D model built from the same dimensions (frame, motors, spinning props, battery, landing gear and camera, or fuselage, wing, tail and prop). Drag to rotate, scroll to zoom, switch between 3D, top, front and side views. Open it directly with `#3d` in the URL. Uses three.js, loaded from a CDN when the tab is first opened.
- **Analysis:** weight breakdown, key engineering figures (hover efficiency, disk loading, lift-to-drag, energy per km) and a sensitivity chart.
- **Trade study:** vary one part across a range and chart flight time and thrust-to-weight.
- **Conditions:** altitude, air temperature and landing battery reserve feed the physics.
- **Pre-built designs:** 10 example builds from 5-inch FPV to a 3 m endurance wing.
- **Frame layouts:** Quad X / + / H / Deadcat, Hex X / + / Y6 coaxial, Octo X / + / X8 coaxial, and Conventional / V-tail / Flying wing airframes. The layout changes the drawing, 3D model and parts list; flight numbers come from motor count, props, battery and payload. Browse them all with `#frames`.
- **Saved builds and comparison**, **PDF design report**, **parts list CSV** and keyboard shortcuts (press `?`).

## Run it locally

Needs Python 3.9 to 3.12. scikit-learn must be **1.6.1**, the version the model was trained with.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python drone_app.py
```

The app opens at <http://127.0.0.1:8547>. On Windows you can also double-click `Drone Design Studio.bat` once the `.venv` exists.

Other entry points:

- `python drone_design.py` prints three example designs.
- `python run_designer.py` is an interactive command-line version.

Locally, saved builds are written to `saved_builds.json`. On the hosted site they are kept in your browser.

## Deploy

The repo is set up for Vercel: `api/index.py` runs the same server as a Python function that serves both the page and the API. Pushing to `main` deploys automatically; to deploy by hand:

```bash
vercel deploy --prod
```

## Files

| File | Purpose |
|---|---|
| `drone_design_model.joblib` | Trained model bundle (do not edit) |
| `drone_design.py` | Model loader, physics simulation, `design_drone()` and `check_build()` |
| `drone_app.py` | Web server and API |
| `drone_web/index.html` | The interface |
| `api/index.py`, `vercel.json` | Vercel deployment |
| `run_designer.py` | Command-line designer |

## Accuracy

Results are estimates for comparing designs. The ML model has no altitude or temperature input, so only the physics reflects operating conditions. Confirm thrust, current and endurance with bench and flight tests before relying on them.
