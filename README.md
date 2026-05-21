# Orbit viewer v1.0.0

PySide6 desktop app for visualizing elliptical orbits in 3D (ECI) and ground-track (lat/lon) views, powered by VisPy and the `Orbit` simulation in `orbit.py`.

## Setup

```bash
cd orbit-app
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Cartopy may download Natural Earth coastline data on first run (requires network once).

## Run

```bash
python app.py
```

## Project layout

| File | Role |
|------|------|
| `app.py` | GUI, VisPy views, animation controls |
| `orbit.py` | Orbit model, ECI integration, ground-track simulation |
| `utils.py` | Coordinate helpers, dateline breaks |
