"""PySide6 orbit viewer with embedded VisPy simulation panels."""

import sys
import time

import numpy as np
from PySide6 import QtCore, QtWidgets
from vispy import app as vispy_app
from vispy import scene
from vispy.app import use_app
from vispy.geometry import Rect, create_sphere

from orbit import DEFAULT_ANIMATION_SECONDS, R_EARTH, Orbit
from utils import break_at_dateline, xyz_to_latlon

DEFAULT_H_MAX = 100_000e3
DEFAULT_H_MIN = 500e3
DEFAULT_STEP_S = 60.0
DEFAULT_N_ORBITS = 4
DEFAULT_ANIMATION_SPEED = 1.0
# Chase camera defaults: standoff (km) and roll / azimuth / pitch (degrees, ±180).
DEFAULT_CHASE_DISTANCE_KM = 5000.0
CHASE_DISTANCE_MIN_KM = 100
CHASE_DISTANCE_MAX_KM = 100_000
MAP_GRATICULE_SPACING_DEG = 30
MAP_IMAGE_SIZE = (1920, 960)
DEFAULT_CHASE_ROLL_DEG = 0.0
DEFAULT_CHASE_AZIMUTH_DEG = -108.0
DEFAULT_CHASE_PITCH_DEG = 20.0
# Arrow length = |v| × this many seconds of flight.
VELOCITY_ARROW_DURATION_S = 90.0
DEFAULT_RAAN_DEG = 0.0
DEFAULT_INC_DEG = 0.0
DEFAULT_AOP_DEG = 0.0
TIMER_INTERVAL_S = 1 / 120
VIEW_BG = "#1e1e1e"
MAP_BOUNDS = (-180.0, -90.0, 360.0, 180.0)  # left, bottom, width, height (degrees)
ORIGIN = np.zeros(3, dtype=np.float32)

# Draw order for 3D scene (back → front). Translucent layers use depth_mask=False.
_GL_ORDER_GRID = 0
_GL_ORDER_EARTH = 1
_GL_ORDER_ORBIT = 3
_GL_ORDER_TRAIL = 8
_GL_ORDER_SATELLITE = 10
_GL_ORDER_VELOCITY = 11


def _configure_scene_visual(visual, *, order, alpha=1.0):
    """Apply consistent depth/blend state to avoid transparency artifacts (3D)."""
    visual.order = order
    if alpha >= 1.0:
        visual.set_gl_state("opaque", depth_test=True, depth_mask=True)
    else:
        visual.set_gl_state("translucent", depth_test=True, depth_mask=False)


def _configure_ground_track_basemap(visual):
    """Map image behind the track; never writes depth (2D overlay)."""
    visual.order = 0
    visual.set_gl_state("translucent", depth_test=False, depth_mask=False)


def _configure_ground_track_overlay(visual, *, order, alpha=1.0):
    """Trail and marker above the basemap without depth fighting (2D overlay)."""
    visual.order = order
    visual.set_gl_state("translucent", depth_test=False, depth_mask=False)


def _equatorial_grid(half, divisions=12):
    lines = []
    step = (2 * half) / divisions
    for k in range(divisions + 1):
        t = -half + k * step
        lines.append([[t, -half, 0.0], [t, half, 0.0]])
        lines.append([[-half, t, 0.0], [half, t, 0.0]])
    pos = np.vstack(lines).astype(np.float32)
    connect = np.arange(2 * len(lines), dtype=np.uint32).reshape(-1, 2)
    return pos, connect


def _lonlat_line_strip(lon, lat):
    """Lon/lat polyline for VisPy ``connect='strip'`` (NaN breaks at dateline)."""
    lon = np.asarray(lon, dtype=np.float64).ravel()
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon, lat = break_at_dateline(lat=lat, lon=lon)
    if lon.size == 0:
        return None, "strip"
    if lon.size == 1:
        lon = np.array([lon[0], lon[0]])
        lat = np.array([lat[0], lat[0]])
    return np.column_stack([lon, lat]).astype(np.float32), "strip"


def _lonlat_to_line_segments(lon, lat):
    """Split lon/lat at dateline into separate segment groups for multi-path lines."""
    lon = np.asarray(lon, dtype=np.float64).ravel()
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon, lat = break_at_dateline(lat=lat, lon=lon)
    segments = []
    seg_lon, seg_lat = [], []
    for lo, la in zip(lon, lat):
        if np.isnan(lo) or np.isnan(la):
            if len(seg_lon) >= 2:
                segments.append(np.column_stack([seg_lon, seg_lat]).astype(np.float32))
            seg_lon, seg_lat = [], []
        else:
            seg_lon.append(lo)
            seg_lat.append(la)
    if len(seg_lon) >= 2:
        segments.append(np.column_stack([seg_lon, seg_lat]).astype(np.float32))
    if not segments:
        return None, None
    if len(segments) == 1:
        return segments[0], "strip"
    pos = np.vstack(segments)
    connect = np.arange(2 * len(segments), dtype=np.uint32).reshape(-1, 2)
    return pos, connect


def _map_border_polyline():
    left, bottom, width, height = MAP_BOUNDS
    right = left + width
    top = bottom + height
    return np.array(
        [
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
            [left, bottom],
        ],
        dtype=np.float32,
    )


def _graticule_line_data(spacing=MAP_GRATICULE_SPACING_DEG):
    """Lon/lat grid lines for the ground-track view."""
    lines = []
    for lon in range(-180, 181, spacing):
        lines.append([[lon, -90.0], [lon, 90.0]])
    for lat in range(-90, 91, spacing):
        lines.append([[-180.0, lat], [180.0, lat]])
    pos = np.vstack(lines).astype(np.float32)
    connect = np.arange(2 * len(lines), dtype=np.uint32).reshape(-1, 2)
    return pos, connect


def _build_ground_track_map_rgb(size=MAP_IMAGE_SIZE):
    """Raster map with coastlines and 30° graticule (Plate Carrée)."""
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    width, height = size
    dpi = 100
    fig = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor=VIEW_BG,
    )
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.set_facecolor("#152535")
    ax.add_feature(cfeature.OCEAN, facecolor="#152535")
    ax.add_feature(cfeature.LAND, facecolor="#2a3540")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#7a92a8")
    ax.gridlines(
        draw_labels=False,
        linewidth=0.45,
        color="#556575",
        alpha=0.85,
        linestyle="-",
        xlocs=np.arange(-180, 181, MAP_GRATICULE_SPACING_DEG),
        ylocs=np.arange(-90, 91, MAP_GRATICULE_SPACING_DEG),
    )
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    return rgba[:, :, :3][::-1]


def _interp_positions(positions, times, t_query):
    """Linear interpolation along columns of ``positions`` (3, N) at time ``t_query``."""
    pos = np.asarray(positions, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if t_query <= t[0]:
        return pos[:, 0].astype(np.float32)
    if t_query >= t[-1]:
        return pos[:, -1].astype(np.float32)
    idx = int(np.searchsorted(t, t_query, side="right") - 1)
    idx = min(idx, len(t) - 2)
    dt = t[idx + 1] - t[idx]
    frac = 0.0 if dt == 0 else (t_query - t[idx]) / dt
    return ((1.0 - frac) * pos[:, idx] + frac * pos[:, idx + 1]).astype(np.float32)


def _chase_camera_offset_direction(vel, azimuth_deg, pitch_deg):
    """Unit vector from satellite toward camera in the velocity-local frame.

    Azimuth rotates about +velocity from behind toward right; pitch tilts toward
    local up. Both are in degrees on [-180, 180].
    """
    vel = np.asarray(vel, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(vel))
    if speed < 1e-6:
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)

    forward = vel / speed
    behind = -forward
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        right = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), forward)
        right_norm = float(np.linalg.norm(right))
    right /= right_norm
    up = np.cross(forward, right)
    up /= float(np.linalg.norm(up))

    az = np.deg2rad(float(azimuth_deg))
    pitch = np.deg2rad(float(pitch_deg))
    offset = (
        behind * np.cos(pitch) * np.cos(az)
        + right * np.cos(pitch) * np.sin(az)
        + up * np.sin(pitch)
    )
    norm = float(np.linalg.norm(offset))
    if norm < 1e-12:
        return behind
    return offset / norm


def _offset_to_turntable_angles(offset):
    offset = np.asarray(offset, dtype=np.float64).reshape(3)
    elevation = float(np.degrees(np.arcsin(np.clip(offset[2], -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(offset[1], offset[0])))
    return azimuth, elevation


def _update_mesh_position(visual, position, scale):
    visual.transform = scene.transforms.STTransform(
        translate=np.asarray(position, dtype=np.float32).reshape(3),
        scale=scale,
    )
    visual.update()


def _orbital_speed_km_s(vel):
    """Scalar speed in km/s from an ECI velocity vector (m/s)."""
    return float(np.linalg.norm(np.asarray(vel, dtype=np.float64))) / 1000.0


def _update_velocity_arrow(arrow, pos, vel, *, duration_s=VELOCITY_ARROW_DURATION_S):
    """Draw a fixed-time velocity arrow anchored at the satellite."""
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    vel = np.asarray(vel, dtype=np.float64).reshape(3)
    vec = vel * float(duration_s)
    length = float(np.linalg.norm(vec))
    if length < 1.0:
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        length = 1.0
    tip = pos + vec
    head_frac = 0.12
    head_base = tip - head_frac * vec
    line_pos = np.vstack([pos, tip]).astype(np.float32)
    arrows = np.array([np.concatenate([head_base, tip])], dtype=np.float32)
    arrow.set_data(pos=line_pos, arrows=arrows)
    arrow.update()


class GroundTrackCamera(scene.cameras.PanZoomCamera):
    """Lat/lon map: pan when zoomed in, zoom, clamped to world bounds."""

    def __init__(self, **kwargs):
        super().__init__(rect=MAP_BOUNDS, aspect=None, **kwargs)

    def _clamp_rect(self):
        wb = Rect(*MAP_BOUNDS)
        r = Rect(self.rect)
        if r.width >= wb.width:
            r.left = wb.left
            r.width = wb.width
        else:
            r.left = float(np.clip(r.left, wb.left, wb.right - r.width))
        if r.height >= wb.height:
            r.bottom = wb.bottom
            r.height = wb.height
        else:
            r.bottom = float(np.clip(r.bottom, wb.bottom, wb.top - r.height))
        self.rect = r

    def viewbox_mouse_event(self, event):
        if event.handled or not self.interactive:
            return
        scene.cameras.BaseCamera.viewbox_mouse_event(self, event)
        if event.type == "mouse_wheel":
            center = self._scene_transform.imap(event.pos)
            self.zoom((1 + self.zoom_factor) ** (-event.delta[1] * 30), center)
            event.handled = True
        elif event.type == "gesture_zoom":
            center = self._scene_transform.imap(event.pos)
            self.zoom(1 - event.scale, center)
            event.handled = True
        elif event.type == "mouse_move":
            if event.press_event is None:
                return
            modifiers = event.mouse_event.modifiers
            if 1 in event.buttons and not modifiers:
                p1 = np.array(event.last_event.pos)[:2]
                p2 = np.array(event.pos)[:2]
                p1s = self._transform.imap(p1)
                p2s = self._transform.imap(p2)
                self.pan(p1s - p2s)
                event.handled = True
            elif 2 in event.buttons and not modifiers:
                p1c = np.array(event.last_event.pos)[:2]
                p2c = np.array(event.pos)[:2]
                scale = (1 + self.zoom_factor) ** ((p1c - p2c) * np.array([1, -1]))
                center = self._transform.imap(event.press_event.pos[:2])
                self.zoom(scale, center)
                event.handled = True
            else:
                event.handled = False
        elif event.type == "mouse_press":
            event.handled = event.button in (1, 2)
        else:
            event.handled = False
        self._clamp_rect()

    def viewbox_resize_event(self, event):
        super().viewbox_resize_event(event)
        self._clamp_rect()


class AnimationController:
    """Time-based animation driver for 3D and ground-track views."""

    def __init__(self):
        self._t0 = None
        self._speed = DEFAULT_ANIMATION_SPEED
        self._duration = DEFAULT_ANIMATION_SECONDS
        self._running = False
        self._tick_callbacks = []
        self._timer = vispy_app.Timer(
            interval=TIMER_INTERVAL_S,  # pyright: ignore[reportArgumentType]
            start=False,
            connect=self._on_tick,
        )

    def add_tick_callback(self, callback):
        self._tick_callbacks.append(callback)

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = max(float(value), 0.01)

    def set_duration(self, seconds):
        self._duration = max(float(seconds), 0.1)

    def start(self):
        if not self._running:
            self._running = True
            if self._t0 is None:
                self._t0 = time.perf_counter()
            self._timer.start()

    def stop(self):
        self._running = False
        self._timer.stop()
        self._t0 = None
        for callback in self._tick_callbacks:
            callback(0.0, reset=True)

    def pause(self):
        if self._running:
            self._running = False
            self._timer.stop()

    def reset(self):
        self._t0 = None
        for callback in self._tick_callbacks:
            callback(0.0, reset=True)

    def _elapsed(self):
        if self._t0 is None:
            return 0.0
        return (time.perf_counter() - self._t0) * self._speed

    def _on_tick(self, _event):
        if not self._running:
            return
        if self._t0 is None:
            self._t0 = time.perf_counter()
        elapsed = self._elapsed()
        for callback in self._tick_callbacks:
            callback(elapsed, reset=False)


class OrbitApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Orbit viewer")
        self.resize(1100, 700)

        self._orbit = None
        self._t = None
        self._period = None
        self._r_eci = None
        self._v_eci = None
        self._r_ecef = None
        self._orbit_camera_distance = R_EARTH * 4
        self._trail_lon = None
        self._trail_lat = None
        self._sat_scale_3d = (1.0, 1.0, 1.0)
        self._last_sat_pos = None
        self._last_sat_vel = None

        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self._animation = AnimationController()
        self._setup_3d_panel()
        self._setup_2d_panel()
        self._animation.add_tick_callback(self._tick_3d)
        self._animation.add_tick_callback(self._tick_2d)

        self._view_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self._view_splitter.addWidget(self._panel_3d)
        self._view_splitter.addWidget(self._panel_2d)

        root_layout.addWidget(self._build_sidebar())
        root_layout.addWidget(self._view_splitter, stretch=1)

        self.setCentralWidget(central)
        self._update_panel_visibility()
        self._run_simulation()

    def _setup_3d_panel(self):
        self._panel_3d = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self._panel_3d)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas_3d = scene.SceneCanvas(
            keys="interactive",
            bgcolor=VIEW_BG,
            show=False,
        )
        grid = self._canvas_3d.central_widget.add_grid(spacing=0, margin=0)
        self._view_3d = grid.add_view(row=0, col=0, row_span=3, col_span=3)
        self._view_3d.camera = scene.cameras.TurntableCamera(
            fov=45,
            azimuth=225,
            elevation=25,
            distance=R_EARTH * 4,
            center=(0.0, 0.0, 0.0),
        )
        self._hook_axis_camera_sync()

        self._axis_view = grid.add_view(row=2, col=0)
        self._axis_view.width_max = 110
        self._axis_view.height_max = 110
        self._axis_view.camera = scene.cameras.TurntableCamera(
            fov=0,
            distance=2.5,
            elevation=25,
            azimuth=225,
        )
        self._axis_view.camera.interactive = False
        scene.visuals.XYZAxis(parent=self._axis_view.scene)
        self._sync_axis_camera()

        self._ref_parent = self._view_3d.scene
        self._earth_visual = None
        self._grid_visual = None
        self._orbit_path_3d = None
        self._trail_3d = None
        self._satellite_3d = None
        self._velocity_arrow_3d = None

        canvas_host = QtWidgets.QWidget()
        canvas_grid = QtWidgets.QGridLayout(canvas_host)
        canvas_grid.setContentsMargins(0, 0, 0, 0)
        canvas_grid.addWidget(self._canvas_3d.native, 0, 0)
        self._orbital_speed_label = QtWidgets.QLabel("v = — km/s", canvas_host)
        self._orbital_speed_label.setStyleSheet(
            "color: #e8e8e8; background-color: rgba(30, 30, 30, 0.82);"
            "padding: 6px 10px; border-radius: 4px; font-size: 13px;"
        )
        canvas_grid.addWidget(
            self._orbital_speed_label,
            0,
            0,
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(canvas_host)

    def _hook_axis_camera_sync(self):
        """Keep the corner axis gizmo aligned with the main 3D camera."""
        main_cam = self._view_3d.camera
        orig_view_changed = main_cam.view_changed

        def view_changed_with_axis_sync(*args, **kwargs):
            orig_view_changed(*args, **kwargs)
            self._sync_axis_camera()

        main_cam.view_changed = view_changed_with_axis_sync

    def _sync_axis_camera(self, _event=None):
        if self._axis_view is None or self._view_3d is None:
            return
        src = self._view_3d.camera
        dst = self._axis_view.camera
        dst.azimuth = src.azimuth
        dst.elevation = src.elevation
        dst.roll = src.roll

    def _update_orbital_speed_label(self, vel):
        if not hasattr(self, "_orbital_speed_label"):
            return
        self._orbital_speed_label.setText(
            f"v = {_orbital_speed_km_s(vel):.3f} km/s"
        )

    def _setup_2d_panel(self):
        self._panel_2d = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self._panel_2d)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas_2d = scene.SceneCanvas(
            keys="interactive",
            bgcolor=VIEW_BG,
            show=False,
        )
        self._view_2d = self._canvas_2d.central_widget.add_view()
        self._view_2d_map = None
        self._view_2d.camera = GroundTrackCamera()
        self._view_2d.camera.set_range(
            x=(MAP_BOUNDS[0], MAP_BOUNDS[0] + MAP_BOUNDS[2]),
            y=(MAP_BOUNDS[1], MAP_BOUNDS[1] + MAP_BOUNDS[3]),
        )

        self._add_ground_track_basemap()
        self._trail_2d = None
        self._marker_2d = scene.visuals.Markers(
            parent=self._view_2d.scene,
            symbol="disc",
            size=10,
            face_color=(1.0, 0.55, 0.1, 1.0),
        )
        _configure_ground_track_overlay(self._marker_2d, order=2, alpha=1.0)
        self._marker_2d.set_data(
            pos=np.array([[0.0, 0.0]], dtype=np.float32),
            edge_width=0,
        )
        layout.addWidget(self._canvas_2d.native)

    def _add_ground_track_basemap(self):
        """Coastlines and 30° graticule behind the ground-track overlay."""
        left, bottom, width, height = MAP_BOUNDS
        parent = self._view_2d.scene
        try:
            rgb = _build_ground_track_map_rgb()
            map_image = scene.visuals.Image(
                rgb,
                interpolation="linear",
                parent=parent,
            )
            map_image.transform = scene.transforms.STTransform(
                translate=(left, bottom),
                scale=(width / rgb.shape[1], height / rgb.shape[0]),
            )
            _configure_ground_track_basemap(map_image)
        except Exception:
            grid_pos, grid_connect = _graticule_line_data()
            grid = scene.visuals.Line(
                grid_pos,
                connect=grid_connect,
                color=(0.35, 0.42, 0.5, 0.65),
                width=1,
                parent=parent,
            )
            grid.order = 0
            scene.visuals.Line(
                _map_border_polyline(),
                color=(0.35, 0.4, 0.5, 0.8),
                width=1,
                connect="strip",
                parent=parent,
            )

    def _dispose_visual(self, attr):
        visual = getattr(self, attr, None)
        if visual is not None:
            visual.parent = None
        setattr(self, attr, None)

    def _build_3d_reference(self, half):
        self._dispose_visual("_earth_visual")
        self._dispose_visual("_grid_visual")

        grid_pos, grid_connect = _equatorial_grid(half)
        self._grid_visual = scene.visuals.Line(
            grid_pos,
            connect=grid_connect,
            color=(0.45, 0.5, 0.6, 0.35),
            width=1,
            parent=self._ref_parent,
        )
        _configure_scene_visual(self._grid_visual, order=_GL_ORDER_GRID, alpha=0.35)

        earth_mesh = create_sphere(rows=32, cols=32, radius=R_EARTH)
        self._earth_visual = scene.visuals.Mesh(
            meshdata=earth_mesh,
            color=(0.25, 0.45, 0.75, 1.0),
            shading="smooth",
            parent=self._ref_parent,
        )
        self._earth_visual.transform = scene.transforms.STTransform(translate=ORIGIN)
        _configure_scene_visual(self._earth_visual, order=_GL_ORDER_EARTH, alpha=1.0)

    def _run_simulation(self):
        h_max = self.h_max()
        h_min = self.h_min()
        step = self.step_size()

        self._orbit = Orbit(
            h_min=h_min,
            h_max=h_max,
            raan=self.raan(),
            inc=self.inc(),
            aop=self.aop(),
        )
        self._orbit.simulation_data = {}
        self._orbit.simulate_ground_track(n_orbits=self.n_orbits(), step=step)

        self._t = self._orbit.simulation_data["t"]
        self._period = float(self._orbit.params["T"])
        self._r_eci = self._orbit.slice_data(
            self._orbit.simulation_data["r_ECI"], n_orbits=1
        )
        self._v_eci = self._orbit.slice_data(
            self._orbit.simulation_data["v_ECI"], n_orbits=1
        )
        self._r_ecef = self._orbit.simulation_data["r_ECEF"]
        lon_lat = xyz_to_latlon(self._r_ecef)
        self._trail_lon = lon_lat[0]
        self._trail_lat = lon_lat[1]

        self._animation.set_duration(DEFAULT_ANIMATION_SECONDS)
        self._rebuild_scene_visuals()
        self._animation.reset()

    def _rebuild_scene_visuals(self):
        for attr in (
            "_orbit_path_3d",
            "_trail_3d",
            "_satellite_3d",
            "_velocity_arrow_3d",
        ):
            self._dispose_visual(attr)

        extent = float(np.max(np.linalg.norm(self._r_eci, axis=0)))
        half = max(extent, R_EARTH) * 1.05
        self._build_3d_reference(half)

        static_orbit = self._orbit.draw_ECI(n_points=200).T.astype(np.float32)
        self._orbit_path_3d = scene.visuals.Line(
            static_orbit,
            color=(0.2, 0.85, 0.95, 0.9),
            width=2,
            connect="strip",
            parent=self._ref_parent,
        )
        _configure_scene_visual(self._orbit_path_3d, order=_GL_ORDER_ORBIT, alpha=0.9)

        self._trail_3d = scene.visuals.Line(
            static_orbit[:1],
            color=(1.0, 0.55, 0.1, 0.55),
            width=2,
            connect="strip",
            parent=self._ref_parent,
        )
        _configure_scene_visual(self._trail_3d, order=_GL_ORDER_TRAIL, alpha=0.55)

        sat_radius = extent * 0.008
        self._sat_scale_3d = (sat_radius, sat_radius, sat_radius)
        sat_mesh = create_sphere(rows=10, cols=10, radius=1.0)
        self._satellite_3d = scene.visuals.Mesh(
            meshdata=sat_mesh,
            color=(1.0, 0.55, 0.1, 1.0),
            shading="smooth",
            parent=self._ref_parent,
        )
        _configure_scene_visual(
            self._satellite_3d, order=_GL_ORDER_SATELLITE, alpha=1.0
        )
        _update_mesh_position(self._satellite_3d, self._r_eci[:, 0], self._sat_scale_3d)

        pos0 = self._r_eci[:, 0]
        vel0 = self._v_eci[:, 0]
        vec0 = vel0 * VELOCITY_ARROW_DURATION_S
        tip0 = pos0 + vec0
        head_base0 = tip0 - 0.12 * vec0
        self._velocity_arrow_3d = scene.visuals.Arrow(
            pos=np.vstack([pos0, tip0]).astype(np.float32),
            color=(0.35, 1.0, 0.55, 1.0),
            width=2,
            arrows=np.array([np.concatenate([head_base0, tip0])], dtype=np.float32),
            arrow_size=14,
            arrow_color=(0.35, 1.0, 0.55, 1.0),
            parent=self._ref_parent,
        )
        _configure_scene_visual(
            self._velocity_arrow_3d, order=_GL_ORDER_VELOCITY, alpha=1.0
        )
        self._update_orbital_speed_label(vel0)

        self._orbit_camera_distance = max(extent * 2.5, R_EARTH * 3)
        if not self._chk_chase.isChecked():
            self._view_3d.camera.center = (0.0, 0.0, 0.0)
            self._view_3d.camera.distance = self._orbit_camera_distance
        else:
            self._update_chase_camera(self._r_eci[:, 0], self._v_eci[:, 0])
        self._sync_axis_camera()

        self._dispose_visual("_trail_2d")
        self._trail_2d = scene.visuals.Line(
            np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            color=(1.0, 0.85, 0.2, 1.0),
            width=3,
            connect="strip",
            parent=self._view_2d.scene,
        )
        _configure_ground_track_overlay(self._trail_2d, order=1, alpha=1.0)

        self._canvas_3d.update()
        self._canvas_2d.update()

    def _sim_time_3d(self, elapsed):
        duration = DEFAULT_ANIMATION_SECONDS
        phase = (elapsed % duration) / duration
        return phase * self._period

    def _sim_time_2d(self, elapsed):
        duration = DEFAULT_ANIMATION_SECONDS * (
            self._r_ecef.shape[1] / max(self._r_eci.shape[1], 1)
        )
        phase = (elapsed % duration) / duration
        return phase * (self._t[-1] if len(self._t) else self._period)

    def _chase_camera_enabled(self):
        return self._chk_chase.isChecked()

    def _update_chase_camera(self, pos, vel):
        cam = self._view_3d.camera
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        cam.center = tuple(float(x) for x in pos)
        cam.distance = self.chase_distance_m()
        offset = _chase_camera_offset_direction(
            vel, self.chase_azimuth_deg(), self.chase_pitch_deg()
        )
        cam.azimuth, cam.elevation = _offset_to_turntable_angles(offset)
        cam.roll = self.chase_roll_deg()
        self._sync_axis_camera()

    def _set_chase_controls_enabled(self, enabled):
        for widget in (
            self._chase_distance_slider,
            self._chase_roll,
            self._chase_azimuth,
            self._chase_pitch,
        ):
            widget.setEnabled(enabled)

    def _on_chase_settings_changed(self, *_args):
        if not self._chase_camera_enabled() or self._last_sat_pos is None:
            return
        self._update_chase_camera(self._last_sat_pos, self._last_sat_vel)
        self._canvas_3d.update()

    def _restore_orbit_camera(self):
        cam = self._view_3d.camera
        cam.center = (0.0, 0.0, 0.0)
        cam.distance = self._orbit_camera_distance
        cam.azimuth = 225
        cam.elevation = 25
        cam.roll = 0
        self._sync_axis_camera()

    def _on_chase_toggled(self, checked):
        self._set_chase_controls_enabled(checked)
        if self._r_eci is None:
            return
        if checked:
            pos = (
                self._last_sat_pos
                if self._last_sat_pos is not None
                else self._r_eci[:, 0]
            )
            vel = (
                self._last_sat_vel
                if self._last_sat_vel is not None
                else self._v_eci[:, 0]
            )
            self._update_chase_camera(pos, vel)
        else:
            self._restore_orbit_camera()
        self._canvas_3d.update()

    def _tick_3d(self, elapsed, *, reset):
        if self._r_eci is None or self._satellite_3d is None:
            return
        t_eci = self._t[: self._r_eci.shape[1]]
        if reset:
            pos = self._r_eci[:, 0]
            vel = self._v_eci[:, 0]
            self._trail_3d.set_data(
                pos=self._r_eci[:, :1].T.astype(np.float32),
                connect="strip",
            )
        else:
            t_query = self._sim_time_3d(elapsed)
            pos = _interp_positions(self._r_eci, t_eci, t_query)
            vel = _interp_positions(self._v_eci, t_eci, t_query)
            idx = int(np.searchsorted(t_eci, t_query, side="right"))
            idx = max(1, min(idx, self._r_eci.shape[1]))
            self._trail_3d.set_data(
                pos=self._r_eci[:, :idx].T.astype(np.float32),
                connect="strip",
            )
        self._last_sat_pos = pos
        self._last_sat_vel = vel
        self._update_orbital_speed_label(vel)
        _update_mesh_position(self._satellite_3d, pos, self._sat_scale_3d)
        if self._velocity_arrow_3d is not None:
            _update_velocity_arrow(self._velocity_arrow_3d, pos, vel)
        if self._chase_camera_enabled():
            self._update_chase_camera(pos, vel)
        elif reset:
            self._restore_orbit_camera()
        self._canvas_3d.update()

    def _update_trail_2d(self, idx):
        lon = self._trail_lon[: idx + 1]
        lat = self._trail_lat[: idx + 1]
        pos, connect = _lonlat_line_strip(lon, lat)
        if pos is None:
            pos = np.array([[lon[0], lat[0]], [lon[0], lat[0]]], dtype=np.float32)
            connect = "strip"
        self._trail_2d.set_data(pos=pos, connect=connect)
        self._trail_2d.update()

    def _tick_2d(self, elapsed, *, reset):
        if self._r_ecef is None or self._trail_2d is None:
            return
        n = self._r_ecef.shape[1]
        if reset:
            idx = 0
        else:
            t_query = self._sim_time_2d(elapsed)
            idx = int(np.searchsorted(self._t, t_query, side="right") - 1)
            idx = int(np.clip(idx, 0, n - 1))
        self._update_trail_2d(idx)
        self._marker_2d.set_data(
            pos=np.array(
                [[self._trail_lon[idx], self._trail_lat[idx]]], dtype=np.float32
            ),
            edge_width=0,
        )
        self._canvas_2d.update()

    def _build_sidebar(self):
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(260)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        display_group = QtWidgets.QGroupBox("Simulation panels")
        display_layout = QtWidgets.QVBoxLayout(display_group)
        self._chk_3d = QtWidgets.QCheckBox("3D (ECI)")
        self._chk_2d = QtWidgets.QCheckBox("Ground track")
        self._chk_3d.setChecked(True)
        self._chk_2d.setChecked(True)
        self._chk_3d.toggled.connect(self._update_panel_visibility)
        self._chk_2d.toggled.connect(self._update_panel_visibility)
        display_layout.addWidget(self._chk_3d)
        display_layout.addWidget(self._chk_2d)
        layout.addWidget(display_group)

        params_group = QtWidgets.QGroupBox("Parameters")
        params_form = QtWidgets.QFormLayout(params_group)

        self._h_max = self._altitude_spin(DEFAULT_H_MAX)
        self._h_min = self._altitude_spin(DEFAULT_H_MIN)
        self._step_input = QtWidgets.QLineEdit(str(int(DEFAULT_STEP_S)))
        self._step_input.setPlaceholderText("seconds")

        self._n_orbits = QtWidgets.QSpinBox()
        self._n_orbits.setRange(1, 50)
        self._n_orbits.setValue(DEFAULT_N_ORBITS)

        params_form.addRow("h_max", self._h_max)
        params_form.addRow("h_min", self._h_min)
        params_form.addRow("step", self._step_input)
        params_form.addRow("n_orbits", self._n_orbits)
        layout.addWidget(params_group)

        angles_group = QtWidgets.QGroupBox("Euler angles")
        angles_form = QtWidgets.QFormLayout(angles_group)
        self._raan = self._angle_spin(DEFAULT_RAAN_DEG)
        self._inc = self._angle_spin(DEFAULT_INC_DEG, maximum=180.0)
        self._aop = self._angle_spin(DEFAULT_AOP_DEG)
        angles_form.addRow("RAAN", self._raan)
        angles_form.addRow("inc", self._inc)
        angles_form.addRow("aop", self._aop)
        layout.addWidget(angles_group)

        self._btn_run = QtWidgets.QPushButton("Run simulation")
        self._btn_run.clicked.connect(self._on_run_simulation)
        layout.addWidget(self._btn_run)

        speed_group = QtWidgets.QGroupBox("Animation")
        speed_layout = QtWidgets.QVBoxLayout(speed_group)
        self._speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 50)
        self._speed_slider.setValue(int(DEFAULT_ANIMATION_SPEED * 10))
        self._speed_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self._speed_slider.setTickInterval(10)
        self._speed_label = QtWidgets.QLabel(f"Speed: {DEFAULT_ANIMATION_SPEED:.1f}×")
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self._speed_label)
        speed_layout.addWidget(self._speed_slider)
        layout.addWidget(speed_group)

        chase_group = QtWidgets.QGroupBox("Chase camera")
        chase_layout = QtWidgets.QVBoxLayout(chase_group)
        self._chk_chase = QtWidgets.QCheckBox("Enable chase view")
        self._chk_chase.toggled.connect(self._on_chase_toggled)
        chase_layout.addWidget(self._chk_chase)
        chase_form = QtWidgets.QFormLayout()
        distance_row = QtWidgets.QWidget()
        distance_layout = QtWidgets.QHBoxLayout(distance_row)
        distance_layout.setContentsMargins(0, 0, 0, 0)
        self._chase_distance_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self._chase_distance_slider.setRange(
            int(CHASE_DISTANCE_MIN_KM), int(CHASE_DISTANCE_MAX_KM)
        )
        self._chase_distance_slider.setValue(int(DEFAULT_CHASE_DISTANCE_KM))
        self._chase_distance_slider.setSingleStep(50)
        self._chase_distance_slider.setTickPosition(
            QtWidgets.QSlider.TickPosition.TicksBelow
        )
        self._chase_distance_slider.setTickInterval(10_000)
        self._chase_distance_label = QtWidgets.QLabel(
            f"{DEFAULT_CHASE_DISTANCE_KM:.0f} km"
        )
        self._chase_distance_slider.valueChanged.connect(
            self._on_chase_distance_changed
        )
        distance_layout.addWidget(self._chase_distance_slider, stretch=1)
        distance_layout.addWidget(self._chase_distance_label)
        self._chase_roll = self._chase_angle_spin(DEFAULT_CHASE_ROLL_DEG)
        self._chase_azimuth = self._chase_angle_spin(DEFAULT_CHASE_AZIMUTH_DEG)
        self._chase_pitch = self._chase_angle_spin(DEFAULT_CHASE_PITCH_DEG)
        chase_form.addRow("Distance", distance_row)
        chase_form.addRow("Roll", self._chase_roll)
        chase_form.addRow("Azimuth", self._chase_azimuth)
        chase_form.addRow("Pitch", self._chase_pitch)
        chase_layout.addLayout(chase_form)
        for spin in (self._chase_roll, self._chase_azimuth, self._chase_pitch):
            spin.valueChanged.connect(self._on_chase_settings_changed)
        self._set_chase_controls_enabled(False)
        layout.addWidget(chase_group)

        controls_group = QtWidgets.QGroupBox("Controls")
        controls_layout = QtWidgets.QGridLayout(controls_group)
        self._btn_start = QtWidgets.QPushButton("Start")
        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_pause = QtWidgets.QPushButton("Pause")
        self._btn_reset = QtWidgets.QPushButton("Reset")
        self._btn_start.clicked.connect(self._animation.start)
        self._btn_stop.clicked.connect(self._animation.stop)
        self._btn_pause.clicked.connect(self._animation.pause)
        self._btn_reset.clicked.connect(self._animation.reset)
        controls_layout.addWidget(self._btn_start, 0, 0)
        controls_layout.addWidget(self._btn_stop, 0, 1)
        controls_layout.addWidget(self._btn_pause, 1, 0)
        controls_layout.addWidget(self._btn_reset, 1, 1)
        layout.addWidget(controls_group)

        layout.addStretch(1)
        return panel

    def _on_run_simulation(self):
        self._animation.stop()
        self._run_simulation()

    def _update_panel_visibility(self):
        show_3d = self._chk_3d.isChecked()
        show_2d = self._chk_2d.isChecked()

        self._panel_3d.setVisible(show_3d)
        self._panel_2d.setVisible(show_2d)

        if show_3d and show_2d:
            self._view_splitter.setSizes([1, 1])
        elif show_3d:
            self._view_splitter.setSizes([1, 0])
        elif show_2d:
            self._view_splitter.setSizes([0, 1])

        self._canvas_3d.update()
        self._canvas_2d.update()

    @staticmethod
    def _altitude_spin(value):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, DEFAULT_H_MAX / 1000)
        spin.setDecimals(0)
        spin.setSuffix(" km")
        spin.setSingleStep(1)
        spin.setValue(value / 1000)
        return spin

    @staticmethod
    def _angle_spin(value, minimum=0.0, maximum=360.0):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(1)
        spin.setSuffix(" °")
        spin.setSingleStep(1.0)
        spin.setWrapping(maximum >= 360.0)
        spin.setValue(value)
        return spin

    def _on_speed_changed(self, value):
        speed = value / 10.0
        self._speed_label.setText(f"Speed: {speed:.1f}×")
        self._animation.speed = speed

    def _on_chase_distance_changed(self, value):
        self._chase_distance_label.setText(f"{value} km")
        self._on_chase_settings_changed()

    def h_max(self):
        return self._h_max.value() * 1000

    def h_min(self):
        return self._h_min.value() * 1000

    def chase_distance_m(self):
        return self._chase_distance_slider.value() * 1000.0

    @staticmethod
    def _chase_angle_spin(value):
        return OrbitApp._angle_spin(value, minimum=-180.0, maximum=180.0)

    def chase_roll_deg(self):
        return self._chase_roll.value()

    def chase_azimuth_deg(self):
        return self._chase_azimuth.value()

    def chase_pitch_deg(self):
        return self._chase_pitch.value()

    def raan(self):
        return self._raan.value()

    def inc(self):
        return self._inc.value()

    def aop(self):
        return self._aop.value()

    def n_orbits(self):
        return self._n_orbits.value()

    def step_size(self):
        text = self._step_input.text().strip()
        if not text:
            return DEFAULT_STEP_S
        try:
            step = float(text)
        except ValueError:
            return DEFAULT_STEP_S
        return max(step, 1.0)


def main():
    use_app("pyside6")
    qt_app = QtWidgets.QApplication(sys.argv)
    window = OrbitApp()
    window.show()
    return qt_app.exec()


if __name__ == "__main__":
    sys.exit(main())
