import numpy as np
from scipy.integrate import solve_ivp

DTYPE = np.float32
INTEGRATE_DTYPE = np.float64


def radial_to_xyz(r, theta):
    theta = np.asarray(theta, dtype=DTYPE)
    r = np.asarray(r, dtype=DTYPE)
    return r * np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])


def tangential_to_xyz(r, theta):
    theta = np.asarray(theta, dtype=DTYPE)
    r = np.asarray(r, dtype=DTYPE)
    return r * np.vstack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])


def rk4_step(f, x, h):
    k1 = f(x)
    k2 = f(x + h * k1 / 2)
    k3 = f(x + h * k2 / 2)
    k4 = f(x + h * k3)
    return np.asarray(h / 6 * (k1 + 2 * (k2 + k3) + k4), dtype=INTEGRATE_DTYPE)


def integrate(f, x, h):
    """Advance state by ``h`` with SciPy RK45; returns the increment only (drop-in for rk4_step)."""
    x = np.asarray(x, dtype=INTEGRATE_DTYPE)
    h = INTEGRATE_DTYPE(h)

    def fun(_t, y):
        return f(y)

    sol = solve_ivp(
        fun,
        (0.0, h),
        x,
        method="RK45",
        t_eval=[h],
        rtol=1e-9,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    return np.asarray(sol.y[:, -1], dtype=INTEGRATE_DTYPE)


def xyz_to_latlon(xyz):
    """
    Converts position vector(s) in ECEF or ECI x, y, z coordinates to (latitude, longitude) in degrees.
    Accepts 1D (len=3) or 2D (shape=(3,N)) numpy arrays or lists.

    Returns:
        lat (float or np.ndarray): latitude in degrees
        lon (float or np.ndarray): longitude in degrees
    """
    xyz = np.asarray(xyz)
    if xyz.ndim == 1:
        x, y, z = xyz
    else:
        x, y, z = xyz[0], xyz[1], xyz[2]

    # Longitude: arctan2(y, x)
    lon = np.arctan2(y, x)

    # Hypotenuse projected onto x-y plane
    hyp = np.sqrt(x**2 + y**2)

    # Latitude: arctan2(z, hyp)
    lat = np.arctan2(z, hyp)

    # Convert from radians to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    return np.vstack(
        [
            lon_deg,
            lat_deg,
        ]
    )


def break_at_dateline(lat, lon, threshold=180):
    lat = np.asarray(lat).copy()
    lon = np.asarray(lon).copy()
    jumps = np.abs(np.diff(lon)) > threshold

    lon[1:][jumps] = np.nan
    lat[1:][jumps] = np.nan

    return lon, lat
