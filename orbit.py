# %%
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import GM_earth, R_earth

from utils import (
    DTYPE,
    INTEGRATE_DTYPE,
    break_at_dateline,
    integrate,
    radial_to_xyz,
    tangential_to_xyz,
    xyz_to_latlon,
)

MU = GM_earth.to_value()
R_EARTH = R_earth.to_value()
# Wall-clock seconds for one animation loop along the sampled trajectory.
DEFAULT_ANIMATION_SECONDS = 45.0


class Orbit:
    def __init__(
        self, h_min, h_max, raan=0.0, inc=0.0, aop=0.0, convert_to_rad=True
    ) -> None:
        ra = R_EARTH + h_max
        rp = R_EARTH + h_min
        a = (ra + rp) / 2
        e = (ra - rp) / (2 * a)
        b = a * np.sqrt(1 - e**2)
        c = e * a
        T = 2 * np.pi * np.sqrt(a**3 / MU)
        E = -MU / (2 * a)

        if convert_to_rad:
            raan = np.deg2rad(raan)
            inc = np.deg2rad(inc)
            aop = np.deg2rad(aop)

        self.params = {
            "h_max": h_max,
            "h_min": h_min,
            "ra": ra,
            "rp": rp,
            "a": a,
            "b": b,
            "c": c,
            "e": e,
            "T": T,
            "E": E,
            "p": a * (1 - e**2),
            "RAAN": raan,
            "INC": inc,
            "AOP": aop,
        }

        self.simulation_data = {}

    @property
    def rotation_matrix(self):
        raan, inc, aop = self.params["RAAN"], self.params["INC"], self.params["AOP"]

        raan_mat = np.array(
            [
                [np.cos(raan), -np.sin(raan), 0],
                [np.sin(raan), np.cos(raan), 0],
                [0, 0, 1],
            ],
            dtype=DTYPE,
        )

        inc_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(inc), -np.sin(inc)],
                [0, np.sin(inc), np.cos(inc)],
            ],
            dtype=DTYPE,
        )

        aop_mat = np.array(
            [
                [np.cos(aop), -np.sin(aop), 0],
                [np.sin(aop), np.cos(aop), 0],
                [0, 0, 1],
            ],
            dtype=DTYPE,
        )

        return raan_mat @ inc_mat @ aop_mat

    def radius(self, theta):
        return self.params["p"] / (1 + np.cos(theta) * self.params["e"])

    def draw_ECI(self, n_points=100):
        ta = np.linspace(0, 2 * np.pi, n_points)
        return self.rotation_matrix @ radial_to_xyz(r=self.radius(ta), theta=ta)

    def simulate_orbit_ECI(self, n_orbits=1, step=None):
        # Simulation should not be run if n_orbits is smaller than or equal to simulation_data["n_orbits"]

        if self.simulation_data != {} and self.simulation_data["n_orbits"] >= n_orbits:
            print(
                f"Simulation was already run for {self.simulation_data['n_orbits']}, larger than {n_orbits}."
            )

            return

        self.simulation_data["step_size"] = step
        self.simulation_data["n_orbits"] = n_orbits

        t_stop = self.params["T"] * n_orbits
        if step is None:
            step = max(self.params["T"] / 600.0, 30.0)

        r0 = float(self.radius(0.0))
        a = self.params["a"]
        speed = np.sqrt(MU * (2.0 / r0 - 1.0 / a))
        X = np.array([r0, 0.0, 0.0, speed / r0], dtype=INTEGRATE_DTYPE)

        def state_derivative(X):
            r = max(float(X[0]), 1.0)
            return np.array(
                [
                    X[1],
                    X[0] * X[3] ** 2 - MU / r**2,
                    X[3],
                    -2.0 * X[1] * X[3] / r,
                ],
                dtype=INTEGRATE_DTYPE,
            )

        i = 0
        states = []

        print("Simulation started.")

        while i * step < t_stop:
            states.append(X.copy())
            X = integrate(state_derivative, X, step)
            i += 1
            if not np.all(np.isfinite(X)):
                raise RuntimeError(
                    "Orbit integration diverged; try a smaller step size."
                )

        print("Simulation complete.")

        timestamps = np.arange(i, dtype=INTEGRATE_DTYPE) * step
        states = np.vstack(states).T
        R = self.rotation_matrix.astype(INTEGRATE_DTYPE)
        vct_r = (R @ radial_to_xyz(states[0], states[2])).astype(DTYPE)
        vct_v = (
            R
            @ (
                radial_to_xyz(states[1], states[2])
                + tangential_to_xyz(states[0] * states[3], states[2])
            )
        ).astype(DTYPE)

        print("New simulation data set.")

        self.simulation_data["t"] = timestamps
        self.simulation_data["r_ECI"] = vct_r
        self.simulation_data["v_ECI"] = vct_v

    def simulate_ground_track(self, n_orbits=1, step=None):
        ang_freq = 2 * np.pi / (24 * 3600)

        self.simulate_orbit_ECI(n_orbits, step)

        t = self.simulation_data["t"]
        r_ECI = self.simulation_data["r_ECI"]

        r_ECEF = []

        for t, r in zip(t, r_ECI.T):
            R = np.array(
                [
                    [np.cos(ang_freq * t), np.sin(ang_freq * t), 0],
                    [-np.sin(ang_freq * t), np.cos(ang_freq * t), 0],
                    [0, 0, 1],
                ]
            )

            r_ECEF.append(R @ r.reshape(3, 1))

        r_ECEF = np.hstack(r_ECEF).astype(DTYPE)

        self.simulation_data["r_ECEF"] = r_ECEF

    def to_lat_lon(self, r_ECEF=None, for_orbits=None):
        if r_ECEF is None:
            assert self.simulation_data["r_ECEF"] is not None, (
                "Ground track must be simulated first, or provide r_ECEF!"
            )

            r_ECEF = self.simulation_data["r_ECEF"]

        lon_lat = xyz_to_latlon(r_ECEF)

        sliced = self.slice_data(lon_lat, for_orbits)

        return break_at_dateline(lon=sliced[0], lat=sliced[1])

    def slice_data(self, data, n_orbits=None):
        """
        `data` should have shape `(..., N)`, where `N` is the number of data points.
        """
        if n_orbits is None:
            n_orbits = self.simulation_data["n_orbits"]

        last_index = int(
            (self.params["T"] * n_orbits) / self.simulation_data["step_size"]
        )

        return data[..., :last_index]


# %%
if __name__ == "__main__":
    orbit = Orbit(h_min=500e3, h_max=36000e3, inc=98.0, aop=270.0)

    orbit.simulate_ground_track(step=60, n_orbits=4)

    plt.plot(*orbit.to_lat_lon(for_orbits=1))
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.grid(True)
    plt.show()
