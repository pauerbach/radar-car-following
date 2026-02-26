from typing import List, Dict, TYPE_CHECKING, Optional, Union
from gymnasium import spaces
import gymnasium as gym

# gym.logger.set_level(40)
import numpy as np
import pandas as pd
from scipy import signal
from numba import jit, complex64, prange
import pyfftw
import matplotlib.pyplot as plt
import cv2

from highway_env import utils

# from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.controller import MDPVehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

c = 3e8  # Speed of light (m/s)


def ca_cfar_2d(power_map, guard=1, train=4, pfa=1e-3):
    """
    2D Cell-Averaging CFAR on linear power map.
    guard: guard cells on each side (square guard region)
    train: training cells thickness around guard
    pfa: desired probability of false alarm
    Returns boolean detection map of same shape.
    """
    H, W = power_map.shape
    half = guard + train  # half window radius including guard+train

    # Integral image for fast rectangular sums
    I = np.pad(power_map, ((1, 0), (1, 0)), mode="constant")
    I = np.cumsum(np.cumsum(I, axis=0), axis=1)

    def rect_sum(y0, x0, y1, x1):
        return I[y1 + 1, x1 + 1] - I[y0, x1 + 1] - I[y1 + 1, x0] + I[y0, x0]

    det = np.zeros_like(power_map, dtype=bool)
    N_train = (2 * half + 1) ** 2 - (2 * guard + 1) ** 2
    alpha = N_train * (pfa ** (-1.0 / max(N_train, 1)) - 1.0)

    for y in range(half, H - half):
        for x in range(half, W - half):
            y0, x0 = y - half, x - half
            y1, x1 = y + half, x + half
            total_sum = rect_sum(y0, x0, y1, x1)

            yg0, xg0 = y - guard, x - guard
            yg1, xg1 = y + guard, x + guard
            guard_sum = rect_sum(yg0, xg0, yg1, xg1)

            noise_sum = total_sum - guard_sum
            noise_mean = noise_sum / max(N_train, 1)
            threshold = alpha * noise_mean

            if power_map[y, x] > threshold:
                det[y, x] = True
    return det


class Draw:
    # Represents drawing for example
    #
    # Draw is done for each antenna, and each antenna is represented for
    # other subplot

    def __init__(self, max_speed_m_s, max_range_m, title=""):
        # max_range_m:   maximum supported range
        # max_speed_m_s: maximum supported speed
        self._h = None
        self._max_speed_m_s = max_speed_m_s
        self._max_range_m = max_range_m
        self.title = title

        plt.ion()

        self._fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((2) // 2, 2))
        self._ax = ax

        self._fig.canvas.manager.set_window_title("Doppler")
        self._fig.canvas.mpl_connect("close_event", self.close)
        self._is_window_open = True

    def _draw_first_time(self, data):
        # First time draw
        #
        # It computes minimal, maximum value and draw data for all antennas
        # in same scale

        minmin = np.min(data)
        maxmax = np.max(data)

        h = self._ax.imshow(
            data,
            vmin=minmin,
            vmax=maxmax,
            extent=(
                -self._max_speed_m_s,
                self._max_speed_m_s,
                0,
                self._max_range_m / 2,
            ),
            aspect="auto",
            origin="lower",
        )
        self._h = h

        self._ax.set_xlabel("velocity (m/s)")
        self._ax.set_ylabel("distance (m)")
        self._ax.set_title(self.title)
        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])

        cbar = self._fig.colorbar(self._h, cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (dB)")

    def _draw_next_time(self, data):
        self._h.set_data(data)

    def draw(self, data):
        if self._is_window_open:
            if self._h is None:  # handle the first run
                self._draw_first_time(data)
            else:
                self._draw_next_time(data)

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self, event=None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close("all")  # Needed for Matplotlib ver: 3.4.0 and 3.4.1
            print("Application closed!")

    def is_open(self):
        return self._is_window_open


@jit(nopython=True, parallel=False, fastmath=False)
def simulate_ground_clutter(
    fc,
    B,
    T_chirp,
    N_r,
    N_c,
    N_rx,
    ego_velocity,
    antenna_height=0.08,
    R_min=0.02,
    R_max=0.25,
    n_scatterers=100,
    clutter_power=0.05,
    doppler_spread=0.1,
):
    """
    Simulate distributed ground clutter for a forward-looking FMCW radar.
    """

    c = 3e8
    lambda_radar = c / fc
    d = c / (2 * fc)

    t = (np.arange(N_r) / N_r) * T_chirp
    chirps = np.arange(N_c) * T_chirp

    clutter_cube = np.zeros((N_r, N_c, N_rx), dtype=np.complex64)

    for _ in range(n_scatterers):
        # Random ground scatterer range
        R = np.random.uniform(R_min, R_max)

        # Doppler centered at ego velocity with spread
        v = ego_velocity + np.random.randn() * doppler_spread

        # Beat and Doppler frequencies
        f_b = 2 * B * R / (c * T_chirp)
        f_d = 2 * v * fc / c

        # Path loss and grazing angle attenuation
        path_loss = clutter_power / (R**2 + antenna_height**2)

        # Random phase
        phi0 = np.random.uniform(0, 2 * np.pi)

        fast_phase = np.exp(1j * (2 * np.pi * f_b * t + phi0))
        doppler_phase = np.exp(1j * (2 * np.pi * f_d * chirps))

        for rx in range(N_rx):
            rx_phase = np.exp(1j * 2 * np.pi * rx * d / lambda_radar)
            for c_idx in range(N_c):
                clutter_cube[:, c_idx, rx] += (
                    path_loss * rx_phase * doppler_phase[c_idx] * fast_phase
                )

    return clutter_cube


@jit(nopython=True, parallel=False, fastmath=False)
def simulate_ground_clutter_fast(
    fc,
    B,
    T_chirp,
    N_r,
    N_c,
    N_rx,
    ego_velocity,
    antenna_height=0.08,
    R_min=0.02,
    R_max=0.25,
    n_scatterers=100,
    clutter_power=0.05,
    doppler_spread=0.1,
):
    c = 3e8
    lambda_radar = c / fc
    d = c / (2 * fc)

    t = (np.arange(N_r) / N_r) * T_chirp  # (N_r,)
    chirps = np.arange(N_c) * T_chirp  # (N_c,)
    rx_idx = np.arange(N_rx)  # (N_rx,)

    clutter_cube = np.zeros((N_r, N_c, N_rx), dtype=np.complex64)

    for _ in range(n_scatterers):
        R = np.random.uniform(R_min, R_max)
        v = ego_velocity + np.random.randn() * doppler_spread

        f_b = 2 * B * R / (c * T_chirp)
        f_d = 2 * v * fc / c

        path_loss = clutter_power / (R**2 + antenna_height**2)
        phi0 = np.random.uniform(0, 2 * np.pi)

        fast_phase = np.exp(1j * (2 * np.pi * f_b * t + phi0))  # (N_r,)
        doppler_phase = np.exp(1j * 2 * np.pi * f_d * chirps)  # (N_c,)
        rx_phase = np.exp(1j * 2 * np.pi * rx_idx * d / lambda_radar)  # (N_rx,)

        # Rank-1 outer products
        clutter_cube += (
            path_loss
            * fast_phase[:, None, None]
            * doppler_phase[None, :, None]
            * rx_phase[None, None, :]
        )

    return clutter_cube


@jit(nopython=True, parallel=False, fastmath=False)
def simulate_ground_clutter_fast_vectorized(
    fc,
    B,
    T_chirp,
    N_r,
    N_c,
    N_rx,
    ego_velocity,
    antenna_height=0.08,
    R_min=0.02,
    R_max=0.25,
    n_scatterers=100,
    clutter_power=0.05,
    doppler_spread=0.1,
):
    """
    Vectorized ground clutter simulation.
    Equivalent to simulate_ground_clutter_fast, but without Python loops.
    """

    c = 3e8
    lambda_radar = c / fc
    d = c / (2 * fc)

    # Axes
    t = (np.arange(N_r) / N_r) * T_chirp  # (N_r,)
    chirps = np.arange(N_c) * T_chirp  # (N_c,)
    rx_idx = np.arange(N_rx)  # (N_rx,)

    # ------------------------------------------------
    # Sample scatterers (S)
    # ------------------------------------------------
    S = n_scatterers

    R = np.random.uniform(R_min, R_max, S)  # (S,)
    v = ego_velocity + np.random.randn(S) * doppler_spread  # (S,)
    phi0 = np.random.uniform(0, 2 * np.pi, S)  # (S,)

    path_loss = clutter_power / (R**2 + antenna_height**2)  # (S,)

    f_b = 2 * B * R / (c * T_chirp)  # (S,)
    f_d = 2 * v * fc / c  # (S,)

    # ------------------------------------------------
    # Phase terms
    # ------------------------------------------------
    # Fast time: (S, N_r)
    fast_phase = np.exp(1j * (2 * np.pi * f_b[:, None] * t[None, :] + phi0[:, None]))

    # Doppler slow time: (S, N_c)
    doppler_phase = np.exp(1j * 2 * np.pi * f_d[:, None] * chirps[None, :])

    # RX phase: (N_rx,)
    rx_phase = np.exp(1j * 2 * np.pi * rx_idx * d / lambda_radar)

    # ------------------------------------------------
    # Combine via broadcasting
    # ------------------------------------------------
    # Result before summation:
    # (S, N_r, N_c, N_rx)
    clutter = (
        path_loss[:, None, None, None]
        * fast_phase[:, :, None, None]
        * doppler_phase[:, None, :, None]
        * rx_phase[None, None, None, :]
    )

    # Sum over scatterers
    clutter_cube = np.sum(clutter, axis=0).astype(np.complex64)

    return clutter_cube


class RadarSimulator:
    def __init__(self, fc, B, T_chirp, n_samples, n_chirp, n_rx, add_zero_noise):
        # ----------------------------
        # FMCW waveform
        # ----------------------------
        self.fc = fc  # Carrier frequency (Hz)
        self.B = B  # Bandwidth (Hz)
        self.T_chirp = T_chirp  # Chirp duration (s)
        self.n_samples = n_samples  # Number of range samples per chirp
        self.n_chirp = n_chirp  # Number of chirps (for Doppler)
        self.n_rx = n_rx  # number or receive antennas (ULA)

        self.add_zero_noise = add_zero_noise

        # ----------------------------
        # Derived parameters
        # ----------------------------
        range_res = c / (2 * B)
        max_range = range_res * n_samples
        lambda_radar = c / fc
        v_max = lambda_radar / (4 * T_chirp)
        v_res = lambda_radar / (2 * n_chirp * T_chirp)
        # print(
        #     f"Range resolution: {range_res:.2f} m, Max range: {max_range:.2f} m, Max velocity: {v_max:.2f} m/s, Vel resolution: {v_res:.2f}"
        # )

    def get_max_range(self):
        range_res = c / (2 * self.B)
        max_range = range_res * self.n_samples

        return max_range

    def get_max_velocity(self):
        lambda_radar = c / self.fc
        v_max = lambda_radar / (4 * self.T_chirp)

        return v_max

    def add_thermal_noise(self, radar_cube, SNR_dB):
        # ----------------------------
        # Add thermal noise
        # ----------------------------
        signal_power = np.mean(np.abs(radar_cube) ** 2)
        # print(signal_power)
        # signal_power = 500
        noise_power = signal_power / (10 ** (SNR_dB / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*radar_cube.shape) + 1j * np.random.randn(*radar_cube.shape)
        )
        radar_cube += noise

        return radar_cube

    def simulate_radar(self, target, ego_vel):
        radar_cube = RadarSimulator._simulate_radar_optimized(
            target,
            self.fc,
            self.B,
            self.T_chirp,
            self.n_samples,
            self.n_chirp,
            self.n_rx,
            self.add_zero_noise,
        )

        ground_clutter = simulate_ground_clutter_fast_vectorized(
            fc=self.fc,
            B=self.B,
            T_chirp=self.T_chirp,
            N_r=self.n_samples,
            N_c=self.n_chirp,
            N_rx=self.n_rx,
            # ego_velocity=target[1],  # vehicle speed
            ego_velocity=ego_vel,  # vehicle speed
            clutter_power=0.007,
        )

        return radar_cube + ground_clutter

    @jit(nopython=True, parallel=False, fastmath=False)
    def _simulate_radar_optimized(
        target, fc, B, T_chirp, N_r, N_c, N_rx, add_zero_noise
    ):
        # ----------------------------
        # Internal Multipath Model for Corner Reflector
        # ----------------------------
        # Models multiple internal bounces between reflector plates
        # producing delayed replicas (ghost peaks in range profile)
        def corner_internal_reflections(R, base_amplitude, v_base, num_bounces=3):
            """
            Simulate multiple internal reflections in a moving trihedral corner reflector.
            Each internal bounce adds:
            - extra path length  -> shifted range
            - small Doppler offset -> spectral spreading

            Returns a list of (effective_range, amplitude_scale, effective_velocity)
            """
            reflections = [(R, base_amplitude, v_base)]

            for i in range(1, num_bounces + 1):
                # extra path due to internal reflection geometry (meters)
                extra_path = (
                    0.08 * i
                )  # ~8cm per bounce #TODO make this a configurable parmater
                effective_R = R + extra_path

                # amplitude decay with each bounce
                amp_scale = base_amplitude * (
                    0.3**i
                )  # TODO make this a configurable parmater

                # each bounce creates a slight Doppler offset (micro-motion effect)
                v_eff = v_base * (
                    1 + 0.25 * i
                )  # TODO make this a configurable parmater

                reflections.append((effective_R, amp_scale, v_eff))

            return reflections

        c = 3e8

        lambda_radar = c / fc
        d = c / (2 * fc)

        # Fast time & slow time
        t = (np.arange(N_r, dtype=np.float32) / N_r) * T_chirp
        chirps = np.arange(N_c, dtype=np.float32) * T_chirp

        radar_cube = np.zeros((N_r, N_c, N_rx), dtype=complex64)

        # Target parameters
        R, v, angle_deg, RCS = target
        angle_center = np.deg2rad(angle_deg)

        # create internal multipath reflections (ghost ranges + Doppler spreads)
        reflection_paths = corner_internal_reflections(R, RCS, v, num_bounces=2)

        # Corner reflector multipath (fixed small count)
        # reflection_paths = [(R, RCS, v),
        #                     (R + 0.08, RCS * 0.3, v * 1.25),
        #                     (R + 0.16, RCS * 0.09, v * 1.5)]

        # Clutter
        sensor_clutter_rcs = 1.5
        sensor_range_base = 0.01
        f_b_clutter = 2 * B * sensor_range_base / (c * T_chirp)
        clutter_fast = np.exp(1j * (2 * np.pi * f_b_clutter * t))

        # Precompute RX phase
        rx_phase = np.empty(N_rx, dtype=complex64)
        for n_rx in range(N_rx):
            phase_angle = 2 * np.pi * n_rx * d * np.sin(angle_center) / lambda_radar
            rx_phase[n_rx] = np.exp(1j * phase_angle)

        for eff_R, eff_RCS, eff_v in reflection_paths:
            f_b = 2 * B * eff_R / (c * T_chirp)
            f_d = 2 * eff_v * fc / c

            # Fast-time exponential (ONCE)
            fast_phase = np.exp(1j * (2 * np.pi * f_b * t))

            # Doppler slow-time phase (ONCE)
            doppler_phase = np.exp(1j * (2 * np.pi * f_d * chirps))

            path_loss = 1.0 / (eff_R**2)
            # path_loss = 1.0

            for n_rx in range(N_rx):
                rx_p = rx_phase[n_rx]

                for n_c in range(N_c):
                    dp = doppler_phase[n_c]

                    radar_cube[:, n_c, n_rx] += (
                        path_loss * eff_RCS * rx_p * dp * fast_phase
                    )

                    if add_zero_noise:
                        # clutter (both Doppler signs)
                        radar_cube[:, n_c, n_rx] += (
                            sensor_clutter_rcs * dp * clutter_fast
                        )
                        radar_cube[:, n_c, n_rx] += (
                            sensor_clutter_rcs * np.conj(dp) * clutter_fast
                        )

        return radar_cube

    @jit(nopython=True, parallel=True, fastmath=False)
    def _simulate_radar(target, fc, B, T_chirp, N_r, N_c, N_rx, add_zero_noise):
        c = 3e8  # Speed of light (m/s)

        # Derived parameters
        lambda_radar = c / fc
        d = c / (2 * fc)  # half-wavelength spacing

        # ----------------------------
        # Internal Multipath Model for Corner Reflector
        # ----------------------------
        # Models multiple internal bounces between reflector plates
        # producing delayed replicas (ghost peaks in range profile)
        def corner_internal_reflections(R, base_amplitude, v_base, num_bounces=3):
            """
            Simulate multiple internal reflections in a moving trihedral corner reflector.
            Each internal bounce adds:
            - extra path length  -> shifted range
            - small Doppler offset -> spectral spreading

            Returns a list of (effective_range, amplitude_scale, effective_velocity)
            """
            reflections = [(R, base_amplitude, v_base)]

            for i in range(1, num_bounces + 1):
                # extra path due to internal reflection geometry (meters)
                extra_path = (
                    0.08 * i
                )  # ~8cm per bounce #TODO make this a configurable parmater
                effective_R = R + extra_path

                # amplitude decay with each bounce
                amp_scale = base_amplitude * (
                    0.3**i
                )  # TODO make this a configurable parmater

                # each bounce creates a slight Doppler offset (micro-motion effect)
                v_eff = v_base * (
                    1 + 0.25 * i
                )  # TODO make this a configurable parmater

                reflections.append((effective_R, amp_scale, v_eff))

            return reflections

        # ----------------------------
        # Simulation setup
        # ----------------------------
        t = np.linspace(0, T_chirp, N_r)  # fast time
        chirps = np.arange(N_c) * T_chirp  # slow time

        radar_cube = np.zeros((N_r, N_c, N_rx), dtype=complex64)

        # ----------------------------
        # Corner Reflector Clutter Model with Reflective Patterns (Trihedral Signature)
        # ----------------------------
        # Simulates realistic angular reflective patterns from trihedral corner reflectors
        sensor_clutter_rcs = 1.5  # TODO make this a configurable parmater
        sensor_range_base = 0.01  # TODO make this a configurable parmater

        # for target in targets:
        R, v, angle_deg, RCS = target
        angle_center = np.deg2rad(angle_deg)

        # create internal multipath reflections (ghost ranges + Doppler spreads)
        reflection_paths = corner_internal_reflections(R, RCS, v, num_bounces=2)

        for eff_R, eff_RCS, eff_v in reflection_paths:
            f_b = 2 * B * eff_R / (c * T_chirp)
            f_d = 2 * eff_v * fc / c

            R_eff_clutter = sensor_range_base
            f_b_clutter = 2 * B * R_eff_clutter / (c * T_chirp)

            for n_rx in range(N_rx):
                # for n_c, t_slow in enumerate(chirps):
                for n_c in prange(chirps.shape[0]):
                    t_slow = chirps[n_c]
                    phase_doppler = 2 * np.pi * f_d * t_slow

                    # simulate angular scattering across beam (micro angular spread)
                    theta = angle_center

                    phase_angle = 2 * np.pi * n_rx * d * np.sin(theta) / lambda_radar

                    path_loss = 1.0 / (eff_R**2)
                    # path_loss = 1.0

                    radar_cube[:, n_c, n_rx] += (
                        eff_RCS
                        * path_loss
                        * np.exp(
                            1j * (2 * np.pi * f_b * t + phase_doppler + phase_angle)
                        )
                    )

                    if add_zero_noise:
                        # add zero distance clutter
                        radar_cube[:, n_c, n_rx] += sensor_clutter_rcs * np.exp(
                            1j * (2 * np.pi * f_b_clutter * t + phase_doppler + 0)
                        )
                        radar_cube[:, n_c, n_rx] += sensor_clutter_rcs * np.exp(
                            1j * (2 * np.pi * f_b_clutter * t + -phase_doppler + 0)
                        )
        return radar_cube


class RadarPipeline:
    def __init__(self, num_rx, num_chirps, num_samples):
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.num_samples = num_samples

        Nr = num_samples
        Nc = num_chirps
        Nd = 2 * Nc

        # ----------------------------
        # Windows (float32 for speed)
        # ----------------------------
        self.range_window = (
            signal.windows.blackmanharris(Nr).astype(np.float32).reshape(Nr, 1, 1)
        )

        self.doppler_window = (
            signal.windows.blackmanharris(Nc).astype(np.float32).reshape(1, Nc, 1)
        )

        self.range_in = pyfftw.empty_aligned((2 * Nr, Nc, num_rx), dtype=np.float32)

        # rfft output shape: N//2 + 1 = Nr + 1
        self.range_out = pyfftw.empty_aligned((Nr + 1, Nc, num_rx), dtype=np.complex64)

        # ---------- Doppler FFT (complex -> complex) ----------
        self.doppler_in = pyfftw.empty_aligned(
            (Nr + 1, 2 * Nc, num_rx), dtype=np.complex64
        )

        self.doppler_out = pyfftw.empty_aligned(
            (Nr + 1, 2 * Nc, num_rx), dtype=np.complex64
        )

        # ==========================================================
        # FFTW plans (created ONCE)
        # ==========================================================

        # Range FFT: real -> complex
        self.range_fft_plan = pyfftw.FFTW(
            self.range_in,
            self.range_out,
            axes=(0,),
            direction="FFTW_FORWARD",
            flags=("FFTW_MEASURE",),
            threads=1,
        )

        # Doppler FFT: complex -> complex
        self.doppler_fft_plan = pyfftw.FFTW(
            self.doppler_in,
            self.doppler_out,
            axes=(1,),
            direction="FFTW_FORWARD",
            flags=("FFTW_MEASURE",),
            threads=1,
        )

        # ==========================================================
        # Doppler index remap (no fftshift)
        # ==========================================================
        self.doppler_index = (np.arange(Nd) + Nd // 2) % Nd

    def run(self, radar_cube):
        Nr = self.num_samples
        Nc = self.num_chirps

        # ==========================================================
        # Range FFT (REAL)
        # ==========================================================

        radar_real = radar_cube.real.astype(np.float32, copy=False)

        # Zero padding
        self.range_in[Nr:, :, :] = 0.0

        # Window + copy (no temporaries)
        np.multiply(radar_real, self.range_window, out=self.range_in[:Nr, :, :])

        # Execute FFTW plan (in-place)
        self.range_fft_plan()

        # ==========================================================
        # Doppler FFT (COMPLEX)
        # ==========================================================

        self.doppler_in[:, Nc:, :] = 0.0

        np.multiply(self.range_out, self.doppler_window, out=self.doppler_in[:, :Nc, :])

        self.doppler_fft_plan()

        return np.abs(self.doppler_out[:, self.doppler_index, :])


# class RadarPipeline:
#     def __init__(self, num_rx, num_chirps, num_samples):
#         self.num_rx = num_rx
#         self.num_chirps = num_chirps
#         self.num_samples = num_samples
#
#         self.range_window = signal.windows.blackmanharris(self.num_samples).reshape(
#             self.num_samples, 1, 1
#         )
#         self.doppler_window = signal.windows.blackmanharris(self.num_chirps).reshape(
#             1, self.num_chirps, 1
#         )
#
#         # pyfftw.interfaces.cache.enable()
#         # pyfftw.interfaces.cache.set_keepalive_time(60)
#
#         # Byte-aligned inputs are supposedly faster
#         # radar_cube_shape = (num_samples * 2, num_chirps, num_rx)
#         # range_in_array = pyfftw.empty_aligned(radar_cube_shape, dtype=np.complex128)
#         # range_out_array = pyfftw.empty_aligned(radar_cube_shape, dtype=np.complex128)
#         #
#         # range_cube_shape = (num_samples, num_chirps * 2, num_rx)
#         # doppler_in_array = pyfftw.empty_aligned(range_cube_shape, dtype=np.complex128)
#         # doppler_out_array = pyfftw.empty_aligned(range_cube_shape, dtype=np.complex128)
#         #
#         # self.range_fftw_object = pyfftw.FFTW(
#         #     range_in_array,
#         #     range_out_array,
#         #     axes=(0,),
#         #     direction="FFTW_FORWARD",
#         #     flags=("FFTW_ESTIMATE",),
#         #     threads=1,
#         # )
#         # self.doppler_fftw_object = pyfftw.FFTW(
#         #     doppler_in_array,
#         #     doppler_out_array,
#         #     axes=(1,),
#         #     direction="FFTW_FORWARD",
#         #     flags=("FFTW_ESTIMATE",),
#         #     threads=1,
#         # )
#
#     def run(self, radar_cube):
#         # ----------------------------
#         # Range FFT
#         # ----------------------------
#         radar_cube = np.multiply(radar_cube, self.range_window)
#         radar_cube = np.pad(
#             radar_cube, ((0, self.num_samples), (0, 0), (0, 0)), "constant"
#         )
#
#         range_fft = np.fft.fft(radar_cube, axis=0, norm="forward")  # / N_r
#         # range_fft = self.range_fftw_object(radar_cube)  # / N_r
#         # range_fft = pyfftw.interfaces.numpy_fft.fft(radar_cube, axis=0) / N_r
#
#         range_fft = 2 * range_fft[range(int(self.num_samples)), :, :]
#
#         # ----------------------------
#         # Doppler FFT
#         # ----------------------------
#         range_fft = np.multiply(range_fft, self.doppler_window)
#         range_fft = np.pad(
#             range_fft, ((0, 0), (0, self.num_chirps), (0, 0)), "constant"
#         )
#
#         doppler_fft = np.fft.fft(range_fft, axis=1, norm="forward")  # / N_c
#         # doppler_fft = self.doppler_fftw_object(range_fft)  # / N_c
#         # doppler_fft = pyfftw.interfaces.numpy_fft.fft(range_fft, axis=1) / N_c
#         doppler_fft = np.fft.fftshift(doppler_fft, axes=1)
#
#         return doppler_fft
#


class ObservationType(object):
    def __init__(self, env: "AbstractEnv", **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class GrayscaleObservation(ObservationType):
    """
    An observation class that collects directly what the simulator renders

    Also stacks the collected frames as in the nature DQN.
    Specific keys are expected in the configuration dictionary passed.

    Example of observation dictionary in the environment config:
        observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
            "stack_size": 4,
            "observation_shape": (84, 84)
        }

    Also, the screen_height and screen_width of the environment should match the
    expected observation_shape.
    """

    def __init__(self, env: "AbstractEnv", config: dict) -> None:
        super().__init__(env)
        self.config = config
        self.observation_shape = config["observation_shape"]
        self.shape = self.observation_shape + (config["stack_size"],)
        self.state = np.zeros(self.shape)

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(shape=self.shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        new_obs = self._record_to_grayscale()
        new_obs = np.reshape(new_obs, self.observation_shape)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[:, :, -1] = new_obs
        return self.state

    def _record_to_grayscale(self) -> np.ndarray:
        # TODO: center rendering on the observer vehicle
        raw_rgb = self.env.render("rgb_array")
        return np.dot(raw_rgb[..., :3], self.config["weights"])


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: "AbstractEnv", horizon: int = 10, **kwargs: dict) -> None:
        super().__init__(env)
        self.horizon = horizon

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(
                shape=self.observe().shape, low=0, high=1, dtype=np.float32
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(
                (3, 3, int(self.horizon * self.env.config["policy_frequency"]))
            )
        grid = compute_ttc_grid(
            self.env,
            vehicle=self.observer_vehicle,
            time_quantization=1 / self.env.config["policy_frequency"],
            horizon=self.horizon,
        )
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.observer_vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.observer_vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0 : lf + 1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_speeds = 3
        v0 = grid.shape[0] + self.observer_vehicle.speed_index - obs_speeds // 2
        vf = grid.shape[0] + self.observer_vehicle.speed_index + obs_speeds // 2
        clamped_grid = padded_grid[v0 : vf + 1, :, :]
        return clamped_grid


class RadarObservation(ObservationType):
    def __init__(
        self,
        env: "AbstractEnv",
        normalize: bool = False,
        discretice: bool = True,
        dist_only: bool = False,
        use_radar_simulation: bool = False,
        use_cfar: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)

        self.vel_limit = 0.68
        self.vel_resolution = 0.09

        self.normalize = normalize
        self.discretice = discretice
        self.dist_only = dist_only
        self.use_radar_simulation = use_radar_simulation
        self.use_cfar = use_cfar

        self.vel_bins = np.arange(-self.vel_limit, self.vel_limit, self.vel_resolution)

        if self.use_radar_simulation or self.use_cfar:
            # FMCW waveform
            fc = kwargs["fc"]  # Carrier frequency (Hz)
            B = kwargs["B"]  # Bandwidth (Hz)
            # B = 5.56e9  # Bandwidth (Hz)
            T_chirp = kwargs["T_chirp"]  # Chirp duration (s)
            # T_chirp = 0.000591125  # Chirp duration (s)
            # self.N_r = 128  # Number of range samples per chirp
            self.N_r = kwargs["N_r"]  # Number of range samples per chirp
            self.N_c = kwargs["N_c"]  # Number of chirps (for Doppler)
            self.N_rx = kwargs["N_rx"]  # number of receive antennas (ULA)

            self.add_zero_noise = False

            self.thermal_noise_snr = kwargs["thermal_noise_snr"]

            self.radar_simulator = RadarSimulator(
                fc,
                B,
                T_chirp,
                self.N_r,
                self.N_c,
                self.N_rx,
                self.add_zero_noise,
            )
            self.radar_pipeline = RadarPipeline(
                self.N_rx,
                self.N_c,
                self.N_r,
            )

            self.max_distance = self.radar_simulator.get_max_range()

    def space(self) -> spaces.Space:
        if self.dist_only or self.use_cfar:
            return spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float64)
            # return spaces.Box(shape=(2,), low=-1, high=1, dtype=np.float64)
        elif self.use_radar_simulation:
            return spaces.Box(
                shape=(1, self.N_r, self.N_c * 2),
                low=0,
                high=255,
                dtype=np.uint8,
            )
        else:
            return spaces.Box(shape=(4,), low=-1, high=1, dtype=np.float64)

    def normalize_obs(self, obs):
        if self.use_radar_simulation:
            minmin = np.min(obs)
            maxmax = np.max(obs)

            obs = (obs - minmin) * 255 / (maxmax - minmin)

            obs = obs[np.newaxis, ...]

            return obs.astype(np.uint8)

        # relative x position of leader to ego vehicle
        obs[0] = utils.lmap(obs[0], [0.0, 1.0], [-1, 1])

        # relative y position of leader to ego vehicle
        obs[1] = utils.lmap(obs[1], [-1.0, 1.0], [-1, 1])

        if not self.dist_only and not self.use_cfar:
            # speed of leader towards ego vehicle
            obs[2] = utils.lmap(obs[2], [-1.0, 1.0], [-1, 1])

            # absolute speed of ego vehicle
            obs[3] = utils.lmap(obs[3], [0.2, 1.0], [-1, 1])

        return obs

    def observe(self):
        vehicles = self.env.road.vehicles

        veh = None
        for v in vehicles:
            if v.id == 1:
                veh = v

        if self.dist_only or self.use_radar_simulation or self.use_cfar:
            dist = (
                np.linalg.norm(self.observer_vehicle.position - veh.position)
                - self.observer_vehicle.LENGTH
            )

        else:
            leader = veh.to_dict(self.observer_vehicle)

            # convert to coordinate system (rotation) of ego vehicle
            ego = self.observer_vehicle.to_dict()
            x = leader["x"] * np.cos(-ego["heading"]) - leader["y"] * np.sin(
                -ego["heading"]
            )
            y = leader["x"] * np.sin(-ego["heading"]) + leader["y"] * np.cos(
                -ego["heading"]
            )

            x -= (
                self.observer_vehicle.LENGTH
            )  # account for radar offset from center of car

        # calculate speed along ray
        v = veh.velocity - self.observer_vehicle.velocity
        direction = veh.position - self.observer_vehicle.position

        speed = np.dot(v, direction / np.linalg.norm(direction))

        if self.discretice:
            # discretize speed the to match the actual radar outputs
            ind = np.digitize(speed, self.vel_bins) - 1
            speed = self.vel_bins[ind]

        if self.dist_only:
            # obs = np.array([dist, speed, self.observer_vehicle.speed])
            obs = np.array([dist, speed])
        elif not self.use_radar_simulation and not self.use_cfar:
            obs = np.array([x, y, speed, self.observer_vehicle.speed])

        if self.use_radar_simulation or self.use_cfar:
            # print(f"Real: {speed} {dist}")
            target = [
                dist,
                speed,
                10.0,
                0.9,
            ]  # TODO calculate angle and make RCS a parameter
            radar_cube = self.radar_simulator.simulate_radar(
                target, -self.observer_vehicle.speed
            )
            radar_cube = self.radar_simulator.add_thermal_noise(
                radar_cube, SNR_dB=self.thermal_noise_snr
            )
            doppler_fft = self.radar_pipeline.run(radar_cube)
            # obs = 20 * np.log10(np.abs(np.average(doppler_fft, 2)) + 1e-12)
            # obs = obs[:-1, :]
            obs = np.average(doppler_fft, 2)

            if self.use_cfar:
                detections = ca_cfar_2d(obs, guard=4, train=3, pfa=1e-3)
                detections = detections.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    detections,
                    cv2.RETR_LIST | cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if len(contours) > 0:
                    cnt = max(contours, key=cv2.contourArea)
                    (x, y), _ = cv2.minEnclosingCircle(cnt)
                    w, h = detections.shape
                    det_speed = (x - w / 2) / (w / 2)
                    det_dist = y * (self.max_distance / 2) / h
                    obs = np.array([det_speed, det_dist])
                    # print(f"CFAR {obs}")
                else:
                    # print("No contours found")
                    obs = np.array([0, 0])
            else:
                obs = 20 * np.log10(np.abs(obs) + 1e-12)
                obs = obs[:-1, :]

        if self.normalize:
            obs = self.normalize_obs(obs)

        return obs


class KinematicObservation(ObservationType):
    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ["presence", "x", "y", "vx", "vy"]

    def __init__(
        self,
        env: "AbstractEnv",
        features: List[str] = None,
        vehicles_count: int = 5,
        features_range: Dict[str, List[float]] = None,
        absolute: bool = False,
        order: str = "sorted",
        normalize: bool = True,
        clip: bool = False,
        see_behind: bool = True,
        observe_intentions: bool = False,
        **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

    def space(self) -> spaces.Space:
        # return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-1, high=1, dtype=np.float32)
        return spaces.Box(
            shape=(self.vehicles_count * len(self.features),),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def normalize_obs2(self, data: List[dict]):
        if not self.features_range:
            self.features_range = {
                # "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
                # "y": [-12, 12],
                # "vx": [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
                # "vy": [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
                "vx": [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
                "vy": [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
                "x": [-6.7, 11.0],
                # "y": [0.0, 2.0],
                "y": [-0.43, 0.43],
            }

        for veh in data:
            for feature, _ in veh.items():
                if feature in self.features_range.keys():
                    veh[feature] = utils.lmap(
                        veh[feature],
                        [
                            self.features_range[feature][0],
                            self.features_range[feature][1],
                        ],
                        [-1, 1],
                    )
                    # veh[feature] = np.interp(veh[feature], [self.features_range[feature][0], self.features_range[feature][1]], [-1, 1])

        return data

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            # side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            # self.features_range = {
            #     "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
            #     "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
            #     "vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
            #     "vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX]
            # }
            self.features_range = {
                # "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
                "x": [-190, 310],
                "y": [-12, 12],
                "vx": [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
                "vy": [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                # df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                df[feature] = np.interp(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe_old(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        # sort = self.order == "sorted"
        close_vehicles = self.env.road.close_vehicles_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(
                pd.DataFrame.from_records(
                    [
                        v.to_dict(origin, observe_intentions=self.observe_intentions)
                        for v in close_vehicles[-self.vehicles_count + 1 :]
                    ]
                )[self.features],
                ignore_index=True,
            )
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(
                pd.DataFrame(data=rows, columns=self.features), ignore_index=True
            )
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs.astype(self.space().dtype)

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Collect nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
        )
        obs_list = []

        # Add ego-vehicle
        obs = self.observer_vehicle.to_dict()
        # extract only the features we want
        obs = {k: obs[k] for k in self.features if k in obs}

        # replace x position with relative position to merge end
        obs["x"] = (
            # 2.4 - self.observer_vehicle.position[0]
            2.0 - self.observer_vehicle.position[0]
        )  # adjusted for circular track
        obs_list.append(obs)

        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None

            # close_veh = [
            #     v.to_dict(origin, observe_intentions=self.observe_intentions)
            #     for v in close_vehicles[-self.vehicles_count + 1 :]
            # ]

            close_veh = []
            for v in close_vehicles[-self.vehicles_count + 1 :]:
                v_dict = v.to_dict(origin, observe_intentions=self.observe_intentions)

                lane_dist = -v.lane_distance_to(self.observer_vehicle)
                v_dict["x"] = lane_dist

                # print(f"{v.id} {lane_dist}")

                v_dict["y"] = v.lane.local_coordinates(v.position)[1]

                # if v.lane_index == 1567:
                #     v_dict["y"] += 1.0
                # elif v.lane_index == 1568:
                #     v_dict["y"] += 1.3
                # else:
                #     v_dict["y"] += 0.7
                #
                if v.lane_index == 1234:
                    v_dict["y"] += 0
                elif v.lane_index == 1235:
                    v_dict["y"] += 0.14
                else:
                    v_dict["y"] -= 0.14

                v_dict["y"] -= origin.position[1]

                close_veh.append(v_dict)

            # extract only the features we want
            for idx, veh in enumerate(close_veh):
                close_veh[idx] = {k: veh[k] for k in self.features if k in veh}
                obs_list.append(close_veh[idx])

        # Normalize and clip
        if self.normalize:
            obs_list[1:] = self.normalize_obs2(obs_list[1:])

            # special traetment for ego
            # obs_list[0]["x"] = utils.lmap(obs_list[0]["x"], [0.9, 5], [-1, 1])
            obs_list[0]["x"] = utils.lmap(obs_list[0]["x"], [0.0, 5], [-1, 1])
            # obs_list[0]["y"] = utils.lmap(obs_list[0]["y"], [0.0, 2.0], [-1, 1])
            obs_list[0]["y"] = utils.lmap(obs_list[0]["y"], [-0.43, 0.43], [-1, 1])
            obs_list[0]["vx"] = utils.lmap(
                obs_list[0]["vx"],
                [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
                [-1, 1],
            )
            obs_list[0]["vy"] = utils.lmap(
                obs_list[0]["vy"],
                [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
                [-1, 1],
            )

        # Fill missing rows
        if len(obs_list) < self.vehicles_count:
            empty_row = {k: 0 for k in self.features}
            for i in range(self.vehicles_count - len(obs_list)):
                obs_list.append(empty_row)

        # Convert to 2D Array
        res = [[item.get(key, "") for key in self.features] for item in obs_list]

        # return res
        return np.array(res).flatten()


class OccupancyGridObservation(ObservationType):
    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: List[str] = ["presence", "vx", "vy"]
    GRID_SIZE: List[List[float]] = [[-5.5 * 5, 5.5 * 5], [-5.5 * 5, 5.5 * 5]]
    GRID_STEP: List[int] = [5, 5]

    def __init__(
        self,
        env: "AbstractEnv",
        features: Optional[List[str]] = None,
        grid_size: Optional[List[List[float]]] = None,
        grid_step: Optional[List[int]] = None,
        features_range: Dict[str, List[float]] = None,
        absolute: bool = False,
        **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = (
            np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        )
        self.grid_step = (
            np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        )
        grid_shape = np.asarray(
            np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / grid_step),
            dtype=np.int,
        )
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.grid.shape, low=-1, high=1, dtype=np.float32)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2 * MDPVehicle.SPEED_MAX, 2 * MDPVehicle.SPEED_MAX],
                "vy": [-2 * MDPVehicle.SPEED_MAX, 2 * MDPVehicle.SPEED_MAX],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # Add nearby traffic
            self.grid.fill(0)
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles]
            )
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                for _, vehicle in df.iterrows():
                    x, y = vehicle["x"], vehicle["y"]
                    # Recover unnormalized coordinates for cell index
                    if "x" in self.features_range:
                        x = utils.lmap(
                            x,
                            [-1, 1],
                            [self.features_range["x"][0], self.features_range["x"][1]],
                        )
                    if "y" in self.features_range:
                        y = utils.lmap(
                            y,
                            [-1, 1],
                            [self.features_range["y"][0], self.features_range["y"][1]],
                        )
                    cell = (
                        int((x - self.grid_size[0, 0]) / self.grid_step[0]),
                        int((y - self.grid_size[1, 0]) / self.grid_step[1]),
                    )
                    if (
                        0 <= cell[1] < self.grid.shape[-2]
                        and 0 <= cell[0] < self.grid.shape[-1]
                    ):
                        self.grid[layer, cell[1], cell[0]] = vehicle[feature]
            # Clip
            obs = np.clip(self.grid, -1, 1)
            return obs


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env: "AbstractEnv", scales: List[float], **kwargs: dict) -> None:
        self.scales = np.array(scales)
        super().__init__(env, **kwargs)

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(
                dict(
                    desired_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["desired_goal"].shape,
                        dtype=np.float32,
                    ),
                    achieved_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["achieved_goal"].shape,
                        dtype=np.float32,
                    ),
                    observation=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["observation"].shape,
                        dtype=np.float32,
                    ),
                )
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        if not self.observer_vehicle:
            return {
                "observation": np.zeros((len(self.features),)),
                "achieved_goal": np.zeros((len(self.features),)),
                "desired_goal": np.zeros((len(self.features),)),
            }

        obs = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        )
        goal = np.ravel(
            pd.DataFrame.from_records([self.env.goal.to_dict()])[self.features]
        )
        obs = {
            "observation": obs / self.scales,
            "achieved_goal": obs / self.scales,
            "desired_goal": goal / self.scales,
        }
        return obs


class AttributesObservation(ObservationType):
    def __init__(
        self, env: "AbstractEnv", attributes: List[str], **kwargs: dict
    ) -> None:
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(
                {
                    attribute: spaces.Box(
                        -np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float32
                    )
                    for attribute in self.attributes
                }
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        return {
            attribute: getattr(self.env, attribute) for attribute in self.attributes
        }


class LidarObservation(ObservationType):
    DISTANCE = 0
    SPEED = 1

    def __init__(
        self,
        env,
        cells: int = 64,
        maximum_range: float = 150,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.cells = cells
        self.maximum_range = maximum_range
        self.normalize = normalize
        self.angle = np.array(2 * np.pi / self.cells)
        self.grid = np.ones((self.cells, 1)) * float("inf")
        self.origin = None

        self.directions = [
            np.array([np.cos(index * self.angle), np.sin(index * self.angle)])
            for index in range(self.cells)
        ]

    def space(self) -> spaces.Space:
        high = 1 if self.normalize else self.maximum_range
        # return spaces.Box(shape=(self.cells, 2), low=-high, high=high, dtype=np.float32)
        # return spaces.Box(shape=(self.cells+1, 2), low=-high, high=high, dtype=np.float32)
        return spaces.Dict(
            {
                "lidar": spaces.Box(
                    shape=(self.cells, 2), low=-high, high=high, dtype=np.float32
                ),
                "ego": spaces.Box(shape=(4,), low=-1, high=1),
            }
        )

    def observe(self):
        # obs = self.trace(
        # self.observer_vehicle.position, self.observer_vehicle.velocity
        # ).copy()
        # if self.normalize:
        # obs /= self.maximum_range
        # return obs

        self.grid = utils.trace(
            self.observer_vehicle.position,
            self.observer_vehicle.velocity,
            self.maximum_range,
            self.cells,
            self.angle,
            self.env.road.vehicles + self.env.road.objects,
            self.observer_vehicle,
        )
        self.origin = self.observer_vehicle.position.copy()
        obs = self.grid.copy()
        if self.normalize:
            # normalize distances
            # obs[:, 0] /= self.maximum_range
            obs[:, 0] /= self.maximum_range / 30
            # normalize velocities
            obs[:, 1] = [
                utils.lmap(
                    x,
                    [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
                    [-1, 1],
                )
                for x in obs[:, 1]
            ]

        # add the current position of the ego vehicle to the observation
        ego_obs = self.observer_vehicle.to_dict()
        ego_pos = np.array(
            [ego_obs["x"] - 66, ego_obs["y"], ego_obs["vx"], ego_obs["vy"]]
        )

        ego_pos[0] = utils.lmap(ego_pos[0], [-3, 2], [0, 310])

        # normalize ego_pos same as in KinematicObservation
        # ego_pos[0] = utils.lmap(ego_pos[0], [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX], [-1, 1])
        # ego_pos[0] = utils.lmap(ego_pos[0], [-460, 460], [-1, 1])
        # ego_pos[1] = utils.lmap(ego_pos[1], [-12, 12], [-1, 1])
        # ego_pos[2] = utils.lmap(ego_pos[2], [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX], [-1, 1])
        # ego_pos[3] = utils.lmap(ego_pos[3], [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX], [-1, 1])

        # special traetment for ego
        # replace x position with relative position to merge end
        # ego_pos[0] = (2.0 - self.observer_vehicle.position[0])  # adjusted for circular track
        # obs_list[0]["x"] = utils.lmap(obs_list[0]["x"], [0.9, 5], [-1, 1])
        # ego_pos[0] = utils.lmap(ego_pos[0], [0.0, 5], [-1, 1])
        ego_pos[0] = utils.lmap(ego_pos[0], [-460, 460], [-1, 1])
        # obs_list[0]["y"] = utils.lmap(obs_list[0]["y"], [0.0, 2.0], [-1, 1])
        ego_pos[1] = utils.lmap(ego_pos[1], [-0.43, 0.43], [-1, 1])
        ego_pos[2] = utils.lmap(
            ego_pos[2],
            [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
            [-1, 1],
        )
        ego_pos[3] = utils.lmap(
            ego_pos[3],
            [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
            [-1, 1],
        )

        # obs = np.vstack([obs, ego_pos])
        obs = {"lidar": obs, "ego": ego_pos}

        return obs

    def trace(self, origin: np.ndarray, origin_velocity: np.ndarray) -> np.ndarray:
        self.origin = origin.copy()
        self.grid = np.ones((self.cells, 2)) * self.maximum_range

        for obstacle in self.env.road.vehicles + self.env.road.objects:
            if obstacle is self.observer_vehicle:  # or not obstacle.solid:
                continue
            center_distance = np.linalg.norm(obstacle.position - origin)
            if center_distance > self.maximum_range:
                continue
            center_angle = self.position_to_angle(obstacle.position, origin)
            center_index = self.angle_to_index(center_angle)
            distance = center_distance - obstacle.WIDTH / 2
            if distance <= self.grid[center_index, self.DISTANCE]:
                direction = self.index_to_direction(center_index)
                velocity = (obstacle.velocity - origin_velocity).dot(direction)
                self.grid[center_index, :] = [distance, velocity]

            # Angular sector covered by the obstacle
            corners = utils.rect_corners(
                obstacle.position, obstacle.LENGTH, obstacle.WIDTH, obstacle.heading
            )
            angles = [self.position_to_angle(corner, origin) for corner in corners]
            min_angle, max_angle = min(angles), max(angles)
            if (
                min_angle < -np.pi / 2 < np.pi / 2 < max_angle
            ):  # Object's corners are wrapping around +pi
                min_angle, max_angle = max_angle, min_angle + 2 * np.pi
            start, end = self.angle_to_index(min_angle), self.angle_to_index(max_angle)
            if start < end:
                indexes = np.arange(start, end + 1)
            else:  # Object's corners are wrapping around 0
                indexes = np.hstack(
                    [np.arange(start, self.cells), np.arange(0, end + 1)]
                )

            # Actual distance computation for these sections
            for index in indexes:
                direction = self.index_to_direction(index)
                ray = (origin, origin + self.maximum_range * direction)
                # distance = utils.distance_to_rect(*ray, corners)
                distance = utils.distance_to_rect2(*ray, corners, self.maximum_range)
                # distance2 = utils.distance_to_rect(*ray, corners)
                # assert abs(distance - distance2) < 0.0001 or distance == distance2
                if distance <= self.grid[index, self.DISTANCE]:
                    velocity = (obstacle.velocity - origin_velocity).dot(direction)
                    self.grid[index, :] = [distance, velocity]
        return self.grid

    def position_to_angle(self, position: np.ndarray, origin: np.ndarray) -> float:
        return (
            np.arctan2(position[1] - origin[1], position[0] - origin[0])
            + self.angle / 2
        )

    def position_to_index(self, position: np.ndarray, origin: np.ndarray) -> int:
        return self.angle_to_index(self.position_to_angle(position, origin))

    def angle_to_index(self, angle: float) -> int:
        return int(np.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index: np.ndarray) -> np.ndarray:
        # return np.array([np.cos(index * self.angle), np.sin(index * self.angle)])
        return self.directions[index]


class MultiAgentObservation(ObservationType):
    def __init__(self, env: "AbstractEnv", observation_config: dict, **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple(
            [obs_type.space() for obs_type in self.agents_observation_types]
        )

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)


def observation_factory(env: "AbstractEnv", config: dict) -> ObservationType:
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif config["type"] == "LidarObservation":
        return LidarObservation(env, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    elif config["type"] == "Radar":
        return RadarObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
