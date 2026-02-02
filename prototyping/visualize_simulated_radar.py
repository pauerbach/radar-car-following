import sys
import numpy as np

sys.path.append("./highway-env/")

from highway_env.envs.common.observation import Draw, RadarPipeline, RadarSimulator

fc = 60.75e9
B = 5.36e9
T_chirp = 0.00123371
N_r = 64
N_c = 32
N_rx = 3
thermal_noise_snr = 6

radar_simulator = RadarSimulator(
    fc,
    B,
    T_chirp,
    N_r,
    N_c,
    N_rx,
    False,
)
radar_pipeline = RadarPipeline(
    N_rx,
    N_c,
    N_r,
)

max_range = radar_simulator.get_max_range()
max_doppler = radar_simulator.get_max_velocity()
draw = Draw(max_doppler, max_range)

target = [
    0.35,  # dist
    0.0,  # speed
    10.0,
    0.9,
]

ego_speed = 0.5

radar_cube = radar_simulator.simulate_radar(target, -ego_speed)
radar_cube = radar_simulator.add_thermal_noise(radar_cube, SNR_dB=thermal_noise_snr)
doppler_fft = radar_pipeline.run(radar_cube)
obs = 20 * np.log10(np.abs(np.average(doppler_fft, 2)) + 1e-12)

minmin = np.min(obs)
maxmax = np.max(obs)

obs = (obs - minmin) * 255 / (maxmax - minmin)

draw.draw(obs)
draw.save("test_out.png")

# while True:
#     draw.draw(obs)
