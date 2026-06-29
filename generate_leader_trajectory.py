import numpy as np
import matplotlib.pyplot as plt

# Fixed control points for the leader speeds
# fixed_points = np.array(
#     [
#         [0, 0.6],
#         [10, 0.5],
#         [18, 0.6],
#         [21, 0.6],
#         [30, 0.0],
#         [40, 0.4],
#         [50, 0.4],
#         [53, 0.0],
#         [60, 0.0],
#         [65, 0.6],
#         [80, 0.6],
#         [82, 0.0],
#         [90, 0.0],
#     ]
# )

## New set of points with no stopping
fixed_points = np.array(
    [
        [0, 0.7],
        [10, 0.6],
        [18, 0.7],
        [21, 0.7],
        [30, 0.2],
        [40, 0.5],
        [50, 0.5],
        [53, 0.2],
        [60, 0.2],
        [65, 0.7],
        [80, 0.7],
        [82, 0.2],
        [90, 0.2],
    ]
)
# 1. Extract time (t) and speed (v) arrays
t_fixed = fixed_points[:, 0]
v_fixed = fixed_points[:, 1]

# 2. Create a time array with a discrete spacing of 0.1 seconds
# We add a tiny buffer (0.01) to the max time to ensure the final point (90.0) is included
t_interp = np.arange(t_fixed.min(), t_fixed.max() + 0.01, 0.1)

# 3. Perform linear interpolation using numpy
v_interp = np.interp(t_interp, t_fixed, v_fixed)

# 4. Stack time and velocity together and save to a .npy file
np.save("leader_trajectory_simple_no_stopping.npy", v_interp)

# 5. Plot the results to verify
plt.figure(figsize=(10, 5))

# Plot the interpolated line
plt.plot(t_interp, v_interp, label="Interpolated Trajectory (0.1s steps)", color="blue")

# Plot the original control points as red dots
plt.plot(t_fixed, v_fixed, "ro", label="Control Points")

# Add formatting and labels
plt.title("Leader Speed Trajectory over Time")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.grid(True)
plt.legend()

# Display the plot
plt.show()
