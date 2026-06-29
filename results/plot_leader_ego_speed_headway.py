import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional theme and context scales
sns.set_theme()
sns.set_context("paper", font_scale=1.7)

# Parameters
dt = 0.1  # Time delta between samples (0.1 seconds)
# plot_until = 50 * 10  # 500 samples total (50 seconds)
plot_until = 90 * 10  # 500 samples total (50 seconds)

# --- 1. LOAD DATA FOR FOLLOWER VEHICLE A (end-to-end)---
# ego_speed_A = np.load("./ego_speeds_for_trajectory_plot.npy")[:plot_until]
# headways_A = np.load("./headway_radar_for_trajectory_plot.npy")[:plot_until]
# ttcs_A = np.load("./ttc_radar_for_trajectory_plot.npy")[:plot_until]
# ego_speed_A = np.load("ego_speeds_radar_for_trajectory_plot_seed42_simple.npy")[ :plot_until ]
# headways_A = np.load("headway_radar_for_trajectory_plot_seed42_simple.npy")[:plot_until]
# ttcs_A = np.load("ttc_radar_for_trajectory_plot_seed42_simple.npy")[:plot_until]
ego_speed_A = np.load("ego_speeds_radar_min0_speed_no_stopping.npy")[:plot_until]
headways_A = np.load("headway_radar_min0_speed_no_stopping.npy")[:plot_until]
ttcs_A = np.load("ttc_radar_min0_speed_no_stopping.npy")[:plot_until]
# ego_speed_A = np.load("ego_speeds_direct_min0_speed_no_stopping.npy")[:plot_until]
# headways_A = np.load("headway_direct_min0_speed_no_stopping.npy")[:plot_until]
# ttcs_A = np.load("ttc_direct_min0_speed_no_stopping.npy")[:plot_until]
# ego_speed_A = np.load("ego_speeds_cfar_min0_speed_no_stopping.npy")[:plot_until]
# headways_A = np.load("headway_cfar_min0_speed_no_stopping.npy")[:plot_until]
# ttcs_A = np.load("ttc_cfar_min0_speed_no_stopping.npy")[:plot_until]

# --- 2. LOAD DATA FOR FOLLOWER VEHICLE B (IDM)---
# ego_speed_B = np.load("./ego_speeds_idm_for_trajectory_plot.npy")[:plot_until]
# headways_B = np.load("./headway_idm_for_trajectory_plot.npy")[:plot_until]
# ttcs_B = np.load("./ttc_idm_for_trajectory_plot.npy")[:plot_until]
# ego_speed_B = np.load("ego_speeds_idm_for_trajectory_plot_seed42_simple.npy")[ :plot_until ]
# headways_B = np.load("headway_idm_for_trajectory_plot_seed42_simple.npy")[:plot_until]
# ttcs_B = np.load("ttc_idm_for_trajectory_plot_seed42_simple.npy")[:plot_until]
ego_speed_B = np.load("ego_speeds_idm_min0_speed_no_stopping.npy")[:plot_until]
headways_B = np.load("headway_idm_min0_speed_no_stopping.npy")[:plot_until]
ttcs_B = np.load("ttc_idm_min0_speed_no_stopping.npy")[:plot_until]

# --- 3. LOAD DATA FOR FOLLOWER VEHICLE C (Direct Agent) ---
ego_speed_C = np.load("ego_speeds_direct_min0_speed_no_stopping.npy")[:plot_until]
headways_C = np.load("headway_direct_min0_speed_no_stopping.npy")[:plot_until]
ttcs_C = np.load("ttc_direct_min0_speed_no_stopping.npy")[:plot_until]

# Load the common leader speed baseline
# leader_speed = np.load("./leader_speeds_for_trajectory_plot.npy")[:plot_until]
# leader_speed = np.load("../leader_trajectory_simple.npy")[:plot_until]
# leader_speed = np.load("leader_speeds_radar_for_trajectory_plot_seed42_simple.npy")[ :plot_until ]
leader_speed = np.load("../leader_trajectory_simple_no_stopping.npy")[:plot_until]

actions = np.load("actions.npy")[:plot_until]


# Create accurate time array based on 0.1s resolution
time = np.arange(len(leader_speed)) * dt

# Create figure with 3 subplots sharing the same X-axis
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# ==========================================
# SUBPLOT 1: Vehicle Velocities Comparison
# ==========================================
axs[0].plot(time, ego_speed_B, label="IDM", linewidth=2, color="#ff7f0e", linestyle="--")
axs[0].plot(time, ego_speed_A, label="End-to-End", linewidth=2, color="#1f77b4")
axs[0].plot(time, ego_speed_C, label="Direct", linewidth=2, color="#48af0d")

axs[0].plot( time, leader_speed, label="Leader", linewidth=2, color="gray", linestyle=":")

axs[0].set_ylabel("Velocity [m/s]")
# axs[0].set_title("Vehicle Velocities", fontsize=14, fontweight="bold")
axs[0].legend(loc="lower left", frameon=True, bbox_to_anchor=(0.07, 0.1), fontsize=14)
axs[0].grid(True, linewidth=1.5)

# ==========================================
# SUBPLOT 2: Time Headway Comparison
# ==========================================
axs[1].plot(time, np.maximum(np.minimum(headways_B, 4.0), 0.0), label="IDM", linewidth=2, color="#ff7f0e", linestyle="--")
axs[1].plot(time, np.maximum(np.minimum(headways_A, 4.0), 0.0), label="End-to-End", linewidth=2, color="#1f77b4")
axs[1].plot(time, np.maximum(np.minimum(headways_C, 4.0), 0.0), label="Direct RL", linewidth=2, color="#48af0d")

# Safety threshold line
axs[1].axhline(y=0.9, color="#006400", linestyle="--", linewidth=1.5, label="Safe Threshold")
axs[1].set_ylabel("Headway [s]")
# axs[1].set_title("Time Headway", fontsize=14, fontweight="bold")
axs[1].legend(loc="upper left", bbox_to_anchor=(0.07, 0.97), frameon=True, fontsize=14)
axs[1].grid(True, linewidth=1.5)

# ==========================================
# SUBPLOT 3: Time-to-Collision (TTC) Comparison
# ==========================================
axs[2].plot(time, np.minimum(ttcs_B, 10), label="IDM", linewidth=2, color="#ff7f0e", linestyle="--")
axs[2].plot(time, np.minimum(ttcs_A, 10), label="End-to-End", linewidth=2, color="#1f77b4")
axs[2].plot(time, np.minimum(ttcs_C, 10), label="Direct", linewidth=2, color="#48af0d")
# Safety threshold line
axs[2].axhline(y=1.5, color="#006400", linestyle="--", linewidth=2, label="Safe Threshold")
axs[2].set_xlabel("Time [s]")
axs[2].set_ylabel("TTC [s]")
# axs[2].set_title("Time-to-Collision (TTC)", fontsize=14, fontweight="bold")
axs[2].legend(loc="lower left", bbox_to_anchor=(0.07, 0.1), frameon=True, fontsize=14)
axs[2].grid(True, linewidth=1.5)

# Optimize spacing and layout adjustments
plt.tight_layout()

# Save and display
plt.savefig("comparison_fixed_trajecotry_no_stopping_end-to-end_and_direct.png", dpi=300)
# plt.show()
