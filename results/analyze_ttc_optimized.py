import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration & Style ---
sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2)

TTC_THRESHOLD = 1.5
HEADWAY_THRESHOLD = 0.9
DT = 0.1
NUMBER_OF_RUNS = 50000

# Define your datasets and their corresponding file paths
datasets = {
    "IDM": {
        # "ttc": "./ttc_idm_without_action_noise.npy",
        # "headway": "./headway_idm_without_action_noise.npy",
        # "ttc": "./ttc_idm_with_action_noise.npy",
        # "headway": "./headway_idm_with_action_noise.npy",
        "ttc": "./ttc_idm_with_action_noise_08.npy",
        "headway": "./headway_idm_with_action_noise_08.npy",
        # "ttc": "./ttc_idm_with_action_noise_08_banana.npy",
        # "headway": "./headway_idm_with_action_noise_08_banana.npy",
    },
    "Direct": {
        # "ttc": "./ttc_direct_no_action_noise.npy",
        # "headway": "./headway_direct_no_action_noise.npy",
        "ttc": "./ttc_direct_with_action_noise.npy",
        "headway": "./headway_direct_with_action_noise.npy",
        # "ttc": "./ttc_direct_with_action_noise_banana.npy",
        # "headway": "./headway_direct_with_action_noise_banana.npy",
    },
    "CFAR": {
        # "ttc": "./ttc_cfar_direct_agent_no_action_noise.npy",
        # "headway": "./headway_cfar_direct_agent_no_action_noise.npy",
        "ttc": "./ttc_cfar_direct_agent_with_action_noise.npy",
        "headway": "./headway_cfar_direct_agent_with_action_noise.npy",
        # "ttc": "./ttc_cfar_direct_agent_with_action_noise_banana.npy",
        # "headway": "./headway_cfar_direct_agent_with_action_noise_banana.npy",
    },
    "End-to-End": {
        # "ttc": "./ttc_radar_no_action_noise.npy",
        # "headway": "./headway_radar_no_action_noise.npy",
        "ttc": "./ttc_radar_new.npy",
        "headway": "./headway_radar_new.npy",
        # "ttc": "./ttc_radar_min0_speed.npy",
        # "headway": "./headway_radar_min0_speed.npy",
        # "ttc": "./ttc_radar_with_action_noise_banana.npy",
        # "headway": "./headway_radar_with_action_noise_banana.npy",
        # "ttc": "./ttc_radar_with_action_noise_banana_smooth.npy",
        # "headway": "./headway_radar_with_action_noise_banana_smooth.npy",
        # "ttc": "./ttc_radar_with_action_noise_rectangle.npy",
        # "headway": "./headway_radar_with_action_noise_rectangle.npy",
        # "ttc": "./ttc_attention.npy",
        # "headway": "./headway_attention.npy",
        # "ttc": "./ttc_attention_circular.npy",
        # "headway": "./headway_attention_circular.npy",
    },
}

# --- Plot Setup ---
fig, axs = plt.subplots(2, figsize=(20, 10))

# Create the inset zoom for the TTC plot (axs[0])
x1, x2, y1, y2 = 0, 1.5, 0.0, 0.025
axins = axs[0].inset_axes([0.25, 0.07, 0.27, 0.43], xlim=(x1, x2), ylim=(y1, y2))
axins.set(xlabel="", ylabel=" ")

# --- Data Processing & Plotting Loop ---
for name, paths in datasets.items():
    # 1. Load Data
    ttc = np.load(paths["ttc"])
    headway = np.load(paths["headway"])

    # 2. Statistical Output
    print(f"--- {name} ---")
    # Safety margin fraction (calculated before filtering)
    print(
        f"TTC Safety Threshold Fraction: {ttc[ttc < TTC_THRESHOLD].shape[0] / NUMBER_OF_RUNS:.4f}"
    )
    print(
        f"Headway Safety Threshold Fraction: {headway[headway < HEADWAY_THRESHOLD].shape[0] / NUMBER_OF_RUNS:.4f}"
    )

    is_below = ttc < TTC_THRESHOLD
    # Pad with False to safely catch sequences starting at index 0 or ending at the last index
    padded = np.concatenate(([False], is_below, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    max_consecutive_steps = np.max(ends - starts) if len(starts) > 0 else 0
    max_consecutive_time = max_consecutive_steps * DT
    print(f"Max Consecutive Time Below TTC Threshold: {max_consecutive_time:.2f}s")

    is_below_hw = headway < HEADWAY_THRESHOLD
    padded_hw = np.concatenate(([False], is_below_hw, [False]))
    diffs_hw = np.diff(padded_hw.astype(int))
    starts_hw = np.where(diffs_hw == 1)[0]
    ends_hw = np.where(diffs_hw == -1)[0]

    max_consecutive_steps_hw = np.max(ends_hw - starts_hw) if len(starts_hw) > 0 else 0
    max_consecutive_time_hw = max_consecutive_steps_hw * DT
    print(
        f"Max Consecutive Time Below Headway Threshold: {max_consecutive_time_hw:.2f}s"
    )

    # 3. Filtering
    # Apply filtering for TTC statistics and plotting
    ttc_filtered = ttc[(ttc < 10) & (ttc > 0.3)]
    # ttc_filtered = ttc[(ttc < 10) & (ttc > 0.0)]

    print(f"Mean TTC: {np.mean(ttc_filtered):.2f}")
    print(f"Min TTC: {np.min(ttc_filtered):.2f}")
    print(f"1st percentile TTC: {np.percentile(ttc_filtered, 1):.2f}")
    print(f"Mean Headway: {np.mean(headway):.2f}")
    print(f"Min Headway: {np.min(headway):.2f}")
    print(f"1st percentile Headway: {np.percentile(headway, 1):.2f}\n")

    # 4. Adding to Plots
    # TTC KDEs
    sns.kdeplot(ttc_filtered, ax=axs[0], label=name, linewidth=4, bw_adjust=0.2)
    # TTC Inset
    sns.kdeplot(ttc_filtered, ax=axins, linewidth=4, bw_adjust=0.2)
    # Headway KDEs
    sns.kdeplot(headway, ax=axs[1], label=name, linewidth=4, bw_adjust=0.2)

# --- Final Plot Polishing ---
# TTC Axis (Top)
axs[0].indicate_inset_zoom(axins, edgecolor="black")
axs[0].set(xlabel="TTC [s]", ylabel="Density")
axs[0].yaxis.labelpad = 20
axs[0].set_ylim(bottom=0)
axs[0].legend()

# Headway Axis (Bottom)
axs[1].set(xlabel="Headway [s]", ylabel="Density")
axs[1].yaxis.labelpad = 20
axs[1].set_xlim(0.25, 2.5)
axs[1].set_ylim(bottom=0)
axs[1].legend()

plt.subplots_adjust(hspace=0.3)
plt.show()
# plt.savefig( "ttc_headway_comparison_kde_inset_with_action_noise_08_banana.png", dpi=300, bbox_inches="tight",)
print("Plot saved successfully.")
