import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2)

TTC_THRESHOLD = 1.5
number_of_runs = 50000
# (means/medians, variances, percentile safety margins, time-below-threshold fractions,

ttc_idm = np.load("./ttc_idm_with_action_noise.npy")
headway_idm = np.load("./headway_idm_with_action_noise.npy")
print("IDM")
print(ttc_idm[ttc_idm < TTC_THRESHOLD].shape[0] / number_of_runs)
ttc_idm = ttc_idm[(ttc_idm < 10) & (ttc_idm > 0)]
print(np.mean(ttc_idm))
print(np.min(ttc_idm))
print(np.mean(headway_idm))
print(np.min(headway_idm))
print()

ttc_direct = np.load("./ttc_direct_distance_vel.npy")
headway_direct = np.load("./headway_direct_distance_vel.npy")
print("Direct")
print(ttc_direct[ttc_direct < TTC_THRESHOLD].shape[0] / number_of_runs)
ttc_direct = ttc_direct[(ttc_direct < 10) & (ttc_direct > 0)]
print(np.mean(ttc_direct))
print(np.min(ttc_direct))
print(np.mean(headway_direct))
print(np.min(headway_direct))
print()

# ttc_cfar = np.load("./ttc_cfar.npy")
# headway_cfar = np.load("./headway_cfar.npy")
ttc_cfar = np.load("./ttc_cfar_new.npy")
headway_cfar = np.load("./headway_cfar_new.npy")
# ttc_cfar = np.load("./ttc_cfar_kalman.npy")
# headway_cfar = np.load("./headway_cfar_kalman.npy")
print("CFAR")
print(ttc_cfar[ttc_cfar < TTC_THRESHOLD].shape[0] / number_of_runs)
ttc_cfar = ttc_cfar[(ttc_cfar < 10) & (ttc_cfar > 0)]
print(np.mean(ttc_cfar))
print(np.min(ttc_cfar))
print(np.mean(headway_cfar))
print(np.min(headway_cfar))
print()

# ttc = np.load("./ttc_radar.npy")
# headway = np.load("./headway_radar.npy")
ttc = np.load("./ttc_radar_new.npy")
headway = np.load("./headway_radar_new.npy")
print("End-to-end")
print(ttc[ttc < TTC_THRESHOLD].shape[0] / number_of_runs)
ttc = ttc[(ttc < 10) & (ttc > 0)]
print(np.mean(ttc))
print(np.min(ttc))
print(np.mean(headway))
print(np.min(headway))
print()

# --- Filtering (as in original script) ---
# headway_idm = headway_idm[headway_idm < 5]

# headway_direct = headway_direct[headway_direct < 5]

# headway = headway[headway < 5]

# ttc_cfar = ttc_cfar[ttc_cfar > 0]  # only use ttcs below 10s as in paper
# headway_cfar = headway_cfar[headway_cfar < 5]  # only use headways below 5 as in paper

# --- Plotting ---
fig, axs = plt.subplots(2, figsize=(20, 10))

# TTC KDEs
sns.kdeplot(ttc_idm, ax=axs[0], label="IDM", linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc_direct, ax=axs[0], label="Direct", linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc_cfar, ax=axs[0], label="CFAR", linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc, ax=axs[0], label="End-to-End", linewidth=4, bw_adjust=0.2)

## Inset zoom of tailend
x1, x2, y1, y2 = 0, 1.5, 0.0, 0.025  # subregion of the original image
axins = axs[0].inset_axes(
    [0.25, 0.07, 0.27, 0.43],
    xlim=(x1, x2),
    ylim=(y1, y2),
    # xticklabels=[],
    # yticklabels=[],
)
axins.set(xlabel="", ylabel=" ")
sns.kdeplot(ttc_idm, ax=axins, linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc_direct, ax=axins, linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc_cfar, ax=axins, linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc, ax=axins, linewidth=4, bw_adjust=0.2)

axs[0].indicate_inset_zoom(axins, edgecolor="black")
## End inset

axs[0].set(xlabel="TTC [s]", ylabel="Density")
axs[0].yaxis.labelpad = 20
# axs[0].set_xlim(0, 10)
axs[0].set_ylim(bottom=0)
axs[0].legend()

# Headway KDEs
sns.kdeplot(headway_idm, ax=axs[1], label="IDM", linewidth=4, bw_adjust=0.2)
sns.kdeplot(headway_direct, ax=axs[1], label="Direct", linewidth=4, bw_adjust=0.2)
sns.kdeplot(headway_cfar, ax=axs[1], label="CFAR", linewidth=4, bw_adjust=0.2)
sns.kdeplot(headway, ax=axs[1], label="End-to-End", linewidth=4, bw_adjust=0.2)

axs[1].set(xlabel="Headway [s]", ylabel="Density")
axs[1].yaxis.labelpad = 20
axs[1].set_xlim(0.25, 2.5)
axs[1].set_ylim(bottom=0)
axs[1].legend()

plt.subplots_adjust(hspace=0.3)
plt.show()
# plt.savefig("ttc_headway_comparison_kde_inset.png", dpi=300, bbox_inches="tight")
