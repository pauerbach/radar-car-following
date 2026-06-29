import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2)

# ttc_idm = np.load("./ttc_idm_with_action_noise.npy")
# headway_idm = np.load("./headway_idm_with_action_noise.npy")
# ttc_idm = np.load("./ttc_circular_oncoming_2.npy")
# headway_idm = np.load("./headway_circular_oncoming_2.npy")
ttc_idm = np.load("./ttc_circular_same_direction.npy")
headway_idm = np.load("./headway_circular_same_direction.npy")

print(headway_idm[headway_idm < -0.4].shape)

# ttc_direct = np.load("./ttc_direct_distance_vel.npy")
# headway_direct = np.load("./headway_direct_distance_vel.npy")
ttc_direct = np.load("./ttc_radar.npy")
headway_direct = np.load("./headway_radar.npy")

ttc_cfar = np.load("./ttc_cfar.npy")
headway_cfar = np.load("./headway_cfar.npy")
# ttc_cfar = np.load("./ttc_bananaangleshorter_oncoming.npy")
# headway_cfar = np.load("./headway_bananaangleshorter_oncoming.npy")
print(headway_cfar.shape)

# ttc = np.load("./ttc_radar.npy")
# headway = np.load("./headway_radar.npy")
# ttc = np.load("./ttc_radar_longer_termination_headway_2.npy")
# headway = np.load("./headway_radar_longer_termination_headway_2.npy")
# ttc = np.load("./ttc_bananaangleshorter.npy")
# headway = np.load("./headway_bananaangleshorter.npy")
ttc = np.load("./ttc_cfar_kalman.npy")
headway = np.load("./headway_cfar_kalman.npy")
# ttc = np.load("./ttc_radar_new_pixi.npy")
# headway = np.load("./headway_radar_new_pixi.npy")
print(headway.shape)

# --- Filtering (as in original script) ---
ttc_idm = ttc_idm[(ttc_idm < 10) & (ttc_idm > 0)]
# headway_idm = headway_idm[headway_idm < 5]

ttc_direct = ttc_direct[(ttc_direct < 10) & (ttc_direct > 0)]
# headway_direct = headway_direct[headway_direct < 5]

ttc = ttc[(ttc < 10) & (ttc > 0)]
# headway = headway[headway < 5]

ttc_cfar = ttc_cfar[ttc_cfar < 10]  # only use ttcs below 10s as in paper
ttc_cfar = ttc_cfar[ttc_cfar > 0]  # only use ttcs below 10s as in paper
# headway_cfar = headway_cfar[headway_cfar < 5]  # only use headways below 5 as in paper

# --- Plotting ---
fig, axs = plt.subplots(2, figsize=(20, 10))

# TTC KDEs
sns.kdeplot(ttc_idm, ax=axs[0], label="circular oncoming", linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc_direct, ax=axs[0], label="Circular", linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc_cfar, ax=axs[0], label="CFAR", linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc, ax=axs[0], label="CFAR KF", linewidth=4, bw_adjust=0.2)

axs[0].set(xlabel="TTC [s]", ylabel="Density")
axs[0].yaxis.labelpad = 20
axs[0].set_xlim(0, 10)
axs[0].set_ylim(bottom=0)
axs[0].legend()

# Headway KDEs
sns.kdeplot(
    headway_idm, ax=axs[1], label="circular oncoming", linewidth=4, bw_adjust=0.2
)
sns.kdeplot(headway_direct, ax=axs[1], label="Circular", linewidth=4, bw_adjust=0.2)
sns.kdeplot(headway_cfar, ax=axs[1], label="CFAR", linewidth=4, bw_adjust=0.2)
sns.kdeplot(headway, ax=axs[1], label="CFAR KF", linewidth=4, bw_adjust=0.2)

axs[1].set(xlabel="Headway [s]", ylabel="Density")
axs[1].yaxis.labelpad = 20
# axs[1].set_xlim(0, 5)
axs[1].set_ylim(bottom=0)
axs[1].legend()

plt.subplots_adjust(hspace=0.3)
plt.show()
# plt.savefig("ttc_headway_comparison_kde.png", dpi=300, bbox_inches="tight")
