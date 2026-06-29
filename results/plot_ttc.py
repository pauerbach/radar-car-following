import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2)

# ttc_idm = np.load("./ttc_idm_with_action_noise.npy")
# headway_idm = np.load("./headway_idm_with_action_noise.npy")
ttc_idm = np.load("./ttc_circular_oncoming.npy")
headway_idm = np.load("./headway_circular_oncoming.npy")

# ttc_direct = np.load("./ttc_direct_distance_vel.npy")
# headway_direct = np.load("./headway_direct_distance_vel.npy")
ttc_direct = np.load("./ttc_radar.npy")
headway_direct = np.load("./headway_radar.npy")

# ttc_cfar = np.load("./ttc_cfar.npy")
# headway_cfar = np.load("./headway_cfar.npy")
ttc_cfar = np.load("./ttc_bananaangleshorter_oncoming.npy")
headway_cfar = np.load("./headway_bananaangleshorter_oncoming.npy")

# ttc = np.load("./ttc_radar.npy")
# headway = np.load("./headway_radar.npy")
# ttc = np.load("./ttc_radar_longer_termination_headway_2.npy")
# headway = np.load("./headway_radar_longer_termination_headway_2.npy")
# ttc = np.load("./ttc_oncoming.npy")
# headway = np.load("./headway_oncoming.npy")
# ttc = np.load("./ttc_rectangle.npy")
# headway = np.load("./headway_rectangle.npy")
ttc = np.load("./ttc_bananaangleshorter.npy")
headway = np.load("./headway_bananaangleshorter.npy")
# ttc = np.load("./ttc_radar_new_pixi.npy")
# headway = np.load("./headway_radar_new_pixi.npy")

ttc_idm = ttc_idm[ttc_idm < 10]  # only use ttcs below 10s as in paper
ttc_idm = ttc_idm[ttc_idm > 0]  # only use ttcs below 10s as in paper
headway_idm = headway_idm[headway_idm < 5]  # only use headways below 5 as in paper

ttc_direct = ttc_direct[ttc_direct < 10]  # only use ttcs below 10s as in paper
ttc_direct = ttc_direct[ttc_direct > 0]  # only use ttcs below 10s as in paper
headway_direct = headway_direct[
    headway_direct < 5
]  # only use headways below 5 as in paper

ttc_cfar = ttc_cfar[ttc_cfar < 10]  # only use ttcs below 10s as in paper
ttc_cfar = ttc_cfar[ttc_cfar > 0]  # only use ttcs below 10s as in paper
headway_cfar = headway_cfar[headway_cfar < 5]  # only use headways below 5 as in paper
headway_cfar = headway_cfar[headway_cfar > 0]  # only use headways below 5 as in paper

ttc = ttc[ttc < 10]  # only use ttcs below 10s as in paper
ttc = ttc[ttc > 0]  # only use ttcs below 10s as in paper
headway = headway[headway < 5]  # only use headways below 5 as in paper

# Plot the result
fig, axs = plt.subplots(2, figsize=(20, 10))

# axs[0].plot(ttc)
axs[0].hist(ttc_idm, bins=100, density=True, label="Circular oncoming")
axs[0].hist(ttc_direct, bins=100, density=True, label="Circular", alpha=0.7)
axs[0].hist(ttc_cfar, bins=100, density=True, label="Banana oncoming", alpha=0.7)
axs[0].hist(ttc, bins=100, density=True, label="Banana", alpha=0.7)
axs[0].set(xlabel="TTC [s]", ylabel="Density")
axs[0].yaxis.labelpad = 20
# axs[0].set_ylim(0, 10)
axs[0].legend()

# axs[1].plot(headway)
axs[1].hist(headway_idm, bins=100, density=True, label="Circular oncoming")
axs[1].hist(headway_direct, bins=100, density=True, label="Circular", alpha=0.7)
axs[1].hist(headway_cfar, bins=100, density=True, label="Banana oncoming", alpha=0.7)
axs[1].hist(headway, bins=100, density=True, label="Banana", alpha=0.7)
axs[1].set(xlabel="Headway [s]", ylabel="Density")
axs[1].yaxis.labelpad = 20
# axs[1].set_ylim(0, 5)
axs[1].legend()

plt.subplots_adjust(hspace=0.3)
plt.show()
# plt.savefig("ttc_headway_comparison_with_cfar.png", dpi=300, bbox_inches="tight")
