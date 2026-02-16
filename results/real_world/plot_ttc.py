import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2)

# ttc_idm = np.load("./results/ttc_idm_with_action_noise.npy")
# headway_idm = np.load("./results/headway_idm_with_action_noise.npy")

ttc_sim = np.load("./ttc_radar_sim.npy")
headway_sim = np.load("./headway_radar_sim.npy")

ttc = np.load("./ttc_radar.npy")
headway = np.load("./headway_radar.npy")

# ttc_idm = ttc_idm[ttc_idm < 10]  # only use ttcs below 10s as in paper
# ttc_idm = ttc_idm[ttc_idm > 0]  # only use ttcs below 10s as in paper
# headway_idm = headway_idm[headway_idm < 5]  # only use headways below 5 as in paper

ttc_sim = ttc_sim[ttc_sim < 10]  # only use ttcs below 10s as in paper
ttc_sim = ttc_sim[ttc_sim > 0]  # only use ttcs below 10s as in paper
headway_sim = headway_sim[headway_sim < 5]  # only use headways below 5 as in paper

ttc = ttc[ttc < 10]  # only use ttcs below 10s as in paper
ttc = ttc[ttc > 0]  # only use ttcs below 10s as in paper
headway = headway[headway < 5]  # only use headways below 5 as in paper

# Plot the result
fig, axs = plt.subplots(2, figsize=(20, 10))

# axs[0].plot(ttc)
# axs[0].hist(ttc_idm, bins=100, density=True, label="IDM")
axs[0].hist(ttc_sim, bins=100, density=True, label="Simulated Radar", alpha=0.7)
axs[0].hist(ttc, bins=100, density=True, label="Real Radar", alpha=0.7)
axs[0].set(xlabel="TTC [s]", ylabel="Density")
axs[0].yaxis.labelpad = 20
# axs[0].set_ylim(0, 10)
axs[0].legend()

# axs[1].plot(headway)
# axs[1].hist(headway_idm, bins=100, density=True, label="IDM")
axs[1].hist(headway_sim, bins=100, density=True, label="Simulated Radar", alpha=0.7)
axs[1].hist(headway, bins=100, density=True, label="Real Radar", alpha=0.7)
axs[1].set(xlabel="Headway [s]", ylabel="Density")
axs[1].yaxis.labelpad = 20
# axs[1].set_ylim(0, 5)
axs[1].legend()

plt.subplots_adjust(hspace=0.3)
plt.show()
# plt.savefig("ttc_headway_comparison.png", dpi=300, bbox_inches="tight")
