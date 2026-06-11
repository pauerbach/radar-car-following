import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2)

ttc_idm = np.load("./ttc_idm_callbak_new.npy")
headway_idm = np.load("./headway_idm_callbak_new.npy")

# ttc_sim = np.load("./ttc_radar_sim.npy")
# headway_sim = np.load("./headway_radar_sim.npy")
ttc_sim = np.load("./ttc_simulated-radar-new.npy")
headway_sim = np.load("./headway_simulated-radar-new.npy")

# ttc_cfar = np.load("./ttc_cfar_new.npy")
# headway_cfar = np.load("./headway_cfar_new.npy")
# ttc_cfar = np.load("./ttc_cfar.npy")
# headway_cfar = np.load("./headway_cfar.npy")
# ttc_cfar = np.load("./ttc_cfar_new2.npy")
# headway_cfar = np.load("./headway_cfar_new2.npy")
ttc_cfar = np.load("./ttc_cfar_kalmann_real.npy")
headway_cfar = np.load("./headway_cfar_kalmann_real.npy")

# ttc = np.load("./ttc_radar.npy")
# headway = np.load("./headway_radar.npy")
# ttc = np.load("./ttc_real_new.npy")
# headway = np.load("./headway_real_new.npy")
ttc = np.load("./ttc_end-to-end-new.npy")
headway = np.load("./headway_end-to-end-new.npy")
# ttc = np.load("./ttc_end-to-end-bananasmooth.npy")
# headway = np.load("./headway_end-to-end-bananasmooth.npy")
# ttc = np.load("./ttc_end-to-end-bananaangle.npy")
# headway = np.load("./headway_end-to-end-bananaangle.npy")

# --- Filtering (as in original script) ---
ttc_idm = ttc_idm[(ttc_idm < 10) & (ttc_idm > 0)]
headway_idm = headway_idm[headway_idm < 5]

ttc_sim = ttc_sim[(ttc_sim < 10) & (ttc_sim > 0)]
headway_sim = headway_sim[headway_sim < 5]

ttc_cfar = ttc_cfar[ttc_cfar < 10]  # only use ttcs below 10s as in paper
ttc_cfar = ttc_cfar[ttc_cfar > 0]  # only use ttcs below 10s as in paper
headway_cfar = headway_cfar[headway_cfar < 5]  # only use headways below 5 as in paper

ttc = ttc[(ttc < 10) & (ttc > 0)]
headway = headway[headway < 5]

# --- Plotting ---
fig, axs = plt.subplots(2, figsize=(20, 10))

# TTC KDEs
sns.kdeplot(ttc_idm, ax=axs[0], label="IDM", linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc_sim, ax=axs[0], label="Simulated Radar", linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc_cfar, ax=axs[0], label="CFAR", linewidth=4, bw_adjust=0.2)
sns.kdeplot(ttc, ax=axs[0], label="Range-Doppler Input", linewidth=4, bw_adjust=0.2)

axs[0].set(xlabel="TTC [s]", ylabel="Density")
axs[0].yaxis.labelpad = 20
axs[0].set_xlim(0, 10)
axs[0].set_ylim(bottom=0)
axs[0].legend()

# Headway KDEs
sns.kdeplot(headway_idm, ax=axs[1], label="IDM", linewidth=4, bw_adjust=0.2)
sns.kdeplot(headway_sim, ax=axs[1], label="Simulated Radar", linewidth=4, bw_adjust=0.2)
sns.kdeplot(headway_cfar, ax=axs[1], label="CFAR", linewidth=4, bw_adjust=0.2)
sns.kdeplot(headway, ax=axs[1], label="Range-Doppler Input", linewidth=4, bw_adjust=0.2)

axs[1].set(xlabel="Headway [s]", ylabel="Density")
axs[1].yaxis.labelpad = 20
axs[1].set_xlim(0, 3)
axs[1].set_ylim(bottom=0)
axs[1].legend()

plt.subplots_adjust(hspace=0.3)
# plt.show()
plt.savefig(
    "ttc_headway_comparison_real_world_kde_new_runs.png", dpi=300, bbox_inches="tight"
)
