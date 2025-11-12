import numpy as np
import matplotlib.pyplot as plt

# ttc = np.load("ttc_circular.npy")
# headway = np.load("headway_circular.npy")
# ttc = np.load("ttc_straight.npy")
# headway = np.load("headway_straight.npy")
# ttc = np.load("ttc_waving.npy")
# headway = np.load("headway_waving.npy")
ttc = np.load("ttc_straight_10hz.npy")
headway = np.load("headway_straight_10hz.npy")

ttc = ttc[ttc < 10]  # only use ttcs below 10s as in paper
ttc = ttc[ttc > 0]  # only use ttcs below 10s as in paper
headway = headway[headway < 5]  # only use headways below 5 as in paper

# Plot the result
fig, axs = plt.subplots(2, figsize=(20, 10))

# axs[0].plot(ttc)
axs[0].hist(ttc, bins=100, density=True)
axs[0].set(xlabel="Step", ylabel="TTC")
# axs[0].set_ylim(0, 10)

# axs[1].plot(headway)
axs[1].hist(headway, bins=100, density=True)
axs[1].set(xlabel="Step", ylabel="Headway")
# axs[1].set_ylim(0, 5)

plt.show()
