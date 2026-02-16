import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("paper")

# plot_until = 50 * 10
commanded_speeds = np.load("./commanded_speeds_ego.npy")
actual_speeds = np.load("./actual_speeds_ego.npy")
# x = np.linspace(0, 50, plot_until)
#
print(commanded_speeds.shape)
print(actual_speeds.shape)

# plt.plot(x, speeds[:plot_until])
sns.lineplot(commanded_speeds)
sns.lineplot(actual_speeds)
plt.xlabel("Time [sec]")
plt.ylabel("Velocity [m/s]")

# plt.gca().set_aspect(20)

plt.show()
# plt.savefig("leader_speeds.png", dpi=600, bbox_inches="tight")
