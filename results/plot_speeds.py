import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("paper")

plot_until = 50 * 10
speeds = np.load("./leader_speeds.npy")
x = np.linspace(0, 50, plot_until)

# plt.plot(x, speeds[:plot_until])
sns.lineplot(x=x, y=speeds[:plot_until])
plt.xlabel("Time [sec]")
plt.ylabel("Velocity [m/s]")

plt.gca().set_aspect(20)

plt.show()
# plt.savefig("leader_speeds.png", dpi=600, bbox_inches="tight")
