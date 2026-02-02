import numpy as np
from matplotlib import pyplot as plt

T = 5
N = 1000
mu = 0.4226
sigma = 0.4365
# h = (d + leader.LENGTH) / vehicle.speed

h = np.linspace(0, T, N)
r_eff = (
    1
    / (np.sqrt(2 * np.pi) * h * sigma)
    * np.exp(-((np.log(h) - mu) ** 2) / (2 * sigma**2))
)

ttc = np.linspace(0, T, N)
# if ttc > 0 and ttc < 1.5:
r_safe = np.log(ttc / 1.5)

plt.plot(np.linspace(0, T, N), r_eff)
# plt.plot(np.linspace(0, T, N), r_safe)
plt.title("Reward shape for efficiency")
plt.xlabel("Headway [s]")
plt.ylabel("Reward")
plt.show()
