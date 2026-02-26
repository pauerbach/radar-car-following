import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import pandas as pd

sns.set_theme()
sns.set_context("paper")
# sns.set(font_scale=1.1)


def fmt_two_digits(x, pos):
    x /= 1e6
    return f"{x:.0f}"


df = pd.read_csv("./reward2.csv")
ax = sns.lineplot(x=df.columns[0], y=df.columns[1], data=df)
ax.set(xlabel="Timesteps (1e6)", ylabel="Average Return")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_two_digits))
plt.gca().set_aspect(20000)


# plt.subplots_adjust(hspace=0.3)
# plt.show()
plt.savefig("reward_over_time.png", dpi=300, bbox_inches="tight")
