import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="ticks", color_codes=True)

data = pd.read_csv("./dataset/time_task.csv")

sns.catplot(x="time", y="task", kind="swarm", data=data)
plt.show()
