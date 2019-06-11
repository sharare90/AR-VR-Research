import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime


file_name = "house1"

sns.set(style="ticks", color_codes=True)

data = pd.read_csv("./dataset/" + file_name + "/time_task_train.csv")

sns.boxplot(x="time", y="action", data=data,
            whis="range", palette="vlag")

sns.swarmplot(x="time", y="action", data=data,
              size=2, color=".3", linewidth=0)

plt.grid(True)
plt.ylabel("", fontsize=16)
sns.despine(trim=True, left=True)

plt.show()
