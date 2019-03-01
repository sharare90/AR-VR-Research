import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


file_name = "house2"

sns.set(style="ticks", color_codes=True)

data = pd.read_csv("./dataset/" + file_name + "/generated_data.csv")
data = data.sort_values(by=["time"])
sns.catplot(x="time", y="task", kind="swarm", data=data)
plt.show()
