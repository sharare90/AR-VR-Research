import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime


file_name = "house2"

sns.set(style="ticks", color_codes=True)

data = pd.read_csv("./dataset/" + file_name + "/generated_data_train.csv")
# data = data.sort_values(by=["time"])
# sns.catrplot(x="time", y="action", kind="swarm", data=data)
sns.distplot(data)
plt.show()
