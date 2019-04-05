import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

loss_train = np.loadtxt("./saved_models_d2/train_f1score.txt")
loss_validation = np.loadtxt("./saved_models_d2/val_f1score.txt")

sns.set_context("paper")
plt.plot(loss_train, label="train")
plt.plot(loss_validation, label="validation")
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()