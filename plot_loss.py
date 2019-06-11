import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import seaborn as sns
from setting import saved_model_folder

# loss_train = np.loadtxt("./saved_models_for_plot/" + saved_model_folder + "_1h/train_f1score.txt")
# loss_validation = np.loadtxt("./saved_models_for_plot/" + saved_model_folder + "_1h/val_f1score.txt")
#
#
# def smooth_data(data, smoothing_weight):
#     last = data[0]
#     for i in range(1, data.shape[0]):
#         data[i] = last * smoothing_weight + (1 - smoothing_weight) * data[i]
#         last = data[i]
#
#     return data
#
#
# sns.set_context("paper")
# plt.plot(smooth_data(loss_train, 0.0), label="train")
# plt.plot(smooth_data(loss_validation, 0.0), label="validation")
# plt.xlabel("Number of epochs", fontsize=16)
# plt.ylabel("Accuracy", fontsize=16)
# plt.legend(loc="best", prop={'size': 16})
# plt.xlim(0, 225)
# plt.ylim(0, 1)
# plt.grid(True, color="darkseagreen", linewidth=0.5)
# # plt.title("Accuracy on Dataset 2", fontsize=16)
# plt.show()

# Plot loss based on size of data

x = np.array([2, 4, 8, 12, 16, 20, 24, 27, 32, 36, 40, 42, 45])
loss_train = np.array([0.0039, 0.0149, 0.0253, 0.0424, 0.0390, 0.0455, 0.0526, 0.0579, 0.0668, 0.0713, 0.0760, 0.0781, 0.0785])
loss_validation = np.array([0.2556, 0.2523, 0.2587, 0.1673, 0.1814, 0.1626, 0.1396, 0.1124, 0.1021, 0.0863, 0.0875, 0.0885, 0.0864])

plt.plot(x, loss_train, label="Train loss")
plt.plot(x, loss_validation, label="Validation loss")
plt.legend(loc="best", prop={'size': 16})
plt.xlabel("Number of data", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.grid(True, color="darkseagreen", linewidth=0.5)
plt.show()


