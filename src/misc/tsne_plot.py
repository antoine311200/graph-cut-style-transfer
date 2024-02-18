import numpy as np
import matplotlib.pyplot as plt

features_embedded = np.load("graph-cut-style-transfer/data/misc/features_embedded2.npy")
kmeans_labels = np.load("graph-cut-style-transfer/data/misc/kmeans2.npy")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features_embedded[:, 0], features_embedded[:, 1], features_embedded[:, 2], c=kmeans_labels, cmap='cool', s=4)
# Rotate the axes and update
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)
plt.show()