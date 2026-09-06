import numpy as np
import matplotlib.pyplot as plt


A=np.load(
    "data/processed/ddtw_similarity.npy"
)


plt.figure(
    figsize=(8,7)
)


plt.imshow(
    A,
    cmap="hot"
)


plt.colorbar(
    label="DDTW Similarity"
)


plt.xlabel(
    "Station"
)

plt.ylabel(
    "Station"
)


plt.title(
    "DDTW Functional Similarity Matrix"
)


plt.savefig(
    "ddtw_matrix.png",
    dpi=300
)


plt.show()