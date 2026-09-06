import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors



embedding_path = "data/processed/station_semantic_embedding.npy"

save_path = "data/processed/semantic_adj.npy"



# =========================
# 1. load embedding
# =========================

Z=np.load(
    embedding_path
)


print(
    "Embedding:",
    Z.shape
)



# =========================
# 2. cosine similarity
# =========================

S=cosine_similarity(Z)


print(
    "Similarity:",
    S.shape
)



# =========================
# 3. Top-K semantic graph
# =========================

k=10


semantic_adj=np.zeros_like(S)



for i in range(S.shape[0]):

    # 排除自己
    idx=np.argsort(
        S[i]
    )[-k-1:-1]


    semantic_adj[i,idx]=S[i,idx]



# 对称化
semantic_adj=np.maximum(
    semantic_adj,
    semantic_adj.T
)



# =========================
# 4. save
# =========================

np.save(
    save_path,
    semantic_adj
)


print("="*50)

print(
    "Saved:",
    save_path
)


print(
    "Edges:",
    np.sum(
        semantic_adj>0
    )
)

print("="*50)