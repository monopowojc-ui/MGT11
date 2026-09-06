import os
import numpy as np
import torch

from torch_geometric.utils import dense_to_sparse

from graphmae_model import GraphMAE



# ===============================
# 1. 路径
# ===============================

BASE_DIR = "data/processed"


feature_path = os.path.join(
    BASE_DIR,
    "poi_feature_semantic.npy"
)


adj_path = os.path.join(
    BASE_DIR,
    "topology_adj.npy"
)


save_path = os.path.join(
    BASE_DIR,
    "station_semantic_embedding.npy"
)



# ===============================
# 2. 读取数据
# ===============================


X = np.load(feature_path)

A = np.load(adj_path)



print("="*50)
print("POI Feature:")
print(X.shape)

print("Topology:")
print(A.shape)
print("="*50)



# ===============================
# 3. adjacency转换edge_index
# ===============================


edge_index, edge_weight = dense_to_sparse(
    torch.tensor(
        A,
        dtype=torch.float32
    )
)


edge_index=edge_index.long()



# 节点特征

x=torch.tensor(
    X,
    dtype=torch.float32
)



print(
    "edge_index:",
    edge_index.shape
)



# ===============================
# 4. GraphMAE模型
# ===============================


model=GraphMAE(
    input_dim=x.shape[1],
    hidden_dim=64,
    embed_dim=32
)



device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("device:",device)



model=model.to(device)

x=x.to(device)

edge_index=edge_index.to(device)



# ===============================
# 5. optimizer
# ===============================


optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)



# ===============================
# 6. Feature Mask
# ===============================


def mask_feature(
        x,
        mask_ratio=0.3
):

    num_nodes=x.shape[0]

    mask_num=int(
        num_nodes*mask_ratio
    )


    mask_nodes=torch.randperm(
        num_nodes
    )[:mask_num]


    x_mask=x.clone()


    # mask token
    mask_token=torch.zeros(
        x.shape[1],
        device=x.device
    )


    x_mask[
        mask_nodes
    ]=mask_token


    return (
        x_mask,
        mask_nodes
    )



# ===============================
# 7. cosine loss
# ===============================


def cosine_loss(
        pred,
        target
):

    loss=1-torch.cosine_similarity(
        pred,
        target,
        dim=1
    )

    return loss.mean()



# ===============================
# 8. Training
# ===============================


epochs=300


for epoch in range(
    epochs
):

    model.train()


    optimizer.zero_grad()



    # mask feature

    x_mask, mask_nodes = mask_feature(
        x,
        mask_ratio=0.3
    )


    # encoder

    z, x_hat = model(
        x_mask,
        edge_index
    )



    # only reconstruct masked nodes

    loss=cosine_loss(
        x_hat[mask_nodes],
        x[mask_nodes]
    )


    loss.backward()


    optimizer.step()



    if epoch % 20 ==0:

        print(
            f"Epoch {epoch:03d} "
            f"Loss={loss.item():.6f}"
        )



# ===============================
# 9. 提取embedding
# ===============================


model.eval()


with torch.no_grad():


    z, _ = model(
        x,
        edge_index
    )


z=z.cpu().numpy()



print(
    "Embedding:",
    z.shape
)



np.save(
    save_path,
    z
)



print("="*50)
print("Saved:")
print(save_path)
print("="*50)