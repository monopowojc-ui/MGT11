"""
build_od_graph.py

Construct Static Origin-Destination Graph

Input
-----
data/processed/od_edges.csv
data/processed/station_master.csv

Output
------
data/processed/od_adj.npy
data/processed/od_adj_norm.npy
data/processed/od_edges_weight.csv

Visualization
-------------
data/processed/od_heatmap.png
data/processed/od_network.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# ======================================================
# Configuration
# ======================================================

DATA_DIR = Path("data/processed")

OD_FILE = DATA_DIR / "od_edges.csv"
STATION_FILE = DATA_DIR / "station_master.csv"

OUTPUT_ADJ = DATA_DIR / "od_adj.npy"
OUTPUT_ADJ_NORM = DATA_DIR / "od_adj_norm.npy"
OUTPUT_EDGE = DATA_DIR / "od_edges_weight.csv"

OUTPUT_HEATMAP = DATA_DIR / "od_heatmap.png"
OUTPUT_NETWORK = DATA_DIR / "od_network.png"

# ======================================================
# Load Data
# ======================================================

print("=" * 60)
print("Building Static OD Graph")
print("=" * 60)

station = pd.read_csv(STATION_FILE)
NUM_NODES = len(station)

print(f"Stations   : {NUM_NODES}")

od = pd.read_csv(OD_FILE)

print(f"OD Records : {len(od)}")
print("Columns    :", list(od.columns))

# ======================================================
# Aggregate Flow
# ======================================================

edge_weight = (
    od.groupby(["source", "target"], as_index=False)["Flow"]
      .sum()
      .rename(columns={"Flow": "weight"})
)

print(f"Unique OD Pairs : {len(edge_weight)}")

# ======================================================
# Build Adjacency Matrix
# ======================================================

adj = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

adj[
    edge_weight["source"].astype(int),
    edge_weight["target"].astype(int)
] = edge_weight["weight"]

# 保存原始矩阵
np.save(OUTPUT_ADJ, adj)

# ======================================================
# Symmetric Normalization
# Â = D^(-1/2)(A+I)D^(-1/2)
# ======================================================

adj_norm = np.log1p(adj)

# Add Self-loop
adj_norm = adj_norm + np.eye(NUM_NODES, dtype=np.float32)

degree = np.sum(adj_norm, axis=1)

degree_inv_sqrt = np.power(degree, -0.5)

degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

D_inv_sqrt = np.diag(degree_inv_sqrt)

adj_norm = D_inv_sqrt @ adj_norm @ D_inv_sqrt

np.save(OUTPUT_ADJ_NORM, adj_norm)

# ======================================================
# Save Weighted Edge List
# ======================================================

edge_weight.to_csv(OUTPUT_EDGE, index=False)

# ======================================================
# Visualization 1
# Heatmap
# ======================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    adj_norm,
    cmap="viridis",
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(label="Normalized Weight")

plt.title("Normalized Static OD Graph")

plt.xlabel("Destination")

plt.ylabel("Origin")

plt.tight_layout()

plt.savefig(OUTPUT_HEATMAP, dpi=300)

plt.close()

# ======================================================
# Visualization 2
# Network Graph
# ======================================================

TOP_EDGE = 50

top_edges = edge_weight.sort_values(
    by="weight",
    ascending=False
).head(TOP_EDGE)

G = nx.DiGraph()

G.add_nodes_from(range(NUM_NODES))

for _, row in top_edges.iterrows():

    G.add_edge(
        int(row["source"]),
        int(row["target"]),
        weight=row["weight"]
    )

plt.figure(figsize=(12, 12))

pos = nx.spring_layout(
    G,
    seed=42,
    k=0.35
)

weights = np.array(
    [G[u][v]["weight"] for u, v in G.edges()]
)

weights = weights / weights.max() * 4

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=30,
    node_color="steelblue"
)

nx.draw_networkx_edges(
    G,
    pos,
    width=weights,
    alpha=0.6,
    arrows=False
)

plt.title(f"Top {TOP_EDGE} OD Connections")

plt.axis("off")

plt.tight_layout()

plt.savefig(OUTPUT_NETWORK, dpi=300)

plt.close()

# ======================================================
# Statistics
# ======================================================

density = np.count_nonzero(adj) / (NUM_NODES * NUM_NODES)

print("=" * 60)
print("Static OD Graph Built")
print("=" * 60)

print(f"Nodes          : {NUM_NODES}")
print(f"Edges          : {len(edge_weight)}")
print(f"Total Flow     : {adj.sum():,.0f}")
print(f"Density        : {density:.6f}")

print("=" * 60)

print("Saved:")

print(OUTPUT_ADJ)
print(OUTPUT_ADJ_NORM)
print(OUTPUT_EDGE)
print(OUTPUT_HEATMAP)
print(OUTPUT_NETWORK)

print("=" * 60)