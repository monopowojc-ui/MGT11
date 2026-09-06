"""
可视化地铁网络拓扑图
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def visualize_topology(graph, station_file, save_path=None):
    """
    Parameters
    ----------
    graph : networkx.Graph
        build_topology.py生成的拓扑图

    station_file : str
        station_master.csv路径

    save_path : str
        图片保存路径（可选）
    """

    station = pd.read_csv(station_file)

    # 使用真实经纬度作为节点位置
    pos = {
        row.node_index: (row.longitude, row.latitude)
        for _, row in station.iterrows()
    }

    plt.figure(figsize=(10, 10))

    # 绘制边
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="gray",
        width=0.8,
        alpha=0.7
    )

    # 绘制节点
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=18,
        node_color="red"
    )

    plt.title("Shanghai Metro Topology Graph", fontsize=15)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.axis("equal")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )
        print(f"Saved to: {save_path}")

    plt.show()