"""
构建上海地铁网络拓扑图

输入:
    data/processed/station_master.csv

输出:
    data/processed/topology_adj.npy
    data/processed/topology_edges.csv
"""

import ast
import os

import networkx as nx
import numpy as np
import pandas as pd


class TopologyBuilder:

    def __init__(self, station_file):

        self.station = pd.read_csv(station_file)

        # 将字符串转换成list
        self.station["neighbors"] = self.station["neighbors"].apply(ast.literal_eval)

        # station_id -> node_index
        self.id2node = dict(
            zip(
                self.station.station_id,
                self.station.node_index
            )
        )

        self.graph = nx.Graph()

    def build_graph(self):

        # 添加节点
        for _, row in self.station.iterrows():

            self.graph.add_node(
                row.node_index,
                station_id=row.station_id,
                station_name=row.station_name,
                longitude=row.longitude,
                latitude=row.latitude
            )

        # 添加边
        for _, row in self.station.iterrows():

            u = row.node_index

            for neighbor in row.neighbors:

                if neighbor in self.id2node:

                    v = self.id2node[neighbor]

                    self.graph.add_edge(u, v)

        print("=" * 50)
        print("Topology Graph Built")
        print(f"Nodes : {self.graph.number_of_nodes()}")
        print(f"Edges : {self.graph.number_of_edges()}")
        print("=" * 50)

    def adjacency_matrix(self):

        A = nx.to_numpy_array(
            self.graph,
            nodelist=range(len(self.station)),
            dtype=np.float32
        )

        return A

    def edge_dataframe(self):

        edges = []

        for u, v in self.graph.edges():

            edges.append([u, v])

        return pd.DataFrame(
            edges,
            columns=["source", "target"]
        )

    def save(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)

        A = self.adjacency_matrix()

        np.save(
            os.path.join(save_dir, "topology_adj.npy"),
            A
        )

        edge_df = self.edge_dataframe()

        edge_df.to_csv(
            os.path.join(save_dir, "topology_edges.csv"),
            index=False
        )

        print("Saved:")
        print(os.path.join(save_dir, "topology_adj.npy"))
        print(os.path.join(save_dir, "topology_edges.csv"))


if __name__ == "__main__":

    builder = TopologyBuilder(
        "data/processed/station_master.csv"
    )

    builder.build_graph()

    builder.save(
        "data/processed"
    )

from visualizatioo.visualize_graph import visualize_topology

if __name__ == "__main__":

    builder = TopologyBuilder(
        "data/processed/station_master.csv"
    )

    builder.build_graph()

    builder.save("data/processed")

    visualize_topology(
        builder.graph,
        "data/processed/station_master.csv",
        save_path="picture/topology_graph.png"
    )