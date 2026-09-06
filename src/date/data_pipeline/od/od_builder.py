import pandas as pd
import numpy as np


class ODBuilder:


    def __init__(
        self,
        od_file,
        time_file,
        station_file
    ):

        self.od_file = od_file
        self.time_file = time_file
        self.station_file = station_file


        self.station_map = {}
        self.time_map = {}



    def load_mapping(self):

        station = pd.read_csv(
            self.station_file
        )

        station.columns = (
            station.columns
            .str.strip()
        )


        # station_id → node_index

        self.station_map = dict(
            zip(
                station.station_id,
                station.node_index
            )
        )


        time = pd.read_csv(
            self.time_file
        )


        self.time_map = dict(
            zip(
                zip(
                    time.date,
                    time.timeslot
                ),
                time.global_timeslot
            )
        )



    def process(self):

        self.load_mapping()

        T = len(self.time_map)
        N = len(self.station_map)


        flow_tensor = np.zeros(
            (T, N, 2),
            dtype=np.float32
        )


        edge_list = []


        for chunk_id, chunk in enumerate(
            pd.read_csv(
                self.od_file,
                chunksize=500000
            )
        ):

            print(
                "processing chunk:",
                chunk_id
            )


            chunk.columns = (
                chunk.columns
                .str.strip()
            )


            # 时间映射

            chunk["global_time"] = [
                self.time_map.get(
                    (d, t),
                    -1
                )
                for d, t in zip(
                    chunk["date"],
                    chunk["timeslot"]
                )
            ]


            chunk = chunk[
                chunk["global_time"] >= 0
            ]


            # OD站点映射

            chunk["source"] = (
                chunk["originStation"]
                .map(
                    self.station_map
                )
            )


            chunk["target"] = (
                chunk["destinationStation"]
                .map(
                    self.station_map
                )
            )


            chunk = chunk.dropna()


            chunk["source"] = (
                chunk["source"]
                .astype(int)
            )

            chunk["target"] = (
                chunk["target"]
                .astype(int)
            )


            flow = (
                chunk["Flow"]
                .astype(np.float32)
            )


            # 起点流量

            np.add.at(
                flow_tensor[:, :, 0],
                (
                    chunk["global_time"],
                    chunk["source"]
                ),
                flow
            )


            # 终点流量

            np.add.at(
                flow_tensor[:, :, 1],
                (
                    chunk["global_time"],
                    chunk["target"]
                ),
                flow
            )


            edge = (
                chunk[
                    [
                        "global_time",
                        "source",
                        "target",
                        "Flow"
                    ]
                ]
                .groupby(
                    [
                        "global_time",
                        "source",
                        "target"
                    ],
                    as_index=False
                )
                .agg(
                    {
                        "Flow":"sum"
                    }
                )
            )

        edge_list.append(edge)
    
        return flow_tensor, edge_list