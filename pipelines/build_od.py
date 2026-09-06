from src.data_pipeline.od.od_builder import ODBuilder
import numpy as np
import pandas as pd


builder = ODBuilder(
    "data/raw/metroData_ODFlow.csv",
    "data/processed/time_index.csv",
    "data/processed/station_master.csv"
)


flow_tensor, edges = builder.process()


np.save(
    "data/processed/flow_tensor.npy",
    flow_tensor
)


od_edges = pd.concat(
    edges,
    ignore_index=True
)


od_edges.to_csv(
    "data/processed/od_edges.csv",
    index=False
)


print("flow tensor:")
print(flow_tensor.shape)


print("OD edges:")
print(len(od_edges))