import pandas as pd
import numpy as np


A=np.load(
    "data/processed/semantic_adj.npy"
)


stations=pd.read_csv(
    "data/processed/shanghai_station_poi.csv"
)


print(stations.columns)


station_col=stations.columns[0]


for i in [0,50,100]:

    idx=np.argsort(A[i])[-6:-1]


    print("\nStation:",
          stations.iloc[i][station_col])


    for j in idx:

        print(
            stations.iloc[j][station_col],
            round(A[i,j],3)
        )