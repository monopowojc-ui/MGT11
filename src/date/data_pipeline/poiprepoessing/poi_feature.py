import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


path="data/processed/shanghai_station_poi.csv"


df=pd.read_csv(path)


# 自动保留数值型POI特征
X=df.select_dtypes(
    include=["int64","float64"]
).values


print("POI feature shape:",X.shape)


scaler=StandardScaler()

X_norm=scaler.fit_transform(X)


np.save(
"data/processed/poi_feature.npy",
X_norm
)