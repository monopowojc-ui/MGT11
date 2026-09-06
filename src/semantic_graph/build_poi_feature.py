import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


input_path = "data/processed/shanghai_station_poi.csv"

output_path = "data/processed/poi_feature_semantic.npy"



# ==========================
# 读取
# ==========================

df=pd.read_csv(
    input_path
)


print("Original:")
print(df.shape)



# ==========================
# 选择语义功能变量
# ==========================

feature_cols=[

    "entropy",

    "Restaurant",

    "Shopping",

    "Office",

    "Education",

    "Medical",

    "Residential",

    "Transport",

    "Entertainment",

    "Hotel",

    "Finance"

]


X=df[feature_cols].copy()



print("Before weight:")
print(X.head())



# ==========================
# 降低Transport影响
# ==========================

X["Transport"] = (
    X["Transport"] * 0.5
)



# ==========================
# 标准化
# ==========================

scaler=StandardScaler()


X_norm=scaler.fit_transform(
    X
)



print("Final feature:")
print(
    X_norm.shape
)



# ==========================
# 保存
# ==========================

np.save(
    output_path,
    X_norm
)


print("="*50)

print("Saved:")
print(output_path)

print("="*50)