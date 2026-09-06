import pandas as pd


input_file = (
    "data/processed/station_master.csv"
)

output_file = (
    "data/processed/station_master.csv"
)


df = pd.read_csv(input_file)


# 按站点编号排序
df = df.sort_values(
    "station_id"
).reset_index(drop=True)


# 创建连续节点编号
df.insert(
    0,
    "node_index",
    range(len(df))
)


df.to_csv(
    output_file,
    index=False
)


print(df.head())

print("\n节点数量:")
print(len(df))