import pandas as pd
import os


input_file = "data/raw/metroData_ODFlow.csv"

output_file = "data/processed/time_index.csv"


# 保存所有时间组合
time_list = []


# 分块读取
for i, chunk in enumerate(
        pd.read_csv(
            input_file,
            chunksize=500000
        )
):

    print("processing chunk:", i)

    # 清除字段空格
    chunk.columns = chunk.columns.str.strip()


    # 只保留时间相关字段
    temp = chunk[
        [
            "date",
            "timeslot",
            "startTime",
            "endTime"
        ]
    ]


    time_list.append(temp.drop_duplicates())


# 合并时间表
time_df = pd.concat(
    time_list,
    ignore_index=True
)


# 删除重复时间
time_df = (
    time_df
    .drop_duplicates()
    .sort_values(
        [
            "date",
            "timeslot"
        ]
    )
    .reset_index(drop=True)
)


# 创建全局时间编号
time_df.insert(
    0,
    "global_timeslot",
    range(len(time_df))
)


# 转换时间

def convert_time(x):

    x = str(x).zfill(4)

    return int(x[:2]), int(x[2:])


time_df[
    ["hour","minute"]
] = time_df["startTime"].apply(
    lambda x:
    pd.Series(convert_time(x))
)


# 星期

time_df["date"] = (
    time_df["date"]
    .astype(str)
)


time_df["datetime"] = pd.to_datetime(
    time_df["date"]
)


time_df["weekday"] = (
    time_df["datetime"]
    .dt.weekday
)


# 保存

time_df.to_csv(
    output_file,
    index=False
)


print("\n完成")
print(time_df.head())

print(
    "时间节点数量:",
    len(time_df)
)