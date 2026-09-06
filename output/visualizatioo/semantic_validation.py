import os
import numpy as np
import pandas as pd


# ==============================
# 路径
# ==============================

DATA_DIR = "data/processed"


semantic_path = os.path.join(
    DATA_DIR,
    "semantic_adj.npy"
)


poi_path = os.path.join(
    DATA_DIR,
    "shanghai_station_poi.csv"
)


save_path = os.path.join(
    DATA_DIR,
    "semantic_validation_cn.csv"
)



# ==============================
# 读取
# ==============================

semantic_adj = np.load(
    semantic_path
)


poi = pd.read_csv(
    poi_path
)


print("语义图:", semantic_adj.shape)
print("POI数据:", poi.shape)



# ==============================
# 自动识别站名
# ==============================

station_col = poi.columns[0]


stations = poi[station_col].values



# ==============================
# POI字段中文映射
# 根据你的csv修改
# ==============================

poi_translate = {

    "Restaurant":"餐饮",

    "Shopping":"商业购物",

    "Office":"办公",

    "Education":"教育",

    "Medical":"医疗",

    "Residential":"居住",

    "Transport":"交通",

    "Entertainment":"休闲娱乐",

    "POI_total":"POI总量",

    "density":"POI密度"

}



# ==============================
# 获取站点功能
# ==============================

def get_function(row):


    ignore=[
        station_col,
        "longitude",
        "latitude",
        "lon",
        "lat"
    ]


    values={}


    for col in poi.columns:


        if col not in ignore:

            values[col]=row[col]


    # 排序

    values=sorted(
        values.items(),
        key=lambda x:x[1],
        reverse=True
    )


    result=[]


    for k,v in values[:3]:

        cname=poi_translate.get(
            k,
            k
        )


        result.append(
            f"{cname}({int(v)})"
        )


    return "、".join(result)



# ==============================
# 指定典型站点
# ==============================

test_stations=[

    "East Changji Road",

    "Longcao Road",

    "Liziyuan",

    "Jing'an Temple",

    "Shanghai Automobile City"

]


sample_ids=[]


for s in test_stations:


    idx=np.where(
        stations==s
    )[0]


    if len(idx)>0:

        sample_ids.append(
            idx[0]
        )



# 如果没有找到，则随机20个

if len(sample_ids)==0:


    np.random.seed(42)


    sample_ids=np.random.choice(
        len(stations),
        20,
        replace=False
    )



# ==============================
# 生成结果
# ==============================

results=[]


for i in sample_ids:


    similarity=semantic_adj[i].copy()


    similarity[i]=0


    neighbors=np.argsort(
        similarity
    )[-5:][::-1]


    row={}


    row["站点"]=stations[i]


    for rank,n in enumerate(neighbors,1):


        row[
            f"相似站点{rank}"
        ]=stations[n]


        row[
            f"相似度{rank}"
        ]=round(
            similarity[n],
            3
        )


    row["主要城市功能"]=get_function(
        poi.iloc[i]
    )


    results.append(row)



# ==============================
# 保存中文CSV
# ==============================


result=pd.DataFrame(
    results
)


result.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)


print("="*50)

print("已生成:")

print(save_path)


print(result)

print("="*50)