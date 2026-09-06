import numpy as np
import matplotlib.pyplot as plt
import random


# ==================================================
# 参数
# ==================================================

FLOW_PATH = "data/processed/flow_tensor.npy"

SIM_PATH = "data/processed/ddtw_similarity.npy"


TOP_K = 3


# ==================================================
# 读取数据
# ==================================================

flow = np.load(
    FLOW_PATH
)


sim = np.load(
    SIM_PATH
)


print("Flow:",flow.shape)

print("Similarity:",sim.shape)



# ==================================================
# 生成典型日客流曲线
# ==================================================

POINTS_PER_DAY = 103


T,N,C = flow.shape


days = T // POINTS_PER_DAY


print("Days:",days)



# 去除不完整日期

flow = flow[
    :days*POINTS_PER_DAY
]


# reshape

flow_day = flow.reshape(
    days,
    POINTS_PER_DAY,
    N,
    C
)



# 每日平均

daily_flow = np.mean(
    flow_day[:,:,:,0]
    +
    flow_day[:,:,:,1],
    axis=0
)


# 
# shape:
# 103 ×302
#

station_curve = daily_flow.T


print(
    "Station curve:",
    station_curve.shape
)



# ==================================================
# 随机选择站点
# ==================================================

target = random.randint(
    0,
    N-1
)


print("\nTarget station:")
print(target)



# ==================================================
# 找Top3相似站
# ==================================================

similarity = sim[target].copy()


# 排除自己

similarity[target]=-1



top_indices=np.argsort(
    similarity
)[::-1][:TOP_K]



print("\nTop similar stations:")


for idx in top_indices:

    print(
        "Station:",
        idx,
        "Similarity:",
        similarity[idx]
    )



# ==================================================
# 时间轴
# ==================================================

time_labels=[]


hour=6
minute=0


for i in range(POINTS_PER_DAY):


    time_labels.append(
        f"{hour:02d}:{minute:02d}"
    )


    minute+=10


    if minute>=60:

        hour+=1
        minute-=60



# ==================================================
# 绘图
# ==================================================

plt.figure(
    figsize=(12,6)
)



# 原站点

plt.plot(
    station_curve[target],
    linewidth=3,
    label=f"Target {target}"
)



# 相似站点

for idx in top_indices:


    plt.plot(
        station_curve[idx],
        linestyle="--",
        label=f"Similar {idx}"
    )



plt.xticks(
    range(
        0,
        POINTS_PER_DAY,
        6
    ),
    time_labels[::6],
    rotation=45
)


plt.xlabel(
    "Time"
)


plt.ylabel(
    "Passenger Flow"
)


plt.title(
    f"DDTW Similarity Check\nTarget Station {target}"
)



plt.legend()


plt.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    "ddtw_similarity_check.png",
    dpi=300
)


plt.show()