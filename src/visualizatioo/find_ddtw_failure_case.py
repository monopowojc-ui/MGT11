import numpy as np
import matplotlib.pyplot as plt


# ==============================
# 参数
# ==============================

FLOW_PATH = "data/processed/flow_tensor.npy"


POINTS_PER_DAY = 103


STATION_A = 63

STATION_B = 151



# ==============================
# 读取数据
# ==============================

flow = np.load(
    FLOW_PATH
)


T,N,C = flow.shape



days = T // POINTS_PER_DAY


flow = flow[:days*POINTS_PER_DAY]


flow_day = flow.reshape(
    days,
    POINTS_PER_DAY,
    N,
    C
)



# ==============================
# 构造典型日曲线
# ==============================


daily_flow = np.mean(
    flow_day[:,:,:,0]
    +
    flow_day[:,:,:,1],
    axis=0
)


# time × station

curve = daily_flow.T



station_a_curve = curve[STATION_A]

station_b_curve = curve[STATION_B]



# ==============================
# 峰值
# ==============================


peak_a = np.argmax(
    station_a_curve
)


peak_b = np.argmax(
    station_b_curve
)



value_a = station_a_curve[peak_a]

value_b = station_b_curve[peak_b]



peak_shift = abs(
    peak_a-peak_b
)*10



intensity_diff = abs(
    value_a-value_b
)/max(
    value_a,
    value_b
)



# ==============================
# 时间轴
# ==============================


time=[]


hour=6
minute=0


for i in range(POINTS_PER_DAY):

    time.append(
        f"{hour:02d}:{minute:02d}"
    )


    minute+=10

    if minute>=60:

        hour+=1

        minute-=60



x=np.arange(
    POINTS_PER_DAY
)



# ==============================
# 绘图
# ==============================


plt.figure(
    figsize=(12,6)
)



plt.plot(
    x,
    station_a_curve,
    linewidth=3,
    label=f"Station {STATION_A}"
)



plt.plot(
    x,
    station_b_curve,
    linewidth=3,
    linestyle="--",
    label=f"Station {STATION_B}"
)



# 峰值点

plt.scatter(
    peak_a,
    value_a,
    s=100
)


plt.scatter(
    peak_b,
    value_b,
    s=100
)



# 标注峰值时间

plt.annotate(
    f"Peak\n{time[peak_a]}\n{value_a:.1f}",
    xy=(peak_a,value_a),
    xytext=(peak_a-15,value_a+5),
    arrowprops=dict(
        arrowstyle="->"
    )
)


plt.annotate(
    f"Peak\n{time[peak_b]}\n{value_b:.1f}",
    xy=(peak_b,value_b),
    xytext=(peak_b-20,value_b-8),
    arrowprops=dict(
        arrowstyle="->"
    )
)



plt.xticks(
    range(0,POINTS_PER_DAY,6),
    time[::6],
    rotation=45
)



plt.xlabel(
    "Time"
)


plt.ylabel(
    "Passenger Flow"
)



plt.title(
    "DDTW Similarity Failure Case\n"
    f"Station {STATION_A} vs Station {STATION_B}"
)



# 信息框

text = (
    f"DDTW Similarity: 0.999980\n"
    f"Peak shift: {peak_shift} min\n"
    f"Peak intensity diff: {intensity_diff:.2%}"
)


plt.text(
    0.02,
    0.95,
    text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round"
    )
)



plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()



plt.savefig(
    "DDTW_failure_case_63_151.png",
    dpi=300,
    bbox_inches="tight"
)



plt.show()