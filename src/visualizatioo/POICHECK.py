import os
import numpy as np
import matplotlib.pyplot as plt



# =====================================================
# 参数
# =====================================================

FLOW_PATH = (
    "data/processed/flow_tensor.npy"
)


DIST_PATH = (
    "data/processed/"
    "directional_adaptive_distance.npy"
)


STATION_ID = 44


TOP_K = 3


POINTS_PER_DAY = 103



OUTPUT_DIR = (
    "data/processed/"
    "station_similarity_check"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# =====================================================
# 1. 读取典型日进出站曲线
# =====================================================

def load_directional_series():


    flow=np.load(
        FLOW_PATH
    )


    T,N,C=flow.shape


    days=T//POINTS_PER_DAY


    flow=flow[
        :days*POINTS_PER_DAY
    ]


    flow_day=flow.reshape(
        days,
        POINTS_PER_DAY,
        N,
        C
    )


    inbound=np.mean(
        flow_day[:,:,:,0],
        axis=0
    )


    outbound=np.mean(
        flow_day[:,:,:,1],
        axis=0
    )


    # station,time

    return inbound.T, outbound.T



# =====================================================
# 2. 找相似站点
# =====================================================

def find_similar_station(
        dist,
        station_id,
        k=3
):


    d=dist[
        station_id
    ].copy()


    # 自己排除

    d[station_id]=np.inf



    idx=np.argsort(
        d
    )[:k]


    return idx,d[idx]



# =====================================================
# 3. 绘制比较
# =====================================================

def plot_compare(
        inbound,
        outbound,
        station_id,
        similar_ids
):


    x=np.arange(
        inbound.shape[1]
    )


    # -------------------
    # inbound
    # -------------------

    plt.figure(
        figsize=(12,5)
    )


    plt.plot(
        x,
        inbound[station_id],
        linewidth=3,
        label=f"Station {station_id}"
    )


    for sid in similar_ids:


        plt.plot(
            x,
            inbound[sid],
            linestyle="--",
            label=f"Station {sid}"
        )


    plt.title(
        f"Inbound flow similarity: Station {station_id}"
    )


    plt.xlabel(
        "Time index (10min)"
    )


    plt.ylabel(
        "Inbound flow"
    )


    plt.legend()

    plt.grid()


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"station_{station_id}_inbound_similarity.png"
        ),
        dpi=300
    )


    plt.show()



    # -------------------
    # outbound
    # -------------------


    plt.figure(
        figsize=(12,5)
    )


    plt.plot(
        x,
        outbound[station_id],
        linewidth=3,
        label=f"Station {station_id}"
    )


    for sid in similar_ids:


        plt.plot(
            x,
            outbound[sid],
            linestyle="--",
            label=f"Station {sid}"
        )


    plt.title(
        f"Outbound flow similarity: Station {station_id}"
    )


    plt.xlabel(
        "Time index (10min)"
    )


    plt.ylabel(
        "Outbound flow"
    )


    plt.legend()


    plt.grid()


    plt.tight_layout()



    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"station_{station_id}_outbound_similarity.png"
        ),
        dpi=300
    )


    plt.show()



# =====================================================
# Main
# =====================================================


if __name__=="__main__":


    print("="*50)

    print(
        "Load distance matrix"
    )


    dist=np.load(
        DIST_PATH
    )


    print(
        "Distance shape:",
        dist.shape
    )


    similar_ids,values=find_similar_station(
        dist,
        STATION_ID,
        TOP_K
    )



    print("="*50)

    print(
        f"Station {STATION_ID} top {TOP_K} similar stations:"
    )


    for sid,d in zip(
        similar_ids,
        values
    ):

        print(
            f"Station {sid}, distance={d:.4f}"
        )



    print("="*50)



    inbound,outbound=load_directional_series()



    plot_compare(
        inbound,
        outbound,
        STATION_ID,
        similar_ids
    )


    print(
        "Finished"
    )