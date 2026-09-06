import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os


# =====================================================
# 参数
# =====================================================

FLOW_PATH = "data/processed/flow_tensor.npy"

OUTPUT_DIR = "data/processed"


POINTS_PER_DAY = 103     # 06:00-23:00 10min

TOP_K = 10               # 每个节点保留10个相似节点

N_JOBS = -1              # 使用全部CPU



# =====================================================
# DDTW derivative
# =====================================================

def derivative(ts):

    n=len(ts)

    d=np.zeros(n)


    for i in range(n):

        if i==0:

            d[i]=ts[1]-ts[0]


        elif i==n-1:

            d[i]=ts[-1]-ts[-2]


        else:

            d[i]=(
                (ts[i]-ts[i-1])
                +
                (ts[i+1]-ts[i-1])/2
            )/2


    return d



# =====================================================
# DDTW距离
# =====================================================

def ddtw_distance(x,y):


    x=derivative(x)

    y=derivative(y)


    m=len(x)

    n=len(y)



    cost=np.zeros(
        (m,n)
    )


    for i in range(m):

        for j in range(n):

            cost[i,j]=(x[i]-y[j])**2



    acc=np.zeros_like(cost)



    acc[0,0]=cost[0,0]


    for i in range(1,m):

        acc[i,0]=(
            cost[i,0]
            +
            acc[i-1,0]
        )


    for j in range(1,n):

        acc[0,j]=(
            cost[0,j]
            +
            acc[0,j-1]
        )


    for i in range(1,m):

        for j in range(1,n):

            acc[i,j]=cost[i,j]+min(
                acc[i-1,j],
                acc[i,j-1],
                acc[i-1,j-1]
            )


    return acc[-1,-1]



# =====================================================
# 并行计算pair
# =====================================================

def compute_pair(i,j,series):


    d=ddtw_distance(
        series[i],
        series[j]
    )


    return i,j,d




def build_ddtw_matrix(series):


    N=len(series)


    pairs=[]


    for i in range(N):

        for j in range(i+1,N):

            pairs.append(
                (i,j)
            )


    print(
        "Pairs:",
        len(pairs)
    )


    results=Parallel(
        n_jobs=N_JOBS,
        backend="loky"
    )(
        delayed(compute_pair)(
            i,
            j,
            series
        )
        for i,j in tqdm(
            pairs
        )
    )



    dist=np.zeros(
        (N,N)
    )


    for i,j,d in results:

        dist[i,j]=d

        dist[j,i]=d



    return dist



# =====================================================
# 距离转相似度
# =====================================================

def distance_to_similarity(dist):


    sigma=np.std(
        dist
    )


    sim=np.exp(
        -(dist**2)
        /
        (2*sigma*sigma)
    )


    np.fill_diagonal(
        sim,
        0
    )


    return sim



# =====================================================
# Top-K图
# =====================================================

def topk_graph(sim,k):


    N=sim.shape[0]


    A=np.zeros_like(sim)



    for i in range(N):

        idx=np.argpartition(
            sim[i],
            -k
        )[-k:]


        A[i,idx]=sim[i,idx]



    # 无向化

    A=np.maximum(
        A,
        A.T
    )


    return A



# =====================================================
# GCN归一化
# =====================================================

def normalize_adj(A):


    A=A+np.eye(
        A.shape[0]
    )


    degree=np.sum(
        A,
        axis=1
    )


    D=np.diag(
        degree**(-0.5)
    )


    return D@A@D



# =====================================================
# 保存边
# =====================================================

def save_edges(A,path):


    rows=[]


    N=A.shape[0]


    for i in range(N):

        for j in range(N):

            if A[i,j]>0 and i!=j:

                rows.append(
                    [
                        i,
                        j,
                        A[i,j]
                    ]
                )


    df=pd.DataFrame(
        rows,
        columns=[
            "source",
            "target",
            "weight"
        ]
    )


    df.to_csv(
        path,
        index=False
    )



# =====================================================
# Main
# =====================================================

if __name__=="__main__":



    print("="*60)

    print(
        "Build DDTW Functional Similarity Graph"
    )

    print("="*60)



    # -------------------------------
    # 1.读取flow
    # -------------------------------


    flow=np.load(
        FLOW_PATH
    )


    print(
        "Original flow:",
        flow.shape
    )



    T,N,C=flow.shape



    # -------------------------------
    # 2.恢复日期
    # -------------------------------


    days=T//POINTS_PER_DAY


    print(
        "Days:",
        days
    )



    flow=flow[
        :days*POINTS_PER_DAY
    ]



    flow_day=flow.reshape(
        days,
        POINTS_PER_DAY,
        N,
        C
    )



    # -------------------------------
    # 3.典型日曲线
    # -------------------------------


    daily=np.mean(
        flow_day[:,:, :,0]
        +
        flow_day[:,:, :,1],
        axis=0
    )



    # 103 ×302

    series=daily.T



    print(
        "Station series:",
        series.shape
    )



    # -------------------------------
    # 4.DDTW
    # -------------------------------


    dist=build_ddtw_matrix(
        series
    )


    np.save(
        os.path.join(
            OUTPUT_DIR,
            "ddtw_distance.npy"
        ),
        dist
    )



    # -------------------------------
    # 5.similarity
    # -------------------------------


    sim=distance_to_similarity(
        dist
    )


    np.save(
        os.path.join(
            OUTPUT_DIR,
            "ddtw_similarity.npy"
        ),
        sim
    )



    # -------------------------------
    # 6.TopK
    # -------------------------------


    A=topk_graph(
        sim,
        TOP_K
    )


    print(
        "Edges:",
        np.sum(A>0)
    )



    # -------------------------------
    # 7.normalize
    # -------------------------------


    A_norm=normalize_adj(
        A
    )



    np.save(
        os.path.join(
            OUTPUT_DIR,
            "ddtw_adj.npy"
        ),
        A_norm
    )



    save_edges(
        A,
        os.path.join(
            OUTPUT_DIR,
            "ddtw_edges.csv"
        )
    )



    print("="*60)

    print("Finished")

    print("="*60)