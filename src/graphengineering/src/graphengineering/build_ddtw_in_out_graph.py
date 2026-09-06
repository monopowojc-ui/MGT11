import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os


# ======================================================
# 参数
# ======================================================

FLOW_PATH = "data/processed/flow_tensor.npy"

OUTPUT_DIR = "data/processed"


POINTS_PER_DAY = 103

TOP_K = 10

N_JOBS = -1



# ======================================================
# DDTW导数
# ======================================================

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
                0.5*(ts[i+1]-ts[i-1])
            )/2


    return d



# ======================================================
# DDTW距离
# ======================================================

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

            cost[i,j]=abs(
                x[i]-y[j]
            )



    acc=np.zeros_like(cost)


    acc[0,0]=cost[0,0]


    for i in range(1,m):

        acc[i,0]=(
            acc[i-1,0]
            +
            cost[i,0]
        )


    for j in range(1,n):

        acc[0,j]=(
            acc[0,j-1]
            +
            cost[0,j]
        )



    for i in range(1,m):

        for j in range(1,n):

            acc[i,j]=cost[i,j]+min(

                acc[i-1,j],

                acc[i,j-1],

                acc[i-1,j-1]

            )


    return acc[-1,-1]



# ======================================================
# 并行计算矩阵
# ======================================================


def calculate_pair(i,j,series):


    d=ddtw_distance(
        series[i],
        series[j]
    )


    return i,j,d




def build_distance_matrix(series):


    N=series.shape[0]


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
        n_jobs=N_JOBS
    )(
        delayed(calculate_pair)(
            i,
            j,
            series
        )
        for i,j in tqdm(pairs)
    )


    dist=np.zeros(
        (N,N)
    )


    for i,j,d in results:

        dist[i,j]=d

        dist[j,i]=d



    return dist



# ======================================================
# 距离转换相似度
# ======================================================


def distance_to_similarity(dist):


    sigma=np.std(
        dist
    )


    sim=np.exp(
        -dist/sigma
    )


    np.fill_diagonal(
        sim,
        0
    )


    return sim



# ======================================================
# Top-K稀疏
# ======================================================


def build_topk_graph(sim,k):


    N=sim.shape[0]


    A=np.zeros_like(sim)



    for i in range(N):


        index=np.argpartition(
            sim[i],
            -k
        )[-k:]


        A[i,index]=sim[i,index]



    # 无向化

    A=np.maximum(
        A,
        A.T
    )


    return A



# ======================================================
# GCN归一化
# ======================================================


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



# ======================================================
# 保存edge
# ======================================================


def save_edges(A,path):


    edges=[]


    N=A.shape[0]


    for i in range(N):

        for j in range(N):

            if A[i,j]>0 and i!=j:

                edges.append(
                    [
                        i,
                        j,
                        A[i,j]
                    ]
                )



    df=pd.DataFrame(
        edges,
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



# ======================================================
# 单个方向建图
# ======================================================


def build_one_graph(series,name):


    print("\n====================")

    print(
        "Building:",
        name
    )

    print("====================")


    # DDTW距离

    dist=build_distance_matrix(
        series
    )


    np.save(
        f"{OUTPUT_DIR}/ddtw_{name}_distance.npy",
        dist
    )



    # 相似度

    sim=distance_to_similarity(
        dist
    )


    np.save(
        f"{OUTPUT_DIR}/ddtw_{name}_similarity.npy",
        sim
    )



    # TopK

    A=build_topk_graph(
        sim,
        TOP_K
    )


    print(
        "Edges:",
        np.sum(A>0)
    )



    # GCN adj

    A_norm=normalize_adj(
        A
    )


    np.save(
        f"{OUTPUT_DIR}/ddtw_{name}_adj.npy",
        A_norm
    )



    save_edges(
        A,
        f"{OUTPUT_DIR}/ddtw_{name}_edges.csv"
    )



# ======================================================
# main
# ======================================================


if __name__=="__main__":


    print(
        "Load flow tensor"
    )


    flow=np.load(
        FLOW_PATH
    )


    print(
        "Flow:",
        flow.shape
    )



    T,N,C=flow.shape



    # --------------------------
    # 日期重构
    # --------------------------


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



    print(
        "Days:",
        days
    )



    # --------------------------
    # 平均典型日
    # --------------------------


    inflow=np.mean(
        flow_day[:,:,:,0],
        axis=0
    )


    outflow=np.mean(
        flow_day[:,:,:,1],
        axis=0
    )



    # 
    # 103 ×302
    #

    inflow=inflow.T

    outflow=outflow.T



    print(
        "In series:",
        inflow.shape
    )


    print(
        "Out series:",
        outflow.shape
    )



    # --------------------------
    # 分别建图
    # --------------------------


    build_one_graph(
        inflow,
        "in"
    )


    build_one_graph(
        outflow,
        "out"
    )



    print("\nFinished")