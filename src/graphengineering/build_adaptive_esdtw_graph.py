import os
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed



# =====================================================
# 参数
# =====================================================

FLOW_PATH = (
    "data/processed/flow_tensor.npy"
)


OUTPUT_DIR = (
    "data/processed"
)


POINTS_PER_DAY = 103


TOP_K = 10


N_JOBS = -1



# DTW权重

LAMBDA1 = 0.7

LAMBDA2 = 0.3



# 峰参数

THETA = 1.5

PROMINENCE_RATIO = 0.25


WINDOW = 6



ALPHA = 0.5



# =====================================================
# 1. 读取数据并拆工作日/周末
# =====================================================


def load_daytype_flow():


    flow=np.load(
        FLOW_PATH
    )


    print(
        "Flow:",
        flow.shape
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


    # 假设日期顺序连续
    # day index:
    # 0 Monday


    weekday=[]

    weekend=[]



    for d in range(days):


        if d%7 <5:

            weekday.append(d)

        else:

            weekend.append(d)



    weekday=np.array(
        weekday
    )


    weekend=np.array(
        weekend
    )



    print(
        "weekday days:",
        len(weekday)
    )


    print(
        "weekend days:",
        len(weekend)
    )



    # -------------------
    # inbound
    # -------------------

    in_flow=flow_day[:,:,:,0]


    out_flow=flow_day[:,:,:,1]



    weekday_in=np.mean(

        in_flow[weekday],

        axis=0

    )


    weekend_in=np.mean(

        in_flow[weekend],

        axis=0

    )



    weekday_out=np.mean(

        out_flow[weekday],

        axis=0

    )


    weekend_out=np.mean(

        out_flow[weekend],

        axis=0

    )



    # station,time

    return (

        weekday_in.T,

        weekend_in.T,

        weekday_out.T,

        weekend_out.T

    )



# =====================================================
# 2. 通勤峰检测
# =====================================================


def detect_commute_peak(x):


    mu=np.mean(x)

    sigma=np.std(x)+1e-6



    dx=np.diff(x)



    morning=[]

    evening=[]



    for t in range(
        WINDOW,
        len(x)-WINDOW
    ):


        # 局部极值

        if not(
            dx[t-1]>0
            and
            dx[t]<0
        ):

            continue



        # 时间窗口

        if 6<=t<=24:

            period="morning"


        elif 66<=t<=84:

            period="evening"


        else:

            continue



        Rt=(

            x[t]-mu

        )/sigma



        if Rt<THETA:

            continue



        # 峰突出度

        left=np.mean(

            x[
                t-WINDOW:t
            ]

        )


        right=np.mean(

            x[
                t+1:
                t+WINDOW+1
            ]

        )


        prominence=x[t]-(
            left+right
        )/2



        if prominence < (
            PROMINENCE_RATIO*sigma
        ):

            continue



        if period=="morning":

            morning.append(t)

        else:

            evening.append(t)



    # 必须存在通勤峰
    flag=(

        len(morning)>0

        or

        len(evening)>0

    )


    return flag



# =====================================================
# 3. derivative
# =====================================================


def derivative(x):


    d=np.zeros_like(x)



    for i in range(len(x)):


        if i==0:

            d[i]=x[1]-x[0]

        elif i==len(x)-1:

            d[i]=x[-1]-x[-2]

        else:

            d[i]=(

                (x[i]-x[i-1])

                +

                (x[i+1]-x[i-1])/2

            )/2



    return d



# =====================================================
# 4. descriptor
# =====================================================


def descriptor(x):


    dx=np.diff(x)


    points=[]



    for t in range(1,len(dx)):


        if dx[t-1]*dx[t]<0:

            points.append(t)



    result=[]


    for t in points:


        left=max(
            0,
            t-WINDOW
        )


        right=min(
            len(x),
            t+WINDOW+1
        )


        w=x[left:right]



        pad=np.zeros(
            2*WINDOW+1
        )


        pad[:len(w)]=w



        grad=np.diff(pad)



        result.append(

            np.concatenate(
                [
                    pad,
                    grad
                ]
            )

        )



    if len(result)==0:


        return np.zeros(
            (1,2*WINDOW+1)
        )


    return np.array(result)



# =====================================================
# 5. DTW
# =====================================================


def dtw(A,B,x,y):


    m=len(A)

    n=len(B)



    dp=np.ones(
        (m,n)
    )*np.inf



    cost=np.zeros(
        (m,n)
    )


    for i in range(m):

        for j in range(n):


            f=np.linalg.norm(

                A[i]-B[j]

            )


            r=abs(

                x[min(i,len(x)-1)]

                -

                y[min(j,len(y)-1)]

            )


            cost[i,j]=(

                LAMBDA1*f

                +

                LAMBDA2*r

            )



    dp[0,0]=cost[0,0]



    for i in range(1,m):

        dp[i,0]=cost[i,0]+dp[i-1,0]


    for j in range(1,n):

        dp[0,j]=cost[0,j]+dp[0,j-1]


    for i in range(1,m):

        for j in range(1,n):


            dp[i,j]=cost[i,j]+min(

                dp[i-1,j],

                dp[i,j-1],

                dp[i-1,j-1]

            )


    return dp[-1,-1]



# =====================================================
# 6. distance
# =====================================================


def ddtw(x,y):


    return dtw(

        derivative(x).reshape(-1,1),

        derivative(y).reshape(-1,1),

        x,

        y

    )



def esdtw(x,y):


    return dtw(

        descriptor(x),

        descriptor(y),

        x,

        y

    )



def adaptive_distance(x,y,flag1,flag2):


    if flag1 and flag2:

        return esdtw(x,y)

    else:

        return ddtw(x,y)



# =====================================================
# 7. pair
# =====================================================


def pair_distance(

        i,

        j,

        win_in,

        win_out,

        flag_in,

        flag_out

):


    Din=adaptive_distance(

        win_in[i],

        win_in[j],

        flag_in[i],

        flag_in[j]

    )


    Dout=adaptive_distance(

        win_out[i],

        win_out[j],

        flag_out[i],

        flag_out[j]

    )



    return (

        i,

        j,

        ALPHA*Din
        +(1-ALPHA)*Dout

    )



# =====================================================
# 8. matrix
# =====================================================


def build_distance(

        win_in,

        win_out,

        flag_in,

        flag_out

):


    N=len(win_in)


    pairs=[

        (i,j)

        for i in range(N)

        for j in range(i+1,N)

    ]


    print(
        "Pairs:",
        len(pairs)
    )



    results=Parallel(

        n_jobs=N_JOBS

    )(

        delayed(pair_distance)(

            i,

            j,

            win_in,

            win_out,

            flag_in,

            flag_out

        )

        for i,j in tqdm(pairs)

    )



    D=np.zeros(
        (N,N)
    )


    for i,j,d in results:

        D[i,j]=d

        D[j,i]=d



    return D



# =====================================================
# 9. graph
# =====================================================


def build_graph(D):


    sigma=np.std(D)


    A=np.exp(

        -(D**2)

        /

        (2*sigma*sigma)

    )


    np.fill_diagonal(
        A,
        0
    )


    return A



def topk(A):


    N=len(A)


    G=np.zeros_like(A)



    for i in range(N):


        idx=np.argpartition(

            A[i],

            -TOP_K

        )[-TOP_K:]


        G[i,idx]=A[i,idx]



    return np.maximum(
        G,
        G.T
    )



# =====================================================
# Main
# =====================================================


if __name__=="__main__":


    (

        weekday_in,

        weekend_in,

        weekday_out,

        weekend_out

    )=load_daytype_flow()



    # 当前只使用工作日通勤模式

    win_in=weekday_in

    win_out=weekday_out



    peak_in=[]

    peak_out=[]



    for i in tqdm(range(len(win_in))):


        peak_in.append(

            detect_commute_peak(
                win_in[i]
            )

        )


        peak_out.append(

            detect_commute_peak(
                win_out[i]
            )

        )



    peak_in=np.array(
        peak_in
    )


    peak_out=np.array(
        peak_out
    )



    print("================")

    print(
        "Inbound ESDTW:",
        peak_in.sum()
    )


    print(
        "Outbound ESDTW:",
        peak_out.sum()
    )

    print("================")



    np.save(

        OUTPUT_DIR+
        "/weekday_peak_in.npy",

        peak_in

    )


    np.save(

        OUTPUT_DIR+
        "/weekday_peak_out.npy",

        peak_out

    )



    D=build_distance(

        win_in,

        win_out,

        peak_in,

        peak_out

    )



    np.save(

        OUTPUT_DIR+
        "/daytype_direction_distance.npy",

        D

    )



    A=topk(

        build_graph(D)

    )



    np.save(

        OUTPUT_DIR+
        "/daytype_direction_esdtw_adj.npy",

        A

    )



    print(
        "Graph finished"
    )