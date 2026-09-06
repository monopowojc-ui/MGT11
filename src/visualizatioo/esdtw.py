import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# =====================================================
# 参数
# =====================================================

FLOW_PATH = (
    "data/processed/flow_tensor.npy"
)


PEAK_IN_PATH = (
    "data/processed/peak_in_flag.npy"
)


PEAK_OUT_PATH = (
    "data/processed/peak_out_flag.npy"
)


POINTS_PER_DAY = 103


OUTPUT_DIR = (
    "data/processed/"
    "directional_peak_validation"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# =====================================================
# 读取进出站典型日
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



    return inbound.T, outbound.T



# =====================================================
# 峰值检测
# =====================================================


def extract_peaks(
        x,
        theta=1.0
):


    mu=np.mean(x)

    sigma=np.std(x)+1e-6


    dx=np.diff(x)


    peaks=[]



    for t in range(1,len(dx)):


        # 局部极大

        if dx[t-1]>0 and dx[t]<0:



            # 时间约束

            if not(

                (6<=t<=24)

                or

                (66<=t<=84)

            ):

                continue



            Rt=(x[t]-mu)/sigma



            if Rt>theta:


                peaks.append(

                    {

                        "time":t,

                        "value":x[t],

                        "Rt":Rt

                    }

                )



    return peaks



# =====================================================
# 绘图
# =====================================================


def plot_station(

        series,

        station_id,

        direction,

        label

):


    peaks=extract_peaks(
        series
    )


    t=np.arange(
        len(series)
    )



    plt.figure(
        figsize=(10,4)
    )



    plt.plot(

        t,

        series,

        linewidth=2,

        label="flow"

    )



    for p in peaks:


        plt.scatter(

            p["time"],

            p["value"],

            s=60

        )


        plt.text(

            p["time"],

            p["value"],

            f'Rt={p["Rt"]:.2f}',

            fontsize=8

        )



    # 早峰区域

    plt.axvspan(

        6,

        24,

        alpha=0.2

    )


    # 晚峰区域

    plt.axvspan(

        66,

        84,

        alpha=0.2

    )



    plt.title(

        f"{direction} Station {station_id} {label}"

    )


    plt.xlabel(
        "Time index (10min)"
    )


    plt.ylabel(
        "Flow"
    )


    plt.grid()



    plt.tight_layout()



    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            f"{direction}_{label}_{station_id}.png"

        ),

        dpi=300

    )


    plt.close()



# =====================================================
# 统计
# =====================================================


def analyze_direction(

        series,

        flags,

        direction

):


    print("="*60)

    print(direction)



    print(

        "Total stations:",

        len(flags)

    )


    print(

        "ESDTW stations:",

        flags.sum()

    )


    print(

        "DDTW stations:",

        len(flags)-flags.sum()

    )



    records=[]



    for i,x in enumerate(series):


        peaks=extract_peaks(x)



        records.append(

            [

                i,

                flags[i],

                len(peaks),

                max(

                    [

                        p["Rt"]

                        for p in peaks

                    ],

                    default=0

                )

            ]

        )



    df=pd.DataFrame(

        records,

        columns=[

            "station",

            "ESDTW",

            "peak_number",

            "max_Rt"

        ]

    )



    df.to_csv(

        os.path.join(

            OUTPUT_DIR,

            f"{direction}_peak_statistics.csv"

        ),

        index=False

    )



    # 随机展示

    es_ids=np.where(

        flags==1

    )[0]


    dd_ids=np.where(

        flags==0

    )[0]



    np.random.seed(42)



    if len(es_ids)>=3:


        sample_es=np.random.choice(

            es_ids,

            3,

            replace=False

        )

    else:

        sample_es=es_ids



    if len(dd_ids)>=3:


        sample_dd=np.random.choice(

            dd_ids,

            3,

            replace=False

        )

    else:

        sample_dd=dd_ids



    print(

        "ESDTW examples:",

        sample_es

    )


    print(

        "DDTW examples:",

        sample_dd

    )



    for sid in sample_es:


        plot_station(

            series[sid],

            sid,

            direction,

            "ESDTW"

        )



    for sid in sample_dd:


        plot_station(

            series[sid],

            sid,

            direction,

            "DDTW"

        )





# =====================================================
# Main
# =====================================================


if __name__=="__main__":



    inbound,outbound=load_directional_series()



    peak_in=np.load(

        PEAK_IN_PATH

    )


    peak_out=np.load(

        PEAK_OUT_PATH

    )



    analyze_direction(

        inbound,

        peak_in,

        "Inbound"

    )



    analyze_direction(

        outbound,

        peak_out,

        "Outbound"

    )



    print("="*60)

    print(

        "Directional peak validation finished!"

    )
