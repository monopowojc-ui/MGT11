# ==========================================================
# Check Adaptive ESDTW-DDTW Peak Detection
#
# Validate:
#
# double peak station  -> ESDTW
# other station        -> DDTW
#
# Input:
# station_features.pkl
#
# Output:
# 1. station classification csv
# 2. example plots
#
# ==========================================================


import os
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



# ==========================================================
# Path
# ==========================================================


FEATURE_PATH = (
    "data/processed/station_features.pkl"
)


OUTPUT_DIR = (
    "data/processed/peak_detection_check"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ==========================================================
# Load feature
# ==========================================================


def load_features(path):


    with open(
        path,
        "rb"
    ) as f:

        features=pickle.load(f)


    return features



# ==========================================================
# Statistics
# ==========================================================


def statistics(features):


    total=len(features)


    esdtw_num=sum(
        [
            f["peak"]
            for f in features
        ]
    )


    ddtw_num=total-esdtw_num



    print("="*50)

    print(
        "Total stations:",
        total
    )


    print(
        "ESDTW stations:",
        esdtw_num
    )


    print(
        "DDTW stations:",
        ddtw_num
    )


    print(
        "ESDTW ratio:",
        esdtw_num/total
    )

    print("="*50)



# ==========================================================
# Save classification
# ==========================================================


def save_station_type(features):


    rows=[]


    for i,f in enumerate(features):


        rows.append(
            [
                i,
                int(f["peak"])
            ]
        )



    df=pd.DataFrame(

        rows,

        columns=[
            "station_id",
            "ESDTW"
        ]

    )


    path=os.path.join(
        OUTPUT_DIR,
        "esdtw_station_type.csv"
    )


    df.to_csv(
        path,
        index=False
    )


    print(
        "Saved:",
        path
    )



# ==========================================================
# Plot station
# ==========================================================


def plot_station(
        feature,
        station_id,
        label):


    series=feature["series"]

    derivative=feature["derivative"]



    # 时间索引

    t=np.arange(
        len(series)
    )



    plt.figure(
        figsize=(12,5)
    )


    plt.plot(
        t,
        series,
        marker="o",
        linewidth=1.5
    )



    # 早晚峰区域

    plt.axvspan(
        6,
        30,
        alpha=0.2
    )


    plt.axvspan(
        60,
        90,
        alpha=0.2
    )



    # 最大早峰

    morning_idx=np.argmax(
        series[6:31]
    )+6


    evening_idx=np.argmax(
        series[60:91]
    )+60



    plt.scatter(
        [
            morning_idx,
            evening_idx
        ],
        [
            series[morning_idx],
            series[evening_idx]
        ]
    )



    plt.text(
        morning_idx,
        series[morning_idx],
        "Morning"
    )


    plt.text(
        evening_idx,
        series[evening_idx],
        "Evening"
    )



    plt.title(
        f"Station {station_id} - {label}"
    )


    plt.xlabel(
        "Time index (10min)"
    )


    plt.ylabel(
        "Passenger flow"
    )


    plt.grid()



    plt.tight_layout()



    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            f"station_{station_id}_{label}.png"
        ),

        dpi=300

    )


    plt.close()



    # ----------------------
    # derivative
    # ----------------------


    plt.figure(
        figsize=(12,4)
    )


    plt.plot(
        derivative
    )


    plt.axhline(
        0,
        linestyle="--"
    )


    plt.title(
        f"Station {station_id} derivative"
    )


    plt.xlabel(
        "Time index"
    )


    plt.ylabel(
        "Derivative"
    )


    plt.grid()


    plt.tight_layout()


    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            f"station_{station_id}_{label}_derivative.png"
        ),

        dpi=300

    )


    plt.close()



# ==========================================================
# Find examples
# ==========================================================


def find_examples(features):


    esdtw_station=None

    ddtw_station=None



    for i,f in enumerate(features):


        if f["peak"] and esdtw_station is None:

            esdtw_station=i



        if (not f["peak"]) and ddtw_station is None:

            ddtw_station=i



        if (
            esdtw_station is not None
            and
            ddtw_station is not None
        ):

            break



    return (
        esdtw_station,
        ddtw_station
    )



# ==========================================================
# Main
# ==========================================================


if __name__=="__main__":


    features=load_features(
        FEATURE_PATH
    )



    statistics(
        features
    )



    save_station_type(
        features
    )



    esdtw_station,ddtw_station=find_examples(
        features
    )



    print(
        "Example ESDTW station:",
        esdtw_station
    )


    print(
        "Example DDTW station:",
        ddtw_station
    )



    if esdtw_station is not None:


        plot_station(

            features[esdtw_station],

            esdtw_station,

            "ESDTW"

        )



    if ddtw_station is not None:


        plot_station(

            features[ddtw_station],

            ddtw_station,

            "DDTW"

        )



    print(
        "Peak detection check finished!"
    )