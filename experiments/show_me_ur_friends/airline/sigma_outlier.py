from definitions import ROOT_DIR
import pandas as pd
from clustering.dbscan import DBScan
from ots_eval.outlier_detection.doots import DOOTS
from visualizations.seaborn.oneD.Plotter import Plotter
from sklearn import preprocessing
from outlier_detection.outlier_detector import OutlierDetector
import numpy as np

SETTINGS = {
    "Name": "flights",
    "filename": f"{ROOT_DIR}/data/DelayedFlights.csv",
    "identifier": "UniqueCarrier",
    "features": ["Distance"],
    "feature_renames": ["Distance"],
    #'features':['feature1'],
    #'feature_renames':['feature1'],
    "time_column_name": "DayofMonth",
    "no_timepoints": 8,
    "normalization": "yes",
    "min_pref": 0.45,
    "sw": 4,
    "minpts": 3,
    "eps": 0.03,
    "tau": 0.4,
    "plot": {
        "time_axis_name": "time stamp",
        "col_wrap": 4,
        "no_timepoints": 8,
        "title": "Flights Data",
        "1DPlot": True,
    },
}


def normalize(data, feature):
    f1 = data[[feature]].values.astype(float)

    min_max_scaler_dep = preprocessing.MinMaxScaler()
    f1_scaled = min_max_scaler_dep.fit_transform(f1)

    data[feature] = f1_scaled
    return data


def get_data():
    o_data = pd.read_csv(SETTINGS["filename"])

    o_data["DayofMonth_prefix"] = o_data[SETTINGS["time_column_name"]].map(str)
    temp = o_data["DayofMonth_prefix"].loc[o_data.DayofMonth < 10]
    temp = "0" + temp.map(str)
    o_data.loc[o_data.DayofMonth < 10, "DayofMonth_prefix"] = temp

    o_data["Month_prefix"] = o_data["Month"].map(str)
    temp = o_data["Month_prefix"].loc[o_data.Month < 10]
    temp = "0" + temp.map(str)
    o_data.loc[o_data.Month < 10, "Month_prefix"] = temp

    o_data["Datetime"] = (
        o_data["Year"].map(str)
        + "-"
        + o_data["Month_prefix"].map(str)
        + "-"
        + o_data["DayofMonth_prefix"].map(str)
    )
    o_data["Epoch"] = pd.to_datetime(o_data["Datetime"])
    o_data["Epoch"] = o_data["Epoch"].values.astype(np.int64) // 10 ** 6

    if len(SETTINGS["features"]) > 1:
        t_data = pd.concat(
            [
                o_data[SETTINGS["identifier"]],
                o_data["Epoch"],
                o_data[SETTINGS["features"][0]],
                o_data[SETTINGS["features"][1]],
            ],
            axis=1,
            keys=[
                "ObjectID",
                "Time",
                SETTINGS["feature_renames"][0],
                SETTINGS["feature_renames"][1],
            ],
        )
    else:
        t_data = pd.concat(
            [
                o_data[SETTINGS["identifier"]],
                o_data["Epoch"],
                o_data[SETTINGS["features"][0]],
            ],
            axis=1,
            keys=["ObjectID", "Time", SETTINGS["feature_renames"][0]],
        )

    t_data = t_data.groupby(["ObjectID", "Time"]).mean().reset_index()

    if SETTINGS["normalization"] == "yes":
        for feature in SETTINGS["feature_renames"]:
            t_data = normalize(t_data, feature)

    time_points = t_data["Time"].unique().tolist()
    time_points.sort()
    time_points = time_points[: SETTINGS["no_timepoints"]]

    t_data = t_data[t_data["Time"] <= time_points[len(time_points) - 1]]
    times = t_data["Time"].unique()
    t_data["Time"] = t_data["Time"].map(lambda t: np.where(times == t)[0][0] + 1)

    return t_data


def get_data_s():
    folder_path = f"{ROOT_DIR}/data/"
    csv_name = "DelayedFlights.csv"
    df = pd.read_csv(folder_path + csv_name)
    df = df[["UniqueCarrier", "DayofMonth", "Distance"]]
    df = df.rename(
        columns={
            "UniqueCarrier": "object_id",
            "DayofMonth": "time",
            "Distance": "feature1",
        }
    )

    df = df.groupby(by=["object_id", "time"]).mean().reset_index()
    df["feature1"] = (df["feature1"] - df["feature1"].min()) / (
        df["feature1"].max() - df["feature1"].min()
    )
    return df


def cluster_data_dbscan(data):
    dbscan = DBScan(data, SETTINGS["minpts"], SETTINGS["eps"], SETTINGS)
    clustered_data = dbscan.create_clusters()
    return clustered_data


if __name__ == "__main__":
    data = get_data()

    clustered_data = cluster_data_dbscan(data)

    clustered_data = (
        clustered_data.reset_index()
        .rename(
            columns={"ObjectID": "object_id", "Time": "time", "cluster": "cluster_id"}
        )
        .drop(columns=["idx"])
    )

    clustered_data = clustered_data.rename(columns={"Distance": "feature1"})

    outlierDetector = OutlierDetector()
    result = outlierDetector.detect_outliers(clustered_data, sigma=1)
    result = result.rename(columns={"feature1": "Distance", "time": "Time"})
    df_mapping = dict(
        time_col="Time",
        object_id_col="object_id",
        f1_col="Distance",
        group_col="cluster_id",
        outlier_col="outlier",
    )
    plotter = Plotter(df=result, df_mapping=df_mapping, outlier_method="sigma")
    fig = plotter.generate_fig()
    fig.savefig("results/sigma.png")
