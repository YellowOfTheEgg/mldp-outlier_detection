from clustering.dbscan import DBScan
from ots_eval.outlier_detection.doots import DOOTS
from visualizations.seaborn.twoD.Plotter import Plotter
from definitions import ROOT_DIR
import pandas as pd

SETTINGS = {
    "Name": "generated",
    "filename": "generated_data_outlier.csv",
    "identifier": "object_id",
    "features": ["feature1", "feature2"],
    "feature_renames": ["feature1", "feature2"],
    "time_column_name": "time",
    "no_timepoints": 8,
    "normalization": "no",
    "minpts": 4,
    "eps": 0.11,
    "tau": 0.5,
    "plot": {
        "time_axis_name": "year",
        "col_wrap": 4,
        "no_timepoints": 8,
        "title": "Generated Data 2D",
        "1DPlot": False,
    },
}


def get_data():
    folder_path = f"{ROOT_DIR}/data/"
    csv_name = "generated_data_outlier.csv"
    df = pd.read_csv(folder_path + csv_name)
    return df


def cluster_data_dbscan(data):
    dbscan = DBScan(data, SETTINGS["minpts"], SETTINGS["eps"], SETTINGS)
    clustered_data = dbscan.create_clusters()
    return clustered_data


if __name__ == "__main__":
    data = get_data()
    data["object_id"] = data["object_id"].map(str)
    data.rename(columns={"object_id": "ObjectID", "time": "Time"}, inplace=True)

    clustered_data = cluster_data_dbscan(data)

    clustered_data = (
        clustered_data.reset_index()
        .rename(
            columns={"ObjectID": "object_id", "Time": "time", "cluster": "cluster_id"}
        )
        .drop(columns=["idx"])
    )

    prepared_data = clustered_data.drop(columns=["feature1", "feature2"])

    outlier_detector = DOOTS(prepared_data, weighting=False, jaccard=False)
    outlier_result = outlier_detector.calc_outlier_degree()
    clusters, outlier_result = outlier_detector.mark_outliers(tau=0.5)

    result = clustered_data.merge(
        clusters, on=["object_id", "time", "cluster_id"], how="left"
    )
    result["outlier"] = result["outlier"].map(lambda x: 0 if x == 1 else x)

    plotter = Plotter(
        df=result,
        plot_settings=dict(col_wrap=4, bbox_to_anchor=(-1.2, -0.4)),
        outlier_method="tau",
    )
    fig = plotter.generate_fig()
    fig.savefig("results/tau.png")
