from experiments.mongodb_provider import MongoDB
from clustering.dbscan import DBScan
from outlier_detection.outlier_detector import OutlierDetector

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

    data.rename(columns={"object_id": "ObjectID", "time": "Time"}, inplace=True)

    clustered_data = cluster_data_dbscan(data)
    clustered_data = clustered_data.reset_index()

    clustered_data = clustered_data.rename(
        columns={"ObjectID": "object_id", "Time": "time", "cluster": "cluster_id"}
    ).drop(columns=["idx"])
    outlierDetector = OutlierDetector()
    result = outlierDetector.detect_outliers(clustered_data, sigma=1)
    result.rename(columns={"time": "year"}, inplace=True)

    df_mapping = df_mapping = dict(
        time_col="year",
        object_id_col="object_id",
        f1_col="feature1",
        f2_col="feature2",
        group_col="cluster_id",
        outlier_col="outlier",
    )
    plotter = Plotter(
        df=result,
        df_mapping=df_mapping,
        plot_settings=dict(col_wrap=4, bbox_to_anchor=(-1.2, -0.4)),
    )
    fig = plotter.generate_fig()
    fig.savefig("results/sigma.png")
