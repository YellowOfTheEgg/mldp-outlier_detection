from experiments.mongodb_provider import MongoDB
from clustering.dbscan import DBScan
from outlier_detection.outlier_detector import OutlierDetector
from visualizations.seaborn.twoD.Plotter import Plotter

SETTINGS = {
    "Name": "financial",
    "features": ["TR-TtlPlanExpectedReturn", "TR-NetSales"],
    "feature_renames": ["ExpectedReturn", "NetSales"],
    "normalization_feature": "TR-TotalAssetsReported",
    "no_companies": 50,
    "start_year": 2008,
    "end_year": 2013,
    "minpts": 2,
    "eps": 0.15,
    "tau": 0.6,
    "plot": {
        "time_axis_name": "year",
        "col_wrap": 3,
        "no_timepoints": 6,
        "title": "Financial Data",
        "1DPlot": True,
    },
    "fill_missing_values": False,
}


def get_data():
    db = MongoDB(SETTINGS)
    data = db.get_financial_data(True)
    return data


def cluster_data_dbscan(data):
    dbscan = DBScan(data, SETTINGS["minpts"], SETTINGS["eps"], SETTINGS)
    clustered_data = dbscan.create_clusters()
    return clustered_data


if __name__ == "__main__":
    data = get_data()

    clustered_data = cluster_data_dbscan(data)
    clustered_data = clustered_data.reset_index()

    clustered_data = clustered_data.rename(
        columns={
            "ObjectID": "object_id",
            "Time": "time",
            "cluster": "cluster_id",
            "ExpectedReturn": "feature1",
            "NetSales": "feature2",
        }
    ).drop(columns=["idx"])

    outlierDetector = OutlierDetector()
    result = outlierDetector.detect_outliers(clustered_data, sigma=1)
    result = result.rename(
        columns={"feature1": "ExpectedReturn", "feature2": "NetSales", "time": "year"}
    )

    df_mapping = df_mapping = dict(
        time_col="year",
        object_id_col="object_id",
        f1_col="ExpectedReturn",
        f2_col="NetSales",
        group_col="cluster_id",
        outlier_col="outlier",
    )
    plotter = Plotter(df=result, df_mapping=df_mapping, outlier_method="sigma")
    fig = plotter.generate_fig()
    fig.savefig("results/sigma.png")
