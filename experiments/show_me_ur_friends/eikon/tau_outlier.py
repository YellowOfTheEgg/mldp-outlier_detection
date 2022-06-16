from experiments.mongodb_provider import MongoDB
from clustering.dbscan import DBScan
from ots_eval.outlier_detection.doots import DOOTS

# from dact import DACT
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

    prepared_data = clustered_data.drop(columns=["feature1", "feature2"])

    outlier_detector = DOOTS(prepared_data, weighting=False, jaccard=False)

    outlier_result = outlier_detector.calc_outlier_degree()
    clusters, outlier_result = outlier_detector.mark_outliers(tau=0.6)

    result = clustered_data.merge(
        clusters, on=["object_id", "time", "cluster_id"], how="left"
    )
    result["outlier"] = result["outlier"].map(lambda x: 0 if x == 1 else x)
    result = result.rename(
        columns={"time": "year", "feature1": "ExpectedReturn", "feature2": "NetSales"}
    )

    # res = result.groupby(["outlier"]).count()

    df_mapping = dict(
        time_col="year",
        object_id_col="object_id",
        f1_col="ExpectedReturn",
        f2_col="NetSales",
        group_col="cluster_id",
        outlier_col="outlier",
    )
    plotter = Plotter(df=result, df_mapping=df_mapping, outlier_method="tau")
    fig = plotter.generate_fig()
    fig.savefig("results/tau.png")
