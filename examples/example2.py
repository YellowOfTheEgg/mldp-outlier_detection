from ots_eval.clustering.cots import COTS


from visualizations.seaborn import Plotter
import pandas as pd
from outlier_detection.outlier_detector import OutlierDetector
from definitions import ROOT_DIR



def get_data():
    folder_path = f"{ROOT_DIR}/data/"
    csv_name = "pub1_generated_datasetB.csv"
    df = pd.read_csv(folder_path + csv_name)
    return df


def get_clustering(df):
    cots = COTS(df)
    clusters = cots.get_clusters_df(min_cf=0.2, sw=3)
    return clusters


if __name__ == "__main__":
    data = get_data()
    clustering = get_clustering(data)
    clustering.rename(columns={"cluster": "cluster_id"}, inplace=True)   
    outlier_detector = OutlierDetector()
    outlier_df = outlier_detector.detect_outliers(clustering,sigma=1)
    outlier_df['outlier'] = outlier_df['outlier'].apply(lambda x: 1 if x is True  else 0)
    outlier_df['preoutlier']= outlier_df['preoutlier'].apply(lambda x: 1 if x is True  else 0)
  
  
    plotter=Plotter(df=outlier_df)
    fig=plotter.generate_fig()
    fig.savefig('exampleOutlier.png')
  