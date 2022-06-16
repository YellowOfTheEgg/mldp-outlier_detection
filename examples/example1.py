


from visualizations.seaborn import Plotter
import pandas as pd
from outlier_detection.outlier_detector import OutlierDetector
from definitions import ROOT_DIR



def get_data():
    test_data = [
        [1, 1, 1, 1 / 3, 1 / 6],
        [2, 1, 1, 2 / 3, 1 / 6],
        [3, 1, 1, 1 / 3, 2 / 6],
        [4, 1, 2, 2 / 3, 4 / 6],
        [5, 1, 2, 3 / 3, 4 / 6],
        [6, 1, 2, 2 / 3, 5 / 6],
        [7, 1, 7, 0.5, 0.5],
        [1, 2, 3, 2 / 3, 1 / 6],
        [2, 2, 3, 3 / 3, 1 / 6],
        [3, 2, 3, 2 / 3, 2 / 6],
        [4, 2, 4, 2 / 3, 5 / 6],
        [5, 2, 4, 3 / 3, 5 / 6],
        [6, 2, 4, 2 / 3, 6 / 6],
        [7, 2, 7, 0.5, 0.5],
        [1, 3, 5, 2 / 3, 1 / 6],
        [2, 3, 5, 2 / 3, 2 / 6],
        [3, 3, 5, 1 / 3, 1 / 6],
        [4, 3, 6, 2 / 3, 5 / 6],
        [5, 3, 6, 3 / 3, 4 / 6],
        [6, 3, 6, 1 / 3, 6 / 6],
        [7, 3, 7, 0.5, 0.5],
        [8, 1, -1, 0.99, 0.99],
        [8,2,-1,0.99,0.99],
        [8,3,-1,0.99,0.99],
        [9,1,-1,0.01,0.01],
        [9,2,-1,0.01,0.01],
        [9,3,-1,0.01,0.01],
    ]

    data = pd.DataFrame(
        test_data, columns=["object_id", "time", "cluster_id", "feature1", "feature2"]
    )
    return data




if __name__ == "__main__":
    clustering = get_data()   
    outlier_detector = OutlierDetector()
    outlier_df = outlier_detector.detect_outliers(clustering,sigma=1)
    outlier_df['outlier'] = outlier_df['outlier'].apply(lambda x: 1 if x is True  else 0)
    outlier_df['preoutlier']= outlier_df['preoutlier'].apply(lambda x: 1 if x is True  else 0)
  
  
    plotter=Plotter(df=outlier_df)
    fig=plotter.generate_fig()
    fig.savefig('example1.png')
  