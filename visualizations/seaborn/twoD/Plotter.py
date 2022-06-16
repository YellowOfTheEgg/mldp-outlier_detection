from visualizations.seaborn.twoD.Stage import Stage
from visualizations.seaborn.twoD.Subplot import Subplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


class Plotter:
    def __init__(
        self,
        df,
        df_mapping=dict(
            time_col="time",
            object_id_col="object_id",
            f1_col="feature1",
            f2_col="feature2",
            group_col="cluster_id",
            outlier_col="outlier"
        ),
        plot_settings=dict(col_wrap=3, bbox_to_anchor=(-1.2, -0.4)),
        outlier_method='sigma'
    ):
        self.df = df
        self.df_mapping = df_mapping
        self.plot_settings = plot_settings
        self.outlier_method=outlier_method
        mpl.rc('font',family='serif',serif='CMU Serif')

    def generate_fig(self):      
        stage = Stage(df=self.df, df_mapping=self.df_mapping, plot_settings=self.plot_settings)
        subplot = Subplot(df_mapping=self.df_mapping,outlier_method=self.outlier_method)
        g = stage.getStage()       
        g = subplot.addSubplots(g)       
        return g


