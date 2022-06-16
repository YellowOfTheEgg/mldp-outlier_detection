import seaborn as sns
from .PointStyle import PointStyle

class Subplot:
    def __init__(
        self,
        outlier_method='sigma',
        df_mapping=dict(
            time_col="time",
            object_id_col="object_id",
            f1_col="feature1",
            f2_col="feature2",
            group_col="cluster_id",
            outlier_col="outlier"
        ),
    ):
        self.f1_col = df_mapping["f1_col"]
        self.f2_col = df_mapping["f2_col"]
        self.group_col = df_mapping["group_col"]
        self.object_id_col = df_mapping["object_id_col"]
        self.outlier_col="outlier"
        self.preoutlier_col="preoutlier"
        self.outlier_method=outlier_method

    def get_style_ts_patch(self, color):
        style_ts_patch = dict(
            fontsize=10,
            xytext=(-2, -1.5),
            textcoords="offset points",
            bbox=dict(boxstyle="square", alpha=0.1, color=color),
            va="top",
            ha="right",
            alpha=0.4,
        )
        return style_ts_patch




    def _get_sub_plot(self, x, y, z, outlier,preoutlier=None, **kwargs):
        point_style=PointStyle()        
        ax = sns.scatterplot(
            x=x, y=y, **kwargs, marker="s", s=10
        )  # s=0 to hide scatter points
        number_of_points = len(x)       
        for i in range(number_of_points):            
            if outlier.values[i]==1 or outlier.values[i]==-1 :
                point_style.set_style(ax,i,x,y,z,kwargs,style_type='outlier')
             

            elif (preoutlier is not None and preoutlier.values[i]==1) or outlier.values[i]==-2 or outlier.values[i]==-3:
                point_style.set_style(ax,i,x,y,z,kwargs,style_type='preoutlier')

            elif outlier.values[i]==0:
               point_style.set_style(ax,i,x,y,z,kwargs,style_type='default')
        return

    def addSubplots(self, g):      
        if self.outlier_method=='sigma':
            g.map(
                self._get_sub_plot,
                self.f1_col,
                self.f2_col,
                self.object_id_col,
                self.outlier_col,
                self.preoutlier_col
            )
        elif self.outlier_method=='tau':
            g.map(
                self._get_sub_plot,
                self.f1_col,
                self.f2_col,
                self.object_id_col,
                self.outlier_col,             
            )
        return g
