import seaborn as sns


class LinePlot:
    def __init__(
        self,
        stage,
        df,
        outlier_method,
        df_mapping=dict(
            time_col="time",
            object_id_col="object_id",
            f1_col="feature1",
            group_col="cluster_id",
            outlier_col="outlier",
        ),
    ):
        self.df = df
        self.df_mapping = df_mapping
        self.outlier_method = outlier_method
        self.stage = stage
        self.color_counter = 0

    def get_subseq(self, time_interval, ts):
        time_a = time_interval[0]
        time_b = time_interval[1]
        time_col = self.df_mapping["time_col"]
        subseq = ts.loc[(ts[time_col] == time_a) | (ts[time_col] == time_b)]
        return subseq

    def get_type_of_subseq(self, subseq):
        time_col = self.df_mapping["time_col"]
        outlier_col = self.df_mapping["outlier_col"]
        ordered_subseq = subseq.sort_values(by=[time_col])
        type_a = ordered_subseq[outlier_col].values[0]
        type_b = ordered_subseq[outlier_col].values[1]
        type_of_subseq = None
        if type_a == -1 and type_b != 0:
            type_of_subseq = -1
        elif type_a == 0 or type_b == 0:
            type_of_subseq = 0
        elif type_a == -2 and type_b == -2:
            type_of_subseq = -2
        elif type_a == -3 and type_b == -3:
            type_of_subseq = -3
        elif type_a == -3 and type_b == -1:
            type_of_subseq = -1
        return type_of_subseq

    def get_time_intervals(self, ts):
        time_col = self.df_mapping["time_col"]
        times = ts[time_col].unique()
        time_intervals = []
        for i in range(0, len(times)):
            if i == 0:
                tpl = (times[i], times[i + 1])
                time_intervals.append(tpl)
            elif i > 1:
                tpl = (times[i - 1], times[i])
                time_intervals.append(tpl)
        return time_intervals

    def get_color(self):
        object_id_col = self.df_mapping["object_id_col"]
        color_palette = sns.color_palette(
            "muted", n_colors=len(self.df[object_id_col].unique())
        )
        color_id = self.color_counter % len(color_palette)
        self.color_counter += 1
        color = color_palette[color_id]
        return color

    def add_line_plot(self, subseq, type_of_subseq, color):
        linewidth = 0.8
        time_col = self.df_mapping["time_col"]
        feature_col = self.df_mapping["f1_col"]
        object_id_col = self.df_mapping["object_id_col"]
        if type_of_subseq == 0:
            sns.lineplot(
                x=time_col,
                y=feature_col,
                hue=object_id_col,
                data=subseq,
                palette=[color],
                legend=False,
                linewidth=linewidth,
                ax=self.stage,
            )
        elif type_of_subseq == -1:
            sns.lineplot(
                x=time_col,
                y=feature_col,
                hue=object_id_col,
                data=subseq,
                palette=["black"],
                legend=False,
                linewidth=linewidth,
                ax=self.stage,
            )
        elif type_of_subseq == -2:
            sns.lineplot(
                x=time_col,
                y=feature_col,
                hue=object_id_col,
                data=subseq,
                palette=[color],
                legend=False,
                linestyle="-.",
                linewidth=linewidth,
                ax=self.stage,
            )
        elif type_of_subseq == -3:
            sns.lineplot(
                x=time_col,
                y=feature_col,
                hue=object_id_col,
                data=subseq,
                palette=["black"],
                legend=False,
                linestyle="-.",
                linewidth=linewidth,
                ax=self.stage,
            )

    def get_type_of_subseq_sigma(self, subseq):
        time_col = self.df_mapping["time_col"]
        outlier_col = self.df_mapping["outlier_col"]
        ordered_subseq = subseq.sort_values(by=[time_col])
        outlier_b = ordered_subseq[outlier_col].values[1]
        type_of_subseq = None
        if outlier_b:
            type_of_subseq = -1
        else:
            type_of_subseq = 0
        return type_of_subseq

    def add_line_plot_sigma(self, subseq, type_of_subseq, color):
        linewidth = 0.8
        time_col = self.df_mapping["time_col"]
        feature_col = self.df_mapping["f1_col"]
        object_id_col = self.df_mapping["object_id_col"]
        if type_of_subseq == 0:
            sns.lineplot(
                x=time_col,
                y=feature_col,
                hue=object_id_col,
                data=subseq,
                palette=[color],
                legend=False,
                linewidth=linewidth,
                ax=self.stage,
            )
        elif type_of_subseq == -1:
            sns.lineplot(
                x=time_col,
                y=feature_col,
                hue=object_id_col,
                data=subseq,
                palette=["black"],
                legend=False,
                linewidth=linewidth,
                ax=self.stage,
            )

    def addLinePlots(self):
        object_id_col = self.df_mapping["object_id_col"]
      #  self.df=self.df.loc[self.df['object_id']=='24']
       
        object_ids = self.df[object_id_col].unique()
        for object_id in object_ids:
            color = self.get_color()
            ts = self.df.loc[self.df[object_id_col] == object_id]
            time_intervals = self.get_time_intervals(ts)
            for time_interval in time_intervals:
                subseq = self.get_subseq(time_interval, ts)
                if self.outlier_method == "tau":
                    type_of_subseq = self.get_type_of_subseq(subseq)
                    self.add_line_plot(subseq, type_of_subseq, color)
                elif self.outlier_method == "sigma":
                    type_of_subseq = self.get_type_of_subseq_sigma(subseq)
                    self.add_line_plot_sigma(subseq, type_of_subseq, color)
