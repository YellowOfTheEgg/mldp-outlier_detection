import pandas as pd
import numpy as np


class OutlierDetector:    
    def restructure_df(self, data_df):
        times = data_df.time.unique()
        object_ids = data_df.object_id.unique()
        pp_data = []
        for i in range(0, len(times) - 1):
            for object_id in object_ids:
                first_qry = f"object_id=='{object_id}' & time=={times[i]}"
                second_qry = f"object_id=='{object_id}' & time=={times[i+1]}"
                first_row = data_df.query(first_qry).head()   
                second_row = data_df.query(second_qry).head()
    
                if len(first_row)>0 and len(second_row)>0:
                    row = {
                        "object_id": object_id,
                        "cluster_id_first": first_row["cluster_id"].values[0],
                        "cluster_id_next": second_row["cluster_id"].values[0],
                        "time_first": first_row["time"].values[0],
                        "time_next": second_row["time"].values[0],
                    }
                    pp_data.append(row)
            
                   # print(first_row)
        df = pd.DataFrame(pp_data)

        return df

    def add_number_of_peers(self, filtered_data):
        peers = (
            filtered_data.groupby(
                by=["cluster_id_first", "cluster_id_next", "time_first", "time_next"]
            )["object_id"]
            .count()
            .reset_index(name="count")
        )
        result = filtered_data.merge(
            peers,
            on=["cluster_id_first", "cluster_id_next", "time_first", "time_next"],
            how="left",
        )
        return result

    def merge_dfs(self, data_df, outlier):
        result = data_df.merge(
            outlier,
            left_on=["object_id", "cluster_id", "time"],
            right_on=["object_id", "cluster_id_next", "time_next"],
            how="left",
        ).drop(
            columns=["cluster_id_first", "cluster_id_next", "time_first", "time_next"]
        )
        return result

    def mark_outliers(self, merged_df, sigma):
        def mark_row(row, sigma):
            if row["count"] <= sigma:
                return True
            else:
                return False

        merged_df["outlier"] = merged_df.apply(lambda row: mark_row(row, sigma), axis=1)
        return merged_df

    def mark_preoutliers(self,marked_outliers):
        def get_preoutlier(x):
            preoutlier=x
            preoutlier['time']=x['time']-1
            return preoutlier
        outliers = marked_outliers.query('outlier==True').drop(['feature1','feature2','cluster_id','count','outlier'],axis=1,errors='ignore')
     
        preoutliers=outliers.apply(lambda x: get_preoutlier(x),axis=1)        
        preoutliers['preoutlier']=True        
     
        marked_preoutliers=marked_outliers.merge(preoutliers, on=['object_id','time'],how='left')
        marked_preoutliers['preoutlier']=[False if preout is np.nan else True for preout in marked_preoutliers['preoutlier']]        
        return marked_preoutliers

    def set_cluster_outliers_as_own_clusters(self, data_df):
        cluster_id=-1
        cluster_outliers=data_df.query("cluster_id==-1")
        for index, row in cluster_outliers.iterrows():          
            data_df.at[index, 'cluster_id']=cluster_id
            cluster_id-=1       
        return data_df

    # set cluster_outliers_as_own_clusters to false if outliers found through clustering per timestamp should not be own clusters
    def detect_outliers(self, data_df, sigma=1, cluster_outliers_as_own_clusters=True):
        if cluster_outliers_as_own_clusters:
            data_df=self.set_cluster_outliers_as_own_clusters(data_df)    
        data_df['object_id'] = data_df['object_id'].map(str)
        restructured_df = self.restructure_df(data_df)
     
        number_of_peers = self.add_number_of_peers(restructured_df)
        merged_df = self.merge_dfs(data_df, number_of_peers)
        marked_outliers = self.mark_outliers(merged_df, sigma)
        fully_marked=self.mark_preoutliers(marked_outliers)
        return fully_marked
