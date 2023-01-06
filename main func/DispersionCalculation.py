import numpy as np
import pandas as pd


class Cell_Dispersion_GPU:

    def __init__(self):

        import tensorflow as tf
    
    def squared_dist(self, X):
        """
        :param X: PC coordinates for cells, shape = n_cells * n_PCs
        """
        expanded_a = tf.expand_dims(X, 1)
        expanded_b = tf.expand_dims(X, 0)
        distances = tf.reduce_sum(tf.math.squared_difference(expanded_a, expanded_b), 2)
        return distances

    def bootstrap_avg_dist(self, pc_df, n_size, target_cells, n_times):
        """
        :param pc_df: Dataframe contains samples, cell types, disease states, PC coordinates.
                      The form should be: cells * (sample, cell type, disease state, PCs...)
        :param n_size: the size for each sampling, integer
        :param target_cells: Only perform calculation on wanted cell types, string list
        :param n_times: perform random sampling for n times
        """
        self.DIST = {}
        # Rename columns for avoiding errors
        pc_df.columns = ['sample','cell_type','disease_state']+[f'PC{i+1}' for i in range(pc_df.shape[1]-3)]

        for s in pc_df['sample'].unique():

            CT_AVG_DIST = {}

            if target_cells == None:
                target_cells = list(pc_df['cell_type'].unique())

            for ct in target_cells:
                mask_s = pc_df['sample']==s
                mask_c = pc_df['cell_type']==c
                sub_df = pc_df[mask_s & mask_c].iloc[:,3:]

                # If the number of cells is smaller than n_size, cell dispersion = NA
                if len(sub_df) > n_size:

                    avg_dist_ls = []
                    for _ in range(n_times):
                        sampled = sub_df.sample(n=n_size)
                        dist_mtx = self.squared_dist(sampled)
                        dist_mtx = np.log(tf.linalg.set_diag(dist_mtx,np.ones(n_size))).reshape(-1)
                        avg_dist_ls.append(np.sum(dist_mtx)/(n_size**2-n_size))
                    
                    AVG_DIST = np.mean(avg_dist_ls)
                    CT_AVG_DIST.setdefault(ct, [AVG_DIST])

                else:
                    CT_AVG_DIST.setdefault(ct, [np.nan])
            self.DIST.setdefault(s, CT_AVG_DIST)
        

    def to_df(self):
        
        out_df = pd.DataFrame()
        for sample, result in self.DIST.items():
            result = pd.DataFrame(result, index=[sample])
            out_df = pd.concat([out_df, result])
        return out_df


    
            
















