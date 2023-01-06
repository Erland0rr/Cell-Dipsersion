import pandas as pd
import numpy as np
import scipy as sp


class DifferentialAnalysis_GPU:

    def __init__(self, EXP, cell_type, state1, state2, n_times):
        """
        Perform differential analysis on expression profiles of a given cell type from two disease states.
        1. Create permutation distributions and ground truth statistics for each gene.
        2. Apply permutation test on the gene expression distributions.

        :param EXP: Expression profile dataframe for a cell type, shape: cells * (disease state, genes)
        :param state1: one label for disease state 
        :param state2: the other label for disease state
        """
        self.EXP = EXP
        self.ct = cell_type
        self.state1 = state1
        self.state2 = state2
        self.n_times = n_times
    
    
    def permutation_distribution(self, save_path=None):

        if save_path == None:
            save_path = '.'
        else:
            if save_path[-1]=='/':
                save_path = save_path[:-1]

        
        # Exclude disease state column
        s1_exp = self.EXP[self.EXP.iloc[:,0]==self.state1].iloc[:,1:]
        s2_exp = self.EXP[self.EXP.iloc[:,0]==self.state2].iloc[:,1:]
        
        s1_mean, s2_mean = s1_exp.mean(), s2_exp.mean()
        s1_var, s2_var = s1_exp.var(), s2_exp.var()

        # Calculate ground truth statistics for each gene
        self.TRUE = pd.DataFrame(
            {
             'MEAN':s1_mean-s2_mean,
             'VAR':s1_var-s2_var,
             'DI':(s1_var/s1_mean)-(s2_var/s2_mean)
            }
        )


        size = len(s1_exp)
        all_exp = pd.concat([s1_exp, s2_exp])

        self.MEAN, self.VAR, self.DI = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Permute n times
        for _ in range(self.n_times):

            mean, var, di = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            # 1,000 genes for each batch
            for i in range(0,len(s1_exp),1000):
                shuffled = tf.convert_to_tensor(all_exp.iloc[:,j:j+1000].sample(frac=1).T)
                group1 = shuffled[:,0:size]        # 1 ~ state1 size belongs to group1
                group2 = shuffled[:,size:]         # status1 size+1 ~ last belongs to group2
                shuffled = 0                       # Free memory
                group1_mean, group1_var = tf.reduce_mean(group1, 1), tf.math.reduce_variance(group1, 1)
                group2_mean, group2_var = tf.reduce_mean(group2, 1), tf.math.reduce_variance(group2, 1)
                group1, group2 = 0, 0              # Free memory   
                group1_di, group2_di = group1_var/group1_mean, group2_var/group2_mean

                mean = pd.concat([mean, pd.DataFrame(np.array(group2_mean-group1_mean).tolist())])
                var = pd.concat([var, pd.DataFrame(np.array(group2_var-group1_var).tolist())])
                di = pd.concat([di, pd.DataFrame(np.array(group2_di-group1_di).tolist())])

            self.MEAN = pd.concat([self.MEAN, mean.T])
            self.VAR  = pd.concat([self.VAR, var.T]) 
            self.DI   = pd.concat([self.DI, di.T])

        self.MEAN.columns, self.VAR.columns, self.DI.columns = all_exp.columns, all_exp.columns, all_exp.columns


        self.MEAN.to_csv(f'{save_path}/{self.ct}_mean_perm_distribution.csv', index=None)
        self.VAR.to_csv(f'{save_path}/{self.ct}_var_perm_distribution.csv', index=None)
        self.DI.to_csv(f'{save_path}/{self.ct}_di_perm_distribution.csv', index=None)


    
    def permutation_test(self, save_path=None, TRUE=None, cell_type=None, MEAN=None, VAR=None, DI=None,
                        state1=None, state2=None):

        if TRUE==None:
            TRUE = self.TRUE
            MEAN = self.MEAN
            VAR = self.MEAN
            DI = self.DI
            cell_type = self.ct
            state1 = self.state1
            state2 = self.state2

        if save_path == None:
            save_path = '.'
        else:
            if save_path[-1]=='/':
                save_path = save_path[:-1]


        GENES = list(MEAN.columns)
        STATS = {'Gene':[], 'TRUE/ΔMEAN':[], 'TRUE/ΔVAR':[], 'TRUE/ΔDI':[], 'MEAN_pval':[], 'VAR_pval':[], 'DI_pval':[]}

        for g in GENES:

            mean_loc, mean_scale = sp.stats.norm.fit(MEAN[g])
            var_loc, var_scale = sp.stats.norm.fit(VAR[g])
            di_loc, di_scale = sp.stats.norm.fit(DI[g].dropna())

            STATS['Gene'].append(g)
            STATS['TRUE/ΔMEAN'].append(TRUE.loc[g,'MEAN'])
            STATS['TRUE/ΔVAR'].append(TRUE.loc[g,'VAR'])
            STATS['TRUE/ΔDI'].append(TRUE.loc[g,'DI'])
            STATS['MEAN_pval'].append(sp.stats.norm.sf(abs(TRUE.loc[g,'MEAN']), scale=mean_scale, loc=mean_loc)*2)
            STATS['VAR_pval'].append(sp.stats.norm.sf(abs(TRUE.loc[g,'VAR']), scale=var_scale, loc=var_loc)*2)
            STATS['DI_pval'].append(sp.stats.norm.sf(abs(TRUE.loc[g,'DI']), scale=di_scale, loc=di_loc)*2)

        META = pd.DataFrame(
            {
            'Cell Type':[cell_type]*len(STATS['Gene']),
            'Group':[f'{state1} vs {state2}']*len(STATS['Gene'])
            }
        )
        STATS = pd.concat([META, pd.DataFrame(STATS)], axis=1)


        STATS.to_csv(f'{save_path}/{cell_type}.stats.csv', index=None)
        return STATS






