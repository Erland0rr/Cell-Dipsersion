import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import RepeatedKFold
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.impute import KNNImputer
import pickle
import os
import random
import sklearn.metrics as sk_eval
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


class SLE_MODULE_SCORE_REGRESSION_MODEL:

    def __init__(self):

        print('Initialize...')

        self.score_name = ['Up-B_score', 'Down-Pan_score', 'Up-Monocyte_score', 'Up-Pan_score', 'Up-T_score', 'Down-Monocyte_score']

        self.prop = pd.read_csv('../result/Cell Proportion/SLE.prop_df.csv',index_col=0)
        self.disp = pd.read_csv('../result/Cell Dispersion/SLE_sPC_disp.csv',index_col=0)
        self.score = pd.read_csv('SLE.score_df.csv',index_col=0)

        self.result_df = pd.DataFrame()
        
        self.params_id = 'params_0'
        self.loss_dict = {'Parameter_ID':[],'loss':[]}

        print('Complete.')
    

    def Drop_Thirteen_Cell_Types(self):

        keep_ct = self.disp.isna().sum().sort_values()[0:17].index
        self.disp = self.disp.loc[:,keep_ct]


    def KNN_Imputation(self):

        self.disp_knn = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(self.disp), index=self.disp.index, columns=self.disp.columns)
    

    def Extract_Status_Index(self):

        self.ctl_idx = [indi for indi in self.score.index if indi.split(sep=':')[1]=='Healthy']
        self.mng_idx = [indi for indi in self.score.index if indi.split(sep=':')[1]=='Managed']
        self.flr_idx = [indi for indi in self.score.index if indi.split(sep=':')[1]=='Flare']
        self.trt_idx = [indi for indi in self.score.index if indi.split(sep=':')[1]=='Treated']

    
    def Rename_Feature_Columns(self):

        def rename_cols(df,add):
            df.columns = [c+add for c in df.columns]
            return df

        self.disp_knn = rename_cols(self.disp_knn,'_disp')
        self.prop = rename_cols(self.prop,'_prop')

    
    def Generate_Training_and_Testing_List(self,seed):

        random.seed(seed)
        ctl_test, mng_test = random.sample(self.ctl_idx,k=36), random.sample(self.mng_idx,k=36)
        ctl_train = list(set(self.ctl_idx).difference(ctl_test))+random.choices(list(set(self.ctl_idx).difference(ctl_test)),k=47)
        mng_train = list(set(self.mng_idx).difference(mng_test))

        return [ctl_train, ctl_test], [mng_train, mng_test]


    def Result_df_Update(self, model_result, feature_name, comb_name, regressor_name, module_name,
                               ctl_mng_feature, ctl_mng_score, flr_trt_feature, flr_trt_score):

        out_df = {'MSE':[],'R2':[],'Feature':[],'Status':[],'Combination':[], 'Model':[], 'Module':[]}
    
        for i in range(15):
            ctl_mng_mse       = sk_eval.mean_squared_error(ctl_mng_score, model_result['estimator'][i].predict(ctl_mng_feature))
            ctl_mnh_r2        = sk_eval.r2_score(ctl_mng_score, model_result['estimator'][i].predict(ctl_mng_feature))
            flr_trt_mse       = sk_eval.mean_squared_error(flr_trt_score, model_result['estimator'][i].predict(flr_trt_feature))
            flr_trt_r2        = sk_eval.r2_score(flr_trt_score, model_result['estimator'][i].predict(flr_trt_feature))
            out_df['MSE']    += [ctl_mng_mse, flr_trt_mse]
            out_df['R2']     += [ctl_mnh_r2, flr_trt_r2]
            out_df['Status'] += ['Healthy/Managed','Flare/Treated']
        
        out_df['Feature']     += [feature_name]*30
        out_df['Combination'] += [comb_name]*30
        out_df['Model']       += [regressor_name]*30
        out_df['Module']      += [module_name]*30
        
        return pd.DataFrame(out_df)


    def Model_for_One_Fold(self, regressor, regressor_name, module_score, comb_name, seed):
        ctl, mng = self.Generate_Training_and_Testing_List(seed)
    
        disp_train,   disp_test = self.disp_knn.loc[ctl[0]+mng[0],:], self.disp_knn.loc[ctl[1]+mng[1],:]
        prop_train,   prop_test = self.prop.loc[ctl[0]+mng[0],:], self.prop.loc[ctl[1]+mng[1],:]
        score_train, score_test = self.score.loc[ctl[0]+mng[0],module_score], self.score.loc[ctl[1]+mng[1],module_score]
        
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
        
        disp_result   = cross_validate(regressor, disp_train, score_train, cv=cv, scoring=['neg_mean_squared_error','r2'], return_estimator=True)
        prop_result   = cross_validate(regressor, prop_train, score_train, cv=cv, scoring=['neg_mean_squared_error','r2'], return_estimator=True)
        concat_result = cross_validate(regressor, pd.concat([disp_train,prop_train],axis=1), score_train, cv=cv, scoring=['neg_mean_squared_error','r2'], return_estimator=True)
        
        result = [disp_result, prop_result, concat_result]
        
        feature_name = ['Dispersion','Proportion','Concat']
        ctl_mng_feature = [disp_test,prop_test,pd.concat([disp_test,prop_test],axis=1)]
        flr_trt_feature = [self.disp_knn.loc[self.flr_idx+self.trt_idx,:],
                           self.prop.loc[self.flr_idx+self.trt_idx,:],
                           pd.concat([self.disp_knn.loc[self.flr_idx+self.trt_idx,:],self.prop.loc[self.flr_idx+self.trt_idx,:]],axis=1)]

        for i in range(3):
            tmp_df = self.Result_df_Update(result[i], feature_name[i], comb_name, regressor_name, module_score,
                                           ctl_mng_feature[i], score_test, flr_trt_feature[i], self.score.loc[self.flr_idx+self.trt_idx,module_score])
            self.result_df = pd.concat([self.result_df,tmp_df])

    
    def Objective(self, space):
        
        print(self.params_id)

        if space['regressor']=='XGB/Regressor':
            regressor = xgb.XGBRegressor()
        elif space['regressor']=='XGB/Regressor':
            regressor = xgb.XGBRFRegressor()

        regressor.n_estimators     = int(space['n_estimators'])
        regressor.max_depth        = int(space['max_depth'])
        regressor.eta              = space['eta']
        regressor.subsample        = space['subsample']
        regressor.colsample_bytree = space['colsample_bytree']
        regressor.reg_alpha        = space['reg_alpha']
        regressor.gamma            = space['gamma']
        regressor.reg_lambda       = space['reg_lambda']
        regressor.min_child_weight = space['min_child_weight']
        
        for i in range(5):
            seed = random.randint(0,1000)
            comb_name = 'Combination {0}'.format(i+1)
            for module_score in self.score_name:
                print(comb_name, module_score, '......')
                self.Model_for_One_Fold(regressor,space['regressor'],module_score,comb_name,seed) # self.result_df shape (90,7), fifteen models for six module scores
            # self.result_df shape (540,7)
        # self.result_df shape (2700,7)

        objective_df = self.result_df.iloc[-2700:,:]
        loss = objective_df['MSE'].sum()-objective_df['R2'].sum()

        print(self.params_id, ' loss:', loss)
        self.loss_dict['Parameter_ID'].append(self.params_id)
        self.loss_dict['loss'].append(loss)

        self.params_id = 'params_'+str(int(self.params_id.split(sep='_')[1])+1)
        
        return {'loss':loss, 'status':STATUS_OK}




def main():

    TASK = SLE_MODULE_SCORE_REGRESSION_MODEL()
    TASK.Drop_Thirteen_Cell_Types()
    TASK.KNN_Imputation()
    TASK.Extract_Status_Index()
    TASK.Rename_Feature_Columns()
    
    path2models = '../result/ScoreModel_params_search_score'

    space={'max_depth': hp.quniform("max_depth", 3, 15, 1),
           'gamma': hp.uniform ('gamma', 0,2),
           'reg_alpha' : hp.uniform('reg_alpha', 0,5),
           'reg_lambda' : hp.uniform('reg_lambda', 0,1),
           'colsample_bytree' : hp.uniform('colsample_bytree', 0.2,1),
           'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
           'n_estimators': hp.quniform('n_estimators',1200,3000,1),
           'eta': hp.uniform('eta',0,0.5),
           'subsample' : hp.uniform('subsample',0,0.8),
           'path2models' : path2models,
           'regressor': 'XGB/Regressor'}
    
    trials = Trials()

    best_hyperparams = fmin(fn = TASK.Objective, space = space, algo = tpe.suggest, max_evals = 1200, trials = trials)
    print('Best parameters:',best_hyperparams)

    
    pd.DataFrame(TASK.result_df).to_csv(path2models+'/model_result.csv', index=None)
    pd.DataFrame(TASK.loss_dict).to_csv(path2models+'/params_loss.csv', index=None)


main()




