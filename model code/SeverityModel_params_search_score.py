import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import SeverityModel as SM
import os
from statannotations.Annotator import Annotator
import random
import sklearn.metrics as sk_eval
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import warnings
warnings.filterwarnings('ignore')


class COVID_SEVERITY_CLASSIFICATION_MODEL:

    def __init__(self):

        self.S_disp = pd.read_csv('../result/Cell Dispersion/stephenson_sPC_disp.csv', index_col=0)
        self.S_prop = pd.read_csv('../result/Cell Proportion/stephenson.prop_df.csv', index_col=0)
        self.R_disp = pd.read_csv('../result/Cell Dispersion/ren_sPC_disp.csv', index_col=0)
        self.R_prop = pd.read_csv('../result/Cell Proportion/ren.prop_df.csv', index_col=0)

        self.S_meta = pd.read_csv('../Original datasets/metadata/stephenson_metadata.csv', index_col=0)
        self.R_meta = pd.read_csv('../Original datasets/metadata/ren_metadata.csv', index_col=0)

        self.S_sample_severity_dict = dict(zip(self.S_meta['sample_id'], self.S_meta['Status_on_day_collection_summary']))
        self.R_sample_severity_dict = dict(zip(self.R_meta['sampleID'], self.R_meta['CoVID-19 severity']))

        self.to_numeric = {'5class':{'Healthy':0, 'Mild':1, 'Moderate':1, 'Severe':2, 'Critical':2},
                           '3class':{'control':0, 'mild/moderate':1, 'severe/critical':2}}

        self.result_df = {'Combination':[], 'Train_on':[], 'Feature':[], 'Parameter_ID':[], 'params':[]}
        self.result_df.update({'{0}_{1}'.format(s,metric):[] for s in ['CV','S','R'] for metric in ['acc', 'f1', 'auc', 'precision', 'recall']})
        self.result_df.update({'{0}_{1}'.format(s,metric):[] for s in ['S','R'] for metric in ['ctl_acc','mm_acc','sc_acc']})

        self.params_id = 'params_0'
        self.loss_dict = {'Parameter_ID':[],'loss':[]}
    
    def Drop_ASP_Samples(self):

        self.S_disp = SM.Drop_ASP_samples(self.S_disp, self.S_sample_severity_dict, dtype='true')
        self.S_prop = SM.Drop_ASP_samples(self.S_prop, self.S_sample_severity_dict, dtype='true')


    def Drop_CVS_Samples(self):
        
        R_sample_time_dict = dict(zip(self.R_meta['sampleID'],self.R_meta['Sample time']))
        R_CVS_samples = [k for k, v in R_sample_time_dict.items() if v == 'convalescence']

        self.R_disp.drop(R_CVS_samples, inplace=True)
        self.R_prop.drop(R_CVS_samples, inplace=True)


    def Drop_Seven_Cell_Types(self):

        self.S_disp = SM.Drop_seven_celltypes_lacking_of_cell_dispersion(self.S_disp, dtype='true')
        self.R_disp = SM.Drop_seven_celltypes_lacking_of_cell_dispersion(self.R_disp, dtype='true')


    def KNN_Imputation_for_Cell_Dispersion(self):

        self.S_disp_knn = SM.KNN_imputation(self.S_disp)
        self.R_disp_knn = SM.KNN_imputation(self.R_disp)


    def Severity_Status_Counts(self):

        self.S_sample_severity_df = pd.DataFrame({'sample':list(self.S_sample_severity_dict.keys()), 
                                                  'severity':list(self.S_sample_severity_dict.values())})
        self.R_sample_severity_df = pd.DataFrame({'sample':list(self.R_sample_severity_dict.keys()), 
                                                  'severity':list(self.R_sample_severity_dict.values())})
        
        # Remain samples which are not CVS or ASP
        self.S_sample_severity_df = self.S_sample_severity_df[self.S_sample_severity_df['sample'].isin(self.S_disp.index)]
        self.R_sample_severity_df = self.R_sample_severity_df[self.R_sample_severity_df['sample'].isin(self.R_disp.index)]

    
    def Generate_Testing_Sample_List(self, source):

        testing_sample_list = []
        if source == 'stephenson':
            for severity in self.to_numeric['5class'].keys():
                sub_df = self.S_sample_severity_df[self.S_sample_severity_df['severity']==severity]

                if severity == 'Healthy':
                    sub_df = sub_df.sample(n=5)

                elif severity == 'Mild':
                    sub_df = sub_df.sample(n=5)

                elif severity == 'Moderate':
                    sub_df = sub_df.sample(n=6)

                elif severity == 'Severe':
                    sub_df = sub_df.sample(n=3)

                elif severity == 'Critical':
                    sub_df = sub_df.sample(n=3)

                testing_sample_list += list(sub_df['sample'])

        else:
            for severity in self.to_numeric['3class'].keys():
                sub_df = self.R_sample_severity_df[self.R_sample_severity_df['severity']==severity]

                if severity == 'control':
                    sub_df = sub_df.sample(n=4)

                elif severity == 'mild/moderate':
                    sub_df = sub_df.sample(n=4)

                elif severity == 'severe/critical':
                    sub_df = sub_df.sample(n=9)

                testing_sample_list += list(sub_df['sample'])

        return testing_sample_list

    
    def Generate_Training_Set(self, disp, prop, sample_severity_dict, sample_severity_df,
                                    testing_sample_list, source, oversampling_size_list):

        # Drop testing samples
        disp.drop(testing_sample_list, inplace=True)
        prop.drop(testing_sample_list, inplace=True)
        sample_severity_df = sample_severity_df[~sample_severity_df['sample'].isin(testing_sample_list)]

        # Oversampling
        if source == 'stephenson':
            mask_healthy  = sample_severity_df['severity']=='Healthy'
            mask_mild     = sample_severity_df['severity']=='Mild'
            mask_moderate = sample_severity_df['severity']=='Moderate'
            mask_severe   = sample_severity_df['severity']=='Severe'
            mask_critical = sample_severity_df['severity']=='Critical'

            CTL_samples = random.choices(list(sample_severity_df[mask_healthy]['sample']), k=oversampling_size_list[0])+list(sample_severity_df[mask_healthy]['sample'])
            MM_samples = list(sample_severity_df[mask_mild|mask_moderate]['sample'])
            SC_samples = random.choices(list(sample_severity_df[mask_severe|mask_critical]['sample']), k=oversampling_size_list[2])+list(sample_severity_df[mask_severe|mask_critical]['sample'])

            key = '5class' # to_numeric key

        else:
            mask_ctl = sample_severity_df['severity']=='control'
            mask_mm  = sample_severity_df['severity']=='mild/moderate'
            mask_sc  = sample_severity_df['severity']=='severe/critical'
            
            CTL_samples = random.choices(list(sample_severity_df[mask_ctl]['sample']), k=oversampling_size_list[0])+list(sample_severity_df[mask_ctl]['sample'])
            MM_samples = random.choices(list(sample_severity_df[mask_mm]['sample']), k=oversampling_size_list[1])+list(sample_severity_df[mask_mm]['sample'])
            SC_samples = list(sample_severity_df[mask_sc]['sample'])

            key = '3class' # to_numeric key
        
        # Rename feature names, otherwise, error occurs during model training process
        disp.columns = [c+'_disp' for c in disp.columns]
        prop.columns = [c+'_prop' for c in prop.columns]
        
        # Generate train_X, train_Y
        X = pd.concat([disp.loc[CTL_samples+MM_samples+SC_samples,:], prop.loc[CTL_samples+MM_samples+SC_samples,:]], axis=1)
        Y = pd.DataFrame({'Severity':[self.to_numeric[key][sample_severity_dict[sample]] for sample in X.index]})
        
        return X, Y

    
    def Model_for_One_Fold(self, source1_sample_severity_df, source1_sample_severity_dict, source1_to_numeric, source1_disp, source1_prop, source1, s1_testing_samples,
                                                             source2_sample_severity_dict, source2_to_numeric, source2_disp, source2_prop, source2, classifier):
        
        # Generate test_X, test_Y
        s1_testing_X, s1_testing_Y = SM.True_df_from_excluded_samples(source1_disp, source1_prop, s1_testing_samples,       source1_sample_severity_dict, source1_to_numeric)
        s2_testing_X, s2_testing_Y = SM.True_df_from_excluded_samples(source2_disp, source2_prop, list(source2_disp.index), source2_sample_severity_dict, source2_to_numeric)

        # Generate oversampled train_X, train_Y
        if source1=='stephenson':
            oversampling_size = [28, 0, 21]
        else:
            oversampling_size = [23, 24, 0]
        s1_training_X, s1_training_Y = self.Generate_Training_Set(source1_disp.copy(), source1_prop.copy(), source1_sample_severity_dict, source1_sample_severity_df, s1_testing_samples, source1, oversampling_size)
        
        #print('{0} train_X shape: {1}\n{2}\n{0} test_X shape : {3}\n{4}\n{5} test_X shape : {6}\n{7}\n'.format(source1, s1_training_X.shape, s1_training_X.head(2), s1_testing_X.shape, s1_testing_X.head(2), source2, s2_testing_X.shape, s2_testing_X.head(2)))
        
        # Make sure the order of features is correct
        feature_order = list(s1_training_X.columns)
        s1_testing_X = s1_testing_X.loc[:,feature_order]
        s2_testing_X = s2_testing_X.loc[:,feature_order]
        
        # Model CV strategy
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        cv_scores = ['accuracy', 'f1_macro', 'roc_auc_ovr', 'precision_macro', 'recall_macro']
        
        # Dispersion model
        print('Dispersion Model...')
        disp_model_result   = cross_validate(classifier, s1_training_X.iloc[:,0:23], s1_training_Y, cv=cv, scoring=cv_scores, return_estimator=True)
        # Proportion model
        print('Proportion Model...')
        prop_model_result   = cross_validate(classifier, s1_training_X.iloc[:,23:], s1_training_Y, cv=cv, scoring=cv_scores, return_estimator=True)
        # Concat model
        print('Concat Model...')
        concat_model_result = cross_validate(classifier, s1_training_X,             s1_training_Y, cv=cv, scoring=cv_scores, return_estimator=True)
        
        # Model performance output
        info_ls = []
        for source in [source1,source2]:
            for metric in ['acc','f1','auc','precision','recall','ctl_acc','mm_acc','sc_acc']:
                info_ls.append('{0}_testing_{1}'.format(source,metric))
        
        model_eval = {'Dispersion':{info:[] for info in info_ls}, 'Proportion':{info:[] for info in info_ls}, 'Concat':{info:[] for info in info_ls}}
        
        def eval_update(feature_type, model_result, idx_r, idx_l, cv):
            for info in info_ls:
                metric = info.split(sep='_')[2]

                if info.split(sep='_')[0]==source1:
                    X, Y = s1_testing_X.iloc[:,idx_r:idx_l], s1_testing_Y
                else:
                    X, Y = s2_testing_X.iloc[:,idx_r:idx_l], s2_testing_Y

                if metric == 'acc':
                    value = model_result['estimator'][cv].score(X, Y)

                elif metric == 'f1':
                    value = sk_eval.f1_score(Y, model_result['estimator'][cv].predict(X), average='macro')

                elif metric == 'auc':
                    value = sk_eval.roc_auc_score(Y, model_result['estimator'][cv].predict_proba(X), average='macro', multi_class='ovr')

                elif metric == 'precision':
                    value = sk_eval.precision_score(Y, model_result['estimator'][cv].predict(X), average='macro')

                elif metric == 'recall':
                    value = sk_eval.recall_score(Y, model_result['estimator'][cv].predict(X), average='macro')

                elif metric == 'ctl':
                    mask_ctl = Y['Severity']==0
                    value = model_result['estimator'][cv].score(X[mask_ctl], Y[mask_ctl])

                elif metric == 'mm':
                    mask_mm = Y['Severity']==1
                    value = model_result['estimator'][cv].score(X[mask_mm], Y[mask_mm])

                elif metric == 'sc':
                    mask_sc = Y['Severity']==2
                    value = model_result['estimator'][cv].score(X[mask_sc], Y[mask_sc])
                    
                model_eval[feature_type][info].append(value)
        
        for j in range(5):
            eval_update('Dispersion', disp_model_result, 0, 23, j)
            eval_update('Proportion', prop_model_result, 23, 53, j)
            eval_update('Concat', concat_model_result, 0, 53, j)
        

        return [disp_model_result, prop_model_result, concat_model_result], model_eval


    def Result_df_Update(self, comb, train_on, models, results, params):
        self.result_df['Combination'] += [comb]*15
        self.result_df['Train_on']    += [train_on]*15
        self.result_df['Feature']     += ['Dispersion']*5+['Proportion']*5+['Concat']*5
        self.result_df['Parameter_ID']+= [self.params_id]*15
        self.result_df['params']      += [params]*15
        
        self.result_df['CV_acc']       += list(models[0]['test_accuracy'])+list(models[1]['test_accuracy'])+list(models[2]['test_accuracy'])
        self.result_df['CV_f1']        += list(models[0]['test_f1_macro'])+list(models[1]['test_f1_macro'])+list(models[2]['test_f1_macro'])
        self.result_df['CV_auc']       += list(models[0]['test_roc_auc_ovr'])+list(models[1]['test_roc_auc_ovr'])+list(models[2]['test_roc_auc_ovr'])
        self.result_df['CV_precision'] += list(models[0]['test_precision_macro'])+list(models[1]['test_precision_macro'])+list(models[2]['test_precision_macro'])
        self.result_df['CV_recall']    += list(models[0]['test_recall_macro'])+list(models[1]['test_recall_macro'])+list(models[2]['test_recall_macro'])
        
        d = {'disp':'Dispersion', 'prop':'Proportion', 'concat':'Concat'}
        for f in ['disp', 'prop', 'concat']:
            for m in ['acc', 'f1', 'auc', 'precision', 'recall','ctl_acc','mm_acc','sc_acc']:
                self.result_df['{0}_{1}'.format('S', m)] += results[d[f]]['{0}_testing_{1}'.format('stephenson', m)]
                self.result_df['{0}_{1}'.format('R', m)] += results[d[f]]['{0}_testing_{1}'.format('ren', m)]

        
    def Save_Models(self, path2models, models, source, comb):
        features = ['disp','prop','concat']
        for i in range(3):
            f = '{0}_{1}_{2}.pickle'.format(source,features[i],self.params_id)
            print('Saving... {0}'.format(f))
            pickle.dump(models[i],open('{0}/{1}'.format(path2models ,f), 'wb'))

    
    def Main(self, classifier, path2models):

        for i in range(5):
            print('#'*80+'\nStephenson Combination {0}...\n'.format(i+1)+'#'*80)
            S_testing_samples = self.Generate_Testing_Sample_List('stephenson')
            
            S_knn_models, S_knn_result = self.Model_for_One_Fold(self.S_sample_severity_df, self.S_sample_severity_dict, self.to_numeric['5class'], self.S_disp_knn, self.S_prop, 'stephenson', S_testing_samples,
                                                                                            self.R_sample_severity_dict, self.to_numeric['3class'], self.R_disp_knn, self.R_prop, 'ren', classifier)
            self.Result_df_Update('Combination '+str(i+1), 'Stephenson', S_knn_models, S_knn_result, str(classifier.get_xgb_params()))
            #self.Save_Models(path2models, S_knn_models, 'S', 'comb'+str(i+1))
            
            
            print('#'*80+'\nRen Combination {0}...\n'.format(i+1)+'#'*80)
            R_testing_samples = self.Generate_Testing_Sample_List('ren')
            
            R_knn_models, R_knn_result = self.Model_for_One_Fold(self.R_sample_severity_df, self.R_sample_severity_dict, self.to_numeric['3class'], self.R_disp_knn, self.R_prop, 'ren', R_testing_samples,
                                                                                            self.S_sample_severity_dict, self.to_numeric['5class'], self.S_disp_knn, self.S_prop, 'stephenson', classifier)
            
            self.Result_df_Update('Combination '+str(i+1), 'Ren', R_knn_models, R_knn_result, str(classifier.get_xgb_params()))
            #self.Save_Models(path2models, R_knn_models, 'R', 'comb'+str(i+1))

        return pd.DataFrame(self.result_df).iloc[-150:,:] # The newest result : 2(datasets) X 5 (cross-validations) X 3 (Features) X 5 (Combinations)


    def Objective(self, space):
        
        print(self.params_id)

        classifier = xgb.XGBRFClassifier(n_estimators=int(space['n_estimators']), 
                                         max_depth=int(space['max_depth']), 
                                         eta=space['eta'], 
                                         subsample=space['subsample'], 
                                         colsample_bytree=space['colsample_bytree'], 
                                         reg_alpha=space['reg_alpha'], 
                                         gamma=space['gamma'],
                                         reg_lambda=space['reg_lambda'],
                                         min_child_weight=space['min_child_weight'])

        objective_df = self.Main(classifier, space['path2models'])
        loss = -objective_df.iloc[:,10:].sum().sum()

        print(self.params_id, ' loss:', loss)
        self.loss_dict['Parameter_ID'].append(self.params_id)
        self.loss_dict['loss'].append(loss)

        self.params_id = 'params_'+str(int(self.params_id.split(sep='_')[1])+1)
        
        return {'loss':loss, 'status':STATUS_OK}




def main():

    TASK = COVID_SEVERITY_CLASSIFICATION_MODEL()
    TASK.Drop_ASP_Samples()
    TASK.Drop_CVS_Samples()
    TASK.Drop_Seven_Cell_Types()
    TASK.KNN_Imputation_for_Cell_Dispersion()
    TASK.Severity_Status_Counts()
    
    path2models = '../result/SeverityModel_params_search_score'

    space={'max_depth': hp.quniform("max_depth", 3, 15, 1),
           'gamma': hp.uniform ('gamma', 0,2),
           'reg_alpha' : hp.uniform('reg_alpha', 0,5),
           'reg_lambda' : hp.uniform('reg_lambda', 0,1),
           'colsample_bytree' : hp.uniform('colsample_bytree', 0.2,1),
           'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
           'n_estimators': hp.quniform('n_estimators',1200,3000,1),
           'eta': hp.uniform('eta',0,0.5),
           'subsample' : hp.uniform('subsample',0,0.8),
           'path2models' : path2models}
    
    trials = Trials()

    best_hyperparams = fmin(fn = TASK.Objective, space = space, algo = tpe.suggest, max_evals = 1000, trials = trials)
    print('Best parameters:',best_hyperparams)

    
    pd.DataFrame(TASK.result_df).to_csv(path2models+'/model_result.csv', index=None)
    pd.DataFrame(TASK.loss_dict).to_csv(path2models+'/params_loss.csv', index=None)


main()