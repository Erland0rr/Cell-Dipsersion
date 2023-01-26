import pandas as pd
import numpy as np

def Drop_ASP_samples(df, sample_severity, dtype='true'):
    ASP_samples = [sample for sample, status in sample_severity.items() if status=='Asymptomatic']
    if dtype == 'true':
        df = df[~df.index.isin(ASP_samples)]
    elif dtype == 'aug':
        df = df[~df['sample'].isin(ASP_samples)]
    return df

def Select_overlapped_samples(disp, prop):
    df_overlapped_samples = list(set(disp['sample'].unique()).intersection(prop['sample'].unique()))
    disp = disp[disp['sample'].isin(df_overlapped_samples)]
    prop = prop[prop['sample'].isin(df_overlapped_samples)]
    return disp, prop

def Drop_seven_celltypes_lacking_of_cell_dispersion(df, dtype='true'):
    drop_types = ['CD8 Proliferating', 'Eryth', 'dnT', 'CD4 Proliferating', 'ASDC', 'HSPC', 'cDC1']
    if dtype == 'true':
        df = df.drop(drop_types, axis=1)
    elif dtype == 'aug':
        df = df[~df['cell_type'].isin(drop_types)]
    return df

def KNN_imputation(df):
    from sklearn.impute import KNNImputer
    df = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df), index=df.index, columns=df.columns)
    return df

def Generate_testing_sample_list(sample_severity_df, to_numeric, source):
    testing_sample_list = []
    if source == 'stephenson':
        for severity in to_numeric.keys():
            sub_df = sample_severity_df[sample_severity_df.severity==severity]
            if severity == 'Healthy':
                sub_df = sub_df.sample(n=8)
            elif severity == 'Mild':
                sub_df = sub_df.sample(n=9)
            elif severity == 'Moderate':
                sub_df = sub_df.sample(n=10)
            elif severity == 'Severe':
                sub_df = sub_df.sample(n=5)
            elif severity == 'Critical':
                sub_df = sub_df.sample(n=5)
            testing_sample_list += list(sub_df['sample'])
    else:
        for severity in to_numeric.keys():
            sub_df = sample_severity_df[sample_severity_df.severity==severity]
            if severity == 'control':
                sub_df = sub_df.sample(n=7)
            elif severity == 'mild/moderate':
                sub_df = sub_df.sample(n=6)
            elif severity == 'severe/critical':
                sub_df = sub_df.sample(n=16)
            testing_sample_list += list(sub_df['sample'])
    return testing_sample_list

def True_df_from_excluded_samples(disp, prop, testing_sample_list, sample_severity, to_numeric):
    
    disp = disp.loc[testing_sample_list,:]
    disp.columns = [c+'_disp' for c in disp.columns]
    
    prop = prop.loc[testing_sample_list,:]
    prop.columns = [c+'_prop' for c in prop.columns]
    
    out_X = pd.concat([disp, prop], axis=1)
    out_Y = pd.DataFrame({'Severity':[to_numeric[sample_severity[s]] for s in testing_sample_list]}, index=testing_sample_list)
    
    return out_X, out_Y

def True_df_from_ren(disp, prop, sample_severity, to_numeric, first):
    
    if first:
        disp.columns = [c+'_disp' for c in disp.columns]
        prop.columns = [c+'_prop' for c in prop.columns]
    
    out_X = pd.concat([disp, prop], axis=1)
    out_Y = pd.DataFrame({'Severity':[to_numeric[sample_severity[s]] for s in disp.index]}, index=disp.index)
    
    return out_X, out_Y

def Generate_trainingNtesting_sets(disp, prop, sample_size, testing_sample_list, sample_severity, to_numeric):
    from sklearn.impute import KNNImputer
    drop_types = ['CD8 Proliferating', 'Eryth', 'dnT', 'CD4 Proliferating', 'ASDC', 'HSPC', 'cDC1']
    out_disp_X = {'{0}_disp'.format(c):[] for c in disp['cell_type'].unique()}
    out_disp_X.update({'sample':[]})
    out_prop_X = {'{0}_prop'.format(c):[] for c in prop['cell_type'].unique()}
    out_prop_X.update({'sample':[]})
    out_Y = {'Severity':[],'sample':[]}
    
    for s in disp['sample'].unique(): # same in disp and prop
        for c in prop['cell_type'].unique(): # 23 cell types in disp, 30 cell types in prop
            prop_mask_s = prop['sample']==s
            prop_mask_c = prop['cell_type']==c
            sub_prop = prop[prop_mask_s&prop_mask_c].sample(n=sample_size)
            out_prop_X[c+'_prop'] += list(sub_prop['proportion'])
            
            if c not in drop_types: 
                disp_mask_s = disp['sample']==s
                disp_mask_c = disp['cell_type']==c
                try:
                    sub_disp = disp[disp_mask_s&disp_mask_c].sample(n=sample_size)
                    out_disp_X[c+'_disp'] += list(sub_disp['distance'])
                except:
                    out_disp_X[c+'_disp'] += [np.nan]*sample_size
        out_prop_X['sample'] += [s]*sample_size
        out_disp_X['sample'] += [s]*sample_size
                
        out_Y['Severity'] += [to_numeric[sample_severity[s]]]*sample_size
        out_Y['sample'] += [s]*sample_size
        
    out_disp_X = pd.DataFrame(out_disp_X)
    out_disp_X = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(out_disp_X.iloc[:,0:23]), index=out_disp_X['sample'], columns=out_disp_X.columns[0:23])
    
    out_prop_X = pd.DataFrame(out_prop_X)
    out_prop_X.set_index('sample', inplace=True)    
        
    out_X = pd.concat([out_disp_X, out_prop_X], axis=1)
    out_Y = pd.DataFrame(out_Y).set_index('sample')
    
    
    train_X, train_Y = out_X[~out_X.index.isin(testing_sample_list)], out_Y[~(out_Y.index.isin(testing_sample_list))]
    test_X, test_Y = out_X[out_X.index.isin(testing_sample_list)], out_Y[out_Y.index.isin(testing_sample_list)]
    
    
    return train_X, train_Y, test_X, test_Y
