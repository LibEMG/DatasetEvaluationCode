from libemg.datasets import *
from Models.MLPR import MLPR

# (1) Get all the regression datasets 
c_ds = list(get_dataset_list('REGRESSION').keys())
cu_ds = EMG2POSECU()

# (2) Create feature dictionaries for all necessary parameters (WENG feature needs the sampling frequency)
feature_dics = [{}, {}, [{'WENG_fs': get_dataset_list('REGRESSION')[d]().sampling} for d in c_ds] ]
cu_feature_dics = [{}, {}, {'WENG_fs': 2000}]

# (3) Evaluate LR and MLP across three feature sets (HTD, RMSPHASOR, and WENG)
for f_i, f in enumerate([['MAV', 'SSC', 'ZC', 'WL'], ['RMSPHASOR'], ['WENG']]):
    evaluate('LR', 200, 50, f, metrics=['MAE'], included_datasets=c_ds, output_file='NewResults/lr_' + str(f_i) + '.pkl', regression=True, feature_dic=feature_dics[f_i])
    evaluate(MLPR(), 200, 50, f, metrics=['MAE'], included_datasets=c_ds, output_file='NewResults/mlpr_' + str(f_i) + '.pkl', normalize_features=True, regression=True, feature_dic=feature_dics[f_i])

    # (4) Evaluate cross-user EMG2POSE 
    ds = EMG2POSECU()
    for m in ds.mapping.keys():
       print(m)
       if m == 'Unconstrained':
           continue
       evaluate_crossuser(MLPR(), 200, 50, feature_list=f, metrics=['MAE'], output_file='NewResults/MetaResults/' + m + '_mlp_' + str(f_i) + '.pkl', memory_efficient=True, included_datasets=[EMG2POSECU(stage=m)], regression=True, normalize_features=True, feature_dic=cu_feature_dics[f_i])
       evaluate_crossuser("LR", 200, 50, feature_list=f, metrics=['MAE'], output_file='NewResults/MetaResults/' + m + '_lr_' + str(f_i) + '.pkl', memory_efficient=True, included_datasets=[EMG2POSECU(stage=m)], regression=True, feature_dic=cu_feature_dics[f_i])