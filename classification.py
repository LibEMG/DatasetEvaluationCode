from libemg.datasets import *
from Models.MLP import MLP

# (1) Get all the classification datasets 
c_ds = list(get_dataset_list().keys())
cu_ds = list(get_dataset_list(cross_user=True).keys())

# (2) Create feature dictionaries for all necessary parameters (WENG feature needs the sampling frequency)
feature_dics = [{}, {}, [{'WENG_fs': get_dataset_list()[d]().sampling} for d in c_ds]]
cu_feature_dics = [{}, {}, {'WENG_fs': 200}]

# (3) Evaluate LDA and MLP across three feature sets (HTD, RMSPHASOR, and WENG)
for f_i, f in enumerate([['MAV', 'SSC', 'ZC', 'WL'], ['RMSPHASOR'], ['WENG']]):
    evaluate('LDA', 200, 50, f, included_datasets=c_ds, output_file='NewResults/lda_' + str(f_i) + '.pkl', feature_dic=feature_dics[f_i])
    evaluate(MLP(), 200, 50, f, included_datasets=c_ds, output_file='NewResults/mlp_' + str(f_i) + '.pkl', normalize_features=True, feature_dic=feature_dics[f_i])
    evaluate_crossuser('LDA', 200, 50, f, included_datasets=cu_ds, output_file='NewResults/lda_cu_' + str(f_i) + '.pkl', feature_dic=cu_feature_dics[f_i])
    evaluate_crossuser(MLP(), 200, 50, f, included_datasets=cu_ds, output_file='NewResults/mlp_cu' + str(f_i) + '.pkl', normalize_features=True, feature_dic=cu_feature_dics[f_i])