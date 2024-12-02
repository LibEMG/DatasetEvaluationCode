from libemg.datasets import *
from Models.MLP import MLP
from Models.MLPR import MLPR
from Models.AEMLP import AE_MLP

# Evaluate all classification datasets (LDA and MLP) 
c_ds = list(get_dataset_list().keys())
evaluate('LDA', 200, 50, ['MAV','ZC','SSC','WL'], included_datasets=c_ds, output_file='Results/lda.pkl')
evaluate(MLP(), 200, 50, ['MAV','ZC','SSC','WL'], included_datasets=c_ds, output_file='Results/mlp.pkl', normalize_features=True)

# Evaluate all regression datasets (LR and MLPR) 
c_ds = list(get_dataset_list(cross_user=True).keys())
evaluate_crossuser('LDA', 200, 50, ['MAV','ZC','SSC','WL'], included_datasets=c_ds, output_file='Results/lda.pkl')
evaluate_crossuser(MLP(), 200, 50, ['MAV','ZC','SSC','WL'], included_datasets=c_ds, output_file='Results/mlp.pkl', normalize_features=True)

# Evaluate all regression datasets (LR and MLPR) 
c_ds = list(get_dataset_list().keys())
evaluate('LDA', 200, 50, ['MAV','ZC','SSC','WL'], included_datasets=c_ds, output_file='Results/lr.pkl', regression=True)
evaluate(MLPR(), 200, 50, ['MAV','ZC','SSC','WL'], included_datasets=c_ds, output_file='Results/mlpr.pkl', normalize_features=True, regression=True)

# Evaluate cross-user EMG2POSE
ds = EMG2POSECU()
for m in ds.mapping.keys():
    print(m)
    if m == 'Unconstrained':
        continue
    evaluate_crossuser(MLPR(), 200, 50, feature_list=['MAV', 'ZC', 'SSC', 'WL'], metrics=['MSE', 'MAE'], output_file='Results/MetaResults/' + m + '_mlp.pkl', memory_efficient=True, included_datasets=[EMG2POSECU(stage=m)], regression=True, normalize_features=True)

# Evaluate weakly supervised dataset
mdl = AE_MLP(input_shape=32, latent_dims=16, num_classes=5, num_decoder_layers=3, num_encoder_layers=3)
evaluate_weaklysupervised(mdl, 300, 100, feature_list=['MAV','SSC','ZC','WL'], included_datasets=['CIIL_WeaklySupervised'], normalize_features=True)