from libemg.datasets import *
from Models.AEMLP import AE_MLP

feature_dics = [{}, {}, {'WENG_fs': 1000}]
input_shapes = [32, 8*7, 64]

for f_i, f in enumerate([['MAV', 'SSC', 'ZC', 'WL'], ['RMSPHASOR'], ['WENG']]):
    ws_model = AE_MLP(input_shape=input_shapes[f_i], latent_dims=16, num_classes=5, num_decoder_layers=3, num_encoder_layers=3)
    evaluate_weaklysupervised(ws_model, 200, 50, feature_list=f, included_datasets=['CIIL_WeaklySupervised'], normalize_features=True, output_file='NewResults/weakly_' + str(f_i) + '.pkl', feature_dic=feature_dics[f_i])