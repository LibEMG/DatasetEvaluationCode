import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class AE_MLP(nn.Module):
    def __init__(self, input_shape, latent_dims, num_classes, num_encoder_layers, num_decoder_layers):
        super(AE_MLP, self).__init__()
        self.input_shape = input_shape
        self.latent_dims = latent_dims 
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.make_encoder()
        self.make_decoder()
        self.make_mlp()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.mlp.to(self.device)

    def make_encoder(self):
        self.encoder_construction = []
        encoder_parameter_step = (32 - self.latent_dims)/self.num_encoder_layers
        input_size = self.input_shape
        for i in range(self.num_encoder_layers):
            output_size = int(32 - (i+1)*encoder_parameter_step)
            self.encoder_construction.extend([
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(),
                nn.Dropout1d(0.1)
            ])
            input_size = output_size

        self.encoder = nn.Sequential(
            *self.encoder_construction
        )
    
    def make_decoder(self):
        self.decoder_construction = []
        decoder_parameter_step = (32 - self.latent_dims)//self.num_decoder_layers
        input_size = self.latent_dims
        for i in range(self.num_decoder_layers-1):
            output_size = int(self.latent_dims + (i+1)*decoder_parameter_step)
            self.decoder_construction.extend([
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(),
                nn.Dropout1d(0.1)
            ])
            input_size = output_size
        self.decoder_construction.extend([
                nn.Linear(input_size, self.input_shape)
            ])
        self.decoder = nn.Sequential(
            *self.decoder_construction
        )

    def make_mlp(self):
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dims, self.num_classes),
            nn.ReLU(),
            nn.Linear(self.num_classes, self.num_classes),
            nn.ReLU(),
            nn.Linear(self.num_classes, self.num_classes)
        )

    def fit(self, ds):
        pretrain_dataclass = PretrainDataClass(ds['pretraining_features'])
        pretrain_dataloader = DataLoader(pretrain_dataclass,
                                         64, shuffle=True)
        train_dataclass    = TrainDataClass(ds['training_features'], ds['training_labels'])
        train_dataloader   = DataLoader(train_dataclass,
                                        batch_size=64, shuffle=True)

        self.train()
        self.fit_ae(pretrain_dataloader)

        self.fit_mlp(train_dataloader)

        self.eval()

    def fit_ae(self, pretrain_dataloader, verbose=0):
        # unfreeze encoder/decoder
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        for param in self.decoder.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                     lr = 1e-3
        )

        loss_function = nn.L1Loss()
        for e in range(25):
            for i, data in enumerate(pretrain_dataloader):
                optimizer.zero_grad()
                data = data.to(self.device)

                latent = self.encoder(data)
                
                data_hat = self.decoder(latent)

                loss = loss_function(data, data_hat)
                loss.backward()

                optimizer.step()
                # print(f'{e}:\t{i} {loss.item()}')
        
        if verbose:
            pca = PCA(2)
            latent = self.encoder(pretrain_dataloader.dataset.inputs.to(self.device))
            t_data = pca.fit_transform(latent.detach().cpu().numpy())
            plt.scatter(t_data[:,0], t_data[:,1])
            plt.show()
        # freeze encoder/decoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.decoder.parameters():
            param.requires_grad = False

    def fit_mlp(self, train_dataloader, verbose=0):
        optimizer = torch.optim.Adam(self.mlp.parameters(),
                                        lr = 1e-3
        )
        for param in self.mlp.parameters():
            param.requires_grad = True
        loss_function = nn.CrossEntropyLoss()
        for e in range(1000):
            for i, (data, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                data = data.to(self.device)
                labels = labels.to(self.device)

                latent = self.encoder(data)
                labels_hat = self.mlp(latent)

                loss = loss_function(labels_hat, labels)
                loss.backward()
                optimizer.step()
                # print(f'{e}:\t{i} {loss.item()}')
        
        if verbose:
            pca = PCA(2)
            t_data = pca.fit_transform(train_dataloader.dataset.inputs.numpy())
            predictions = self.encoder(train_dataloader.dataset.inputs.to(self.device))
            predictions = self.mlp(predictions)
            predictions = torch.argmax(predictions,1)
            plt.scatter(t_data[:,0], t_data[:,1], c=predictions.detach().cpu().numpy())
            plt.show()


        
        for param in self.mlp.parameters():
            param.requires_grad = False


    def predict(self, test_features):
        proba = self.predict_proba(test_features)
        preds = np.argmax(proba, 1)
        return preds

    def predict_proba(self, test_features):
        if type(test_features) == np.ndarray:
            test_features = torch.tensor(test_features, dtype=torch.float32)
        test_features = test_features.to(self.device)
        latent = self.encoder(test_features)
        proba = self.mlp(latent)
        return np.array(proba.detach().cpu())


class PretrainDataClass:
    def __init__(self,inputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx]
    
class TrainDataClass:
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]