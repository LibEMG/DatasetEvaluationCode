import torch.nn as nn
import torch.optim as optim
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from libemg.emg_predictor import EMGClassifier


class MLPR(nn.Module):
    def __init__(self, layers=[128, 64, 32]):
        super().__init__()
        self.og_layers = layers 
        fix_random_seed(0)

    def setup(self, n_labels, n_features):
        self.layers = np.array(self.og_layers) * n_features // 32
        self.dropout = nn.Dropout(p=0.2)
        self.initial_layer = nn.Linear(n_features, self.layers[0])
        self.layer1 = nn.Linear(self.layers[0], self.layers[1])
        self.layer2 = nn.Linear(self.layers[1], self.layers[2])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.layers[-1], n_labels) 
        self.n_labels = n_labels

    def forward_once(self, out):
        out = self.initial_layer(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

    def forward(self, x):
        out = self.forward_once(x)
        out = self.output_layer(out)
        return out

    def fit(self, train_feats, train_labels, learning_rate=1e-3, num_epochs=10000, verbose=False):
        training_losses = []
        if type(train_feats) is dict:
            train_feats = EMGClassifier()._format_data(train_feats)

        self.setup(train_labels.shape[1], len(train_feats[0]))        
        tr_dl = make_data_loader(train_feats, train_labels)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        # get the optimizer and loss function ready
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        loss_function = nn.L1Loss()
        # Logger:
        self.log = {"training_loss":[], "training_accuracy": []} 
        # now start the training
        epoch = 0
        while epoch < num_epochs:
            tr_loss = []
            #training set
            self.train()
            for data, labels in tr_dl:
                optimizer.zero_grad()
                data = data.to(device)
                labels = labels.to(device)
                output = self.forward(data)
                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()
                tr_loss.append(loss.item())

                # log it
                self.log["training_loss"] += [(epoch, loss.item())]
            scheduler.step()
            training_losses.append(np.mean(tr_loss))
            if len(training_losses) > 50: # Used 50 for evertyhing other than Meta
                has_improved = False
                for i in range(2,6):
                    if training_losses[-i] - training_losses[-1] > 0.01:
                        has_improved = True
                if not has_improved:
                    epoch = 10000000 # End 
            if verbose:
                epoch_trloss = np.mean([i[1] for i in self.log['training_loss'] if i[0]==epoch])
                print(f"{epoch}: trloss:{epoch_trloss:.5f}")
            epoch+=1
        self.eval()
        self.to('cpu')

    def predict(self, x, device='cpu'):
        self.to(device)
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        preds = self.forward(x.to(device))
        return preds.detach().cpu().numpy()

    def predict_proba(self, x):
        return None

class DL_input_data(Dataset):
    def __init__(self, windows, labels):
        self.data = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.labels[idx]
        return data, label

    def __len__(self):
        return self.data.shape[0]

def make_data_loader(windows, labels, batch_size=1000):
    obj = DL_input_data(windows, labels)
    dl = DataLoader(obj,
    batch_size=batch_size,
    shuffle=True)
    return dl

def fix_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
