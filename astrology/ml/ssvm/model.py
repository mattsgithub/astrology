from itertools import product

import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

class Coder(nn.Module):
    def __init__(self, n_input, n_output):
        
        super(Coder, self).__init__()
        self.dense = nn.Linear(n_input, n_output)

    def forward(self, x):
        x = self.dense(x)
        x = F.relu(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(EncoderDecoder, self).__init__()
        
        self.encoder = Coder(n_input, n_hidden)
        self.decoder = Coder(n_hidden, n_input)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def vector(self, x):
         return self.encoder(x)


class AutoEncoder:
    def __init__(self, n_epochs, batch_size, dim):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dim = dim

    def predict(self, x):
        # Return the y prediction
        min_energy = float('inf')
        y_pred = None
        for y in self.y_space:
            X = np.concatenate((x, y)).astype('float32')
            X = torch.tensor(X)
            X_reconstruct = self.encoder_decoder(X)
            energy = F.mse_loss(X, X_reconstruct).item()

            if energy < min_energy:
                y_pred = y
                min_energy = energy

        return y_pred

    def fit(self, X_train, y_train):
        self.y_space = np.array(list(product([0, 1], repeat=y_train.shape[1])))

        # Cocatenate inputs together
        X_train = np.concatenate((X_train, y_train), axis=1).astype('float32')
        n_input = X_train.shape[1]

        self.encoder_decoder = EncoderDecoder(n_input=n_input,
                                              n_hidden=self.dim)

        loss = nn.MSELoss()
        optimizer = optim.SGD(self.encoder_decoder.parameters(), lr=0.01)

        # Target is itself
        train_dataset = Dataset(X_train, X_train)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=32,
                                      shuffle=True)

        m = 0
        time = []
        loss_at_each_epoch = []

        for epoch in range(1, self.n_epochs + 1):
            losses = []
            for i, data in enumerate(train_dataloader, 0):
                X_train_, y_train_ = data
                
                optimizer.zero_grad()
                
                y_hat = self.encoder_decoder(X_train_)
                out = loss(y_hat, y_train_)
                out.backward()
                optimizer.step()
                        
                # Update number of examples trained on
                m += X_train_.shape[0]
                
                # Record data
                losses.append(out.item())
                
            mean_loss = np.mean(losses)
            
            time.append(m)
            loss_at_each_epoch.append(mean_loss)

    def vector(self, x, y):
        
        x = np.concatenate((x, y)).astype('float32')
        x = torch.tensor(x)
        vector = self.encoder_decoder.vector(x).detach().numpy()

        return vector