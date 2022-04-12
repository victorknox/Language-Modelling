# train.py

import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader


file_path = 'drive/MyDrive/intro-to-nlp-assign3'
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataset, model, args):
    # Early stopping
    last_loss = 100
    patience = 2
    triggertimes = 0
    
    model.train()
    

    learning_rate = 0.001
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            if batch % 100 == 0 or batch == len(dataloader):
                print('[Epoch: {}/{}, Batch: {}/{}] loss: {:.8}'.format(epoch, args.max_epochs, batch, len(dataloader), loss.item()))
            
            curr_loss = loss.item()
            last_loss = curr_loss
            if curr_loss > last_loss:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!')
                    return model

            else:
                trigger_times = 0

            last_loss = curr_loss

# train(dataset, model, args)

# torch.save(model, 'model.pth')