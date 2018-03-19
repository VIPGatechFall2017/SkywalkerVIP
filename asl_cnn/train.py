import os
import shutil
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboard_logger import configure, log_value
from model import CNN
from utils import *

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser(description='SFXNet Trainer')

parser.add_argument('--train_data', default='', metavar='DIR', help='Path to training hdf5 file')

parser.add_argument('--test_data', default='', metavar='DIR', help='Path to test hdf5 file')

parser.add_argument('--val_data', default='', metavar='DIR', help='Path to validation hdf5 file')

parser.add_argument('--save_path', default='', metavar='DIR', help='Path to save folder')

parser.add_argument('--batch_size', default=100, type=int, help='Batch size')

parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')

parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate for optimizer')

def main():
    args = parser.parse_args()
    print (args)

    train_waveforms, train_angles, train_labels = data_loader(args.train_data)
    test_waveforms, test_angles, test_labels = data_loader(args.test_data)

    model = CNN()
    model = train(model, 
        train_waveforms, train_angles, train_labels, 
        test_waveforms, test_angles, test_labels, 
        args.lr, args.batch_size, args.epochs, args.save_path)

def train(model, 
    train_waveforms, train_angles, train_labels, 
    test_waveforms, test_angles, test_labels, 
    lr, batch_size, epochs, save_path):
    num_batches = int(train_waveforms.shape[0]/batch_size)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    log_step = 0
    step_counter = 0
    best_val_loss = 1
    check_every = 100
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch))
        model.train()
        total_loss = 0.0
        for batch_id in range(num_batches):
            batch_x = train_waveforms[batch_id*batch_size:(batch_id+1)*batch_size]
            batch_y = train_labels[batch_id*batch_size:(batch_id+1)*batch_size]
            out = model(batch_x)
            loss = criterion(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data[0]
            if step_counter % check_every == 0 and step_counter > 0:
                log_value('Train Loss (MSE)', total_loss/batch_id, log_step)
                validation_loss = evaluate(model, test_waveforms, test_angles, test_labels, batch_size)
                log_value('Validation Loss (MSE)', validation_loss, log_step)
                log_step += 1
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pkl'))
            print(epoch+1, batch_id+1, num_batches, loss.data[0])
            step_counter += 1
    return model

def evaluate(model, test_waveforms, test_angles, test_labels, batch_size):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    num_batches = int(data['y'].shape[0]/batch_size) + 1
    x = []
    expect = []
    predict = []
    for batch_id in range(num_batches):
        batch_x = test_waveforms[batch_id*batch_size:(batch_id+1)*batch_size]
        batch_y = test_labels[batch_id*batch_size:(batch_id+1)*batch_size]
        out = model(batch_x, hidden)
        loss = criterion(out, batch_y)
        total_loss += loss.data[0]
    loss = total_loss/num_batches
    return loss

# model.load_state_dict(torch.load(path, map_location=lambda storage, loc:storage))

if __name__ == '__main__':
    main()