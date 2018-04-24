import os
import shutil
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import glob, re

from config_visualizer import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.autograd import Variable
from tensorboard_logger import configure, log_value
from model import CNN
from utils import *

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser(description='ASL_CNN')

parser.add_argument('--train_data', default='', metavar='DIR', help='Path to training hdf5 file')

parser.add_argument('--test_data', default='', metavar='DIR', help='Path to test hdf5 file')

parser.add_argument('--val_data', default='', metavar='DIR', help='Path to validation hdf5 file')

parser.add_argument('--save_path', default='', metavar='DIR', help='Path to save folder')

parser.add_argument('--batch_size', default=100, type=int, help='Batch size')

parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')

parser.add_argument('--lr', default=0.1, type=float, help='Learning rate for optimizer')

parser.add_argument('--id', default=1, type=float, help='Learning rate for optimizer')

def main():
    args = parser.parse_args()
    print (args)
    fingers = ['mixed']#['thumb', 'pinky', 'mixed', 'index', 'middle', 'ring']
    types = ['train', 'test']
    ids = [args.id]
    regex = re.compile(r'\d+')
    train_set = dict()
    test_set = dict()
    for finger in fingers:
        for t in types:
            for name in glob.glob('data/*%s*%s'%(t, finger)):
                curr_id = int(regex.findall(name)[3])
                if curr_id in ids:
                    dw, da, dl = unpackSingle_h5(name)
                    if t == 'train':
                        if len(train_set.keys()) == 0:
                            train_set['dw'] = dw
                            train_set['da'] = da
                            train_set['dl'] = dl
                        else:
                            train_set['dw'] = np.concatenate((train_set['dw'], dw))
                            train_set['da'] = np.concatenate((train_set['da'], da))
                            train_set['dl'] = np.concatenate((train_set['dl'], dl))
                    else:
                        if len(test_set.keys()) == 0:
                            test_set['dw'] = dw
                            test_set['da'] = da
                            test_set['dl'] = dl
                        else:
                            test_set['dw'] = np.concatenate((test_set['dw'], dw))
                            test_set['da'] = np.concatenate((test_set['da'], da))
                            test_set['dl'] = np.concatenate((test_set['dl'], dl))

    train_set['dw'], train_set['da'], train_set['dl'] = \
        data_preprocessing(train_set['dw'], train_set['da'], train_set['dl'], train=True, balance=True)
    test_set['dw'], test_set['da'], test_set['dl'] = \
        data_preprocessing(test_set['dw'], test_set['da'], test_set['dl'])

    model = CNN()
    # if torch.cuda.is_available():
    #     model.cuda()

    batch_size = 100

    model = train(model, 
        processData(train_set['dw']), train_set['da'], train_set['dl'], 
        processData(test_set['dw']), test_set['da'], test_set['dl'], 
        args.lr, args.batch_size, args.epochs, 10, '.')

    _, test_true, test_pred = \
        evaluate(model, processData(test_set['dw']), test_set['da'], test_set['dl'], batch_size)
    _, train_true, train_pred = \
        evaluate(model, processData(train_set['dw']), train_set['da'], train_set['dl'], batch_size)

    test_true = torch.cat(test_true).data.numpy()
    test_pred = torch.max(torch.cat(test_pred), 1)[1].data.numpy()
    train_true = torch.cat(train_true).data.numpy()
    train_pred = torch.max(torch.cat(train_pred), 1)[1].data.numpy()


    cnf_matrix = confusion_matrix(test_true, test_pred)
    plot_confusion_matrix(cnf_matrix, classes=np.unique(test_true), title='Confusion matrix, without normalization')
    print "training acc", accuracy_score(train_true, train_pred)
    print "testing acc", accuracy_score(test_true, test_pred)
    print classification_report(test_true, test_pred)

def train(model, 
    train_waveforms, train_angles, train_labels, 
    test_waveforms, test_angles, test_labels, 
    lr, batch_size, epochs, print_step, save_path):
    num_batches = int(train_waveforms.shape[0]/batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    log_step = 0
    step_counter = 0
    best_val_loss = 1
    check_every = 100
    # make_dir(save_path)
    # configure(run_path , flush_secs = 2)
    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch))
        model.train()
        total_loss = 0.0
        for batch_id in range(num_batches):
            batch_x = to_var(train_waveforms[batch_id*batch_size:(batch_id+1)*batch_size]).unsqueeze(1)
            batch_y = to_var(train_labels[batch_id*batch_size:(batch_id+1)*batch_size], 'Long')
            angle = to_var(train_angles[batch_id*batch_size:(batch_id+1)*batch_size])
            out = model(batch_x, angle)
            loss = criterion(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data[0]
            if step_counter % check_every == 0 and step_counter > 0:
                # log_value('Train Loss (MSE)', total_loss/batch_id, log_step)
                validation_loss, _, _ = evaluate(model, test_waveforms, test_angles, test_labels, batch_size)
                # log_value('Validation Loss (MSE)', validation_loss, log_step)
                log_step += 1
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pkl'))
            if batch_id % print_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Total Loss: %.4f'
                      %(epoch, epochs, batch_id, num_batches, total_loss))
            step_counter += 1
    return model

def evaluate(model, waveforms, angles, labels, batch_size):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    num_batches = int(waveforms.shape[0]/batch_size)
    ground = []
    predict = []
    for batch_id in range(num_batches):
        batch_x = to_var(waveforms[batch_id*batch_size:(batch_id+1)*batch_size]).unsqueeze(1)
        batch_y = to_var(labels[batch_id*batch_size:(batch_id+1)*batch_size], 'Long')
        angle = to_var(angles[batch_id*batch_size:(batch_id+1)*batch_size])
        out = model(batch_x, angle)
        ground.append(batch_y)
        predict.append(out)
        loss = criterion(out, batch_y)
        total_loss += loss.data[0]s
    loss = total_loss/num_batches
    return loss, ground, predict

if __name__ == '__main__':
    main()