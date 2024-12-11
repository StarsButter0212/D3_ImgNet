#!/usr/bin/env python3

import pickle
import sys
import torch
import torch.nn as nn

sys.path.append('../')
import predict_args as args
from train import train_imgnet as train


if __name__ == "__main__":

    data_dir = args.dataset_dir
    dataset_trained = args.dataset_trained
    dataset_predict = args.dataset_predict
    basis_set = args.basis_set
    dim = args.dim
    seq_len = args.seq_len

    batch_size = args.batch_size
    operation = args.operation
    lr = args.lr
    iteration = args.iteration
    dropout = args.dropout
    num_workers = args.num_workers

    task = args.task
    transfer_flag = args.transfer_flag
    dipole_mul = args.dipole_mul

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dir_trained = '../dataset/' + data_dir + dataset_trained + '/' + 'create_data' + '_' + basis_set + '/'
    dir_predict = '../dataset/' + data_dir + dataset_predict + '/' + 'create_data' + '_' + basis_set + '/'

    field = '_'.join([basis_set + '/'])
    dataset_test = train.MyDataset(dir_predict + 'test_' + field)
    dataloader_test = train.mydataloader(dataset_test, batch_size=batch_size,
                                         num_workers=num_workers)

    with open(dir_trained + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)

    N_orbitals = len(orbital_dict)
    N_output = len(dataset_test[0][-2][0])

    model = train.D3_ImgNet(device, basis_set, N_orbitals, dim, seq_len,
                            task, transfer_flag, dropout=dropout).to(device)

    if transfer_flag:
        if task == 'Force':
            model.res_output = nn.Linear(256, 3).to(device)
        else:
            model.res_output = nn.Linear(256, 1).to(device)

    file_model = '../model/{}/best_model.pth'.format(task)
    model.load_state_dict(torch.load(file_model, map_location=device))

    tester = train.Tester(model, task=task, operation=operation, dipole_mul=dipole_mul)

    print('Start predicting for', dataset_predict, 'dataset using the \n'
          'pretrained model with', dataset_trained, 'dataset.\n'
          'The prediction result is saved in the output directory.\n'
          'Wait for a while...')

    MAE, prediction, _, _ = tester.test(dataloader_test)
    filename = ('../output/{}/prediction--'.format(task) + dataset_predict + '.txt')
    tester.save_prediction(prediction, filename)

    print('MAE:', MAE)
    print()
    print('The prediction has finished.')
