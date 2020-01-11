import argparse
import logging
import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from pytorchtools import EarlyStopping
from tqdm import tqdm

import utils
import model.net as net
from evaluate import evaluate
from dataloader import *


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          test_loader: DataLoader,
          params: utils.Params,
          epoch: int) -> float:
    '''
    Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        train_loader: load train data and labels
        test_loader: load test data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()
    loss_epoch = np.zeros(len(train_loader))

    # Train_loader:
    # train_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
    # idx ([batch_size]): one integer denoting the time series id;
    # labels_batch ([batch_size, train_window]): z_{1:T}.
    
    for i, (train_batch, idx, labels_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_size = train_batch.shape[0]

        train_batch = train_batch.permute(1, 0, 2).to(torch.float32).to(params.device)  # not scaled
        labels_batch = labels_batch.permute(1, 0).to(torch.float32).to(params.device)  # not scaled
        idx = idx.unsqueeze(0).to(params.device)

        loss = torch.zeros(1, device=params.device)
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)

        for t in range(params.train_window):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (train_batch[t, :, 0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                train_batch[t, zero_index, 0] = mu[zero_index]
            mu, sigma, hidden, cell = model(train_batch[t].unsqueeze_(0).clone(), idx, hidden, cell)
            loss += loss_fn(mu, sigma, labels_batch[t])

        loss.backward()
        optimizer.step()
        loss = loss.item() / params.train_window  # loss per timestep
        loss_epoch[i] = loss

    return loss_epoch


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       val_loader: DataLoader,
                       optimizer: optim, loss_fn,
                       params: utils.Params,
                       restore_file: str = None) -> None:  # 箭头无意义，提示函数返回值为None
    '''
    Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        test_loader: load test data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''

    print('Begin training')
    print(model)
    train_len = len(train_loader)

    loss_summary = np.zeros((train_len * params.num_epochs))
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(params.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, params.num_epochs))

        # train
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader, 
                                                                        test_loader, params, epoch)
        print(f"train_loss: {np.mean(loss_summary[epoch * train_len:(epoch + 1) * train_len])}")

        # evaluate
        val_metrics = evaluate(model, loss_fn, val_loader, params, sample=params.sampling)
        test_metrics = evaluate(model, loss_fn, test_loader, params, sample=params.sampling)

        # early stop
        early_stopping(val_metrics['test_loss'], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # save weights
        # utils.save_checkpoint({'epoch': epoch + 1,
        #                        'state_dict': model.state_dict(),
        #                        'optim_dict': optimizer.state_dict()},
        #                       epoch=epoch,
        #                       is_best=is_best,
        #                       checkpoint=params.model_dir)
