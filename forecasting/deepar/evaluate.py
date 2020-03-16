import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import model.net as net
from dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def evaluate(model, loss_fn, test_loader, params, sample=True):
    '''
    Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():

        summary_metric = {}
        raw_metrics = utils.init_metrics(sample=sample)

        # Test_loader: 
        # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
        # id_batch ([batch_size]): one integer denoting the time series id;
        # v ([batch_size, 2]): scaling factor for each window;
        # labels ([batch_size, train_window]): z_{1:T}.

        result_mu = []
        result_sigma = []
        for i, (test_batch, id_batch, v, labels) in enumerate(test_loader):
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
            id_batch = id_batch.unsqueeze(0).to(params.device)
            v_batch = v.to(torch.float32).to(params.device)
            labels = labels.to(torch.float32).to(params.device)
            batch_size = test_batch.shape[1]
            input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
            input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
            hidden = model.init_hidden(batch_size)
            cell = model.init_cell(batch_size)

            for t in range(params.test_predict_start):  # 先计算encoder部分
                # if z_t is missing, replace it by output mu from the last time step
                # 如果z_t缺失，用前一步预测值代替真实值作为输入
                zero_index = (test_batch[t, :, 0] == 0)
                if t > 0 and torch.sum(zero_index) > 0:
                    test_batch[t, zero_index, 0] = mu[zero_index]

                mu, sigma, hidden, cell = model(test_batch[t].unsqueeze(0), id_batch, hidden, cell)
                input_mu[:, t] = v_batch[:, 0] * mu + v_batch[:, 1]  # v_batch[:, 1] == 0, useless
                input_sigma[:, t] = v_batch[:, 0] * sigma
            
            if not params.one_step:
                test_batch[params.test_predict_start, :, 0] = mu
            
            # 计算decoder部分
            if sample:
                samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling=True, one_step=params.one_step)
                raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
            else:
                sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, one_step=params.one_step)
                raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)
            result_mu.append(sample_mu)
            result_sigma.append(sample_sigma)

        summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
        metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
        # print('test metrics: ' + metrics_string)
    return summary_metric, result_mu, result_sigma
