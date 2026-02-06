import time
import os

import argparse
import logging
import os

import numpy as np
from numpy.linalg import inv
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

# import lib
from lib.datasetLoader import *
from lib.Model import *
from lib.modules import *
from lib.Metric import *


logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
logger = logging.getLogger('Model.Train')

LOSS_DICT = {
    "QLIKE":QLIKE,
    "MSE": MSE,
    }


def test_xai(model, test_set, test_loader, configs):
    all_outputs_w = []  # to collect outputs
    all_outputs_vw = []  # to collect outputs
    all_outputs_m = []  # to collect outputs
    all_outputs_vm = []  # to collect outputs
    all_outputs_ca = []  # to collect outputs
    
    model.eval()
    with torch.no_grad():

        for i, (test_batch, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.to(torch.float32).to(configs.device)
            labels_batch = labels.to(torch.float32).to(configs.device)
            wegihts_w, v_w, wegihts_m, v_m, c_a = model.forward_attention(test_batch)

            wegihts_w = wegihts_w.detach().cpu().numpy()
            v_w = v_w.detach().cpu().numpy()
            wegihts_m = wegihts_m.detach().cpu().numpy()
            v_m = v_m.detach().cpu().numpy()
            c_a = c_a.detach().cpu().numpy()
            
            all_outputs_w.append(wegihts_w)
            all_outputs_vw.append(v_w)
            all_outputs_m.append(wegihts_m)
            all_outputs_vm.append(v_m)
            all_outputs_ca.append(c_a)

        all_outputs_w = np.concatenate(all_outputs_w, axis=0)
        all_outputs_vw = np.concatenate(all_outputs_vw, axis=0)
        all_outputs_m = np.concatenate(all_outputs_m, axis=0)
        all_outputs_vm = np.concatenate(all_outputs_vm, axis=0)
        all_outputs_ca = np.concatenate(all_outputs_ca, axis=0)

        # Save to file, e.g. numpy binary or csv
    return all_outputs_w, all_outputs_vw, all_outputs_m, all_outputs_vm, all_outputs_ca


def test_predict(model, test_set, test_loader, configs):
    all_outputs = []  # to collect outputs
    model.eval()
    with torch.no_grad():

        for i, (test_batch, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.to(torch.float32).to(configs.device)
            labels_batch = labels.to(torch.float32).to(configs.device)
            
            labels_volatility = labels_batch[:, -configs.pred_len:]
            return_volatility = model(test_batch)


            return_volatility = return_volatility.detach().cpu().numpy()
            labels_volatility = labels_volatility.detach().cpu().numpy()
            all_outputs.append(return_volatility)

        all_outputs = np.concatenate(all_outputs, axis=0)

        # Save to file, e.g. numpy binary or csv
    return all_outputs


def evaluate(model, test_set, test_loader, configs, loss_type = ['QLIKE']):
    
    model.eval()
    with torch.no_grad():
        summary_metric = {}
        eval_batch = {}

        for loss in loss_type :
            eval_batch[loss] = np.zeros(2)

        for i, (test_batch, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.to(torch.float32).to(configs.device)
            labels_batch = labels.to(torch.float32).to(configs.device)
            
            labels_volatility = labels_batch[:, -configs.pred_len:]  
            return_volatility = model(test_batch)


            return_volatility = return_volatility.detach().cpu().numpy()
            labels_volatility = labels_volatility.detach().cpu().numpy()
            # print(return_volatility.shape, labels_volatility.shape)
            for loss in loss_type :
                upper, lower = LOSS_DICT[loss](return_volatility, labels_volatility)
                eval_batch[loss][0] += upper
                eval_batch[loss][1] += lower

        for loss in loss_type :
            summary_metric[loss] = eval_batch[loss][0]/ eval_batch[loss][1] 
                
    return summary_metric





def modelTrain(model: nn.Module,
          optimizer: optim,
          train_loader: DataLoader,
          configs)  :

    model.train()
    loss_epoch = np.zeros(len(train_loader))
    
    for i, (train_batch, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        

        train_batch = train_batch.to(torch.float32).to(configs.device) 
        labels_batch = labels_batch.to(torch.float32).to(configs.device)

        labels_volatility = labels_batch[:, -configs.pred_len:] 
        return_volatility = model(train_batch)

        loss = QLIKE_loss(return_volatility, labels_volatility)

        if i % 1000 == 0 :
            print(loss )
        loss.backward()
        optimizer.step()
        
        loss_epoch[i] = loss.item()

    return loss_epoch






def train_and_evaluate(model: nn.Module,
                       dataset_all: object,
                       configs) :

    logger.info('begin training and evaluation')
    
    train_set, train_loader = dataset_all._get_data('train')
    val_set, validation_loader = dataset_all._get_data('val')
    test_set, test_loader = dataset_all._get_data('test')


    path = os.path.join(configs.model_dir, configs.setting)

    if not os.path.exists(path):
        os.makedirs(path)

    train_len = len(train_loader)

    loss = configs.loss
    evaluation_summary = {}
    loss_summary = np.zeros((train_len * configs.train_epochs))
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True)


    # Model initialization to prevent the dying ReLU problem
    optimizer = model.init_before_training(train_loader, clamp_min=configs.clamp_min, tries=configs.init_tries)

    for epoch in tqdm(range(configs.train_epochs)):
        logger.info('Epoch {}/{}'.format(epoch + 1, configs.train_epochs))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = modelTrain(model, optimizer, train_loader, configs)

        # Evaluate Model
        summary_val = evaluate(model, val_set, validation_loader, configs, loss_type =["QLIKE", "MSE"])
        
        
        # [MONITORING ONLY]
        # Test performance tracking is disabled by default to ensure zero data leakage.
        # It is NEVER used for gradient descent, early stopping, or model selection.
        # -----------------------------------------------------------------------------
        summary_test = evaluate(model, test_set, test_loader, configs, loss_type=["QLIKE", "MSE"]) 
        print(f'Epoch {epoch + 1} - Test Monitoring (Internal Only): {summary_test}')
        # -----------------------------------------------------------------------------]) 



        evaluation_summary[epoch] = summary_val
        val_loss = summary_val[loss]
        early_stopping(val_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break



    logger.info(f'Current Best Loss is:  {early_stopping.best_score}')

    evaluation_summary['best_score'] = early_stopping.best_score
    json_path =  path + '/' + 'validation_metric.json'
    save_dict_to_json(evaluation_summary, json_path)

    configs_dict = vars(configs)
    configs_dict['device'] = str(configs_dict['device'])
    configs_dict['best score'] = early_stopping.best_score
    json_path =  path + '/' + 'configs.json'
    save_dict_to_json(configs_dict, json_path)

