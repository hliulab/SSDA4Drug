import pandas as pd
import torch
import json
import os
import argparse
import random
import pickle
from collections import defaultdict
import itertools
import numpy as np
import data_process

import fine_tuning
from copy import deepcopy
import train_code_adv
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold


def generate_encoded_features(encoder, dataloader, normalize_flag=False):
    """
    :param normalize_flag:
    :param encoder:
    :param dataloader:
    :return:
    """
    encoder.eval()
    raw_feature_tensor = dataloader.dataset.tensors[0].cpu()
    label_tensor = dataloader.dataset.tensors[1].cpu()

    encoded_feature_tensor = encoder.cpu()(raw_feature_tensor)
    if normalize_flag:
        encoded_feature_tensor = torch.nn.functional.normalize(encoded_feature_tensor, p=2, dim=1)
    return encoded_feature_tensor, label_tensor


def load_pickle(pickle_file):
    data = []
    with open(pickle_file, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    return data


def wrap_training_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])

    return aux_dict


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')


def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])


def main(args, drug, update_params_dict):
    for i in range(50):
        train_fn = train_code_adv.train_code_adv
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(os.path.join('model_save/train_params.json'), 'r') as f:
            training_params = json.load(f)

        training_params['unlabeled'].update(update_params_dict)
        param_str = dict_to_str(update_params_dict)

        if not args.norm_flag:
            method_save_folder = os.path.join('model_save', f'{args.drug}', args.method)
        else:
            method_save_folder = os.path.join('model_save', f'{args.drug}', f'{args.method}_norm')

        x_source = pd.read_csv("../../Datasets/processedData" + '/' + args.drug + "/source_data/source_scaled" + args.gene + ".csv",
                               index_col=0)
        y_source = pd.read_csv("../../Datasets/processedData" + '/' + args.drug + "/source_data/source_meta_data.csv",
                               index_col=0)
        x_target = pd.read_csv("../../Datasets/processedData" + '/' + args.drug + "/target_data/target_scaled" + args.gene + ".csv",
                               index_col=0)
        y_target = pd.read_csv("../../Datasets/processedData" + '/' + args.drug + "/target_data/target_meta_data.csv",
                               index_col=0)
        training_params.update(
            {
                'device': device,
                'input_dim': x_source.T.shape[-1],
                'model_save_folder': os.path.join(method_save_folder, param_str),
                'es_flag': False,
                'retrain_flag': args.retrain_flag,
                'norm_flag': args.norm_flag
            })
        if args.pdtc_flag:
            task_save_folder = os.path.join(f'{method_save_folder}', args.metric, 'pdtc', drug)
        else:
            task_save_folder = os.path.join(f'{method_save_folder}', args.metric, drug)

        safe_make_dir(training_params['model_save_folder'])
        safe_make_dir(task_save_folder)

        #######################################load data############################################################
        x_source = x_source.T
        x_source_train, x_source_test = train_test_split(x_source, test_size=0.2, random_state=i)

        train_source_dateset = TensorDataset(
            torch.from_numpy(x_source_train.values.astype('float32')))
        test_source_dateset = TensorDataset(
            torch.from_numpy(x_source_test.values.astype('float32')))
        source_dataset = TensorDataset(
            torch.from_numpy(x_source.values.astype('float32'))
        )
        train_source_dataloader = DataLoader(train_source_dateset,
                                             batch_size=training_params['unlabeled']['batch_size'],
                                             shuffle=True, drop_last=True)
        test_source_dataloader = DataLoader(test_source_dateset,
                                            batch_size=training_params['unlabeled']['batch_size'],
                                            shuffle=True)
        source_dataloader = DataLoader(source_dataset,
                                       batch_size=training_params['unlabeled']['batch_size'],
                                       shuffle=True,
                                       drop_last=True
                                       )
        s_dataloaders, t_dataloaders = (source_dataloader, test_source_dataloader), (
            source_dataloader, test_source_dataloader)

        # start unlabeled training
        encoder, historys = train_fn(s_dataloaders=s_dataloaders,
                                     t_dataloaders=t_dataloaders,
                                     **wrap_training_params(training_params, type='unlabeled'))
        if args.retrain_flag:
            with open(os.path.join(training_params['model_save_folder'], f'unlabel_train_history.pickle'),
                      'wb') as f:
                for history in historys:
                    pickle.dump(dict(history), f)

        ft_evaluation_metrics = defaultdict(list)
        #############################################load data#########################################
        train_source_dataloader, test_source_dataloader, test_target_dataloader = data_process.get_labeled_dataloaders(x_source, y_source, x_target, y_target,
                                                                            batch_size=training_params['labeled']['batch_size'],i=i)
        ft_encoder = deepcopy(encoder)
        print(train_source_dataloader.dataset.tensors[1].sum())
        print(test_source_dataloader.dataset.tensors[1].sum())
        print(test_target_dataloader.dataset.tensors[1].sum())

        target_classifier, ft_historys = fine_tuning.fine_tune_encoder(
            encoder=ft_encoder,
            train_dataloader=train_source_dataloader,
            val_dataloader=test_source_dataloader,
            test_dataloader=test_target_dataloader,
            seed=i,
            normalize_flag=args.norm_flag,
            metric_name=args.metric,
            task_save_folder=task_save_folder,
            **wrap_training_params(training_params, type='labeled')
        )
        ft_evaluation_metrics['best_index'].append(ft_historys[-2]['best_index'])
        for metric in ['auroc', 'acc', 'aps', 'f1', 'auprc']:
            ft_evaluation_metrics[metric].append(ft_historys[-1][metric][ft_historys[-2]['best_index']])

        test_file = task_save_folder + '/' + drug + '.tsv'
        with open(test_file, 'a') as p:
            p.write("num\t{}\tAUC\t{}\tAUPR\t{}\n".format(i, ft_historys[-1]["auroc"][ft_historys[-2]['best_index']], ft_historys[-1]["aps"][ft_historys[-2]['best_index']]))
        with open(os.path.join(task_save_folder, f'{param_str}_ft_evaluation_results.json'), 'w') as f:
            json.dump(ft_evaluation_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADSN training and evaluation')
    parser.add_argument('--method', dest='method', nargs='?', default='code_adv')
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc', choices=['auroc', 'auprc'])
    parser.add_argument('--drug', type=str, default='Afatinib',
                        help='choose drug,Can be PLX4720_451Lu,Gefitinib,Erlotinib,PLX4720,Vorinostat, Cetuximab,AR-42,and Etoposide.')
    parser.add_argument('--gene', type=str, default="_tp4k",
                        help='choose data type. Can be all , _tp4k or _gene. Default: _tp4k')
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--a_thres', dest='a_thres', nargs='?', type=float, default=None)
    parser.add_argument('--d_thres', dest='days_thres', nargs='?', type=float, default=None)

    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=False)

    train_group.add_argument('--pdtc', dest='pdtc_flag', action='store_true')
    train_group.add_argument('--no-pdtc', dest='pdtc_flag', action='store_false')
    parser.set_defaults(pdtc_flag=False)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    args = parser.parse_args()

    params_grid = {
        "pretrain_num_epochs": [0],
        "train_num_epochs": [100],
        "dop": [0.0]
    }

    if args.method not in ['code_adv', 'adsn', 'adae', 'dsnw']:
        params_grid.pop('pretrain_num_epochs')

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param_dict in update_params_dict_list:
        main(args=args, drug=args.drug, update_params_dict=param_dict)
