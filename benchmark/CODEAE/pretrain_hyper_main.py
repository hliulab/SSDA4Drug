import pandas as pd
import torch
import json
import os
import argparse
import random
import pickle
import itertools

import train_code_adv
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


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


def main(args, update_params_dict):
    train_fn = train_code_adv.train_code_adv
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join('model_save/train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params['unlabeled'].update(update_params_dict)
    param_str = dict_to_str(update_params_dict)
    x_source = pd.read_csv("../../data/SCAD" + '/' + args.drug + "/source_data/source_scaled" + args.gene + ".csv",
                           index_col=0)
    if not args.norm_flag:
        method_save_folder = os.path.join('model_save', f'{args.drug}', f'{args.method}')
    else:
        method_save_folder = os.path.join('model_save', f'{args.drug}', f'{args.method}_norm')
    training_params.update(
        {
            'device': device,
            'input_dim': x_source.T.shape[-1],
            'model_save_folder': os.path.join(method_save_folder, param_str),
            'es_flag': False,
            'retrain_flag': args.retrain_flag,
            'norm_flag': args.norm_flag
        })

    safe_make_dir(training_params['model_save_folder'])
    random.seed(2020)
    #######################################load data############################################################
    x_source = x_source.T
    x_source_train, x_source_test = train_test_split(x_source, test_size=0.1)

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
    with open(os.path.join(training_params['model_save_folder'], f'unlabel_train_history.pickle'),
              'wb') as f:
        for history in historys:
            pickle.dump(dict(history), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADSN training and evaluation')
    parser.add_argument('--method', dest='method', nargs='?', default='code_adv')
    parser.add_argument('--drug', type=str, default='Vorinostat',
                        help='choose drug,Can be PLX4720_451Lu,Gefitinib,Erlotinib,PLX4720,Vorinostat, Cetuximab,AR-42,and Etoposide.')
    parser.add_argument('--gene', type=str, default="_tp4k",
                        help='choose data type. Can be all , _tp4k or _gene. Default: _tp4k')
    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=True)

    train_group.add_argument('--pdtc', dest='pdtc_flag', action='store_true')
    train_group.add_argument('--no-pdtc', dest='pdtc_flag', action='store_false')
    parser.set_defaults(pdtc_flag=False)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    args = parser.parse_args()
    print(f'current config is {args}')
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
        main(args=args, update_params_dict=param_dict)
