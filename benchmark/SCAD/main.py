import argparse
import time
import warnings
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import modules as m
import matplotlib
import random
import itertools
sys.path.append("/data/hks/idea_1/")
import utils
import trainer

# 设置随机数种子，方便实验复现
seed = 42
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# slower and more reproducible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict_label(XTestCells, gen_model, map_model, device):
    """
    Inputs:
    :param XTestCells - X_target_test
    :param gen_model
    :param map_model

    Output:
        - Predicted (binary) labels (Y for input target test data)
    """

    gen_model.eval()
    gen_model.to(device)
    map_model.eval()  ## predictor
    map_model.to(device)

    F_xt_test = gen_model(XTestCells.to(device))

    yhatt_test = map_model(F_xt_test)
    return yhatt_test


def evaluate_model(XTestCells, YTestCells, gen_model, map_model, device):
    """
    Inputs:
    :param XTestCells - single cell test data
    :param YTestCells - true class labels (binary) for single cell test data
    :param path_to_models - path to the saved models from training

    Outputs:
        - test loss
        - test accuracy (AUC)
    """
    XTestCells = XTestCells.to(device)
    YTestCells = YTestCells.view(-1,1).to(device)
    y_predicted = predict_label(XTestCells, gen_model, map_model,device)

    # #LOSSES
    C_loss_eval = torch.nn.BCELoss()
    closs_test = C_loss_eval(y_predicted, YTestCells.float())

    YTestCells = YTestCells.to(device)

    yt_true_test = YTestCells.view(-1,1)
    yt_true_test = yt_true_test.cpu()
    y_predicted = y_predicted.cpu()

    print("{} and {}".format(yt_true_test, y_predicted))   # yt_true_test = binary, y_predicted = float/decimal
    AUC_test = roc_auc_score(yt_true_test.detach().numpy(), y_predicted.detach().numpy())

    # Precision Recall
    APR_test = average_precision_score(yt_true_test.detach().numpy(), y_predicted.detach().numpy())

    return closs_test, AUC_test, APR_test, y_predicted

def main(args):
    for i in range(50):
        epochs = args.epochs
        lr = args.lr
        dropout = args.dropout
        batch_size = args.batch_size

        preditor_hdims = args.predictor_h_dims.split(",")
        preditor_hdims = list(map(int, preditor_hdims))

        encoder_hdims = args.encoder_h_dims.split(",")
        encoder_hdims = list(map(int, encoder_hdims))

        para = "_drug_" + str(args.drug) + "_method_" + str(args.method) + "_gene_" + str(args.gene) + "_DAEdim_" + str(
            args.encoder_h_dims) + "_Predim_" + str(args.predictor_h_dims) + "_dropout_" + str(
            args.dropout) + "_lr_" + str(
            args.lr) + "_bs_" + str(args.batch_size)  # (para)
        print("para:", para)
        if args.gene == "":
            gene = "all"
        elif args.gene == "_tp4k":
            gene = "tp4k"
        elif args.gene == "_gene":
            gene = "gene"

        ####################################################################################################################
        # data loading
        ####################################################################################################################
        # Loading Source Data
        x_source = pd.read_csv("../../data/SCAD" + '/' + args.drug + "/source_data/source_scaled" + args.gene + ".csv",
                               index_col=0)
        y_source = pd.read_csv("../../data/SCAD" + '/' + args.drug + "/source_data/source_meta_data.csv",
                               index_col=0)
        # train
        x_source = x_source.T
        x_train_source, x_val_source, y_train_source, y_val_source = train_test_split(x_source, y_source,
                                                                                      test_size=0.2, random_state=i)
        x_train_source = x_train_source.T
        x_val_source = x_val_source.T

        from collections import Counter
        class_sample_count = np.array([Counter(y_source['response'])[0] / len(y_source['response']),
                                       Counter(y_source['response'])[1] / len(y_source['response'])])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train_source["response"].values])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.reshape(-1)
        from torch.utils.data.sampler import WeightedRandomSampler
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)

        source_train = utils.create_dataset(x=x_train_source, y=y_train_source, batch_size=batch_size, shuffle=False,
                                            sampler=sampler)

        # valid
        source_valid = utils.create_dataset(x=x_val_source, y=y_val_source, batch_size=batch_size, shuffle=False)

        dataloader_source = {'train': source_train, 'val': source_valid}

        ###################################################################################################################
        # Loading Target Data
        x_target = pd.read_csv("../../data/SCAD" + '/' + args.drug + "/target_data/target_scaled" + args.gene + ".csv",
                               index_col=0)
        y_target = pd.read_csv("../../data/SCAD" + '/' + args.drug + "/target_data/target_meta_data.csv",
                               index_col=0)

        x_target = x_target.T
        x_train_target, x_val_target, y_train_target, y_val_target = train_test_split(x_target, y_target,
                                                                                      test_size=0.2,
                                                                                      random_state=i)
        x_train_target = x_train_target.T
        x_val_target = x_val_target.T
        target_train = utils.create_dataset(x=x_train_target, y=y_train_target, batch_size=batch_size, shuffle=False)
        target_valid = utils.create_dataset(x=x_val_target, y=y_val_target, batch_size=batch_size, shuffle=False)
        dataloader_target = {'train': target_train, 'val': target_valid}
        ####################################################################################################################
        # data loading finished
        ####################################################################################################################
        bulk_tasks, sc_tasks = utils.cell_dim(drug=args.drug, gene=args.gene)
        dim_model_out = 2

        for path in [args.umap_path + args.drug, args.sc_all + args.drug, args.result + '/' + args.method, "./results/SCAD" + '/' + args.drug]:
            if not os.path.exists(path):
                # Create a new directory because it does not exist
                os.makedirs(path)
                print("The new directory is created!")
        # target model path
        sc_all_path = args.sc_all + args.drug + '/' + para
        result_path = args.result + '/' + args.method + '/' + args.drug + '/'
        # Select the Training device
        if (args.device == "gpu"):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("device:", device)

        # encoder
        encoder = m.FX(h_dim=sc_tasks['pathway'], input_dim=bulk_tasks['expression'], dropout_rate=dropout, z_dim=32)
        encoder.to(device)

        # predictor
        predictor = m.MTLP(h_dim=sc_tasks["pathway"], z_dim=32, dropout_rate=dropout)
        predictor.to(device)

        # discriminator
        discriminator = m.Discriminator(dropout_rate=dropout, h_dim=sc_tasks["pathway"], z_dim=32)
        discriminator.to(device)

        loss_c = torch.nn.BCELoss()
        optimizer_d = torch.optim.Adagrad(itertools.chain(encoder.parameters(),
                                                          predictor.parameters(), discriminator.parameters()), lr=lr)

        encoder_f, predictor_f, discriminator_f = trainer.train_scad(encoder, predictor, discriminator,
                                                                     dataloader_source,
                                                                     dataloader_target,
                                                                     optimizer_d,
                                                                     loss_c, epochs, start_epoch=0,
                                                                     save_path=sc_all_path + ".pkl",
                                                                     device=device, auc_path=result_path)
        torch.save({
            'encoder_state_dict': encoder_f.state_dict(),
            'predictor_state_dict': predictor_f.state_dict(),
            'discriminator_state_dict': discriminator_f.state_dict()
        }, sc_all_path + '.pth')
        if args.gene == "_tp4k":
            Gene = "tp4k"
        elif args.gene == "":
            Gene = "all"
        elif args.gene == "_gene":
            Gene = "gene"
        print("Sampling Method:", args.sampling_method)
        print("Gene:", Gene)
        print("Method:", args.method)
        print("Drug:", args.drug)
        loss_class = torch.nn.BCELoss()
        print("Transfer " + args.method + " finished")
        x_target = torch.FloatTensor(x_target.values)
        y_target = y_target['response']
        y_target = torch.LongTensor(y_target.values)
        result = evaluate_model(x_target, y_target, encoder_f, predictor_f, device)

        test_file = "./results/SCAD" + '/' + args.drug + '/' + args.drug + '.tsv'
        with open(test_file, 'a') as p:
            p.write("num\t{}\tAUC\t{}\tAUPR\t{}\n".format(i, result[1], result[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--path', type=str, default='data/SCAD',
                        help='Path of the dataset used for model training')
    parser.add_argument('--drug', type=str, default='Vorinostat',
                        help='choose drug,Can be PLX4720_451Lu,Gefitinib,Erlotinib,PLX4720,Vorinostat, Cetuximab,AR-42,and Etoposide.')
    parser.add_argument('--gene', type=str, default="_tp4k",
                        help='choose data type. Can be all , _tp4k or _gene. Default: _tp4k')
    parser.add_argument('--sampling_method', type=str, default="weight",
                        help='choose sampling type. Can be weight or smote. Default: weight')
    parser.add_argument('--shot_method', type=str, default="3-shot",
                        help='choose shot type. Can be n-shot. Default: 3-shot')
    # save
    parser.add_argument('--umap_path', type=str, default='save/figure/',
                        help='Path of the model in the bulk level')
    parser.add_argument('--result', type=str, default='save/results/sc/',
                        help='Path of the training result report files')
    parser.add_argument('--sc_all', type=str, default='save/sc/all_path/',
                        help='Path of the model in the sc level')

    # train
    parser.add_argument('--device', type=str, default="gpu",
                        help='Device to train the model. Can be cpu or gpu. Default: gpu')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epoches training. Default: 500')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size when training. Default: 200')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout of neural network. Default: 0.3')
    parser.add_argument('--fix_source', type=int, default=0,
                        help='Fix the parameters in the bulk model. 0: do not freeze, 1: freeze. Default: 0')
    # model
    parser.add_argument('--encoder', type=str, default="DAE",
                        help='choose model type. Can be MLP or DAE. Default: MLP')
    parser.add_argument('--method', type=str, default="adv",
                        help='choose model type. Can be DANN or CDAN. Default: DANN')
    parser.add_argument('--encoder_h_dims', type=str, default="512,256", help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                                Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--bottleneck', type=int, default=256,
                        help='Size of the bottleneck layer of the model. Default: 128')
    parser.add_argument('--predictor_h_dims', type=str, default="64,32", help='Shape of the predictor. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 128')
    parser.add_argument('--load_source_model', type=int, default=0,
                        help='Load a trained bulk level or not. 0: do not load, 1: load. Default: 0')
    parser.add_argument('--printpathway', type=str, default='F',
                        help='Print the cirtical pathway list: T: print. Default: T')
    warnings.filterwarnings("ignore")

    args, unknown = parser.parse_known_args()
    matplotlib.use('Agg')

    main(args)
