import argparse
import time
import warnings
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
import utils
import trainer
import model as m
import matplotlib
import random
import itertools

# Set a random number seed to facilitate experiment reproduction
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
            args.encoder_h_dims) + "_Predim_" + str(args.predictor_h_dims) + "_dropout_" + str(args.dropout) + "_lr_" + str(
            args.lr) + "_bs_" + str(args.batch_size) # (para)
        print("para:", para)
        if args.gene == "":
            gene = "all"
        elif args.gene == "_tp4k":
            gene = "tp4k"

        ####################################################################################################################
        # data loading
        ####################################################################################################################
        # Loading Source Data
        x_source = pd.read_csv("./baselineData/SCAD" + '/' + args.drug + "/source_data/source_scaled" + args.gene + ".csv",
                               index_col=0)
        y_source = pd.read_csv("./baselineData/SCAD" + '/' + args.drug + "/source_data/source_meta_data.csv",
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
        x_target = pd.read_csv("./baselineData/SCAD" + '/' + args.drug + "/target_data/target_scaled" + args.gene + ".csv",
                               index_col=0)
        y_target = pd.read_csv("./baselineData/SCAD" + '/' + args.drug + "/target_data/target_meta_data.csv",
                               index_col=0)

        x_target = x_target.T
        x_train_target, x_val_target, y_train_target, y_val_target = train_test_split(x_target, y_target,
                                                                                      test_size=0.2,
                                                                                      random_state=i)
        x_train_target = x_train_target.T
        x_val_target = x_val_target.T

        #############train#####################
        random.seed(i)
        sample_0_train = random.sample(y_train_target[y_train_target["response"] == 0].index.tolist(), args.n)
        sample_1_train = random.sample(y_train_target[y_train_target["response"] == 1].index.tolist(), args.n)
        y_train_labeled = y_train_target.loc[sample_0_train + sample_1_train]
        x_train_labeled = x_train_target.loc[:, y_train_labeled.index.tolist()]
        y_train_unlabeled = y_train_target.drop(sample_0_train + sample_1_train, axis=0)
        x_train_unlabeled = x_train_target.loc[:, y_train_unlabeled.index.tolist()]

        #############valid#####################
        random.seed(i)
        sample_0_val = random.sample(y_val_target[y_val_target["response"] == 0].index.tolist(), args.n)
        sample_1_val = random.sample(y_val_target[y_val_target["response"] == 1].index.tolist(), args.n)
        y_val_labeled = y_val_target.loc[sample_0_val + sample_1_val]
        x_val_labeled = x_val_target.loc[:, y_val_labeled.index.tolist()]
        y_val_unlabeled = y_val_target.drop(sample_0_val + sample_1_val, axis=0)
        x_val_unlabeled = x_val_target.loc[:, y_val_unlabeled.index.tolist()]

        # Loading Unlabeled Target Data
        # train
        target_train_unlabeled = utils.create_dataset(x=x_train_unlabeled, y=y_train_unlabeled, batch_size=batch_size, shuffle=True)

        # valid
        target_valid_unlabeled = utils.create_dataset(x=x_val_unlabeled, y=y_val_unlabeled, batch_size=batch_size, shuffle=False)

        dataloader_unlabeled_target = {'train': target_train_unlabeled, 'val': target_valid_unlabeled}

        # Loading labeled Target Data
        # train
        target_train_labeled = utils.create_dataset(x=x_train_labeled, y=y_train_labeled, batch_size=batch_size, shuffle=True)

        # valid
        target_valid_labeled = utils.create_dataset(x=x_val_labeled, y=y_val_labeled, batch_size=batch_size, shuffle=False)

        dataloader_labeled_target = {'train': target_train_labeled, 'val': target_valid_labeled}

        # test
        y_test_target = y_target.drop(y_train_labeled.index, axis=0)
        x_target = x_target.T
        x_test_target = x_target.loc[:, y_test_target.index.tolist()]

        test_dataset = utils.create_dataset(x=x_test_target, y=y_test_target, batch_size=batch_size, shuffle=False)

        ####################################################################################################################
        # data loading finished
        ####################################################################################################################
        bulk_tasks, sc_tasks = utils.cell_dim(drug=args.drug, gene=args.gene)
        dim_model_out = 2

        for path in [args.umap_path + args.drug, args.sc_all + args.drug, args.result + '/' + args.method]:
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
        if args.encoder == "DAE":
            encoder = m.DAE(input_dim=sc_tasks['pathway'], fc_dim=256,
                            AE_input_dim=bulk_tasks['expression'], AE_h_dims=encoder_hdims,
                            pretrained_weights=None,drop=dropout)
            encoder.to(device)
        elif args.encoder == "MLP":
            encoder = m.MLP(input_dim=sc_tasks["expression"], latent_dim=sc_tasks["pathway"], h_dims=encoder_hdims,
                            drop_out=dropout)
            encoder.to(device)
        else:
            raise ValueError('encoder cannot be recognized.')

        # predictor
        predictor = m.Predictor(input_dim=sc_tasks["pathway"],
                                output_dim=32,
                                drop_out=dropout)
        predictor.to(device)

        Predictor_adentropy = m.Predictor_adentropy(num_class=dim_model_out, inc=32)
        Predictor_adentropy.to(device)

        fgm = m.FGM([encoder,predictor,Predictor_adentropy])

        loss_c = nn.CrossEntropyLoss()
        loss_e = nn.MSELoss()
        optimizer_d = torch.optim.Adagrad(
            itertools.chain(encoder.parameters(), predictor.parameters(), Predictor_adentropy.parameters()), lr=lr)
        if args.encoder == "DAE":
            encoder_f, predictor_f, Predictor_adentropy_f = trainer.train_semi_dae(fgm, encoder, predictor, Predictor_adentropy,
                                                                                   dataloader_source,
                                                                                   dataloader_unlabeled_target,
                                                                                   dataloader_labeled_target,
                                                                                   args.method,
                                                                                   optimizer_d,
                                                                                   loss_c,
                                                                                   loss_e, epochs, start_epoch=0,
                                                                                   save_path=sc_all_path + ".pkl",
                                                                                   device=device, auc_path=result_path)
            torch.save({
                'encoder_state_dict': encoder_f.state_dict(),
                'predictor_state_dict': predictor_f.state_dict(),
                'Predictor_adentropy_state_dict': Predictor_adentropy_f.state_dict()
            }, sc_all_path + '.pth')
        elif args.encoder == "MLP":
            encoder_f, predictor_f, Predictor_adentropy_f = trainer.train_semi_mlp(encoder, predictor, Predictor_adentropy,
                                                                                   dataloader_source,
                                                                                   dataloader_unlabeled_target,
                                                                                   dataloader_labeled_target,
                                                                                   args.method,
                                                                                   optimizer_d,
                                                                                   loss_c,
                                                                                   loss_e, epochs, start_epoch=0,
                                                                                   save_path=sc_all_path + ".pkl",
                                                                                   device=device, auc_path=result_path)
            torch.save({
                'encoder_state_dict': encoder_f.state_dict(),
                'predictor_state_dict': predictor_f.state_dict(),
                'Predictor_adentropy_state_dict': Predictor_adentropy_f.state_dict()
            }, sc_all_path + 'pth')

        if args.gene == "_tp4k":
            Gene = "tp4k"
        elif args.gene == "":
            Gene = "all"
        print("Sampling Method:", args.sampling_method)
        print("Gene:", Gene)
        print("Method:", args.method)
        print("Shot-Method:", args.shot_method)
        print("Drug:", args.drug)
        loss_class = nn.CrossEntropyLoss()
        print("Transfer " + args.method + " finished")
        test_model = m.Test_Double_Model(predictor=predictor_f, encoder=encoder_f, adentropy_p=Predictor_adentropy_f)
        test_file = "./baselineData/SCAD" + '/' + args.drug + '/' + str(args.n) + '.tsv'
        trainer.test_shot(test_model, test_dataset, loss_class, device, test_file,i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--path', type=str, default='data/SCAD',
                        help='Path of the dataset used for model training')
    parser.add_argument('--drug', type=str, default='PLX4720',
                        help='choose drug,Can be PLX4720_451Lu,Gefitinib,Erlotinib,PLX4720,Vorinostat, Cetuximab,AR-42,and Etoposide.')
    parser.add_argument('--n', type=int, default=3, help='Number of shot. Default: 3')
    parser.add_argument('--gene', type=str, default="_tp4k",
                        help='choose data type. Can be all or _tp4k. Default: _tp4k')
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