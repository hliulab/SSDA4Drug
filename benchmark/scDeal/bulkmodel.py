import argparse
import time
import warnings
import os
import numpy as np
import pandas as pd
import torch
import trainers
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score,
                             classification_report, mean_squared_error, r2_score, roc_auc_score)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from models import (AEBase,PretrainedPredictor, PretrainedVAEPredictor, VAEBase)
import matplotlib
import random
seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

def run_main(args):
    if(args.checkpoint not in ["False","True"]):
        selected_model = args.checkpoint
        split_name = selected_model.split("/")[-1].split("_")
        para_names = (split_name[1::2])
        paras = (split_name[0::2])
        args.encoder_hdims = paras[4]
        args.predictor_h_dims = paras[5]
        args.bottleneck = int(paras[3])
        args.drug = paras[2]
        args.dropout = float(paras[7])
        args.dimreduce = paras[6]
    # Extract parameters
    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    select_drug = args.drug.upper()
    test_size = args.test_size
    valid_size = args.valid_size
    batch_size = args.batch_size
    encoder_hdims = args.encoder_h_dims.split(",")
    preditor_hdims = args.predictor_h_dims.split(",")
    reduce_model = args.dimreduce

    encoder_hdims = list(map(int, encoder_hdims) )
    preditor_hdims = list(map(int, preditor_hdims) )
    load_model = bool(args.load_source_model)

    para = str(args.bulk) + "_drug_"+str(args.drug)+"_bottle_"+str(args.bottleneck)+"_edim_"+str(args.encoder_h_dims)+"_pdim_"+str(args.predictor_h_dims)+"_model_"+reduce_model+"_dropout_"+str(args.dropout)+"_gene_"+str(args.printgene)+"_lr_"+str(args.lr)+"_mod_"+str(args.mod)+"_sam_"+str(args.sampling_method)    #(para)
    now = time.strftime("%Y-%m-%d-%H-%M-%S")

    for path in [args.log,args.bulk_model,args.bulk_encoder,'save/ori_result','save/figures']:
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")

    # Load model from checkpoint
    if(args.checkpoint not in ["False","True"]):
        para = os.path.basename(selected_model).split("_DaNN.pkl")[0]
        args.checkpoint = 'True'

    preditor_path = args.bulk_model + para 
    bulk_encoder = args.bulk_encoder + para

    #####################################################################################################################
    # Loading Source Data
    x_source = pd.read_csv("../../Datasets/processedData" + '/' + args.drug + "/source_data/source_scaled" + args.gene + ".csv",
                           index_col=0)
    y_source = pd.read_csv("../../Datasets/processedData/" + args.drug + "/source_data/source_meta_data.csv", index_col=0)

    x_source = x_source.T
    X_train_all, X_test, Y_train_all, Y_test = train_test_split(x_source, y_source, test_size=test_size, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_all, Y_train_all, test_size=valid_size,
                                                          random_state=42)
    if args.sampling_method == "weight":
        from collections import Counter
        class_sample_count = np.array([Counter(y_source['response'])[0] / len(y_source['response']),
                                       Counter(y_source['response'])[1] / len(y_source['response'])])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in Y_train["response"].values])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.reshape(-1)
        from torch.utils.data.sampler import WeightedRandomSampler
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)
    elif args.sampling_method == "smote":

        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline
        from imblearn.under_sampling import RandomUnderSampler
        over = SMOTE(sampling_strategy=0.5)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_train = X_train.T
        X_train, X_train = pipeline.fit_resample(X_train, Y_train)
        X_train = X_train.T
        sampler = None

    ##################################################################################################################

    dim_model_out = 2

    # Select the Training device
    if(args.device == "gpu"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    print(device)
    # Construct datasets and data loaders
    X_trainTensor = torch.FloatTensor(X_train.values).to(device)
    X_validTensor = torch.FloatTensor(X_valid.values).to(device)
    X_testTensor = torch.FloatTensor(X_test.values).to(device)

    Y_train = Y_train['response']
    Y_trainTensor = torch.LongTensor(Y_train.values).to(device)

    Y_valid = Y_valid['response']
    Y_validTensor = torch.LongTensor(Y_valid.values).to(device)

    # Preprocess data to tensor
    train_dataset = TensorDataset(X_trainTensor, X_trainTensor)
    valid_dataset = TensorDataset(X_validTensor, X_validTensor)

    X_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    X_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    # construct TensorDataset
    trainreducedDataset = TensorDataset(X_trainTensor, Y_trainTensor)
    validreducedDataset = TensorDataset(X_validTensor, Y_validTensor)

    trainDataLoader_p = DataLoader(dataset=trainreducedDataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    validDataLoader_p = DataLoader(dataset=validreducedDataset, batch_size=batch_size, shuffle=False)

    bulk_X_allTensor = torch.FloatTensor(x_source.values).to(device)

    y_source = y_source['response']
    bulk_Y_allTensor = torch.LongTensor(y_source.values).to(device)
    dataloaders_train = {'train':trainDataLoader_p,'val':validDataLoader_p}
    print("bulk_X_allRensor",bulk_X_allTensor.shape)
    if(str(args.pretrain)!="False"):
        dataloaders_pretrain = {'train':X_trainDataLoader,'val':X_validDataLoader}
        if reduce_model == "VAE":
            encoder = VAEBase(input_dim=x_source.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
        if reduce_model == 'AE':
            encoder = AEBase(input_dim=x_source.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
        if reduce_model == 'DAE':
            encoder = AEBase(input_dim=x_source.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)

        encoder.to(device)
        optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
        loss_function_e = nn.MSELoss()
        exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

        if(args.checkpoint != "False"):
            load = bulk_encoder
        else:
            load = False

        if reduce_model == "AE":
            encoder,loss_report_en = trainers.train_AE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                        optimizer=optimizer_e,loss_function=loss_function_e,load=load,
                                        n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=bulk_encoder + '.pkl')
        elif reduce_model == "VAE":
            encoder,loss_report_en = trainers.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                            optimizer=optimizer_e,load=False,
                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=bulk_encoder + '.pkl')
        if reduce_model == "DAE":
            encoder,loss_report_en = trainers.train_DAE_model(net=encoder,task='bulk',data_loaders=dataloaders_pretrain,
                                        optimizer=optimizer_e,loss_function=loss_function_e,load=load,
                                        n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=bulk_encoder + '.pkl')

        print("Pretrained finished")

    # Defined the model of predictor
    print("X_train.shape[1]:", X_train.shape[1])
    if reduce_model == "AE":
        model = PretrainedPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                                hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                                pretrained_weights=bulk_encoder + '.pkl',freezed=bool(args.freeze_pretrain),drop_out=args.dropout,drop_out_predictor=args.dropout)
    if reduce_model == "DAE":
        model = PretrainedPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                                hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                                pretrained_weights=bulk_encoder + '.pkl',freezed=bool(args.freeze_pretrain),drop_out=args.dropout,drop_out_predictor=args.dropout)
    elif reduce_model == "VAE":
        model = PretrainedVAEPredictor(input_dim=X_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims, 
                        hidden_dims_predictor=preditor_hdims,output_dim=dim_model_out,
                        pretrained_weights=bulk_encoder + '.pkl',freezed=bool(args.freeze_pretrain),z_reparam=bool(args.VAErepram),drop_out=args.dropout,drop_out_predictor=args.dropout)

    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    loss_function = nn.CrossEntropyLoss()

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    if(args.checkpoint != "False"):
        load = True
    else:
        load = False
    model,report = trainers.train_predictor_model(model,dataloaders_train,
                                            optimizer,loss_function,epochs,exp_lr_scheduler,load=load,save_path=preditor_path + '.pkl')

    dl_result = model(X_testTensor).detach().cpu().numpy()
    lb_results = np.argmax(dl_result,axis=1)
    pb_results = dl_result[:,1]

    Y_test = Y_test["response"]
    acc = accuracy_score(Y_test, lb_results)
    precision = precision_score(Y_test, lb_results)
    recall = recall_score(Y_test, lb_results)
    f1 = f1_score(Y_test, lb_results)
    auc = roc_auc_score(Y_test, pb_results)
    print("Bulk Data: {} ACC: {:.6f} AUC: {:.6f} F1: {:.6f} Precision: {:.6f} Recall: {:.6f}".format(args.drug, acc, auc, f1, precision, recall))

    report_dict = classification_report(Y_test, lb_results, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    ap_score = average_precision_score(Y_test, pb_results)
    auroc_score = roc_auc_score(Y_test, pb_results)

    report_df['auroc_score'] = auroc_score
    report_df['ap_score'] = ap_score

    report_df.to_csv("save/logs/" + reduce_model + select_drug + now + '_report.csv')

    model = DummyClassifier(strategy='stratified')
    model.fit(X_train, Y_train)
    yhat = model.predict_proba(X_test)
    naive_probs = yhat[:, 1]
    print("bulk_model finished")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--drug', type=str, default='Vorinostat',
                        help='choose drug,Can be PLX4720_451Lu,Gefitinib,Erlotinib,PLX4720,Vorinostat, Cetuximab,AR-42,and Etoposide.')
    parser.add_argument('--gene', type=str, default="_tp4k",
                        help='choose data type. Can be all , _tp4k or _gene. Default: _tp4k')
    parser.add_argument('--result', type=str, default='save/results/result_',help='Path of the training result report files')
    parser.add_argument('--missing_value', type=int, default=1,help='The value filled in the missing entry in the drug screening annotation, default: 1')
    parser.add_argument('--test_size', type=float, default=0.2,help='Size of the test set for the bulk model traning, default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2,help='Size of the validation set for the bulk model traning, default: 0.2')
    # train
    parser.add_argument('--sampling_method', type=str, default="weight",
                        help='choose sampling type. Can be weight or smote. Default: weight')
    parser.add_argument('--device', type=str, default="gpu",help='Device to train the model. Can be cpu or gpu. Deafult: cpu')
    parser.add_argument('--bulk_encoder','-e', type=str, default='save/bulk_encoder_my/',help='Path of the pre-trained encoder in the bulk level')
    parser.add_argument('--pretrain', type=str, default="True",help='Whether to perform pre-training of the encoder,str. False: do not pretraing, True: pretrain. Default: True')
    parser.add_argument('--lr', type=float, default=0.5,help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=50,help='Number of epoches training. Default: 500')
    parser.add_argument('--batch_size', type=int, default=200,help='Number of batch size when training. Default: 200')
    parser.add_argument('--bottleneck', type=int, default=256,help='Size of the bottleneck layer of the model. Default: 32')
    parser.add_argument('--dimreduce', type=str, default="DAE",help='Encoder model type. Can be AE or VAE. Default: AE')
    parser.add_argument('--freeze_pretrain', type=int, default=0,help='Fix the prarmeters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0')
    parser.add_argument('--encoder_h_dims', type=str, default="512,256",help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--predictor_h_dims', type=str, default="256,128",help='Shape of the predictor. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 16,8')
    parser.add_argument('--VAErepram', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='False',help='Load weight from checkpoint files, can be True,False, or file path. Checkpoint files can be paraName1_para1_paraName2_para2... Default: True')

    # misc
    parser.add_argument('--bulk_model', '-p',  type=str, default='save/bulk_pre_my/',help='Path of the trained prediction model in the bulk level')
    parser.add_argument('--log', '-l',  type=str, default='save/logs/',help='Path of training log')
    parser.add_argument('--load_source_model',  type=int, default=0,help='Load a trained bulk level or not. 0: do not load, 1: load. Default: 0')
    parser.add_argument('--mod', type=str, default='new',help='Embed the cell type label to regularized the training: new: add cell type info, ori: do not add cell type info. Default: new')
    parser.add_argument('--printgene', type=str, default='T',help='Print the cirtical gene list: T: print. Default: T')
    parser.add_argument('--dropout', type=float, default=0.1,help='Dropout of neural network. Default: 0.3')
    parser.add_argument('--bulk', type=str, default='integrate',help='Selection of the bulk database.integrate:both dataset. old: GDSC. new: CCLE. Default: integrate')
    warnings.filterwarnings("ignore")

    args, unknown = parser.parse_known_args()
    matplotlib.use('Agg')

    run_main(args)

