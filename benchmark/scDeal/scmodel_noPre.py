#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import DaNN.mmd as mmd
import trainers as t
from models import (AEBase, DaNN, PretrainedPredictor,
                    PretrainedVAEPredictor, VAEBase, CVAEBase)
from scipy.spatial import distance_matrix, minkowski_distance, distance
import random
seed = 42
torch.manual_seed(seed)
#np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#from transformers import *
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TargetModel(nn.Module):
    def __init__(self, source_predcitor,target_encoder):
        super(TargetModel, self).__init__()
        self.source_predcitor = source_predcitor
        self.target_encoder = target_encoder

    def forward(self, X_target,C_target=None):

        if(type(C_target)==type(None)):
            x_tar = self.target_encoder.encode(X_target)
        else:
            x_tar = self.target_encoder.encode(X_target,C_target)
        y_src = self.source_predcitor.predictor(x_tar)
        return y_src
        
def run_main(args):
    for i in range(50):
        if(args.checkpoint not in ["False","True"]):
            selected_model = args.checkpoint
            split_name = selected_model.split("/")[-1].split("_")
            para_names = (split_name[1::2])
            paras = (split_name[0::2])
            args.bulk_h_dims = paras[4]
            args.sc_h_dims = paras[4]
            args.predictor_h_dims = paras[5]
            args.bottleneck = int(paras[3])
            # args.drug = 'ERLOTINIB'
            args.drug = paras[2]
            args.dropout = float(paras[7])
            args.dimreduce = paras[6]

        # Load parameters from args
        epochs = args.epochs
        dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512

        test_size = args.test_size
        select_drug = args.drug.upper()
        freeze = args.freeze_pretrain
        valid_size = args.valid_size
        batch_size = args.batch_size
        encoder_hdims = args.bulk_h_dims.split(",")
        encoder_hdims = list(map(int, encoder_hdims))
        reduce_model = args.dimreduce
        predict_hdims = args.predictor_h_dims.split(",")
        predict_hdims = list(map(int, predict_hdims))
        mod = args.mod

        # Merge parameters as string for saving model and logging
        para = str(args.bulk)+"_drug_"+str(args.drug)+"_bottle_"+str(args.bottleneck)+"_edim_"+str(args.bulk_h_dims)+"_pdim_"+str(args.predictor_h_dims)+"_model_"+reduce_model+"_dropout_"+str(args.dropout)+"_gene_"+str(args.printgene)+"_lr_"+str(args.lr)+"_mod_"+str(args.mod)+"_sam_"+str(args.sampling_method)

        # Record time
        now=time.strftime("%Y-%m-%d-%H-%M-%S")
        # Initialize logging and std out

        # Create directories if they do not exist
        for path in [args.logging_file,args.bulk_model_path,args.sc_model_path,args.sc_encoder_path,"save/adata/","save/results/sc" + '/' + args.drug]:
            if not os.path.exists(path):
                # Create a new directory because it does not exist
                os.makedirs(path)
                print("The new directory is created!")

        # Save arguments
        # Overwrite params if checkpoint is provided
        if(args.checkpoint not in ["False","True"]):
            para = os.path.basename(selected_model).split("_DaNN.pkl")[0]
            args.checkpoint = "True"

        sc_encoder_path = args.sc_encoder_path + para
        source_model_path = args.bulk_model_path + para + ".pkl"
        print(source_model_path)
        #print(source_model_path)
        target_model_path = args.sc_model_path + para

        ## target data
        x_target = pd.read_csv("../../Datasets/processedData" + '/' + args.drug + "/target_data/target_scaled" + args.gene + ".csv",
                                   index_col=0)

        y_target = pd.read_csv("../../Datasets/processedData" + '/' + args.drug + "/target_data/target_meta_data.csv",
                               index_col=0)

        ## Add adata
        count = x_target.T
        adata = sc.AnnData(count)
        adata.obs['response'] = y_target["response"]
        ## source data
        x_source = pd.read_csv("../../Datasets/processedData" + '/' + args.drug + "/source_data/source_scaled" + args.gene + ".csv",
                               index_col=0)
        y_source = pd.read_csv("../../Datasets/processedData/" + args.drug + "/source_data/source_meta_data.csv", index_col=0)

        # Split data to train and valid set
        x_target = x_target.T
        Xtarget_train, Xtarget_valid = train_test_split(x_target, test_size=valid_size, random_state=i)

        # Select the device of gpu
        if(args.device == "gpu"):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)

        # Construct datasets and data loaders
        Xtarget_trainTensor = torch.FloatTensor(Xtarget_train.values).to(device)
        Xtarget_validTensor = torch.FloatTensor(Xtarget_valid.values).to(device)
        #print(Xtarget_validTensor.shape)

        #print("C",Ctarget_validTensor )
        X_allTensor = torch.FloatTensor(x_target.values).to(device)

        train_dataset = TensorDataset(Xtarget_trainTensor,Xtarget_trainTensor)
        valid_dataset = TensorDataset(Xtarget_validTensor,Xtarget_validTensor)

        Xtarget_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        Xtarget_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

        dataloaders_pretrain = {'train':Xtarget_trainDataLoader,'val':Xtarget_validDataLoader}
        #print('START SECTION OF LOADING SC DATA TO THE TENSORS')
    ################################################# START SECTION OF LOADING SC DATA TO THE TENSORS #################################################

        # Split source data
        x_source = x_source.T
        Xsource_train_all, Xsource_test, Ysource_train_all, Ysource_test = train_test_split(x_source,y_source, test_size=test_size, random_state=i)
        Xsource_train, Xsource_valid, Ysource_train, Ysource_valid = train_test_split(Xsource_train_all,Ysource_train_all, test_size=valid_size, random_state=i)

        Ysource_train = Ysource_train["response"]
        Ysource_valid = Ysource_valid["response"]
        Ysource_test = Ysource_test["response"]
        # Transform source data
        # Construct datasets and data loaders
        Xsource_trainTensor = torch.FloatTensor(Xsource_train.values).to(device)
        Xsource_validTensor = torch.FloatTensor(Xsource_valid.values).to(device)

        Ysource_trainTensor = torch.LongTensor(Ysource_train.values).to(device)
        Ysource_validTensor = torch.LongTensor(Ysource_valid.values).to(device)

        sourcetrain_dataset = TensorDataset(Xsource_trainTensor, Ysource_trainTensor)
        sourcevalid_dataset = TensorDataset(Xsource_validTensor, Ysource_validTensor)

        Xsource_trainDataLoader = DataLoader(dataset=sourcetrain_dataset, batch_size=batch_size, shuffle=True)
        Xsource_validDataLoader = DataLoader(dataset=sourcevalid_dataset, batch_size=batch_size, shuffle=True)

        dataloaders_source = {'train':Xsource_trainDataLoader,'val':Xsource_validDataLoader}
        #print('END SECTION OF LOADING BULK DATA')
    ################################################# END SECTION OF LOADING BULK DATA  #################################################
        # print("data.shape:", data.shape)
        # exit()
    ################################################# START SECTION OF MODEL CUNSTRUCTION  #################################################
        # Construct target encoder
        if reduce_model == "AE":
            encoder = AEBase(input_dim=x_source.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
            loss_function_e = nn.MSELoss()
        elif reduce_model == "VAE":
            encoder = VAEBase(input_dim=x_source.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
        if reduce_model == "DAE":
            encoder = AEBase(input_dim=x_source.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
            loss_function_e = nn.MSELoss()
        elif reduce_model == "CVAE":
            encoder = CVAEBase(input_dim=x_source.shape[1],n_conditions=1,latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)

        encoder.to(device)
        optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
        loss_function_e = nn.MSELoss()
        exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

        # Binary classification
        dim_model_out = 2
        # Load the trained source encoder and predictor
        print("Xsource_train.shape[1]:",Xsource_train.shape[1])
        if reduce_model == "AE":
            source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,
                    hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                    pretrained_weights=None,freezed=freeze,drop_out=args.dropout,drop_out_predictor=args.dropout)
            source_encoder = source_model
        if reduce_model == "DAE":
            source_model = PretrainedPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,
                    hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                    pretrained_weights=None,freezed=freeze,drop_out=args.dropout,drop_out_predictor=args.dropout)
            source_encoder = source_model
        # Load VAE model
        elif reduce_model == "VAE":
            source_model = PretrainedVAEPredictor(input_dim=Xsource_train.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,
                    hidden_dims_predictor=predict_hdims,output_dim=dim_model_out,
                    pretrained_weights=None,freezed=freeze,z_reparam=bool(args.VAErepram),drop_out=args.dropout,drop_out_predictor=args.dropout)
            source_encoder = source_model

        source_encoder.to(device)
    ################################################# END SECTION OF MODEL CUNSTRUCTION  #################################################

    ################################################# START SECTION OF SC MODEL PRETRAININIG  #################################################
        # Pretrain target encoder training
        # Pretain using autoencoder is pretrain is not False
        if(str(args.sc_encoder_path)!='False'):
            # Pretrained target encoder if there are not stored files in the harddisk
            train_flag = True
            sc_encoder_path = str(sc_encoder_path)
            print("Pretrain=="+sc_encoder_path)

            # If pretrain is not False load from check point
            if(args.checkpoint!="False"):
                # if checkpoint is not False, load the pretrained model
                try:
                    encoder.load_state_dict(torch.load(sc_encoder_path))
                    print("Load pretrained target encoder from "+sc_encoder_path)
                    train_flag = False

                except:
                    print("Loading failed, procceed to re-train model")
                    train_flag = True

            # If pretrain is not False and checkpoint is False, retrain the model
            if train_flag == True:
                if reduce_model == "AE":
                    encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                                optimizer=optimizer_e,loss_function=loss_function_e,load=False,
                                                n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=sc_encoder_path + '.pkl')
                if reduce_model == "DAE":
                    encoder,loss_report_en = t.train_DAE_model(net=encoder,task='sc',data_loaders=dataloaders_pretrain,
                                                optimizer=optimizer_e,loss_function=loss_function_e,load=False,
                                                n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=sc_encoder_path + '.pkl')

                elif reduce_model == "VAE":
                    encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                    optimizer=optimizer_e,load=False,
                                    n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=sc_encoder_path + '.pkl')
                if reduce_model == "CVAE":
                    encoder,loss_report_en = t.train_CVAE_model(net=encoder,task='sc',data_loaders=dataloaders_pretrain,
                                    optimizer=optimizer_e,load=False,
                                    n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=sc_encoder_path + '.pkl')
                #print(loss_report_en)
                print("Pretrained finished")

            # Before Transfer learning, we test the performance of using no transfer performance:
            # Use vae result to predict
            embeddings_pretrain = encoder.encode(X_allTensor)
            print(embeddings_pretrain)
            pretrain_prob_prediction = source_model.predict(embeddings_pretrain).detach().cpu().numpy()

            adata.obs["sens_preds_pret"] = pretrain_prob_prediction[:,1]
            adata.obs["sens_label_pret"] = pretrain_prob_prediction.argmax(axis=1)

            # Add embeddings to the adata object
            embeddings_pretrain = embeddings_pretrain.detach().cpu().numpy()
            adata.obsm["X_pre"] = embeddings_pretrain
    ################################################# END SECTION OF SC MODEL PRETRAININIG  #################################################

    ################################################# START SECTION OF TRANSFER LEARNING TRAINING #################################################
        # Using DaNN transfer learning
        # DaNN model
        # Set predictor loss
        loss_d = nn.CrossEntropyLoss()
        optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
        exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)

        # Set DaNN model
        #DaNN_model = DaNN(source_model=source_encoder,target_model=encoder)
        DaNN_model = DaNN(source_model=source_encoder,target_model=encoder,fix_source=bool(args.fix_source))
        DaNN_model.to(device)

        # Set distribution loss
        def loss(x,y,GAMMA=args.mmd_GAMMA):
            result = mmd.mmd_loss(x,y,GAMMA)
            return result

        loss_disrtibution = loss

        # Train DaNN model
        print("Trainig using " + mod + " model")
        target_model = TargetModel(source_model,encoder)
        # Switch to use regularized DaNN model or not
        if mod == 'ori':
            if args.checkpoint == 'True':
                DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
                                    dataloaders_source,dataloaders_pretrain,
                                    # Should here be all optimizer d?
                                    optimizer_d, loss_d,
                                    epochs,exp_lr_scheduler_d,
                                    dist_loss=loss_disrtibution,
                                    load=target_model_path+"_DaNN.pkl",
                                    weight = args.mmd_weight,
                                    save_path=target_model_path+"_DaNN.pkl")
            else:
                DaNN_model, report_ = t.train_DaNN_model(DaNN_model,
                                    dataloaders_source,dataloaders_pretrain,
                                    # Should here be all optimizer d?
                                    optimizer_d, loss_d,
                                    epochs,exp_lr_scheduler_d,
                                    dist_loss=loss_disrtibution,
                                    load=False,
                                    weight = args.mmd_weight,
                                    save_path=target_model_path+"_DaNN.pkl")
        # Train DaNN model with new loss function
        if mod == 'new':
            #args.checkpoint = 'False'
            if args.checkpoint == 'True':
                DaNN_model, report_, _, _ = t.train_DaNN_model2(DaNN_model,
                                dataloaders_source,dataloaders_pretrain,
                                # Should here be all optimizer d?
                                optimizer_d, loss_d,
                                epochs,exp_lr_scheduler_d,
                                dist_loss=loss_disrtibution,
                                load=selected_model,
                                weight = args.mmd_weight,
                                save_path=target_model_path+"_DaNN.pkl")
            else:
                DaNN_model, report_, _, _ = t.train_DaNN_model2(DaNN_model,
                                    dataloaders_source,dataloaders_pretrain,
                                    # Should here be all optimizer d?
                                    optimizer_d, loss_d,
                                    epochs,exp_lr_scheduler_d,
                                    dist_loss=loss_disrtibution,
                                    load=False,
                                    weight = args.mmd_weight,
                                    save_path=target_model_path+"_DaNN.pkl",
                                    device=device)

        encoder = DaNN_model.target_model
        source_model = DaNN_model.source_model
        print("Transfer DaNN finished")
    ################################################# END SECTION OF TRANSER LEARNING TRAINING #################################################


    ################################################# START SECTION OF PREPROCESSING FEATURES #################################################
        # Extract feature embeddings
        # Extract prediction probabilities
        embedding_tensors = encoder.encode(X_allTensor)
        prediction_tensors = source_model.predictor(embedding_tensors)
        embeddings = embedding_tensors.detach().cpu().numpy()
        predictions = prediction_tensors.detach().cpu().numpy()
        print("predictions",predictions.shape)
        # Transform predict8ion probabilities to 0-1 labels

        adata.obs["sens_preds"] = predictions[:,1]
        adata.obs["sens_label"] = predictions.argmax(axis=1)
        adata.obs["sens_label"] = adata.obs["sens_label"].astype('category')
        adata.obs["rest_preds"] = predictions[:,0]

    ################################################# END SECTION OF ANALYSIS AND POST PROCESSING #################################################

    ################################################# START SECTION OF ANALYSIS FOR scRNA-Seq DATA #################################################
        # Save adata
        adata.write("save/adata/"+select_drug+para+".h5ad")
    ################################################# END SECTION OF ANALYSIS FOR scRNA-Seq DATA #################################################
        from sklearn.metrics import (average_precision_score,
                                 classification_report, mean_squared_error, r2_score, roc_auc_score)
        report_df = {}
        Y_test = adata.obs['response']
        sens_pb_results = adata.obs['sens_preds']
        lb_results = adata.obs['sens_label']

        acc = accuracy_score(Y_test, lb_results)
        precision = precision_score(Y_test, lb_results)
        recall = recall_score(Y_test, lb_results)
        f1 = f1_score(Y_test, lb_results)
        auc = roc_auc_score(Y_test, sens_pb_results)
        aupr = average_precision_score(Y_test, sens_pb_results)
        print("Data:{} ACC: {:.6f} AUC: {:.6f} AUPR: {:.6f} F1: {:.6f} Precision: {:.6f} Recall: {:.6f}".format(args.drug,acc, auc, aupr, f1, precision, recall))

        test_file = "./save/results/sc" + '/' + args.drug + '/' + para + '_noPre.tsv'
        with open(test_file, 'a') as p:
            p.write("num\t{}\tAUC\t{}\tAUPR\t{}\n".format(i, auc, aupr))

        #Y_test ture label
        report_dict = classification_report(Y_test, lb_results, output_dict=True)
        f1score = report_dict['weighted avg']['f1-score']
        report_df['f1_score'] = f1score
        file = 'save/bulk_f_'+select_drug+'_f1_score_ori.txt'
        with open(file, 'a+') as f:
             f.write(para+'\t'+str(f1score)+'\n')
        print("sc model finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--drug', type=str, default='Vorinostat',
                        help='choose drug,Can be PLX4720_451Lu,Gefitinib,Erlotinib,PLX4720,Vorinostat, Cetuximab,AR-42,and Etoposide.')
    parser.add_argument('--gene', type=str, default="_tp4k",
                        help='choose data type. Can be all , _tp4k or _gene. Default: _tp4k')
    parser.add_argument('--missing_value', type=int, default=1,
                        help='The value filled in the missing entry in the drug screening annotation, default: 1')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Size of the test set for the bulk model traning, default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2,
                        help='Size of the validation set for the bulk model traning, default: 0.2')
    parser.add_argument('--mmd_weight', type=float, default=0.25,
                        help="Weight of the MMD loss of the transfer learning, default: 0.25")
    parser.add_argument('--mmd_GAMMA', type=int, default=1000,
                        help="Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000")

    # train
    parser.add_argument('--sampling_method', type=str, default="weight",
                        help='choose sampling type. Can be weight or smote. Default: weight')
    parser.add_argument('--device', type=str, default="gpu",
                        help='Device to train the model. Can be cpu or gpu. Deafult: cpu')
    parser.add_argument('--bulk_model_path', '-s', type=str, default='save/bulk_pre_my/',
                        help='Path of the trained predictor in the bulk level')
    parser.add_argument('--sc_model_path', '-p', type=str, default='save/sc_pre_my/',
                        help='Path (prefix) of the trained predictor in the single cell level')
    parser.add_argument('--sc_encoder_path', type=str, default='save/sc_encoder_my/',
                        help='Path of the pre-trained encoder in the single-cell level')
    parser.add_argument('--checkpoint', type=str, default='False',
                        help='Load weight from checkpoint files, can be True,False, or a file path. Checkpoint files can be paraName1_para1_paraName2_para2... Default: True')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epoches training. Default: 500')
    parser.add_argument('--batch_size', type=int, default=200, help='Number of batch size when training. Default: 200')
    parser.add_argument('--bottleneck', type=int, default=256,
                        help='Size of the bottleneck layer of the model. Default: 32')
    parser.add_argument('--dimreduce', type=str, default="DAE", help='Encoder model type. Can be AE or VAE. Default: AE')
    parser.add_argument('--freeze_pretrain', type=int, default=0,
                        help='Fix the prarmeters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0')
    parser.add_argument('--bulk_h_dims', type=str, default="512,256", help='Shape of the source encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--sc_h_dims', type=str, default="512,256", help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--predictor_h_dims', type=str, default="256,128", help='Shape of the predictor. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 16,8')
    parser.add_argument('--VAErepram', type=int, default=1)
    parser.add_argument('--batch_id', type=str, default="HN137", help="Batch id only for testing")
    parser.add_argument('--load_sc_model', type=int, default=0,
                        help='Load a trained model or not. 0: do not load, 1: load. Default: 0')

    parser.add_argument('--mod', type=str, default='new',
                        help='Embed the cell type label to regularized the training: new: add cell type info, ori: do not add cell type info. Default: new')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of neural network. Default: 0.3')
    # miss
    parser.add_argument('--fix_source', type=int, default=0, help='Fix the bulk level model. Default: 0')
    parser.add_argument('--bulk', type=str, default='integrate',
                        help='Selection of the bulk database.integrate:both dataset. old: GDSC. new: CCLE. Default: integrate')
    #
    args, unknown = parser.parse_known_args()
    run_main(args)