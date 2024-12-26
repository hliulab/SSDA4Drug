import os
import numpy as np
import torch
from utils import save_AUCs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from torch import nn
import utils


def test_experiment(net, dataset, loss_c, device, test_predicted_y_file, YTestCells):
    net.eval()
    Y_pre = []
    Y_true = []
    AUC_scores = []
    APR_scores = []
    running_loss = []
    correct = 0

    for _, data in enumerate(zip(dataset)):
        data = data[0]
        x = data[0].to(device)
        y_true = data[1].to(device)
        y_out = net(x)
        loss = loss_c(y_out, y_true.long())
        running_loss.append(loss.item())
        y_out = nn.Softmax(dim=1)(y_out)

        y_true = y_true.cpu()
        y_out = y_out.cpu()
        y_score = y_out[:, 1]
        pred = y_out.max(1, keepdim=True)[1]
        correct += pred.eq(y_true.view_as(pred)).sum().item()

        AUC_train = utils.roc_auc_score_trainval(y_true.detach().numpy(), y_score.detach().numpy())
        APR_train = average_precision_score(y_true.detach().numpy(), y_score.detach().numpy())

        Y_pre.append(y_score)
        Y_true.append(y_true)

        AUC_scores.append(AUC_train)
        APR_scores.append(APR_train)

    Y_pre = torch.cat(Y_pre, dim=0).detach().numpy()
    Y_true = torch.cat(Y_true, dim=0).detach().numpy()
    auc = np.mean(np.array(AUC_scores))
    aupr = np.mean(np.array(APR_scores))
    loss = np.mean(np.array(running_loss))
    print("Test Performance -> Loss: {:.6f} AUC:{:.4f} AUPR:{:.4f} Accuracy: {}/{} ({:.0f}%) ".format(loss, auc, aupr,
                                                                                    correct, len(dataset.dataset),
                                                                                100. * correct / len(dataset.dataset)))

    with open(test_predicted_y_file, 'w') as p:
        TYTestCells = YTestCells['response'].values
        for tp in range(TYTestCells.shape[0]):
            p.write("{}\t{}\t{}\n".format(YTestCells.index[tp], YTestCells['response'].values[tp], Y_pre[tp]))


def test_shot(net, dataset, loss_c, device, test_file,i):
    net.eval()
    Y_pre = []
    Y_true = []
    AUC_scores = []
    APR_scores = []
    running_loss = []
    correct = 0

    for _, data in enumerate(zip(dataset)):
        data = data[0]
        x = data[0].to(device)
        y_true = data[1].to(device)
        y_out = net(x)
        loss = loss_c(y_out, y_true.long())
        running_loss.append(loss.item())
        y_out = nn.Softmax(dim=1)(y_out)

        y_true = y_true.cpu()
        y_out = y_out.cpu()
        y_score = y_out[:, 1]
        pred = y_out.max(1, keepdim=True)[1]
        correct += pred.eq(y_true.view_as(pred)).sum().item()

        AUC_train = utils.roc_auc_score_trainval(y_true.detach().numpy(), y_score.detach().numpy())
        APR_train = average_precision_score(y_true.detach().numpy(), y_score.detach().numpy())

        Y_pre.append(y_score)
        Y_true.append(y_true)

        AUC_scores.append(AUC_train)
        APR_scores.append(APR_train)

    Y_pre = torch.cat(Y_pre, dim=0).detach().numpy()
    Y_true = torch.cat(Y_true, dim=0).detach().numpy()
    auc = np.mean(np.array(AUC_scores))
    aupr = np.mean(np.array(APR_scores))
    loss = np.mean(np.array(running_loss))
    print("Test Performance -> Loss: {:.6f} AUC:{:.4f} AUPR:{:.4f} Accuracy: {}/{} ({:.0f}%) ".format(loss, auc, aupr,
                                                                                    correct, len(dataset.dataset),
                                                                                100. * correct / len(dataset.dataset)))

    with open(test_file, 'a') as p:
        p.write("num\t{}\tAUC\t{}\tAUPR\t{}\n".format(i, auc, aupr))


def test_single_model(net, dataset, loss_c, device, test_predicted_y_file, YTestCells):
    net.eval()
    AUC = []
    APR = []
    Y_pre = []
    Y_true = []
    running_loss = []
    correct = 0

    for _, data in enumerate(zip(dataset)):
        data = data[0]
        x = data[0].to(device)
        y_true = data[1].to(device)
        y_out = net(x)
        loss = loss_c(y_out, y_true.long())
        running_loss.append(loss.item())
        y_out = nn.Softmax(dim=1)(y_out)

        y_true = y_true.cpu()
        y_out = y_out.cpu()
        y_score = y_out[:, 1]
        pred = y_out.max(1, keepdim=True)[1]
        correct += pred.eq(y_true.view_as(pred)).sum().item()

        Y_pre.append(y_score)
        Y_true.append(y_true)

        auc_test = utils.roc_auc_score_trainval(y_true.detach().numpy(), y_score.detach().numpy())
        apr_test = average_precision_score(y_true.detach().numpy(), y_score.detach().numpy())

        AUC.append(auc_test)
        APR.append(apr_test)

    Y_pre = torch.cat(Y_pre, dim=0).detach().numpy()
    Y_true = torch.cat(Y_true, dim=0).detach().numpy()

    auc = np.mean(np.array(AUC))
    aupr = np.mean(np.array(APR))
    loss = np.mean(np.array(running_loss))
    print("Test Performance -> Loss: {:.6f}  Accuracy: {}/{} ({:.0f}%) AUC: {:.3f} AUPR: {:.3f}".format(loss, correct,
                                                                                                        len(dataset.dataset),
                                                                                                        100. * correct / len(
                                                                                                            dataset.dataset),
                                                                                                        auc, aupr))

    with open(test_predicted_y_file, 'w') as p:
        TYTestCells = YTestCells['response'].values
        for tp in range(TYTestCells.shape[0]):
            p.write("{}\t{}\t{}\n".format(YTestCells.index[tp], YTestCells['response'].values[tp], Y_pre[tp]))


def train_semi_dae(fgm, encoder, predictor, adentropy_p, source_loader, unlabeled_loader, labeled_loader, method,
                   optimizer, loss_class, loss_e, n_epochs, start_epoch=0,
                   save_path="save/model.pkl",
                   device="cuda", auc_path=''):
    best_loss = np.inf
    best_auc = float('-inf')
    best_encoder = None
    best_predictor = None
    best_adentropy_p = None

    if not os.path.exists(auc_path):
        os.makedirs(auc_path)
    file_AUCs = auc_path + '/' + '_train_AUCs_' + '.txt'

    AUCs = ('Epoch\tloss\tloss_dann\tloss_c\tloss_dae_s\tloss_dae_t\tphase\tauc\taupr')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    for epoch in range(n_epochs):

        AUCtrain_source = []
        AUCval_source = []

        APRtrain_source = []
        APRval_source = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #  Set model to training mode
                encoder.train()
                predictor.train()
                adentropy_p.train()
            else:
                # Set model to evaluate mode
                encoder.eval()
                predictor.eval()
                adentropy_p.eval()

            running_loss_train = []
            running_loss_valid = []
            running_dann_train = []
            running_dann_valid = []
            running_c_train = []
            running_c_valid = []
            running_e_train_s = []
            running_e_valid_s = []
            running_e_train_t = []
            running_e_valid_t = []

            len_source = len(source_loader[phase])
            len_unlabeled = len(unlabeled_loader[phase])
            len_labeled = len(labeled_loader[phase])

            num_iter = max(len_source, len_unlabeled, len_labeled)

            for batch_idx in range(num_iter):
                if batch_idx % len_source == 0:
                    iter_source = iter(source_loader[phase])
                if batch_idx % len_unlabeled == 0:
                    iter_target_unl = iter(unlabeled_loader[phase])
                if batch_idx % len_labeled == 0:
                    iter_target = iter(labeled_loader[phase])
                xs, ys = iter_source.__next__()
                xt, yt = iter_target.__next__()
                xt_unl, _ = iter_target_unl.__next__()
                xt_unl = xt_unl.to(device)
                xs = xs.to(device)
                ys = ys.to(device)
                xt = xt.to(device)
                yt = yt.to(device)

                xs.requires_grad_(True)
                xt.requires_grad_(True)

                data = torch.cat((xs, xt), 0)
                target = torch.cat((ys, yt), 0)

                feature, ae_output = encoder(data)

                output = predictor(feature)
                output = adentropy_p(output)
                loss_c = loss_class(output, target.long())
                softmax_output = nn.Softmax(dim=1)(output)

                ## DAE
                loss_ae_t = loss_e(xt, ae_output.narrow(0, xs.size(0), xt.size(0)))
                loss_ae_s = loss_e(xs, ae_output.narrow(0, 0, xs.size(0)))

                loss = loss_c + loss_ae_t + loss_ae_s

                loss.backward(retain_graph=True)
                if method == "adv":
                    fgm.attack()
                    # optimizer.zero_grad() # Don’t want to accumulate gradients
                    feature, ae_output = encoder(data)

                    output = predictor(feature)
                    output = adentropy_p(output)
                    loss_c_2 = loss_class(output, target.long())
                    loss_ae_t_2 = loss_e(xt, ae_output.narrow(0, xs.size(0), xt.size(0)))
                    loss_ae_s_2 = loss_e(xs, ae_output.narrow(0, 0, xs.size(0)))
                    loss_sum = loss_c_2 + loss_ae_t_2 + loss_ae_s_2
                    loss_sum.backward(retain_graph=True)
                    fgm.restore()
                optimizer.step()
                optimizer.zero_grad()

                ## MME
                output, ae_output = encoder(xt_unl)
                output = predictor(output)

                loss_t = utils.adentropy(adentropy_p, output, 0.1)
                loss_t.backward()
                optimizer.step()
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    ys = ys.cpu()
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()
                    y_pre = y_pre[:, 1]
                    AUC_train = utils.roc_auc_score_trainval(ys.detach().numpy(), y_pre.detach().numpy())
                    APR_train = average_precision_score(ys.detach().numpy(), y_pre.detach().numpy())
                    AUCtrain_source.append(AUC_train)
                    APRtrain_source.append(APR_train)

                    running_loss_train.append(loss.item())
                    running_dann_train.append(-loss_t.item())
                    running_c_train.append(loss_c.item())
                    running_e_train_s.append(loss_ae_s.item())
                    running_e_train_t.append(loss_ae_t.item())
                if phase == 'val':
                    ys = ys.cpu()
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()
                    y_pre = y_pre[:, 1]
                    AUC_val = utils.roc_auc_score_trainval(ys.detach().numpy(), y_pre.detach().numpy())
                    APR_val = average_precision_score(ys.detach().numpy(), y_pre.detach().numpy())
                    AUCval_source.append(AUC_val)
                    APRval_source.append(APR_val)

                    running_loss_valid.append(loss.item())
                    running_dann_valid.append(-loss_t.item())
                    running_c_valid.append(loss_c.item())
                    running_e_valid_s.append(loss_ae_s.item())
                    running_e_valid_t.append(loss_ae_t.item())

            # Average epoch loss
            if phase == 'train':
                epoch_loss = np.mean(np.array(running_loss_train))
                epoch_dann = np.mean(np.array(running_dann_train))
                epoch_c = np.mean(np.array(running_c_train))
                epoch_ae_s = np.mean(np.array(running_e_train_s))
                epoch_ae_t = np.mean(np.array(running_e_train_t))
                auc = np.mean(np.array(AUCtrain_source))
                aupr = np.mean(np.array(APRtrain_source))

            if phase == 'val':
                epoch_loss = np.mean(np.array(running_loss_valid))
                epoch_dann = np.mean(np.array(running_dann_valid))
                epoch_c = np.mean(np.array(running_c_valid))
                epoch_ae_s = np.mean(np.array(running_e_valid_s))
                epoch_ae_t = np.mean(np.array(running_e_valid_t))
                auc = np.mean(np.array(AUCval_source))
                aupr = np.mean(np.array(APRval_source))

            train_AUCs = [epoch, epoch_loss, epoch_dann, epoch_c, epoch_ae_s, epoch_ae_t, phase, auc, aupr]
            print(
                "Epoch:{} Phase:{} AUC: {:.3f} AUPR: {:.3f} ".format(
                    epoch, phase, auc, aupr))
            save_AUCs(train_AUCs, file_AUCs)

            if (phase == 'val') and (epoch_loss < best_loss or best_auc < auc):
                best_auc = auc
                best_loss = epoch_loss

    return encoder, predictor, adentropy_p


def train_semi_mlp(encoder, predictor, adentropy_p, source_loader, unlabeled_loader, labeled_loader, method,
                   optimizer, loss_class, loss_e, n_epochs, start_epoch=0,
                   save_path="save/model.pkl",
                   device="cuda", auc_path=''):
    best_loss = np.inf
    best_auc = float('-inf')

    if not os.path.exists(auc_path):
        os.makedirs(auc_path)
    file_AUCs = auc_path + '/' + '_train_AUCs_' + '.txt'

    AUCs = ('Epoch\tloss\tloss_dann\tloss_c\tphase\tauc\taupr')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    for epoch in range(n_epochs):

        AUCtrain_source = []
        AUCval_source = []

        APRtrain_source = []
        APRval_source = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #  Set model to training mode
                encoder.train()
                predictor.train()
                adentropy_p.train()
            else:
                # Set model to evaluate mode
                encoder.eval()
                predictor.eval()
                adentropy_p.eval()

            running_loss_train = 0.0
            running_loss_valid = 0.0
            running_dann_train = 0.0
            running_dann_valid = 0.0
            running_c_train = 0.0
            running_c_valid = 0.0

            len_source = len(source_loader[phase])
            len_unlabeled = len(unlabeled_loader[phase])
            len_labeled = len(labeled_loader[phase])

            num_iter = max(len_source, len_unlabeled, len_labeled)

            for batch_idx in range(num_iter):
                if batch_idx % len_source == 0:
                    iter_source = iter(source_loader[phase])
                if batch_idx % len_unlabeled == 0:
                    iter_target_unl = iter(unlabeled_loader[phase])
                if batch_idx % len_labeled == 0:
                    iter_target = iter(labeled_loader[phase])
                xs, ys = iter_source.__next__()
                xt, yt = iter_target.__next__()
                xt_unl, _ = iter_target_unl.__next__()
                xt_unl = xt_unl.to(device)
                xs = xs.to(device)
                ys = ys.to(device)
                xt = xt.to(device)
                yt = yt.to(device)

                xs.requires_grad_(True)
                xt.requires_grad_(True)

                data = torch.cat((xs, xt), 0)
                target = torch.cat((ys, yt), 0)

                feature = encoder(data)

                output = predictor(feature)
                output = adentropy_p(output)
                loss_c = loss_class(output, target.long())
                softmax_output = nn.Softmax(dim=1)(output)

                loss = loss_c

                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

                ## MME
                output = encoder(xt_unl)
                output = predictor(output)

                loss_t = utils.adentropy(adentropy_p, output, 0.1)
                loss_t.backward()
                optimizer.step()
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    ys = ys.cpu()
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()
                    y_pre = y_pre[:, 1]
                    AUC_train = utils.roc_auc_score_trainval(ys.detach().numpy(), y_pre.detach().numpy())
                    APR_train = average_precision_score(ys.detach().numpy(), y_pre.detach().numpy())
                    AUCtrain_source.append(AUC_train)
                    APRtrain_source.append(APR_train)

                    running_loss_train += loss.item()
                    running_dann_train += -loss_t.item()
                    # running_sc_train += loss_s.item()
                    running_c_train += loss_c.item()
                if phase == 'val':
                    ys = ys.cpu()
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()
                    y_pre = y_pre[:, 1]
                    AUC_val = utils.roc_auc_score_trainval(ys.detach().numpy(), y_pre.detach().numpy())
                    APR_val = average_precision_score(ys.detach().numpy(), y_pre.detach().numpy())
                    AUCval_source.append(AUC_val)
                    APRval_source.append(APR_val)

                    running_loss_valid += loss.item()
                    running_dann_valid += -loss_t.item()
                    running_c_valid += loss_c.item()

            # Average epoch loss
            if phase == 'train':
                epoch_loss = np.mean(np.array(running_loss_train))
                epoch_dann = np.mean(np.array(running_dann_train))
                # epoch_sc = running_sc_train / n_iters
                epoch_c = np.mean(np.array(running_c_train))
                auc = np.mean(np.array(AUCtrain_source))
                aupr = np.mean(np.array(APRtrain_source))

            if phase == 'val':
                epoch_loss = np.mean(np.array(running_loss_valid))
                epoch_dann = np.mean(np.array(running_dann_valid))
                # epoch_sc = running_e_train_s / n_iters
                epoch_c = np.mean(np.array(running_c_valid))
                auc = np.mean(np.array(AUCval_source))
                aupr = np.mean(np.array(APRval_source))

            train_AUCs = [epoch, epoch_loss, epoch_dann, epoch_c, phase, auc, aupr]
            print(
                "Epoch:{} Phase:{} AUC: {:.6f} AUPR: {:.6f} ".format(
                    epoch, phase, auc, aupr))
            save_AUCs(train_AUCs, file_AUCs)

            if (phase == 'val') and (epoch_loss < best_loss or best_auc < auc):
                best_auc = auc
                best_loss = epoch_loss

    return encoder, predictor, adentropy_p


def train_baseline_dae(encoder, predictor, adentropy_p, source_loader, method,
                   optimizer, loss_class, loss_e, n_epochs, start_epoch=0,
                   save_path="save/model.pkl",
                   device="cuda", auc_path=''):
    best_loss = np.inf
    best_auc = float('-inf')
    best_encoder = None
    best_predictor = None
    best_adentropy_p = None

    if not os.path.exists(auc_path):
        os.makedirs(auc_path)
    file_AUCs = auc_path + '/' + '_train_AUCs_' + '.txt'

    AUCs = ('Epoch\tloss\tloss_dann\tloss_c\tloss_dae\tphase\tauc\taupr')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    for epoch in range(n_epochs):

        AUCtrain_source = []
        AUCval_source = []

        APRtrain_source = []
        APRval_source = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #  Set model to training mode
                encoder.train()
                predictor.train()
                adentropy_p.train()
            else:
                # Set model to evaluate mode
                encoder.eval()
                predictor.eval()
                adentropy_p.eval()

            running_loss_train = []
            running_loss_valid = []
            running_dann_train = []
            running_dann_valid = []
            running_c_train = []
            running_c_valid = []
            running_e_train = []
            running_e_valid = []

            len_source = len(source_loader[phase])

            for batch_idx in range(len_source):
                iter_source = iter(source_loader[phase])
                xs, ys = iter_source.__next__()

                xs = xs.to(device)
                ys = ys.to(device)

                xs.requires_grad_(True)
                feature, ae_output = encoder(xs)

                output = predictor(feature)
                output = adentropy_p(output)
                loss_c = loss_class(output, ys.long())
                softmax_output = nn.Softmax(dim=1)(output)
                loss_ae = loss_e(xs, ae_output.narrow(0, 0, xs.size(0)))

                loss = loss_c + loss_ae

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    ys = ys.cpu()
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()
                    y_pre = y_pre[:, 1]
                    AUC_train = utils.roc_auc_score_trainval(ys.detach().numpy(), y_pre.detach().numpy())
                    APR_train = average_precision_score(ys.detach().numpy(), y_pre.detach().numpy())
                    AUCtrain_source.append(AUC_train)
                    APRtrain_source.append(APR_train)

                    running_loss_train.append(loss.item())
                    running_c_train.append(loss_c.item())
                    running_e_train.append(loss_ae.item())
                if phase == 'val':
                    ys = ys.cpu()
                    y_pre = softmax_output.narrow(0, 0, xs.size(0)).cpu()
                    y_pre = y_pre[:, 1]
                    AUC_val = utils.roc_auc_score_trainval(ys.detach().numpy(), y_pre.detach().numpy())
                    APR_val = average_precision_score(ys.detach().numpy(), y_pre.detach().numpy())
                    AUCval_source.append(AUC_val)
                    APRval_source.append(APR_val)

                    running_loss_valid.append(loss.item())
                    running_c_valid.append(loss_c.item())
                    running_e_valid.append(loss_ae.item())

            # Average epoch loss
            if phase == 'train':
                epoch_loss = np.mean(np.array(running_loss_train))
                epoch_dann = np.mean(np.array(running_dann_train))
                epoch_c = np.mean(np.array(running_c_train))
                epoch_ae = np.mean(np.array(running_e_train))
                auc = np.mean(np.array(AUCtrain_source))
                aupr = np.mean(np.array(APRtrain_source))

            if phase == 'val':
                epoch_loss = np.mean(np.array(running_loss_valid))
                epoch_dann = np.mean(np.array(running_dann_valid))
                epoch_c = np.mean(np.array(running_c_valid))
                epoch_ae = np.mean(np.array(running_e_valid))
                auc = np.mean(np.array(AUCval_source))
                aupr = np.mean(np.array(APRval_source))

            train_AUCs = [epoch, epoch_loss, epoch_dann, epoch_c, epoch_ae, phase, auc, aupr]
            print(
                "Epoch:{} Phase:{} AUC: {:.3f} AUPR: {:.3f} ".format(
                    epoch, phase, auc, aupr))
            save_AUCs(train_AUCs, file_AUCs)

            if (phase == 'val') and (epoch_loss < best_loss or best_auc < auc):
                best_auc = auc
                best_loss = epoch_loss
    return encoder, predictor, adentropy_p


def train_scad(encoder, predictor, discriminator, source_loader, target_loader,
                   optimizer, loss_class, n_epochs, start_epoch=0,
                   save_path="save/model.pkl",
                   device="cuda", auc_path=''):
    best_loss = np.inf
    best_auc = float('-inf')
    best_encoder = None
    best_predictor = None
    best_adentropy_p = None

    if not os.path.exists(auc_path):
        os.makedirs(auc_path)
    file_AUCs = auc_path + '/' + '_train_AUCs_' + '.txt'

    AUCs = ('Epoch\tloss\tloss_dann\tloss_c\tphase\tauc\taupr')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    for epoch in range(n_epochs):

        AUCtrain_source = []
        AUCval_source = []

        APRtrain_source = []
        APRval_source = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #  Set model to training mode
                encoder.train()
                predictor.train()
                discriminator.train()
            else:
                # Set model to evaluate mode
                encoder.eval()
                predictor.eval()
                discriminator.eval()

            running_loss_train = []
            running_loss_valid = []
            running_dann_train = []
            running_dann_valid = []
            running_c_train = []
            running_c_valid = []

            len_source = len(source_loader[phase])
            len_target = len(target_loader[phase])

            num_iter = max(len_source, len_target)

            for batch_idx in range(num_iter):
                if batch_idx % len_source == 0:
                    iter_source = iter(source_loader[phase])
                if batch_idx % len_target == 0:
                    iter_target = iter(target_loader[phase])
                xs, ys = iter_source.__next__()
                xt, yt = iter_target.__next__()

                xs = xs.to(device)
                ys = ys.view(-1,1).to(device)
                xt = xt.to(device)
                yt = yt.view(-1,1).to(device)

                xs.requires_grad_(True)
                xt.requires_grad_(True)

                F_xs = encoder(xs)
                F_xt = encoder(xt)

                yhat_xs = predictor(F_xs)
                # print("yhat_xs.shape:",yhat_xs.shape)
                loss1 = loss_class(yhat_xs, ys.float())

                Labels = torch.ones(F_xs.size(0), 1)
                Labelt = torch.zeros(F_xt.size(0), 1)
                Lst = torch.cat([Labels, Labelt], 0).to(device)
                Xst = torch.cat([F_xs, F_xt], 0).to(device)

                yhat_DG = discriminator(Xst)
                DG_loss = loss_class(yhat_DG, Lst)

                loss2 = 0.1 * DG_loss

                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if phase == 'train':
                    ys = ys.cpu()
                    y_pre = yhat_xs.cpu()
                    AUC_train = utils.roc_auc_score_trainval(ys.detach().numpy(), y_pre.detach().numpy())
                    APR_train = average_precision_score(ys.detach().numpy(), y_pre.detach().numpy())
                    AUCtrain_source.append(AUC_train)
                    APRtrain_source.append(APR_train)

                    running_loss_train.append(loss.item())
                    running_c_train.append(loss1.item())
                    running_dann_train.append(loss2.item())
                if phase == 'val':
                    ys = ys.cpu()
                    y_pre = yhat_xs.cpu()
                    AUC_val = utils.roc_auc_score_trainval(ys.detach().numpy(), y_pre.detach().numpy())
                    APR_val = average_precision_score(ys.detach().numpy(), y_pre.detach().numpy())
                    AUCval_source.append(AUC_val)
                    APRval_source.append(APR_val)

                    running_loss_valid.append(loss.item())
                    running_c_valid.append(loss1.item())
                    running_dann_valid.append(loss2.item())

            # Average epoch loss
            if phase == 'train':
                epoch_loss = np.mean(np.array(running_loss_train))
                epoch_dann = np.mean(np.array(running_dann_train))
                epoch_c = np.mean(np.array(running_c_train))
                auc = np.mean(np.array(AUCtrain_source))
                aupr = np.mean(np.array(APRtrain_source))

            if phase == 'val':
                epoch_loss = np.mean(np.array(running_loss_valid))
                epoch_dann = np.mean(np.array(running_dann_valid))
                epoch_c = np.mean(np.array(running_c_valid))
                auc = np.mean(np.array(AUCval_source))
                aupr = np.mean(np.array(APRval_source))

            train_AUCs = [epoch, epoch_loss, epoch_dann, epoch_c, phase, auc, aupr]
            print(
                "Epoch:{} Phase:{} AUC: {:.3f} AUPR: {:.3f} ".format(
                    epoch, phase, auc, aupr))
            save_AUCs(train_AUCs, file_AUCs)

            if (phase == 'val') and (epoch_loss < best_loss or best_auc < auc):
                best_auc = auc
                best_loss = epoch_loss
    return encoder, predictor, discriminator