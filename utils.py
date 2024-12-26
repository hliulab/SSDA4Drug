import numpy as np
import torch
from sklearn.metrics import *
import torch.nn.functional as F

def adentropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent


def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')


def roc_auc_score_trainval(y_true, y_predicted):
    # https://stackoverflow.com/questions/45139163/roc-auc-score-only-one-class-present-in-y-true?rq=1 #
    if len(np.unique(y_true)) == 1:
        return accuracy_score(y_true, np.rint(y_predicted))
    return roc_auc_score(y_true, y_predicted)


def cell_dim(drug, gene):
    global bulk_tasks, sc_tasks
    if drug == 'GSE149383':
        if gene == "":
            bulk_tasks = {'expression': 10668, 'pathway': 128}
            sc_tasks = {'expression': 10668, 'pathway': 128}
        if gene == "tp4k":
            bulk_tasks = {'expression': 3994, 'pathway': 128}
            sc_tasks = {'expression': 3994, 'pathway': 128}

    if drug == 'GSE140440':
        if gene == "":
            bulk_tasks = {'expression': 13762, 'pathway': 128}
            sc_tasks = {'expression': 13762, 'pathway': 128}
        if gene == "tp4k":
            bulk_tasks = {'expression': 3994, 'pathway': 128}
            sc_tasks = {'expression': 3994, 'pathway': 128}

    if drug == 'Cetuximab':
        if gene == "":
            bulk_tasks = {'expression': 10684, 'pathway': 128}
            sc_tasks = {'expression': 10684, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2876, 'pathway': 128}
            sc_tasks = {'expression': 2876, 'pathway': 128}

    if drug == 'Etoposide':
        if gene == "":
            bulk_tasks = {'expression': 9738, 'pathway': 128}
            sc_tasks = {'expression': 9738, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2668, 'pathway': 128}
            sc_tasks = {'expression': 2668, 'pathway': 128}

    if drug == 'PLX4720_451Lu':
        if gene == "":
            bulk_tasks = {'expression': 11839, 'pathway': 128}
            sc_tasks = {'expression': 11839, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2856, 'pathway': 128}
            sc_tasks = {'expression': 2856, 'pathway': 128}

    if drug == 'PLX4720':
        if gene == "":
            bulk_tasks = {'expression': 11937, 'pathway': 128}
            sc_tasks = {'expression': 11937, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2873, 'pathway': 128}
            sc_tasks = {'expression': 2873, 'pathway': 128}

    if drug == 'Vorinostat':
        if gene == "":
            bulk_tasks = {'expression': 10610, 'pathway': 128}
            sc_tasks = {'expression': 10610, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2933, 'pathway': 128}
            sc_tasks = {'expression': 2933, 'pathway': 128}

    if drug == 'Gefitinib':
        if gene == "":
            bulk_tasks = {'expression': 10610, 'pathway': 128}
            sc_tasks = {'expression': 10610, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2933, 'pathway': 128}
            sc_tasks = {'expression': 2933, 'pathway': 128}

    if drug == 'AR-42':
        if gene == "":
            bulk_tasks = {'expression': 10610, 'pathway': 128}
            sc_tasks = {'expression': 10610, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2933, 'pathway': 128}
            sc_tasks = {'expression': 2933, 'pathway': 128}

    if drug == 'NVP-TAE684':
        if gene == "":
            bulk_tasks = {'expression': 10684, 'pathway': 128}
            sc_tasks = {'expression': 10684, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2876, 'pathway': 128}
            sc_tasks = {'expression': 2876, 'pathway': 128}

    if drug == 'Afatinib':
        if gene == "":
            bulk_tasks = {'expression': 10684, 'pathway': 128}
            sc_tasks = {'expression': 10684, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2876, 'pathway': 128}
            sc_tasks = {'expression': 2876, 'pathway': 128}

    if drug == 'Sorafenib':
        if gene == "":
            bulk_tasks = {'expression': 10684, 'pathway': 128}
            sc_tasks = {'expression': 10684, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2876, 'pathway': 128}
            sc_tasks = {'expression': 2876, 'pathway': 128}

    if drug == 'bortezomib':
        if gene == "":
            bulk_tasks = {'expression': 10470, 'pathway': 128}
            sc_tasks = {'expression': 10470, 'pathway': 128}
        if gene == "_tp4k":
            bulk_tasks = {'expression': 2876, 'pathway': 128}
            sc_tasks = {'expression': 2876, 'pathway': 128}

    return bulk_tasks,sc_tasks


def cellline_dim(cell):
    global bulk_tasks, sc_tasks
    if cell == 'MCF7':
        bulk_tasks = {'expression': 12863, 'pathway': 128}
        sc_tasks = {'expression': 12863, 'pathway': 128}
    if cell == 'OSCC':
        bulk_tasks = {'expression': 14543, 'pathway': 128}
        sc_tasks = {'expression': 14543, 'pathway': 128}
    if cell == 'PC9':
        bulk_tasks = {'expression': 2245, 'pathway': 128}
        sc_tasks = {'expression': 2245, 'pathway': 128}
    if cell == 'PCa':
        bulk_tasks = {'expression': 13762, 'pathway': 128}
        sc_tasks = {'expression': 13762, 'pathway': 128}

    return bulk_tasks,sc_tasks

def create_dataset(x,y,batch_size,shuffle,sampler=None):
    x = x.T
    x = torch.FloatTensor(x.values)
    y = y['response']
    y = torch.LongTensor(y.values)
    dataset = torch.utils.data.TensorDataset(x, y)
    dataset = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,sampler=sampler)
    return dataset



